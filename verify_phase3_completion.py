"""Verify completed Phase 3 from saved data; never run BLIP, OCR or fitting."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import uuid
import zlib

from capture_phase3_features import candidate, checked_entry
from evaluate_phase2_dev import PROTOCOL
from evaluate_phase3_dev import DEV_IMAGES, THRESHOLDS, confusion_metrics, select_candidate, summarize
from evaluate_vqa import OfficialScorer
from modules.answerability_features import features
from modules.answerability_gate import Gate, decide
from phase3_common import PHASE2_RECORDS, ROOT, checked_splits, digest, safe_run, sha, verify_freeze, write_new
from prepare_phase3_data import OFFICIAL_URL


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label):
    """Compare nested metric summaries, allowing only floating-point roundoff."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual)==set(expected), label+' keys differ')
        for key in expected:
            same(actual[key], expected[key], label+'.'+str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual)==len(expected), label+' length differs')
        for i,(left,right) in enumerate(zip(actual,expected)):
            same(left,right,label+'.'+str(i))
    elif isinstance(expected, (float,int)) and not isinstance(expected,bool):
        require(isinstance(actual,(float,int)) and math.isfinite(actual) and
                math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-8), label+' differs')
    else:
        require(actual==expected,label+' differs')


def reference_identity(saved, original):
    for key in ('image','question','answers','answerable'):
        require(saved[key]==original[key], 'Original development '+key+' changed')
    require(len(saved['answers'])==10,'All ten original references are required')


def risk_points(configurations, family):
    # Conditional risk is undefined when a configuration returns zero answers.
    return [c for c in configurations if c['name']==family and c['answered_examples']>0]


def audit(run_dir):
    import numpy as np
    require(Path(sys.prefix).name=='.venv-gpu','Use .venv-gpu')
    root=safe_run(run_dir)
    _,train,dev,payload=checked_splits()
    frozen=verify_freeze(full=True)
    gate=Gate(root/'models')
    identity=json.loads((root/'captures/identity.json').read_text())
    prepared=json.loads((root/'data/prepared.json').read_text())
    archive=json.loads((root/'data/archive-identity.json').read_text())
    summary=json.loads((root/'results/summary.json').read_text())
    policy=json.loads((root/'selected-policy.json').read_text())
    require(identity['protocol']==PROTOCOL and identity['ocr_threshold']==.9,'Capture protocol changed')
    require(identity['device']=='cuda' and identity['ocr_device']==identity['object_device']=='cpu','Capture devices changed')
    for name,expected in identity['source_sha256'].items():
        require(sha(ROOT/name)==expected,'Capture source changed: '+name)
    packages={name:importlib.metadata.version(name) for name in identity['packages']}
    require(packages==identity['packages'],'GPU environment package versions changed')
    require(archive['url']==OFFICIAL_URL and archive['selected_images']==5000,'Official image source/count changed')
    wanted={r['image'] for r in train}
    require(set(archive['members'])==wanted==set(identity['image_sha256']),'Original training membership changed')
    require({p.name for p in (root/'data/train-images').glob('*.jpg')}==wanted,'Training image filenames changed')
    require(prepared['images']==5000 and prepared['all_readable_and_crc_verified'],'Image preparation incomplete')
    require(json.loads((root/'inputs/train-only.json').read_text())['rows']==train,'Exported training split changed')
    require(json.loads((root/'inputs/dev-only.json').read_text())==payload,'Exported development split changed')
    for row in train+dev:
        require(all(isinstance(a.get('answer'),str) and a['answer'].strip() for a in row['answers']), 'Invalid original reference')
    scorer=OfficialScorer(payload['official_rules'])
    training=[]; capture_hashes=[]; image_hashes=[]
    forbidden={'answers','answerable','answer_type','baseline_vqa_score','final_vqa_score','score_delta'}
    for index,row in enumerate(train):
        name=row['image']; info=json.loads((root/'data/image-integrity'/(name+'.json')).read_text())
        data=(root/'data/train-images'/name).read_bytes()
        actual=hashlib.sha256(data).hexdigest()
        require(actual==info['sha256']==identity['image_sha256'][name],'Training image checksum changed')
        require(len(data)==info['bytes']==archive['members'][name]['bytes'],'Training image size changed')
        require(zlib.crc32(data)&0xffffffff==info['crc32']==archive['members'][name]['crc32'],'Training image CRC changed')
        require(info['source']==OFFICIAL_URL and info['image']==name,'Training image provenance changed')
        path=root/'captures/train'/f'{index:04d}.json'
        record=checked_entry(path,row,actual)
        prediction=checked_entry(root/'captures/blip'/f'{index:04d}.json',row,actual)
        require(record['baseline_answer']==prediction['baseline_answer'],'Cached BLIP candidate changed')
        require(not forbidden.intersection(record),'Annotations/scores present in inference capture')
        expected=features(record,record['image_quality'],scorer.normalize)
        require(record['features']==expected,'Training inference features changed')
        poisoned={**record,'answers':[{'answer':'SENTINEL_LABEL'}]*10,'answerable':1-row['answerable'],
                  'answer_type':'SENTINEL','baseline_vqa_score':1,'final_vqa_score':0,'score_delta':999}
        require(features(poisoned,record['image_quality'],scorer.normalize)==expected,'Ground-truth leakage into features')
        replay=candidate(record)
        for key in ('final_answer','decision','decision_reason','answer_source','supporting_evidence_ids','candidate_trace'):
            require(record[key]==replay[key],'Frozen OCR decision changed: '+key)
        training.append(record);capture_hashes.append(sha(path));image_hashes.append(actual)
        if (index+1)%1000==0:
            print('AUDIT_TRAIN',index+1,'/5000 image CRC/SHA, captures, policy and features verified',flush=True)
    verification=json.loads((root/'captures/training-verification.json').read_text())
    require(verification['examples']==5000 and verification['capture_digest']==digest(capture_hashes),'Training capture digest changed')
    table=json.loads((root/'training-features.json').read_text())
    require(table['rows']==[{'image':r['image'],'features':r['features']} for r in training],'Fitted feature table differs from captures')
    require(gate.metadata['fitted_ids_sha256']==digest([r['image'] for r in train]),'Fitted original training IDs changed')
    x=np.asarray([[r['features'][name] for name in gate.names] for r in training])
    active=np.asarray([bool(r['final_answer'].strip()) and scorer.normalize(r['final_answer'])!='unanswerable' for r in training])
    target=np.asarray([int(row['answerable']==1 and scorer.score(record['final_answer'],row['answers'])>=1-1e-10)
                       for row,record in zip(train,training)])
    for name,values,labels in [('answerability',x,[r['answerable'] for r in train]),
                               ('reliability',x[active],target[active].tolist())]:
        model=gate.models[name]
        require(model is not None,'Required trained head unavailable')
        scale=model.calibrated_classifiers_[0].estimator.named_steps['scale']
        require(np.allclose(scale.mean_,values.mean(axis=0),rtol=1e-10,atol=1e-10),'Scaler was not fitted on original training features')
        require(np.allclose(scale.var_,values.var(axis=0),rtol=1e-9,atol=1e-9),'Scaler variance differs from original training features')
        require(gate.metadata[name]['class_counts']=={str(k):v for k,v in Counter(labels).items()},'Training targets changed')
    require(int(active.sum())==gate.metadata['reliability_examples'],'Reliability training subset changed')
    originals=json.loads(PHASE2_RECORDS.read_text())
    records=json.loads((root/'results/selected_records.json').read_text())
    require(len(originals)==len(records)==len(dev)==1000,'Exact development count required')
    saved_features=[]
    for index,(original,row,record) in enumerate(zip(originals,dev,records)):
        reference_identity(original,row);reference_identity(record,row)
        saved=json.loads((root/'dev-features'/f'{index:04d}.json').read_text())
        require(saved['image']==row['image'] and saved['phase2_record_sha256']==digest(original),'Development feature identity changed')
        require(saved['image_sha256']==sha(DEV_IMAGES/row['image']),'Development image changed')
        require(record['baseline_answer']==original['baseline_answer'] and record['phase2_final_answer']==original['final_answer'],'Saved baseline changed')
        for key in ('ocr_evidence','selected_evidence'):
            require(record[key]==original[key],'Development cached evidence changed')
        saved_features.append(saved['features'])
    matrix=np.asarray([[r[n] for n in gate.names] for r in saved_features])
    pa=gate.models['answerability'].predict_proba(matrix)[:,1]
    original_answered=np.asarray([bool(r['final_answer'].strip()) and scorer.normalize(r['final_answer'])!='unanswerable' for r in originals])
    pr=np.full(1000,np.nan);pr[original_answered]=gate.models['reliability'].predict_proba(matrix[original_answered])[:,1]
    baseline_scores=np.asarray([scorer.score(r['final_answer'],r['answers']) for r in originals])
    require(abs(baseline_scores.mean()*100-59.15)<1e-8,'Frozen Phase 2 score changed')
    for config in summary['comparisons']:
        saved=json.loads((root/'results'/(config['name']+'_records.json')).read_text())
        require(len(saved)==1000,'Configuration count changed')
        scores=[];answered=[]
        for i,(row,record) in enumerate(zip(dev,saved)):
            reference_identity(record,row)
            decision=decide(originals[i]['final_answer'],float(pa[i]),float(pr[i]) if np.isfinite(pr[i]) else None,
                            config['threshold_answerable'],config['threshold_reliable'])
            score=scorer.score(decision['answer'],row['answers'])
            require(record['final_answer']==decision['answer'] and abs(score-record['final_vqa_score'])<1e-10,'Configuration replay changed')
            require(abs(record['answerability_probability']-pa[i])<1e-10,'Saved answerability prediction differs from trained head')
            if np.isfinite(pr[i]):
                require(abs(record['candidate_full_agreement_probability']-pr[i])<1e-10,'Saved reliability prediction differs from trained head')
            scores.append(score);answered.append(not decision['abstained'])
        measured=summarize(originals,scores,answered,original_answered,baseline_scores)
        measured.update({key:config[key] for key in ('name','threshold_answerable','threshold_reliable')})
        same(measured,config,'configuration '+config['name'])
    require({(c['threshold_answerable'],c['threshold_reliable']) for c in summary['threshold_sweep']}==
            {(a,r) for a in THRESHOLDS for r in (None,*THRESHOLDS)},'Predeclared threshold grid changed')
    abstain_scores=np.asarray([scorer.score('unanswerable',r['answers']) for r in originals])
    for config in summary['threshold_sweep']:
        answered=original_answered & (pa>=config['threshold_answerable'])
        if config['threshold_reliable'] is not None:
            answered &= pr>=config['threshold_reliable']
        measured=summarize(originals,np.where(answered,baseline_scores,abstain_scores),answered,original_answered,baseline_scores)
        measured.update({key:config[key] for key in ('name','threshold_answerable','threshold_reliable')})
        same(measured,config,'threshold sweep')
    chosen,_=select_candidate(summary['threshold_sweep'],summary['comparisons'][0])
    same(chosen,summary['selected'],'selected policy')
    require(policy['threshold_answerable']==chosen['threshold_answerable'] and policy['threshold_reliable']==chosen['threshold_reliable'] and policy['ocr_threshold']==.9,'Published thresholds changed')
    for i,record in enumerate(records):
        decision=decide(originals[i]['final_answer'],float(pa[i]),float(pr[i]) if np.isfinite(pr[i]) else None,
                        policy['threshold_answerable'],policy['threshold_reliable'])
        require(record['final_answer']==decision['answer'],'Selected final answer changed')
        require(abs(scorer.score(record['final_answer'],dev[i]['answers'])-record['final_vqa_score'])<1e-10,'Selected official score changed')
    logs=list((root/'execution').glob('final-regressions-*.log'))
    require(logs and '133 passed' in max(logs,key=lambda p:p.stat().st_mtime).read_text(),'Completed regression result missing')
    completed=list((root/'execution').glob('completed-*.json'))
    require(completed and json.loads(completed[-1].read_text())['complete'],'Runner did not finish')
    output=root/('completion-audit-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    standalone={}
    labels=np.asarray([r['answerable'] for r in dev])
    full=np.asarray([int(r['answerable']==1 and score>=1-1e-10) for r,score in zip(dev,baseline_scores)])
    standalone['answerability_at_selected_0_90']=confusion_metrics(labels,pa>=policy['threshold_answerable'])
    standalone['full_credit_at_selected_0_25']=confusion_metrics(full[original_answered],pr[original_answered]>=policy['threshold_reliable'])
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'original_training_images_and_captures':5000,
            'original_development_examples':1000,'all_ten_reference_answers_preserved':True,'train_only_scaler_means_and_variances_verified':True,
            'ground_truth_mutation_invariance_checked_training_examples':5000,'training_image_sha256_digest':digest(image_hashes),
            'training_capture_digest':digest(capture_hashes),'frozen_phase2_integrity':frozen,'classifiers_sha256':sha(root/'models/heads.joblib'),
            'official_vqa_scores_recomputed_all_three_configurations':True,'saved_cpu_classifier_predictions_reproduced':True,
            'new_blip_calls':0,'new_ocr_calls':0,'new_fitting_calls':0,'holdout_inspected_or_evaluated':False,
            'standalone_at_selected_thresholds':standalone,'regression_tests_passed':133,'package_versions':packages}
    write_new(output/'VERIFICATION.json',result)
    corrected_risk_plot(root,output,summary)
    write_final_report(root,output,summary,result,json.loads(completed[-1].read_text()))
    sources=('phase3_common.py','prepare_phase3_data.py','capture_phase3_features.py','train_answerability.py','evaluate_phase3_dev.py',
             'run_phase3_local.py','check_phase3.py','verify_phase3_completion.py','phase3_pipeline.py','phase3_infer.py','app_phase3.py',
             'modules/answerability_features.py','modules/answerability_gate.py','PHASE3_GUIDE.md')
    protected={str(path.relative_to(root)):sha(path) for path in (root/'results').rglob('*') if path.is_file()}
    protected.update({str(path.relative_to(root)):sha(path) for path in (root/'models').iterdir() if path.is_file()})
    protected.update({str(path.relative_to(root)):sha(path) for path in (root/'selected-policy.json',root/'PHASE3_FINAL_REPORT.md')})
    write_new(output/'PHASE3_FREEZE.json',{'policy':policy,'development_vqa_percent':summary['selected']['vqa_percent'],
             'protected_result_and_model_sha256':protected,'implementation_sha256':{name:sha(ROOT/name) for name in sources},
             'training_capture_digest':result['training_capture_digest'],'holdout_inspected_or_evaluated':False})
    print('PHASE3_COMPLETION_AUDIT_PASSED',output,flush=True)
    return output


def corrected_risk_plot(root,output,summary):
    os.environ['MPLCONFIGDIR']=str(root/'runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for family,color in [('answerability_only','#527da3'),('both_heads','#827698')]:
        values=risk_points(summary['threshold_sweep'],family)
        ax.scatter([c['answered_examples']/10 for c in values],[c['answered_subset_risk']*100 for c in values],label=family,color=color)
    base=summary['comparisons'][0];selected=summary['selected']
    ax.scatter([base['answered_examples']/10],[base['answered_subset_risk']*100],marker='*',s=180,color='black',label='Frozen Phase 2')
    ax.scatter([selected['answered_examples']/10],[selected['answered_subset_risk']*100],marker='D',s=100,color='#468d83',label='Selected Phase 3')
    ax.set(xlabel='Answered coverage (%)',ylabel='1 − returned-answer VQA agreement (%)')
    ax.legend(fontsize=9);fig.tight_layout();fig.savefig(output/'coverage_risk_verified.png',dpi=180);plt.close(fig)


def write_final_report(root,output,summary,verification,execution):
    destination=root/'PHASE3_FINAL_REPORT.md'
    require(not destination.exists(),'Existing final report preserved; do not overwrite it')
    text=(root/'results/PHASE3_FINAL_REPORT.md').read_text(encoding='utf-8').replace('](plots/','](results/plots/')
    relative=output.relative_to(root).as_posix()
    text=text.replace('(results/plots/coverage_risk.png)',f'({relative}/coverage_risk_verified.png)')
    base,answer,joint=summary['comparisons']
    text+='\n## Completion audit and practical interpretation\n\n'
    text+=f'Completed: 5,000 original training images and captures; both fitted classifiers; all 1,000 original development examples; **133 relevant regression tests passed**. A fresh verification checked every training image CRC/SHA, capture digest, frozen OCR replay, all saved feature vectors, training-only scaler means/variances and all three development configurations using the official scorer. No neural inference, OCR extraction or model fitting was repeated for this audit. The optional Gradio UI construction test passed without starting a server or inference.\n\n'
    text+=f'Runner wall time was {execution["wall_seconds"]/3600:.2f} hours, including dataset waiting and all pipeline overhead. Recorded BLIP inference averaged {summary["training_capture"]["recorded_component_seconds"]["blip"]/5000:.3f} seconds per training example. Summed component times exclude other preparation, hashing and scheduling overhead.\n\n'
    text+=f'The selected joint policy returns **{joint["answered_examples"]} answers**, compared with **{base["answered_examples"]}** for Phase 2. It adds {joint["new_abstentions"]} abstentions, including {base["TP"]-joint["TP"]} questions labelled answerable and {base["FP"]-joint["FP"]} labelled unanswerable. Answerability recall falls from {base["recall_percent"]:.2f}% to {joint["recall_percent"]:.2f}%, and final-system F1 falls from {base["f1_percent"]:.2f}% to {joint["f1_percent"]:.2f}%. Thus the measured VQA gain does not establish a better answerability F1 or better coverage.\n\n'
    text+=f'Agreement among returned answers rises from {base["answered_subset_vqa_percent"]:.2f}% to {joint["answered_subset_vqa_percent"]:.2f}%; {joint["answered_subset_zero_credit"]} of the {joint["answered_examples"]} retained replies still receive zero credit. The OCR-supported subgroup returns only {joint["subgroups"]["ocr_supported"]["answered_examples"]} answers out of 137, compared with {base["subgroups"]["ocr_supported"]["answered_examples"]} previously. Its higher overall VQA score includes reference credit for abstention; it does not demonstrate improved text-reading answer generation.\n\n'
    text+='VizWiz answerability labels and reference-answer consensus are separate signals. Some examples labelled answerable have human `unanswerable` references. Replacing a wrong reply with abstention can therefore increase VQA credit even on an answerable-labelled question. The examples and per-question scores preserve this distinction.\n\n'
    text+='### Standalone heads at their selected thresholds\n\n| Head and target | Examples | Precision % | Recall % | F1 % | TN / FP / FN / TP |\n|---|---:|---:|---:|---:|---|\n'
    for name,metrics in verification['standalone_at_selected_thresholds'].items():
        count=1000 if name.startswith('answerability') else 431
        text+=f'| {name} | {count} | {metrics["precision_percent"]:.2f} | {metrics["recall_percent"]:.2f} | {metrics["f1_percent"]:.2f} | {metrics["TN"]} / {metrics["FP"]} / {metrics["FN"]} / {metrics["TP"]} |\n'
    text+='\nThe candidate-reliability head is weak: its development average precision is 0.3792, and its diagnostic F1 at an unselected 0.50 cutoff is only 4.48%. The selected cutoff is 0.25, chosen on development data. Neither head provides a guarantee of factual correctness or visual grounding.\n\n'
    text+='### Recommended next steps\n\n'
    text+='1. Keep the selected joint policy as the documented development research configuration, with answerability cutoff 0.90, full-credit cutoff 0.25 and frozen OCR cutoff 0.90. Preserve Phase 2 as an independently selectable control.\n'
    text+=f'2. Review the 22 harmful rejections and the coverage reduction before choosing the user-facing default. The tested answerability-only alternative scores {answer["vqa_percent"]:.2f}% while returning {answer["answered_examples"]} replies; the joint gate gains another {joint["vqa_percent"]-answer["vqa_percent"]:.2f} percentage points but rejects {answer["answered_examples"]-joint["answered_examples"]} additional replies. No production-policy claim follows from development selection alone.\n'
    text+='3. Test the optional application with your own photographs. End-to-end interactive user testing and speech-output integration for the new wrapper remain separate from this completed development experiment.\n'
    text+='4. Treat recovery of existing abstentions, stronger question-specific visual support and more robust candidate calibration as future work. Freeze any revised method using development data before requesting a separate holdout evaluation. No holdout evaluation was performed or requested by this workflow.\n\n'
    text+=f'Audit evidence: [{relative}/VERIFICATION.json]({relative}/VERIFICATION.json). The corrected coverage/risk view excludes zero-answer configurations, for which conditional risk is undefined; original plots and results are preserved. The audit directory also contains `PHASE3_FREEZE.json` with result, model and implementation hashes.\n'
    with destination.open('x',encoding='utf-8') as stream:
        stream.write(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    audit(parser.parse_args().run_dir)
