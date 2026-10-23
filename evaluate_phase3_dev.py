"""Development-only cached Phase 3 replay, threshold selection and reports."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time
import uuid

from evaluate_vqa import OfficialScorer
from modules.answerability_features import features, image_quality
from modules.answerability_gate import Gate, decide
from phase3_common import checked_splits, digest, ensure_new_or_equal, PHASE2_RECORDS, ROOT, safe_run, sha, verify_freeze, write_new
from utils.image_utils import normalize_image

THRESHOLDS=(.25,.5,.75,.9,.95)
DEV_IMAGES=ROOT/'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/data/images'


def training_capture_summary(root):
    """Report actual costs from completed captures, never refit or rerun them."""
    _,train,_,_=checked_splits()
    from capture_phase3_features import checked_entry
    records=[]
    paths=[root/'captures/train'/f'{i:04d}.json' for i in range(5000)]
    verification=json.loads((root/'captures/training-verification.json').read_text())
    if verification['examples']!=5000 or verification['capture_digest']!=digest([sha(p) for p in paths]):
        raise ValueError('Completed training capture digest changed')
    for path,row in zip(paths,train):
        ledger=json.loads((root/'data/image-integrity'/(row['image']+'.json')).read_text())
        records.append(checked_entry(path,row,ledger['sha256']))
    seconds={key:sum(r['timing_seconds'].get(key,0.) for r in records)
             for key in ('image_preparation','blip','evidence_extraction','selection')}
    return {'examples':5000,'capture_digest_verified':True,'recorded_component_seconds':seconds,
            'recorded_component_total_seconds':sum(seconds.values()),
            'peak_vram_allocated_mb':max(r['peak_vram_mb'] for r in records),
            'ocr_routed_examples':sum(bool(r['ocr_supported_question']) for r in records),
            'ocr_nonempty_examples':sum(bool(r['ocr_evidence']) for r in records),
            'ocr_answer_source_examples':sum(r['answer_source']=='ocr' for r in records),
            'extraction_status_counts':{source:dict(Counter(r['extraction_status'].get(source,{}).get('status','unknown') for r in records))
                                       for source in ('ocr','object')},
            'note':'Summed recorded components, not wall time; setup, hashing and quality-feature overhead are excluded.'}


def confusion_metrics(labels, answered):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    matrix=confusion_matrix(labels,answered,labels=[0,1]).tolist()
    precision,recall,f1,_=precision_recall_fscore_support(labels,answered,average='binary',zero_division=0)
    return {'precision_percent':float(precision*100),'recall_percent':float(recall*100),'f1_percent':float(f1*100),
            'confusion_matrix_true_rows_predicted_columns_0_1':matrix,
            'TN':matrix[0][0],'FP':matrix[0][1],'FN':matrix[1][0],'TP':matrix[1][1]}


def summarize(rows, scores, answered, original_answered, original_scores):
    import numpy as np
    scores=np.asarray(scores); answered=np.asarray(answered,dtype=bool)
    changed=original_answered & ~answered
    def group(mask):
        return {'examples':int(np.sum(mask)),'vqa_percent':float(100*np.mean(scores[mask])) if np.sum(mask) else None,
                'answered_examples':int(np.sum(answered[mask]))}
    labels=np.asarray([r['answerable'] for r in rows])
    result={'examples':len(rows),'vqa_percent':float(scores.mean()*100),'abstention_percent':float(100*(~answered).mean()),
            'answered_examples':int(answered.sum()),'new_abstentions':int(changed.sum()),
            'beneficial_changes':int(np.sum(changed & (scores>original_scores+1e-10))),
            'harmful_changes':int(np.sum(changed & (scores<original_scores-1e-10))),
            'neutral_changes':int(np.sum(changed & (np.abs(scores-original_scores)<=1e-10))),
            'unchanged_answers':int(len(rows)-changed.sum()),
            'answered_subset_vqa_percent':float(100*scores[answered].mean()) if answered.any() else None,
            'answered_subset_zero_credit':int(np.sum(answered & (scores<=1e-10))),
            'answered_subset_partial_credit':int(np.sum(answered & (scores>1e-10) & (scores<1-1e-10))),
            **confusion_metrics(labels,answered)}
    result['answered_subset_risk']=1-result['answered_subset_vqa_percent']/100 if answered.any() else 1.
    result['subgroups']={'answerable':group(labels==1),'unanswerable':group(labels==0),
                        'ocr_supported':group(np.asarray([r['ocr_supported_question'] for r in rows],dtype=bool))}
    return result


def select_candidate(candidates, baseline):
    """Predeclared selection: non-worse VQA and strictly lower answered risk."""
    feasible=[r for r in candidates if r['vqa_percent']>=baseline['vqa_percent']-1e-9 and
              r['answered_subset_risk']<baseline['answered_subset_risk']-1e-10 and r['new_abstentions']>0]
    key=lambda r:(r['vqa_percent'],-r['answered_subset_risk'],-r['new_abstentions'])
    return (max(feasible,key=key) if feasible else None),max(candidates,key=key) if candidates else None


def evaluate(run_dir):
    import numpy as np
    from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,precision_recall_curve
    root=safe_run(run_dir); verify_freeze()
    os.environ['MPLCONFIGDIR']=str(root/'runtime/matplotlib')
    _,_,dev,payload=checked_splits()
    records=json.loads(PHASE2_RECORDS.read_text())
    if len(records)!=1000 or any(r['image']!=s['image'] or r['question']!=s['question'] or r['answers']!=s['answers'] or r['answerable']!=s['answerable'] for r,s in zip(records,dev)):
        raise ValueError('Phase 2 records differ from original development rows')
    destination=root/'results'
    if (destination/'PHASE3_FINAL_REPORT.md').exists():
        print('Completed results preserved; verify final hashes instead of rerunning evaluation',flush=True);return
    if destination.exists(): raise FileExistsError('Existing result destination preserved; use a separate result version')
    output=root/'evaluation-attempts'/uuid.uuid4().hex
    output.mkdir(parents=True,exist_ok=False)
    scorer=OfficialScorer(payload['official_rules']); gate=Gate(root/'models')
    dataset=[]; quality_seconds=0.
    for i,r in enumerate(records):
        fpath=root/'dev-features'/f'{i:04d}.json'
        fpath.parent.mkdir(exist_ok=True)
        if fpath.exists():
            saved=json.loads(fpath.read_text())
            if saved['image']!=r['image'] or saved['phase2_record_sha256']!=digest(r) or saved['image_sha256']!=sha(DEV_IMAGES/r['image']):
                raise ValueError('Cached development features changed')
        else:
            began=time.perf_counter()
            image=normalize_image(DEV_IMAGES/r['image'],thumbnail=False)
            quality=image_quality(image); del image
            value=features(r,quality,scorer.normalize)
            saved={'image':r['image'],'phase2_record_sha256':digest(r),'image_sha256':sha(DEV_IMAGES/r['image']),
                   'features':value,'feature_seconds':time.perf_counter()-began}
            write_new(fpath,saved)
        if sorted(saved['features'])!=gate.names: raise ValueError('Training/development feature schemas differ')
        dataset.append(saved);quality_seconds+=saved['feature_seconds']
    x=np.asarray([[r['features'][n] for n in gate.names] for r in dataset])
    labels=np.asarray([r['answerable'] for r in records])
    base_scores=np.asarray([scorer.score(r['final_answer'],r['answers']) for r in records])
    if abs(base_scores.mean()*100-59.15)>1e-8: raise ValueError('Frozen baseline does not reproduce 59.15%')
    abstain_scores=np.asarray([scorer.score('unanswerable',r['answers']) for r in records])
    original_answered=np.asarray([scorer.normalize(r['final_answer'])!='unanswerable' for r in records])
    began=time.perf_counter(); pa=gate.models['answerability'].predict_proba(x)[:,1]
    pr=np.full(1000,np.nan)
    if gate.models.get('reliability') is not None: pr[original_answered]=gate.models['reliability'].predict_proba(x[original_answered])[:,1]
    probability_seconds=time.perf_counter()-began
    per_query_times=[]
    for i in range(20):
        began=time.perf_counter(); query_pa,query_pr=gate.predict(dataset[i]['features'],records[i]['final_answer'])
        per_query_times.append(time.perf_counter()-began)
        if abs(query_pa-pa[i])>1e-8 or (query_pr is not None and abs(query_pr-pr[i])>1e-8):
            raise ValueError('Single-query and batched gate predictions disagree')
    baseline=summarize(records,base_scores,original_answered,original_answered,base_scores)
    baseline.update(name='Phase2_frozen',threshold_answerable=None,threshold_reliable=None)
    candidates=[]
    for ta in THRESHOLDS:
        for tr in (None,*THRESHOLDS) if gate.models.get('reliability') is not None else (None,):
            answered=original_answered & (pa>=ta)
            if tr is not None: answered &= pr>=tr
            scores=np.where(answered,base_scores,abstain_scores)
            result=summarize(records,scores,answered,original_answered,base_scores)
            result.update(name='answerability_only' if tr is None else 'both_heads',threshold_answerable=ta,threshold_reliable=tr)
            candidates.append(result)
    selected,best=select_candidate(candidates,baseline)
    chosen=selected or baseline
    answer_selected,answer_best=select_candidate([r for r in candidates if r['name']=='answerability_only'],baseline)
    joint_selected,joint_best=select_candidate([r for r in candidates if r['name']=='both_heads'],baseline)
    comparisons=[baseline]
    if answer_best: comparisons.append(answer_selected or answer_best)
    if joint_best: comparisons.append(joint_selected or joint_best)
    head_metrics={'answerability':{'average_precision':float(average_precision_score(labels,pa)),
                  'brier_score':float(brier_score_loss(labels,pa)),'log_loss':float(log_loss(labels,pa)),
                  'standalone_classifier_at_0_5':confusion_metrics(labels,pa>=.5)}}
    reliable_labels=np.asarray([int(r['answerable']==1 and score>=1-1e-10) for r,score in zip(records,base_scores)])
    if gate.models.get('reliability') is not None:
        head_metrics['candidate_full_credit']={'examples':int(original_answered.sum()),
                 'average_precision':float(average_precision_score(reliable_labels[original_answered],pr[original_answered])),
                 'brier_score':float(brier_score_loss(reliable_labels[original_answered],pr[original_answered])),
                 'log_loss':float(log_loss(reliable_labels[original_answered],pr[original_answered],labels=[0,1])),
                 'standalone_classifier_at_0_5':confusion_metrics(reliable_labels[original_answered],pr[original_answered]>=.5)}
    selected_records=[]; corrected=[]; worsened=[]; neutral=[]
    decision_begin=time.perf_counter()
    for i,r in enumerate(records):
        decision=decide(r['final_answer'],float(pa[i]),float(pr[i]) if np.isfinite(pr[i]) else None,
                        chosen['threshold_answerable'],chosen['threshold_reliable'])
        score=scorer.score(decision['answer'],r['answers'])
        item={**r,'phase2_final_answer':r['final_answer'],'phase2_vqa_score':float(base_scores[i]),
              'phase2_answer_source':r['answer_source'],'phase2_decision_reason':r['decision_reason'],
              'phase2_supporting_evidence_ids':r['supporting_evidence_ids'],
              'final_answer':decision['answer'],'final_vqa_score':score,'gate_reason':decision['reason'],
              'answer_source':'abstention' if decision['abstained'] else r['answer_source'],
              'supporting_evidence_ids':[] if decision['abstained'] else r['supporting_evidence_ids'],
              'answerability_probability':float(pa[i]),'candidate_full_agreement_probability':float(pr[i]) if np.isfinite(pr[i]) else None,
              'new_gate_abstention':bool(original_answered[i] and decision['abstained']),'phase3_score_delta':float(score-base_scores[i])}
        selected_records.append(item)
        if item['new_gate_abstention']:
            (corrected if item['phase3_score_delta']>1e-10 else worsened if item['phase3_score_delta']<-1e-10 else neutral).append(item)
    decision_seconds=time.perf_counter()-decision_begin
    if abs(sum(r['final_vqa_score'] for r in selected_records)/10-chosen['vqa_percent'])>1e-8:
        raise ValueError('Vectorized threshold replay and official final scoring differ')
    summary={'purpose':'development tuning only; no holdout inference or inspection','dev_examples':1000,
             'rows_sha256':payload['rows_sha256'],'source_manifest_sha256':payload['source_manifest_sha256'],
             'frozen_phase2_vqa_percent':59.15,'frozen_ocr_threshold':.9,'threshold_grid':list(THRESHOLDS),
             'selection_rule':'VQA >= frozen baseline, strictly lower answered-subset risk, then highest VQA / lower risk / fewer added abstentions',
             'selected':chosen,'qualified_phase3_candidate':selected is not None,'best_experimental_candidate':best,
             'comparisons':comparisons,'heads':head_metrics,'threshold_sweep':candidates,
             'latency':{'dev_image_quality_total_seconds':quality_seconds,'batch_probability_seconds':probability_seconds,
                        'single_query_probability_mean_seconds_20_examples':float(np.mean(per_query_times)),
                        'decision_and_official_scoring_total_seconds':decision_seconds,
                        'historical_phase2_mean_seconds':sum(r['processing_seconds'] for r in records)/1000,
                        'note':'New CPU gate work measured separately. Historical Phase 2 inference/extraction timings were reused, not rerun.'},
             'holdout_accessed':False,'new_blip_calls':0,'new_ocr_calls':0,'fitted_split':'train',
             'model_metadata':gate.metadata,'phase2_records_sha256':sha(PHASE2_RECORDS)}
    summary['training_capture']=training_capture_summary(root)
    for filename,data in [('summary.json',summary),('selected_records.json',selected_records),('corrected_examples.json',corrected),('worsened_examples.json',worsened),('neutral_changes.json',neutral)]: write_new(output/filename,data)
    # Preserve per-example predictions for both experimental families even
    # when neither meets the acceptance rule and Phase 2 remains selected.
    for configuration in comparisons:
        per_example=[]
        for i,r in enumerate(records):
            decision=decide(r['final_answer'],float(pa[i]),float(pr[i]) if np.isfinite(pr[i]) else None,
                            configuration['threshold_answerable'],configuration['threshold_reliable'])
            score=float(abstain_scores[i] if decision['abstained'] else base_scores[i])
            per_example.append({'image':r['image'],'question':r['question'],'answers':r['answers'],'answerable':r['answerable'],
                'blip_answer':r['baseline_answer'],'phase2_answer':r['final_answer'],'final_answer':decision['answer'],
                'gate_reason':decision['reason'],'answerability_probability':float(pa[i]),
                'candidate_full_agreement_probability':float(pr[i]) if np.isfinite(pr[i]) else None,
                'phase2_vqa_score':float(base_scores[i]),'final_vqa_score':score,'delta':float(score-base_scores[i]),
                'new_abstention':bool(original_answered[i] and decision['abstained']),
                'phase2_supporting_evidence_ids':r['supporting_evidence_ids']})
        write_new(output/(configuration['name']+'_records.json'),per_example)
    with (output/'comparison.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['name','threshold_answerable','threshold_reliable','vqa_percent','precision_percent','recall_percent','f1_percent','abstention_percent','new_abstentions','beneficial_changes','harmful_changes','neutral_changes','answered_subset_vqa_percent']
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows({key:r[key] for key in fields} for r in comparisons)
    ensure_new_or_equal(root/'selected-policy.json',{'configuration':chosen['name'],'threshold_answerable':chosen['threshold_answerable'],
                 'threshold_reliable':chosen['threshold_reliable'],'development_selected':True,'ocr_threshold':.9,'qualified_phase3_candidate':selected is not None})
    plot_report(output,summary,records,pa,pr,labels,original_answered,reliable_labels)
    verified=verify_freeze(full=True)
    write_new(output/'VERIFICATION.json',{'frozen_integrity':verified,'original_dev_ids_and_all_ten_references_match':True,
               'original_training_fit_only':True,'holdout_accessed':False,'new_blip_calls':0,'new_ocr_calls':0})
    report(output,summary,corrected,worsened,neutral)
    if root not in output.resolve().parents or root not in destination.resolve().parents or destination.exists():
        raise ValueError('Result publication target changed; attempt preserved')
    output.rename(destination)
    print('PHASE3_EVALUATION_COMPLETE',json.dumps({'selected':chosen['name'],'vqa_percent':chosen['vqa_percent'],'new_abstentions':chosen['new_abstentions'],'beneficial':chosen['beneficial_changes'],'harmful':chosen['harmful_changes']}),flush=True)


def plot_report(output,summary,records,pa,pr,labels,active,reliable_labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve
    plots=output/'plots';plots.mkdir()
    configs=summary['comparisons']; names=[c['name'].replace('_','\n') for c in configs]
    fig,ax=plt.subplots(figsize=(8,4)); bars=ax.bar(names,[c['vqa_percent'] for c in configs],color=['#527da3','#72908d','#827698'][:len(configs)])
    ax.set_ylabel('Official development VQA agreement (%)'); ax.set_ylim(0,100)
    for b,c in zip(bars,configs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{c["vqa_percent"]:.2f}%',ha='center')
    fig.tight_layout();fig.savefig(plots/'vqa_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,len(configs),figsize=(4*len(configs),4),squeeze=False)
    for ax,c in zip(axes[0],configs):
        matrix=np.asarray(c['confusion_matrix_true_rows_predicted_columns_0_1']);ax.imshow(matrix,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        ax.set_xticks([0,1],['Abstain','Answer']);ax.set_yticks([0,1],['Unanswerable','Answerable']);ax.set_xlabel('System decision');ax.set_ylabel('Ground-truth label');ax.set_title(c['name'])
    fig.tight_layout();fig.savefig(plots/'confusion_matrices.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    precision,recall,_=precision_recall_curve(labels,pa);axes[0].plot(recall,precision,label='Answerability')
    true,pred=calibration_curve(labels,pa,n_bins=8,strategy='quantile');axes[1].plot(pred,true,'o-',label='Answerability')
    if np.isfinite(pr[active]).all():
        p,r,_=precision_recall_curve(reliable_labels[active],pr[active]);axes[0].plot(r,p,label='Candidate full credit')
        true,pred=calibration_curve(reliable_labels[active],pr[active],n_bins=8,strategy='quantile');axes[1].plot(pred,true,'o-',label='Candidate full credit')
    axes[0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));axes[0].legend()
    axes[1].plot([0,1],[0,1],'k--');axes[1].set(xlabel='Mean predicted probability',ylabel='Observed target fraction',xlim=(0,1),ylim=(0,1));axes[1].legend()
    fig.tight_layout();fig.savefig(plots/'precision_recall_calibration.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for family,color in [('answerability_only','#527da3'),('both_heads','#827698')]:
        values=[c for c in summary['threshold_sweep'] if c['name']==family]
        if values: ax.scatter([c['answered_examples']/1000*100 for c in values],[c['answered_subset_risk']*100 for c in values],label=family,color=color)
    baseline=summary['comparisons'][0];ax.scatter([baseline['answered_examples']/10],[baseline['answered_subset_risk']*100],marker='*',s=160,label='Frozen Phase 2',color='black')
    ax.set(xlabel='Answered coverage (%)',ylabel='1 − answered-subset official VQA agreement (%)');ax.legend();fig.tight_layout();fig.savefig(plots/'coverage_risk.png',dpi=180);plt.close(fig)


def report(output,s,corrected,worsened,neutral):
    chosen=s['selected']; lines=['# VisionBridge Phase 3 final development report','',
    'This report contains actual development-set results. No holdout annotations, images or predictions were inspected or evaluated.','',
    f'Frozen Phase 2: **59.15%** official VQA agreement, OCR threshold **0.90**. Selected configuration: **{chosen["name"]}**, scoring **{chosen["vqa_percent"]:.2f}%** on the same original 1,000 development examples.',
    f'Change: **{chosen["vqa_percent"]-59.15:+.2f} percentage points**. This is threshold selection on development data, not independent or holdout accuracy.','',
    '| Configuration | Gate thresholds A / R | VQA % | Precision % | Recall % | F1 % | Abstention % | Added abstentions | Beneficial / harmful / neutral |',
    '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for c in s['comparisons']:
        lines.append(f'| {c["name"]} | {c["threshold_answerable"]} / {c["threshold_reliable"]} | {c["vqa_percent"]:.2f} | {c["precision_percent"]:.2f} | {c["recall_percent"]:.2f} | {c["f1_percent"]:.2f} | {c["abstention_percent"]:.2f} | {c["new_abstentions"]} | {c["beneficial_changes"]} / {c["harmful_changes"]} / {c["neutral_changes"]} |')
    lines+=['','Head rows show the best selected experimental operating point for that family. Only the explicitly named selected configuration is accepted; if no point satisfies the acceptance rule, Phase 2 remains selected.']
    lines+=['','A denotes the learned answerability threshold; R denotes the learned candidate full-credit threshold. These gate thresholds are separate from the unchanged OCR confidence threshold of 0.90.']
    lines+=['','Precision/recall/F1 above measure actual return-versus-abstain decisions, with answerable as the positive class. They do not measure answer correctness.','',
    '## Data and fitting','',f'Both learned heads and their scalers/sigmoid calibrators used original-training data only. Answerability used exactly 5,000 rows; candidate reliability used {s["model_metadata"]["reliability_examples"]} non-abstaining training candidates. Targets were answerability labels and full official reference agreement respectively. No labels, answers, answer types or VQA scores entered inference features.',
    'Regularization C was selected by training cross-validation. Development selected from the predeclared gate grid 0.25, 0.50, 0.75, 0.90, 0.95. The OCR policy stayed unchanged.','',
    '## Selected-policy subgroup results','', '| Subgroup | Examples | Phase 2 VQA % | Selected VQA % | Change (points) | Answered |','|---|---:|---:|---:|---:|---:|']
    for name,g in chosen['subgroups'].items():
        before=s['comparisons'][0]['subgroups'][name]['vqa_percent']
        lines.append(f'| {name} | {g["examples"]} | {before:.2f} | {g["vqa_percent"]:.2f} | {g["vqa_percent"]-before:+.2f} | {g["answered_examples"]} |')
    lines+=['',f'Both heads use {len(s["model_metadata"]["feature_names"])} fixed numerical inference features.']
    for name in ('answerability','reliability'):
        model=s['model_metadata'][name]
        if 'unavailable' in model:
            lines.append(f'- {name}: unavailable — {model["unavailable"]}.')
        else:
            lines.append(f'- {name}: selected C = {model["C"]:g}; training target counts {model["class_counts"]}; grid search plus sigmoid calibration took {model["fit_seconds"]:.2f} seconds.')
    lines+=['','## Probability diagnostics','', 'These scores estimate their specified targets, not guaranteed correctness or direct visual grounding.']
    for name,m in s['heads'].items(): lines.append(f'- {name}: average precision {m["average_precision"]:.4f}; Brier score {m["brier_score"]:.4f}; log loss {m["log_loss"]:.4f}.')
    lines+=['','## Confusion matrix for the selected answer/abstain policy','',
            '| True label | Abstain | Answer |','|---|---:|---:|',
            f'| Unanswerable | {chosen["TN"]} | {chosen["FP"]} |',
            f'| Answerable | {chosen["FN"]} | {chosen["TP"]} |',
            '', 'The standalone answerability classifier and the final system have different confusion matrices: the final gate also preserves all existing Phase 2 abstentions. Both are included in `summary.json`.']
    lines+=['','## Error analysis','',f'The selected policy added {chosen["new_abstentions"]} abstentions: {len(corrected)} improved reference agreement, {len(worsened)} reduced it, and {len(neutral)} left it unchanged. {chosen["answered_subset_zero_credit"]} retained answers still earned zero credit; {chosen["answered_subset_partial_credit"]} earned partial credit.','',
    'The answerability/reliability heads cannot restore answers to the 288 answerable cases already rejected by BLIP/Phase 2. Incorrect retained answers remain a separate problem from the VizWiz answerability label. No unsupported-answer claim is made solely from lack of OCR or object evidence.','']
    for heading,examples in [('Beneficial abstentions',corrected),('Harmful abstentions',worsened)]:
        lines+=['### '+heading,'']
        if not examples:lines+=['None for the selected policy.','']
        for r in examples[:5]:
            answers=Counter(a['answer'] for a in r['answers'])
            lines.append(f'- **{r["image"]}** — {r["question"]} Candidate: `{r["phase2_final_answer"]}` → `{r["final_answer"]}`. Official score {r["phase2_vqa_score"]:.2f} → {r["final_vqa_score"]:.2f}; references {dict(answers)}. Reason: {r["gate_reason"]}')
        lines+=['']
    lines+=['## Runtime and preservation','',f'New development image-quality work: {s["latency"]["dev_image_quality_total_seconds"]:.2f} seconds total. Probability prediction: {s["latency"]["batch_probability_seconds"]:.4f} seconds for 1,000 rows. Decision plus final official scoring: {s["latency"]["decision_and_official_scoring_total_seconds"]:.2f} seconds. Historical Phase 2 mean latency: {s["latency"]["historical_phase2_mean_seconds"]:.3f} seconds. These components are not a newly timed end-to-end GPU evaluation.',
    'All frozen source/checkpoint/artifact hashes were verified. No Phase 1/2 files or original environment were changed, and no holdout was used. Training captures, image CRC/SHA verification, models, threshold sweep and per-example results are stored separately in this run. No commits or pushes were performed.','',
    '## Limitations and recommended next steps','',
    '- BLIP v2 was fine-tuned on the same 5,000 training rows; its cached training predictions may be optimistic. Gate cross-validation and sigmoid calibration do not remove this base-model bias. Development reliability plots must be read with this limitation.',
    '- Development has already selected earlier models/policies and now these gate thresholds. The selected score is a tuning result; paired changes and subgroup tradeoffs are shown instead of claiming generalization.',
    '- Full-reference agreement is a conservative reliability target. Partial-credit answers are treated as negative during reliability fitting. It is not a learned verification of question-relevant visual evidence.',
    '- Both gates only abstain; they do not improve the content of an incorrect answer or recover an existing abstention.',
    '- Keep the frozen Phase 2 default if no candidate meets the predeclared acceptance rule. Use Phase 3 only as an explicitly selected option; inspect harmful development rejections and calibration before deployment.',
    '- Freeze the chosen Phase 3 configuration before requesting any separate holdout evaluation. No holdout run is authorized or performed here.','',
    '## Artifacts','', '[Comparison graph](plots/vqa_comparison.png), [confusion matrices](plots/confusion_matrices.png), [precision/recall and calibration](plots/precision_recall_calibration.png), [coverage/risk](plots/coverage_risk.png).',
    'See `summary.json`, `comparison.csv`, `selected_records.json`, `corrected_examples.json`, `worsened_examples.json`, `neutral_changes.json`, and `VERIFICATION.json`.','',
    'Regression test results and full execution timing are recorded by the local runner alongside this report.']
    t=s['training_capture']; costs=t['recorded_component_seconds']
    lines+=['','## Training feature-extraction cost','',
            f'All {t["examples"]} original training captures passed their saved digest checks. Recorded BLIP CUDA inference: {costs["blip"]/60:.2f} minutes; CPU evidence extraction: {costs["evidence_extraction"]/60:.2f} minutes. Peak allocated GPU memory: {t["peak_vram_allocated_mb"]:.1f} MiB. These are summed component times, not total wall time.',
            f'OCR routing covered {t["ocr_routed_examples"]} training questions; {t["ocr_nonempty_examples"]} had nonempty OCR; {t["ocr_answer_source_examples"]} candidates used OCR as their answer source.',
            'The available features are image quality, question intent, candidate/BLIP agreement, abstention indicators, extraction status and OCR confidence/geometry/matching. No BLIP token probabilities were available in the preserved development caches, so neither head uses them.']
    with (output/'PHASE3_FINAL_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();evaluate(args.run_dir)
