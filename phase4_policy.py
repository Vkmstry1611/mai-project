"""Cached development policy search and immutable pre-holdout protocol."""
import argparse
import csv
import json
from pathlib import Path
import math

from evaluate_phase3_dev import summarize
from evaluate_vqa import OfficialScorer
from modules.answerability_gate import decide
from phase3_common import ROOT, MANIFEST, CHECKPOINT, PHASE2_RECORDS, checked_splits, digest, sha, verify_freeze, write_new

P3 = ROOT/'reports/phase3_answerability/run-local-20261012-v1'
BASE = ROOT/'reports/phase4_policy'
RULE = {'minimum_relative_answerability_recall': .95,
        'minimum_relative_answer_coverage': .90,
        'minimum_relative_ocr_question_coverage': .90,
        'nondecreasing_vqa_and_answerability_precision': True,
        'ranking': 'highest VQA, then highest recall, then coverage, then lower thresholds'}
GRID_A = tuple(round(i/20, 2) for i in range(20))
GRID_R = (None, *tuple(round(i/20, 2) for i in range(11)))


def run_path(path):
    path=Path(path).resolve()
    if BASE not in path.parents: raise ValueError('Use a new versioned directory under reports/phase4_policy')
    return path


def verify_previous():
    verify_freeze(full=True)
    lock=json.loads(next(P3.glob('completion-audit-*/PHASE3_FREEZE.json')).read_text())
    for group,base in [('protected_result_and_model_sha256',P3),('implementation_sha256',ROOT)]:
        for name,expected in lock[group].items():
            if sha(base/name)!=expected: raise ValueError('Preserved Phase 3 artifact changed: '+name)
    return lock


def admissible(c, b):
    return (c['vqa_percent']>=b['vqa_percent']-1e-9 and
            c['precision_percent']>=b['precision_percent']-1e-9 and
            c['recall_percent']>=RULE['minimum_relative_answerability_recall']*b['recall_percent']-1e-9 and
            c['answered_examples']>=RULE['minimum_relative_answer_coverage']*b['answered_examples'] and
            c['subgroups']['ocr_supported']['answered_examples']>=RULE['minimum_relative_ocr_question_coverage']*b['subgroups']['ocr_supported']['answered_examples'])


def pareto(candidates):
    axes=('vqa_percent','recall_percent','precision_percent','answered_examples')
    return [c for c in candidates if not any(
        all(d[k]>=c[k]-1e-9 for k in axes) and any(d[k]>c[k]+1e-9 for k in axes)
        for d in candidates)]


def replay(rows, scorer, pa, pr, ta, tr):
    import numpy as np
    answers=[decide(r['final_answer'],a,p,ta,tr) for r,a,p in zip(rows,pa,pr)]
    original=np.array([scorer.normalize(r['final_answer'])!='unanswerable' for r in rows])
    scores=np.array([scorer.score(r['final_answer'],r['answers']) for r in rows])
    final=np.array([scorer.score(d['answer'],r['answers']) for r,d in zip(rows,answers)])
    answered=np.array([not d['abstained'] for d in answers])
    metric=summarize(rows,final,answered,original,scores)
    metric.update(threshold_answerable=ta,threshold_reliable=tr,answer_coverage_percent=100*answered.mean())
    return metric,answers,final


def optimize(path):
    root=run_path(path);root.mkdir(parents=True,exist_ok=True)
    if (root/'POLICY_FREEZE.json').exists(): raise FileExistsError('Frozen run preserved; do not optimize it again')
    verify_previous()
    _,_,dev,payload=checked_splits()
    # Record safeguards before reading probabilities or alternative metrics.
    write_new(root/'SELECTION_PLAN.json',{'development_only':True,'rule':RULE,'grid_answerable':GRID_A,'grid_reliable':GRID_R,
        'holdout_protocol':'One selected system only; original 3319 rows; no post-holdout selection, retraining or thresholds.',
        'original_protocol_permission':'COLAB_T4_GUIDE.md allows final evaluation after completed model selection; current user authorizes final holdout.'})
    rows=json.loads(PHASE2_RECORDS.read_text())
    cached=json.loads((P3/'results/both_heads_records.json').read_text())
    if len(rows)!=1000 or any(r['image']!=d['image'] or r['question']!=d['question'] or r['answers']!=d['answers'] or r['answerable']!=d['answerable'] for r,d in zip(rows,dev)):
        raise ValueError('Development membership/references changed')
    if any(c['image']!=r['image'] or c['phase2_answer']!=r['final_answer'] for c,r in zip(cached,rows)) or len(cached)!=1000:
        raise ValueError('Cached probability identity changed')
    pa=[c['answerability_probability'] for c in cached];pr=[c['candidate_full_agreement_probability'] for c in cached]
    # Reproduce saved probabilities from features and frozen heads on CPU.
    import numpy as np
    from modules.answerability_gate import Gate
    gate=Gate(P3/'models')
    vectors=[json.loads((P3/'dev-features'/f'{i:04d}.json').read_text()) for i in range(1000)]
    if any(v['image']!=r['image'] or v['phase2_record_sha256']!=digest(r) for v,r in zip(vectors,rows)): raise ValueError('Feature identity mismatch')
    x=np.asarray([[v['features'][n] for n in gate.names] for v in vectors])
    if not np.allclose(gate.models['answerability'].predict_proba(x)[:,1],pa,rtol=0,atol=1e-9):raise ValueError('Answerability probabilities changed')
    active=[i for i,p in enumerate(pr) if p is not None]
    if not np.allclose(gate.models['reliability'].predict_proba(x[active])[:,1],[pr[i] for i in active],rtol=0,atol=1e-9):raise ValueError('Reliability probabilities changed')
    scorer=OfficialScorer(payload['official_rules'])
    baseline,_,_=replay(rows,scorer,pa,pr,None,None)
    if abs(baseline['vqa_percent']-59.15)>1e-9:raise ValueError('Frozen baseline mismatch')
    # Vectorized cached scores, identical inclusive finite gates, no neural calls.
    original=np.array([scorer.normalize(r['final_answer'])!='unanswerable' for r in rows])
    base_scores=np.array([scorer.score(r['final_answer'],r['answers']) for r in rows])
    abs_scores=np.array([scorer.score('unanswerable',r['answers']) for r in rows])
    a=np.asarray(pa);p=np.asarray([float('nan') if v is None else v for v in pr])
    sweep=[]
    for ta in GRID_A:
        for tr in GRID_R:
            answered=original & (a>=ta)
            if tr is not None:answered &= p>=tr
            m=summarize(rows,np.where(answered,base_scores,abs_scores),answered,original,base_scores)
            m.update(threshold_answerable=ta,threshold_reliable=tr,answer_coverage_percent=float(100*answered.mean()))
            sweep.append(m)
    candidates=[baseline,*sweep]
    feasible=[c for c in candidates if admissible(c,baseline)]
    chosen=max(feasible,key=lambda c:(c['vqa_percent'],c['recall_percent'],c['answered_examples'],-(c['threshold_answerable'] or 0),-(c['threshold_reliable'] or 0)))
    names=[('Phase2',None,None),('Phase3_answerability',.9,None),('Phase3_both',.9,.25),('Phase4_balanced',chosen['threshold_answerable'],chosen['threshold_reliable'])]
    comparisons=[]
    for name,ta,tr in names:
        m,decisions,scores=replay(rows,scorer,pa,pr,ta,tr);m['name']=name;comparisons.append(m)
        if name=='Phase4_balanced':
            records=[{**r,'phase2_final_answer':r['final_answer'],'final_answer':d['answer'],'gate_reason':d['reason'],
                'answerability_probability':aa,'candidate_full_agreement_probability':pp,'phase2_vqa_score':float(bs),'final_vqa_score':float(fs),
                'new_abstention':bool(orig and d['abstained']),'score_delta':float(fs-bs)}
                for r,d,aa,pp,bs,fs,orig in zip(rows,decisions,pa,pr,base_scores,scores,original)]
            write_new(root/'development_records.json',records)
    summary={'split':'development','examples':1000,'rule':RULE,'selected':chosen,'comparisons':comparisons,'sweep':sweep,
        'pareto':pareto(candidates),'holdout_accessed':False,'new_blip_calls':0,'new_ocr_calls':0}
    write_new(root/'development_summary.json',summary)
    with (root/'comparison.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['name','threshold_answerable','threshold_reliable','vqa_percent','precision_percent','recall_percent','f1_percent','answer_coverage_percent','abstention_percent','beneficial_changes','harmful_changes','neutral_changes']
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows({k:c[k] for k in fields} for c in comparisons)
    policy={'schema':1,'development_selected':True,'threshold_answerable':chosen['threshold_answerable'],'threshold_reliable':chosen['threshold_reliable'],
        'ocr_threshold':.9,'phase3_run':str(P3),'models_sha256':sha(P3/'models/heads.joblib'),'manifest_sha256':sha(MANIFEST),
        'dev_rows_sha256':digest(dev),'selected_metrics':chosen,'selection_rule':RULE}
    write_new(root/'selected-policy.json',policy)
    print(json.dumps({'comparisons':comparisons,'pareto_configurations':len(summary['pareto'])},indent=2),flush=True)


def freeze(path):
    root=run_path(path);verify_previous()
    if (root/'POLICY_FREEZE.json').exists():return verify_policy(root)
    if not (root/'PRE_HOLDOUT_TESTS.json').exists():raise RuntimeError('Passing pre-holdout tests required')
    tests=json.loads((root/'PRE_HOLDOUT_TESTS.json').read_text())
    if tests['exit_code']!=0:raise ValueError('Tests failed')
    sources=('phase4_policy.py','phase4_holdout.py','phase4_pipeline.py','phase4_infer.py','app_phase4.py','app.py')
    from evaluate_phase2_dev import PROTOCOL
    write_new(root/'POLICY_FREEZE.json',{'policy_sha256':sha(root/'selected-policy.json'),
        'source_sha256':{name:sha(ROOT/name) for name in sources},'development_summary_sha256':sha(root/'development_summary.json'),
        'model_sha256':sha(P3/'models/heads.joblib'),'checkpoint_sha256':sha(CHECKPOINT/'model.safetensors'),
        'manifest_sha256':sha(MANIFEST),'scorer_sha256':digest(checked_splits()[3]['official_rules']),
        'protocol':PROTOCOL,'batch_size':1,'cuda':True,'ocr_cpu':True,'object_cpu':True,
        'holdout_expected_examples':3319,'holdout_evaluation_limit':1,
        'comparison_controls':'BLIP v2 and frozen Phase 2 scored from the same single capture; no alternative gate policies on holdout',
        'error_categories':'existing abstention, new false abstention, correctly rejected unanswerable, retained zero/partial/full VQA',
        'no_holdout_tuning':True})
    return verify_policy(root)


def verify_policy(path):
    root=run_path(path);lock=json.loads((root/'POLICY_FREEZE.json').read_text());policy=json.loads((root/'selected-policy.json').read_text())
    pairs=[(root/'selected-policy.json',lock['policy_sha256']),(root/'development_summary.json',lock['development_summary_sha256']),
        (P3/'models/heads.joblib',lock['model_sha256']),(CHECKPOINT/'model.safetensors',lock['checkpoint_sha256']),(MANIFEST,lock['manifest_sha256'])]
    pairs.extend((ROOT/n,h) for n,h in lock['source_sha256'].items())
    if any(sha(p)!=h for p,h in pairs):raise ValueError('Frozen Phase 4 policy, implementation or input changed')
    if policy['ocr_threshold']!=.9 or not policy['development_selected']:raise ValueError('Invalid frozen policy')
    return policy


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    freeze(args.run_dir) if args.freeze else optimize(args.run_dir)
