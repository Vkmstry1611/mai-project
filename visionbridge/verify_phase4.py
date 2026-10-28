"""Read-only inference audit plus new verification ledger; no neural calls."""
import argparse
import json
import math
import time
import zlib
from phase4_policy import run_path, verify_policy, verify_previous, P3
from phase4_holdout import holdout_rows
from phase3_common import sha,digest,write_new


def audit(path):
    root=run_path(path);policy=verify_policy(root);verify_previous()
    if not (root/'holdout/summary.json').exists():raise RuntimeError('Complete holdout required')
    from evaluate_vqa import OfficialScorer
    from modules.answerability_gate import Gate,decide
    from capture_phase3_features import checked_entry
    from evaluate_phase3_dev import summarize
    import numpy as np
    rows,payload=holdout_rows(root);scorer=OfficialScorer(payload['official_rules']);gate=Gate(P3/'models')
    scored=json.loads((root/'holdout/scored_records.json').read_text());summary=json.loads((root/'holdout/summary.json').read_text())
    if len(scored)!=3319:raise ValueError('Incomplete scored membership')
    records=[]
    for i,(row,saved) in enumerate(zip(rows,scored)):
        if any(saved[k]!=row[k] for k in ('image','question','answers','answerable')):raise ValueError('Original holdout references or labels changed')
        ledger=json.loads((root/'holdout/image-integrity'/(row['image']+'.json')).read_text())
        p=root/'holdout/images'/row['image'];raw=p.read_bytes()
        if len(raw)!=ledger['bytes'] or zlib.crc32(raw)&0xffffffff!=ledger['crc32'] or sha(p)!=ledger['sha256']:raise ValueError('Image integrity mismatch')
        r=checked_entry(root/'holdout/records'/f'{i:04d}.json',row,ledger['sha256'])
        if any(k in r for k in ('answers','answerable','final_vqa_score','phase2_vqa_score')):raise ValueError('Ground truth entered inference capture')
        b=checked_entry(root/'holdout/blip'/f'{i:04d}.json',row,ledger['sha256'])
        if r['baseline_answer']!=b['baseline_answer']:raise ValueError('Original BLIP capture mismatch')
        decision=decide(r['phase2_final_answer'],r['answerability_probability'],r['candidate_full_agreement_probability'],policy['threshold_answerable'],policy['threshold_reliable'])
        if decision['answer']!=r['final_answer'] or decision['reason']!=r['gate_reason']:raise ValueError('Final policy replay differs')
        if abs(scorer.score(r['final_answer'],row['answers'])-saved['final_vqa_score'])>1e-9:raise ValueError('Official score mismatch')
        if any(not math.isfinite(v) for v in r['features'].values()):raise ValueError('Nonfinite features')
        records.append(r)
    x=np.asarray([[r['features'][n] for n in gate.names] for r in records]);pa=gate.models['answerability'].predict_proba(x)[:,1]
    if not np.allclose(pa,[r['answerability_probability'] for r in records],atol=1e-9,rtol=0):raise ValueError('Cached answerability changed')
    active=[i for i,r in enumerate(records) if r['candidate_full_agreement_probability'] is not None]
    pr=gate.models['reliability'].predict_proba(x[active])[:,1]
    if not np.allclose(pr,[records[i]['candidate_full_agreement_probability'] for i in active],atol=1e-9,rtol=0):raise ValueError('Cached reliability changed')
    final=np.asarray([r['final_vqa_score'] for r in scored]);base=np.asarray([r['phase2_vqa_score'] for r in scored])
    answered=np.asarray([scorer.normalize(r['final_answer'])!='unanswerable' for r in records]);original=np.asarray([scorer.normalize(r['phase2_final_answer'])!='unanswerable' for r in records])
    computed=summarize(scored,final,answered,original,base)
    for key in ('vqa_percent','precision_percent','recall_percent','f1_percent','answered_examples','new_abstentions','beneficial_changes','harmful_changes','neutral_changes'):
        if abs(computed[key]-summary['selected'][key])>1e-9:raise ValueError('Summary differs: '+key)
    if len(list((root/'holdout/blip').glob('*.json')))!=3319 or len(list((root/'holdout/records').glob('*.json')))!=3319:raise ValueError('Capture file count changed')
    if digest([sha(root/'holdout/records'/f'{i:04d}.json') for i in range(3319)])!=summary['capture_digest']:raise ValueError('Capture aggregate mismatch')
    verify_policy(root);verify_previous()
    write_new(root/('FINAL_VERIFICATION-'+str(time.time_ns())+'.json'),{'holdout_examples':3319,'original_ids_questions_labels_ten_raw_reference_slots_verified':True,
        'train_dev_holdout_disjoint':True,'all_image_crc_sha_capture_digests_verified':True,'fixed_policy_and_official_scores_reproduced':True,
        'head_probabilities_reproduced_cpu':True,'original_checkpoints_reports_sources_unchanged':True,
        'new_blip_calls':0,'new_ocr_calls':0,'new_classifier_fits':0,'alternative_holdout_policies_tested':0})
    print('PHASE4_FINAL_AUDIT_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();audit(a.run_dir)
