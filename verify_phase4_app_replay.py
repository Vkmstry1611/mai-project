"""Cached development-only parity of application features and benchmark gates."""
from copy import deepcopy
import argparse
import json
import time

from phase4_policy import run_path, verify_policy, P3
from phase3_common import PHASE2_RECORDS, checked_splits, write_new


def check(path):
    root=run_path(path);policy=verify_policy(root)
    from phase4_pipeline import Phase4Pipeline
    from modules.answerability_gate import Gate,decide
    from evaluate_vqa import OfficialScorer
    from evaluate_phase3_dev import DEV_IMAGES
    from PIL import Image
    rows=json.loads(PHASE2_RECORDS.read_text());cached=json.loads((P3/'results/both_heads_records.json').read_text())
    # Debug coverage, never score-based selection: first 20 plus OCR-source cases.
    indices=sorted(set(range(20))|{i for i,r in enumerate(rows) if r['answer_source']=='ocr'})
    wrapper=Phase4Pipeline.__new__(Phase4Pipeline);wrapper.policy=policy;wrapper.gate=Gate(P3/'models')
    wrapper.scorer=OfficialScorer(checked_splits()[3]['official_rules'])
    class Base:
        def __init__(self,row):self.row=row
        def run(self,image,question,**kwargs):
            row=deepcopy(self.row);row['grounded']=row['answer_source']=='ocr'
            row['extraction_status']={('objects' if k=='object' else k):v for k,v in row['extraction_status'].items()}
            return row['final_answer'],None,'Cached parity only',row,image
    for i in indices:
        wrapper.base=Base(rows[i])
        with Image.open(DEV_IMAGES/rows[i]['image']) as image:
            shown,_,_,debug,_=wrapper.run(image,rows[i]['question'])
        expected=decide(rows[i]['final_answer'],cached[i]['answerability_probability'],cached[i]['candidate_full_agreement_probability'],policy['threshold_answerable'],policy['threshold_reliable'])
        if debug['final_answer']!=expected['answer'] or abs(debug['answerability_probability']-cached[i]['answerability_probability'])>1e-9:
            raise ValueError('Application and benchmark answerability differ')
        if cached[i]['candidate_full_agreement_probability'] is not None and abs(debug['candidate_full_agreement_probability']-cached[i]['candidate_full_agreement_probability'])>1e-9:
            raise ValueError('Application and benchmark reliability differ')
        if debug['phase2_final_answer']!=rows[i]['final_answer'] or debug['baseline_answer']!=rows[i]['baseline_answer']:
            raise ValueError('Original candidates lost')
        if debug['phase2_supporting_evidence_ids']!=rows[i]['supporting_evidence_ids']:raise ValueError('Original evidence IDs lost')
    write_new(root/('APP_PARITY-'+str(time.time_ns())+'.json'),{'examples':len(indices),'split':'development',
        'application_and_cached_evaluation_heads_decisions_match':True,'new_blip_calls':0,'new_ocr_calls':0})
    print('PHASE4_APP_REPLAY_PARITY_PASSED',len(indices),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();check(a.run_dir)
