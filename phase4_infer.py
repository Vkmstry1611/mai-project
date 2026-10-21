"""One personal-photo request using the frozen Phase 4 policy."""
import argparse
import json
from pathlib import Path
import sys
from phase4_pipeline import Phase4Pipeline

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--image',required=True);p.add_argument('--question',required=True)
    args=p.parse_args()
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu for inference')
    # Reserved benchmark files go through the audited evaluator, not UI requests.
    if Path(args.image).name.startswith('VizWiz_'):raise ValueError('Use personal photos here; use the frozen benchmark runner for dataset images')
    shown,_,status,debug,_=Phase4Pipeline(args.run_dir).run(Path(args.image),args.question)
    print(json.dumps({'display_answer':shown,'status':status,'details':debug},ensure_ascii=False))
