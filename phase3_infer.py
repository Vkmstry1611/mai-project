"""Single-image Phase 3 CLI; no Gradio installation or original app changes."""
import argparse
import json
from pathlib import Path
import sys

from phase3_common import checked_splits
from phase3_pipeline import Phase3Pipeline


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--image',required=True);p.add_argument('--question',required=True)
    args=p.parse_args()
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu')
    path=Path(args.image)
    # Never run a supplied validation file outside the known development IDs
    # while this Phase 3 development-only workflow is active.
    if path.name.startswith('VizWiz_val_'):
        _,_,dev,_=checked_splits()
        if path.name not in {r['image'] for r in dev}:raise ValueError('Validation filename is outside the approved development split')
    pipeline=Phase3Pipeline(args.run_dir)
    shown,_,status,debug,_=pipeline.run(path,args.question)
    print(json.dumps({'display_answer':shown,'status':status,'details':debug},indent=2,ensure_ascii=False))
