"""Autonomous resumable local Phase 3 execution after verified data preparation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from phase3_common import ROOT, safe_run, verify_freeze, write_new


def run(root):
    root=safe_run(root)
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu')
    root.mkdir(parents=True,exist_ok=True)
    log_dir=root/'execution';log_dir.mkdir(exist_ok=True)
    started=time.perf_counter()
    try:
        # Dataset is being fetched independently; never start inference on an
        # unverified or substituted split. Persistent ledgers make this resumable.
        while not (root/'data/prepared.json').exists():
            print('Waiting for verified exact training-image preparation',flush=True)
            time.sleep(30)
        for stage,script,args in [
            ('training-pilot','capture_phase3_features.py',['--limit','20']),
            ('training-captures','capture_phase3_features.py',['--limit','5000']),
            ('classifier-fitting','train_answerability.py',[]),
            ('development-evaluation','evaluate_phase3_dev.py',[]),
            ('final-regressions','check_phase3.py',[])]:
            stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
            logfile=log_dir/(stage+'-'+stamp+'.log')
            print(f'START {stage}: {logfile}',flush=True)
            with logfile.open('x',encoding='utf-8') as log:
                target=['-m','visionbridge.check_phase3'] if script=='check_phase3.py' else [str(ROOT/script)]
                subprocess.run([sys.executable,'-u',*target,'--run-dir',str(root),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'COMPLETE {stage}',flush=True)
        integrity=verify_freeze(full=True)
        result={'complete':True,'wall_seconds':time.perf_counter()-started,'frozen_integrity':integrity,'holdout_accessed':False}
        write_new(log_dir/('completed-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.json'),result)
        print('PHASE3_ALL_STAGES_COMPLETE',flush=True)
    except Exception:
        stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
        text=traceback.format_exc()
        with (log_dir/('BLOCKER-'+stamp+'.md')).open('x',encoding='utf-8') as f:
            f.write('# Phase 3 execution interrupted\n\n'+text+'\nCompleted images, BLIP predictions and evidence remain cached. No holdout or replacement split is permitted.\n')
        print(text,flush=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();run(args.run_dir)
