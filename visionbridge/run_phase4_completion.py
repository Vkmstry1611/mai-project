"""Finish audits/report after the already-running single holdout execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def finish(path):
    root=Path(path).resolve()
    if (ROOT/'reports/phase4_policy') not in root.parents:raise ValueError('Invalid output')
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu')
    if (root/'COMPLETED.json').exists():print('Completed experiment preserved');return
    logs=root/'execution';logs.mkdir(exist_ok=True)
    print('Waiting for the frozen holdout summary; no extra inference will be launched.',flush=True)
    while not (root/'holdout/summary.json').exists():time.sleep(10)
    for script in ('verify_phase4.py','check_phase4.py','phase4_report.py'):
        if script=='phase4_report.py' and (root/'PHASE4_FINAL_REPORT.md').exists():continue
        stamp=str(time.time_ns());log=logs/(script[:-3]+'-'+stamp+'.log')
        print('Executing',script,flush=True)
        with log.open('x',encoding='utf-8') as stream:
            code=subprocess.run([sys.executable,'-u','-m','visionbridge.'+script.removesuffix('.py'),'--run-dir',str(root)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:
            from evaluate_phase2_dev import write_new
            write_new(logs/('failure-'+stamp+'.json'),{'script':script,'exit_code':code,'log':str(log)})
            raise RuntimeError('Completion check failed; inspect '+str(log))
    from phase4_policy import verify_policy,verify_previous
    from phase3_common import write_new,sha
    verify_policy(root);verify_previous()
    write_new(root/'COMPLETED.json',{'completed':True,'report_sha256':sha(root/'PHASE4_FINAL_REPORT.md'),
        'single_holdout_examples':3319,'no_holdout_tuning':True,'previous_artifacts_preserved':True,'no_commits_or_pushes':True})
    print('PHASE4_ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();finish(a.run_dir)
