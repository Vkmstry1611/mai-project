"""Regression checks, integrity and pre-holdout gate without neural inference."""
import argparse
import json
import py_compile
import tempfile
import time
from phase3_common import ROOT, write_new
from phase4_policy import run_path, verify_previous
from visionbridge.check_phase3 import TESTS


def check(path):
    import pytest
    root=run_path(path);folder=root/'tests';folder.mkdir(exist_ok=True)
    stamp=str(time.time_ns());temporary=ROOT/('.pytest-phase4-'+stamp);temporary.mkdir();tempfile.tempdir=str(temporary)
    for name in ('phase4_policy','phase4_holdout','phase4_pipeline','phase4_infer','app_phase4','phase4_gpu_infer','launch_phase4','check_phase4','phase4_report','verify_phase4','verify_phase4_app_replay','run_phase4_completion','app'):
        py_compile.compile(str(ROOT/(name+'.py')),doraise=True)
    code=pytest.main([str(ROOT/'tests'/(t+'.py')) for t in (*TESTS,'test_phase3_completion','test_phase4_policy','test_phase4_gpu_bootstrap')]+
        ['-q','--basetemp',str(folder/('regression-'+stamp)),'--override-ini','cache_dir='+str(folder/'pytest-cache')])
    if code:raise RuntimeError('Regressions failed')
    verify_previous()
    name='PRE_HOLDOUT_TESTS.json' if not (root/'PRE_HOLDOUT_TESTS.json').exists() else 'REGRESSIONS-'+stamp+'.json'
    write_new(root/name,{'exit_code':int(code),'timestamp_ns':stamp,'module_count':len(TESTS)+3,'no_neural_inference':True,'preserved_phase2_and_phase3_hashes_verified':True})
    print('PHASE4_REGRESSIONS_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();check(a.run_dir)
