"""Repeatable lightweight regressions and final integrity verification."""
import argparse
import json
from pathlib import Path
import py_compile
import tempfile
import time

from phase3_common import ROOT,safe_run,verify_freeze,write_new

TESTS=('test_answerability_features','test_answerability_gate','test_phase3_dev','test_phase3_pipeline','test_ocr_targeted','test_phase2_replay',
       'test_phase2_ocr_coverage','test_phase2','test_phase2_dev','test_phase2_local_setup','test_training_workflow','test_vqa_checkpoint_loading')


def check(run_dir):
    import pytest
    root=safe_run(run_dir); folder=root/'tests';folder.mkdir(exist_ok=True)
    stamp=str(time.time_ns());temp=ROOT/('.pytest-phase3-temp-'+stamp);temp.mkdir();tempfile.tempdir=str(temp)
    modules=('phase3_common','prepare_phase3_data','capture_phase3_features','train_answerability','evaluate_phase3_dev','run_phase3_local','check_phase3','phase3_pipeline','phase3_infer','app_phase3')
    for module in modules:py_compile.compile(str(ROOT/(module+'.py')),doraise=True)
    args=[str(ROOT/'tests'/(name+'.py')) for name in TESTS]+['-q','--basetemp',str(folder/('final-'+stamp)),
          '--override-ini','cache_dir='+str(folder/'pytest-cache')]
    code=pytest.main(args)
    if code:raise RuntimeError(f'Regression tests failed with exit code {code}')
    integrity=verify_freeze(full=True)
    write_new(folder/('verification-'+stamp+'.json'),{'pytest_exit_code':int(code),'test_modules':list(TESTS),
                'frozen_integrity':integrity,'holdout_accessed':False})
    print('PHASE3_REGRESSIONS_AND_INTEGRITY_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();check(args.run_dir)
