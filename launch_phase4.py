"""Reuse the existing app/UI, routing model requests through CUDA bootstrap."""
import argparse
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


class GPUWorker:
    """Inject only this controller's subprocess dependency, not the global API."""
    @staticmethod
    def run(command,**options):
        command=list(command)
        if len(command)<2 or Path(command[1]).resolve()!=ROOT/'phase4_infer.py':
            raise ValueError('Unexpected model worker; no fallback')
        command[1]=str(ROOT/'phase4_gpu_infer.py')
        return subprocess.run(command,**options)


def install_worker():
    import app_phase4
    app_phase4.subprocess=GPUWorker()
    return app_phase4


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    if Path(sys.prefix).name!='.venv':raise RuntimeError('Use the existing .venv for Gradio; GPU inference stays in .venv-gpu')
    root=Path(a.run_dir).resolve()
    if (ROOT/'reports/phase4_policy') not in root.parents or not (root/'APPLICATION_FREEZE.json').exists():raise ValueError('Select the completed frozen application integration')
    install_worker()
    sys.argv=[str(ROOT/'app.py'),'--phase4-run',str(root)]
    runpy.run_path(str(ROOT/'app.py'),run_name='__main__')
