"""Application-only CUDA bootstrap over the unchanged frozen inference policy."""
import argparse
import json
import os
from pathlib import Path
import sys

from phase4_policy import run_path,verify_policy
from phase3_common import ROOT,sha


def verify_application(path):
    root=run_path(path);verify_policy(root)
    lock=json.loads((root/'APPLICATION_FREEZE.json').read_text())
    if sha(root/'selected-policy.json')!=lock['policy_sha256']:raise ValueError('Application policy changed')
    for name,expected in lock['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen application integration changed: '+name)
    return root


def prime_cuda(torch_module=None):
    """Prime BEFORE CPU YOLO, which otherwise hides CUDA via its environment."""
    if torch_module is None:import torch as torch_module
    if not torch_module.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU BLIP fallback is forbidden')
    if os.getenv('VISIONBRIDGE_LOW_MEMORY_VQA')=='1':raise RuntimeError('Keep the frozen FP32-weight/FP16-autocast protocol')
    from utils.model_loader import resolve_device
    torch_module.cuda.init()
    resolve_device.cache_clear()
    if resolve_device('cuda')!='cuda':raise RuntimeError('CUDA device resolution failed')
    torch_module.set_num_threads(4);torch_module.manual_seed(42);torch_module.cuda.manual_seed_all(42)
    return torch_module


def run(args):
    if Path(sys.prefix).name!='.venv-gpu':raise RuntimeError('Use .venv-gpu for model inference')
    root=verify_application(args.run_dir)
    if Path(args.image).name.startswith('VizWiz_'):raise ValueError('Use personal photos in the UI; use the audited runner for benchmark files')
    os.environ['HF_HUB_OFFLINE']='1'
    torch=prime_cuda()
    from phase4_pipeline import Phase4Pipeline
    shown,_,status,debug,_=Phase4Pipeline(root).run(Path(args.image),args.question)
    # Remove only the preserved Phase 2 UI's obsolete implementation-status note.
    debug['warnings']=[w for w in debug.get('warnings',[]) if 'Learned answerability is not implemented.' not in w]
    debug['gpu_runtime']={'device':torch.cuda.get_device_name(0),'cuda_initialized_before_cpu_evidence':True,
        'precision':'FP32 model weights, CUDA FP16 autocast','CPU_evidence':True}
    print(json.dumps({'display_answer':shown,'status':status,'details':debug},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--image',required=True);p.add_argument('--question',required=True)
    run(p.parse_args())
