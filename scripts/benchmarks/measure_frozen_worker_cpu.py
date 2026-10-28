"""Measure the actual Python worker CPU, not its Windows venv launcher."""
import argparse
import json
from pathlib import Path
import time


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--image',required=True)
    p.add_argument('--question',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    wall=time.perf_counter();cpu=time.process_time()
    from phase4_gpu_infer import run
    run(a)
    elapsed=time.perf_counter()-wall;cpu=time.process_time()-cpu
    import torch
    with output.open('x',encoding='utf-8') as stream:
        json.dump({'actual_worker_cpu_seconds':cpu,'actual_worker_wall_seconds':elapsed,
                   'average_cpu_core_equivalents':cpu/elapsed,'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/1024**2,
                   'gpu':torch.cuda.get_device_name(0),'scope':'Actual Python worker body, including runner imports/loading/inference. Not Gradio latency.',
                   'image':'synthetic-latency-sign.png','holdout_accessed':False,'frozen_runner_unchanged':True},stream,indent=2)
