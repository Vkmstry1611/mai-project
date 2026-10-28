"""Synthetic-only latency measurement; never evaluates a benchmark dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def stats(values):
    ordered = sorted(values)
    # Nearest-rank p95; small samples must not be treated as stable tail estimates.
    import math
    return {'count':len(values), 'mean_seconds':statistics.mean(values),
            'median_seconds':statistics.median(values), 'min_seconds':min(values),
            'max_seconds':max(values), 'p95_seconds_nearest_rank':ordered[math.ceil(.95*len(values))-1]}


def main(args):
    if args.repeats < 2 or args.warmup < 1:
        raise ValueError('Use at least one warmup and two timed requests')
    expected_env = '.venv-gpu' if args.mode == 'pipeline' else '.venv'
    if Path(sys.prefix).name != expected_env:
        raise RuntimeError('Use '+expected_env+' for this benchmark mode')
    out = Path(args.output).resolve()
    if ROOT/'reports/submission_preparation' not in out.parents:
        raise ValueError('Use a new directory inside reports/submission_preparation')
    out.mkdir(parents=True, exist_ok=False)
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB',(900,480),'white')
    font_path = Path('C:/Windows/Fonts/arialbd.ttf')
    ImageDraw.Draw(image).text((80,140),'WELCOME',fill='black',font=ImageFont.truetype(str(font_path),80))
    image.save(out/'synthetic-welcome.png')
    question = 'What does this sign say?'
    start = time.perf_counter()
    if args.mode == 'pipeline':
        import os
        os.environ['HF_HUB_OFFLINE']='1'
        from phase4_gpu_infer import prime_cuda,verify_application
        torch = prime_cuda()
        root = verify_application(args.run_dir)
        from phase4_pipeline import Phase4Pipeline
        pipeline = Phase4Pipeline(root)
        # Cache location only: isolate benchmark writes from earlier evidence/results.
        from dataclasses import replace
        pipeline.base.config=replace(pipeline.base.config,evidence_cache_dir=out/'evidence-cache')
        pipeline.config=pipeline.base.config
        def call():
            return pipeline.run(image.copy(),question)
        def synchronize():
            torch.cuda.synchronize()
    else:
        from launch_phase4 import install_worker
        app = install_worker()
        def call():
            return app.ask(image.copy(),question,False,args.run_dir)
        def synchronize():
            pass  # GPU subprocess has completed before the callback returns.
    setup = time.perf_counter()-start
    records=[]
    for index in range(args.warmup+args.repeats):
        synchronize(); start=time.perf_counter()
        result=call()
        synchronize(); elapsed=time.perf_counter()-start
        shown,_,status,details,_=result
        if details.get('baseline_answer') is None:
            raise RuntimeError('Base inference failed: '+status)
        record={'index':index,'warmup':index<args.warmup,'seconds':elapsed,'answer':shown,
                'baseline_answer':details['baseline_answer'],'final_answer':details['final_answer'],
                'decision':details['decision'],'gate_reason':details.get('phase4_gate_reason'),
                'supporting_evidence_ids':details.get('supporting_evidence_ids',[]),
                'extraction_status':details.get('extraction_status',{})}
        records.append(record)
        with (out/('request-%02d.json'%index)).open('x',encoding='utf-8') as stream:
            json.dump(record,stream,indent=2)
        print('warmup' if record['warmup'] else 'timed',index,round(elapsed,3),'seconds',shown,flush=True)
    measured=[r for r in records if not r['warmup']]
    signatures=[(r['baseline_answer'],r['final_answer'],r['decision'],r['supporting_evidence_ids']) for r in records]
    report={'mode':args.mode,'environment':sys.prefix,'setup_seconds':setup,'warmup_requests':args.warmup,
            'timed_requests':args.repeats,'latency':stats([r['seconds'] for r in measured]),
            'prediction_and_support_ids_stable':all(s==signatures[0] for s in signatures),
            'fixture_sha256':hashlib.sha256((out/'synthetic-welcome.png').read_bytes()).hexdigest(),
            'font_sha256':hashlib.sha256(font_path.read_bytes()).hexdigest(),'question':question,
            'policy_sha256':hashlib.sha256((Path(args.run_dir)/'selected-policy.json').read_bytes()).hexdigest(),
            'scope':('Persistent in-process pipeline, warmed models and cached evidence; NOT Gradio request latency.'
                     if args.mode=='pipeline' else 'Actual Gradio callback with warm file/cache state; BLIP reloads in every subprocess.'),
            'holdout_accessed':False,'training_performed':False,'audio_enabled':False,
            'records':records,'caveat':'Synthetic cached-image microbenchmark; small-sample p95 is descriptive only.'}
    if args.mode=='pipeline':
        report.update(gpu=torch.cuda.get_device_name(0),peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        verify_application(args.run_dir)
    with (out/'BENCHMARK.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['pipeline','ui'],default='pipeline')
    p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1')
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',default='reports/submission_preparation/latency-'+time.strftime('%Y%m%d-%H%M%S'))
    main(p.parse_args())
