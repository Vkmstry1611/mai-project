"""Comparable sequential localhost measurements on synthetic, cached images."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
import uuid


def summarize(values):
    return dict(individual_seconds=values,mean_seconds=statistics.mean(values),median_seconds=statistics.median(values))


def comparable_details(details):
    """Cache-hit bookkeeping may differ; answer/evidence/policy must not."""
    value=json.loads(json.dumps(details))
    for status in value.get('extraction_status',{}).values():status.pop('cache_hit',None)
    return value


def main(args):
    os.environ['GRADIO_ANALYTICS_ENABLED']='False'
    out=Path('reports/persistent_demo')/('benchmark-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new('RGB',(900,480),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((45,95,680,290),outline='black',width=3)
    draw.text((80,140),'WELCOME',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',80),fill='black')
    fixture=out/'synthetic-sign.png';image.save(fixture)
    import launch_reading_demo as ui
    original_request=ui.request
    from gradio_client import Client,handle_file
    rows=[];summaries=[]
    backends=['existing','persistent'] if args.backend=='both' else [args.backend]
    routes=['frozen','reading'] if args.route=='both' else [args.route]
    try:
        for backend in backends:
            for route in routes:
                session=None;ui.request=original_request
                if backend=='persistent':
                    from launch_persistent_demo import build_app
                    demo,session=build_app(Path(args.run_dir).resolve())
                else:demo=ui.build_app(Path(args.run_dir).resolve())
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                try:
                    _,url,_=demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port,
                        share=False,inbrowser=False,prevent_thread_lock=True,quiet=True)
                    client=Client(url,verbose=False);group=[]
                    for index in range(args.repeats+1):
                        print('Starting',backend,route,index,'warmup' if index==0 else 'measured repeat',flush=True)
                        tick=time.perf_counter()
                        result=client.predict(handle_file(str(fixture)),'What is written in this image?',True,route=='reading','0',api_name='/ask')
                        elapsed=time.perf_counter()-tick
                        timing=result[-1]
                        if isinstance(timing,str):timing=json.loads(Path(timing).read_text(encoding='utf-8'))
                        details=result[3]
                        if isinstance(details,str):details=json.loads(Path(details).read_text(encoding='utf-8'))
                        row=dict(backend=backend,route=route,index=index,warmup=index==0,end_to_end_seconds=elapsed,
                            answer=result[0],details=details,timings=timing,gradio_transfer_queue_overhead_seconds=elapsed-timing['callback_seconds'])
                        rows.append(row);group.append(row)
                        (out/f'{backend}-{route}-{index}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                        print('Completed',backend,route,index,round(elapsed,3),'seconds',flush=True)
                    summaries.append(dict(backend=backend,route=route,first_request_seconds=group[0]['end_to_end_seconds'],
                        repeated=summarize([r['end_to_end_seconds'] for r in group[1:]])))
                finally:
                    demo.close()
                    if session:session.close()
        comparisons=[]
        for route in routes:
            groups={backend:[r for r in rows if r['backend']==backend and r['route']==route] for backend in backends}
            if 'existing' in groups and 'persistent' in groups:
                reference=groups['existing'][0]
                equal=all(r['answer']==reference['answer'] and comparable_details(r['details'])==comparable_details(reference['details']) for group in groups.values() for r in group)
                comparisons.append(dict(route=route,answer_and_details_identical_except_cache_hit=equal))
                if not equal:print('WARNING: semantic parity mismatch; inspect saved details before use',flush=True)
        report=dict(rows=rows,summaries=summaries,comparisons=comparisons,repeats=args.repeats,
            scope='Sequential localhost Gradio requests, speech on, same synthetic pixels/question/config, one first request excluded from repeat statistics. OCR cache should be warm for both backends; verify saved extraction_status. Fresh worker per route for persistent; no flushed OS cold test. Browser rendering/audio playback excluded.',
            holdout_used=False,training=False)
        (out/'RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Saved',out,flush=True)
    finally:ui.request=original_request


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--backend',choices=['both','existing','persistent'],default='both')
    p.add_argument('--route',choices=['both','frozen','reading'],default='both');args=p.parse_args()
    if args.repeats<3:raise ValueError('At least three measured repeats are required')
    main(args)
