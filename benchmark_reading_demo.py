"""Small synthetic Gradio API benchmark; never reads an evaluation dataset."""
import argparse
import json
import os
from pathlib import Path
import socket
import time
import uuid


def main(args):
    os.environ['GRADIO_ANALYTICS_ENABLED']='False'
    out=Path('reports/experimental_reading')/('benchmark-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    photo=Image.new('RGB',(900,480),'white');draw=ImageDraw.Draw(photo)
    draw.rectangle((45,95,680,290),outline='black',width=3)
    draw.text((80,140),'WELCOME',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',80),fill='black')
    fixture=out/'synthetic-sign.png';photo.save(fixture)
    from launch_reading_demo import build_app
    demo=build_app(Path(args.run_dir).resolve())
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    records=[]
    try:
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port,
            share=False,inbrowser=False,prevent_thread_lock=True,quiet=True)
        from gradio_client import Client,handle_file
        client=Client(url,verbose=False)
        cases=[('first_frozen_request',False),('repeat_frozen_request',False),('experimental_reading',True)]
        if args.mode=='reading-only':cases=cases[-1:]
        for label,experimental in cases:
            print('Starting',label,flush=True);start=time.perf_counter()
            result=client.predict(handle_file(str(fixture)),'What is written in this image?',True,experimental,'0',api_name='/ask')
            timing=result[-1]
            if isinstance(timing,str):timing=json.loads(Path(timing).read_text(encoding='utf-8'))
            record=dict(label=label,end_to_end_seconds=time.perf_counter()-start,answer=result[0],audio_file=result[1],timings=timing)
            records.append(record)
            (out/(label+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
            print('Completed',label,round(record['end_to_end_seconds'],3),'seconds',flush=True)
        summary=dict(records=records,holdout_used=False,training=False,
            scope='Localhost Gradio upload, queue, callback, TTS and result transfer. Excludes browser rendering and audible playback. First request is process/cache cold, not flushed OS cold. Frozen worker restarts on every request; repeat is not model-warm.')
        (out/'RESULTS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('Saved',out,flush=True)
    finally:demo.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1')
    p.add_argument('--mode',choices=['all','reading-only'],default='all');main(p.parse_args())
