"""Localhost Gradio API timing on synthetic images; frozen worker unchanged."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import pstats
import socket
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent


def cpu_seconds(process):
    times=[wintypes.FILETIME() for _ in range(4)]
    if not ctypes.windll.kernel32.GetProcessTimes(int(process._handle),*[ctypes.byref(t) for t in times]):
        return None
    return sum((t.dwHighDateTime<<32)+t.dwLowDateTime for t in times[2:])/1e7


def main(args):
    os.environ['GRADIO_ANALYTICS_ENABLED']='False'
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    from PIL import Image,ImageDraw,ImageFont
    photo=Image.new('RGB',(900,480),'white');d=ImageDraw.Draw(photo)
    d.rectangle((50,100,650,275),outline='black',width=3)
    d.text((80,140),'WELCOME',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',80),fill='black')
    fixture=out/'synthetic-latency-sign.png';photo.save(fixture)
    from launch_phase4 import install_worker
    app=install_worker();frozen_ask=app.ask
    callback_records=[];workers=[];current={'profile':False};samples=[];stop=threading.Event()

    class ObservedWorker:
        @staticmethod
        def run(command,**options):
            command=list(command)
            if Path(command[1]).resolve()!=ROOT/'phase4_infer.py':raise ValueError('Unexpected worker')
            command[1]=str(ROOT/'phase4_gpu_infer.py')
            if current['profile']:
                command[1:1]=['-m','cProfile','-o',str(out/'worker-profile.pstats')]
            options.pop('capture_output',None)
            timeout=options.pop('timeout')
            started=time.time();wall=time.perf_counter()
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000,**options) as process:
                try:stdout,stderr=process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill();process.communicate();raise
                cpu=cpu_seconds(process)
                workers.append({'pid':process.pid,'started_unix':started,'seconds':time.perf_counter()-wall,
                                'launcher_cpu_seconds':cpu,'returncode':process.returncode,'profiled':current['profile']})
                return subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
    app.subprocess=ObservedWorker

    def observed_ask(*values):
        start=time.time();wall=time.perf_counter()
        result=frozen_ask(*values)
        callback_records.append({'started_unix':start,'seconds':time.perf_counter()-wall})
        return result
    app.ask=observed_ask

    def sample_gpu():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used',
                    '--format=csv,noheader,nounits'],text=True,creationflags=0x08000000,timeout=8)
                samples.append({'unix':time.time(),'values':raw.strip()})
            except Exception as error:samples.append({'unix':time.time(),'error':type(error).__name__})
            stop.wait(2)
    monitor=threading.Thread(target=sample_gpu,daemon=True);monitor.start()
    demo=app.build_app(Path(args.run_dir).resolve())
    server_start=time.perf_counter()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    records=[]
    try:
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port,
            share=False,inbrowser=False,prevent_thread_lock=True,quiet=True)
        server_seconds=time.perf_counter()-server_start
        from gradio_client import Client,handle_file
        client=Client(url,verbose=False)
        endpoint=next(dep['api_name'] for dep in demo.config['dependencies'] if dep.get('api_name') and len(dep['inputs'])==3)
        for i,question in enumerate(['What does this sign say?']*3+['What does the bottle say?']):
            current['profile']=i==3
            print('Starting',i,'profiled YOLO route' if i==3 else 'unprofiled Gradio request',flush=True)
            start=time.time();wall=time.perf_counter();transitions=[]
            job=client.submit(handle_file(str(fixture)),question,False,api_name='/'+endpoint)
            last=None
            while not job.done():
                status=job.status();code=str(status.code)
                if code!=last:transitions.append({'seconds':time.perf_counter()-wall,'code':code,'rank':status.rank});last=code
                time.sleep(.25)
            result=job.result();elapsed=time.perf_counter()-wall
            record={'index':i,'label':'profiled_yolo' if i==3 else ('first_server_request' if i==0 else 'repeat_cache_warm'),
                    'end_to_end_seconds':elapsed,'callback':callback_records[-1],'worker':workers[-1],
                    'pre_callback_seconds_including_upload_and_queue':callback_records[-1]['started_unix']-start,
                    'post_callback_seconds':start+elapsed-callback_records[-1]['started_unix']-callback_records[-1]['seconds'],
                    'question':question,'answer':result[0],'details':result[3],'queue_status_transitions':transitions}
            records.append(record)
            (out/('request-%d.json'%i)).write_text(json.dumps(record,indent=2),encoding='utf-8')
            print('Completed',i,round(elapsed,3),'seconds',result[0],flush=True)
        stats=pstats.Stats(str(out/'worker-profile.pstats'))
        selected=[]
        for (filename,line,function),(primitive,calls,self_time,total_time,callers) in stats.stats.items():
            if (str(ROOT).lower() in filename.lower() and '.venv' not in filename.lower()) or function in {'_find_and_load','init','_lazy_init'}:
                selected.append({'file':filename,'line':line,'function':function,'calls':calls,'self_seconds':self_time,'inclusive_seconds':total_time})
        summary={'server_start_seconds':server_seconds,'records':records,'profile_functions':sorted(selected,key=lambda x:x['inclusive_seconds'],reverse=True),
                 'gpu_samples':samples,'scope':'Localhost upload, Gradio queue/callback and output transfer; excludes browser rendering/audio. First request is process/cache cold, NOT flushed OS cold.',
                 'profile_request_excluded_from_cold_warm_statistics':True,'holdout_accessed':False,'frozen_sources_modified':False,'manual_tests_passed':False}
        (out/'GRADIO_LATENCY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    finally:
        stop.set();monitor.join(timeout=10);demo.close()
        (out/'gpu-samples.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1')
    p.add_argument('--output',default='reports/submission_preparation/gradio-20261016-v1')
    main(p.parse_args())
