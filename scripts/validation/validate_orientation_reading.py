"""Permitted synthetic/manual-image validation; never reads VizWiz examples.

CPU stage: .venv-gpu ... --cpu
Real API stage: .venv ... --api <CPU-report-directory>
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'reports/phase4_policy/run-local-20261013-v1'
FAILED = RUN/'application-requests/9079468924344f5eb70fbc42e8ba4f22'


def save(path, data):
    with path.open('x', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def axis_hint(image):
    """CHEAP EXPERIMENT ONLY. Not imported or used by the production reader.

    Horizontal vs vertical directional gradient energy; cannot resolve 180,
    and graphics/mixed layouts can dominate. Deliberately does not skip angles.
    """
    import cv2
    import numpy as np
    small=image.convert('L');small.thumbnail((512,512))
    a=np.asarray(small)
    x=float(np.abs(cv2.Sobel(a,cv2.CV_32F,1,0)).mean())
    y=float(np.abs(cv2.Sobel(a,cv2.CV_32F,0,1)).mean())
    return dict(axis='horizontal' if x>y else 'vertical', gradient_x=x, gradient_y=y,
                limitation='Not a text-only estimator; cannot resolve upright versus upside down.')


def cpu():
    from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageEnhance
    import torch
    from config import Config
    from modules.ocr import load_reader
    from modules.evidence_store import descriptor
    from experimental_orientation import read,THRESHOLD,process_memory_mib
    out=ROOT/'reports/experimental_reading'/('orientation-validation-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    print('REPORT_DIRECTORY',out,flush=True)
    image=Image.new('RGB',(600,250),'white')
    ImageDraw.Draw(image).text((70,70),'HELLO',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',90),fill='black')
    fixtures={f'hello-{a}':image.rotate(a,expand=True) for a in (0,90,180,270)}
    mixed=Image.new('RGB',(900,500),'white')
    ImageDraw.Draw(mixed).text((30,170),'HELLO',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',65),fill='black')
    patch=Image.new('RGB',(400,130),'white')
    ImageDraw.Draw(patch).text((10,15),'NOTICE',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',65),fill='black')
    mixed.paste(patch.rotate(90,expand=True),(650,40));fixtures['mixed']=mixed
    fixtures['low-quality']=ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(12))).enhance(.02)
    multiline=Image.new('RGB',(820,340),'white')
    ImageDraw.Draw(multiline).multiline_text((40,40),'HELLO WORLD\nREAD THIS SIGN',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',65),fill='black',spacing=24)
    fixtures['multiline']=multiline
    fixtures['saved-failure']=Image.open(FAILED/'input.png').convert('RGB')
    cfg=Config();source=descriptor('ocr',cfg)
    records={};checks={};hints={}
    for label,photo in fixtures.items():
        photo.save(out/(label+'.png'))
        tick=time.perf_counter();hint=axis_hint(photo);hint['seconds']=time.perf_counter()-tick;hints[label]=hint
        tick=time.perf_counter()
        result=read(photo,RUN/'experimental-orientation-cache',source,
                    lambda:load_reader(str(cfg.cache_dir/'easyocr'),'cpu'))
        record=dict(wall_seconds=time.perf_counter()-tick,details=result,memory=process_memory_mib())
        records[label]=record;save(out/(label+'.json'),record)
        print(label,round(record['wall_seconds'],3),'seconds',result['orientation']['selected_ccw_degrees'],result['answer'],flush=True)
        if label.startswith('hello-'):
            a=int(label.split('-')[1]);checks[label]=(result['orientation']['selected_ccw_degrees']==(-a)%360 and result['answer']=='Recognized text fragments: HELLO.')
        elif label in ('mixed','low-quality'):
            checks[label]=result['answer_source']=='abstention'
        elif label=='multiline':
            # The second line scores 0.7369 in this environment. Allow either
            # the exact correct reading or a clearly labeled truthful partial.
            # Never relax 0.90 or insert missing words to pass this fixture.
            checks[label]=(result['orientation']['selected_ccw_degrees']==0 and
                (result['lines']==['HELLO WORLD','READ THIS SIGN'] or
                 (result['lines']==['HELLO WORLD'] and result['partial'])))
        else:
            checks[label]=result['orientation']['selected_ccw_degrees']==90
        save(out/(label+'-check.json'),dict(passed=checks[label]))
    save(out/'CPU_RESULTS.json',dict(checks=checks,records=records,axis_hint_experiment=hints,
        axis_hint_adopted=False,threshold=THRESHOLD,cuda_initialized=torch.cuda.is_initialized(),
        scope='Synthetic images and one exact saved manual input; no evaluation dataset.'))
    print('CPU checks:',checks,flush=True)
    if not all(checks.values()):raise AssertionError('Inspect failed CPU fixtures before API validation: '+str(out))


def api(out):
    import os,socket,wave
    os.environ['GRADIO_ANALYTICS_ENABLED']='False'
    from PIL import Image
    from launch_workspace import build_app,launch_options
    from orientation_workspace_session import OrientationSession
    from workspace_ui import ASSISTANT,READING
    from gradio_client import Client,handle_file
    prior=json.loads((out/'CPU_RESULTS.json').read_text(encoding='utf-8'))
    if not all(prior['checks'].values()):raise ValueError('CPU fixture checks must pass first')
    session=OrientationSession(RUN);demo,_=build_app(RUN,session=session)
    folder=out/('api-'+uuid.uuid4().hex[:8]);folder.mkdir(exist_ok=False)
    checks={};records=[]
    def check(name,passed):
        checks[name]=bool(passed)
        if not passed:raise AssertionError(name)
    def call(label,path,mode=ASSISTANT,rotation='0',speak=False,question='What is written in the image?'):
        tick=time.perf_counter()
        response=client.predict(handle_file(str(path)),question,not speak,mode,rotation,api_name='/analyze')
        details=next((v for v in response if isinstance(v,dict) and 'final_answer' in v),{})
        timings=next((v for v in response if isinstance(v,dict) and 'persistent_worker' in v),{})
        row=dict(label=label,wall_seconds=time.perf_counter()-tick,response=response,details=details,timings=timings)
        records.append(row);save(folder/(label+'.json'),row)
        print(label,round(row['wall_seconds'],3),'seconds',response[0],flush=True)
        return row
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        client=Client(url,verbose=False);check('server_started',bool(client))
        # Same text/size, different harmless corner pixel: controlled cache misses.
        nonce=uuid.uuid4().bytes
        def marked(index):
            image=Image.open(out/'hello-0.png').convert('RGB');image.putpixel((0,0),(index,255-index,255))
            # Unique outside-text nonce makes repeated benchmark runs real
            # cache misses without changing the visible HELLO fixture.
            for pixel in range(4):image.putpixel((pixel+1,0),tuple(nonce[pixel*3:pixel*3+3]))
            path=folder/('request-'+str(index)+'.png');image.save(path);return path
        cold=call('cold-auto',marked(11));check('cold_auto_hello',cold['response'][0]=='Recognized text fragments: HELLO.')
        warm_auto=[call('warm-miss-auto-'+str(i),marked(20+i)) for i in range(3)]
        warm_manual=[call('warm-miss-manual-'+str(i),marked(30+i),READING) for i in range(3)]
        check('cold_auto_cache_miss',not cold['details']['orientation']['cache_hit'])
        check('automatic_benchmark_cache_misses',all(not r['details']['orientation']['cache_hit'] for r in warm_auto))
        check('manual_benchmark_cache_misses',all(not r['details']['extraction_status']['ocr']['cache_hit'] for r in warm_manual))
        repeat=folder/'request-22.png'
        warm_hit=[call('warm-hit-auto-'+str(i),repeat) for i in range(3)]
        check('all_warm_hello',all(r['response'][0]=='Recognized text fragments: HELLO.' for r in warm_auto+warm_manual+warm_hit))
        check('warm_cache_hits',all(r['details']['orientation']['cache_hit'] for r in warm_hit))
        for angle in (0,90,180,270):
            row=call('api-angle-'+str(angle),out/(f'hello-{angle}.png'),speak=angle==0)
            check('api-correction-'+str(angle),row['details']['orientation']['selected_ccw_degrees']==(-angle)%360 and row['response'][0]=='Recognized text fragments: HELLO.')
            if angle==0:
                with wave.open(row['response'][1]) as wav:check('speech_audio_generated',wav.getnframes()>0)
        manual=call('manual-rotation',out/'hello-90.png',READING,'270')
        check('manual_rotation_bypasses_auto',manual['response'][0]=='Recognized text fragments: HELLO.' and 'orientation' not in manual['details'])
        for label in ('mixed','low-quality'):
            row=call('api-'+label,out/(label+'.png'))
            check('api-'+label+'-abstains',row['details']['answer_source']=='abstention')
        row=call('exact-saved-failure',FAILED/'input.png',speak=True)
        check('saved_failure_90',row['details']['orientation']['selected_ccw_degrees']==90)
        check('saved_failure_same_as_cpu',row['response'][0]==prior['records']['saved-failure']['details']['answer'])
        check('saved_original_and_corrected_boxes',all('original_polygon' in e and 'corrected_polygon' in e for e in row['details']['evidence']))
        with wave.open(row['response'][1]) as wav:check('saved_failure_audio_generated',wav.getnframes()>0)
        before=session.client.process.pid
        row=call('specific-field-notice',out/'hello-0.png',question='What is the expiry date?')
        check('specific_field_still_unsupported',row['details'].get('decision')=='workspace_unsupported_field')
        check('worker_reused',session.client.process.pid==before)
        check('missing_image_rejected','Check your input' in str(client.predict(None,'Read this',False,ASSISTANT,'0',api_name='/analyze')))
        def stats(rows):
            values=[r['wall_seconds'] for r in rows]
            return dict(individual_seconds=values,mean_seconds=statistics.mean(values),median_seconds=statistics.median(values))
        summary=dict(checks=checks,cold_auto_seconds=cold['wall_seconds'],warm_auto_cache_miss=stats(warm_auto),
            warm_manual_cache_miss=stats(warm_manual),warm_auto_cache_hit=stats(warm_hit),
            comparison='Same local API, persistent reader, synthetic HELLO at same size; TTS off for timings.',
            browser_interaction='UNVERIFIED; backend/API only',audible_playback='UNVERIFIED',
            holdout_inference=False,training=False)
        save(folder/'API_RESULTS.json',summary)
        print('API_RESULT_DIRECTORY',folder,flush=True);print(json.dumps(summary),flush=True)
    finally:
        if not (folder/'API_RESULTS.json').exists():save(folder/'INCOMPLETE.json',dict(checks=checks,completed_requests=len(records)))
        demo.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--api',type=Path);a=p.parse_args()
    if a.cpu:cpu()
    elif a.api:api(a.api.resolve())
    else:p.error('Choose --cpu or --api REPORT_DIRECTORY')
