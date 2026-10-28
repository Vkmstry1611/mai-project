"""Final controlled website cold/warm HTTP benchmark and reference edge checks."""
import json
from pathlib import Path
import socket
import statistics
import time
import wave

def main():
    from gradio_client import Client,handle_file
    from visionbridge.full_text_session import FullTextSession
    from launch_workspace import build_app,launch_options
    from visionbridge.workspace_ui import ASSISTANT
    from visionbridge.workspace_speech import ResponseSpeech
    from scripts.validation.validate_image_conversation import OUT,ROOT
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1');demo,_=build_app(session.run_dir,session)
    calls=[];original=session.reader.read
    def observed(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    session.reader.read=observed
    records=[];checks={}
    def run(label,photo,q):
        started=time.perf_counter();r=client.predict(handle_file(str(photo)),q,False,ASSISTANT,'0',api_name='/analyze')
        d=next(x for x in r if isinstance(x,dict) and 'final_answer' in x)
        t=next(x for x in r if isinstance(x,dict) and 'workspace_generator_wall_seconds' in x)
        with wave.open(r[1]) as f:assert f.getnframes()>0
        records.append({'label':label,'question':q,'text':r[0],'wall_seconds':time.perf_counter()-started,'details':d,'timings':t})
        print(label,round(records[-1]['wall_seconds'],3),repr(r[0][:110]),flush=True)
        return records[-1]
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue().launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options());client=Client(url,verbose=False)
        ufo=ROOT/'071b6d6407ff4814ab6a6b46b08404e2/input.png'
        for n in range(4):
            r=run('website-cold' if n==0 else 'website-cached-'+str(n),ufo,'what is website in the image?')
            assert 'nolongerquivering.com' in r['text'] and 'cannot confirm' in r['text']
            assert r['details']['assistant_trace']['blip_status']=='not_requested'
        checks['four-website-responses-grounded']=True;checks['one-extraction-for-four-questions']=len(calls)==1
        assert checks['one-extraction-for-four-questions']
        hello=Path('reports/full_text_reading/api-17b203b2/hello.png')
        run('hello',hello,'Read text.')
        count=len(calls);clarify=run('explain-single-word',hello,'Explain that.')
        assert 'Please specify' in clarify['text'] and len(calls)==count
        checks['single-word-reference-clarifies-without-inference']=True
        client.predict('EasyOCR',api_name='/set_ocr_engine')
        easy=run('verified-runtime-easy',hello,'Read text.')
        assert easy['text']=='HELLO' and easy['details']['source']['engine']=='EasyOCR'
        checks['final-verified-weights-easy-worker']=True
        assert all(r['details']['assistant_trace'].get('tts_seconds',-1)>=0 for r in records)
        assert all(r['details']['assistant_trace'].get('total_api_generator_seconds',-1)>=0 for r in records)
        checks['stage-trace-and-default-speech']=True
        client.predict(api_name='/reset_session')
        clear=run('after-reset-clarify',hello,'Read the rest.')
        assert 'Do you mean' in clear['text'];checks['reset-clears-reference']=True
        warm=[r['wall_seconds'] for r in records[1:4]]
        payload={'records':records,'checks':checks,'warm_individual':warm,'warm_mean':statistics.mean(warm),'warm_median':statistics.median(warm),
            'scope':'Fresh CPU worker for website, then same-browser cached repeats; no GPU model or dataset inference.',
            'native_raw_calls':len(calls),'browser_rendering':'UNVERIFIED','autoplay':'UNVERIFIED'}
        with (OUT/'FINAL_CONTROLLED_API.json').open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2,ensure_ascii=False)
        print('Final controlled checks passed',len(checks),flush=True)
    finally:demo.close();session.close()

if __name__=='__main__':main()
