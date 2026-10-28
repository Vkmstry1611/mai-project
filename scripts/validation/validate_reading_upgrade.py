"""Real localhost OCR/speech validation, permitted manual photos only.

No GPU request is started while the user's existing GPU worker is resident.
Frozen VQA routing is covered by regression tests and read-only integrity checks.
"""
import json
from pathlib import Path
import socket
import statistics
import time
import uuid
import wave

def main():
    from gradio_client import Client,handle_file
    from launch_workspace import build_app,launch_options
    from visionbridge.full_text_session import FullTextSession
    from visionbridge.workspace_speech import ResponseSpeech
    from visionbridge.workspace_ui import ASSISTANT
    from visionbridge.investigate_reading_upgrade import OUT,PHOTOS
    out=OUT/('api-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1');demo,_=build_app(session.run_dir,session)
    original=ResponseSpeech.__call__;spoken=[];checks={};records=[];fail=[False]
    def voice(self,text):
        spoken.append(text)
        if fail[0]:raise RuntimeError('Injected test voice failure')
        return original(self,text)
    ResponseSpeech.__call__=voice
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    def call(name,path,q,mute=False):
        before=len(spoken);tick=time.perf_counter()
        job=client.submit(handle_file(str(path)),q,mute,ASSISTANT,'0',api_name='/analyze');r=job.result()
        details=next(x for x in r if isinstance(x,dict) and 'final_answer' in x)
        timings=next(x for x in r if isinstance(x,dict) and 'callback_seconds' in x) if any(isinstance(x,dict) and 'callback_seconds' in x for x in r) else {}
        record=dict(name=name,wall_seconds=time.perf_counter()-tick,text=r[0],details=details,timings=timings,tts_text=spoken[before:])
        records.append(record)
        check(name+'-text',bool(r[0]))
        check(name+'-one-speech',len(spoken)-before==(0 if mute else 1))
        if not mute and not fail[0]:
            with wave.open(r[1]) as f:check(name+'-audio',f.getnframes()>0)
            check(name+'-factual-consistency',spoken[-1]==r[0])
            check(name+'-text-first',any(x[0]==r[0] and x[1] is None for x in job.outputs()))
        elif mute:check(name+'-muted',r[1] is None)
        else:check(name+'-failure-keeps-text',r[1] is None)
        with (out/(name+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
        print(name,round(record['wall_seconds'],3),repr(r[0][:150]),flush=True)
        return record
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        client=Client(url,verbose=False)
        photos={k:Path('reports/phase4_policy/run-local-20261013-v1/application-requests')/v/'input.png' for k,v in PHOTOS.items()}
        wine=call('wine-cold',photos['wine'],'What is written in this image?')
        check('wine-no-junk',not any(x in wine['text'] for x in ('[uncertain','[unreadable','by wol','Experimental transcription')))
        warm=[]
        for n in range(3):
            session.reader.cache_namespace='warm-'+uuid.uuid4().hex
            # A fresh browser state also prevents the new in-memory image
            # conversation cache from turning this into an extraction hit.
            client=Client(url,verbose=False)
            warm.append(call('wine-warm-'+str(n),photos['wine'],'What is written in this image?')['wall_seconds'])
        call('wine-cache',photos['wine'],'What is written in this image?')
        ufo=call('ufo-main',photos['ufo'],'What is written in this image?')
        check('ufo-main-no-interface',all(x not in ufo['text'] for x in ('Safari','http','Amazon','skeptic.gif')))
        check('ufo-raw-preserved',any('http' in r['text'] for r in ufo['details']['evidence']))
        call('ufo-all',photos['ufo'],'Read every visible word.')
        call('ufo-summary',photos['ufo'],'Summarize this passage.')
        call('ufo-question',photos['ufo'],'What happened in the story?')
        call('ufo-website',photos['ufo'],'What website is shown?')
        hello=Path('reports/full_text_reading/api-17b203b2/hello.png')
        client.predict('EasyOCR',api_name='/set_ocr_engine')
        easy=call('easy-hello',hello,'What is written in this image?')
        check('easy-accessible',easy['details']['source']['engine']=='EasyOCR' and 'HELLO' in easy['text'])
        call('easy-wine',photos['wine'],'What is written in this image?',mute=True)
        client.predict('RapidOCR',api_name='/set_ocr_engine')
        call('rapid-hello',hello,'Read text.')
        client.predict('Automatic',api_name='/set_ocr_engine')
        call('clarification',hello,'Help')
        call('muted',hello,'Read text.',mute=True)
        fail[0]=True;call('speech-failure',hello,'Read text.');fail[0]=False
        fields=Path('reports/full_text_reading/api-17b203b2/labeled-fields.png')
        for label,q in [('expiry','What is the expiry date?'),('manufacturing','What is the manufacturing date?'),('price','What is the price?'),('brand','What brand is this?')]:
            r=call(label,fields,q,mute=True);check(label+'-supported',r['details']['decision']=='experimental_labeled_field')
        speaker=ResponseSpeech();audio=speaker('What words are written in this image?')
        transcription=client.predict(handle_file(audio),api_name='/transcribe_question')
        check('real-whisper-generated-audio',bool(transcription[0].strip()))
        check('retake-state-wired',any(c['type']=='state' for c in demo.config['components']))
        check('gpu-worker-not-started',session.legacy.client.process is None)
        summary=dict(checks=checks,records=records,warm_individual=warm,warm_mean=statistics.mean(warm),warm_median=statistics.median(warm),
            scope='Permitted private photos, real CPU OCR, SAPI and Whisper; no training or evaluation datasets.',
            gpu_validation='Not rerun: user-owned GPU worker remains active; frozen hashes and regression routing checked.',
            browser_autoplay='UNVERIFIED',screen_reader='UNVERIFIED',physical_microphone='UNVERIFIED')
        with (out/'RESULTS.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
        print('REPORT',out,'CHECKS',len(checks),flush=True)
    finally:
        ResponseSpeech.__call__=original
        if not (out/'RESULTS.json').exists():
            (out/'INCOMPLETE.json').write_text(json.dumps(dict(checks=checks,completed=len(records)),indent=2),encoding='utf-8')
        demo.close();session.close()

if __name__=='__main__':main()
