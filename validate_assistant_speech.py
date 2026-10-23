"""Real localhost, CPU OCR, local GPU VQA and SAPI checks on permitted fixtures."""
import json
import os
from pathlib import Path
import socket
import time
import uuid
import wave


def main():
    os.environ['GRADIO_ANALYTICS_ENABLED']='False';os.environ['HF_HUB_OFFLINE']='1'
    from PIL import Image,ImageDraw,ImageFont
    from gradio_client import Client,handle_file
    from launch_workspace import build_app,launch_options
    from full_text_session import FullTextSession
    from workspace_ui import ASSISTANT,READING
    from workspace_speech import ResponseSpeech
    out=Path('reports/full_text_reading')/('api-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    base=Path('reports/full_text_reading/fixtures-83eeca7a')
    label=Image.new('RGB',(900,400),'white')
    ImageDraw.Draw(label).multiline_text((40,40),'BRAND: ACME\nEXP: 12/2027\nMFG: 01/2026\nPRICE: Rs 99.00',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',48),fill='black',spacing=24)
    label.save(out/'labeled-fields.png')
    hello=Image.new('RGB',(600,250),'white')
    ImageDraw.Draw(hello).text((70,70),'HELLO',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',90),fill='black');hello.save(out/'hello.png')
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1');demo,_=build_app(session.run_dir,session)
    checks={};records=[];speech_calls=[];fail=[False]
    original=ResponseSpeech.__call__
    def observed(self,text):
        speech_calls.append(text)
        if fail[0]:raise RuntimeError('Injected voice failure for API test')
        return original(self,text)
    ResponseSpeech.__call__=observed
    def save(name,data):
        with (out/(name+'.json')).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False)
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise AssertionError(name)
    def call(name,image,question,muted=False,mode=ASSISTANT,rotation='0',defaults=False):
        before=len(speech_calls);tick=time.perf_counter()
        args=[handle_file(str(image)),question] if defaults else [handle_file(str(image)),question,muted,mode,rotation]
        job=client.submit(*args,api_name='/analyze');result=job.result();stream=job.outputs()
        details=next(v for v in result if isinstance(v,dict) and 'final_answer' in v)
        record=dict(name=name,seconds=time.perf_counter()-tick,result=result,details=details,
            speech_attempts=len(speech_calls)-before,streamed_responses=len(stream))
        records.append(record);save(name,record);print(name,round(record['seconds'],3),result[0][:180],flush=True)
        check(name+'-text',bool(result[0]))
        if muted:check(name+'-mute',result[1] is None and record['speech_attempts']==0)
        elif fail[0]:check(name+'-failure-keeps-text',result[1] is None and 'Speech generation failed' in str(result))
        else:
            with wave.open(result[1]) as audio:check(name+'-audio',audio.getnframes()>0)
            check(name+'-once',record['speech_attempts']==1)
            check(name+'-text-before-audio',any(v[0]==result[0] and v[1] is None for v in stream))
        return record
    try:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        client=Client(url,verbose=False)
        components={c['props'].get('elem_id'):c for c in demo.config['components']}
        check('mute-default-off',components['vb-mute']['props']['value'] is False)
        check('autoplay-configured',components['vb-playback']['props']['autoplay'] is True)
        check('old-speech-checkbox-removed',not any(c['props'].get('label')=='Generate a spoken answer' for c in demo.config['components']))
        call('default-full-text',base/'paragraph-90.png','Read the complete paragraph.',defaults=True)
        call('muted-full-text',base/'paragraph-90.png','Read everything written here.',muted=True)
        call('clarification',out/'hello.png','Help',defaults=True)
        blank=call('blank-abstention',base/'blank.png','Read this page.',defaults=True)
        replacement=call('replacement-retake',base/'paragraph-0.png','Read this page.',muted=True)
        guidance=next(v for v in replacement['result'] if isinstance(v,dict) and 'comparison' in v)
        check('retake-previous-image-preserved',guidance['comparison'].get('previous_image_id')==blank['details']['image_id'])
        for question in ['What is the expiry date?','What is the price?','What brand is this?']:
            row=call('field-'+str(len(records)),out/'labeled-fields.png',question)
            check('supported-field-'+str(len(records)),row['details']['decision']=='experimental_labeled_field')
        fail[0]=True
        call('tts-failure',out/'hello.png','Help')
        fail[0]=False
        call('manual-conservative',out/'hello.png','What is written in the image?',mode=READING)
        call('manual-full-rotation',base/'paragraph-90.png','Read this page.',mode='Full Text Reading (experimental)',rotation='270',muted=True)
        print('Running one permitted real GPU VQA request, no evaluation dataset.',flush=True)
        vqa=call('real-frozen-vqa',out/'hello.png','What color is the background?',defaults=True)
        check('real-vqa-preserved',bool(vqa['details'].get('baseline_answer')) and 'orientation' not in vqa['details'])
        speaker=ResponseSpeech();spoken=speaker('What words are written in this image?')
        transcript=client.predict(handle_file(spoken),api_name='/transcribe_question')
        save('transcription',{'result':transcript})
        check('whisper-transcription',bool(transcript[0].strip()))
        reset=client.predict(api_name='/reset_session');check('reset-clears-answer',not any(isinstance(v,str) and 'Experimental transcription' in v for v in reset))
        save('RESULTS',dict(checks=checks,records=records,scope='Real API OCR, one GPU VQA image, SAPI and synthetic spoken-question transcription.',
            browser_autoplay='UNVERIFIED',microphone_physical='UNVERIFIED',audible_playback='UNVERIFIED',holdout_inference=False,training=False))
        print('API_REPORT',out,'Checks passed',len(checks),flush=True)
    finally:
        ResponseSpeech.__call__=original
        if not (out/'RESULTS.json').exists():save('INCOMPLETE',dict(checks=checks,completed_requests=len(records)))
        demo.close();session.close()


if __name__=='__main__':main()
