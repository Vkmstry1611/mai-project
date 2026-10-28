"""Actual HTTP/CPU OCR/speech tests, with explicitly replayed frozen visual capture.

Never loads a competing GPU worker or evaluates a dataset. Personal fixtures only.
"""
import json
from pathlib import Path
import socket
import time
import wave

OUT=Path('reports/assistant_followup/run-20261020-v1')
ROOT=Path('reports/phase4_policy/run-local-20261013-v1/application-requests')

def main():
    from gradio_client import Client,handle_file
    from launch_workspace import build_app,launch_options
    from visionbridge.full_text_session import FullTextSession
    from visionbridge.workspace_speech import ResponseSpeech
    from visionbridge.workspace_ui import ASSISTANT
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1');counts={'rapid':0,'easy':0,'saved_visual':0};records=[];checks={};spoken=[]
    for attr,name in [('reader','rapid'),('easy','easy')]:
        reader=getattr(session,attr);original=reader.read
        def observed(*args,_original=original,_name=name,**kwargs):counts[_name]+=1;return _original(*args,**kwargs)
        reader.read=observed
    saved=json.loads((ROOT/'d57ffbb579ea495d86fb10c36cc87b93/stdout.log').read_text(encoding='utf-8'))
    def frozen_replay(*args):
        from modules.evidence_store import image_identity
        from PIL import Image
        if image_identity(args[0])!=image_identity(Image.open(ROOT/'d57ffbb579ea495d86fb10c36cc87b93/input.png').convert('RGB')):raise ValueError('Replay only the exact verified blurry-image capture')
        counts['saved_visual']+=1
        return saved['display_answer'],None,saved['status'],saved['details'],None,'Preserved original Phase 4 output.',{'callback_seconds':0.,'saved_frozen_capture_replay':True,'model_inference_performed':False}
    session.legacy.request=frozen_replay
    original_voice=ResponseSpeech.__call__;fail=[False]
    def voice(self,text):
        spoken.append(text)
        if fail[0]:raise RuntimeError('Injected TTS failure')
        return original_voice(self,text)
    ResponseSpeech.__call__=voice
    demo,_=build_app(session.run_dir,session)
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:raise AssertionError(name)
    def call(client,name,image,q,muted=False):
        tick=time.perf_counter();before=len(spoken)
        job=client.submit(handle_file(str(image)),q,muted,ASSISTANT,'0',api_name='/analyze');result=job.result()
        details=next(x for x in result if isinstance(x,dict) and 'final_answer' in x)
        timings=next(x for x in result if isinstance(x,dict) and 'workspace_generator_wall_seconds' in x)
        record={'name':name,'question':q,'text':result[0],'seconds':time.perf_counter()-tick,'details':details,'timings':timings,
                'tts_text':spoken[before:],'counts':dict(counts),'history':result[-1]}
        records.append(record)
        check(name+'-text',bool(result[0]));check(name+'-speech-count',len(spoken)-before==(0 if muted else 1))
        if not muted and not fail[0]:
            with wave.open(result[1]) as f:check(name+'-audio',f.getnframes()>0)
            check(name+'-same-facts',spoken[-1]==result[0])
            check(name+'-text-first',any(x[0]==result[0] and x[1] is None for x in job.outputs()))
        elif fail[0]:check(name+'-failure-text',result[1] is None)
        with (OUT/(name+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
        print(name,round(record['seconds'],3),repr(record['text'][:170]),counts,flush=True)
        return record
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue(default_concurrency_limit=1).launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        a=Client(url,verbose=False);b=Client(url,verbose=False)
        ufo=ROOT/'071b6d6407ff4814ab6a6b46b08404e2/input.png';blurry=ROOT/'d57ffbb579ea495d86fb10c36cc87b93/input.png'
        call(a,'01-detect-every-text',ufo,'Detect every text')
        website=call(a,'02-exact-website-question',ufo,'what is website in the image?')
        check('exact-domain-recovered','nolongerquivering.com' in website['text'] and 'cannot confirm' in website['text'])
        check('website-no-main-passage','BEAUTY PARLOR' not in website['text'])
        check('website-no-blip',counts['saved_visual']==0 and session.legacy.client.process is None)
        main=call(a,'03-main-reading',ufo,'What is written in the image?')
        check('main-no-browser',all(x not in main['text'] for x in ('http','Safari','Amazon')))
        call(a,'04-url',ufo,'What is the URL?')
        call(a,'05-contextual-website',ufo,'What website is that?')
        call(a,'06-passage-question',ufo,'What happened in the passage?')
        call(a,'07-contextual-overview',ufo,'Explain that.')
        call(a,'08-contextual-rest',ufo,'Read the rest.')
        call(a,'09-guidance',ufo,'What should I do to make it clearer?')
        check('same-image-one-ocr',counts['rapid']==1 and counts['easy']==0)
        isolated=call(b,'10-isolated-browser',ufo,'Which domain is visible?',muted=True)
        check('browser-history-isolated',isolated['history'].count('<li>')==1 and 'Detect every' not in isolated['history'])
        check('independent-raw-evidence',len(isolated['details']['evidence'])==len(website['details']['evidence']))
        visual=call(a,'11-blurry-general-replay',blurry,'what is in the image?')
        check('blurry-frozen-preserved',visual['details']['baseline_answer']==visual['details']['final_answer']=='unanswerable')
        check('blurry-no-invented-scene','person holding' not in visual['text'])
        check('general-no-extra-ocr',counts['rapid']==2 and counts['easy']==0)
        call(a,'12-repeat-visual-cache',blurry,'what is in the image?',muted=True)
        check('visual-cache-reused',counts['saved_visual']==1)
        label=call(a,'13-blurry-label-reading',blurry,'Can you read it?')
        call(a,'14-exact-product',blurry,'What exact product is this?')
        check('product-abstention',records[-1]['details']['answer_source']=='abstention')
        before_counts=dict(counts)
        fields=Path('reports/full_text_reading/api-17b203b2/labeled-fields.png')
        retake=call(a,'15-clear-retake-product',fields,'What exact product is this?')
        check('retake-clears-old-history','11-blurry' not in retake['history'] and retake['history'].count('<li>')==1)
        check('retake-product-explicit','ACME' in retake['text'])
        for name,q in [('expiry','What is the expiry date?'),('price','What is the price?'),('manufacturing','What is the manufacturing date?')]:
            field=call(a,'16-field-'+name,fields,q,muted=True);check(name+'-explicit-supported',field['details']['decision']=='experimental_labeled_field')
        check('field-cache-one-extraction',counts['rapid']==before_counts['rapid']+1)
        hello=Path('reports/full_text_reading/api-17b203b2/hello.png')
        a.predict('EasyOCR',api_name='/set_ocr_engine');easy=call(a,'17-easy-hello',hello,'Read text.')
        check('easy-retained',easy['details']['source']['engine']=='EasyOCR' and easy['text']=='HELLO')
        a.predict('RapidOCR',api_name='/set_ocr_engine');rapid=call(a,'18-rapid-hello',hello,'Read text.')
        check('rapid-retained',rapid['details']['source']['engine']=='RapidOCR' and rapid['text']=='HELLO')
        a.predict('Automatic',api_name='/set_ocr_engine')
        call(a,'19-ambiguous-reference',hello,'Explain that.')
        fail[0]=True;call(a,'20-voice-failure',hello,'Read text.');fail[0]=False
        call(a,'21-muted',hello,'Read text.',muted=True)
        speaker=ResponseSpeech();audio=speaker('What words are written in this image?')
        transcript=a.predict(handle_file(audio),api_name='/transcribe_question');check('real-whisper',bool(transcript[0].strip()))
        a.predict(api_name='/reset_session')
        cleared=call(a,'22-after-clear',hello,'Read the rest.',muted=True)
        check('clear-forgets-context','Do you mean' in cleared['text'])
        check('no-owned-gpu-started',session.legacy.client.process is None)
        summary={'checks':checks,'records':records,'engine_calls':counts,'scope':'Real HTTP, CPU OCR, speech and Whisper. Exact frozen blurry result replayed; no new GPU or dataset inference.',
            'browser_autoplay':'UNVERIFIED','screen_reader':'UNVERIFIED','physical_microphone':'UNVERIFIED'}
        with (OUT/'API_RESULTS.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
        print('API checks passed',len(checks),flush=True)
    finally:
        ResponseSpeech.__call__=original_voice;demo.close();session.close()
        if not (OUT/'API_RESULTS.json').exists():
            (OUT/('API_INCOMPLETE-'+str(len(records))+'.json')).write_text(json.dumps({'checks':checks,'completed':len(records)},indent=2))

if __name__=='__main__':main()
