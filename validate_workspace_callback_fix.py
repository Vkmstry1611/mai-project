"""Actual upload/state HTTP tests on personal fixtures; never dataset inference."""
import json
from pathlib import Path
import socket
import time
import wave

OUT=Path('reports/workspace_callback_fix/run-20261020-v1')


def main():
    from gradio_client import Client,handle_file
    from full_text_session import FullTextSession
    from launch_workspace import build_app,launch_options
    from workspace_ui import ASSISTANT,VISUAL
    from workspace_speech import ResponseSpeech
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    demo,_=build_app(session.run_dir,session)
    for fn in demo.fns.values():
        if fn.fn and fn.fn.__name__=='upload_reset':fn.api_visibility='undocumented'
    records=[];checks={};fail_tts=[False]
    voice=ResponseSpeech.__call__
    def speech(self,text):
        if fail_tts[0]:raise RuntimeError('Injected TTS failure')
        return voice(self,text)
    ResponseSpeech.__call__=speech
    root=Path('reports/phase4_policy/run-local-20261013-v1/application-requests')
    blurry=root/'d57ffbb579ea495d86fb10c36cc87b93/input.png'
    ufo=root/'071b6d6407ff4814ab6a6b46b08404e2/input.png'
    hello=Path('reports/full_text_reading/api-17b203b2/hello.png')
    def save():
        (OUT/'API_RESULTS.json').write_text(json.dumps({'records':records,'checks':checks,'browser_interactions':'UNVERIFIED'},indent=2,ensure_ascii=False),encoding='utf-8')
    def run(name,photo,q,mute=False,mode=ASSISTANT,upload=False):
        if upload:client.predict(handle_file(str(photo)),api_name='/upload_reset')
        tick=time.perf_counter()
        result=client.predict(handle_file(str(photo)) if photo else None,q,mute,mode,'0',api_name='/analyze')
        schema=client.endpoints[1].output_component_types
        assert len(result)==sum(not c.skip for c in schema)
        public=iter(result)
        result=tuple(None if c.skip else next(public) for c in schema)
        assert len(result)==14
        d=result[3];assert isinstance(d,dict)
        if result[1]:
            with wave.open(result[1]) as audio:assert audio.getnframes()>0
        row={'name':name,'answer':result[0],'audio_generated':bool(result[1]),'status':result[2],
             'details':d,'timings':result[6],'guidance':result[11],'seconds':time.perf_counter()-tick}
        records.append(row);checks[name+'-contract']=True;save()
        print(name,round(row['seconds'],3),repr(row['answer'][:160]),flush=True)
        return row
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue().launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        client=Client(url,verbose=False)
        a=run('exact-blurry-upload',blurry,'what is written in the image?',upload=True)
        assert a['answer'] and a['audio_generated'] and 'application_error' not in a['details']
        checks['first-upload-has-no-fake-quality-baseline']='No earlier completed analysis' in str(a['guidance'])
        b=run('blurry-followup','' if False else blurry,'Read the rest.',mute=True)
        assert b['answer']
        c=run('exact-ufo-new-upload',ufo,'What website is in the image?',upload=True)
        assert 'nolongerquivering.com' in c['answer'] and c['audio_generated']
        h=run('hello-new-upload',hello,'Read the text.',upload=True)
        assert 'HELLO' in h['answer'] and h['audio_generated']
        cached=run('hello-cached',hello,'Read the text.',mute=True)
        assert cached['answer']==h['answer']
        checks['ocr-cache-hit']=cached['details']['assistant_trace']['ocr_cache_hit']
        fail_tts[0]=True
        t=run('tts-failure',hello,'Read the text.')
        assert t['answer']==h['answer'] and not t['audio_generated'] and 'text answer is still available' in t['status']
        fail_tts[0]=False
        run('missing-image',None,'Read the text.',mute=True)
        run('empty-question',hello,'',mute=True)
        client.predict(api_name='/reset_session')
        reset=run('after-reset',hello,'Read the rest.',mute=True)
        assert 'Do you mean' in reset['answer']
        # Real frozen GPU invocation, not a saved-answer stub. No VizWiz image.
        v=run('real-frozen-vqa',blurry,'What is in the image?',mute=True,mode=VISUAL,upload=True)
        assert v['details'].get('final_answer') and not v['details'].get('application_error')
        checks['real-frozen-vqa']=True
        original=session.legacy.request
        def broken(*a):raise RuntimeError('Injected worker failure')
        session.legacy.request=broken
        bad=run('worker-failure',blurry,'What is in the image?',mute=True,mode=VISUAL)
        assert not bad['answer'] and bad['details']['application_error']['stage']=='analysis-worker'
        session.legacy.request=original
        checks['all']=all(checks.values());assert checks['all'];save()
    finally:
        ResponseSpeech.__call__=voice
        demo.close();session.close()


if __name__=='__main__':main()
