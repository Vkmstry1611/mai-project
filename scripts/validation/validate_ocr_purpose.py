"""Real personal/dev-fixture callbacks, original BLIP + OCR + speech. No evaluation."""
import json
from pathlib import Path
import socket
import time
import wave


def main():
    from visionbridge.full_text_session import FullTextSession
    from launch_workspace import build_app,launch_options
    from gradio_client import Client,handle_file
    from visionbridge.workspace_ui import ASSISTANT,VISUAL
    from visionbridge.workspace_speech import ResponseSpeech
    out=Path('reports/ocr_purpose/run-20261021-v1/api');out.mkdir(exist_ok=False)
    photo=Path('reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/data/images/VizWiz_val_00002603.jpg')
    manifest=json.loads(Path('reports/phase2_local_dev_evaluation/drive_inputs/split_manifest.json').read_text())
    assert photo.name in json.dumps(manifest['dev'])
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    calls=[];reader=session.reader.read
    def read(*args,**kwargs):calls.append(True);return reader(*args,**kwargs)
    session.reader.read=read
    voice=ResponseSpeech.__call__;spoken=[]
    def speak(self,text):spoken.append(text);return voice(self,text)
    ResponseSpeech.__call__=speak
    demo,_=build_app(session.run_dir,session)
    for fn in demo.fns.values():
        if fn.fn and fn.fn.__name__=='upload_reset':fn.api_visibility='undocumented'
    records=[];checks={}
    def save():
        (out/'RESULTS.json').write_text(json.dumps({'records':records,'checks':checks,'ocr_provider_calls':len(calls),
            'scope':'Actual BLIP CUDA, CPU OCR and TTS on one original development image. No dataset evaluation.',
            'browser_interactions':'UNVERIFIED'},indent=2,ensure_ascii=False),encoding='utf-8')
    def check(name,value):checks[name]=bool(value);save();assert value,name
    def run(name,q,mute=True,mode=ASSISTANT):
        tick=time.perf_counter();n=len(spoken)
        public=client.predict(handle_file(str(photo)),q,mute,mode,'0',api_name='/analyze')
        schema=client.endpoints[1].output_component_types;assert len(public)==sum(not c.skip for c in schema)
        values=iter(public);r=tuple(None if c.skip else next(values) for c in schema)
        record={'name':name,'question':q,'answer':r[0],'audio':bool(r[1]),'details':r[3],'timings':r[6],
                'seconds':time.perf_counter()-tick,'spoken_text':spoken[n:]}
        if r[1]:
            with wave.open(r[1]) as f:assert f.getnframes()>0
        records.append(record);check(name+'-contract',len(r)==14)
        print(name,round(record['seconds'],3),repr(r[0]),flush=True);return record
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue().launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        client=Client(url,verbose=False);client.predict(handle_file(str(photo)),api_name='/upload_reset')
        q='What is the image about?'
        before=run('before-cached-ocr',q)
        check('no-ocr-on-first-purpose',not calls)
        run('native-label-reading','Read the cooking instructions.')
        n=len(calls)
        after=run('purpose-after-reading',q,mute=False)
        check('cached-no-extra-ocr',len(calls)==n)
        if before['details']['final_answer']=='unanswerable':
            check('fallback-source',after['details']['answer_source']=='ocr_content_description')
            check('grounded-food-text','ground beef' in after['answer'] and 'Pasta' in after['answer'])
            frozen=after['details']['frozen_vqa_result']
            check('frozen-answer-preserved',frozen['final_answer']==before['details']['final_answer'])
            check('frozen-policy-preserved',frozen['phase4_policy']==before['details']['phase4_policy'])
        else:check('success-never-overridden',after['answer']==before['answer'])
        check('speech-matches-text',after['audio'] and after['spoken_text']==[after['answer']])
        again=run('cached-purpose-repeat',q)
        check('stable-cached-answer',again['answer']==after['answer'] and len(calls)==n)
        manual=run('manual-frozen-mode',q,mode=VISUAL)
        check('manual-vqa-unchanged',manual['details']['final_answer']==before['details']['final_answer'] and 'purpose_fallback' not in manual['details'])
        client.predict('EasyOCR',api_name='/set_ocr_engine')
        changed=run('different-engine-cache-miss',q)
        check('different-config-no-extraction',len(calls)==n and changed['details']['final_answer']==before['details']['final_answer'])
        check('all',all(checks.values()))
    finally:ResponseSpeech.__call__=voice;demo.close();session.close()


if __name__=='__main__':main()
