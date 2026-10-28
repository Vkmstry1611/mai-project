"""Real Gradio/TTS with immutable native OCR replay; no fresh OCR, BLIP or evaluation."""
import argparse
import json
from pathlib import Path
import socket
import statistics
import time
from uuid import uuid4
import wave


def main(out):
    from PIL import Image,ImageDraw,ImageFont
    from gradio_client import Client,handle_file
    from visionbridge.full_text_session import FullTextSession
    from launch_workspace import build_app,launch_options
    from visionbridge.workspace_ui import ASSISTANT
    from visionbridge.workspace_speech import ResponseSpeech
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    image_path=Path('reports/phase4_policy/run-local-20261013-v1/application-requests/071b6d6407ff4814ab6a6b46b08404e2/input.png')
    # Large, separated source paragraphs without instruction keywords.
    columns=Image.new('RGB',(1800,400),'white');draw=ImageDraw.Draw(columns)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
    for x,lines in [(40,['This is the first passage about trees.','Green leaves grow in the spring.','Branches provide shade in summer.']),
                    (950,['This is the second passage about books.','A reader opens a book each day.','The pages contain a short story.'])]:
        for n,line in enumerate(lines):draw.text((x,40+n*60),line,font=font,fill='black')
    column_path=out/'two-passages.png';columns.save(column_path)
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    import copy
    raw={'result':json.loads(Path('reports/assistant_followup/run-20261020-v1/API_RESULTS.json').read_text(encoding='utf-8'))['records'][0]['details'],'memory':{}}
    counts=[];spoken=[];voice=ResponseSpeech.__call__
    def read(*a,**kw):counts.append(True);return copy.deepcopy(raw)
    def speech(self,text):spoken.append(text);return voice(self,text)
    session.reader.read=read;ResponseSpeech.__call__=speech
    demo,_=build_app(session.run_dir,session)
    for fn in demo.fns.values():
        if fn.fn and fn.fn.__name__=='upload_reset':fn.api_visibility='undocumented'
    records=[];checks={}
    def save():
        (out/'RESULTS.json').write_text(json.dumps({'records':records,'checks':checks,'raw_read_calls':len(counts),'ocr_source':'Saved exact UFO screenshot OCR; no new extraction',
            'browser_interactions':'UNVERIFIED','fixture_split':'existing browser screenshot (no dataset evaluation)','holdout_inference':False},indent=2,ensure_ascii=False),encoding='utf-8')
    def check(name,condition):
        checks[name]=bool(condition);save();assert condition,name
    def call(name,photo,q,mute=True,upload=False,c=None):
        c=c or client
        if upload:c.predict(handle_file(str(photo)),api_name='/upload_reset')
        before=len(spoken);tick=time.perf_counter()
        values=c.predict(handle_file(str(photo)),q,mute,ASSISTANT,'0',api_name='/analyze')
        schema=c.endpoints[1].output_component_types;assert len(values)==sum(not v.skip for v in schema)
        public=iter(values);r=tuple(None if v.skip else next(public) for v in schema)
        assert len(r)==14
        if r[1]:
            with wave.open(r[1]) as f:assert f.getnframes()>0
        item={'name':name,'question':q,'answer':r[0],'audio':bool(r[1]),'details':r[3],'timings':r[6],
              'history':r[13],'seconds':time.perf_counter()-tick,'spoken_text':spoken[before:]}
        records.append(item);check(name+'-contract',True)
        check(name+'-tts',spoken[before:]==([] if mute or not r[0] else [r[0]]))
        print(name,round(item['seconds'],3),repr(r[0][:160]),flush=True)
        return item
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        _,url,_=demo.queue().launch(server_port=port,prevent_thread_lock=True,quiet=True,**launch_options())
        client=Client(url,verbose=False)
        r=call('exact-passage-about',image_path,'What is the passage about?',upload=True,mute=False)
        check('summary-route',r['details']['reading_intent']=='summary')
        check('grounded-summary',r['answer'].startswith('The passage describes') and bool(r['details']['supporting_evidence_ids']))
        main=call('main-passage',image_path,'Read the main passage',mute=False)
        check('main-evidence',bool(main['details']['supporting_evidence_ids']))
        web=call('domain',image_path,'What website is shown?')
        check('domain-evidence','nolongerquivering.com' in web['answer'])
        follow=call('followup-summary',image_path,'What is the passage about?',mute=False)
        check('same-summary',follow['answer']==r['answer'])
        check('cache-reuse',len(counts)==1)
        check('history-structure','<br>' in follow['history'] and 'display:block' in follow['history'])
        check('no-gpu-worker',session.legacy.client.process is None)
        check('all',all(checks.values()))
    finally:ResponseSpeech.__call__=voice;demo.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/screenshot_errors/api-'+uuid4().hex[:8]);main(p.parse_args().out)
