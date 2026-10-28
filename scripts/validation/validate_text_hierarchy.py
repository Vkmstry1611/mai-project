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
    image_path=Path('reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/data/images/VizWiz_val_00002603.jpg')
    manifest=json.loads(Path('reports/phase2_local_dev_evaluation/drive_inputs/split_manifest.json').read_text())
    assert image_path.name in json.dumps(manifest['dev'])
    # Large, separated source paragraphs without instruction keywords.
    columns=Image.new('RGB',(1800,400),'white');draw=ImageDraw.Draw(columns)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
    for x,lines in [(40,['This is the first passage about trees.','Green leaves grow in the spring.','Branches provide shade in summer.']),
                    (950,['This is the second passage about books.','A reader opens a book each day.','The pages contain a short story.'])]:
        for n,line in enumerate(lines):draw.text((x,40+n*60),line,font=font,fill='black')
    column_path=out/'two-passages.png';columns.save(column_path)
    session=FullTextSession('reports/phase4_policy/run-local-20261013-v1')
    import copy
    raw=json.loads(Path('reports/reading_sequences/run-20261020-v1/ORIGINAL_RAPID_RAW.json').read_text(encoding='utf-8'))
    counts=[];spoken=[];voice=ResponseSpeech.__call__
    def read(*a,**kw):counts.append(True);return copy.deepcopy(raw)
    def speech(self,text):spoken.append(text);return voice(self,text)
    session.reader.read=read;ResponseSpeech.__call__=speech
    demo,_=build_app(session.run_dir,session)
    for fn in demo.fns.values():
        if fn.fn and fn.fn.__name__=='upload_reset':fn.api_visibility='undocumented'
    records=[];checks={}
    def save():
        (out/'RESULTS.json').write_text(json.dumps({'records':records,'checks':checks,'raw_read_calls':len(counts),'ocr_source':'Saved exact native cooking fixture; no new extraction',
            'browser_interactions':'UNVERIFIED','fixture_split':'dev','holdout_inference':False},indent=2,ensure_ascii=False),encoding='utf-8')
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
        first=call('first-request',image_path,'Read the labels',upload=True,mute=False)
        second=call('second-step',image_path,'Read the second cooking step',mute=False)
        check('second-only','Stir in hot water' in second['answer'] and 'Brown' not in second['answer'] and 'Reduce' not in second['answer'])
        third=call('third-step',image_path,'Repeat step three')
        check('third-only','Reduce heat' in third['answer'] and 'Stir in hot water' not in third['answer'])
        one=call('first-step',image_path,'Read the first step')
        rest=call('remaining',image_path,'Read the remaining instructions')
        check('rest-not-repeat','Brown' not in rest['answer'])
        label=call('first-label',image_path,'I want first label to be read only')
        check('label-not-step','cannot reliably distinguish' in label['answer'] and not label['details']['supporting_evidence_ids'])
        absent=call('second-label',image_path,'Read the second label',mute=False)
        check('missing-label-clarification','cannot reliably distinguish' in absent['answer'] and not absent['details']['supporting_evidence_ids'])
        heading=call('heading',image_path,'Read the heading')
        check('heading-only','You Will Need' in heading['answer'] and 'Brown' not in heading['answer'])
        check('history-labels','User:' in heading['history'] and 'VisionBridge:' in heading['history'] and 'tabindex' in heading['history'])
        check('cache-reused',len(counts)==1)
        client.predict(api_name='/reset_session')
        fresh=call('after-reset',image_path,'Read the second cooking step',upload=True)
        check('reset-new-cache',len(counts)==2)
        check('reset-correct',fresh['answer']==second['answer'])
        check('no-gpu-worker',session.legacy.client.process is None)
        check('all',all(checks.values()))
    finally:ResponseSpeech.__call__=voice;demo.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/reading_hierarchy/api-'+uuid4().hex[:8]);main(p.parse_args().out)
