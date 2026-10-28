"""Real Gradio/OCR/TTS sequence checks; no BLIP or dataset evaluation."""
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
    from full_text_session import FullTextSession
    from launch_workspace import build_app,launch_options
    from workspace_ui import ASSISTANT
    from workspace_speech import ResponseSpeech
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
    counts=[];reader=session.reader.read;spoken=[];voice=ResponseSpeech.__call__
    def read(*a,**kw):counts.append(True);return reader(*a,**kw)
    def speech(self,text):spoken.append(text);return voice(self,text)
    session.reader.read=read;ResponseSpeech.__call__=speech
    demo,_=build_app(session.run_dir,session)
    for fn in demo.fns.values():
        if fn.fn and fn.fn.__name__=='upload_reset':fn.api_visibility='undocumented'
    records=[];checks={}
    def save():
        (out/'RESULTS.json').write_text(json.dumps({'records':records,'checks':checks,'raw_read_calls':len(counts),
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
        q='Tell me the cooking instructions mentioned in the image?'
        cooking=call('cooking-native-first',image_path,q,upload=True,mute=False)
        check('cooking-route-reading',cooking['details']['assistant_trace']['route']=='reading')
        check('cooking-partial-no-ingredient','Brown ground beef' in cooking['answer'] and 'Stir in hot water' in cooking['answer'] and '1Lb' not in cooking['answer'])
        original=json.loads(Path('reports/reading_sequences/run-20261020-v1/ORIGINAL_RAPID_RAW.json').read_text())['result']['evidence']
        check('exact-raw-evidence-unchanged',cooking['details']['evidence']==original)
        cleared=client.predict(q,api_name='/clear_submitted_question')
        clear_value=cleared[0].get('value') if isinstance(cleared[0],dict) else cleared[0]
        check('question-cleared-after-completion',clear_value=='')
        draft=client.predict('New question being edited',api_name='/clear_submitted_question')
        check('new-draft-not-cleared',isinstance(draft[0],dict) and 'value' not in draft[0])
        check('retake-restores-question',client.predict('',api_name='/reuse_last_question')==q)
        n=len(counts)
        follow=call('cooking-read-label',image_path,'Could you read the label?',mute=False)
        check('label-no-unnecessary-clarification','Brown ground beef' in follow['answer'] and 'multiple possible' not in follow['answer'])
        check('followup-no-native-ocr',len(counts)==n)
        every=call('cooking-all',image_path,'Read every visible word')
        check('all-retains-ingredient','1Lb' in every['answer'] and 'Brown ground beef' in every['answer'])
        ambiguity=call('two-passages',column_path,'Could you read the label?',upload=True)
        check('genuine-ambiguity-stored',ambiguity['details']['clarification_pending'] and len(ambiguity['details']['passage_candidates'])==2)
        n=len(counts)
        first=call('first-passage',column_path,'Start with first passage.',mute=False)
        check('first-resolved','first passage' in first['answer'] and 'second passage' not in first['answer'])
        ordinal=call('numeric-ordinal',column_path,'1st passage.')
        check('numeric-same-scope',ordinal['answer']==first['answer'])
        second=call('second-passage',column_path,'Second passage')
        check('second-resolved','second passage' in second['answer'] and 'first passage' not in second['answer'])
        rest=call('continue',column_path,'Continue')
        check('continue-no-loop','no additional' in rest['answer'] and 'multiple possible' not in rest['answer'])
        beginning=call('beginning',column_path,'Start from the beginning')
        check('beginning-active-scope',beginning['answer']==second['answer'])
        all_passages=call('all-passages',column_path,'Read all passages')
        check('all-passages-complete','first passage' in all_passages['answer'] and 'second passage' in all_passages['answer'])
        check('ordinal-reuses-evidence',len(counts)==n)
        isolated=call('isolated-browser',column_path,'First passage',c=Client(url,verbose=False))
        check('state-isolated','no stored passage' in isolated['answer'])
        client.predict(api_name='/reset_session')
        reset=call('reset-forgets',column_path,'Second passage')
        check('reset-forgets-passages','no stored passage' in reset['answer'])
        check('no-gpu-worker',session.legacy.client.process is None)
        check('all',all(checks.values()))
    finally:ResponseSpeech.__call__=voice;demo.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/reading_sequences/api-'+uuid4().hex[:8]);main(p.parse_args().out)
