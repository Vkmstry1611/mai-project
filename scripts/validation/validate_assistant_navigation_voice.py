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
    import copy
    raw=json.loads(Path('reports/reading_sequences/run-20261020-v1/ORIGINAL_RAPID_RAW.json').read_text(encoding='utf-8'))
    counts=[];spoken=[];voice=ResponseSpeech.__call__
    def read(*a,**kw):counts.append(True);return copy.deepcopy(raw)
    def speech(self,text):spoken.append(text);return voice(self,text)
    session.reader.read=read;ResponseSpeech.__call__=speech
    frozen=json.loads(Path('reports/ocr_purpose/run-20261021-v1/api/RESULTS.json').read_text(encoding='utf-8'))['records'][0]
    def saved_vqa(*a,**kw):
        return (frozen['answer'],None,'Saved exact frozen fixture replay',copy.deepcopy(frozen['details']),None,'',copy.deepcopy(frozen['timings']))
    session.legacy.request=saved_vqa
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
        for name,q,term in [('first','Read only the first cooking step','Brown'),('second','Read the second cooking step','Stir'),
            ('third','Read the third step','Reduce'),('previous','Read the previous step','Stir'),('repeat','Repeat that step','Stir'),('next','Read the next step','Reduce')]:
            r=call(name,image_path,q,upload=name=='first',mute=False)
            check(name+'-scoped',term in r['answer'] and len(r['details']['selected_hierarchy_ids'])==1)
        purpose=call('intervening-purpose',image_path,'What is the image about?')
        check('purpose-supported','cooking' in purpose['answer'])
        previous=call('previous-after-purpose',image_path,'Read the previous step')
        check('cursor-preserved','Stir' in previous['answer'] and 'Brown' not in previous['answer'])
        ingredient=call('ingredients',image_path,'What ingredients are mentioned?',mute=False)
        check('ingredients-supported',ingredient['details']['ingredient_evidence']['preparation_mentions']==['ground beef','hot water','milk','Sauce Mix','uncooked Pasta'])
        for word in ('milk','pasta'):
            r=call('mention-'+word,image_path,'Does it mention '+word+'?',mute=False)
            check('mention-'+word+'-yes',r['answer'].startswith('Yes.'))
        second=call('second-after-ingredients',image_path,'Read the second cooking step')
        check('no-blip-for-ingredients','Stir' in second['answer'])
        check('one-ocr-read',len(counts)==1)
        speaker=ResponseSpeech();audio_path=speaker('What is written in this image?')
        before=time.perf_counter()
        recognized=client.predict(handle_file(audio_path),api_name='/transcribe_validated')
        check('real-whisper',bool(recognized[0].strip()) and 'written' in recognized[0].lower())
        print('Whisper',repr(recognized[0]),round(time.perf_counter()-before,3),flush=True)
        (out/'WHISPER.json').write_text(json.dumps({'transcript':recognized[0],'status':recognized[1],'synthetic_audio_path':str(audio_path),'physical_microphone':False},indent=2))
        again=client.predict(handle_file(audio_path),api_name='/transcribe_validated')
        check('whisper-cache',again[0]==recognized[0] and 'reused' in again[1])
        inserted=client.predict('new typed draft','older draft',recognized[0],api_name='/insert_reviewed_transcript')
        check('new-draft-preserved',not isinstance(inserted[0],str) or inserted[0]=='new typed draft')
        missing=client.predict(None,api_name='/transcribe_validated')
        check('empty-audio-error','No recording' in missing[1])
        client.predict(api_name='/reset_session')
        cleared=call('reset-previous',image_path,'Read the previous step')
        check('reset-clears-cursor','do not have a current step' in cleared['answer'])
        check('no-gpu-worker',session.legacy.client.process is None)
        check('all',all(checks.values()))
    finally:ResponseSpeech.__call__=voice;demo.close();session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='reports/assistant_navigation_voice/api-'+uuid4().hex[:8]);main(p.parse_args().out)
