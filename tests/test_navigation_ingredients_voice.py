import json
from pathlib import Path
import pytest
from PIL import Image
from visionbridge.image_conversation import prepare,ConversationSession,resolve
from visionbridge.reading_relevance import compose
from test_image_conversation import fake_base,turn
from test_text_hierarchy import evidence
from test_reading_relevance import row
from visionbridge.workspace_voice import VoiceReview,insert_transcript


def test_acceptance_navigation(tmp_path):
    base,calls=fake_base(tmp_path,evidence());im=Image.new('RGB',(800,600));state=prepare(None,im)
    for q,expected in [('Read only the first cooking step','Brown'),('Read the second cooking step','Stir'),
                       ('Read the third step','Reduce'),('Read the previous step','Stir'),('Repeat that step','Stir'),('Read the next step','Reduce')]:
        r=turn(base,state,im,q);assert expected in r[0]
        assert len(r[3]['selected_hierarchy_ids'])==1
    # A purpose/general turn cannot erase the stable step cursor.
    turn(base,state,im,'What is the image about?')
    # Repetition recalls the last spoken step; navigation moves the cursor.
    assert 'Reduce' in turn(base,state,im,'Repeat the previous step')[0]
    assert 'Stir' in turn(base,state,im,'Go back one step')[0]
    assert 'Brown' in turn(base,state,im,'Go back one step')[0]
    facade=ConversationSession(base,state,im)
    assert facade.classify('Previous step').task=='clarify'
    assert 'first' in resolve('Previous step',state['conversation'])[1]
    assert calls.count('ocr')==1
    fresh=prepare(None,im)
    assert ConversationSession(base,fresh,im).classify('Next step').task=='clarify'


def test_continue_and_all(tmp_path):
    base,_=fake_base(tmp_path,evidence());im=Image.new('RGB',(800,600));state=prepare(None,im)
    turn(base,state,im,'Read the second step')
    answer=turn(base,state,im,'Continue from here')[0]
    assert 'Stir' in answer and 'Reduce' in answer and 'Brown' not in answer
    assert 'Brown' in turn(base,state,im,'Read all steps')[0]
    assert 'Brown' in turn(base,state,im,'Start from the first step')[0]
    assert 'Brown' not in turn(base,state,im,'Read the remaining steps')[0]


@pytest.mark.parametrize('q',['What ingredients are mentioned?','Which ingredients are listed?','What do I need to prepare this?',
    'What ingredients are used in the instructions?','Read the ingredient list.','Does it mention milk?','Is pasta mentioned?'])
def test_ingredient_route(q):
    from visionbridge.full_text_session import FullTextSession
    assert object.__new__(FullTextSession).classify(q).task=='reading'


def test_exact_saved_ingredient_evidence():
    d=json.loads(Path('reports/reading_sequences/run-20261020-v1/ORIGINAL_RAPID_RAW.json').read_text(encoding='utf-8'))['result']
    r=compose('What ingredients are mentioned?',d)
    assert r['ingredient_evidence']['preparation_mentions']==['ground beef','hot water','milk','Sauce Mix','uncooked Pasta']
    assert 'not the complete' in r['answer']
    assert 'Yes.' in compose('Does it mention milk?',d)['answer']
    assert 'Yes.' in compose('Is pasta mentioned?',d)['answer']
    assert 'does not establish' in compose('Does it mention peanuts?',d)['answer']
    assert 'cannot verify a formal' in compose('Read the ingredient list',d)['answer']


def test_incomplete_and_formal():
    assert not compose('Does it mention milk?',{'evidence':[row('milk',0,0,score=.4)]})['supporting_evidence_ids']
    r=compose('Read the ingredient list',{'evidence':[row('Ingredients: oats, raisins',0,0)]})
    assert 'Recognized ingredients section' in r['answer'] and 'oats' in r['answer']
    assert 'cannot reliably extract' in compose('What ingredients?',{'evidence':[]})['answer']


def wav(tmp_path,silent=False):
    import numpy as np
    import wave
    p=tmp_path/'voice.wav';t=np.arange(16000)/16000
    with wave.open(str(p),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(((np.zeros(16000) if silent else .1*np.sin(2*np.pi*400*t))*32767).astype('<i2').tobytes())
    return str(p)


def test_voice_valid_cached_draft(tmp_path):
    calls=[];v=VoiceReview(lambda p:(calls.append(p) or 'Read the text','Ready'))
    text,status,state=v.transcribe(wav(tmp_path));assert text=='Read the text'
    assert v.transcribe(wav(tmp_path),state)[0]==text and len(calls)==1
    assert insert_transcript('new draft','old draft',text)[0] is None
    assert insert_transcript('same','same',text)[0]==text


def test_voice_errors(tmp_path):
    v=VoiceReview(lambda p:('', 'No speech'))
    assert 'No recording' in v.transcribe(None)[1]
    assert 'No audible signal' in v.transcribe(wav(tmp_path,True))[1]
    bad=tmp_path/'bad.wav';bad.write_text('bad')
    assert 'format' in v.transcribe(str(bad))[1]
    assert 'no text' in v.transcribe(wav(tmp_path))[1]
    v=VoiceReview(lambda p:(_ for _ in ()).throw(RuntimeError('private payload')))
    assert 'private payload' not in v.transcribe(wav(tmp_path))[1]


def test_voice_zero_frames_missing_file_and_session_dedup(tmp_path):
    import wave
    p=tmp_path/'empty.wav'
    with wave.open(str(p),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
    calls=[];v=VoiceReview(lambda p:(calls.append(True) or 'question','ok'))
    assert 'Empty' in v.transcribe(str(p))[1]
    assert 'No captured audio' in v.transcribe(str(tmp_path/'missing.wav'))[1]
    path=wav(tmp_path)
    v.transcribe(path,None,'session-a');v.transcribe(path,None,'session-a')
    assert len(calls)==1
    v.transcribe(path,None,'session-b');assert len(calls)==2


def test_voice_event_wiring_and_clear_cancel(tmp_path):
    from launch_workspace import build_app
    base,_=fake_base(tmp_path,[]);base.close=lambda:None
    demo,_=build_app(tmp_path,base)
    try:
        demo.queue(default_concurrency_limit=1).validate_queue_settings()
        config=demo.get_config_file();components={c['id']:c for c in config['components']}
        rec=next(c for c in components.values() if c['type']=='audio' and c['props'].get('sources')==['microphone','upload'])
        assert rec['props']['format']=='wav'
        assert any(tuple(t)==(rec['id'],'input') for d in config['dependencies'] for t in d['targets'])
        assert any(d.get('cancels') for d in config['dependencies'])
    finally:demo.close()
