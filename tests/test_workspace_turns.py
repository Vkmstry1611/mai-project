"""Current text/transcript editing must not become conversation history."""
from pathlib import Path
from PIL import Image
from workspace_turns import question_after_submission,question_for_retake
from test_workspace_callback_contract import callbacks,contract
from workspace_ui import ASSISTANT,WorkspaceController


def test_clear_completed_only_and_preserve_new_edits():
    state={'conversation':{'image_id':'image-a'},'submission':{'completed':True,'image_id':'image-a','question':'Read this label'},'last_submitted_question':'Read this label'}
    assert question_after_submission('Read this label',state)==''
    assert question_after_submission('What color is it?',state) is None
    state['submission']['completed']=False
    assert question_after_submission('Read this label',state) is None
    state['submission']['completed']=True;state['conversation']['image_id']='image-b'
    assert question_after_submission('Read this label',state) is None
    assert question_for_retake('',state)=='Read this label'
    assert question_for_retake('New typed question',state)=='New typed question'
    assert question_for_retake('',None)==''


def test_separate_typed_turns_clear_without_losing_evidence(tmp_path):
    from test_image_conversation import row
    f,base,calls=callbacks(tmp_path,[row('HELLO')]);image=Image.new('RGB',(320,240))
    state=f['upload_reset'](image,None)[12]
    for q in ['Read the label','Read every visible word']:
        result=list(f['analyze_browser'](image,q,True,ASSISTANT,'0',state))[-1];contract(result);state=result[12]
        assert f['clear_submitted_question'](q,state)[0]['value']==''
        assert 'value' not in f['clear_submitted_question']('Different next question',state)[0]
    assert [t['question'] for t in state['conversation']['turns']]==['Read the label','Read every visible word']
    assert calls==['ocr'] and f['reuse_last_question']('',state)=='Read every visible word'


def test_failed_submission_retains_current_question(tmp_path):
    f,_,_=callbacks(tmp_path);image=Image.new('RGB',(320,240))
    result=list(f['analyze_browser'](None,'Read text',True,ASSISTANT,'0',None))[-1]
    assert not result[12]['submission']['completed']
    assert 'value' not in f['clear_submitted_question']('Read text',result[12])[0]


def test_microphone_transcript_replaces_current_text_and_can_be_edited(tmp_path):
    from types import SimpleNamespace
    c=WorkspaceController(SimpleNamespace(close=lambda:None),transcriber=lambda _:('Read the cooking instructions','Review your transcript'))
    result=list(c.transcribe('synthetic-recording.wav'))[-1]
    assert result[0]=='Read the cooking instructions'
    assert 'previous question' not in result[0]


def test_clear_event_follows_first_text_and_clicks_do_not_duplicate(tmp_path):
    from test_image_conversation import fake_base
    from launch_workspace import build_app
    base,_=fake_base(tmp_path,[]);base.close=lambda:None
    demo,_=build_app(tmp_path,base)
    answer=next(c['id'] for c in demo.config['components'] if c['props'].get('elem_id')=='vb-answer')
    clear=next(d for d in demo.config['dependencies'] if d.get('api_name')=='clear_submitted_question')
    assert [answer,'change'] in [list(t) for t in clear['targets']]
    analyze=next(d for d in demo.config['dependencies'] if d.get('api_name')=='analyze')
    assert analyze['trigger_mode']=='once' and len(analyze['outputs'])==14
