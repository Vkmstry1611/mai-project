"""Exercise real workspace event functions, including upload-created state."""
import json
import pytest
from PIL import Image
from test_image_conversation import fake_base
from visionbridge.workspace_ui import ASSISTANT, VISUAL
from launch_workspace import build_app


def callbacks(tmp_path, rows=()):
    base, calls = fake_base(tmp_path, list(rows))
    base.close = lambda: None
    demo, _ = build_app(tmp_path, base)
    functions = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}
    return functions, base, calls


def contract(value):
    assert len(value) == 14
    for index in (0, 2, 5, 7, 9, 10, 13):
        assert isinstance(value[index], str)
    for index in (3, 6, 8, 11):
        assert isinstance(value[index], dict)
    assert value[1] is None or isinstance(value[1], str)
    assert value[4] is None or isinstance(value[4], Image.Image)
    json.dumps(value[3]); json.dumps(value[6]); json.dumps(value[11])


@pytest.mark.parametrize('question', ['What is written in the image?', 'What is in the image?', 'Help', ''])
def test_upload_then_first_analysis_contract(tmp_path, question):
    f, _, _ = callbacks(tmp_path)
    image = Image.new('RGB', (320, 240), 'gray')
    uploaded = f['upload_reset'](image, None)
    contract(uploaded)
    results = list(f['analyze_browser'](image, question, True, ASSISTANT, '0', uploaded[12]))
    for result in results: contract(result)
    assert results[-1][8]['interactive']


def test_followup_new_image_reset_and_retake_contract(tmp_path):
    f, _, _ = callbacks(tmp_path)
    image = Image.new('RGB', (320, 240), 'gray')
    state = f['upload_reset'](image, None)[12]
    for question in ['Read the text', 'Read the rest', 'What should I do to make it clearer?']:
        result = list(f['analyze_browser'](image, question, True, ASSISTANT, '0', state))[-1]
        contract(result); state = result[12]
    changed = f['upload_reset'](Image.new('RGB', (320, 240), 'blue'), state)
    contract(changed)
    assert not changed[12]['conversation']['turns']
    result = list(f['analyze_browser'](None, 'Read this', True, ASSISTANT, '0', None))[-1]
    contract(result)


@pytest.mark.parametrize('ocr',[False,True])
def test_worker_failure_contract(tmp_path,ocr):
    f, base, _ = callbacks(tmp_path)
    def fail(*args): raise RuntimeError('Injected worker failure')
    base.legacy.request = fail
    base.reader.read=fail; base.easy.read=fail
    image = Image.new('RGB', (320, 240), 'gray')
    result = list(f['analyze_browser'](image, 'Read text' if ocr else 'What is this?', True, ASSISTANT if ocr else VISUAL, '0', f['upload_reset'](image, None)[12]))[-1]
    contract(result)
    assert not result[0] and 'No substitute answer' in result[2]


def test_reset_and_manual_mode_callback_contracts(tmp_path):
    base,_=fake_base(tmp_path,[]);base.close=lambda:None
    demo,_=build_app(tmp_path,base)
    handlers={f.api_name:f.fn for f in demo.fns.values() if f.fn}
    cleared=handlers['reset_session']()
    assert len(cleared)==16 and cleared[0] is None and cleared[1]==''
    contract(cleared[2:]);assert cleared[14] is None
    for mode in (ASSISTANT,VISUAL,'Experimental OCR Reading','Full Text Reading (experimental)'):
        changed=handlers['set_mode'](mode)
        assert len(changed)==16
        contract(changed[2:])
        assert changed[0]['visible']==('Reading' in mode)


@pytest.mark.parametrize('reading,abstain', [(True,False),(True,True),(False,False),(False,True)])
def test_success_abstention_cache_and_speech_failure(tmp_path,monkeypatch,caplog,reading,abstain):
    from test_image_conversation import row
    from visionbridge.workspace_speech import ResponseSpeech
    f,base,calls=callbacks(tmp_path, [] if abstain else [row('HELLO')])
    if not reading and not abstain:
        base.legacy.request=lambda *a:('A cup',None,'Complete',{'baseline_answer':'cup','final_answer':'cup','answer_source':'blip','evidence':[]},None,'',{'callback_seconds':0.})
    def fail(self,text):raise RuntimeError('PRIVATE OCR TEXT must not be logged')
    monkeypatch.setattr(ResponseSpeech,'__call__',fail)
    image=Image.new('RGB',(320,240),'gray');state=f['upload_reset'](image,None)[12]
    question='Read text' if reading else 'What is in the image?'
    first=list(f['analyze_browser'](image,question,False,ASSISTANT,'0',state))
    for value in first:contract(value)
    result=first[-1]
    assert result[0] and result[1] is None and 'text answer is still available' in result[2]
    assert 'exception=RuntimeError' in caplog.text and 'PRIVATE OCR TEXT' not in caplog.text
    count=len(calls)
    repeated=list(f['analyze_browser'](image,question,True,ASSISTANT,'0',result[12]))[-1]
    contract(repeated);assert repeated[0]==result[0] and len(calls)==count
    import copy
    copy.deepcopy(repeated[12])  # Gradio State is server-side and deepcopy-able.
    json.dumps(repeated[12]['conversation']['ocr'])


def test_outer_callback_failure_is_logged_not_abstention(tmp_path,monkeypatch,caplog):
    from visionbridge import image_conversation
    f,_,_=callbacks(tmp_path)
    def fail(*a):raise ValueError('PRIVATE IMAGE CONTENT')
    monkeypatch.setattr(image_conversation,'prepare',fail)
    result=list(f['analyze_browser'](Image.new('RGB',(100,100)),'Read text',True,ASSISTANT,'0',None))[-1]
    contract(result)
    assert result[3]['application_error']['stage']=='workspace-callback'
    assert 'Application error' in result[2] and 'exception=ValueError' in caplog.text
    assert 'PRIVATE IMAGE CONTENT' not in caplog.text
    uploaded=f['upload_reset'](Image.new('RGB',(100,100)),None)
    contract(uploaded)
    assert uploaded[12] is None and uploaded[3]['application_error']['stage']=='image-upload-state'
