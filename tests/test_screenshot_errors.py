import pytest
from PIL import Image
from reading_relevance import compose,intent
from full_text_session import FullTextSession
from test_reading_relevance import row
from test_workspace_callback_contract import callbacks,contract
from workspace_ui import ASSISTANT


@pytest.mark.parametrize('q',['What is the passage about?','What is this article about?','What is that document about?'])
def test_passage_about_is_extractive_ocr(q):
    assert intent(q)=='summary'
    assert object.__new__(FullTextSession).classify(q).task=='reading'
    r=compose(q,{'evidence':[row('A witness describes a light in the sky.',0,0),row('The report records the observations.',0,35)]})
    assert r['answer'].startswith('The passage describes') and r['supporting_evidence_ids']
    assert 'A witness describes' in r['answer']


def test_summary_no_evidence_uncertainty():
    r=compose('What is the passage about?',{'evidence':[]})
    assert 'cannot read sufficient' in r['answer'] and not r['supporting_evidence_ids']


def test_outer_error_preserves_history_and_clears_audio(tmp_path,monkeypatch):
    import image_conversation
    f,_,_=callbacks(tmp_path)
    state={'conversation':{'turns':[{'question':'Earlier question','answer':'First line\nSecond line'}]}}
    monkeypatch.setattr(image_conversation,'prepare',lambda *a:(_ for _ in ()).throw(ValueError('private')))
    r=list(f['analyze_browser'](Image.new('RGB',(100,100)),'Read text',True,ASSISTANT,'0',state))[-1]
    contract(r);assert r[1] is None and r[12]==state
    assert 'Earlier question' in r[13] and 'First line<br>Second line' in r[13]


def test_worker_guard_error_not_hidden(tmp_path):
    f,base,_=callbacks(tmp_path)
    base.legacy.request=lambda *a:(_ for _ in ()).throw(RuntimeError('Session artifact changed; restart and reverify: private-path'))
    r=list(f['analyze_browser'](Image.new('RGB',(100,100)),'What is in the image?',True,ASSISTANT,'0',None))[-1]
    contract(r);assert r[3]['application_error']['code']=='worker_source_changed'
    assert 'Restart VisionBridge' in r[2] and r[1] is None and 'private-path' not in r[2]


def test_explicit_microphone_probe_releases_tracks():
    from workspace_voice import MICROPHONE_CHECK_JS
    assert 'getUserMedia' in MICROPHONE_CHECK_JS and 'track.stop()' in MICROPHONE_CHECK_JS
    for code in ('NotAllowedError','NotFoundError','NotReadableError'):assert code in MICROPHONE_CHECK_JS
    assert 'video:false' in MICROPHONE_CHECK_JS and 'MediaRecorder' not in MICROPHONE_CHECK_JS
