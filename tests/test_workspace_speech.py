import wave
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from visionbridge.workspace_speech import chunks,ResponseSpeech
from visionbridge.workspace_ui import WorkspaceController,ASSISTANT


def test_chunks_preserve_nonwhitespace_and_limit():
    text=('A sentence. Another sentence!\nA new line. '*100)+'X'*2400
    parts=chunks(text)
    assert all(0<len(x)<=1200 for x in parts)
    assert ''.join(text.split())==''.join(''.join(parts).split())


def test_long_audio_joins_in_order(tmp_path):
    calls=[]
    def synth(text,cfg):
        calls.append(text);p=cfg.output_dir/f'{len(calls)}.wav'
        with wave.open(str(p),'wb') as f:f.setparams((1,2,8000,0,'NONE','not compressed'));f.writeframes(bytes([len(calls),0])*10)
        return str(p)
    speech=ResponseSpeech(synth)
    try:
        path=speech('One. '*600)
        with wave.open(path) as f:
            data=f.readframes(f.getnframes());assert data[:2]==bytes([1,0]);assert data[-2:]==bytes([len(calls),0])
        assert len(calls)>1
    finally:speech.close()


@pytest.mark.parametrize('question',['What color is this?','Read this','Help','What is the expiry date?'])
def test_one_request_text_then_audio_and_no_backend_duplicate(question):
    calls=[];spoken=[]
    def request(*a):
        calls.append(a);return ('test response',None,'Complete',{'final_answer':'test response'},None,'',{})
    c=WorkspaceController(SimpleNamespace(request=request));c.response_speech=lambda t:spoken.append(t) or 'answer.wav'
    result=list(c.analyze_ui(Image.new('RGB',(10,10)),question))
    assert result[-1][0] and result[-1][1]=='answer.wav'
    assert len(spoken)==1
    assert all(a[2] is False for a in calls)
    assert any(x[0]==result[-1][0] and x[1] is None for x in result[:-1])


def test_mute_and_failure_keep_text():
    c=WorkspaceController(SimpleNamespace(request=lambda *a:('unanswerable',None,'Complete',{},None,'',{})))
    def fail(t):raise RuntimeError('voice unavailable')
    c.response_speech=fail
    out=list(c.analyze_ui(Image.new('RGB',(10,10)),'What is this?',True))[-1]
    assert out[0]=='unanswerable' and out[1] is None and 'voice unavailable' not in out[2]
    out=list(c.analyze_ui(Image.new('RGB',(10,10)),'What is this?'))[-1]
    assert out[0]=='unanswerable' and out[1] is None and 'Speech generation failed' in out[2]
