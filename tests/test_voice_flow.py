"""Real event callbacks: late results, resets, retries and explicit insertion."""
import wave
import math
import struct

from launch_workspace import build_app
from test_image_conversation import fake_base


def setup_flow(tmp_path,monkeypatch):
    calls=[]
    def transcribe(path):
        calls.append(path)
        return 'What is in the image?', 'Ready'
    monkeypatch.setattr('workspace_voice.transcribe_local',transcribe)
    base,_=fake_base(tmp_path,[])
    base.close=lambda:None
    demo,_=build_app(tmp_path,base)
    demo.queue().validate_queue_settings()
    handlers={f.fn.__name__:f.fn for f in demo.fns.values() if f.fn}
    audio=tmp_path/'question.wav'
    with wave.open(str(audio),'wb') as stream:
        stream.setparams((1,2,16000,0,'NONE','not compressed'))
        stream.writeframes(b''.join(struct.pack('<h',int(4000*math.sin(i/8))) for i in range(16000)))
    return demo,handlers,calls,str(audio)


def test_recording_review_never_automatically_replaces_question(tmp_path,monkeypatch):
    demo,f,calls,audio=setup_flow(tmp_path,monkeypatch)
    job=f['begin'](audio)[0]
    result=f['run'](job,None)
    committed=f['finish'](job,result)
    assert committed[0]['value']=='What is in the image?'
    assert committed[0]['interactive']
    assert committed[3]['interactive'] and committed[4]['interactive']
    assert f['use_question']('  My reviewed question  ')[0]=='My reviewed question'
    components={c['props'].get('elem_id'):c['id'] for c in demo.config['components']}
    finish=next(d for d in demo.config['dependencies'] if d.get('backend_fn') and
        demo.fns[d['id']].fn.__name__=='finish')
    assert components['vb-question'] not in finish['outputs']
    # An explicit use click is the only new UI insertion action.
    assert f['use_question']('')[0]=={'__type__':'update'}


def test_late_result_is_ignored_after_clear_or_new_recording(tmp_path,monkeypatch):
    _,f,_,audio=setup_flow(tmp_path,monkeypatch)
    old=f['begin'](audio)[0]
    result=f['run'](old,None)
    fresh=f['reset_voice']()[0]
    assert all(item=={'__type__':'update'} for item in f['finish'](fresh,result))
    newer=f['begin'](audio)[0]
    assert newer['id']!=old['id']
    assert all(item=={'__type__':'update'} for item in f['finish'](newer,result))
    cleared=f['discard']()
    assert cleared[0]['value'] is None and cleared[0]['recording'] is False
    assert cleared[2]['value']=='' and not cleared[2]['interactive']
    assert cleared[-1] is None


def test_retries_reuse_validated_audio_and_failure_disables_use(tmp_path,monkeypatch):
    _,f,calls,audio=setup_flow(tmp_path,monkeypatch)
    job=f['begin'](audio)[0]
    result=f['run'](job,None)
    again=f['run'](f['begin'](audio)[0],result['cache'])
    assert again['text']==result['text'] and len(calls)==1
    # Gradio can emit a temporary empty value before file upload finishes.
    missing=f['begin'](None)[0]
    result=f['run'](missing,None)
    assert result['pending']
    assert all(item=={'__type__':'update'} for item in f['finish'](missing,result))
    assert not f['begin'](None)[3]['interactive']
    starting=f['starting']()
    assert not starting[3]['interactive'] and not starting[4]['interactive']
