import base64
from io import BytesIO

from PIL import Image
import pytest

from image_conversation import prepare
from modules import local_vision


def response(text='Answer from the supplied image.'):
    return {'message': {'content': text}, 'done_reason': 'stop',
            'eval_duration': 1000000000}


@pytest.mark.parametrize('question', ['What is the kcal per serving?',
    'Explain this circuit.', 'Translate this sign.', 'Compare the two charts.',
    'What is the colour of this object?', 'What does this medical term mean?'])
def test_questions_reach_model_with_image_without_keyword_rejection(monkeypatch,question):
    calls=[]
    monkeypatch.setattr(local_vision,'post_json',lambda path,payload: (calls.append(payload) or response()))
    session=local_vision.LocalVisionSession()
    result=session.request(Image.new('RGB',(100,70),'blue'),question)
    message=calls[0]['messages'][-1]
    assert message['content']==question
    image=Image.open(BytesIO(base64.b64decode(message['images'][0])))
    assert image.size==(100,70)
    assert result[0]=='Answer from the supplied image.'
    assert result[3]['answer_source']=='local_vision_model'
    assert not result[3]['supporting_evidence_ids']
    assert session.classify(question).task=='vqa'


def test_followups_are_scoped_to_current_image_and_browser(monkeypatch):
    calls=[]
    monkeypatch.setattr(local_vision,'post_json',lambda path,payload: (calls.append(payload) or response()))
    session=local_vision.LocalVisionSession()
    image=Image.new('RGB',(100,70),'blue')
    state=prepare(None,image)
    session.for_conversation(state,image).request(image,'Describe this.')
    session.for_conversation(state,image).request(image,'Explain that.')
    assert calls[-1]['messages'][1]['content']=='Describe this.'
    assert calls[-1]['messages'][2]['content']=='Answer from the supplied image.'
    different=Image.new('RGB',(100,70),'red')
    fresh=prepare(state,different)
    session.for_conversation(fresh,different).request(different,'What is this?')
    assert len(calls[-1]['messages'])==2
    other_browser=prepare(None,image)
    session.for_conversation(other_browser,image).request(image,'What is this?')
    assert len(calls[-1]['messages'])==2


def test_model_keep_alive_can_be_disabled_for_low_memory_bda_host(monkeypatch):
    calls = []
    monkeypatch.setenv('VISIONBRIDGE_OLLAMA_KEEP_ALIVE', '0')
    monkeypatch.setattr(local_vision, 'post_json',
                        lambda path, payload: (calls.append(payload) or response()))
    local_vision.LocalVisionSession().request(Image.new('RGB', (20, 20)), 'What is shown?')
    assert calls[0]['keep_alive'] == '0'


def test_empty_response_is_application_error_not_model_abstention(monkeypatch):
    monkeypatch.setattr(local_vision,'post_json',lambda *args:response(''))
    with pytest.raises(local_vision.LocalVisionError,match='empty response'):
        local_vision.LocalVisionSession().request(Image.new('RGB',(20,20)),'Read this')


def test_auto_backend_uses_local_when_frozen_runtime_absent(tmp_path):
    from launch_workspace import create_session
    assert isinstance(create_session(tmp_path),local_vision.LocalVisionSession)
    with pytest.raises(RuntimeError,match='Frozen backend is incomplete'):
        create_session(tmp_path,backend='frozen')


def test_reasoning_budget_exhaustion_retries_without_exposing_thinking(monkeypatch):
    budgets=[]
    def ask(path,payload):
        budgets.append(payload['options']['num_predict'])
        if len(budgets)==1:
            return {'message':{'content':'','thinking':'internal reasoning'},'done_reason':'length'}
        return response('109.8 kcal per serving.')
    monkeypatch.setattr(local_vision,'post_json',ask)
    result=local_vision.LocalVisionSession().request(Image.new('RGB',(20,20)),'How many kcal?')
    assert result[0]=='109.8 kcal per serving.'
    assert budgets==[2048,4096]
    assert 'internal reasoning' not in str(result)
