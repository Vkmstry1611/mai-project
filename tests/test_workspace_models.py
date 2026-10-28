"""Model choices route through existing callbacks without sharing browser state."""
from types import SimpleNamespace

from PIL import Image
import pytest

from launch_workspace import build_app
from test_image_conversation import fake_base
from visionbridge.workspace_intent import Route
from visionbridge.workspace_models import BLIP, QWEN, FrontendModels
from visionbridge.workspace_ui import ASSISTANT


def test_lazy_models_validate_choices_and_close_once():
    created = []
    closed = []

    def factory(name):
        created.append(name)
        return SimpleNamespace(close=lambda: closed.append(name))

    models = FrontendModels(factory)
    assert models.get(BLIP) is models.get(BLIP)
    assert created == [BLIP]
    with pytest.raises(ValueError):
        models.get('unknown')
    models.get(QWEN)
    models.close()
    models.close()
    assert closed == [BLIP, QWEN]


def test_selector_isolates_models_and_clears_context_but_preserves_inputs(tmp_path):
    blip, _ = fake_base(tmp_path, [])
    blip.close = lambda: None
    blip.legacy.request = lambda *args: ('bus', None, 'Complete',
        {'answer_source': 'blip', 'evidence': []}, None, '', {})
    qwen = SimpleNamespace(close=lambda: None)
    qwen.classify = lambda q: Route('vqa', 'Direct image question')
    qwen.for_conversation = lambda state, image: qwen
    qwen.request = lambda *args: ('A blue bus', None, 'Complete',
        {'answer_source': 'local_vision_model', 'evidence': []}, None, '', {})
    models = FrontendModels(lambda name: blip if name == BLIP else qwen)
    demo, _ = build_app(tmp_path, blip, models)
    handlers = {f.api_name: f.fn for f in demo.fns.values() if f.fn}
    functions = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}
    components = {c['id']: c for c in demo.config['components']}
    by_elem = {c['props'].get('elem_id'): c for c in demo.config['components']}
    assert by_elem['vb-model-choice']['type'] == 'dropdown'
    assert by_elem['vb-photo']['id'] < by_elem['vb-question']['id'] < by_elem['vb-analyze']['id'] < by_elem['vb-answer']['id']
    assert by_elem['vb-answer']['id'] < by_elem['vb-audio-controls']['id']
    assert {'vb-contrast', 'vb-pause-answer', 'vb-resume-answer', 'vb-stop-answer'} <= by_elem.keys()
    select = next(d for d in demo.config['dependencies'] if d['api_name'] == 'select_model')
    # A selection never writes to the image or question components.
    assert not {'vb-photo', 'vb-question'} & {
        components[i]['props'].get('elem_id') for i in select['outputs']}
    analyze = next(d for d in demo.config['dependencies'] if d['api_name'] == 'analyze')
    assert len(analyze['inputs']) == 6  # Existing public API contract.
    photo = Image.new('RGB', (320, 240), 'gray')
    selected = handlers['select_model'](QWEN)
    assert len(selected) == 17
    state = selected[12]
    assert state == {'workspace_model': QWEN}
    assert selected[-3]['value'] == ASSISTANT
    assert selected[-1]['visible'] is False
    uploaded = functions['upload_reset'](photo, state)[12]
    result = list(handlers['analyze'](photo, 'What is in the image?', True, ASSISTANT, '0', uploaded))[-1]
    assert result[0] == 'A blue bus'
    assert result[12]['workspace_model'] == QWEN
    # A second browser with no selection still uses the launch default.
    other = list(handlers['analyze'](photo, 'What is in the image?', True, ASSISTANT, '0', None))[-1]
    assert other[0] == 'bus'
    cleared = handlers['reset_session'](QWEN)
    assert cleared[14] == {'workspace_model': QWEN}
    assert 'conversation' not in cleared[14]
    back = handlers['select_model'](BLIP)
    assert back[12] == {'workspace_model': BLIP}
    assert back[-1]['visible'] is True
    assert back[0] == '' and back[3] == {} and back[4] is None
    models.close()


def test_cloud_default_champion_can_switch_to_qwen(tmp_path):
    qwen = SimpleNamespace(close=lambda: None, backend_label='Cloud Qwen')
    qwen.for_conversation = lambda state, image: qwen
    qwen.classify = lambda question: Route('vqa', 'Cloud vision')
    qwen.request = lambda *args: ('Qwen answer', None, 'Complete',
        {'answer_source': 'local_vision_model', 'evidence': []}, None, '', {})
    blip = SimpleNamespace(close=lambda: None)
    blip.classify = lambda question: Route('vqa', 'Cloud BLIP')
    blip.request = lambda *args: ('stop', None, 'Complete',
        {'answer_source': 'dvc_blip', 'evidence': []}, None, '', {})
    models = FrontendModels(lambda name: blip if name == BLIP else qwen,
                            default=BLIP, initial=blip)
    demo, _ = build_app(tmp_path, qwen, models, cloud_mode=True)
    handlers = {f.api_name: f.fn for f in demo.fns.values() if f.fn}
    photo = Image.new('RGB', (320, 240), 'gray')
    result = list(handlers['analyze'](photo, 'What word is on the sign?',
                        True, ASSISTANT, '0', None))[-1]
    assert result[0] == 'stop'
    assert result[12]['conversation']['image_id']
    selected = handlers['select_model'](QWEN)
    switched = list(handlers['analyze'](photo, 'What word is on the sign?',
                        True, ASSISTANT, '0', selected[12]))[-1]
    assert switched[0] == 'Qwen answer'
    models.close()
