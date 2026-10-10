import importlib
from unittest.mock import patch
from PIL import Image
from pipeline import Pipeline
from modules.qges import select
from utils.image_utils import normalize_image

def test_independent_imports():
    for name in ['vqa', 'speech_to_text', 'text_to_speech', 'object_detection', 'ocr', 'visual_features', 'qges', 'fusion', 'answerability']:
        importlib.import_module('modules.' + name)

def test_validation():
    pipeline = Pipeline()
    assert 'upload' in pipeline.run(None, 'what?')[2]
    assert 'question' in pipeline.run(Image.new('RGB', (20, 20)), '')[2]

def test_baseline_and_missing_checkpoints():
    pipeline = Pipeline()
    pipeline.config.enable_trained = True
    with patch('modules.vqa.answer', return_value='bottle'):
        answer, audio, status, debug, image = pipeline.run(Image.new('RGB', (30, 30)), 'what is here?', speak=False, enhanced=True)
    assert answer == 'bottle'
    assert audio is None and image is not None
    assert debug['grounded'] is False and debug['answerability']['probability'] is None
    assert len(debug['warnings']) == 3

def test_optional_failures_and_tts_failure():
    with patch('modules.vqa.answer', return_value='bottle'), patch('modules.ocr.extract', side_effect=RuntimeError('offline')), patch('modules.text_to_speech.synthesize', side_effect=RuntimeError('voice missing')):
        result = Pipeline().run(Image.new('RGB', (30, 30)), 'what?', use_ocr=True)
    assert 'bottle' in result[0] and result[1] is None
    assert 'offline' in result[3]['components']['ocr']
    assert result[3]['warnings']

def test_qges_and_image():
    assert select('bottle', [{'text': 'cat'}, {'text': 'bottle'}])[0]['text'] == 'bottle'
    assert normalize_image(Image.new('RGBA', (10, 20))).mode == 'RGB'

def test_cpu_fallback():
    from utils.model_loader import resolve_device
    resolve_device.cache_clear()
    with patch('torch.cuda.is_available', return_value=False):
        assert resolve_device('cuda') == 'cpu'

def test_gate_prevents_vqa():
    from modules.answerability import GateResult
    pipeline = Pipeline()
    pipeline.config.enable_trained = True
    class Gate:
        def predict(self, inputs):
            return GateResult(False, 0.1, 'test fixture')
    pipeline.adapters['answerability'] = Gate()
    with patch('modules.vqa.answer') as model:
        result = pipeline.run(Image.new('RGB', (20, 20)), 'what?', speak=False, enhanced=True)
    model.assert_not_called()
    assert 'cannot answer' in result[0]

def test_ui_build():
    from app import build_app
    assert build_app() is not None
