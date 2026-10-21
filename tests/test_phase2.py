"""Lightweight Phase 2 tests: fake extractors/checkpoints; no model downloads."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

from PIL import Image
import pytest
from config import Config
from modules import evidence_store, qges, vqa
from modules.evidence_decision import decide
from modules.fusion import FusionInput
from modules.question_intent import classify
from pipeline import Pipeline
from utils.checkpoint_import import import_checkpoint
from utils.image_utils import normalize_image


def evidence(text, identity='ocr-a', confidence=.99, box=None):
    return {'id': identity, 'kind': 'ocr', 'text': text, 'confidence': confidence,
            'box': box or [1, 1, 15, 10], 'status': 'complete'}


def checkpoint(folder):
    folder.mkdir()
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
        (folder / name).write_text(json.dumps({'architectures': ['BlipForQuestionAnswering']}) if name == 'config.json' else 'fixture')
    return folder


def archive(path, root='blip_vizwiz_v2', weight=b'fixture'):
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr(root + '/config.json', json.dumps({'architectures': ['BlipForQuestionAnswering']}))
        for name in ('preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
            bundle.writestr(root + '/' + name, '{}')
        bundle.writestr(root + '/model.safetensors', weight)
    return path


def test_v2_requires_separate_complete_checkpoint(tmp_path):
    original = checkpoint(tmp_path / 'original')
    cfg = Config(trained_blip_dir=original, improved_blip_dir=tmp_path / 'blip_vizwiz_v2', vqa_mode='v2')
    with pytest.raises(FileNotFoundError, match='No fallback'):
        vqa.model_source(cfg)
    checkpoint(cfg.improved_blip_dir)
    assert vqa.model_source(cfg)[0] == str(cfg.improved_blip_dir)
    cfg.improved_blip_dir = original
    with pytest.raises(ValueError, match='separate'):
        vqa.model_source(cfg)


def test_import_refuses_overwrite_mixed_root_and_bad_hash(tmp_path):
    original = checkpoint(tmp_path / 'original')
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    source = archive(tmp_path / 'model.zip')
    with pytest.raises(FileExistsError):
        import_checkpoint(source, original)
    destination = tmp_path / 'blip_vizwiz_v2'
    with pytest.raises(ValueError, match='checksum'):
        import_checkpoint(source, destination, '0' * 64)
    assert not destination.exists()
    import_checkpoint(source, destination, hashlib.sha256(b'fixture').hexdigest())
    assert destination.joinpath('model.safetensors').read_bytes() == b'fixture'
    with pytest.raises(FileExistsError):
        import_checkpoint(source, destination)
    assert before == {p.name: p.read_bytes() for p in original.iterdir()}
    mixed = archive(tmp_path / 'mixed.zip')
    with zipfile.ZipFile(mixed, 'a') as bundle:
        bundle.writestr('blip_vizwiz/vocab.txt', '{}')
    with pytest.raises(ValueError):
        import_checkpoint(mixed, tmp_path / 'another')


def test_lfs_pointer_not_imported(tmp_path):
    source = archive(tmp_path / 'pointer.zip', weight=b'version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='LFS pointer'):
        import_checkpoint(source, tmp_path / 'blip_vizwiz_v2')
    assert not (tmp_path / 'blip_vizwiz_v2').exists()


def test_v2_load_failure_never_uses_original(tmp_path):
    cfg = Config(improved_blip_dir=checkpoint(tmp_path / 'blip_vizwiz_v2'), vqa_mode='v2')
    with patch('modules.vqa.load_blip', side_effect=RuntimeError('bad weights')) as loader:
        with pytest.raises(RuntimeError, match='bad weights'):
            vqa.answer(Image.new('RGB', (20, 20)), 'What?', cfg)
    assert loader.call_count == 1
    assert loader.call_args.args[0] == str(cfg.improved_blip_dir)


def test_phase1_preprocessing_and_decoding_contract():
    image = Image.new('RGB', (2000, 500))
    assert normalize_image(image.copy(), thumbnail=False).size == (2000, 500)
    processor = Mock()
    processor.return_value.to.return_value = {}
    processor.decode.return_value = '2'
    model = Mock()
    model.generate.return_value = [[1, 2]]
    with patch('modules.vqa.load_blip', return_value=(processor, model, 'cpu')):
        assert vqa.answer(image, 'How many?', Config(vqa_mode='pretrained')) == '2'
    assert processor.call_args.kwargs['max_length'] == 128
    assert model.generate.call_args.kwargs == {'max_new_tokens': 20, 'num_beams': 1, 'do_sample': False}


def test_ids_cache_and_image_source_invalidation(tmp_path):
    image = Image.new('RGB', (30, 30))
    cfg = Config(evidence_cache_dir=tmp_path / 'cache', cache_dir=tmp_path / 'weights')
    rows = [evidence('STOP', 'ignored'), evidence('SECOND', 'ignored', box=[1, 12, 15, 20])]
    component = SimpleNamespace(extract=Mock(return_value=rows))
    first, metadata = evidence_store.extract(image, 'ocr', component, cfg)
    second, cached = evidence_store.extract(image, 'ocr', component, cfg)
    assert first == second and cached['cache_hit'] and component.extract.call_count == 1
    assert first[0]['source']['extractor'] == 'easyocr'
    assert all(item['id'] and item['image_id'] and item['status'] == 'complete' for item in first)
    assert component.extract.call_args.args[1].device == 'cpu'
    reordered = evidence_store.standardize(image, list(reversed(rows)), 'ocr', metadata['source'])
    assert reordered['items'] == first
    evidence_store.extract(Image.new('RGB', (30, 30), 'red'), 'ocr', component, cfg)
    assert component.extract.call_count == 2
    cfg.ocr_device = 'auto'
    evidence_store.extract(image, 'ocr', component, cfg)
    assert component.extract.call_count == 3
    weights = cfg.cache_dir / 'easyocr'
    weights.mkdir(parents=True)
    (weights / 'english.pth').write_bytes(b'first test weights')
    _, before = evidence_store.extract(image, 'ocr', component, cfg)
    (weights / 'english.pth').write_bytes(b'changed test weights with a new size')
    _, after = evidence_store.extract(image, 'ocr', component, cfg)
    assert before['cache_key'] != after['cache_key'] and component.extract.call_count == 5


def test_failures_empty_partial_and_cache_corruption(tmp_path):
    image = Image.new('RGB', (30, 30))
    cfg = Config(evidence_cache_dir=tmp_path / 'cache', cache_dir=tmp_path / 'weights')
    component = SimpleNamespace(extract=Mock(side_effect=RuntimeError('offline')))
    items, metadata = evidence_store.extract(image, 'ocr', component, cfg)
    assert not items and metadata['status'] == 'unavailable'
    assert not cfg.evidence_cache_dir.exists()
    component.extract = Mock(return_value=[])
    items, metadata = evidence_store.extract(image, 'ocr', component, cfg)
    assert metadata['status'] == 'empty'
    component.extract.assert_called_once()
    path = next(cfg.evidence_cache_dir.glob('*.json'))
    original = json.loads(path.read_text())
    original['record']['status'] = 'modified'
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='integrity'):
        evidence_store.extract(image, 'ocr', component, cfg)
    result = evidence_store.standardize(image, [evidence('OK'), evidence('BAD', box=[0, 0, 100, 100])], 'ocr', {})
    assert result['status'] == 'partial' and result['rejected_items'] == 1


@pytest.mark.parametrize('question,kind', [('What does this say?', 'reading'), ('How much does it cost?', 'price'),
    ('What brand is this?', 'brand'), ('When does it expire?', 'date'), ('What is the manufacture date?', 'date'),
    ('What colour are the letters?', 'general'), ('What language is the text?', 'general'), ('Is the price $5?', 'general')])
def test_question_intents(question, kind):
    assert classify(question).kind == kind


@pytest.mark.parametrize('question,text,expected', [('What does this say?', 'STOP', 'STOP'),
    ('What is the price?', 'Price: $5.99', '$5.99'), ('What brand is this?', 'Brand: Acme', 'Acme'),
    ('When does it expire?', 'EXP 12/2027', '12/2027')])
def test_unique_ocr_span_changes_answer_with_trace(question, text, expected):
    selected = qges.select(question, [evidence(text)], top_k=None)
    result = decide(FusionInput(question, 'wrong', selected, {}))
    assert result.answer == expected and result.baseline_answer == 'wrong'
    assert result.supporting_evidence_ids == ['ocr-a'] and result.source == 'ocr'


@pytest.mark.parametrize('question,items', [('What is the price?', []),
    ('What is the price?', [evidence('$5'), evidence('$7', 'ocr-b')]),
    ('What brand is this?', [evidence('ACME')]),
    ('When does it expire?', [evidence('MFG 12/2027')]),
    ('When does it expire?', [evidence('EXP 99/2027')]),
    ('When does it expire?', [evidence('MFG/EXP 12/2027')]),
    ('What is the price?', [evidence('$5.999')]),
    ('What does this say?', [evidence('STOP', confidence=.2)]),
    ('What colour is this?', [evidence('STOP')])])
def test_missing_conflicting_weak_or_wrong_evidence_retains_blip(question, items):
    selected = qges.select(question, items, top_k=None)
    result = decide(FusionInput(question, 'blue', selected, {}))
    assert result.answer == 'blue' and result.supporting_evidence_ids == [] and not result.grounded


def test_selection_does_not_hide_conflicts_and_preserves_spatial_target():
    items = [evidence('$' + str(i), 'ocr-' + str(i)) for i in range(8)]
    assert len(qges.select('What is the price?', items, top_k=None)) == 8
    assert decide(FusionInput('What is the price?', '5', qges.select('What is the price?', items, None), {})).source == 'blip'
    question = 'What is written on the bottle?'
    assert qges.select(question, [evidence('STOP')], None) == []
    bottle = {'kind': 'object', 'text': 'bottle', 'box': [0, 0, 20, 20], 'id': 'object-a'}
    assert qges.select(question, [bottle, evidence('ACME')], None)[0]['id'] == 'ocr-a'
    assert not decide(FusionInput('What does this say?', 'yes', [evidence('STOP')], {}, selection_truncated=True)).grounded
    assert decide(FusionInput('What does this say?', 'unanswerable', [evidence('STOP')], {})).answer == 'unanswerable'
    repeated = [evidence('STOP'), evidence('stop', 'ocr-b')]
    result = decide(FusionInput('What does this say?', 'sign', repeated, {}))
    assert result.supporting_evidence_ids == ['ocr-a', 'ocr-b']


def test_end_to_end_phase2_trace_cache_and_no_learned_gate(tmp_path):
    cfg = Config(improved_blip_dir=checkpoint(tmp_path / 'blip_vizwiz_v2'),
                 evidence_cache_dir=tmp_path / 'cache', cache_dir=tmp_path / 'weights', vqa_mode='v2')
    pipeline = Pipeline(cfg)
    with patch('modules.ocr.extract', return_value=[evidence('Price: $5.99')]) as extractor, \
         patch('modules.vqa.answer', return_value='milk') as blip, \
         patch.object(pipeline, 'adapter', side_effect=AssertionError('no trained adapter')):
        result = pipeline.run(Image.new('RGB', (30, 30)), 'What is the price?', speak=False, evidence_aware=True)
        assert result[0] == '$5.99'
        debug = result[3]
        assert debug['baseline_answer'] == 'milk' and debug['final_answer'] == '$5.99'
        assert debug['supporting_evidence_ids'] == [debug['selected_evidence'][0]['id']]
        assert debug['answerability']['probability'] is None
        pipeline.run(Image.new('RGB', (30, 30)), 'What is the price?', speak=False, evidence_aware=True)
        assert extractor.call_count == 1 and blip.call_count == 2


def test_phase2_rejects_wrong_model_and_handles_partial_extraction(tmp_path):
    cfg = Config(improved_blip_dir=checkpoint(tmp_path / 'blip_vizwiz_v2'),
                 evidence_cache_dir=tmp_path / 'cache', cache_dir=tmp_path / 'weights', vqa_mode='v2')
    with patch('modules.vqa.answer', return_value='milk') as blip:
        wrong = Pipeline(cfg).run(Image.new('RGB', (30, 30)), 'What?', vqa_mode='pretrained', speak=False, evidence_aware=True)
        assert 'explicit Improved BLIP v2' in wrong[2]
        blip.assert_not_called()
        with patch('modules.ocr.extract', return_value=[evidence('$5'), evidence('BAD', box=[0, 0, 100, 100])]):
            result = Pipeline(cfg).run(Image.new('RGB', (30, 30)), 'What is the price?', speak=False, evidence_aware=True)
        assert result[0] == 'milk' and result[3]['extraction_status']['ocr']['status'] == 'partial'
