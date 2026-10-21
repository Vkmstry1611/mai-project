"""No model downloads, training, real dev evaluation or holdout access."""
from argparse import Namespace
import copy
import json
from unittest.mock import Mock

from PIL import Image
import pytest
from config import Config
from evaluate_vqa import OfficialScorer, REFERENCE_POLICY
from export_phase2_dev import export_dev, validate_dev
from evaluate_phase2_dev import capture, replay, compare, summarize, protect_output, parser, run
from modules.evidence_store import digest


def rules():
    return {'source': 'synthetic-test-only', 'source_sha256': 'fixture', 'punct': [],
            'manualMap': {}, 'articles': [], 'contractions': {}}


def row(index=0, answer='welcome', question='What text is written?', answerable=1):
    return {'image': f'VizWiz_val_{index:08d}.jpg', 'question': question,
            'answers': [{'answer': answer, 'answer_confidence': 'yes'} for _ in range(10)],
            'answerable': answerable, 'answer_type': 'other'}


def payload():
    rows = [row(i) for i in range(1000)]
    return {'schema': 1, 'split': 'dev', 'source_run_name': 'vizwiz-5000-lr1e5-run02',
            'source_run': '/phase1/vizwiz-5000-lr1e5-run02', 'source_manifest_sha256': 'fixture',
            'reference_policy': REFERENCE_POLICY, 'rows': rows,
            'rows_sha256': digest(rows), 'official_rules': rules()}


def item(text='WELCOME', confidence=.92, identity='ocr-one'):
    return {'id': identity, 'kind': 'ocr', 'text': text, 'confidence': confidence,
            'status': 'complete', 'box': [1, 1, 8, 8]}


def captured(r, baseline='hello', evidence=None, incomplete=False):
    evidence = evidence if evidence is not None else [item()]
    return {'image': r['image'], 'question': r['question'], 'baseline_answer': baseline,
            'selected_evidence': evidence, 'evidence': evidence, 'ocr_evidence': evidence,
            'ocr_supported_question': True, 'ocr_question_intent': 'reading',
            'incomplete': incomplete, 'extraction_status': {'ocr': {'status': 'complete', 'cache_hit': True}},
            'timing_seconds': {'image_preparation': .1, 'blip': 1., 'evidence_extraction': 2., 'selection': .1}}


def test_export_preserves_exact_rows_duplicates_invalid_slots_and_source(tmp_path):
    p = payload()
    p['rows'][0]['answers'][0]['answer'] = ''
    p['rows'][0]['answers'][1]['answer'] = 'WELCOME'
    source = tmp_path / 'vizwiz-5000-lr1e5-run02'
    source.mkdir()
    # Other split values deliberately unusable: exporter must not traverse them.
    original = {'dev': p['rows'], 'train': None, 'holdout': 'DO NOT USE', 'reference_policy': REFERENCE_POLICY}
    manifest = source / 'split_manifest.json'
    manifest.write_text(json.dumps(original), encoding='utf-8')
    (source / 'official_normalization.json').write_text(json.dumps(rules()), encoding='utf-8')
    before = manifest.read_bytes()
    dest = export_dev(source, tmp_path / 'dev.json')
    exported = json.loads(dest.read_text())
    assert validate_dev(exported) == p['rows']
    assert manifest.read_bytes() == before
    assert not any(k in exported for k in ('train', 'holdout'))
    with pytest.raises(FileExistsError):
        export_dev(source, dest)
    with pytest.raises(ValueError, match='outside'):
        export_dev(source, source / 'dev.json')


@pytest.mark.parametrize('mutation', ['holdout', 'train', 'split', 'size', 'checksum', 'duplicate', 'unsafe', 'run', 'refs'])
def test_dev_manifest_guards(mutation):
    p = payload()
    if mutation in ('holdout', 'train'):
        p[mutation] = []
    elif mutation == 'split':
        p['split'] = 'holdout'
    elif mutation == 'size':
        p['rows'].pop()
    elif mutation == 'checksum':
        p['rows'][0]['question'] = 'changed'
    elif mutation == 'duplicate':
        p['rows'][1]['image'] = p['rows'][0]['image']
    elif mutation == 'unsafe':
        p['rows'][0]['image'] = '..\\outside.jpg'
    elif mutation == 'run':
        p['source_run_name'] = 'run01'
    else:
        p['rows'][0]['answers'] = [{'answer': ''}] * 10
    if mutation != 'checksum':
        p['rows_sha256'] = digest(p['rows'])
    with pytest.raises(ValueError):
        validate_dev(p)


def test_capture_one_inference_no_labels_no_thumbnail_and_replay_shared(tmp_path):
    r = row()
    Image.new('RGB', (1700, 20)).save(tmp_path / r['image'])
    def predict(image, question, cfg):
        assert image.size == (1700, 20)
        assert question == r['question']
        return 'hello'
    predict_mock = Mock(side_effect=predict)
    extract_mock = Mock(return_value=([item()], {'status': 'complete', 'cache_hit': False}))
    c = capture(r, tmp_path, Config(vqa_mode='v2'), predict_mock, extract_mock)
    before = copy.deepcopy(c)
    report, records = compare([r], [c], OfficialScorer(rules()), True, [.8, .9, .95])
    assert predict_mock.call_count == extract_mock.call_count == 1
    assert records['A'][0]['final_answer'] == 'hello'
    assert records['B'][0]['final_answer'] == 'WELCOME'
    assert records['C'][0]['final_answer'] == 'hello'
    assert 'vqa_agreement_percent' not in report['configurations']['B']
    assert c == before  # Policies cannot mutate their shared capture.
    assert 'answers' not in c and 'answerable' not in c


@pytest.mark.parametrize('evidence,incomplete', [([], False), ([item(), item('NOTICE', .99, 'two')], False), ([item()], True)])
def test_unsafe_overrides_retain_baseline(evidence, incomplete):
    r = row()
    result = replay(r, captured(r, evidence=evidence, incomplete=incomplete), OfficialScorer(rules()), .9)
    assert result['final_answer'] == 'hello'
    assert not result['ocr_override']


def test_metrics_beneficial_harmful_neutral_unchanged_and_subgroup_denominators():
    scorer = OfficialScorer(rules())
    rows = [row(0), row(1, answer='hello', answerable=0), row(2, answer='neither'), row(3)]
    captures = [captured(r) for r in rows]
    captures[3]['selected_evidence'] = []
    report, records = compare(rows, captures, scorer, False, [.9, .95])
    b = report['configurations']['B']
    assert (b['ocr_overrides'], b['beneficial_overrides'], b['harmful_overrides'], b['neutral_overrides'], b['unchanged_answers']) == (3, 1, 1, 1, 1)
    assert b['subgroups']['answerable']['examples'] == 3
    assert b['subgroups']['unanswerable']['examples'] == 1
    assert b['subgroups']['ocr_supported_questions']['examples'] == 4
    assert b['vqa_agreement_percent'] == 25
    assert records['B'][0]['answers'] == rows[0]['answers']
    assert records['B'][0]['supporting_evidence_ids'] == ['ocr-one']
    assert records['B'][0]['processing_seconds'] > records['A'][0]['processing_seconds']
    assert not report['development_candidate']['application_setting_changed']


def test_same_answer_not_counted_override_and_redundant_C_removed():
    r = row()
    c = captured(r, baseline='welcome', evidence=[item(confidence=.99)])
    report, records = compare([r], [c], OfficialScorer(rules()), False, [.9])
    assert report['C_redundant_on_this_run']
    assert 'C' not in report['configurations']
    assert not records['B'][0]['ocr_override']
    assert records['B'][0]['answer_source'] == 'ocr'


def test_partial_reference_frequency_same_scorer_all_policies():
    r = row()
    r['answers'] = [{'answer': 'welcome'}] * 2 + [{'answer': ''}] * 8
    s = OfficialScorer(rules())
    assert replay(r, captured(r), s, .9)['final_vqa_score'] == pytest.approx(.6)
    assert replay(r, captured(r, baseline='welcome'), s)['final_vqa_score'] == pytest.approx(.6)


def test_missing_capture_never_silently_skipped():
    with pytest.raises(ValueError, match='one capture'):
        compare([row()], [], OfficialScorer(rules()), True, [.9])
    with pytest.raises(ValueError, match='order'):
        compare([row()], [captured(row(1))], OfficialScorer(rules()), True, [.9])


def test_output_protection(tmp_path):
    cfg = Config(trained_blip_dir=tmp_path / 'original', improved_blip_dir=tmp_path / 'v2')
    p = payload()
    p['source_run'] = str(tmp_path / 'vizwiz-5000-lr1e5-run02')
    for target in (cfg.trained_blip_dir, cfg.improved_blip_dir / 'results', tmp_path / 'vizwiz-5000-lr1e5-run02' / 'new'):
        with pytest.raises(ValueError):
            protect_output(target, cfg, p)
    protect_output(tmp_path / 'phase2', cfg, p)


def test_defaults_are_pilot_not_full():
    args = parser().parse_args(['--dev-manifest', 'dev.json', '--images', 'images', '--checkpoint', 'v2',
                               '--output', 'phase2', '--evidence-cache', 'cache'])
    assert args.mode == 'pilot' and args.limit == 20


def test_twenty_example_synthetic_pilot_resume_and_artifact_protection(tmp_path, monkeypatch):
    import evaluate_phase2_dev as evaluator
    p = payload()
    manifest = tmp_path / 'dev.json'
    manifest.write_text(json.dumps(p), encoding='utf-8')
    model = tmp_path / 'blip_vizwiz_v2'
    model.mkdir()
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
        (model / name).write_text(json.dumps({'architectures': ['BlipForQuestionAnswering']}) if name == 'config.json' else 'fixture')
    model_before = {f.name: f.read_bytes() for f in model.iterdir()}
    images = tmp_path / 'images'
    images.mkdir()
    for r in p['rows'][:20]:
        Image.new('RGB', (20, 20)).save(images / r['image'])
    predictor = Mock(return_value='hello')
    extractor = Mock(return_value=([item()], {'status': 'complete', 'cache_hit': True}))
    actual_capture = evaluator.capture
    capture_mock = Mock(side_effect=lambda r, im, cfg: actual_capture(r, im, cfg, predictor, extractor))
    monkeypatch.setattr(evaluator, 'capture', capture_mock)
    args = parser().parse_args(['--dev-manifest', str(manifest), '--images', str(images),
                               '--checkpoint', str(model), '--output', str(tmp_path / 'pilot'),
                               '--evidence-cache', str(tmp_path / 'cache'), '--device', 'cpu'])
    first = run(args)
    report = json.loads((first / 'summary.json').read_text())
    assert report['configurations']['A']['examples'] == 20
    assert report['configurations']['B']['beneficial_overrides'] == 20
    assert 'vqa_agreement_percent' not in report['configurations']['B']
    assert predictor.call_count == extractor.call_count == 20
    first_bytes = (first / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(args)
    args.resume = True
    second = run(args)
    assert second != first and capture_mock.call_count == 20
    assert (first / 'summary.json').read_bytes() == first_bytes
    assert model_before == {f.name: f.read_bytes() for f in model.iterdir()}
    Image.new('RGB', (21, 20)).save(images / p['rows'][0]['image'])
    with pytest.raises(ValueError, match='identity changed'):
        run(args)


def test_interrupted_capture_is_resumed_without_repeating_completed_images(tmp_path, monkeypatch):
    import evaluate_phase2_dev as evaluator
    p = payload()
    manifest = tmp_path / 'dev.json'
    manifest.write_text(json.dumps(p), encoding='utf-8')
    model = tmp_path / 'v2'
    model.mkdir()
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
        (model / name).write_text(json.dumps({'architectures': ['BlipForQuestionAnswering']}) if name == 'config.json' else 'fixture')
    images = tmp_path / 'images'
    images.mkdir()
    for r in p['rows'][:3]:
        Image.new('RGB', (20, 20)).save(images / r['image'])
    actual_capture = evaluator.capture
    def fake(r, im, cfg):
        return actual_capture(r, im, cfg, Mock(return_value='hello'), Mock(return_value=([], {'status': 'empty'})))
    interrupted = Mock(side_effect=[fake(p['rows'][0], images, Config()), RuntimeError('disconnect')])
    monkeypatch.setattr(evaluator, 'capture', interrupted)
    args = parser().parse_args(['--dev-manifest', str(manifest), '--images', str(images), '--checkpoint', str(model),
                               '--output', str(tmp_path / 'pilot'), '--evidence-cache', str(tmp_path / 'cache'),
                               '--limit', '3', '--device', 'cpu'])
    with pytest.raises(RuntimeError, match='disconnect'):
        run(args)
    completed = (tmp_path / 'pilot' / 'captures' / '0000.json').read_bytes()
    resumed = Mock(side_effect=fake)
    monkeypatch.setattr(evaluator, 'capture', resumed)
    args.resume = True
    run(args)
    assert resumed.call_count == 2
    assert (tmp_path / 'pilot' / 'captures' / '0000.json').read_bytes() == completed


def test_optional_missing_package_has_explicit_metadata(monkeypatch):
    import evaluate_phase2_dev as evaluator
    def missing(name):
        raise evaluator.evidence_store.PackageNotFoundError(name)
    monkeypatch.setattr(evaluator.evidence_store, 'version', missing)
    assert evaluator.package_version('ultralytics') == 'not-installed'


def test_incomplete_extraction_is_reported_without_changing_denominators():
    r = row()
    c = captured(r, incomplete=True)
    c['extraction_status']['ocr']['status'] = 'partial'
    rec = replay(r, c, OfficialScorer(rules()), .9)
    report = summarize([rec])
    assert report['examples'] == report['partial_extractions'] == 1
    assert report['unchanged_answers'] == 1
