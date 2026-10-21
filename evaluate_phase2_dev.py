"""Development-only Phase 2 capture and offline policy comparison. No training."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from config import Config
from evaluate_vqa import OfficialScorer
from export_phase2_dev import validate_dev
from modules import evidence_store, evidence_decision, fusion, qges, vqa, ocr, object_detection
from modules.question_intent import classify
from modules.reliability import mentioned_objects
from utils.image_utils import normalize_image
from utils.model_loader import resolve_device
from visionbridge_training import sha256, verify_checkpoint, guard_output

PROTOCOL = {'image': 'EXIF transpose, RGB, no thumbnail', 'question_tokens': 128,
            'max_new_tokens': 20, 'num_beams': 1, 'do_sample': False,
            'batch_size': 1, 'precision': 'CUDA autocast FP16; CPU FP32', 'seed': 42}
CODE_FILES = ('evaluate_phase2_dev.py', 'export_phase2_dev.py', 'evaluate_vqa.py', 'config.py',
              'modules/vqa.py', 'modules/qges.py', 'modules/evidence_store.py',
              'modules/evidence_decision.py', 'modules/question_intent.py', 'modules/ocr_phrases.py', 'modules/fusion.py',
              'modules/ocr.py', 'modules/object_detection.py', 'modules/reliability.py',
              'utils/image_utils.py', 'utils/model_loader.py')


def package_version(package):
    try:
        return evidence_store.version(package)
    except evidence_store.PackageNotFoundError:
        return 'not-installed'


def write_new(path, value):
    """Exclusive creation: existing experiment files are never overwritten."""
    import uuid
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
    if path.exists():
        raise FileExistsError('Concurrent writer; partial file preserved. Use one evaluator process per output.')
    temporary.rename(path)


def protect_output(path, cfg, payload):
    path = Path(path).resolve()
    for checkpoint in (cfg.trained_blip_dir, cfg.improved_blip_dir):
        guard_output(checkpoint, path)
    source = Path(payload['source_run']).resolve()
    if path == source or source in path.parents or path in source.parents:
        raise ValueError('Output/cache must be separate from the Phase 1 run.')
    if 'vizwiz-5000-lr1e5-run02' in path.parts:
        raise ValueError('Do not write Phase 2 outputs into run02.')


def capture(row, images, cfg, predict=None, extract=None):
    """Inference sees only image/question, never reference answers or labels."""
    predict, extract = predict or vqa.answer, extract or evidence_store.extract
    start = time.perf_counter()
    image = normalize_image(Path(images) / row['image'], thumbnail=False)
    prepared = time.perf_counter()
    baseline = predict(image, row['question'], cfg)
    generated = time.perf_counter()
    intent = classify(row['question'])
    needs_objects = intent.supported and bool(mentioned_objects(row['question']))
    evidence = []
    statuses = {'ocr': {'status': 'disabled'}, 'object': {'status': 'disabled'}}
    for kind, enabled, component in [('ocr', intent.supported, ocr), ('object', needs_objects, object_detection)]:
        if enabled:
            items, status = extract(image, kind, component, cfg)
            evidence.extend(items)
            statuses[kind] = {k: v for k, v in status.items() if k != 'items'}
    extracted = time.perf_counter()
    selected = qges.select(row['question'], evidence, top_k=None)
    incomplete = statuses['ocr']['status'] in ('partial', 'unavailable') or (
        needs_objects and statuses['object']['status'] in ('partial', 'unavailable'))
    finished = time.perf_counter()
    return {'image': row['image'], 'question': row['question'], 'baseline_answer': baseline,
            'image_sha256': sha256(Path(images) / row['image']), 'image_identity': evidence_store.image_identity(image),
            'ocr_question_intent': intent.kind, 'ocr_supported_question': intent.supported,
            'evidence': evidence, 'ocr_evidence': [e for e in evidence if e['kind'] == 'ocr'],
            'selected_evidence': selected, 'extraction_status': statuses, 'incomplete': incomplete,
            'timing_seconds': {'image_preparation': prepared - start, 'blip': generated - prepared,
                               'evidence_extraction': extracted - generated, 'selection': finished - extracted}}


def replay(row, captured, scorer, threshold=None):
    started = time.perf_counter()
    inputs = fusion.FusionInput(row['question'], captured['baseline_answer'], captured['selected_evidence'], {},
                               selection_truncated=captured.get('selection_truncated',
                                   captured['incomplete'] and not any(s.get('status') in ('partial', 'unavailable')
                                       for s in captured.get('extraction_status', {}).values())),
                               ocr_min_score=.9 if threshold is None else threshold,
                               extraction_status=captured.get('extraction_status', {}))
    result = fusion.baseline(inputs) if threshold is None else evidence_decision.decide(inputs)
    decision_time = time.perf_counter() - started
    base_score = scorer.score(captured['baseline_answer'], row['answers'])
    final_score = scorer.score(result.answer, row['answers'])
    changed = scorer.normalize(result.answer) != scorer.normalize(captured['baseline_answer'])
    times = captured['timing_seconds']
    latency = times['image_preparation'] + times['blip'] + decision_time
    if threshold is not None:
        latency += times['evidence_extraction'] + times['selection']
    return {**captured, 'answers': row['answers'], 'answerable': row['answerable'],
            'answer_type': row['answer_type'], 'final_answer': result.answer,
            'decision': result.decision, 'decision_reason': result.reason, 'answer_source': result.source,
            'candidate_trace': result.candidate_trace,
            'supporting_evidence_ids': result.supporting_evidence_ids,
            'baseline_vqa_score': base_score, 'final_vqa_score': final_score,
            'answer_changed': changed, 'ocr_override': changed and result.source == 'ocr',
            'ocr_corrected_blip': changed and final_score > base_score,
            'ocr_worsened_blip': changed and final_score < base_score,
            'score_delta': final_score - base_score, 'threshold': threshold,
            'processing_seconds': latency, 'decision_seconds': decision_time}


def summarize(records, pilot=False):
    n = len(records)
    if not n:
        raise ValueError('No examples; never silently skip missing images or failed BLIP inference.')
    report = {'examples': n, 'pilot_debug_only': pilot,
              'ocr_overrides': sum(r['ocr_override'] for r in records),
              'beneficial_overrides': sum(r['ocr_corrected_blip'] for r in records),
              'harmful_overrides': sum(r['ocr_worsened_blip'] for r in records),
              'neutral_overrides': sum(r['ocr_override'] and r['score_delta'] == 0 for r in records),
              'unchanged_answers': sum(not r['answer_changed'] for r in records),
              'ocr_sourced_answers': sum(r['answer_source'] == 'ocr' for r in records),
              'average_processing_seconds': sum(r['processing_seconds'] for r in records) / n,
              'extraction_failures': sum(any(s['status'] == 'unavailable' for s in r['extraction_status'].values()) for r in records),
              'partial_extractions': sum(any(s['status'] == 'partial' for s in r['extraction_status'].values()) for r in records),
              'ocr_cache_hits': sum(r['extraction_status']['ocr'].get('cache_hit', False) for r in records)}
    # Pilot scores remain per-example diagnostics, never aggregate accuracy.
    if not pilot:
        def group(rows):
            return {'examples': len(rows), 'vqa_agreement_percent':
                    100 * sum(r['final_vqa_score'] for r in rows) / len(rows) if rows else None}
        report.update(group(records))
        report['subgroups'] = {
            'answerable': group([r for r in records if r['answerable'] == 1]),
            'unanswerable': group([r for r in records if r['answerable'] == 0]),
            'ocr_supported_questions': group([r for r in records if r['ocr_supported_question']]),
            **{f'ocr_intent_{kind}': group([r for r in records if r['ocr_question_intent'] == kind])
               for kind in ('reading', 'price', 'brand', 'date')}}
    return report


def compare(rows, captures, scorer, pilot, thresholds):
    if not rows or len(rows) != len(captures):
        raise ValueError('Exactly one capture per example is required; no silent skips.')
    if any(r['image'] != c['image'] or r['question'] != c['question'] for r, c in zip(rows, captures)):
        raise ValueError('Capture order does not match the frozen development rows.')
    policies = {'A': None, 'B': .90, 'C': .95}
    records = {name: [replay(r, c, scorer, threshold) for r, c in zip(rows, captures)]
               for name, threshold in policies.items()}
    redundant = all(scorer.normalize(b['final_answer']) == scorer.normalize(c['final_answer'])
                    for b, c in zip(records['B'], records['C']))
    report = {'purpose': 'pilot debugging only; not an accuracy result' if pilot else 'development tuning only; not held-out accuracy',
              'policies': {'A': 'BLIP v2 only', 'B': 'Existing restricted OCR policy, confidence >= 0.90',
                           'C': 'Same restricted OCR policy, confidence >= 0.95'},
              'C_redundant_on_this_run': redundant,
              'configurations': {name: summarize(value, pilot) for name, value in records.items()
                                 if name != 'C' or not redundant},
              'threshold_sweep': [], 'threshold_conclusion':
              'Keep 0.90 provisional. Do not adjust from a pilot; review the complete development sweep and override errors first.'}
    for threshold in thresholds:
        swept = [replay(r, c, scorer, threshold) for r, c in zip(rows, captures)]
        report['threshold_sweep'].append({'threshold': threshold, **summarize(swept, pilot)})
    if not pilot:
        sweep = report['threshold_sweep']
        best = max(sweep, key=lambda s: (s['vqa_agreement_percent'], -s['harmful_overrides'], s['threshold']))
        baseline_agreement = report['configurations']['A']['vqa_agreement_percent']
        report['development_candidate'] = {
            'threshold': best['threshold'] if best['vqa_agreement_percent'] >= baseline_agreement else None,
            'delta_vs_A_percentage_points': best['vqa_agreement_percent'] - baseline_agreement,
            'delta_vs_B_percentage_points': best['vqa_agreement_percent'] - report['configurations']['B']['vqa_agreement_percent'],
            'rule': 'Highest dev agreement; ties prefer fewer harmful overrides, then higher threshold.',
            'application_setting_changed': False,
            'recommendation': ('Review this candidate and its errors before freezing it; development selection is optimistic.'
                               if best['vqa_agreement_percent'] >= baseline_agreement else
                               'Every tested OCR policy scored below A. Retain BLIP-only pending development error analysis.')}
        report['threshold_conclusion'] = (
            'Development evidence is available, but no threshold is automatically deployed. Compare agreement, harmful/beneficial '
            'overrides, subgroup sizes and extraction failures; inspect development errors before choosing a policy. '
            'These OCR scores are readability filters, not calibrated probabilities. Freeze any chosen threshold before holdout evaluation.')
    report['latency_note'] = ('A excludes evidence time; B/C add the same observed extraction/selection time and measured policy time. '
                              'Shared-capture estimates, not three independent end-to-end runs. Lazy model loading/downloads and cache hits '
                              'affect these measurements; timing excludes scoring, file writes and checkpoint hashing.')
    return report, records


def run(args):
    payload = json.loads(Path(args.dev_manifest).read_text(encoding='utf-8'))
    all_rows = validate_dev(payload)
    pilot = args.mode == 'pilot'
    if not 1 <= args.limit <= 20:
        raise ValueError('Pilot limit must be 1..20; full mode always uses all 1,000 rows.')
    rows = all_rows[:args.limit] if pilot else all_rows
    cfg = replace(Config(), vqa_mode='v2', improved_blip_dir=Path(args.checkpoint),
                  device=args.device, ocr_device='cpu', object_device='cpu',
                  evidence_cache_dir=Path(args.evidence_cache), cache_evidence=True)
    if args.model_cache:
        cfg.cache_dir = Path(args.model_cache)
    if args.yolo_model:
        cfg.yolo_model = args.yolo_model
    for location in (args.output, cfg.evidence_cache_dir, cfg.cache_dir, cfg.yolo_model):
        protect_output(location, cfg, payload)
    vqa.model_source(cfg)  # Explicit v2 validation, no fallback.
    checkpoint_hashes = verify_checkpoint(cfg.improved_blip_dir)
    for name in ('generation_config.json', 'special_tokens_map.json'):
        if (cfg.improved_blip_dir / name).is_file():
            checkpoint_hashes[name] = sha256(cfg.improved_blip_dir / name)
    if args.device == 'cuda' and resolve_device('cuda') != 'cuda':
        raise RuntimeError('CUDA requested but unavailable. Enable T4 or explicitly choose --device cpu.')
    images = Path(args.images).resolve()
    image_hashes = {}
    for row in rows:
        image_hashes[row['image']] = sha256(images / row['image'])  # All selected images required before model work.
    root = Path(__file__).resolve().parent
    scorer = OfficialScorer(payload['official_rules'])
    identity = {'schema': 1, 'mode': args.mode, 'rows_sha256': evidence_store.digest(rows),
                'dev_manifest_sha256': sha256(args.dev_manifest), 'checkpoint_hashes': checkpoint_hashes,
                'image_hashes': image_hashes, 'protocol': PROTOCOL, 'scorer': scorer.identity,
                'code_hashes': {p: sha256(root / p) for p in CODE_FILES},
                'device': resolve_device(cfg.device), 'ocr_device': cfg.ocr_device,
                'vqa_weight_dtype': 'float16' if __import__('os').environ.get('VISIONBRIDGE_LOW_MEMORY_VQA') == '1' else 'float32',
                'object_device': cfg.object_device, 'yolo_model': str(Path(cfg.yolo_model).resolve()),
                'packages': {p: package_version(p) for p in ('torch', 'transformers', 'Pillow', 'easyocr', 'ultralytics')}}
    supported = [r for r in rows if classify(r['question']).supported]
    needed = ['easyocr'] if supported else []
    if any(mentioned_objects(r['question']) for r in supported):
        needed.append('ultralytics')
    if any(identity['packages'][p] == 'not-installed' for p in needed):
        raise RuntimeError('Missing evidence dependency. Install requirements-evidence.txt locally or requirements-phase2-colab.txt in Colab.')
    output = Path(args.output)
    if output.exists():
        if not args.resume:
            raise FileExistsError('Experiment exists. Use --resume for identical capture settings or a fresh output directory.')
        if json.loads((output / 'identity.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('Capture identity changed; preserve this experiment and use a fresh directory.')
    else:
        if args.resume:
            raise FileNotFoundError('Resume requires an existing experiment.')
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / 'identity.json', identity)
        (output / 'captures').mkdir()
    import torch
    import gc
    torch.manual_seed(PROTOCOL['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PROTOCOL['seed'])
    captures = []
    for index, row in enumerate(rows):
        path = output / 'captures' / f'{index:04d}.json'
        if path.exists():
            entry = json.loads(path.read_text(encoding='utf-8'))
            value = entry['capture']
            if entry['sha256'] != evidence_store.digest(value) or value['image'] != row['image'] or value['question'] != row['question'] or value['image_sha256'] != image_hashes[row['image']]:
                raise ValueError('Capture integrity mismatch; preserve it and use a new experiment.')
            for kind, status in value['extraction_status'].items():
                if 'source' in status and status['source'] != evidence_store.descriptor(kind, cfg):
                    raise ValueError('Evidence model/settings changed since capture; use a new experiment.')
        else:
            if resolve_device(cfg.device) == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            value = capture(row, images, cfg)
            if resolve_device(cfg.device) == 'cuda':
                torch.cuda.synchronize()
                value['peak_vram_mb'] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            write_new(path, {'capture': value, 'sha256': evidence_store.digest(value)})
        captures.append(value)
        print(f'{index + 1}/{len(rows)} {row["image"]}', flush=True)
        del value
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report, records = compare(rows, captures, scorer, pilot, args.thresholds)
    # Each replay is a new immutable results directory, even after resume.
    import uuid
    result_dir = output / ('comparison-' + uuid.uuid4().hex[:12])
    result_dir.mkdir(exist_ok=False)
    write_new(result_dir / 'summary.json', report)
    for name, value in records.items():
        write_new(result_dir / f'{name}_records.json', value)
    import csv
    with (result_dir / 'comparison.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['configuration', 'examples', 'vqa_agreement_percent', 'ocr_overrides', 'beneficial_overrides',
                  'harmful_overrides', 'neutral_overrides', 'unchanged_answers', 'average_processing_seconds']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, summary in report['configurations'].items():
            writer.writerow({'configuration': name, **summary})
    print(json.dumps(report, indent=2))
    print(f'Results: {result_dir}')
    return result_dir


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev-manifest', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--evidence-cache', required=True)
    p.add_argument('--model-cache')
    p.add_argument('--yolo-model')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--mode', choices=('pilot', 'full'), default='pilot')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--thresholds', type=float, nargs='+', default=[.80, .85, .90, .95, .99])
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if not args.thresholds or any(not 0 <= t <= 1 for t in args.thresholds):
        raise ValueError('OCR thresholds must be in [0, 1].')
    run(args)
