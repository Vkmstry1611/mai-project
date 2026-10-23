"""Replay exactly saved run02 development captures; never load inference models.

The policy is fixed at 0.90. Missing OCR remains missing. Input captures and
original evaluations are immutable; every invocation creates a new directory.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import time

from evaluate_phase2_dev import replay, summarize, write_new, CODE_FILES
from evaluate_vqa import OfficialScorer
from export_phase2_dev import validate_dev
from modules import qges, evidence_store
from modules.question_intent import classify
from visionbridge_training import sha256

POLICY = 'targeted-ocr-v1-confidence-0.90'


def reselect_capture(captured):
    """Selection receives inference-only fields, never references/labels."""
    if any(k in captured for k in ('answers', 'answerable', 'final_vqa_score')):
        raise ValueError('Use raw inference captures, not scored result records.')
    updated = copy.deepcopy(captured)
    start = time.perf_counter()
    updated['selected_evidence'] = qges.select(captured['question'], captured['evidence'], top_k=None)
    updated['timing_seconds']['selection'] = time.perf_counter() - start
    intent = classify(captured['question'])
    updated['ocr_supported_question'] = intent.supported
    updated['ocr_question_intent'] = intent.kind
    return updated


def group(records):
    return {'examples': len(records), 'vqa_agreement_percent':
            100 * sum(r['final_vqa_score'] for r in records) / len(records) if records else None}


def run_replay(dev_manifest, captures_dir, comparison_dir, output, checkpoint=None):
    dev_manifest, captures_dir, comparison_dir, output = map(Path,
        (dev_manifest, captures_dir, comparison_dir, output))
    if output.exists():
        raise FileExistsError('Preserve existing replay results; choose a new output directory.')
    for protected in (dev_manifest.parent, captures_dir.parent, comparison_dir):
        if output.resolve() == protected.resolve() or protected.resolve() in output.resolve().parents:
            raise ValueError('Output must be outside source inputs/evaluation directories.')
    if checkpoint and (Path(checkpoint).resolve() == output.resolve()
                       or Path(checkpoint).resolve() in output.resolve().parents):
        raise ValueError('Never write inside a checkpoint.')
    payload = json.loads(dev_manifest.read_text(encoding='utf-8'))
    rows = validate_dev(payload)
    scorer = OfficialScorer(payload['official_rules'])
    protected_files = [dev_manifest] + [comparison_dir / f'{k}_records.json' for k in 'AB']
    if checkpoint:
        protected_files += [p for p in Path(checkpoint).iterdir() if p.is_file()]
    originals = {k: json.loads((comparison_dir / f'{k}_records.json').read_text(encoding='utf-8')) for k in 'AB'}
    identity_path = captures_dir.parent / 'identity.json'
    if identity_path.is_file():
        protected_files.append(identity_path)
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        if identity['rows_sha256'] != payload['rows_sha256'] or identity['scorer'] != scorer.identity:
            raise ValueError('Capture identity differs from frozen development/scoring protocol.')
        if checkpoint and any(sha256(Path(checkpoint) / name) != value
                              for name, value in identity['checkpoint_hashes'].items()):
            raise ValueError('Checkpoint differs from original evaluation; never replace it during replay.')
    if any(len(v) != 1000 for v in originals.values()):
        raise ValueError('Original A/B must contain exactly 1,000 results.')
    captures = []
    cache_files = set()
    for index, row in enumerate(rows):
        path = captures_dir / f'{index:04d}.json'
        protected_files.append(path)
        captured = json.loads(path.read_text(encoding='utf-8'))
        if 'capture' in captured:
            if evidence_store.digest(captured['capture']) != captured.get('sha256'):
                raise ValueError('Saved capture checksum differs.')
            captured = captured['capture']
        if (captured['image'], captured['question']) != (row['image'], row['question']):
            raise ValueError('Capture order differs from the unchanged development export.')
        for k in 'AB':
            original = originals[k][index]
            if (original['image'], original['question'], original['answers'], original['answerable']) != (
                    row['image'], row['question'], row['answers'], row['answerable']):
                raise ValueError('Original records do not match frozen development references.')
            if captured['baseline_answer'] != original['baseline_answer']:
                raise ValueError('Shared BLIP prediction changed.')
            if k == 'A' and original['final_answer'] != captured['baseline_answer']:
                raise ValueError('A must be the unchanged BLIP-only baseline.')
            if scorer.score(original['final_answer'], row['answers']) != original['final_vqa_score']:
                raise ValueError('Original official VQA score differs.')
        if captured['evidence'] != originals['B'][index]['evidence']:
            raise ValueError('Raw evidence differs from original evaluation.')
        captures.append(captured)
    # Cache integrity is already protected in captures. Verify linked cache files
    # when a standard run layout is available; never rewrite or refresh them.
    cache_root = captures_dir.parents[3] / 'evidence-cache'
    for captured in captures:
        for kind, status in captured['extraction_status'].items():
            if status.get('cache_key') and cache_root.is_dir():
                path = cache_root / (status['cache_key'] + '.json')
                entry = json.loads(path.read_text(encoding='utf-8'))
                if evidence_store.digest(entry['record']) != entry['record_sha256']:
                    raise ValueError('Evidence cache checksum differs.')
                if entry['record']['items'] != [i for i in captured['evidence'] if i['kind'] == kind]:
                    raise ValueError('Capture/cache evidence mismatch.')
                cache_files.add(path)
    protected_files += sorted(cache_files)
    before = {str(p): sha256(p) for p in protected_files}
    code_files = list(CODE_FILES) + ['replay_phase2_dev.py']
    code_hashes = {p: sha256(p) for p in code_files}
    started = time.perf_counter()
    targeted = []
    for index, (row, captured) in enumerate(zip(rows, captures)):
        selected = reselect_capture(captured)
        result = replay(row, selected, scorer, .90)
        result['policy'] = POLICY
        result['original_ocr_supported_question'] = captured['ocr_supported_question']
        result['original_ocr_question_intent'] = captured['ocr_question_intent']
        result['cached_ocr_available'] = captured['extraction_status']['ocr']['status'] != 'disabled'
        targeted.append(result)
        if (index + 1) % 100 == 0:
            print(f'Cached replay: {index + 1}/1000', flush=True)
    summary = {'purpose': 'development-only cached replay; not held-out accuracy', 'policy': POLICY,
        'threshold': .90, 'baseline': summarize(originals['A']), 'original_ocr': summarize(originals['B']),
        'targeted': summarize(targeted), 'remaining_reasons': {}, 'replay_seconds': time.perf_counter()-started,
        'rows_sha256': payload['rows_sha256'], 'input_sha256': before, 'code_sha256': code_hashes,
        'verified_cache_files': len(cache_files), 'gpu_inference_calls': 0,
        'scope_note': 'Only the dev-only export was read. No original multi-split manifest or holdout was opened.',
        'latency_note': 'BLIP/OCR times are historical shared-capture observations. Only new selection/decision wall time was measured.'}
    from collections import Counter
    summary['remaining_reasons'] = dict(Counter(r['decision_reason'] for r in targeted if r['answer_source'] == 'blip'))
    summary['delta_vqa_percentage_points'] = summary['targeted']['vqa_agreement_percent'] - summary['baseline']['vqa_agreement_percent']
    old_ids = {r['image'] for r in originals['A'] if r['ocr_supported_question']}
    summary['frozen_original_ocr_subgroup'] = {
        'baseline': group([r for r in originals['A'] if r['image'] in old_ids]),
        'targeted': group([r for r in targeted if r['image'] in old_ids])}
    new_ids = {r['image'] for r in targeted if r['ocr_supported_question']}
    summary['revised_router_ocr_subgroup'] = {
        'baseline': group([r for r in originals['A'] if r['image'] in new_ids]),
        'targeted': group([r for r in targeted if r['image'] in new_ids]),
        'missing_cached_ocr': sum(r['image'] in new_ids and not r['cached_ocr_available'] for r in targeted)}
    corrected = [r for r in targeted if r['ocr_corrected_blip']]
    worsened = [r for r in targeted if r['ocr_worsened_blip']]
    if before != {str(p): sha256(p) for p in protected_files} or code_hashes != {p: sha256(p) for p in code_files}:
        raise RuntimeError('Inputs or policy changed during replay; refusing to publish results.')
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('summary.json', summary), ('targeted_records.json', targeted),
                       ('corrected_examples.json', corrected), ('worsened_examples.json', worsened)]:
        write_new(output / name, data)
    with (output / 'comparison.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['configuration', 'examples', 'official_vqa_percent', 'overrides', 'beneficial', 'harmful', 'neutral'])
        for name, s in [('A: saved BLIP v2', summary['baseline']), ('B: saved original OCR', summary['original_ocr']), ('D: targeted OCR 0.90', summary['targeted'])]:
            writer.writerow([name, s['examples'], s['vqa_agreement_percent'], s['ocr_overrides'],
                             s['beneficial_overrides'], s['harmful_overrides'], s['neutral_overrides']])
    write_report(output, summary, corrected, worsened)
    return output


def write_report(output, summary, corrected, worsened):
    b, t = summary['baseline'], summary['targeted']
    delta = summary['delta_vqa_percentage_points']
    lines = ['# VisionBridge targeted OCR development replay', '',
        f'Exactly 1,000 unchanged run02 development examples; threshold **0.90**. Saved BLIP predictions and raw OCR/object evidence were reused. No GPU inference or holdout inspection.', '',
        '| Metric | Saved BLIP v2 | Targeted OCR |', '|---|---:|---:|',
        f'| Official VQA agreement | {b["vqa_agreement_percent"]:.2f}% | {t["vqa_agreement_percent"]:.2f}% |',
        f'| OCR overrides | 0 | {t["ocr_overrides"]} |',
        f'| Beneficial / harmful / neutral overrides | 0 / 0 / 0 | {t["beneficial_overrides"]} / {t["harmful_overrides"]} / {t["neutral_overrides"]} |',
        f'| Unchanged answers | 1,000 | {t["unchanged_answers"]} |', '',
        f'Net agreement change: **{delta:+.2f} percentage points**. ' + ('This replay confirms a development-set increase only.' if delta > 0 else 'This replay does not confirm an improvement; retain BLIP-only as the validated baseline.'), '',
        f'The saved original OCR policy is {summary["original_ocr"]["vqa_agreement_percent"]:.2f}%; its outputs remain untouched. The historical 59.48% holdout result is a different split and was not re-evaluated.', '',
        '| Subgroup | n | Baseline | Targeted |', '|---|---:|---:|---:|']
    def percent(value):
        return 'N/A' if value is None else f'{value:.2f}%'
    for name in ('answerable', 'unanswerable'):
        a=b['subgroups'][name];z=t['subgroups'][name]
        lines.append(f'| {name} | {a["examples"]} | {percent(a["vqa_agreement_percent"])} | {percent(z["vqa_agreement_percent"])} |')
    for name, field in [('Original OCR-supported questions (fixed subgroup)', 'frozen_original_ocr_subgroup'),
                         ('Revised router supported questions', 'revised_router_ocr_subgroup')]:
        a=summary[field]['baseline'];z=summary[field]['targeted']
        lines.append(f'| {name} | {a["examples"]} | {percent(a["vqa_agreement_percent"])} | {percent(z["vqa_agreement_percent"])} |')
    lines += ['', f'Revised routing includes {summary["revised_router_ocr_subgroup"]["missing_cached_ocr"]} supported questions with no saved OCR. These retain BLIP; routing coverage is not fully measured without new extraction. No missing observations were invented.', '',
        '## Corrected and worsened examples', '', 'Beneficial means a strictly higher official VQA score, not necessarily a completely correct answer. Harmful means a lower score. The JSON files contain every changed example and all ten original reference slots.']
    from collections import Counter
    for title, records in [('Corrected', corrected), ('Worsened', worsened)]:
        lines += ['', f'### {title}', '']
        if not records:
            lines.append('None in this replay.')
        for r in records[:10]:
            refs=dict(Counter(a['answer'] for a in r['answers']))
            lines += [f'- **{r["image"]}** — {r["question"].strip()}',
                      f'  BLIP: `{r["baseline_answer"]}` ({r["baseline_vqa_score"]:.2f}); final: `{r["final_answer"]}` ({r["final_vqa_score"]:.2f}). References: {refs}.',
                      f'  Evidence IDs: {r["supporting_evidence_ids"]}. Decision: {r["decision_reason"]}']
    lines += ['', '## Remaining first rejection reasons', '', '| Reason | Examples |', '|---|---:|']
    for reason, count in sorted(summary['remaining_reasons'].items(), key=lambda x:-x[1]):
        lines.append(f'| {reason} | {count} |')
    lines += ['', '## Policy and verification', '',
        '- Confidence is applied per candidate. Separate high-confidence candidates still trigger ambiguity; unreadable fragments are not treated as readable competing answers.',
        '- Phrase assembly uses left-to-right aligned boxes and compact top-to-bottom lines. It does not spell-correct text, join distant regions or bridge an unreadable intervening word. Curved, overlapping or uncertain layouts may remain unresolved.',
        '- Brand fallback requires a unique dominant alphabetic literal on a populated label (area at least twice other alphabetic text). This is an explicit heuristic, not proof of brand identity; harm remains possible. No brand names or reference labels are hard-coded.',
        '- Partial extraction requires all rejected boxes to be demonstrably outside a detected target region and complete object extraction. Old partial caches lack those details and retain BLIP. BLIP abstention remains protected.',
        f'- Dev row digest: `{summary["rows_sha256"]}`. A/B references and saved official scores matched; {summary["verified_cache_files"]} linked cache files passed integrity checks.',
        '- All input/checkpoint file hashes matched before and after replay. Source code hashes record this single frozen policy. No thresholds were swept or adjusted after seeing replay scores.',
        f'- Cached replay wall time: {summary["replay_seconds"]:.2f} seconds, excluding hashing/output. Historical BLIP/OCR durations are not newly measured inference latency.',
        '- Stop here before holdout evaluation. Review errors and validate any further changes on development data only.']
    with (output/'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dev-manifest', 'captures', 'original-comparison', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkpoint', help='Optional checkpoint directory: hash only, never load.')
    args = parser.parse_args()
    print(run_replay(args.dev_manifest, args.captures, args.original_comparison, args.output, args.checkpoint))
