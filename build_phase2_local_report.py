"""Build development-only report and charts from verified Phase 2 outputs."""
import argparse
import csv
import json
from pathlib import Path

from evaluate_phase2_dev import summarize
from export_phase2_dev import validate_dev
from modules.evidence_store import digest
from visionbridge_training import sha256, verify_checkpoint


def score_group(rows):
    return {'n': len(rows), 'agreement_percent': 100 * sum(r['final_vqa_score'] for r in rows) / len(rows) if rows else None}


def write_chart(report, records, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['A: BLIP v2', 'B: OCR ≥0.90', 'C: OCR ≥0.95']
    names = 'ABC'
    scores = [report['configurations'].get(k, {}).get('vqa_agreement_percent') for k in names]
    times = [report['configurations'].get(k, {}).get('average_processing_seconds') for k in names]
    if report['C_redundant_on_this_run']:
        labels[2] += ' (same as B)'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = ['#5865a8', '#4a9b91', '#8368a8']
    for ax, values, title, ylabel in ((axes[0], scores, 'Official VizWiz VQA agreement', 'Agreement (%)'),
                                      (axes[1], times, 'Shared-capture processing latency', 'Seconds / example')):
        bars = ax.bar(labels, values, color=palette)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=15)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom')
    fig.suptitle('VisionBridge Phase 2 — run02 development set only (n=1,000)')
    fig.tight_layout()
    fig.savefig(destination, dpi=170, bbox_inches='tight')
    plt.close(fig)


def build(run_dir: Path, profile: str):
    prepared = json.loads((run_dir / 'prepared.json').read_text(encoding='utf-8'))
    payload = json.loads((run_dir / 'inputs' / 'run02-dev-only.json').read_text(encoding='utf-8'))
    rows = validate_dev(payload)
    if len(rows) != 1000 or 'holdout' in payload or 'train' in payload:
        raise ValueError('Report input must be exactly the dev-only 1,000-row export.')
    eval_root = run_dir / 'evaluation' / profile / 'full'
    comparison_dirs = sorted(eval_root.glob('comparison-*'), key=lambda p: p.stat().st_mtime_ns)
    if not comparison_dirs:
        raise FileNotFoundError(f'No full comparison output under {eval_root}')
    result = comparison_dirs[-1]
    identity = json.loads((eval_root / 'identity.json').read_text(encoding='utf-8'))
    summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
    if identity.get('mode') != 'full' or identity.get('device') != 'cuda' or identity.get('ocr_device') != 'cpu':
        raise ValueError('Expected full 1,000-row CUDA/CPU evaluation identity.')
    if identity['rows_sha256'] != digest(rows) or set(identity['image_hashes']) != set(prepared['dev_ids']):
        raise ValueError('Full evaluation did not use exactly the saved development rows and images.')
    checkpoint = verify_checkpoint(Path(__file__).resolve().parent / 'models' / 'blip_vizwiz_v2')
    required_hashes = prepared['checkpoint_hashes']
    if checkpoint != required_hashes or any(identity['checkpoint_hashes'].get(k) != v for k, v in required_hashes.items()):
        raise ValueError('Original improved BLIP v2 checkpoint hash changed; report stopped.')
    records = {}
    for config in 'ABC':
        records[config] = json.loads((result / f'{config}_records.json').read_text(encoding='utf-8'))
        if len(records[config]) != 1000:
            raise ValueError(f'{config}: expected exactly 1,000 results.')
        for source, item in zip(rows, records[config]):
            if source['image'] != item['image'] or source['question'] != item['question'] or source['answers'] != item['answers']:
                raise ValueError(f'{config}: record order/reference mismatch.')
    report_root = run_dir / 'report'
    report_root.mkdir(exist_ok=True)
    recomputed = {k: summarize(v) for k, v in records.items()}
    # Use the evaluator's official-scorer-derived summaries; recomputed summaries are a consistency check.
    for k in 'ABC':
        declared = summary['configurations'].get(k)
        if declared is not None and declared != recomputed[k]:
            raise ValueError(f'{k}: persisted summary differs from per-example records.')
        if declared is None and not (k == 'C' and summary['C_redundant_on_this_run']):
            raise ValueError(f'{k}: missing configuration summary.')
        if declared is None:
            recomputed[k] = dict(recomputed['B'])
    # C is retained in the per-example data and explicitly labelled redundant where applicable.
    with (report_root / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['configuration', 'examples', 'vqa_agreement_percent', 'ocr_overrides', 'beneficial_overrides',
                  'harmful_overrides', 'unchanged_answers', 'average_processing_seconds', 'total_processing_seconds', 'redundant']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for k, label in [('A', 'Improved BLIP v2 only'), ('B', 'BLIP v2 + OCR, threshold 0.90'), ('C', 'BLIP v2 + OCR, threshold 0.95')]:
            stats = recomputed[k]
            writer.writerow({'configuration': label, **stats,
                             'vqa_agreement_percent': 100 * sum(r['final_vqa_score'] for r in records[k]) / 1000,
                             'total_processing_seconds': sum(r['processing_seconds'] for r in records[k]),
                             'redundant': k == 'C' and summary['C_redundant_on_this_run']})
    # Reference counts are kept raw in the records; subgroups derive only from dev labels/questions.
    subgroup = {}
    for config in 'ABC':
        subgroup[config] = {
            'answerable': score_group([r for r in records[config] if r['answerable'] == 1]),
            'unanswerable': score_group([r for r in records[config] if r['answerable'] == 0]),
            'ocr_supported_questions': score_group([r for r in records[config] if r['ocr_supported_question']]),
            'ocr_intents': {kind: score_group([r for r in records[config] if r['ocr_question_intent'] == kind])
                            for kind in ('reading', 'price', 'brand', 'date')},
        }
    corrected = [r for r in records['B'] if r['ocr_corrected_blip']]
    worsened = [r for r in records['B'] if r['ocr_worsened_blip']]
    (report_root / 'ocr_corrected_examples.json').write_text(json.dumps(corrected, indent=2, ensure_ascii=False), encoding='utf-8')
    (report_root / 'ocr_worsened_examples.json').write_text(json.dumps(worsened, indent=2, ensure_ascii=False), encoding='utf-8')
    errors = {'BLIP_no_agreement': sum(r['baseline_vqa_score'] == 0 for r in records['A']),
              'BLIP_partial_agreement': sum(0 < r['baseline_vqa_score'] < 1 for r in records['A']),
              'BLIP_full_agreement': sum(r['baseline_vqa_score'] == 1 for r in records['A']),
              'OCR_policy_B_no_agreement': sum(r['final_vqa_score'] == 0 for r in records['B']),
              'OCR_policy_B_partial_agreement': sum(0 < r['final_vqa_score'] < 1 for r in records['B']),
              'OCR_policy_B_full_agreement': sum(r['final_vqa_score'] == 1 for r in records['B'])}
    answerability_confusion = {}
    for config in 'ABC':
        predicted_unanswerable = lambda r: r['final_answer'].strip().lower() == 'unanswerable'
        answerability_confusion[config] = {
            'reference_unanswerable_predicted_unanswerable': sum(r['answerable'] == 0 and predicted_unanswerable(r) for r in records[config]),
            'reference_unanswerable_predicted_answer': sum(r['answerable'] == 0 and not predicted_unanswerable(r) for r in records[config]),
            'reference_answerable_predicted_unanswerable': sum(r['answerable'] == 1 and predicted_unanswerable(r) for r in records[config]),
            'reference_answerable_predicted_answer': sum(r['answerable'] == 1 and not predicted_unanswerable(r) for r in records[config]),
        }
    threshold_text = ', '.join(format(x['threshold'], '.2f') for x in summary['threshold_sweep'])
    chart_summary = dict(summary)
    chart_summary['configurations'] = {k: {**recomputed[k], 'vqa_agreement_percent':
                                100 * sum(r['final_vqa_score'] for r in records[k]) / 1000}
                                for k in 'ABC'}
    write_chart(chart_summary, recomputed, report_root / 'configuration_comparison.png')
    lines = [
        '# VisionBridge Phase 2 development evaluation', '',
        '**Split:** run02 development only; exactly 1,000 examples. **No holdout examples were evaluated or used for tuning.**',
        '**This is a development result, not the Phase 1 59.48% holdout result and not a holdout comparison.**', '',
        'All three policies reuse identical BLIP v2 predictions, images, questions, ten original reference answers, and official VizWiz scoring. OCR/object evidence is shared and cached. Latency is the evaluator’s shared-capture estimate; model loading and output writing are excluded.', '',
        '| Configuration | VQA agreement | OCR overrides | Beneficial | Harmful | Unchanged | Mean sec/example |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k, label in [('A','A — Improved BLIP v2 only'),('B','B — OCR policy ≥0.90'),('C','C — OCR policy ≥0.95')]:
        stats = recomputed[k]
        val = 100 * sum(r['final_vqa_score'] for r in records[k]) / 1000
        suffix = ' (redundant with B on these examples)' if k == 'C' and summary['C_redundant_on_this_run'] else ''
        lines.append(f"| {label}{suffix} | {val:.2f}% | {stats['ocr_overrides']} | {stats['beneficial_overrides']} | {stats['harmful_overrides']} | {stats['unchanged_answers']} | {stats['average_processing_seconds']:.3f} |")
    lines += ['', '## Development subgroups', '', 'Agreement is official VizWiz VQA agreement, calculated separately within each development subgroup.', '',
              '| Configuration | Answerable n/agreement | Unanswerable n/agreement | OCR-supported n/agreement | Reading | Price | Brand | Date |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in 'ABC':
        s = subgroup[k]
        fmt = lambda x: f"{x['n']}/{x['agreement_percent']:.2f}%" if x['agreement_percent'] is not None else '0/—'
        lines.append(f"| {k} | {fmt(s['answerable'])} | {fmt(s['unanswerable'])} | {fmt(s['ocr_supported_questions'])} | {fmt(s['ocr_intents']['reading'])} | {fmt(s['ocr_intents']['price'])} | {fmt(s['ocr_intents']['brand'])} | {fmt(s['ocr_intents']['date'])} |")
    lines += ['', '## Decision and error analysis', '',
              f"- C is {'redundant' if summary['C_redundant_on_this_run'] else 'distinct'} from B on this 1,000-example set.",
              f"- Policy B overrides: {recomputed['B']['ocr_overrides']}; beneficial: {recomputed['B']['beneficial_overrides']}; harmful: {recomputed['B']['harmful_overrides']}.",
              f"- BLIP score cases: {errors['BLIP_full_agreement']} full, {errors['BLIP_partial_agreement']} partial, {errors['BLIP_no_agreement']} zero-agreement.",
              f"- OCR-policy B score cases: {errors['OCR_policy_B_full_agreement']} full, {errors['OCR_policy_B_partial_agreement']} partial, {errors['OCR_policy_B_no_agreement']} zero-agreement.",
              f"- Evidence extraction status: {recomputed['B']['extraction_failures']} unavailable examples and {recomputed['B']['partial_extractions']} partial-extraction examples across the shared captures.",
              '- Exact-answer answerability confusion counts (a generated answer is considered an abstention only when it is exactly `unanswerable`; this is descriptive, not a calibrated classifier):',
              '', '| Configuration | Ref unanswerable → abstain | Ref unanswerable → answer | Ref answerable → abstain | Ref answerable → answer |', '|---|---:|---:|---:|---:|']
    for k in 'ABC':
        m = answerability_confusion[k]
        lines.append(f"| {k} | {m['reference_unanswerable_predicted_unanswerable']} | {m['reference_unanswerable_predicted_answer']} | {m['reference_answerable_predicted_unanswerable']} | {m['reference_answerable_predicted_answer']} |")
    lines += [
              f"- Threshold sweep ({threshold_text}): no OCR policy produced an override; all scored {summary['configurations']['A']['vqa_agreement_percent']:.2f}% on dev. There is no development evidence to adjust the provisional 0.90 threshold. The sweep's tie-break selects 0.99 only as a numeric candidate; keep 0.90 frozen unless a reviewed future development experiment finds a meaningful difference.",
              f"- Mean processing time: A {recomputed['A']['average_processing_seconds']:.3f}s, B {recomputed['B']['average_processing_seconds']:.3f}s, C {recomputed['C']['average_processing_seconds']:.3f}s. Total measured time: A {sum(r['processing_seconds'] for r in records['A']):.1f}s, B {sum(r['processing_seconds'] for r in records['B']):.1f}s, C {sum(r['processing_seconds'] for r in records['C']):.1f}s.",
              '- Thresholds are decision filters, not calibrated answer probabilities. Any threshold selection is development-only and must be frozen before a separate final holdout evaluation.',
              '- Per-example predictions, all ten references, OCR evidence, selections, reasons, sources, scores and timing are in `evaluation/.../comparison-*/{A,B,C}_records.json`.',
              '- Corrected and worsened B examples are saved in `ocr_corrected_examples.json` and `ocr_worsened_examples.json`.', '',
              '## Artifacts and integrity', '',
              f"- Frozen dev row checksum: `{prepared['dev_ids_sha256']}`.",
              f"- Original split manifest SHA256: `{prepared['source_manifest_sha256']}`; official validation archive SHA256: `{prepared['validation_archive_sha256']}`.",
              f"- BLIP v2 checkpoint SHA256s matched before and after evaluation: `{json.dumps(checkpoint, sort_keys=True)}`.",
              '- Official scoring normalization follows the run02 Phase 1 rules.',
              '- The Phase 1 59.48% holdout baseline is preserved as historical context only; no Phase 2 holdout result is claimed.', '',
              '![Development comparison](configuration_comparison.png)', '']
    (report_root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report written to {report_root / "REPORT.md"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--profile', choices=('fp32','fp16'), required=True)
    a = p.parse_args()
    build(a.run_dir, a.profile)
