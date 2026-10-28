"""Read completed dev results and create a new report; no inference or tuning."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import os

from evaluate_phase2_dev import summarize
from evaluate_vqa import OfficialScorer, usable_answers


def report(root, results):
    root, results = Path(root), Path(results)
    verification = json.loads((results / 'verification.json').read_text())
    if verification['mode'] != 'full' or verification['examples'] != 1000:
        raise ValueError('A full dev report requires exactly 1,000 verified dev predictions.')
    payload = json.loads((root / 'dev-only.json').read_text())
    scorer = OfficialScorer(payload['official_rules'])
    records = {n: json.loads((results / f'{n}_records.json').read_text()) for n in ('A', 'B', 'C')}
    stats = {n: summarize(rows) for n, rows in records.items()}
    redundant = all(scorer.normalize(b['final_answer']) == scorer.normalize(c['final_answer'])
                    for b, c in zip(records['B'], records['C']))
    output = root / ('report-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(exist_ok=False)
    fields = ['configuration', 'examples', 'vqa_agreement_percent', 'ocr_overrides', 'beneficial_overrides',
              'harmful_overrides', 'neutral_overrides', 'unchanged_answers', 'average_processing_seconds', 'total_processing_seconds']
    with (output / 'comparison.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name in ('A', 'B', 'C'):
            writer.writerow({'configuration': name, **stats[name],
                             'total_processing_seconds': sum(r['processing_seconds'] for r in records[name])})
    groups = ['answerable', 'unanswerable', 'ocr_supported_questions', 'ocr_intent_reading',
              'ocr_intent_price', 'ocr_intent_brand', 'ocr_intent_date']
    lines = ['# VisionBridge Phase 2 — local development evaluation', '',
             '**Development data only: exact 1,000 run02 dev rows.** No held-out examples were evaluated.', '',
             'The Phase 1 59.48% score on 3,319 held-out examples is a separate result. These development scores do not establish a change in held-out accuracy.', '',
             '| Policy | Official VQA agreement (%) | OCR overrides | Beneficial | Harmful | Neutral | Unchanged | Mean seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('A', 'B', 'C'):
        s = stats[name]
        lines.append(f"| {name} | {s['vqa_agreement_percent']:.2f} | {s['ocr_overrides']} | {s['beneficial_overrides']} | {s['harmful_overrides']} | {s['neutral_overrides']} | {s['unchanged_answers']} | {s['average_processing_seconds']:.3f} |")
    lines += ['', 'A: BLIP v2 alone; B: restricted OCR at 0.90; C: restricted OCR at 0.95.',
              'C is redundant: identical normalized answers to B.' if redundant else 'C produces some different answers from B.', '',
              'An override changes the normalized answer. Beneficial/harmful refers to a higher/lower fractional VQA agreement, not a binary correctness guarantee.', '',
              '## Subgroups', '', '| Subgroup | N | A (%) | B (%) | C (%) |', '|---|---:|---:|---:|---:|']
    for group in groups:
        values = [stats[n]['subgroups'][group] for n in ('A', 'B', 'C')]
        scores = ['n/a' if v['vqa_agreement_percent'] is None else f"{v['vqa_agreement_percent']:.2f}" for v in values]
        lines.append(f"| {group} | {values[0]['examples']} | {' | '.join(scores)} |")
    lines += ['', 'OCR-related cohorts are defined by the pre-answer question router, including failed/missing evidence. They are not a human-verified list of all OCR questions.', '',
              '## Error analysis', '', '| Policy | Zero agreement | Partial agreement | Full agreement |', '|---|---:|---:|---:|']
    errors = {}
    for name, rows in records.items():
        counts = [sum(r['final_vqa_score'] == 0 for r in rows),
                  sum(0 < r['final_vqa_score'] < 1 for r in rows), sum(r['final_vqa_score'] == 1 for r in rows)]
        lines.append(f"| {name} | {' | '.join(map(str, counts))} |")
        confusion = Counter((int(r['answerable']), scorer.normalize(r['final_answer']) != 'unanswerable') for r in rows)
        errors[name] = {'agreement_counts': counts, 'answerability_output_confusion':
                        [{'reference_answerable': a, 'predicted_answered': p, 'count': c} for (a, p), c in sorted(confusion.items())]}
    lines += ['', 'Answerability-output counts below describe whether BLIP/OCR emits `unanswerable`; no learned gate or calibrated probability is present. Wrong answers do not prove absence of visual support.', '',
              '| Policy | True answerable, answered | True answerable, abstained | True unanswerable, answered | True unanswerable, abstained |',
              '|---|---:|---:|---:|---:|']
    for name, rows in records.items():
        counter = Counter((int(r['answerable']), scorer.normalize(r['final_answer']) != 'unanswerable') for r in rows)
        lines.append(f"| {name} | {counter[1, True]} | {counter[1, False]} | {counter[0, True]} | {counter[0, False]} |")
    for kind, flag in [('beneficial', 'ocr_corrected_blip'), ('harmful', 'ocr_worsened_blip')]:
        selected = [{**r, 'configuration': n} for n in ('B', 'C') for r in records[n] if r[flag]]
        with (output / f'{kind}_overrides.json').open('x', encoding='utf-8') as stream:
            json.dump(selected, stream, indent=2)
        lines += ['', f'## {kind.title()} override examples', '']
        for r in selected[:10]:
            lines.append(f"- **{r['configuration']} / {r['image']}** — {r['question']}\n  BLIP: {r['baseline_answer']!r}; final: {r['final_answer']!r}; score: {r['baseline_vqa_score']:.2f} → {r['final_vqa_score']:.2f}; evidence: {', '.join(r['supporting_evidence_ids'])}.")
        if not selected:
            lines.append('None observed.')
    lines += ['', '## Timing and reproducibility', '',
              f"- Whole full-run wall time, including validation/hashing/writes: {verification['wall_seconds']:.1f} seconds.",
              f"- Model weight precision: {verification['weight_precision']}; CUDA FP16 autocast, batch size 1; EasyOCR and object detection on CPU.",
              '- Mean/total policy times are shared-capture estimates, not three independent timed model runs. First-use loading and cache hits affect them. Pilot timings are separate.',
              '- All policies share each image/question, unchanged ten answer slots, one BLIP candidate and the existing official scorer.',
              '- Model/framework versions may differ from Colab Phase 1. The dev A score is the local baseline for this experiment.',
              '- No threshold is automatically deployed. Development threshold exploration is not held-out validation.',
              '- Exact dev order/references and checkpoint hashes were verified after execution. Original artifacts were not overwritten.', '',
              f'Raw records and verification: `{results}`', '', '![Development agreement](agreement.png)', '', '![OCR overrides](overrides.png)']
    os.environ['MPLCONFIGDIR'] = str(root / 'model-cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['A', 'B', 'C']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, [stats[n]['vqa_agreement_percent'] for n in names], color=['#607d9b', '#8d80af', '#559d98'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Official VQA agreement (%)')
    ax.set_xlabel('Development configuration (1,000 examples each)')
    ax.set_title('VisionBridge — development set only')
    ax.bar_label(bars, fmt='%.2f%%')
    fig.tight_layout()
    fig.savefig(output / 'agreement.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0, 1]
    ax.bar([v-.18 for v in x], [stats[n]['beneficial_overrides'] for n in ('B','C')], .36, label='Higher VQA agreement', color='#559d98')
    ax.bar([v+.18 for v in x], [stats[n]['harmful_overrides'] for n in ('B','C')], .36, label='Lower VQA agreement', color='#8d80af')
    ax.set_xticks(x, ['B (0.90)', 'C (0.95)'])
    ax.set_ylabel('Number of changed answers')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'overrides.png', dpi=160)
    plt.close(fig)
    with (output / 'error_analysis.json').open('x', encoding='utf-8') as stream:
        json.dump(errors, stream, indent=2)
    with (output / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('REPORT=' + str(output / 'REPORT.md'))
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    report(args.root, args.results)
