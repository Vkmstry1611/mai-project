"""Compare existing Phase 1 and Phase 4 predictions; no model calls."""
import hashlib
import json
from pathlib import Path
import sys

from evaluate_vqa import OfficialScorer


def main(folder):
    root = Path(__file__).resolve().parents[2]
    folder = Path(folder)
    historical_path = folder / 'provenance/improved_blip_predictions.json'
    historical = json.loads(historical_path.read_text(encoding='utf-8'))
    metadata = json.loads((folder / 'provenance/summary.json').read_text())
    current = json.loads((root / 'reports/phase4_policy/run-local-20261013-v1/holdout/scored_records.json').read_text())
    dev_input = root / 'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/inputs/run02-dev-only.json'
    scorer = OfficialScorer(json.loads(dev_input.read_text())['official_rules'])
    changes = []
    old_total = new_total = 0
    for old, new in zip(historical, current, strict=True):
        for key in ('image', 'question', 'answers', 'answerable'):
            assert old[key] == new[key], (key, old['image'])
        assert len(old['answers']) == 10
        old_score = scorer.score(old['prediction'], old['answers'])
        new_score = scorer.score(new['baseline_answer'], new['answers'])
        assert abs(old_score - old['score']) < 1e-9
        old_total += old_score
        new_total += new_score
        if scorer.normalize(old['prediction']) != scorer.normalize(new['baseline_answer']):
            changes.append({'image': old['image'], 'question': old['question'],
                            'historical_answer': old['prediction'], 'current_answer': new['baseline_answer'],
                            'historical_score': old_score, 'current_score': new_score,
                            'delta': new_score-old_score, 'answers': old['answers']})
    hashes = {name: hashlib.sha256((root / 'models/blip_vizwiz_v2' / name).read_bytes()).hexdigest()
              for name in metadata['protocol']['export_sha256']}
    assert hashes == metadata['protocol']['export_sha256']
    report = {'examples': len(current), 'all_membership_questions_labels_references_identical': True,
              'export_files_identical': True, 'historical_vqa_percent': 100*old_total/len(current),
              'current_blip_vqa_percent': 100*new_total/len(current),
              'changed_normalized_predictions': len(changes),
              'beneficial_prediction_changes': sum(c['delta'] > 0 for c in changes),
              'harmful_prediction_changes': sum(c['delta'] < 0 for c in changes),
              'neutral_prediction_changes': sum(c['delta'] == 0 for c in changes),
              'inference_calls': 0, 'thresholds_changed': False,
              'differences': changes}
    (folder / 'BASELINE_PROVENANCE_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'differences'}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
