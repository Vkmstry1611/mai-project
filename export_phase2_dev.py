"""Copy the saved run02 dev split, never sample a new split or evaluate holdout."""
import argparse
import json
from pathlib import Path

from evaluate_vqa import OfficialScorer, REFERENCE_POLICY, usable_answers
from modules.evidence_store import digest
from visionbridge_training import sha256


def validate_dev(payload):
    if payload.get('split') != 'dev' or any(k in payload for k in ('train', 'holdout', 'validation')):
        raise ValueError('Only a dev-only export is accepted; do not supply the Phase 1 split manifest.')
    if payload.get('reference_policy') != REFERENCE_POLICY:
        raise ValueError('Reference policy differs from Phase 1.')
    if payload.get('source_run_name') != 'vizwiz-5000-lr1e5-run02':
        raise ValueError('Expected the saved run02 development split.')
    rows = payload['rows']
    if not isinstance(rows, list) or len(rows) != 1000 or payload.get('rows_sha256') != digest(rows):
        raise ValueError('Expected exactly 1,000 unchanged development rows with a matching checksum.')
    names = set()
    for row in rows:
        name = row['image']
        if not isinstance(name, str) or not name or any(c in name for c in '/\\:') or name in ('.', '..') or name in names:
            raise ValueError('Unsafe or duplicate image ID.')
        names.add(name)
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('Missing development question.')
        if not isinstance(row['answers'], list) or len(row['answers']) != 10 or not usable_answers(row['answers']) or row['answerable'] not in (0, 1):
            raise ValueError('Invalid saved reference slots or answerability label; do not repair/resample here.')
    OfficialScorer(payload['official_rules'])
    return rows


def export_dev(run_dir, destination):
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if run_dir.name != 'vizwiz-5000-lr1e5-run02':
        raise ValueError('Select the existing run02 directory.')
    if destination == run_dir or run_dir in destination.parents:
        raise ValueError('Export outside Phase 1 artifacts.')
    if destination.exists():
        raise FileExistsError('Export already exists; preserve it and choose a new filename.')
    manifest_path = run_dir / 'split_manifest.json'
    # This source JSON contains multiple splits. Only the saved dev field is
    # selected; no train/holdout row is traversed, inspected, exported or scored.
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    payload = {'schema': 1, 'split': 'dev', 'source_run_name': run_dir.name,
               'source_run': str(run_dir), 'source_manifest_sha256': sha256(manifest_path),
               'reference_policy': manifest['reference_policy'], 'rows': manifest['dev'],
               'rows_sha256': digest(manifest['dev']),
               'official_rules': json.loads((run_dir / 'official_normalization.json').read_text(encoding='utf-8'))}
    validate_dev(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(export_dev(args.run_dir, args.output))
