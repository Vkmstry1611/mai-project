"""Integrity gate for the local 20-example Phase 2 pilot."""
import argparse
import json
from pathlib import Path

from modules.evidence_store import digest
from visionbridge_training import sha256, verify_checkpoint


def verify(run_dir: Path, result_dir: Path):
    prepared = json.loads((run_dir / 'prepared.json').read_text(encoding='utf-8'))
    dev = json.loads((run_dir / 'inputs' / 'run02-dev-only.json').read_text(encoding='utf-8'))
    identity = json.loads((result_dir.parent / 'identity.json').read_text(encoding='utf-8'))
    summary = json.loads((result_dir / 'summary.json').read_text(encoding='utf-8'))
    if prepared['dev_examples'] != 1000 or identity.get('mode') != 'pilot':
        raise ValueError('Pilot must originate from the prepared 1,000-row saved run02 dev split.')
    rows = dev['rows'][:20]
    if len(rows) != 20 or identity['rows_sha256'] != digest(rows):
        raise ValueError('Pilot row IDs/references differ from the first 20 exact dev rows.')
    if set(identity['image_hashes']) != {r['image'] for r in rows}:
        raise ValueError('Pilot image set does not match exact dev rows.')
    if identity.get('device') != 'cuda' or identity.get('ocr_device') != 'cpu':
        raise ValueError('Expected BLIP CUDA and OCR CPU.')
    current = verify_checkpoint(Path(__file__).resolve().parent / 'models' / 'blip_vizwiz_v2')
    expected = prepared['checkpoint_hashes']
    if current != expected:
        raise ValueError('Improved BLIP v2 checkpoint changed during evaluation.')
    if any(identity['checkpoint_hashes'].get(k) != v for k, v in expected.items()):
        raise ValueError('Pilot did not use the prepared BLIP v2 checkpoint.')
    if 'holdout' in dev or 'train' in dev or len(rows) != 20:
        raise ValueError('Dev-only export unexpectedly contains other splits.')
    if 'accuracy' in summary or 'vqa_agreement_percent' in summary.get('configurations', {}).get('A', {}):
        raise ValueError('Pilot must not be reported as an aggregate accuracy result.')
    for config in 'ABC':
        records = json.loads((result_dir / f'{config}_records.json').read_text(encoding='utf-8'))
        if len(records) != 20:
            raise ValueError(f'{config} must contain all 20 pilot records.')
        for row, record in zip(rows, records):
            if row['image'] != record['image'] or row['question'] != record['question'] or row['answers'] != record['answers']:
                raise ValueError(f'{config} pilot record differs from frozen source references.')
            if record['ocr_supported_question'] and record['extraction_status']['ocr']['status'] == 'unavailable':
                raise ValueError('OCR failed on a pilot question that requires OCR; do not start full evaluation.')
    return {'examples': 20, 'development_ids_sha256': identity['rows_sha256'],
            'checkpoint_unchanged': True, 'device': 'CUDA BLIP / CPU OCR',
            'C_redundant_on_pilot': summary['C_redundant_on_this_run']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.result_dir), indent=2))
