"""Phase 3 integrity and split guards. Never decode holdout annotations."""
import hashlib
import json
from pathlib import Path

from evaluate_phase2_dev import write_new
from modules.evidence_store import digest

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/phase3_answerability'
FREEZE = BASE / 'plan-20261012-023211-3dbd1268/PHASE2_FREEZE.json'
MANIFEST = ROOT / 'reports/phase2_local_dev_evaluation/drive_inputs/split_manifest.json'
DEV_INPUT = ROOT / 'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/inputs/run02-dev-only.json'
PHASE2_RECORDS = ROOT / 'reports/phase2_local_dev_evaluation/coverage-targeted-v1-20261012-021213/results/expanded_records.json'
MODEL_CACHE = ROOT / 'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/model-cache'
CHECKPOINT = ROOT / 'models/blip_vizwiz_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_selected_splits(path=MANIFEST):
    """Select train/dev at root; skip all other values without decoding rows."""
    raw = Path(path).read_text(encoding='utf-8-sig')
    decoder = json.JSONDecoder()

    def skip(i):
        if raw[i] in '[{':
            depth, quoted, escaped = 0, False, False
            for j in range(i, len(raw)):
                c = raw[j]
                if quoted:
                    if escaped:
                        escaped = False
                    elif c == '\\':
                        escaped = True
                    elif c == '"':
                        quoted = False
                elif c == '"':
                    quoted = True
                elif c in '[{':
                    depth += 1
                elif c in ']}':
                    depth -= 1
                    if depth == 0:
                        return j + 1
            raise ValueError('Incomplete source JSON')
        if raw[i] == '"':
            return decoder.raw_decode(raw, i)[1]
        while i < len(raw) and raw[i] not in ',}':
            i += 1
        return i

    selected, i = {}, raw.index('{') + 1
    while True:
        while raw[i].isspace() or raw[i] == ',':
            i += 1
        if raw[i] == '}':
            break
        key, i = decoder.raw_decode(raw, i)
        while raw[i].isspace():
            i += 1
        if raw[i] != ':':
            raise ValueError('Malformed manifest')
        i += 1
        while raw[i].isspace():
            i += 1
        if key in ('train', 'dev', 'reference_policy', 'sampling', 'seed'):
            if key in selected:
                raise ValueError('Duplicate source key')
            selected[key], i = decoder.raw_decode(raw, i)
        else:
            i = skip(i)
    return selected


def validate_splits(selected, train_count=5000, dev_count=1000):
    train, dev = selected['train'], selected['dev']
    if len(train) != train_count or len(dev) != dev_count:
        raise ValueError('Original train/development sizes changed; no resampling allowed')
    for split, rows in (('train', train), ('dev', dev)):
        names = set()
        for row in rows:
            name = row['image']
            if not isinstance(name, str) or any(c in name for c in '/\\:') or name in names or not name.startswith('VizWiz_train_' if split == 'train' else 'VizWiz_val_'):
                raise ValueError('Unsafe, duplicate or wrong-split filename')
            if row['answerable'] not in (0, 1) or len(row['answers']) != 10 or not row['question'].strip():
                raise ValueError('Invalid original label/reference/question')
            names.add(name)
    if {r['image'] for r in train} & {r['image'] for r in dev}:
        raise ValueError('Training/development overlap')
    return train, dev


def checked_splits():
    freeze = json.loads(FREEZE.read_text())
    expected = freeze['protected_artifact_sha256'][str(MANIFEST.relative_to(ROOT))]
    if sha(MANIFEST) != expected:
        raise ValueError('Original manifest hash changed')
    selected = read_selected_splits()
    train, dev = validate_splits(selected)
    payload = json.loads(DEV_INPUT.read_text())
    if dev != payload['rows'] or digest(dev) != payload['rows_sha256']:
        raise ValueError('Development rows changed')
    return selected, train, dev, payload


def verify_freeze(full=False):
    lock = json.loads(FREEZE.read_text())
    if lock['ocr_threshold'] != .9:
        raise ValueError('OCR policy must remain frozen at 0.90')
    pairs = dict(lock['frozen_source_sha256'])
    pairs.update({k: v for k, v in lock['protected_artifact_sha256'].items()
                  if full or k.startswith('models\\blip_vizwiz_v2') or k in (str(MANIFEST.relative_to(ROOT)),str(PHASE2_RECORDS.relative_to(ROOT)),str(DEV_INPUT.relative_to(ROOT)))
                  or ('model-cache' in Path(k).parts and Path(k).suffix in ('.pt','.pth'))})
    for path, expected in pairs.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen artifact/source changed: {path}')
    # The detector is referenced in frozen development extraction provenance.
    # Validate it even if its binary was not included by an older artifact lock.
    records=json.loads(PHASE2_RECORDS.read_text())
    for record in records:
        source=record.get('extraction_status',{}).get('object',{}).get('source',{})
        expected=source.get('weights_sha256')
        if expected and expected!='not-present':
            if sha(MODEL_CACHE/'yolov8n.pt')!=expected:
                raise ValueError('Detector weights differ from frozen Phase 2 provenance')
            break
    return {'verified_files': len(pairs), 'full': full, 'threshold': .9}


def safe_run(path):
    path = Path(path).resolve()
    if BASE not in path.parents or path == BASE or 'plan-' in path.name:
        raise ValueError('Use a separate experiment under reports/phase3_answerability')
    return path


def ensure_new_or_equal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing artifact differs; preserve it: {path}')
    else:
        write_new(path, value)
