"""Versioned, image-specific extraction cache. Failures are never cached."""
from dataclasses import replace
import hashlib
from importlib.metadata import version, PackageNotFoundError
from functools import lru_cache
import json
import math
from pathlib import Path
import uuid

SCHEMA = 1


@lru_cache(maxsize=64)
def _weights_digest(path, size, modified_ns):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def weights_digest(path):
    stat = path.stat()
    return _weights_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def image_identity(image):
    return hashlib.sha256(str(image.size).encode() + image.convert('RGB').tobytes()).hexdigest()


def descriptor(kind, config):
    package = 'easyocr' if kind == 'ocr' else 'ultralytics'
    try:
        installed = version(package)
    except PackageNotFoundError:
        installed = 'not-installed'
    device = config.ocr_device if kind == 'ocr' else config.object_device
    source = {'extractor': package, 'version': installed, 'device': device,
              'settings': 'english-readtext-defaults-v1' if kind == 'ocr' else 'yolov8n-predict-defaults-v1'}
    from utils.model_loader import resolve_device
    source['resolved_device'] = resolve_device(device)
    if kind == 'object':
        model = Path(config.yolo_model)
        source['model'] = str(model)
        source['weights_sha256'] = weights_digest(model) if model.is_file() else 'not-present'
    else:
        source['weights_sha256'] = {p.name: weights_digest(p)
                                   for p in sorted((Path(config.cache_dir) / 'easyocr').glob('*.pth'))}
    return source


def standardize(image, rows, kind, source):
    image_id = image_identity(image)
    valid, rejected, rejections = [], 0, []
    for row in rows:
        box = None
        reason = 'invalid_item'
        try:
            text = row['text']
            box = [float(v) for v in row['box']]
            score = float(row['confidence'])
            if not isinstance(text, str) or not text.strip() or len(box) != 4:
                raise ValueError('Invalid text or box')
            if not all(math.isfinite(v) for v in box + [score]) or not 0 <= score <= 1:
                raise ValueError('Invalid numeric data')
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                reason = 'outside_image'
                raise ValueError('Box outside image')
            valid.append({'kind': kind, 'text': text, 'box': box, 'confidence': score,
                          'polygon': row.get('polygon')})
        except (KeyError, TypeError, ValueError):
            rejected += 1
            safe_box = box if box and len(box) == 4 and all(math.isfinite(v) for v in box) else None
            rejections.append({'box': safe_box, 'reason': reason})
    valid.sort(key=lambda row: (row['box'], row['text'], row['confidence']))
    items = []
    for index, row in enumerate(valid):
        # Duplicate boxes/text still get distinct, deterministic IDs.
        identity = {'image': image_id, 'source': source, 'row': row, 'index': index}
        items.append({**row, 'id': kind + '-' + digest(identity)[:24], 'source': source,
                      'image_id': image_id, 'status': 'complete', 'coordinate_space': 'exif-corrected-rgb-pixels'})
    return {'schema': SCHEMA, 'image_id': image_id, 'source': source, 'items': items,
            'status': 'partial' if rejected else ('complete' if items else 'empty'),
            'rejected_items': rejected, 'rejections': rejections}


def extract(image, kind, component, config):
    if kind not in ('ocr', 'object'):
        raise ValueError('Unsupported extractor')
    source = descriptor(kind, config)
    identity = {'schema': SCHEMA, 'image_id': image_identity(image), 'source': source, 'kind': kind}
    key = digest(identity)
    path = Path(config.evidence_cache_dir) / (key + '.json')
    if config.cache_evidence and path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached['identity'] != identity or digest(cached['record']) != cached['record_sha256']:
            raise ValueError('Evidence cache integrity mismatch; preserve file and select another cache directory')
        return cached['record']['items'], {**cached['record'], 'cache_hit': True, 'cache_key': key}
    try:
        cfg = replace(config, device=config.ocr_device if kind == 'ocr' else config.object_device)
        rows = component.extract(image, cfg)
        source = descriptor(kind, config)  # Include weights downloaded on first use.
        identity = {'schema': SCHEMA, 'image_id': image_identity(image), 'source': source, 'kind': kind}
        key = digest(identity)
        path = Path(config.evidence_cache_dir) / (key + '.json')
        record = standardize(image, rows, kind, source)
    except Exception as error:
        return [], {'status': 'unavailable', 'source': source, 'error': str(error), 'cache_hit': False}
    if config.cache_evidence:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'identity': identity, 'record': record, 'record_sha256': digest(record)}
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, allow_nan=False)
        if path.exists():
            raise FileExistsError('Concurrent cache writer; partial file preserved')
        temporary.rename(path)
    return record['items'], {**record, 'cache_hit': False, 'cache_key': key}
