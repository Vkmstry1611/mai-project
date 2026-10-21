"""Prepare a verified run02 dev-only input set for the Windows GPU evaluator."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from evaluate_vqa import OfficialScorer
from export_phase2_dev import export_dev, validate_dev
from visionbridge_training import DATA_BASE, sha256, verify_checkpoint

ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / 'reports' / 'phase2_local_dev_evaluation'
OFFICIAL_VAL_URL = DATA_BASE + 'images/val.zip'


def under(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f'Path must stay under {root}: {path}')
    return path


def validate_source_manifest(path: Path):
    path = Path(path)
    size = path.stat().st_size
    payload = json.loads(path.read_text(encoding='utf-8'))
    expected = {'train': 5000, 'dev': 1000, 'holdout': 3319}
    counts = {name: len(payload[name]) for name in expected}
    if payload.get('dataset') != 'official VizWiz-VQA' or counts != expected:
        raise ValueError(f'Expected the original run02 manifest with counts {expected}; got {counts}.')
    if payload.get('reference_policy') != 'vizwiz-raw-ten-slots-valid-targets-v1':
        raise ValueError('Unexpected answer-reference policy; preserve the file and investigate.')
    # Iterate only the selected development rows. Holdout is counted above but
    # never used for filenames, model input, tuning, scoring or reporting.
    names, rows = set(), payload['dev']
    for row in rows:
        name = row.get('image')
        if not isinstance(name, str) or Path(name).name != name or '\\' in name or '/' in name:
            raise ValueError('Unsafe development image filename.')
        if name in names:
            raise ValueError(f'Duplicate dev image ID: {name}')
        names.add(name)
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'Missing question for {name}.')
        if not isinstance(row.get('answers'), list) or len(row['answers']) != 10:
            raise ValueError(f'Expected the original ten references for {name}.')
    return payload, {'source_bytes': size, 'source_sha256': sha256(path), 'counts': counts,
                     'dev_rows_sha256': __import__('modules.evidence_store', fromlist=['digest']).digest(rows),
                     'dev_ids': [r['image'] for r in rows]}


def save_source_copy(source: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Preserving existing source copy: {dest}')
    with source.open('rb') as src, dest.open('xb') as out:
        shutil.copyfileobj(src, out)
    if sha256(source) != sha256(dest):
        raise ValueError(f'Copied file hash mismatch: {dest}')


def download_archive(destination: Path, url=OFFICIAL_VAL_URL, opener=urllib.request.urlopen):
    destination = Path(destination)
    if destination.exists():
        if zipfile.is_zipfile(destination):
            print(f'Reusing existing validation archive: {destination}', flush=True)
            return destination
        raise ValueError(f'Existing file is not a ZIP; preserve it and choose a fresh run: {destination}')
    if destination.with_name(destination.name + '.partial').exists():
        raise FileExistsError('An interrupted download is preserved as .partial. Start a fresh run folder; it will not be overwritten.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    print(f'Downloading official VizWiz validation images: {url}', flush=True)
    started = time.monotonic()
    try:
        with opener(url, timeout=60) as response, partial.open('xb') as out:
            expected = response.headers.get('Content-Length')
            expected = int(expected) if expected else None
            total, last_report = 0, 0
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
                total += len(block)
                if total - last_report >= 512 * 1024 * 1024:
                    rate = total / max(time.monotonic() - started, 0.01) / (1024 * 1024)
                    suffix = f'/{expected / (1024**3):.2f} GiB' if expected else ''
                    print(f'Downloaded {total / (1024**3):.2f} GiB{suffix} ({rate:.1f} MiB/s)', flush=True)
                    last_report = total
        if expected is not None and total != expected:
            raise IOError(f'Incomplete response: expected {expected} bytes, received {total}. Partial retained.')
        if not zipfile.is_zipfile(partial):
            raise ValueError('Downloaded response is not a ZIP archive. Partial retained.')
        partial.rename(destination)
    except Exception:
        print(f'Download incomplete; preserved for diagnosis at {partial}', flush=True)
        raise
    print(f'Validation archive ready: {total:,} bytes, SHA256 {sha256(destination)}', flush=True)
    return destination


def inspect_archive(archive: Path, dev_names):
    with zipfile.ZipFile(archive) as bundle:
        by_basename = {}
        for member in bundle.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename.replace('\\', '/')).name
            by_basename.setdefault(name, []).append(member)
        for name in dev_names:
            matches = by_basename.get(name, [])
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one official validation archive member named {name}; found {len(matches)}.')
            if matches[0].file_size <= 0:
                raise ValueError(f'Empty archived image: {name}')
    return len(dev_names)


def check_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        fixed = ImageOps.exif_transpose(image)
        if fixed.width * fixed.height > 40_000_000:
            raise ValueError(f'Image exceeds evaluator limit: {path.name}')
        fixed.load()
        if fixed.width <= 0 or fixed.height <= 0:
            raise ValueError(f'Invalid image dimensions: {path.name}')


def prepare(manifest, rules, archive, run_dir):
    run_dir = under(Path(run_dir), REPORT_ROOT)
    if run_dir.exists():
        if not run_dir.is_dir() or any(run_dir.iterdir()):
            raise FileExistsError(f'Run folder is not empty. Preserve it and select a new unique run name: {run_dir}')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    model = ROOT / 'models' / 'blip_vizwiz_v2'
    checkpoint = verify_checkpoint(model)
    print('Verified local BLIP v2 checkpoint (no model download):', model, flush=True)
    device_check()
    original, audit = validate_source_manifest(Path(manifest))
    src = run_dir / 'source' / 'vizwiz-5000-lr1e5-run02'
    src.mkdir(parents=True, exist_ok=False)
    copied_manifest = src / 'split_manifest.json'
    copied_rules = src / 'official_normalization.json'
    save_source_copy(Path(manifest), copied_manifest)
    save_source_copy(Path(rules), copied_rules)
    source_payload, source_audit = validate_source_manifest(copied_manifest)
    if source_audit['dev_rows_sha256'] != audit['dev_rows_sha256']:
        raise ValueError('Saved development rows changed while preparing inputs.')
    rule_data = json.loads(copied_rules.read_text(encoding='utf-8'))
    OfficialScorer(rule_data)
    dev_path = run_dir / 'inputs' / 'run02-dev-only.json'
    export_dev(src, dev_path)
    payload = json.loads(dev_path.read_text(encoding='utf-8'))
    rows = validate_dev(payload)
    if rows != original['dev']:
        raise ValueError('Exporter changed development IDs, questions or raw reference answers.')
    names = [row['image'] for row in rows]
    print('Original run02 counts verified:', audit['counts'], flush=True)
    print('Frozen development IDs/reference rows:', len(names), audit['dev_rows_sha256'], flush=True)
    print('Holdout is excluded. Only exact dev filenames will be looked up/extracted.', flush=True)
    data = run_dir / 'data'
    image_dir = data / 'images'
    image_dir.mkdir(parents=True, exist_ok=False)
    # A user-supplied existing archive is read-only and may live outside this
    # run folder; only generated outputs are constrained to the new run.
    zip_path = Path(archive).resolve() if archive else data / 'official-vizwiz-validation.zip'
    if not zipfile.is_zipfile(zip_path):
        download_archive(zip_path)
    inspected = inspect_archive(zip_path, names)
    print(f'Official validation ZIP contains exactly one member for each of {inspected} requested dev IDs.', flush=True)
    from visionbridge_training import extract_named
    extract_named(zip_path, names, image_dir)
    for index, name in enumerate(names, 1):
        check_image(image_dir / name)
        if index % 100 == 0 or index == len(names):
            print(f'Image integrity: {index}/{len(names)} verified', flush=True)
    state = {'schema': 1, 'run_name': run_dir.name, 'dev_examples': len(rows),
             'dev_ids_sha256': audit['dev_rows_sha256'], 'dev_ids': names,
             'source_manifest_sha256': sha256(copied_manifest),
             'scoring_rules_sha256': sha256(copied_rules), 'dev_export_sha256': sha256(dev_path),
             'checkpoint_hashes': checkpoint, 'validation_archive_sha256': sha256(zip_path),
             'extraction': 'Only exact 1,000 saved run02 development filenames extracted; no holdout inference/scoring.',
             'device': 'cuda BLIP; CPU OCR/object', 'batch_size': 1}
    state_path = run_dir / 'prepared.json'
    with state_path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
    print('Preparation complete:', run_dir, flush=True)
    return state


def device_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in .venv-gpu. No CPU inference fallback is allowed.')
    if '1650' not in torch.cuda.get_device_name(0):
        raise RuntimeError(f'Expected the GTX 1650; detected {torch.cuda.get_device_name(0)}.')
    if not torch.version.cuda:
        raise RuntimeError('This PyTorch build has no CUDA support.')
    missing = []
    from importlib.metadata import PackageNotFoundError, version
    for package in ('transformers', 'easyocr', 'ultralytics', 'safetensors', 'Pillow'):
        try:
            version(package)
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError('Missing .venv-gpu dependencies: ' + ', '.join(missing) +
                           '. Install only those missing packages in .venv-gpu, pinning Torch/torchvision in a constraints file.')
    if version('torch') != '2.5.1+cu121' or version('transformers') != '4.46.3':
        raise RuntimeError('Unexpected GPU environment versions; preserve .venv and .venv-gpu and investigate before evaluation.')
    from modules import ocr
    if not callable(ocr.extract):
        raise RuntimeError('EasyOCR extraction is unavailable.')
    print('CUDA:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
    print('Transformers:', version('transformers'), '| EasyOCR:', version('easyocr'), '| YOLO:', version('ultralytics'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--rules', required=True)
    p.add_argument('--archive')
    p.add_argument('--run-dir', required=True)
    args = p.parse_args()
    prepare(args.manifest, args.rules, args.archive, args.run_dir)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'PREPARATION FAILED: {type(error).__name__}: {error}', file=sys.stderr)
        raise
