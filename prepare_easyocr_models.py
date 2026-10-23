"""Fetch and verify EasyOCR's official English detector/recognizer weights."""
import argparse
import hashlib
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


def file_md5(path):
    h = hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ensure_model(directory: Path, filename: str, url: str, expected_md5: str):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.exists():
        if file_md5(target) == expected_md5:
            print(f'Reusing verified EasyOCR weight: {filename}', flush=True)
            return target
        raise FileExistsError(f'Existing EasyOCR weight has the wrong hash; preserved: {target}')
    archive = directory / (filename + '.download.zip')
    archive_partial = archive.with_name(archive.name + '.partial')
    if archive.exists() or archive_partial.exists():
        raise FileExistsError(f'Incomplete prior EasyOCR download is preserved: {archive}')
    print(f'Downloading official EasyOCR weight {filename}: {url}', flush=True)
    separator = '&' if '?' in url else '?'
    request = urllib.request.Request(url + separator + 'visionbridge_cache=' + str(time.time_ns()),
                                    headers={'User-Agent': 'VisionBridge-Evaluation/1.0', 'Cache-Control': 'no-cache'})
    with urllib.request.urlopen(request, timeout=60) as response, archive_partial.open('xb') as destination:
        shutil.copyfileobj(response, destination, length=4 * 1024 * 1024)
    if not zipfile.is_zipfile(archive_partial):
        raise ValueError(f'Official EasyOCR download is not a ZIP; partial retained: {archive_partial}')
    os.rename(archive_partial, archive)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [m for m in bundle.infolist() if not m.is_dir() and Path(m.filename.replace('\\', '/')).name == filename]
            if len(members) != 1:
                raise ValueError(f'Expected one {filename} in official EasyOCR ZIP; found {len(members)}')
            temporary = target.with_name(target.name + '.partial')
            if temporary.exists():
                raise FileExistsError(f'Preserving existing partial model: {temporary}')
            with bundle.open(members[0]) as source, temporary.open('xb') as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            if file_md5(temporary) != expected_md5:
                raise ValueError(f'Official EasyOCR weight failed MD5 verification: {filename}; partial preserved at {temporary}')
            os.rename(temporary, target)
        archive.unlink()
    except Exception:
        print(f'Archive retained for diagnosis: {archive}', flush=True)
        raise
    print(f'Verified EasyOCR weight: {target}', flush=True)
    return target


def prepare(directory):
    from easyocr.config import detection_models, recognition_models
    craft = detection_models['craft']
    english = recognition_models['gen2']['english_g2']
    result = [ensure_model(directory, item['filename'], item['url'], item['md5sum'])
              for item in (craft, english)]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    for item in prepare(args.directory.resolve()):
        print(f'OK {item.name} {file_md5(item)}')
