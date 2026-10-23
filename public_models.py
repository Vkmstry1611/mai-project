"""Verified public model setup. Never overwrites conflicting files or loads pickle."""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import urllib.request
import uuid
import zipfile

from public_runtime import ROOT, read_manifest, safe_path, sha, verify


def install_bytes(path, data, expected):
    path = Path(path)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Download hash mismatch: ' + path.name)
    if path.exists():
        if sha(path) != expected:
            raise FileExistsError('Conflicting existing file preserved: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive destination and preserved partial make interruptions visible.
    with path.open('xb') as target:
        target.write(data)


def download(root=ROOT):
    manifest = verify(root, assets=False)
    for row in manifest['assets']:
        path = safe_path(root, row['path'])
        if path.exists():
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise FileExistsError('Conflicting asset preserved: ' + row['path'])
            print('Verified:', row['path'], flush=True)
            continue
        source = row.get('download')
        if source is None:
            raise FileNotFoundError('Missing approved LFS/source asset: ' + row['path'] + '; run git lfs pull')
        url = source['url']
        if not url.startswith('https://'):
            raise ValueError('HTTPS model source required')
        print('Downloading:', row['path'], flush=True)
        # All downloaded payloads are bounded (largest is ~83 MB). BLIP is LFS.
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read(150 * 1024 * 1024 + 1)
        if len(data) > 150 * 1024 * 1024:
            raise ValueError('Download exceeds reviewed size bound')
        if source.get('member'):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if archive.namelist().count(source['member']) != 1:
                    raise ValueError('Expected unique OCR model member')
                member = archive.getinfo(source['member'])
                if member.file_size != row['bytes']:
                    raise ValueError('Unexpected model ZIP member size')
                data = archive.read(member)
        if len(data) != row['bytes']:
            raise ValueError('Download size mismatch')
        install_bytes(path, data, row['sha256'])
    verify(root)
    print('All public runtime assets verified. No private evaluation files required.')


def verify_whisper(root=ROOT):
    manifest = read_manifest(root)
    for row in manifest['assets']:
        if row['path'].startswith('models/cache/whisper-tiny.en/'):
            path = safe_path(root, row['path'])
            if not path.is_file() or sha(path) != row['sha256']:
                raise ValueError('Pinned Whisper snapshot missing or changed; run public_models.py download')
    return Path(root) / 'models/cache/whisper-tiny.en'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download', 'verify', 'check-source'])
    args = parser.parse_args()
    if args.command == 'download':
        download()
    else:
        manifest = verify(assets=args.command == 'verify')
        print('Verified public runtime:', len(manifest['assets']), 'declared assets; sources verified.')


if __name__ == '__main__':
    main()
