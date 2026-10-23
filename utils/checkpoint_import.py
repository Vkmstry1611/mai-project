"""Import the supplied Hugging Face BLIP export without executing archive code."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path
import shutil
import zipfile
from config import ROOT

def import_checkpoint(archive, destination, expected_sha256=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint directory: {destination}')
    with zipfile.ZipFile(archive) as bundle:
        members = [member for member in bundle.infolist() if not member.is_dir()]
        required = {'config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'}
        found, roots = set(), set()
        for member in members:
            parts = Path(member.filename.replace('\\', '/')).parts
            if len(parts) != 2 or parts[0] not in ('blip_vizwiz', 'blip_vizwiz_v2') or '..' in parts:
                raise ValueError(f'Unexpected archive member: {member.filename}')
            if Path(parts[1]).suffix not in {'.json', '.txt', '.safetensors'} or parts[1] in found:
                raise ValueError(f'Unsupported checkpoint file: {member.filename}')
            found.add(parts[1])
            roots.add(parts[0])
        if len(roots) != 1 or not required <= found:
            raise ValueError('Checkpoint archive is missing required model or processor files.')
        root = next(iter(roots))
        if root == 'blip_vizwiz_v2' and destination.name != 'blip_vizwiz_v2':
            raise ValueError('v2 must use a separate blip_vizwiz_v2 destination')
        cfg = json.loads(bundle.read(root + '/config.json'))
        if cfg.get('architectures') != ['BlipForQuestionAnswering']:
            raise ValueError('Expected BlipForQuestionAnswering export')
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(destination.name + '.import-' + uuid.uuid4().hex)
        staging.mkdir(exist_ok=False)
        for member in members:
            target = staging / Path(member.filename).name
            with bundle.open(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
        weights = staging / 'model.safetensors'
        with weights.open('rb') as stream:
            if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f'LFS pointer is not model weights; staging preserved: {staging}')
        digest = hashlib.sha256()
        with weights.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if expected_sha256 and digest.hexdigest() != expected_sha256.lower():
            raise ValueError(f'Weight checksum mismatch; staging preserved: {staging}')
        if destination.exists():
            raise FileExistsError(f'Destination appeared during import; staging preserved: {staging}')
        staging.rename(destination)
        print(f'Imported {len(members)} files into {destination}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'models' / 'visionbridge_model_only.zip')
    parser.add_argument('--destination', type=Path, default=ROOT / 'models' / 'blip_vizwiz')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    import_checkpoint(args.archive, args.destination, args.expected_sha256)
