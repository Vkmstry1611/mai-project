"""Import the supplied Hugging Face BLIP export without executing archive code."""
from pathlib import Path
import shutil
import zipfile
from config import ROOT

def import_checkpoint(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        required = {'config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'}
        found = set()
        for member in members:
            parts = Path(member.filename.replace('\\', '/')).parts
            if len(parts) != 2 or parts[0] != 'blip_vizwiz' or '..' in parts:
                raise ValueError(f'Unexpected archive member: {member.filename}')
            if Path(parts[1]).suffix not in {'.json', '.txt', '.safetensors'}:
                raise ValueError(f'Unsupported checkpoint file: {member.filename}')
            found.add(parts[1])
        if not required <= found:
            raise ValueError('Checkpoint archive is missing required model or processor files.')
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / Path(member.filename).name
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite existing checkpoint: {target}')
        for member in members:
            target = destination / Path(member.filename).name
            with bundle.open(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
        print(f'Imported {len(members)} files into {destination}')

if __name__ == '__main__':
    import_checkpoint(ROOT / 'models' / 'visionbridge_model_only.zip', ROOT / 'models' / 'blip_vizwiz')
