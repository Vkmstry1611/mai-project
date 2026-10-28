"""Read-only resolution of existing verified EasyOCR weights; no downloads."""
from pathlib import Path
from modules.evidence_store import weights_digest

WEIGHTS={'craft_mlt_25k.pth':'4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17',
         'english_g2.pth':'e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8'}

def easyocr_directory(root):
    root=Path(root)
    choices=[root/'models/cache/easyocr',root/'reports/phase2_local_dev_evaluation/run-local-20261011-193140-9b32cf49/model-cache/easyocr']
    for folder in choices:
        if all((folder/n).is_file() and weights_digest(folder/n)==expected for n,expected in WEIGHTS.items()):return folder
    raise FileNotFoundError('Verified EasyOCR weights missing or changed. Restore the private runtime bundle; no automatic model download or overwrite was attempted.')
