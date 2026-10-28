"""Verified, exclusive download of the optional English ONNX recognizer."""
import hashlib
from pathlib import Path
import urllib.request
import uuid
from visionbridge.full_text_ocr import ENGLISH,ENGLISH_SHA


def main():
    if ENGLISH.exists():
        if hashlib.sha256(ENGLISH.read_bytes()).hexdigest()!=ENGLISH_SHA:raise ValueError('Existing experimental English model differs; preserved. Investigate before continuing.')
        print('Existing English OCR model verified.');return
    ENGLISH.parent.mkdir(parents=True,exist_ok=True)
    part=ENGLISH.with_name(ENGLISH.name+'.'+uuid.uuid4().hex+'.download')
    url='https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx'
    with urllib.request.urlopen(url,timeout=60) as source,part.open('xb') as target:
        while chunk:=source.read(1024*1024):target.write(chunk)
    if hashlib.sha256(part.read_bytes()).hexdigest()!=ENGLISH_SHA:raise ValueError('Downloaded SHA256 mismatch. Partial preserved: '+str(part))
    if ENGLISH.exists():raise FileExistsError('Concurrent model download; existing target preserved')
    part.rename(ENGLISH);print('English OCR model verified:',ENGLISH)


if __name__=='__main__':main()
