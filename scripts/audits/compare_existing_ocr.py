"""Lightweight real legacy OCR comparison on rendered known ground truth only."""
import json
from pathlib import Path
import time
from dataclasses import replace


def main(folder):
    from PIL import Image
    from config import Config
    from modules import ocr,evidence_store
    from experimental_reading import reconstruct
    from evaluate_full_text import errors
    source=json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    cfg=replace(Config(),device='cpu',ocr_device='cpu');records={}
    for label in ('paragraph-0','paragraph-90','paragraph-180','paragraph-270','small-text'):
        image=Image.open(folder/(label+'.png')).convert('RGB');tick=time.perf_counter()
        rows=ocr.extract(image,cfg);elapsed=time.perf_counter()-tick
        evidence=evidence_store.standardize(image,rows,'ocr',evidence_store.descriptor('ocr',cfg))
        decision=reconstruct(evidence['items']);text='\n'.join(decision['lines'])
        records[label]=dict(seconds=elapsed,prediction=text,decision=decision,
            metrics=errors(source['records'][label]['reference'],text))
        print(label,round(elapsed,3),records[label]['metrics'],repr(text),flush=True)
    with (folder/'LEGACY_COMPARISON.json').open('x',encoding='utf-8') as f:json.dump(records,f,indent=2,ensure_ascii=False)


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
