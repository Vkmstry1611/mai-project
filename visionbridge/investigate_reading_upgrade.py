"""Private permitted-photo probes; no evaluation dataset or training access."""
import argparse
import hashlib
import json
from pathlib import Path
import time

OUT=Path('reports/reading_upgrade/run-20261020-v1')
PHOTOS={'wine':'cd057018186141148538ed6013c5302f','ufo':'9079468924344f5eb70fbc42e8ba4f22'}

def main(engine):
    from PIL import Image
    OUT.mkdir(parents=True,exist_ok=True)
    if engine=='snapshot':
        paths=list(Path('.').glob('*.py'))+list(Path('modules').glob('*.py'))+list(Path('utils').glob('*.py'))
        data={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with (OUT/'SOURCE_BEFORE.json').open('x') as f:json.dump(data,f,indent=2)
        return
    if engine=='rapid':
        from visionbridge.full_text_ocr import FullTextEngine
        reader=FullTextEngine()
    else:
        from modules.ocr import load_reader
        tick=time.perf_counter();reader=load_reader('models/cache/easyocr','cpu');loading=time.perf_counter()-tick
    records={}
    for label,identity in PHOTOS.items():
        p=Path('reports/phase4_policy/run-local-20261013-v1/application-requests')/identity/'input.png'
        image=Image.open(p).convert('RGB');tick=time.perf_counter()
        if engine=='rapid':result=reader.read(image,OUT/'rapid-baseline-cache')
        else:
            import numpy as np
            rows=reader.readtext(np.array(image.rotate(90,expand=True) if label=='ufo' else image))
            result={'rows':[{'text':t,'confidence':float(c),'polygon':[[float(x),float(y)] for x,y in b]} for b,t,c in rows],
                    'manual_ccw':90 if label=='ufo' else 0,'loading_seconds':loading}
        records[label]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seconds':time.perf_counter()-tick,'result':result}
        print(engine,label,records[label]['seconds'],flush=True)
        if engine=='rapid':
            corrected=image.rotate(result['orientation']['selected_ccw_degrees'] or 0,expand=True)
            for n,row in enumerate(result['evidence']):
                if label=='wine':corrected.crop(tuple(row['corrected_box'])).save(OUT/f'wine-crop-{n}.png')
    with (OUT/(engine+'-baseline.json')).open('x',encoding='utf-8') as f:json.dump(records,f,indent=2,ensure_ascii=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['snapshot','rapid','easy']);main(p.parse_args().engine)
