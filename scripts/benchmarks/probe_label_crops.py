"""Compare bounded local preprocessing without enabling it in production."""
import json
from pathlib import Path
import time

def main():
    import numpy as np
    from PIL import Image,ImageEnhance
    from full_text_ocr import FullTextEngine
    from investigate_reading_upgrade import OUT
    baseline=json.loads((OUT/'rapid-baseline.json').read_text(encoding='utf-8'))['wine']
    image=Image.open(baseline['path']).convert('RGB');engine=FullTextEngine()
    rows=baseline['result']['evidence'];polygons=np.asarray([r['corrected_polygon'] for r in rows],dtype=np.float32)
    crops=engine.engine.get_crop_img_list(engine.engine.load_img(image),polygons);records=[]
    for row,crop in zip(rows,crops):
        if len(row['text'])<3:continue
        record={'id':row['id'],'raw':row['text'],'raw_confidence':row['confidence'],'variants':{}}
        for name,prepared in [('rectified',Image.fromarray(crop)),('upscale2',Image.fromarray(crop).resize((crop.shape[1]*2,crop.shape[0]*2))),
             ('contrast',ImageEnhance.Contrast(Image.fromarray(crop)).enhance(1.4))]:
            tick=time.perf_counter();result,_=engine.engine.text_rec([np.asarray(prepared)])
            record['variants'][name]={'result':result,'seconds':time.perf_counter()-tick}
        records.append(record)
    with (OUT/'LOCAL_CROP_PROBE.json').open('x',encoding='utf-8') as f:json.dump(records,f,indent=2,ensure_ascii=False)
    print(json.dumps(records,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
