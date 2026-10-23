"""Optional CPU-only EasyOCR worker; frozen modules and models are reused read-only."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

def serve(base):
    protocol=sys.stdout;base=Path(base).resolve()
    def send(value):protocol.write(json.dumps(value,ensure_ascii=False)+'\n');protocol.flush()
    try:
        tick=time.perf_counter()
        with redirect_stdout(sys.stderr):
            from modules.ocr import load_reader
            from experimental_orientation import transform,bounds,matrices,process_memory_mib
            from modules.evidence_store import image_identity,digest,descriptor
            from config import Config
            from dataclasses import replace
            from PIL import Image,ImageOps
            import numpy as np
            import torch
            from assistant_runtime import easyocr_directory
            # Match the existing persistent worker's bounded CPU threading;
            # avoid oversubscribing this laptop during detector convolutions.
            torch.set_num_threads(4)
            weight_folder=easyocr_directory(Path(__file__).resolve().parent)
            reader=load_reader(str(weight_folder),'cpu')
        source=dict(descriptor('ocr',replace(Config(),device='cpu',ocr_device='cpu',cache_dir=weight_folder.parent)),engine='EasyOCR',policy='original-readtext-0.90',cpu_threads=4)
        send(dict(ready=True,startup_seconds=time.perf_counter()-tick,source=source))
    except Exception as e:send(dict(ready=False,error=str(e)));return
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action')=='shutdown':send({'stopped':True});break
            tick=time.perf_counter()
            path=Path(request['image']).resolve()
            if base not in path.parents:raise ValueError('Image must belong to this temporary session')
            image=ImageOps.exif_transpose(Image.open(path)).convert('RGB');angle=int(request.get('manual') or 0)
            if image.width*image.height>40_000_000:raise ValueError('Resize below 40 megapixels.')
            if angle not in (0,90,180,270):raise ValueError('Unsupported rotation')
            identity={'version':'easy-reading-original-v1','image':image_identity(image),'angle':angle,'source':source}
            namespace=request.get('cache_namespace','cache')
            if not isinstance(namespace,str) or not namespace or not all(c.isalnum() or c=='-' for c in namespace):raise ValueError('Invalid cache namespace')
            folder=base/namespace;folder.mkdir(parents=True,exist_ok=True)
            cache=folder/(digest(identity)+'.json')
            if cache.exists():
                payload=json.loads(cache.read_text(encoding='utf-8'))
                if payload['identity']!=identity or digest(payload['result'])!=payload['sha256']:raise ValueError('EasyOCR cache integrity mismatch')
                result=dict(payload['result'],cache_hit=True,original_extraction_seconds=payload['result']['request_seconds'],request_seconds=time.perf_counter()-tick)
            else:
                corrected=image.rotate(angle,expand=True)
                with redirect_stdout(sys.stderr):values=reader.readtext(np.array(corrected))
                rows=[]
                for n,(polygon,text,confidence) in enumerate(values):
                    polygon=[[float(x),float(y)] for x,y in polygon];original=[transform(p,image.width,image.height,angle,True) for p in polygon]
                    rows.append(dict(id='easy-'+digest([identity,n,text,polygon])[:24],kind='ocr',text=text,confidence=float(confidence),status='complete',source=source,
                        corrected_polygon=polygon,corrected_box=bounds(polygon),original_polygon=original,original_box=bounds(original),box=bounds(original),polygon=original,coordinate_space='original-EXIF-RGB'))
                result=dict(evidence=rows,selected_evidence=[],supporting_evidence_ids=[],source=source,cache_hit=False,image_id=identity['image'],ocr_lines=len(rows),
                    transcription='\n'.join(r['text'] for r in rows),orientation={'selected_ccw_degrees':angle,'reason':'manual cardinal rotation; original EasyOCR detection','original_size':list(image.size),'working_size':list(image.size),'transform':matrices(image.size,image.size,angle)},
                    extraction_status={'ocr':{'status':'complete'}},request_seconds=time.perf_counter()-tick)
                with cache.open('x',encoding='utf-8') as f:json.dump({'identity':identity,'result':result,'sha256':digest(result)},f,ensure_ascii=False)
            send(dict(ok=True,result=result,memory=process_memory_mib()))
        except Exception as e:send(dict(ok=False,error=str(e)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session-dir',required=True);serve(parser.parse_args().session_dir)
