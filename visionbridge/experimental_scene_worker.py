"""Optional CPU YOLO enumeration; existing local weights, no downloads or BLIP."""
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout


def serve(base):
    base=Path(base).resolve();protocol=sys.stdout
    def send(value):protocol.write(json.dumps(value)+'\n');protocol.flush()
    try:
        from config import Config
        from dataclasses import replace
        from modules.object_detection import extract
        from modules.evidence_store import digest
        from visionbridge.experimental_orientation import process_memory_mib
        from PIL import Image
        from phase3_common import MODEL_CACHE
        cfg=replace(Config(),device='cpu',yolo_model=str(MODEL_CACHE/'yolov8n.pt'))
        if not Path(cfg.yolo_model).is_file():raise FileNotFoundError('Existing local YOLO weights unavailable; no download permitted.')
        send({'ready':True})
    except Exception as e:send({'ready':False,'error':type(e).__name__});return
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='shutdown':send({'stopped':True});break
            p=Path(r['image']).resolve()
            if r.get('action')!='read' or base not in p.parents:raise ValueError('Invalid scene request')
            with redirect_stdout(sys.stderr):items=extract(Image.open(p).convert('RGB'),cfg)
            for i,item in enumerate(items):
                item.update(id='scene-'+digest([i,item])[:24],source={'engine':'YOLOv8n','device':'cpu'},status='complete')
            send({'ok':True,'result':{'evidence':items,'source':{'engine':'YOLOv8n','device':'cpu'}},'memory':process_memory_mib()})
        except Exception as e:send({'ok':False,'error':type(e).__name__})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--session-dir',required=True);serve(p.parse_args().session_dir)
