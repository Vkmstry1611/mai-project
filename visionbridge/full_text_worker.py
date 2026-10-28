"""Persistent CPU ONNX reader in .venv-ocr, isolated from CUDA and frozen VQA."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time


def serve(base):
    from visionbridge.full_text_ocr import FullTextEngine
    from visionbridge.experimental_orientation import process_memory_mib
    base=Path(base).resolve();protocol=sys.stdout
    def send(x):protocol.write(json.dumps(x,ensure_ascii=False)+'\n');protocol.flush()
    try:
        tick=time.perf_counter()
        with redirect_stdout(sys.stderr):engine=FullTextEngine()
        send(dict(ready=True,startup_seconds=time.perf_counter()-tick,source=engine.source))
    except Exception as e:send(dict(ready=False,error=str(e)));return
    for line in sys.stdin:
        try:
            r=json.loads(line)
            if r.get('action')=='shutdown':send({'stopped':True});break
            if r.get('action')!='read':raise ValueError('Unsupported full-text action')
            path=Path(r['image']).resolve()
            if base not in path.parents:raise ValueError('Use this worker session temporary image directory')
            from PIL import Image
            namespace=r.get('cache_namespace','cache')
            if not isinstance(namespace,str) or not namespace or not all(c.isalnum() or c=='-' for c in namespace):raise ValueError('Invalid cache namespace')
            with redirect_stdout(sys.stderr):result=engine.read(Image.open(path),base/namespace,r.get('manual'))
            send(dict(ok=True,result=result,memory=process_memory_mib()))
        except Exception as e:send(dict(ok=False,error=str(e)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session-dir',required=True);serve(p.parse_args().session_dir)
