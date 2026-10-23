"""Persistent CUDA worker for the public runtime; no private report reads."""
from contextlib import redirect_stdout
import argparse
import json
import os
from pathlib import Path
import sys
import time

from persistent_demo_worker import Engine, FileGuard


class PublicEngine(Engine):
    def __init__(self, run_dir):
        from public_runtime import ROOT, verify, guarded_paths, pipeline
        from phase4_gpu_infer import prime_cuda
        start = time.perf_counter()
        self.timings = {}
        self.originals = []
        self.root = Path(run_dir).resolve()
        if ROOT / 'runtime-local' != self.root:
            raise ValueError('Public requests must use checkout runtime-local directory')
        self.root.mkdir(exist_ok=True)
        verify()
        self.verification_seconds = time.perf_counter() - start
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['YOLO_CONFIG_DIR'] = str(self.root / 'ultralytics')
        import torch
        self.reading_threads = torch.get_num_threads()
        tick = time.perf_counter()
        self.torch = prime_cuda()
        self.cuda_seconds = time.perf_counter() - tick
        from modules import vqa, ocr, object_detection, qges, evidence_store, evidence_decision, answerability_gate
        self.ocr, self.evidence_store = ocr, evidence_store
        for module, name, label in [
            (vqa, 'answer', 'blip_seconds_including_loading'), (vqa, 'load_blip', 'blip_loader_seconds'),
            (ocr, 'extract', 'ocr_seconds_including_loading'), (ocr, 'load_reader', 'ocr_loader_seconds'),
            (object_detection, 'extract', 'yolo_seconds_including_loading'),
            (qges, 'select', 'evidence_selection_seconds'), (evidence_decision, 'decide', 'ocr_answer_selection_seconds'),
            (answerability_gate, 'decide', 'learned_gate_decision_seconds'),
            (evidence_store, 'extract', 'evidence_cache_and_extraction_seconds')]:
            self.wrap(module, name, label)
        self.pipeline = pipeline(session_dir=self.root)
        self.guard = FileGuard(guarded_paths())
        self.startup_seconds = time.perf_counter() - start


def serve(run_dir):
    protocol = sys.stdout
    def send(value):
        protocol.write(json.dumps(value, ensure_ascii=False) + '\n')
        protocol.flush()
    try:
        with redirect_stdout(sys.stderr):
            engine = PublicEngine(run_dir)
        send(dict(ready=True, startup_seconds=engine.startup_seconds,
                  verification_seconds=engine.verification_seconds, cuda_seconds=engine.cuda_seconds))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        send(dict(ready=False, error=str(error)))
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('action') == 'shutdown':
                send({'stopped': True})
                break
            if request.get('action') != 'infer':
                raise ValueError('Unknown worker action')
            image = Path(request['image']).resolve()
            if engine.root not in image.parents:
                raise ValueError('Image must belong to this public session')
            with redirect_stdout(sys.stderr):
                result, timings = engine.infer(image, request['question'], bool(request.get('reading')))
            send(dict(ok=True, result=result, timings=timings))
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            send(dict(ok=False, error=str(error)))
            if 'out of memory' in str(error).lower():
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    serve(parser.parse_args().run_dir)
