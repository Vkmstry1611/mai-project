"""Separate opt-in worker. Original frozen/manual requests delegate unchanged."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

from persistent_demo_worker import Engine


class OrientationEngine(Engine):
    def infer_auto(self, path, question):
        from PIL import Image, ImageOps
        from config import Config
        from experimental_reading import is_reading_request
        from experimental_orientation import read, process_memory_mib
        if Path(path).name.startswith('VizWiz_'):
            raise ValueError('Demonstration inputs only; evaluation dataset requests forbidden')
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000 or not is_reading_request(question):
            raise ValueError('Explicit literal text-reading request required')
        self.guard.check()
        self.timings = {}
        start = time.perf_counter()
        self.torch.set_num_threads(self.reading_threads)
        self.torch.cuda.reset_peak_memory_stats()
        cfg = Config()
        if not all((cfg.cache_dir/'easyocr'/name).is_file() for name in ('craft_mlt_25k.pth', 'english_g2.pth')):
            raise FileNotFoundError('Verified local EasyOCR weights are required; no download attempted')
        image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        source = self.evidence_store.descriptor('ocr', cfg)
        if source['resolved_device'] != 'cpu':
            raise ValueError('Automatic orientation OCR must use CPU')
        details = read(image, self.root/'experimental-orientation-cache', source,
                       lambda: self.ocr.load_reader(str(cfg.cache_dir/'easyocr'), 'cpu'))
        orientation = details['orientation']
        self.timings.update(orientation_cache_hit=orientation['cache_hit'],
            orientation_selected_ccw=orientation['selected_ccw_degrees'],
            orientation_seconds=orientation['request_seconds'], inference_seconds=time.perf_counter()-start,
            peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=self.torch.cuda.max_memory_reserved()/2**20, **process_memory_mib())
        status = ('Experimental OCR fragments; automatic rotation '+str(orientation['selected_ccw_degrees'])+' degrees CCW. '
                  'Reading may be incomplete.') if orientation['selected_ccw_degrees'] is not None else 'Experimental OCR abstained: '+orientation['decision_reason']
        return dict(display_answer=details['answer'], status=status, details=details), dict(self.timings)


def serve(run_dir):
    if Path(sys.prefix).name != '.venv-gpu':
        raise RuntimeError('Use .venv-gpu for this worker')
    protocol = sys.stdout
    def send(value):
        protocol.write(json.dumps(value, ensure_ascii=False)+'\n')
        protocol.flush()
    try:
        with redirect_stdout(sys.stderr):
            engine = OrientationEngine(run_dir)
        send(dict(ready=True, startup_seconds=engine.startup_seconds,
                  verification_seconds=engine.verification_seconds, cuda_seconds=engine.cuda_seconds))
    except Exception as error:
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
            with redirect_stdout(sys.stderr):
                if request.get('automatic_orientation'):
                    if not request.get('reading'):
                        raise ValueError('Automatic orientation is forbidden for frozen VQA')
                    result, timings = engine.infer_auto(request['image'], request['question'])
                else:
                    result, timings = engine.infer(request['image'], request['question'], bool(request.get('reading')))
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
