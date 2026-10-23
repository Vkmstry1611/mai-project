"""Opt-in component smoke test; synthetic inputs only, no evaluation dataset."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', action='store_true', help='Load CUDA BLIP, both OCR readers, YOLO and classifier')
    parser.add_argument('--speech', action='store_true', help='Initialize pinned CPU Whisper and synthesize a SAPI WAV')
    parser.add_argument('--deny-private-project', type=Path, help='Validation guard: deny reads from another checkout models/reports/.deps')
    args = parser.parse_args()
    if args.deny_private_project:
        import sys
        original = args.deny_private_project.resolve()
        here = Path(__file__).resolve().parent
        def guard(event, values):
            if event == 'open' and values and isinstance(values[0], (str, bytes)):
                p = Path(values[0]).resolve()
                if here == p or here in p.parents:
                    if here / 'reports' in p.parents:
                        raise PermissionError('Private worktree report access denied')
                    return
                if any(original / n in p.parents for n in ('models', 'reports', '.deps')):
                    raise PermissionError('Private original artifact access denied')
        sys.addaudithook(guard)
    from public_runtime import verify, ROOT, load_gate, pipeline
    verify()
    results = {'public_integrity': 'passed', 'historical_reports_required': False}
    if args.models:
        from phase4_gpu_infer import prime_cuda
        torch = prime_cuda()
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('RGB', (640, 240), 'white')
        ImageDraw.Draw(image).text((60, 70), 'HELLO', fill='black', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 80))
        start = time.perf_counter()
        instance = pipeline()
        result = instance.run(image, 'What color is the background?')
        if result[3].get('baseline_answer') is None:
            raise RuntimeError('BLIP smoke test failed: ' + str(result[3]))
        results['vqa'] = {'answer': result[0], 'seconds': time.perf_counter() - start}
        from modules.ocr import load_reader
        import numpy as np
        reader = load_reader(str(ROOT / 'models/cache/easyocr'), 'cpu')
        results['easyocr'] = [r[1] for r in reader.readtext(np.array(image))]
        if not any('HELLO' in t.upper() for t in results['easyocr']):
            raise RuntimeError('EasyOCR did not read synthetic HELLO')
        from modules.object_detection import extract
        results['yolo_candidates'] = len(extract(image, instance.config))
        results['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        # Run RapidOCR separately with .venv-ocr; do not mix ONNX envs here.
    if args.speech:
        from public_models import verify_whisper
        from modules.speech_to_text import load_whisper
        load_whisper(str(verify_whisper()), str(ROOT / 'models/cache'))
        from modules.text_to_speech import synthesize
        from config import Config
        audio = synthesize('VisionBridge public runtime test.', Config())
        from public_runtime_session import transcribe
        text, _ = transcribe(audio)
        if not text.strip():
            raise RuntimeError('Synthetic SAPI-to-Whisper transcription empty')
        results['speech'] = {'wav_generated': True, 'synthetic_transcript': text,
                             'physical_microphone_or_audible_playback_tested': False}
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
