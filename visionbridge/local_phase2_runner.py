"""Windows GPU wrapper around the existing evaluator. No training or fallback model."""
import argparse
import json
from pathlib import Path
import sys
import time

import evaluate_phase2_dev as evaluation
from export_phase2_dev import validate_dev
from visionbridge_training import verify_checkpoint, sha256


def verify_results(results, rows, mode):
    expected = rows[:20] if mode == 'pilot' else rows
    records = {}
    for name in ('A', 'B', 'C'):
        values = json.loads((results / f'{name}_records.json').read_text(encoding='utf-8'))
        assert len(values) == len(expected)
        assert all(r['image'] == s['image'] and r['question'] == s['question'] and
                   r['answers'] == s['answers'] and r['answerable'] == s['answerable']
                   for r, s in zip(values, expected)), 'Development rows/references changed.'
        records[name] = values
    assert all(a['baseline_answer'] == b['baseline_answer'] == c['baseline_answer']
               for a, b, c in zip(records['A'], records['B'], records['C']))
    if any(s['status'] == 'unavailable' for r in records['B'] for s in r['extraction_status'].values()):
        raise RuntimeError('Extraction failures detected. Results preserved; fix these before the full evaluation.')
    # Partial evidence is an expected conservative-retain case in the existing
    # app. Keep it in denominators and report it rather than rejecting the data.
    return records


def execute(root, mode, half=False):
    import os
    os.environ['YOLO_CONFIG_DIR'] = str(Path(root).resolve() / 'model-cache/ultralytics-settings')
    os.environ['MPLCONFIGDIR'] = str(Path(root).resolve() / 'model-cache/matplotlib')
    import torch
    from modules import vqa
    root = Path(root).resolve()
    rows = validate_dev(json.loads((root / 'dev-only.json').read_text(encoding='utf-8')))
    checkpoint = Path(__file__).resolve().parents[1] / 'models/blip_vizwiz_v2'
    before = verify_checkpoint(checkpoint)
    for name in ('generation_config.json', 'special_tokens_map.json'):
        before[name] = sha256(checkpoint / name)
    evaluation.CODE_FILES += ('visionbridge/local_phase2_runner.py',)
    evaluation.PROTOCOL['weight_precision'] = 'FP16' if half else 'FP32'
    if half:
        original = vqa.answer
        def predict(image, question, cfg):
            source, _ = vqa.model_source(cfg)
            _, model, _ = vqa.load_blip(source, str(cfg.cache_dir), cfg.device)
            model.half()  # In-memory only; all rows in this attempt use this precision.
            return original(image, question, cfg)
        vqa.answer = predict
    output = root / (mode + ('-fp16' if half else ''))
    args = ['--dev-manifest', str(root / 'dev-only.json'), '--images', str(root / 'data/images'),
            '--checkpoint', str(checkpoint), '--output', str(output),
            '--evidence-cache', str(root / 'evidence-cache'), '--model-cache', str(root / 'model-cache'),
            '--yolo-model', str(root / 'model-cache/yolov8n.pt'), '--device', 'cuda',
            '--mode', mode, '--limit', '20']
    if output.exists():
        args.append('--resume')
    start = time.perf_counter()
    try:
        results = evaluation.run(evaluation.parser().parse_args(args))
    except torch.cuda.OutOfMemoryError:
        print('CUDA OOM: attempt preserved. Retry with --half in a fresh process/directory; no mixed precision captures.', flush=True)
        return 42
    records = verify_results(results, rows, mode)
    assert all(sha256(checkpoint / name) == value for name, value in before.items()), 'Checkpoint changed!'
    evidence = {'mode': mode, 'examples': len(records['A']), 'exact_dev_order_and_references': True,
                'holdout_evaluated': False, 'checkpoint_unchanged': True, 'checkpoint_hashes': before,
                'weight_precision': 'FP16' if half else 'FP32', 'wall_seconds': time.perf_counter() - start}
    evaluation.write_new(results / 'verification.json', evidence)
    print('VERIFIED_RESULTS=' + str(results), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--mode', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()
    if Path(sys.prefix).name != '.venv-gpu':
        raise RuntimeError('Use .venv-gpu/Scripts/python.exe; original .venv must remain untouched.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback permitted.')
    print('GPU:', torch.cuda.get_device_name(0), 'free/total bytes:', torch.cuda.mem_get_info(), flush=True)
    sys.exit(execute(args.root, args.mode, args.half))
