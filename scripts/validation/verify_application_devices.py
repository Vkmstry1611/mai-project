"""Inspect actual loaded model devices without generating VQA predictions."""
import json
from pathlib import Path
import sys

from phase4_gpu_infer import prime_cuda, verify_application


def main(output):
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch = prime_cuda()
    from phase4_pipeline import Phase4Pipeline
    from modules import ocr, object_detection, vqa
    from PIL import Image
    pipeline = Phase4Pipeline('reports/phase4_policy/run-local-20261013-v1')
    cfg = pipeline.config
    object_detection.load_yolo(cfg.yolo_model).predict(Image.new('RGB', (64, 64), 'white'), device='cpu', verbose=False)
    detector = object_detection.load_yolo(cfg.yolo_model)
    reader = ocr.load_reader(str(cfg.cache_dir / 'easyocr'), 'cpu')
    _, model, resolved = vqa.load_blip(str(cfg.improved_blip_dir), str(cfg.cache_dir), 'cuda')
    result = {'blip_resolved_device': resolved, 'blip_parameter_device': str(next(model.parameters()).device),
              'blip_parameter_dtype': str(next(model.parameters()).dtype), 'ocr_device': reader.device,
              'yolo_device': str(next(detector.model.parameters()).device),
              'gpu': torch.cuda.get_device_name(0), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'cuda_primed_before_cpu_yolo': True, 'vqa_generation_calls': 0}
    assert resolved == 'cuda' and result['blip_parameter_device'].startswith('cuda')
    assert result['blip_parameter_dtype'] == 'torch.float32'
    assert result['ocr_device'] == 'cpu' and result['yolo_device'] == 'cpu'
    verify_application('reports/phase4_policy/run-local-20261013-v1')
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
