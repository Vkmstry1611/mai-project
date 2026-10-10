from utils.model_loader import cached_loader, resolve_device

@cached_loader
def load_reader(cache_dir, device):
    import easyocr
    return easyocr.Reader(['en'], gpu=device == 'cuda', model_storage_directory=cache_dir, verbose=False)

def extract(image, config):
    import numpy as np
    rows = load_reader(str(config.cache_dir / 'easyocr'), resolve_device(config.device)).readtext(np.array(image))
    return [{'kind': 'ocr', 'text': text, 'polygon': [[float(x), float(y)] for x, y in polygon],
             'box': [float(min(p[0] for p in polygon)), float(min(p[1] for p in polygon)),
                     float(max(p[0] for p in polygon)), float(max(p[1] for p in polygon))],
             'confidence': float(score)} for polygon, text, score in rows]
