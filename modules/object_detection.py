from utils.model_loader import cached_loader, resolve_device

@cached_loader
def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)

def extract(image, config):
    result = load_yolo(config.yolo_model).predict(image, device=resolve_device(config.device), verbose=False)[0]
    return [{'kind': 'object', 'text': result.names[int(b.cls.item())],
             'box': b.xyxy[0].tolist(), 'confidence': float(b.conf.item())} for b in result.boxes]
