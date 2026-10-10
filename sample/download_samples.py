"""Download four individual demonstration images, not a dataset."""
from pathlib import Path
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parent
URLS = {
    'bus.jpg': 'https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg',
    'people.jpg': 'https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg',
    'cats.png': 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png',
    'stop_sign.png': 'https://upload.wikimedia.org/wikipedia/commons/8/81/Stop_sign.png',
}

if __name__ == '__main__':
    for name, url in URLS.items():
        path = ROOT / name
        if path.exists():
            print('Already present:', name)
            continue
        request = urllib.request.Request(url, headers={'User-Agent': 'VisionBridge-local-samples/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError('Unexpectedly large sample download')
        path.write_bytes(data)
        with Image.open(path) as image:
            image.verify()
        print(name, len(data), 'bytes')
