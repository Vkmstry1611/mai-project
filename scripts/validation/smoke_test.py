"""Real local inference checks. Downloads only the enabled pretrained models."""
import argparse
from PIL import Image, ImageDraw
from config import Config
from modules import vqa, text_to_speech, speech_to_text, object_detection, ocr, visual_features
from utils.model_loader import resolve_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', action='store_true')
    parser.add_argument('--evidence', action='store_true')
    args = parser.parse_args()
    config = Config()
    image = Image.new('RGB', (400, 300), 'white')
    ImageDraw.Draw(image).rectangle((80, 60, 300, 220), fill='red')
    print('Device:', resolve_device(config.device), flush=True)
    print('BLIP:', vqa.answer(image, 'What color is the rectangle?', config), flush=True)
    wav = text_to_speech.synthesize('What color is the rectangle?', config)
    print('Playable WAV:', wav, flush=True)
    if args.speech:
        print('Whisper:', speech_to_text.transcribe(wav, config), flush=True)
    if args.evidence:
        evidence = object_detection.extract(image, config) + ocr.extract(image, config)
        print('Evidence:', evidence, flush=True)
        print('CLIP embedding dimensions:', len(visual_features.extract(image, 'red rectangle', evidence, config)['image_embedding']), flush=True)
