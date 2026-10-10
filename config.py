from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    device: str = field(default_factory=lambda: os.getenv('VISIONBRIDGE_DEVICE', 'auto'))
    blip_model: str = 'Salesforce/blip-vqa-base'
    whisper_model: str = 'tiny.en'
    clip_model: str = 'openai/clip-vit-base-patch32'
    yolo_model: str = str(ROOT / 'models' / 'yolov8n.pt')
    cache_dir: Path = ROOT / 'models' / 'cache'
    output_dir: Path = ROOT / 'outputs'
    qges_checkpoint: Path = ROOT / 'models' / 'qges'
    fusion_checkpoint: Path = ROOT / 'models' / 'fusion'
    answerability_checkpoint: Path = ROOT / 'models' / 'answerability'
    enable_trained: bool = False
    enable_qges: bool = True
    max_regions: int = 5
