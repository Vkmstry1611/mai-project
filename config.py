from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    device: str = field(default_factory=lambda: os.getenv('VISIONBRIDGE_DEVICE', 'auto'))
    blip_model: str = 'Salesforce/blip-vqa-base'
    trained_blip_dir: Path = ROOT / 'models' / 'blip_vizwiz'
    improved_blip_dir: Path = field(default_factory=lambda: Path(os.getenv('VISIONBRIDGE_V2_DIR', str(ROOT / 'models' / 'blip_vizwiz_v2'))))
    vqa_mode: str = field(default_factory=lambda: os.getenv('VISIONBRIDGE_VQA_MODE', 'auto'))
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
    enable_reliability_checks: bool = True
    object_presence_threshold: float = 0.35
    max_regions: int = 5
    ocr_device: str = 'cpu'
    object_device: str = 'cpu'
    evidence_cache_dir: Path = ROOT / 'models' / 'cache' / 'evidence'
    cache_evidence: bool = True
    ocr_min_score: float = 0.9  # Provisional readability filter; NOT calibrated or dev-selected.
