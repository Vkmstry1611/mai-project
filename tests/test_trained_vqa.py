from pathlib import Path
from unittest.mock import patch
import zipfile
import pytest
from PIL import Image
from config import Config
from modules.vqa import model_source
from pipeline import Pipeline
from utils.checkpoint_import import import_checkpoint

def local_config(tmp_path):
    config = Config(trained_blip_dir=tmp_path)
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'):
        (tmp_path / name).write_text('fixture')
    return config

def test_auto_and_explicit_selection(tmp_path):
    config = Config(trained_blip_dir=tmp_path)
    assert model_source(config)[0] == config.blip_model
    with pytest.raises(FileNotFoundError):
        model_source(config, 'trained')
    config = local_config(tmp_path)
    assert model_source(config)[0] == str(tmp_path)
    assert model_source(config, 'pretrained')[0] == config.blip_model

def test_request_choice_and_status(tmp_path):
    pipeline = Pipeline(local_config(tmp_path))
    with patch('modules.vqa.answer', return_value='two') as infer:
        result = pipeline.run(Image.new('RGB', (20, 20)), 'how many?', speak=False, vqa_mode='trained')
    assert 'Local trained BLIP' in result[3]['mode']
    assert infer.call_args.args[2].vqa_mode == 'trained'
    assert pipeline.config.vqa_mode == 'auto'
    assert result[3]['grounded'] is False
    assert result[3]['answerability']['probability'] is None

def test_model_abstention(tmp_path):
    with patch('modules.vqa.answer', return_value='unanswerable'):
        result = Pipeline(local_config(tmp_path)).run(Image.new('RGB', (20, 20)), 'what?', speak=False)
    assert 'returned unanswerable' in result[0]
    assert result[3]['answerability']['answerable'] is None

def test_archive_path_rejection(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('blip_vizwiz/../../outside.py', 'bad')
    with pytest.raises(ValueError):
        import_checkpoint(archive, tmp_path / 'target')
    assert not (tmp_path / 'target').exists()
