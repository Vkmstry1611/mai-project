from utils.model_loader import cached_loader, resolve_device, MODEL_LOCK
from pathlib import Path

def model_source(config, choice=None):
    mode = choice or config.vqa_mode
    if mode not in {'auto', 'trained', 'pretrained'}:
        raise ValueError('VQA mode must be auto, trained or pretrained.')
    ready = all((config.trained_blip_dir / name).is_file() for name in
                ('config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt'))
    if mode == 'trained' and not ready:
        raise FileNotFoundError('Trained BLIP checkpoint is missing. Import the ZIP or select pretrained BLIP.')
    if mode == 'trained' or (mode == 'auto' and ready):
        return str(config.trained_blip_dir), 'Local trained BLIP (blip_vizwiz)'
    return config.blip_model, 'Pretrained BLIP baseline'

@cached_loader
def load_blip(model_id, cache_dir, requested_device):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    device = resolve_device(requested_device)
    local = Path(model_id).is_dir()
    processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=local, use_fast=False)
    model, loading_info = BlipForQuestionAnswering.from_pretrained(
        model_id, cache_dir=cache_dir, local_files_only=local, use_safetensors=True,
        output_loading_info=True)
    if loading_info['missing_keys'] or loading_info['unexpected_keys'] or loading_info.get('mismatched_keys'):
        raise RuntimeError('BLIP checkpoint architecture does not match the loader; refusing partial weights.')
    return processor, model.to(device).eval(), device

def answer(image, question, config):
    import torch
    source, _ = model_source(config)
    processor, model, device = load_blip(source, str(config.cache_dir), config.device)
    with MODEL_LOCK, torch.inference_mode():
        inputs = processor(image, question, return_tensors='pt', truncation=True, max_length=512).to(device)
        tokens = model.generate(**inputs, max_new_tokens=20, num_beams=1)
        return processor.decode(tokens[0], skip_special_tokens=True).strip()
