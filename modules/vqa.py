from utils.model_loader import cached_loader, resolve_device, MODEL_LOCK
from pathlib import Path

_BLIP_TIED_DECODER_WEIGHT = 'text_decoder.cls.predictions.decoder.weight'


def validate_blip_loading(model, loading_info):
    """Restore BLIP's omitted tied LM head in Transformers 4.46 safetensors loads.

    The BLIP VQA checkpoint stores the shared token embedding once. Transformers
    4.46 lists the decoder projection as missing, while newer BLIP metadata maps
    that projection to the decoder input embedding. Restore that exact tie and
    continue to reject every other incomplete or mismatched checkpoint.
    """
    missing = set(loading_info.get('missing_keys') or ())
    allowed = {_BLIP_TIED_DECODER_WEIGHT}
    if missing == allowed:
        text_config = getattr(model.config, 'text_config', None)
        if not getattr(text_config, 'tie_word_embeddings', False):
            raise RuntimeError('BLIP checkpoint omits the decoder head but does not declare tied token embeddings.')
        embeddings = model.text_decoder.bert.embeddings.word_embeddings
        decoder = model.text_decoder.cls.predictions.decoder
        if tuple(embeddings.weight.shape) != tuple(decoder.weight.shape):
            raise RuntimeError('BLIP tied decoder head and token embeddings have incompatible shapes.')
        decoder.weight = embeddings.weight
        missing.clear()
    remaining = dict(loading_info)
    remaining['missing_keys'] = sorted(missing)
    if any(remaining.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('BLIP checkpoint architecture does not match the loader; refusing partial weights.')
    if missing:
        raise RuntimeError('BLIP checkpoint has unresolved missing weights.')
    return model


def model_source(config, choice=None):
    mode = choice or config.vqa_mode
    if mode not in {'auto', 'trained', 'pretrained', 'v2'}:
        raise ValueError('VQA mode must be auto, trained, pretrained or v2.')
    if mode == 'v2':
        folder = Path(config.improved_blip_dir)
        if folder.resolve() == Path(config.trained_blip_dir).resolve():
            raise ValueError('BLIP v2 must use a separate checkpoint directory, not the original model.')
        required = ('config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.txt')
        if not all((folder / name).is_file() for name in required):
            raise FileNotFoundError(f'BLIP v2 checkpoint incomplete: {folder}. No fallback is permitted.')
        return str(folder), 'Improved BLIP v2'
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
    # Opt-in fallback used only after the launcher observes a CUDA OOM. Half
    # precision weights halve model VRAM; normal evaluation retains Phase 1's
    # FP32 weights and autocast behaviour.
    import os
    load_options = {'torch_dtype': __import__('torch').float16} if os.environ.get('VISIONBRIDGE_LOW_MEMORY_VQA') == '1' else {}
    model, loading_info = BlipForQuestionAnswering.from_pretrained(
        model_id, cache_dir=cache_dir, local_files_only=local, use_safetensors=True,
        output_loading_info=True, **load_options)
    model = validate_blip_loading(model, loading_info)
    return processor, model.to(device).eval(), device

def answer(image, question, config):
    import torch
    source, _ = model_source(config)
    processor, model, device = load_blip(source, str(config.cache_dir), config.device)
    with MODEL_LOCK, torch.inference_mode():
        inputs = processor(image, question, return_tensors='pt', truncation=True, max_length=128).to(device)
        from contextlib import nullcontext
        precision = torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else nullcontext()
        with precision:
            tokens = model.generate(**inputs, max_new_tokens=20, num_beams=1, do_sample=False)
        return processor.decode(tokens[0], skip_special_tokens=True).strip()
