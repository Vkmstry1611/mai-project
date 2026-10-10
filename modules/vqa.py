from utils.model_loader import cached_loader, resolve_device, MODEL_LOCK

@cached_loader
def load_blip(model_id, cache_dir, requested_device):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    device = resolve_device(requested_device)
    processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = BlipForQuestionAnswering.from_pretrained(model_id, cache_dir=cache_dir)
    return processor, model.to(device).eval(), device

def answer(image, question, config):
    import torch
    processor, model, device = load_blip(config.blip_model, str(config.cache_dir), config.device)
    with MODEL_LOCK, torch.inference_mode():
        inputs = processor(image, question, return_tensors='pt').to(device)
        tokens = model.generate(**inputs, max_new_tokens=20, num_beams=1)
        return processor.decode(tokens[0], skip_special_tokens=True).strip()
