from utils.model_loader import cached_loader, resolve_device

@cached_loader
def load_clip(model_id, cache_dir, requested_device):
    from transformers import CLIPModel, CLIPProcessor
    device = resolve_device(requested_device)
    return (CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir),
            CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval(), device)

def extract(image, question, evidence, config):
    import torch
    processor, model, device = load_clip(config.clip_model, str(config.cache_dir), config.device)
    regions = [image] + [image.crop(item['box']) for item in evidence[:config.max_regions] if item.get('box')]
    with torch.inference_mode():
        images = model.get_image_features(**processor(images=regions, return_tensors='pt').to(device))
        text = model.get_text_features(**processor(text=[question], return_tensors='pt', padding=True, truncation=True).to(device))
        images = images / images.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
    return {'image_embedding': images[0].cpu().tolist(), 'region_embeddings': images[1:].cpu().tolist(),
            'question_similarity': (images @ text.T).flatten().cpu().tolist()}
