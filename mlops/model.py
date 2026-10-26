"""Strict, offline BLIP checkpoint loading for CPU serving and demo evaluation."""

import hashlib
from pathlib import Path


REQUIRED = (
    "config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer_config.json", "vocab.txt",
)


def model_sha256(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in REQUIRED:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_model(model_dir: Path, *, device: str = "cpu"):
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor

    torch.set_num_threads(int(__import__("os").environ.get("TORCH_NUM_THREADS", "2")))
    model_dir = Path(model_dir)
    model_sha256(model_dir)  # also rejects incomplete downloads
    processor = BlipProcessor.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)
    model, loading = BlipForQuestionAnswering.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True,
        output_loading_info=True,
    )
    # Transformers 4.46 does not restore this omitted tied projection on its own.
    missing = set(loading.get("missing_keys") or ())
    tied = "text_decoder.cls.predictions.decoder.weight"
    if missing == {tied} and getattr(model.config.text_config, "tie_word_embeddings", False):
        embedding = model.text_decoder.bert.embeddings.word_embeddings.weight
        decoder = model.text_decoder.cls.predictions.decoder
        if tuple(embedding.shape) != tuple(decoder.weight.shape):
            raise RuntimeError("BLIP tied weight shape mismatch")
        decoder.weight = embedding
        missing.clear()
    if missing or any(loading.get(k) for k in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Incomplete BLIP checkpoint: {loading}")
    return processor, model.to(device).eval()


def predict(processor, model, image, question: str) -> str:
    import torch

    inputs = processor(image, question, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=20, num_beams=1, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True).strip()
