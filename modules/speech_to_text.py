from utils.model_loader import cached_loader

@cached_loader
def load_whisper(model_name, cache_dir):
    from faster_whisper import WhisperModel
    # CPU int8 avoids CUDA/cuDNN compatibility issues and suits small Windows machines.
    return WhisperModel(model_name, device='cpu', compute_type='int8', download_root=cache_dir)

def transcribe(audio_path, config):
    if not audio_path:
        raise ValueError('Record or upload audio first.')
    segments, _ = load_whisper(config.whisper_model, str(config.cache_dir)).transcribe(audio_path, beam_size=1)
    text = ' '.join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise ValueError('No speech was recognized. Try again or type your question.')
    return text
