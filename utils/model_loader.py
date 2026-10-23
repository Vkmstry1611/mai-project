from functools import lru_cache
from threading import RLock

MODEL_LOCK = RLock()

@lru_cache(maxsize=3)
def resolve_device(requested='auto'):
    import torch
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu, or cuda.')
    return 'cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu'

def cached_loader(function):
    cached = lru_cache(maxsize=4)(function)
    last_blip = None
    def load(*args):
        nonlocal last_blip
        with MODEL_LOCK:
            # Clear old BLIP weights before allocating a different source/device.
            # Otherwise switching models can retain several models on a 4 GB GPU.
            if function.__name__ == 'load_blip' and args != last_blip:
                cached.cache_clear()
                last_blip = args
            return cached(*args)
    return load
