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
    def load(*args):
        with MODEL_LOCK:
            return cached(*args)
    return load
