"""Explicit adapter contract for future checkpoints; no assumed tensor architecture."""
import importlib.util

def load_adapter(folder, device):
    source = folder / 'adapter.py'
    if not source.is_file():
        raise FileNotFoundError(f'No adapter at {source}')
    spec = importlib.util.spec_from_file_location('visionbridge_adapter_' + folder.name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load(folder, device)
