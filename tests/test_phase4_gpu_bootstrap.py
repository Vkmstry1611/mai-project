from types import SimpleNamespace
import pytest
from phase4_gpu_infer import prime_cuda
from launch_phase4 import GPUWorker,ROOT


def test_cuda_is_primed_before_device_resolution(monkeypatch):
    import utils.model_loader
    events=[]
    def resolve(device):
        assert events[0]=='cuda-init'
        events.append('resolve-'+device);return 'cuda'
    resolve.cache_clear=lambda:events.append('cache-clear')
    monkeypatch.setattr(utils.model_loader,'resolve_device',resolve)
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,init=lambda:events.append('cuda-init'),manual_seed_all=lambda s:events.append('cuda-seed')),
        set_num_threads=lambda n:events.append('threads-'+str(n)),manual_seed=lambda n:events.append('seed'))
    prime_cuda(fake)
    assert events[:3]==['cuda-init','cache-clear','resolve-cuda'] and 'threads-4' in events


def test_missing_cuda_is_not_a_cpu_fallback():
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    with pytest.raises(RuntimeError,match='fallback is forbidden'):prime_cuda(fake)


def test_worker_redirect_preserves_arguments_and_real_subprocess_api(monkeypatch):
    import launch_phase4,subprocess
    calls=[]
    monkeypatch.setattr(launch_phase4.subprocess,'run',lambda command,**options:calls.append((command,options)) or 'ok')
    command=['gpu-python',str(ROOT/'phase4_infer.py'),'--question','What brand?']
    assert GPUWorker.run(command,capture_output=True)=='ok'
    assert calls[0][0][1]==str(ROOT/'phase4_gpu_infer.py')
    assert calls[0][0][2:]==command[2:]
    assert command[1]==str(ROOT/'phase4_infer.py')


def test_unknown_worker_is_rejected():
    with pytest.raises(ValueError):GPUWorker.run(['python','other-script.py'])
