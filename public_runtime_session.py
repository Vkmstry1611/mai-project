"""Public workspace transport; reuses all experimental conversation/OCR routes."""
from pathlib import Path
import json
import os
import subprocess
import threading
import time
import uuid

from full_text_session import FullTextSession, FullTextClient, EasyTextClient, SceneClient
from persistent_demo_client import WorkerClient, ROOT


class PublicWorkerClient(WorkerClient):
    def start(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is not None:
                    raise RuntimeError('Public worker stopped; restart workspace. No fallback.')
                return
            folder = self.run_dir / 'worker-logs' / uuid.uuid4().hex
            folder.mkdir(parents=True, exist_ok=False)
            self.log = (folder / 'stderr.log').open('x', encoding='utf-8')
            executable = Path(os.environ.get('VISIONBRIDGE_GPU_PYTHON', ROOT / '.venv-gpu/Scripts/python.exe'))
            if not executable.is_file():
                raise FileNotFoundError('Create .venv-gpu with setup_public_runtime.ps1')
            tick = time.perf_counter()
            self.process = subprocess.Popen([str(executable), '-u', str(ROOT / 'public_runtime_worker.py'),
                '--run-dir', str(self.run_dir)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding='utf-8', env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                creationflags=0x08000000 if os.name == 'nt' else 0)
            threading.Thread(target=self._read, daemon=True).start()
            self.startup = self.receive()
            self.startup['observed_process_startup_seconds'] = time.perf_counter() - tick
            if not self.startup.get('ready'):
                self.close(force=True)
                raise RuntimeError('Public runtime initialization failed: ' + self.startup.get('error', 'unknown'))


class PublicVisualSession:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.client = PublicWorkerClient(run_dir)

    def request(self, image, question, speak, experimental, rotation, run_dir=None):
        from PIL import ImageOps
        from experimental_reading import is_reading_request
        from submission_ui_helpers import input_error, with_evidence_text
        from utils.image_utils import annotate
        start = time.perf_counter()
        error = input_error(image, question)
        if error:
            raise ValueError(error)
        reading = bool(experimental and is_reading_request(question))
        normalized = ImageOps.exif_transpose(image).convert('RGB')
        if reading:
            normalized = normalized.rotate(int(rotation), expand=True)
        folder = self.run_dir / 'requests' / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / 'input.png'
        normalized.save(path)
        value = self.client.infer(path, question, reading)
        result = value['result']
        audio, status = None, result['status']
        if speak:
            try:
                from modules.text_to_speech import synthesize
                from config import Config
                from dataclasses import replace
                audio = synthesize(result['display_answer'], replace(Config(), output_dir=folder / 'speech'))
            except Exception as error:
                status += '; spoken answer unavailable: ' + str(error)
        details = result['details']
        timings = dict(persistent_worker=True, worker_stages=value['timings'], startup=self.client.startup,
                       callback_seconds=time.perf_counter() - start, experimental_reading_active=reading)
        output = (result['display_answer'], audio, status, details, annotate(normalized, details.get('evidence', [])))
        return (*with_evidence_text(output), timings)

    def close(self):
        self.client.close()


class PublicSceneClient(SceneClient):
    worker = 'public_scene_worker.py'


class PublicSession(FullTextSession):
    def __init__(self, run_dir=None):
        self.run_dir = ROOT / 'runtime-local'
        self.legacy = PublicVisualSession(self.run_dir)
        self.reader = FullTextClient(self.run_dir)
        self.easy = EasyTextClient(self.run_dir)
        self.scene = PublicSceneClient(self.run_dir)
        self.engine_choice = 'Automatic'
        # Explicit interpreter overrides support isolated validation using the
        # installed packages. Normal installation uses the documented local venvs.
        gpu = os.environ.get('VISIONBRIDGE_GPU_PYTHON')
        cpu = os.environ.get('VISIONBRIDGE_OCR_PYTHON')
        if gpu:
            self.easy.environment = self.scene.environment = str(Path(gpu).parent.parent)
        if cpu:
            self.reader.environment = str(Path(cpu).parent.parent)

    def for_conversation(self, state, image):
        # Cache hits must also invalidate when deployment artifacts change.
        # Keep the same startup-SHA / per-request-stat guard as the worker.
        from public_runtime import verify, guarded_paths
        from persistent_demo_worker import FileGuard
        if not hasattr(self, '_public_guard'):
            verify()
            self._public_guard = FileGuard(guarded_paths())
        self._public_guard.check()
        return super().for_conversation(state, image)


def transcribe(recording):
    from config import Config
    from dataclasses import replace
    from modules.speech_to_text import transcribe as existing_transcribe
    from public_models import verify_whisper
    folder = verify_whisper()
    text = existing_transcribe(recording, replace(Config(), whisper_model=str(folder)))
    return text, 'Transcribed; review and edit your question.'
