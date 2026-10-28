"""Opt-in transport extension; no monkey-patches to original worker sources."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from visionbridge.persistent_demo_client import ROOT, WorkerClient
from visionbridge.launch_persistent_demo import Session


class OrientationClient(WorkerClient):
    automatic = False

    def start(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is not None:
                    raise RuntimeError('Worker stopped; restart this workspace')
                return
            out = ROOT/'reports/experimental_reading'/('worker-'+uuid.uuid4().hex)
            out.mkdir(parents=True, exist_ok=False)
            self.log = (out/'worker-stderr.log').open('x', encoding='utf-8')
            tick = time.perf_counter()
            self.process = subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'), '-u',
                '-m', 'visionbridge.orientation_reading_worker', '--run-dir', str(self.run_dir)], cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf-8',
                env=dict(os.environ, PYTHONIOENCODING='utf-8'), creationflags=0x08000000)
            threading.Thread(target=self._read, daemon=True).start()
            ready = self.receive()
            self.startup = dict(ready, observed_worker_startup_seconds=time.perf_counter()-tick, log_directory=str(out))
            (out/'startup.json').write_text(json.dumps(self.startup, indent=2), encoding='utf-8')
            if not ready.get('ready'):
                self.close(force=True)
                raise RuntimeError('Orientation worker initialization failed: '+ready.get('error', 'unknown'))

    def infer(self, image, question, reading):
        with self.lock:
            self.start()
            self.process.stdin.write(json.dumps(dict(action='infer', image=str(image), question=question,
                reading=bool(reading), automatic_orientation=bool(reading and self.automatic)))+'\n')
            self.process.stdin.flush()
            value = self.receive()
            if not value.get('ok'):
                raise RuntimeError('Inference failed; no substitute: '+value.get('error', 'unknown'))
            return value


class OrientationSession(Session):
    def __init__(self, run_dir, client=None):
        super().__init__(run_dir, client or OrientationClient(run_dir))
        self.mode_lock = threading.RLock()

    def request(self, *args, **kwargs):
        # Advanced manual rotation and all frozen VQA requests remain original.
        with self.mode_lock:
            self.client.automatic = False
            return super().request(*args, **kwargs)

    def request_auto(self, image, question, speak, experimental, rotation='0'):
        if not experimental:
            return self.request(image, question, speak, False, '0')
        with self.mode_lock:
            self.client.automatic = True
            try:
                response = super().request(image, question, speak, True, '0')
            finally:
                self.client.automatic = False
        details = response[3]
        orientation = details.get('orientation', {})
        angle = orientation.get('selected_ccw_degrees')
        if angle is None:
            return response
        from PIL import ImageOps
        from utils.image_utils import annotate
        from visionbridge.experimental_orientation import MAX_SIDE
        corrected = ImageOps.exif_transpose(image).convert('RGB')
        corrected.thumbnail((MAX_SIDE, MAX_SIDE))
        corrected = corrected.rotate(angle, expand=True)
        rows = [dict(e, box=e['corrected_box']) for e in details['evidence']]
        values = list(response)
        values[4] = annotate(corrected, rows)
        values[5] += '\nAutomatic rotation: '+str(angle)+' degrees counterclockwise. Preview boxes use corrected coordinates; JSON also preserves original coordinates.'
        return tuple(values)
