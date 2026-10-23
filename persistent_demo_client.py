"""Single local stdio worker, serialized requests and explicit shutdown."""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parent


class WorkerClient:
    def __init__(self,run_dir,timeout=240):
        self.run_dir=Path(run_dir).resolve();self.timeout=timeout;self.process=None
        self.lock=threading.RLock();self.messages=queue.Queue();self.startup={};self.log=None

    def _read(self):
        try:
            for line in self.process.stdout:
                try:self.messages.put(json.loads(line))
                except ValueError:self.messages.put({'protocol_error':'Worker emitted non-JSON output'})
        finally:self.messages.put({'protocol_error':'Worker exited; restart the demo. No fallback.'})

    def receive(self):
        try:value=self.messages.get(timeout=self.timeout)
        except queue.Empty:
            self.close(force=True);raise TimeoutError('Worker timeout; stopped owned process. Restart demo; no fallback.')
        if 'protocol_error' in value:raise RuntimeError(value['protocol_error'])
        return value

    def start(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is not None:raise RuntimeError('Worker stopped; restart demo')
                return
            out=ROOT/'reports/persistent_demo'/('session-'+uuid.uuid4().hex);out.mkdir(parents=True,exist_ok=False)
            self.log=(out/'worker-stderr.log').open('x',encoding='utf-8')
            tick=time.perf_counter()
            self.process=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-u',str(ROOT/'persistent_demo_worker.py'),
                '--run-dir',str(self.run_dir)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=0x08000000)
            threading.Thread(target=self._read,daemon=True).start()
            ready=self.receive()
            self.startup=dict(ready,observed_worker_startup_seconds=time.perf_counter()-tick,log_directory=str(out))
            (out/'startup.json').write_text(json.dumps(self.startup,indent=2),encoding='utf-8')
            if not ready.get('ready'):
                self.close(force=True);raise RuntimeError('Worker initialization failed: '+ready.get('error','unknown'))

    def infer(self,image,question,reading):
        with self.lock:
            self.start()
            self.process.stdin.write(json.dumps(dict(action='infer',image=str(image),question=question,reading=bool(reading)))+'\n')
            self.process.stdin.flush();value=self.receive()
            if not value.get('ok'):raise RuntimeError('Persistent inference failed; no fallback: '+value.get('error','unknown'))
            return value

    def close(self,force=False):
        with self.lock:
            if self.process is None:return
            if self.process.poll() is None:
                try:
                    if not force:
                        self.process.stdin.write('{"action":"shutdown"}\n');self.process.stdin.flush()
                        self.process.wait(timeout=20)
                    else:raise subprocess.TimeoutExpired('worker',0)
                except (OSError,subprocess.TimeoutExpired):
                    # Only our owned venv launcher/process tree; never other apps.
                    subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],capture_output=True,creationflags=0x08000000)
                    self.process.wait(timeout=10)
            for stream in (self.process.stdin,self.process.stdout):
                if stream:stream.close()
            if self.log:self.log.close()


class PersistentTransport:
    """Adapter for the unchanged app_phase4.ask subprocess contract."""
    def __init__(self,client,reading,timings):self.client=client;self.reading=reading;self.timings=timings

    def run(self,command,**options):
        if Path(command[1]).resolve()!=ROOT/'phase4_infer.py':raise ValueError('Unexpected worker command')
        path=Path(command[command.index('--image')+1]);question=command[command.index('--question')+1]
        self.timings['request_directory']=str(path.parent)
        tick=time.perf_counter();value=self.client.infer(path,question,self.reading)
        self.timings['worker_roundtrip_seconds']=time.perf_counter()-tick
        self.timings['worker_stages']=value['timings'];self.timings['startup']=self.client.startup
        return subprocess.CompletedProcess(command,0,json.dumps(value['result'],ensure_ascii=False),'')
