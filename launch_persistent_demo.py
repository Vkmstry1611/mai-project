"""Explicit opt-in persistent demo; existing launchers remain unchanged."""
import argparse
import atexit
import json
from pathlib import Path
import threading
import time

from persistent_demo_client import WorkerClient,PersistentTransport


class Session:
    def __init__(self,run_dir,client=None):
        self.run_dir=Path(run_dir).resolve();self.client=client or WorkerClient(self.run_dir);self.lock=threading.Lock()

    def request(self,image,question,speak,experimental,rotation,run_dir=None):
        start=time.perf_counter()
        from PIL import ImageOps
        from experimental_reading import is_reading_request
        from submission_ui_helpers import input_error,with_evidence_text
        error=input_error(image,question)
        if error:raise ValueError(error)
        if len(question)>1000:raise ValueError('Please keep the question under 1000 characters.')
        import app_phase4 as app
        from modules import text_to_speech
        reading=bool(experimental and is_reading_request(question))
        source=ImageOps.exif_transpose(image).convert('RGB')
        if reading:source=source.rotate(int(rotation),expand=True)
        timings=dict(persistent_worker=True,experimental_reading_active=reading,rotation_ccw_degrees=int(rotation) if reading else 0,tts_requested=bool(speak))
        with self.lock:
            previous=app.subprocess;tts=text_to_speech.synthesize
            def observed_tts(*a,**kw):
                tick=time.perf_counter()
                try:return tts(*a,**kw)
                finally:timings['tts_seconds']=time.perf_counter()-tick
            app.subprocess=PersistentTransport(self.client,reading,timings);text_to_speech.synthesize=observed_tts
            try:result=app.ask(source,question,speak,self.run_dir)
            finally:
                app.subprocess=previous;text_to_speech.synthesize=tts
                timings['callback_seconds']=time.perf_counter()-start
                if timings.get('request_directory'):
                    with (Path(timings['request_directory'])/'persistent-callback-timings.json').open('x',encoding='utf-8') as f:json.dump(timings,f,indent=2)
        return (*with_evidence_text(result),timings)

    def close(self):self.client.close()


def build_app(run_dir):
    import launch_reading_demo as ui
    session=Session(run_dir)
    # This dedicated Python process opts in to transport reuse. No file patches.
    ui.request=session.request
    demo=ui.build_app(run_dir)
    demo.title='VisionBridge — opt-in persistent worker prototype'
    import gradio as gr
    with demo:
        gr.Markdown('**Opt-in persistent-worker prototype.** Models stay initialized between requests. '
                    'Experimental OCR is still off by default. Press Ctrl+C in the launching terminal to stop the UI and release its worker/GPU memory.')
    atexit.register(session.close)
    return demo,session


if __name__=='__main__':
    import sys
    if Path(sys.prefix).name!='.venv':raise RuntimeError('Use .venv for UI; worker stays in .venv-gpu')
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1');args=p.parse_args()
    demo,session=build_app(Path(args.run_dir).resolve())
    print('OPT-IN persistent worker prototype. Ctrl+C shuts down this UI and its worker. First request initializes models.',flush=True)
    try:demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',share=False,show_error=True)
    finally:session.close()
