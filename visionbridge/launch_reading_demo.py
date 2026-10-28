"""Optional OCR readout and stage timing; the default remains frozen Phase 4."""
import argparse
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
_LOCK=threading.Lock()


def request(image,question,speak,experimental,rotation,run_dir):
    callback_start=time.perf_counter()
    import json
    from PIL import ImageOps
    from visionbridge.experimental_reading import is_reading_request,route_request
    from submission_ui_helpers import input_error,with_evidence_text
    error=input_error(image,question)
    if error:raise ValueError(error)
    if len(question)>1000:raise ValueError('Please keep the question under 1000 characters.')
    # Controller module alias only; do not patch Python's global subprocess API.
    import app_phase4 as app
    from modules import text_to_speech
    timings={};reading=bool(experimental and is_reading_request(question))
    source=ImageOps.exif_transpose(image).convert('RGB')
    if reading:source=source.rotate(int(rotation),expand=True)

    class ObservedWorker:
        @staticmethod
        def run(command,**options):
            command=list(command)
            if Path(command[1]).resolve()!=ROOT/'phase4_infer.py':raise ValueError('Unexpected worker')
            command[1]=str(ROOT/'visionbridge/reading_demo_worker.py')
            if reading:command.append('--reading')
            image_path=Path(command[command.index('--image')+1]);timings['request_directory']=str(image_path.parent)
            start=time.perf_counter()
            # Windows pipes otherwise use the worker's legacy code page while
            # the frozen controller explicitly decodes UTF-8. Preserve Unicode
            # literal OCR (and experimental labels) without changing predictions.
            options['env']=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONPATH=str(ROOT))
            try:return subprocess.run(command,**options,creationflags=0x08000000)
            finally:timings['worker_process_seconds']=time.perf_counter()-start

    with _LOCK:
        previous=app.subprocess;previous_tts=text_to_speech.synthesize
        def observed_tts(*a,**kw):
            start=time.perf_counter()
            try:return previous_tts(*a,**kw)
            finally:timings['tts_seconds']=time.perf_counter()-start
        app.subprocess=ObservedWorker();text_to_speech.synthesize=observed_tts
        start=time.perf_counter()
        try:
            # Same exact frozen controller call when off/non-reading. Timings are
            # returned separately; never inject fields into frozen result details.
            call=lambda:app.ask(source,question,speak,run_dir)
            result=route_request(experimental,question,call,call)
        finally:
            app.subprocess=previous;text_to_speech.synthesize=previous_tts
            timings['callback_seconds']=time.perf_counter()-callback_start
            timings['experimental_reading_active']=reading
            timings['rotation_ccw_degrees']=int(rotation) if reading else 0
            timings['tts_requested']=bool(speak)
            folder=timings.get('request_directory')
            if folder:
                stages=Path(folder)/'stage-timings.json'
                if stages.exists():timings['worker_stages']=json.loads(stages.read_text(encoding='utf-8'))
                with (Path(folder)/'callback-timings.json').open('x',encoding='utf-8') as file:json.dump(timings,file,indent=2)
    return (*with_evidence_text(result),timings)


def build_app(run_dir):
    import gradio as gr
    from app_phase4 import transcribe
    with gr.Blocks(title='VisionBridge — experimental reading demo') as demo:
        gr.Markdown('# VisionBridge\nDefault: unchanged frozen Phase 4. Optional reading mode is **experimental** and has no verified VQA accuracy claim. '
                    'It reads literal OCR fragments; missing text is not reconstructed from guesses.')
        image=gr.Image(label='Photo',type='pil',sources=['upload','webcam'])
        question=gr.Textbox(label='Question',lines=2)
        experimental=gr.Checkbox(label='EXPERIMENTAL: literal OCR reading for explicit reading questions',value=False)
        rotation=gr.Dropdown(['0','90','180','270'],value='0',label='Experimental reading only: rotate counterclockwise (180 for upside down)')
        gr.Markdown('Rotation is ignored in frozen mode. For tilted/curved labels, take a straight-on photo. Reading fragments are not a guaranteed complete transcription.')
        with gr.Accordion('Speech input',open=False):
            recording=gr.Audio(label='Record a question',sources=['microphone','upload'],type='filepath')
            transcribe_button=gr.Button('Transcribe and review')
        speak=gr.Checkbox(label='Generate spoken answer',value=False)
        button=gr.Button('Ask',variant='primary')
        answer=gr.Textbox(label='Answer',interactive=False,lines=4)
        audio=gr.Audio(label='Spoken answer — press Play',autoplay=False)
        status=gr.Textbox(label='Status',interactive=False)
        details=gr.JSON(label='Decision details (experimental route is labelled)')
        annotated=gr.Image(label='Evidence in displayed image coordinates',interactive=False)
        evidence_text=gr.Textbox(label='Evidence text alternative',interactive=False,lines=5)
        timing=gr.JSON(label='Request timings in seconds — nested durations overlap')
        transcribe_button.click(transcribe,recording,[question,status])
        def ask(i,q,s,e,r):
            try:return request(i,q,s,e,r,run_dir)
            except Exception as error:raise gr.Error('No substitute answer was generated. '+str(error)) from error
        button.click(ask,[image,question,speak,experimental,rotation],[answer,audio,status,details,annotated,evidence_text,timing],api_name='ask')
    return demo


if __name__=='__main__':
    import sys
    if Path(sys.prefix).name!='.venv':raise RuntimeError('Launch UI with .venv; workers use .venv-gpu')
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1');args=p.parse_args()
    build_app(Path(args.run_dir).resolve()).queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',share=False,show_error=True)
