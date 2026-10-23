"""Existing Gradio environment; frozen-policy inference in GPU subprocesses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT=Path(__file__).resolve().parent


def ask(image,question,speak,run_dir):
    from PIL import ImageOps
    root=Path(run_dir).resolve()
    if (ROOT/'reports/phase4_policy') not in root.parents or not (root/'POLICY_FREEZE.json').is_file():raise ValueError('Select a frozen Phase 4 run')
    if image is None or not isinstance(question,str) or not question.strip():raise ValueError('Upload a photo and type a question')
    folder=root/'application-requests'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    normalized=ImageOps.exif_transpose(image).convert('RGB');path=folder/'input.png';normalized.save(path)
    process=subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),str(ROOT/'phase4_infer.py'),'--run-dir',str(root),
        '--image',str(path),'--question',question.strip()],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=240)
    for name,value in [('stderr.log',process.stderr),('stdout.log',process.stdout)]:
        with (folder/name).open('x',encoding='utf-8') as f:f.write(value)
    if process.returncode:raise RuntimeError('Inference failed; no fallback. Logs: '+str(folder))
    decoder=json.JSONDecoder();result=None
    for i,char in enumerate(process.stdout):
        if char=='{':
            try:
                value,_=decoder.raw_decode(process.stdout,i)
                if isinstance(value,dict) and 'display_answer' in value and 'details' in value:result=value;break
            except json.JSONDecodeError:pass
    if result is None:raise RuntimeError('No valid inference response; logs preserved')
    audio=None;status=result['status']
    if speak:
        try:
            from modules import text_to_speech
            from config import Config
            from dataclasses import replace
            audio=text_to_speech.synthesize(result['display_answer'],replace(Config(),output_dir=folder/'speech'))
        except Exception as error:status+='; spoken answer unavailable: '+str(error)
    from utils.image_utils import annotate
    return result['display_answer'],audio,status,result['details'],annotate(normalized,result['details'].get('evidence',[]))


def transcribe(recording):
    from modules import speech_to_text
    from config import Config
    try:
        text=speech_to_text.transcribe(recording,Config());return text,'Transcribed; correct the question if needed.'
    except Exception as error:return '',str(error)


def build_app(run_dir):
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
    import gradio as gr
    with gr.Blocks(title='VisionBridge — frozen Phase 4') as demo:
        gr.Markdown('# VisionBridge — Visual Assistant\nBLIP v2, question-guided OCR and a frozen answer-or-abstain policy. Answers can still be incorrect. Learned scores do not guarantee visual grounding.')
        with gr.Row():
            with gr.Column():
                image=gr.Image(label='Photo — upload or camera',sources=['upload','webcam'],type='pil')
                recording=gr.Audio(label='Record a question',sources=['microphone','upload'],type='filepath')
                transcribe_button=gr.Button('Transcribe recording')
                question=gr.Textbox(label='Question — type or correct transcription',lines=2)
                speak=gr.Checkbox(label='Generate spoken answer',value=False)
                button=gr.Button('Ask about image',variant='primary',size='lg')
            with gr.Column():
                answer=gr.Textbox(label='Answer',interactive=False,lines=3)
                audio=gr.Audio(label='Spoken answer',autoplay=True)
                status=gr.Textbox(label='Status',interactive=False)
                annotated=gr.Image(label='Question-relevant visual evidence',interactive=False)
                with gr.Accordion('Answer decision and evidence',open=False):details=gr.JSON()
        transcribe_button.click(transcribe,recording,[question,status])
        button.click(lambda i,q,s:ask(i,q,s,run_dir),[image,question,speak],[answer,audio,status,details,annotated])
    return demo


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    build_app(a.run_dir).queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',share=False)
