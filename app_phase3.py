"""Separate UI in original environment; all model inference uses .venv-gpu."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT=Path(__file__).resolve().parent


def ask(image,question,run_dir):
    from PIL import ImageOps
    root=Path(run_dir).resolve()
    if (ROOT/'reports/phase3_answerability') not in root.parents:
        raise ValueError('Use the completed versioned Phase 3 experiment')
    if not (root/'results/PHASE3_FINAL_REPORT.md').is_file():
        raise RuntimeError('Phase 3 development evaluation is not complete yet')
    if image is None or not isinstance(question,str) or not question.strip():
        raise ValueError('Upload an image and enter a question')
    folder=root/'application-requests'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    normalized=ImageOps.exif_transpose(image).convert('RGB');path=folder/'input.png';normalized.save(path)
    # A subprocess keeps the working UI environment and CUDA evaluator separate.
    proc=subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),str(ROOT/'phase3_infer.py'),
              '--run-dir',str(root),'--image',str(path),'--question',question.strip()],cwd=ROOT,
              capture_output=True,text=True,encoding='utf-8',timeout=240)
    with (folder/'stderr.log').open('x',encoding='utf-8') as f:f.write(proc.stderr)
    with (folder/'stdout.log').open('x',encoding='utf-8') as f:f.write(proc.stdout)
    if proc.returncode:raise RuntimeError('Phase 3 inference failed; logs preserved in '+str(folder))
    # Libraries may write startup notices before the final JSON object.
    decoder=json.JSONDecoder(); result=None
    for index,char in enumerate(proc.stdout):
        if char=='{':
            try:
                parsed,end=decoder.raw_decode(proc.stdout,index)
                if isinstance(parsed,dict) and 'details' in parsed and 'display_answer' in parsed:
                    result=parsed;break
            except json.JSONDecodeError:pass
    if result is None:raise RuntimeError('Inference output was not valid result JSON; logs preserved')
    from utils.image_utils import annotate
    return result['display_answer'],result['status'],result['details'],annotate(normalized,result['details'].get('evidence',[]))


def build_app(run_dir):
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
    import gradio as gr
    with gr.Blocks(title='VisionBridge Phase 3') as demo:
        gr.Markdown('# VisionBridge — answer or abstain\nAsk a question about your photo. The system may decline when its estimated reliability is insufficient. Answers can still be incorrect.')
        image=gr.Image(label='Photo',sources=['upload','webcam'],type='pil')
        question=gr.Textbox(label='Question',lines=2)
        button=gr.Button('Ask about image',variant='primary')
        answer=gr.Textbox(label='Answer',interactive=False,lines=3)
        status=gr.Textbox(label='Status',interactive=False)
        annotated=gr.Image(label='Detected visual evidence',interactive=False)
        with gr.Accordion('Answer decision and evidence',open=False):details=gr.JSON()
        button.click(lambda image,question:ask(image,question,run_dir),[image,question],[answer,status,details,annotated])
    return demo


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();build_app(args.run_dir).queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',share=False)
