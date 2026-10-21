"""Optional presentation/accessibility shell around the frozen controller."""
import argparse
from pathlib import Path
import sys

from submission_ui_helpers import input_error, with_evidence_text


def build_app(run_dir):
    from launch_phase4 import install_worker
    app=install_worker()
    import gradio as gr

    def ask(image,question,speak):
        error=input_error(image,question)
        if error:
            raise gr.Error(error)
        try:
            return with_evidence_text(app.ask(image,question,speak,run_dir))
        except Exception as error:
            # UI presentation only: do not generate a substitute answer.
            raise gr.Error('No answer was returned. '+str(error)) from error

    with gr.Blocks(title='VisionBridge — submission demonstration') as demo:
        gr.Markdown('# VisionBridge — Visual Assistant\nUpload a photo, type a question and press Ask. '
                    'Requests may take about two minutes on this laptop. Answers can still be incorrect. '
                    'OCR and learned scores do not guarantee visual grounding.')
        with gr.Row():
            with gr.Column():
                image=gr.Image(label='1. Photo — upload or camera',sources=['upload','webcam'],type='pil')
                question=gr.Textbox(label='2. Question — type or correct transcription',lines=2)
                with gr.Accordion('Optional speech input',open=False):
                    recording=gr.Audio(label='Record or upload a spoken question',sources=['microphone','upload'],type='filepath')
                    transcribe=gr.Button('Transcribe recording')
                    gr.Markdown('Review the transcript before asking. Whisper uses CPU; the first use may need its cached model or a download.')
                speak=gr.Checkbox(label='Generate an optional spoken answer',value=False)
                button=gr.Button('3. Ask about image',variant='primary',size='lg')
            with gr.Column():
                answer=gr.Textbox(label='Answer or uncertainty message',interactive=False,lines=3)
                status=gr.Textbox(label='Processing status',interactive=False)
                audio=gr.Audio(label='Spoken answer — press Play to listen',autoplay=False)
                text=gr.Textbox(label='Selected evidence — text alternative',interactive=False,lines=6)
                annotated=gr.Image(label='Evidence boxes — also described in the text panel',interactive=False)
                with gr.Accordion('Technical decision details',open=False):
                    details=gr.JSON(label='Original candidate, evidence and frozen decision')
        transcribe.click(app.transcribe,recording,[question,status])
        button.click(ask,[image,question,speak],[answer,audio,status,details,annotated,text])
    return demo


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default='reports/phase4_policy/run-local-20261013-v1')
    args=p.parse_args()
    if Path(sys.prefix).name!='.venv':
        raise RuntimeError('Use .venv for the UI; the frozen worker uses .venv-gpu')
    root=Path(args.run_dir).resolve()
    from launch_phase4 import ROOT
    if ROOT/'reports/phase4_policy' not in root.parents or not (root/'APPLICATION_FREEZE.json').is_file():
        raise ValueError('Select the completed frozen application run')
    build_app(root).queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',share=False,show_error=True,
        theme=__import__('gradio').themes.Base(primary_hue='blue',neutral_hue='slate'),
        css='body,input,textarea {font-size:18px !important} button {min-height:48px !important} button:focus-visible,input:focus-visible,textarea:focus-visible {outline:3px solid #2563eb !important; outline-offset:3px}')
