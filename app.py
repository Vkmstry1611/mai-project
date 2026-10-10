"""Run with python app.py. Models load only when requested."""
import logging
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
from pipeline import Pipeline

def build_app(pipeline=None):
    # Initialize native inference libraries on the main thread on Windows.
    # Weights still load lazily on the first request.
    import torch
    from transformers import BlipProcessor, BlipForQuestionAnswering, CLIPModel, CLIPProcessor
    try:
        import faster_whisper
    except ImportError:
        pass
    import gradio as gr
    pipeline = pipeline or Pipeline()
    with gr.Blocks(title='VisionBridge — AI Visual Assistant') as demo:
        gr.Markdown('# VisionBridge — AI Visual Assistant\nUpload an image or take a photo. Type a question, or record and transcribe it. Then select **Ask about image**.')
        from modules.vqa import model_source
        _, active_label = model_source(pipeline.config)
        gr.Markdown(f'**Default VQA:** {active_label}. **Safeguards:** unsupported weather/identity questions are declined; YOLO checks recognized objects mentioned in questions. **Available:** Whisper and offline speech. **Pending:** trained fusion, learned QGES and a calibrated answerability gate. Answers may still be incorrect.')
        with gr.Row():
            with gr.Column():
                image = gr.Image(label='Image — upload or camera', sources=['upload', 'webcam'], type='pil')
                recording = gr.Audio(label='Record your question', sources=['microphone', 'upload'], type='filepath')
                transcribe = gr.Button('Transcribe recording', size='lg')
                transcript = gr.Textbox(label='Recognized speech', interactive=False)
                question = gr.Textbox(label='Question — type or correct transcription', lines=2)
                vqa_model = gr.Dropdown(label='VQA model', choices=[('Automatic — use local trained checkpoint when available', 'auto'),
                    ('Trained BLIP — local blip_vizwiz checkpoint', 'trained'),
                    ('Original pretrained BLIP baseline', 'pretrained')], value=pipeline.config.vqa_mode)
                reliability_checks = gr.Checkbox(label='Check question support before answering (recommended)', value=pipeline.config.enable_reliability_checks)
                with gr.Accordion('Optional processing', open=False):
                    objects = gr.Checkbox(label='Detect objects with YOLO', value=False)
                    text = gr.Checkbox(label='Read visible text with EasyOCR', value=False)
                    clip = gr.Checkbox(label='Extract CLIP image and region features', value=False)
                    enhanced = gr.Checkbox(label='Use configured custom adapters (requires trusted checkpoints)', value=False)
                speak = gr.Checkbox(label='Generate spoken answer', value=True)
                ask = gr.Button('Ask about image', variant='primary', size='lg')
            with gr.Column():
                answer = gr.Textbox(label='Answer', interactive=False, lines=4)
                audio = gr.Audio(label='Spoken answer — play or replay', autoplay=True)
                status = gr.Textbox(label='Processing status and errors', interactive=False)
                with gr.Accordion('Visual evidence and component status', open=False):
                    debug = gr.JSON(label='Evidence, warnings and active components')
                    annotated = gr.Image(label='Evidence bounding boxes', interactive=False)
        transcribe.click(pipeline.transcribe, recording, [transcript, question, status])
        ask.click(pipeline.run, [image, question, objects, text, clip, speak, enhanced, vqa_model, reliability_checks], [answer, audio, status, debug, annotated])
    return demo

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build_app().queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', share=False,
        theme=__import__('gradio').themes.Base(primary_hue='blue', neutral_hue='slate'),
        css='body, input, textarea {font-size:18px !important} button {min-height:48px !important} .gradio-container {max-width:1200px !important} button:focus-visible, input:focus-visible, textarea:focus-visible {outline:3px solid #2563eb !important; outline-offset:3px}')
