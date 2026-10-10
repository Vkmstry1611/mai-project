"""Exercise actual image upload and handlers through a local Gradio server."""
from pathlib import Path
from PIL import Image, ImageDraw
from app import build_app
from gradio_client import Client, handle_file

if __name__ == '__main__':
    path = Path('outputs/test-image.png')
    path.parent.mkdir(exist_ok=True)
    image = Image.new('RGB', (400, 300), 'white')
    ImageDraw.Draw(image).rectangle((80, 60, 300, 220), fill='red')
    image.save(path)
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7861, prevent_thread_lock=True, share=False)
    try:
        client = Client('http://127.0.0.1:7861', verbose=False)
        result = client.predict(handle_file(str(path)), 'What color is the rectangle?', False, False, False, True, False, api_name='/run')
        print('Answer:', result[0], flush=True)
        print('Audio:', result[1], flush=True)
        print('Status:', result[2], flush=True)
        assert 'red' in result[0].lower()
        assert result[1] and Path(result[1]).is_file()
        assert result[3]['grounded'] is False
        transcript = client.predict(handle_file(result[1]), api_name='/transcribe')
        print('Transcription:', transcript[0], flush=True)
        assert transcript[0]
        print('Local upload, VQA, TTS and STT integration passed.', flush=True)
    finally:
        app.close()
