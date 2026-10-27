"""One local speech request, isolated from Gradio's native libraries."""
import json
import sys
from contextlib import redirect_stdout


def main():
    from config import Config
    from modules.speech_to_text import transcribe
    with redirect_stdout(sys.stderr):
        text=transcribe(sys.argv[1],Config())
    print(json.dumps({'text':text},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
