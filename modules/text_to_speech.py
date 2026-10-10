"""Offline Windows SAPI voices via pyttsx3; each call owns its COM engine."""
from pathlib import Path
from threading import Lock
from uuid import uuid4
import sys
import wave
import time

_lock = Lock()

def synthesize(text, config):
    if not text.strip():
        raise ValueError('No answer to speak.')
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        for old in config.output_dir.glob('speech-*.wav'):
            if time.time() - old.stat().st_mtime > 86400:
                old.unlink(missing_ok=True)
        path = config.output_dir / f'speech-{uuid4().hex}.wav'
        engine = None
        if sys.platform == 'win32':
            import pythoncom
            pythoncom.CoInitialize()
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.save_to_file(text, str(path))
            engine.runAndWait()
            with wave.open(str(path), 'rb') as audio:
                if audio.getnframes() == 0:
                    raise RuntimeError('The installed voice produced empty audio.')
            return str(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            if engine:
                engine.stop()
            if sys.platform == 'win32':
                pythoncom.CoUninitialize()
