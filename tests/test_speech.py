from unittest.mock import patch
from types import SimpleNamespace
from modules.speech_to_text import transcribe
from config import Config
import pytest

def test_transcript_and_empty_recording():
    model = SimpleNamespace(transcribe=lambda *a, **k: (iter([SimpleNamespace(text=' What is this? ')]), None))
    with patch('modules.speech_to_text.load_whisper', return_value=model):
        assert transcribe('fixture.wav', Config()) == 'What is this?'
    with pytest.raises(ValueError, match='Record'):
        transcribe(None, Config())

def test_silence():
    model = SimpleNamespace(transcribe=lambda *a, **k: (iter([]), None))
    with patch('modules.speech_to_text.load_whisper', return_value=model), pytest.raises(ValueError, match='No speech'):
        transcribe('fixture.wav', Config())
