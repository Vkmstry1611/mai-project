"""The two model calls must receive the same image and question."""
import sys
from types import SimpleNamespace

from bda.model_dispatch import run_models, transcribe_audio


def test_run_models_uses_one_input_for_both(monkeypatch):
    calls = []

    class Client:
        def __init__(self, url, verbose=False):
            self.url = url

        def predict(self, image, question, muted, override, angle, api_name):
            from pathlib import Path
            calls.append((self.url, Path(image["path"]).read_bytes(), question, muted,
                          override, angle, api_name))
            return (self.url, None, None, {}, None, None, {"callback_seconds": 1.0})

    def handle_file(path):
        return {"path": str(path), "meta": {"_type": "gradio.FileData"}}

    monkeypatch.setitem(sys.modules, "gradio_client",
                        SimpleNamespace(Client=Client, handle_file=handle_file))
    result = run_models(b"same image", "sample.jpg", "What is shown?",
                        {"Qwen": "http://qwen", "BLIP": "http://blip"})
    assert [row["model"] for row in result] == ["Qwen", "BLIP"]
    assert all(row["status"] == "completed" for row in result)
    assert [call[1:3] for call in calls] == [(b"same image", "What is shown?")] * 2
    assert all(call[3:] == (True, "VisionBridge Assistant", "0", "/analyze") for call in calls)


def test_speech_request_returns_playable_audio(tmp_path, monkeypatch):
    recording = tmp_path / "answer.wav"
    recording.write_bytes(b"wave bytes")
    muted = []

    class Client:
        def __init__(self, url, verbose=False):
            pass

        def predict(self, image, question, mute, override, angle, api_name):
            muted.append(mute)
            return ("blue", str(recording), None, {}, None, None, {})

    monkeypatch.setitem(sys.modules, "gradio_client", SimpleNamespace(
        Client=Client, handle_file=lambda path: {"path": str(path)}))
    result = run_models(b"image", "sample.jpg", "What color?",
                        {"Qwen": "http://qwen"}, speak=True)[0]
    assert muted == [False]
    assert result["answer"] == "blue"
    assert result["audio"] == b"wave bytes"


def test_transcription_uploads_audio_and_returns_text(monkeypatch):
    seen = []

    class Client:
        def __init__(self, url, verbose=False):
            pass

        def predict(self, file, api_name):
            from pathlib import Path
            seen.append((Path(file["path"]).read_bytes(), api_name))
            return "What is shown?"

    monkeypatch.setitem(sys.modules, "gradio_client", SimpleNamespace(
        Client=Client, handle_file=lambda path: {"path": str(path)}))
    assert transcribe_audio(b"spoken question", "recording.wav", "http://qwen") == "What is shown?"
    assert seen == [(b"spoken question", "/transcribe_for_dashboard")]
