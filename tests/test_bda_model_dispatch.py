"""The two model calls must receive the same image and question."""
import sys
from types import SimpleNamespace

from bda.model_dispatch import run_models


def test_run_models_uses_one_input_for_both(monkeypatch):
    calls = []

    class Client:
        def __init__(self, url, verbose=False):
            self.url = url

        def predict(self, image_path, question, muted, override, angle, api_name):
            from pathlib import Path
            calls.append((self.url, Path(image_path).read_bytes(), question, muted,
                          override, angle, api_name))
            return (self.url, None, None, {}, None, None, {"callback_seconds": 1.0})

    monkeypatch.setitem(sys.modules, "gradio_client", SimpleNamespace(Client=Client))
    result = run_models(b"same image", "sample.jpg", "What is shown?",
                        {"Qwen": "http://qwen", "BLIP": "http://blip"})
    assert [row["model"] for row in result] == ["Qwen", "BLIP"]
    assert all(row["status"] == "completed" for row in result)
    assert [call[1:3] for call in calls] == [(b"same image", "What is shown?")] * 2
    assert all(call[3:] == (True, "VisionBridge Assistant", "0", "/analyze") for call in calls)
