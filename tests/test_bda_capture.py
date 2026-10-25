import json
import pytest

from PIL import Image

import bda.capture as capture_module
from bda.capture import capture_request
from bda.prepare_labels import prepare


def test_capture_creates_one_event_and_content_addressed_image(tmp_path):
    photo = Image.new("RGB", (32, 20), "blue")
    details = {"model": "qwen3-vl:4b", "backend": "ollama", "final_answer": "blue",
               "answer_source": "local_vision_model", "evidence": []}
    first = capture_request(photo, "What color?", "blue", details,
                            {"callback_seconds": 1.25}, root=tmp_path)
    second = capture_request(photo, " What   color? ", "blue", details,
                             {"callback_seconds": 1.0}, root=tmp_path)
    a = json.loads(first.read_text(encoding="utf-8"))
    b = json.loads(second.read_text(encoding="utf-8"))
    assert a["request_id"] != b["request_id"]
    assert a["pair_id"] == b["pair_id"]
    assert a["image_id"] == b["image_id"]
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 1
    assert a["final_answer"] == "blue"
    assert a["processing_seconds"] == 1.25
    assert a["run_id"] == "ollama:qwen3-vl:4b"


def test_prepared_labels_match_captured_pair(tmp_path):
    photo = Image.new("RGB", (32, 20), "green")
    photo_path = tmp_path / "sample.png"
    photo.save(photo_path)
    event_path = capture_request(photo, "What is shown?", "green", {"backend": "blip"},
                                 {"callback_seconds": 2}, root=tmp_path / "spool")
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"image_path": str(photo_path),
                                  "question": "What is shown?",
                                  "answers": [{"answer": "green"}] * 10}) + "\n", encoding="utf-8")
    output = tmp_path / "labels.jsonl"
    assert prepare(source, output) == 1
    assert json.loads(output.read_text())["pair_id"] == json.loads(event_path.read_text())["pair_id"]


def test_capture_explicit_run_id(tmp_path):
    path = capture_request(Image.new("RGB", (8, 8), "red"), "What color?", "red",
                           {"backend": "ollama"}, {}, root=tmp_path, run_id="qwen-demo-v1")
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "qwen-demo-v1"


def test_failed_event_write_never_publishes_incomplete_json(tmp_path, monkeypatch):
    def interrupted_dump(value, output, **kwargs):
        output.write('{"request_id":')
        raise OSError("interrupted write")

    monkeypatch.setattr(capture_module.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="interrupted write"):
        capture_request(Image.new("RGB", (8, 8), "red"), "What color?", "red",
                        {"backend": "ollama"}, {}, root=tmp_path,
                        request_id="interrupted")
    assert list((tmp_path / "events").iterdir()) == []
