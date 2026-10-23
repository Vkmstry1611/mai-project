import json

from PIL import Image

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
