"""A submitted model result is captured on the VM before cluster processing."""
from io import BytesIO
import json
import sys
from types import SimpleNamespace

from PIL import Image

from bda.live_store import store_result


def test_store_result_saves_event_and_upserts_mongo(tmp_path, monkeypatch):
    saved = []

    class Collection:
        def replace_one(self, selector, document, upsert=False):
            saved.append((selector, document, upsert))

    class Client:
        def __init__(self, uri, serverSelectionTimeoutMS):
            self.uri = uri

        def __getitem__(self, key):
            return {"requests": Collection()}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pymongo", SimpleNamespace(MongoClient=Client))
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
    event = store_result(buffer.getvalue(), "What color?",
                         {"model": "Qwen", "status": "completed", "answer": "blue",
                          "details": {"model": "qwen3-vl:4b"}, "timings": {"callback_seconds": 2}},
                         tmp_path, "mongodb://localhost:27017",
                         run_ids={"Qwen": "qwen-demo-v1"})
    assert event["run_id"] == "qwen-demo-v1"
    assert event["backend"] == "ollama"
    assert event["final_answer"] == "blue"
    assert json.loads(next((tmp_path / "events").glob("*.json")).read_text())["request_id"] == event["request_id"]
    assert saved[0][0] == {"request_id": event["request_id"]}
    assert saved[0][2] is True
