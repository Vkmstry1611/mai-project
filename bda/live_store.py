"""Store a Streamlit model result on the VM as soon as it completes."""
from io import BytesIO
import json

from PIL import Image

from bda.capture import capture_request


def store_result(image_bytes, question, result, spool, mongo_uri, database="visionbridge_bda",
                 run_ids=None):
    """Capture locally first, then upsert MongoDB; the VM watcher imports HDFS."""
    from pymongo import MongoClient

    model = result["model"]
    backend = "ollama" if model == "Qwen" else "blip"
    details = dict(result.get("details") or {})
    details.setdefault("backend", backend)
    for field in ("processing_started_at_utc", "processing_completed_at_utc"):
        if result.get(field):
            details[field] = result[field]
    if result["status"] != "completed":
        details["application_error"] = details.get("application_error") or {
            "stage": "model_dispatch", "message": result.get("error", "No answer returned")
        }
    with Image.open(BytesIO(image_bytes)) as image:
        path = capture_request(image, question, result.get("answer", ""), details,
                               result.get("timings") or {}, root=spool,
                               run_id=(run_ids or {}).get(model))
    event = json.loads(path.read_text(encoding="utf-8"))
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        collection = client[database]["requests"]
        collection.replace_one({"request_id": event["request_id"]}, event, upsert=True)
    finally:
        client.close()
    return event
