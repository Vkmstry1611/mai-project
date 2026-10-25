"""Persist one immutable request event and its image for later HDFS ingestion."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from PIL import ImageOps


def _jsonable(value):
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _sync_directory(directory):
    if os.name == "posix":
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def image_and_pair_id(image, question):
    photo = ImageOps.exif_transpose(image).convert("RGB")
    stream = BytesIO()
    photo.save(stream, format="JPEG", quality=95)
    image_bytes = stream.getvalue()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    normalized_question = " ".join(str(question).strip().lower().split())
    pair_id = hashlib.sha256((image_sha + "\n" + normalized_question).encode()).hexdigest()
    return image_bytes, image_sha, pair_id


def capture_request(image, question, answer, details, timings, *, root=None, request_id=None,
                    run_id=None):
    """Write an image and matching JSON; return the event path.

    The request path is local spool storage. VM ingestion copies it to HDFS and
    MongoDB. This function never sends a request over the network.
    """
    if image is None or not str(question or '').strip():
        raise ValueError("An image and question are required for BDA capture")
    root = Path(root or os.getenv("VISIONBRIDGE_BDA_SPOOL", "reports/bda/spool"))
    images, events = root / "images", root / "events"
    images.mkdir(parents=True, exist_ok=True)
    events.mkdir(parents=True, exist_ok=True)
    request_id = request_id or uuid4().hex
    image_bytes, image_sha, pair_id = image_and_pair_id(image, question)
    model = str(details.get("model") or timings.get("local_vision_model") or
                os.getenv("VISIONBRIDGE_BDA_MODEL", "blip-vizwiz-v2"))
    backend = str(details.get("backend") or ("ollama" if details.get("answer_source") == "local_vision_model" else "blip"))
    run_id = run_id or os.getenv("VISIONBRIDGE_BDA_RUN_ID") or backend + ":" + model
    evidence = details.get("evidence") or details.get("selected_evidence") or []
    route = (details.get("assistant_trace") or {}).get("route") or (details.get("workspace_routing") or {}).get("task") or "vqa"
    event = {
        "schema_version": 1, "request_id": request_id, "pair_id": pair_id,
        "image_id": image_sha, "image_file": f"images/{image_sha}.jpg",
        "question": str(question).strip(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "backend": backend, "model": model, "route": route,
        "prediction": details.get("final_answer", answer), "final_answer": answer,
        "status": "error" if details.get("application_error") else "completed",
        "abstained": details.get("answer_source") == "abstention" or
                     str(answer).strip().lower() == "unanswerable",
        "ocr_evidence": _jsonable(evidence), "processing_details": _jsonable(details),
        "timings": _jsonable(timings), "processing_seconds": timings.get("callback_seconds"),
    }
    for field in ("processing_started_at_utc", "processing_completed_at_utc"):
        if details.get(field):
            event[field] = str(details[field])
    image_path = images / f"{image_sha}.jpg"
    if not image_path.is_file() or image_path.read_bytes() != image_bytes:
        handle, name = tempfile.mkstemp(suffix=".tmp", prefix=f".{image_sha}-", dir=images)
        temporary_image = Path(name)
        try:
            with os.fdopen(handle, "wb") as output:
                output.write(image_bytes)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_image, image_path)
            _sync_directory(images)
        finally:
            temporary_image.unlink(missing_ok=True)
    event_path = events / f"{request_id}.json"
    # Publish a complete, durable record in one filesystem operation. Writing
    # directly to the final .json path can leave an empty file after a VM reset.
    handle, name = tempfile.mkstemp(suffix=".tmp", prefix=f".{request_id}-", dir=events)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(event, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        if os.name == "nt":
            # Windows may deny hard links to files created by mkstemp.
            if event_path.exists():
                raise FileExistsError(event_path)
            os.replace(temporary, event_path)
        else:
            os.link(temporary, event_path)  # fails if an immutable ID exists
        _sync_directory(events)
    finally:
        temporary.unlink(missing_ok=True)
    return event_path
