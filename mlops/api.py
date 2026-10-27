"""Minimal CPU VQA API for the DVC-selected VisionBridge checkpoint."""

import io
import base64
import binascii
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from mlops.model import load_model, model_sha256, predict


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "mlops/registry/champion"))
UI_PATH = Path(__file__).with_name("ui.html")
app = FastAPI(title="VisionBridge VQA API", version="1.0")
_lock = threading.Lock()
_rate_lock = threading.Lock()
_recent_predictions = deque()
MAX_PREDICTIONS_PER_MINUTE = 30
_state = {"processor": None, "model": None, "metadata": None,
          "requests": 0, "errors": 0, "latency_total_seconds": 0.0}


class PredictionRequest(BaseModel):
    image_base64: str
    question: str


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(UI_PATH, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.on_event("startup")
def startup() -> None:
    metadata = json.loads((MODEL_DIR / "registry.json").read_text(encoding="utf-8"))
    if model_sha256(MODEL_DIR) != metadata["model_sha256"]:
        raise RuntimeError("Registry model hash mismatch")
    processor, model = load_model(MODEL_DIR)
    _state.update(processor=processor, model=model, metadata=metadata)


@app.get("/health")
async def health() -> dict:
    metadata = _state["metadata"]
    if metadata is None:
        raise HTTPException(503, "Model is not loaded")
    return {"status": "ok", "candidate": metadata["candidate"],
            "model_sha256": metadata["model_sha256"]}


@app.get("/metrics")
async def metrics() -> dict:
    with _lock:
        count = _state["requests"]
        return {
            "requests": count,
            "errors": _state["errors"],
            "mean_latency_seconds": round(_state["latency_total_seconds"] / count, 3) if count else 0,
        }


@app.post("/predict")
async def prediction(request: PredictionRequest) -> dict:
    if _state["model"] is None:
        raise HTTPException(503, "Model is not loaded")
    question = request.question.strip()
    if not question or len(question) > 500:
        raise HTTPException(400, "Question must contain 1 to 500 characters")
    if len(request.image_base64) > (MAX_UPLOAD_BYTES * 4 // 3 + 8):
        raise HTTPException(413, "Image exceeds 5 MiB")
    try:
        payload = base64.b64decode(request.image_base64, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(400, "Invalid base64 image") from None
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds 5 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            rgb = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Invalid image") from None
    now = time.monotonic()
    with _rate_lock:
        while _recent_predictions and now - _recent_predictions[0] >= 60:
            _recent_predictions.popleft()
        if len(_recent_predictions) >= MAX_PREDICTIONS_PER_MINUTE:
            raise HTTPException(429, "Demo is busy; retry in a minute")
        _recent_predictions.append(now)
    started = time.perf_counter()
    try:
        with _lock:
            answer = predict(_state["processor"], _state["model"], rgb, question)
            elapsed = time.perf_counter() - started
            _state["requests"] += 1
            _state["latency_total_seconds"] += elapsed
    except Exception:
        with _lock:
            _state["errors"] += 1
        print(json.dumps({"event": "prediction_error", "model_sha256": _state["metadata"]["model_sha256"]}), flush=True)
        raise
    print(json.dumps({"event": "prediction", "latency_seconds": round(elapsed, 3),
                      "model_sha256": _state["metadata"]["model_sha256"]}), flush=True)
    return {"answer": answer, "model_sha256": _state["metadata"]["model_sha256"],
            "latency_seconds": round(elapsed, 3)}
