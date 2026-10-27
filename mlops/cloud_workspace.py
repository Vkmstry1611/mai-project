"""Serve the mai-project VisionBridge workspace with models on the EC2 host."""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

from PIL import ImageOps

from launch_workspace import build_app, launch_options
from modules.local_vision import LocalVisionSession
from workspace_intent import Route
from workspace_models import BLIP, QWEN, FrontendModels


class CloudBlipSession:
    """The DVC-selected BLIP checkpoint served by the existing local API."""

    backend_label = "BLIP v2 on AWS"

    def classify(self, question):
        return Route("vqa", "The DVC-selected BLIP model answers the image question.")

    def request(self, image, question, speak=False, experimental=False, rotation="0"):
        if image is None or not str(question or "").strip():
            raise ValueError("Upload an image and enter a question.")
        if experimental:
            raise ValueError("BLIP cloud mode does not include the private OCR pipeline. Select Qwen for reading questions.")
        started = time.perf_counter()
        photo = ImageOps.exif_transpose(image).convert("RGB")
        photo.thumbnail((1600, 1600))
        stream = BytesIO()
        photo.save(stream, format="JPEG", quality=90)
        payload = json.dumps({"question": question.strip(),
                              "image_base64": base64.b64encode(stream.getvalue()).decode("ascii")}).encode("utf-8")
        request = Request("http://127.0.0.1:8000/predict", payload,
                          {"Content-Type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
            result = json.load(response)
        answer = result["answer"]
        details = {"final_answer": answer, "answer_source": "dvc_blip_v2",
                   "model_sha256": result["model_sha256"], "backend": "AWS EC2 CPU",
                   "decision": "generated_visual_answer", "selected_evidence": [],
                   "supporting_evidence_ids": [], "evidence": [],
                   "limitation": "BLIP VQA only; no private Phase 4 OCR or frozen gating."}
        timings = {"callback_seconds": time.perf_counter() - started,
                   "model_inference_performed": True,
                   "model_latency_seconds": result["latency_seconds"],
                   "experimental_reading_active": False}
        return answer, None, "Complete. BLIP v2 on AWS; verify important details.", details, None, "", timings

    def close(self):
        pass


def main():
    os.environ.setdefault("VISIONBRIDGE_CLOUD", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/visionbridge-gradio")
    os.environ.setdefault("VISIONBRIDGE_VISION_MODEL", "qwen3-vl:2b-instruct-q4_K_M")
    qwen = LocalVisionSession()
    models = FrontendModels(lambda name: CloudBlipSession() if name == BLIP else LocalVisionSession(),
                            default=QWEN, initial=qwen)
    demo, _ = build_app(Path("/app"), session=qwen, model_sessions=models, cloud_mode=True)
    try:
        demo.queue(default_concurrency_limit=1).launch(server_port=7865,
            **launch_options("127.0.0.1"))
    finally:
        models.close()


if __name__ == "__main__":
    main()
