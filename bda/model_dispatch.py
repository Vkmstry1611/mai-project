"""Submit one image and question to the existing model-specific Gradio apps."""
from pathlib import Path
from tempfile import TemporaryDirectory


def run_models(image_bytes, filename, question, endpoints):
    """Return one result per requested model; each app captures its own event."""
    from gradio_client import Client

    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"
    results = []
    with TemporaryDirectory(prefix="visionbridge-comparison-") as directory:
        image_path = Path(directory) / ("image" + suffix)
        image_path.write_bytes(image_bytes)
        for model, url in endpoints.items():
            try:
                outputs = Client(url, verbose=False).predict(
                    str(image_path), question, True, "VisionBridge Assistant", "0",
                    api_name="/analyze",
                )
                answer = str(outputs[0] or "").strip()
                details = outputs[3] if isinstance(outputs[3], dict) else {}
                results.append({
                    "model": model,
                    "answer": answer,
                    "status": "error" if details.get("application_error") or not answer else "completed",
                    "details": details,
                    "timings": outputs[6] if isinstance(outputs[6], dict) else {},
                })
            except Exception as error:
                results.append({"model": model, "answer": "", "status": "error",
                                "error": str(error), "details": {}, "timings": {}})
    return results
