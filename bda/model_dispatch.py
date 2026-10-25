"""Submit one image and question to the existing model-specific Gradio apps."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def transcribe_audio(audio_bytes, filename, service_url):
    """Upload a WAV recording to the validated Windows speech endpoint."""
    from gradio_client import Client, handle_file

    if not filename.lower().endswith(".wav"):
        raise ValueError("Upload a WAV recording.")
    with TemporaryDirectory(prefix="visionbridge-speech-input-") as directory:
        audio_path = Path(directory) / "question.wav"
        audio_path.write_bytes(audio_bytes)
        transcript = Client(service_url, verbose=False).predict(
            handle_file(audio_path), api_name="/transcribe_for_dashboard")
    text = str(transcript or "").strip()
    if not text:
        raise ValueError("No speech was recognized. Try again or type your question.")
    return text


def run_models(image_bytes, filename, question, endpoints, *, speak=False):
    """Return one result per requested model; each app captures its own event."""
    from gradio_client import Client, handle_file

    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"
    results = []
    with TemporaryDirectory(prefix="visionbridge-comparison-") as directory:
        image_path = Path(directory) / ("image" + suffix)
        image_path.write_bytes(image_bytes)
        for model, url in endpoints.items():
            started_at = now_utc()
            try:
                outputs = Client(url, verbose=False).predict(
                    handle_file(image_path), question, not speak, "VisionBridge Assistant", "0",
                    api_name="/analyze",
                )
                answer = str(outputs[0] or "").strip()
                details = outputs[3] if isinstance(outputs[3], dict) else {}
                audio = None
                speech_error = None
                if speak and outputs[1]:
                    try:
                        audio_file = outputs[1].get("path") if isinstance(outputs[1], dict) else outputs[1]
                        audio = Path(audio_file).read_bytes()
                    except (OSError, TypeError, ValueError) as error:
                        speech_error = str(error)
                results.append({
                    "model": model,
                    "answer": answer,
                    "status": "error" if details.get("application_error") or not answer else "completed",
                    "details": details,
                    "timings": outputs[6] if isinstance(outputs[6], dict) else {},
                    "audio": audio,
                    "speech_error": speech_error,
                    "processing_started_at_utc": started_at,
                    "processing_completed_at_utc": now_utc(),
                })
            except Exception as error:
                results.append({"model": model, "answer": "", "status": "error",
                                "error": str(error), "details": {}, "timings": {},
                                "processing_started_at_utc": started_at,
                                "processing_completed_at_utc": now_utc()})
    return results
