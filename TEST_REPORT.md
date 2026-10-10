# Local verification — 10 October 2026

Environment: Windows, Python 3.10.11, torch 2.4.1+cpu, torchvision 0.19.1+cpu, transformers 4.44.0, Gradio 6.13.0. No CUDA was available. Missing speech/OCR packages were installed in the project virtual environment. Pretrained weights were cached locally; no training or dataset download ran.

- Automated tests: **10 passed**. Covered module imports, missing image, empty question, RGB normalization, lexical selection, CPU fallback, absent custom checkpoints, negative gate behavior using a fixture, optional extractor/TTS failures, speech/silence handling and Gradio construction.
- Real BLIP CPU inference: answered `red` for a synthetic red rectangle.
- Real offline Windows speech: produced nonempty valid WAV files.
- Real Whisper tiny.en: transcribed synthesized `What color is the rectangle?` correctly.
- YOLO nano and EasyOCR: loaded and processed the rectangle image without errors; empty detections were expected for this artificial image.
- EasyOCR readable sample: extracted `VISION BRIDGE` with bounding coordinates.
- CLIP: returned a 512-dimensional image embedding.
- Gradio HTTP: returned 200 and registered image upload/webcam and transcription/question handlers.
- Full localhost Gradio client integration: uploaded a PNG, asked a typed question, received `Possible answer: red. I cannot assess its reliability.`, downloaded generated audio, and submitted it for transcription. The round-trip transcription rendered `red` as `read`; speech recognition is imperfect.

Fixed during verification: importing native Transformers dependencies for the first time in a Gradio worker caused a Windows access violation. The app initializes native inference libraries on the main thread before building the UI; model weights still load only when needed. The complete integration test passed after this fix.

Not verified: physical webcam/microphone capture, audible speaker playback, screen-reader and visual contrast audits, CUDA execution, meaningful object-detection accuracy on natural photos, general VQA accuracy or reliability calibration. No trained custom checkpoints exist, so enhanced model accuracy/grounding is not claimed. Automated gate fixtures are only contract tests, not a functioning trained classifier.

Run tests with `.\.venv\Scripts\python.exe -m pytest -q`; real model smoke checks with `.\.venv\Scripts\python.exe smoke_test.py --speech --evidence`; Gradio upload/handler integration with `.\.venv\Scripts\python.exe integration_test.py`.
