# VisionBridge — AI Visual Assistant

Local, inference-only Gradio application for visual questions, microphone transcription and spoken answers. No model training or VizWiz dataset download is performed.

The supplied `VisionBridge_VizWiz_Baseline_VQA (2).ipynb` was inspected. Its BlipProcessor/BlipForQuestionAnswering inference, RGB conversion, CUDA/CPU choice and decoded generation were reused. The notebook is not modified. Dataset preparation and evaluation are not copied into the app.

## Windows setup

Python 3.10 or 3.11 is recommended. This project was built with Python 3.10.11 and CPU PyTorch 2.4.1 / torchvision 0.19.1.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. Camera and microphone permission is granted in the browser. Upload an image, type a question and click Ask. For speech, record, click Transcribe, correct the recognized question and then click Ask. The audio player supports replay; browser autoplay policy may require pressing Play.

Install optional evidence dependencies with `pip install -r requirements-evidence.txt`. Enable the desired checkboxes. Evidence is diagnostic: BLIP does not consume OCR/YOLO evidence, and displaying it does not make an answer grounded. Lexical QGES is a training-free prototype; CLIP image-question similarities are provided separately in the debug panel.

## Models and limitations

### Conservative reliability safeguards

The recommended **Check question support before answering** setting is on by default. Weather questions return “I can't determine the weather from this image.” This deliberately declines all weather questions, including outdoor scenes; the app does not measure weather or fetch a forecast. Personal/animal names and some future/unseen-information questions also abstain.

When a question mentions a recognizable YOLO/COCO category (for example bottle, cat, dog, bus, phone or sofa), the app runs YOLO automatically even if the optional detection checkbox is off. If that category is not observed above the configured detection threshold, it says “I can't see that item in this image” with a note that it may be obscured or too small. Non-detection does not prove absence. If detection is unavailable, the app says it cannot verify the item instead of inventing its color, count or location. Install `requirements-evidence.txt` for these object checks. Categories outside the supported detector vocabulary may not be checked.

Detected presence does not validate the answer. Normal answers contain only the answer; limitations remain in the UI notice and debug panel; the debug panel separates these heuristic checks from the still-unavailable trained gate, and supplies no answer confidence. The checkbox can be disabled to compare raw VQA behavior. These safeguards need broader validation and may abstain on objects the detector misses.

### Your trained BLIP checkpoint

The provided `models/visionbridge_model_only.zip` is integrated as `models/blip_vizwiz/`. The app's **VQA model** selector defaults to Automatic: it uses this local checkpoint when all required files are available. Choose Original pretrained BLIP baseline to compare, or Trained BLIP to require the local checkpoint explicitly. The debug panel reports the actual model source. Local checkpoint loading does not download a replacement model. A missing explicitly selected checkpoint produces an error; Automatic uses the pretrained baseline if the local checkpoint is absent.

On a fresh checkout, install Git LFS and run `git lfs install` followed by `git lfs pull` to download the versioned checkpoint. Alternatively, copy the ZIP into `models/` and run `python -m utils.checkpoint_import` once if the checkpoint directory is absent. This imports only the expected model/tokenizer files and refuses to overwrite an existing checkpoint. The ZIP is excluded from Git. The trained safetensors file is tracked with Git LFS; tokenizer and configuration files are stored in normal Git. Transformers is pinned to 4.57.6 to match the checkpoint export; loading rejects missing or mismatched weights.

The checkpoint contains `BlipForQuestionAnswering` weights, not learned QGES, fusion or a separate answerability classifier. Selecting this model does not enable those adapters or establish grounded answers. If VQA outputs the exact label `unanswerable`, the app presents an uncertainty response; this is model output handling, not a calibrated gate. Other answers are displayed and spoken without a repetitive reliability suffix. Use `$env:VISIONBRIDGE_VQA_MODE = 'pretrained'` or `'trained'` to change the startup default.

BLIP uses Salesforce/blip-vqa-base. Whisper uses tiny.en with CPU int8; set `Config.whisper_model` to `tiny` for multilingual transcription. Speech output uses installed offline Windows SAPI voices through pyttsx3. First use downloads model files; later inference can operate offline if all required files are cached. BLIP is roughly a gigabyte; downloads and CPU inference can take time. CUDA is detected for BLIP, YOLO, EasyOCR and CLIP; Whisper intentionally uses CPU. Force CPU with `$env:VISIONBRIDGE_DEVICE = 'cpu'`.

The baseline cannot reliably detect unanswerable questions. The UI notice and debug panel explain that answer reliability is not calibrated. No arbitrary answer confidence is produced. Missing custom checkpoints fall back with warnings. Future gate rejection returns an uncertainty response. Do not interpret baseline output as a verified reading of medication, hazards or other critical details.

The app binds to localhost and does not create a public share link. Images/audio are processed locally, though Gradio creates temporary upload files. Speech WAV files are uniquely named in outputs; files older than a day are removed on the next synthesis. Delete local uploads/cache according to your own retention needs.

## Future Colab integration

See [models/README.md](models/README.md) for the adapter contract, checkpoint locations and enablement. Baseline stays usable without trained weights. Config.py controls model names, paths, device and QGES. Custom fusion and gate semantics must match the architecture trained in Colab.

## Verification

```powershell
python -m pytest -q
python smoke_test.py
python integration_test.py
python integration_test.py --trained
```

Unit tests use controlled model doubles for pipeline behavior and do not download weights. Smoke tests use real BLIP and Windows TTS; `--speech` also transcribes the synthesized WAV using Whisper, and `--evidence` tests YOLO, OCR and CLIP. These tests are functionality checks, not accuracy evaluation. A synthetic image does not establish VQA quality. Browser camera/microphone hardware still require a manual check.

## Troubleshooting

- PowerShell activation blocked: run `.\.venv\Scripts\python.exe app.py` directly; activation is optional.
- DLL or CUDA errors: use the matched CPU torch/torchvision pair above in a clean environment. Whisper on CPU does not require cuDNN. CUDA installs should follow https://pytorch.org/get-started/locally/.
- Model download failed: check connectivity, proxy and free disk space; retry. No weights are bundled.
- No voice or no WAV: install an English Windows speech voice in Settings. The text answer survives TTS failures.
- Slow processing: disable optional evidence and use CPU defaults; first requests load models. Requests are queued one at a time.
- Port busy: close the previous app or change the launch port in app.py.
- OCR/YOLO package errors: install requirements-evidence.txt; avoid mixing several OpenCV wheel variants in one environment.

The UI uses named controls, large buttons and the Gradio keyboard controls. Screen reader, contrast and hardware access should be checked with the intended users before deployment.

See TEST_REPORT.md for actual executed checks. The provided development environment uses `.venv` with existing system packages shared; for a reproducible personal installation, use the clean virtual environment setup above.
