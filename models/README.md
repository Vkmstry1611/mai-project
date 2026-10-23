# Runtime models

For the public workspace use `setup_public_runtime.ps1` from the checkout root; see README. BLIP v2 and classifier heads have exact LFS paths; OCR, YOLO and Whisper use hash-verified public downloads. The original BLIP checkpoint remains unchanged and is not needed for v2 inference. No private evaluation archive is required by the public adapter.

The following historical model documentation remains for compatibility:

# Checkpoints and historical adapter contracts

## Current frozen Phase 4 application

The final launch path is `launch_phase4.py` (or the presentation-only `launch_submission.py`). It explicitly loads the separate `models/blip_vizwiz_v2/` run02 checkpoint in `.venv-gpu`, uses CPU OCR/YOLO, and applies the verified Phase 3 classifier heads with the frozen Phase 4 policy. A failed explicit v2 load does not fall back to the original model. The frozen GPU environment uses Transformers 4.46.3 with an exact tied-decoder compatibility check; the original environment below is a different historical workflow.

The classifier heads live under the preserved Phase 3 report run, not `models/answerability/`. Generic learned QGES/fusion adapters described below were not trained or used in the final system. Current OCR selection is rule-based and evidence-aware; calibrated lightweight gates run after the candidate decision. See the root README and Phase 4 guide for required artifact paths and supported features. The original checkpoint remains unchanged. Do not normalize frozen artifact bytes or overwrite either model directory.

## Historical application: original BLIP export

The integrated checkpoint is versioned with Git LFS. After cloning, run `git lfs install` and `git lfs pull` to retrieve `blip_vizwiz/model.safetensors`. Processor/tokenizer configuration is stored in normal Git. The original ZIP remains ignored; do not import it over an existing checkout of the checkpoint.

`visionbridge_model_only.zip` contains a Hugging Face `BlipForQuestionAnswering` export and tokenizer/processor files. Import it with `python -m utils.checkpoint_import`; the app automatically selects `models/blip_vizwiz/` when available. Transformers 4.57.6 matches the exported version. Model weights are loaded locally with safetensors and checked for missing/unexpected keys. Use the UI model selector to switch back to pretrained BLIP. This checkpoint is independent of the future adapter contracts below.

The remaining paragraphs describe the historical custom-adapter extension contract, not the frozen Phase 4 inference policy. Put trusted checkpoints and an `adapter.py` in `models/qges/`, `models/fusion/`, or `models/answerability/`. No weights for these generic adapters are supplied. Set `Config.enable_trained = True` and select custom adapters in the historical UI. Missing adapters produce warnings and fall back to BLIP; an absent generic adapter gate never produces a probability. These fallback semantics do not apply to the explicit final v2 checkpoint or required Phase 3 heads.

Each adapter exports `load(checkpoint_directory: Path, device: str)` and returns an object with `predict(inputs: FusionInput)`. QGES returns a list of evidence dictionaries; fusion returns `FusionOutput`; answerability returns `GateResult`. The gate receives the question, evidence and optional CLIP features before baseline generation; `baseline_answer` is empty at this stage. A negative gate decision prevents VQA and yields an uncertainty response.

Implement preprocessing, architecture construction and checkpoint loading inside your adapter to match your Colab training. Use `model.eval()` and inference mode. Prefer safetensors or `torch.load(..., weights_only=True)`. Adapter Python code executes locally: only use files you trust. Calibrate thresholds on development/calibration data, never the project's final holdout; document probability semantics. Set `grounded=True` only when the implementation supports that claim.

Evidence dictionaries have `kind`, `text`, `box` (xyxy pixels in the normalized image), and extractor `confidence`; OCR also has a polygon. QGES adds lexical `relevance`, which is not answer confidence. CLIP returns normalized image/region embeddings and cosine similarities. Regions follow the first five boxed evidence items. The original image is resized to fit 1600 pixels before inference.

BLIP and Whisper download only when used. YOLO, OCR and CLIP download only when individually enabled. Cached files live here and are ignored by Git. No dataset is needed.
