# Future trained components

## Integrated BLIP export

`visionbridge_model_only.zip` contains a Hugging Face `BlipForQuestionAnswering` export and tokenizer/processor files. Import it with `python -m utils.checkpoint_import`; the app automatically selects `models/blip_vizwiz/` when available. Transformers 4.57.6 matches the exported version. Model weights are loaded locally with safetensors and checked for missing/unexpected keys. Use the UI model selector to switch back to pretrained BLIP. This checkpoint is independent of the future adapter contracts below.

Put trusted checkpoints and an `adapter.py` in `models/qges/`, `models/fusion/`, or `models/answerability/`. No custom weights are supplied. Set `Config.enable_trained = True` and select custom adapters in the UI. Missing adapters produce warnings and fall back to BLIP; an absent gate never produces a probability.

Each adapter exports `load(checkpoint_directory: Path, device: str)` and returns an object with `predict(inputs: FusionInput)`. QGES returns a list of evidence dictionaries; fusion returns `FusionOutput`; answerability returns `GateResult`. The gate receives the question, evidence and optional CLIP features before baseline generation; `baseline_answer` is empty at this stage. A negative gate decision prevents VQA and yields an uncertainty response.

Implement preprocessing, architecture construction and checkpoint loading inside your adapter to match your Colab training. Use `model.eval()` and inference mode. Prefer safetensors or `torch.load(..., weights_only=True)`. Adapter Python code executes locally: only use files you trust. Calibrate the gate threshold using held-out data; document its probability semantics. Set `grounded=True` only when your actual model supports that claim.

Evidence dictionaries have `kind`, `text`, `box` (xyxy pixels in the normalized image), and extractor `confidence`; OCR also has a polygon. QGES adds lexical `relevance`, which is not answer confidence. CLIP returns normalized image/region embeddings and cosine similarities. Regions follow the first five boxed evidence items. The original image is resized to fit 1600 pixels before inference.

BLIP and Whisper download only when used. YOLO, OCR and CLIP download only when individually enabled. Cached files live here and are ignored by Git. No dataset is needed.
