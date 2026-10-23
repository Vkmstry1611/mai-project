# VisionBridge — Multimodal AI Assistant for Visually Impaired Users

VisionBridge is a university AI/ML accessibility prototype that answers questions about images and reads question-relevant printed text. It combines a fine-tuned BLIP model with visual evidence, conservative answer-or-abstain decisions, and an experimental voice-first workspace.

Blind and visually impaired users may need a scene description, a label reading or a specific printed field from the same photograph. These tasks need different evidence. VisionBridge routes questions automatically and keeps uncertainty visible rather than guessing missing information.

## Problem statement and objectives

Photographs, packaging, signs and documents often contain information that is difficult to access without sight. Small print, poor focus and incomplete visual context make an unsupported answer especially unhelpful. VisionBridge is an academic prototype for exploring accessible access to this information; it is not a validated assistive device.

The objectives are to answer visual questions, read supported text, provide voice input/output, preserve evidence for decisions, support contextual follow-ups, and offer useful capture guidance when evidence is insufficient. These objectives are implemented with conservative limitations described below.

## Features and project contributions

- Frozen BLIP v2 VQA with preserved Phase 2–4 evidence/gating policies.
- Separate retained EasyOCR and RapidOCR CPU readers, orientation correction and question-aware region selection.
- Experimental task-specific quality hints, guided retake and explicitly labeled field extraction.
- Experimental evidence-aware follow-up conversations: upload once, ask multiple questions, reuse raw evidence and clarify ambiguous references.
- Whisper transcription review, automatic spoken answers, mute, Play Answer, selectable/copied text and a navy Gradio workspace.
- Complete raw evidence, source IDs, selected/excluded candidates, boxes and stage traces in collapsed Technical Details.
- Cooking-step and paragraph selection, next/back navigation, ingredient mentions and literal word-presence questions. Layout-supported blocks are not presented as verified printed step numbers.
- Session-only recall of exact earlier questions and answers. The last 60 non-recall questions are retained separately from the six visible exchanges, with the existing 30-minute image-context expiry.
- Experimental serving-size, calorie and net-quantity extraction when explicit labels and a unique supported value are available. Serving size is never used as package weight.
- Experimental scene overviews supplement a successful frozen VQA answer with existing CPU YOLO candidates. They do not override a frozen abstention or constitute a verified object inventory.

See the system architecture and known limitations directly below. Experimental features have no new official VQA accuracy claim. Historical local validation reports are not required public documentation; current measured checks are summarized here.

## Launch on the configured laptop

From the project directory, use one workspace at a time:

```powershell
.\.venv\Scripts\python.exe launch_workspace.py --port 7865
```

Open `http://127.0.0.1:7865`. Stop an older workspace with Ctrl+C in its terminal before restarting; duplicate GPU workers can exhaust a 4 GB GTX 1650. Source guards must remain enabled.

If that port is occupied, use `--port 7866` and open `http://127.0.0.1:7866/`. Run from the checkout root. Restart after source edits so initialized workers can reverify their protected files.

Upload/capture an image and type or transcribe a question. Select **Analyze Image**. Text appears first, followed by one generated audio response. Speech is enabled by default. If autoplay is blocked, press **Play Answer**; mute disables synthesis for subsequent requests. Microphone and camera activation are explicitly user-controlled.

Ask follow-ups without re-uploading the image. **Clear image, question and conversation** clears active context. See the [detailed user manual](docs/USER_MANUAL.md).

| Request | Route |
|---|---|
| What is in this image? | Frozen visual answer or abstention |
| Read the passage | Main recognized passage |
| Detect every text | Accepted text from all regions |
| What is website in the image? | Qualified visible domain evidence |
| What is the URL? | Observed URL or supported partial answer |
| What happened in the passage? | Recognized passage quotation |
| Explain that / Read the rest | Grounded context resolution or clarification |
| What is the expiry date? | Experimental explicit label/value extraction |
| What exact product is this? | Printed product/brand evidence, otherwise abstention |
| Read the second cooking step | One supported detected instruction block |
| Repeat the previous step | Repeat the last selected step; does not navigate backward |
| Go back one step | Move the instruction cursor backward |
| What ingredients are mentioned? | Recognized ingredient section or qualified preparation mentions |
| What is the serving size? | Explicit printed label/value evidence |
| How can I take a clearer picture? | Advisory image-quality guidance without BLIP |
| What was my second question? | Exact available session question; no inference |

Advanced / Developer retains manual Visual Q&A, original Experimental OCR Reading, Full Text Reading, rotation and engine overrides. The original conservative EasyOCR policy remains 0.90. The experimental full-line RapidOCR acceptance remains its pre-existing 0.75; neither is a calibrated correctness probability.

## Installation and model setup

The public deployment adapter loads the same BLIP v2 checkpoint, classifier and selected Phase 4 thresholds without private annotations or prediction records. Historical evaluation commands still require the preserved private experiment archive; do not run them as installation steps. Public-runtime deployment parity tests and environment limitations are recorded in the preparation report; an entirely new-machine installation is not yet certified.

### Fresh Windows checkout

Supported target: Windows 11, Python **3.11.9** (Python 3.11), Git with Git LFS, and an NVIDIA CUDA-capable GPU. The tested GTX 1650 has 4 GB VRAM. Frozen VQA requires CUDA: CPU-only machines can inspect source and use isolated CPU OCR, but cannot launch the complete VQA workflow with identical behavior. Install an NVIDIA driver compatible with CUDA 12.1; a separate CUDA toolkit is not needed for the PyTorch CUDA wheels.

After this PR is merged, clone the team repository. Skip automatic LFS smudging to avoid downloading the historical BLIP checkpoint, then fetch only the two deployment assets:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/Vkmstry1611/mai-project.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location mai-project
git lfs install --local
powershell -ExecutionPolicy Bypass -File .\setup_public_runtime.ps1
.\.venv\Scripts\python.exe launch_workspace.py --port 7865
```

Open `http://127.0.0.1:7865`. Stop with Ctrl+C; the workspace shuts down its owned workers. Run only one GPU workspace on a 4 GB card. Before merge, review the proposed branch on the fork instead; do not expect these new setup files on team master yet.

The setup creates separate `.venv` (Gradio/speech), `.venv-gpu` (CUDA BLIP/EasyOCR/YOLO/classifier), and `.venv-ocr` (CPU RapidOCR) environments. It installs Torch **2.5.1+cu121** / torchvision **0.20.1**, Transformers **4.46.3**, scikit-learn **1.9.1**, Gradio **6.13.0**, EasyOCR **1.7.2** and the pinned full-text profile. Do not run setup over an unrelated customized environment; it can install/update packages in those named environments. The existing working laptop environments are not changed by development validation.

### Model retrieval and integrity

- **LFS:** seven files in `models/blip_vizwiz_v2/` (only `model.safetensors` uses LFS), plus trusted `runtime/classifier/heads.joblib` (LFS). The original checkpoint is left untouched and is not selected by v2 inference.
- **Public downloads:** EasyOCR detector/English recognizer, YOLOv8n, RapidOCR English ONNX recognizer, and the four required Faster Whisper `tiny.en` files. Whisper is pinned to revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba`; each file also has a trusted SHA256/size in `runtime/manifest.json`. RapidOCR's other models are supplied by the pinned package.
- **Runtime metadata:** `runtime/policy.json`, `runtime/normalization.json`, classifier metadata and source/asset hashes. The manifest is itself anchored in `runtime_trust.py`. Do not edit hashes or disable verification to bypass a failure.
- The installer skips byte-identical files and refuses conflicting assets. Initial setup needs Internet. Read [third-party notices](THIRD_PARTY_NOTICES.md) before distributing model files. No private bundle, split manifest, development examples, personal images or recordings are needed by this public launch path.

```powershell
# Retry downloads after dependencies and LFS retrieval are complete:
.\.venv\Scripts\python.exe public_models.py download
# Read-only deployment verification:
.\.venv\Scripts\python.exe public_models.py verify
# Synthetic component checks; these do not run an evaluation dataset:
.\.venv-gpu\Scripts\python.exe validate_public_runtime.py --models
.\.venv\Scripts\python.exe validate_public_runtime.py --speech
```

Models and trusted joblib artifacts must come from the reviewed repository/LFS version. SHA256 verifies consistency with that version; it does not prove licensing permission or independently reconstruct the private training audit. The public adapter preserves exact gate feature order and normalization; original provenance hashes are attestations only. Historical freeze files and evaluation programs are unchanged.

### Speech and troubleshooting

Microphone transcription runs Faster Whisper on CPU/int8. Windows needs an installed SAPI voice for pyttsx3 spoken output. Recorded speech remains editable before analysis. Browser microphone permissions, audible playback, autoplay and screen-reader behavior need manual testing; typed questions and text responses remain available if speech fails. No microphone or camera starts automatically.

If LFS pointers remain, run `git lfs pull --include="models/blip_vizwiz_v2/model.safetensors,runtime/classifier/heads.joblib"`. If a checksum differs, preserve the file and investigate rather than overwrite it. A CUDA error requires checking the NVIDIA driver and `.venv-gpu` Torch pair; no silent CPU/older-model fallback is used. A missing dependency requires the correct environment, not mixing historical `requirements.txt` into the final profiles. For dependency checks run `python -m pip check` with each environment's interpreter.

Downloaded caches stay under ignored `models/cache/` and `.deps/`. Public request images/logs stay under ignored `runtime-local/`; these can contain personal content and must not be published. This is a local single-user prototype, not a multi-user deployment. Existing historical launchers and the private bundle installer remain available for provenance; their private requirements do not apply to the new public workspace adapter.

## System architecture

```mermaid
flowchart TD
    I[Image upload or camera] --> P[Input validation]
    Q[Typed question or explicit microphone recording] --> W[Whisper and editable transcript]
    W --> P
    P --> R[Intent router and temporary image context]
    R --> H[Session recall or advisory capture guidance]
    R --> V[Frozen BLIP v2 and Phase 4 policy]
    R --> O[Persistent CPU RapidOCR or EasyOCR]
    O --> E[Complete raw evidence cache]
    E --> S[Per-question regions, website retrieval or labeled fields]
    S --> C[Evidence-grounded experimental composition]
    V --> A[Preserved visual answer or abstention]
    V --> Y[Optional experimental CPU YOLO scene candidates]
    Y --> A
    C --> A
    H --> A
    A --> T[Text then automatic speech]
    E --> D[Collapsed technical evidence]
```

BLIP uses vision-language cross-attention, an intermediate-fusion-style approach. OCR is a separate evidence decision layer, not text appended arbitrarily to the BLIP question. The unified assistant adds experimental routing/composition around frozen inference. It cannot recover a rejected frozen visual answer by inventing a description. A qualified domain identifies visible OCR evidence, not necessarily the active browser website.

## Methodology and verified results

The original run02 manifest fixes **5,000 training**, **1,000 development** and **3,319 held-out** examples. The holdout is a reserved portion of VizWiz validation, not the official VizWiz test set. BLIP fine-tuning completed five epochs on Colab T4; the best development checkpoint was selected. Classifiers were fitted on training data only. OCR policy changes and classifier/threshold selection used development data. The final policy and protocol were frozen before the single final holdout capture. No later training or holdout tuning occurred.

Official VQA agreement uses all ten original annotator slots, preserving reference frequencies and official prediction normalization. It is a soft agreement metric, not binary answer accuracy. `unanswerable` outputs can earn agreement; therefore report answer coverage and recall alongside VQA.

| Stage / configuration | Dataset | N | Official VQA agreement |
|---|---|---:|---:|
| Phase 1 pretrained BLIP | Historical Colab holdout | 3,319 | 21.60% |
| Phase 1 existing fine-tuned BLIP | Historical Colab holdout | 3,319 | 55.38% |
| Phase 1 improved BLIP v2 | Historical Colab holdout | 3,319 | 59.48% |
| BLIP v2 development control | Original development | 1,000 | 59.05% |
| Frozen Phase 2 OCR | Original development | 1,000 | 59.15% |
| Phase 3 answerability-only, A=0.90 | Original development | 1,000 | 60.32% |
| Phase 3 both gates, A=0.90/R=0.25 | Original development | 1,000 | 60.70% |
| Phase 4 balanced, A=0.00/R=0.05 | Original development | 1,000 | 59.31% |
| BLIP v2 shared final control | Final Windows holdout | 3,319 | 58.72% |
| Frozen Phase 2 shared final control | Final Windows holdout | 3,319 | 58.75% |
| Frozen Phase 4 final system | Final Windows holdout | 3,319 | **58.89%** |

Phase 3's higher development agreement came with lower answerability recall (43.74%) and coverage (31.70%). Phase 4 prioritizes a balance: development recall **57.44%**, coverage **42.90%**. On holdout, final precision/recall/F1 are **90.50% / 60.00% / 72.16%**, answer coverage **45.04%**, abstention **54.96%**. These classification metrics describe returning an answer for an answerable-labelled question; they do not measure the factual correctness of that answer.

The final paired gain over its shared Phase 2 control is **0.1386 percentage points**, without a significance claim. It does **not** establish an improvement over historical 59.48%. The archived Colab predictions reproduce **59.4848%**; current saved BLIP predictions reproduce **58.7225%**. IDs, questions, references and checkpoint/processor export hashes match. There are 237 changed normalized answers (40 beneficial, 77 harmful, 120 neutral). Colab used batch size 4, Torch 2.11.0+cu130, Transformers 4.57.6 and a T4; Windows used batch size 1, Torch 2.5.1+cu121, Transformers 4.46.3 and a GTX 1650. These verified differences do not isolate the cause. Keep both results with their provenance; no holdout rerun was used in the investigation.

See [the public submission results and provenance](submission/RESULTS_AND_PROVENANCE.md) and its comparison/confusion plots. Detailed local reports, freeze files and per-example artifacts remain in the separately preserved artifact bundle under `reports/`; they are excluded from the proposed Git commit. Recorded holdout component latency was **7.769 s/example**, peak allocated GPU memory **2,107.7 MiB**; this excludes cold application startup.

The [16 October discrepancy update](BASELINE_DISCREPANCY_UPDATE.md) identifies a concrete batching confound: both inspected BLIP versions give the decoder an all-ones question-state mask even when encoder inputs were padded. A synthetic tokenizer check confirmed extra positions under batching. This is a plausible contributor, not proof of the cause of all 237 changed predictions. No holdout inference or frozen-result edits were used to investigate it.


## Repository structure

```text
VisionBridge/
├── launch_workspace.py           # unified opt-in assistant
├── workspace_ui.py               # Gradio presentation and safe response handling
├── workspace_intent.py          # deterministic task routing
├── image_conversation.py        # per-image context and cache reuse
├── conversation_recall.py       # earlier question/answer retrieval
├── text_hierarchy.py            # paragraphs, detected steps and navigation
├── reading_relevance.py         # OCR region selection and bounded overviews
├── ingredient_reading.py        # literal ingredient/word mentions
├── assistive_tools.py           # quality hints and explicit printed fields
├── experimental_scene*.py       # existing CPU YOLO candidate supplement
├── full_text_session.py         # persistent OCR orchestration
├── workspace_voice.py           # microphone payload checks/transcription review
├── workspace_speech.py          # automatic speech and long-text chunking
├── modules/                     # model, evidence and classifier components
├── utils/                       # checkpoint loading and image utilities
├── docs/USER_MANUAL.md          # detailed end-user instructions
├── assets/workspace/            # CSS and SVG UI assets
├── tests/                       # unit, regression and integration checks
├── submission/                  # reviewed public provenance/preparation materials
├── models/                      # local checkpoints; source alone is insufficient
└── reports/                     # private local runs/data/cache, ignored by Git
```

| Files | Purpose |
|---|---|
| `launch_workspace.py`, `workspace_ui.py` | Unified Gradio presentation |
| `image_conversation.py`, `website_evidence.py` | Session follow-ups and qualified website retrieval |
| `reading_relevance.py`, `assistive_tools.py` | Experimental selection, quality and labeled fields |
| `full_text_session.py`, `full_text_worker.py`, `easy_reading_worker.py` | Isolated persistent OCR workers and adapters |
| `persistent_demo_worker.py`, `persistent_demo_client.py` | Preserved frozen GPU transport |
| `phase4_pipeline.py`, `phase4_gpu_infer.py`, `app_phase4.py` | Frozen application/policy integration |
| `modules/`, `utils/` | Original model loading, inference, evidence and gates |
| `mai_model_train_improved.ipynb`, `visionbridge_training.py` | Preserved training workflow; not run during assistant development |
| `tests/`, `submission/` | Regression tests and small curated submission materials |
| `models/`, `runtime/` | Public deployment assets and reviewed metadata; retrieve LFS files and downloads before launch |
| `reports/` | Historical private evaluation artifacts; not needed by public runtime |

## Testing and performance

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_answerability_gate.py --ignore=tests/test_phase3_dev.py
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_answerability_gate.py tests/test_phase3_dev.py -q
```

Install test-only pytest tooling in a fresh UI environment if needed; do not replace neural dependencies. Tests use mocks, permitted synthetic fixtures and integrity checks, not a new holdout evaluation. The dedicated `validate_image_conversation.py` and `validate_followup_final.py` tools use permitted local manual fixtures and produce new reports; their saved-visual replay is explicitly labeled. They are not general dataset benchmarks.

Recorded 21 October validation: **428 regression tests and 45 subtests passed**, plus **14 classifier tests**. There were 32 initial native five-image HTTP callback completions; later checks used saved OCR/BLIP evidence and separately identified real CPU YOLO/TTS. These counts are not OCR accuracy or accessibility certification. Frozen Phase 2–4 integrity checks passed. Preparation source-only checks on 22 October passed 27 tests and 40 subtests; those are a separate, narrower scope.

Many complete-suite tests require ignored private manifests, freeze files and saved manual fixtures. A source-only checkout cannot reproduce those checks without the separately supplied artifacts, and some historical fixtures are not in the runtime-only bundle. Do not substitute data or regenerate splits. For a source-only check using installed UI/test dependencies and no model inference:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_workspace_intent.py tests/test_workspace_speech.py tests/test_artifact_bundle.py -q
```

The recorded copied-tree check passed 27 tests and 40 subtests. Artifact tests use temporary fake payloads, not real checkpoints. `pytest>=8,<10` is test-only tooling for a fresh UI environment; subtest support must match the installed pytest setup. This check does not validate model loading, private fixtures, hardware or a new-machine installation.

API latency is not browser rendering or audio duration. Cold initialization, CPU OCR fallback and speech can substantially increase end-to-end response time. Cached recall often takes around 1–4 seconds over HTTP in the recorded local checks, but different tasks and mute settings are not comparable benchmarks. Persistent initialization/caching is enabled in the experimental workspace, while original frozen launchers remain unchanged.

Other retained entry points: `launch_phase4.py`, `launch_submission.py`, `launch_reading_demo.py`, `launch_persistent_demo.py`, `launch_orientation_workspace.py` and `launch_full_text_workspace.py`. Run them with the documented `.venv` interpreter, one GPU interface at a time.

## Known limitations, privacy and responsible use

- OCR can omit, join or misrecognize text, especially curved labels, glare, small print and mixed layouts. Missing characters are not invented. Known rendered-text results are not real-photo accuracy.
- Summaries are explicitly extractive; unsupported semantic questions clarify or abstain. Many answerable visual questions are still declined. No experimental assistant holdout gain is claimed.
- A summary is a bounded reading-based overview of up to six physical OCR lines, not a reliable semantic explanation. Joined OCR words and missing content are not repaired by guessing.
- Dense soup-label ingredients and net quantity were not recovered in the recorded test. Serving size was recovered; ambiguous calorie-column values abstained. Missing text is not proof that an ingredient or allergen is absent.
- CPU YOLO scene candidates may be incomplete or wrong; the living-room check included a questionable TV candidate. This is not a verified detailed scene description or safety/navigation aid.
- Quality scores and classifier outputs do not prove correctness or product safety. This prototype is not a verified aid for medication or hazards.
- Conversation history is bounded and browser-scoped. Raw OCR/speech worker files are temporary. **Original frozen VQA callbacks still retain private request images/logs and evidence caches under ignored `reports/`. Reset does not delete preserved artifacts. The public adapter stores its request files under ignored `runtime-local/` instead.** Do not publish these files or the private runtime bundle.
- Engine overrides are workspace-wide in this local single-user prototype. It is not a production multi-user service. Camera/microphone access remains explicit; mute is browser-session scoped where storage is available.
- Browser autoplay, physical microphone/camera, audible speech, keyboard-only use and screen-reader behavior require human verification. Follow [MANUAL_TEST_CHECKLIST.md](MANUAL_TEST_CHECKLIST.md); backend tests do not certify accessibility.

## Future improvements

- Evaluate dense and curved-label text recognition on annotated real photographs before adopting extra OCR passes or changing experimental policies.
- Assess lightweight, evidence-verified semantic summarization; no new summarization model is currently implemented.
- Improve verification of scene-detector candidates and measure latency/coverage alongside answer quality.
- Add production retention controls and validate keyboard, speech and screen-reader workflows with intended users.
- Use independent development/evaluation data for future methods. The completed holdout must not become a tuning set.

## Contribution and reproducibility boundaries

Keep datasets, private runtime bundles, request recordings/images, caches, credentials and generated logs out of source changes. Preserve frozen checkpoints, thresholds and recorded results. Changes to experimental assistant behavior need their own tests and clearly separated evaluation claims. The original tracked LFS checkpoint and sample files remain unchanged; repository licensing and unresolved sample provenance still need team decisions. See [sample provenance](SAMPLE_PROVENANCE.md).

## Screenshots

Actual local UI screenshots exist, but none has been newly approved for public publication in this preparation. No private image, synthetic substitute or unapproved screenshot is embedded. Add only an approved, reviewed UI capture with a relative repository path.
