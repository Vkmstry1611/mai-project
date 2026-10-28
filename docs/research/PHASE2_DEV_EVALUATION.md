# Phase 2 development evaluation

This workflow tunes only the **exact 1,000 saved development rows from run02**.
It does not read or evaluate held-out examples during inference or policy tuning.
The 59.48% result on 3,319 held-out examples remains the Phase 1 baseline; it is
not a development score and must not be compared directly with these dev scores.

## What is compared

| Configuration | Behaviour |
| --- | --- |
| A | Improved BLIP v2, raw decoded answer, no legacy gate or speech |
| B | Same BLIP candidate + existing question-guided restricted OCR decision, threshold 0.90 |
| C | Same as B with threshold 0.95; omitted from the main comparison when all normalized answers equal B |

The current application policy is reused, including object-box association for
questions that name a supported object. CLIP and learned answerability are not
used. OCR is extracted only for reading/price/brand/date intents supported by
the current router. No evidence, partial/failed extraction, ambiguity, unsupported
intent or BLIP's `unanswerable` response retains the BLIP candidate.

All configurations share **one** BLIP prediction and evidence capture per row.
Images use EXIF correction and RGB without the UI thumbnail. Questions use
128 tokens; decoding is greedy, 20 new tokens, one beam, no sampling. CUDA uses
FP16 autocast; CPU uses FP32. This follows Phase 1's processing/decoding settings,
using batch size one to match the application. Small numeric differences from
Phase 1's batched inference are possible. A/B/C comparisons are exact because
they reuse the same candidate. Do not interpret a newly measured development A
as a reproduction of the held-out 59.48% result.

The exported v2 processor is used for every policy. Checkpoint hashes, processor
files, optional generation/tokenizer files, image bytes, source code hashes,
package versions, scoring rules and capture settings are recorded. Resume
rejects changed inputs. OCR/object cache entries carry IDs, source, boxes,
extraction status and integrity checks. Interrupted `.partial` files are ignored
and preserved; there is no automatic cleanup or overwrite.

## 1. Prepare the current code for Colab

The new code is uncommitted: cloning GitHub will not retrieve it. In PowerShell:

```powershell
Set-Location '.'
$bundle = Join-Path $PWD '.deps\visionbridge-phase2-dev-code.zip'
if (Test-Path -LiteralPath $bundle) { throw 'Bundle exists; choose a new filename.' }
Compress-Archive -LiteralPath @('config.py', 'evaluate_vqa.py', 'visionbridge_training.py', 'export_phase2_dev.py', 'evaluate_phase2_dev.py', 'requirements-training.txt', 'requirements-phase2-colab.txt', 'modules', 'utils') -DestinationPath $bundle
```

Create a **separate Colab notebook**, enable Runtime → Change runtime type → T4
GPU, and upload this ZIP. Do not execute the training or Phase 1 final-evaluation
cells. The original notebooks and checkpoint remain unchanged.

Run these setup cells in order:

```python
from google.colab import drive, files
from pathlib import Path
import zipfile

drive.mount('/content/drive')
uploaded = files.upload()  # Select visionbridge-phase2-dev-code.zip
assert len(uploaded) == 1, 'Upload only the code ZIP.'
code = Path('/content/visionbridge_phase2_dev')
assert not code.exists(), 'Use a fresh directory; do not overwrite a previous upload.'
code.mkdir()
with zipfile.ZipFile(next(iter(uploaded))) as archive:
    archive.extractall(code)  # Only the ZIP you created from your own project.
```

```python
%cd /content/visionbridge_phase2_dev
%pip install -r requirements-phase2-colab.txt
```

If Colab asks for a restart, restart the session, remount Drive and run `%cd`
again. **Do not install `requirements.txt` in Colab**: it pins the local Torch
pair. The Colab requirements retain Colab's CUDA Torch/torchvision pair.

```python
import torch
import easyocr
import ultralytics
print('Torch:', torch.__version__)
assert torch.cuda.is_available(), 'Enable the T4 GPU runtime.'
print('GPU:', torch.cuda.get_device_name(0))

from pathlib import Path
RUN02 = Path('/content/drive/MyDrive/VisionBridge/runs/vizwiz-5000-lr1e5-run02')
V2 = RUN02 / 'exports/export-001564-fde95532/blip_vizwiz_v2'
PHASE2 = Path('/content/drive/MyDrive/VisionBridge/phase2-dev')
PHASE2.mkdir(parents=True, exist_ok=True)
DEV = PHASE2 / 'run02-dev-only.json'
assert (RUN02 / 'split_manifest.json').is_file()
assert (RUN02 / 'official_normalization.json').is_file()
from visionbridge_training import verify_checkpoint
print(verify_checkpoint(V2))  # Read-only; never loads or changes training state.
```

## 2. Export the saved dev split once

```python
from export_phase2_dev import export_dev, validate_dev
import json
if not DEV.exists():
    export_dev(RUN02, DEV)
payload = json.loads(DEV.read_text())
dev_rows = validate_dev(payload)
assert len(dev_rows) == 1000
print('Saved dev rows:', len(dev_rows), 'Checksum:', payload['rows_sha256'])
```

The Phase 1 JSON stores all splits together. The exporter reads that container
and selects only its saved `dev` field; it never traverses, prints, exports or
scores train/holdout rows. It copies the original ten answer slots, duplicates,
empty slots and answerability labels unchanged. No resampling, label repair,
new annotation download or repeated split creation occurs. Subsequent evaluator
runs accept **only the dev-only export**, and reject manifests containing other
splits. The original manifest is hashed for provenance and never rewritten.

## 3. Obtain only the required development images

If the original Colab session still has `/content/visionbridge_data/images`,
reuse those files. Only image IDs in `dev_rows` will be opened. After a fresh
session, use the existing validation ZIP if available, or download that archive
into a separate Phase 2 data directory. The archive contains the official
validation images; only saved dev IDs are extracted/opened, never held-out IDs.
Downloading an archive is not held-out evaluation.

```python
from visionbridge_training import download_once, extract_named, DATA_BASE
IMAGES = Path('/content/visionbridge_data/images')
missing = [r['image'] for r in dev_rows if not (IMAGES / r['image']).is_file()]
if missing:
    DATA = PHASE2 / 'data'
    archive = download_once(DATA_BASE + 'images/val.zip', DATA / 'val.zip')
    IMAGES = DATA / 'images'
    extract_named(archive, [r['image'] for r in dev_rows], IMAGES)
assert all((IMAGES / r['image']).is_file() for r in dev_rows)
print('Development images ready:', len(dev_rows))
```

The archive is large. No new image downloader or approximate dataset is used.
If there is insufficient Drive space, use `/content/phase2-data` for the archive
and images and recreate them after disconnect. Keep results and caches on Drive.

## 4. Run the real 20-example pilot

```python
import subprocess, sys
pilot_command = [sys.executable, 'evaluate_phase2_dev.py',
    '--dev-manifest', str(DEV), '--images', str(IMAGES), '--checkpoint', str(V2),
    '--output', str(PHASE2 / 'pilot-001'),
    '--evidence-cache', str(PHASE2 / 'evidence-cache'),
    '--model-cache', str(PHASE2 / 'model-cache'),
    '--yolo-model', str(PHASE2 / 'model-cache/yolov8n.pt'),
    '--device', 'cuda', '--mode', 'pilot', '--limit', '20']
subprocess.run(pilot_command, check=True)
```

The pilot deterministically uses the first 20 rows in the saved dev order. It
checks loading, image paths, evidence status, cache use, sample decisions and
artifact saving. It is **not an accuracy experiment**. Per-example scores are
debugging information; pilot CSV/summary files omit aggregate VQA agreement.
Inspect `extraction_failures`, supporting IDs, and examples where OCR changes
the answer. Do not select thresholds from the pilot.

If a session disconnects, reconstruct the same variables/command and append
`'--resume'`. Keep the same device, packages, code, images, model and mode.
Completed captures are reused; each policy replay gets a new comparison
directory. For changed settings/code or corrupt final JSON, preserve the old
directory and select `pilot-002`. Failures of BLIP or missing images stop the
run; extraction failures are explicitly recorded and retain BLIP. Do not launch
two evaluator processes against the same output/cache concurrently.

**Stop after the pilot. Review its diagnostics and obtain approval before the
full development run.** After approval, copy the pilot command, change output to
`PHASE2 / 'dev-001'` and mode to `'full'`. Full mode always uses all 1,000 rows;
the pilot limit is ignored. The evidence cache can be reused. BLIP is run once
per full-run row; captures and results are saved on Drive after each example.

## Windows execution

For a local pilot, download `run02-dev-only.json` from Drive. Also download a
dev-only images ZIP; this cell includes exactly the saved dev IDs:

```python
from google.colab import files
with zipfile.ZipFile('/content/run02-dev-images.zip', 'x', compression=zipfile.ZIP_STORED) as archive:
    for r in dev_rows:
        archive.write(IMAGES / r['image'], arcname=r['image'])
files.download(str(DEV))
files.download('/content/run02-dev-images.zip')
```

Then in PowerShell, adjust Downloads paths if needed:

```powershell
Set-Location '.'
$inputDir = Join-Path $PWD 'outputs\phase2-dev-inputs'
if (Test-Path -LiteralPath $inputDir) { throw 'Inputs exist; reuse them or choose a fresh folder.' }
New-Item -ItemType Directory -Path $inputDir | Out-Null
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\run02-dev-only.json" -Destination "$inputDir\run02-dev-only.json"
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\run02-dev-images.zip" -DestinationPath "$inputDir\images"
# Your app environment already has EasyOCR. Install evidence dependencies if missing:
# .\.venv\Scripts\python.exe -m pip install -r requirements-evidence.txt
.\.venv\Scripts\python.exe -B evaluate_phase2_dev.py `
  --dev-manifest "$inputDir\run02-dev-only.json" `
  --images "$inputDir\images" `
  --checkpoint '.\models\blip_vizwiz_v2' `
  --output '.\outputs\phase2-dev-pilot-001' `
  --evidence-cache '.\models\cache\evidence' `
  --device auto --mode pilot --limit 20
```

Use the same Python environment that loaded your real checkpoint. The repository
`.venv` currently has CPU Torch; `auto` will use CPU there. For your separate
CUDA application environment, use its Python executable and `--device cuda`.
An explicit CUDA request fails if unavailable. OCR and object detection run on
CPU to conserve the GTX 1650's approximately 4 GB VRAM. Local CPU is acceptable
for the pilot but slow for all 1,000; use Colab T4 for the full run after approval.

## Reading the output and choosing a threshold

Each experiment contains immutable `identity.json`, `captures/0000.json`, etc.,
and `comparison-<unique ID>/` containing `summary.json`, `comparison.csv`, and
`A_records.json`, `B_records.json`, `C_records.json`. C's records are retained for
audit even if its main summary is omitted as redundant.

Per-example records contain image/question, unchanged references and labels,
BLIP answer, OCR and selected evidence, extraction status, final answer, reason,
answer source, supporting IDs, official scores, delta, timing, and flags for
beneficial/harmful overrides. All ten reference slots and original frequencies
are scored by the existing Phase 1 `OfficialScorer`, without replacement labels.

Full summaries include overall **official VQA agreement (%)**, answerable and
unanswerable groups, and reading/price/brand/date groups with their denominators.
`ocr_supported_questions` is determined by the question router before inference,
not by successful OCR, reference answers, or the final decision. This is a
router-defined cohort, not a manually verified list of all OCR questions.
Answerability annotations are used for reporting only; no learned gate is added.

An override means an OCR-sourced answer that differs from BLIP **after official
normalization**. Beneficial/harmful means a higher/lower fractional VQA score;
beneficial does not imply complete correctness. Changed answers with equal score
are neutral. Identical normalized answers count as unchanged, even if their
source is OCR. Failed extractions are not silently removed from denominators.

The sweep uses 0.80, 0.85, 0.90, 0.95, 0.99 against the same captures. It selects
a **review candidate** by highest dev agreement, then fewer harmful overrides,
then higher threshold. If every tested policy is worse than A, it recommends
BLIP-only pending error analysis. It never changes the application's setting.
Review counts, subgroup effects, failed extraction and the actual development
errors before retaining/adjusting 0.90. Threshold search on dev is tuning, not
unbiased evidence of improvement. OCR confidence is not a calibrated answer
probability. Freeze the policy before any later held-out comparison.

Latency records measure preparation, BLIP, extraction, selection and policy
decision. A excludes extraction/selection; B/C add the shared measured evidence
time. They are estimates from shared capture, not three separate timed model
runs. First-use downloads/model loading, CPU versus GPU and cache hits affect
the mean; scoring, file writes and model hashing are excluded. Report the
environment and cache status alongside latency.

No full development evaluation or real-data accuracy result was produced while
implementing this workflow. The automated 20-example integration pilot uses
synthetic images, fake BLIP/OCR and test references only.
