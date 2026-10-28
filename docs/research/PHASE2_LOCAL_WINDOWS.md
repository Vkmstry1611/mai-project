# Local VisionBridge Phase 2 evaluation

Use the existing `.venv-gpu` only. Verified locally: Python 3.11.9,
Torch 2.5.1+cu121, torchvision 0.20.1+cu121, Transformers 4.46.3,
EasyOCR 1.7.2, Ultralytics 8.3.40, CUDA and GTX 1650. Do not install the
Colab or training requirements into this environment. The original `.venv`
and Gradio files are not changed by this workflow.

The connected Drive supplied the original `split_manifest.json` and
`official_normalization.json` from the exact run02 folder. They are retained
in a new report-run folder under `inputs/vizwiz-5000-lr1e5-run02/`.
`export_phase2_dev.py` creates `dev-only.json` outside that source folder.
No random split creation or annotation replacement occurs. The original
container includes other splits, but only its `dev` rows are traversed/exported.

For a later run with manually downloaded inputs:

```powershell
Set-Location '.'
# Set $runRoot to a fresh, dedicated report-run folder with an inputs/run02 folder.
.\.venv-gpu\Scripts\python.exe -B export_phase2_dev.py `
  --run-dir "$runRoot\inputs\vizwiz-5000-lr1e5-run02" `
  --output "$runRoot\dev-only.json"
```

Run the 20-example pilot using:

```powershell
.\run_phase2_local.ps1 -RunRoot $runRoot -Prepare
```

`-Prepare` downloads the official validation archive linked by the
[VizWiz VQA dataset page](https://vizwiz.org/tasks-and-datasets/vqa/), checks
ZIP structure, extracts only the named 1,000 dev images, checks each selected
member's CRC on reuse, and verifies images with Pillow. The archive contains
all validation image bytes but held-out image members are never extracted or
opened. Missing images are errors, never substituted. No annotations are downloaded.

For the already-authorized pilot followed by all 1,000 development examples:

```powershell
.\run_phase2_local.ps1 -RunRoot $runRoot -Prepare -AutoFull
```

Omit `-Prepare` when images are already verified. Interrupted download bytes
remain at `data/val.zip.download`; rerunning preparation resumes when the
server accepts byte ranges. A server refusing resume or a corrupt final ZIP
causes an error; preserve the files and select a new data folder.

The evaluator remains `evaluate_phase2_dev.py`. `visionbridge/local_phase2_runner.py` adds
local environment checks, post-run reference/ID checks and immutable verification
files. CUDA BLIP uses batch size 1, EXIF-corrected RGB images without a thumbnail,
128 question tokens, greedy generation and 20 new tokens. OCR and object
detection use CPU. No CLIP, learned answerability, legacy pre-answer gate,
training or new BLIP download is involved.

The default GPU weights remain FP32 with FP16 autocast, matching the existing
inference configuration. On CUDA OOM, the failed attempt is retained and the
launcher starts a new process and a separate `pilot-fp16` or `full-fp16` capture
folder, with uniformly FP16 in-memory weights. The original checkpoint is never
written. Precision is recorded in the protocol and verification. Failed and
successful precision attempts must not be merged. If FP16 still fails, the run
stops with logs; there is no silent CPU or alternate-model fallback. Close other
GPU applications yourself if needed; this workflow does not terminate them.

Resume: rerun the same launcher/root without changing code, environment,
checkpoint, images or configuration. The evaluator verifies identity and reuses
completed captures. OCR/object caches and BLIP captures are shared across A/B/C;
the full experiment has its own captures and can reuse extraction caches from
the pilot. Each replay produces a new results directory, preserving previous
results. Logs are uniquely named. Only one process may use a capture folder at a time.

A is BLIP v2 only. B uses the current restricted OCR policy at 0.90 and C at
0.95. All share a candidate and scorer per image/question. C's records are
retained and its redundancy is explicitly reported if it matches B. All ten
reference slots, including duplicates and invalid individual slots, are retained.
The same `OfficialScorer` as Phase 1 scores them without replacement labels.

The pilot reports diagnostic counts only, without aggregate accuracy. If an
extraction failed, the wrapper stops before the full evaluation, preserving
results for repair. Partial evidence is reported and retains BLIP through the
existing conservative policy. Empty but successful OCR is a
valid result and retains BLIP. BLIP failures and data mismatches stop the run.

Full records include question/references, BLIP and final answers, source/reason,
evidence IDs/boxes, extraction status, VQA scores, beneficial/harmful/neutral
override flags and measured stage time. A fresh `report-*` directory contains
`REPORT.md`, `comparison.csv`, `error_analysis.json`, beneficial/harmful examples,
and labelled agreement/override plots. Mean/total policy times are shared-capture
estimates, not independent timing runs. Answerability confusion counts describe
output behaviour, not a trained gate. OCR-related cohorts use the existing router.

No development outcome establishes improved held-out accuracy. The separate
59.48% Phase 1 result on 3,319 held-out examples must remain labelled as such.
No held-out rows are scored or used to tune thresholds. Existing model/framework
versions differ from Colab; A is the correct local comparator for B/C.
