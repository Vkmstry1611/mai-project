# Phase 3 local answerability and reliability

Use `.venv-gpu` only. The original application, `.venv`, BLIP checkpoints,
run02 manifest and all frozen Phase 2 modules/results remain unchanged.

Current experiment: `reports/phase3_answerability/run-local-20261012-v1`.
Original training membership: 5,000 examples. Development: the saved 1,000
examples and 17 additional OCR overlays. No holdout access is implemented.

The official training ZIP is accessed by HTTPS byte ranges. Only named
training entries are decoded and stored. Every image has archive CRC,
readability and SHA256 checks; HTTP range responses must match archive size
and ETag. Completed images are reused and changed images are refused.

## Resume preparation

```powershell
Set-Location .
& .\.venv-gpu\Scripts\python.exe -u .\prepare_phase3_data.py `
  --run-dir .\reports\phase3_answerability\run-local-20261012-v1 --workers 4
```

Do not run two preparers concurrently. Missing entries or download failures
stop preparation rather than substitute images. Existing verified files are
never overwritten. Interrupted temporary files are retained for diagnosis.

## Execute or resume

```powershell
& .\.venv-gpu\Scripts\python.exe -u -m visionbridge.run_phase3_local `
  --run-dir .\reports\phase3_answerability\run-local-20261012-v1
```

The runner waits for completed image verification, runs a 20-training-example
debugging pilot, captures the complete training split, fits two small CPU
classifiers, replays cached development examples, and runs regression tests.
Use one runner process. Progress logs are in `execution/`.

BLIP uses batch size 1, FP32 weights and CUDA FP16 autocast as in the existing
development protocol. OCR and the existing conditionally needed detector use
CPU. CPU threads are limited to four. An OOM first clears unused allocations
and retries the unchanged protocol once; further failure is recorded rather
than silently changing the model/precision. BLIP answers are saved before
OCR, allowing an interrupted OCR extraction to resume without regeneration.

Targets and features are separate. The answerability target is the original
binary label. Candidate reliability is full official VQA credit and answerable
label 1 for a non-abstaining original-training candidate; partial credit is
negative for this conservative target. References/scores never enter inference
features. Scaling, regularization fitting and sigmoid probability calibration
use original-training examples only. Development chooses thresholds from a
small fixed grid. Neither output probability proves visual grounding.
The existing question-support rule is included only as a learned numeric hint,
not a new hard veto. Model and final result folders are published only after a
complete attempt; interrupted attempts remain preserved and are safely retried.
Features also check whether the candidate matches selected high-confidence OCR
fragments/phrases and their tokens. They compare inference outputs with evidence,
never with human reference answers.

## Results

`results/PHASE3_FINAL_REPORT.md` contains actual development results. Supporting
files include comparison CSV, full threshold sweep, selected per-example
records, corrected/worsened/neutral abstentions, subgroup metrics, calibration
diagnostics and plots. `selected-policy.json` records the selected configuration.
If no gated policy meets the predeclared acceptance rule, the frozen Phase 2
baseline remains selected; no improvement is forced.

BLIP was trained on these same training examples, so its training predictions
can be optimistic. Gate cross-validation does not remove that base-model bias.
Development results are selection results, not independent or holdout accuracy.
Existing abstentions are preserved; the gates cannot recover missed answers.

## Optional single-image application check after completion

The original Gradio app is preserved. Gradio is not installed in `.venv-gpu`,
so a separate CLI wrapper makes the completed gate usable without changing
either environment. It applies the development-selected policy after the
unchanged Phase 2 candidate, with no silent model fallback:

```powershell
& .\.venv-gpu\Scripts\python.exe -m visionbridge.phase3_infer `
  --run-dir .\reports\phase3_answerability\run-local-20261012-v1 `
  --image C:\path\to\your-photo.jpg --question "What is written here?"
```

The output preserves BLIP and Phase 2 answers, decision reason and evidence
IDs, and adds gate scores/reason and final answer. These probabilities estimate
answerability/reference agreement; they do not certify visual grounding.
No extra single-image GPU inference is run automatically after evaluation.

A separate optional UI can reuse the existing Gradio installation in the
original environment while running all model work in `.venv-gpu` subprocesses:

```powershell
& .\.venv\Scripts\python.exe -m visionbridge.app_phase3 `
  --run-dir .\reports\phase3_answerability\run-local-20261012-v1
```

This does not change `app.py`, install packages, or load a model in `.venv`.
It refuses requests until the development evaluation is complete. Each request
loads a fresh GPU process, so cold-start latency is higher than the evaluation.
Use your own photos; do not use the reserved holdout for tuning.

## Verify again without model inference

```powershell
& .\.venv-gpu\Scripts\python.exe -m visionbridge.check_phase3 `
  --run-dir .\reports\phase3_answerability\run-local-20261012-v1
```

The runner preserves partial results instead of overwriting them. If evaluation
or model saving was interrupted after partially publishing final artifacts,
inspect the error and create a separate result/model version; completed
training captures still remain reusable. No commits/pushes are performed.
