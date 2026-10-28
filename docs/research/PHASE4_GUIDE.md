# Phase 4: balanced frozen policy

Run directory: `reports/phase4_policy/run-local-20261013-v1`.

The exact original development split and saved probabilities choose the policy.
Before holdout access, require nondecreasing development VQA and answerability
precision, at least 95% of Phase 2 answerability recall, 90% of answer coverage
and 90% of OCR-question answer coverage. Choose highest VQA among feasible
policies, then recall and coverage. This is a transparent engineering trade-off,
not a user-validated safety standard. Development has been reused for tuning.

The selected answerability threshold is 0.00, reliability threshold 0.05; OCR
stays 0.90. Both heads run, but the answerability cutoff is effectively disabled
for valid probabilities. Reliability is a weak estimate of full reference credit,
not proof of correctness. Original Phase 3 policies remain untouched.

The holdout runner refuses unfrozen or modified policies and only resumes the
same immutable experiment. It downloads exact official validation ZIP members
with CRC/SHA/readability checks. Each BLIP prediction is saved before CPU OCR.
No references or labels reach generation or inference features. Once all final
predictions are fixed, the official scorer reports the selected system and its
BLIP/Phase 2 controls. No other thresholds are tested on holdout.

```powershell
Set-Location .
& .\.venv-gpu\Scripts\python.exe -u .\phase4_holdout.py --run-dir .\reports\phase4_policy\run-local-20261013-v1
```

Completed results are never rerun or overwritten. An interrupted run reuses
verified image and prediction caches; run only one writer. OOM clears unused
memory and retries the same protocol once, then stops rather than changing
precision. BLIP uses CUDA, batch 1; EasyOCR and conditional YOLO use CPU.

Launch the integrated Gradio mode using the existing UI environment:

```powershell
& .\.venv\Scripts\python.exe .\launch_phase4.py --run-dir .\reports\phase4_policy\run-local-20261013-v1
```

The launcher reuses `app.py --phase4-run` and the existing UI, with an
application-only GPU worker bootstrap. YOLO's CPU selector can hide CUDA before
BLIP loads; this bootstrap initializes CUDA and caches explicit GPU resolution
first. It refuses CPU fallback, preserves the frozen model/policy/evaluator,
and removes an obsolete Phase 2 warning about missing learned answerability.
`APPLICATION_FREEZE.json` records the added integration source hashes.

Running `app.py` without a flag still opens the previous app. The UI supports photos,
typed or transcribed questions, optional offline speech, evidence boxes and
decision details. GPU inference runs in `.venv-gpu`; no dependencies are installed
in either environment. Each request loads a GPU process, so cold-start latency
is higher than batch evaluation. Missing models or policy failures never fall
back silently. Use personal photos for UI testing.

Results are in `PHASE4_FINAL_REPORT.md`, `development_summary.json`,
`development_records.json`, `comparison.csv`, `POLICY_FREEZE.json`, `plots/`,
and `holdout/`. Freeze before holdout; never alter thresholds using its results.
