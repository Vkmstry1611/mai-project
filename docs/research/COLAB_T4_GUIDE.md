# Run the improved VisionBridge BLIP workflow in Google Colab

The code is prepared for a Colab GPU run, but no improved-model result exists yet.
This guide uses the local, uncommitted files on `feature/visionbridge-model-enhancement`.
The original notebook, checkpoint and historical results are preserved.

If Section 4 failed with `Empty answer: VizWiz_train_00023642.jpg`, follow
[DATASET_FIX_GUIDE.md](DATASET_FIX_GUIDE.md). Preserve the old run01 and start
a fresh run02 with the corrected uploaded code; completed local data is reusable.

## Files to have ready

Open `notebooks/mai_model_train_improved.ipynb` through Colab's **File → Upload notebook**.
Keep these four files ready on your computer for the upload in notebook section 2:

- `scripts/research/colab_train_t4.py` (upload it to Colab with the basename `colab_train_t4.py`)
- `visionbridge_training.py`
- `evaluate_vqa.py`
- `requirements-training.txt`

Do not upload or execute `mai_model_train.ipynb`. The notebook retrieves the
existing `models/blip_vizwiz` checkpoint from the fork with Git LFS. There is no
need to push the new code first: the four local files supply the updated workflow.
Cloning the remote default branch supplies checkpoint files, not this local branch's
uncommitted training code. If the repository is private, authenticate the clone in
Colab; do not place credentials in a notebook that you share.

## First training run

1. Choose **Runtime → Change runtime type → T4 GPU**, then connect. Execute
   sections separately; do not start with Run all. Colab GPU availability varies.
2. Run **section 1: Install loading dependencies** (code cell index 2).
   It adds the pinned BLIP loading stack and Matplotlib, retaining Colab's CUDA
   PyTorch. If you imported Transformers/Pillow before installation, use
   **Runtime → Restart session**, then continue with section 2. Do not install
   the application's CPU PyTorch requirements in this runtime.
3. Run **section 2: Retrieve the original checkpoint and upload the improved
   code** (code cell 4). Select all four listed local files. Wait for Git LFS to
   finish. The checkpoint is about 1.45 GB; a small LFS pointer is not a model.
4. Run **section 3: Choose a persistent run** (code cell 6). Authorise Drive
   mounting. Keep `RESUME = False`. Choose a new `RUN_NAME`, such as
   `vizwiz-5000-lr1e5-run01`; change it if that folder already exists. Confirm the
   printed GPU is a T4 and CUDA is available. Confirm the original model hash
   passes and the persistent output starts with `/content/drive/MyDrive/`.
5. Keep the default configuration for the first planned run: 5,000 training
   examples, 1,000 development examples, batch 4, accumulation 4, LR 1e-5,
   at most five epochs, and early stopping after two checks without a meaningful
   score gain. To change settings, edit `TrainConfig(...)` **before creating the
   run**, choose another run name and keep its recorded configuration fixed.
6. Check the Google Drive storage page for **at least 10 GB free**. Locally,
   reserve approximately **30 GB**. Official train/validation ZIPs total about
   14.8 GB; selected extracted images, models and caches need additional space.
   Downloads are full archives even though only 5,000 training images are selected.
7. In **section 4: Prepare reproducible official data** (code cell 8), change
   `PREPARE_DATA = True` and execute it. Wait for all downloads and extraction.
   Expected sizes are `train: 5000`, `dev: 1000`, `holdout: 3319`. The notebook
   prints the local free space and refuses preparation below 30 GB. The 186 MB
   official API archive supplies the scorer's normalisation tables.
8. Run **section 5: Lightweight real-model check** (code cell 10). It loads the
   original checkpoint, verifies a finite padded-answer loss and generates two
   answers on the GPU. It does **not** perform a backward pass or update weights.
   Do not start training if this check fails. The printed sample answers are
   smoke checks, not an accuracy result.
9. In **section 6: Start or resume training explicitly** (code cell 12), set
   `RUN_TRAINING = True` and run the cell. It first measures the starting
   checkpoint on the development set, then trains. Every epoch prints development
   VQA score and loss; history and checkpoints go to Drive. A two-hour training
   stage budget excludes setup and initial evaluation; validation/checkpoint
   operations can extend past its boundary. GPU performance has not been measured
   locally. If CUDA runs out of memory, create a **new run** with batch 2 and
   accumulation 8 (same effective batch 16); do not alter a run's saved settings.
10. Check the export's `training_summary.json`. If `stop_reason` is
    `time_budget`, resume this run before final evaluation. Final comparison is
    allowed only after `max_epochs` or `early_stopping`. If the starting
    checkpoint remains best, the export keeps it and reports that explicitly.
11. Once selection is complete, in **section 7: Final comparison** (code cell
    14), enable `RUN_FINAL_EVALUATION = True`. Three models use the **same 3,319
    held-out examples, order, processor, token limits, greedy decoding and scorer**:
    pretrained BLIP, existing VizWiz checkpoint and the best new export.
    This cell saves the comparison table, all questions/reference answers/model
    answers, subgroup scores, losses and graph to Drive and displays five examples.
12. Run **Optional: inspect training and development curves** (code cell 18).
    Curves are already saved automatically; this cell displays and refreshes them
    from saved history. Use development results to diagnose overfitting, not final
    holdout results to choose more epochs.
13. Run **section 8: Export and use updated weights** (code cell 16) to display
    the model path. `MAKE_ZIP = False` is the default. Optionally enable it once
    to create a new ZIP on Drive. A duplicate ZIP is refused, not replaced.
    The standalone best-model directory is named `blip_vizwiz_v2`; never copy it
    over the original `models/blip_vizwiz` directory. App integration is a later step.

## What is saved on Drive

All persistent output is inside:
`/content/drive/MyDrive/VisionBridge/runs/<RUN_NAME>/`

| Path | Purpose |
| --- | --- |
| `config.json`, `input_checkpoint_hashes.json` | Fixed settings and original checkpoint identity |
| `code_source/`, `code_hashes.json`, `environment.json` | Exact uploaded code and runtime versions |
| `split_manifest.json`, `official_normalization.json`, `experiment_identity.json` | Examples, ten answers and evaluation identity |
| `starting_dev_predictions.json` | Existing checkpoint's initial development predictions |
| `history.json`, `history.csv`, `training_progress.json` | Loss, development VQA score, updates, saved training-stage wall time |
| `checkpoints/step-.../` | Trainable weights, optimiser/scheduler/scaler/RNG state and checksums |
| `latest.json`, `best.json` | Latest resume point and best development checkpoint |
| `plots/training_validation.png` and `.pdf` | Training loss, development loss and development VQA curves |
| `exports/export-.../blip_vizwiz_v2/` | Standalone best model, tokenizer and processor |
| `exports/export-.../training_summary.json`, `export.json` | Stop reason, recorded time and selected export |
| `comparison-.../` | Each model's predictions/metrics JSON, `model_comparison.csv`, `sample_comparison.csv`, PNG/PDF comparison graphs |
| `final_comparison.json` | Completed three-model comparison |

The comparison CSV uses VQA scores on a 0–1 scale; graphs show percentages.
`sample_comparison.csv` contains **all** holdout records, not only the five displayed
examples; its answers field preserves all ten human answers as JSON.
Final `validation_loss` is token loss on this validation holdout; development loss
is separately recorded in history. Discrete `unanswerable` output rates are not
a learned/calibrated answerability classifier.

## If Colab disconnects or the training time budget expires

1. Reconnect/select a T4 and rerun installation, checkpoint retrieval and upload.
   Upload the exact four files saved under your Drive run's `code_source/`.
2. Use the **same RUN_NAME**, set `RESUME = True`, mount Drive and run section 3.
   It restores the recorded config and rejects changed code or input weights.
3. Set `PREPARE_DATA = True` and run section 4 again. Local `/content` data may
   have disappeared; the manifest stays fixed on Drive and the same images are
   selected. Unique partial files from interrupted work are retained; rerunning
   extraction does not treat a partial image as complete. Do not overwrite a
   different/corrupt existing final file: preserve it and choose another DATA_ROOT.
4. Run the lightweight check, then section 6 with `RUN_TRAINING = True`.
   It resumes the latest complete snapshot. If disconnection happened **before
   any initial snapshot**, it starts training from the original checkpoint using
   the existing run's configuration, rather than requesting a nonexistent resume.
   After completed early stopping/max epochs, resume only re-exports the best
   model and performs no extra updates. Each resumed training invocation has a new
   120-minute budget. Unsaved work after the latest snapshot is replayed.
5. Enable final evaluation after completed model selection. If evaluation itself
   disconnected, the cell reuses fully saved per-model results and finishes the
   remaining models. When comparison is already complete, it displays the saved
   result without another evaluation.

Snapshots are written every 500 successful updates and each epoch; an abrupt
disconnect can lose progress since the last save. Training time is cumulative
saved-session elapsed time; unsaved work/time cannot be reconstructed exactly.
The workflow does not bypass Colab limits or guarantee a session stays alive.

## How to interpret the comparison

The 3,319 records are withheld from **this new model-selection run**. The old model
may already have been examined on some official validation examples; this is not
a claim of a never-seen official test benchmark. All three models still use the
same evaluation inputs and scorer for a controlled project comparison.
Historical 26.80%/48.40% or 25.13% notebook scores used different samples and
normalisation and must not be mixed with the new numbers.

An improved accuracy claim is justified only after actual Colab training and
the completed comparison. The new export can score lower, or the original
checkpoint can remain the best. Learned evidence-aware fusion and answerability
are outside this BLIP-training stage.
