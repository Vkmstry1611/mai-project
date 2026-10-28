# VisionBridge BLIP training comparison and improved workflow

## The original experiment

`mai_model_train.ipynb` has four cells: dependency installation, the full training
and evaluation block, model-only ZIP export, and an empty cell. All cells and
saved textual results were inspected. The notebook is preserved byte-for-byte.

| Setting | Vidish's notebook | Previous `colab_train_t4.py` |
|---|---|---|
| Architecture | Salesforce BLIP base, `BlipForQuestionAnswering` | Same |
| Starting model | Original pretrained BLIP | Same |
| Trainable weights | Last two decoder blocks and LM head, including tied token embeddings | Same |
| Dataset | `Raluck/vizwiz-vqa-en`, streamed | Same |
| Train / evaluation size | 800 / 100 | 800 / 100 |
| Sampling | Shuffle buffer 64, then first usable records | Same |
| Epochs requested | **15** | **2** |
| LR / batch | 2e-5 / 4 | Same |
| Training-stage cap | **30 minutes**, despite a comment saying 25 | **25 minutes** |
| Additional cap | Stop training by minute 45 from training-cell start | Stop by minute 45 from cell start, including installation |
| Data preparation limit | 20 minutes from second cell start | 20 minutes including installation |
| Optimizer | AdamW, weight decay .01 | Same |
| Precision / clipping | FP16 with scaler / 1.0 | Same |
| Question / answer limits | 48 / 24 tokens | Same |
| Images | RGB; thumbnail to 640; JPEG quality 90 | Same |
| Evaluation | Before and after only; approximate normalization and ten leave-one-out scores | Same |
| Checkpoints | Final full model + processor before final evaluation | Same |
| Resume state / best model / early stopping | Absent | Absent |
| Results packaging | Separate model-only ZIP; metadata includes LR and batch | Combined model/results ZIP |

The saved notebook run reports a T4, 15 epochs, 2,998 successful updates,
11.9 minutes elapsed, no timer stop, and diagnostic agreement improving from
26.80% to 48.40%. Its 100-example evaluation set has 82 answerable and 18
unanswerable examples; scores improve from 0% to 79.44% on the unanswerable
group and from 32.68% to 41.59% on the answerable group. The gain is uneven,
so overall score alone is insufficient. These are saved results, not reruns.

The notebook is the more complete **recorded experiment**. The script was a
shorter-run implementation of the same design. File modification dates do not
establish which source was authored most recently. The notebook export layout
matches the imported checkpoint, but the checkpoint alone cannot prove which
exact run produced it.

Keep BLIP, partial fine-tuning, ten reference answers, disjoint split checks,
the separate decoder input IDs and -100 label mask, FP16, clipping, frozen
encoder evaluation mode, before/after predictions, and model-plus-processor
export. Preserve the original results as a historical baseline. Improve both
the maintained Python implementation and a new notebook wrapper so their
training logic cannot drift apart.

## Implemented settings

- Start from the existing local `models/blip_vizwiz` checkpoint, with strict
  loading, LFS pointer rejection, and recorded input file hashes.
- Select 5,000 examples from the complete official training annotation list.
  Sampling is reproducible and stratified by answerability and answer type.
- Reserve 1,000 official validation examples for development; the other 3,319
  are for the final comparison. Record all filenames, all ten answers, seed,
  and annotation hashes in a persistent manifest.
- Preserve original image bytes; use RGB/EXIF correction and BLIP preprocessing
  without an intermediate lossy JPEG resize.
- Train the last two decoder blocks, LM head and tied decoder embeddings.
  Batch 4 with accumulation 4 gives an effective batch of 16, except the short
  final group. Accumulation is weighted by valid answer tokens.
- Use LR 1e-5, AdamW weight decay .01, FP16, gradient clipping 1.0, 5% warmup,
  linear LR decay, and at most five epochs.
- Sample one human answer per image per epoch with frequency weighting and
  deterministic sampling. Use the majority answer for validation loss.
- Evaluate development VQA score and loss every epoch. Save the highest-scoring
  model. Stop after two checks without a gain greater than .002 (0.2 percentage
  points). Smaller score gains still update the best model but not patience.
- Save resumable snapshots every 500 successful updates, each epoch, and at
  the training time boundary. Preserve optimizer, scheduler, scaler, RNG,
  epoch, exact next saved batch, best score, and patience. AMP-overflow-skipped
  updates do not advance the scheduler.
- The 120-minute budget starts after initial checkpoint evaluation. Setup,
  downloads and final export/evaluation are outside it. A new resume invocation
  has a new budget. Timing is checked between accumulated updates; it is not
  a hard process kill.
- Export a complete best model to a unique `blip_vizwiz_v2` directory. If the
  starting model is best, report that explicitly instead of claiming a gain.
- Compare pretrained BLIP, the existing checkpoint and the best new export on
  the same final images with identical preprocessing and greedy decoding.
  Save predictions and answerability/answer-type breakdowns.

No learned evidence selection, OCR fusion, or calibrated answerability
classifier is introduced by this BLIP-only stage.

## Evaluation rules

`evaluate_vqa.py` uses the literal normalization tables from the
[official VizWiz API](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip),
records their source hash, and implements its leave-one-out accuracy block.
Only predictions are normalized, matching that API; supplied references remain
unchanged. This includes number words, articles, contractions and punctuation,
instead of the original notebook's approximate regex. Downloaded Python code
is not executed by the training pipeline. Caption metrics are outside scope.

The official API ZIP is about 186 MB because it includes optional caption
evaluation assets. Data preparation downloads it once, but only reads the
normalization tables. An offline test compares this scorer with the official
accuracy implementation from the cached ZIP.

The existing checkpoint may already have been inspected on part of validation.
Therefore the held-out remainder is untouched by **this new model-selection
run**, not a claim of a never-seen test benchmark. For a final project report,
consider the official released test annotations with a predeclared protocol.
Never tune on final evaluation results. The original 48.4% score cannot be
directly compared with scores on a different split or evaluator.

## Run in Colab

Follow [COLAB_T4_GUIDE.md](COLAB_T4_GUIDE.md) for the complete beginner guide,
cell numbers, persistent output paths and disconnect recovery.

1. Open `notebooks/mai_model_train_improved.ipynb`; choose a T4 runtime.
2. Run installation. Keep Colab's CUDA PyTorch; do not install application CPU
   requirements. Restart if dependencies had already been imported.
3. Clone the fork and retrieve its existing LFS checkpoint. Upload the three
   locally changed Python files and `requirements-training.txt` into the separate code directory. Uncommitted
   changes are not yet on GitHub.
4. Mount Drive. Choose a unique run name and create the run. A pre-existing run
   is refused unless explicitly resuming. Confirm the checkpoint's SHA-256.
5. Enable the data-preparation switch. Official image ZIPs total about 14.8 GB;
   only selected training images and the validation images are extracted.
   Images and archives stay on Colab disk; run metadata/checkpoints stay on
   Drive. Complete existing files are reused, never silently overwritten.
6. Run the two-example check. It computes an evaluation loss and generates
   answers without a backward pass or optimizer update.
7. Enable `RUN_TRAINING` only when ready. The notebook defaults it to False.
8. On disconnect, restore setup/data and use the same run name with RESUME=True.
   Training continues from the latest **saved** update; unsaved work is replayed.
   A completed early-stopped run is only exported again, not retrained.
9. After early stopping or maximum epochs, enable final comparison once. A
   time-budget pause must first be resumed. All three models use identical
   held-out records, the starting-checkpoint processor, decoding and scorer.
   Saved per-model results are reused if comparison is interrupted.
10. Keep the run folder. Optionally ZIP the best full model. Later install it
    as `models/blip_vizwiz_v2` and deliberately configure the app to select it.
    Do not replace the original model. Git LFS rules for that new path should
    be added when a real export is ready to version.

Snapshots contain only trainable weights, and require the unchanged starting
checkpoint to restore frozen weights. Their optimizer state is not an app model.
The full exported model is standalone. Each snapshot can use several hundred MB;
no automatic deletion of older snapshots is performed. Reserve at least 10 GB
free in Drive and about 30 GB on Colab disk. Longer runs with repeated exports
can require more space. Exact code copies, environment information, loss/VQA/time
histories, prediction tables and PNG/PDF graphs are saved in the Drive run folder.

For a prepared run, the CLI supports:

```bash
python -m scripts.research.colab_train_t4 prepare --run-dir RUN_DIR --data-root DATA_ROOT
python -m scripts.research.colab_train_t4 train --run-dir RUN_DIR --data-root DATA_ROOT
python -m scripts.research.colab_train_t4 train --run-dir RUN_DIR --data-root DATA_ROOT --resume
python -m scripts.research.colab_train_t4 evaluate --run-dir RUN_DIR --data-root DATA_ROOT
```

The old checkpoint provides weights, but no old optimizer state. Continuing
from it starts a new optimizer; exact state resumption applies to new snapshots.

## Lightweight verification

Offline tests use tiny synthetic models for accumulation/checkpoint behavior.
They never download images or weights or run full BLIP training:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_training_workflow.py -v
```

Also check all new Python/notebook cells compile, perform a real local BLIP
evaluation forward pass with padded answers, and verify the original notebook
and checkpoint hashes. CUDA performance, FP16 behavior on a T4, full dataset
downloads and end-to-end training still require the explicit Colab run.

Executed locally on 10 October 2026: **21 tests passed**, including an isolated
official-API scoring comparison, skipped-AMP-step behavior, best-model export,
resuming an already completed run without more updates, interrupted download/
extraction recovery, and resuming the fair three-model comparison. All new Python and
notebook code compiled. A real CPU BLIP evaluation forward pass with padded
answers produced finite loss **2.9425**; its 42,968,636 trainable parameters match
the original partial fine-tuning setup. No real BLIP optimizer updates ran.
Both original artifact SHA-256 hashes remained unchanged. Temporary-file tests
required permission outside the sandbox; the final suite had no test failures.
