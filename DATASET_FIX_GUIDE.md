# Recover from the VizWiz empty-answer preparation error

## What was wrong

The previous loader assumed every annotator response was nonempty. Inspection of
the official `Annotations.zip` on 11 October 2026 found two empty-string responses
in the 20,523 training records and none in the 4,319 validation records:

| Image | Empty slot (zero-based) | Original answerable |
| --- | --- | --- |
| `VizWiz_train_00023642.jpg` | 3 | 1 |
| `VizWiz_train_00023682.jpg` | 6 | 0 |

Both records have ten answer slots and nine usable responses. The first blank
response has `answer_confidence: "yes"`; this flag does not make its text usable.
The archive does not identify why either annotator response is empty. It is an
annotation edge case, not evidence of a corrupt image/archive download. No
example in this archive has zero usable responses.

Official dataset format and evaluation:
[VizWiz-VQA documentation](https://vizwiz.org/tasks-and-datasets/vqa/).

## Reference policy

Policy identifier: `vizwiz-raw-ten-slots-valid-targets-v1`.

- Preserve every original answer slot, confidence field, answer type and binary
  answerability label. Do not replace blanks with `unanswerable`, deduplicate
  answers, or coerce null/numeric/malformed responses into strings.
- A usable response is an answer dictionary whose `answer` is a string with at
  least one non-whitespace character. Confidence is not used as an extra filter.
- Training samples uniformly from the list of usable responses, retaining
  repetitions. Thus a response appearing four times has four chances of selection.
  Development loss uses their deterministic majority; original order breaks ties.
  Targets retain the existing outer-whitespace trimming behavior.
- Scoring keeps all ten annotator slots and the official leave-one-out denominator.
  Valid reference strings retain their exact values and original frequencies.
  Invalid slots contribute no matches; they are not removed to create an easier
  nine-annotator metric. Predictions use the existing official normalization.
- An empty normalized prediction receives zero. This is an explicit robustness
  rule: the official API can give credit if a blank prediction matches a blank
  reference. For nonempty predictions on the official string annotations, the
  score remains the official API score. The policy is recorded in scorer identity.
- If a ten-slot example has no usable responses, exclude it **before sampling**
  from both official source splits. It is not used for training, development or
  holdout evaluation. Record its image, original answerability, answer type and
  exclusion reason in the manifest audit. Never create a replacement target.
  Counts must be reported from that manifest, not assumed from historical counts.
- Structural errors (wrong number of slots, duplicate/unsafe filenames or an
  invalid binary answerability value) still fail explicitly. No slot is fabricated
  to repair a non-ten-slot example.

All three models use the same eligible holdout rows and scorer. Seeded sampling
remains deterministic and disjoint. On the inspected official archive, no whole
rows are excluded, so the default split sizes remain 5,000 / 1,000 / 3,319.
The manifest includes annotation SHA-256 hashes, the reference policy and audits.
An existing manifest with a different/missing policy is refused, never rewritten.

## Update the already-running Colab session

Use a **fresh run**, for example `vizwiz-5000-lr1e5-run02`, with `RESUME = False`.
Keep `vizwiz-5000-lr1e5-run01` intact. Its saved code hashes/source files describe
the old loader; bypassing that check or editing its metadata would weaken
reproducibility. A failed dataset preparation has no trained state to resume.

1. Save/download your current Colab notebook if it contains notes you want to keep.
   Do not delete the old Drive run directory or change its saved code/configuration.
2. In the **same running Colab session**, rerun Section 2. Upload all four local
   files: `colab_train_t4.py`, `visionbridge_training.py`, `evaluate_vqa.py`,
   `requirements-training.txt`. The two Python files containing this fix are
   `visionbridge_training.py` and `evaluate_vqa.py`. Section 2 creates a new upload
   folder and clears their module cache; it does not overwrite the prior upload.
3. Rerun Section 3. Its import statement loads the updated module. Before running,
   set `RUN_NAME = 'vizwiz-5000-lr1e5-run02'` and `RESUME = False`. Choose a different
   unused name if run02 already exists. Keep `DATA_ROOT` unchanged so local data is
   reused. Do not resume run01 or copy new source files over its saved code.
4. Set `PREPARE_DATA = True` and rerun Section 4. The original notebook's Section 4
   code can call the corrected module; alternatively copy the updated Section 4
   cell from the revised local notebook. Expected split sizes: 5000/1000/3319.
5. Inspect the audit without changing anything:

   ```python
   print(manifest['reference_policy'])
   print(json.dumps(manifest['annotation_audit'], indent=2))
   ```

   Expected: two invalid training slots, zero invalid validation slots, and empty
   exclusion lists for both official source splits.
6. Run Section 5's GPU smoke check. Then continue using the regular Colab guide.
   Keep the training switch disabled until you deliberately start training.

No dependency installation/restart is needed just for this code fix. If imports
still resolve to an old copy, inspect `visionbridge_training.__file__` and
`evaluate_vqa.__file__`; both must point to the new Section 2 upload folder.
After an actual runtime restart, rerun Sections 1–4 as usual with the new run's
saved code and `RESUME = True` (only after run02 was successfully created).

The updated local notebook defaults to run02 and prints the audit in Section 4.
Opening a new notebook/session can discard `/content` downloads, so the same
running session is the simplest way to retain the downloaded annotations.

## Are partial files a problem?

At the reported failure, preparation has finished downloading `Annotations.zip`
and extracting `annotations/train.json` and `annotations/val.json`. It fails
while validating train annotations, **before** writing `split_manifest.json`,
downloading image archives or extracting scoring rules. Existing setup metadata
in Drive remains intact. The function writes a manifest only after both source
splits have been validated; this exception does not publish a partial manifest.

Optional read-only check in Colab, before changing the run name:

```python
old_run = Path('/content/drive/MyDrive/VisionBridge/runs/vizwiz-5000-lr1e5-run01')
for name in ('config.json', 'split_manifest.json', 'experiment_identity.json',
             'latest.json', 'export.json'):
    print(name, (old_run / name).exists())
print('Local partial files:', list(DATA_ROOT.rglob('*.partial')))
```

For this failure alone, expect configuration to exist and no manifest/checkpoint
or export. If you see later-stage files, preserve them too and use a fresh run.
This local agent cannot inspect the remote Colab filesystem directly.

**No cleanup is required.** Completed annotation files can be reused: extraction
checks their sizes and CRCs against the archive. Completed image archives/images
from any earlier attempts can also be reused if valid. Unique `.partial` files
are ignored, preserved and never promoted automatically; downloads restart
rather than continuing their bytes. If an existing final file fails integrity
validation, preserve it and select a different `DATA_ROOT` rather than overwriting
or deleting it. No files are automatically deleted by this fix.

## Verification

25 lightweight tests passed on 11 October 2026. They cover malformed/empty
individual references, preserved frequencies and labels, all-invalid exclusions,
deterministic/disjoint splits, refusal to migrate an old manifest, actual official
annotation preparation and official scorer parity for a nonempty prediction with
a blank reference. The annotation test uses a locally cached archive and never
downloads data during testing. No image downloads or full BLIP training ran.
