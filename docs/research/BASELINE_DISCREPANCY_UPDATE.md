# BLIP baseline discrepancy — read-only update, 16 October 2026

No original report, prediction, threshold or result was changed. No BLIP training, holdout inference or threshold tuning occurred. The only new tokenization used two invented questions, not dataset questions, and loaded no BLIP model weights.

## Established facts from existing artifacts

- Archived Colab run02 predictions reproduce **59.4847846%** on 3,319 examples. Current saved BLIP control predictions reproduce **58.7225068%**.
- Image IDs, questions, answerability labels and ten raw reference slots match. All five core exported model/processor/tokenizer/vocabulary hashes match the archived protocol. Re-scoring reproduces the saved per-example scores.
- **237 normalized predictions differ**: 40 beneficial, 77 harmful, 120 neutral changes relative to Colab. They explain **−0.7623 percentage points**. The older −0.7575 number used rounded 59.48.
- Colab recorded batch size 4, Python 3.13.15, Torch 2.11.0+cu130, Transformers 4.57.6 and T4. Windows used batch size 1, Python 3.11.9, Torch 2.5.1+cu121, Transformers 4.46.3 and GTX 1650.
- Greedy decoding, 128 question tokens, 20 new answer tokens and FP32 weights/CUDA FP16 autocast are consistent in inspected configurations. The historical comparison also computed target loss before generation; the script removes label/decoder-target inputs before `generate`.

## Further preprocessing/source checks

Original and v2 image-processor, special-token map and generation configuration files are byte-identical locally. Tokenizer configuration differs only by an explicit `use_fast: false` entry in v2; both implementations explicitly request `use_fast=False`. This rules out an obvious saved configuration mismatch, but not every historical runtime difference.

Static inspection of the installed **4.57.6 and 4.46.3** BLIP VQA implementations found the same important pattern: the question encoder receives the supplied padding mask, but the decoder cross-attention mask is rebuilt as **all ones** across `question_embeds` positions. Padding positions can therefore remain visible to the decoder. Phase 1 pads each four-example question batch to its longest question; local evaluation processes each question alone.

A CPU tokenizer-only illustration:

| Synthetic question | Individual tokens | Width in two-question batch | Padding |
|---|---:|---:|---:|
| What is this? | 6 | 15 | 9 |
| Please read the text printed on this large sign in the photograph. | 15 | 15 | 0 |

The decoder-mask construction gives all 15 positions a one in the batched case. Although encoder keys are masked, padded query-position hidden states are not guaranteed to be zero. This establishes a real **batching/padding path capable of changing generation**, beyond ordinary GPU numerical differences. It does not prove this caused all historical changes: the original executed library source was not archived and no controlled historical-runtime model comparison was run. The inspected local source corresponds to the recorded version numbers, not proof of an untouched historical installation.

The cause is **partially explained but not causally isolated**. Framework/precision implementation, batching and hardware can still interact. The CPU-YOLO bootstrap issue does not explain the final benchmark, which initialized CUDA before evidence. Existing scoring checks give no evidence that a score-formula change caused the gap.

## Submission interpretation

Keep historical **59.48%** and current **58.72%** as different recorded execution configurations. Final Phase 4 is **58.89%**: it gains 0.1386 points over the shared current Phase 2 control, but does not exceed historical 59.48. Do not silently change masks/batching/decoding after viewing holdout results. Any future reproducibility study should start with independent/development inputs under a separately versioned protocol.

Preserved evidence: `reports/application_validation/run-local-20261014-024220/provenance/`, `BASELINE_PROVENANCE_AUDIT.json`, Phase 4 capture/freeze/scoring files, and `reports/submission_preparation/baseline-20261016-v1/PADDING_METADATA.json`. All original numerical results remain intact.
