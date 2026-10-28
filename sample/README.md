# Small manual test set

These are historical demonstration inputs, not the original VizWiz splits. In the final Phase 4 UI, upload a photo and ask a question: evidence is triggered automatically by question intent, and the frozen candidate/gates can return an answer or abstain. The final UI does not expose the historical optional-processing/model-selection checkboxes described below. Use synthetic signs or consented photos for current submission checks, and follow the root `MANUAL_TEST_CHECKLIST.md`. The local reference answers here do not establish model accuracy.

Four downloaded images and 16 **locally authored** questions for VisionBridge. These are demonstration samples, not a VizWiz dataset download or official evaluation annotations. Reference answers were written after visually inspecting the downloaded images; they are not model predictions. Equivalent wording is acceptable.

## Historical application instructions

Open VisionBridge, upload one of the images from this folder and paste a question below. Leave optional processing off first. Enable YOLO for objects, EasyOCR for the stop sign, or CLIP for diagnostic features. The current BLIP baseline does not use the optional evidence to generate its answer.

| Image | Question | Manual reference |
|---|---|---|
| cats.png | What animals are in the image? | cats |
| cats.png | How many cats are visible? | two |
| cats.png | What color is the couch? | pink |
| cats.png | What objects are next to the cats? | remote controls |
| bus.jpg | What large vehicle is in the image? | bus |
| bus.jpg | What color is the lower side of the bus? | blue |
| bus.jpg | Are there people in front of the bus? | yes |
| people.jpg | How many people are in the foreground? | two |
| people.jpg | What is the person on the left doing with his arm? | pointing |
| people.jpg | What color are their jackets? | black |
| stop_sign.png | What word is written on the sign? | STOP |
| stop_sign.png | What color is the sign? | red |
| stop_sign.png | What shape is the sign? | octagon |

Questions to expose unsupported guesses:

- cats.png: What are the names of these cats?
- bus.jpg: At what exact time will this bus arrive at its next stop?
- stop_sign.png: What is printed on the back of this sign?

These last three cannot be answered from the pixels. The baseline may hallucinate; its existing reliability disclaimer is not a trained abstention gate. `questions.json` includes categories, references and expected behavior for these cases.

## Sources

- bus.jpg and people.jpg: [Ultralytics demonstration assets](https://github.com/ultralytics/assets/tree/main/im), originals named bus.jpg and zidane.jpg. Retained unmodified; the source repository's license and contributor information apply.
- cats.png: [Hugging Face documentation sample](https://huggingface.co/datasets/huggingface/documentation-images/blob/main/coco_sample.png), original coco_sample.png (COCO example). Original rights remain with their owners; source attribution is retained here.
- stop_sign.png: [Wikimedia Commons, Stop sign.png](https://commons.wikimedia.org/wiki/File:Stop_sign.png), uploaded originally by Denelson83; source marks the graphic public domain. Retained unmodified.

Downloaded on 10 October 2026. Only four individual image files were fetched (about 867 KiB total). `download_samples.py` records the direct URLs and skips existing files.

Source verification on 16 October 2026 found that `vqa-examples1.jpg` and `vqa-examples1 (1).jpg` are exact duplicates of the [official VizWiz VQA illustration](https://vizwiz.org/wp-content/uploads/2019/05/vqa-examples1.jpg), each SHA256 `6d98ab1b12ca68c3d64efe992b0b669c9b0729ad8035604965127fa94b018a36`. The earlier small-crop description was incorrect. Attribute the VizWiz project/Gurari et al. and link the [official VQA page](https://vizwiz.org/tasks-and-datasets/vqa/), which labels the work CC BY 4.0. No changes were made to either image. They illustrate VQA; they are not inputs for final validation or a new evaluation.

`money.png` still has unverified provenance. Do not claim an aggregator's “free PNG” label establishes permission. Prefer an original synthetic price card for OCR demos, a consented own photograph, or the already sourced stop-sign image. Removing the unknown image and redundant JPEG from Git tracking is recommended only after approval; keep their local copies. See [SAMPLE_PROVENANCE.md](../docs/research/SAMPLE_PROVENANCE.md).
