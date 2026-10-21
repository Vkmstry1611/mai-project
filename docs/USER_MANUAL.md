# VisionBridge user manual

This is the main detailed user manual for the unified experimental workspace. Developer setup, architecture, testing, known limitations and future improvements are maintained in [README.md](../README.md). An answer or a high OCR score is not a guarantee of correctness.

## Current follow-up and field behavior

- Ask “What was my second question?” to recall exact available wording from this image session. Recall uses a separate last-60-question ledger; only six exchanges are displayed. The existing image-context expiry is 30 minutes. Replacing/clearing the image clears its history.
- “Could you repeat the previous step?” repeats the last selected cooking step. “Go back one step” moves backward. Missing step numbers and unclear boundaries are not reconstructed.
- “What is the serving size?” and quantity/calorie requests need explicit printed labels and a unique supported value. Serving weight is not package weight. Conflicting calorie columns, unknown product identity and unreadable ingredients remain uncertain.
- “How should I take a better picture?” uses advisory capture checks, not BLIP. Better image measurements do not prove the new answer is correct.
- Passage summaries are limited reading-based overviews, not unrestricted explanations. Explicit scene descriptions may include experimental CPU YOLO candidates alongside a successful frozen answer; candidates can be wrong.
- Find **Check microphone access** inside **Speak your question → Microphone help · optional**. Recording starts only when you press Record. No successful physical-device claim is implied by automated payload/Whisper tests.

## Start and stop

Complete the README setup first, then start one server from the project directory:

```powershell
.\.venv\Scripts\python.exe launch_workspace.py --port 7865
```

If an older workspace is running, stop it with Ctrl+C in **its** terminal first. Do not disable a source-integrity error; a running frozen worker must restart after source changes. Open the printed localhost address in Chrome.

The first request can take much longer than a cached follow-up. The existing worker's default response timeout is 240 seconds. Click Analyze Image once and wait instead of queuing duplicates; after a timeout, record the status and restart the owned workspace as directed. This timeout is not a promised response time. Do not create a public share link for private images or recordings.

## Basic workflow

1. Upload an image, or explicitly choose camera capture. Type a question, or open **Speak your question**, choose Record, speak and stop. Completed recording is transcribed automatically; **Transcribe & review** provides an explicit retry/review control. Correct the question before analysis. Recording and transcription do not automatically submit an analysis request.
2. Choose **Analyze Image**. Text appears before one spoken response is attached. If autoplay is blocked, use **Play Answer** without re-analysis. Mute Spoken Answers disables synthesis on later requests and is remembered in the browser tab where storage is available.
3. Keep the same image and ask follow-up questions. You do not need to upload it again. **Recent questions** contains up to six exchanges with separate User/VisionBridge labels and complete answers. Technical Details contains the route, cache status, complete raw evidence, selected/excluded regions, original/corrected boxes and timings.
4. Use **Clear image, question and conversation** to clear active context. It does not delete preserved historical VQA request logs. Closing the server cleans its owned temporary OCR/speech files.

The question box clears when a completed text response arrives. Each typed or
transcribed question is a separate turn; old questions stay in Recent questions,
not in the composer. A different draft typed while processing is retained. Input
or worker failures leave your question available for correction. Review a
microphone transcript before submitting it; transcription replaces the current
question instead of appending history.

## Questions and follow-ups

Responses depend on actual recognized evidence, not the example wording.

| Ask | Expected behavior |
|---|---|
| What is in the image? | Frozen BLIP answer or preserved abstention; unreadable text alone does not block it |
| What does the label say? / Can you read it? | Read reliable visible text; no invented product identity |
| Read the passage | Main passage, excluding surrounding browser clutter |
| Detect every text / Read every visible word | Accepted text across regions; not a guarantee of complete OCR |
| What is website in the image? / Which domain is visible? | Qualified domain/URL evidence, not an assertion of the active website |
| What is the URL? | Observed URL, or a qualified partial answer; no missing-character completion |
| What happened in the passage? | Quote recognized passage evidence rather than invent a story |
| Explain that | Extractive overview only with a supported multi-line passage; otherwise clarify |
| Read the rest | Remaining recognized text within the previous reading scope |
| Tell me the cooking instructions mentioned in the image? | Select layout-supported preparation directions; preserve recognized wording and warn about incomplete text |
| First passage / 1st passage / Read the first one | Select the first stored passage choice in the order announced by a clarification |
| Second passage / Read all passages | Select an existing choice, or read all stored regions in order |
| Continue / Start from the beginning | Read remaining evidence in the active passage, or reread that passage; no invented missing text |
| What should I do to make it clearer? | Task-specific capture tips without new inference |
| What exact product is this? | Explicit printed product/brand field only; abstain without it |
| What is the expiry date? | Require the visible label/value relationship; this does not establish safety |

## Replace an image or clear context

For retakes, choose **Retake / replace image · reuse last question** to restore
your last completed question if the composer is empty, then replace the image.
An existing draft is preserved. Image-specific context and passage choices are
invalidated; the previous quality snapshot supports a before/after comparison.
Better quality or more OCR detections does not prove a correct answer. If OCR
misses step numbers or words, the assistant will not fill them in. A partial
cooking transcription is not a verified complete set of preparation directions.

## Advanced controls

Advanced / Developer retains **Visual Q&A**, **Experimental OCR Reading**, **Full Text Reading**, manual rotation and optional Automatic/EasyOCR/RapidOCR selection. The engine override applies to this local workspace, while conversation history is browser-scoped. The original conservative fragment reader remains available with its unchanged 0.90 policy. Original launchers are unchanged; see [installation](../README.md#installation-and-model-setup) and [architecture](../README.md#system-architecture).

Manual checks still needed: physical microphone/camera, audible speech, autoplay fallback, keyboard navigation, large text/zoom, Copy Text, screen-reader labels/announcements, replacement upload and clearing while a request is running. Automated API checks do not certify these interactions.

## Experimental image-purpose descriptions from existing text evidence

After reading a label or document, ask **“What is the image about?”**. The frozen
VQA model still runs first (or reuses its unchanged prediction cache). If it
abstains and the current image/engine already has sufficiently coherent OCR,
the assistant can describe the **recognized printed content** with short source
quotations. It does not claim that mentioned objects are visible in the scene.

This fallback is deliberately **cache-only**. On a fresh session, read the label
first; without compatible cached evidence the original abstention remains. It
does not start expensive OCR for general questions. Object/colour questions,
the frozen VQA result and manual Visual Q&A mode retain their existing behavior. A separate explicitly experimental scene overview may supplement successful VQA for a detailed-description request with detector candidates; it does not establish that every candidate is correct.
Small signs, incidental brand text, conflicting passages and unreliable OCR may
not qualify. Speech is generated from the same final text as usual.

Technical Details identifies `answer_source=ocr_content_description`, supporting
OCR IDs and the original `frozen_vqa_result`. Frozen gate probabilities describe
the original VQA candidate, not the new experimental content description.


## Speech, failures and advanced rotation

Spoken answers are enabled by default. Text appears first, then audio is attached. If autoplay is blocked, use the output player's Play control or **Play Answer** without analyzing again. **Mute Spoken Answers** stops playback and disables generation for later requests; the preference is kept in the current browser session where storage is available. Speech generation failure must leave a valid text answer visible. Long OCR readings may be synthesized in chunks; replay uses the existing audio.

The microphone diagnostic is under **Speak your question → Microphone help · optional → Check microphone access**. It checks access and releases tracks; it does not start a saved recording. If no device is available, type the question or upload a WAV recording. Check browser permission/device selection before retrying. Do not repeatedly resubmit analysis when only audio playback is blocked.

Keep **VisionBridge Assistant** selected for automatic routing. Manual modes are optional developer tests. In a manual OCR mode, choose a counterclockwise rotation only when needed; rotation does not alter frozen Visual Q&A. Automatic experimental OCR considers orientation using layout/recognition evidence and can still be uncertain. EasyOCR/RapidOCR engine overrides change extraction configuration and invalidate incompatible cached evidence. The local engine setting is shared within this single-user workspace, not a private per-user server setting.

If an input is missing, supply the image/question and retry. If analysis fails, keep the diagnostic reference and check the server log; do not treat the failure as a verified model abstention. Stale audio should not be reused as the answer to a failed request. Do not share private images, OCR or recordings in bug reports without review. A source-change error requires stopping/restarting the matching workspace, never bypassing its guard.

## Manual accessibility checks

These steps consolidate the former physical walkthrough for the current workspace. They are **instructions, not newly passed tests**. Record Pass / Fail / Not supported / Not tested, browser/version, UI variant and observed behavior. Earlier results remain in the [manual checklist](../MANUAL_TEST_CHECKLIST.md). Backend tests cannot establish real-user accessibility.

### Microphone and editable transcription

1. Use a real microphone in Chrome on localhost, open Speak your question and choose Record. Allow only the intended local browser permission.
2. Say “What does this sign say?”, stop, and check for one editable transcription. Whisper may need its first model download. Use Transcribe & review if a retry is needed.
3. Edit it to “What word is written on this sign?” before Analyze Image. Verify the reviewed wording is submitted, not the earlier transcript.
4. Try no recording and a silent recording; expect a useful status with typed input still available. Deny microphone permission and confirm typing still works.
5. Type a newer draft while transcription is processing; verify the draft is not overwritten. Test clearing and recording again for duplicate transcription.

### Audible output and mute

1. Ask once with a permitted image and leave mute off. Expect visible text and a nonempty audio player when the Windows voice is available.
2. Listen using Play Answer. Confirm words match the displayed answer or uncertainty, including multiline OCR. Test pause and replay without rerunning inference.
3. Enable mute and submit another question; verify no new speech is generated. Unmute for a subsequent request.
4. Confirm browser-blocked autoplay can be handled using Play Answer. A WAV file existing alone does not verify audible correctness.

### Keyboard and screen reader

1. Put the mouse aside. Use Tab/Shift+Tab through image upload, question, microphone controls, Analyze Image, answer, audio, history and evidence. Use Enter/Space where appropriate.
2. Expect visible focus, useful labels, sensible order and no keyboard trap. Confirm invalid-input errors can be understood and focus can return to the input.
3. With your screen reader, read controls, User/VisionBridge exchanges and technical evidence text alternatives. Windows Narrator can be toggled with Win+Ctrl+Enter; do not leave it enabled unintentionally.
4. Submit a question and check whether completion/error updates are announced. If not, test navigating manually to the answer and record the limitation. Live announcements are not guaranteed.
5. Verify the answer is understandable without interpreting colour or bounding-box overlays. Candidate evidence must not be announced as guaranteed correctness. These checks do not certify WCAG compliance.

### Camera, zoom, retake and privacy

1. Explicitly activate camera capture if supported; verify a preview using a consented object/sign. Otherwise record Not supported and use upload. The camera must not activate automatically.
2. At 200% zoom and a narrower browser width, verify question, analysis, response, history and Advanced controls remain usable without horizontal overflow. Test Larger input & answer text and the browser's supported appearance settings.
3. Inspect contrast with available browser tools rather than relying only on visual judgment.
4. Use Retake / replace image, replace the photo and re-analyze. Check the intended question is preserved, old-image context is cleared, and better image metrics are not presented as proof of correctness.
5. Clear the session while idle and test the documented behavior during processing. Do not rename checkpoints or modify thresholds to simulate failures; use existing regression tests for artifact failures.

Report precise failing controls and expected/actual behavior; never mark a physical test passed simply because automated checks succeeded. Keep photos/audio and request logs private. Clear resets active context but does not erase preserved frozen request logs under ignored reports; closing the server cleans its owned temporary OCR/speech files.
