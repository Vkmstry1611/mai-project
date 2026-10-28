# VisionBridge manual application testing checklist

## Speech-by-default and Full Text Reading — 20 October 2026, human checks pending

Automated API checks confirm text/audio generation, not browser playback or accessibility. Do not mark the following passed without a person testing them.

- [ ] Launch `launch_workspace.py` or `python -m visionbridge.launch_full_text_workspace`. Confirm Mute Spoken Answers starts unchecked and no Generate Spoken Answer checkbox is needed.
- [ ] Analyze one visual question and one reading request. Text should appear first and speech should become available without a second analysis.
- [ ] After clicking Analyze, check whether Chrome autoplays. If blocked, use Play Answer/native Play and confirm the existing audio plays without re-analysis.
- [ ] Mute during playback; confirm audio stops. Submit while muted; confirm no new speech is generated. Unmute and submit another request.
- [ ] Reload the same tab and confirm its mute preference persists. Test storage-blocked/private-browser behavior separately.
- [ ] Listen to a long transcription; verify chunk order, pauses, completeness and absence of duplicated speech.
- [ ] Test clarification, blank-image abstention, and a labeled field. Confirm uncertainty and safety limitations are understandable when spoken.
- [ ] Test physical microphone recording and transcription review. Automated generated-audio transcription does not verify the microphone.
- [ ] Copy a long answer, check line breaks, and paste into a text editor. If clipboard permission is blocked, use text selection and Ctrl+C.
- [ ] Replace a poor photo, keep the question, and re-analyze. Check before/after metrics and OCR counts without assuming the answer became correct.
- [ ] Reset and confirm image, question and comparison are cleared. A second browser session must not inherit the first session's retake state.
- [ ] Test keyboard-only navigation, visible focus, 200% zoom and NVDA labels/live announcements throughout analysis and speech generation.
- [ ] Test real pages with columns, sideways text, glare, mixed directions, small print and faint labels. Record failures; the clean synthetic paragraph result is not a general OCR accuracy claim.

Human tester / browser / date / results: ___________________________________

## New workspace launcher — browser checks pending

Unified Assistant update: confirm there is no required main-page mode decision. With Advanced / Developer closed, try a visual question, a literal reading request, an expiry-date request and “Help”. Expect automatic VQA, automatic OCR, an explicit field-extraction limitation and clarification respectively. Check task explanations in the advanced evidence panel. Open Advanced / Developer only to test the preserved manual overrides; return to Assistant afterward. None of these browser checks is pre-marked passed.

The separate redesigned launcher is `launch_workspace.py`. Its Chrome checklist and expected outcomes are in `WORKSPACE_UI_GUIDE.md`. Construction/API/controller validation does not mark its visual layout, microphone interaction, audible playback, larger-text toggle or accessibility passed. Record those results manually; do not transfer earlier launcher confirmations automatically.

## Latest confirmation: updated grouping mode (reported 19 October 2026)

| Input | User-confirmed observation | Result | Audio generation | Audible playback |
|---|---|---|---|---|
| Wine bottle | THREE, Cabernet, California; about 40 seconds end to end | Partial success | Confirmed | Not separately confirmed |
| Upright potato container | Easy, Sauteed, Bacon and instruction fragments; about 40 seconds end to end | Partial success | Confirmed | Not separately confirmed |
| HELLO | Prior output `Recognized text fragments: HELLO.` | PASS, carried forward from earlier test | Not specified for HELLO | Not specified |

The bottle/container results confirm manual use of the updated grouping mode. Timing is approximate and user-reported. Do not mark full transcription, screen-reader accessibility or audio playback passed from these observations. HELLO has not been reported as rerun in this latest update. Exact browser/source snapshot remains to be recorded.

- [x] User confirms updated-mode bottle/container tests; both remain partial successes.
- [x] User confirms spoken-answer audio generation for those two tests.
- [ ] Separately verify audible playback in the updated experimental mode.
- [ ] Manually verify the new optional persistent-worker launcher; earlier confirmations do not automatically pass a new launcher.
- [ ] Run `.\.venv\Scripts\python.exe -m visionbridge.launch_persistent_demo`, repeat reading/frozen-mode inputs, then use Ctrl+C in its terminal and confirm the worker stops. Cached requests should reuse models; a new photo can still require CPU OCR.

## Experimental OCR results reported 19 October 2026

| Manual input | User-confirmed result | Notes |
|---|---|---|
| Wine bottle | Partial success | Exact new output not supplied; do not mark complete label reading passed. |
| Upright potato container | Partial success | Exact new output not supplied; do not mark complete label reading passed. |
| Simple HELLO image | **PASS** | `Recognized text fragments: HELLO.` |

These earlier results apply to the previously tested experimental application. Browser/source snapshot and actual test date were not recorded. The subsequent updated-grouping confirmation is recorded above; the exact HELLO output is also covered by an automated regression. Persistent-worker work is now authorized as a separate opt-in prototype. Existing unconfirmed accessibility checks remain untested.

- [ ] Launch `python -m visionbridge.launch_reading_demo`, enable experimental reading, and repeat HELLO; expect the exact same sentence.
- [ ] Repeat the bottle and upright container; record the complete answer, omitted text, rotation and evidence panel. Partial success must not be reported as complete transcription.
- [ ] Test separate lines/columns and an unclear word between readable words; expect separated regions or an explicit gap, never invented text.
- [ ] Disable experimental mode; verify the familiar frozen answer/details, with experimental rotation ignored.
- [ ] Check speech playback in the updated launcher and record browser/source version.

Tester: __________  Date: __________  Windows/browser: __________

User-reported initial testing (16 October 2026): image upload, microphone recording/playback, Faster Whisper transcription, BLIP inference, OCR extraction and TTS/spoken playback work in the previously tested application. Exact browser/source snapshot was not recorded. This confirms those reported observations only; accessibility and the separate experimental reading launcher are not marked passed. Reported wine/container answer failures remain model limitations to record, not successful answer-quality tests.
Commit/source snapshot: __________  Frozen run: `run-local-20261013-v1`

Record Pass / Fail / Not supported / Not tested for every row. Do not mark a feature passed just because its control is visible. Use newly created synthetic signs or consented personal photos; never use training/development/holdout examples for this manual check. Do not change thresholds or checkpoints in response to a result.

See [the user manual's accessibility checks](../USER_MANUAL.md#manual-accessibility-checks) for exact microphone, playback, keyboard, screen-reader and camera steps. All physical results remain Not tested until confirmed by the tester.

## Start and record the environment

- [ ] Run `.\.venv\Scripts\python.exe launch_phase4.py --run-dir reports/phase4_policy/run-local-20261013-v1`.
- [ ] Optionally test the presentation-only wrapper with `.\.venv\Scripts\python.exe launch_submission.py`. Identify which UI was tested; it calls the same frozen controller.
- [ ] Confirm localhost access, no public share link, CUDA BLIP, CPU OCR/YOLO, and the expected frozen run.
- [ ] Note that each UI request reloads model work; allow approximately two minutes. Record actual latency and errors rather than promise a fixed time.
- [ ] Use speech off initially. Do not reinstall packages in the verified environments.

## Photos and questions

| Check | Expected behavior | Result / observations |
|---|---|---|
| Upload a PNG or JPEG | Preview appears; image can be replaced | |
| Upload a portrait/phone image with EXIF rotation | Preview and evidence boxes remain understandable after normalization | |
| Ask a simple question about a consented photo | Answer or clear abstention appears; technical panel preserves BLIP candidate and final answer | |
| Ask a second question about the same image | UI updates without showing the first answer as the new result | |
| Replace the image and repeat | Correct new image reaches the request; no stale evidence | |
| Camera capture, if browser/hardware supports it | Permission requested; capture previews correctly; upload remains an alternative | |
| Two requests submitted close together | Requests queue; responses are associated with the right question | |

An incorrect generated answer is an observed model limitation, not automatically a UI failure. Record it without tuning the frozen system.

## OCR and evidence

| Check | Expected behavior | Result / observations |
|---|---|---|
| Clear single WELCOME sign; “What does this sign say?” | OCR text and boxes appear, source/status/IDs are recorded; selected evidence can support the answer | |
| Separate STOP and NOTICE signs; same question | Distinct OCR candidates do not cause an arbitrary OCR override; inspect retention reason | |
| Blurred or blank sign | Missing/weak evidence is reported; no invented replacement OCR text | |
| Synthetic price/brand/date labels | Routing and decision reason are visible; if ambiguous, no blind OCR override | |
| Object mentioned but absent from image | YOLO may return empty detection; abstention/decision must not claim non-detection proves absence | |
| Evidence boxes versus text | Text, IDs and selected/supporting roles agree with the technical panel | |

Confidence is an extraction score, not calibrated answer correctness. OCR-supported does not mean the whole system is universally grounded.

## Invalid inputs and failure handling

- [ ] Ask without an image: clear validation error, no stale answer presented as new.
- [ ] Ask with an empty/whitespace question: clear validation error.
- [ ] Attempt a non-image file: upload rejects it or explains the failure; app remains usable.
- [ ] Remove microphone permission using browser controls: typing still works.
- [ ] Interrupt/cancel a request or close/reopen the page: record behavior and any background worker; do not delete artifacts or stop unrelated processes.
- [ ] Inspect logs after a failed request. Do not deliberately rename/delete checkpoints to simulate failure; missing-artifact behavior already has regression tests.
- [ ] If a natural timeout occurs, record it. The frozen worker has a 240-second timeout; do not force an expensive hanging model run.

## Microphone and speech — conditional features

- [ ] Browser microphone permission and an actual microphone work; otherwise mark **Not supported** and use typed input.
- [ ] Record “What does this sign say?”, press Transcribe and verify editable question text.
- [ ] Correct a transcription error before asking; the corrected text reaches the model.
- [ ] Silence/no recording yields an informative message and does not disable typed questions.
- [ ] Whisper `tiny.en` uses CPU. First use requires cached weights or a download; offline transcription is only supported after weights are cached.
- [ ] Enable spoken answer; verify a nonempty WAV and audible Windows voice. Installed SAPI voice and pyttsx3 support are required.
- [ ] Play, pause and replay audio. The optional submission wrapper deliberately does not autoplay.
- [ ] Speech failure leaves the text answer available and gives a status message.
- [ ] A microphone round trip and audible speech require human verification; automated UI construction does not prove either works.

## Accessibility and usability

- [ ] Complete upload → question → Ask → answer using keyboard only, with no keyboard trap.
- [ ] Tab order and visible focus are sensible; Enter/Space operate buttons where supported.
- [ ] At 200% browser zoom and a narrow window, text and controls remain usable without clipping.
- [ ] Check light/dark themes and contrast using an appropriate browser tool; do not claim WCAG compliance from visual inspection alone.
- [ ] Test with a real screen reader, e.g. NVDA: labels, question, answer, error and evidence text are understandable.
- [ ] Verify whether newly completed answers/errors are announced. Automatic live announcements are not guaranteed by this implementation.
- [ ] Check that evidence boxes have a usable textual alternative in `launch_submission.py`; colour is not the only signal.
- [ ] Audio is optional and can be paused; typed questions work without hearing/speech.
- [ ] Ask a consenting intended user to review the workflow before claiming accessibility effectiveness.

## Finish and submission decision

- [ ] Record unresolved failures, input type, observed response, timing and logs path without copying sensitive photos into Git.
- [ ] Confirm original checkpoints and policy hashes are unchanged; no training/holdout evaluation took place.
- [ ] Agree on retention of local request images/audio/logs. They remain local and are excluded from the proposed commit.
- [ ] Explain the historical 59.48% versus current 58.72% baseline discrepancy during the presentation.

Blocking issues / owner / action: __________________________________________

Manual sign-off for supervised demo: __________  Date: __________

## Optional automatic-orientation workspace — manual checks pending

Use the [user manual's advanced controls](../USER_MANUAL.md#advanced-controls) for the current experimental workspace. The retained `python -m visionbridge.launch_orientation_workspace --port 7866` is the earlier separate orientation demo. These checks are NOT marked passed by synthetic OCR or backend/API tests.

- [ ] Upload upright, sideways and upside-down HELLO images; ask “What is written in the image?” Confirm the answer and displayed corrected preview agree.
- [ ] Inspect the automatic correction angle and original/corrected boxes in Advanced details. Confirm selected evidence corresponds to the visible words.
- [ ] Repeat the saved sideways comic. Expect incomplete fragments and possible recognition errors; do not treat it as a complete transcript.
- [ ] Try mixed directions and unreadable text. Confirm the fallback explains how to use manual rotation or upload a clearer upright photo.
- [ ] Repeat the same image/configuration and check that cache reuse preserves the answer and support IDs.
- [ ] Select manual Experimental OCR Reading in Advanced; confirm manual rotation remains available and bypasses automatic orientation.
- [ ] Generate speech and press Play; record audible playback separately from audio-file generation.
- [ ] Check keyboard focus and screen-reader announcements while the longer orientation search runs. Do not claim browser accessibility from API results.
- [ ] Stop only the new workspace terminal with Ctrl+C and confirm its worker exits without closing another running application.

Tester / date / observations: __________________________________________
# Question-aware reading update — human checks pending

- [ ] Stop the old workspace with Ctrl+C in its terminal, then restart using the current guide. Do not start a duplicate GPU worker.
- [ ] Wine: ask “What is written?” Expect a partial label reading without decoration glyphs or an invented alcohol line. Listen to the generated audio; generation alone is not playback verification.
- [ ] UFO: ask “What is written?” Expect the main passage without browser menus/URLs; inspect retained errors rather than assuming a full correct transcript.
- [ ] Ask “Read every visible word.” Confirm accepted surrounding interface text becomes available in labeled groups.
- [ ] Ask “What website is shown?” Expect a unique recognized address or an honest insufficiency message.
- [ ] Ask for a passage summary and a story question. Confirm extractive/quoted wording is identified honestly; unsupported semantic questions should abstain.
- [ ] Open Technical Details. Check selected/excluded candidates, source engine, original/corrected boxes, region scores and raw transcription.
- [ ] In Advanced / Developer, test Automatic, EasyOCR and RapidOCR on HELLO. Also test the original Experimental OCR Reading override and its unchanged fragment-style output.
- [ ] Confirm speech starts without a checkbox, Play Answer works if autoplay is blocked, mute persists for the tab, and a speech failure leaves text available.
- [ ] Test physical microphone capture, Whisper review, keyboard focus, 200% zoom, large text, NVDA announcements, Copy Text and replacement-image question preservation.

Do not mark these passed without human confirmation. Earlier user-confirmed manual results below remain unchanged.

# Unified retrieval and follow-up conversations — pending human checks

- [ ] Stop the old workspace with Ctrl+C in its terminal. Restart using the current user guide; do not disable frozen source guards.
- [ ] Upload the saved UFO screenshot once. Ask “Detect every text,” then “what is website in the image?”, “Read the passage,” “What is the URL?” and “What happened in the passage?” Do not upload again. Expect independent selections from the same raw evidence.
- [ ] Confirm the website response qualifies the visible domain/image-source URL and does not claim certainty about the current website. Main reading should omit browser clutter.
- [ ] After a multi-line passage, ask “Explain that” and “Read the rest.” Expect an explicitly extractive overview or remaining recognized passage, not invented details. After only HELLO, “Explain that” should clarify.
- [ ] On the exact blurry container photo, ask a general visual question, label reading and exact product identity. Frozen abstention must remain visible in Technical Details; no person/container/brand description should be invented.
- [ ] Ask “What should I do to make it clearer?” Check task-specific, non-blocking guidance. Replace the photo without changing the question; check that image-specific history resets and the quality comparison remains appropriately qualified.
- [ ] Use a truly sharper photo of the **same** container when available. The synthetic replacement API test did not establish real-photo recognition improvement.
- [ ] Check Recent questions, then open a separate private browser session. Its history must start empty. Clear image, question and conversation; ambiguous follow-ups must not retrieve prior context.
- [ ] Confirm text and generated audio convey the same answer, no duplicate speech occurs, mute persists appropriately, and Play Answer works without rerunning analysis when autoplay is blocked.
- [ ] Test physical microphone/camera permissions, Whisper review, keyboard-only navigation, visible focus, large text/200% zoom, Copy Text and NVDA announcements.
- [ ] Test clearing/replacing an image while processing. Confirm the final displayed result belongs to the current input and no old context reappears.
- [ ] Review private legacy request logs before publication. Clearing active context does not delete preserved report artifacts.

No item above is marked passed by automated tests. Preserve previous user-confirmed results separately.

## Cooking sequences, passage choices and composer — new manual checks

The user confirmed that the **previous** callback crash is resolved and website
retrieval works. This confirmation does not establish that the new features below
have passed browser or accessibility testing. Automated HTTP/CPU OCR/TTS checks
are recorded separately in `reports/reading_sequences/run-20261020-v1/`.

- [ ] Hard-refresh the restarted workspace in Chrome. Upload the exact development
  fixture `VizWiz_val_00002603.jpg`. Ask “Tell me the cooking instructions mentioned
  in the image?” Expect a partial reading beginning with recognized preparation
  text, not an unsupported scene answer or unnecessary passage clarification.
- [ ] Confirm missing circled numbers and unreadable words are not invented.
  “Read every visible word” should also include recognized heading/ingredient
  regions. Technical Details should retain the original raw OCR and boxes.
- [ ] Watch the question field as the text answer arrives, before long speech
  synthesis finishes. It should clear; Recent questions should retain the turn.
- [ ] Type a different next question while processing. Confirm the newer draft is
  not erased or combined with the previous question. Rapid double-clicks should
  not queue duplicate analysis of the same pending click.
- [ ] Record and transcribe a new question. Review and edit it; confirm it replaces
  the current composer text rather than adding prior conversation history.
- [ ] Use a genuinely ambiguous two-column document (the synthetic test image is
  saved under `api-v2/two-passages.png`). Check the announced passage order.
  “Start with first passage,” “1st passage” and “Read the first one” should select
  the first choice without repeating the original clarification.
- [ ] Ask “Second passage,” “Continue,” “Read the rest,” “Start from the beginning”
  and “Read all passages.” Expect the selected scope or an honest no-more-text
  response. The assistant must not silently switch to an unrelated region.
- [ ] Confirm text and generated audio agree, autoplay/Play Answer works under
  your browser settings, mute works and no duplicate speech is heard.
- [ ] Use Retake with an empty question field. The previous question should be
  restored. A newer typed draft must remain unchanged. Replace the image and
  check that old passage choices cannot be retrieved from the new photograph.
- [ ] Clear the session and open a separate private browser session. Ordinal
  references must not access the old conversation's candidates or evidence.
- [ ] Repeat keyboard-only, visible-focus, large-text/200% zoom and NVDA tests;
  confirm question clearing is announced without stealing focus or activating
  the microphone/camera. Mark results only after physical testing.

Browser automation was attempted for this update, but the in-app browser webview
did not attach. None of these manual items is marked passed by that attempt.

## Cached OCR image-purpose fallback — pending manual checks (21 October 2026)

- [ ] Hard-refresh the restarted workspace. Read the cooking label, then ask
  “What is the image about?” Expect an explicitly OCR-grounded description with
  actual quoted preparation text; no claim that the named foods are visible.
- [ ] Confirm text and spoken output match, and Play Answer works if autoplay is
  blocked. Automated checks confirm synthesis input/file generation, not hearing.
- [ ] In Technical Details, inspect `ocr_content_description`, supporting IDs,
  cache-only status and the unchanged original frozen answer/gates nested under
  `frozen_vqa_result`.
- [ ] Repeat in a fresh session without reading first: expect the original VQA
  behavior with no automatic OCR extraction. General object/colour questions and
  Advanced manual Visual Q&A must not acquire the new content fallback.
- [ ] Replace the image or change the OCR engine. Old incompatible evidence must
  not be used. Test a photo with only incidental words and an unreadable image;
  neither should produce a guessed purpose or object identity.

These items remain unverified in a browser until the user confirms them.

## Step navigation, ingredient retrieval and speech input — 21 October 2026

Automated API tests cover navigation, cached OCR, TTS and real Whisper. The
in-app browser successfully uploaded a generated WAV and automatically populated
the editable question. Its device selector reported **No microphone found**;
physical Chrome recording and screen-reader behavior are not verified.

- [ ] Hard-refresh `http://127.0.0.1:7865/` in Chrome. Open **Speak your question**.
  At the site-controls icon beside the URL, allow Microphone, select the intended
  input device, and reload if Chrome requests it. If capture still fails, check
  Windows Settings → Privacy & security → Microphone and the selected input's
  level in Sound settings. Do not change permissions for unrelated sites.
- [ ] Choose Record, say “What is written in this image?”, and Stop. Expect
  Recording → Processing speech → Transcription ready. Transcription starts
  automatically; **Transcribe & review** is a retry, not a required extra step.
- [ ] Confirm the recognized text replaces the old question rather than appends
  to it. Edit it before selecting Analyze. Type a newer draft while Whisper is
  processing: the newer draft should remain, with the transcript separately
  available below the recording controls.
- [ ] Test denied permission, disconnected microphone and silence. Expect a
  browser capture error or an explicit empty/inaudible-recording explanation;
  typing/uploading audio remains available. Browser permission errors occur
  before an audio payload reaches Python; do not report these as Whisper errors.
- [ ] Upload the cooking image and ask first, second, third, previous, repeat
  that step, next. Expect blocks 1 → 2 → 3 → 2 → 2 → 3, with no invented printed
  numbers. At the first/last block, previous/next must explain the boundary.
- [ ] Ask the image purpose between step requests. The selected step must
  remain available for subsequent relative navigation. Test “Continue from here,”
  “Read the remaining steps,” and “Read all steps.”
- [ ] Ask “What ingredients are mentioned?”, “Does it mention milk?”, and “Is
  pasta mentioned?” Expect preparation mentions/literal OCR quotations, not a
  claim of the complete packaged-product ingredient list. Unknown mentions must
  state that incomplete OCR cannot establish absence or food safety.
- [ ] Confirm spoken output contains only the selected answer. Starting a new
  microphone recording should pause the old answer audio. Microphone/camera
  must never start automatically.
- [ ] Clear the session during speech processing and after several step turns.
  Old text/audio input and step references must not reappear. Repeat with
  keyboard-only controls and NVDA, including the error and transcription states.

All checkboxes above remain pending human confirmation. Do not mark the project
fully Git-ready until physical microphone and accessibility checks are resolved.

# Consolidated assistant checks — 21 October 2026

These are pending human checks unless explicitly confirmed elsewhere. Automated HTTP checks do not verify microphone capture, speech intelligibility or screen-reader usability.

- [ ] With a real microphone in Chrome, record and stop a question. Verify one editable transcription appears without replacing a newer typed draft.
- [ ] Confirm the compact **Microphone help · optional** accordion opens and its diagnostic releases microphone access after checking.
- [ ] Ask at least eight questions, then “What was my second question?” Check exact original wording. Question recall retains the last 60 non-recall questions for this image session; the displayed history retains six exchanges.
- [ ] Read the third cooking step, then “Could you repeat the previous step?” Expect repetition of the last selected step. “Go back one step” should navigate backward instead.
- [ ] Ask “What is the passage about?” Check that any overview is grounded in recognized text; it remains a limited reading-based overview, not unrestricted semantic explanation.
- [ ] On a product label, compare serving size against net quantity. Missing package weight must not be replaced with serving weight. Multiple calorie values must trigger uncertainty.
- [ ] Ask how to heat a label with only a heading and percentages recognized. Expect uncertainty, not a recipe or a list of nutrition percentages.
- [ ] Ask for a scene description. If detector candidates are shown, verify the experimental label and that the original VQA result remains in Technical Details. Simple colour/presence questions retain frozen VQA behavior.
- [ ] Verify matching spoken text, mute, Play Answer, keyboard focus, 200% zoom, small-screen layout and screen-reader reading order.
- [ ] Replace the image and reset the session; confirm previous-image questions and evidence cannot be recalled.
