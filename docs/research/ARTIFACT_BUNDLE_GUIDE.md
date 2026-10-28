# Verified private runtime bundle — 16 October 2026

Use **`reports/artifact_bundles/VisionBridge-runtime-20261016-v2/`**. Its ZIP contains exactly **21 reviewed runtime files**, not datasets/images or experiment directories. The initial v1 packaging draft omitted YOLO because the older lock did not enumerate that file; it is marked SUPERSEDED and retained rather than deleted. Do not deploy that draft. The frozen model/policy did not change.

Final ZIP: `VisionBridge-runtime.zip`, **1,574,881,084 bytes** (about 1.47 GiB).

SHA256: **`933acfb18a8bfd3afaf4e15dd7fa760491beeaf5ba76216ee764e175b0ace2fc`**

`SHA256SUMS.txt` also covers the external `artifact-manifest.json`. Every payload file has a size/SHA256 entry, and source-code pins ensure the matching project version is used. Entries are sorted with fixed ZIP timestamps and stored without compression for deterministic packaging. Raw JSON bytes/line endings are preserved.

## Structure and scope

```text
VisionBridge-runtime-20261016-v2/
  VisionBridge-runtime.zip
    artifact-manifest.json
    payload/
      models/blip_vizwiz_v2/                  # 7 checkpoint/processor/tokenizer files
      reports/phase2_local_dev_evaluation/
        drive_inputs/split_manifest.json
        .../inputs/run02-dev-only.json
        .../model-cache/easyocr/              # craft and English recognition weights
        .../model-cache/yolov8n.pt
        .../results/expanded_records.json
      reports/phase3_answerability/
        plan-.../PHASE2_FREEZE.json
        run-local-20261012-v1/models/           # heads.joblib and metadata.json
        run-local-20261012-v1/selected-policy.json
      reports/phase4_policy/run-local-20261013-v1/
        selected-policy.json
        development_summary.json
        POLICY_FREEZE.json
        APPLICATION_FREEZE.json
  artifact-manifest.json
  SHA256SUMS.txt
```

The full original split manifest and development records are required by existing integrity checks, so this is a **private bundle containing annotations and development predictions**. It contains no VizWiz images, no holdout prediction table, no personal request images/audio/logs, no virtual environments, and no duplicate original BLIP checkpoint. Copying the manifest for integrity did not evaluate its holdout examples. The preserved original `models/blip_vizwiz` is not needed for explicit frozen v2 inference and was not repacked or modified.

Optional Whisper weights were not found in the configured cache and are not bundled. Microphone transcription needs its first download/cached model; speech output needs an installed Windows SAPI voice. Do not claim fully offline speech input from this core bundle.

## Load on a matching Windows source checkout

1. Use the approved matching source files and Python environments described in README. GPU Torch remains 2.5.1+cu121/torchvision 0.20.1; classifier scikit-learn must be 1.9.1. Do not install/replace packages on the already verified laptop.
2. Copy the ZIP to a private local location. Obtain its expected checksum through a trusted channel, not from an untrusted ZIP itself:
   ```powershell
   Get-FileHash 'C:\path\to\VisionBridge-runtime.zip' -Algorithm SHA256
   ```
3. From the project folder, verify archive bytes, source pins and any existing destination files:
   ```powershell
   .\.venv-gpu\Scripts\python.exe artifact_bundle.py verify --archive 'C:\path\to\VisionBridge-runtime.zip' --expected-sha256 933acfb18a8bfd3afaf4e15dd7fa760491beeaf5ba76216ee764e175b0ace2fc
   ```
4. To restore missing runtime files on a matching checkout:
   ```powershell
   .\.venv-gpu\Scripts\python.exe artifact_bundle.py install --archive 'C:\path\to\VisionBridge-runtime.zip' --expected-sha256 933acfb18a8bfd3afaf4e15dd7fa760491beeaf5ba76216ee764e175b0ace2fc
   .\.venv\Scripts\python.exe launch_phase4.py --run-dir reports/phase4_policy/run-local-20261013-v1
   ```

The installer accepts only the 21 reviewed relative paths, rejects traversal/duplicates/symlink members, verifies every archive member and every existing destination before writing, skips identical files and refuses different existing files. It never overwrites a checkpoint or moves existing data. An interrupted extraction may leave a partial **new** file; do not force-overwrite it—investigate and preserve it before recovery. SHA256 proves consistency with a trusted checksum, not authorship by itself. Only use the locally generated, trusted classifier artifact; joblib must not be loaded from arbitrary sources.

This task built and verified the archive against the current project. It did not run a clean-machine GUI or any holdout evaluation. Keep the bundle outside Git; sharing/uploading it requires your approval.

This is a runtime subset, not the complete historical evaluation archive. Do not invoke training, `phase4_holdout.py` or `run_phase4_completion.py` on a restored runtime-only checkout; it deliberately excludes historical evaluation completion/result files. Use only the documented application launch and safe verification commands.
