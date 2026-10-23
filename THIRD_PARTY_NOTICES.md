# Public runtime asset notices

These notices identify reviewed sources; they do not choose a license for the team's own project or certify permission for every derivative asset. Team approval is required before uploading the proposed fine-tuned checkpoint and classifier. Existing sample provenance decisions remain unchanged.

| Component | Exact distribution / source | Review |
|---|---|---|
| BLIP v2 | Existing VisionBridge fine-tuned Salesforce/blip-vqa-base export; SHA256 `9b6f315df6b4d81c1fc85a41e1882fa15279421317636f1bed20c33d88843c8c` | Base model card declares BSD-3-Clause. Preserve Salesforce copyright/license; confirm team authority for the fine-tuned derivative before upload. |
| Classifier heads | Existing two calibrated logistic heads; SHA256 `9ac285dcd45ef7b67ec91f17ca10a7dae4f50289574fbf0485d40876fc6417ae` | Locally trained, trusted joblib only. Review found parameters/scaler statistics/calibration, not raw training rows or personal strings. Pickle is executable: verify before loading. This is not a formal privacy or memorization guarantee. |
| VizWiz | [Official dataset page](https://vizwiz.org/tasks-and-datasets/vqa/), Gurari et al. | Page declares CC BY 4.0. No dataset examples/annotations/predictions distributed here. Derived-model terms and team licensing still require review. |
| Official VQA normalization tables | Original API URL and source hash in normalization.json | Retained exactly for deployment parity. [VQA terms](https://visualqa.org/terms.html) include a software redistribution notice; see third_party/VQA-SOFTWARE-NOTICE.txt. |
| EasyOCR CRAFT / English G2 | Official JaidedAI GitHub release ZIPs, exact payload hashes in manifest | EasyOCR uses Apache-2.0; CRAFT attribution and model-specific sources must be retained. Download from upstream rather than republishing weight files here. |
| RapidOCR | rapidocr-onnxruntime 1.4.4 plus SWHL/RapidOCR English PP-OCRv3 recognizer | Package declares Apache-2.0. Upstream download is pinned by payload SHA256. Package detector/classifier assets remain supplied by its wheel. |
| YOLOv8n | Existing hash-locked model from Ultralytics assets v8.3.0; ultralytics 8.3.40 | AGPL-3.0 notice retained. Downloading rather than bundling does not waive applicable obligations. Team must review project licensing before merging/distribution. |
| Faster Whisper tiny.en | Systran/faster-whisper-tiny.en at revision `0d3d19a32d3338f10357c0889762bd8d64bbdeba` | Model card declares MIT. Faster Whisper MIT notice retained; original OpenAI Whisper MIT notice retained separately. |
| Windows SAPI | Installed operating-system voices | Not redistributed. Availability/voice terms depend on the Windows installation. |

Upstream notices are in `third_party/`. The project still has no newly selected root license. No rights are inferred from a checksum, a public download URL or this notice table. No claim is made that learned weights cannot memorize training information. The full private ZIP must never be uploaded.
