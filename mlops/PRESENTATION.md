# Five minute MLOps presentation

1. **Goal and scope (30 seconds).** VisionBridge is an accessible visual question answering project. The public page is the MAI branch workspace, with image upload/camera, follow-up questions, voice controls, and two cloud model choices. Its default is the DVC-selected BLIP champion on EC2; Qwen3-VL 2B is an optional choice for broader questions. The private frozen Phase 4 OCR/gating bundle is not part of this CPU deployment.
2. **Reproducible ML pipeline (60 seconds).** Git versions code. DVC versions four attributed demo images, the BLIP v2 checkpoint, preprocessing, two evaluation stages, model selection, and the champion artifact. Show `dvc.yaml`, `dvc.lock`, `dvc status -c`, and the private S3 remote. Both checkpoints scored 10/13 on this small manually authored demo set. This is a pipeline test, not an official VizWiz result.
3. **Tracking and selection (45 seconds).** Show `mlops/experiments/*.json` and `mlops/registry/selection.json`. The candidates use the same input hashes. The declared rule chooses highest exact match, then v2 on a tie. The champion manifest includes the model SHA256 to catch accidental artifact swaps.
4. **CI/CD and deployment (60 seconds).** Show the [successful MAI workspace deployment](https://github.com/Vkmstry1611/mai-project/actions/runs/40696627943): validate, publish both images, deploy. Point out the separate `ci.yml` and reusable `cd.yml` files; the CD job appears inside the same successful run. GitHub Container Registry stores SHA-tagged images. EC2 pulls them; DVC pulls the selected BLIP champion from private S3; Ollama holds Qwen3-VL 2B in a persistent EC2 volume. Caddy serves the HTTPS workspace; Systems Manager provides operator access. The Free Plan blocks ECR and GitHub OIDC creation here, so GHCR and the EC2 runner fill those roles.
5. **Live proof and operations (60 seconds).** Open [the public MAI workspace](https://15-135-22-228.sslip.io/), show that BLIP VQA is selected by default, upload `sample/stop_sign.png`, and ask “What word is on the sign?” Show the selected candidate and hash in the answer details and `/health`. Then switch to Qwen and repeat; this can take roughly 1–2 minutes on the CPU instance. Show both CloudWatch model metrics, error alarms, and deployment rollback.
6. **Cost and limits (25 seconds).** The workspace runs on one `m7i-flex.large` and a 50 GiB encrypted EBS volume in `ap-southeast-2`; stop it between presentations. Qwen is a generated visual answer, not verified OCR evidence. The manual 13-question BLIP score is too small for a quality claim; official VizWiz results are documented separately in `submission/RESULTS_AND_PROVENANCE.md`.

## Viva answers

| Question | Short answer |
|---|---|
| Why Git and DVC? | Git tracks code and small metadata; DVC hashes and remotes large datasets and model files. |
| What makes a run reproducible? | Locked stage dependencies, data/model hashes, pinned package versions, a container image tagged with the Git SHA, and a fixed selection rule. |
| Why two models? | The DVC experiment compared two BLIP checkpoints and selected v2. The public workspace additionally offers a smaller cloud Qwen vision model for broader questions; its generated answers have no official VizWiz score here. |
| Is 10/13 the project's accuracy? | No. It is a tiny, manually authored serving demonstration; official VizWiz evaluation is separate. |
| How are the models fetched? | DVC reads the private BLIP artifact from S3 through the EC2 instance profile. Ollama pulls Qwen3-VL 2B into a persistent volume on the same EC2 host. No model inference depends on the laptop. |
| How does CI/CD work? | A branch push tests code, builds and pushes a SHA-tagged GHCR image, then the EC2 runner pulls and health-checks it. |
| How is inference monitored? | The BLIP API records requests, errors and latency. The Qwen workspace emits CloudWatch latency/error metrics with a Model dimension. Each model has an error alarm. |
| What if deployment fails? | The deploy script tests `/health`, restores the prior local image when possible, and fails the GitHub job visibly. |
| How is cost controlled? | One CPU instance, no load balancer or managed endpoint, seven day logs, and stop/start between demos. The 50 GiB disk and Elastic IP can still incur charges while the instance is stopped. |
