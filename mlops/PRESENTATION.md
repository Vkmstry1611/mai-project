# Five minute MLOps presentation

1. **Goal and scope (30 seconds).** VisionBridge is an accessible visual question answering project. This deployment serves its selected BLIP VQA checkpoint through a CPU API. The frozen research dashboard includes additional OCR and voice features that this API does not serve.
2. **Reproducible ML pipeline (60 seconds).** Git versions code. DVC versions four attributed demo images, the BLIP v2 checkpoint, preprocessing, two evaluation stages, model selection, and the champion artifact. Show `dvc.yaml`, `dvc.lock`, `dvc status -c`, and the private S3 remote. Both checkpoints scored 10/13 on this small manually authored demo set. This is a pipeline test, not an official VizWiz result.
3. **Tracking and selection (45 seconds).** Show `mlops/experiments/*.json` and `mlops/registry/selection.json`. The candidates use the same input hashes. The declared rule chooses highest exact match, then v2 on a tie. The champion manifest includes the model SHA256 to catch accidental artifact swaps.
4. **CI/CD and deployment (60 seconds).** Show the [successful GitHub Actions run](https://github.com/Vkmstry1611/mai-project/actions/runs/40368682335): validate, publish, deploy. GitHub Container Registry stores the immutable image tag. EC2 pulls it, then DVC pulls the pinned model from private S3 using temporary instance credentials. Systems Manager gives access without inbound ports. The AWS Free Plan blocks ECR and GitHub OIDC creation in this account, so GHCR and the narrow EC2 runner fill those roles.
5. **Live proof and operations (60 seconds).** Run `mlops/aws/smoke-test.ps1` and show `/health`, `/predict` answering “stop,” and `/metrics` reporting one request, zero errors, and 1.193 second latency in the verified run. Show the CloudWatch prediction event, latency metric, and error alarm. Explain the seven day log retention and previous-image rollback.
6. **Cost and limits (25 seconds).** The demo runs on one `m7i-flex.large` and a 30 GiB encrypted EBS volume in `ap-southeast-2`; stop it between presentations. Current credits can lag actual usage. The manual 13-question score is too small for a quality claim; official VizWiz results are documented separately in `submission/RESULTS_AND_PROVENANCE.md`.

## Viva answers

| Question | Short answer |
|---|---|
| Why Git and DVC? | Git tracks code and small metadata; DVC hashes and remotes large datasets and model files. |
| What makes a run reproducible? | Locked stage dependencies, data/model hashes, pinned package versions, a container image tagged with the Git SHA, and a fixed selection rule. |
| Why two models? | They are two existing BLIP checkpoints evaluated on identical demo inputs, with separate metric files and a declared tie rule. |
| Is 10/13 the project's accuracy? | No. It is a tiny, manually authored serving demonstration; official VizWiz evaluation is separate. |
| How is the model fetched securely? | DVC reads the private S3 remote through the EC2 instance profile and IMDSv2 temporary credentials. |
| How does CI/CD work? | A branch push tests code, builds and pushes a SHA-tagged GHCR image, then the EC2 runner pulls and health-checks it. |
| How is inference monitored? | The API records request count, errors and latency; CloudWatch stores structured events, emits latency/error metrics, and alarms on errors. |
| What if deployment fails? | The deploy script tests `/health`, restores the prior local image when possible, and fails the GitHub job visibly. |
| How is cost controlled? | One CPU instance, no load balancer or managed endpoint, no inbound public API, seven day logs, and stop/start between demos. |
