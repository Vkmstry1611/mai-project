# VisionBridge MLOps course implementation

This directory implements the professor's data → pipeline → experiments → selected model → DVC remote → Docker API → container registry → EC2 flow. The AWS project uses `ap-southeast-2`. The Agent Toolkit control service runs in `us-east-1`; application resources stay in `ap-southeast-2`.

## What is measured

The 16 locally authored demo questions and four attributed images are a **pipeline and serving demonstration**. Thirteen have reference answers. The two existing BLIP checkpoints each matched 10/13 references using simple normalized exact match. This small set is not VizWiz and does not estimate real-world VQA accuracy. The selected BLIP v2 checkpoint won the declared tie rule. The earlier official VizWiz results and their caveats remain in `submission/RESULTS_AND_PROVENANCE.md`; they are not recomputed or relabeled here.

## Architecture

```mermaid
flowchart TD
  A[Attributed demo images and questions] --> B[DVC raw data pointer]
  B --> C[Prepare and verify hashes]
  C --> D[Evaluate BLIP v1 and v2]
  D --> E[Metrics and deterministic selection]
  E --> F[DVC champion model artifact]
  F --> G[Private S3 remote]
  H[Git push] --> I[GitHub Actions tests and Docker build]
  I --> J[GitHub Container Registry image]
  J --> K[EC2 pulls image]
  G --> L[Container pulls pinned model]
  K --> M[CPU VQA API]
  L --> M
  M --> N[HTTPS Caddy proxy and public browser demo]
  M --> O[CloudWatch logs and API metrics]
```

## Local reproduction

Use a Python environment with DVC 3.67.1, Pillow, CPU or CUDA Torch, Transformers 4.46.3, tokenizers 0.20.3, and safetensors. The existing `.venv-gpu` inference environment supplies the neural dependencies locally; DVC is installed separately. Activate an environment containing the neural dependencies before running DVC stages.

```powershell
$env:Path = (Join-Path (Get-Location) '.venv-gpu\Scripts') + ';' + $env:Path
dvc repro
dvc status
dvc metrics show
```

`dvc.yaml` has five stages: `prepare`, `evaluate_v1`, `evaluate_v2`, `select`, and `register`. Raw images and BLIP v2 are DVC tracked. BLIP v1 remains in Git LFS. The lock file records exact dependency and output hashes. To compare changed pipeline versions later, run `dvc exp run` and `dvc exp show`; keep experiments based on the same data before drawing a model comparison.

The selected artifact is `mlops/registry/champion`, including `registry.json`. It contains the model, processor, selection rule, and source hashes. The deployment image contains code only; `mlops/serve.sh` pulls the pinned artifact from S3 on startup. No AWS keys are stored in Git or Docker.

## API

Open the [public HTTPS demo](https://15-135-22-228.sslip.io/) to upload an image, ask a question, and see the answer. The CPU API also exposes `GET /health`, `POST /predict`, and `GET /metrics`. Prediction accepts JSON:

```json
{"question":"What word is on the sign?","image_base64":"<base64-encoded image>"}
```

It returns the answer, selected model hash, and latency. The image and question are not logged; logs contain only result timing, model hash, and errors. The API binds to localhost on EC2; Caddy exposes the browser demo and API on HTTPS ports 80/443. SSM remains an operator access path. It is a BLIP-only VQA service; it does **not** claim to serve the frozen Phase 4 OCR, gating, voice, or Gradio workspace.

## AWS deployment

The private S3 bucket `visionbridge-mlops-20261026-1a167b3` holds DVC objects. GitHub Container Registry holds immutable code image tags. One `m7i-flex.large` CPU EC2 instance hosts the demo in `ap-southeast-2` with 8 GiB RAM. It uses an instance profile for S3/SSM/CloudWatch access and IMDSv2 hop limit 2 so the container can retrieve temporary credentials. Public ingress is limited to ports 80 and 443 for the HTTPS proxy; the API's port 8000 stays local. The AWS Free Plan's managed policy blocks creating an ECR private repository, CodeBuild, CodePipeline, CloudFormation stacks, and IAM OIDC providers, so GitHub Actions and GHCR provide the CI and image registry.

GitHub Actions validates on pushes to this branch, builds and pushes the image using its built-in GitHub token, then deploys on a narrowly labeled self-hosted runner on the EC2 demo instance. The container restarts automatically. Deployment waits for `/health` and can restore the previous local image after a failed health check. No static AWS keys are placed in GitHub. Pull requests do not run on the self-hosted runner because this is a public repository.

For the presentation, open the public demo above and ask the supplied stop-sign image “What word is on the sign?” See the deployment guide for resource IDs, evidence, and operator access.

## Rubric evidence

| Rubric item | Evidence to show |
|---|---|
| Cloud services and MLOps | Private S3 remote, GHCR image, EC2 instance, IAM role |
| Pipeline and CI/CD | `dvc.yaml`, `dvc.lock`, successful `dvc repro`, GitHub Actions runs |
| Tracking, monitoring, optimization | Both metric JSON files, selection manifest, `/metrics`, CloudWatch logs, measured latency |
| Documentation and presentation | This architecture, deployment record, screenshots in `ss/`, demo request and rollback |
| Viva | Explain Git vs DVC vs GHCR versions, fixed data split, CPU deployment limits, and cost controls |

## Cost and cleanup

The AWS project has Free Plan credits, but EC2, its EBS volume, S3 storage/requests, public IPv4, and logs consume them. The public link works while the instance runs. Stop the instance between demos; the Elastic IP remains allocated and can still incur charges. Terminate the instance and release the Elastic IP when finished. Delete old unneeded GHCR images and S3 DVC objects only after confirming the artifacts are no longer required. Keep the private bucket blocked from public access. Review AWS Settings billing and Free Tier usage during the project.
