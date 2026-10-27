# VisionBridge AWS demo deployment

## Project resources

All regional resources are in `ap-southeast-2` under AWS CLI profile `mlops-visionbridge`.

| Purpose | Resource |
|---|---|
| DVC object storage | Private, encrypted S3 bucket `visionbridge-mlops-20261026-1a167b3`, prefix `dvc/` |
| CPU API host | EC2 `i-0e44bd4d11fa9f312`, `m7i-flex.large` |
| Reproducible host configuration | Launch template `visionbridge-mlops-demo`, version 2 (default) |
| Public demo | [https://15-135-22-228.sslip.io/](https://15-135-22-228.sslip.io/) through Caddy HTTPS proxy |
| Stable address | Elastic IP `15.135.22.228`, allocation `eipalloc-08bb7769aa2b1beda` |
| Access | Systems Manager for operators; security group `sg-035359fd71e75616d` allows public ports 80 and 443; API port 8000 binds to localhost |
| Temporary AWS credentials | Instance profile `VisionBridgeMLOpsEC2Profile` with scoped S3 read, CloudWatch log write, and SSM permissions |
| Application logs | `/visionbridge/mlops/api`, seven day retention |
| Inference monitoring | CloudWatch namespace `VisionBridge/MLOps`: `PredictionLatencySeconds`, `PredictionErrors`, and alarm `VisionBridgePredictionErrors` |
| CI image | `ghcr.io/vkmstry1611/visionbridge-mlops:<full-git-sha>` |
| CI/CD | [VisionBridge MLOps GitHub Actions](https://github.com/Vkmstry1611/mai-project/actions/workflows/mlops.yml) |

The [AWS Free Plan managed policy](https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html) blocks ECR repository creation and IAM OIDC providers. GitHub Actions builds the image and GitHub Container Registry holds it. An EC2 self-hosted runner deploys it using the workflow's short-lived GitHub token. The container pulls the selected model from S3 using temporary instance-role credentials. No static AWS keys are stored in GitHub.

## Reproduce or inspect

The code, raw data DVC pointer, model pointer, and lock file are on branch `codex/mlops-agent-toolkit`. Locally, a Python environment with the dependencies described in [`../README.md`](../README.md) can run:

```powershell
dvc repro
dvc status -c
dvc metrics show
```

The data remote status was `Cache and remote 'aws' are in sync.` Both BLIP candidates scored 10/13 on the small manual demo set; this is an engineering demonstration, not an official VizWiz evaluation.

## Verified public run on 2026-10-27

[GitHub Actions run 6](https://github.com/Vkmstry1611/mai-project/actions/runs/40543121064) completed `validate`, `publish`, and `deploy` successfully for commit `21ebd9f80e5195316959e6abff393349981bd853`. The public HTTPS `/health` returned HTTP 200 and reported `blip_vizwiz_v2`. An external HTTPS `POST /predict` using `sample/stop_sign.png` and “What word is on the sign?” returned `stop` in 1.116 seconds. The page at the public URL returned HTTP 200. Caddy handles TLS and forwards to the API on instance localhost; the image and model are pulled separately from GHCR and private S3. The hostname is provided by third-party sslip.io DNS, so confirm it resolves before a presentation.

## Earlier private smoke test on 2026-10-26

[GitHub Actions run 3](https://github.com/Vkmstry1611/mai-project/actions/runs/40368682335) completed with `validate`, `publish`, and `deploy` all successful for source commit `353c05bc19b59d18ed821ea1c871bd4f9549ff9d`. The image pulled `mlops/registry/champion` from S3 at container startup; CloudWatch showed `9 files fetched and 8 files added`, application startup complete, and `/health` HTTP 200.

The Systems Manager smoke test `mlops/aws/smoke-test.ps1` sent the tracked `sample/stop_sign.png` and asked “What word is on the sign?” The response was:

```json
{"answer":"stop","model_sha256":"ed7f698a1c79043019ece5df9b3cbd073fc0f050e97196d123ab83950862cc34","latency_seconds":1.193}
```

`GET /metrics` returned one request, zero errors, and mean latency 1.193 seconds. CloudWatch Logs received the JSON prediction event, and the `PredictionLatencySeconds` metric contained the 1.193 second datapoint. The bucket's four public-access blocks were true, and bucket encryption was AES256. Evidence screenshots are in `ss/`; the linked GitHub run and commands above provide the deployment record.

To inspect the live container through Systems Manager:

```powershell
aws ssm send-command --instance-ids i-0e44bd4d11fa9f312 --document-name AWS-RunShellScript --parameters commands="curl -fsS http://127.0.0.1:8000/health" --region ap-southeast-2 --profile mlops-visionbridge --query Command.CommandId --output text
aws ssm get-command-invocation --command-id <id-from-previous-command> --instance-id i-0e44bd4d11fa9f312 --region ap-southeast-2 --profile mlops-visionbridge --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
```

The API binds to the instance's localhost. For a private operator check, [install the AWS Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) and open a port-forwarding session to the instance:

```powershell
aws ssm start-session --target i-0e44bd4d11fa9f312 --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["8000"],"localPortNumber":["8000"]}' --region ap-southeast-2 --profile mlops-visionbridge
```

In another terminal, send JSON to `http://127.0.0.1:8000/predict` with `question` and base64 encoded `image_base64`. `GET /health` reports model identity; `GET /metrics` reports request count, errors, and mean latency. The workflow's deployment step tests `/health` after every image change and attempts to restore the prior local image if the new one fails.

## Demo and rollback

1. Show `dvc.yaml`, `dvc.lock`, candidate metric files, and `mlops/registry/selection.json`.
2. Show the successful run 6 `validate`, `publish`, and `deploy` jobs, the GHCR image tag, the private S3 remote, and the EC2 role/security group.
3. Open the public HTTPS demo, upload `sample/stop_sign.png`, ask “What word is on the sign?”, and show `stop`. Show `/health`, `/metrics`, CloudWatch logs and the latency/error metrics.
4. Explain that Git versions source, DVC versions large inputs and champion output, GHCR versions the API image, and the Git SHA pins one deployment.
5. For a manual rollback, rerun `bash mlops/deploy-runner.sh Vkmstry1611 <previous-successful-git-sha>` on the EC2 runner with GHCR login available. The deployment script also restores the prior local image on a failed health check.

## Credits and shutdown

The AWS project showed an active Free Plan and `$98.57` remaining credits when checked on 2026-10-27. Usage reporting can lag. EC2, 30 GiB EBS, the Elastic IP, S3 objects/requests, and logs consume credits. Stop the instance between demos to limit EC2 usage, and start it before showing the live API; the runner and container resume automatically. Stopping does not remove EBS, S3, or allocated public IPv4 charges. The public link is unavailable while the instance is stopped.

```powershell
aws ec2 stop-instances --instance-ids i-0e44bd4d11fa9f312 --region ap-southeast-2 --profile mlops-visionbridge
aws ec2 start-instances --instance-ids i-0e44bd4d11fa9f312 --region ap-southeast-2 --profile mlops-visionbridge
```

At project end, remove the runner from GitHub settings, terminate the instance, and delete unneeded images, log groups, and S3 objects only after saving required grading evidence. Screenshots go in [`../../ss`](../../ss).
