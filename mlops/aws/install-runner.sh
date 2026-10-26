#!/bin/bash
set -euo pipefail
: "${RUNNER_TOKEN:?A fresh GitHub runner registration token is required}"
runner_version=2.338.2
runner_sha256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
runner_archive="actions-runner-linux-x64-${runner_version}.tar.gz"
runner_dir=/opt/visionbridge/actions-runner
dnf install -y libicu
install -d -o ec2-user -g ec2-user "$runner_dir"
curl -fL --retry 3 -o "/tmp/$runner_archive" \
  "https://github.com/actions/runner/releases/download/v${runner_version}/${runner_archive}"
printf '%s  %s\n' "$runner_sha256" "/tmp/$runner_archive" | sha256sum -c -
tar -xzf "/tmp/$runner_archive" -C "$runner_dir"
chown -R ec2-user:ec2-user "$runner_dir"
cd "$runner_dir"
runuser -u ec2-user -- ./config.sh --unattended \
  --url https://github.com/Vkmstry1611/mai-project \
  --token "$RUNNER_TOKEN" \
  --name visionbridge-ec2 \
  --labels visionbridge-mlops \
  --work /opt/visionbridge/_work \
  --replace
./svc.sh install ec2-user
./svc.sh start
rm -f "/tmp/$runner_archive"
