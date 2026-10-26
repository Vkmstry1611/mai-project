#!/bin/bash
set -euxo pipefail
dnf install -y docker git jq tar gzip libicu
systemctl enable --now docker
systemctl enable --now amazon-ssm-agent
usermod -aG docker ec2-user
mkdir -p /opt/visionbridge
chown ec2-user:ec2-user /opt/visionbridge
