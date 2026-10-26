#!/bin/bash
set -euo pipefail
curl -fsS http://127.0.0.1:8000/health
printf '\n'
test -s /tmp/visionbridge-stop-sign.png
image_base64="$(base64 -w 0 /tmp/visionbridge-stop-sign.png)"
jq -n --arg question 'What word is on the sign?' \
  --arg image_base64 "$image_base64" \
  '{question:$question,image_base64:$image_base64}' |
  curl -fsS -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8000/predict
printf '\n'
curl -fsS http://127.0.0.1:8000/metrics
printf '\n'
