#!/usr/bin/env bash
set -euo pipefail
owner="${1:?Usage: deploy-runner.sh <github-owner> <git-sha>}"
image_tag="${2:?Usage: deploy-runner.sh <github-owner> <git-sha>}"
[[ "$owner" =~ ^[A-Za-z0-9-]+$ ]] || { echo 'Invalid owner' >&2; exit 2; }
[[ "$image_tag" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a 40-character Git SHA' >&2; exit 2; }
image_uri="ghcr.io/${owner,,}/visionbridge-mlops:${image_tag}"
workspace_image_uri="ghcr.io/${owner,,}/visionbridge-workspace:${image_tag}"
vision_model="qwen3-vl:2b-instruct-q4_K_M"
docker pull "$image_uri"
docker pull "$workspace_image_uri"
previous_image="$(docker inspect --format '{{.Image}}' visionbridge-mlops 2>/dev/null || true)"
docker rm -f visionbridge-mlops 2>/dev/null || true
run_api() {
  docker run -d --name visionbridge-mlops --restart unless-stopped \
  --memory 7g --cpus 2 \
  -p 127.0.0.1:8000:8000 \
  -e AWS_DEFAULT_REGION=ap-southeast-2 \
  --log-driver awslogs --log-opt awslogs-region=ap-southeast-2 \
  --log-opt awslogs-group=/visionbridge/mlops/api \
  "$1"
}
restore_previous() {
  if [[ -n "$previous_image" ]]; then
    docker rm -f visionbridge-mlops 2>/dev/null || true
    run_api "$previous_image"
    echo "Restored previous image $previous_image after failed health check" >&2
  fi
}
if ! run_api "$image_uri"; then
  restore_previous
  exit 1
fi
for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 10
done
if ! curl -fsS http://127.0.0.1:8000/health >/dev/null; then
  docker logs --tail 100 visionbridge-mlops || true
  restore_previous
  exit 1
fi

# Both inference backends remain on this EC2 host. Ollama is loopback-only.
docker pull ollama/ollama:0.12.7
docker rm -f visionbridge-ollama 2>/dev/null || true
docker run -d --name visionbridge-ollama --restart unless-stopped --network host \
  --memory 4g --cpus 2 -v visionbridge-ollama-models:/root/.ollama \
  -e OLLAMA_HOST=127.0.0.1:11434 -e OLLAMA_NUM_PARALLEL=1 \
  -e OLLAMA_MAX_LOADED_MODELS=1 \
  ollama/ollama:0.12.7
for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then break; fi
  sleep 2
done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
docker exec visionbridge-ollama ollama pull "$vision_model"
docker exec visionbridge-ollama ollama show "$vision_model" >/dev/null

previous_workspace_image="$(docker inspect --format '{{.Image}}' visionbridge-workspace 2>/dev/null || true)"
docker rm -f visionbridge-workspace 2>/dev/null || true
run_workspace() {
  docker run -d --name visionbridge-workspace --restart unless-stopped --network host \
    --memory 1500m --cpus 2 \
    -e VISIONBRIDGE_CLOUD=1 -e VISIONBRIDGE_VISION_MODEL="$vision_model" \
    -e VISIONBRIDGE_NUM_CTX=2048 -e VISIONBRIDGE_NUM_PREDICT=384 \
    --log-driver awslogs --log-opt awslogs-region=ap-southeast-2 \
    --log-opt awslogs-group=/visionbridge/mlops/api \
    "$1"
}
if ! run_workspace "$workspace_image_uri"; then
  restore_previous
  exit 1
fi
for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:7865/ | grep -i VisionBridge >/dev/null; then break; fi
  sleep 5
done
if ! curl -fsS http://127.0.0.1:7865/ | grep -i VisionBridge >/dev/null; then
  docker logs --tail 100 visionbridge-workspace || true
  docker rm -f visionbridge-workspace 2>/dev/null || true
  if [[ -n "$previous_workspace_image" ]]; then run_workspace "$previous_workspace_image"; fi
  restore_previous
  exit 1
fi

# Load the lighter vision model with an image before switching the public page.
image64="$(base64 -w0 sample/stop_sign.png)"
jq -n --arg image "$image64" --arg model "$vision_model" \
  '{model:$model,stream:false,think:false,messages:[{role:"user",content:"What word is on the sign?",images:[$image]}],options:{num_ctx:2048,num_predict:64}}' \
  | curl -fsS --max-time 300 -H 'Content-Type: application/json' \
      -d @- http://127.0.0.1:11434/api/chat \
  | jq -e '.message.content | length > 0' >/dev/null
echo 'Cloud vision model answered the image warm check.'

install -m 644 mlops/aws/Caddyfile /opt/visionbridge/Caddyfile
docker pull caddy:2.11.7-alpine
docker run --rm -v /opt/visionbridge/Caddyfile:/etc/caddy/Caddyfile:ro \
  caddy:2.11.7-alpine caddy validate --config /etc/caddy/Caddyfile
docker rm -f visionbridge-proxy 2>/dev/null || true
docker run -d --name visionbridge-proxy --restart unless-stopped --network host \
  -v /opt/visionbridge/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v visionbridge-caddy-data:/data -v visionbridge-caddy-config:/config \
  caddy:2.11.7-alpine
for attempt in $(seq 1 60); do
  if curl -fsS --resolve 15-135-22-228.sslip.io:443:127.0.0.1 \
      https://15-135-22-228.sslip.io/health >/dev/null && \
      curl -fsS --resolve 15-135-22-228.sslip.io:443:127.0.0.1 \
      https://15-135-22-228.sslip.io/ | grep -i VisionBridge >/dev/null; then
    exit 0
  fi
  sleep 5
done
docker logs --tail 100 visionbridge-proxy || true
exit 1
