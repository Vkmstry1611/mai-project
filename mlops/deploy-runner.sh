#!/usr/bin/env bash
set -euo pipefail
owner="${1:?Usage: deploy-runner.sh <github-owner> <git-sha>}"
image_tag="${2:?Usage: deploy-runner.sh <github-owner> <git-sha>}"
[[ "$owner" =~ ^[A-Za-z0-9-]+$ ]] || { echo 'Invalid owner' >&2; exit 2; }
[[ "$image_tag" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a 40-character Git SHA' >&2; exit 2; }
image_uri="ghcr.io/${owner,,}/visionbridge-mlops:${image_tag}"
docker pull "$image_uri"
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
      https://15-135-22-228.sslip.io/health >/dev/null; then
    exit 0
  fi
  sleep 5
done
docker logs --tail 100 visionbridge-proxy || true
exit 1
