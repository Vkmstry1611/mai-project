#!/bin/sh
set -eu
cd /app
dvc pull mlops/registry/champion
exec uvicorn mlops.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
