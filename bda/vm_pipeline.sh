#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "Usage: bash bda/vm_pipeline.sh BLIP_RUN_ID QWEN_RUN_ID [SPOOL] [HDFS_ROOT] [OUTPUT]" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

blip_run_id="$1"
qwen_run_id="$2"
spool="${3:-spool}"
hdfs_root="${4:-/visionbridge/raw}"
output="${5:-/visionbridge/analytics}"

if [[ ! -d "$spool/events" ]]; then
  echo "No capture events directory at $spool/events" >&2
  exit 1
fi

/usr/bin/python3 -m bda.ingest \
  --spool "$spool" \
  --hdfs-root "$hdfs_root" \
  --mongo-uri "${MONGODB_URI:-mongodb://localhost:27017}" \
  --database "${MONGODB_DATABASE:-visionbridge_bda}"

export PATH="$project_dir/.venv/bin:$PATH"
export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-/usr/local/hadoop/etc/hadoop}"
export YARN_CONF_DIR="${YARN_CONF_DIR:-$HADOOP_CONF_DIR}"
export PYSPARK_DRIVER_PYTHON="$project_dir/.venv/bin/python"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-/usr/bin/python3}"

exec spark-submit --master yarn --deploy-mode client \
  --conf spark.executor.instances=2 \
  --conf spark.executor.memory=1g \
  --conf spark.executor.cores=1 \
  bda/spark_analysis.py \
  --events "hdfs:///${hdfs_root#/}/events/*.json" \
  --output "hdfs:///${output#/}" \
  --blip-run-id "$blip_run_id" \
  --qwen-run-id "$qwen_run_id"
