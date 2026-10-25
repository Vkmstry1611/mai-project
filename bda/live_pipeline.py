"""Watch VM captures and refresh HDFS, MongoDB and Spark after new requests."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def event_snapshot(spool):
    files = sorted((Path(spool) / "events").glob("*.json"))
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}\n".encode())
    return digest.hexdigest(), len(files), max((path.stat().st_mtime for path in files), default=0)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def spark_counts(log):
    for line in reversed(log.splitlines()):
        if line.startswith("{'raw':"):
            try:
                return ast.literal_eval(line)
            except (SyntaxError, ValueError):
                return {}
    return {}


def ensure_cluster():
    """Restart Hadoop controllers after a VM reboot when their ports are down."""
    for port, command in ((9000, "start-dfs.sh"), (8032, "start-yarn.sh")):
        try:
            with socket.create_connection(("h-primary", port), timeout=2):
                continue
        except OSError:
            subprocess.run([command], check=True, timeout=120)
    # The NameNode may still be leaving safe mode; a failed job is retried.


def export_cluster_summary(hdfs_output, target, counts):
    merged = subprocess.run(
        ["hdfs", "dfs", "-cat", hdfs_output.rstrip("/") + "/summary/part-*"],
        check=True, text=True, capture_output=True,
    )
    runs = [json.loads(line) for line in merged.stdout.splitlines() if line.strip()]
    write_json(target, {"refreshed_at_utc": now_utc(), "counts": counts, "runs": runs})


def run_pipeline(project, spool, hdfs_root, output, blip_run_id, qwen_run_id,
                 status_path, summary_path, signature, count):
    write_json(status_path, {"state": "running", "event_count": count,
                             "started_at_utc": now_utc()})
    ensure_cluster()
    command = ["bash", "bda/vm_pipeline.sh", blip_run_id, qwen_run_id,
               str(spool), hdfs_root, output]
    completed = subprocess.run(command, cwd=project, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    log_path = Path(project) / "bda-data" / "analytics" / "live_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Pipeline exited {completed.returncode}; see {log_path}")
    counts = spark_counts(completed.stdout)
    export_cluster_summary(output, summary_path, counts)
    write_json(status_path, {"state": "complete", "event_count": count,
                             "signature": signature, "completed_at_utc": now_utc(),
                             "counts": counts})


def watch(project, spool, hdfs_root, output, blip_run_id, qwen_run_id,
          status_path, summary_path, poll_seconds=3):
    completed_signature = read_json(status_path).get("signature")
    while True:
        signature, count, newest_mtime = event_snapshot(spool)
        if count and signature != completed_signature and time.time() - newest_mtime >= 2:
            try:
                run_pipeline(project, spool, hdfs_root, output, blip_run_id, qwen_run_id,
                             status_path, summary_path, signature, count)
            except Exception as error:
                write_json(status_path, {"state": "error", "event_count": count,
                                         "failed_at_utc": now_utc(), "error": str(error)})
                time.sleep(15)
            else:
                completed_signature = signature
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--spool", default="spool")
    parser.add_argument("--hdfs-root", default="/visionbridge/raw")
    parser.add_argument("--output", default="/visionbridge/analytics")
    parser.add_argument("--blip-run-id", default="blip-demo-v1")
    parser.add_argument("--qwen-run-id", default="qwen-demo-v1")
    parser.add_argument("--poll-seconds", type=float, default=3)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    spool = Path(args.spool)
    if not spool.is_absolute():
        spool = project / spool
    analytics = project / "bda-data" / "analytics"
    watch(project, spool, args.hdfs_root, args.output, args.blip_run_id,
          args.qwen_run_id, analytics / "pipeline_status.json",
          analytics / "cluster_summary.json", args.poll_seconds)


if __name__ == "__main__":
    main()
