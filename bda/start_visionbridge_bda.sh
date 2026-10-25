#!/usr/bin/env bash
# Start the existing three-node Hadoop cluster and VisionBridge BDA services.
# Run on the primary VM. Ctrl+C or q closes monitoring; services keep running.

set -u
export HADOOP_HOME="${HADOOP_HOME:-/usr/local/hadoop}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-8-openjdk-amd64}"
export PATH="$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$JAVA_HOME/bin:$PATH"

PROJECT="${VISIONBRIDGE_PROJECT:-/home/admin/Desktop/BDA/project}"
DASHBOARD_URL="${VISIONBRIDGE_DASHBOARD_URL:-http://127.0.0.1:8501}"
QWEN_URL="${VISIONBRIDGE_QWEN_URL:-http://192.168.56.1:7865}"
BLIP_URL="${VISIONBRIDGE_BLIP_URL:-http://192.168.56.1:7866}"
WORKERS=(secondary1 secondary2)

has_local() { jps 2>/dev/null | awk '{print $2}' | grep -Fxq "$1"; }
has_remote() {
    ssh -o BatchMode=yes -o ConnectTimeout=5 "admin@$1" \
        "jps 2>/dev/null | grep -Eq '[[:space:]]${2}\$'" 2>/dev/null
}

ensure_local() {
    local role="$1" command="$2" daemon="$3"
    if has_local "$role"; then
        printf '  ✓ %s already running on primary\n' "$role"
    else
        printf '  Starting %s on primary...\n' "$role"
        "$HADOOP_HOME/bin/$command" --daemon start "$daemon" || true
    fi
}

ensure_remote() {
    local host="$1" role="$2" command="$3" daemon="$4"
    if has_remote "$host" "$role"; then
        printf '  ✓ %s already running on %s\n' "$role" "$host"
    else
        printf '  Starting %s on %s...\n' "$role" "$host"
        ssh -o BatchMode=yes -o ConnectTimeout=5 "admin@$host" \
            "export JAVA_HOME='$JAVA_HOME' HADOOP_HOME='$HADOOP_HOME'; '$HADOOP_HOME/bin/$command' --daemon start '$daemon'" \
            || printf '  ! Could not start %s on %s; check SSH and that VM is on.\n' "$role" "$host"
    fi
}

start_all() {
    echo 'Starting HDFS...'
    ensure_local NameNode hdfs namenode
    for host in "${WORKERS[@]}"; do ensure_remote "$host" DataNode hdfs datanode; done
    ensure_local SecondaryNameNode hdfs secondarynamenode

    echo 'Starting YARN...'
    ensure_local ResourceManager yarn resourcemanager
    for host in "${WORKERS[@]}"; do ensure_remote "$host" NodeManager yarn nodemanager; done

    echo 'Starting MongoDB and Streamlit...'
    sudo docker start mongodb visionbridge-bda-dashboard >/dev/null || true
    echo 'Starting automatic HDFS and Spark refresh...'
    sudo systemctl start visionbridge-bda-refresh.service || true
}

http_state() {
    if curl --silent --output /dev/null --max-time 3 --fail "$1"; then printf 'UP'; else printf 'DOWN'; fi
}

status_once() {
    printf '\n=== VisionBridge BDA · %s ===\n' "$(TZ=Asia/Kolkata date '+%d %b %Y %I:%M:%S %p IST')"
    printf 'Primary: NameNode %s | ResourceManager %s | SecondaryNameNode %s\n' \
        "$(has_local NameNode && echo UP || echo DOWN)" \
        "$(has_local ResourceManager && echo UP || echo DOWN)" \
        "$(has_local SecondaryNameNode && echo UP || echo DOWN)"
    for host in "${WORKERS[@]}"; do
        printf '%s: DataNode %s | NodeManager %s\n' "$host" \
            "$(has_remote "$host" DataNode && echo UP || echo DOWN)" \
            "$(has_remote "$host" NodeManager && echo UP || echo DOWN)"
    done
    printf 'HDFS live DataNodes: '
    hdfs dfsadmin -report 2>/dev/null | sed -n 's/^Live datanodes (\([0-9]*\)):.*/\1/p' | head -n 1
    printf 'YARN active NodeManagers: '
    yarn node -list 2>/dev/null | sed -n 's/^Total Nodes:\([0-9]*\).*/\1/p' | head -n 1
    printf 'MongoDB: %s | Dashboard: %s (%s)\n' \
        "$(sudo -n docker inspect -f '{{.State.Status}}' mongodb 2>/dev/null || echo DOWN)" \
        "$(http_state "$DASHBOARD_URL/_stcore/health")" "$DASHBOARD_URL"
    local qwen_state blip_state
    qwen_state="$(http_state "$QWEN_URL/")"
    blip_state="$(http_state "$BLIP_URL/")"
    printf 'Windows models: Qwen %s | BLIP %s\n' "$qwen_state" "$blip_state"
    if [[ "$qwen_state" != UP || "$blip_state" != UP ]]; then
        echo '  ! Start the missing Windows model service; the VM cannot launch it.'
    fi
    printf 'Spark refresh worker: %s\n' "$(systemctl is-active visionbridge-bda-refresh.service 2>/dev/null || echo DOWN)"
    python3 - "$PROJECT/bda-data/analytics/pipeline_status.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
    print(f"Latest Spark refresh: {data.get('state', 'unknown')} · {data.get('event_count', 0)} requests")
except (OSError, ValueError):
    print('Latest Spark refresh: no status yet')
PY
    printf 'Dashboard: %s\n' "$DASHBOARD_URL"
    echo 'Press q to close this monitor. Services will keep running.'
}

case "${1:-}" in
    --status) status_once; exit 0 ;;
    --start-only) start_all; status_once; exit 0 ;;
    "") ;;
    *) echo "Usage: $0 [--status|--start-only]" >&2; exit 2 ;;
esac

start_all
if [[ ! -t 0 ]]; then status_once; exit 0; fi
trap 'echo; echo "Monitor closed. Cluster services remain running."; exit 0' INT
while true; do
    printf '\033[H\033[2J'
    status_once
    if read -r -s -n 1 -t 12 key && [[ "$key" == [qQ] ]]; then break; fi
done
echo 'Monitor closed. Cluster services remain running.'
