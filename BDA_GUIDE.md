# VisionBridge BDA branch

For the complete architecture, measured results, demonstration commands and
technical marking check, see [BDA_RUBRIC_EVIDENCE.md](BDA_RUBRIC_EVIDENCE.md).
For a single operating manual covering both the MAI app and VM deployment,
see [PROJECT_DETAILS.md](PROJECT_DETAILS.md).

This branch captures live VisionBridge requests and prepares a distributed
analysis of **BLIP and Qwen runs on the same image/question pairs**. The models
still answer on Windows. Streamlit saves each submission immediately on the
primary VM, and a background service refreshes HDFS and Spark on the cluster.
No VM credential is embedded in the repository.

## Current VM layout

The primary VM is `192.168.1.4` (`admin`), with workers
`192.168.1.2` and `192.168.1.5`. The BDA scripts are deployed to
`/home/admin/Desktop/BDA/project` on the primary VM. The folder name is
case-sensitive. HDFS and YARN each report two live workers as of October 25,
2026. Inference remains on the Windows laptop; only captures and analytics
need to move to the VM.

### One-command VM startup

After booting all three VMs, double-click **Start VisionBridge BDA** on the
primary desktop or run `bash ~/Desktop/start_visionbridge_bda.sh`. The script
starts missing HDFS/YARN daemons, MongoDB, Streamlit and the automatic Spark
refresh service, then monitors their status in the terminal. Press **q** to
close the monitor without stopping services. Qwen and BLIP run on Windows;
the monitor checks them and reports if either Windows service is down. The
version-controlled script is `bda/start_visionbridge_bda.sh`.

## BDA dashboard and Docker

The Python Streamlit dashboard runs in Docker with MongoDB. The VM deployment
writes each model result to its local spool and MongoDB as soon as it finishes.
BLIP/Qwen inference remains on Windows and is not included in this image.

For a standalone local Docker setup, run:

```powershell
docker compose -f compose.bda.yaml up -d --build mongo dashboard
```

Open `http://localhost:8501`. The **Ask VisionBridge** tab has one Streamlit
upload and question form for Qwen, BLIP, or Both; the **BDA analytics** tab
contains the captured results. Select
**Local capture** or **MongoDB** in the sidebar. The dashboard shows actual
request counts, routes, OCR coverage, abstention, timing, individual records
and matched BLIP/Qwen answers. It labels answer disagreement separately from
accuracy. The analytics tab refreshes every five seconds while open.
**Request history** shows every selected run newest first, with processing
date and time in India Standard Time, question, answer, status, duration and
request ID. Search filters the visible history. New requests store model start
and completion timestamps; older records use their existing capture timestamp
and are labelled **Captured** in the time-source column.

To ask, upload an image, type a question **or** record it, then click
**Analyze image** once. WAV upload is in the optional expander. A recorded
question is transcribed and sent through the selected model in that same
submission; the recognized text appears above the result. If both typed text
and audio are present, the typed question takes priority. **Read answers
aloud** adds a player to each model answer. The existing Windows Whisper and
text-to-speech modules handle speech. The stored request contains the final
text question; temporary recording bytes are not saved to HDFS or MongoDB.

Browsers can restrict microphones on the VM's plain HTTP LAN address. Upload
a WAV recording instead, or open the dashboard through a localhost tunnel.
The dashboard hides the recorder on the LAN address and gives the localhost
address, avoiding Streamlit's generic recording error:

```powershell
ssh -N -L 127.0.0.1:8501:127.0.0.1:8501 admin@192.168.1.4
```

Keep that PowerShell window open and browse to `http://127.0.0.1:8501`.

The **Cluster evaluation** section shows the background job's status and the
latest Spark run summary. An official VQA score appears only when VizWiz labels
and scoring rules were supplied; model disagreement is not an accuracy score.

### Dashboard running on the primary VM

The primary VM has a built `visionbridge-bda-dashboard` Docker image and a
running container with the same name. It uses host networking to reach the
existing `mongodb` container on port 27017, mounts `spool` read-write at
`/data/spool`, and mounts `bda-data/analytics` read-only at `/data/analytics`.
Open `http://192.168.1.4:8501` from Windows. After a VM reboot, check both
containers and start them if needed:

```bash
sudo docker ps --format '{{.Names}} {{.Status}}'
sudo docker start mongodb visionbridge-bda-dashboard
systemctl status visionbridge-bda-refresh.service --no-pager
curl --fail http://127.0.0.1:8501/_stcore/health
```

The image was built with
`sudo docker build -f Dockerfile.bda -t visionbridge-bda-dashboard .`.
Its HTTP health endpoint returned `200` from
the VM and Windows. Streamlit's `AppTest` rendered the dashboard title with
zero exceptions. The real data views will populate after paired app requests
are captured and ingested.

To ask a question inside this page, open the **Ask VisionBridge** tab and
select **Qwen**, **BLIP**, or **Both**, then upload an image and ask a question
in the same Streamlit form. The model runtimes remain on Windows as separate
services; their original Gradio pages are no longer shown in the dashboard. The default
URLs are `http://192.168.1.83:7865` for Qwen and
`http://192.168.1.83:7866` for BLIP; edit them in **Connection settings** if
the Windows IP changes. Start each service in its own Windows PowerShell
window:

```powershell
$env:GRADIO_SERVER_NAME = '0.0.0.0'
$env:VISIONBRIDGE_BDA_RUN_ID = 'qwen-demo-v1'
.\.venv\Scripts\python.exe launch_workspace.py --backend local --port 7865
```

```powershell
$env:GRADIO_SERVER_NAME = '0.0.0.0'
$env:VISIONBRIDGE_BDA_RUN_ID = 'blip-demo-v1'
.\.venv\Scripts\python.exe launch_workspace.py --backend frozen --port 7866
```

**Both** calls Qwen and BLIP sequentially with the same image bytes and
question, then displays their answers side by side. Each model service writes its own request event
to the Windows capture spool; the events share a pair ID and have distinct
run IDs. A model error is displayed for that model while the other model's
result is preserved. A disagreement between answers is not an accuracy score.

Each submission from the Streamlit page also creates a VM capture event and
upserts MongoDB immediately. The enabled `visionbridge-bda-refresh.service`
detects new events, imports them into HDFS, runs the YARN Spark analysis, and
exports `bda-data/analytics/cluster_summary.json`. The dashboard shows
**pending**, **running**, **complete**, or **error** while the cluster refreshes.
No manual transfer or request to rerun analytics is needed for this page.

The separate Gradio pages still capture on Windows. If using those pages
directly rather than the Streamlit page, copy the Windows
`reports/bda/spool/events` and `reports/bda/spool/images` directories into
`/home/admin/Desktop/BDA/project/spool/` on the primary VM. Then run
`/usr/bin/python3 -m bda.ingest --spool spool --hdfs-root /visionbridge/raw
--mongo-uri mongodb://localhost:27017 --database visionbridge_bda` from the
VM project directory. Run `bda/vm_pipeline.sh` after both model runs to
refresh Spark analytics.

## 1. Capture directly on the laptop (optional)

Run the app twice, once with `--backend local` and once with `--backend frozen`.
The latter needs the existing private Phase 4 runtime. Ask the **same exact
question about the same image bytes** in both runs. Each completed request
creates one JSON event under `reports/bda/spool/events/` and one content-addressed
JPEG under `reports/bda/spool/images/`. Those paths are ignored by Git.

```powershell
$env:VISIONBRIDGE_BDA_RUN_ID = 'qwen-demo-v1'
.\.venv\Scripts\python.exe launch_workspace.py --backend local --port 7865
```

In a second terminal:

```powershell
$env:VISIONBRIDGE_BDA_RUN_ID = 'blip-demo-v1'
.\.venv\Scripts\python.exe launch_workspace.py --backend frozen --port 7866
```

Set `VISIONBRIDGE_BDA_SPOOL` to change the spool directory. Every request has
a unique `request_id`. `pair_id` is a hash of the saved image and normalized
question, so identical inputs can be matched across runs. `run_id` identifies
the backend experiment. The event preserves route, OCR evidence, full processing
details, answer/abstention and timings. A BDA capture error is logged as an app
diagnostic and does not replace a successful answer. Raw microphone audio is
not captured; the reviewed question text is. Failed analysis requests with a
valid image/question are captured with `status: error` for failure-rate analysis.

**Check privacy before using personal images.** This branch intentionally
retains submitted images and questions for BDA analysis. Do not put the spool,
dataset, private checkpoints or MongoDB credentials into Git.

## 2. Transfer captures made outside Streamlit (optional)

Copy the `spool` directory from Windows into the primary VM's project folder
after actual app requests have been captured. From PowerShell in the Windows
repository root:

```powershell
scp -r reports/bda/spool admin@192.168.1.4:Desktop/BDA/project/
```

On the primary VM, use its system Python with `python3-pymongo` installed.
The existing MongoDB container is named `mongodb` and publishes port 27017.
From the VM project folder:

```bash
cd ~/Desktop/BDA/project
sudo docker start mongodb
/usr/bin/python3 -m bda.ingest --spool spool --hdfs-root /visionbridge/raw \
  --mongo-uri mongodb://localhost:27017
hdfs dfs -ls /visionbridge/raw/events
hdfs fsck /visionbridge/raw -files -blocks -locations
```

Ingestion is safe to repeat: HDFS paths are replaced with the same source files
and MongoDB upserts by `request_id`. The `requests` collection holds nested
event data and HDFS references. Example MongoDB queries:

```javascript
db.requests.countDocuments({ run_id: "qwen-demo-v1" })
db.requests.aggregate([
  { $group: { _id: "$run_id", requests: { $sum: 1 },
      meanSeconds: { $avg: "$processing_seconds" } } }
])
db.requests.findOne({ run_id: "blip-demo-v1" },
  { question: 1, route: 1, ocr_evidence: 1, final_answer: 1 })
```

## 3. Run Spark across the cluster

Run from the project folder on the primary VM. PySpark 3.5.3 is installed in
`.venv`. This cluster uses YARN, so submit with `--master yarn`. Do not use
`local[*]` as evidence of distributed execution. The runner sets the Python
paths required by the pip-installed Spark distribution:

```bash
cd ~/Desktop/BDA/project
bash bda/vm_pipeline.sh blip-demo-v1 qwen-demo-v1
```

The equivalent manual Spark command is:

```bash
cd ~/Desktop/BDA/project
export PATH="$PWD/.venv/bin:$PATH"
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
export YARN_CONF_DIR="$HADOOP_CONF_DIR"
export PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_PYTHON=/usr/bin/python3
spark-submit --master yarn --deploy-mode client \
  bda/spark_analysis.py \
  --events 'hdfs:///visionbridge/raw/events/*.json' \
  --output hdfs:///visionbridge/analytics \
  --blip-run-id blip-demo-v1 --qwen-run-id qwen-demo-v1
hdfs dfs -cat /visionbridge/analytics/summary/part-*.json
hdfs dfs -ls /visionbridge/analytics/matched_blip_qwen
```

The job writes `rejected`, `duplicates`, `curated`, `summary` and
`matched_blip_qwen` datasets. It validates required fields, rejects negative
processing times, deduplicates request IDs, then permits at most one record per
`pair_id`/`run_id`. The matched output contains only pairs present in both
selected runs. It reports answer disagreement and latency. Live user requests
have no reference answers, so they cannot establish VQA accuracy.

For a labelled VizWiz evaluation, prepare a private JSONL where every row has
`image_path`, `question` and all ten original `answers` objects. Use the **same
image files and question text** in the UI runs. Then prepare labels with the
same pair-ID algorithm and upload them to HDFS:

```bash
python -m bda.prepare_labels --source /private/vizwiz_eval.jsonl \
  --output /private/vizwiz_pair_labels.jsonl
hdfs dfs -mkdir -p /visionbridge/labels
hdfs dfs -put -f /private/vizwiz_pair_labels.jsonl /visionbridge/labels/
spark-submit --master yarn --deploy-mode client \
  bda/spark_analysis.py --events 'hdfs:///visionbridge/raw/events/*.json' \
  --output hdfs:///visionbridge/analytics \
  --blip-run-id blip-demo-v1 --qwen-run-id qwen-demo-v1 \
  --labels hdfs:///visionbridge/labels/vizwiz_pair_labels.jsonl \
  --rules /private/official_vqa_rules.json
hdfs dfs -cat /visionbridge/analytics/matched_score_summary/part-*.json
```

`--rules` is the verified official normalization table used by
`evaluate_vqa.py`; it is not committed publicly. The scored summary uses only
pairs present in **both** selected runs and the labels, with ten reference
slots per pair. Do not mix historical BLIP scores with new Qwen runs unless
the examples and scoring protocol match.

## Completion checks on the VMs

1. A new UI request creates an image and JSON event locally.
2. The same request appears in HDFS and MongoDB with the same `request_id`.
3. `hdfs fsck` shows block placement across the two worker DataNodes.
4. YARN's application UI shows containers on the cluster, and raw = curated +
   rejected + duplicate rows for the run.
5. Matched BLIP/Qwen pairs are counted at one pair per selected run.

Real request ingestion and the distributed Spark results must be run after
the laptop has captured paired BLIP and Qwen requests. Never present a smoke
test dataset as real project results.

On October 24, 2026, a separate two-record synthetic smoke fixture passed
HDFS ingestion, MongoDB upsert, and a YARN Spark job. YARN reported a
`SUCCEEDED` application with executors on both worker VMs. The smoke paths
are `/visionbridge/smoke/raw`, `/visionbridge/smoke/analytics`, and MongoDB
database `visionbridge_bda_smoke`. These are infrastructure checks only.
