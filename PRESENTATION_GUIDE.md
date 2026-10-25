# VisionBridge: complete project and presentation guide

**Project:** VisionBridge — image question answering with a Big Data Analytics extension

**Branch:** bda-project

**Verified deployment snapshot:** 25 October 2026
**Audience:** Mini project presentation, live demonstration, viva, and report preparation

This is the single guide to study before presenting. The live counts can increase whenever a new question is submitted. Run the verification commands in Section 13 before quoting numbers in a presentation.

## 1. Thirty-second introduction

VisionBridge helps a user ask questions about a photograph. In the current BDA interface, the user uploads an image, types or records a question, and selects Qwen, BLIP, or Both. The system returns an answer, optional speech, processing details, and a saved request ID. It stores images and structured request records, then uses a three-node Hadoop cluster and PySpark to check data quality and compare the two model runs on the same image and question.

**One-sentence BDA objective:** Measure how Qwen and BLIP differ in answers, recorded OCR evidence, abstention, and processing time for identical image-question pairs, while preserving auditable records in MongoDB and HDFS.

## 2. Problem statement and scope

A photograph may contain objects, packaging, small printed text, dates, or a chart. An answer alone does not reveal which model gave it, what visible evidence was used, whether the model abstained, or how long the response took. Comparing answers also requires a fair pairing rule: a Qwen result for one question must not be compared with a BLIP result for a different question or image.

VisionBridge addresses this by:

1. Capturing each model call as an identifiable request.
2. Keeping the image and raw JSON record for audit.
3. Making the nested record queryable in MongoDB.
4. Distributing raw records and a partitioned VizWiz annotation selection in HDFS.
5. Running PySpark validation, deduplication, summaries, and matched model comparisons on YARN.
6. Showing the latest results and request history in one Streamlit dashboard.

**Why this is a BDA project:** The design handles images plus nested JSON (variety), automatically captures new requests (velocity), and processes partitioned annotations and growing request logs across multiple nodes (scalable volume). The current 19-request live set is a demonstration of the pipeline; it is not, by itself, a large dataset. The separate distributed annotation selection has 9,319 rows.

## 3. What the user sees

The visible BDA page is Streamlit at **http://192.168.56.10:8501/**. It has:

- **Ask VisionBridge:** image upload, typed question or recorded/WAV question, Qwen/BLIP/Both choice, one Analyze image button, answer cards, optional spoken output, processing details, timestamp, and request ID.
- **BDA analytics:** request counts, run comparison, matched answers, searchable request history in India Standard Time, Spark refresh status, and the distributed dataset audit.

The two Windows Gradio applications are used as **internal APIs** by Streamlit. Users do not need to operate two separate model pages for the BDA demo.

Typing and WAV upload work from the VM address. Browsers generally require HTTPS or loopback for microphone recording. For recording, open the same dashboard through this Windows PowerShell tunnel, then browse to **http://127.0.0.1:8501/** in Chrome or Edge:

~~~powershell
ssh -N -L 127.0.0.1:8501:127.0.0.1:8501 admin@192.168.56.10
~~~

The recognized question text is submitted to inference. The raw microphone recording is temporary and is not stored in HDFS or MongoDB.

## 4. Architecture: where each component runs

~~~mermaid
flowchart LR
    U[User browser] --> S[Streamlit in Docker<br/>primary VM]
    S -->|image + question| Q[Qwen API<br/>Windows]
    S -->|same input if Both| B[BLIP API<br/>Windows]
    Q --> S
    B --> S
    S --> P[VM spool<br/>JPEG + JSON]
    S --> M[(MongoDB)]
    P --> W[Automatic refresh service]
    W --> H[(HDFS<br/>2 DataNodes)]
    W --> Y[PySpark on YARN<br/>2 NodeManagers]
    Y --> A[Analytics files]
    A --> S
~~~

| Computer | Stable address | Main work |
| --- | --- | --- |
| Windows laptop | 192.168.56.1 | Ollama Qwen model, frozen BLIP runtime, speech recognition and speech synthesis |
| Primary Ubuntu VM | 192.168.56.10 | Streamlit Docker container, MongoDB Docker container, HDFS NameNode, YARN ResourceManager, Spark driver, refresh service |
| Worker 1 (secondary1) | 192.168.56.11 | HDFS DataNode and YARN NodeManager |
| Worker 2 (secondary2) | 192.168.56.12 | HDFS DataNode and YARN NodeManager |

VirtualBox adapter 2 is a fixed host-only network for these addresses. Adapter 1 remains bridged for internet access. The primary has 4 GB RAM and 8 virtual CPUs; each worker has 2 GB RAM and 4 virtual CPUs. The primary also has a 2 GB swap file. HDFS uses replication factor **2**, so its blocks are placed on the two DataNodes.

**Important viva distinction:** Qwen and BLIP inference runs on **Windows**, where the weights and GPU runtime are installed. Hadoop, MongoDB, and Spark run on the **VMs**. The single Streamlit page on the primary calls the Windows models over the host-only network.

## 5. Models and accessibility modules

| Module | Implementation | What to say |
| --- | --- | --- |
| Qwen | Local Ollama qwen3-vl:4b | General visual question answering; may read printed text internally, but its BDA record does not imply independently verified OCR evidence |
| BLIP | Frozen VizWiz-tuned BLIP v2 research runtime | Separate VQA baseline with its own evidence and answerability behavior |
| OCR in the frozen path | Existing EasyOCR/RapidOCR and question-aware evidence selection | Used for supported printed-text routes; recorded OCR evidence depends on the route |
| Speech to text | Faster Whisper on Windows, CPU int8 | Transcribes a recorded/WAV question before model dispatch |
| Text to speech | Windows SAPI via pyttsx3 | Creates spoken answer audio when enabled |
| Routing and decisions | Existing VisionBridge research policies | Preserve route/evidence/abstention information; do not claim every answer uses OCR |

Selecting **Both** makes two sequential API calls with the same image bytes and question. This avoids loading both inference paths simultaneously on the limited laptop. The page stores a separate result for each model.

The Qwen model is configured to unload after each BDA answer with VISIONBRIDGE_OLLAMA_KEEP_ALIVE=0. This reduces memory pressure during Spark processing; the next Qwen request may take longer to load. The frozen BLIP runtime requires its private, verified model bundle; a source-only Git checkout is insufficient to run it.

**Do not merge the two model claims:** This BDA version compares Qwen and BLIP. It does not train Qwen on BLIP outputs or combine their answers into a new model.

### Earlier MAI research versus this BDA demo

The original MAI work includes a frozen BLIP v2 pipeline, OCR/evidence policies, answerability gates, and an experimental Gradio workspace with follow-up questions. The BDA dashboard reuses its frozen BLIP runtime as one model service and adds Qwen as a second service. The BDA contribution is capture, storage, distributed analysis, and a unified comparison interface.

The archived MAI evaluation reports **58.89% official VQA agreement** for the frozen Phase 4 system on a 3,319-item Windows holdout. This is an earlier labelled evaluation, **not the accuracy of the current live Qwen/BLIP BDA requests**. See submission/RESULTS_AND_PROVENANCE.md for the exact evaluation boundary and baseline differences.

## 6. Exact processing flow for one submission

1. The user uploads a JPG/PNG/WEBP image and enters a question. Typed text takes priority if audio and text are both present.
2. If audio is used, Streamlit uploads the WAV to a Windows speech endpoint and displays the recognized text.
3. Streamlit sends the image file and final question to the selected Windows model API. Both calls Qwen and BLIP sequentially.
4. Each model returns an answer, processing details, timing fields, and optional spoken audio.
5. The primary VM assigns a unique request ID to each model call. It writes a normalized JPEG and a JSON event to its local spool, then upserts the nested event into MongoDB.
6. The refresh service notices the new event and copies its image and JSON to HDFS. It also records HDFS paths in MongoDB.
7. The same service submits a PySpark job to YARN. Spark validates records, flags duplicates, creates summaries, and joins matching Qwen/BLIP requests.
8. Streamlit shows the live request immediately. The Spark panel changes from pending/running to complete after the distributed job finishes.

The VM spool and MongoDB update before Spark. A short delay between a new answer and refreshed cluster metrics is expected.

## 7. How request IDs and matched pairs work

The application normalizes the uploaded image by applying EXIF orientation, converting it to RGB, and saving a JPEG. It hashes those JPEG bytes with SHA-256 to form the **image ID**. It lowercases the question and collapses repeated whitespace, then hashes image ID plus normalized question to form the **pair ID**.

~~~text
image_id  = SHA256(normalized JPEG bytes)
pair_id   = SHA256(image_id + newline + normalized question)
request_id = a new UUID for each model call
run_id    = the experiment/model run, e.g. qwen-demo-v1 or blip-demo-v1
~~~

Two records share a pair ID only when the image content and normalized question match. They still have different request IDs and run IDs. Reusing the same image with a different question creates a different pair. Punctuation changes can also change the pair ID.

**Example:** Upload the Glucon-D package image, select Both, and ask “What is the net quantity?” The Qwen and BLIP records share a pair ID. One response may refer to the printed **200 g** net quantity. If the next question asks “What is six times the 30 g serving size?”, that is a new pair and the arithmetic result is **180 g**. Those are different questions and should not be mixed in one accuracy calculation.

## 8. What is stored and where

Each event stores: request ID, pair ID, image ID/reference, question, UTC timestamp, run/model/backend, route, prediction, final answer, completion/error status, abstention flag, OCR evidence, nested processing details, model start/completion timestamps, and processing time.

| Location | Contents | Purpose |
| --- | --- | --- |
| VM spool/events | One raw JSON event per model call | Local durable capture; watched for new work |
| VM spool/images | Content-addressed normalized JPEGs | Avoid duplicate image files |
| MongoDB visionbridge_bda.requests | Queryable nested request documents | Live queries, filters, audit details |
| HDFS /visionbridge/raw/events | Distributed raw event JSON | Source for PySpark request analysis |
| HDFS /visionbridge/raw/images | Distributed submitted images | Preserves unstructured source data |
| HDFS /visionbridge/analytics | Rejected, duplicate, curated, summary and matched output | Distributed analysis results |
| HDFS /visionbridge/dataset/vizwiz_annotations | 32 annotation data partitions plus manifest | Dataset distribution and Spark audit |

The JPEG and JSON files are published atomically, with an fsync before the final path is made visible. This was added after an interrupted VM run left empty final JSON files. MongoDB upserts use request ID as the unique key. The refresh worker retries failures and reports its state to the dashboard.

**MongoDB is not the same as HDFS:** MongoDB supports direct queries on nested request fields. HDFS stores the raw files and distributed Spark outputs. Docker packages the Streamlit Python application and runs MongoDB as a second container on the primary VM.

## 9. PySpark analytics and fairness rule

The request job reads raw HDFS JSON and checks that request ID, pair ID, run ID, backend, question, image ID, and timestamp are nonempty. Negative processing time is invalid. It writes rejected rows to a separate output.

It then keeps one row per unique request ID and one row per **pair ID + run ID**, flagging repeated rows. This prevents repeated submissions, multiple OCR candidates, or multiple human references from inflating run averages or matched comparisons. It writes a curated Parquet dataset.

From curated rows, Spark computes:

- Request count by run and backend.
- OCR coverage: how many records have nonempty recorded OCR evidence.
- Abstention rate.
- Mean and median recorded processing time.
- Matched BLIP/Qwen answers and whether their normalized answer strings differ.

The stored processing_seconds field comes from the model service's
callback_seconds timing. It describes that model callback, not the later HDFS
copy and Spark refresh time. OCR coverage counts nonempty **recorded** OCR
evidence; Qwen may read text internally without producing that field.

**Disagreement is not accuracy.** An official VizWiz score needs a prediction paired with the correct image/question and the ten original human answers under the verified scoring rules. The current live questions do not provide those labelled references, so no official accuracy percentage is claimed for the live comparison.

The optional scoring path checks for ten answers per labelled pair and one label record per pair. Its scoring unit is **one image-question-run row**, even if the underlying annotation contains ten human answers.

## 10. Distributed dataset: exact claim

The uploaded distributed selection consists of **9,319 VizWiz VQA annotation rows**, split as follows:

| Split | Annotation rows |
| --- | ---: |
| Train | 5,000 |
| Development | 1,000 |
| Holdout | 3,319 |
| **Total** | **9,319** |

The exporter validates the image name, question, and ten answer slots, then hashes rows into **32 JSONL partitions**. HDFS stores those partitions plus an export manifest with source checksum. A Spark audit on YARN reported **9,319 raw, 9,319 valid, zero rejected**. HDFS replication factor 2 distributes blocks across the two workers.

**Say this precisely:** These 9,319 rows are **annotations**, not 9,319 uploaded image files. The live images submitted in the app are separate files in HDFS /visionbridge/raw/images. Do not claim the full VizWiz image corpus is present in HDFS.

## 11. Verified results and what they mean

The most recent verified request snapshot after recovery on **25 October 2026** was:

| Check | Observed value |
| --- | ---: |
| Saved MongoDB requests | 19 |
| Valid raw event JSON files in VM spool/HDFS | 19 |
| Spark valid request rows | 19 |
| Rows flagged by deduplication | 1 |
| Curated request rows | 18 |
| Matched Qwen/BLIP pairs | 6 |
| Live HDFS DataNodes | 2 |
| Live YARN NodeManagers | 2 |
| Missing or under-replicated HDFS blocks | 0 |
| Distributed annotation rows / partitions | 9,319 / 32 |

The dashboard, Qwen endpoint and BLIP endpoint responded after recovery. Spark completed the 19-record refresh. These numbers are a snapshot, not fixed constants: a new answer should increase the saved request count, and the Spark figures update after the next job.

**Recovery note:** A primary VM freeze interrupted a Spark refresh. All 19 saved MongoDB requests survived. Two empty spool files were reconstructed from MongoDB; one additional zero-byte file had no MongoDB/HDFS copy and represented a request that had not completed capture. Its marker is preserved under spool/quarantine. The primary now has 2 GB swap and atomically publishes future events. Never promise that an in-progress request survives a machine failure; users can resubmit it.

## 12. Live demonstration plan (about 7–10 minutes)

### Before the examiner arrives

1. Start all three VMs and confirm the primary can reach both workers.
2. Start the Qwen and BLIP Windows model services.
3. On the primary, run the desktop launcher and watch that two DataNodes, two NodeManagers, MongoDB, dashboard, and model endpoints show UP.
4. Open the Streamlit page and prepare a sample image whose answer can be read directly from the image. Keep a second sample for an edge case.
5. If using microphone recording, open the localhost SSH tunnel first. A typed question is a reliable fallback.
6. Leave the terminal showing service status and the browser dashboard ready.

### Suggested speaking sequence

1. **Introduction (45 seconds):** State the accessibility and data-analysis problem from Section 2.
2. **Architecture (1 minute):** Point to Windows model inference, primary VM services, and two distributed workers. Explain why images/JSON go to HDFS and nested documents go to MongoDB.
3. **Input/output demo (2 minutes):** Upload the Glucon-D image, choose Both, ask “What is the net quantity printed on the package?”, click Analyze image once. Show both answer cards, optional audio, processing details, and two request IDs.
4. **Data capture (1 minute):** Open BDA analytics. Show live history, model/run filter, two answers with the same pair ID, and one nested stored document.
5. **Distributed proof (2 minutes):** Show two DataNodes, two NodeManagers, MongoDB count, HDFS raw event count, and 32 dataset partitions. Show the Spark job status changing to complete; it may take several minutes, so explain the delay.
6. **Results and limitations (1 minute):** State the measured snapshot, explain that disagreement is not accuracy, and distinguish the 9,319 annotations from live images.

### Useful edge-case questions

| Question on the Glucon-D image | What it tests |
| --- | --- |
| What is the net quantity? | Direct printed field; expected visible value 200 g |
| What is the serving size? | Correct label association; visible value 30 g |
| If there are six 30 g servings, what is their total? | Calculation: 180 g; do not confuse it with printed net quantity |
| What is the expiry date? | OCR/date reading and uncertainty about format if the print is unclear |
| What is the customer-care email? | Small-text reading; model should not invent characters it cannot read |
| Is the product safe to consume? | Requires more than a photo; an answer should state the limitation |

Treat these as demonstrations of behavior, not guaranteed correct model outputs. If a model gets a visible fact wrong, show the saved answer and discuss it as an error case rather than changing the claim.

## 13. Commands to prove the implementation

Run these on the **primary VM**:

~~~bash
bash ~/Desktop/start_visionbridge_bda.sh --status
hdfs dfsadmin -report | grep -E 'Live datanodes|Under replicated blocks|Missing blocks:|Hostname:'
yarn node -list
sudo docker ps --format '{{.Names}} {{.Status}}'
systemctl is-active visionbridge-bda-refresh.service
hdfs dfs -count /visionbridge/raw/events /visionbridge/raw/images
hdfs dfs -count /visionbridge/dataset/vizwiz_annotations
cat ~/Desktop/BDA/project/bda-data/analytics/pipeline_status.json
curl --fail http://127.0.0.1:8501/_stcore/health
~~~

Query MongoDB on the **primary VM**:

~~~bash
sudo docker exec -it mongodb mongosh visionbridge_bda
~~~

~~~javascript
db.requests.countDocuments({})
db.requests.aggregate([{ $group: { _id: "$run_id", requests: { $sum: 1 } } }])
db.requests.findOne({}, { request_id: 1, pair_id: 1, question: 1,
  final_answer: 1, ocr_evidence: 1, processing_details: 1, hdfs_event: 1 })
~~~

The Mongo document example proves that a request has queryable nested evidence and an HDFS event reference. The HDFS commands prove two live storage workers and actual distributed files. The YARN command proves two compute workers. The pipeline status file proves the latest request refresh completed.

### Starting the Windows model services

From the repository folder, start each service in its own Windows PowerShell window:

~~~powershell
$env:GRADIO_SERVER_NAME = '0.0.0.0'
$env:VISIONBRIDGE_BDA_RUN_ID = 'qwen-demo-v1'
$env:VISIONBRIDGE_OLLAMA_KEEP_ALIVE = '0'
.\.venv\Scripts\python.exe launch_workspace.py --backend local --port 7865
~~~

~~~powershell
$env:GRADIO_SERVER_NAME = '0.0.0.0'
$env:VISIONBRIDGE_BDA_RUN_ID = 'blip-demo-v1'
.\.venv\Scripts\python.exe launch_workspace.py --backend frozen --port 7866
~~~

The VM desktop script starts the cluster services. It only **checks** these Windows model endpoints; it cannot launch Windows processes remotely.

## 14. Rubric cross-check

| Criterion | Marks | What to show or say |
| --- | ---: | --- |
| Big Data problem statement | 2 | Section 2: a clear same-image/question model comparison, mixed image/JSON data, continuous capture, distributed validation |
| Docker and NoSQL | 2 | Running Streamlit and MongoDB containers; one nested request document and count/group queries |
| Streamlit interface | 1 | One-submit image/question form, Qwen/BLIP/Both, answer and audio, filters, history, Spark status |
| Dataset on Hadoop/PySpark | 2 | Two DataNodes, replication 2, raw images/events in HDFS, 32 annotation partitions, 9,319 valid Spark audit rows |
| Presentation and viva | 5 | Follow Section 12; explain every service and answer the questions below honestly |
| Formal report | 3 | Use the outline in Section 16, include measured outputs and your own explanation |

The evidence supports the technical criteria, but marks are awarded by the examiner. The rubric's Docker row also mentions NoSQL conversion/querying, so demonstrate **both** containerization and real MongoDB queries. If asked whether the entire VizWiz image dataset is uploaded, answer **no**: the annotation selection is distributed, and live sample images are stored separately.

## 15. Viva questions with concise answers

**1. What is the project's main goal?** Answer image questions and analyze the behavior of two model runs using stored, auditable data.

**2. Why is this Big Data Analytics rather than just a chatbot?** It captures unstructured images plus nested events, distributes data in HDFS, and uses PySpark to validate and summarize many request records. The current live count is only a pipeline demonstration.

**3. Where do Qwen and BLIP actually run?** On the Windows laptop. Streamlit, MongoDB, Hadoop controllers and the Spark driver run on the primary VM; storage and compute workers run on the other two VMs.

**4. Why not run the models inside the VMs?** The weights and GPU runtime are on Windows. The VMs have limited RAM and are allocated to Hadoop, Spark, and storage.

**5. What is the difference between Qwen and BLIP here?** Qwen is a general local vision-language model served by Ollama. BLIP is the project's frozen VizWiz-tuned VQA runtime. They are compared separately; one does not refine the other.

**6. Why use Both?** It sends the same uploaded image and question to both models and creates a fair matched pair.

**7. What is a pair ID?** A hash of the normalized image ID and normalized question. It groups model runs for identical input.

**8. Why is request ID different from pair ID?** Request ID identifies one model call. Pair ID identifies the shared input across runs.

**9. What is a run ID?** The experiment/model version, such as qwen-demo-v1 or blip-demo-v1. It prevents results from different model runs being mixed.

**10. Why MongoDB?** It stores nested request records and supports direct queries and indexes on IDs, runs, and evidence.

**11. Why HDFS?** It distributes raw images/events and analysis outputs across the two DataNodes with replication factor 2.

**12. Why Docker?** It packages the Python Streamlit dashboard and runs MongoDB predictably on the primary VM. Model inference remains on Windows.

**13. What does YARN do?** It schedules the Spark work on the worker NodeManagers.

**14. What does PySpark do?** It reads HDFS events, validates fields, flags duplicates, writes curated data, calculates run metrics, and joins matching Qwen/BLIP results.

**15. Why deduplicate before scoring?** Repeated requests or multiple evidence/answer rows must not inflate a model's average or matched pair count.

**16. Why is model disagreement not accuracy?** Two different answers can both be wrong. Accuracy requires verified reference answers under a defined scoring rule.

**17. What is in the 9,319-row dataset?** VizWiz annotation records with questions and human answers, distributed across 32 HDFS data partitions. It is not 9,319 image files.

**18. What happens after a new request?** The VM saves it immediately to the spool and MongoDB. The watcher later copies it to HDFS and runs Spark, then the dashboard displays completed cluster metrics.

**19. Why does the microphone work on localhost but not the VM HTTP address?** Browsers require a secure context for recording; loopback HTTP is treated specially. An SSH tunnel exposes the same Streamlit page through localhost.

**20. What are the project's main limitations?** Models can misread small/blurred text; live requests lack official reference labels; Qwen has no separately verified OCR evidence; the three VMs share one laptop and therefore have limited resources.

**21. What happened during the VM freeze?** A primary freeze interrupted capture/analysis. All 19 saved MongoDB records survived, two spool records were restored from MongoDB, an incomplete capture had no saved copy, and the 19-record Spark job completed after recovery. Atomic file publication and swap were added.

**22. What would you improve next?** More labelled paired examples, better OCR verification, resource isolation or a larger cluster, end-to-end monitoring, and a proper deployment with HTTPS for recording.

## 16. Report outline to write in your own words

1. **Title page:** project title, your name, course, institution, and date.
2. **Abstract:** accessibility problem, two-model comparison, three-node data pipeline, and measured result.
3. **Introduction and objectives:** user problem and the scoped BDA question.
4. **Related work and dataset:** VizWiz, VQA, OCR, Hadoop/Spark; cite original sources you actually used.
5. **System architecture:** the diagram in Section 4 plus machine roles.
6. **Tools and configuration:** Windows model services, Streamlit, Docker, MongoDB, HDFS, YARN, PySpark, network addresses and replication.
7. **Methodology:** input handling, capture schema, pair-ID rule, HDFS/Mongo storage, Spark validation, metrics and optional official scoring.
8. **Implementation:** key source files and commands; include a screenshot of the single Streamlit page, Mongo query, two DataNodes, and Spark status.
9. **Results:** current verified counts, dataset audit, one matched answer example, and qualitative error analysis. Do not invent an accuracy result for unlabelled live questions.
10. **Limitations and future work:** model errors, partial capture during failure, small live demonstration set, no full image corpus in HDFS.
11. **Conclusion and references:** restate what was demonstrated; provide citations for external datasets/models/frameworks.

Write the final report in your own voice, cite sources and screenshots, and check the similarity report required by your course. A tool cannot guarantee a particular plagiarism or AI-detection percentage.

## 17. Source map and follow-up reading

| File | Role |
| --- | --- |
| bda/dashboard.py | Streamlit input, output, history and analytics interface |
| bda/model_dispatch.py | Internal model API and audio transcription calls |
| bda/capture.py | Image normalization, IDs and atomic JPEG/JSON capture |
| bda/live_store.py | Immediate VM spool and MongoDB write |
| bda/ingest.py | HDFS file import and MongoDB HDFS references |
| bda/live_pipeline.py | Automatic refresh, retries and status publication |
| bda/vm_pipeline.sh | HDFS ingestion and YARN Spark submission |
| bda/spark_analysis.py | Request validation, deduplication and model comparison |
| bda/export_vizwiz_annotations.py | 32-partition annotation export |
| bda/spark_dataset_audit.py | Distributed annotation validation |
| bda/start_visionbridge_bda.sh | Primary desktop start and status monitor |
| modules/local_vision.py | Qwen/Ollama request and configurable model keep-alive |
| Dockerfile.bda and compose.bda.yaml | Dashboard container and standalone Docker setup |

For detailed operations, see [PROJECT_DETAILS.md](PROJECT_DETAILS.md) and [HOST_ONLY_NETWORK.md](HOST_ONLY_NETWORK.md). For archived MAI model measurements, see [submission/RESULTS_AND_PROVENANCE.md](submission/RESULTS_AND_PROVENANCE.md). [BDA_RUBRIC_EVIDENCE.md](BDA_RUBRIC_EVIDENCE.md) contains the initial technical evidence snapshot; some counts and bridged-network addresses in that older snapshot predate the stable network and 19-request recovery.
