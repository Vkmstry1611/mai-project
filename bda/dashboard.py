"""Streamlit dashboard for captured requests and later Spark exports."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from bda.dashboard_data import history_rows, read_mongo, read_spool, summarize
from bda.live_store import store_result
from bda.model_dispatch import run_models, transcribe_audio


def model_url(model):
    if model == "Qwen":
        return os.getenv("VISIONBRIDGE_QWEN_URL", "http://192.168.1.83:7865")
    return os.getenv("VISIONBRIDGE_BLIP_URL", "http://192.168.1.83:7866")


def valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def microphone_origin_available(page_url):
    """Browsers permit recording on HTTPS and loopback HTTP origins."""
    if not page_url:
        return True  # Streamlit's AppTest has no browser URL.
    parsed = urlparse(page_url)
    return parsed.scheme == "https" or parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def assistant_panel():
    st.subheader("Ask VisionBridge")
    st.caption("Upload an image, ask a question, and choose which model answers.")
    model = st.radio("Model", ["Qwen", "BLIP", "Both"], horizontal=True,
                     help="Select the model that will answer your image question.")
    with st.expander("Connection settings"):
        qwen_url = st.text_input("Qwen service URL", value=model_url("Qwen")).strip().rstrip("/")
        blip_url = st.text_input("BLIP service URL", value=model_url("BLIP")).strip().rstrip("/")
    endpoints = {name: url for name, url in (("Qwen", qwen_url), ("BLIP", blip_url))
                 if model in (name, "Both")}
    if not all(valid_url(url) for url in endpoints.values()):
        st.error("Enter a full http:// or https:// URL for each selected model in Connection settings.")
        return
    with st.form("visionbridge_question"):
        photo = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
        question = st.text_area("Type your question", max_chars=1000,
                                placeholder="Ask about the image, or record your question below")
        if microphone_origin_available(getattr(st.context, "url", None)):
            recorded = st.audio_input("Or record your question")
        else:
            recorded = None
            st.info("To record, open this page at http://127.0.0.1:8501 in Chrome or Edge. "
                    "Keep the SSH tunnel running. Typing and WAV upload work here.")
        with st.expander("Have a WAV recording? Upload it here"):
            uploaded_audio = st.file_uploader("WAV recording", type=["wav"], key="question_audio")
        speak = st.checkbox("Read answers aloud", value=True)
        submit = st.form_submit_button("Analyze image")
    if submit:
        if photo is None:
            st.error("Upload an image first.")
        else:
            final_question = question.strip()
            audio_input = recorded or uploaded_audio
            if not final_question and audio_input is not None:
                try:
                    speech_url = qwen_url if model in ("Qwen", "Both") else blip_url
                    with st.spinner("Transcribing your question…"):
                        final_question = transcribe_audio(
                            audio_input.getvalue(), audio_input.name, speech_url)
                    st.caption(f"Heard: {final_question}")
                except Exception as error:
                    st.error(f"Could not transcribe the recording: {error}")
            if not final_question:
                st.error("Type or record a question, then try again.")
                return
            image_bytes = photo.getvalue()
            spool = os.getenv("VISIONBRIDGE_BDA_SPOOL", "reports/bda/spool")
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            run_ids = {"Qwen": os.getenv("VISIONBRIDGE_QWEN_RUN_ID", "qwen-demo-v1"),
                       "BLIP": os.getenv("VISIONBRIDGE_BLIP_RUN_ID", "blip-demo-v1")}
            results = []
            with st.spinner(f"Running {model}…"):
                for name, url in endpoints.items():
                    result = run_models(image_bytes, photo.name, final_question,
                                        {name: url}, speak=speak)[0]
                    try:
                        event = store_result(image_bytes, final_question, result, spool,
                                             mongo_uri, run_ids=run_ids)
                        result["stored_request_id"] = event["request_id"]
                    except Exception as error:
                        result["storage_error"] = str(error)
                    results.append(result)
            st.session_state["last_results"] = results
            st.session_state["last_input"] = (photo.name, final_question, model)
    if "last_results" in st.session_state:
        filename, previous_question, previous_model = st.session_state["last_input"]
        st.caption(f"Last run · {previous_model} · {filename} · {previous_question}")
        for column, result in zip(st.columns(len(st.session_state["last_results"])),
                                  st.session_state["last_results"]):
            with column:
                st.subheader(result["model"])
                if result["status"] == "completed":
                    st.write(result["answer"])
                    if result.get("audio"):
                        st.audio(result["audio"], format="audio/wav")
                    elif result.get("speech_error"):
                        st.caption(f"Audio unavailable: {result['speech_error']}")
                else:
                    st.error(result.get("error") or "The model did not return an answer. Check its service and request record.")
                processed = history_rows([result])[0]
                if processed["Time source"] == "Model completed":
                    st.caption(f"Processed {processed['Date (IST)']} at {processed['Time (IST)']} IST")
                if result.get("storage_error"):
                    st.warning(f"Could not confirm storage: {result['storage_error']}")
                elif result.get("stored_request_id"):
                    st.caption(f"Saved request {result['stored_request_id'][:12]} · HDFS/Spark refresh queued")
                with st.expander("Processing details"):
                    st.json(result["details"])
                    st.json(result["timings"])
        st.caption("Each result is saved on the VM immediately. HDFS and Spark refresh in the background.")


def analytics_panel():
    st.caption("Request evidence and matched BLIP/Qwen comparisons. Accuracy is shown only after labelled Spark evaluation.")
    source = st.sidebar.radio("Data source", ["Local capture", "MongoDB"], index=0)
    analytics_results(source)


@st.fragment(run_every="5s")
def analytics_results(source):
    spool = os.getenv("VISIONBRIDGE_BDA_SPOOL", "reports/bda/spool")
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    try:
        events = read_spool(spool) if source == "Local capture" else read_mongo(uri)
    except Exception as error:
        st.error(f"Could not read {source}: {error}")
        return
    if not events:
        st.info("No requests are stored yet. Submit an image question in Ask VisionBridge.")
        return
    choices = sorted({str(row.get("run_id") or "unknown") for row in events})
    selected = st.multiselect("Run ID", choices, default=choices)
    rows = [row for row in events if str(row.get("run_id") or "unknown") in selected]
    summary = summarize(rows)
    a, b, c, d = st.columns(4)
    a.metric("Requests", summary["requests"])
    b.metric("Completed", summary["completed"])
    c.metric("Errors", summary["errors"])
    d.metric("Mean processing time", f'{summary["mean_seconds"]:.2f} s' if summary["mean_seconds"] is not None else "—")
    st.subheader("Runs")
    st.dataframe(summary["runs"], use_container_width=True, hide_index=True)
    if summary["runs"]:
        st.bar_chart(summary["runs"], x="run_id", y="requests")
    st.subheader("Matched BLIP and Qwen requests")
    st.caption("Only identical image/question pair IDs appear here. Answer disagreement is not an accuracy measure.")
    if summary["matched"]:
        st.dataframe(summary["matched"], use_container_width=True, hide_index=True)
    else:
        st.info("No matched pair yet. Run the same image and question through each backend.")
    st.subheader("Request history")
    st.caption("Newest first · India Standard Time. Older records show capture time; new records show model completion time.")
    search = st.text_input("Search question, answer or request ID", key="history_search")
    history = history_rows(rows)
    if search.strip():
        term = search.strip().casefold()
        history = [row for row in history if any(term in str(row[key]).casefold()
                   for key in ("Question", "Answer", "Request ID"))]
    st.dataframe(history, use_container_width=True, hide_index=True)
    with st.expander("Inspect one stored request"):
        ids = [row["Request ID"] for row in history]
        if ids:
            chosen = st.selectbox("Request ID", ids)
            st.json(next(row for row in rows if row["request_id"] == chosen))
        else:
            st.info("No requests match the current filters.")
    st.subheader("Cluster evaluation")
    status_path = Path(os.getenv("VISIONBRIDGE_PIPELINE_STATUS", "bda-data/analytics/pipeline_status.json"))
    summary_path = Path(os.getenv("VISIONBRIDGE_CLUSTER_SUMMARY", "bda-data/analytics/cluster_summary.json"))
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = status.get("state")
            processed = status.get("event_count", 0)
            if state == "running":
                st.info(f"Spark refresh running for {processed} requests.")
            elif state == "complete" and processed >= len(events):
                st.success(f"Spark is up to date for {processed} requests.")
            elif state == "error":
                st.error(f"Spark refresh failed: {status.get('error', 'unknown error')}")
            else:
                st.info(f"Spark refresh pending: {len(events)} captured, {processed} processed.")
        except (OSError, ValueError) as error:
            st.error(f"Could not read pipeline status: {error}")
    else:
        st.info("Spark refresh has not run yet.")
    if summary_path.is_file():
        try:
            cluster = json.loads(summary_path.read_text(encoding="utf-8"))
            st.caption(f"Last cluster refresh: {cluster.get('refreshed_at_utc', 'unknown')}")
            st.dataframe(cluster.get("runs", []), use_container_width=True, hide_index=True)
        except (OSError, ValueError) as error:
            st.error(f"Could not read Spark summary: {error}")
    dataset_path = Path(os.getenv("VISIONBRIDGE_DATASET_AUDIT", "bda-data/analytics/dataset_audit.json"))
    if dataset_path.is_file():
        try:
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            st.subheader("Distributed VizWiz annotations")
            st.caption(f"{dataset.get('total_rows', 0):,} annotation rows across "
                       f"{dataset.get('partitions', 0)} HDFS data partitions")
            st.dataframe(dataset.get("by_split", []), use_container_width=True, hide_index=True)
        except (OSError, ValueError) as error:
            st.error(f"Could not read dataset audit: {error}")
    spark_summary = Path(os.getenv("VISIONBRIDGE_SPARK_SUMMARY", "bda-data/analytics/matched_score_summary.json"))
    if spark_summary.is_file():
        st.caption("Imported Spark output from the VM cluster")
        try:
            st.json(json.loads(spark_summary.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            st.error(f"Could not read the Spark export: {error}")
    else:
        st.info("No Spark score export has been imported yet. This section will populate after the VM run.")


def main():
    st.set_page_config(page_title="VisionBridge BDA", page_icon="📊", layout="wide")
    st.title("VisionBridge · Big Data Analytics")
    ask, analytics = st.tabs(["Ask VisionBridge", "BDA analytics"])
    with ask:
        assistant_panel()
    with analytics:
        analytics_panel()


if __name__ == "__main__":
    main()
