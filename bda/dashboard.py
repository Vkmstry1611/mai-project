"""Streamlit dashboard for captured requests and later Spark exports."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from bda.dashboard_data import read_mongo, read_spool, summarize


def assistant_panel():
    st.subheader("Ask VisionBridge")
    st.caption("Upload an image and ask a question in the assistant below. The model runs on your Windows machine.")
    model = st.radio("Model", ["Qwen", "BLIP"], horizontal=True,
                     help="Select the model that will answer your image question.")
    default_url = (os.getenv("VISIONBRIDGE_QWEN_URL", "http://192.168.1.83:7865")
                   if model == "Qwen" else
                   os.getenv("VISIONBRIDGE_BLIP_URL", "http://192.168.1.83:7866"))
    url = st.text_input("VisionBridge assistant URL",
                        value=default_url, key=f"assistant_url_{model.lower()}",
                        help="Use the Windows machine's LAN address and the selected model's port.")
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        st.error("Enter a full http:// or https:// assistant URL.")
        return
    st.caption(f"Selected model: {model}. Requests from this assistant are captured with its run ID.")
    st.link_button(f"Open {model} assistant in its own tab", url)
    iframe = getattr(st, "iframe", components.iframe)
    iframe(url, height=1100)
    with st.expander("If the assistant does not load"):
        st.write("Start the selected VisionBridge model on Windows with GRADIO_SERVER_NAME set to 0.0.0.0, then use that machine's LAN address above. The assistant must stay running while you use this page.")
        st.write("New answers are captured on Windows. Transfer the capture spool to the VM and run the BDA pipeline to update the analytics tab.")


def analytics_panel():
    st.caption("Request evidence and matched BLIP/Qwen comparisons. Accuracy is shown only after labelled Spark evaluation.")
    source = st.sidebar.radio("Data source", ["Local capture", "MongoDB"], index=0)
    spool = os.getenv("VISIONBRIDGE_BDA_SPOOL", "reports/bda/spool")
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    try:
        events = read_spool(spool) if source == "Local capture" else read_mongo(uri)
    except Exception as error:
        st.error(f"Could not read {source}: {error}")
        return
    if not events:
        st.info("No requests are stored in this VM data source yet. Ask a question, transfer the Windows capture spool to the VM, then refresh. Run the VM pipeline to add HDFS and Spark results.")
        return
    choices = sorted({str(row.get("run_id") or "unknown") for row in events})
    selected = st.sidebar.multiselect("Run ID", choices, default=choices)
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
    st.subheader("Request records")
    st.dataframe([{key: row.get(key) for key in ("request_id", "run_id", "backend", "route", "question", "status", "abstained", "processing_seconds")}
                  for row in rows], use_container_width=True, hide_index=True)
    with st.expander("Inspect one stored request"):
        ids = [row["request_id"] for row in rows]
        chosen = st.selectbox("Request ID", ids)
        st.json(next(row for row in rows if row["request_id"] == chosen))
    spark_summary = Path(os.getenv("VISIONBRIDGE_SPARK_SUMMARY", "bda-data/analytics/matched_score_summary.json"))
    st.subheader("Cluster evaluation")
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
