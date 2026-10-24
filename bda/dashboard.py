"""Streamlit dashboard for captured requests and later Spark exports."""
import json
import os
from pathlib import Path

import streamlit as st

from bda.dashboard_data import read_mongo, read_spool, summarize


def main():
    st.set_page_config(page_title="VisionBridge BDA", page_icon="📊", layout="wide")
    st.title("VisionBridge · Big Data Analytics")
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
        st.info("No requests found. Ask VisionBridge a question, then refresh this dashboard. The VM pipeline can be run later.")
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


if __name__ == "__main__":
    main()
