"""Read captured requests and calculate clearly labelled local dashboard metrics."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import mean


IST = timezone(timedelta(hours=5, minutes=30))


def event_time(row):
    """Prefer exact model completion; older events only have capture time."""
    value = row.get("processing_completed_at_utc") or row.get("timestamp_utc")
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment
    except ValueError:
        return None


def history_rows(events):
    """Newest first, with readable India date/time and the original request ID."""
    ordered = sorted(events, key=lambda row: event_time(row) or datetime.min.replace(
        tzinfo=timezone.utc), reverse=True)
    result = []
    for row in ordered:
        moment = event_time(row)
        local = moment.astimezone(IST) if moment else None
        seconds = row.get("processing_seconds")
        backend = str(row.get("backend") or "").lower()
        model = "Qwen" if "ollama" in backend or "qwen" in backend else "BLIP" if "blip" in backend else (row.get("model") or backend or "Unknown")
        result.append({
            "Date (IST)": local.strftime("%d %b %Y") if local else "—",
            "Time (IST)": local.strftime("%I:%M:%S %p") if local else "—",
            "Model": model,
            "Question": row.get("question") or "",
            "Answer": row.get("final_answer") or "",
            "Status": row.get("status") or "unknown",
            "Duration (s)": round(float(seconds), 2) if isinstance(seconds, (int, float)) else None,
            "Time source": "Model completed" if row.get("processing_completed_at_utc") else "Captured",
            "Request ID": row.get("request_id") or "",
        })
    return result


def read_spool(spool):
    folder = Path(spool) / "events"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]


def read_mongo(uri, database="visionbridge_bda"):
    from pymongo import MongoClient

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        rows = list(client[database]["requests"].find({}, {"_id": 0}))
    finally:
        client.close()
    return rows


def summarize(events):
    """Metrics are descriptive request counts, not model accuracy."""
    rows = list(events)
    seconds = [float(row["processing_seconds"]) for row in rows
               if isinstance(row.get("processing_seconds"), (int, float)) and
               row["processing_seconds"] >= 0]
    completed = [row for row in rows if row.get("status") == "completed"]
    runs = defaultdict(list)
    for row in rows:
        runs[row.get("run_id") or "unknown"].append(row)
    run_summary = []
    for run_id, group in sorted(runs.items()):
        successful = [row for row in group if row.get("status") == "completed"]
        run_seconds = [float(row["processing_seconds"]) for row in group
                       if isinstance(row.get("processing_seconds"), (int, float)) and
                       row["processing_seconds"] >= 0]
        run_summary.append({"run_id": run_id, "requests": len(group),
                            "completed": len(successful),
                            "abstentions": sum(bool(row.get("abstained")) for row in successful),
                            "with_ocr": sum(bool(row.get("ocr_evidence")) for row in group),
                            "mean_seconds": round(mean(run_seconds), 2) if run_seconds else None})
    pairs = defaultdict(dict)
    for row in completed:
        backend = str(row.get("backend", "")).lower()
        family = "qwen" if "ollama" in backend or "qwen" in backend else "blip" if "blip" in backend else None
        if family and row.get("pair_id"):
            pairs[row["pair_id"]].setdefault(family, row)
    matched = []
    for pair_id, pair in pairs.items():
        if {"blip", "qwen"}.issubset(pair):
            blip, qwen = pair["blip"], pair["qwen"]
            matched.append({"pair_id": pair_id[:12], "question": blip.get("question", ""),
                            "blip": blip.get("final_answer", ""),
                            "qwen": qwen.get("final_answer", ""),
                            "different": str(blip.get("final_answer", "")).strip().lower() !=
                                         str(qwen.get("final_answer", "")).strip().lower()})
    return {"requests": len(rows), "completed": len(completed),
            "errors": len(rows) - len(completed),
            "mean_seconds": round(mean(seconds), 2) if seconds else None,
            "runs": run_summary, "matched": matched}
