import json

from bda.dashboard_data import history_rows, read_spool, summarize


def test_dashboard_uses_real_events_and_matches_pairs(tmp_path):
    folder = tmp_path / "events"
    folder.mkdir()
    base = {"pair_id": "same", "question": "What is it?", "status": "completed",
            "ocr_evidence": [], "abstained": False}
    rows = [dict(base, request_id="1", run_id="blip-v1", backend="blip",
                 final_answer="cat", processing_seconds=2.0),
            dict(base, request_id="2", run_id="qwen-v1", backend="ollama",
                 final_answer="dog", processing_seconds=4.0)]
    for row in rows:
        (folder / f'{row["request_id"]}.json').write_text(json.dumps(row), encoding="utf-8")
    actual = summarize(read_spool(tmp_path))
    assert actual["requests"] == 2
    assert actual["mean_seconds"] == 3.0
    assert len(actual["matched"]) == 1
    assert actual["matched"][0]["different"] is True


def test_history_uses_completion_time_and_falls_back_for_old_events():
    rows = [
        {"request_id": "old", "backend": "blip", "timestamp_utc": "2026-10-25T06:00:00Z",
         "question": "What is it?", "final_answer": "cat", "status": "completed"},
        {"request_id": "new", "backend": "Ollama on this computer", "timestamp_utc": "2026-10-25T06:01:00Z",
         "processing_completed_at_utc": "2026-10-25T06:02:03+00:00",
         "question": "What color?", "final_answer": "blue", "status": "completed",
         "processing_seconds": 2.345},
    ]
    history = history_rows(rows)
    assert [row["Request ID"] for row in history] == ["new", "old"]
    assert history[0]["Date (IST)"] == "25 Oct 2026"
    assert history[0]["Time (IST)"] == "11:32:03 AM"
    assert history[0]["Time source"] == "Model completed"
    assert history[0]["Model"] == "Qwen"
    assert history[0]["Duration (s)"] == 2.35
    assert history[1]["Time source"] == "Captured"
    assert history[1]["Time (IST)"] == "11:30:00 AM"
