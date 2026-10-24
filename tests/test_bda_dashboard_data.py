import json

from bda.dashboard_data import read_spool, summarize


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
