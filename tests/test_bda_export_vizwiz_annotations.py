import json

from bda.export_vizwiz_annotations import export_annotations


def test_export_annotations_preserves_rows_and_ten_answers(tmp_path):
    row = {"image": "one.jpg", "question": "What is shown?",
           "answers": [{"answer": "cat"}] * 10, "answer_type": "other", "answerable": 1}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps({"dataset": "official VizWiz-VQA",
                                  "train": [row], "dev": [row], "holdout": [row]}))
    output = tmp_path / "parts"
    summary = export_annotations(source, output, partitions=4)
    records = [json.loads(line) for part in output.glob("part-*.jsonl")
               for line in part.read_text(encoding="utf-8").splitlines()]
    assert summary["total_rows"] == 3
    assert {item["split"] for item in records} == {"train", "dev", "holdout"}
    assert all(len(item["answers"]) == 10 for item in records)
