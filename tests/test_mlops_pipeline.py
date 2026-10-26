"""Small checks for MLOps data integrity and deterministic model selection."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mlops.prepare import prepare
from mlops.select import select


class PipelineTests(unittest.TestCase):
    def test_prepare_records_image_hash_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            raw.mkdir()
            Image.new("RGB", (8, 8), "red").save(raw / "sign.png")
            (raw / "questions.json").write_text(
                json.dumps([{"image": "sign.png", "question": "What color?", "reference_answer": "red"}]),
                encoding="utf-8",
            )
            output = Path(temp) / "processed.json"
            result = prepare(raw, output)
            self.assertEqual(len(result["rows"][0]["image_sha256"]), 64)
            self.assertTrue(output.is_file())
            (raw / "questions.json").write_text(
                json.dumps([{"image": "../outside.png", "question": "What color?"}]), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                prepare(raw, output)

    def test_selection_requires_same_dataset_and_uses_declared_tie_break(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / "v1.json", Path(temp) / "v2.json"]
            for path, candidate in zip(paths, ("blip_vizwiz", "blip_vizwiz_v2")):
                path.write_text(json.dumps({
                    "candidate": candidate, "dataset_sha256": "same",
                    "model_sha256": candidate, "scored_questions": 13,
                    "exact_match": 10 / 13, "mean_latency_seconds": 1.0,
                }), encoding="utf-8")
            result = select(paths, Path(temp) / "selection.json")
            self.assertEqual(result["candidate"], "blip_vizwiz_v2")
            changed = json.loads(paths[1].read_text())
            changed["dataset_sha256"] = "different"
            paths[1].write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                select(paths, Path(temp) / "selection.json")


if __name__ == "__main__":
    unittest.main()
