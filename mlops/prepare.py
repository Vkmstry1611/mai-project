"""Validate the small, attributed demonstration dataset for pipeline checks.

This is not the official VizWiz evaluation split.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def prepare(raw_dir: Path, output: Path) -> dict:
    questions = json.loads((raw_dir / "questions.json").read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a nonempty questions list")
    rows = []
    for row in questions:
        name = row["image"]
        if Path(name).name != name:
            raise ValueError(f"Unsafe image path: {name}")
        image = raw_dir / name
        if not image.is_file():
            raise FileNotFoundError(image)
        with Image.open(image) as opened:
            opened.verify()
        question = str(row["question"]).strip()
        if not question:
            raise ValueError("Empty question")
        rows.append({
            "image": name,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "question": question,
            "reference_answer": row.get("reference_answer"),
            "category": row.get("category", "unspecified"),
        })
    result = {
        "schema": 1,
        "dataset_kind": "attributed_manual_demo_not_official_vizwiz",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("mlops/data/raw"))
    parser.add_argument("--output", type=Path, default=Path("mlops/data/processed.json"))
    args = parser.parse_args()
    print(f"Prepared {len(prepare(args.raw, args.output)['rows'])} demo questions")
