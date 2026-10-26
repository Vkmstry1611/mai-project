"""Run one DVC-tracked candidate on the attributed manual demo set."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from mlops.model import load_model, model_sha256, predict


CANDIDATES = {
    "blip_vizwiz": Path("models/blip_vizwiz"),
    "blip_vizwiz_v2": Path("models/blip_vizwiz_v2"),
}


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def evaluate(candidate: str, processed: Path, raw_dir: Path, output: Path) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate}")
    payload = processed.read_bytes()
    data = json.loads(payload)
    if data["dataset_kind"] != "attributed_manual_demo_not_official_vizwiz":
        raise ValueError("Unexpected evaluation dataset")
    processor, model = load_model(CANDIDATES[candidate])
    records = []
    for row in data["rows"]:
        image_path = raw_dir / row["image"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed after preparation: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        started = time.perf_counter()
        answer = predict(processor, model, image, row["question"])
        elapsed = time.perf_counter() - started
        reference = row["reference_answer"]
        records.append({
            "image": row["image"], "question": row["question"],
            "reference": reference, "prediction": answer,
            "exact_match": None if reference is None else normalized(answer) == normalized(reference),
            "latency_seconds": round(elapsed, 3),
        })
    scored = [r for r in records if r["exact_match"] is not None]
    result = {
        "schema": 1,
        "candidate": candidate,
        "model_sha256": model_sha256(CANDIDATES[candidate]),
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "dataset_kind": data["dataset_kind"],
        "scored_questions": len(scored),
        "exact_match": sum(r["exact_match"] for r in scored) / len(scored),
        "mean_latency_seconds": sum(r["latency_seconds"] for r in records) / len(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--processed", type=Path, default=Path("mlops/data/processed.json"))
    parser.add_argument("--raw", type=Path, default=Path("mlops/data/raw"))
    parser.add_argument("--output", type=Path, default=Path("mlops/experiments/metrics.json"))
    args = parser.parse_args()
    result = evaluate(args.candidate, args.processed, args.raw, args.output)
    print(f"{result['candidate']}: {result['exact_match']:.3f} demo exact match")
