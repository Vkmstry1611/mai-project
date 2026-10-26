"""Select the demo champion using a declared, deterministic comparison rule."""

import argparse
import json
from pathlib import Path


def select(inputs: list[Path], output: Path) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if len(runs) < 2 or len({run["candidate"] for run in runs}) != len(runs):
        raise ValueError("At least two distinct candidate runs are required")
    if len({run["dataset_sha256"] for run in runs}) != 1:
        raise ValueError("Candidates must use the same prepared dataset")
    # Ties are resolved by a predeclared preference for the current research
    # checkpoint, not by reusing the official holdout to tune this demo.
    priority = {"blip_vizwiz": 0, "blip_vizwiz_v2": 1}
    champion = max(runs, key=lambda run: (run["exact_match"], priority[run["candidate"]]))
    result = {
        "schema": 1,
        "scope": "manual_demo_only_not_official_vizwiz_model_selection",
        "selection_rule": "highest_demo_exact_match_then_predeclared_v2_priority",
        "dataset_sha256": champion["dataset_sha256"],
        "candidate": champion["candidate"],
        "model_sha256": champion["model_sha256"],
        "exact_match": champion["exact_match"],
        "comparisons": [
            {k: run[k] for k in ("candidate", "model_sha256", "scored_questions", "exact_match", "mean_latency_seconds")}
            for run in runs
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("mlops/registry/selection.json"))
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print("Selected", select(args.runs, args.output)["candidate"])
