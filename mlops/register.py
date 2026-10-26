"""Materialize a pinned DVC registry artifact from the selected local model."""

import argparse
import json
import shutil
from pathlib import Path

from mlops.evaluate import CANDIDATES
from mlops.model import model_sha256


def register(selection_path: Path, destination: Path) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidate = selection["candidate"]
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate}")
    source = CANDIDATES[candidate]
    if model_sha256(source) != selection["model_sha256"]:
        raise ValueError("Selected checkpoint changed after evaluation")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite registry artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if model_sha256(destination) != selection["model_sha256"]:
        raise RuntimeError("Copied checkpoint hash mismatch")
    metadata = {
        "schema": 1,
        "candidate": candidate,
        "model_sha256": selection["model_sha256"],
        "selection_rule": selection["selection_rule"],
        "selection_dataset_sha256": selection["dataset_sha256"],
        "evaluation_scope": selection["scope"],
    }
    (destination / "registry.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("mlops/registry/selection.json"))
    parser.add_argument("--destination", type=Path, default=Path("mlops/registry/champion"))
    args = parser.parse_args()
    print("Registered", register(args.selection, args.destination)["candidate"])
