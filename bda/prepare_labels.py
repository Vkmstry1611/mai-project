"""Prepare private VizWiz reference labels for the BDA Spark job.

Input JSONL rows: {"image_path": "...", "question": "...", "answers":
[{"answer": "..."}, ... ten original slots]}. Images are not copied here.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from bda.capture import image_and_pair_id


def prepare(source, destination):
    count = 0
    with Path(source).open(encoding="utf-8") as stream, Path(destination).open("x", encoding="utf-8") as output:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row["answers"]) != 10:
                raise ValueError("Each VizWiz row needs the original ten reference slots")
            with Image.open(row["image_path"]) as image:
                _, _, pair_id = image_and_pair_id(image, row["question"])
            output.write(json.dumps({"pair_id": pair_id, "answers": row["answers"]}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.output)} labelled pairs")


if __name__ == "__main__":
    main()
