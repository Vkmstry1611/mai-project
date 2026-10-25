"""Partition the existing VizWiz annotation selection for HDFS ingestion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def export_annotations(manifest_path, output_dir, partitions=32):
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    source = manifest_path.read_bytes()
    manifest = json.loads(source)
    if manifest.get("dataset") != "official VizWiz-VQA":
        raise ValueError("Expected the project's official VizWiz-VQA selection")
    if partitions < 2:
        raise ValueError("Use at least two partitions for distributed storage")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"part-{index:03d}.jsonl" for index in range(partitions)]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    counts = Counter()
    try:
        for split in ("train", "dev", "holdout"):
            for row in manifest[split]:
                if not row.get("image") or not row.get("question") or len(row.get("answers", [])) != 10:
                    raise ValueError(f"Invalid {split} annotation row")
                record = dict(row, split=split)
                key = f"{split}/{row['image']}/{row['question']}".encode("utf-8")
                index = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % partitions
                handles[index].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for handle in handles:
            handle.close()
    summary = {"dataset": manifest["dataset"], "selection": "annotation records only",
               "source_sha256": hashlib.sha256(source).hexdigest(),
               "source_manifest": str(manifest_path), "partitions": partitions,
               "split_counts": dict(counts), "total_rows": sum(counts.values())}
    (output_dir / "export_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output_dir")
    parser.add_argument("--partitions", type=int, default=32)
    args = parser.parse_args()
    print(export_annotations(args.manifest, args.output_dir, args.partitions))


if __name__ == "__main__":
    main()
