"""Copy captured requests into HDFS and MongoDB from a configured VM.

Usage: python -m bda.ingest --spool /path/to/spool --hdfs-root /visionbridge/raw
Requires the `hdfs` CLI and `pymongo` on the machine running this command.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def hdfs(*args):
    subprocess.run(["hdfs", "dfs", *map(str, args)], check=True)


def ingest(spool, hdfs_root, mongo_uri, database="visionbridge_bda"):
    from pymongo import MongoClient

    spool = Path(spool)
    events = sorted((spool / "events").glob("*.json"))
    if not events:
        raise ValueError(f"No captured events under {spool / 'events'}")
    root = hdfs_root.rstrip("/")
    hdfs("-mkdir", "-p", f"{root}/images", f"{root}/events")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    collection = client[database]["requests"]
    collection.create_index("request_id", unique=True)
    collection.create_index([("run_id", 1), ("pair_id", 1)])
    count = 0
    for path in events:
        event = json.loads(path.read_text(encoding="utf-8"))
        image = spool / event["image_file"]
        if not image.is_file():
            raise FileNotFoundError(image)
        # Images are content addressed; repeats can safely overwrite identical bytes.
        hdfs("-put", "-f", image, f"{root}/images/{image.name}")
        hdfs("-put", "-f", path, f"{root}/events/{path.name}")
        document = dict(event, hdfs_image=f"{root}/images/{image.name}",
                        hdfs_event=f"{root}/events/{path.name}")
        collection.replace_one({"request_id": event["request_id"]}, document, upsert=True)
        count += 1
    client.close()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", default=os.getenv("VISIONBRIDGE_BDA_SPOOL", "reports/bda/spool"))
    parser.add_argument("--hdfs-root", default=os.getenv("VISIONBRIDGE_HDFS_ROOT", "/visionbridge/raw"))
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--database", default="visionbridge_bda")
    args = parser.parse_args()
    print(f"Ingested {ingest(args.spool, args.hdfs_root, args.mongo_uri, args.database)} requests")


if __name__ == "__main__":
    main()
