"""Import the local capture spool into MongoDB before the VMs are connected."""
import argparse
import os

from bda.dashboard_data import read_spool


def import_spool(spool, uri, database="visionbridge_bda"):
    from pymongo import MongoClient

    rows = read_spool(spool)
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    try:
        collection = client[database]["requests"]
        collection.create_index("request_id", unique=True)
        collection.create_index([("run_id", 1), ("pair_id", 1)])
        for row in rows:
            collection.replace_one({"request_id": row["request_id"]}, row, upsert=True)
    finally:
        client.close()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", default=os.getenv("VISIONBRIDGE_BDA_SPOOL", "reports/bda/spool"))
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    args = parser.parse_args()
    print(f"Imported {import_spool(args.spool, args.mongo_uri)} records")


if __name__ == "__main__":
    main()
