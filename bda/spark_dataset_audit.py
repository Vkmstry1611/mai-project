"""Audit distributed VizWiz annotation partitions on the YARN cluster."""
import argparse


def audit(spark, source, output):
    from pyspark.sql import functions as F

    rows = spark.read.json(source)
    required = {"image", "question", "answers", "split", "answerable"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing annotation fields: {sorted(required - set(rows.columns))}")
    clean = rows.filter(F.col("image").isNotNull() & F.col("question").isNotNull()
                        & (F.size("answers") == 10))
    summary = clean.groupBy("split").agg(
        F.count("*").alias("rows"),
        F.sum(F.when(F.col("answerable") == 1, 1).otherwise(0)).alias("answerable_rows"),
        F.countDistinct("image").alias("distinct_images"),
    )
    summary.write.mode("overwrite").json(output.rstrip("/") + "/by_split")
    return {"raw": rows.count(), "valid": clean.count(), "rejected": rows.count() - clean.count()}


def main():
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="hdfs:///visionbridge/dataset/vizwiz_annotations/part-*.jsonl")
    parser.add_argument("--output", default="hdfs:///visionbridge/dataset-audit")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("VisionBridgeVizWizAudit").getOrCreate()
    try:
        print(audit(spark, args.source, args.output))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
