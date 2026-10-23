"""Distributed VisionBridge request validation and BLIP/Qwen comparison.

Run with spark-submit on the VM cluster. No pretrained weights are required.
"""
import argparse
import json
from pathlib import Path


def analyze(spark, events_path, output_path, blip_run_id=None, qwen_run_id=None,
            labels_path=None, rules_path=None):
    from pyspark.sql import Window, functions as F

    raw = spark.read.option("multiLine", "true").json(events_path)
    required = ["request_id", "pair_id", "run_id", "backend", "question", "image_id", "timestamp_utc"]
    for field in required:
        if field not in raw.columns:
            raise ValueError(f"Missing event field: {field}")
    valid_condition = F.lit(True)
    for field in required:
        valid_condition = valid_condition & F.col(field).isNotNull() & (F.length(F.trim(F.col(field))) > 0)
    valid_condition = valid_condition & (F.col("processing_seconds").isNull() |
                                         (F.col("processing_seconds") >= 0))
    raw.withColumn("reject_reason", F.when(~valid_condition, "missing_or_invalid_field")) \
        .filter(F.col("reject_reason").isNotNull()) \
        .write.mode("overwrite").json(output_path.rstrip("/") + "/rejected")
    valid = raw.filter(valid_condition)
    # One request ID is one immutable event. A second event for the same
    # image/question/run is flagged, never allowed to inflate model averages.
    request_window = Window.partitionBy("request_id").orderBy("timestamp_utc")
    unique_requests = valid.withColumn("duplicate_request", F.row_number().over(request_window))
    duplicates = unique_requests.filter(F.col("duplicate_request") > 1)
    pair_window = Window.partitionBy("pair_id", "run_id").orderBy("timestamp_utc", "request_id")
    ranked = unique_requests.filter(F.col("duplicate_request") == 1) \
        .withColumn("pair_rank", F.row_number().over(pair_window))
    repeated = ranked.filter(F.col("pair_rank") > 1)
    duplicates.unionByName(repeated, allowMissingColumns=True) \
        .write.mode("overwrite").json(output_path.rstrip("/") + "/duplicates")
    curated = ranked.filter(F.col("pair_rank") == 1).drop("duplicate_request", "pair_rank")
    curated.write.mode("overwrite").parquet(output_path.rstrip("/") + "/curated")
    summary = curated.groupBy("run_id", "backend").agg(
        F.count("*").alias("requests"),
        F.avg(F.col("abstained").cast("double")).alias("abstention_rate"),
        F.avg("processing_seconds").alias("mean_processing_seconds"),
        F.expr("percentile_approx(processing_seconds, 0.5)").alias("median_processing_seconds"),
        F.sum(F.when(F.coalesce(F.size("ocr_evidence"), F.lit(0)) > 0, 1).otherwise(0)).alias("requests_with_ocr"),
    )
    summary.write.mode("overwrite").json(output_path.rstrip("/") + "/summary")
    # Matched comparison only: same pair_id present for both backends.
    completed = curated.filter(F.col("status") == "completed")
    blip_rows = completed.filter(F.col("run_id") == blip_run_id) if blip_run_id else completed.filter(F.col("backend").contains("blip"))
    qwen_rows = completed.filter(F.col("run_id") == qwen_run_id) if qwen_run_id else completed.filter(F.col("backend").contains("ollama"))
    if blip_rows.select("run_id").distinct().count() > 1 or qwen_rows.select("run_id").distinct().count() > 1:
        raise ValueError("Multiple BLIP or Qwen runs found; pass --blip-run-id and --qwen-run-id")
    blip = blip_rows \
        .select("pair_id", F.col("final_answer").alias("blip_answer"),
                F.col("processing_seconds").alias("blip_seconds"))
    qwen = qwen_rows \
        .select("pair_id", F.col("final_answer").alias("qwen_answer"),
                F.col("processing_seconds").alias("qwen_seconds"))
    matched = blip.join(qwen, "pair_id", "inner") \
        .withColumn("answers_differ", F.lower(F.trim("blip_answer")) != F.lower(F.trim("qwen_answer")))
    matched.write.mode("overwrite").parquet(output_path.rstrip("/") + "/matched_blip_qwen")
    labelled_pairs = None
    if labels_path:
        if not rules_path:
            raise ValueError("--rules is required when --labels is supplied")
        from evaluate_vqa import OfficialScorer
        rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        # Distribute the official scorer module to executor Python processes.
        spark.sparkContext.addPyFile(str(Path(__file__).resolve().parents[1] / "evaluate_vqa.py"))
        broadcast_rules = spark.sparkContext.broadcast(rules)

        @F.udf("double")
        def official_score(prediction, answers):
            if prediction is None or answers is None:
                return None
            refs = [answer.asDict(recursive=True) if hasattr(answer, "asDict") else answer
                    for answer in answers]
            return OfficialScorer(broadcast_rules.value).score(prediction, refs)

        labels = spark.read.json(labels_path)
        if not {"pair_id", "answers"}.issubset(labels.columns):
            raise ValueError("Labels need pair_id and ten answers")
        if labels.filter(F.size("answers") != 10).limit(1).count():
            raise ValueError("Every labelled pair needs ten original answer slots")
        if labels.groupBy("pair_id").count().filter(F.col("count") != 1).limit(1).count():
            raise ValueError("Duplicate pair_id in labels")
        labelled = matched.join(labels, "pair_id", "inner")
        scored = labelled.withColumn("blip_vqa", official_score("blip_answer", "answers")) \
            .withColumn("qwen_vqa", official_score("qwen_answer", "answers"))
        scored.write.mode("overwrite").parquet(output_path.rstrip("/") + "/matched_scored")
        scored.agg(F.count("*").alias("matched_labelled_pairs"),
                   F.avg("blip_vqa").alias("blip_official_vqa"),
                   F.avg("qwen_vqa").alias("qwen_official_vqa")) \
            .write.mode("overwrite").json(output_path.rstrip("/") + "/matched_score_summary")
        labelled_pairs = scored.count()
    return {"raw": raw.count(), "valid_requests": valid.count(),
            "duplicate_requests": duplicates.count() + repeated.count(),
            "curated": curated.count(), "matched_pairs": matched.count(),
            "matched_labelled_pairs": labelled_pairs}


def main():
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="hdfs:///visionbridge/raw/events/*.json")
    parser.add_argument("--output", default="hdfs:///visionbridge/analytics")
    parser.add_argument("--blip-run-id")
    parser.add_argument("--qwen-run-id")
    parser.add_argument("--labels", help="Optional HDFS JSONL with pair_id and ten VizWiz references")
    parser.add_argument("--rules", help="Local official VQA rules JSON used by evaluate_vqa.py")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("VisionBridgeBDA").getOrCreate()
    try:
        print(analyze(spark, args.events, args.output, args.blip_run_id, args.qwen_run_id,
                      args.labels, args.rules))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
