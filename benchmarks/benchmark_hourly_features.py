"""Benchmark hourly feature generation from the SW-088 Parquet subset."""

import argparse
import json
import platform
import shutil
import statistics
from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession, functions as F

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, choices=[1, 2], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

input_path = Path("/home/jovyan/work/data/processed/sw088_raw.parquet")
if not input_path.is_dir():
    raise FileNotFoundError(input_path)

# A new directory prevents overwriting an earlier benchmark.
args.output.mkdir(parents=True, exist_ok=False)

spark = (
    SparkSession.builder
    .master(f"local[{args.threads}]")
    .appName(f"HourlyFeaturesBenchmark-{args.threads}")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.sql.files.maxPartitionBytes", str(16 * 1024**2))
    .config("spark.sql.parquet.compression.codec", "snappy")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

sensors = ["S3", "S39", "S40", "S41", "S125", "S181", "S172", "S173"]


def hourly_features():
    readings = (
        spark.read.parquet(str(input_path))
        .filter(F.col("metric").isin(sensors))
        .withColumn("timestamp", F.timestamp_millis("when"))
    )

    return (
        readings
        .groupBy(
            "hwid",
            "metric",
            F.window("timestamp", "1 hour").alias("time_window"),
        )
        .agg(
            F.count("*").alias("n_readings"),
            F.countDistinct(
                F.date_trunc("minute", "timestamp")
            ).alias("minutes_with_readings"),
            F.avg("value").alias("mean_value"),
            F.min("value").alias("min_value"),
            F.max("value").alias("max_value"),
            F.stddev_samp("value").alias("std_value"),
        )
        .select(
            "hwid",
            "metric",
            F.col("time_window.start").alias("window_start"),
            F.col("time_window.end").alias("window_end"),
            "n_readings",
            "minutes_with_readings",
            "mean_value",
            "min_value",
            "max_value",
            "std_value",
        )
    )


timings = []

try:
    # One warm-up, then three measured runs in the same Spark process.
    for run in range(4):
        destination = args.output / f"features_run_{run}"
        spark.catalog.clearCache()

        start = perf_counter()
        hourly_features().write.mode("errorifexists").parquet(
            str(destination)
        )
        elapsed = perf_counter() - start

        phase = "warmup" if run == 0 else "measured"
        timings.append({
            "run": run,
            "phase": phase,
            "seconds": elapsed,
        })
        print(
            f"threads={args.threads} {phase} "
            f"run={run}: {elapsed:.3f} seconds",
            flush=True,
        )

        # Cleanup is outside the timer. Keep the last result for comparison.
        if run < 3:
            shutil.rmtree(destination)

    measured = [r["seconds"] for r in timings if r["phase"] == "measured"]

    report = {
        "threads": args.threads,
        "spark_version": spark.version,
        "python_version": platform.python_version(),
        "input": str(input_path),
        "input_bytes": sum(
            p.stat().st_size
            for p in input_path.rglob("*")
            if p.is_file()
        ),
        "shuffle_partitions": 8,
        "default_parallelism": 8,
        "max_input_partition_bytes": 16 * 1024**2,
        "adaptive_execution": False,
        "driver_memory": spark.sparkContext.getConf().get(
            "spark.driver.memory", "not explicitly set"
        ),
        "timing_scope": "Read, aggregate, and write all hourly features",
        "cache_policy": "No Spark persistence; OS cache is not cleared",
        "runs": timings,
        "median_seconds": statistics.median(measured),
        "min_seconds": min(measured),
        "max_seconds": max(measured),
        "retained_features": str(args.output / "features_run_3"),
    }

    (args.output / "timings.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"\nMedian: {report['median_seconds']:.3f} seconds")

finally:
    spark.stop()