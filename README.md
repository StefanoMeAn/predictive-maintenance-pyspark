# Predictive Maintenance with PySpark

I used PySpark to explore approximately **142 million industrial sensor records** and investigate whether recent sensor measurements could help predict recorded overheating alarms.

The analysis focuses on device **SW-088** and alarm channels **A5 and A9**. It covers data quality, hourly feature engineering, chronological model evaluation, and the limitations of working with sparse alarm observations.

The main lesson: a large sensor dataset does not necessarily provide enough independent failure examples for reliable prediction.

## Analysis

The main notebook is [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb).

1. Read the CSV with an explicit schema and check missing values, timestamps, and device coverage.
2. Save the SW-088 subset as Parquet for reuse.
3. Decode the alarm flags and examine observation gaps and repeated active states.
4. Build hourly features for eight sensors, requiring at least **45 recorded minutes per sensor**.
5. Compare logistic-regression models with persistence and constant predictions.
6. Investigate six-hour sensor histories around observed A9 state changes.

**Tools:** Python, PySpark/Spark ML, pandas, Matplotlib, Docker, and VS Code notebooks. The current notebook runs Spark locally with two worker threads.

## What is being predicted?

At each hourly cutoff, the models use the preceding complete hour of sensor data to predict the **first observed alarm state in the following clock hour**. Hours without alarm readings remain unlabelled.

This evaluates observed alarm states; it does **not** establish advance warning of new failures. The time between the feature cutoff and the target reading varies within the hour.

Features come from `S3`, `S39`, `S40`, `S41`, `S125`, `S181`, `S172`, and `S173`. Models use either hourly means or expanded statistics: mean, minimum, maximum, and standard deviation.

## Benchmark

Training uses data before February 2021, validation uses February, and March is reported as **exploratory evaluation** because it was inspected repeatedly during development. Scaling and class weights are fitted using training data only.

Six methods are compared on identical evaluation rows: always inactive, always active, persistence, logistic regression, weighted logistic regression, and expanded weighted logistic regression. Persistence uses the latest alarm reading strictly before the feature cutoff.

Selected March results:

| Channel | Method | TP | FP | FN | TN | F1 | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 | Always inactive | 0 | 0 | 2 | 15 | 0.000 | 0.500 |
| A5 | Weighted logistic | 2 | 6 | 0 | 9 | 0.400 | 0.800 |
| A5 | Expanded weighted logistic | 2 | 4 | 0 | 11 | 0.500 | 0.867 |
| A9 | Always active | 17 | 3 | 0 | 0 | 0.919 | 0.500 |
| A9 | Persistence | 16 | 3 | 1 | 0 | 0.889 | 0.471 |
| A9 | Expanded weighted logistic | 1 | 0 | 16 | 3 | 0.111 | 0.529 |

For **A5**, expanded features reduced false positives while detecting the three active evaluation readings across February and March. That sample is too small for a strong generalization claim.

For **A9**, always-active outperformed persistence on accuracy and F1 in both months. The weighted models missed 16 of 17 active March readings. High F1 from a constant predictor reflects the class imbalance, not useful discrimination.

The notebook calculates all metrics directly from predictions and exports `predictions.csv`, `metrics.csv`, and `protocol.json` to timestamped folders under `benchmarks/results/`.

## Limitations

- Missing alarm readings are unknown, not evidence of normal operation.
- Consecutive active readings may belong to the same observed spell; some spells cross split boundaries and remain in the benchmark.
- Operating conditions change over time, and some evaluation groups contain very few examples of one class.
- Only two distinct transition-associated A9 histories had complete six-hour sensor coverage. One also preceded inactive readings; the other followed a 38-hour alarm-observation gap. No consistent precursor was established.

Reliable early-warning evaluation would require clearer incident ground truth and a new, untouched period with more independent incidents.

## Running the notebook

1. Use a Python notebook environment with PySpark, a compatible Java runtime, pandas, and Matplotlib. This analysis was run through Docker and VS Code; the reported rerun used Spark 4.2.0.
2. Place the source dataset at `data/raw/sensor_data.csv`. Required columns are `when` (Unix milliseconds), `hwid`, `metric`, and `value`. The source data is needed to reproduce the full analysis.
3. Set `PROJECT_ROOT` in the first cell to your repository location **inside the container**. The existing setup uses `/home/jovyan/work`.
4. Open `notebooks/01_data_exploration.ipynb`, restart the kernel, and run all cells.

Full CSV scans take time. The notebook reuses `data/processed/sw088_raw.parquet` when present; rebuild that subset if the source data changes. The cleaned notebook was successfully rerun end to end in the author's Docker environment.

## Local Spark performance benchmark

I benchmarked hourly feature generation from the approximately 51 MB
SW-088 Parquet subset, comparing one and two local Spark worker threads.

Each run reads the data, computes all hourly statistics, and writes the
results to Parquet. Spark startup is excluded. Each configuration runs
in a fresh Spark process with one warm-up followed by three measured runs.

| Worker threads | Median runtime | Measured range |
| --- | ---: | ---: |
| 1 | 5.433 s | 5.424–5.590 s |
| 2 | 4.121 s | 3.918–4.215 s |

Two threads reduced median runtime by **24.1%**, a **1.32× speedup**.

Both configurations used a 2 GiB driver heap, eight shuffle partitions,
and disabled adaptive execution. The Docker container was limited to
two CPUs and 6 GiB RAM.

Output keys and counts matched exactly; floating-point statistics matched
within tolerance (`rtol=1e-7`, `atol=1e-9`).

These are local warm-run measurements on a small device subset, not a
distributed scaling benchmark on the full dataset. OS file caching was
not cleared, and configurations were run sequentially, so run order and
background activity may affect timings.

Script: `benchmarks/benchmark_hourly_features.py`.
