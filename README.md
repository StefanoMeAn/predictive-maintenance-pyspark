# Predictive Maintenance with PySpark

I used PySpark to explore approximately **142 million industrial sensor records** and investigate whether recent sensor measurements could help predict recorded overheating alarms. The analysis focuses on device **SW-088** and alarm channels **A5 and A9**.

The results show why a large sensor dataset does not necessarily provide enough independent failure examples for reliable prediction. The evaluation measures observed alarm states; it does not establish advance warning of new failures.

## Project Overview

The project covers data quality checks, alarm-state analysis, hourly feature engineering, chronological model evaluation, and a local Spark performance benchmark. The main notebook is [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb).

The analysis proceeds through these steps:

1. Read the CSV with an explicit schema and inspect missing values, timestamps, and device coverage.
2. Save the SW-088 subset as Parquet for reuse.
3. Decode alarm flags and examine observation gaps and repeated active states.
4. Build hourly features for eight sensors, requiring at least **45 recorded minutes per sensor**.
5. Compare logistic-regression models with persistence and constant-prediction baselines.
6. Investigate six-hour sensor histories around observed A9 state changes.

## Dataset

The source data contains approximately **142 million industrial sensor records**. The analysis filters the data to device SW-088 and uses alarm channels A5 and A9. The full source dataset is not included in the repository.

| Data | Description |
|---|---|
| Required CSV columns | `when` (Unix milliseconds), `hwid`, `metric`, and `value` |
| Device analysed | SW-088 |
| Alarm channels | A5 and A9 |
| Sensor features | S3, S39, S40, S41, S125, S181, S172, and S173 |
| Reusable subset | `data/processed/sw088_raw.parquet` |

## Methodology

### Prediction target

At each hourly cutoff, the models use the preceding complete hour of sensor data to predict the **first observed alarm state in the following clock hour**. Hours without alarm readings remain unlabelled.

This evaluates observed alarm states. It does **not** establish advance warning of new failures, and the delay between the feature cutoff and target reading varies within the following hour.

### Features and models

Features are computed from `S3`, `S39`, `S40`, `S41`, `S125`, `S181`, `S172`, and `S173`. I compare hourly means with expanded statistics consisting of mean, minimum, maximum, and standard deviation. Each sensor must have at least 45 recorded minutes in the hour.

The benchmark compares six methods on identical evaluation rows: always inactive, always active, persistence, logistic regression, weighted logistic regression, and expanded weighted logistic regression. Persistence uses the latest alarm reading strictly before the feature cutoff.

### Chronological evaluation

Training uses data before February 2021, validation uses February, and March is reported as **exploratory evaluation** because it was inspected repeatedly during development. Scaling and class weights are fitted using training data only.

## Results

### Selected March benchmark results

| Channel | Method | TP | FP | FN | TN | F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| A5 | Always inactive | 0 | 0 | 2 | 15 | 0.000 | 0.500 |
| A5 | Weighted logistic | 2 | 6 | 0 | 9 | 0.400 | 0.800 |
| A5 | Expanded weighted logistic | 2 | 4 | 0 | 11 | 0.500 | 0.867 |
| A9 | Always active | 17 | 3 | 0 | 0 | 0.919 | 0.500 |
| A9 | Persistence | 16 | 3 | 1 | 0 | 0.889 | 0.471 |
| A9 | Expanded weighted logistic | 1 | 0 | 16 | 3 | 0.111 | 0.529 |

For A5, expanded features reduced false positives while detecting the three active evaluation readings across February and March. That sample is too small to support a strong generalization claim.

For A9, always-active outperformed persistence on accuracy and F1 in both months. The weighted models missed 16 of 17 active March readings. A high F1 from a constant predictor reflects class imbalance, not useful discrimination.

The notebook calculates metrics directly from predictions and exports `predictions.csv`, `metrics.csv`, and `protocol.json` to timestamped folders under `benchmarks/results/`.

### Local Spark performance benchmark

I benchmarked hourly feature generation from the approximately **51 MB** SW-088 Parquet subset using one and two local Spark worker threads. Each run reads the data, computes hourly statistics, and writes the results to Parquet. Spark startup is excluded. Each configuration ran in a fresh Spark process, with one warm-up and three measured runs.

| Worker threads | Median runtime | Measured range |
|---|---:|---:|
| 1 | 5.433 s | 5.424–5.590 s |
| 2 | 4.121 s | 3.918–4.215 s |

Two threads reduced median runtime by **24.1%**, a **1.32× speedup**. Both configurations used a 2 GiB driver heap, eight shuffle partitions, and disabled adaptive execution. The Docker container was limited to two CPUs and 6 GiB RAM. Output keys and counts matched exactly; floating-point statistics matched within `rtol=1e-7` and `atol=1e-9`.

These are local warm-run measurements on a small device subset, not a distributed scaling benchmark on the full dataset. OS file caching was not cleared, and configurations ran sequentially, so run order and background activity may affect timings. The benchmark script is `benchmarks/benchmark_hourly_features.py`.

## Repository Structure

```text
.
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                       # Source CSV (not included)
│   └── processed/
│       └── sw088_raw.parquet      # Reusable SW-088 subset
└── benchmarks/
    ├── benchmark_hourly_features.py
    └── results/                   # Timestamped metrics and predictions
```

The tree shows the main paths referenced by the analysis; generated benchmark results and the processed Parquet subset may be created locally.

## Installation

Use a Python notebook environment with **PySpark**, a compatible **Java runtime**, **pandas**, and **Matplotlib**. The analysis was run through Docker and VS Code; the reported rerun used **Spark 4.2.0**. The local Spark benchmark used Docker with a 2 GiB driver heap and a 6 GiB container memory limit.

## Usage

1. Place the source dataset at `data/raw/sensor_data.csv`. The required columns are `when` (Unix milliseconds), `hwid`, `metric`, and `value`.
2. Set `PROJECT_ROOT` in the first notebook cell to the repository location inside your environment. The existing container setup uses `/home/jovyan/work`.
3. Open `notebooks/01_data_exploration.ipynb`, restart the kernel, and run all cells.

Full CSV scans take time. The notebook reuses `data/processed/sw088_raw.parquet` when it is present; rebuild the subset if the source data changes. The cleaned notebook was rerun end to end in the author’s Docker environment.

To run the local hourly-feature benchmark, use:

```bash
python benchmarks/benchmark_hourly_features.py
```

## Limitations and Future Work

- Missing alarm readings are unknown, not evidence of normal operation.
- Consecutive active readings may belong to the same observed spell; some spells cross split boundaries and remain in the benchmark.
- Operating conditions change over time, and some evaluation groups contain very few examples of one class.
- Only two distinct transition-associated A9 histories had complete six-hour sensor coverage. One also preceded inactive readings; the other followed a 38-hour alarm-observation gap. No consistent precursor was established.
- March was inspected repeatedly during development and is therefore exploratory evaluation, not an untouched final test set.

Reliable early-warning evaluation would require clearer incident ground truth and a new, untouched period containing more independent incidents.

## Technologies

Python · PySpark / Spark ML · pandas · Matplotlib · Docker · VS Code notebooks

## Author

**Stefano Meza** — Physicist with a Master’s degree in Physics of Data from the University of Padova, interested in machine learning, deep learning, computer vision, NLP, and scientific computing.

[LinkedIn](https://www.linkedin.com/in/stefanomean/) · [GitHub](https://github.com/StefanoMeAn)
