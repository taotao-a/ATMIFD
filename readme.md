# ATMIFD: Multi-modal Microservice Anomaly Detection

ATMIFD is a research implementation for anomaly detection in microservice systems. It combines runtime metrics with trace-derived service interaction features, applies imbalance-aware learning, and selects the anomaly threshold on a validation set.

This repository accompanies the paper **“A Multi-modal Imbalance-Aware Method for Microservice Anomaly Detection”** (Yumiao Tao, GAIIS 2026). It is intended as a transparent research and portfolio project rather than a production monitoring service.

## Method overview

ATMIFD represents each time step as a service dependency graph and processes a sequence of graphs with the inherited graph-temporal encoder-decoder.

The project adds three main components to the MSTGAD foundation:

- **Metric-trace feature fusion:** five metric features are combined with six trace-derived propagation descriptors for each service.
- **Imbalance-aware classification:** class weights are computed from labeled training samples using `min(normal / anomaly, cap)`.
- **Validation-based threshold selection:** the threshold that maximizes validation F1 is applied once to the test set.

The current `main` branch also enforces training-only normalization, non-overlapping temporal splits, validation-based checkpoint selection, and anomaly-positive AUC/AP calculation.

## Upstream foundation and attribution

The graph-temporal encoder-decoder, reconstruction objective, and substantial parts of the data pipeline are derived from the official [MSTGAD implementation](https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection) and its [ASE 2023 paper](https://arxiv.org/abs/2310.04701). These inherited components are not claimed as original ATMIFD contributions.

Repository branches serve different purposes:

- `main`: corrected and tested portfolio implementation.
- `legacy`: preserved pre-correction repository state for tracing historical behavior.

Historical paper values belong to the legacy experiment pipeline. Because the corrected pipeline changes data handling, evaluation, and the controlled loss ablation, it must be retrained before reporting new results.

## Repository structure

```text
.
├── main.py                    # Train, select threshold, and evaluate
├── thr_only.py                # Search a threshold for a saved experiment
├── scripts/
│   ├── run_ablation.py        # Four controlled configurations
│   └── smoke_test.py          # Synthetic end-to-end runtime check
├── src/
│   ├── model.py               # Graph-temporal model
│   └── model_util.py          # Attention and graph layers
├── util/
│   ├── data_MSDS.py           # Window cache and feature construction
│   ├── features.py            # Trace-derived node features
│   ├── pre_MSDS.py            # Raw MSDS preprocessing
│   ├── parser_MSDS.py         # Command-line configuration
│   ├── runtime.py             # Paths, splits, normalization, and loaders
│   ├── train.py               # Training, checkpointing, and evaluation
│   └── util.py                # Metrics and experiment utilities
├── tests/                     # Regression tests
├── docs/running.md            # Detailed execution notes
├── requirements-cpu.txt
└── requirements-cu113.txt
```

## Environment

The reference environment uses Python 3.8.20, PyTorch 1.12.0, and PyTorch Geometric 2.2.0. Create a dedicated environment from the repository root.

CPU:

```bash
conda create -n atmifd python=3.8.20 pip
conda activate atmifd
python -m pip install -r requirements-cpu.txt
```

Historical CUDA 11.3 stack:

```bash
python -m pip install -r requirements-cu113.txt
```

See [docs/running.md](docs/running.md) for dependency, checkpoint, and path details.

## Dataset preparation

Download the [Multi-Source Distributed System Data (MSDS)](https://zenodo.org/records/3549604) and place the raw files under:

```text
data/MSDS/concurrent_data/
├── metrics/
├── logs/
└── traces/
```

Generate the prepared metric, trace, and graph files:

```bash
python util/pre_MSDS.py \
  --raw_path data/MSDS/concurrent_data \
  --save_path data/MSDS-pre
```

The preprocessor does not generate anomaly labels. Add the matching `label.pkl` separately:

```text
data/MSDS-pre/
├── metric.csv
├── trace.csv
├── trace_path.pkl
└── label.pkl
```

Datasets, caches, labels, checkpoints, and experiment outputs are excluded from Git. Only load pickle files and checkpoints from trusted sources.

## Training and evaluation

Train the full corrected ATMIFD configuration:

```bash
python main.py --device auto
```

The program creates versioned feature caches and saves each experiment under `result/`, including its parameters, training-only normalization statistics, checkpoints, logs, and final evaluation.

Evaluate a saved experiment:

```bash
python main.py \
  --evaluate true \
  --model_path result/EXPERIMENT_DIRECTORY \
  --device cpu \
  --eval_stage f1
```

Search only for a validation threshold:

```bash
python thr_only.py \
  --model_path result/EXPERIMENT_DIRECTORY \
  --device cpu \
  --eval_stage f1
```

## Controlled ablations

Preview or run the four configurations:

```bash
python scripts/run_ablation.py --dry-run
python scripts/run_ablation.py --seeds 42 --device auto
```

| Configuration | Trace-derived node features | Threshold search | Classification loss |
| --- | --- | --- | --- |
| `NoTraceNodeFeatures` | No | No | Unweighted cross-entropy |
| `MetricTraceFusion` | Yes | No | Unweighted cross-entropy |
| `MetricTraceFusion+ThrSearch` | Yes | Yes | Unweighted cross-entropy |
| `ATMIFD` | Yes | Yes | Class-weighted cross-entropy |

`NoTraceNodeFeatures` removes the six trace-derived node descriptors but retains the inherited trace-edge branch and graph connectivity. The name intentionally avoids claiming that this configuration is a completely trace-free metric-only model.

## Historical manuscript-linked results

The following F1 values were matched to archived local seed-42 experiment logs during repository preparation. They are included for provenance only: the underlying data, caches, checkpoints, and logs are not distributed, and these values have not been reproduced with the corrected `main` branch.

| Manuscript configuration | Historical F1 |
| --- | ---: |
| MetricTraceIsolation | 0.4023 |
| MetricTraceFusion | 0.6238 |
| MetricTraceFusion + threshold search | 0.7448 |
| ATMIFD | 0.7572 |

Do not compare these legacy values directly with future corrected-pipeline results. The corrected pipeline uses a different evaluation and data-processing protocol.

## Verification

Run the regression and synthetic end-to-end checks:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cpu
```

The checks cover configuration validation, non-overlapping splits, training-only normalization, anomaly-positive metrics, controlled weighted/unweighted losses, checkpoint selection, evaluation, threshold search, partial batches, and cache generation. They verify runtime behavior but do not reproduce the paper’s MSDS metrics.

## Scope and limitations

- Evaluation is window-level: node probabilities are aggregated with the configured window rule, which defaults to maximum anomaly probability.
- The inherited log branch remains in the network, while this implementation supplies zero-filled log inputs.
- `NoTraceNodeFeatures` is not a fully trace-free architecture.
- Corrected MSDS training results are not included; formal reproduction requires the original aligned labels and a fresh full run.
- No production guarantees are provided.

## Citation

ATMIFD manuscript:

```bibtex
@inproceedings{tao2026atmifd,
  author    = {Yumiao Tao},
  title     = {A Multi-modal Imbalance-Aware Method for Microservice Anomaly Detection},
  booktitle = {2026 International Conference on Generative Artificial Intelligence and Information Security (GAIIS)},
  year      = {2026}
}
```

MSTGAD foundation:

```bibtex
@inproceedings{huang2023mstgad,
  author    = {Jun Huang and Yang Yang and Hang Yu and Jianguo Li and Xiao Zheng},
  title     = {Twin Graph-Based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System},
  booktitle = {2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages     = {66--78},
  year      = {2023}
}
```

## Redistribution notice

This repository currently contains no license grant. Redistribution and reuse terms for inherited upstream code must be confirmed before others reuse the project. The repository is provided for academic review and portfolio demonstration.
