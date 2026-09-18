# ATMIFD: Multi-modal Microservice Anomaly Detection

ATMIFD is a research project for anomaly detection in microservice systems. It combines runtime metrics with trace-derived service interactions, learns from highly imbalanced labels, and selects anomaly thresholds on validation data.

The project accompanies the paper **“A Multi-modal Imbalance-Aware Method for Microservice Anomaly Detection”** by **Yumiao Tao** (GAIIS 2026).

## Highlights

- **Dual-modal representation:** combines service metrics with trace-derived propagation features.
- **Graph-temporal modeling:** captures service dependencies and their evolution across time windows.
- **Imbalance-aware learning:** increases the contribution of rare anomaly samples during training.
- **Adaptive threshold selection:** selects the decision threshold on the validation set instead of using a fixed heuristic.
- **Reproducible workflow:** includes preprocessing, training, evaluation, ablation, and automated runtime checks.

## Method

ATMIFD represents the microservice system as a temporal sequence of dependency graphs.

```text
Metrics ───────────────┐
                      ├─> Fused node features ─┐
Traces ─> Propagation ┘                        │
                                              ├─> Graph-temporal
Traces ─> Edge features ──────────────────────┘   encoder-decoder
                                                       │
                                                       ├─> Reconstruction loss
                                                       └─> Anomaly probability
                                                                  │
Validation set ─> Threshold selection ────────────────────────────┘
```

For each service, five metric features are fused with six trace-derived descriptors:

- outgoing and incoming interaction strength;
- outgoing and incoming degree;
- outgoing and incoming interaction concentration.

The model jointly optimizes reconstruction and classification objectives. Class weights are estimated from labeled training samples, and the final anomaly threshold is selected by maximizing validation F1.

## Paper results

The paper reports the following ATMIFD performance on the MSDS dataset:

| Precision | Recall | AUC | AP | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.807 | 0.713 | 0.962 | 0.739 | **0.757** |

The ablation study shows the contribution of feature fusion, threshold selection, and imbalance-aware learning:

| Configuration | F1 |
| --- | ---: |
| Without trace-derived node features | 0.402 |
| Metric-trace fusion | 0.624 |
| Metric-trace fusion + threshold search | 0.745 |
| **ATMIFD** | **0.757** |

These are the seed-42 experimental results reported in the manuscript. Dataset files, labels, generated caches, and model checkpoints are not distributed in this repository.

## Foundation and contributions

ATMIFD is developed on top of the official [MSTGAD implementation](https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection) and its [ASE 2023 paper](https://arxiv.org/abs/2310.04701).

The graph-temporal encoder-decoder, reconstruction framework, and parts of the data pipeline originate from MSTGAD. The main ATMIFD additions are:

1. trace-derived node propagation descriptors;
2. metric-trace node feature fusion;
3. training-derived class weighting;
4. validation-based anomaly threshold selection;
5. a four-configuration ablation workflow.

## Repository structure

```text
.
├── main.py                    # Training and evaluation
├── thr_only.py                # Validation threshold search
├── scripts/
│   ├── run_ablation.py        # Four ablation configurations
│   └── smoke_test.py          # Synthetic end-to-end runtime check
├── src/
│   ├── model.py               # Graph-temporal model
│   └── model_util.py          # Attention and graph layers
├── util/
│   ├── data_MSDS.py           # Window cache and feature construction
│   ├── features.py            # Trace-derived node features
│   ├── pre_MSDS.py            # MSDS preprocessing
│   ├── parser_MSDS.py         # Experiment configuration
│   ├── runtime.py             # Data splits, normalization, and loaders
│   ├── train.py               # Training and evaluation logic
│   └── util.py                # Metrics and experiment utilities
├── tests/                     # Regression tests
├── docs/running.md            # Detailed execution guide
├── requirements-cpu.txt
└── requirements-cu113.txt
```

## Environment

The reference environment uses:

- Python 3.8.20
- PyTorch 1.12.0
- PyTorch Geometric 2.2.0

Create a dedicated CPU environment:

```bash
conda create -n atmifd python=3.8.20 pip
conda activate atmifd
python -m pip install -r requirements-cpu.txt
```

For the CUDA 11.3 environment:

```bash
python -m pip install -r requirements-cu113.txt
```

See [docs/running.md](docs/running.md) for detailed execution notes.

## Dataset

Download the [Multi-Source Distributed System Data (MSDS)](https://zenodo.org/records/3549604) and place it under:

```text
data/MSDS/concurrent_data/
├── metrics/
├── logs/
└── traces/
```

Preprocess the raw files:

```bash
python util/pre_MSDS.py \
  --raw_path data/MSDS/concurrent_data \
  --save_path data/MSDS-pre
```

The prepared directory must contain:

```text
data/MSDS-pre/
├── metric.csv
├── trace.csv
├── trace_path.pkl
└── label.pkl
```

`label.pkl` is not generated by the preprocessor and must be supplied separately with the correct time alignment.

## Training

Run the complete ATMIFD configuration:

```bash
python main.py --device auto
```

`--device auto` uses CUDA when available and otherwise runs on CPU.

Each experiment is saved under `result/` with:

- experiment parameters;
- normalization statistics;
- loss-selected and F1-selected checkpoints;
- training and evaluation logs.

## Evaluation

Evaluate a saved experiment:

```bash
python main.py \
  --evaluate true \
  --model_path result/EXPERIMENT_DIRECTORY \
  --device cpu \
  --eval_stage f1
```

Search for a validation threshold without running final test evaluation:

```bash
python thr_only.py \
  --model_path result/EXPERIMENT_DIRECTORY \
  --device cpu \
  --eval_stage f1
```

## Ablation experiments

Preview the four configurations:

```bash
python scripts/run_ablation.py --dry-run
```

Run them with seed 42:

```bash
python scripts/run_ablation.py --seeds 42 --device auto
```

| Configuration | Trace-derived node features | Threshold search | Class weighting |
| --- | --- | --- | --- |
| `NoTraceNodeFeatures` | No | No | No |
| `MetricTraceFusion` | Yes | No | No |
| `MetricTraceFusion+ThrSearch` | Yes | Yes | No |
| `ATMIFD` | Yes | Yes | Yes |

The first configuration removes the six trace-derived node descriptors while retaining the inherited trace-edge graph branch.

## Verification

Run the automated checks:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cpu
```

The tests cover data splitting, normalization, metric calculation, class weighting, checkpoint selection, threshold search, cache generation, training, and evaluation.

## Notes

- ATMIFD performs window-level anomaly detection. Node anomaly probabilities are aggregated into one score per time window.
- Log inputs are zero-filled in this dual-modal metric-trace implementation.
- Data files, labels, caches, checkpoints, and experiment outputs are excluded from Git.
- Only trusted pickle files and model checkpoints should be loaded.

## Citation

ATMIFD:

```bibtex
@inproceedings{tao2026atmifd,
  author    = {Yumiao Tao},
  title     = {A Multi-modal Imbalance-Aware Method for Microservice Anomaly Detection},
  booktitle = {2026 International Conference on Generative Artificial Intelligence and Information Security (GAIIS)},
  year      = {2026}
}
```

MSTGAD:

```bibtex
@inproceedings{huang2023mstgad,
  author    = {Jun Huang and Yang Yang and Hang Yu and Jianguo Li and Xiao Zheng},
  title     = {Twin Graph-Based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System},
  booktitle = {2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages     = {66--78},
  year      = {2023}
}
```

## Use and attribution

This repository is provided for academic review and portfolio demonstration. It contains substantial components derived from MSTGAD; please retain the upstream attribution when reviewing or discussing the project. Licensing and redistribution terms should be confirmed before reuse.
