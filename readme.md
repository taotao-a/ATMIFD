# ATMIFD: Microservice Anomaly Detection

Research code for **A Multi-modal Imbalance-Aware Method for Microservice Anomaly Detection**, by **Yumiao Tao** (2026). This academic extension of **MSTGAD** demonstrates research implementation, experiment configuration, and ablation analysis; it is not a production monitoring service.

## Research contributions and upstream foundation

Based on the official [MSTGAD code](https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection) and [ASE 2023 paper](https://arxiv.org/abs/2310.04701). Its graph-temporal encoder-decoder, reconstruction-based classifier, and much of the data-processing framework are inherited, not claimed as ATMIFD contributions.

ATMIFD adds:

- **Trace-derived node features:** six incoming/outgoing strength, degree, and concentration descriptors, concatenated with metrics.
- **Imbalance-aware training:** weighted cross-entropy using class weights estimated from labeled training samples.
- **Validation-based thresholds:** grid search maximizing validation F1 before final test evaluation.
- **Ablation launcher:** four configurations examining the incremental effects of these changes.

## Implementation overview

Metric/trace graph windows pass through the encoder-decoder; reconstruction discrepancies feed a classifier producing node anomaly probabilities. The archived implementation scores each window by the maximum node probability and labels it anomalous if any node is anomalous.

Sequential train/validation/test split: **0.6 / 0.1 / 0.3**; thresholds use validation, not test windows. Historical settings: 5 services, 5 metric features, 7 trace channels, window 10, batch 50. Log inputs are zeroed, but the inherited log branch remains.

## Project structure

```text
.
├── main.py                  # Training, threshold selection, and evaluation
├── thr_only.py              # Legacy checkpoint threshold-search helper
├── readme.md
├── requirements.txt         # Legacy auxiliary dependency list
├── .gitignore
├── scripts/
│   └── run_ablation.py       # Four-configuration experiment launcher
├── src/
│   ├── model.py             # Inherited model and classification architecture
│   └── model_util.py        # Graph and temporal attention building blocks
├── util/
│   ├── data_MSDS.py         # Graph windows and trace-derived node features
│   ├── pre_MSDS.py          # Legacy dataset preprocessing
│   ├── parser_MSDS.py       # Experiment parameters and switches
│   ├── train.py             # Training, class weights, and window evaluation
│   ├── util.py              # Metrics, threshold search, and experiment logging
│   ├── constant.py          # Service identifiers
│   └── msds.ini             # Required by legacy log preprocessing
└── data/
    └── MSDS-pre/            # Local preprocessed data; excluded from Git
```

## Current status

Source-to-history matching for all four ablations, directory cleanup, and Python syntax checks are complete. **Clean-environment training/inference have not been rerun**; matching historical records is not independent reproduction. Environment validation, runtime fixes, and manuscript/implementation review remain pending.

## Environment and data preparation

- NVIDIA CUDA is currently required by unconditional CUDA calls; `--gpu false` is not reliable CPU support.
- The ablation launcher requires Python 3.9+ for `list[str]` annotations.
- No fully tested environment specification exists yet. `requirements.txt` lists legacy auxiliary dependencies, omitting PyTorch, PyTorch Geometric, and an explicit `cachetools` dependency.

Install listed dependencies only (**not the full runtime**):

```bash
python -m pip install -r requirements.txt
```

Download **Multi-Source Distributed System Data (MSDS)** from [Zenodo](https://zenodo.org/record/3549604) into:

```text
data/MSDS/concurrent_data/
├── metrics/
├── logs/
└── traces/
```

Legacy preprocessing:

```bash
python util/pre_MSDS.py
```

Preprocessing still handles logs despite zeroed model log inputs. Review its fixed interval and local-time conversions before rerunning.

Training requires `metric.csv`, `trace.csv`, `trace_path.pkl`, and `label.pkl` in `data/MSDS-pre/`. Preprocessing does **not** create labels: obtain them separately and check interval alignment. The local label file is Git-ignored. Data, caches, weights, and historical logs are not distributed. Never load untrusted pickle, JSON-pickle, or checkpoint files.

## Experiment commands

Run from the project root with compatible dependencies and complete data. These archived configurations are **not newly verified reproduction commands**.

### Full ATMIFD configuration

```bash
python main.py --random_seed 42 --trace_node_dim 6 --imb_loss true --thr_search true --win_agg max --eval_stage f1 --dataset_path ./data/MSDS-save-fusion
```

Plain `python main.py` defaults to `imb_loss=false`, not full ATMIFD. The inherited `MSTGAD` run label is an output identifier, not model selection.

### Four experiment configurations

```bash
python scripts/run_ablation.py --seeds 42
```

Historical results use seed 42; the launcher defaults to 0, 1, 2 without automatic averaging. Feature variants use separate caches. Rebuild stale caches after preprocessing, window, or label changes; settings are not validated on load. `thr_only.py` has a machine-specific checkpoint path and is not yet portable.

## Historical experiment results

**Historical single-run log values**, corresponding to manuscript Table III; not fresh results or multi-seed averages.

| Configuration | Extra trace node features | Validation threshold search | Classification loss | Historical F1 |
| --- | --- | --- | --- | ---: |
| `MetricOnly` | No | No | Legacy weighted BCE | 0.4023 |
| `MetricTraceFusion` | Yes | No | Legacy weighted BCE | 0.6238 |
| `MetricTraceFusion+ThrSearch` | Yes | Yes | Legacy weighted BCE | 0.7448 |
| `ATMIFD` | Yes | Yes | Train-derived weighted cross-entropy | 0.7572 |

- `MetricOnly` is **MetricTraceIsolation** in the manuscript. It disables only the six extra node features, retaining trace edges/connectivity; it is **not trace-free**.
- Legacy BCE is already weighted. ATMIFD changes both loss type and weight rule, not simply unweighted to weighted supervision. Historical settings: seed 42, `imb_alpha=0.5`, `imb_wmax=5`, weights `[1, 5]`.

## Known limitations and manuscript differences

The archived implementation is unchanged for traceability:

- Normalization uses the full input interval, not training-only statistics as described in the manuscript.
- Metrics are window-level; several manuscript equations describe node-level decisions.
- Zero log inputs retain the log network and reconstruction branch.
- `f1` checkpoints use training F1. Test metrics are logged during training but do not select checkpoints.
- AUC/AP output selection switches between normal/anomaly classes. AP needs anomaly-positive review; no validated AP is claimed.
- Some manuscript narrative numbers differ from its tables. Only matched historical F1 is presented; superiority over every baseline is not claimed.

Preprocessing, scoring, or loss revisions may change results and require separate evaluation.

## Citation and acknowledgments

ATMIFD manuscript:

```bibtex
@inproceedings{tao2026atmifd,
  author    = {Yumiao Tao},
  title     = {A Multi-modal Imbalance-Aware Method for Microservice Anomaly Detection},
  booktitle = {2026 International Conference on Generative Artificial Intelligence and Information Security (GAIIS)},
  year      = {2026}
}
```

DOI, pages, and official publication link are not supplied; citation metadata is limited to the manuscript.

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

Redistribution terms need confirmation before public release; this README grants no license for inherited code.
