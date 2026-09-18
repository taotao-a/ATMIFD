# Running ATMIFD

## Environment

The reference stack is Python 3.8.20, PyTorch 1.12.0 and PyG 2.2.0. The
requirements retain the research project's versions; they are not a production
deployment recommendation. Use a dedicated environment on Windows x86-64 or
Linux x86-64 and run these commands from the repository root:

```text
conda create -n atmifd python=3.8.20 pip
conda activate atmifd
python -m pip install -r requirements-cpu.txt
```

For the historical NVIDIA CUDA 11.3 stack, replace the last command with:

```text
python -m pip install -r requirements-cu113.txt
```

Choose one variant in a fresh environment. The variant-specific files select
matching PyTorch/PyG extension wheels and avoid compiling the extensions from
source. Installation commands follow the [PyTorch archive](https://pytorch.org/get-started/previous-versions/)
and [PyG 2.2 installation guide](https://pytorch-geometric.readthedocs.io/en/2.2.0/notes/installation.html).

## Data

Datasets, caches and checkpoints are not included in the public repository.
To generate a new window cache, supply the following prepared inputs:

```text
data/MSDS-pre/
  label.pkl       # binary labels: timestamps x 5 services
  metric.csv      # timestamp column "now" and service metric columns
  trace.csv       # trace statistics, timestamps and durations
  trace_path.pkl  # binary 5 x 5 service adjacency matrix
```

The trace channel count must match `--raw_edge` (default 7). Logs are zero-filled
in this implementation; `log.csv` is not required by the model loader.

If starting from raw MSDS files, use:

```text
python util/pre_MSDS.py --raw_path data/MSDS/concurrent_data --save_path data/MSDS-pre
```

The raw folder must contain `metrics/`, `traces/`, and
`logs/logs_aggregated_concurrent.csv`. This preprocessor produces the CSV files
and graph, but does **not** create labels. Supply the matching `label.pkl`
separately. Existing generated files are protected unless `--overwrite` is
explicitly supplied. Original date defaults and local-time/one-hour alignment
are retained; use the historical preprocessing timezone or prepared data when
comparing historical experiments.

Only load trusted pickle files and checkpoints. PyTorch 1.12 does not support
restricted weights-only checkpoint loading.

## Train and evaluate

```text
python main.py --device auto
```

Defaults select the full ATMIFD configuration: seed 42, six trace-derived node
features, imbalance-aware classification loss and validation threshold search.
`--device auto` uses CUDA when available, otherwise CPU. Explicit `--device cuda`
requires working CUDA; `--device cpu` does not require a GPU. The legacy
`--gpu false` flag selects CPU when `--device` is `auto`.

Relative paths are resolved against the repository root. Use `--data_path`,
`--dataset_path`, and `--result_dir` to select other locations. Without an
explicit cache path, trace feature settings select separate versioned
`MSDS-save-v2-metric` and `MSDS-save-v2-fusion` directories. New caches contain
`cache_config.json`;
configuration mismatches are rejected instead of silently reusing the cache.
Legacy caches are rejected because their full-dataset normalization cannot be
verified; select a new cache path to rebuild them.
An existing empty cache folder is an error, not a request to overwrite it.

The chronological split is 60% training, 10% validation and 30% testing on the
raw-time axis. Windows are formed inside each split, so adjacent splits do not
share the `window - 1` boundary observations.
Metric and trace normalization statistics are fitted only on the training split
and saved in `normalization.json`; evaluation reuses those exact statistics.
Training drops incomplete batches. F1 checkpoint selection uses validation-set F1
and starts after `rec_down`; a default F1 run therefore needs at least 3 epochs.
For a shorter runtime check, use `--epochs 1 --eval_stage loss`.

Each experiment directory contains `params.json`, `normalization.json`, `running.log`,
`my_loss_stage.ckpt`, `my_f1_stage.ckpt` and `evaluation.log` when the corresponding
training stages succeed. The searched threshold is recorded in `evaluation.log`.

Evaluate a saved experiment (replace the path):

```text
python main.py --evaluate true --model_path result/EXPERIMENT --device cpu --eval_stage f1
```

Saved training parameters are restored. Explicit CLI values override them;
device and checkpoint stage are chosen for the current invocation. When moving
an experiment, override any saved data/cache paths that no longer exist.

Validation-only threshold search, without testing or writing back the threshold:

```text
python thr_only.py --model_path result/EXPERIMENT --device cpu --eval_stage f1
```

Only the selected checkpoint is required. This command prints the threshold;
it does not modify `params.json` or checkpoint files. To use a printed threshold
without searching again, pass `--thr_search false --threshold VALUE` to evaluation.

## Ablations and runtime checks

```text
python scripts/run_ablation.py --dry-run
python scripts/run_ablation.py --seeds 42 --device auto
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cpu
```

The ablation launcher runs four controlled configurations, gives each variant
its own experiment name, and separates node-feature caches. The first variant is
named `NoTraceNodeFeatures` because it removes the six trace-derived node
features but retains the inherited trace-edge branch; it is not a trace-free
model. Weighted and unweighted variants both use cross-entropy so that the loss
ablation changes only the class weights. The smoke
test creates temporary synthetic data, checks training, checkpoints, evaluation,
threshold search and cache generation, then removes its own temporary files.
These runtime checks do not establish reproduction of the paper's metrics.
