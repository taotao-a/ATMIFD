"""Shared entry-point configuration and dataset checks (no ML imports)."""

import json
from pathlib import Path
import sys

from util.parser_MSDS import validate_args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = 2
CACHE_KEYS = ('window', 'step', 'num_nodes', 'metric_len', 'trace_node_dim',
              'raw_edge', 'log_len', 'label_percent')


def project_path(value):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def prepare_args(args, argv=None, threshold_only=False):
    argv = sys.argv[1:] if argv is None else argv
    explicit = {x.split('=', 1)[0][2:] for x in argv if x.startswith('--')}
    cli = args.copy()
    if threshold_only:
        args['evaluate'] = True
    if args['evaluate']:
        if not args['model_path']:
            raise ValueError('--model_path is required for evaluation or threshold search')
        model_path = project_path(args['model_path'])
        params = Path(model_path) / 'params.json'
        if not params.is_file():
            raise FileNotFoundError('Missing training configuration: {}'.format(params))
        with params.open(encoding='utf-8') as handle:
            saved = json.load(handle)
        args.update(saved)
        for key in explicit:
            if key in cli:
                args[key] = cli[key]
        for key in ('device', 'gpu', 'eval_stage'):
            args[key] = cli[key]
        args.update(model_path=model_path, result_dir=model_path, evaluate=True)
        if explicit.intersection({'metric_len', 'trace_node_dim'}) and 'raw_node' not in explicit:
            args['raw_node'] = None
    validate_args(args)
    if threshold_only and args['eval_stage'] == 'both':
        raise ValueError('Threshold-only mode requires --eval_stage f1 or loss')
    if args['dataset_path'] is None:
        args['dataset_path'] = './data/MSDS-save-v2-' + ('fusion' if args['trace_node_dim'] else 'metric')
    for key in ('data_path', 'dataset_path', 'result_dir'):
        args[key] = project_path(args[key])
    if args['model_path']:
        args['model_path'] = project_path(args['model_path'])
    validate_data_paths(args)
    if args['evaluate']:
        stages = ('loss', 'f1') if args['eval_stage'] == 'both' else (args['eval_stage'],)
        for stage in stages:
            checkpoint = Path(args['model_path']) / ('my_{}_stage.ckpt'.format(stage))
            if not checkpoint.is_file():
                raise FileNotFoundError('Missing checkpoint: {}'.format(checkpoint))
    return args


def validate_data_paths(args):
    data, cache = Path(args['data_path']), Path(args['dataset_path'])
    required = ['trace_path.pkl']
    if cache.exists():
        if not cache.is_dir() or not any(p.stem.isdigit() for p in cache.glob('*.pkl')):
            raise ValueError('Dataset cache is empty or invalid: {}. Use a new cache directory to rebuild it.'.format(cache))
        manifest = cache / 'cache_config.json'
        if not manifest.is_file():
            raise ValueError('Legacy cache has no corrected-pipeline metadata: {}. Select a new --dataset_path to rebuild it.'.format(cache))
        with manifest.open(encoding='utf-8') as handle:
            saved = json.load(handle)
        if saved.get('pipeline_version') != PIPELINE_VERSION:
            raise ValueError('Cache pipeline version is not {}: {}. Select a new --dataset_path.'.format(PIPELINE_VERSION, cache))
        mismatches = [key for key in CACHE_KEYS if saved.get(key) != args[key]]
        if mismatches:
            raise ValueError('Cache configuration mismatch ({}). Select a separate --dataset_path.'.format(', '.join(mismatches)))
    else:
        required.extend(['label.pkl', 'metric.csv', 'trace.csv'])
    missing = [str(data / name) for name in required if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing MSDS inputs: {}. Supply prepared data with --data_path; labels must be provided separately.'.format(', '.join(missing)))


def split_window_ranges(n_windows, window):
    """Return non-overlapping window-index ranges for a 60/10/30 raw-time split.

    Window i covers raw positions [i, i + window). Splitting the already-built
    windows without this gap would share window - 1 observations at each boundary.
    """
    n_raw = n_windows + window - 1
    train_raw_end = int(n_raw * 0.6)
    val_raw_end = train_raw_end + int(n_raw * 0.1)
    return {
        'train': (0, max(0, train_raw_end - window + 1)),
        'val': (train_raw_end, max(train_raw_end, val_raw_end - window + 1)),
        'test': (val_raw_end, n_windows),
    }


def _unique_timeline(records, key):
    """Reconstruct a step-1 timeline from consecutive overlapping windows."""
    import numpy as np

    first = np.asarray(records[0][key])
    if len(records) == 1:
        return first.copy()
    tails = np.stack([np.asarray(record[key])[-1] for record in records[1:]], axis=0)
    return np.concatenate([first, tails], axis=0)


def _fit_normalization(train_records, args):
    import numpy as np

    metric = _unique_timeline(train_records, 'data_node')[..., :args['metric_len']]
    trace = _unique_timeline(train_records, 'data_edge')
    return {
        'normalization_version': 2,
        'metric_min': metric.min(axis=0).tolist(),
        'metric_max': metric.max(axis=0).tolist(),
        'trace_mean': trace.mean(axis=0).tolist(),
    }


def _apply_normalization(dataset, args, stats):
    import numpy as np

    if stats.get('normalization_version') != 2:
        raise ValueError('Unsupported normalization metadata version')
    metric_min = np.asarray(stats['metric_min'], dtype=np.float32)
    metric_max = np.asarray(stats['metric_max'], dtype=np.float32)
    trace_mean = np.asarray(stats['trace_mean'], dtype=np.float32)
    expected_metric = (args['num_nodes'], args['metric_len'])
    expected_trace = (args['num_nodes'], args['num_nodes'], args['raw_edge'])
    if metric_min.shape != expected_metric or metric_max.shape != expected_metric:
        raise ValueError('Saved metric normalization shape does not match this model')
    if trace_mean.shape != expected_trace:
        raise ValueError('Saved trace normalization shape does not match this model')

    metric_scale = metric_max - metric_min
    for record in dataset:
        metric = np.asarray(record['data_node'][..., :args['metric_len']], dtype=np.float32)
        metric = np.divide(metric - metric_min, metric_scale,
                           out=np.zeros_like(metric), where=metric_scale > 1e-8)
        edge = np.asarray(record['data_edge'], dtype=np.float32)
        edge = edge / (trace_mean + 1e-6)
        record['data_edge'] = edge
        if args['trace_node_dim'] == 6:
            from util.features import trace_node_features
            node = np.concatenate([metric, trace_node_features(edge)], axis=-1)
        else:
            node = metric
        record['data_node'] = node.astype(np.float32, copy=False)


def build_loaders(processed, args):
    from torch.utils.data import DataLoader

    n_total = len(processed.dataset)
    ranges = split_window_ranges(n_total, args['window'])
    train_start, train_end = ranges['train']
    val_start, val_end = ranges['val']
    test_start, test_end = ranges['test']
    if train_end <= train_start or val_end <= val_start or test_end <= test_start:
        raise ValueError('Dataset is too short for non-overlapping train/validation/test windows')
    n_train = train_end - train_start
    if not args['evaluate'] and n_train < args['batch_size']:
        raise ValueError('Training split is smaller than batch_size; reduce --batch_size')
    if args['evaluate']:
        normalization_path = Path(args['model_path']) / 'normalization.json'
        if not normalization_path.is_file():
            raise FileNotFoundError('Missing corrected normalization metadata: {}'.format(normalization_path))
        with normalization_path.open(encoding='utf-8') as handle:
            normalization = json.load(handle)
    else:
        normalization = _fit_normalization(processed.dataset[train_start:train_end], args)
    _apply_normalization(processed.dataset, args, normalization)
    processed.normalization_stats = normalization
    train_set = processed.dataset[train_start:train_end]
    val_set = processed.dataset[val_start:val_end]
    test_set = processed.dataset[test_start:test_end]
    train_dl = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                          pin_memory=False, drop_last=True)
    val_dl = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                        pin_memory=False, drop_last=False)
    test_dl = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False,
                         pin_memory=False, drop_last=False)
    return train_dl, val_dl, test_dl
