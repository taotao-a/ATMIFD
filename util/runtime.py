"""Shared entry-point configuration and dataset checks (no ML imports)."""

import json
from pathlib import Path
import sys

from util.parser_MSDS import validate_args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        args['dataset_path'] = './data/MSDS-save-' + ('fusion' if args['trace_node_dim'] else 'metric')
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
        if manifest.is_file():
            with manifest.open(encoding='utf-8') as handle:
                saved = json.load(handle)
            mismatches = [key for key in CACHE_KEYS if saved.get(key) != args[key]]
            if mismatches:
                raise ValueError('Cache configuration mismatch ({}). Select a separate --dataset_path.'.format(', '.join(mismatches)))
    else:
        required.extend(['label.pkl', 'metric.csv', 'trace.csv'])
    missing = [str(data / name) for name in required if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing MSDS inputs: {}. Supply prepared data with --data_path; labels must be provided separately.'.format(', '.join(missing)))


def build_loaders(processed, args):
    from torch.utils.data import DataLoader

    n_total = len(processed.dataset)
    n_train, n_val = int(n_total * 0.6), int(n_total * 0.1)
    if not n_val or n_total - n_train - n_val <= 0:
        raise ValueError('At least 10 windows are needed for the train/validation/test split')
    if not args['evaluate'] and n_train < args['batch_size']:
        raise ValueError('Training split is smaller than batch_size; reduce --batch_size')
    train_set = processed.dataset[:n_train]
    val_set = processed.dataset[n_train:n_train + n_val]
    test_set = processed.dataset[n_train + n_val:]
    train_dl = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                          pin_memory=False, drop_last=True)
    val_dl = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                        pin_memory=False, drop_last=False)
    test_dl = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False,
                         pin_memory=False, drop_last=False)
    return train_dl, val_dl, test_dl
