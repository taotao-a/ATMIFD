"""Exercise runtime plumbing with synthetic data, not paper-result reproduction."""

import argparse
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    options = parser.parse_args()

    import numpy as np
    import torch
    from main import main as run_experiment
    from thr_only import main as run_threshold
    from util.parser_MSDS import parse_args, validate_args
    from util.runtime import build_loaders, prepare_args
    from util.data_MSDS import Process
    from util.train import MY
    from src.model import MyModel

    torch.set_num_threads(1)
    output_root = PROJECT_ROOT / 'result'
    output_root.mkdir(exist_ok=True)
    # Only this test's temporary directory is removed on completion.
    with tempfile.TemporaryDirectory(prefix='atmifd-smoke-', dir=str(output_root)) as directory:
        root = Path(directory)
        data, cache = root / 'data', root / 'cache'
        data.mkdir()
        cache.mkdir()
        graph = np.ones((5, 5)) - np.eye(5)
        with (data / 'trace_path.pkl').open('wb') as handle:
            pickle.dump(graph, handle)
        rng = np.random.RandomState(42)
        for i in range(30):
            labels = np.zeros(5, dtype=int)
            labels[0] = i % 2
            masked = labels.copy()
            masked[4] = 2
            record = {
                'data_node': rng.uniform(0.1, 1, (3, 5, 11)).astype(np.float32),
                'data_edge': rng.uniform(0.1, 1, (3, 5, 5, 7)).astype(np.float32),
                'data_log': np.zeros((3, 5, 1), dtype=np.float32),
                'groundtruth_cls': np.eye(3, dtype=np.float32)[masked],
                'groundtruth_real': np.eye(2, dtype=np.float32)[labels],
            }
            with (cache / '{}.pkl'.format(i)).open('wb') as handle:
                pickle.dump(record, handle)

        common = ['--data_path', str(data), '--dataset_path', str(cache),
                  '--result_dir', str(root / 'runs'), '--device', options.device,
                  '--window', '3', '--num_layer', '1', '--batch_size', '2',
                  '--epochs', '3', '--thr_steps', '5']
        for imbalance in ('false', 'true'):
            run_experiment(common + ['--imb_loss', imbalance])
        runs = sorted((root / 'runs').glob('*/params.json'))
        assert len(runs) == 2, 'Both loss variants must produce a run'
        for params in runs:
            assert (params.parent / 'my_loss_stage.ckpt').is_file()
            assert (params.parent / 'my_f1_stage.ckpt').is_file()
            assert (params.parent / 'evaluation.log').is_file()
        model_dir = str(runs[-1].parent)
        evaluate = ['--evaluate', 'true', '--model_path', model_dir,
                    '--device', options.device, '--eval_stage', 'both']
        run_experiment(evaluate)
        f1_only = root / 'f1-only'
        f1_only.mkdir()
        for name in ('params.json', 'my_f1_stage.ckpt'):
            shutil.copyfile(Path(model_dir) / name, f1_only / name)
        run_threshold(['--model_path', str(f1_only), '--device', options.device])

        # Explicit overrides must survive restoration of saved training parameters.
        restored = prepare_args(parse_args(evaluate + ['--thr_search', 'false']),
                                evaluate + ['--thr_search', 'false'])
        assert not restored['thr_search']

        # Batch size 1 must retain the batch dimension, including reconstruction loss.
        config = parse_args(common)
        validate_args(config)
        config['batch_size'] = 1
        processed = Process(**config)
        train_dl, _, _ = build_loaders(processed, config)
        trainer = MY(MyModel(graph, **config), **config)
        batch = trainer.input2device(next(iter(train_dl)))
        losses, logits, labels = trainer.model(batch)
        assert torch.isfinite(sum(losses)) and logits.shape == labels.shape
        probabilities, _ = trainer.model(batch, evaluate=True)
        assert tuple(probabilities.shape) == (1, 5, 2)

        # Reject accidental cache reuse between metric and fusion variants.
        config.update(trace_node_dim=0, raw_node=5)
        try:
            Process(**config)
        except ValueError as error:
            assert 'shape mismatch' in str(error)
        else:
            raise AssertionError('Mismatched cache was accepted')

        # Exercise prepared CSV -> fresh window cache -> reload for both feature settings.
        import pandas as pd
        from util.constant import MSDS_pod
        prepared = root / 'prepared'
        prepared.mkdir()
        shutil.copyfile(data / 'trace_path.pkl', prepared / 'trace_path.pkl')
        raw_labels = np.zeros((12, 5), dtype=int)
        raw_labels[1::2, 0] = 1
        with (prepared / 'label.pkl').open('wb') as handle:
            pickle.dump(raw_labels, handle)
        stamps = np.arange(1000, 1012)
        metric = {'now': stamps}
        for pod in MSDS_pod:
            for feature in ('cpu.user', 'mem.used', 'load.min1', 'load.min15', 'load.min5'):
                metric[pod + '_' + feature] = rng.uniform(0.1, 1, 12)
        pd.DataFrame(metric).to_csv(prepared / 'metric.csv', index=False)
        trace = [{'cmbd_id': MSDS_pod[0], 'fatherpod': MSDS_pod[1],
                  'stats': 'type{}'.format(kind), 'end_time': int(stamp), 'duration': 0.1}
                 for stamp in stamps for kind in range(7)]
        pd.DataFrame(trace).to_csv(prepared / 'trace.csv', index=False)
        for dim in (0, 6):
            raw_config = parse_args(common)
            raw_config.update(data_path=str(prepared), dataset_path=str(root / 'fresh-{}'.format(dim)),
                              trace_node_dim=dim, raw_node=5 + dim)
            validate_args(raw_config)
            generated = Process(**raw_config)
            reloaded = Process(**raw_config)
            assert len(generated.dataset) == len(reloaded.dataset) == 10
            for first, second in zip(generated.dataset, reloaded.dataset):
                for key in ('data_node', 'data_edge', 'data_log', 'groundtruth_cls', 'groundtruth_real'):
                    np.testing.assert_array_equal(first[key], second[key])
            manifest = Path(raw_config['dataset_path']) / 'cache_config.json'
            assert json.loads(manifest.read_text(encoding='utf-8'))['trace_node_dim'] == dim
        logging.shutdown()
    print('PASS: BCE/CE training, loss/F1 checkpoints, evaluation, threshold search, '
          'CLI overrides, batch size 1, partial batches, cache checks and CSV-to-cache generation.')
    print('Synthetic runtime test only; no paper metrics are asserted.')


if __name__ == '__main__':
    main()
