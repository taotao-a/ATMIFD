"""Configuration regressions; these tests do not require PyTorch."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from util.parser_MSDS import parse_args, validate_args
from util.runtime import CACHE_KEYS, PROJECT_ROOT, prepare_args, project_path


class RuntimeTests(unittest.TestCase):
    def test_import_does_not_parse_command_line(self):
        command = [sys.executable, '-c',
                   'import sys; sys.argv=["test", "--not-a-real-option"]; import util.parser_MSDS']
        result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_entry_point_help(self):
        for name in ('main.py', 'thr_only.py', 'scripts/run_ablation.py', 'scripts/smoke_test.py'):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, str(PROJECT_ROOT / name), '--help'],
                                        cwd=str(PROJECT_ROOT.parent), capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(b'usage:', result.stdout)

    def test_full_model_defaults(self):
        args = parse_args([])
        validate_args(args)
        self.assertEqual(args['raw_node'], 11)
        self.assertEqual(args['main_model'], 'ATMIFD')
        self.assertTrue(args['imb_loss'] and args['thr_search'])

    def test_invalid_boolean_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--imb_loss', 'maybe'])

    def test_invalid_configuration(self):
        for options in (['--batch_size', '0'], ['--step', '2'], ['--trace_node_dim', '3'],
                        ['--raw_node', '5'], ['--feature_node', '3'], ['--thr_steps', '1'],
                        ['--threshold', '2'], ['--num_heads_n2e', '3'], ['--epochs', '1']):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_args(parse_args(options))

    def test_short_loss_run_is_allowed(self):
        validate_args(parse_args(['--epochs', '1', '--eval_stage', 'loss']))

    def test_paths_are_project_relative(self):
        self.assertEqual(Path(project_path('data/MSDS-pre')), PROJECT_ROOT / 'data/MSDS-pre')

    def test_evaluation_needs_model_path(self):
        with self.assertRaisesRegex(ValueError, 'model_path'):
            prepare_args(parse_args([]), [], threshold_only=True)

    def test_restore_config_and_cache_checks(self):
        with tempfile.TemporaryDirectory(prefix='atmifd-config-') as directory:
            root = Path(directory)
            data, cache, model = root / 'data', root / 'cache', root / 'model'
            for path in (data, cache, model):
                path.mkdir()
            (data / 'trace_path.pkl').touch()
            (cache / '0.pkl').touch()
            # Only an F1 checkpoint is present; threshold search must not require loss.
            (model / 'my_f1_stage.ckpt').touch()
            saved = parse_args([])
            validate_args(saved)
            saved.update(data_path=str(data), dataset_path=str(cache), imb_loss=False,
                         main_model='LegacyRun', gpu=True)
            (model / 'params.json').write_text(json.dumps(saved), encoding='utf-8')
            options = ['--model_path', str(model), '--device', 'cpu', '--threshold', '0.4',
                       '--thr_search', 'false']
            args = prepare_args(parse_args(options), options, threshold_only=True)
            self.assertFalse(args['imb_loss'])
            self.assertFalse(args['thr_search'])
            self.assertEqual(args['main_model'], 'LegacyRun')
            self.assertEqual(args['threshold'], 0.4)
            self.assertEqual(args['device'], 'cpu')
            (cache / 'cache_config.json').write_text(
                json.dumps({key: args[key] for key in CACHE_KEYS}), encoding='utf-8')
            prepare_args(parse_args(options), options, threshold_only=True)
            with self.assertRaisesRegex(ValueError, 'Cache configuration mismatch'):
                prepare_args(parse_args(options + ['--label_percent', '0.2']),
                             options + ['--label_percent', '0.2'], threshold_only=True)
            (cache / '0.pkl').unlink()
            with self.assertRaisesRegex(ValueError, 'empty or invalid'):
                prepare_args(parse_args(options), options, threshold_only=True)

    def test_ablation_dry_run_has_four_named_variants(self):
        result = subprocess.run([sys.executable, '-X', 'utf8', str(PROJECT_ROOT / 'scripts/run_ablation.py'), '--dry-run'],
                                cwd=str(PROJECT_ROOT.parent), capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('>>>'), 4)
        for name in ('MetricOnly', 'MetricTraceFusion', 'MetricTraceFusion+ThrSearch', 'ATMIFD'):
            self.assertIn('--main_model ' + name, result.stdout)
        self.assertEqual(result.stdout.count('--random_seed 42'), 4)


if __name__ == '__main__':
    unittest.main()
