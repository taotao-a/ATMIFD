"""Regression tests for training-only feature normalization."""

import copy
import unittest

import numpy as np

from util.runtime import (_apply_normalization, _fit_normalization,
                          split_window_ranges)


class NormalizationTests(unittest.TestCase):
    def test_validation_and_test_values_do_not_affect_fitted_statistics(self):
        window, n_raw, n_nodes, n_metrics, n_edges = 3, 32, 5, 5, 7
        metric = np.broadcast_to(
            np.arange(n_raw, dtype=np.float32)[:, None, None],
            (n_raw, n_nodes, n_metrics)).copy()
        edge = np.ones((n_raw, n_nodes, n_nodes, n_edges), dtype=np.float32)
        metric[19:] = 1000.0
        edge[19:] = 10.0
        records = []
        for start in range(n_raw - window + 1):
            records.append({
                'data_node': np.concatenate([
                    metric[start:start + window],
                    np.zeros((window, n_nodes, 6), dtype=np.float32),
                ], axis=-1),
                'data_edge': edge[start:start + window].copy(),
            })

        args = {'window': window, 'num_nodes': n_nodes, 'metric_len': n_metrics,
                'raw_edge': n_edges, 'trace_node_dim': 6}
        ranges = split_window_ranges(len(records), window)
        train_start, train_end = ranges['train']
        stats = _fit_normalization(records[train_start:train_end], args)

        np.testing.assert_allclose(stats['metric_min'], 0.0)
        np.testing.assert_allclose(stats['metric_max'], 18.0)
        np.testing.assert_allclose(stats['trace_mean'], 1.0)

        normalized = copy.deepcopy(records)
        _apply_normalization(normalized, args, stats)
        self.assertLessEqual(normalized[train_end - 1]['data_node'][..., :n_metrics].max(), 1.0)
        val_start = ranges['val'][0]
        self.assertGreater(normalized[val_start]['data_node'][..., :n_metrics].max(), 1.0)
        np.testing.assert_allclose(normalized[val_start]['data_edge'], 10.0, rtol=1e-5)
        self.assertTrue(np.isfinite(normalized[val_start]['data_node']).all())

    def test_saved_statistics_are_deterministic(self):
        args = {'num_nodes': 1, 'metric_len': 1, 'raw_edge': 1, 'trace_node_dim': 0}
        records = [{
            'data_node': np.array([[[1.0]], [[3.0]]], dtype=np.float32),
            'data_edge': np.array([[[[2.0]]], [[[4.0]]]], dtype=np.float32),
        }]
        stats = _fit_normalization(records, args)
        first, second = copy.deepcopy(records), copy.deepcopy(records)
        _apply_normalization(first, args, stats)
        _apply_normalization(second, args, copy.deepcopy(stats))
        np.testing.assert_array_equal(first[0]['data_node'], second[0]['data_node'])
        np.testing.assert_array_equal(first[0]['data_edge'], second[0]['data_edge'])


if __name__ == '__main__':
    unittest.main()
