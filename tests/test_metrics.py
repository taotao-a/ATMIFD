"""Regression tests for corrected evaluation semantics."""

import inspect
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from util.train import MY, class_weights_from_counts
from util.util import calc_index


class MetricSemanticsTests(unittest.TestCase):
    def test_auc_and_ap_always_use_anomaly_class(self):
        labels = np.array([0, 0, 0, 1])
        anomaly_score = np.array([0.2, 0.1, 0.4, 0.3])
        probabilities = np.column_stack([1.0 - anomaly_score, anomaly_score])

        _, result = calc_index(probabilities, labels, threshold=0.25)

        self.assertAlmostEqual(result['auc'], roc_auc_score(labels, anomaly_score))
        self.assertAlmostEqual(result['ap'], average_precision_score(labels, anomaly_score))
        normal_ap = average_precision_score(1 - labels, 1.0 - anomaly_score)
        self.assertNotAlmostEqual(result['ap'], normal_ap)

    def test_fit_accepts_validation_not_test_loader(self):
        parameters = inspect.signature(MY.fit).parameters
        self.assertIn('val_loader', parameters)
        self.assertNotIn('test_loader', parameters)
        source = inspect.getsource(MY.fit)
        self.assertIn('self.evaluate(val_loader)', source)

    def test_class_weight_matches_manuscript_formula(self):
        weights = class_weights_from_counts(100, 10, 20)
        self.assertEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 100 / (10 + 1e-6))
        self.assertEqual(class_weights_from_counts(1000, 10, 5), (1.0, 5.0))
        self.assertEqual(class_weights_from_counts(0, 10, 5), (1.0, 1.0))

    def test_loss_ablation_keeps_cross_entropy_form(self):
        source = inspect.getsource(MY.fit)
        self.assertNotIn('BCEWithLogitsLoss', source)
        self.assertIn('CrossEntropyLoss()', source)


if __name__ == '__main__':
    unittest.main()
