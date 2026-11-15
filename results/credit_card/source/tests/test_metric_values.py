"""Small hand-calculated cases pin down metric meaning and threshold behavior."""

import unittest

import numpy as np

from creditrisk.metrics import metrics
from creditrisk.verify import independent_metrics


class MetricValueTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_counts(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.3, 0.1])
        result = metrics(y, p, 0.5)
        for name in ["tn", "fp", "fn", "tp"]:
            self.assertEqual(result[name], 1)
        for name in ["precision", "recall", "f1", "accuracy", "balanced_accuracy"]:
            self.assertAlmostEqual(result[name], 0.5)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_inclusive_cutoff_and_ranking_threshold_independence(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.3, 0.1])
        result = metrics(y, p, 0.3)
        self.assertEqual(result["tp"], 2)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], 0.75)

    def test_manual_counts_match_independent_sklearn_calls(self):
        rng = np.random.default_rng(35)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        for cutoff in [0.05, 0.5, 0.95]:
            a, b = metrics(y, p, cutoff), independent_metrics(y, p, cutoff)
            for key in a:
                self.assertAlmostEqual(a[key], b[key])
