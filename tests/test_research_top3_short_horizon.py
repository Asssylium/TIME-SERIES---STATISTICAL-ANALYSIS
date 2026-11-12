import unittest

import numpy as np

from scripts.research.research_top3_short_horizon import expected_top_k_metrics


class ExpectedTopKMetricsTests(unittest.TestCase):
    def test_tie_neutral_expectations_match_uniform_subset(self):
        scores = np.zeros(100)
        actual = np.zeros(100, dtype=np.int64)
        actual[4] = 1

        (
            expected_occurrences, expected_hit, order, stable_occurrences,
            stable_hit, tie_size,
        ) = expected_top_k_metrics(scores, actual)

        self.assertAlmostEqual(expected_occurrences, 0.03)
        self.assertAlmostEqual(expected_hit, 0.03)
        self.assertEqual(order.tolist(), [0, 1, 2])
        self.assertEqual(stable_occurrences, 0)
        self.assertFalse(stable_hit)
        self.assertEqual(tie_size, 100)

    def test_scores_above_cutoff_are_always_selected(self):
        scores = np.zeros(100)
        scores[:2] = [2, 1]
        actual = np.zeros(100, dtype=np.int64)
        actual[1] = 1
        actual[4] = 1

        (
            expected_occurrences, expected_hit, order, stable_occurrences,
            stable_hit, tie_size,
        ) = expected_top_k_metrics(scores, actual)

        self.assertAlmostEqual(expected_occurrences, 1 + 1 / 98)
        self.assertEqual(expected_hit, 1.0)
        self.assertEqual(order.tolist(), [0, 1, 2])
        self.assertEqual(stable_occurrences, 1)
        self.assertTrue(stable_hit)
        self.assertEqual(tie_size, 98)
