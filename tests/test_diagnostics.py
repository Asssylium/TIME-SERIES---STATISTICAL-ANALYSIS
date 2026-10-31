"""Payout arithmetic, uncertainty, and causal research feature checks."""
import unittest

import numpy as np
import pandas as pd

from src.data_loader import NUMBER_COLUMNS
from src.diagnostics import (
    block_bootstrap_mean_interval, evaluate_payouts, probability_diagnostics, rank_top_k,
)
from src.features import build_feature_dataset
from research import apply_sigmoid, normalized_features, simple_forecasts


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.zeros((2, 100), dtype=int)
        self.counts[0, [1, 2, 99]] = [2, 1, 24]
        self.counts[1, 99] = 27
        self.selected = np.zeros_like(self.counts, dtype=bool)
        self.selected[:, [1, 2]] = True

    def test_repeated_payout_is_not_binary_precision(self):
        once = evaluate_payouts(self.counts, self.selected, payout_mode="once_per_day")
        repeated = evaluate_payouts(self.counts, self.selected, payout_mode="per_occurrence")
        self.assertEqual(once.summary["cost"], 92_000)
        self.assertEqual(once.summary["profit"], 68_000)
        self.assertEqual(repeated.summary["profit"], 148_000)
        self.assertEqual(once.summary["precision_at_k"], 0.5)
        self.assertEqual(repeated.summary["precision_at_k"], 0.5)
        self.assertEqual(repeated.summary["occurrences_per_selection"], 0.75)
        self.assertEqual(repeated.summary["break_even_units_per_selection"], 0.2875)

    def test_zero_selections_and_drawdown_include_initial_capital(self):
        empty = evaluate_payouts(self.counts, self.selected * 0, payout_mode="per_occurrence")
        self.assertEqual(empty.summary["profit"], 0)
        self.assertTrue(np.isnan(empty.summary["roi"]))
        losers = np.zeros_like(self.selected)
        losers[:, 98] = True
        loss = evaluate_payouts(self.counts, losers, payout_mode="once_per_day")
        self.assertEqual(loss.summary["max_drawdown"], 46_000)

    def test_probability_metrics_match_hand_calculation(self):
        y = np.zeros((2, 100))
        y[0] = 1
        p = np.full_like(y, 0.5)
        summary, table = probability_diagnostics(y, p)
        self.assertEqual(summary["brier"], 0.25)
        self.assertAlmostEqual(summary["log_loss"], np.log(2))
        self.assertEqual(summary["ece"], 0)
        self.assertEqual(table["count"].sum(), 200)
        perfect, _ = probability_diagnostics(y, y)
        self.assertEqual(perfect["brier"], 0)

    def test_invalid_financial_and_probability_inputs(self):
        for kwargs in [{"payout_mode": "bad"}, {"payout_mode": "once_per_day", "stake": -1}]:
            with self.assertRaises(ValueError):
                evaluate_payouts(self.counts, self.selected, **kwargs)
        with self.assertRaises(ValueError):
            evaluate_payouts(self.counts * 0, self.selected, payout_mode="once_per_day")
        with self.assertRaises(ValueError):
            probability_diagnostics(self.selected, np.full((2, 100), 1.1))
        with self.assertRaises(ValueError):
            rank_top_k(np.zeros((2, 100)), k=True)

    def test_deterministic_ranking_and_whole_day_bootstrap(self):
        selected = rank_top_k(np.zeros((4, 100)), k=5)
        self.assertTrue(selected[:, :5].all())
        self.assertFalse(selected[:, 5:].any())
        self.assertEqual(block_bootstrap_mean_interval(np.full(30, -7000.0)), (-7000.0, -7000.0))
        values = np.arange(100, dtype=float) - 50
        self.assertEqual(block_bootstrap_mean_interval(values), block_bootstrap_mean_interval(values))

    def test_simple_models_cannot_see_future(self):
        rng = np.random.default_rng(42)
        frequency = pd.DataFrame(
            rng.multinomial(27, np.full(100, 0.01), size=100),
            index=pd.date_range("2020-01-01", periods=100), columns=NUMBER_COLUMNS,
        ).drop(pd.Timestamp("2020-02-01"))
        whole = simple_forecasts(frequency)
        prefix = simple_forecasts(frequency.iloc[:60])
        for name in whole:
            for full_values, prefix_values in zip(whole[name], prefix[name]):
                np.testing.assert_allclose(full_values[:60], prefix_values)

    def test_sigmoid_identity_and_feature_normalization(self):
        p = np.linspace(0.01, 0.99, 100).reshape(1, 100)
        np.testing.assert_allclose(apply_sigmoid(p, {"enabled": True, "slope": 1.0, "intercept": 0.0}), p)
        np.testing.assert_array_equal(apply_sigmoid(p, {"enabled": False}), p)
        frequency = np.zeros((4, 100), dtype=int)
        frequency[:, 0] = 27
        dataset = build_feature_dataset(frequency, frequency > 0)
        before = dataset.copy(deep=True)
        x = normalized_features(dataset)
        self.assertNotIn("target_next_day", x.columns)
        self.assertNotIn("history_days", x.columns)
        self.assertTrue(x.loc[(slice(None), "00"), "rolling_frequency_7d"].eq(27).all())
        pd.testing.assert_frame_equal(dataset, before)


if __name__ == "__main__":
    unittest.main()
