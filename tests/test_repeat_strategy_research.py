"""Causal Top-3 strategy selection and hit-rate research helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.analyze_repeat_strategies import (
    evaluate_period,
    longest_true_streak,
    score_daily,
    top_numbers,
    wilson_interval,
)


class RepeatStrategyResearchTests(unittest.TestCase):
    def test_top_numbers_is_stable_and_respects_repeat_exclusions(self):
        scores = np.zeros(100)
        scores[:6] = [3, 5, 5, 4, 2, 1]
        self.assertEqual(top_numbers(scores), (1, 2, 3))
        self.assertEqual(top_numbers(scores, excluded={1, 2, 3}), (0, 4, 5))

    def test_daily_score_separates_hit_days_and_occurrence_payout(self):
        actual = np.zeros(100, dtype=int)
        actual[4], actual[8], actual[90] = 2, 1, 24
        result = score_daily((4, 5, 6), actual, previous_pick_count={4, 6})
        self.assertTrue(result["hit"])
        self.assertEqual(result["distinct_hits"], 1)
        self.assertEqual(result["occurrences"], 2)
        self.assertEqual(result["repeated_selections"], 2)
        self.assertEqual(result["repeated_occurrences"], 2)
        self.assertEqual(result["repeated_profit_vnd"], 144_000)
        self.assertEqual(result["cost_vnd"], 81_000)
        self.assertEqual(result["receipts_vnd"], 198_000)
        self.assertEqual(result["profit_vnd"], 117_000)

    def test_wilson_interval_and_streaks(self):
        lower, upper = wilson_interval(50, 100)
        self.assertAlmostEqual(lower, 0.4038, places=3)
        self.assertAlmostEqual(upper, 0.5962, places=3)
        self.assertEqual(longest_true_streak([False, True, True, False, True]), 2)
        self.assertEqual(
            longest_true_streak([True, True], pd.date_range("2025-01-01", periods=2).tolist()),
            2,
        )
        self.assertEqual(
            longest_true_streak([True, True], [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03")]),
            1,
        )
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_repeat_filter_only_uses_prior_forecasts(self):
        dates = pd.date_range("2025-01-01", periods=3)
        actual = pd.DataFrame(np.zeros((3, 100), dtype=int), index=dates, columns=[f"{n:02d}" for n in range(100)])
        actual.iloc[:, 99] = 27
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            for day in ("2025-01-02", "2025-01-03"):
                poisson = np.arange(100, 0, -1, dtype=float)
                binary = poisson[::-1].copy()
                np.savez_compressed(cache / f"{day}.npz", target_date=np.asarray(day), poisson=poisson, binary=binary)
            rows = evaluate_period(
                cache, actual, pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03"),
                period_name="test",
            )
        current = rows.loc[(rows.date == "2025-01-03") & (rows.strategy == "poisson_no_repeat_1d")].iloc[0]
        self.assertEqual(current.prediction, "03 04 05")
        self.assertEqual(current.repeated_selections, 0)
        self.assertEqual(len(rows), 10)

    def test_frequency_baseline_does_not_use_target_day_outcome(self):
        dates = pd.date_range("2025-01-01", periods=3)
        actual = pd.DataFrame(np.zeros((3, 100), dtype=int), index=dates, columns=[f"{n:02d}" for n in range(100)])
        actual.iloc[0, 0], actual.iloc[0, 1], actual.iloc[0, 2] = 27, 0, 0
        actual.iloc[1, 1], actual.iloc[1, 2], actual.iloc[1, 3] = 27, 0, 0
        actual.iloc[2, 99] = 27
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            for day in ("2025-01-02", "2025-01-03"):
                scores = np.arange(100, 0, -1, dtype=float)
                np.savez_compressed(cache / f"{day}.npz", target_date=np.asarray(day), poisson=scores, binary=scores)
            first = evaluate_period(
                cache, actual, pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-03"),
                period_name="test",
            )
            changed_future = actual.copy()
            changed_future.loc[pd.Timestamp("2025-01-03"), :] = 0
            changed_future.loc[pd.Timestamp("2025-01-03"), "98"] = 27
            second = evaluate_period(
                cache, changed_future, pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-03"),
                period_name="test",
            )
        pick_first = first.loc[first.strategy == "frequency_90d", "prediction"].iloc[0]
        pick_second = second.loc[second.strategy == "frequency_90d", "prediction"].iloc[0]
        self.assertEqual(pick_first, pick_second)


if __name__ == "__main__":
    unittest.main()
