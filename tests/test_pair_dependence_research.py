"""Tests for pairwise draw co-occurrence research."""

import unittest

import numpy as np
import pandas as pd

from scripts.research.analyze_pair_dependence import (
    FAIR_PAIR_PROBABILITY,
    _recommendation,
    _report_text,
    compare_periods,
    holm_adjust,
    screen_pairs,
    split_periods,
)
from src.data_loader import NUMBER_COLUMNS


class PairDependenceResearchTests(unittest.TestCase):
    def test_fair_pair_probability_matches_inclusion_exclusion(self) -> None:
        expected = 1 - 2 * (0.99**27) + (0.98**27)
        self.assertAlmostEqual(FAIR_PAIR_PROBABILITY, expected, places=14)
        self.assertGreater(FAIR_PAIR_PROBABILITY, 0)
        self.assertLess(FAIR_PAIR_PROBABILITY, 1)

    def test_holm_adjustment_is_monotone_and_restores_input_order(self) -> None:
        adjusted = holm_adjust(np.array([0.04, 0.01, 0.03]))
        np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])
        self.assertTrue(np.all(adjusted >= np.array([0.04, 0.01, 0.03])))

    def test_screen_detects_planted_pair_and_returns_all_hypotheses(self) -> None:
        rng = np.random.default_rng(42)
        days = 500
        counts = np.zeros((days, 100), dtype=np.uint8)
        counts[:, 0] = 1
        counts[:, 1] = 1
        other_draws = rng.integers(2, 100, size=(days, 25))
        for row, draws in enumerate(other_draws):
            counts[row] += np.bincount(draws, minlength=100).astype(np.uint8)
        frequency = pd.DataFrame(
            counts,
            index=pd.date_range("2020-01-01", periods=days),
            columns=NUMBER_COLUMNS,
        )

        result = screen_pairs(frequency)
        planted = result.loc[result.number_a.eq("00") & result.number_b.eq("01")].iloc[0]

        self.assertEqual(len(result), 4_950)
        self.assertEqual(planted.cooccurrence_days, days)
        self.assertLessEqual(planted.p_value_holm, 0.05)
        self.assertTrue(np.isfinite(result.p_value_holm).all())

    def test_split_is_chronological_and_stability_requires_both_periods(self) -> None:
        discovery_dates = pd.date_range("2022-01-01", periods=365)
        replication_dates = pd.date_range("2023-01-01", periods=365)
        dates = discovery_dates.append(replication_dates)
        counts = np.zeros((len(dates), 100), dtype=np.uint8)
        counts[:, 0] = 27
        frequency = pd.DataFrame(counts, index=dates, columns=NUMBER_COLUMNS)
        discovery, replication = split_periods(frequency)
        self.assertEqual(discovery.index[-1], pd.Timestamp("2022-12-31"))
        self.assertEqual(replication.index[0], pd.Timestamp("2023-01-01"))

        discovery_scores = pd.DataFrame({
            "number_a": ["00", "00"],
            "number_b": ["01", "02"],
            "effect_percentage_points": [2.0, 2.0],
            "p_value_holm": [0.01, 0.01],
        })
        replication_scores = pd.DataFrame({
            "number_a": ["00", "00"],
            "number_b": ["01", "02"],
            "effect_percentage_points": [1.0, -1.0],
            "p_value_holm": [0.02, 0.02],
        })
        compared = compare_periods(discovery_scores, replication_scores).set_index("number_b")
        self.assertTrue(compared.loc["01", "replicated_significant"])
        self.assertFalse(compared.loc["02", "replicated_significant"])

    def test_screen_rejects_noncanonical_draw_totals(self) -> None:
        counts = np.zeros((2, 100), dtype=np.uint8)
        counts[:, 0] = 26
        frequency = pd.DataFrame(
            counts, index=pd.date_range("2020-01-01", periods=2), columns=NUMBER_COLUMNS
        )
        with self.assertRaisesRegex(ValueError, "totaling 27"):
            screen_pairs(frequency)

    def test_report_preserves_two_digit_numbers_and_cautious_recommendation(self) -> None:
        summary = {
            "discovery": {"start": "2005-10-01", "end": "2022-12-31"},
            "replication": {"start": "2023-01-01", "end": "2026-10-29"},
            "discovery_significant_pairs": 0,
            "replication_significant_pairs": 0,
            "replicated_same_direction_pairs": 0,
            "recommendation": _recommendation(0),
        }
        pairs = pd.DataFrame({
            "number_a": [0],
            "number_b": [1],
            "cooccurrence_days": [5],
            "days": [100],
            "effect_percentage_points": [0.0],
            "p_value_holm": [1.0],
        })

        report = _report_text(summary, pairs, pairs)

        self.assertIn("| Khám phá | 00–01 |", report)
        self.assertIn("Không thêm luật dự đoán theo cặp", report)


if __name__ == "__main__":
    unittest.main()
