"""Tests for the paired 000-099 Top-3 experiment."""

import unittest

import numpy as np
import pandas as pd

from scripts.research.research_top3_000_099 import (
    ALL_CANDIDATES,
    MODEL_NAMES,
    RESTRICTED_CANDIDATES,
    evaluate_scores,
    poisson_binomial_upper_tail,
    random_pick_daily_hit_probabilities,
    random_pick_hit_probability,
    rank_candidates,
)
from scripts.research.research_top3_3digit import NUMBER_COLUMNS_3D


class RestrictedTop3ResearchTests(unittest.TestCase):
    def test_ranking_never_leaves_000_099(self) -> None:
        scores = np.zeros((2, 1000), dtype=float)
        scores[0, [999, 98, 97, 96]] = [100, 4, 3, 2]
        scores[1, [500, 5, 4, 3]] = [100, 4, 3, 2]

        selected, order = rank_candidates(scores, RESTRICTED_CANDIDATES)

        self.assertTrue((order < 100).all())
        self.assertEqual(order[0].tolist(), [98, 97, 96])
        self.assertEqual(order[1].tolist(), [5, 4, 3])
        self.assertTrue(selected.sum(axis=1).tolist() == [3, 3])

    def test_random_reference_uses_distinct_actual_numbers(self) -> None:
        actual = np.zeros((1, 1000), dtype=int)
        actual[0, 0] = 2
        actual[0, 1] = 1
        expected = 1 - (98 * 97 * 96) / (100 * 99 * 98)

        observed = random_pick_hit_probability(actual, RESTRICTED_CANDIDATES)
        daily = random_pick_daily_hit_probabilities(actual, RESTRICTED_CANDIDATES)

        self.assertAlmostEqual(observed, expected)
        self.assertEqual(daily.shape, (1,))
        self.assertAlmostEqual(daily[0], expected)

    def test_poisson_binomial_tail_handles_certain_trials(self) -> None:
        probabilities = np.array([1.0, 0.0, 0.5])

        self.assertEqual(poisson_binomial_upper_tail(probabilities, 1), 1.0)
        self.assertAlmostEqual(poisson_binomial_upper_tail(probabilities, 2), 0.5)
        self.assertEqual(poisson_binomial_upper_tail(probabilities, 3), 0.0)

    def test_evaluation_pairs_same_scores_under_both_scopes(self) -> None:
        dates = pd.date_range("2026-01-01", periods=2)
        values = np.zeros((2, 1000), dtype=np.uint8)
        values[:, 400] = 23
        values[0, 1] = 1
        values[0, 400] -= 1
        counts = pd.DataFrame(values, index=dates, columns=NUMBER_COLUMNS_3D)
        scores = np.zeros((2, 1000), dtype=float)
        scores[:, [400, 1, 2, 3]] = [10, 9, 8, 7]
        score_sets = {model: scores for model in MODEL_NAMES}

        daily = evaluate_scores(dates, counts, score_sets)

        self.assertEqual(len(daily), 12)
        restricted = daily.loc[daily.scope.eq("restricted_000_099")]
        unrestricted = daily.loc[daily.scope.eq("all_000_999")]
        self.assertTrue(restricted.prediction_top3.str.fullmatch(r"\d{3} \d{3} \d{3}").all())
        self.assertTrue(restricted.prediction_top3.str.split().map(
            lambda picks: all(int(pick) < 100 for pick in picks)
        ).all())
        self.assertTrue(restricted.groupby("model").hit_day.sum().eq(1).all())
        self.assertTrue(unrestricted.groupby("model").hit_day.sum().eq(2).all())

    def test_all_candidate_ranking_can_select_999(self) -> None:
        scores = np.zeros((1, 1000), dtype=float)
        scores[0, 999] = 1
        _, order = rank_candidates(scores, ALL_CANDIDATES)
        self.assertEqual(order[0, 0], 999)


if __name__ == "__main__":
    unittest.main()
