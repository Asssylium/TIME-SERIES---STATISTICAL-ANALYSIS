"""Verify metric definitions, input contracts, reporting, and CV compatibility."""

import io
import unittest

import numpy as np
import pandas as pd

from src.data_loader import NUMBER_COLUMNS
from src.evaluator import evaluate_test_set, evaluate_top_k, print_summary_report
from src.models import BaselinePredictor, cross_validate


class EvaluatorTests(unittest.TestCase):
    def test_daily_metrics_with_duplicates_and_zero(self) -> None:
        result = evaluate_top_k(["00", "00", 1, "02"], [0, "01", "99"], k=3)
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["n_actual"], 4)
        self.assertEqual(result["n_unique_actual"], 3)
        self.assertEqual(result["hit_rate_at_k"], 1.0)
        self.assertEqual(result["accuracy_at_k"], 1.0)
        self.assertAlmostEqual(result["precision_at_k"], 2 / 3)
        self.assertAlmostEqual(result["recall_at_k"], 2 / 3)

    def test_default_k_and_no_hits(self) -> None:
        result = evaluate_top_k([99] * 27, list(range(15)))
        self.assertEqual(result["k"], 15)
        for key in ["hits", "hit_rate_at_k", "accuracy_at_k", "precision_at_k", "recall_at_k"]:
            self.assertEqual(result[key], 0)

    def test_full_domain(self) -> None:
        result = evaluate_top_k(["00", "99", "99"], list(range(100)), k=100)
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["precision_at_k"], 0.02)
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_scored_predictions_preserve_ranking_and_ignore_tail(self) -> None:
        # Scores are metadata; use the supplied ranking and only the first K.
        result = evaluate_top_k(["01"], [("02", 0.1), ("01", 0.9)], k=1)
        self.assertEqual(result["hits"], 0)
        result = evaluate_top_k(["01"], ["01", "01", "ignored"], k=1)
        self.assertEqual(result["hits"], 1)

    def test_batch_metrics_and_macro_recall(self) -> None:
        result = evaluate_test_set(
            [[1, 1, 2, 3], [4]], [[1, 8], [4, 9]], k=2,
            dates=["2026-01-01", "2026-01-03"],
        )
        self.assertEqual(result.summary["n_days"], 2)
        self.assertEqual(result.summary["days_with_hit"], 2)
        self.assertEqual(result.summary["total_hits"], 2)
        self.assertEqual(result.summary["mean_hits"], 1.0)
        self.assertEqual(result.summary["precision_at_k"], 0.5)
        self.assertAlmostEqual(result.summary["recall_at_k"], (1 / 3 + 1) / 2)
        self.assertIsInstance(result.per_day.index, pd.DatetimeIndex)
        self.assertEqual(result.per_day.index.name, "date")
        mixed = evaluate_test_set([[1], [2]], [[1], [3]], k=1)
        self.assertEqual(mixed.summary["hit_rate_at_k"], 0.5)
        self.assertEqual(mixed.summary["accuracy_at_k"], 0.5)
        self.assertEqual(mixed.summary["mean_hits"], 0.5)

    def test_report_is_printed_and_returned(self) -> None:
        result = evaluate_test_set([[1], [2]], [[1], [3]], k=1)
        output = io.StringIO()
        report = print_summary_report(result, file=output)
        self.assertEqual(output.getvalue(), report + "\n")
        self.assertIn("Hit Rate / Accuracy@1: 50.00%", report)
        self.assertIn("Mean distinct hits per day: 0.500", report)
        self.assertIn("Days with at least one hit: 1/2", report)

    def test_arrays_series_and_inputs_not_mutated(self) -> None:
        truth = np.array([0, 0, 99])
        prediction = pd.Series(["00", "99"])
        before_truth = truth.copy()
        before_prediction = prediction.copy()
        result = evaluate_top_k(truth, prediction, k=np.int64(2))
        self.assertEqual(result["recall_at_k"], 1)
        np.testing.assert_array_equal(truth, before_truth)
        pd.testing.assert_series_equal(prediction, before_prediction)
        result = evaluate_test_set(np.array([[0, 0], [1, 1]]), np.array([[0], [2]]), k=1)
        self.assertEqual(result.summary["hit_rate_at_k"], 0.5)

    def test_reject_invalid_numbers_and_predictions(self) -> None:
        for value in [-1, 100, "100", "001", "abc", True, np.bool_(False), 1.0, np.nan, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_top_k([value], [1], k=1)
            with self.subTest(prediction=value), self.assertRaises(ValueError):
                evaluate_top_k([1], [value], k=1)
        for predictions in [["1", "01"], [1], [(1, np.inf), (2, 0.1)], [(1,)], [(1, True), (2, 0.1)]]:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_top_k([1], predictions, k=2)
        for k in [0, 101, -1, True, 1.5, "15"]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                evaluate_top_k([1], [1], k=k)

    def test_reject_missing_truth_and_misaligned_batches(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_top_k([], [1], k=1)
        for actual, predicted in [([], []), ([[1]], []), ([1], [[1]]), ([[1]], [1]), ([[1]], [[]])]:
            with self.subTest(actual=actual, predicted=predicted), self.assertRaises(ValueError):
                evaluate_test_set(actual, predicted, k=1)
        with self.assertRaises(ValueError):
            evaluate_top_k("01", [1], k=1)
        with self.assertRaises(ValueError):
            evaluate_top_k([1], {1}, k=1)

    def test_reject_invalid_dates(self) -> None:
        for dates in [["2026-01-01"], ["2026-01-01"] * 2,
                      ["2026-01-02", "2026-01-01"], [None, "2026-01-01"],
                      ["wrong", "2026-01-01"], "2026-01-01",
                      pd.date_range("2026-01-01", periods=2, tz="UTC")]:
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                evaluate_test_set([[1], [2]], [[1], [2]], k=1, dates=dates)

    def test_metrics_match_existing_cross_validation(self) -> None:
        rng = np.random.default_rng(13)
        frequency = pd.DataFrame(
            rng.multinomial(27, np.full(100, 0.01), size=40),
            index=pd.date_range("2026-01-01", periods=40), columns=NUMBER_COLUMNS,
        )
        cv = cross_validate(BaselinePredictor(), frequency, frequency.gt(0), n_splits=2, top_k=15)
        for fold, predictions in cv.predictions.groupby("fold"):
            truths, ranked, dates = [], [], []
            for date, day in predictions.groupby(level=0):
                numbers = day.index.get_level_values(1)
                truths.append(numbers[day.target_next_day.to_numpy(dtype=bool)].tolist())
                order = np.argsort(-day.score.to_numpy(), kind="stable")[:15]
                ranked.append([(numbers[i], float(day.score.iloc[i])) for i in order])
                dates.append(date + pd.Timedelta(days=1))
            result = evaluate_test_set(truths, ranked, k=15, dates=dates)
            for key in ["precision_at_k", "recall_at_k", "hit_rate_at_k", "mean_hits"]:
                self.assertAlmostEqual(result.summary[key], cv.fold_metrics.loc[fold, key])

    def test_vectorized_batch_matches_reference_sets(self) -> None:
        rng = np.random.default_rng(27)
        truths = rng.integers(0, 100, size=(1000, 27))
        ranked = np.argsort(rng.random((1000, 100)), axis=1)[:, :15]
        result = evaluate_test_set(truths, ranked)
        expected_hits = [len(set(truth) & set(prediction)) for truth, prediction in zip(truths, ranked)]
        expected_recall = [hits / len(set(truth)) for hits, truth in zip(expected_hits, truths)]
        np.testing.assert_array_equal(result.per_day.hits, expected_hits)
        np.testing.assert_allclose(result.per_day.recall_at_k, expected_recall)


if __name__ == "__main__":
    unittest.main()
