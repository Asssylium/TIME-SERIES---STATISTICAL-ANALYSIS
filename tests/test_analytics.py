"""Stable API analytics schema for chart consumers."""

import unittest

import numpy as np

from src.analytics import build_analytics_payload
from src.diagnostics import evaluate_payouts
from src.evaluator import evaluate_test_set


class AnalyticsPayloadTests(unittest.TestCase):
    def test_daily_fields_and_summary_have_stable_names(self) -> None:
        evaluation = evaluate_test_set(
            [[1, 1, 2], [4]], [[1, 8], [3, 4]], k=2,
            dates=["2026-01-01", "2026-01-02"],
        )
        frequency = np.zeros((2, 100), dtype=int)
        frequency[0, 1], frequency[0, 2], frequency[0, 99] = 2, 1, 24
        frequency[1, 4], frequency[1, 99] = 1, 26
        selected = np.zeros_like(frequency)
        selected[0, [1, 8]] = 1
        selected[1, [3, 4]] = 1
        payout = evaluate_payouts(
            frequency, selected, payout_mode="per_occurrence",
            stake=27_000, gross_payout=99_000,
        )

        payload = build_analytics_payload({"baseline": evaluation}, {"baseline": payout})

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["period"], {
            "start_date": "2026-01-01", "end_date": "2026-01-02",
        })
        model = payload["models"][0]
        self.assertEqual(model["model_id"], "baseline")
        self.assertEqual(set(model["summary"]), {
            "days", "days_with_hit", "hit_rate", "total_hits", "mean_hits",
            "precision_at_k", "recall_at_k", "total_stake", "total_receipts",
            "total_profit", "roi", "max_drawdown",
        })
        self.assertEqual(set(model["series"][0]), {
            "date", "hits", "hit", "precision_at_k", "recall_at_k",
            "selections", "paid_occurrences", "stake", "receipts", "profit",
            "roi", "cumulative_profit", "cumulative_roi", "drawdown",
        })
        self.assertEqual(model["series"][0]["hits"], 1)
        self.assertEqual(model["series"][0]["paid_occurrences"], 2)
        self.assertEqual(model["series"][0]["stake"], 54_000)
        self.assertEqual(model["summary"]["days"], 2)

    def test_rejects_mismatched_model_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "same models"):
            build_analytics_payload({"baseline": object()}, {})


if __name__ == "__main__":
    unittest.main()
