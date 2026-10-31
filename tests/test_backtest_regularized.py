"""Leakage and cash-flow checks for daily-refit regularized backtests."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from backtest_regularized import evaluation_dates, fit_day, validate_record, write_reports
from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.data_loader import NUMBER_COLUMNS
from src.regularized_models import RegularizedLGBMPredictor


class RegularizedBacktestTests(unittest.TestCase):
    def test_partial_year_is_inclusive_and_does_not_extend_to_future(self) -> None:
        dates = evaluation_dates(2026, "2026-10-29")
        self.assertEqual(len(dates), 302)
        self.assertEqual(dates[0], pd.Timestamp("2026-01-01"))
        self.assertEqual(dates[-1], pd.Timestamp("2026-10-29"))
        self.assertEqual(len(evaluation_dates(2025)), 365)
        self.assertEqual(len(evaluation_dates(2024)), 366)

    def test_invalid_cutoffs_fail_before_any_source_access(self) -> None:
        for value in ["2025-12-31", "2027-01-01", "2026-2-01", "2026-02-30", "2026-01-01T00:00:00", ""]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation_dates(2026, value)

    def test_daily_fit_matches_truncated_source_and_ignores_future(self) -> None:
        rng = np.random.default_rng(51)
        frequency = pd.DataFrame(
            rng.multinomial(27, np.full(100, .01), size=1120),
            index=pd.date_range("2022-01-01", periods=1120), columns=NUMBER_COLUMNS,
        )
        target = frequency.index[-4]
        dataset = build_count_feature_dataset(frequency, frequency.gt(0), include_unlabeled=True)
        record = fit_day(dataset, target)
        validate_record(record, target)
        prefix = frequency.loc[frequency.index < target]
        prefix_data = build_count_feature_dataset(prefix, prefix.gt(0), include_unlabeled=True)
        model = RegularizedLGBMPredictor().fit(prefix_data)
        expected = model.predict_proba(prefix_data.iloc[-100:].drop(columns=COUNT_TARGET)).to_numpy()
        np.testing.assert_array_equal(record["all_scores"], expected)
        self.assertEqual(record["training_days"], 1095)
        self.assertEqual(pd.Timestamp(record["training_label_end"]), target - pd.Timedelta(days=1))
        changed = frequency.copy()
        changed.loc[changed.index >= target] = 0
        changed.loc[changed.index >= target, "99"] = 27
        changed_data = build_count_feature_dataset(changed, changed.gt(0), include_unlabeled=True)
        self.assertEqual(fit_day(changed_data, target), record)

    def test_reports_count_repeats_and_keep_missing_cash_flows_unknown(self) -> None:
        dates = pd.date_range("2025-01-01", "2025-01-04")
        frequency = pd.DataFrame(0, index=pd.to_datetime(["2024-12-31", "2025-01-01", "2025-01-03", "2025-01-04"]), columns=NUMBER_COLUMNS)
        frequency["99"] = 27
        frequency.loc["2025-01-01", ["00", "99"]] = [2, 25]
        frequency.loc["2025-01-04", ["00", "99"]] = [3, 24]
        scores = np.linspace(.9, .1, 100).tolist()
        records = {day: {
            "target_date": str(day.date()), "history_end": str((day-pd.Timedelta(days=1)).date()),
            "training_label_start": "2024-12-01", "training_label_end": "2024-12-31",
            "training_days": 31, "all_scores": scores,
        } for day in [dates[0], dates[-1]]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output = Path(directory)
            summary = write_reports(frequency, dates, records, output, 10)
            daily = pd.read_csv(output / "daily.csv", index_col="target_date")
            self.assertEqual(summary["days"], 2)
            self.assertEqual(summary["excluded_days"], 2)
            self.assertEqual(summary["occurrences"], 5)
            self.assertEqual(summary["distinct_hits"], 2)
            self.assertEqual(summary["cost"], 460_000)
            self.assertEqual(summary["receipts"], 400_000)
            self.assertEqual(summary["profit"], -60_000)
            self.assertEqual(summary["profitable_days"], 1)
            self.assertEqual(summary["losing_days"], 1)
            self.assertEqual(daily.loc["2025-01-02", "status"], "missing_actual")
            self.assertEqual(daily.loc["2025-01-03", "status"], "missing_previous_day")
            self.assertTrue(daily.loc[["2025-01-02", "2025-01-03"], "cost"].isna().all())
            self.assertEqual(daily.loc["2025-01-04", "hits_with_counts"], "00x3")
            self.assertEqual(json.loads((output / "summary.json").read_text())["profit"], -60_000)

    def test_cache_rejects_future_training_and_nonfinite_scores(self) -> None:
        target = pd.Timestamp("2025-01-01")
        record = {"target_date": "2025-01-01", "history_end": "2024-12-31",
                  "training_label_start": "2024-01-01", "training_label_end": "2024-12-31",
                  "all_scores": [.25] * 100}
        validate_record(record, target)
        for update in [{"training_label_end": "2025-01-01"}, {"all_scores": [float("nan")] * 100},
                       {"all_scores": [.25] * 99}, {"history_end": "2024-12-30"}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_record({**record, **update}, target)


if __name__ == "__main__":
    unittest.main()
