"""Tests for the isolated 23-result Top-3 research target."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.research_top3_excluding_prize4 import (
    _target_dates,
    count_results_excluding_prize4,
    replace_target_with_subset,
    run,
)
from src.count_features import COUNT_TARGET
from src.data_loader import NUMBER_COLUMNS
from src.prize_features import PRIZE_COLUMNS, PRIZE_WIDTHS, normalize_prizes


class Top3ExcludingPrizeFourTests(unittest.TestCase):
    def test_subset_counts_exclude_only_prize_four_and_preserve_repeats(self) -> None:
        raw = pd.DataFrame({
            column: ["0" * width]
            for column, width in PRIZE_WIDTHS.items()
        })
        raw.insert(0, "date", ["2026-01-01"])
        raw.loc[0, ["prize1", "prize2_1"]] = ["1", "1"]
        raw.loc[0, ["prize4_1", "prize4_2", "prize4_3", "prize4_4"]] = [
            "1", "2", "3", "4"
        ]

        normalized = normalize_prizes(raw)
        counts = count_results_excluding_prize4(normalized)

        self.assertEqual(tuple(normalized.columns), PRIZE_COLUMNS)
        self.assertEqual(tuple(counts.columns), NUMBER_COLUMNS)
        self.assertEqual(int(counts.loc[pd.Timestamp("2026-01-01")].sum()), 23)
        self.assertEqual(counts.loc[pd.Timestamp("2026-01-01"), "01"], 2)
        self.assertEqual(counts.loc[pd.Timestamp("2026-01-01"), "02"], 0)
        self.assertEqual(counts.loc[pd.Timestamp("2026-01-01"), "03"], 0)
        self.assertEqual(counts.loc[pd.Timestamp("2026-01-01"), "04"], 0)

    def test_subset_labels_align_to_next_calendar_day_without_filling_gaps(self) -> None:
        origins = pd.date_range("2026-01-01", periods=3, freq="D")
        index = pd.MultiIndex.from_product(
            [origins, NUMBER_COLUMNS], names=["date", "number"]
        )
        full = pd.DataFrame({COUNT_TARGET: np.full(len(index), 27)}, index=index)
        available = pd.date_range("2026-01-02", periods=1, freq="D").append(
            pd.DatetimeIndex(["2026-01-04"])
        )
        targets = pd.DataFrame(0, index=available, columns=NUMBER_COLUMNS, dtype=np.uint8)
        targets.loc["2026-01-02", "01"] = 23
        targets.loc["2026-01-04", "02"] = 23

        labeled = replace_target_with_subset(full, targets)

        self.assertEqual(int(labeled.xs(origins[0])[COUNT_TARGET].sum()), 23)
        self.assertTrue(labeled.xs(origins[1])[COUNT_TARGET].isna().all())
        self.assertEqual(int(labeled.xs(origins[2])[COUNT_TARGET].sum()), 23)

    def test_target_dates_require_previous_calendar_day(self) -> None:
        dates = pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
        counts = pd.DataFrame(0, index=dates, columns=NUMBER_COLUMNS)

        result = _target_dates(counts, pd.Timestamp("2026-01-01"))

        pd.testing.assert_index_equal(
            result,
            pd.DatetimeIndex(["2026-01-02", "2026-01-05"]),
        )

    def test_walk_forward_runner_writes_isolated_report_and_forecast(self) -> None:
        rng = np.random.default_rng(23)
        raw = pd.DataFrame({
            column: rng.integers(0, 10**width, 45).astype(str)
            for column, width in PRIZE_WIDTHS.items()
        })
        raw.insert(0, "date", pd.date_range("2026-01-01", periods=len(raw)))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            output = root / "research"
            raw.to_csv(source, index=False)

            protocol = run(source, output, start_date="2026-01-20", history_days=5)

            daily = pd.read_csv(output / "daily.csv")
            cutoffs = pd.read_csv(output / "training_cutoffs.csv")
            forecast = pd.read_csv(output / "latest_forecast.csv", dtype={"number": str})
            self.assertFalse(protocol["production_default_changed"])
            self.assertEqual(set(daily.strategy), {
                "poisson_full_27",
                "binary_full_27",
                "frequency_90d_23",
                "binary_excluding_prize4_23",
            })
            self.assertTrue(daily.groupby("date").size().eq(4).all())
            self.assertTrue((cutoffs.training_label_end < cutoffs.first_target_date).all())
            self.assertEqual(len(forecast), 3)
            self.assertEqual(forecast.forecast_date.nunique(), 1)
            self.assertTrue((output / "report.md").is_file())
            with self.assertRaises(FileExistsError):
                run(source, output, start_date="2026-01-20", history_days=5)

    def test_overlapping_quarter_is_invariant_to_evaluation_start(self) -> None:
        """A quarter replay must not reset cumulative feature history."""
        rng = np.random.default_rng(77)
        raw = pd.DataFrame({
            column: rng.integers(0, 10**width, 300).astype(str)
            for column, width in PRIZE_WIDTHS.items()
        })
        raw.insert(0, "date", pd.date_range("2025-07-01", periods=len(raw)))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            raw.to_csv(source, index=False)
            run(source, root / "full", start_date="2026-01-01", history_days=30)
            run(source, root / "quarter", start_date="2026-04-01", history_days=30)

            full = pd.read_csv(root / "full" / "daily.csv")
            quarter = pd.read_csv(root / "quarter" / "daily.csv")
            overlap = full.loc[full.date >= "2026-04-01"].reset_index(drop=True)
            pd.testing.assert_frame_equal(overlap, quarter.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
