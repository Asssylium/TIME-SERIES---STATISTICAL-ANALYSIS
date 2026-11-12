"""Tests for 000-999 XSMB ending research."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.research_top3_3digit import (
    NUMBER_COLUMNS_3D,
    ThreeDigitFeatureBuilder,
    _rolling_suffix_shrinkage_scores,
    count_three_digit_endings,
    run,
)
from src.prize_features import PRIZE_WIDTHS, normalize_prizes


class ThreeDigitResearchTests(unittest.TestCase):
    def test_counts_use_last_three_digits_and_keep_leading_zeroes(self) -> None:
        raw = pd.DataFrame({
            column: ["0" * width]
            for column, width in PRIZE_WIDTHS.items()
        })
        raw.insert(0, "date", ["2026-01-01"])
        raw.loc[0, ["prize1", "prize2_1", "prize6_1", "prize7_1"]] = [
            "00123", "00001", "007", "01"
        ]
        raw.loc[0, ["prize4_1", "prize4_2", "prize4_3", "prize4_4"]] = [
            "0001", "0002", "0003", "0004"
        ]

        counts = count_three_digit_endings(normalize_prizes(raw))

        self.assertEqual(tuple(counts.columns), NUMBER_COLUMNS_3D)
        self.assertEqual(int(counts.iloc[0].sum()), 23)
        self.assertEqual(counts.iloc[0]["123"], 1)
        self.assertEqual(counts.iloc[0]["001"], 2)
        self.assertEqual(counts.iloc[0]["002"], 1)
        self.assertEqual(counts.iloc[0]["007"], 1)

    def test_features_use_only_observed_past_and_aggregate_suffixes(self) -> None:
        dates = pd.date_range("2026-01-01", periods=3)
        values = np.zeros((3, 1000), dtype=np.uint8)
        values[:, 0] = 23
        values[0, 123] = 2
        values[0, 0] -= 2
        values[1, 123] = 1
        values[1, 0] -= 1
        counts = pd.DataFrame(values, index=dates, columns=NUMBER_COLUMNS_3D)

        features, names = ThreeDigitFeatureBuilder(counts).transform(dates[:2])
        tensor = features.reshape(2, 1000, -1)
        column = {name: index for index, name in enumerate(names)}

        self.assertEqual(tensor.shape[1], 1000)
        self.assertEqual(tensor[1, 123, column["three_count_7d"]], 3)
        self.assertEqual(tensor[1, 123, column["two_digit_suffix_count_90d"]], 3)
        self.assertEqual(tensor[1, 123, column["last_digit_count_90d"]], 3)

    def test_rolling_windows_are_correct_for_microsecond_datetime_indexes(self) -> None:
        dates = pd.date_range("2025-01-01", periods=130).as_unit("us")
        values = np.zeros((len(dates), 1000), dtype=np.uint8)
        values[:, 0] = 23
        for day in (0, 50, 128):
            values[day, 0] -= 1
            values[day, 123] = 1
        counts = pd.DataFrame(values, index=dates, columns=NUMBER_COLUMNS_3D)

        features, names = ThreeDigitFeatureBuilder(counts).transform(dates[-1:])
        row = features.reshape(1, 1000, -1)[0, 123]
        column = {name: index for index, name in enumerate(names)}

        self.assertEqual(row[column["three_count_7d"]], 1)
        self.assertEqual(row[column["three_count_90d"]], 2)
        self.assertEqual(row[column["three_count_1095d"]], 3)

    def test_suffix_shrinkage_uses_only_prior_observations(self) -> None:
        dates = pd.date_range("2026-01-01", periods=5)
        values = np.zeros((5, 1000), dtype=np.uint8)
        values[:, 0] = 23
        for day, number, occurrences in (
            (0, 1, 2),
            (1, 101, 3),
            (2, 201, 1),
            (3, 301, 4),
            (4, 1, 23),
        ):
            values[day, 0] -= occurrences
            values[day, number] += occurrences
        counts = pd.DataFrame(values, index=dates, columns=NUMBER_COLUMNS_3D)

        scores = _rolling_suffix_shrinkage_scores(
            ThreeDigitFeatureBuilder(counts), dates[4:], window=90
        )[0]

        self.assertAlmostEqual(scores[1], 2 + 0.05 * 10)
        self.assertAlmostEqual(scores[101], 3 + 0.05 * 10)
        self.assertAlmostEqual(scores[301], 4 + 0.05 * 10)

    def test_walk_forward_runner_writes_three_digit_report(self) -> None:
        rng = np.random.default_rng(2026)
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

            protocol = run(
                source,
                output,
                start_date="2026-01-20",
                end_date="2026-01-30",
                history_days=5,
            )

            daily = pd.read_csv(output / "daily.csv")
            detail = pd.read_csv(output / "daily_detail.csv")
            for row in daily.itertuples(index=False):
                self.assertEqual(row.stake_units, 69)
                self.assertEqual(row.gross_payout_units, row.target_occurrences * 900)
                self.assertEqual(
                    row.net_profit_units,
                    row.target_occurrences * 900 - 69,
                )
            cutoffs = pd.read_csv(output / "training_cutoffs.csv")
            forecast = pd.read_csv(
                output / "latest_forecast.csv", dtype={"number": str}
            )
            self.assertEqual(protocol["target_digit_count"], 3)
            self.assertEqual(protocol["candidate_numbers"], 1000)
            self.assertEqual(protocol["target_draw_count_per_day"], 23)
            self.assertFalse(protocol["production_default_changed"])
            self.assertEqual(
                protocol["exploratory_strategy"]["suffix_count_weight"], 0.05
            )
            self.assertEqual(set(daily.strategy), {
                "poisson_3digit_23",
                "binary_3digit_23",
                "frequency_90d_3digit_23",
                "frequency_5y_suffix_shrinkage_3digit_23",
            })
            self.assertTrue(daily.groupby("date").size().eq(4).all())
            self.assertEqual(len(detail), 11)
            self.assertTrue(detail.binary_top3.str.fullmatch(r"\d{3} \d{3} \d{3}").all())
            self.assertTrue(detail.actual_23_endings_sorted.str.split().str.len().eq(23).all())
            self.assertTrue((detail.binary_stake == 69).all())
            self.assertTrue(
                (detail.binary_net_profit == detail.binary_gross_payout - 69).all()
            )
            self.assertEqual(daily.date.min(), "2026-01-20")
            self.assertEqual(daily.date.max(), "2026-01-30")
            self.assertEqual(protocol["evaluation_end"], "2026-01-30")
            self.assertTrue((cutoffs.training_label_end < cutoffs.first_target_date).all())
            self.assertEqual(len(forecast), 3)
            self.assertTrue(forecast.number.str.fullmatch(r"\d{3}").all())
            self.assertEqual(forecast.number.nunique(), 3)
            self.assertTrue((output / "report.md").is_file())
            self.assertIn("daily_detail.csv", (output / "report.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                run(
                    source,
                    output,
                    start_date="2026-01-20",
                    end_date="2026-01-30",
                    history_days=5,
                )


if __name__ == "__main__":
    unittest.main()
