"""Regression tests for calendar boundaries, counts, and causal features."""

import unittest

import numpy as np
import pandas as pd

from src.data_loader import NUMBER_COLUMNS, XSMBDataLoader
from src.features import (
    build_feature_dataset,
    compute_co_occurrence,
    compute_gap_days,
    compute_rolling_frequency,
)


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        draws = [[0] * 2 + [1] * 25, [2] * 27, [0] * 27, [1] * 27]
        self.frequency = pd.DataFrame(
            [np.bincount(row, minlength=100) for row in draws],
            index=pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-04", "2026-01-05"]),
            columns=NUMBER_COLUMNS,
            dtype=np.uint8,
        )
        self.binary = self.frequency.gt(0).astype(np.uint8)

    def test_rolling_calendar_boundaries(self) -> None:
        result = compute_rolling_frequency(self.frequency, windows=[1, 2, 7])
        pd.testing.assert_frame_equal(result[1], self.frequency.astype(np.int64))
        pd.testing.assert_series_equal(result[2].iloc[1], self.frequency.iloc[1].astype(np.int64))
        np.testing.assert_array_equal(result[2].iloc[2], self.frequency.iloc[1:3].sum())
        np.testing.assert_array_equal(result[7].iloc[-1], self.frequency.sum())
        self.assertEqual(list(compute_rolling_frequency(self.frequency)), [7, 14, 30, 60])

    def test_gap_calendar_days_and_unseen(self) -> None:
        gaps = compute_gap_days(self.binary)
        np.testing.assert_array_equal(gaps["00"], [0, 2, 0, 1])
        np.testing.assert_array_equal(gaps["01"], [0, 2, 3, 0])
        np.testing.assert_array_equal(gaps["02"], [-1, 0, 1, 2])
        self.assertTrue((gaps["99"] == -1).all())

    def test_co_occurrence_diagonal_and_duplicates(self) -> None:
        counts = compute_co_occurrence(self.binary)
        self.assertEqual(counts.shape, (100, 100))
        self.assertEqual(counts.loc["00", "01"], 1)
        self.assertEqual(counts.loc["00", "00"], 2)
        self.assertEqual(counts.loc["02", "01"], 0)
        pd.testing.assert_frame_equal(counts, counts.T)
        np.testing.assert_array_equal(np.diag(counts), self.binary.sum())

    def test_no_uint8_overflow(self) -> None:
        frequency = np.zeros((400, 100), dtype=np.uint8)
        frequency[:, 0] = 27
        rolling = compute_rolling_frequency(frequency, windows=[60])
        self.assertEqual(rolling[60].iloc[-1, 0], 1620)
        counts = compute_co_occurrence((frequency > 0).astype(np.uint8))
        self.assertEqual(counts.iloc[0, 0], 400)

    def test_dataset_labels_and_missing_dates(self) -> None:
        training = build_feature_dataset(self.frequency, self.binary)
        self.assertEqual(training.shape[0], 200)
        self.assertEqual(training.index.names, ["date", "number"])
        self.assertEqual(str(training.target_next_day.dtype), "UInt8")
        self.assertFalse(training.target_next_day.isna().any())
        for date in training.index.get_level_values("date").unique():
            np.testing.assert_array_equal(
                training.xs(date).target_next_day.to_numpy(dtype=np.uint8),
                self.binary.loc[date + pd.Timedelta(days=1)],
            )
        all_rows = build_feature_dataset(self.frequency, self.binary, include_unlabeled=True)
        self.assertEqual(len(all_rows), 400)
        self.assertTrue(all_rows.xs(self.frequency.index[0]).target_next_day.isna().all())
        self.assertTrue(all_rows.xs(self.frequency.index[-1]).target_next_day.isna().all())

    def test_causal_co_occurrence_summary(self) -> None:
        dataset = build_feature_dataset(self.frequency, self.binary, include_unlabeled=True)
        for position, date in enumerate(self.frequency.index):
            counts = compute_co_occurrence(self.binary.iloc[:position + 1])
            expected = counts.sum(axis=1).to_numpy() - np.diag(counts)
            np.testing.assert_array_equal(dataset.xs(date).co_occurrence_total, expected)

    def test_future_changes_cannot_change_past_features(self) -> None:
        before = build_feature_dataset(self.frequency, self.binary, include_unlabeled=True)
        changed = self.frequency.copy()
        changed.iloc[2:] = 0
        changed.loc[changed.index[2:], "99"] = 27
        after = build_feature_dataset(changed, changed.gt(0), include_unlabeled=True)
        pd.testing.assert_frame_equal(
            before.iloc[:200].drop(columns="target_next_day"),
            after.iloc[:200].drop(columns="target_next_day"),
        )
        prefix = build_feature_dataset(
            self.frequency.iloc[:2], self.binary.iloc[:2], include_unlabeled=True
        )
        pd.testing.assert_frame_equal(
            before.iloc[:200].drop(columns="target_next_day"),
            prefix.drop(columns="target_next_day"),
        )

    def test_single_day_and_ndarrays(self) -> None:
        frequency = self.frequency.iloc[:1].to_numpy()
        binary = self.binary.iloc[:1].to_numpy()
        self.assertTrue(build_feature_dataset(frequency, binary).empty)
        dataset = build_feature_dataset(frequency, binary, include_unlabeled=True)
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.index.names, ["day_index", "number"])
        self.assertTrue(dataset.target_next_day.isna().all())
        gaps = compute_gap_days(self.binary.to_numpy())
        np.testing.assert_array_equal(gaps["00"], [0, 1, 0, 1])

    def test_reordered_columns_and_no_input_mutation(self) -> None:
        snapshot = self.frequency.copy(deep=True)
        expected = build_feature_dataset(self.frequency, self.binary)
        actual = build_feature_dataset(self.frequency.iloc[:, ::-1], self.binary)
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(snapshot, self.frequency)

    def test_invalid_windows(self) -> None:
        for windows in [[], [0], [-1], [1.5], [True], [7, 7], "7", None, 7]:
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                compute_rolling_frequency(self.frequency, windows=windows)

    def test_invalid_values_shapes_and_alignment(self) -> None:
        for value in [np.nan, np.inf, -1, 1.5, 28]:
            bad = self.frequency.astype(float)
            bad.iloc[0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_rolling_frequency(bad)
        for bad in [self.frequency.iloc[:0], self.frequency.iloc[:, :99],
                    self.frequency.iloc[::-1], self.frequency.astype(str),
                    self.frequency * 0]:
            with self.assertRaises(ValueError):
                compute_rolling_frequency(bad)
        bad_binary = self.binary.copy()
        bad_binary.iloc[0, 0] = 2
        with self.assertRaises(ValueError):
            compute_gap_days(bad_binary)
        with self.assertRaises(ValueError):
            build_feature_dataset(self.frequency, self.binary * 0)
        with self.assertRaises(ValueError):
            build_feature_dataset(self.frequency, self.binary.iloc[:2])
        for index in [pd.DatetimeIndex(["2026-01-01"] * 4),
                      self.frequency.index.tz_localize("UTC"),
                      self.frequency.index + pd.Timedelta(hours=1)]:
            bad = self.frequency.copy()
            bad.index = index
            with self.assertRaises(ValueError):
                compute_rolling_frequency(bad)

    def test_loader_integration_and_reference_on_multiyear_data(self) -> None:
        rng = np.random.default_rng(42)
        frequency = rng.multinomial(27, np.full(100, 0.01), size=4000)
        raw = pd.DataFrame(frequency, columns=NUMBER_COLUMNS)
        raw.insert(0, "date", pd.date_range("2010-01-01", periods=len(raw)))
        freq, binary = XSMBDataLoader().preprocess_and_build_matrices(raw)
        actual = compute_rolling_frequency(freq)
        for window, frame in actual.items():
            reference = freq.rolling(f"{window}D", min_periods=1).sum().astype(np.int64)
            pd.testing.assert_frame_equal(frame, reference)
        dataset = build_feature_dataset(freq, binary)
        self.assertEqual(len(dataset), 3999 * 100)
        self.assertFalse(dataset.isna().any().any())


if __name__ == "__main__":
    unittest.main()
