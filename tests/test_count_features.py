"""Checks for causal occurrence targets and observed-history normalization."""

import unittest

import numpy as np
import pandas as pd

from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.data_loader import NUMBER_COLUMNS
from src.features import build_feature_dataset


class CountFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        counts = np.zeros((5, 100), dtype=np.uint8)
        counts[0, [0, 1]] = [2, 25]
        counts[1, [0, 2]] = [3, 24]
        counts[2, [2, 3]] = [20, 7]
        counts[3, [1, 2]] = [5, 22]
        counts[4, [0, 1]] = [1, 26]
        self.frequency = pd.DataFrame(
            counts,
            index=pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05", "2026-01-06"]
            ),
            columns=NUMBER_COLUMNS,
        )
        self.binary = self.frequency.gt(0).astype(np.uint8)

    def test_count_labels_preserve_repeats_and_exact_next_day(self) -> None:
        result = build_count_feature_dataset(self.frequency, self.binary)
        self.assertEqual(len(result), 300)
        self.assertEqual(str(result[COUNT_TARGET].dtype), "UInt8")
        self.assertEqual(result.loc[(pd.Timestamp("2026-01-01"), "00"), COUNT_TARGET], 3)
        self.assertEqual(result.loc[(pd.Timestamp("2026-01-04"), "01"), COUNT_TARGET], 5)
        for date in result.index.get_level_values(0).unique():
            target = result.xs(date)[COUNT_TARGET].to_numpy(dtype=np.uint8)
            np.testing.assert_array_equal(
                target, self.frequency.loc[date + pd.Timedelta(days=1)].to_numpy()
            )
            self.assertEqual(int(target.sum()), 27)

    def test_missing_dates_and_latest_date_are_wholly_unlabeled(self) -> None:
        result = build_count_feature_dataset(
            self.frequency, self.binary, include_unlabeled=True
        )
        self.assertEqual(len(result), 500)
        for day in ["2026-01-02", "2026-01-06"]:
            self.assertTrue(result.xs(pd.Timestamp(day))[COUNT_TARGET].isna().all())
        self.assertFalse(result.xs(pd.Timestamp("2026-01-04"))[COUNT_TARGET].isna().any())

    def test_normalization_uses_observations_and_has_no_binary_target(self) -> None:
        result = build_count_feature_dataset(
            self.frequency, self.binary, include_unlabeled=True
        )
        base = build_feature_dataset(self.frequency, self.binary, include_unlabeled=True)
        self.assertEqual([name for name in result if name.startswith("target_")], [COUNT_TARGET])
        for removed in ["target_next_day", "history_days", "co_occurrence_total"]:
            self.assertNotIn(removed, result)
        for window in [7, 14, 30, 60]:
            self.assertNotIn(f"rolling_frequency_{window}d", result)
            np.testing.assert_allclose(
                result[f"rolling_rate_{window}d"],
                base[f"rolling_frequency_{window}d"] / base[f"observed_days_{window}d"],
            )
        np.testing.assert_allclose(
            result.co_occurrence_per_day, base.co_occurrence_total / base.history_days
        )
        jan4 = result.xs(pd.Timestamp("2026-01-04"))
        self.assertEqual(jan4.loc["00", "observed_days_7d"], 3)
        self.assertAlmostEqual(jan4.loc["00", "rolling_rate_7d"], 5 / 3)
        self.assertTrue(np.isfinite(result.drop(columns=COUNT_TARGET).to_numpy()).all())

    def test_gap_completeness_resets_after_missing_calendar_draw(self) -> None:
        result = build_count_feature_dataset(
            self.frequency, self.binary, include_unlabeled=True
        )
        complete = result.gap_observation_complete.unstack("number")
        self.assertEqual(complete.loc["2026-01-02", "00"], 1)
        self.assertEqual(complete.loc["2026-01-04", "00"], 0)
        self.assertEqual(complete.loc["2026-01-04", "01"], 0)
        self.assertEqual(complete.loc["2026-01-04", "02"], 1)
        self.assertEqual(complete.loc["2026-01-05", "01"], 1)
        self.assertEqual(complete.loc["2026-01-06", "00"], 1)
        self.assertTrue((complete["99"] == 0).all())

    def test_causal_prefix_invariance_and_no_input_mutation(self) -> None:
        snapshot = self.frequency.copy(deep=True)
        full = build_count_feature_dataset(self.frequency, self.binary, include_unlabeled=True)
        prefix = build_count_feature_dataset(
            self.frequency.iloc[:3], self.binary.iloc[:3], include_unlabeled=True
        )
        pd.testing.assert_frame_equal(
            full.iloc[:300].drop(columns=COUNT_TARGET), prefix.drop(columns=COUNT_TARGET)
        )
        changed = self.frequency.copy()
        changed.iloc[3:] = 0
        changed.loc[changed.index[3:], "99"] = 27
        altered = build_count_feature_dataset(changed, changed.gt(0), include_unlabeled=True)
        pd.testing.assert_frame_equal(
            full.iloc[:300].drop(columns=COUNT_TARGET), altered.iloc[:300].drop(columns=COUNT_TARGET)
        )
        pd.testing.assert_frame_equal(self.frequency, snapshot)
        reordered = build_count_feature_dataset(
            self.frequency.iloc[:, ::-1], self.binary, include_unlabeled=True
        )
        pd.testing.assert_frame_equal(reordered, full)

    def test_arrays_assume_consecutive_days_and_single_day_is_unlabeled(self) -> None:
        result = build_count_feature_dataset(self.frequency.to_numpy(), self.binary.to_numpy())
        self.assertEqual(result.index.names, ["day_index", "number"])
        self.assertEqual(len(result), 400)
        np.testing.assert_array_equal(
            result.xs(1)[COUNT_TARGET].to_numpy(dtype=np.uint8), self.frequency.iloc[2]
        )
        self.assertTrue(build_count_feature_dataset(self.frequency.iloc[:1], self.binary.iloc[:1]).empty)
        one = build_count_feature_dataset(
            self.frequency.iloc[:1], self.binary.iloc[:1], include_unlabeled=True
        )
        self.assertEqual(len(one), 100)
        self.assertTrue(one[COUNT_TARGET].isna().all())


if __name__ == "__main__":
    unittest.main()
