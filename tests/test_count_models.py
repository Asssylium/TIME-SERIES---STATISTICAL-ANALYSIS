"""Count-target integrity, causal inference guards, and constrained predictions."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.count_models import CountLGBMPredictor
from src.data_loader import NUMBER_COLUMNS


class CountModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.frequency = pd.DataFrame(
            rng.multinomial(27, np.full(100, 0.01), size=45),
            index=pd.date_range("2026-01-01", periods=45), columns=NUMBER_COLUMNS,
        )
        self.data = build_count_feature_dataset(
            self.frequency, self.frequency.gt(0).astype(int), include_unlabeled=True,
        )

    def small_model(self) -> CountLGBMPredictor:
        return CountLGBMPredictor(n_estimators=5, num_leaves=3, min_child_samples=10)

    def test_repeats_reach_regression_target_and_label_is_excluded(self) -> None:
        self.assertGreater(self.data[COUNT_TARGET].max(), 1)
        with patch("src.count_models.LGBMRegressor") as regressor:
            model = self.small_model().fit(self.data)
            x, y = regressor.return_value.fit.call_args.args
        self.assertNotIn(COUNT_TARGET, x)
        self.assertNotIn("target_next_day", x)
        np.testing.assert_array_equal(y.reshape(-1, 100), self.frequency.iloc[1:].to_numpy())
        self.assertNotIn(COUNT_TARGET, model.feature_columns_)
        self.assertEqual(model.trained_through_, self.frequency.index[-1])

    def test_expected_counts_sum_to_27_and_rank_without_mutating(self) -> None:
        before = self.data.copy(deep=True)
        model = self.small_model().fit(self.data.iloc[:3000])
        latest = self.data.iloc[3000:]
        scores = model.predict_expected_counts(latest)
        np.testing.assert_allclose(scores.groupby(level=0).sum(), 27, rtol=1e-12)
        self.assertTrue(scores.ge(0).all())
        self.assertTrue(scores.le(27).all())
        self.assertEqual(scores.name, "expected_count")
        top = model.predict(latest, top_k=6)
        self.assertEqual(len(top), 6)
        self.assertEqual(len({n for n, _ in top}), 6)
        self.assertEqual([s for _, s in top], sorted([s for _, s in top], reverse=True))
        self.assertEqual(model.prediction_date_, self.frequency.index[-1] + pd.Timedelta(days=1))
        pd.testing.assert_frame_equal(before, self.data)
        reordered = latest.iloc[::-1, ::-1]
        pd.testing.assert_series_equal(scores, model.predict_expected_counts(reordered))
        pd.testing.assert_series_equal(scores, model.predict_expected_counts(latest.drop(columns=COUNT_TARGET)))

    def test_raw_model_total_is_normalized_without_probability_clipping(self) -> None:
        with patch("src.count_models.LGBMRegressor") as regressor:
            model = self.small_model().fit(self.data)
            raw = np.arange(1, 101, dtype=float)
            regressor.return_value.predict.return_value = raw
            scores = model.predict_expected_counts(self.data.iloc[-100:])
            np.testing.assert_allclose(scores, 27 * raw / raw.sum())
            regressor.return_value.predict.return_value = np.full(100, 1e308)
            np.testing.assert_allclose(model.predict_expected_counts(self.data.iloc[-100:]), 0.27)
            regressor.return_value.predict.return_value = np.r_[27.0, np.zeros(99)]
            self.assertEqual(model.predict(self.data.iloc[-100:], 1), [("00", 27.0)])

    def test_whole_day_missing_labels_and_no_labeled_data(self) -> None:
        missing = self.data.copy()
        missing.iloc[100:200, missing.columns.get_loc(COUNT_TARGET)] = pd.NA
        model = self.small_model().fit(missing)
        self.assertEqual(model.trained_through_, self.frequency.index[-1])
        with self.assertRaisesRegex(ValueError, "fully labeled"):
            self.small_model().fit(self.data.iloc[-100:])
        partial = self.data.copy()
        partial.iloc[0, partial.columns.get_loc(COUNT_TARGET)] = pd.NA
        with self.assertRaisesRegex(ValueError, "entire date"):
            self.small_model().fit(partial)

    def test_count_target_values_and_daily_totals_are_strict(self) -> None:
        for value in [-1, 28, 1.5, np.inf]:
            bad = self.data.astype({COUNT_TARGET: "Float64"})
            bad.iloc[0, bad.columns.get_loc(COUNT_TARGET)] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integers in 0..27"):
                self.small_model().fit(bad)
        bad = self.data.copy()
        bad.iloc[:100, bad.columns.get_loc(COUNT_TARGET)] = 0
        with self.assertRaisesRegex(ValueError, "sum to exactly 27"):
            self.small_model().fit(bad)
        with self.assertRaisesRegex(ValueError, "numeric integers"):
            self.small_model().fit(self.data.astype({COUNT_TARGET: object}))

    def test_schema_candidate_alignment_and_future_guard(self) -> None:
        for bad in [self.data.iloc[:-1], self.data.reset_index(), self.data.assign(target_next_day=0)]:
            with self.assertRaises(ValueError):
                self.small_model().fit(bad)
        bad = self.data.copy()
        bad["number_id"] = 99
        with self.assertRaisesRegex(ValueError, "number_id"):
            self.small_model().fit(bad)
        model = self.small_model().fit(self.data)
        with self.assertRaisesRegex(ValueError, "predate"):
            model.predict_expected_counts(self.data.iloc[:100])
        with self.assertRaisesRegex(ValueError, "match training"):
            model.predict_expected_counts(self.data.iloc[-100:].drop(columns="rolling_rate_7d"))
        bad = self.data.iloc[-100:].copy()
        bad["frequency_today"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            model.predict_expected_counts(bad)

    def test_not_fitted_invalid_objective_and_invalid_k(self) -> None:
        with self.assertRaises(NotFittedError):
            self.small_model().predict_expected_counts(self.data)
        with self.assertRaisesRegex(ValueError, "poisson"):
            CountLGBMPredictor(objective="binary")
        for k in [0, 101, True, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.small_model().predict(self.data, k)

    def test_array_origins_and_deterministic_tie_break(self) -> None:
        data = build_count_feature_dataset(
            self.frequency.to_numpy(), self.frequency.gt(0).to_numpy(), include_unlabeled=True,
        )
        with patch("src.count_models.LGBMRegressor") as regressor:
            regressor.return_value.predict.return_value = np.ones(100)
            model = self.small_model().fit(data)
            self.assertEqual(model.predict(data, 2), [("00", 0.27), ("01", 0.27)])
        self.assertEqual(model.trained_through_, 44)
        self.assertEqual(model.prediction_date_, 45)

    def test_invalid_regression_outputs_are_not_silently_replaced(self) -> None:
        with patch("src.count_models.LGBMRegressor") as regressor:
            model = self.small_model().fit(self.data)
            for values in [np.zeros(100), np.full(100, np.nan), -np.ones(100), np.ones(99)]:
                regressor.return_value.predict.return_value = values
                with self.assertRaises(ValueError):
                    model.predict_expected_counts(self.data.iloc[-100:])


if __name__ == "__main__":
    unittest.main()
