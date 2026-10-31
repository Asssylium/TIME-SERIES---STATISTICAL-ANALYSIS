"""Calendar-window training, label mapping, and causal presence predictions."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.data_loader import NUMBER_COLUMNS
from src.regularized_models import RegularizedLGBMPredictor


class RegularizedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(23)
        self.frequency = pd.DataFrame(
            rng.multinomial(27, np.full(100, 0.01), size=45),
            index=pd.date_range("2026-01-01", periods=45), columns=NUMBER_COLUMNS,
        )
        self.data = build_count_feature_dataset(
            self.frequency, self.frequency.gt(0).astype(int), include_unlabeled=True,
        )

    def small_model(self, **params) -> RegularizedLGBMPredictor:
        return RegularizedLGBMPredictor(n_estimators=5, num_leaves=3, min_child_samples=10, **params)

    def test_calendar_cutoff_retains_boundary_and_excludes_older_labels(self) -> None:
        sparse = self.frequency.drop(self.frequency.index[[-4, -3]])
        data = build_count_feature_dataset(sparse, sparse.gt(0).astype(int), include_unlabeled=True)
        with patch("src.regularized_models.LGBMClassifier") as classifier:
            model = self.small_model(history_days=10).fit(data)
            x, _ = classifier.return_value.fit.call_args.args
        last_label = sparse.index[-1]
        target_dates = x.index.get_level_values(0) + pd.Timedelta(days=1)
        first_allowed = last_label + pd.Timedelta(days=1 - 10)
        self.assertEqual(target_dates.min(), first_allowed)
        self.assertEqual(target_dates.max(), last_label)
        self.assertTrue((target_dates >= first_allowed).all())
        self.assertLess(target_dates.nunique(), 10)
        self.assertEqual(model.training_label_start_, first_allowed)
        self.assertEqual(model.trained_through_, last_label)

    def test_default_window_uses_exactly_1095_calendar_days(self) -> None:
        origins = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2025-12-31"])
        # Four complete labeled date blocks suffice to test a multi-year cutoff.
        data = self.data.iloc[:400].copy()
        data.index = pd.MultiIndex.from_product([origins, NUMBER_COLUMNS], names=["date", "number"])
        with patch("src.regularized_models.LGBMClassifier") as classifier:
            model = self.small_model().fit(data)
            x, _ = classifier.return_value.fit.call_args.args
        cutoff = pd.Timestamp("2026-01-01") + pd.Timedelta(days=1 - 1095)
        self.assertEqual(cutoff, pd.Timestamp("2023-01-03"))
        self.assertEqual(x.index.get_level_values(0).unique().tolist(), origins[1:].tolist())
        self.assertEqual(model.training_label_start_, cutoff)

    def test_repeated_count_labels_map_to_presence_and_never_features(self) -> None:
        self.assertGreater(self.data[COUNT_TARGET].max(), 1)
        with patch("src.regularized_models.LGBMClassifier") as classifier:
            model = self.small_model().fit(self.data)
            x, y = classifier.return_value.fit.call_args.args
        np.testing.assert_array_equal(y.reshape(-1, 100), self.frequency.iloc[1:].gt(0).to_numpy())
        self.assertNotIn(COUNT_TARGET, x)
        self.assertNotIn("target_next_day", x)
        self.assertNotIn(COUNT_TARGET, model.feature_columns_)
        self.assertEqual(len(y), 4400)

    def test_real_predictions_schema_and_temporal_guard(self) -> None:
        before = self.data.copy(deep=True)
        model = self.small_model().fit(self.data.iloc[:3000])
        future = self.data.iloc[3000:]
        probabilities = model.predict_proba(future)
        self.assertTrue(probabilities.between(0, 1).all())
        self.assertEqual(probabilities.name, "probability")
        self.assertEqual(probabilities.shape, (1500,))
        self.assertTrue(probabilities.groupby(level=0).sum().gt(1).all())
        pd.testing.assert_series_equal(probabilities, model.predict_proba(future.drop(columns=COUNT_TARGET)))
        pd.testing.assert_series_equal(probabilities, model.predict_proba(future.iloc[::-1, ::-1]))
        top = model.predict(future, 6)
        self.assertEqual(len({n for n, _ in top}), 6)
        self.assertEqual([p for _, p in top], sorted([p for _, p in top], reverse=True))
        self.assertEqual(model.prediction_date_, self.frequency.index[-1] + pd.Timedelta(days=1))
        with self.assertRaisesRegex(ValueError, "predate"):
            model.predict_proba(self.data.iloc[:100])
        with self.assertRaisesRegex(ValueError, "match training"):
            model.predict_proba(future.drop(columns="rolling_rate_7d"))
        with self.assertRaisesRegex(ValueError, "Unexpected feature"):
            model.predict_proba(future.assign(target_next_day=0))
        pd.testing.assert_frame_equal(before, self.data)

    def test_count_validation_happens_before_binarization(self) -> None:
        invalid = self.data.astype({COUNT_TARGET: "Float64"})
        invalid.iloc[0, invalid.columns.get_loc(COUNT_TARGET)] = -1
        with self.assertRaisesRegex(ValueError, "integers in 0..27"):
            self.small_model().fit(invalid)
        invalid = self.data.copy()
        invalid.iloc[0, invalid.columns.get_loc(COUNT_TARGET)] = pd.NA
        with self.assertRaisesRegex(ValueError, "entire date"):
            self.small_model().fit(invalid)
        with self.assertRaisesRegex(ValueError, "fully labeled"):
            self.small_model().fit(self.data.iloc[-100:])

    def test_invalid_parameters_and_unfitted_prediction(self) -> None:
        for history in [0, -1, True, 1.5]:
            with self.subTest(history=history), self.assertRaisesRegex(ValueError, "history_days"):
                self.small_model(history_days=history)
        with self.assertRaisesRegex(ValueError, "binary"):
            RegularizedLGBMPredictor(objective="poisson")
        with self.assertRaises(NotFittedError):
            self.small_model().predict_proba(self.data)
        for k in [0, 101, True, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.small_model().predict(self.data, k)

    def test_array_origins_window_and_tied_ranking(self) -> None:
        data = build_count_feature_dataset(
            self.frequency.to_numpy(), self.frequency.gt(0).to_numpy(), include_unlabeled=True,
        )
        with patch("src.regularized_models.LGBMClassifier") as classifier:
            classifier.return_value.predict_proba.return_value = np.tile([0.75, 0.25], (100, 1))
            model = self.small_model(history_days=3).fit(data)
            x, _ = classifier.return_value.fit.call_args.args
            self.assertEqual(x.index.get_level_values(0).unique().tolist(), [41, 42, 43])
            self.assertEqual(model.predict(data, 2), [("00", 0.25), ("01", 0.25)])
        self.assertEqual(model.training_label_start_, 42)
        self.assertEqual(model.trained_through_, 44)
        self.assertEqual(model.prediction_date_, 45)


if __name__ == "__main__":
    unittest.main()
