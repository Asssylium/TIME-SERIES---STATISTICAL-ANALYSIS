"""Test probability forecasts, baseline counts, and chronological validation."""

import unittest

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.data_loader import NUMBER_COLUMNS
from src.features import build_feature_dataset, compute_rolling_frequency
from src.models import BaselinePredictor, LGBMPredictor, cross_validate


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.frequency = pd.DataFrame(
            rng.multinomial(27, np.full(100, 0.01), size=60),
            index=pd.date_range("2026-01-01", periods=60),
            columns=NUMBER_COLUMNS, dtype=np.uint8,
        )
        self.binary = self.frequency.gt(0).astype(np.uint8)
        self.dataset = build_feature_dataset(
            self.frequency, self.binary, include_unlabeled=True
        )

    def small_model(self) -> LGBMPredictor:
        return LGBMPredictor(n_estimators=8, num_leaves=7, min_child_samples=10)

    def test_baseline_calendar_window_and_tie_breaking(self) -> None:
        frequency = self.frequency.copy()
        frequency.iloc[:] = 0
        frequency.iloc[:30, 99] = 27
        frequency.iloc[30:, 0] = 27
        model = BaselinePredictor()
        top = model.train_predict(frequency, top_k=3)
        self.assertEqual(top, [("00", 810.0), ("01", 0.0), ("02", 0.0)])
        self.assertEqual(model.prediction_date_, pd.Timestamp("2026-03-02"))
        scores = model.predict_scores()
        scores.iloc[0] = -100
        self.assertEqual(model.predict(1), [("00", 810.0)])
        sparse = frequency.iloc[[0, 58, 59]]
        sparse_scores = BaselinePredictor().fit(sparse).predict_scores()
        self.assertEqual(sparse_scores["99"], 0)
        self.assertEqual(sparse_scores["00"], 54)

    def test_lightgbm_probability_and_next_day_prediction(self) -> None:
        model = self.small_model()
        before = self.dataset.copy(deep=True)
        result = model.train_predict(self.dataset, top_k=15)
        self.assertEqual(len(result), 15)
        self.assertEqual(len({number for number, _ in result}), 15)
        self.assertTrue(all(number in NUMBER_COLUMNS and 0 <= p <= 1 for number, p in result))
        self.assertEqual([p for _, p in result], sorted([p for _, p in result], reverse=True))
        self.assertEqual(model.prediction_date_, self.frequency.index[-1] + pd.Timedelta(days=1))
        self.assertNotIn("target_next_day", model.feature_columns_)
        latest = self.dataset.iloc[-100:]
        scores = model.predict_proba(latest)
        self.assertEqual(scores.shape, (100,))
        self.assertTrue(scores.index.equals(latest.index))
        changed_labels = latest.copy()
        changed_labels["target_next_day"] = 1
        pd.testing.assert_series_equal(scores, model.predict_proba(changed_labels))
        reordered = latest.iloc[:, ::-1]
        pd.testing.assert_series_equal(scores, model.predict_proba(reordered))
        pd.testing.assert_frame_equal(before, self.dataset)

    def test_historical_predictions_are_rejected(self) -> None:
        model = self.small_model().fit(self.dataset)
        with self.assertRaisesRegex(ValueError, "predate"):
            model.predict_proba(self.dataset.iloc[:100])
        # Training-only data has no unseen origin for a future forecast.
        with self.assertRaisesRegex(ValueError, "predate"):
            self.small_model().train_predict(self.dataset.iloc[:-100])

    def test_not_fitted_and_invalid_top_k(self) -> None:
        with self.assertRaises(NotFittedError):
            BaselinePredictor().predict()
        with self.assertRaises(NotFittedError):
            self.small_model().predict(self.dataset)
        model = BaselinePredictor().fit(self.frequency)
        for top_k in [0, -1, 101, True, 1.5]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                model.predict(top_k)

    def test_dataset_contract_and_target_validation(self) -> None:
        for bad in [self.dataset.iloc[:-1], self.dataset.reset_index(),
                    self.dataset.assign(future_target=1)]:
            with self.assertRaises(ValueError):
                self.small_model().fit(bad)
        for bad_value in [2, "wrong"]:
            bad = self.dataset.astype({"target_next_day": object})
            bad.iloc[0, bad.columns.get_loc("target_next_day")] = bad_value
            with self.assertRaises(ValueError):
                self.small_model().fit(bad)
        bad = self.dataset.copy()
        bad.iloc[0, bad.columns.get_loc("target_next_day")] = pd.NA
        with self.assertRaisesRegex(ValueError, "entire date"):
            self.small_model().fit(bad)
        bad = self.dataset.copy()
        bad["target_next_day"] = 0
        with self.assertRaisesRegex(ValueError, "both absence"):
            self.small_model().fit(bad)
        bad = self.dataset.copy()
        bad["number_id"] = 99
        with self.assertRaisesRegex(ValueError, "number_id"):
            self.small_model().fit(bad)

    def test_prediction_schema_must_match_training(self) -> None:
        model = self.small_model().fit(self.dataset)
        latest = self.dataset.iloc[-100:].drop(columns="rolling_frequency_7d")
        with self.assertRaisesRegex(ValueError, "match training"):
            model.predict(latest)

    def test_cv_date_groups_gap_and_metrics(self) -> None:
        for model in [BaselinePredictor(), self.small_model()]:
            with self.subTest(model=type(model).__name__):
                result = cross_validate(
                    model, self.frequency, self.binary, n_splits=3, top_k=10,
                    gap=1, test_size=5, max_train_size=20,
                )
                self.assertEqual(len(result.predictions), 3 * 5 * 100)
                self.assertEqual(len(result.fold_metrics), 3)
                self.assertTrue(result.predictions.index.is_unique)
                self.assertTrue(result.predictions.groupby(level=0).size().eq(100).all())
                self.assertTrue(result.predictions.groupby(level=0).selected_top_k.sum().eq(10).all())
                self.assertTrue((result.fold_metrics.train_label_end < result.fold_metrics.validation_start).all())
                self.assertTrue(result.fold_metrics.n_train_days.eq(20).all())
                for name in ["precision_at_k", "recall_at_k", "hit_rate_at_k"]:
                    self.assertTrue(result.fold_metrics[name].between(0, 1).all())
                first_fold = result.predictions.query("fold == 1")
                hits = (first_fold.target_next_day * first_fold.selected_top_k).groupby(level=0).sum()
                actual = first_fold.target_next_day.groupby(level=0).sum()
                metric = result.fold_metrics.loc[1]
                self.assertAlmostEqual(metric.precision_at_k, (hits / 10).mean())
                self.assertAlmostEqual(metric.recall_at_k, (hits / actual).mean())
                self.assertAlmostEqual(metric.hit_rate_at_k, (hits > 0).mean())
                if isinstance(model, BaselinePredictor):
                    self.assertIsNone(model.scores_)
                else:
                    self.assertIsNone(model.model_)

    def test_baseline_cv_matches_causal_rolling(self) -> None:
        result = cross_validate(BaselinePredictor(), self.frequency, self.binary, n_splits=2)
        expected = compute_rolling_frequency(self.frequency, windows=[30])[30]
        for date in result.predictions.index.get_level_values(0).unique():
            np.testing.assert_array_equal(result.predictions.xs(date).score, expected.loc[date])

    def test_future_changes_cannot_affect_earlier_cv_predictions(self) -> None:
        changed = self.frequency.copy()
        changed.iloc[-5:] = 0
        changed.iloc[-5:, 99] = 27
        for model in [BaselinePredictor(), self.small_model()]:
            before = cross_validate(model, self.frequency, self.binary, n_splits=3, test_size=5)
            after = cross_validate(model, changed, changed.gt(0), n_splits=3, test_size=5)
            pd.testing.assert_frame_equal(
                before.predictions.query("fold == 1"), after.predictions.query("fold == 1")
            )

    def test_arrays_and_calendar_gaps(self) -> None:
        frequency, binary = self.frequency.to_numpy(), self.binary.to_numpy()
        dataset = build_feature_dataset(frequency, binary, include_unlabeled=True)
        self.assertEqual(len(self.small_model().train_predict(dataset, top_k=20)), 20)
        result = cross_validate(self.small_model(), frequency, binary, n_splits=2)
        self.assertEqual(result.predictions.index.names, ["day_index", "number"])
        sparse_f = self.frequency.drop(self.frequency.index[[10, 25, 40]])
        sparse_b = sparse_f.gt(0)
        result = cross_validate(BaselinePredictor(), sparse_f, sparse_b, n_splits=2)
        for date in result.predictions.index.get_level_values(0).unique():
            self.assertIn(date + pd.Timedelta(days=1), sparse_f.index)

    def test_invalid_cv_configuration(self) -> None:
        for kwargs in [{"gap": 0}, {"n_splits": 1}, {"test_size": 0},
                       {"max_train_size": -1}, {"n_splits": 100}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cross_validate(BaselinePredictor(), self.frequency, self.binary, **kwargs)


if __name__ == "__main__":
    unittest.main()
