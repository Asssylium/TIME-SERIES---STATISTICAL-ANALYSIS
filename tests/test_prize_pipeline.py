"""Ordered-prize semantics, prefix causality, serialization and model guards."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.count_features import COUNT_TARGET
from src.prize_features import PRIZE_COLUMNS, PRIZE_WIDTHS, build_prize_feature_dataset, normalize_prizes
from src.prize_models import PrizeLGBMPredictor
from scripts.operations.train_prize_model import run


class PrizePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(31)
        self.raw = pd.DataFrame({column: rng.integers(0, 10**width, 75).astype(str)
                                 for column, width in PRIZE_WIDTHS.items()})
        self.raw.insert(0, "date", pd.date_range("2026-01-01", periods=75))

    def test_order_zero_padding_and_digits(self) -> None:
        self.raw.loc[0, "special"] = "1"
        self.raw.loc[0, "prize1"] = "99999"
        normalized = normalize_prizes(self.raw)
        self.assertEqual(tuple(normalized.columns), PRIZE_COLUMNS)
        self.assertEqual(normalized.iloc[0]["special"], "00001")
        data = build_prize_feature_dataset(self.raw, include_unlabeled=True)
        day = data.xs(self.raw.date.iloc[0])
        self.assertEqual(day.loc["99", "slot_01_match"], 1)
        self.assertEqual(day.loc["01", "slot_27_match"], 1)
        self.assertEqual(day.loc["00", "slot_27_prefix_digit_hits"], 3)
        self.assertEqual(day.loc["11", "slot_27_prefix_digit_hits"], 0)
        self.assertTrue(day.slot_01_previous_day_match.eq(-1).all())
        self.assertTrue(data.iloc[-100:][COUNT_TARGET].isna().all())
        self.assertTrue(data.groupby(level=0)[COUNT_TARGET].sum(min_count=1).dropna().eq(27).all())

    def test_prefix_and_future_outcome_invariance(self) -> None:
        before = self.raw.copy(deep=True)
        full = build_prize_feature_dataset(self.raw, include_unlabeled=True)
        prefix = build_prize_feature_dataset(self.raw.iloc[:40], include_unlabeled=True)
        pd.testing.assert_frame_equal(full.iloc[:4000].drop(columns=COUNT_TARGET), prefix.drop(columns=COUNT_TARGET))
        altered = self.raw.copy()
        for column in PRIZE_COLUMNS:
            altered.loc[40:, column] = "0"
        changed = build_prize_feature_dataset(altered, include_unlabeled=True)
        pd.testing.assert_frame_equal(full.iloc[:4000].drop(columns=COUNT_TARGET), changed.iloc[:4000].drop(columns=COUNT_TARGET))
        tail = build_prize_feature_dataset(self.raw, start=self.raw.date.iloc[40], include_unlabeled=True)
        pd.testing.assert_frame_equal(tail, full.iloc[4000:])
        pd.testing.assert_frame_equal(self.raw, before)

    def test_missing_day_never_becomes_previous_calendar_day(self) -> None:
        raw = self.raw.drop(index=15)
        data = build_prize_feature_dataset(raw, include_unlabeled=True)
        self.assertTrue(data.xs(self.raw.date.iloc[14])[COUNT_TARGET].isna().all())
        self.assertTrue(data.xs(self.raw.date.iloc[16]).slot_01_previous_day_match.eq(-1).all())

    def test_prize_validation_rejects_wrong_schema_and_width(self) -> None:
        for raw in [self.raw.drop(columns="special"), self.raw.assign(special="123456"),
                    self.raw.assign(prize7_1="100"), self.raw.assign(prize1="1 2")]:
            with self.subTest(), self.assertRaises(ValueError):
                normalize_prizes(raw)

    def test_both_objectives_time_guards_and_saved_predictions(self) -> None:
        data = build_prize_feature_dataset(self.raw, include_unlabeled=True)
        for objective in ["binary", "poisson"]:
            with self.subTest(objective=objective):
                model = PrizeLGBMPredictor(objective, history_days=30, n_estimators=5, min_child_samples=10).fit(data.iloc[:5000])
                self.assertEqual(model.training_label_start_, self.raw.date.iloc[21])
                self.assertEqual(model.trained_through_, self.raw.date.iloc[50])
                self.assertNotIn(COUNT_TARGET, model.feature_columns_)
                future = data.iloc[5000:].drop(columns=COUNT_TARGET)
                scores = model.predict_scores(future)
                if objective == "binary":
                    self.assertTrue(scores.between(0, 1).all())
                else:
                    np.testing.assert_allclose(scores.groupby(level=0).sum(), 27)
                self.assertEqual(len(model.predict(future, 5)), 5)
                with self.assertRaisesRegex(ValueError, "predate"):
                    model.predict_scores(data.iloc[:100])
                with self.assertRaisesRegex(ValueError, "Unexpected feature"):
                    model.predict_scores(future.assign(target_next_day=0))
                with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
                    path = Path(directory)/"model.joblib"
                    joblib.dump(model, path)
                    pd.testing.assert_series_equal(joblib.load(path).predict_scores(future), scores)

    def test_unknown_and_nonfinite_features_are_rejected(self) -> None:
        data = build_prize_feature_dataset(self.raw, include_unlabeled=True)
        for invalid in [data.assign(slot_01_tail=np.nan), data.assign(future_outcome=1)]:
            with self.subTest(), self.assertRaises(ValueError):
                PrizeLGBMPredictor().fit(invalid)

    def test_training_runner_cuts_future_and_saves_reloadable_models(self) -> None:
        payload = self.raw.to_csv(index=False).encode()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output = Path(directory)/"run"
            with patch("src.data_loader.urlopen", return_value=io.BytesIO(payload)), redirect_stdout(io.StringIO()):
                result = run(forecast_date="2026-02-05", test_start="2026-02-01", top_k=3, output=output)
            self.assertEqual(result["forecast_date"], "2026-02-05")
            self.assertEqual(result["training_label_end"], "2026-02-04")
            self.assertTrue(result["reload_prediction_equality"])
            snapshot = pd.read_csv(output/"source_snapshot.csv", dtype=str)
            self.assertEqual(snapshot.date.max(), "2026-02-04")
            cutoffs = pd.read_csv(output/"training_cutoffs.csv")
            self.assertTrue((cutoffs.training_label_end < cutoffs.first_test_target).all())
            ranking = pd.read_csv(output/"ranking.csv")
            self.assertEqual(len(ranking), 9)
            self.assertTrue(ranking.days.eq(4).all())
            self.assertTrue((ranking.cost == ranking.top_k*4*27_000).all())
            forecasts = pd.read_csv(output/"forecast_top_k.csv", dtype={"number":str})
            self.assertEqual(len(forecasts), 6)
            self.assertEqual(set(forecasts.model), {"prize_binary", "prize_poisson"})
            self.assertTrue((output/"models/prize_binary.joblib").exists())
            # Identical source and protocol must reuse the chronological scores.
            with patch("src.data_loader.urlopen", return_value=io.BytesIO(payload)), redirect_stdout(io.StringIO()) as log:
                run(forecast_date="2026-02-05", test_start="2026-02-01", top_k=3, output=output)
            self.assertNotIn("FIT 2026Q1", log.getvalue())
            with patch("src.data_loader.urlopen", return_value=io.BytesIO(payload)), self.assertRaisesRegex(ValueError, "Protocol/source changed"):
                run(forecast_date="2026-02-05", test_start="2026-02-01", top_k=5, output=output)


if __name__ == "__main__":
    unittest.main()
