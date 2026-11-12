"""The live API source must retain ordered prizes for Prize Poisson."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from api import index as api
from src.data_loader import XSMBDataLoader
from src.prize_features import PRIZE_WIDTHS, RAW_PRIZE_URL


class ApiPrizeSourceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.raw = pd.DataFrame({
            name: rng.integers(0, 10**width, 105).astype(str)
            for name, width in PRIZE_WIDTHS.items()
        })
        self.raw.insert(0, "date", pd.date_range("2026-01-01", periods=105).strftime("%Y-%m-%d"))
        self.frequency, _ = XSMBDataLoader().preprocess_and_build_matrices(self.raw)
        api._load_raw.cache_clear()
        api._load_matrices.cache_clear()
        self.addCleanup(api._load_raw.cache_clear)
        self.addCleanup(api._load_matrices.cache_clear)

    def load_source(self, loader):
        # Reproduce the two real feed formats without network access.
        if loader.url == RAW_PRIZE_URL:
            return self.raw.copy()
        return self.frequency.reset_index()

    def test_live_prize_forecast_uses_ordered_prizes(self):
        with patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source):
            response = api.app.test_client().post("/api/forecast", json={
                "source": "live", "model": "poisson_prize_90d", "top_k": 3,
            })
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["target_date"], "2026-04-16")
        self.assertEqual(len(result["forecasts"]["poisson_prize_90d"]), 3)
        self.assertEqual(result["analytics"]["schema_version"], "1.0")
        self.assertEqual(result["analytics"]["models"], [])

    def test_sources_expose_only_the_live_github_feed(self):
        with patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source):
            response = api.app.test_client().get("/api/sources")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["sources"], [{
            "id": "live",
            "label": "Live GitHub data feed",
            "kind": "remote",
            "first_date": "2026-01-01",
            "last_date": "2026-04-15",
            "next_date": "2026-04-16",
            "days": 105,
        }])

    def test_source_refresh_pulls_and_validates_a_new_revision(self):
        with patch.object(
            XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source,
        ) as load:
            response = api.app.test_client().post("/api/sources/refresh")
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(load.call_count, 2)
        self.assertEqual(payload["source"]["last_date"], "2026-04-15")
        self.assertTrue(payload["source_revision"])
        self.assertEqual(payload["refresh"], {
            "previous_last_date": "2026-04-15",
            "current_last_date": "2026-04-15",
            "added_days": 0,
            "changed": False,
        })

    def test_prize_poisson_range_refits_each_historical_date_without_future_labels(self):
        calls = []
        constructor_calls = []

        class RecordingPredictor:
            def __init__(self, *args, **kwargs):
                constructor_calls.append((args, kwargs))

            def fit(self, train):
                self.train = train
                return self

            def predict_scores(self, inference):
                calls.append((self.train.copy(), inference.copy()))
                return pd.Series(np.arange(100, dtype=float), index=inference.index)

        with (
            patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source),
            patch.object(api, "PrizeLGBMPredictor", RecordingPredictor),
        ):
            response = api.app.test_client().post("/api/forecast", json={
                "source": "live",
                "model": "poisson_prize_90d",
                "top_k": 3,
                "target_date": "2026-04-10",
                "target_end_date": "2026-04-12",
            })

        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["target_start_date"], "2026-04-10")
        self.assertEqual(result["target_end_date"], "2026-04-12")
        self.assertEqual(result["result_count"], 3)
        self.assertEqual([item["target_date"] for item in result["results"]], [
            "2026-04-10", "2026-04-11", "2026-04-12",
        ])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(constructor_calls), 3)
        for args, kwargs in constructor_calls:
            self.assertEqual(args, ("poisson",))
            self.assertEqual(kwargs, {
                "history_days": 90,
                "n_jobs": 1,
                "n_estimators": 100,
            })
        self.assertEqual(result["model_parameters"], {
            "history_days": 90,
            "n_estimators": 100,
        })
        analytics = result["analytics"]
        self.assertEqual(analytics["schema_version"], "1.0")
        self.assertEqual(analytics["period"], {
            "start_date": "2026-04-10", "end_date": "2026-04-12",
        })
        self.assertEqual(analytics["models"][0]["model_id"], "poisson_prize_90d")
        self.assertEqual(len(analytics["models"][0]["series"]), 3)
        for target_text, (train, inference) in zip(
            ["2026-04-10", "2026-04-11", "2026-04-12"], calls, strict=True,
        ):
            target = pd.Timestamp(target_text)
            training_labels = train.index.get_level_values(0) + pd.Timedelta(days=1)
            self.assertTrue((training_labels < target).all())
            self.assertTrue((training_labels >= target - pd.Timedelta(days=90)).all())
            self.assertEqual(inference.index.get_level_values(0).unique().tolist(), [
                target - pd.Timedelta(days=1),
            ])

    def test_prize_poisson_accepts_one_historical_target(self):
        with (
            patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source),
            patch.object(api, "PrizeLGBMPredictor") as predictor,
        ):
            fitted = predictor.return_value.fit.return_value
            fitted.predict_scores.return_value = pd.Series(
                np.arange(100, dtype=float),
                index=pd.MultiIndex.from_product([
                    [pd.Timestamp("2026-04-09")], [f"{number:02d}" for number in range(100)]
                ]),
            )
            response = api.app.test_client().post("/api/forecast", json={
                "source": "live",
                "model": "poisson_prize_90d",
                "target_date": "2026-04-10",
            })

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["target_date"], "2026-04-10")

    def test_prize_poisson_rejects_range_beyond_next_unseen_date(self):
        with patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source):
            response = api.app.test_client().post("/api/forecast", json={
                "source": "live",
                "model": "poisson_prize_90d",
                "target_date": "2026-04-16",
                "target_end_date": "2026-04-17",
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn("latest forecastable date", response.get_json()["error"])

    def test_live_raw_feed_still_produces_frequency_matrices(self):
        with patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source):
            frequency, binary = api._load_matrices("live")
        pd.testing.assert_frame_equal(frequency, self.frequency)
        pd.testing.assert_frame_equal(binary, self.frequency.gt(0).astype("uint8"))
