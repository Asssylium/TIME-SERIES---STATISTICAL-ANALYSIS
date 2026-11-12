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

    def test_live_raw_feed_still_produces_frequency_matrices(self):
        with patch.object(XSMBDataLoader, "load_data", autospec=True, side_effect=self.load_source):
            frequency, binary = api._load_matrices("live")
        pd.testing.assert_frame_equal(frequency, self.frequency)
        pd.testing.assert_frame_equal(binary, self.frequency.gt(0).astype("uint8"))
