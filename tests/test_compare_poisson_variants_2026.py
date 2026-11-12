import unittest

import numpy as np
import pandas as pd

from scripts.backtests.compare_poisson_variants_2026 import (
    COUNT_TARGET,
    _record,
    _training_mask,
)
from src.data_loader import NUMBER_COLUMNS


class ComparePoissonVariants2026Tests(unittest.TestCase):
    def test_training_window_uses_only_labels_before_target(self):
        origins = pd.to_datetime(["2026-02-28", "2026-03-01", "2026-03-02", "2026-03-03"])
        index = pd.MultiIndex.from_product(
            [origins, NUMBER_COLUMNS], names=["date", "number"]
        )
        dataset = pd.DataFrame(
            {COUNT_TARGET: pd.array(np.zeros(len(index)), dtype="UInt8")},
            index=index,
        )

        mask = _training_mask(dataset, pd.Timestamp("2026-03-04"), 2)
        selected_dates = set(dataset.index.get_level_values(0)[mask])

        self.assertEqual(selected_dates, {pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")})

    def test_record_counts_repeats_and_reconciles_cash_flow(self):
        actual = np.zeros(100, dtype=np.int64)
        actual[1] = 2
        actual[3] = 1

        result = _record(
            pd.Timestamp("2026-03-04"),
            "count_90d",
            np.asarray([1, 2, 3]),
            actual,
            training_labels=90,
            training_label_start="2025-12-03",
            training_label_end="2026-03-03",
        )

        self.assertEqual(result["distinct_hits"], 2)
        self.assertEqual(result["occurrences"], 3)
        self.assertEqual(result["cost_vnd"], 81_000)
        self.assertEqual(result["receipts_vnd"], 297_000)
        self.assertEqual(result["profit_vnd"], 216_000)


if __name__ == "__main__":
    unittest.main()
