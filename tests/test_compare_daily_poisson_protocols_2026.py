import unittest

import pandas as pd

from scripts.backtests.compare_daily_poisson_protocols_2026 import aggregate_monthly


class CompareDailyPoissonProtocolsTests(unittest.TestCase):
    def test_monthly_pairing_and_partial_month(self):
        daily = pd.DataFrame([
            {
                "date": "2026-10-28",
                "model": "poisson_90d_daily",
                "hit": True,
                "occurrences": 2,
            },
            {
                "date": "2026-10-28",
                "model": "poisson_all_history_daily",
                "hit": False,
                "occurrences": 0,
            },
            {
                "date": "2026-10-29",
                "model": "poisson_90d_daily",
                "hit": False,
                "occurrences": 0,
            },
            {
                "date": "2026-10-29",
                "model": "poisson_all_history_daily",
                "hit": True,
                "occurrences": 1,
            },
        ])
        month = aggregate_monthly(daily).iloc[0]
        self.assertEqual(month.days_in_scope, 29)
        self.assertEqual(month.evaluated_days, 2)
        self.assertEqual(month.excluded_days, 27)
        self.assertEqual(month.poisson_90d_daily_profit_vnd, 36_000)
        self.assertEqual(month.poisson_all_history_daily_profit_vnd, -63_000)

    def test_rejects_unpaired_model_dates(self):
        daily = pd.DataFrame([
            {
                "date": "2026-10-28",
                "model": "poisson_90d_daily",
                "hit": True,
                "occurrences": 1,
            },
            {
                "date": "2026-10-29",
                "model": "poisson_all_history_daily",
                "hit": True,
                "occurrences": 1,
            },
        ])
        with self.assertRaisesRegex(ValueError, "identical target dates"):
            aggregate_monthly(daily)


if __name__ == "__main__":
    unittest.main()
