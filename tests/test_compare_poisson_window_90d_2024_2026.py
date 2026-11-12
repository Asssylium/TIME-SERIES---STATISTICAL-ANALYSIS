import unittest

import pandas as pd

from scripts.backtests.compare_poisson_window_90d_2024_2026 import build_monthly


class ComparePoissonWindowMonthlyTests(unittest.TestCase):
    def test_builds_paired_monthly_profit_and_partial_month_scope(self):
        daily = pd.DataFrame([
            {
                "date": "2026-11-03",
                "model": "poisson_90d",
                "hit_days": True,
                "occurrences": 2,
            },
            {
                "date": "2026-11-03",
                "model": "poisson_all_history",
                "hit_days": False,
                "occurrences": 0,
            },
            {
                "date": "2026-11-04",
                "model": "poisson_90d",
                "hit_days": False,
                "occurrences": 0,
            },
            {
                "date": "2026-11-04",
                "model": "poisson_all_history",
                "hit_days": True,
                "occurrences": 1,
            },
        ])

        monthly = build_monthly(daily).iloc[0]

        self.assertEqual(monthly.days_in_scope, 4)
        self.assertEqual(monthly.evaluated_days, 2)
        self.assertEqual(monthly.excluded_days, 2)
        self.assertEqual(monthly.poisson_90d_profit_vnd, 36_000)
        self.assertEqual(monthly.poisson_all_history_profit_vnd, -63_000)
        self.assertEqual(monthly.profit_delta_90d_minus_all_history_vnd, 99_000)

    def test_rejects_unpaired_model_dates(self):
        daily = pd.DataFrame([
            {
                "date": "2026-11-03",
                "model": "poisson_90d",
                "hit_days": True,
                "occurrences": 1,
            },
            {
                "date": "2026-11-04",
                "model": "poisson_all_history",
                "hit_days": True,
                "occurrences": 1,
            },
        ])
        with self.assertRaisesRegex(ValueError, "identical target dates"):
            build_monthly(daily)


if __name__ == "__main__":
    unittest.main()
