import unittest

import pandas as pd

from scripts.reports.report_default_poisson_2026 import build_monthly, validate_markdown_tables


class DefaultPoisson2026ReportTests(unittest.TestCase):
    def test_monthly_cashflow_and_partial_final_month(self):
        daily = pd.DataFrame([
            {
                "date": "2026-11-03",
                "hit": True,
                "occurrences": 2,
            },
            {
                "date": "2026-11-04",
                "hit": False,
                "occurrences": 0,
            },
        ])
        month = build_monthly(daily).iloc[0]
        self.assertEqual(month.days_in_scope, 4)
        self.assertEqual(month.evaluated_days, 2)
        self.assertEqual(month.excluded_days, 2)
        self.assertEqual(month.stake_vnd, 162_000)
        self.assertEqual(month.receipts_vnd, 198_000)
        self.assertEqual(month.profit_vnd, 36_000)

    def test_markdown_table_width_validation(self):
        validate_markdown_tables("| A | B |\n|---|---:|\n| x | 1 |\n")
        with self.assertRaisesRegex(ValueError, "inconsistent columns"):
            validate_markdown_tables("| A | B |\n|---|\n| x | 1 |\n")


if __name__ == "__main__":
    unittest.main()
