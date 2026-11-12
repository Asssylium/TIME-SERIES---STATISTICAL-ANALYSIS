import unittest

import pandas as pd

from scripts.reports.report_monthly_comparison_2026 import (
    build_monthly_comparison,
    validate_markdown_tables,
)


class MonthlyComparison2026Tests(unittest.TestCase):
    def test_markdown_table_validation_accepts_equal_width_rows(self):
        validate_markdown_tables(
            "| A | B |\n|---|---:|\n| x | 1 |\n"
        )

    def test_markdown_table_validation_rejects_mismatched_separator(self):
        with self.assertRaisesRegex(ValueError, "inconsistent column counts"):
            validate_markdown_tables(
                "| A | B | C |\n|---|---:|\n| x | 1 | 2 |\n"
            )

    def test_aggregates_paired_monthly_cashflows_and_partial_days(self):
        daily = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-02-01"],
            "frequency_prediction": ["01 02 03"] * 3,
            "frequency_occurrences": [1, 0, 2],
            "frequency_hit": [True, False, True],
            "poisson_prediction": ["04 05 06"] * 3,
            "poisson_occurrences": [0, 1, 1],
            "poisson_hit": [False, True, True],
        })

        monthly = build_monthly_comparison(daily).set_index("month")

        january = monthly.loc["2026-01"]
        self.assertEqual(january.evaluated_days, 2)
        self.assertEqual(january.days_in_scope, 31)
        self.assertEqual(january.excluded_days, 29)
        self.assertEqual(january.stake_vnd, 162_000)
        self.assertEqual(january.frequency_profit_vnd, -63_000)
        self.assertEqual(january.poisson_profit_vnd, -63_000)
        february = monthly.loc["2026-02"]
        self.assertEqual(february.frequency_profit_vnd, 117_000)
        self.assertEqual(february.poisson_profit_vnd, 18_000)
        self.assertEqual(
            february.profit_delta_frequency_minus_poisson_vnd, 99_000
        )

    def test_partial_final_month_does_not_count_future_days_as_excluded(self):
        daily = pd.DataFrame({
            "date": ["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04"],
            "frequency_prediction": ["01 02 03"] * 4,
            "frequency_occurrences": [0] * 4,
            "frequency_hit": [False] * 4,
            "poisson_prediction": ["04 05 06"] * 4,
            "poisson_occurrences": [0] * 4,
            "poisson_hit": [False] * 4,
        })

        november = build_monthly_comparison(daily).iloc[0]

        self.assertEqual(november.calendar_days, 30)
        self.assertEqual(november.days_in_scope, 4)
        self.assertEqual(november.evaluated_days, 4)
        self.assertEqual(november.excluded_days, 0)


if __name__ == "__main__":
    unittest.main()
