import unittest

import pandas as pd

from scripts.backtests.compare_frequency_90d_poisson import summarize


class PoissonComparisonTests(unittest.TestCase):
    def test_summary_uses_paired_days_and_occurrence_payouts(self):
        daily = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=4),
            "frequency_prediction": ["01 02 03", "01 02 03", "01 02 03", "01 02 03"],
            "frequency_occurrences": [1, 1, 2, 0],
            "frequency_hit": [True, True, True, False],
            "poisson_prediction": ["04 05 06", "04 05 06", "04 05 06", "04 05 06"],
            "poisson_occurrences": [1, 0, 1, 0],
            "poisson_hit": [True, False, True, False],
        })

        summary, paired, eras, yearly, yearly_paired = summarize(
            daily, bootstrap_resamples=200
        )

        frequency = summary.loc[summary.model.eq("frequency_90d")].iloc[0]
        poisson = summary.loc[summary.model.eq("poisson")].iloc[0]
        difference = paired.iloc[0]
        self.assertEqual(frequency.occurrences, 4)
        self.assertEqual(poisson.occurrences, 2)
        self.assertEqual(frequency.profit_vnd, 4 * 99_000 - 4 * 3 * 27_000)
        self.assertEqual(poisson.profit_vnd, 2 * 99_000 - 4 * 3 * 27_000)
        self.assertEqual(difference.occurrence_delta_frequency_minus_poisson, 2)
        self.assertEqual(difference.frequency_only_hit_days, 1)
        self.assertEqual(difference.poisson_only_hit_days, 0)
        self.assertEqual(len(eras), 2)
        self.assertEqual(len(yearly), 2)
        self.assertEqual(len(yearly_paired), 1)
        self.assertEqual(yearly_paired.iloc[0].year, 2026)


if __name__ == "__main__":
    unittest.main()
