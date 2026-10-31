"""Positive controls and missing-calendar guards for signal screening."""

import unittest

import numpy as np
import pandas as pd

from diagnose_signal import screen
from src.data_loader import NUMBER_COLUMNS


class SignalDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2020-01-01", periods=400)

    def test_detects_planted_weekday_effect(self) -> None:
        counts = np.zeros((400, 100), dtype=int)
        monday = self.dates.dayofweek == 0
        counts[monday, 0] = 20
        counts[~monday, 1] = 20
        counts[:, 99] = 7
        result = screen(pd.DataFrame(counts, index=self.dates, columns=NUMBER_COLUMNS), permutations=99)
        weekday = result.loc[result.family.eq("weekday")].set_index("number")
        self.assertLessEqual(weekday.loc["00", "adjusted_p"], .05)
        self.assertTrue(np.isfinite(result[["effect", "statistic", "adjusted_p"]].to_numpy()).all())

    def test_detects_planted_serial_dependence(self) -> None:
        counts = np.zeros((400, 100), dtype=int)
        state = (np.arange(400)//20) % 2
        counts[np.arange(400), state] = 27
        result = screen(pd.DataFrame(counts, index=self.dates, columns=NUMBER_COLUMNS), permutations=99)
        effect = result.loc[result.family.eq("self_lag") & result.number.eq("00") & result.lag.eq(1)].iloc[0]
        self.assertGreater(effect.effect, .8)
        self.assertLessEqual(effect.adjusted_p, .05)

    def test_absent_weekday_is_rejected_and_number_bias_is_detected(self) -> None:
        counts = np.zeros((400, 100), dtype=int)
        counts[:, 0] = 27
        frequency = pd.DataFrame(counts, index=self.dates, columns=NUMBER_COLUMNS)
        with self.assertRaisesRegex(ValueError, "all seven weekdays"):
            screen(frequency.loc[frequency.index.dayofweek != 0], permutations=99)
        result = screen(frequency.drop(self.dates[[1, 4, 9]]), permutations=99)
        identity = result.loc[result.family.eq("number_identity") & result.number.eq("00")].iloc[0]
        self.assertLessEqual(identity.adjusted_p, .05)
        self.assertTrue(np.isfinite(result[["effect", "statistic", "adjusted_p"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
