"""Prior-cache prefix validation and preservation of completed protocols."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.backtests.backtest_periods import _verify_prior_history, calendar_periods, run
from src.data_loader import NUMBER_COLUMNS


class BacktestPeriodTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.zeros((5, 100), dtype=np.uint8)
        values[:, 0] = 27
        self.frequency = pd.DataFrame(
            values, index=pd.date_range('2026-01-01', periods=5, name='date'),
            columns=NUMBER_COLUMNS,
        )

    def test_requested_end_before_prior_snapshot_checks_only_relevant_prefix(self) -> None:
        current = self.frequency.iloc[:3].copy()
        # The old expression current.loc[prior.index] raised KeyError here.
        prior = self.frequency.copy()
        prior.iloc[3:, 0], prior.iloc[3:, 1] = 0, 27
        _verify_prior_history(current, prior)
        _verify_prior_history(self.frequency, self.frequency.iloc[:3])

    def test_mismatched_counts_or_missing_prior_dates_reject_cache(self) -> None:
        current = self.frequency.iloc[:3].copy()
        current.iloc[1, 0], current.iloc[1, 1] = 26, 1
        with self.assertRaises(AssertionError):
            _verify_prior_history(current, self.frequency)
        with self.assertRaisesRegex(ValueError, 'missing dates'):
            _verify_prior_history(self.frequency.drop(self.frequency.index[1]), self.frequency)

    def test_newly_filled_historical_gap_also_rejects_cache(self) -> None:
        prior = self.frequency.drop(self.frequency.index[1])
        with self.assertRaises(AssertionError):
            _verify_prior_history(self.frequency, prior)

    def test_changed_protocol_does_not_overwrite_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshot = output / 'source_snapshot.csv'
            self.frequency.to_csv(snapshot)
            protocol = output / 'protocol.json'
            protocol.write_text('{"version": "old-preserve-me"}', encoding='utf-8')
            cache = output / 'cache'
            cache.mkdir()
            sentinel = cache / 'existing.json'
            sentinel.write_text('{"saved": true}', encoding='utf-8')
            before = {path: path.read_bytes() for path in [snapshot, protocol, sentinel]}
            with self.assertRaisesRegex(ValueError, 'separate output directory'):
                run('2026-01-02', '2026-01-03', output, workers=1)
            for path, original in before.items():
                self.assertEqual(path.read_bytes(), original)
            self.assertFalse((output / 'daily.csv').exists())

    def test_periods_are_inclusive_nonoverlapping_with_partial_tail(self) -> None:
        periods = calendar_periods('2026-01-01', '2026-10-29')
        self.assertEqual([(str(a.date()), str(b.date())) for a, b in periods], [
            ('2026-01-01', '2026-03-31'), ('2026-04-01', '2026-06-29'),
            ('2026-06-30', '2026-09-27'), ('2026-09-28', '2026-10-29'),
        ])


if __name__ == '__main__':
    unittest.main()
