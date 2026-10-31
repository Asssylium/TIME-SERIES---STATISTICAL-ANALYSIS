"""Prospective timing, immutable records, real model round-trip and payout tests."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.paper_trading import (Ledger, export_report, initialize, make_models,
                               record_forecast, settle)
from src.prize_features import PRIZE_WIDTHS, normalize_prizes


def moment(hour: int, day: int = 30) -> datetime:
    """UTC instant for a local UTC+7 hour on October 2026."""
    return datetime(2026, 10, day, hour-7, tzinfo=timezone.utc)


class PaperJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(71)
        dates = pd.date_range('2026-08-01', '2026-10-29')
        cls.raw = pd.DataFrame({name: rng.integers(0, 10**width, len(dates)).astype(str)
                               for name, width in PRIZE_WIDTHS.items()})
        cls.raw.insert(0, 'date', dates.strftime('%Y-%m-%d'))
        # Actual LightGBM fitting/serialization exercised once; all other tests
        # reuse this fixed bundle, never replacing inference with fake scores.
        cls.bundle = make_models(normalize_prizes(cls.raw), pd.Timestamp('2026-10-29'), {'history_days':1095})

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.ledger = Ledger(self.path/'journal.sqlite3')
        self.clock = patch('src.paper_trading.utc_now', return_value=moment(16))
        self.now = self.clock.start()
        self.fit = patch('src.paper_trading.make_models', return_value=self.bundle)
        self.fit.start()
        initialize(self.ledger)

    def tearDown(self) -> None:
        self.fit.stop()
        self.clock.stop()
        self.ledger.close()
        self.temp.cleanup()

    def record(self) -> dict:
        record_forecast(self.ledger, self.raw)
        return self.ledger.get('forecast','2026-10-30')

    def result_raw(self, number: str) -> pd.DataFrame:
        return pd.concat([self.raw, pd.DataFrame([{'date':'2026-10-30', **{c:number for c in PRIZE_WIDTHS}}])],ignore_index=True)

    def test_real_inference_repeat_is_unchanged_and_protocol_frozen(self) -> None:
        original = self.record()
        count = self.ledger.verify()
        self.now.return_value = moment(20)
        record_forecast(self.ledger, self.raw)
        self.assertEqual(count, self.ledger.verify())
        self.assertEqual(original, self.ledger.get('forecast','2026-10-30'))
        self.assertEqual(initialize(self.ledger)['start'],'2026-10-30')
        for name, f in original['forecasts'].items():
            self.assertEqual(len(set(f['numbers'])),5)
            self.assertEqual(len(f['scores_00_to_99']),100)
            if name=='prize_poisson':
                self.assertAlmostEqual(sum(f['scores_00_to_99']),27)

    def test_duplicate_payouts_pending_days_and_idempotent_scoring(self) -> None:
        forecast = self.record()
        chosen = forecast['forecasts']['prize_poisson']['numbers'][0]
        actual = self.result_raw(chosen)
        # Complete source alone is insufficient before the 19:00 guard.
        settle(self.ledger,actual)
        report=export_report(self.ledger,self.path)
        self.assertEqual(report['awaiting_results'],1)
        self.assertEqual(report['models']['prize_poisson']['days_settled'],0)
        self.now.return_value=moment(20)
        settle(self.ledger,actual)
        payout=self.ledger.get('result','2026-10-30')['models']['prize_poisson']
        self.assertEqual(payout['paid_occurrences'],27)
        self.assertEqual(payout['distinct_hits'],1)
        self.assertEqual(payout['profit'],27*80000-5*23000)
        before=self.ledger.verify()
        settle(self.ledger,actual)
        self.assertEqual(before,self.ledger.verify())
        report=export_report(self.ledger,self.path)
        self.assertEqual(report['models']['prize_poisson']['profit'],payout['profit'])

    def test_future_results_and_stale_previous_day_rejected(self) -> None:
        for raw in [self.result_raw('00'), self.raw.iloc[:-1]]:
            with self.assertRaises(ValueError):
                record_forecast(self.ledger,raw)
        self.assertIsNone(self.ledger.get('forecast','2026-10-30'))

    def test_post_deadline_no_backdating_and_missing_coverage(self) -> None:
        self.now.return_value=moment(18)
        self.assertIn('deadline',record_forecast(self.ledger,self.raw))
        report=export_report(self.ledger,self.path)
        self.assertEqual(report['missing_forecasts'],1)
        self.assertEqual(report['models']['prize_poisson']['days_settled'],0)

    def test_crossing_deadline_rolls_back_models_and_forecast(self) -> None:
        def slow(*args):
            self.now.return_value=moment(18)
            return self.bundle
        with patch('src.paper_trading.make_models',side_effect=slow):
            with self.assertRaisesRegex(ValueError,'Deadline crossed'):
                self.record()
        self.assertEqual(self.ledger.verify(),1)
        self.assertEqual(self.ledger.all('model'),[])

    def test_source_correction_cannot_rewrite_results(self) -> None:
        self.record()
        self.now.return_value=moment(20)
        settle(self.ledger,self.result_raw('00'))
        old=self.ledger.get('result','2026-10-30')
        with self.assertRaisesRegex(ValueError,'NOT overwritten'):
            settle(self.ledger,self.result_raw('01'))
        self.assertEqual(old,self.ledger.get('result','2026-10-30'))

    def test_historical_revision_blocks_next_forecast(self) -> None:
        self.record()
        self.now.return_value=moment(16,31)
        raw=self.result_raw('00')
        raw.loc[0,'special']='12345' if raw.loc[0,'special']!='12345' else '54321'
        with self.assertRaisesRegex(ValueError,'Historical source revision'):
            record_forecast(self.ledger,raw)

    def test_updates_deletes_and_duplicate_keys_are_blocked(self) -> None:
        for statement in ['UPDATE events SET payload=\'{}\'', 'DELETE FROM events']:
            with self.assertRaises(sqlite3.IntegrityError):
                self.ledger.db.execute(statement)
        with self.assertRaises(sqlite3.IntegrityError), self.ledger.transaction():
            self.ledger.append('protocol','v1',{})
        self.assertEqual(self.ledger.verify(),1)

    def test_integrity_detects_changes_even_if_update_trigger_removed(self) -> None:
        self.ledger.db.execute('DROP TRIGGER no_update')
        self.ledger.db.execute("UPDATE events SET payload='{}'")
        with self.assertRaisesRegex(ValueError,'integrity'):
            self.ledger.verify()

    def test_changed_code_blocks_recording(self) -> None:
        with patch('src.paper_trading.code_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'Frozen code'):
                self.record()

    def test_initialization_after_deadline_starts_tomorrow(self) -> None:
        other=Ledger(self.path/'other.sqlite3')
        try:
            self.now.return_value=moment(18)
            p=initialize(other)
            self.assertEqual(p['start'],'2026-10-31')
            self.assertEqual((pd.Timestamp(p['end'])-pd.Timestamp(p['start'])).days,89)
        finally:
            other.close()


if __name__=='__main__':
    unittest.main()
