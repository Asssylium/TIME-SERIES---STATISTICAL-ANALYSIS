"""Prospective paper-only forecast journal. No CLI option can backdate entries."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.data_loader import XSMBDataLoader
from src.paper_trading import Ledger, checked_protocol, export_report, initialize, record_forecast, settle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'run', 'status', 'verify'])
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent/'reports/paper_trial')
    args = parser.parse_args(argv)
    if args.command != 'init' and not (args.directory/'journal.sqlite3').exists():
        parser.error('No journal exists. Run init first.')
    ledger = Ledger(args.directory/'journal.sqlite3')
    try:
        if args.command == 'init':
            p = initialize(ledger)
            print(f"PAPER ONLY: {p['start']} through {p['end']}; primary Poisson Top 5.")
        elif args.command == 'run':
            p = checked_protocol(ledger, check_code=True)
            raw = XSMBDataLoader(url=p['source_url']).load_data()
            print(settle(ledger, raw))
            print(record_forecast(ledger, raw))
        elif args.command == 'verify':
            print(f"Integrity OK: {ledger.verify()} immutable events.")
        report = export_report(ledger, args.directory)
        print(f"Recorded {report['forecasts_recorded']}; awaiting {report['awaiting_results']}; missed {report['missing_forecasts']}.")
        print(args.directory/'report.md')
        return 0
    except Exception as exc:
        print(f"Paper journal error (existing entries preserved): {exc}", file=sys.stderr)
        return 1
    finally:
        ledger.close()


if __name__ == '__main__':
    raise SystemExit(main())
