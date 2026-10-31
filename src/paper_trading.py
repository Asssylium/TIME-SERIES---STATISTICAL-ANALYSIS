"""Append-only, prospective paper forecasts and settlement; no real wagers."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

import numpy as np
import pandas as pd
from lightgbm import Booster

from .count_features import COUNT_TARGET
from .data_loader import XSMBDataLoader
from .prize_features import EXTRA_FEATURES, RAW_PRIZE_URL, build_prize_feature_dataset, normalize_prizes
from .prize_models import PrizeLGBMPredictor
from .regularized_models import RegularizedLGBMPredictor


LOCAL = timezone(timedelta(hours=7))
ROOT = Path(__file__).resolve().parents[1]
MODELS = ("prize_poisson", "prize_binary", "regularized")
CODE_FILES = ("src/paper_trading.py", "paper_trade.py", "src/prize_features.py",
              "src/prize_models.py", "src/regularized_models.py", "src/count_features.py",
              "src/count_models.py", "src/features.py", "src/data_loader.py")


def utc_now() -> datetime:
    """Read the system clock; the CLI provides no backdating override."""
    return datetime.now(timezone.utc)


def canonical(value: Any) -> str:
    """Stable JSON for hashes, rejecting non-finite numeric values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Ledger:
    """SQLite event journal with unique keys, transactional appends and hashes.

    UPDATE/DELETE triggers deter accidental edits. A privileged filesystem
    owner can bypass them/rewrite the chain: this is not external notarization.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, event_key TEXT NOT NULL,
                created_utc TEXT NOT NULL, payload TEXT NOT NULL,
                previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL,
                UNIQUE(kind, event_key));
            CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'Journal is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'Journal is append-only'); END;
        """)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize writers; a crash cannot commit a partial forecast bundle."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.verify()
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def verify(self) -> int:
        previous = "0" * 64
        count = 0
        for row in self.db.execute("SELECT * FROM events ORDER BY id"):
            count += 1
            expected = digest(canonical([count, row['kind'], row['event_key'], row['created_utc'],
                                         row['payload'], previous]))
            if row['id'] != count or row['previous_hash'] != previous or row['event_hash'] != expected:
                raise ValueError(f"Journal integrity failure at event {row['id']}")
            previous = row['event_hash']
        return count

    def get(self, kind: str, key: str) -> dict | None:
        row = self.db.execute("SELECT payload FROM events WHERE kind=? AND event_key=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind: str) -> list[tuple[str, dict]]:
        return [(row[0], json.loads(row[1])) for row in self.db.execute(
            "SELECT event_key,payload FROM events WHERE kind=? ORDER BY event_key", (kind,))]

    def append(self, kind: str, key: str, payload: dict) -> None:
        if not self.db.in_transaction:
            raise RuntimeError("Appends require a transaction")
        tail = self.db.execute("SELECT id,event_hash FROM events ORDER BY id DESC LIMIT 1").fetchone()
        number, previous = (tail[0]+1, tail[1]) if tail else (1, "0"*64)
        timestamp, encoded = utc_now().isoformat(), canonical(payload)
        event_hash = digest(canonical([number, kind, key, timestamp, encoded, previous]))
        self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                        (number, kind, key, timestamp, encoded, previous, event_hash))


def environment() -> dict:
    return {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "lightgbm", "scikit-learn")}


def code_hashes() -> dict:
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CODE_FILES}


def initialize(ledger: Ledger) -> dict:
    """Freeze a 90-calendar-day protocol, starting today if before 18:00."""
    with ledger.transaction():
        existing = ledger.get("protocol", "v1")
        if existing:
            return existing
        now = utc_now().astimezone(LOCAL)
        start = now.date() + timedelta(days=int(now.time() >= time(18)))
        config = {"schema": 1, "mode": "paper_only", "start": str(start),
                  "end": str(start+timedelta(days=89)), "days": 90, "timezone": "UTC+07:00",
                  "forecast_deadline": "18:00", "settlement_not_before": "19:00",
                  "primary": "prize_poisson", "models": list(MODELS), "top_k": 5,
                  "stake": 23000, "gross_per_occurrence": 80000, "history_days": 1095,
                  "source_url": RAW_PRIZE_URL, "refit": "initial start, then each calendar quarter",
                  "parameters": {"prize_poisson": PrizeLGBMPredictor("poisson").params,
                                 "prize_binary": PrizeLGBMPredictor("binary").params,
                                 "regularized": RegularizedLGBMPredictor().params},
                  "code_sha256": code_hashes(), "environment": environment(),
                  "missing_days": "Missing forecasts are coverage failures, not losses or abstentions.",
                  "selection": "Primary/model/K/evaluation horizon frozen; no winner switching.",
                  "timestamp_limit": "Local system clock; no trusted external timestamp."}
        ledger.append("protocol", "v1", config)
        return config


def checked_protocol(ledger: Ledger, *, check_code: bool = False) -> dict:
    ledger.verify()
    protocol = ledger.get("protocol", "v1")
    if protocol is None:
        raise ValueError("Run init first")
    if check_code and (protocol['code_sha256'] != code_hashes() or protocol['environment'] != environment()):
        raise ValueError("Frozen code/dependencies changed; restore them before recording or scoring. Do not overwrite the protocol.")
    return protocol


def snapshot(prizes: pd.DataFrame) -> str:
    """Canonical normalized CSV, with all 27 values and leading zeros intact."""
    return prizes.reset_index().to_csv(index=False, lineterminator="\n")


def make_models(prizes: pd.DataFrame, cutoff: pd.Timestamp, config: dict) -> dict:
    """Train only on the initial/quarter cutoff, before current target day."""
    raw = prizes.loc[prizes.index <= cutoff]
    if raw.empty:
        raise ValueError("No observations before training cutoff")
    data = build_prize_feature_dataset(raw.reset_index(), include_unlabeled=True,
                                       start=raw.index[-1]-pd.Timedelta(days=config['history_days']))
    bundle = {"requested_cutoff": str(cutoff.date()), "source_sha256": digest(snapshot(raw)), "models": {}}
    for name in MODELS:
        print(f"Training frozen {name}; source through {raw.index[-1].date()}", flush=True)
        frame = data.drop(columns=list(EXTRA_FEATURES)) if name == "regularized" else data
        model = (RegularizedLGBMPredictor() if name == "regularized" else
                 PrizeLGBMPredictor("poisson" if name == "prize_poisson" else "binary"))
        model.fit(frame)
        bundle['models'][name] = {"training_label_end": str(model.trained_through_.date()),
                                  "training_label_start": str(model.training_label_start_.date()),
                                  "features": model.feature_columns_,
                                  "booster": model.model_.booster_.model_to_string()}
    return bundle


def record_forecast(ledger: Ledger, raw: pd.DataFrame) -> str:
    """Record today's forecast once, only before deadline and before results.

    Locks throughout computation to prevent concurrent replacements. Recheck
    the wall clock after expensive feature generation/model fitting.
    """
    config = checked_protocol(ledger, check_code=True)
    now = utc_now().astimezone(LOCAL)
    day = str(now.date())
    with ledger.transaction():
        if ledger.get("forecast", day):
            return f"Forecast {day} already recorded; unchanged."
        if not config['start'] <= day <= config['end']:
            return "Outside the frozen 90-day period; no forecast recorded."
        if now.time() >= time(18):
            return "Forecast deadline passed; no backdated forecast permitted."
        prizes = normalize_prizes(raw)
        target = pd.Timestamp(day)
        if (prizes.index >= target).any():
            raise ValueError("Source already includes today's/future results; refuse a prospective forecast.")
        if prizes.index[-1] != target-pd.Timedelta(days=1):
            raise ValueError("Missing complete previous-day results; no forecast recorded.")
        # A changed historical input must be investigated, not silently accepted.
        for _, previous in ledger.all("forecast"):
            old = prizes.loc[prizes.index <= pd.Timestamp(previous['source_through'])]
            if digest(snapshot(old)) != previous['source_sha256']:
                raise ValueError("Historical source revision detected; saved forecasts remain unchanged.")
        period_start = max(target.to_period('Q').start_time, pd.Timestamp(config['start']))
        cutoff = period_start-pd.Timedelta(days=1)
        model_key = str(period_start.date())
        bundle = ledger.get("model", model_key)
        if bundle is None:
            bundle = make_models(prizes, cutoff, config)
            ledger.append("model", model_key, bundle)
        elif digest(snapshot(prizes.loc[prizes.index <= cutoff])) != bundle['source_sha256']:
            raise ValueError("Training source changed since model freeze")
        current = build_prize_feature_dataset(prizes.reset_index(), include_unlabeled=True,
                                              start=prizes.index[-1]).drop(columns=COUNT_TARGET)
        forecasts = {}
        for name, saved in bundle['models'].items():
            if pd.Timestamp(saved['training_label_end']) >= target:
                raise ValueError("Model trained on current or future outcome")
            booster = Booster(model_str=saved['booster'])
            score = np.asarray(booster.predict(current[saved['features']].astype(float)), dtype=float)
            if score.shape != (100,) or not np.isfinite(score).all() or (score < 0).any():
                raise ValueError("Invalid model scores")
            if name == 'prize_poisson':
                if score.sum() <= 0:
                    raise ValueError("Invalid count total")
                score = score/score.max()
                score = 27*score/score.sum()
            elif (score > 1).any():
                raise ValueError("Invalid presence probability")
            selected = np.argsort(-score, kind='stable')[:config['top_k']]
            forecasts[name] = {"numbers": [f"{n:02d}" for n in selected], "scores_00_to_99": score.tolist(),
                               "score_type": "expected_occurrences" if name == 'prize_poisson' else "uncalibrated_presence_probability"}
        csv = snapshot(prizes)
        completed = utc_now().astimezone(LOCAL)
        if completed.date() != now.date() or completed.time() >= time(18):
            raise ValueError("Deadline crossed during computation; entire forecast transaction rolled back")
        ledger.append("forecast", day, {"date": day, "model_key": model_key, "protocol_sha256": digest(canonical(config)),
                      "source_through": str(prizes.index[-1].date()), "source_sha256": digest(csv),
                      "source_csv_gzip_base64": base64.b64encode(gzip.compress(csv.encode(), mtime=0)).decode(),
                      "forecasts": forecasts})
        return f"Recorded {day} Top 5: " + " | ".join(f"{name}: {' '.join(forecasts[name]['numbers'])}" for name in MODELS)


def settle(ledger: Ledger, raw: pd.DataFrame) -> str:
    """Score all available recorded dates once; conflicting revisions fail closed."""
    config = checked_protocol(ledger, check_code=True)
    prizes = normalize_prizes(raw)
    frequency, _ = XSMBDataLoader().preprocess_and_build_matrices(prizes.reset_index())
    now = utc_now().astimezone(LOCAL)
    added = 0
    with ledger.transaction():
        for day, forecast in ledger.all("forecast"):
            date = pd.Timestamp(day)
            if date not in prizes.index or date.date() > now.date() or (date.date() == now.date() and now.time() < time(19)):
                continue
            actual = [int(x) for x in frequency.loc[date].to_numpy()]
            outcome_hash = digest(canonical(prizes.loc[date].to_dict()))
            existing = ledger.get("result", day)
            if existing:
                if existing['counts_00_to_99'] != actual or existing['raw_prizes_sha256'] != outcome_hash:
                    raise ValueError(f"Source correction on {day}; existing settlement NOT overwritten")
                continue
            result = {"date": day, "counts_00_to_99": actual, "raw_prizes": prizes.loc[date].to_dict(),
                      "raw_prizes_sha256": outcome_hash, "models": {}}
            for name in MODELS:
                numbers = forecast['forecasts'][name]['numbers']
                counts = [actual[int(number)] for number in numbers]
                cost = len(numbers)*config['stake']
                receipts = sum(counts)*config['gross_per_occurrence']
                result['models'][name] = {"numbers": numbers, "counts": counts,
                                        "distinct_hits": sum(x > 0 for x in counts), "paid_occurrences": sum(counts),
                                        "cost": cost, "receipts": receipts, "profit": receipts-cost}
            ledger.append("result", day, result)
            added += 1
    return f"Settled {added} new day(s); previous forecasts/results unchanged."


def export_report(ledger: Ledger, directory: Path) -> dict:
    """Rebuild disposable reports; the SQLite journal is the source of truth."""
    config = checked_protocol(ledger)
    forecasts, results = dict(ledger.all('forecast')), dict(ledger.all('result'))
    now = utc_now().astimezone(LOCAL)
    rows, coverage = [], []
    for day in pd.date_range(config['start'], config['end']):
        key = str(day.date())
        status = ('settled' if key in results else 'awaiting_result' if key in forecasts else
                  'missing_forecast' if day.date() < now.date() or (day.date() == now.date() and now.time() >= time(18)) else
                  'not_recorded_yet' if day.date() == now.date() else 'future')
        coverage.append({'date': key, 'status': status})
        if key in results:
            for name, values in results[key]['models'].items():
                rows.append({'date': key, 'model': name, **{k: v for k, v in values.items() if k not in ('numbers','counts')},
                             'numbers': ' '.join(values['numbers']), 'counts': ' '.join(map(str,values['counts']))})
    columns = ['date','model','distinct_hits','paid_occurrences','cost','receipts','profit','numbers','counts']
    daily = pd.DataFrame(rows, columns=columns)
    summary = {}
    for name in MODELS:
        part = daily.loc[daily.model == name]
        profit = int(part.profit.sum())
        cost = int(part.cost.sum())
        cumulative = np.r_[0, part.profit.to_numpy(dtype=float).cumsum()]
        summary[name] = {'days_settled': len(part), 'cost': cost, 'receipts': int(part.receipts.sum()), 'profit': profit,
                         'roi': profit/cost if cost else None,
                         'max_drawdown': float((np.maximum.accumulate(cumulative)-cumulative).max())}
    coverage_frame = pd.DataFrame(coverage)
    report = {'protocol': config, 'generated_utc': utc_now().isoformat(), 'forecasts_recorded': len(forecasts),
              'awaiting_results': len(forecasts)-len(results),
              'missing_forecasts': int(coverage_frame.status.eq('missing_forecast').sum()), 'models': summary}
    directory.mkdir(parents=True, exist_ok=True)
    daily.to_csv(directory/'daily.csv',index=False,encoding='utf-8-sig')
    coverage_frame.to_csv(directory/'coverage.csv',index=False)
    (directory/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# Sổ dự báo giả lập XSMB', '', f"Thời gian: **{config['start']} đến {config['end']}** (90 ngày lịch).",
             'Phương án chính: **Poisson Top 5**. Binary và Regularized chỉ là đối chứng. Không dùng tiền thật.', '',
             f"Đã ghi {len(forecasts)} ngày; chờ kết quả {len(forecasts)-len(results)}; bỏ lỡ dự báo {report['missing_forecasts']} ngày.", '',
             '| Mô hình | Ngày đã chấm | Vốn giả lập | Thu về | Lãi/lỗ | Sụt giảm tối đa |', '|---|---:|---:|---:|---:|---:|']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['days_settled']} | {s['cost']:,} | {s['receipts']:,} | {s['profit']:+,} | {s['max_drawdown']:,.0f} |")
    lines.extend(['', 'Mỗi mô hình là một tài khoản giả lập riêng. Không cộng vốn/lợi nhuận ba mô hình thành một chiến lược.',
                  'Ngày chưa có kết quả và ngày bỏ lỡ dự báo không được tính thành thua. Báo cáo tỷ lệ bao phủ riêng.',
                  'Sổ dùng đồng hồ máy và chuỗi băm nội bộ; không phải chứng nhận thời gian độc lập. Báo cáo CSV/JSON có thể tái tạo, không phải sổ gốc.', '',
                  '[Kết quả từng ngày](./daily.csv) · [Bao phủ 90 ngày](./coverage.csv) · [Chi tiết](./summary.json)', ''])
    if forecasts:
        day = max(forecasts)
        lines.extend([f'## Dự báo đã khóa gần nhất: {day}', ''])
        for name in MODELS:
            lines.append(f"- {name}: **{' · '.join(forecasts[day]['forecasts'][name]['numbers'])}**")
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report
