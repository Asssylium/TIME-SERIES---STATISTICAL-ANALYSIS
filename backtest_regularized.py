"""Auditable daily-refit backtest of the current regularized Top-K model.

Example: python backtest_regularized.py --year 2025 --top-k 10 --workers 2
Each forecast uses only labels before its target day. Missing calendar days
remain explicit in daily.csv, with unknown cash flows rather than zeroes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as calendar_date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.data_loader import DEFAULT_XSMB_URL, NUMBER_COLUMNS, XSMBDataLoader
from src.diagnostics import evaluate_payouts, rank_top_k
from src.regularized_models import RegularizedLGBMPredictor


def fit_day(dataset: pd.DataFrame, target: pd.Timestamp) -> dict[str, Any]:
    """Fit the default model using known labels and return all 100 scores.

    Feature rows at t predict counts at t+1. Therefore training origins must
    be strictly before target-1, and the inference row must be target-1.
    The precomputed dataset is causal; inference drops its target column.
    Pre-slicing the 1095-day window preserves the model's own window rule
    while avoiding repeated validation of decades of unused training rows.
    """
    cutoff = target - pd.Timedelta(days=1)
    origins = dataset.index.get_level_values(0)
    before = dataset.iloc[:origins.searchsorted(cutoff)]
    available = before[COUNT_TARGET].notna().to_numpy()
    valid_positions = np.flatnonzero(available)
    if not len(valid_positions):
        raise ValueError(f"No known training labels before {target:%Y-%m-%d}.")
    last_origin = before.index.get_level_values(0)[valid_positions[-1]]
    first_origin = last_origin - pd.Timedelta(days=1094)
    lower = before.index.get_level_values(0).searchsorted(first_origin)
    train = before.iloc[lower:]
    train = train.loc[train[COUNT_TARGET].notna()]
    model = RegularizedLGBMPredictor().fit(train)
    latest = dataset.iloc[origins.searchsorted(cutoff):origins.searchsorted(cutoff, side="right")]
    latest = latest.drop(columns=COUNT_TARGET)
    if len(latest) != 100 or model.trained_through_ >= target:
        raise ValueError("Missing preceding day or overlapping training labels.")
    scores = model.predict_proba(latest).droplevel(0).reindex(NUMBER_COLUMNS).to_numpy()
    return {
        "target_date": target.strftime("%Y-%m-%d"),
        "history_end": cutoff.strftime("%Y-%m-%d"),
        "training_label_start": model.training_label_start_.strftime("%Y-%m-%d"),
        "training_label_end": model.trained_through_.strftime("%Y-%m-%d"),
        "training_days": int(train.index.get_level_values(0).nunique()),
        "all_scores": scores.tolist(),
    }


def validate_record(record: dict[str, Any], target: pd.Timestamp) -> None:
    """Reject incompatible or corrupt cached predictions before reuse."""
    scores = np.asarray(record["all_scores"], dtype=float)
    if (record["target_date"] != target.strftime("%Y-%m-%d")
            or pd.Timestamp(record["history_end"]) != target - pd.Timedelta(days=1)
            or not pd.Timestamp(record["training_label_start"]) <= pd.Timestamp(record["training_label_end"]) < target
            or scores.shape != (100,) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError(f"Invalid prediction cache for {target:%Y-%m-%d}.")


def write_reports(
    frequency: pd.DataFrame, dates: pd.DatetimeIndex,
    records: dict[pd.Timestamp, dict[str, Any]], output: Path, top_k: int,
) -> dict[str, Any]:
    """Write calendar-complete daily cash flows, monthly totals and a report."""
    targets = pd.DatetimeIndex(sorted(records))
    if targets.empty:
        raise ValueError("No evaluable dates.")
    scores = np.array([records[t]["all_scores"] for t in targets])
    counts = frequency.loc[targets].to_numpy(dtype=np.int64)
    selected = rank_top_k(scores, top_k)
    payout = evaluate_payouts(counts, selected, payout_mode="per_occurrence")
    evaluated = payout.per_day.copy()
    evaluated.index = targets
    evaluated["prediction"] = [
        " ".join(f"{n:02d}" for n in np.argsort(-s, kind="stable")[:top_k]) for s in scores
    ]
    evaluated["hits_with_counts"] = [
        " ".join(f"{n:02d}x{actual[n]}" for n in np.flatnonzero(mask & (actual > 0)))
        for mask, actual in zip(selected, counts)
    ]
    for key in ["history_end", "training_label_start", "training_label_end", "training_days"]:
        evaluated[key] = [records[t][key] for t in targets]
    daily = evaluated.reindex(dates)
    daily.index.name = "target_date"
    daily.insert(0, "status", [
        "evaluated" if t in records else
        "missing_actual" if t not in frequency.index else "missing_previous_day"
        for t in dates
    ])
    daily.to_csv(output / "daily.csv", encoding="utf-8-sig")
    daily.loc[daily.status != "evaluated", ["status"]].to_csv(output / "excluded_dates.csv", encoding="utf-8-sig")
    monthly_rows = []
    for month in dates.to_period("M").unique():
        mask = targets.to_period("M") == month
        calendar_days = int((dates.to_period("M") == month).sum())
        if mask.any():
            summary = evaluate_payouts(counts[mask], selected[mask], payout_mode="per_occurrence").summary
        else:
            summary = {"days": 0, "occurrences": 0, "cost": 0, "receipts": 0, "profit": 0, "roi": None}
        monthly_rows.append({"month": str(month), "calendar_days": calendar_days,
                             "excluded_days": calendar_days - int(mask.sum()), **summary})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    summary = {"calendar_days": len(dates), "excluded_days": len(dates) - len(targets),
               "top_k": top_k, **payout.summary,
               "losing_days": int((evaluated.profit < 0).sum()),
               "break_even_days": int((evaluated.profit == 0).sum())}
    for key in ["cost", "receipts", "profit", "occurrences"]:
        if monthly[key].sum() != summary[key]:
            raise AssertionError(f"Monthly totals differ for {key}.")
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Regularized Top {top_k}: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}", "",
        "Daily refit of the current RegularizedLGBMPredictor with unchanged default parameters. "
        "Select the highest presence probabilities, not expected occurrence counts. "
        "One unit per selected number per day: cost 23,000 VND; gross receipt 80,000 VND for EACH occurrence. "
        "No separate stake refund, compounding, threshold, or adaptive Top K.", "",
        f"Evaluated {len(targets)}/{len(dates)} calendar days; {len(dates)-len(targets)} excluded. "
        "Missing outcomes or missing preceding days have UNKNOWN cash flows, not zero losses or assumed abstention. "
        "Totals and cumulative profit describe evaluable dates only.", "",
        "| Month | Evaluated days | Excluded | Occurrences | Cost (VND) | Receipts (VND) | Profit (VND) | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in monthly.itertuples():
        roi = f"{row.roi:.2%}" if pd.notna(row.roi) else "N/A"
        lines.append(f"| {row.month} | {row.days} | {row.excluded_days} | {row.occurrences} | {row.cost:,.0f} | {row.receipts:,.0f} | {row.profit:+,.0f} | {roi} |")
    lines.extend([f"| TOTAL | {summary['days']} | {summary['excluded_days']} | {summary['occurrences']} | {summary['cost']:,.0f} | {summary['receipts']:,.0f} | {summary['profit']:+,.0f} | {summary['roi']:.2%} |", "",
        f"Profitable days: {summary['profitable_days']}; losing days: {summary['losing_days']}. "
        f"Maximum drawdown over evaluated days: {summary['max_drawdown']:,.0f} VND.", "",
        "Training labels for each target end before that target. The model retains the preceding 1095 calendar days of labels "
        "relative to the latest known training label. All-history causal features are computed once and sliced by date; "
        "prediction rows contain no target column. Missing dates are never shifted to a different calendar day.", "",
        "This is a retrospective reconstruction using today's model. Historical periods were previously used in model research "
        "and evaluation; this is not an untouched prospective test. No hyperparameters or Top K were selected during this run.", "",
        "[Daily results](./daily.csv) · [Monthly results](./monthly.csv) · [Summary](./summary.csv) · "
        "[Excluded dates](./excluded_dates.csv) · [Protocol](./protocol.json)", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def evaluation_dates(year: int, end: str | None = None) -> pd.DatetimeIndex:
    """Return the year's calendar dates through an optional explicit cutoff."""
    last = calendar_date(year, 12, 31)
    if end is not None:
        try:
            last = calendar_date.fromisoformat(end)
            if last.isoformat() != end or last.year != year:
                raise ValueError("Cutoff must belong to the requested year.")
        except (TypeError, ValueError) as exc:
            raise ValueError("end must be a YYYY-MM-DD date within the requested year.") from exc
    return pd.date_range(calendar_date(year, 1, 1), last)


def run(year: int, top_k: int, output: Path, url: str = DEFAULT_XSMB_URL,
        workers: int = 2, end: str | None = None) -> dict[str, Any]:
    """Freeze the source, resume verified daily fits, and aggregate a full/partial year."""
    if workers not in (1, 2, 3, 4) or not 1 <= top_k <= 100:
        raise ValueError("Use workers 1..4 and top_k 1..100.")
    dates = evaluation_dates(year, end)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    snapshot = output / "source_snapshot.csv"
    loader = XSMBDataLoader(url=snapshot.resolve().as_uri() if snapshot.exists() else url)
    frequency, binary = loader.preprocess_and_build_matrices(loader.load_data())
    if frequency.index.max() < dates[-1]:
        raise ValueError("Source does not yet cover the requested end date.")
    frequency, binary = frequency.loc[:dates[-1]], binary.loc[:dates[-1]]
    root = Path(__file__).resolve().parent
    protocol = {
        "year": year, "top_k": top_k, "model": "regularized", "refit": "daily",
        "start": dates[0].strftime("%Y-%m-%d"), "end": dates[-1].strftime("%Y-%m-%d"),
        "model_parameters": RegularizedLGBMPredictor().params, "history_days": 1095,
        "stake": 23000, "gross_per_occurrence": 80000, "tie_break": "number ascending",
        "frequency_sha256": hashlib.sha256(frequency.to_csv().encode()).hexdigest(),
        "source_url": url,
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm"]}},
        "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in [
            "backtest_regularized.py", "src/count_features.py", "src/count_models.py",
            "src/features.py", "src/data_loader.py", "src/regularized_models.py", "src/diagnostics.py"]},
        "missing_dates": "Require target outcome and preceding calendar day. Exclude unknown cash flows explicitly.",
        "retrospective": "Historical periods were previously used in model research and evaluation; this is not an untouched prospective test.",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Protocol or source changed; choose a new output directory.")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if not snapshot.exists():
        frequency.rename_axis("date").to_csv(snapshot)
    valid = dates.isin(frequency.index) & (dates - pd.Timedelta(days=1)).isin(frequency.index)
    targets = dates[valid]
    print(f"PROTOCOL: {len(targets)}/{len(dates)} evaluable dates; exclusions: {dates[~valid].strftime('%Y-%m-%d').tolist()}", flush=True)
    records = {}
    for target in targets:
        path = cache / f"{target:%Y-%m-%d}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_record(record, target)
            records[target] = record
    pending = [target for target in targets if target not in records]
    print(f"CACHE: {len(records)} available; {len(pending)} daily fits required; workers={workers}", flush=True)
    if pending:
        dataset = build_count_feature_dataset(frequency, binary, include_unlabeled=True)
        began = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fit_day, dataset, target): target for target in pending}
            for done, future in enumerate(as_completed(futures), 1):
                target, record = futures[future], future.result()
                validate_record(record, target)
                path = cache / f"{target:%Y-%m-%d}.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(record), encoding="utf-8")
                temporary.replace(path)
                records[target] = record
                if done == 1 or done % 10 == 0 or done == len(pending):
                    elapsed = time.monotonic() - began
                    print(f"FIT {done}/{len(pending)}; target={target:%Y-%m-%d}; elapsed={elapsed:.0f}s; ETA={(len(pending)-done)*elapsed/done:.0f}s", flush=True)
    summary = write_reports(frequency, dates, records, output, top_k)
    print("SUMMARY " + json.dumps(summary), flush=True)
    print("REPORT " + str(output / "report.md"), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--end", default=None, help="Optional YYYY-MM-DD cutoff within --year; defaults to December 31.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--url", default=DEFAULT_XSMB_URL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("reports/regularized_top10_daily_2025"))
    run(**vars(parser.parse_args()))
