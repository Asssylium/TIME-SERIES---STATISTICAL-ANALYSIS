"""Compare 2026 Top-K payouts with daily refits and a rolling training window.

Downloads the raw prize feed on every run. Results are retrospective and do not
represent a guarantee of future performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.data_loader import XSMBDataLoader
from src.diagnostics import evaluate_payouts, rank_top_k
from src.features import build_feature_dataset
from src.models import LGBMPredictor, TARGET
from src.prize_features import (
    RAW_PRIZE_URL,
    build_prize_feature_dataset,
    normalize_prizes,
)
from src.prize_models import PrizeLGBMPredictor
from src.regularized_models import RegularizedLGBMPredictor


MODEL_NAMES = ("regularized", "poisson", "binary", "lightgbm")
MODEL_OPTIONS = (*MODEL_NAMES, "all")
STAKE = 27_000
GROSS_PAYOUT = 99_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-01", help="First target date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Last target date (YYYY-MM-DD); defaults to the latest source date.")
    parser.add_argument("--history-days", type=int, default=90, help="Rolling calendar days used for each model fit.")
    parser.add_argument("--top-k", type=int, default=3, help="Distinct numbers selected per model and day.")
    parser.add_argument("--model", choices=MODEL_OPTIONS, default="all", help="Model to evaluate, or all.")
    parser.add_argument("--url", default=RAW_PRIZE_URL, help="Live raw prize CSV URL.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory; defaults to a date-stamped reports folder.")
    args = parser.parse_args()
    try:
        start = pd.Timestamp(args.start)
    except (TypeError, ValueError) as exc:
        raise ValueError("--start must be a valid YYYY-MM-DD date.") from exc
    if start.strftime("%Y-%m-%d") != args.start:
        raise ValueError("--start must use YYYY-MM-DD format.")
    if args.end is not None:
        try:
            end = pd.Timestamp(args.end)
        except (TypeError, ValueError) as exc:
            raise ValueError("--end must be a valid YYYY-MM-DD date.") from exc
        if end.strftime("%Y-%m-%d") != args.end:
            raise ValueError("--end must use YYYY-MM-DD format.")
        if start > end:
            raise ValueError("--start must be on or before --end.")
        args.end_date = end
    else:
        args.end_date = None
    if isinstance(args.history_days, bool) or args.history_days < 1:
        raise ValueError("--history-days must be a positive integer.")
    if isinstance(args.top_k, bool) or not 1 <= args.top_k <= 100:
        raise ValueError("--top-k must be an integer in 1..100.")
    args.start_date = start
    return args


def main() -> None:
    args = parse_args()
    print(f"Downloading latest raw prize data from {args.url}", flush=True)
    source = XSMBDataLoader(url=args.url).load_data()
    prizes = normalize_prizes(source)
    frequency, binary = XSMBDataLoader().preprocess_and_build_matrices(prizes.reset_index())
    latest_source_date = frequency.index.max()
    end = min(args.end_date, latest_source_date) if args.end_date is not None else latest_source_date
    if args.end_date is not None and args.end_date > latest_source_date:
        raise ValueError(
            f"Requested end date {args.end_date:%Y-%m-%d} is later than the latest source result "
            f"{latest_source_date:%Y-%m-%d}."
        )
    dates = pd.date_range(args.start_date, end)
    valid = dates.isin(frequency.index) & (dates - pd.Timedelta(days=1)).isin(frequency.index)
    targets = dates[valid]
    if targets.empty:
        raise ValueError("No dates with both a target outcome and its preceding calendar draw.")
    excluded = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "reason": "actual outcome missing" if date not in frequency.index else "preceding calendar day missing",
        }
        for date in dates[~valid]
    ]
    selected_models = MODEL_NAMES if args.model == "all" else (args.model,)
    model_label = args.model
    output = args.output or Path(
        f"work/reports/model_comparison_{args.history_days}d_top{args.top_k}_"
        f"{args.start_date:%Y-%m-%d}_{end:%Y-%m-%d}_{model_label}"
    )
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    print(
        f"Latest result: {end:%Y-%m-%d}. Comparing {len(targets)} shared days "
        f"from {targets[0]:%Y-%m-%d} to {targets[-1]:%Y-%m-%d}; "
        f"{len(excluded)} dates excluded.",
        flush=True,
    )

    feature_start = args.start_date - pd.Timedelta(days=args.history_days + 1)
    datasets = {}
    if "regularized" in selected_models:
        datasets["regularized"] = build_count_feature_dataset(
            frequency, binary, include_unlabeled=True
        )
    if any(name in selected_models for name in ("poisson", "binary")):
        prize = build_prize_feature_dataset(
            source, include_unlabeled=True, start=feature_start
        )
        datasets.update({name: prize for name in ("poisson", "binary") if name in selected_models})
    if "lightgbm" in selected_models:
        datasets["lightgbm"] = build_feature_dataset(
            frequency, binary, include_unlabeled=True
        )
    origins_by_dataset = {
        name: dataset.index.get_level_values(0) for name, dataset in datasets.items()
    }
    params = {}
    if "regularized" in selected_models:
        params["regularized"] = RegularizedLGBMPredictor(history_days=args.history_days).params
    if "poisson" in selected_models:
        params["poisson"] = PrizeLGBMPredictor("poisson", history_days=args.history_days).params
    if "binary" in selected_models:
        params["binary"] = PrizeLGBMPredictor("binary", history_days=args.history_days).params
    if "lightgbm" in selected_models:
        params["lightgbm"] = LGBMPredictor().params
    source_hash = hashlib.sha256(
        prizes.reset_index().to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    protocol = {
        "start": args.start_date.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "days_in_source": len(targets),
        "history_days": args.history_days,
        "history_policy": "Daily rolling calendar-day training labels strictly before target.",
        "top_k": args.top_k,
        "models": list(selected_models),
        "payout": {
            "stake_per_selection_vnd": STAKE,
            "gross_per_occurrence_vnd": GROSS_PAYOUT,
            "separate_stake_refund": False,
        },
        "payout_mode": "per_occurrence",
        "source_url": args.url,
        "source_sha256": source_hash,
        "model_params": params,
        "excluded_dates": excluded,
        "notes": "Retrospective comparison; no model/parameter selection on evaluation outcomes.",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Existing output has a different protocol; select another --output: {output}")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "excluded_dates.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")

    daily_rows: list[dict] = []
    for day_index, target in enumerate(targets, 1):
        cache_path = cache / f"{target:%Y-%m-%d}.npz"
        scores: dict[str, np.ndarray] = {}
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                if str(saved["target_date"].item()) != target.strftime("%Y-%m-%d"):
                    raise ValueError(f"Invalid cached target date in {cache_path}.")
                scores = {name: saved[name] for name in selected_models}
        else:
            first_label = target - pd.Timedelta(days=args.history_days)
            prediction_origin = target - pd.Timedelta(days=1)
            for name in selected_models:
                dataset = datasets[name]
                origins = origins_by_dataset[name]
                labels = origins + pd.Timedelta(days=1)
                target_column = COUNT_TARGET if name in {"regularized", "poisson", "binary"} else TARGET
                training_mask = (
                    dataset[target_column].notna().to_numpy()
                    & (labels >= first_label)
                    & (labels < target)
                )
                train = dataset.loc[training_mask]
                inference = dataset.loc[origins == prediction_origin].drop(columns=target_column)
                if train.empty or len(inference) != 100:
                    raise ValueError(f"Incomplete training/inference rows for {name} at {target:%Y-%m-%d}.")

                if name == "regularized":
                    fitted = RegularizedLGBMPredictor(history_days=args.history_days).fit(train)
                    values = fitted.predict_proba(inference).to_numpy()
                elif name == "poisson":
                    fitted = PrizeLGBMPredictor("poisson", history_days=args.history_days).fit(train)
                    values = fitted.predict_scores(inference).to_numpy()
                elif name == "binary":
                    fitted = PrizeLGBMPredictor("binary", history_days=args.history_days).fit(train)
                    values = fitted.predict_scores(inference).to_numpy()
                else:
                    fitted = LGBMPredictor().fit(train)
                    values = fitted.predict_proba(inference).to_numpy()
                if values.shape != (100,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid prediction scores for {name} at {target:%Y-%m-%d}.")
                scores[name] = values

            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, target_date=np.asarray(target.strftime("%Y-%m-%d")), **scores)
            temporary.replace(cache_path)

        actual = frequency.loc[target].to_numpy(dtype=np.int64)
        for name in selected_models:
            choice = rank_top_k(scores[name].reshape(1, 100), args.top_k)[0]
            payment = evaluate_payouts(
                actual.reshape(1, 100),
                choice.reshape(1, 100),
                payout_mode="per_occurrence",
                stake=STAKE,
                gross_payout=GROSS_PAYOUT,
            ).per_day.iloc[0]
            hits = np.flatnonzero(choice & (actual > 0))
            daily_rows.append(
                {
                    "date": target.strftime("%Y-%m-%d"),
                    "model": name,
                    "top_k": args.top_k,
                    "prediction": " ".join(
                        f"{number:02d}" for number in np.argsort(-scores[name], kind="stable")[:args.top_k]
                    ),
                    "hit_numbers_with_counts": " ".join(f"{number:02d}x{actual[number]}" for number in hits),
                    "distinct_hits": int(payment.distinct_hits),
                    "occurrences": int(payment.paid_occurrences),
                    "cost_vnd": int(payment.cost),
                    "receipts_vnd": int(payment.receipts),
                    "profit_vnd": int(payment.profit),
                }
            )
        if day_index % 20 == 0 or day_index == len(targets):
            print(f"Completed {day_index}/{len(targets)} dates.", flush=True)

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    summary_rows, monthly_rows = [], []
    for name in selected_models:
        rows = daily.loc[daily.model == name]
        summary_rows.append(summarize(name, rows, args.top_k))
        for month, monthly_group in rows.groupby(rows.date.str[:7], sort=True):
            monthly_rows.append(summarize(name, monthly_group, args.top_k, month=month))
    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded, columns=["date", "reason"]).to_csv(
        output / "excluded_dates.csv", index=False, encoding="utf-8-sig"
    )
    write_report(output, summary, monthly, daily, len(dates), len(targets), end, excluded, args)
    print("COMPARISON " + summary.to_json(orient="records"), flush=True)
    print(f"REPORT {output / 'report.md'}", flush=True)


def summarize(model: str, rows: pd.DataFrame, top_k: int, *, month: str | None = None) -> dict:
    cost = int(rows.cost_vnd.sum())
    receipts = int(rows.receipts_vnd.sum())
    return {
        **({"month": month} if month is not None else {}),
        "model": model,
        "days": int(len(rows)),
        "top_k": top_k,
        "occurrences": int(rows.occurrences.sum()),
        "distinct_hits": int(rows.distinct_hits.sum()),
        "profitable_days": int((rows.profit_vnd > 0).sum()),
        "cost_vnd": cost,
        "receipts_vnd": receipts,
        "profit_vnd": receipts - cost,
        "roi": (receipts - cost) / cost if cost else None,
    }


def write_report(
    output: Path, summary: pd.DataFrame, monthly: pd.DataFrame, daily: pd.DataFrame, calendar_days: int,
    evaluated_days: int, end: pd.Timestamp, excluded: list[dict], args: argparse.Namespace,
) -> None:
    lines = [
        f"# Model comparison: 2026 Top {args.top_k}, rolling {args.history_days}-day training window",
        "",
        f"Results cover {args.start_date:%Y-%m-%d} through {end:%Y-%m-%d}. "
        f"{evaluated_days}/{calendar_days} calendar days were evaluable; {len(excluded)} dates were excluded.",
        "",
        f"Downloaded the current raw-prize feed on this run. Each daily fit uses up to {args.history_days} "
        f"calendar days of labels strictly before the target. All models use the same target dates, Top {args.top_k}, "
        f"{STAKE:,} VND stake per selection and {GROSS_PAYOUT:,} VND gross receipt per matching occurrence; "
        "there is no separate stake refund.",
        "",
        "| Model | Days | Occurrences | Profitable days | Cost (VND) | Receipts (VND) | Profit (VND) | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.days} | {row.occurrences} | {row.profitable_days} | "
            f"{row.cost_vnd:,.0f} | {row.receipts_vnd:,.0f} | {row.profit_vnd:+,.0f} | {row.roi:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Monthly profit/loss",
            "",
            "| Month | Model | Days | Cost (VND) | Receipts (VND) | Profit (VND) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in monthly.itertuples(index=False):
        lines.append(
            f"| {row.month} | {row.model} | {row.days} | {row.cost_vnd:,.0f} | "
            f"{row.receipts_vnd:,.0f} | {row.profit_vnd:+,.0f} |"
        )
    if len(summary) == 1:
        lines.extend(
            [
                "",
                "## Daily hit/miss and cash flow",
                "",
                "| Date | Prediction | Matching numbers (count) | Result | Cost (VND) | Receipts (VND) | Profit/loss (VND) |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in daily.itertuples(index=False):
            result = "HIT" if row.distinct_hits else "MISS"
            lines.append(
                f"| {row.date} | {row.prediction} | {row.hit_numbers_with_counts or '—'} | {result} | "
                f"{row.cost_vnd:,.0f} | {row.receipts_vnd:,.0f} | {row.profit_vnd:+,.0f} |"
            )
    lines.extend(
        [
            "",
            "Retrospective backtest results do not prove future advantage. The Poisson score estimates occurrence "
            "counts; binary scores estimate presence and are uncalibrated. Do not interpret these model scores as "
            "reliable real-world odds or select a future model solely from this comparison.",
            "",
            "The report evaluates only the requested date range; 2026 is not yet a complete year. "
            "Dates without the target draw or its preceding calendar draw are excluded, not counted as losses.",
            "",
            "Files: [daily.csv](./daily.csv), [summary.csv](./summary.csv), [monthly.csv](./monthly.csv), "
            "[protocol.json](./protocol.json), [excluded_dates.csv](./excluded_dates.csv).",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
