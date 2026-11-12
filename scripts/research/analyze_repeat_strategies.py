"""Chronologically evaluate Top-3 hit-rate strategies from cached daily scores.

The script does not change the production forecast. Model scores must come
from daily walk-forward comparison caches; recent-frequency and repeat filters
use only information available before each target date.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import XSMBDataLoader
from src.diagnostics import block_bootstrap_mean_interval
from src.prize_features import RAW_PRIZE_URL


STRATEGY_NAMES = (
    "poisson",
    "binary",
    "frequency_90d",
    "poisson_no_repeat_1d",
    "poisson_no_repeat_3d",
)
STAKE = 27_000
GROSS_PAYOUT = 99_000
TOP_K = 3
APPROX_RANDOM_HIT_RATE = 1 - (1 - TOP_K / 100) ** 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-cache", type=Path, required=True, help="Cache folder for earlier selection period.")
    parser.add_argument("--stress-cache", type=Path, required=True, help="Cache folder for previously reviewed stress-test period.")
    parser.add_argument("--url", default=RAW_PRIZE_URL)
    parser.add_argument("--output", type=Path, default=Path("work/reports/repeat_strategy_research_90d"))
    return parser.parse_args()


def top_numbers(scores: Sequence[float] | np.ndarray, *, k: int = TOP_K, excluded: set[int] | None = None) -> tuple[int, ...]:
    values = np.asarray(scores, dtype=float)
    if values.shape != (100,) or not np.isfinite(values).all():
        raise ValueError("Each daily score vector must contain 100 finite values.")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k must be an integer in 1..100.")
    blocked = excluded or set()
    order = np.argsort(-values, kind="stable")
    chosen = tuple(int(number) for number in order if int(number) not in blocked)[:k]
    if len(chosen) != k:
        raise ValueError("Not enough eligible numbers to fill Top-K.")
    return chosen


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if isinstance(successes, bool) or isinstance(trials, bool) or not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Require 0 <= successes <= positive trials.")
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return float(center - radius), float(center + radius)


def longest_true_streak(
    values: Sequence[bool], dates: Sequence[pd.Timestamp] | None = None
) -> int:
    if dates is not None and len(dates) != len(values):
        raise ValueError("Streak values and dates must have equal lengths.")
    longest = current = 0
    previous_date: pd.Timestamp | None = None
    for index, value in enumerate(values):
        current_date = pd.Timestamp(dates[index]) if dates is not None else None
        if previous_date is not None and current_date - previous_date != pd.Timedelta(days=1):
            current = 0
        current = current + 1 if value else 0
        longest = max(longest, current)
        previous_date = current_date
    return longest


def score_daily(
    picks: Sequence[int], actual: Sequence[int], *, previous_pick_count: set[int] | None = None
) -> dict[str, int | bool]:
    selected = tuple(picks)
    raw_counts = np.asarray(actual, dtype=float)
    if (
        raw_counts.shape != (100,)
        or not np.isfinite(raw_counts).all()
        or (raw_counts < 0).any()
        or (raw_counts != np.floor(raw_counts)).any()
        or raw_counts.sum() != 27
    ):
        raise ValueError("Actual daily counts must be 100 nonnegative integers totaling 27.")
    counts = raw_counts.astype(int)
    if len(selected) != TOP_K or len(set(selected)) != TOP_K or any(not 0 <= n < 100 for n in selected):
        raise ValueError(f"Each strategy must select {TOP_K} distinct numbers in 00..99.")
    occurrences = int(counts[list(selected)].sum())
    distinct_hits = int(np.count_nonzero(counts[list(selected)]))
    cost = TOP_K * STAKE
    receipts = occurrences * GROSS_PAYOUT
    repeated_numbers = [number for number in selected if previous_pick_count and number in previous_pick_count]
    repeated = len(repeated_numbers)
    repeated_occurrences = int(counts[repeated_numbers].sum()) if repeated_numbers else 0
    return {
        "distinct_hits": distinct_hits,
        "occurrences": occurrences,
        "hit": distinct_hits > 0,
        "cost_vnd": cost,
        "receipts_vnd": receipts,
        "profit_vnd": receipts - cost,
        "repeated_selections": repeated,
        "repeated_occurrences": repeated_occurrences,
        "repeated_cost_vnd": repeated * STAKE,
        "repeated_receipts_vnd": repeated_occurrences * GROSS_PAYOUT,
        "repeated_profit_vnd": repeated_occurrences * GROSS_PAYOUT - repeated * STAKE,
    }


def _read_cache(cache_dir: Path, target: pd.Timestamp) -> dict[str, np.ndarray]:
    path = cache_dir / f"{target:%Y-%m-%d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction cache for {target:%Y-%m-%d}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["target_date"].item()) != target.strftime("%Y-%m-%d"):
            raise ValueError(f"Cached target date does not match filename: {path}")
        missing = {"poisson", "binary"} - set(archive.files)
        if missing:
            raise ValueError(f"Cache {path} lacks score arrays: {sorted(missing)}")
        return {name: np.asarray(archive[name], dtype=float) for name in ("poisson", "binary")}


def _monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    data = daily.assign(month=daily.date.str[:7])
    for (period, month, strategy), group in data.groupby(["period", "month", "strategy"], sort=True):
        row = _summarize(period, strategy, group)
        row["month"] = month
        rows.append(row)
    return pd.DataFrame(rows)


def _hit_rate_deltas(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, group in daily.groupby("period", sort=True):
        hits = group.pivot(index="date", columns="strategy", values="hit").sort_index()
        if "poisson" not in hits:
            raise ValueError(f"Poisson baseline missing from {period}.")
        for strategy in hits.columns:
            delta = hits[strategy].astype(float).to_numpy() - hits["poisson"].astype(float).to_numpy()
            lower, upper = block_bootstrap_mean_interval(delta, block_length=7)
            rows.append({
                "period": period,
                "strategy": strategy,
                "hit_rate_delta_vs_poisson": float(delta.mean()),
                "block_ci95_low": lower,
                "block_ci95_high": upper,
            })
    return pd.DataFrame(rows)


def _summarize(period: str, strategy: str, group: pd.DataFrame) -> dict[str, int | float | str]:
    n = len(group)
    hits = int(group.hit.sum())
    cost = int(group.cost_vnd.sum())
    profit = int(group.profit_vnd.sum())
    lower, upper = wilson_interval(hits, n)
    dates = pd.to_datetime(group.date).tolist()
    hit_streak = longest_true_streak((~group.hit).tolist(), dates)
    loss_streak = longest_true_streak((group.profit_vnd < 0).tolist(), dates)
    return {
        "period": period,
        "strategy": strategy,
        "days": n,
        "hit_days": hits,
        "hit_rate": hits / n,
        "hit_rate_ci95_low": lower,
        "hit_rate_ci95_high": upper,
        "days_with_2plus_distinct_hits": int((group.distinct_hits >= 2).sum()),
        "days_with_3_distinct_hits": int((group.distinct_hits == 3).sum()),
        "days_with_3plus_occurrences": int((group.occurrences >= 3).sum()),
        "days_with_4plus_occurrences": int((group.occurrences >= 4).sum()),
        "occurrences": int(group.occurrences.sum()),
        "days_with_loss": int((group.profit_vnd < 0).sum()),
        "longest_miss_streak": hit_streak,
        "longest_loss_streak": loss_streak,
        "repeated_selections": int(group.repeated_selections.sum()),
        "repeated_selection_hit_rate": (
            float(group.repeated_selection_hits.sum() / group.repeated_selections.sum())
            if group.repeated_selections.sum()
            else float("nan")
        ),
        "repeated_occurrences": int(group.repeated_occurrences.sum()),
        "repeated_cost_vnd": int(group.repeated_cost_vnd.sum()),
        "repeated_receipts_vnd": int(group.repeated_receipts_vnd.sum()),
        "repeated_profit_vnd": int(group.repeated_profit_vnd.sum()),
        "nonrepeat_selections": n * TOP_K - int(group.repeated_selections.sum()),
        "nonrepeat_occurrences": int(group.occurrences.sum()) - int(group.repeated_occurrences.sum()),
        "nonrepeat_profit_vnd": profit - int(group.repeated_profit_vnd.sum()),
        "cost_vnd": cost,
        "receipts_vnd": int(group.receipts_vnd.sum()),
        "profit_vnd": profit,
        "roi": profit / cost if cost else float("nan"),
    }


def evaluate_period(
    cache_dir: Path,
    actual: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    period_name: str,
    require_complete: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range(start, end)
    observed_dates = set(actual.index)
    valid_targets = [
        target for target in dates
        if target in observed_dates and target - pd.Timedelta(days=1) in observed_dates
    ]
    prediction_dates = []
    scores_by_date: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    missing_cache = []
    for target in valid_targets:
        cache_path = cache_dir / f"{target:%Y-%m-%d}.npz"
        if not cache_path.is_file():
            missing_cache.append(target)
            continue
        scores_by_date[target] = _read_cache(cache_dir, target)
        prediction_dates.append(target)
    if require_complete and missing_cache:
        raise FileNotFoundError(
            f"Missing {len(missing_cache)} walk-forward score caches in {period_name}; "
            f"first missing target is {missing_cache[0]:%Y-%m-%d}."
        )
    if not prediction_dates:
        raise ValueError(f"No matching cached predictions and actual results in {period_name}.")

    daily_rows: list[dict[str, object]] = []
    predictions_by_date: dict[pd.Timestamp, dict[str, tuple[int, ...]]] = {}
    for target in prediction_dates:
        scores = scores_by_date[target]
        history = actual.loc[
            (actual.index >= target - pd.Timedelta(days=90)) & (actual.index < target)
        ]
        if history.empty:
            raise ValueError(f"No past results available to rank frequency baseline on {target:%Y-%m-%d}.")
        frequency_scores = history.sum(axis=0).to_numpy(dtype=float)

        one_day_prior = scores_by_date.get(target - pd.Timedelta(days=1), {}).get("poisson")
        three_day_prior = [
            scores_by_date[d]["poisson"]
            for d in pd.date_range(target - pd.Timedelta(days=3), target - pd.Timedelta(days=1))
            if d in scores_by_date
        ]
        previous_picks = set(top_numbers(one_day_prior)) if one_day_prior is not None else set()
        recent_picks = {
            n for vector in three_day_prior for n in top_numbers(vector)
        }
        selections: Mapping[str, tuple[int, ...]] = {
            "poisson": top_numbers(scores["poisson"]),
            "binary": top_numbers(scores["binary"]),
            "frequency_90d": top_numbers(frequency_scores),
            "poisson_no_repeat_1d": top_numbers(scores["poisson"], excluded=previous_picks),
            "poisson_no_repeat_3d": top_numbers(scores["poisson"], excluded=recent_picks),
        }
        actual_counts = actual.loc[target].to_numpy(dtype=int)
        for strategy, picks in selections.items():
            previous = {
                number
                for prior_date in pd.date_range(target - pd.Timedelta(days=3), target - pd.Timedelta(days=1))
                for number in predictions_by_date.get(prior_date, {}).get(strategy, ())
            }
            metrics = score_daily(picks, actual_counts, previous_pick_count=previous)
            selected_hits = sum(int(actual_counts[n] > 0) for n in picks)
            daily_rows.append({
                "date": target.strftime("%Y-%m-%d"),
                "period": period_name,
                "strategy": strategy,
                "prediction": " ".join(f"{n:02d}" for n in picks),
                "hit": bool(metrics["hit"]),
                "distinct_hits": int(metrics["distinct_hits"]),
                "occurrences": int(metrics["occurrences"]),
                "repeated_selections": sum(n in previous for n in picks),
                "repeated_selection_hits": sum(int(actual_counts[n] > 0) for n in picks if n in previous),
                "repeated_occurrences": int(metrics["repeated_occurrences"]),
                "repeated_cost_vnd": int(metrics["repeated_cost_vnd"]),
                "repeated_receipts_vnd": int(metrics["repeated_receipts_vnd"]),
                "repeated_profit_vnd": int(metrics["repeated_profit_vnd"]),
                "days_with_2plus_distinct_hits": int(selected_hits >= 2),
                "days_with_3_distinct_hits": int(selected_hits == 3),
                "days_with_3plus_occurrences": int(metrics["occurrences"] >= 3),
                "days_with_4plus_occurrences": int(metrics["occurrences"] >= 4),
                "cost_vnd": int(metrics["cost_vnd"]),
                "receipts_vnd": int(metrics["receipts_vnd"]),
                "profit_vnd": int(metrics["profit_vnd"]),
            })
        predictions_by_date[target] = dict(selections)
    return pd.DataFrame(daily_rows)


def write_report(output: Path, daily: pd.DataFrame, selection_period: str, stress_period: str) -> None:
    summary = pd.DataFrame([
        _summarize(period, strategy, group)
        for (period, strategy), group in daily.groupby(["period", "strategy"], sort=True)
    ])
    monthly = _monthly_summary(daily)
    deltas = _hit_rate_deltas(daily)
    summary = summary.merge(deltas, on=["period", "strategy"], how="left", validate="one_to_one")
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(output / "hit_rate_deltas.csv", index=False, encoding="utf-8-sig")

    selection = summary.loc[summary.period == selection_period].copy()
    selection["strategy_order"] = selection.strategy.map(
        {name: index for index, name in enumerate(STRATEGY_NAMES)}
    )
    selection = selection.sort_values(
        ["hit_rate", "strategy_order"], ascending=[False, True], kind="stable"
    )
    stress = summary.loc[summary.period == stress_period]
    best_strategy = str(selection.iloc[0].strategy)
    lines = [
        "# Top-3 repeat strategy research",
        "",
        "## Protocol",
        "",
        f"- Strategy selection period: {selection_period}. The best strategy is selected by hit rate only.",
        f"- Previously reviewed stress-test period: {stress_period}; it is descriptive, not an untouched holdout.",
        "- All predictions use cached daily walk-forward scores. Frequency baselines use only results from the preceding 90 calendar days.",
        "- Repeat filters exclude numbers selected by Poisson on the immediately preceding 1 or 3 calendar days, then fill from the current Poisson ranking. Repeat-pick diagnostics classify a selection as repeated if the same strategy selected it in the prior 3 calendar days; its hit rate is per selected number, not per day.",
        "- A hit-day means at least one distinct Top-3 number appeared. Occurrences affect receipts, not hit-day count.",
        f"- Approximate random-selection hit-rate reference: {APPROX_RANDOM_HIT_RATE:.1%}, assuming 27 independent, uniform prize endings; this is only a rough reference.",
        f"- Payout accounting: {STAKE:,} VND per selection and {GROSS_PAYOUT:,} VND per occurrence; no stake refund.",
        "- Poisson is trained on repeat-inclusive occurrence counts, while Binary is trained on next-day presence. Neither objective is guaranteed to maximize Top-3 hit-days; the repeat filters test whether removing recent picks helps rather than assuming it does.",
        "",
        "## Selection-period results",
        "",
        "| Strategy | Hit days | Hit rate (95% Wilson CI) | Δ vs Poisson (95% 7-observation block CI) | 2+ distinct | 3 distinct | 3+ occurrences | 4+ occurrences | Total occurrences | Repeat picks (hit rate; net) | Other picks net | Miss streak max | Loss days | Loss streak max | Net (VND) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:",
    ]
    for row in selection.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.hit_days}/{row.days} | {row.hit_rate:.1%} "
            f"({row.hit_rate_ci95_low:.1%}–{row.hit_rate_ci95_high:.1%}) | "
            f"{row.hit_rate_delta_vs_poisson:+.1%} "
            f"[{row.block_ci95_low:+.1%}, {row.block_ci95_high:+.1%}] | "
            f"{row.days_with_2plus_distinct_hits} | {row.days_with_3_distinct_hits} | "
            f"{row.days_with_3plus_occurrences} | {row.days_with_4plus_occurrences} | "
            f"{row.occurrences} | {row.repeated_selections} ({row.repeated_selection_hit_rate:.1%}; "
            f"{row.repeated_profit_vnd:+,}) | "
            f"{row.nonrepeat_profit_vnd:+,} | "
            f"{row.longest_miss_streak} | {row.days_with_loss} | "
            f"{row.longest_loss_streak} | {row.profit_vnd:+,} |"
        )
    lines.extend(["", "## Previously reviewed stress-test results", ""])
    lines.extend([
        "| Strategy | Hit days | Hit rate (95% Wilson CI) | Δ vs Poisson (95% 7-observation block CI) | 2+ distinct | 3 distinct | 3+ occurrences | 4+ occurrences | Total occurrences | Repeat picks (hit rate; net) | Other picks net | Miss streak max | Loss days | Loss streak max | Net (VND) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:",
    ])
    for row in stress.sort_values("hit_rate", ascending=False, kind="stable").itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.hit_days}/{row.days} | {row.hit_rate:.1%} "
            f"({row.hit_rate_ci95_low:.1%}–{row.hit_rate_ci95_high:.1%}) | "
            f"{row.hit_rate_delta_vs_poisson:+.1%} "
            f"[{row.block_ci95_low:+.1%}, {row.block_ci95_high:+.1%}] | "
            f"{row.days_with_2plus_distinct_hits} | {row.days_with_3_distinct_hits} | "
            f"{row.days_with_3plus_occurrences} | {row.days_with_4plus_occurrences} | "
            f"{row.occurrences} | {row.repeated_selections} ({row.repeated_selection_hit_rate:.1%}; "
            f"{row.repeated_profit_vnd:+,}) | "
            f"{row.nonrepeat_profit_vnd:+,} | "
            f"{row.longest_miss_streak} | {row.days_with_loss} | "
            f"{row.longest_loss_streak} | {row.profit_vnd:+,} |"
        )
    selected_stress = stress.loc[stress.strategy == best_strategy]
    selected_row = selection.loc[selection.strategy == best_strategy].iloc[0]
    selected_stress_row = selected_stress.iloc[0]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"The selection-period winner by hit rate is **{best_strategy}** "
        f"({selected_row.hit_rate:.1%}, {selected_row.hit_days}/{selected_row.days}). "
        f"On the already-reviewed stress-test it reached {selected_stress_row.hit_rate:.1%} "
        f"({selected_stress_row.hit_days}/{selected_stress_row.days}).",
        "This is not evidence that the rule will outperform prospectively: the stress-test year has already been inspected, and the strategy set is small but still selected using historical outcomes.",
        "Do not switch the production command based on these retrospective results. Freeze the selected rule and log forecasts before results for a prospective paper-trial.",
        "Wilson intervals treat days as independent; paired moving-block intervals resample 7 evaluated observations to allow short-range dependence. These intervals are descriptive and not corrected for trying multiple strategies. Monthly results and calendar-aware daily streaks are in monthly.csv and daily.csv.",
        "",
        "Files: [daily.csv](./daily.csv), [summary.csv](./summary.csv), [monthly.csv](./monthly.csv), [hit_rate_deltas.csv](./hit_rate_deltas.csv).",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps({
        "selection_period": selection_period,
        "stress_period": stress_period,
        "selection_metric": "Top-3 hit rate",
        "selection_tie_break": "fixed strategy order; not profit",
        "top_k": TOP_K,
        "history_days": 90,
        "strategies": list(STRATEGY_NAMES),
        "stake_per_selection_vnd": STAKE,
        "gross_per_occurrence_vnd": GROSS_PAYOUT,
        "stress_period_is_untouched": False,
        "production_default_changed": False,
    }, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = XSMBDataLoader(url=args.url).load_data()
    actual, _ = XSMBDataLoader().preprocess_and_build_matrices(source)
    cache_protocols = {}
    for name, directory in (
        ("selection_2025", args.selection_cache),
        ("known_stress_2026", args.stress_cache),
    ):
        protocol_path = directory.parent / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"Missing comparison protocol: {protocol_path}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (
            protocol.get("history_days") != 90
            or protocol.get("top_k") != TOP_K
            or not {"poisson", "binary"}.issubset(protocol.get("models", []))
        ):
            raise ValueError(f"{name} cache must contain 90-day Top-3 Poisson and Binary scores.")
        cache_protocols[name] = protocol
    selection_protocol = cache_protocols["selection_2025"]
    stress_protocol = cache_protocols["known_stress_2026"]
    if not selection_protocol["start"].startswith("2025-") or not stress_protocol["start"].startswith("2026-"):
        raise ValueError("Selection and stress caches must cover 2025 and 2026 respectively.")
    selection_daily = evaluate_period(
        args.selection_cache, actual, pd.Timestamp(selection_protocol["start"]),
        pd.Timestamp(selection_protocol["end"]), period_name="selection_2025",
        require_complete=True,
    )
    stress_daily = evaluate_period(
        args.stress_cache, actual, pd.Timestamp(stress_protocol["start"]),
        pd.Timestamp(stress_protocol["end"]), period_name="known_stress_2026",
        require_complete=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    daily = pd.concat([selection_daily, stress_daily], ignore_index=True)
    write_report(args.output, daily, "selection_2025", "known_stress_2026")
    print(f"REPORT {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
