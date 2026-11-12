"""Bounded permutation screening for persistent number and temporal effects.

These are retrospective diagnostics, not proof that all possible signals are
absent. Fix the discovery/replication split and test families before computing
results; do not search for the most profitable threshold after seeing outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data_loader import XSMBDataLoader
from src.diagnostics import evaluate_payouts


LAGS = (1, 2, 3, 7)


def screen(frequency: pd.DataFrame, permutations: int = 499, seed: int = 42) -> pd.DataFrame:
    """Test three families, using maximum-statistic permutation corrections.

    Number identity: independently relabel each day's 100 numbers, preserving
    its 27 draws and duplicate pattern. Temporal tests: permute whole daily
    vectors within calendar quarters, preserving quarterly rates and within-day
    dependence. Demean within quarters before computing weekday or lag effects.
    Lags refer to exact calendar dates, never adjacent available observations.
    Max-statistic p-values control all tests within each family under the
    corresponding exchangeability null. Bonferroni covers the three families.
    """
    counts = frequency.to_numpy(dtype=float)
    dates = frequency.index
    quarters = dates.to_period("Q")
    blocks = [np.flatnonzero(quarters == quarter) for quarter in quarters.unique()]
    residual = counts.copy()
    for positions in blocks:
        residual[positions] -= residual[positions].mean(axis=0)
    weekday_rows = [np.flatnonzero(dates.dayofweek == day) for day in range(7)]
    if any(len(positions) == 0 for positions in weekday_rows):
        raise ValueError("Weekday tests require observations for all seven weekdays.")
    pairs = []
    for lag in LAGS:
        destination = dates.get_indexer(dates + pd.Timedelta(days=lag))
        source = np.flatnonzero(destination >= 0)
        if len(source) < 2:
            raise ValueError("Insufficient exact-calendar pairs for the requested lags.")
        pairs.append((source, destination[source]))

    def lag_stats(values: np.ndarray) -> np.ndarray:
        result = []
        for left, right in pairs:
            a, b = values[left], values[right]
            a = a - a.mean(axis=0)
            b = b - b.mean(axis=0)
            denominator = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
            result.append(np.divide((a*b).sum(axis=0), denominator,
                                    out=np.zeros(100), where=denominator > 0))
        return np.asarray(result)

    def weekday_stats(values: np.ndarray) -> np.ndarray:
        variation = values.var(axis=0)
        sum_squares = sum(len(pos) * values[pos].mean(axis=0)**2 for pos in weekday_rows)
        return np.divide(sum_squares, variation, out=np.zeros(100), where=variation > 0)

    observed_identity = np.abs(counts.mean(axis=0) - .27)
    observed_lag = lag_stats(residual)
    observed_weekday = weekday_stats(residual)
    null_identity, null_lag, null_weekday = [], [], []
    rng = np.random.default_rng(seed)
    began = time.monotonic()
    for iteration in range(permutations):
        relabeled = rng.permuted(counts, axis=1)
        null_identity.append(np.abs(relabeled.mean(axis=0) - .27).max())
        order = np.arange(len(counts))
        for positions in blocks:
            order[positions] = rng.permutation(positions)
        shuffled = residual[order]
        null_lag.append(np.abs(lag_stats(shuffled)).max())
        null_weekday.append(weekday_stats(shuffled).max())
        if (iteration+1) % 100 == 0:
            print(f"PERM {iteration+1}/{permutations}; elapsed={time.monotonic()-began:.0f}s", flush=True)

    def corrected(null: list[float], statistic: float) -> tuple[float, float]:
        p = (1 + np.count_nonzero(np.asarray(null) >= statistic)) / (permutations + 1)
        return float(p), float(min(1, p * 3))

    rows = []
    for number in range(100):
        p, total = corrected(null_identity, observed_identity[number])
        rows.append({"family": "number_identity", "number": f"{number:02d}", "lag": 0,
                     "effect": float(counts[:, number].mean() - .27),
                     "statistic": float(observed_identity[number]), "family_p": p, "adjusted_p": total})
        p, total = corrected(null_weekday, observed_weekday[number])
        rows.append({"family": "weekday", "number": f"{number:02d}", "lag": 0,
                     "effect": float(np.ptp([residual[pos, number].mean() for pos in weekday_rows])),
                     "weekday_pattern": json.dumps([float(residual[pos, number].mean()) for pos in weekday_rows]),
                     "statistic": float(observed_weekday[number]), "family_p": p, "adjusted_p": total})
        for i, lag in enumerate(LAGS):
            p, total = corrected(null_lag, abs(observed_lag[i, number]))
            rows.append({"family": "self_lag", "number": f"{number:02d}", "lag": lag,
                         "effect": float(observed_lag[i, number]), "statistic": float(abs(observed_lag[i, number])),
                         "family_p": p, "adjusted_p": total})
    return pd.DataFrame(rows)


def run(source: Path, output: Path, permutations: int = 499) -> None:
    """Screen historical effects and evaluate a fixed discovery-only ranking."""
    loader = XSMBDataLoader(url=source.resolve().as_uri())
    frequency, _ = loader.preprocess_and_build_matrices(loader.load_data())
    discovery = frequency.loc[:"2022-12-31"]
    replication = frequency.loc["2023-01-01":]
    if len(discovery) < 365 or len(replication) < 365 or permutations < 99:
        raise ValueError("Require substantial discovery/replication data and >=99 permutations.")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "discovery_end": "2022-12-31", "replication_start": "2023-01-01",
                "replication_end": replication.index[-1].strftime("%Y-%m-%d"),
                "permutations": permutations, "seeds": [42, 43], "lags": list(LAGS),
                "n_tests_per_period": 600, "families": 3,
                "correction": "Permutation maximum statistic within each family, Bonferroni over three families",
                "limitations": "Historical replication, not untouched prospective data. Permutation tests assume within-day number-label exchangeability or within-quarter day exchangeability under their null. Tests do not cover every nonlinear relationship or prove profitability."}
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = {}
    for label, data, seed in [("discovery", discovery, 42), ("replication", replication, 43)]:
        print(f"SCREEN {label}: {len(data)} dates, {data.index[0]:%Y-%m-%d} through {data.index[-1]:%Y-%m-%d}", flush=True)
        results[label] = screen(data, permutations, seed)
        results[label].to_csv(output/f"{label}.csv", index=False, encoding="utf-8-sig")
        best = results[label].sort_values("adjusted_p").groupby("family", sort=False).head(1)
        print("BEST " + best.to_json(orient="records"), flush=True)
    joined = results["discovery"].merge(results["replication"], on=["family", "number", "lag"], suffixes=("_discovery", "_replication"))
    joined["significant_both"] = (joined.adjusted_p_discovery <= .05) & (joined.adjusted_p_replication <= .05)
    joined["same_direction"] = np.sign(joined.effect_discovery) == np.sign(joined.effect_replication)
    for index in joined.index[joined.family == "weekday"]:
        left = json.loads(joined.loc[index, "weekday_pattern_discovery"])
        right = json.loads(joined.loc[index, "weekday_pattern_replication"])
        joined.loc[index, "same_direction"] = np.dot(left, right) > 0
    joined.to_csv(output/"comparison.csv", index=False, encoding="utf-8-sig")
    # A fixed top 10 screens whether a persistent count-frequency bias pays off.
    # Selection uses only pre-2023 counts; do not optimize K or change the list.
    selected_numbers = np.argsort(-discovery.mean().to_numpy(), kind="stable")[:10]
    selected = np.zeros((len(replication), 100), dtype=bool)
    selected[:, selected_numbers] = True
    payout = evaluate_payouts(replication.to_numpy(), selected, payout_mode="per_occurrence")
    daily = payout.per_day.set_axis(replication.index)
    daily.to_csv(output/"fixed_top10_replication.csv", encoding="utf-8-sig")
    summary = {"discovery_days": len(discovery), "replication_days": len(replication),
               "significant_discovery": int((results['discovery'].adjusted_p <= .05).sum()),
               "significant_replication": int((results['replication'].adjusted_p <= .05).sum()),
               "significant_both_same_direction": int((joined.significant_both & joined.same_direction).sum()),
               "fixed_top10": [f"{n:02d}" for n in selected_numbers],
               "fixed_top10_payout": payout.summary,
               "missing_replication_dates": pd.date_range(replication.index[0], replication.index[-1]).difference(replication.index).strftime("%Y-%m-%d").tolist()}
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("work/reports/signal_diagnostics_2026-10-30"))
    parser.add_argument("--permutations", type=int, default=499)
    run(**vars(parser.parse_args()))
