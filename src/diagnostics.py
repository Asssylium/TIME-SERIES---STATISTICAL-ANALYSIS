"""Probability, payout, and uncertainty diagnostics for chronological forecasts.

The payout is the gross amount received, not profit on top of a refunded stake.
Presence-based precision and count-based payouts intentionally remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


PayoutMode = Literal["once_per_day", "per_occurrence"]


@dataclass(frozen=True)
class PayoutResult:
    """Daily cash flows and aggregate results under an explicit payout rule."""

    per_day: pd.DataFrame
    summary: dict[str, float | int | str]


def rank_top_k(scores: NDArray[np.generic], k: int = 5) -> NDArray[np.bool_]:
    """Select K unique numbers per day, breaking ties by number ascending."""
    values = _matrix(scores)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= 100:
        raise ValueError("k must be an integer in 1..100.")
    selected = np.zeros(values.shape, dtype=bool)
    order = np.argsort(-values, axis=1, kind="stable")[:, :k]
    np.put_along_axis(selected, order, True, axis=1)
    return selected


def evaluate_payouts(
    frequency: NDArray[np.generic],
    selected: NDArray[np.generic],
    *,
    payout_mode: PayoutMode,
    stake: float = 23_000,
    gross_payout: float = 80_000,
) -> PayoutResult:
    """Compute daily receipts, profit and ROI without treating repeats as hits.

    frequency is the actual (N_days, 100) count matrix, each row totaling 27.
    selected is a same-shaped binary selection mask. It can include abstention
    days with no selected numbers. Stake is paid once per selected number/day.
    In per_occurrence mode, each occurrence pays gross_payout; once_per_day
    mode pays only once for each distinct matching number. No stake is returned
    separately. ROI is undefined (NaN) if no money was staked.
    """
    counts = _matrix(frequency)
    choices = _matrix(selected)
    if counts.shape != choices.shape:
        raise ValueError("Frequency and selection shapes must match.")
    if (counts < 0).any() or (counts != np.floor(counts)).any() or not np.all(counts.sum(axis=1) == 27):
        raise ValueError("Frequency rows must contain nonnegative integer counts totaling 27.")
    if not np.isin(choices, [0, 1]).all():
        raise ValueError("selected must be binary.")
    if payout_mode not in {"once_per_day", "per_occurrence"}:
        raise ValueError("Specify once_per_day or per_occurrence.")
    if not np.isfinite(stake) or not np.isfinite(gross_payout) or stake <= 0 or gross_payout <= 0:
        raise ValueError("Stake and gross payout must be finite positive amounts.")
    distinct = ((counts > 0) * choices).sum(axis=1).astype(np.int64)
    occurrences = (counts * choices).sum(axis=1).astype(np.int64)
    n_selected = choices.sum(axis=1).astype(np.int64)
    paid = occurrences if payout_mode == "per_occurrence" else distinct
    cost = n_selected * stake
    receipts = paid * gross_payout
    profit = receipts - cost
    cumulative = np.cumsum(profit)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    drawdown = running_peak - cumulative
    per_day = pd.DataFrame({
        "selections": n_selected, "distinct_hits": distinct,
        "paid_occurrences": occurrences, "payout_units": paid,
        "cost": cost, "receipts": receipts, "profit": profit,
        "cumulative_profit": cumulative, "drawdown": drawdown,
    })
    total_cost = float(cost.sum())
    total_choices = int(n_selected.sum())
    return PayoutResult(per_day, {
        "payout_mode": payout_mode, "days": len(counts), "selections": total_choices,
        "distinct_hits": int(distinct.sum()), "occurrences": int(occurrences.sum()),
        "payout_units": int(paid.sum()), "cost": total_cost,
        "receipts": float(receipts.sum()), "profit": float(profit.sum()),
        "roi": float(profit.sum() / total_cost) if total_cost else float("nan"),
        "precision_at_k": float(distinct.sum() / total_choices) if total_choices else float("nan"),
        "occurrences_per_selection": float(occurrences.sum() / total_choices) if total_choices else float("nan"),
        "days_with_hit": int(np.count_nonzero(distinct)),
        "profitable_days": int(np.count_nonzero(profit > 0)),
        "mean_profit_per_day": float(profit.mean()), "max_drawdown": float(drawdown.max()),
        "break_even_units_per_selection": stake / gross_payout,
    })


def probability_diagnostics(
    actual_binary: NDArray[np.generic], probabilities: NDArray[np.generic], n_bins: int = 10
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return Brier/log loss and a reliability table for presence probabilities.

    Metrics cover all number/day pairs; these are not assumed independent.
    ECE is a descriptive, bin-dependent calibration error, not a ranking score.
    """
    actual, predicted = _matrix(actual_binary), _matrix(probabilities)
    if actual.shape != predicted.shape or not np.isin(actual, [0, 1]).all():
        raise ValueError("Matching shapes and binary ground truth are required.")
    if (predicted < 0).any() or (predicted > 1).any():
        raise ValueError("Probabilities must be in [0, 1].")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    y, p = actual.ravel(), predicted.ravel()
    ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    sizes = np.bincount(ids, minlength=n_bins)
    predicted_sum = np.bincount(ids, weights=p, minlength=n_bins)
    observed_sum = np.bincount(ids, weights=y, minlength=n_bins)
    means = np.divide(predicted_sum, sizes, out=np.full(n_bins, np.nan), where=sizes > 0)
    observed = np.divide(observed_sum, sizes, out=np.full(n_bins, np.nan), where=sizes > 0)
    bounded = np.clip(p, 1e-12, 1 - 1e-12)
    summary = {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
        "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
        "ece": float(np.nansum(sizes * np.abs(means - observed)) / len(p)),
    }
    return summary, pd.DataFrame({
        "bin_low": np.arange(n_bins) / n_bins, "bin_high": (np.arange(n_bins) + 1) / n_bins,
        "count": sizes, "mean_predicted": means, "observed_rate": observed,
    })


def block_bootstrap_mean_interval(
    daily_values: NDArray[np.generic], *, block_length: int = 7,
    n_resamples: int = 2000, confidence: float = 0.95, seed: int = 42,
) -> tuple[float, float]:
    """Approximate a mean interval using circular moving blocks of whole days.

    Preserve all within-day effects by passing daily aggregates or paired daily
    model-minus-reference differences. This descriptive interval assumes local
    temporal stability and is not a multiple-model-selection correction.
    """
    values = np.asarray(daily_values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite daily vector.")
    for value in [block_length, n_resamples]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Block length and resample count must be positive integers.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    block_length = min(block_length, len(values))
    rng = np.random.default_rng(seed)
    n_blocks = (len(values) + block_length - 1) // block_length
    starts = rng.integers(0, len(values), size=(n_resamples, n_blocks, 1))
    indices = (starts + np.arange(block_length)) % len(values)
    means = values[indices.reshape(n_resamples, -1)[:, :len(values)]].mean(axis=1)
    lower, upper = np.quantile(means, [(1 - confidence) / 2, (1 + confidence) / 2])
    return float(lower), float(upper)


def _matrix(value: NDArray[np.generic]) -> NDArray[np.float64]:
    """Validate an explicit numeric matrix with all 100 candidate columns."""
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != 100 or raw.dtype.kind not in "biuf":
        raise ValueError("Expected a nonempty real numeric (N_days, 100) matrix.")
    array = raw.astype(float)
    if not np.isfinite(array).all():
        raise ValueError("Matrix values must be finite.")
    return array
