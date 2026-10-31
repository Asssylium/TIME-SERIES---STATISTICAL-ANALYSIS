"""Causal, normalized features for predicting next-day occurrence counts.

The target counts every occurrence, matching a payout made for each repeated
draw. It is an expected-count regression target, not an appearance probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import MATRIX_WIDTH, NUMBER_COLUMNS
from .features import DEFAULT_WINDOWS, MatrixInput, build_feature_dataset


COUNT_TARGET = "target_next_count"


def build_count_feature_dataset(
    freq_matrix: MatrixInput,
    binary_matrix: MatrixInput,
    *,
    include_unlabeled: bool = False,
) -> pd.DataFrame:
    """Return one causal feature row per (date, number), with a count target.

    Inputs and their validation follow ``build_feature_dataset``: frequency
    rows contain 27 draws over '00'..'99', binary equals frequency > 0, and
    date-indexed matrices preserve missing calendar days. Arrays assume one
    observation per consecutive day. Neither input is modified.

    ``target_next_count`` is nullable UInt8 and contains the count on the exact
    next calendar day; all 100 targets for a labeled date sum to 27. Missing
    next-day results, including the latest date, produce a wholly missing
    target block. Keep these blocks for inference with include_unlabeled=True.
    The binary target is never returned.

    Rolling counts become ``rolling_rate_Wd`` counts per observed day for
    W = 7, 14, 30, 60. Observed-day counts remain explicit coverage features.
    ``co_occurrence_per_day`` divides the causal co-occurrence row sum by the
    observed history length; raw cumulative counters are removed.

    ``gap_observation_complete`` is 1 only when a number has been observed and
    no missing calendar draw falls after its last observed appearance. A gap
    value with this flag equal to 0 must not be interpreted as an exact absence
    duration. Dates before the first observed appearance also receive 0.

    All features use observations through the feature date, predicting t + 1.
    Split whole dates chronologically and exclude COUNT_TARGET when fitting.
    """
    result = build_feature_dataset(
        freq_matrix, binary_matrix, include_unlabeled=True
    )
    label_available = result.pop("target_next_day").notna().to_numpy()

    result["co_occurrence_per_day"] = (
        result.pop("co_occurrence_total") / result.pop("history_days")
    )
    for window in DEFAULT_WINDOWS:
        result[f"rolling_rate_{window}d"] = (
            result.pop(f"rolling_frequency_{window}d")
            / result[f"observed_days_{window}d"]
        )

    # Validation above guarantees the order, numeric domain and daily totals.
    if isinstance(freq_matrix, pd.DataFrame):
        frequency = freq_matrix.loc[:, list(NUMBER_COLUMNS)].to_numpy(dtype=np.uint8)
    else:
        frequency = np.asarray(freq_matrix, dtype=np.uint8)
    dates = result.index.get_level_values(0).unique()
    if isinstance(dates, pd.DatetimeIndex):
        days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    else:
        days = np.arange(len(dates), dtype=np.int64)

    # A missing interval affects only gaps whose last appearance precedes it.
    # An observed appearance after the interval restores complete knowledge.
    missing_interval = np.r_[False, np.diff(days) > 1]
    last_missing_boundary = np.maximum.accumulate(
        np.where(missing_interval, days, days[0] - 1)
    )
    gap_days = result["gap_days"].to_numpy().reshape(-1, MATRIX_WIDTH)
    last_seen_day = days[:, None] - gap_days
    result["gap_observation_complete"] = (
        (gap_days >= 0) & (last_seen_day >= last_missing_boundary[:, None])
    ).astype(np.uint8).ravel()

    next_counts = np.zeros_like(frequency)
    next_counts[:-1] = frequency[1:]
    targets = pd.array(next_counts.ravel(), dtype="UInt8")
    targets[~label_available] = pd.NA
    result[COUNT_TARGET] = targets
    if not include_unlabeled:
        result = result.loc[label_available].copy()
    return result
