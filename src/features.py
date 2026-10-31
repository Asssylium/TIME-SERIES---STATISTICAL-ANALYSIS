"""Causal features for daily XSMB matrices, including the valid outcome '00'.

Inputs are date-indexed DataFrames from XSMBDataLoader, or NumPy arrays with
100 columns in numeric order. Arrays imply consecutive daily observations.
Date-indexed inputs preserve calendar gaps; missing days are never imputed.
Features at t include the completed draw at t and predict t + 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data_loader import DRAW_COUNT, MATRIX_WIDTH, NUMBER_COLUMNS


MatrixInput = pd.DataFrame | NDArray[np.generic]
DEFAULT_WINDOWS = (7, 14, 30, 60)


def compute_rolling_frequency(
    freq_matrix: MatrixInput, windows: Sequence[int] = DEFAULT_WINDOWS
) -> dict[int, pd.DataFrame]:
    """Return {window_days: frequency_sum} with input dates and columns 00..99.

    At date t, window W includes observations in [t - W + 1 day, t]. During
    warm-up, sum the available history. Missing dates contribute no observed
    counts; their true outcomes remain unknown. Arrays use W consecutive rows.
    Accumulation uses int64 to avoid uint8 overflow. Only windows are looped;
    prefix sums compute all dates/numbers together.
    """
    values, index = _validate_matrix(freq_matrix, binary=False)
    selected_windows = _validate_windows(windows)
    days = _day_offsets(index)
    sums = _rolling_sums(values, days, selected_windows)
    return {
        window: pd.DataFrame(total, index=index, columns=NUMBER_COLUMNS)
        for window, total in sums.items()
    }


def compute_gap_days(binary_matrix: MatrixInput) -> pd.DataFrame:
    """Return elapsed calendar days since each number's last observed appearance.

    An appearance today gives 0. Before the first appearance, return -1 because
    the previous appearance is unknown (left-censored history). Missing dates
    count toward elapsed time, but an unobserved appearance cannot be inferred.
    Arrays assume consecutive days. The result is int64, shaped (N_days, 100).
    """
    values, index = _validate_matrix(binary_matrix, binary=True)
    gaps = _gap_values(values, _day_offsets(index))
    return pd.DataFrame(gaps, index=index, columns=NUMBER_COLUMNS)


def compute_co_occurrence(binary_matrix: MatrixInput) -> pd.DataFrame:
    """Return int64 counts C[i,j] of days on which i and j both appeared.

    C is symmetric and shaped (100, 100), with labeled axes '00'..'99'. Its
    diagonal counts days each number appeared. Duplicate draws count once per
    day because the input is binary. Cast before multiplication to prevent
    overflow with uint8 input. This summarizes the ENTIRE supplied period;
    never attach it to earlier training rows, which would leak future data.
    """
    values, _ = _validate_matrix(binary_matrix, binary=True)
    return pd.DataFrame(
        values.T @ values, index=NUMBER_COLUMNS, columns=NUMBER_COLUMNS
    )


def build_feature_dataset(
    freq_matrix: MatrixInput,
    binary_matrix: MatrixInput,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    include_unlabeled: bool = False,
) -> pd.DataFrame:
    """Build one training row per (feature_date, number), including number '00'.

    The MultiIndex is ('date', 'number') for dated input or ('day_index',
    'number') for arrays. Features use only observations through feature_date:

    * number_id, frequency_today, present_today;
    * gap_days (-1 before first appearance) and seen_before;
    * rolling_frequency_Wd and observed_days_Wd for each window;
    * history_days: total observed records through this date;
    * co_occurrence_total: sum of off-diagonal entries in this number's row of
      the cumulative co-occurrence matrix THROUGH THIS DATE. A day with m
      distinct numbers contributes m - 1 to each present number. This compact
      summary avoids allocating a (N_days, 100, 100) tensor.

    target_next_day is nullable UInt8: presence on the exact next calendar day.
    By default, omit dates without that label (the last day and dates before a
    calendar gap). Set include_unlabeled=True to retain these rows with pd.NA,
    including the latest features for inference. One input day thus produces
    an empty training frame by default. Arrays use the next row as next day.

    Split unique dates chronologically for validation, keeping all 100 numbers
    of each date in the same fold. Exclude target_next_day from model inputs.
    Recompute neither normalization nor fitted statistics using future folds.
    """
    frequency, index = _validate_matrix(freq_matrix, binary=False)
    presence, binary_index = _validate_matrix(binary_matrix, binary=True)
    if not index.equals(binary_index):
        raise ValueError("Frequency and binary matrices must have identical date indices.")
    if not np.array_equal(presence, frequency > 0):
        raise ValueError("binary_matrix must equal (freq_matrix > 0).")
    selected_windows = _validate_windows(windows)
    days = _day_offsets(index)
    gaps = _gap_values(presence, days)
    rolling = _rolling_sums(frequency, days, selected_windows)
    n_days = len(index)
    distinct_per_day = presence.sum(axis=1, keepdims=True)
    # Sum_j C_t[i,j] - C_t[i,i], without constructing the cumulative C tensor.
    co_occurrence = np.cumsum(presence * (distinct_per_day - 1), axis=0)

    label_available = np.zeros(n_days, dtype=bool)
    label_available[:-1] = np.diff(days) == 1
    rows = np.arange(n_days) if include_unlabeled else np.flatnonzero(label_available)
    n_rows = len(rows)
    identifiers = pd.MultiIndex.from_product(
        [index.take(rows), NUMBER_COLUMNS],
        names=["date" if isinstance(index, pd.DatetimeIndex) else "day_index", "number"],
    )
    columns: dict[str, NDArray[np.generic]] = {
        "number_id": np.tile(np.arange(MATRIX_WIDTH, dtype=np.uint8), n_rows),
        "frequency_today": frequency[rows].ravel(),
        "present_today": presence[rows].ravel(),
        "gap_days": gaps[rows].ravel(),
        "seen_before": (gaps[rows] >= 0).astype(np.uint8).ravel(),
        "history_days": np.repeat(rows + 1, MATRIX_WIDTH),
        "co_occurrence_total": co_occurrence[rows].ravel(),
    }
    for window, total in rolling.items():
        columns[f"rolling_frequency_{window}d"] = total[rows].ravel()
        observed_days = np.arange(n_days) + 1 - _window_starts(days, window)
        columns[f"observed_days_{window}d"] = np.repeat(observed_days[rows], MATRIX_WIDTH)
    result = pd.DataFrame(columns, index=identifiers)
    targets = pd.array(
        presence[np.minimum(rows + 1, n_days - 1)].ravel(), dtype="UInt8"
    )
    targets[np.repeat(~label_available[rows], MATRIX_WIDTH)] = pd.NA
    result["target_next_day"] = targets
    return result


def _validate_matrix(
    matrix: MatrixInput, *, binary: bool
) -> tuple[NDArray[np.int64], pd.Index]:
    """Check shape, dates, numeric domain and daily totals; never mutate input."""
    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2 or matrix.shape[1] != MATRIX_WIDTH:
            raise ValueError("Matrices must have shape (N_days, 100).")
        frame = pd.DataFrame(matrix, columns=NUMBER_COLUMNS)
    elif isinstance(matrix, pd.DataFrame):
        if not matrix.columns.is_unique or set(matrix.columns) != set(NUMBER_COLUMNS):
            raise ValueError("DataFrame columns must be the unique strings '00'..'99'.")
        frame = matrix.loc[:, list(NUMBER_COLUMNS)]
    else:
        raise TypeError("Expected a pandas DataFrame or NumPy ndarray.")
    if frame.empty:
        raise ValueError("At least one daily observation is required.")
    index = frame.index
    if isinstance(index, pd.DatetimeIndex):
        if (
            index.hasnans or index.tz is not None or not index.is_unique
            or not index.is_monotonic_increasing or not index.equals(index.normalize())
        ):
            raise ValueError("Dates must be unique, sorted, timezone-naive midnight dates.")
    elif not isinstance(index, pd.RangeIndex) or index.start != 0 or index.step != 1:
        raise ValueError("Use a DatetimeIndex or a consecutive RangeIndex starting at 0.")
    if any(
        not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise ValueError("Matrix values must be real numeric counts, not strings or objects.")
    raw = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    upper = 1 if binary else DRAW_COUNT
    if (
        not np.isfinite(raw).all() or (raw < 0).any() or (raw > upper).any()
        or (raw != np.floor(raw)).any()
    ):
        raise ValueError(f"Matrix entries must be finite integers in 0..{upper}.")
    values = raw.astype(np.int64)
    if not binary and not (values.sum(axis=1) == DRAW_COUNT).all():
        raise ValueError("Each frequency row must sum to exactly 27 draw results.")
    return values, index


def _validate_windows(windows: Sequence[int]) -> tuple[int, ...]:
    """Reject empty, duplicate, noninteger, or nonpositive rolling windows."""
    try:
        selected = tuple(windows)
    except TypeError as exc:
        raise ValueError("windows must be a sequence of positive integers.") from exc
    if not selected or any(
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer)) or window <= 0
        for window in selected
    ):
        raise ValueError("windows must contain positive integers.")
    if len(set(selected)) != len(selected):
        raise ValueError("windows must not contain duplicates.")
    return tuple(int(window) for window in selected)


def _day_offsets(index: pd.Index) -> NDArray[np.int64]:
    """Return calendar offsets independent of pandas timestamp resolution."""
    if isinstance(index, pd.DatetimeIndex):
        days = index.to_numpy(dtype="datetime64[D]").astype(np.int64)
        return days - days[0]
    return np.arange(len(index), dtype=np.int64)


def _window_starts(days: NDArray[np.int64], window: int) -> NDArray[np.int64]:
    """Find each inclusive rolling-window start in a sorted calendar axis."""
    effective_window = min(window, int(days[-1]) + 1)
    return np.searchsorted(days, days - effective_window + 1, side="left")


def _rolling_sums(
    values: NDArray[np.int64], days: NDArray[np.int64], windows: tuple[int, ...]
) -> dict[int, NDArray[np.int64]]:
    """Share one int64 prefix-sum array across all requested windows."""
    cumulative = np.empty((len(values) + 1, MATRIX_WIDTH), dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(values, axis=0, dtype=np.int64, out=cumulative[1:])
    return {
        window: cumulative[1:] - cumulative[_window_starts(days, window)]
        for window in windows
    }


def _gap_values(values: NDArray[np.int64], days: NDArray[np.int64]) -> NDArray[np.int64]:
    """Forward-fill last-seen positions using a vectorized cumulative maximum."""
    positions = np.arange(len(values), dtype=np.int64)[:, None]
    last_seen = np.maximum.accumulate(np.where(values > 0, positions, -1), axis=0)
    gaps = days[:, None] - days[np.maximum(last_seen, 0)]
    gaps[last_seen < 0] = -1
    return gaps
