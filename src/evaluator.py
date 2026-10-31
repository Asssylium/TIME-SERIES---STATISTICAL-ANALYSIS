"""Presence-based Top-K evaluation for daily predictions of numbers 00..99.

Actual draws may repeat a number, but each distinct number counts as one hit.
Predictions must contain K distinct candidates in ranked order. This module
scores supplied predictions; use chronological out-of-fold predictions or a
held-out test set, not predictions on the data used to fit the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any, TextIO, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data_loader import MATRIX_WIDTH


Number = str | int | np.integer
DailyNumbers = Sequence[Number] | NDArray[np.generic] | pd.Series
DailyPredictions = Sequence[Number | tuple[Number, float]] | NDArray[np.generic] | pd.Series


class DailyMetrics(TypedDict):
    """Ranking metrics for a single forecast date."""

    k: int
    n_actual: int
    n_unique_actual: int
    hits: int
    hit_rate_at_k: float
    accuracy_at_k: float
    precision_at_k: float
    recall_at_k: float


class SummaryMetrics(TypedDict):
    """Test-set metrics, with equal weight for each observed day."""

    k: int
    n_days: int
    days_with_hit: int
    total_hits: int
    mean_hits: float
    hit_rate_at_k: float
    accuracy_at_k: float
    precision_at_k: float
    recall_at_k: float


@dataclass(frozen=True)
class EvaluationResult:
    """Daily metrics plus their macro-average summary over the test set."""

    per_day: pd.DataFrame
    summary: SummaryMetrics


def evaluate_top_k(
    y_true_list: DailyNumbers,
    y_pred_top_k_list: DailyPredictions,
    k: int = 15,
) -> DailyMetrics:
    """Evaluate ONE day of actual draws against an ordered prediction list.

    Let A be the set of actual numbers and P the first K predicted numbers:
    hits = |A intersection P|, Precision@K = hits / K, Recall@K = hits / |A|.
    Hit Rate@K and Accuracy@K both equal 1.0 if hits > 0, else 0.0. Across days,
    evaluate_test_set averages these indicators and also reports mean_hits.

    Accept integers 0..99 or one/two-digit strings; '1' and '01' are identical.
    Predictions may also be (number, score) tuples returned by the predictors.
    Their supplied ranking is preserved; scores are not re-sorted. Candidates
    beyond K are ignored. Fewer than K candidates or duplicates within Top-K
    raise ValueError rather than changing the denominator or filling slots.

    Actual lists can have any positive length (27-entry validation belongs to
    the loader). Empty actuals mean missing ground truth and are rejected.
    Floats, booleans, missing and out-of-range numbers are rejected.
    """
    k = _validate_k(k)
    actual, predicted, count = _day_masks(y_true_list, y_pred_top_k_list, k)
    frame = _daily_metrics(actual[None, :], predicted[None, :], np.array([count]), k)
    row = frame.iloc[0]
    return DailyMetrics(
        k=k, n_actual=count, n_unique_actual=int(row["n_unique_actual"]),
        hits=int(row["hits"]), hit_rate_at_k=float(row["hit_rate_at_k"]),
        accuracy_at_k=float(row["accuracy_at_k"]),
        precision_at_k=float(row["precision_at_k"]), recall_at_k=float(row["recall_at_k"]),
    )


def evaluate_test_set(
    y_true_list: Sequence[DailyNumbers],
    y_pred_top_k_list: Sequence[DailyPredictions],
    k: int = 15,
    *,
    dates: Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None = None,
) -> EvaluationResult:
    """Evaluate paired daily lists and return per-day metrics and a summary.

    Both outer sequences must contain the same nonzero number of days in the
    same order. Each prediction list follows evaluate_top_k's input contract.
    Optional dates label the ACTUAL/forecast target days, not feature dates;
    they must be unique, chronological, timezone-naive daily dates.

    Precision, recall and hit rate are macro averages (equal weight per day).
    Recall divides by distinct actual numbers, never the number of raw draws.
    Evaluation uses vectorized boolean masks across all days and 100 numbers;
    a single day loop validates the variable-length input lists.
    """
    k = _validate_k(k)
    truths = _sequence(y_true_list, "y_true_list")
    predictions = _sequence(y_pred_top_k_list, "y_pred_top_k_list")
    if not truths or len(truths) != len(predictions):
        raise ValueError("Test inputs must contain the same nonzero number of days.")
    n_days = len(truths)
    index = _test_index(dates, n_days)
    actual = np.zeros((n_days, MATRIX_WIDTH), dtype=bool)
    predicted = np.zeros_like(actual)
    counts = np.empty(n_days, dtype=np.int64)
    for day, (truth, forecast) in enumerate(zip(truths, predictions)):
        try:
            actual[day], predicted[day], counts[day] = _day_masks(truth, forecast, k)
        except ValueError as exc:
            raise ValueError(f"Test day {index[day]!s}: {exc}") from exc
    per_day = _daily_metrics(actual, predicted, counts, k)
    per_day.index = index
    summary = SummaryMetrics(
        k=k, n_days=n_days, days_with_hit=int(per_day["hits"].gt(0).sum()),
        total_hits=int(per_day["hits"].sum()), mean_hits=float(per_day["hits"].mean()),
        hit_rate_at_k=float(per_day["hit_rate_at_k"].mean()),
        accuracy_at_k=float(per_day["accuracy_at_k"].mean()),
        precision_at_k=float(per_day["precision_at_k"].mean()),
        recall_at_k=float(per_day["recall_at_k"].mean()),
    )
    return EvaluationResult(per_day=per_day, summary=summary)


def print_summary_report(result: EvaluationResult, *, file: TextIO | None = None) -> str:
    """Print a test-set report to stdout (or file) and return its text for reuse.

    All rates are fractions in the result and percentages in the report.
    The report distinguishes days with a hit from the average number of hits.
    """
    summary = result.summary
    lines = [
        f"TOP-{summary['k']} EVALUATION REPORT",
        f"Test days: {summary['n_days']}",
        f"Days with at least one hit: {summary['days_with_hit']}/{summary['n_days']}",
        f"Hit Rate / Accuracy@{summary['k']}: {summary['hit_rate_at_k']:.2%}",
        f"Precision@{summary['k']}: {summary['precision_at_k']:.2%}",
        f"Recall@{summary['k']}: {summary['recall_at_k']:.2%}",
        f"Mean distinct hits per day: {summary['mean_hits']:.3f}",
        f"Total distinct daily hits: {summary['total_hits']}",
        "Rates are averaged equally across days; repeated actual draws count once per number.",
    ]
    report = "\n".join(lines)
    print(report, file=file)
    return report


def _day_masks(
    truth: DailyNumbers, forecast: DailyPredictions, k: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int]:
    """Normalize one day without allowing repeated predictions to inflate hits."""
    actual_items = _sequence(truth, "actual draws")
    predicted_items = _sequence(forecast, "predictions")
    if not actual_items:
        raise ValueError("Actual draws must not be empty; missing ground truth cannot be scored.")
    if len(predicted_items) < k:
        raise ValueError(f"Expected at least {k} predictions, got {len(predicted_items)}.")
    actual_numbers = [_normalize_number(number) for number in actual_items]
    predicted_numbers: list[int] = []
    for item in predicted_items[:k]:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("Scored predictions must be (number, score) tuples.")
            number, score = item
            if (
                isinstance(score, (bool, np.bool_))
                or not isinstance(score, (int, float, np.integer, np.floating))
                or not np.isfinite(score)
            ):
                raise ValueError("Prediction scores must be finite real numbers.")
            item = number
        predicted_numbers.append(_normalize_number(item))
    if len(set(predicted_numbers)) != k:
        raise ValueError("Top-K predictions must contain K distinct numbers.")
    actual = np.zeros(MATRIX_WIDTH, dtype=bool)
    predicted = np.zeros(MATRIX_WIDTH, dtype=bool)
    actual[actual_numbers] = True
    predicted[predicted_numbers] = True
    return actual, predicted, len(actual_items)


def _daily_metrics(
    actual: NDArray[np.bool_], predicted: NDArray[np.bool_],
    counts: NDArray[np.int64], k: int,
) -> pd.DataFrame:
    """Compute all daily metrics with NumPy boolean operations."""
    hits = np.count_nonzero(actual & predicted, axis=1)
    n_unique = np.count_nonzero(actual, axis=1)
    hit_rate = (hits > 0).astype(float)
    return pd.DataFrame({
        "k": k, "n_actual": counts, "n_unique_actual": n_unique, "hits": hits,
        "hit_rate_at_k": hit_rate, "accuracy_at_k": hit_rate,
        "precision_at_k": hits / k, "recall_at_k": hits / n_unique,
    })


def _normalize_number(value: Any) -> int:
    """Normalize an integer or one/two-digit ASCII string to the domain 0..99."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean values are not lottery numbers.")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-9]{1,2}", value.strip()):
        number = int(value.strip())
    else:
        raise ValueError(f"Invalid number {value!r}; expected an integer or digit string 00..99.")
    if not 0 <= number < MATRIX_WIDTH:
        raise ValueError(f"Number {value!r} is outside 00..99.")
    return number


def _sequence(value: Any, name: str) -> list[Any]:
    """Accept ordered sequences/arrays without interpreting strings as lists."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, np.ndarray, pd.Series)):
        raise ValueError(f"{name} must be an ordered sequence.")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise ValueError(f"{name} must not be a scalar array.")
    return list(value)


def _validate_k(k: int) -> int:
    """Require a meaningful Top-K size in the 100-number domain."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= MATRIX_WIDTH:
        raise ValueError("k must be an integer between 1 and 100.")
    return int(k)


def _test_index(
    dates: Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None, n_days: int
) -> pd.Index:
    """Validate optional target dates without silently sorting paired results."""
    if dates is None:
        return pd.RangeIndex(n_days, name="day_index")
    if isinstance(dates, (str, bytes)):
        raise ValueError("dates must be a sequence, not a single string.")
    try:
        index = pd.DatetimeIndex(dates, name="date")
    except (TypeError, ValueError) as exc:
        raise ValueError("dates must contain valid daily dates.") from exc
    if (
        len(index) != n_days or index.hasnans or index.tz is not None
        or not index.is_unique or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError("Provide one unique, sorted, timezone-naive midnight date per test day.")
    return index
