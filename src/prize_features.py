"""Causal next-day features preserving all 27 ordered prize positions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .count_features import COUNT_TARGET, build_count_feature_dataset
from .data_loader import DataValidationError, NUMBER_COLUMNS, XSMBDataLoader


RAW_PRIZE_URL = "https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv"
# Group and within-group temporal order were explicitly confirmed by the user.
PRIZE_WIDTHS = {"prize1": 5}
for _group, _count, _width in [(2, 2, 5), (3, 6, 5), (4, 4, 4), (5, 6, 4), (6, 3, 3), (7, 4, 2)]:
    PRIZE_WIDTHS.update({f"prize{_group}_{i}": _width for i in range(1, _count + 1)})
PRIZE_WIDTHS["special"] = 5
PRIZE_COLUMNS = tuple(PRIZE_WIDTHS)
SLOT_FEATURES = ("tail", "match", "tens_match", "units_match", "prefix_digit_hits", "previous_day_match", "rate_30d")
EXTRA_FEATURES = ("candidate_tens", "candidate_units", *(
    f"slot_{position:02d}_{suffix}" for position in range(1, 28) for suffix in SLOT_FEATURES
))


def normalize_prizes(raw: pd.DataFrame) -> pd.DataFrame:
    """Return date-indexed, fixed-width prize strings in confirmed draw order.

    Require exactly one integer string per prize column and all 27 positions.
    Restore leading zeros using each prize's known width. No input mutation,
    missing-value imputation, date duplication or invented calendar rows.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty or not raw.columns.is_unique:
        raise DataValidationError("Expected a nonempty raw prize table with unique columns.")
    # CSV exports occasionally include a UTF-8 BOM or whitespace in the header.
    # Normalize those harmless transport differences before enforcing the schema.
    raw = raw.copy()
    raw.columns = [str(column).lstrip("\ufeff").strip() for column in raw.columns]
    if not raw.columns.is_unique:
        raise DataValidationError("Raw prize table contains duplicate column names after header cleanup.")
    expected_columns = {"date", *PRIZE_COLUMNS}
    actual_columns = set(raw.columns)
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        extra = sorted(actual_columns - expected_columns)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        raise DataValidationError(
            "Raw prize table must contain date and exactly the 27 named prize positions" + suffix + "."
        )
    normalized = raw.loc[:, ["date", *PRIZE_COLUMNS]].copy()
    for column, width in PRIZE_WIDTHS.items():
        text = normalized[column].astype("string").str.strip()
        if not text.str.fullmatch(r"[0-9]{1," + str(width) + "}").fillna(False).all():
            raise DataValidationError(f"Invalid prize value or digit width in {column}.")
        normalized[column] = text.str.zfill(width)
    # Reuse the production date/27-results validation before indexing.
    XSMBDataLoader(prize_columns=PRIZE_COLUMNS).preprocess_and_build_matrices(normalized)
    normalized["date"] = pd.to_datetime(normalized.date, format="%Y-%m-%d")
    return normalized.set_index("date").sort_index()


def build_prize_feature_dataset(
    raw: pd.DataFrame, *, include_unlabeled: bool = False,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build (date, candidate 00..99) rows forecasting the next calendar day.

    Add candidate digits and seven features for each ordered prize position:
    today's two-digit ending, exact/tens/units matches to the candidate, count
    of leading prize digits matching either candidate digit, previous CALENDAR
    day's match (-1 if missing), and the preceding 30-calendar-day ending rate.
    Rates include today and divide by observed days; they never see tomorrow.
    Full prize leading zeros participate in the digit features. start only
    trims output rows; rolling/gap features retain their preceding history.
    COUNT_TARGET remains the exact next-day total occurrence count per number.
    The rows are available only after ALL 27 draws for their feature date.
    """
    prizes = normalize_prizes(raw)
    frequency, binary = XSMBDataLoader().preprocess_and_build_matrices(prizes.reset_index())
    base = build_count_feature_dataset(frequency, binary, include_unlabeled=True)
    if start is not None:
        base = base.loc[base.index.get_level_values(0) >= pd.Timestamp(start)]
    if base.empty:
        raise ValueError("No feature dates at or after start.")
    dates = pd.DatetimeIndex(base.index.get_level_values(0).unique())
    candidates = np.arange(100)
    tens, units = candidates // 10, candidates % 10
    extras: dict[str, np.ndarray] = {
        "candidate_tens": np.tile(tens.astype(np.uint8), len(dates)),
        "candidate_units": np.tile(units.astype(np.uint8), len(dates)),
    }
    for position, column in enumerate(PRIZE_COLUMNS, 1):
        name = f"slot_{position:02d}_"
        full_text = prizes.loc[dates, column]
        endings_all = prizes[column].str[-2:].astype(int)
        endings = endings_all.loc[dates].to_numpy()
        extras[name + "tail"] = np.repeat(endings.astype(np.uint8), 100)
        extras[name + "match"] = (endings[:, None] == candidates).astype(np.uint8).ravel()
        extras[name + "tens_match"] = (endings[:, None] // 10 == tens).astype(np.uint8).ravel()
        extras[name + "units_match"] = (endings[:, None] % 10 == units).astype(np.uint8).ravel()
        prefix_hits = np.zeros((len(dates), 100), dtype=np.uint8)
        for digit_position in range(PRIZE_WIDTHS[column] - 2):
            digit = full_text.str[digit_position].astype(int).to_numpy()[:, None]
            prefix_hits += ((digit == tens) | (digit == units)).astype(np.uint8)
        extras[name + "prefix_digit_hits"] = prefix_hits.ravel()
        previous = endings_all.reindex(dates - pd.Timedelta(days=1)).to_numpy(dtype=float)
        previous_match = np.where(np.isnan(previous[:, None]), -1, previous[:, None] == candidates)
        extras[name + "previous_day_match"] = previous_match.astype(np.int8).ravel()
        indicator = pd.DataFrame(endings_all.to_numpy()[:, None] == candidates,
                                 index=prizes.index, columns=NUMBER_COLUMNS)
        extras[name + "rate_30d"] = indicator.rolling("30D", min_periods=1).mean().loc[dates].to_numpy(dtype=np.float32).ravel()
    extra = pd.DataFrame(extras, index=base.index)
    if tuple(extra.columns) != EXTRA_FEATURES:
        raise AssertionError("Unexpected prize feature order.")
    result = pd.concat([base, extra], axis=1)
    return result if include_unlabeled else result.loc[result[COUNT_TARGET].notna()].copy()
