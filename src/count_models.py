"""Experimental next-day expected-count model for per-occurrence payouts.

Each score is an expected NUMBER OF OCCURRENCES, not a probability. Forecasts
are made after the draw at feature date t and target the calendar day t + 1.
Matching the payout target does not establish better rankings or profitability;
compare this model against frozen chronological baselines before deployment.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.exceptions import NotFittedError

from .count_features import COUNT_TARGET
from .data_loader import DRAW_COUNT, MATRIX_WIDTH, NUMBER_COLUMNS


CountTopK = list[tuple[str, float]]
_BASE_FEATURES = {
    "number_id", "frequency_today", "present_today", "gap_days", "seen_before",
    "co_occurrence_per_day",
}


class CountLGBMPredictor:
    """Poisson LightGBM regression on next-day occurrence counts, including repeats.

    All 100 candidates share a model; number_id is categorical. Predictions for
    a day are rescaled to sum to the known total of 27 draws. This enforces the
    count constraint without changing that day's ranking. The Poisson objective
    estimates a conditional mean; it does not assert independent lottery draws
    or justify converting the score into a probability without validation.
    """

    def __init__(self, **model_params: Any) -> None:
        """Use a small regularized model; tune only inside training periods."""
        self.params: dict[str, Any] = {
            "objective": "poisson", "n_estimators": 100, "learning_rate": 0.03,
            "num_leaves": 7, "min_child_samples": 200, "reg_lambda": 20.0,
            "random_state": 42, "n_jobs": 1, "verbosity": -1,
            "deterministic": True, "force_col_wise": True,
        }
        self.params.update(model_params)
        if self.params["objective"] != "poisson":
            raise ValueError("CountLGBMPredictor requires objective='poisson'.")
        self.model_: LGBMRegressor | None = None
        self.feature_columns_: list[str] = []
        self.trained_through_: pd.Timestamp | int | None = None
        self.prediction_date_: pd.Timestamp | int | None = None

    def fit(self, dataset: pd.DataFrame) -> CountLGBMPredictor:
        """Fit complete labeled dates; drop whole unlabeled dates automatically.

        Every labeled date must contain 100 integer targets in 0..27 whose sum
        is exactly 27. The next-day count label is excluded from model inputs.
        Fitting never shuffles dates or tunes on a supplied test period.
        """
        data = _validate_dataset(dataset, require_target=True)
        train = data.loc[data[COUNT_TARGET].notna()]
        if train.empty:
            raise ValueError("At least one fully labeled training date is required.")
        features = [column for column in train if column != COUNT_TARGET]
        labels = train[COUNT_TARGET].to_numpy(dtype=np.int64)
        model = LGBMRegressor(**self.params)
        model.fit(train[features].astype(float), labels, categorical_feature=["number_id"])
        # Commit state only after a successful fit.
        self.model_ = model
        self.feature_columns_ = features
        self.trained_through_ = _next_day(train.index.get_level_values(0).max())
        self.prediction_date_ = None
        return self

    def predict_expected_counts(self, dataset: pd.DataFrame) -> pd.Series:
        """Return nonnegative E[count at t+1] for each supplied (t, number) row.

        All supplied origins must be on or after the latest training LABEL
        date. A model trained through the completed draw at t can forecast
        t+1 using features at t. Scores sum to 27 separately for every date.
        Return rows in canonical date/number order; never mutate the input.
        """
        if self.model_ is None or self.trained_through_ is None:
            raise NotFittedError("Call fit(training_dataset) before predicting.")
        data = _validate_dataset(dataset, require_target=False)
        first = data.index.get_level_values(0).min()
        if isinstance(first, pd.Timestamp) != isinstance(self.trained_through_, pd.Timestamp):
            raise ValueError("Training and prediction must use the same date/index type.")
        if first < self.trained_through_:
            raise ValueError("Prediction features predate the latest training label.")
        features = [column for column in data if column != COUNT_TARGET]
        if set(features) != set(self.feature_columns_):
            raise ValueError("Prediction feature columns must match training feature columns.")
        raw = np.asarray(self.model_.predict(data[self.feature_columns_].astype(float)), dtype=float)
        if raw.shape != (len(data),) or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("The fitted model returned invalid expected counts.")
        counts = raw.reshape(-1, MATRIX_WIDTH)
        # Scale by each day's maximum first, avoiding overflow in the sum.
        maximum = counts.max(axis=1, keepdims=True)
        if (maximum <= 0).any():
            raise ValueError("The fitted model returned a zero total expected count.")
        scaled = counts / maximum
        expected = DRAW_COUNT * scaled / scaled.sum(axis=1, keepdims=True)
        return pd.Series(expected.ravel(), index=data.index, name="expected_count")

    def predict(self, dataset: pd.DataFrame, top_k: int = 5) -> CountTopK:
        """Rank the latest origin and return (number, expected occurrences)."""
        if (
            isinstance(top_k, (bool, np.bool_))
            or not isinstance(top_k, (int, np.integer))
            or not 1 <= top_k <= MATRIX_WIDTH
        ):
            raise ValueError("top_k must be an integer in 1..100.")
        data = _validate_dataset(dataset, require_target=False)
        latest = data.index.get_level_values(0).max()
        latest_rows = data.loc[data.index.get_level_values(0) == latest]
        scores = self.predict_expected_counts(latest_rows).droplevel(0)
        ranked = scores.sort_values(ascending=False, kind="stable").iloc[:top_k]
        self.prediction_date_ = _next_day(latest)
        return [(str(number), float(score)) for number, score in ranked.items()]

    def train_predict(self, dataset: pd.DataFrame, top_k: int = 5) -> CountTopK:
        """Fit labels and forecast the last date of include_unlabeled=True data."""
        return self.fit(dataset).predict(dataset, top_k)


def _validate_dataset(dataset: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Validate the causal count-feature contract and whole-day target totals."""
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("Expected a nonempty DataFrame from build_count_feature_dataset.")
    if not isinstance(dataset.index, pd.MultiIndex) or dataset.index.nlevels != 2:
        raise ValueError("Dataset must have a (date/day_index, number) MultiIndex.")
    if not dataset.index.is_unique or not dataset.columns.is_unique:
        raise ValueError("Dataset indices and columns must be unique.")
    origins = dataset.index.get_level_values(0)
    if isinstance(origins, pd.DatetimeIndex):
        if origins.hasnans or origins.tz is not None or not origins.equals(origins.normalize()):
            raise ValueError("Feature dates must be timezone-naive midnight dates.")
    elif not pd.api.types.is_integer_dtype(origins.dtype) or (origins < 0).any():
        raise ValueError("Feature dates must be datetime dates or nonnegative integer day indices.")
    numbers = dataset.index.get_level_values(1)
    if set(numbers) != set(NUMBER_COLUMNS) or not dataset.groupby(level=0).size().eq(MATRIX_WIDTH).all():
        raise ValueError("Every date must contain all 100 unique numbers '00'..'99'.")
    if require_target and COUNT_TARGET not in dataset:
        raise ValueError(f"Missing target column: {COUNT_TARGET}")
    features = [column for column in dataset if column != COUNT_TARGET]
    if not _BASE_FEATURES.issubset(features):
        raise ValueError("Missing features; use build_count_feature_dataset.")
    for column in features:
        if column not in _BASE_FEATURES | {"gap_observation_complete"} and not re.fullmatch(
            r"(?:rolling_rate|observed_days)_[1-9][0-9]*d", str(column)
        ):
            raise ValueError(f"Unexpected feature {column!r}; only causal count features are accepted.")
        dtype = dataset[column].dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
            raise ValueError(f"Feature {column!r} must be real numeric data.")
    if not np.isfinite(dataset[features].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Features must be finite; use -1 for unseen gaps.")
    if not np.array_equal(dataset.number_id.to_numpy(), numbers.astype(int)):
        raise ValueError("number_id must match the number in the row index.")
    if COUNT_TARGET in dataset:
        targets = dataset[COUNT_TARGET]
        if not pd.api.types.is_numeric_dtype(targets.dtype) or pd.api.types.is_complex_dtype(targets.dtype):
            raise ValueError("Count targets must be numeric integers or missing.")
        labeled = targets.dropna().to_numpy(dtype=float)
        if (
            not np.isfinite(labeled).all() or (labeled < 0).any() or (labeled > DRAW_COUNT).any()
            or (labeled != np.floor(labeled)).any()
        ):
            raise ValueError("Count targets must be integers in 0..27 or missing.")
        labeled_per_day = targets.notna().groupby(level=0).sum()
        if not labeled_per_day.isin([0, MATRIX_WIDTH]).all():
            raise ValueError("Targets must be labeled or missing for the entire date.")
        totals = targets.groupby(level=0).sum(min_count=1)
        if not totals.loc[labeled_per_day == MATRIX_WIDTH].eq(DRAW_COUNT).all():
            raise ValueError("Labeled target counts must sum to exactly 27 per date.")
    return dataset.sort_index()


def _next_day(value: pd.Timestamp | int) -> pd.Timestamp | int:
    """Advance a calendar date or array day index by exactly one day."""
    return value + pd.Timedelta(days=1) if isinstance(value, pd.Timestamp) else int(value) + 1
