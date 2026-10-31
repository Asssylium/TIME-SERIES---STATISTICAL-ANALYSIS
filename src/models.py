"""Next-day ranking models and chronological validation for XSMB.

Forecasts are made AFTER the draw on feature date t, for t + 1. Consequently,
frequency_today/present_today are lag-1 predictors relative to the target.
Rolling features must be built causally using src.features.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import TimeSeriesSplit

from .data_loader import MATRIX_WIDTH, NUMBER_COLUMNS
from .features import MatrixInput, build_feature_dataset, compute_rolling_frequency


TARGET = "target_next_day"
TopK = list[tuple[str, float]]


class BaselinePredictor:
    """Rank by total draw frequency in the most recent 30 calendar days.

    Scores are COUNTS, not probabilities of daily presence. Within-day repeats
    count individually. Missing dates contribute no observations, and warm-up
    uses all available history. NumPy inputs assume consecutive daily rows.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = _positive_integer(window, "window")
        self.scores_: pd.Series | None = None
        self.prediction_date_: pd.Timestamp | int | None = None

    def fit(self, frequency_matrix: MatrixInput) -> BaselinePredictor:
        """Use history ending at t to prepare a forecast for t + 1."""
        rolling = compute_rolling_frequency(frequency_matrix, windows=[self.window])[self.window]
        self.scores_ = rolling.iloc[-1].astype(float).rename("score")
        self.prediction_date_ = _next_day(rolling.index[-1])
        return self

    def predict_scores(self) -> pd.Series:
        """Return all 100 frequency scores, indexed by two-digit numbers."""
        if self.scores_ is None:
            raise NotFittedError("Call fit(frequency_matrix) before predicting.")
        return self.scores_.copy()

    def predict(self, top_k: int = 10) -> TopK:
        """Return (number, count) pairs in descending score order."""
        return _top_k(self.predict_scores(), top_k)

    def train_predict(self, frequency_matrix: MatrixInput, top_k: int = 10) -> TopK:
        """Convenience method: fit the baseline and return its next-day Top-K."""
        return self.fit(frequency_matrix).predict(top_k)


class LGBMPredictor:
    """Fit one binary LightGBM model across all (date, number) training rows.

    The positive label means at least one appearance on the next day. Returned
    probabilities are independent per-number estimates and do NOT sum to one.
    They are uncalibrated model estimates. Defaults do not rebalance classes.
    number_id is categorical, so its numeric ordering does not imply affinity.
    """

    def __init__(self, **model_params: Any) -> None:
        """Override LightGBM parameters, e.g. n_estimators or learning_rate.

        Fixed tree counts avoid tuning/early stopping on the outer test fold.
        Hyperparameter tuning, if needed, must use a separate chronological
        validation set within the training period.
        """
        self.params: dict[str, Any] = {
            "objective": "binary", "n_estimators": 200, "learning_rate": 0.05,
            "num_leaves": 15, "min_child_samples": 50, "reg_lambda": 1.0,
            "random_state": 42, "n_jobs": 1, "verbosity": -1,
            "deterministic": True, "force_col_wise": True,
        }
        self.params.update(model_params)
        if self.params["objective"] != "binary":
            raise ValueError("LGBMPredictor requires objective='binary'.")
        self.model_: LGBMClassifier | None = None
        self.feature_columns_: list[str] = []
        self.trained_through_: pd.Timestamp | int | None = None
        self.prediction_date_: pd.Timestamp | int | None = None

    def fit(self, dataset: pd.DataFrame) -> LGBMPredictor:
        """Fit labeled rows produced by build_feature_dataset.

        Entire unlabeled dates are excluded automatically. A partially labeled
        date is rejected. The target is removed before fitting. No train/test
        shuffle or random validation split is performed.
        """
        data = _validate_dataset(dataset, require_target=True)
        train = data.loc[data[TARGET].notna()]
        if train.empty:
            raise ValueError("At least one fully labeled training date is required.")
        features = [column for column in train.columns if column != TARGET]
        labels = train[TARGET].to_numpy(dtype=np.uint8)
        if np.unique(labels).size != 2:
            raise ValueError("Training labels must contain both absence (0) and presence (1).")
        # Fit locally first: a failed refit must not leave a half-updated model.
        model = LGBMClassifier(**self.params)
        model.fit(train[features].astype(float), labels, categorical_feature=["number_id"])
        self.model_ = model
        self.feature_columns_ = features
        self.trained_through_ = _next_day(train.index.get_level_values(0).max())
        self.prediction_date_ = None
        return self

    def predict_proba(self, dataset: pd.DataFrame) -> pd.Series:
        """Return P(appearance on t+1) for every supplied (t, number) row.

        All feature dates must be on/after the latest training LABEL date.
        This rejects accidental in-sample/backdated forecasts. A model trained
        through the completed draw at t may forecast using features at t.
        For historical estimates, use cross_validate instead of a final model.
        """
        if self.model_ is None or self.trained_through_ is None:
            raise NotFittedError("Call fit(training_dataset) before predicting.")
        data = _validate_dataset(dataset, require_target=False)
        first_date = data.index.get_level_values(0).min()
        if isinstance(first_date, pd.Timestamp) != isinstance(self.trained_through_, pd.Timestamp):
            raise ValueError("Training and prediction must use the same date/index type.")
        if first_date < self.trained_through_:
            raise ValueError("Prediction features predate the latest training label; use cross_validate.")
        features = [column for column in data.columns if column != TARGET]
        if set(features) != set(self.feature_columns_):
            raise ValueError("Prediction feature columns must match training feature columns.")
        probabilities = self.model_.predict_proba(data[self.feature_columns_].astype(float))[:, 1]
        return pd.Series(probabilities, index=data.index, name="probability")

    def predict(self, dataset: pd.DataFrame, top_k: int = 10) -> TopK:
        """Forecast after the latest supplied feature date; return (number, probability)."""
        data = _validate_dataset(dataset, require_target=False)
        latest = data.index.get_level_values(0).max()
        latest_rows = data.loc[data.index.get_level_values(0) == latest]
        scores = self.predict_proba(latest_rows).droplevel(0)
        result = _top_k(scores, top_k)
        self.prediction_date_ = _next_day(latest)
        return result

    def train_predict(self, dataset: pd.DataFrame, top_k: int = 10) -> TopK:
        """Fit labels and forecast the latest day using include_unlabeled=True data."""
        return self.fit(dataset).predict(dataset, top_k)


@dataclass(frozen=True)
class CrossValidationResult:
    """Out-of-fold predictions and one row of ranking metrics per fold.

    predictions keeps the (feature_date, number) index and includes score,
    target_next_day, selected_top_k, and fold. Scores are baseline counts or
    LightGBM probabilities, respectively. fold_metrics includes chronological
    boundaries, daily-averaged Precision@K, Recall@K, Hit Rate@K and mean hits.
    Recall counts DISTINCT actual numbers; hit rate means at least one hit/day.
    """

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame


def cross_validate(
    predictor: BaselinePredictor | LGBMPredictor,
    frequency_matrix: MatrixInput,
    binary_matrix: MatrixInput,
    *,
    n_splits: int = 5,
    top_k: int = 10,
    gap: int = 1,
    test_size: int | None = None,
    max_train_size: int | None = None,
) -> CrossValidationResult:
    """Compare next-day rankings using TimeSeriesSplit on UNIQUE feature dates.

    All 100 candidates for a date stay together. gap >= 1 purges at least one
    feature date so every training label precedes the first validation feature
    date. test_size/max_train_size/gap count observed, labeled feature dates;
    calendar gaps can make folds cover different calendar durations.

    Each fold fits a fresh LightGBM model once. Validation is sequential
    next-day forecasting: features at each validation date may use completed
    draws earlier in that validation period. They never use t+1 or later data.
    Baseline scores similarly use only history through each forecast origin.
    The caller's predictor is neither fitted nor modified. No test-fold early
    stopping or hyperparameter selection occurs here.
    """
    if not isinstance(predictor, (BaselinePredictor, LGBMPredictor)):
        raise TypeError("predictor must be BaselinePredictor or LGBMPredictor.")
    _validate_top_k(top_k)
    _positive_integer(gap, "gap")
    n_splits = _positive_integer(n_splits, "n_splits")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if test_size is not None:
        _positive_integer(test_size, "test_size")
    if max_train_size is not None:
        _positive_integer(max_train_size, "max_train_size")
    data = _validate_dataset(
        build_feature_dataset(frequency_matrix, binary_matrix), require_target=True
    )
    dates = data.index.get_level_values(0).unique()
    splitter = TimeSeriesSplit(
        n_splits=n_splits, gap=gap, test_size=test_size, max_train_size=max_train_size
    )
    baseline_scores = None
    if isinstance(predictor, BaselinePredictor):
        baseline_scores = compute_rolling_frequency(
            frequency_matrix, windows=[predictor.window]
        )[predictor.window]
    outputs: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for fold, (train_days, validation_days) in enumerate(splitter.split(dates), start=1):
        # Date blocks are complete and sorted: mapping date positions to rows
        # never cuts a day into separate train/validation candidate subsets.
        train_rows = (train_days[:, None] * MATRIX_WIDTH + np.arange(MATRIX_WIDTH)).ravel()
        validation_rows = (validation_days[:, None] * MATRIX_WIDTH + np.arange(MATRIX_WIDTH)).ravel()
        train, validation = data.iloc[train_rows], data.iloc[validation_rows]
        label_end = _next_day(dates[train_days[-1]])
        if label_end >= dates[validation_days[0]]:
            raise ValueError("Training labels overlap validation; increase gap.")
        if isinstance(predictor, LGBMPredictor):
            model = LGBMPredictor(**predictor.params).fit(train)
            scores = model.predict_proba(validation).to_numpy().reshape(-1, MATRIX_WIDTH)
        else:
            assert baseline_scores is not None
            scores = baseline_scores.loc[dates[validation_days]].to_numpy(dtype=float)
        actual = validation[TARGET].to_numpy(dtype=np.uint8).reshape(-1, MATRIX_WIDTH)
        selected = np.zeros_like(actual, dtype=bool)
        order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        np.put_along_axis(selected, order, True, axis=1)
        hits = (actual * selected).sum(axis=1)
        actual_count = actual.sum(axis=1)
        recall = np.divide(hits, actual_count, out=np.zeros(len(hits)), where=actual_count > 0)
        output = pd.DataFrame({
            "fold": fold, "score": scores.ravel(), TARGET: actual.ravel(),
            "selected_top_k": selected.ravel(),
        }, index=validation.index)
        outputs.append(output)
        metrics.append({
            "fold": fold, "train_start": dates[train_days[0]],
            "train_end": dates[train_days[-1]], "train_label_end": label_end,
            "validation_start": dates[validation_days[0]],
            "validation_end": dates[validation_days[-1]],
            "n_train_days": len(train_days), "n_validation_days": len(validation_days),
            "top_k": top_k, "precision_at_k": float(np.mean(hits / top_k)),
            "recall_at_k": float(np.mean(recall)), "hit_rate_at_k": float(np.mean(hits > 0)),
            "mean_hits": float(np.mean(hits)),
        })
    return CrossValidationResult(pd.concat(outputs), pd.DataFrame(metrics).set_index("fold"))


def _validate_dataset(dataset: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Validate the feature-builder contract before training or scoring."""
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("Expected a nonempty DataFrame from build_feature_dataset.")
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
    if set(dataset.index.get_level_values(1)) != set(NUMBER_COLUMNS):
        raise ValueError("Every date must contain all numbers '00'..'99'.")
    if not dataset.groupby(level=0).size().eq(MATRIX_WIDTH).all():
        raise ValueError("Every date must contain exactly 100 unique candidate rows.")
    if require_target and TARGET not in dataset:
        raise ValueError(f"Missing target column: {TARGET}")
    base_features = {
        "number_id", "frequency_today", "present_today", "gap_days", "seen_before",
        "history_days", "co_occurrence_total",
    }
    features = [column for column in dataset.columns if column != TARGET]
    if not base_features.issubset(features):
        raise ValueError("Missing features; supply the output of build_feature_dataset.")
    for column in features:
        if column not in base_features and not re.fullmatch(
            r"(?:rolling_frequency|observed_days)_[1-9][0-9]*d", str(column)
        ):
            raise ValueError(f"Unexpected feature {column!r}; only causal builder features are accepted.")
        dtype = dataset[column].dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
            raise ValueError(f"Feature {column!r} must be real numeric data.")
    if not np.isfinite(dataset[features].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Features must be finite; use -1 for unseen gaps.")
    if not np.array_equal(dataset.number_id.to_numpy(), dataset.index.get_level_values(1).astype(int)):
        raise ValueError("number_id must match the number in the row index.")
    if TARGET in dataset:
        labels = dataset[TARGET]
        if not labels.dropna().isin([0, 1]).all():
            raise ValueError("Targets must be 0, 1, or missing for an unlabeled date.")
        counts = labels.notna().groupby(level=0).sum()
        if not counts.isin([0, MATRIX_WIDTH]).all():
            raise ValueError("Targets must be present for all 100 numbers or absent for the entire date.")
    return dataset.sort_index()


def _next_day(value: pd.Timestamp | int) -> pd.Timestamp | int:
    """Advance a calendar date or consecutive-array day index by one day."""
    return value + pd.Timedelta(days=1) if isinstance(value, pd.Timestamp) else int(value) + 1


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_top_k(top_k: int) -> None:
    if _positive_integer(top_k, "top_k") > MATRIX_WIDTH:
        raise ValueError("top_k must be between 1 and 100.")


def _top_k(scores: pd.Series, top_k: int) -> TopK:
    """Break equal-score ties deterministically by ascending two-digit number."""
    _validate_top_k(top_k)
    ordered = scores.reindex(NUMBER_COLUMNS).sort_values(ascending=False, kind="stable").iloc[:top_k]
    return [(str(number), float(score)) for number, score in ordered.items()]
