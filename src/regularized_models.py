"""Experimental regularized presence ranking with a bounded training history."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import NotFittedError

from .count_features import COUNT_TARGET
from .count_models import _next_day, _validate_dataset
from .data_loader import MATRIX_WIDTH


ProbabilityTopK = list[tuple[str, float]]


class RegularizedLGBMPredictor:
    """Estimate next-day presence using normalized features and recent history.

    The input is build_count_feature_dataset output. Training converts each
    next-day count to a presence indicator; count 1 and count 3 both mean 1.
    Returned scores are uncalibrated probabilities of at least one appearance,
    not expected counts or a claim of improved profitability.

    History is bounded by CALENDAR target dates, not candidate rows or observed
    days. For final training label L and window W, retain labels L-W+1..L.
    All 100 candidates for each retained date remain together.
    """

    def __init__(self, history_days: int = 1095, **model_params: Any) -> None:
        """Use a three-year window and small regularized trees by default."""
        if (
            isinstance(history_days, (bool, np.bool_))
            or not isinstance(history_days, (int, np.integer)) or history_days < 1
        ):
            raise ValueError("history_days must be a positive integer.")
        self.history_days = int(history_days)
        self.params: dict[str, Any] = {
            "objective": "binary", "n_estimators": 100, "learning_rate": 0.03,
            "num_leaves": 7, "min_child_samples": 200, "reg_lambda": 20.0,
            "random_state": 42, "n_jobs": 1, "verbosity": -1,
            "deterministic": True, "force_col_wise": True,
        }
        self.params.update(model_params)
        if self.params["objective"] != "binary":
            raise ValueError("RegularizedLGBMPredictor requires objective='binary'.")
        self.model_: LGBMClassifier | None = None
        self.feature_columns_: list[str] = []
        self.trained_through_: pd.Timestamp | int | None = None
        self.training_label_start_: pd.Timestamp | int | None = None
        self.prediction_date_: pd.Timestamp | int | None = None

    def fit(self, dataset: pd.DataFrame) -> RegularizedLGBMPredictor:
        """Exclude missing labels, restrict the calendar window, and fit presence.

        Validate the complete count-target contract before binarizing so an
        invalid count or partially labeled day cannot silently become absence.
        The count target never becomes a model input. No test-set tuning occurs.
        """
        data = _validate_dataset(dataset, require_target=True)
        labeled = data.loc[data[COUNT_TARGET].notna()]
        if labeled.empty:
            raise ValueError("At least one fully labeled training date is required.")
        origins = labeled.index.get_level_values(0)
        last_label = _next_day(origins.max())
        if isinstance(last_label, pd.Timestamp):
            targets = origins + pd.Timedelta(days=1)
            first_allowed = last_label + pd.Timedelta(days=1 - self.history_days)
        else:
            targets = origins + 1
            first_allowed = last_label + 1 - self.history_days
        train = labeled.loc[targets >= first_allowed]
        features = [column for column in train if column != COUNT_TARGET]
        y = train[COUNT_TARGET].to_numpy(dtype=np.int64) > 0
        model = LGBMClassifier(**self.params)
        model.fit(train[features].astype(float), y.astype(np.uint8), categorical_feature=["number_id"])
        self.model_ = model
        self.feature_columns_ = features
        self.trained_through_ = last_label
        self.training_label_start_ = _next_day(train.index.get_level_values(0).min())
        self.prediction_date_ = None
        return self

    def predict_proba(self, dataset: pd.DataFrame) -> pd.Series:
        """Return uncalibrated P(at least one appearance at t+1) for each row.

        Supplied feature dates must be on or after the latest training label.
        Predictions keep canonical date/number order and do not sum to one.
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
        probabilities = np.asarray(self.model_.predict_proba(data[self.feature_columns_].astype(float)), dtype=float)
        if probabilities.shape != (len(data), 2):
            raise ValueError("The fitted model returned invalid binary probabilities.")
        positive = probabilities[:, 1]
        if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
            raise ValueError("The fitted model returned probabilities outside [0, 1].")
        return pd.Series(positive, index=data.index, name="probability")

    def predict(self, dataset: pd.DataFrame, top_k: int = 5) -> ProbabilityTopK:
        """Rank the latest feature date; ties use the ascending number order."""
        if (
            isinstance(top_k, (bool, np.bool_))
            or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= MATRIX_WIDTH
        ):
            raise ValueError("top_k must be an integer in 1..100.")
        data = _validate_dataset(dataset, require_target=False)
        latest = data.index.get_level_values(0).max()
        latest_rows = data.loc[data.index.get_level_values(0) == latest]
        scores = self.predict_proba(latest_rows).droplevel(0)
        top = scores.sort_values(ascending=False, kind="stable").iloc[:top_k]
        self.prediction_date_ = _next_day(latest)
        return [(str(number), float(score)) for number, score in top.items()]

    def train_predict(self, dataset: pd.DataFrame, top_k: int = 5) -> ProbabilityTopK:
        """Fit labels and forecast the last date of include_unlabeled=True data."""
        return self.fit(dataset).predict(dataset, top_k)
