"""Regularized LightGBM models for next-day ordered-prize features."""

from __future__ import annotations

from typing import Any, Literal

from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .count_features import COUNT_TARGET
from .count_models import _validate_dataset
from .prize_features import EXTRA_FEATURES
from .regularized_models import RegularizedLGBMPredictor


class PrizeLGBMPredictor:
    """Fit presence probabilities or expected counts with a 1095-day window.

    Default tree parameters match RegularizedLGBMPredictor so a binary-model
    comparison isolates the addition of prize features. No tuning/early stop
    uses the evaluation period. Presence probabilities remain uncalibrated.
    """

    def __init__(self, objective: Literal["binary", "poisson"] = "binary",
                 history_days: int = 1095, **model_params: Any) -> None:
        if objective not in ("binary", "poisson"):
            raise ValueError("objective must be binary or poisson.")
        prototype = RegularizedLGBMPredictor(history_days=history_days)
        self.history_days = prototype.history_days
        self.objective = objective
        self.params = {**prototype.params, **model_params, "objective": objective}
        self.model_: LGBMClassifier | LGBMRegressor | None = None
        self.feature_columns_: list[str] = []
        self.trained_through_: pd.Timestamp | None = None
        self.training_label_start_: pd.Timestamp | None = None

    @staticmethod
    def _validate(data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame) or not data.columns.is_unique or not set(EXTRA_FEATURES).issubset(data.columns):
            raise ValueError("Missing ordered-prize features; use build_prize_feature_dataset.")
        # The existing contract still rejects unrecognized columns/target leaks.
        _validate_dataset(data.drop(columns=list(EXTRA_FEATURES)), require_target=require_target)
        if not isinstance(data.index.get_level_values(0), pd.DatetimeIndex):
            raise ValueError("Prize models require calendar date indices.")
        for name in EXTRA_FEATURES:
            if not pd.api.types.is_numeric_dtype(data[name].dtype):
                raise ValueError(f"Prize feature {name} must be numeric.")
        values = data.loc[:, list(EXTRA_FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Prize features must be finite.")
        return data.sort_index()

    def fit(self, dataset: pd.DataFrame) -> PrizeLGBMPredictor:
        """Fit complete known labels from the trailing calendar window."""
        data = self._validate(dataset, require_target=True)
        labeled = data.loc[data[COUNT_TARGET].notna()]
        if labeled.empty:
            raise ValueError("At least one fully labeled date is required.")
        targets = labeled.index.get_level_values(0) + pd.Timedelta(days=1)
        end = targets.max()
        train = labeled.loc[targets >= end - pd.Timedelta(days=self.history_days - 1)]
        features = [column for column in train if column != COUNT_TARGET]
        y = train[COUNT_TARGET].to_numpy(dtype=np.int64)
        constructor = LGBMClassifier if self.objective == "binary" else LGBMRegressor
        fitted = constructor(**self.params)
        fitted.fit(train[features].astype(float), (y > 0).astype(np.uint8) if self.objective == "binary" else y,
                   categorical_feature=["number_id"])
        self.model_, self.feature_columns_ = fitted, features
        self.trained_through_ = end
        self.training_label_start_ = train.index.get_level_values(0).min() + pd.Timedelta(days=1)
        return self

    def predict_scores(self, dataset: pd.DataFrame) -> pd.Series:
        """Return presence probability or expected count, preserving row indices."""
        if self.model_ is None:
            raise NotFittedError("Fit before predicting.")
        data = self._validate(dataset, require_target=False)
        if data.index.get_level_values(0).min() < self.trained_through_:
            raise ValueError("Prediction features predate the latest training label.")
        features = [column for column in data if column != COUNT_TARGET]
        if set(features) != set(self.feature_columns_):
            raise ValueError("Prediction features must match training features.")
        x = data[self.feature_columns_].astype(float)
        if self.objective == "binary":
            scores = self.model_.predict_proba(x)[:, 1]
            name = "uncalibrated_probability"
        else:
            raw = self.model_.predict(x).reshape(-1, 100)
            if not np.isfinite(raw).all() or (raw < 0).any() or (raw.max(axis=1) <= 0).any():
                raise ValueError("Invalid expected counts.")
            scaled = raw / raw.max(axis=1, keepdims=True)
            scores = (27 * scaled / scaled.sum(axis=1, keepdims=True)).ravel()
            name = "expected_count"
        if not np.isfinite(scores).all() or (scores < 0).any() or (self.objective == "binary" and (scores > 1).any()):
            raise ValueError("Invalid model scores.")
        return pd.Series(scores, index=data.index, name=name)

    def predict(self, dataset: pd.DataFrame, top_k: int = 5) -> list[tuple[str, float]]:
        """Rank the latest day; ties use ascending candidate number."""
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer in 1..100.")
        latest = dataset.index.get_level_values(0).max()
        scores = self.predict_scores(dataset.loc[dataset.index.get_level_values(0) == latest]).droplevel(0)
        return [(str(number), float(value)) for number, value in scores.sort_values(ascending=False, kind="stable").iloc[:top_k].items()]
