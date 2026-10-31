"""Bounded probability shrinkage fitted on chronological calibration forecasts.

The caller owns temporal splits: fit on predictions produced without their
labels, then freeze the fitted map before evaluating later dates. This module
does not split dates or promise that calibration transfers to a future period.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ProbabilityShrinkageCalibrator:
    """Minimize calibration Brier loss with a fixed, supplied probability anchor.

    The map is ``anchor + weight_ * (probability - anchor)``, with ``weight_``
    constrained to [0, 1]. A positive weight preserves ordering and existing
    ties; zero gives an unranked constant forecast when there is no fitted
    positive association. It never reverses rankings or amplifies confidence.

    Supply an anchor selected before the evaluation period. It can be a fixed
    reference probability or an empirical rate estimated from training or
    calibration labels; estimating it from future evaluation labels leaks
    information. A supplied empirical anchor is treated as fixed during fit.

    All supplied observations have equal weight. Use out-of-sample predictions
    from chronological folds, keeping dates intact, rather than predictions on
    the estimator's own training labels. Better Brier loss does not establish
    improved Top-K accuracy or profitable decisions.
    """

    def __init__(self, anchor: float) -> None:
        if (
            isinstance(anchor, (bool, np.bool_))
            or not isinstance(anchor, Real)
            or not np.isfinite(anchor)
            or not 0 <= anchor <= 1
        ):
            raise ValueError("anchor must be a finite probability in [0, 1].")
        self.anchor = float(anchor)
        self.weight_: float | None = None

    def fit(
        self, probabilities: ArrayLike, labels: ArrayLike
    ) -> ProbabilityShrinkageCalibrator:
        """Fit the closed-form constrained least-squares weight, without mutation.

        Inputs must have identical nonempty shapes and at least one dimension.
        Probabilities are in [0, 1]; labels are binary. If every probability
        equals the anchor, all weights give the same map and zero is chosen.
        """
        values = _probability_array(probabilities, "probabilities")
        outcomes = _probability_array(labels, "labels")
        if values.shape != outcomes.shape:
            raise ValueError("probabilities and labels must have identical shapes.")
        if not np.isin(outcomes, (0, 1)).all():
            raise ValueError("labels must be binary (0 or 1).")
        deviations = values - self.anchor
        denominator = float(np.sum(deviations * deviations))
        numerator = float(np.sum(deviations * (outcomes - self.anchor)))
        self.weight_ = (
            float(np.clip(numerator / denominator, 0.0, 1.0))
            if denominator > 0
            else 0.0
        )
        return self

    def transform(self, probabilities: ArrayLike) -> NDArray[np.float64]:
        """Apply the fitted map, returning a new array with the original shape.

        A convex combination remains in [0, 1]. The output expresses the fitted
        historical adjustment, not a guarantee of future probability accuracy.
        """
        if self.weight_ is None:
            raise RuntimeError("Call fit before transform.")
        values = _probability_array(probabilities, "probabilities")
        return np.asarray(
            self.anchor + self.weight_ * (values - self.anchor), dtype=np.float64
        )


def _probability_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate real, finite, nonempty probability arrays without modifying them."""
    array = np.asarray(values)
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty array with at least one dimension.")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must contain finite values in [0, 1].")
    return array
