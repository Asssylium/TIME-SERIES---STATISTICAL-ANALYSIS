"""Run the XSMB load -> features -> temporal holdout -> forecast pipeline.

Examples:
    python main.py
    python main.py --model both --test-days 90 --top-k 5
    python main.py --model regularized --top-k 5 --forecast-date 2026-10-30
    python main.py --model baseline --url https://example.com/draws.csv

Test forecasts are made sequentially after each completed draw. LightGBM is
fitted once before the test period; its features can use newly observed test
draws when forecasting the following day. Only after test evaluation is the
final model fitted on all available labels for the next unseen day's forecast.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date as calendar_date
import sys

from lightgbm.basic import LightGBMError
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import DEFAULT_XSMB_URL, NUMBER_COLUMNS, XSMBDataLoader
from src.evaluator import EvaluationResult, evaluate_test_set, print_summary_report
from src.diagnostics import PayoutResult, evaluate_payouts
from src.features import build_feature_dataset
from src.models import BaselinePredictor, LGBMPredictor, TARGET, TopK
from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.regularized_models import RegularizedLGBMPredictor


@dataclass(frozen=True)
class PipelineResult:
    """Reusable test results and final forecasts, also printed by the pipeline."""

    train_feature_dates: pd.DatetimeIndex
    test_feature_dates: pd.DatetimeIndex
    prediction_date: pd.Timestamp
    evaluations: dict[str, EvaluationResult]
    forecasts: dict[str, TopK]
    payouts: dict[str, PayoutResult] = field(default_factory=dict)


def run_pipeline(
    url: str = DEFAULT_XSMB_URL,
    *,
    model: str = "lightgbm",
    top_k: int = 5,
    test_days: int = 90,
    n_splits: int = 5,
    gap: int = 1,
    n_estimators: int | None = None,
    n_jobs: int = 1,
    timeout: float = 30.0,
    inspect: bool = False,
    forecast_date: str | None = None,
) -> PipelineResult:
    """Load a CSV URL, evaluate the last time split, and print next-day Top-K.

    The LAST TimeSeriesSplit fold is the held-out test set. Earlier folds are
    not scored or used for tuning. Sizes count observed, labeled feature dates;
    dates without a next-calendar-day outcome cannot be evaluated and are
    excluded. gap >= 1 keeps the latest training label strictly before the
    first test feature date. All 100 candidate numbers for a day stay together.

    Default features include lag-1 counts/presence, rolling frequencies for
    7/14/30/60 days, gaps, and causal co-occurrence summaries. Missing dates
    remain missing; all target dates are matched by calendar date, not row shift.

    LightGBM forecasts are uncalibrated daily-presence probabilities. The
    optional regularized classifier uses normalized features and the preceding
    1095 calendar days of labels; it remains an experimental alternative.
    Payout reports count every actual occurrence at 23,000/80,000 VND.
    Baseline
    forecasts are 30-day frequency counts and are explicitly labeled as such.
    By default the final forecast date is last observed date + 1, not the
    machine's date. forecast_date accepts YYYY-MM-DD and cuts off all outcomes
    on or after that date before creating features or labels. Its preceding
    calendar day's result must be available. Test metrics cover earlier days;
    they do not evaluate the requested forecast day's result.
    """
    if model not in {"baseline", "lightgbm", "regularized", "both"}:
        raise ValueError("model must be baseline, lightgbm, regularized, or both.")
    if n_estimators is None:
        n_estimators = 100 if model == "regularized" else 200
    for name, value in {
        "top_k": top_k, "test_days": test_days, "n_splits": n_splits,
        "gap": gap, "n_estimators": n_estimators, "n_jobs": n_jobs,
    }.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if top_k > 100 or n_splits < 2:
        raise ValueError("top_k must be <= 100 and n_splits must be >= 2.")

    requested_date: pd.Timestamp | None = None
    if forecast_date is not None:
        try:
            parsed_date = calendar_date.fromisoformat(forecast_date)
            if parsed_date.isoformat() != forecast_date:
                raise ValueError("Date must use YYYY-MM-DD.")
            requested_date = pd.Timestamp(parsed_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("forecast_date must be a valid calendar date in YYYY-MM-DD format.") from exc

    print("[1/5] Loading and validating XSMB data...", flush=True)
    loader = XSMBDataLoader(url=url, timeout=timeout)
    raw = loader.load_data()
    if inspect:
        loader.inspect_data(raw)
    frequency, binary = loader.preprocess_and_build_matrices(raw)
    if requested_date is not None:
        # Cut BOTH matrices before constructing any features, labels, or splits:
        # a historical forecast must never see its own outcome or later draws.
        source_end = frequency.index[-1]
        previous_day = requested_date - pd.Timedelta(days=1)
        frequency = frequency.loc[frequency.index < requested_date]
        binary = binary.loc[binary.index < requested_date]
        if frequency.empty or frequency.index[-1] != previous_day:
            raise ValueError(
                f"Cannot forecast {requested_date:%Y-%m-%d}: require the draw for "
                f"the preceding calendar day {previous_day:%Y-%m-%d}. "
                f"Source data ends on {source_end:%Y-%m-%d}; choose an available "
                "next-day date or update --url with the missing results."
            )
        print(f"Requested forecast: {requested_date:%Y-%m-%d}; using results only through {previous_day:%Y-%m-%d}.")
    print(f"Observed days: {len(frequency):,}; {frequency.index[0]:%Y-%m-%d} to {frequency.index[-1]:%Y-%m-%d}")

    print("[2/5] Building causal rolling and gap features...", flush=True)
    if model == "regularized":
        dataset = build_count_feature_dataset(frequency, binary, include_unlabeled=True)
        target_column = COUNT_TARGET
        print("Experimental regularized classifier: normalized features, trailing 1095-day labels; probabilities remain uncalibrated.")
    else:
        dataset = build_feature_dataset(frequency, binary, include_unlabeled=True)
        target_column = TARGET
    labeled = dataset.loc[dataset[target_column].notna()]
    dates = pd.DatetimeIndex(labeled.index.get_level_values(0).unique())
    minimum_days = n_splits * test_days + gap + 1
    if len(dates) < minimum_days:
        raise ValueError(
            f"Only {len(dates)} labeled dates; need at least {minimum_days} for "
            f"n_splits={n_splits}, test_days={test_days}, gap={gap}. "
            "Reduce --test-days/--n-splits or provide more history."
        )

    print("[3/5] Selecting the final chronological train/test split...", flush=True)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_days, gap=gap)
    train_positions, test_positions = list(splitter.split(dates))[-1]
    train_dates, test_dates = dates[train_positions], dates[test_positions]
    origins = labeled.index.get_level_values(0)
    train = labeled.loc[origins.isin(train_dates)]
    test = labeled.loc[origins.isin(test_dates)]
    one_day = pd.Timedelta(days=1)
    train_label_end = train_dates[-1] + one_day
    if train_label_end >= test_dates[0]:
        raise ValueError("Training labels overlap the test feature period; increase --gap.")
    target_dates = test_dates + one_day
    print(f"Available training feature dates: {train_dates[0]:%Y-%m-%d} to {train_dates[-1]:%Y-%m-%d}")
    print(f"Training labels end: {train_label_end:%Y-%m-%d}")
    print(f"Test target dates: {target_dates[0]:%Y-%m-%d} to {target_dates[-1]:%Y-%m-%d} ({len(target_dates)} days)")
    print("Test protocol: sequential next-day forecasts using only completed draws.")
    if model in {"lightgbm", "regularized", "both"}:
        print("LightGBM test: fitted once before the holdout; no daily refit. Daily-refit backtests use a different policy.")

    # Recover the 27 actual results, retaining duplicates for the evaluator's
    # raw-count diagnostics. Metrics themselves count distinct matching numbers.
    truth = [
        np.repeat(np.arange(100), counts)
        for counts in frequency.loc[target_dates].to_numpy(dtype=np.int64)
    ]
    names = ["baseline", "lightgbm"] if model == "both" else [model]
    model_parameters = {"n_estimators": n_estimators, "n_jobs": n_jobs}
    evaluations: dict[str, EvaluationResult] = {}
    forecasts: dict[str, TopK] = {}
    payouts: dict[str, PayoutResult] = {}

    print("[4/5] Fitting models and evaluating the held-out test dates...", flush=True)
    for name in names:
        print(f"\nModel: {name}", flush=True)
        if name == "baseline":
            baseline = BaselinePredictor(window=30)
            predictions: list[TopK] = []
            for date in test_dates:
                # Refit only the baseline's observed rolling window at each
                # forecast origin; neither tomorrow nor later draws are visible.
                history = frequency.loc[
                    (frequency.index > date - pd.Timedelta(days=30))
                    & (frequency.index <= date)
                ]
                predictions.append(baseline.train_predict(history, top_k=top_k))
        else:
            predictor = RegularizedLGBMPredictor if name == "regularized" else LGBMPredictor
            fitted = predictor(**model_parameters).fit(train)
            if name == "regularized":
                print(f"Actual fitted label dates: {fitted.training_label_start_:%Y-%m-%d} to {fitted.trained_through_:%Y-%m-%d}")
            probabilities = fitted.predict_proba(test).unstack(level=1)
            probabilities = probabilities.reindex(index=test_dates, columns=NUMBER_COLUMNS)
            scores = probabilities.to_numpy()
            order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
            predictions = [
                [(NUMBER_COLUMNS[number], float(scores[row, number])) for number in selected]
                for row, selected in enumerate(order)
            ]
        evaluations[name] = evaluate_test_set(truth, predictions, k=top_k, dates=target_dates)
        print_summary_report(evaluations[name])
        selection = np.zeros((len(target_dates), 100), dtype=bool)
        for row, predicted in enumerate(predictions):
            selection[row, [int(number) for number, _ in predicted]] = True
        payouts[name] = evaluate_payouts(
            frequency.loc[target_dates].to_numpy(dtype=np.int64), selection,
            payout_mode="per_occurrence", stake=23_000, gross_payout=80_000,
        )
        money = payouts[name].summary
        print("Payout rule: 23,000 VND per selected number; gross 80,000 VND per occurrence (including repeats).")
        print(f"Occurrences: {money['occurrences']}; cost: {money['cost']:,.0f}; receipts: {money['receipts']:,.0f}; profit: {money['profit']:+,.0f} VND; ROI: {money['roi']:.2%}")

    # Test metrics are now finalized. It is safe to use all observed labels
    # (including former test outcomes) for a separate, future-facing model.
    prediction_date = frequency.index[-1] + one_day
    print(f"\n[5/5] Fitting forecast models for {prediction_date:%Y-%m-%d}...", flush=True)
    latest = dataset.loc[dataset.index.get_level_values(0) == frequency.index[-1]]
    for name in names:
        if name == "baseline":
            forecasts[name] = BaselinePredictor().train_predict(frequency, top_k=top_k)
            score_label = "30-day count"
            display_scores = [f"{score:.0f}" for _, score in forecasts[name]]
        else:
            predictor = RegularizedLGBMPredictor if name == "regularized" else LGBMPredictor
            final_model = predictor(**model_parameters).fit(dataset)
            forecasts[name] = final_model.predict(latest, top_k=top_k)
            score_label = "Estimated P(at least one appearance), uncalibrated"
            display_scores = [f"{score:.2%}" for _, score in forecasts[name]]
        table = pd.DataFrame({
            "Rank": np.arange(1, top_k + 1),
            "Number": [number for number, _ in forecasts[name]],
            score_label: display_scores,
        })
        print(f"\n{name.upper()} TOP {top_k} FOR {prediction_date:%Y-%m-%d}")
        print(table.to_string(index=False))

    return PipelineResult(train_dates, test_dates, prediction_date, evaluations, forecasts, payouts)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and return an exit code without hiding failures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_XSMB_URL, help="CSV URL (defaults to the XSMB source).")
    parser.add_argument("--model", choices=["baseline", "lightgbm", "regularized", "both"], default="lightgbm")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--forecast-date", default=None, metavar="YYYY-MM-DD",
                        help="Forecast this calendar day using only earlier draws; requires its preceding day's result. Default: last observed day + 1.")
    parser.add_argument("--test-days", type=int, default=90, help="Labeled feature dates per time split.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of chronological splits; use the last as test.")
    parser.add_argument("--gap", type=int, default=1, help="Purged feature dates between train and test (>= 1).")
    parser.add_argument("--n-estimators", type=int, default=None, help="Fixed tree count (default 200; regularized uses 100).")
    parser.add_argument("--n-jobs", type=int, default=1, help="Positive LightGBM thread count.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Download timeout in seconds.")
    parser.add_argument("--inspect", action="store_true", help="Print raw DataFrame info and head.")
    args = parser.parse_args(argv)
    try:
        run_pipeline(**vars(args))
    except (OSError, ValueError, LightGBMError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Pipeline interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
