"""Train ordered-prize models, compare chronological forecasts, and save models.

Example: python train_prize_model.py --forecast-date 2026-10-30 --top-k 5
All three comparison models refit at quarter boundaries using the same known
labels. This differs from earlier daily-refit backtests. No model or parameter
is selected by test profit; both experimental objectives are saved explicitly.
"""

from __future__ import annotations

import argparse
from datetime import date as calendar_date
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

from src.calibration import ProbabilityShrinkageCalibrator
from src.count_features import COUNT_TARGET
from src.data_loader import XSMBDataLoader
from src.diagnostics import block_bootstrap_mean_interval, evaluate_payouts, probability_diagnostics, rank_top_k
from src.prize_features import EXTRA_FEATURES, PRIZE_COLUMNS, PRIZE_WIDTHS, RAW_PRIZE_URL, build_prize_feature_dataset, normalize_prizes
from src.prize_models import PrizeLGBMPredictor
from src.regularized_models import RegularizedLGBMPredictor


MODEL_NAMES = ("regularized", "prize_binary", "prize_poisson")


def iso_date(value: str) -> pd.Timestamp:
    """Require a strict ISO calendar date, without timezone or time component."""
    parsed = calendar_date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD.")
    return pd.Timestamp(parsed)


def run(url: str = RAW_PRIZE_URL, forecast_date: str | None = None, top_k: int = 5,
        test_start: str = "2025-01-01", output: Path | None = None) -> dict:
    """Evaluate fixed configurations, then fit and save separate final models."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer in 1..100.")
    start = iso_date(test_start)
    requested = iso_date(forecast_date) if forecast_date is not None else None
    loader = XSMBDataLoader(url=url)
    prizes = normalize_prizes(loader.load_data())
    target = requested if requested is not None else prizes.index[-1] + pd.Timedelta(days=1)
    prizes = prizes.loc[prizes.index < target]
    if prizes.empty or prizes.index[-1] != target - pd.Timedelta(days=1):
        raise ValueError("The preceding calendar day's complete 27 prizes are required.")
    if start >= target:
        raise ValueError("test_start must precede forecast_date.")
    raw = prizes.reset_index()
    frequency, binary = XSMBDataLoader().preprocess_and_build_matrices(raw)
    known = frequency.index[(frequency.index-pd.Timedelta(days=1)).isin(frequency.index)]
    prior = known[known < start]
    if not len(prior):
        raise ValueError("No known training labels before the evaluation period.")
    feature_start = prior[-1] - pd.Timedelta(days=1095)
    out = output or Path(f"reports/prize_model_{target:%Y-%m-%d}")
    out.mkdir(parents=True, exist_ok=True)
    cache = out/"cache"
    cache.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    protocol = {
        "source_url": url, "prize_snapshot_sha256": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        "forecast_date": str(target.date()), "evaluation_start": str(start.date()),
        "evaluation_end": str(prizes.index[-1].date()), "top_k_forecast": top_k,
        "ranking_k": [1, 5, 10], "history_days": 1095, "refit_policy": "before each calendar quarter",
        "models": list(MODEL_NAMES), "parameters": {name: (
            RegularizedLGBMPredictor().params if name == "regularized" else
            PrizeLGBMPredictor("binary" if name == "prize_binary" else "poisson").params) for name in MODEL_NAMES},
        "draw_columns": list(PRIZE_COLUMNS), "prize_widths": PRIZE_WIDTHS,
        "draw_order_provenance": "Group and within-group order explicitly confirmed by user; not independently verified.",
        "source_cutoff": "Remove all results on or after forecast date BEFORE feature creation.",
        "feature_timing": "All features at date t use completed results through t only and forecast t+1. Exact calendar lag/label matching.",
        "calibration": "Presence shrinkage fitted only to 2025 chronological OOF predictions; evaluated separately on 2026 if available. Raw ranking retained and labeled uncalibrated.",
        "retrospective": "2025 and 2026 have already been examined in previous research; neither is an untouched prospective test.",
        "selection": "No hyperparameter, K, threshold or model search using evaluation results. Both new objectives saved, no automatic replacement of existing model.",
        "payout": {"stake": 27000, "gross_per_occurrence": 99000, "count_duplicates": True},
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "lightgbm", "scikit-learn", "joblib"]}},
        "code_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in [
            "scripts/operations/train_prize_model.py", "src/prize_features.py", "src/prize_models.py", "src/count_features.py",
            "src/count_models.py", "src/features.py", "src/regularized_models.py", "src/data_loader.py", "src/calibration.py", "src/diagnostics.py"]},
    }
    protocol_path = out/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Protocol/source changed; select another --output directory.")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw.to_csv(out/"source_snapshot.csv", index=False, encoding="utf-8-sig")
    print(f"BUILD features from {len(raw)} observed dates; output origins from {feature_start:%Y-%m-%d}", flush=True)
    data = build_prize_feature_dataset(raw, include_unlabeled=True, start=feature_start)
    base = data.drop(columns=list(EXTRA_FEATURES))
    print(f"FEATURES: base={len(base.columns)-1}, ordered-prize={len(data.columns)-1}", flush=True)
    origins = data.index.get_level_values(0)
    labels = origins + pd.Timedelta(days=1)
    has_label = data[COUNT_TARGET].notna().to_numpy()
    calendar = pd.date_range(start, target-pd.Timedelta(days=1))
    valid = calendar.isin(frequency.index) & (calendar-pd.Timedelta(days=1)).isin(frequency.index)
    evaluated_dates = calendar[valid]
    if not len(evaluated_dates):
        raise ValueError("No evaluable target dates.")
    pd.DataFrame({"date": calendar[~valid], "reason": [
        "missing_actual" if day not in frequency.index else "missing_previous_day" for day in calendar[~valid]
    ]}).to_csv(out/"excluded_dates.csv", index=False)
    scores_by_model = {name: [] for name in MODEL_NAMES}
    evaluated_blocks, cutoff_rows = [], []
    began = time.monotonic()
    for period in evaluated_dates.to_period("Q").unique():
        first = max(period.start_time, start)
        block_dates = evaluated_dates[evaluated_dates.to_period("Q") == period]
        train = has_label & (labels < first)
        last_label = labels[train].max()
        train &= labels >= last_label - pd.Timedelta(days=1094)
        test = labels.isin(block_dates)
        np.testing.assert_array_equal(data.loc[test, COUNT_TARGET].to_numpy(dtype=int).reshape(-1, 100), frequency.loc[block_dates].to_numpy())
        evaluated_blocks.append(block_dates)
        for name in MODEL_NAMES:
            frame = base if name == "regularized" else data
            path = cache/f"{first:%Y-%m-%d}_{name}.npz"
            cutoff_rows.append({"model": name, "first_test_target": str(block_dates[0].date()),
                                "last_test_target": str(block_dates[-1].date()),
                                "training_label_start": str(labels[train].min().date()),
                                "training_label_end": str(last_label.date())})
            if last_label >= block_dates[0]:
                raise AssertionError("Training labels overlap test targets.")
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    np.testing.assert_array_equal(cached["dates"], block_dates.strftime("%Y-%m-%d"))
                    score = cached["scores"]
            else:
                print(f"FIT {period} {name}; labels through {last_label:%Y-%m-%d}", flush=True)
                model = RegularizedLGBMPredictor() if name == "regularized" else PrizeLGBMPredictor("binary" if name == "prize_binary" else "poisson")
                model.fit(frame.loc[train])
                inference = frame.loc[test].drop(columns=COUNT_TARGET)
                score = (model.predict_proba(inference) if name == "regularized" else model.predict_scores(inference)).to_numpy().reshape(-1, 100)
                np.savez_compressed(path, scores=score, dates=block_dates.strftime("%Y-%m-%d").to_numpy(dtype=str))
                del model, inference
                gc.collect()
            if score.shape != (len(block_dates), 100) or not np.isfinite(score).all():
                raise ValueError("Invalid cached/model score shape or values.")
            scores_by_model[name].append(score)
        print(f"DONE {period}; elapsed={time.monotonic()-began:.0f}s", flush=True)
    dates = evaluated_blocks[0].append(evaluated_blocks[1:])
    scores_by_model = {name: np.concatenate(parts) for name, parts in scores_by_model.items()}
    counts = frequency.loc[dates].to_numpy(dtype=int)
    metrics, rankings, daily_frames = [], [], []
    calibrators, calibration = {}, {}
    anchor = float(binary.loc[binary.index < start].to_numpy().mean())
    cal_mask = dates.year == 2025
    for name in ("regularized", "prize_binary"):
        if cal_mask.any():
            cal = ProbabilityShrinkageCalibrator(anchor).fit(scores_by_model[name][cal_mask], counts[cal_mask] > 0)
            calibrators[name] = cal
            calibration[name] = {"anchor": anchor, "weight": cal.weight_, "fit_target_year": 2025}
    (out/"calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        reference_daily = {}
        for k in (1, 5, 10):
            reference_daily[k] = evaluate_payouts(counts[mask], rank_top_k(scores_by_model['regularized'][mask], k), payout_mode="per_occurrence").per_day.profit.to_numpy()
        for name, scores in scores_by_model.items():
            row = {"year": int(year), "model": name, "days": int(mask.sum())}
            if name != "prize_poisson":
                raw_stats, _ = probability_diagnostics(counts[mask] > 0, scores[mask])
                row.update(raw_stats)
                if year > 2025 and name in calibrators:
                    calibrated, _ = probability_diagnostics(counts[mask] > 0, calibrators[name].transform(scores[mask]))
                    row["calibrated_brier"] = calibrated["brier"]
                row["constant_anchor_brier"] = float(np.square((counts[mask] > 0)-anchor).mean())
            else:
                np.testing.assert_allclose(scores[mask].sum(axis=1), 27)
                row["count_mse"] = float(np.square(scores[mask]-counts[mask]).mean())
                row["constant_count_mse"] = float(np.square(.27-counts[mask]).mean())
            metrics.append(row)
            for k in (1, 5, 10):
                selected = rank_top_k(scores[mask], k)
                payout = evaluate_payouts(counts[mask], selected, payout_mode="per_occurrence")
                daily = payout.per_day.copy()
                daily.insert(0, "date", dates[mask].strftime("%Y-%m-%d"))
                daily["model"], daily["top_k"] = name, k
                daily["prediction"] = [" ".join(f"{n:02d}" for n in np.argsort(-s, kind='stable')[:k]) for s in scores[mask]]
                daily["hits_with_counts"] = [" ".join(f"{n:02d}x{actual[n]}" for n in np.flatnonzero(choice & (actual>0))) for choice, actual in zip(selected, counts[mask])]
                daily_frames.append(daily)
                delta = daily.profit.to_numpy() - reference_daily[k]
                low, high = block_bootstrap_mean_interval(delta)
                rankings.append({"year": int(year), "model": name, "top_k": k, **payout.summary,
                                 "profit_delta_vs_regularized": float(delta.sum()),
                                 "delta_mean_profit_95_low": low, "delta_mean_profit_95_high": high})
    pd.DataFrame(metrics).to_csv(out/"metrics.csv", index=False)
    ranking_frame = pd.DataFrame(rankings)
    ranking_frame.to_csv(out/"ranking.csv", index=False, encoding="utf-8-sig")
    pd.concat(daily_frames, ignore_index=True).to_csv(out/"daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cutoff_rows).to_csv(out/"training_cutoffs.csv", index=False)
    print("RANKING " + ranking_frame[['year','model','top_k','profit','precision_at_k','delta_mean_profit_95_low','delta_mean_profit_95_high']].to_json(orient='records'), flush=True)
    print("CALIBRATION " + json.dumps(calibration), flush=True)
    models_dir = out/"models"
    models_dir.mkdir(exist_ok=True)
    final_known = has_label & (labels < target)
    last_label = labels[final_known].max()
    final_known &= labels >= last_label - pd.Timedelta(days=1094)
    latest = data.loc[origins == prizes.index[-1]].drop(columns=COUNT_TARGET)
    forecasts, importance = [], []
    for name, objective in [("prize_binary", "binary"), ("prize_poisson", "poisson")]:
        print(f"FINAL FIT {name} for {target:%Y-%m-%d}", flush=True)
        model = PrizeLGBMPredictor(objective).fit(data.loc[final_known])
        score = model.predict_scores(latest).to_numpy()
        model_path = models_dir/f"{name}.joblib"
        joblib.dump(model, model_path)
        model.model_.booster_.save_model(str(models_dir/f"{name}.txt"))
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(restored.predict_scores(latest).to_numpy(), score)
        order = np.argsort(-score, kind="stable")
        calibrated = calibrators[name].transform(score) if name in calibrators else None
        for rank, number in enumerate(order, 1):
            forecasts.append({"forecast_date": str(target.date()), "model": name, "rank": rank,
                              "number": f"{number:02d}", "score": float(score[number]),
                              "score_type": "uncalibrated_presence_probability" if objective == 'binary' else "expected_occurrences",
                              "calibrated_probability": float(calibrated[number]) if calibrated is not None else None})
        gains = model.model_.booster_.feature_importance(importance_type="gain")
        importance.extend({"model": name, "feature": column, "gain": float(gain)} for column, gain in zip(model.feature_columns_, gains))
        del model, restored
        gc.collect()
    pd.DataFrame(forecasts).to_csv(out/"forecast_all.csv", index=False, encoding="utf-8-sig")
    top = pd.DataFrame(forecasts).loc[lambda frame: frame['rank'] <= top_k]
    top.to_csv(out/"forecast_top_k.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(importance).to_csv(out/"feature_importance.csv", index=False)
    manifest = {"forecast_date": str(target.date()), "training_label_end": str(last_label.date()),
                "feature_count": len(latest.columns), "feature_columns": list(latest.columns),
                "models": ["prize_binary", "prize_poisson"], "reload_prediction_equality": True,
                "calibration": calibration, "protocol_file": "protocol.json"}
    (models_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Ordered-prize model training and chronological comparison", "",
             f"Forecast date: {target:%Y-%m-%d}; training results through {last_label:%Y-%m-%d}. "
             f"{len(latest.columns)} input features including 27 ordered prize positions and all available leading digits.", "",
             "Both experimental models were saved and reload predictions were checked for exact equality. "
             "Binary estimates appearance probability; Poisson estimates occurrence count, NOT probability. "
             "Raw ranking is retained even if calibration collapses scores to a constant.", "",
             "Evaluation: chronological quarter-by-quarter refits with identical training cutoffs and 1095-day label windows. "
             "This is a different policy from daily-refit reports. 2025/2026 were already examined, so this is retrospective. "
             "No tuning, profit-driven model selection, or automatic replacement of the existing model.", "",
             "| Year | Model | K | Days | Precision@K | Paid occurrences | Profit (VND) | ROI |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in ranking_frame.itertuples():
        lines.append(f"| {row.year} | {row.model} | {row.top_k} | {row.days} | {row.precision_at_k:.2%} | {row.occurrences} | {row.profit:+,.0f} | {row.roi:.2%} |")
    lines.extend(["", "Cash flow: 27,000 VND per selected number/day; gross 99,000 VND EACH occurrence, counting repeats. "
                  "Missing target or preceding days are excluded, not imputed as losses/abstentions. "
                  "Profit-difference intervals in ranking.csv use 2,000 seven-observation-day block bootstrap samples; "
                  "they are approximate, cross calendar gaps, and are not adjusted for comparing multiple models/K.", "",
                  "Feature importances describe tree usage; they do not establish causality or a hidden physical rule. "
                  "Temporal order follows the user's confirmation, not independent video/time verification.", "",
                  "[Ranking](./ranking.csv) · [Metrics](./metrics.csv) · [Daily results](./daily.csv) · "
                  "[Forecast](./forecast_top_k.csv) · [Excluded dates](./excluded_dates.csv) · "
                  "[Protocol](./protocol.json) · [Model manifest](./models/manifest.json)", ""])
    (out/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print("FORECAST " + top.to_json(orient='records'), flush=True)
    print("REPORT " + str(out/"report.md"), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Parse training options and return a readable failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=RAW_PRIZE_URL)
    parser.add_argument("--forecast-date", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--output", type=Path, default=None)
    try:
        run(**vars(parser.parse_args(argv)))
    except (ValueError, OSError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
