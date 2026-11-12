"""Vercel-compatible HTTP API for the forecasting project.

This module serves the browser dashboard locally and is the stateless
serverless entry point used by Vercel. It exposes only the
operations needed by the browser dashboard.
"""

from __future__ import annotations

from datetime import date as calendar_date
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.native_runtime import load_bundled_openmp  # noqa: E402

load_bundled_openmp()

from main import run_pipeline  # noqa: E402
from src.count_features import COUNT_TARGET  # noqa: E402
from src.data_loader import XSMBDataLoader  # noqa: E402
from src.prize_features import RAW_PRIZE_URL, build_prize_feature_dataset, normalize_prizes  # noqa: E402
from src.prize_models import PrizeLGBMPredictor  # noqa: E402


ALLOWED_MODELS = ("baseline", "lightgbm", "regularized", "poisson", "poisson_prize_90d", "both", "all")
SOURCE_DIRS = (ROOT / "data" / "processed", ROOT / "data" / "raw")
MAX_PRIZE_RANGE_DAYS = 31

app = Flask(__name__)


@app.get("/")
def home() -> Any:
    """Serve the same dashboard when the function is run directly."""
    return send_from_directory(ROOT / "public", "index.html")


def _source_catalog() -> dict[str, dict[str, str]]:
    sources = {
        "live": {
            "id": "live",
            "label": "Live GitHub data feed",
            "kind": "remote",
            # Ordered prizes support Prize Poisson and can also be aggregated
            # by XSMBDataLoader for the other models and source-health display.
            "location": RAW_PRIZE_URL,
        }
    }
    for folder in SOURCE_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
            relative = path.relative_to(ROOT).as_posix()
            source_id = f"bundled:{relative}"
            sources[source_id] = {
                "id": source_id,
                "label": relative,
                "kind": "bundled",
                "location": relative,
            }
    return sources


def _source_url(source_id: str) -> str:
    source = _source_catalog().get(source_id)
    if source is None:
        raise ValueError("Unknown data source.")
    if source["kind"] == "remote":
        return source["location"]
    path = (ROOT / source["location"]).resolve()
    if ROOT not in path.parents or path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError("The selected bundled source is not available.")
    return path.as_uri()


@lru_cache(maxsize=8)
def _load_matrices(source_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = XSMBDataLoader(url=_source_url(source_id), timeout=60)
    return loader.preprocess_and_build_matrices(loader.load_data())


@lru_cache(maxsize=4)
def _load_raw(source_id: str) -> pd.DataFrame:
    loader = XSMBDataLoader(url=_source_url(source_id), timeout=60)
    return loader.load_data()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, calendar_date)):
        return value.isoformat()
    return value


def _forecast_payload(result: Any, source_id: str, model: str, top_k: int, test_days: int) -> dict[str, Any]:
    return _json_safe(
        {
            "target_date": result.prediction_date.strftime("%Y-%m-%d"),
            "source": source_id,
            "model": model,
            "top_k": top_k,
            "test_days": test_days,
            "forecasts": {
                name: [{"number": str(number), "score": float(score)} for number, score in values]
                for name, values in result.forecasts.items()
            },
            "evaluations": {name: dict(item.summary) for name, item in result.evaluations.items()},
            "payouts": {name: dict(item.summary) for name, item in result.payouts.items()},
        }
    )


def _prize_poisson_90d_result(
    data: pd.DataFrame, target: pd.Timestamp, top_k: int,
) -> dict[str, Any]:
    """Refit the causal 90-day model and forecast one requested target."""
    origins = data.index.get_level_values(0)
    labels = origins + pd.Timedelta(days=1)
    origin = target - pd.Timedelta(days=1)
    training_mask = (
        data[COUNT_TARGET].notna().to_numpy()
        & (labels >= target - pd.Timedelta(days=90))
        & (labels < target)
    )
    train = data.loc[training_mask]
    training_labels = labels[training_mask]
    inference = data.loc[origins == origin].drop(columns=COUNT_TARGET)
    if train.empty:
        raise ValueError(f"No labeled training data is available before {target:%Y-%m-%d}.")
    if len(inference) != 100:
        raise ValueError(
            f"The completed draw for {origin:%Y-%m-%d} is required to forecast {target:%Y-%m-%d}."
        )
    model = PrizeLGBMPredictor("poisson", history_days=90, n_jobs=1, n_estimators=80).fit(train)
    scores = model.predict_scores(inference).droplevel(0)
    ranked = scores.sort_values(ascending=False, kind="stable").head(top_k)
    return {
        "target_date": target.strftime("%Y-%m-%d"),
        "training_start_date": training_labels.min().strftime("%Y-%m-%d"),
        "training_end_date": training_labels.max().strftime("%Y-%m-%d"),
        "forecasts": {
            "poisson_prize_90d": [
                {"number": str(number), "score": float(score)}
                for number, score in ranked.items()
            ]
        },
        "evaluations": {},
        "payouts": {},
    }


def _prize_poisson_90d_payload(
    source_id: str, top_k: int, target_date: str | None, target_end_date: str | None = None,
) -> dict[str, Any]:
    """Forecast one date or an inclusive range with independent walk-forward refits."""
    raw = _load_raw(source_id)
    prizes = normalize_prizes(raw)
    latest = prizes.index[-1]
    start = pd.Timestamp(target_date) if target_date is not None else latest + pd.Timedelta(days=1)
    end = pd.Timestamp(target_end_date) if target_end_date is not None else start
    if end < start:
        raise ValueError("target_end_date must be on or after target_date.")
    targets = pd.date_range(start, end, freq="D")
    if len(targets) > MAX_PRIZE_RANGE_DAYS:
        raise ValueError(
            f"A Poisson Prize range may contain at most {MAX_PRIZE_RANGE_DAYS} calendar days."
        )
    latest_target = latest + pd.Timedelta(days=1)
    if end > latest_target:
        raise ValueError(
            f"The latest forecastable date for this source is {latest_target:%Y-%m-%d}."
        )
    feature_start = start - pd.Timedelta(days=91)
    data = build_prize_feature_dataset(raw, include_unlabeled=True, start=feature_start)
    results = [_prize_poisson_90d_result(data, target, top_k) for target in targets]
    common = {
        "source": source_id,
        "model": "poisson_prize_90d",
        "top_k": top_k,
        "test_days": 90,
    }
    if target_end_date is None:
        return _json_safe({**common, **results[0]})
    return _json_safe({
        **common,
        "target_start_date": start.strftime("%Y-%m-%d"),
        "target_end_date": end.strftime("%Y-%m-%d"),
        "result_count": len(results),
        "results": results,
    })


@app.get("/api/health")
def health() -> Any:
    return jsonify({"status": "ok", "service": "xsmb-forecast-api", "models": ALLOWED_MODELS})


@app.get("/api/sources")
def sources() -> Any:
    items = []
    for source in _source_catalog().values():
        item = {key: source[key] for key in ("id", "label", "kind")}
        try:
            frequency, _ = _load_matrices(source["id"])
            item["last_date"] = frequency.index[-1].strftime("%Y-%m-%d")
            item["next_date"] = (frequency.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            item["days"] = int(len(frequency))
        except Exception as exc:  # source health should not hide other sources
            item["error"] = str(exc)
        items.append(item)
    return jsonify({"sources": items})


@app.post("/api/forecast")
def forecast() -> Any:
    body = request.get_json(silent=True) or {}
    source_id = str(body.get("source", "live"))
    model = str(body.get("model", "baseline"))
    try:
        top_k = int(body.get("top_k", 3))
        test_days = int(body.get("test_days", 90))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k and test_days must be integers."}), 400
    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"model must be one of: {', '.join(ALLOWED_MODELS)}."}), 400
    if not 1 <= top_k <= 20:
        return jsonify({"error": "top_k must be between 1 and 20."}), 400
    if not 30 <= test_days <= 300:
        return jsonify({"error": "test_days must be between 30 and 300."}), 400
    target_date = body.get("target_date") or None
    target_end_date = body.get("target_end_date") or None
    for field, value in (("target_date", target_date), ("target_end_date", target_end_date)):
        if value is None:
            continue
        try:
            parsed = calendar_date.fromisoformat(str(value))
            if parsed.isoformat() != str(value):
                raise ValueError
        except ValueError:
            return jsonify({"error": f"{field} must use YYYY-MM-DD format."}), 400
    if target_end_date is not None and target_date is None:
        return jsonify({"error": "target_date is required when target_end_date is provided."}), 400
    if target_end_date is not None and model != "poisson_prize_90d":
        return jsonify({"error": "Date ranges are currently supported only by poisson_prize_90d."}), 400
    try:
        if model == "poisson_prize_90d":
            return jsonify(_prize_poisson_90d_payload(source_id, top_k, target_date, target_end_date))
        result = run_pipeline(
            url=_source_url(source_id),
            model=model,
            top_k=top_k,
            test_days=test_days,
            n_splits=5,
            n_estimators=80,
            n_jobs=1,
            timeout=60,
            forecast_date=target_date,
        )
        return jsonify(_forecast_payload(result, source_id, model, top_k, test_days))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Forecast request failed")
        return jsonify({"error": "Forecast failed. Check the selected source and target date."}), 500


@app.get("/api")
def api_index() -> Any:
    return jsonify({"service": "xsmb-forecast-api", "health": "/api/health", "sources": "/api/sources"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
