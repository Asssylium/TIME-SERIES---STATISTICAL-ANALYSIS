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
from src.data_loader import DEFAULT_XSMB_URL, XSMBDataLoader  # noqa: E402
from src.prize_features import build_prize_feature_dataset, normalize_prizes  # noqa: E402
from src.prize_models import PrizeLGBMPredictor  # noqa: E402


ALLOWED_MODELS = ("baseline", "lightgbm", "regularized", "poisson", "poisson_prize_90d", "both", "all")
SOURCE_DIRS = (ROOT / "data" / "processed", ROOT / "data" / "raw")

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
            "location": DEFAULT_XSMB_URL,
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


def _prize_poisson_90d_payload(source_id: str, top_k: int, target_date: str | None) -> dict[str, Any]:
    """Run the same rolling 90-day prize Poisson model as the CLI script."""
    raw = _load_raw(source_id)
    prizes = normalize_prizes(raw)
    latest = prizes.index[-1]
    target = latest + pd.Timedelta(days=1)
    if target_date is not None and pd.Timestamp(target_date) != target:
        raise ValueError(
            f"poisson_prize_90d forecasts the next unseen date only: {target:%Y-%m-%d}."
        )
    feature_start = target - pd.Timedelta(days=91)
    data = build_prize_feature_dataset(raw, include_unlabeled=True, start=feature_start)
    origins = data.index.get_level_values(0)
    labels = origins + pd.Timedelta(days=1)
    training_mask = (
        data[COUNT_TARGET].notna().to_numpy()
        & (labels >= target - pd.Timedelta(days=90))
        & (labels < target)
    )
    train = data.loc[training_mask]
    inference = data.loc[origins == latest].drop(columns=COUNT_TARGET)
    if train.empty or len(inference) != 100:
        raise ValueError(f"Incomplete prize data for forecast {target:%Y-%m-%d}.")
    model = PrizeLGBMPredictor("poisson", history_days=90, n_jobs=1, n_estimators=80).fit(train)
    scores = model.predict_scores(inference).droplevel(0)
    ranked = scores.sort_values(ascending=False, kind="stable").head(top_k)
    return _json_safe(
        {
            "target_date": target.strftime("%Y-%m-%d"),
            "source": source_id,
            "model": "poisson_prize_90d",
            "top_k": top_k,
            "test_days": 90,
            "forecasts": {
                "poisson_prize_90d": [
                    {"number": str(number), "score": float(score)}
                    for number, score in ranked.items()
                ]
            },
            "evaluations": {},
            "payouts": {},
        }
    )


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
    if target_date is not None:
        try:
            parsed = calendar_date.fromisoformat(str(target_date))
            if parsed.isoformat() != str(target_date):
                raise ValueError
        except ValueError:
            return jsonify({"error": "target_date must use YYYY-MM-DD format."}), 400
    try:
        if model == "poisson_prize_90d":
            return jsonify(_prize_poisson_90d_payload(source_id, top_k, target_date))
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
