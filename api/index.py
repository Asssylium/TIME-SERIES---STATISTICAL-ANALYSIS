"""Vercel-compatible HTTP API for the forecasting project.

This module serves the browser dashboard locally and is the stateless
serverless entry point used by Vercel. It exposes only the
operations needed by the browser dashboard.
"""

from __future__ import annotations

from datetime import date as calendar_date, datetime, timezone
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
from src.analytics import build_analytics_payload  # noqa: E402
from src.count_features import COUNT_TARGET  # noqa: E402
from src.data_loader import XSMBDataLoader  # noqa: E402
from src.diagnostics import evaluate_payouts  # noqa: E402
from src.evaluator import evaluate_test_set  # noqa: E402
from src.prize_features import RAW_PRIZE_URL, build_prize_feature_dataset, normalize_prizes  # noqa: E402
from src.prize_models import PrizeLGBMPredictor  # noqa: E402


ALLOWED_MODELS = ("baseline", "lightgbm", "regularized", "poisson", "poisson_prize_90d", "both", "all")
MAX_PRIZE_RANGE_DAYS = 31
PRIZE_POISSON_HISTORY_DAYS = 90
PRIZE_POISSON_N_ESTIMATORS = 100

app = Flask(__name__)


@app.get("/")
def home() -> Any:
    """Serve the same dashboard when the function is run directly."""
    return send_from_directory(ROOT / "public", "index.html")


@app.get("/assets/<path:filename>")
def public_asset(filename: str) -> Any:
    """Serve dashboard assets during local Flask development."""
    return send_from_directory(ROOT / "public" / "assets", filename)


def _source_catalog() -> dict[str, dict[str, str]]:
    return {
        "live": {
            "id": "live",
            "label": "Live GitHub data feed",
            "kind": "remote",
            # Ordered prizes support Prize Poisson and can also be aggregated
            # by XSMBDataLoader for the other models and source-health display.
            "location": RAW_PRIZE_URL,
        }
    }


def _source_url(source_id: str) -> str:
    source = _source_catalog().get(source_id)
    if source is None:
        raise ValueError("Only the live GitHub data feed is supported.")
    return source["location"]


@lru_cache(maxsize=8)
def _load_matrices(
    source_id: str, source_revision: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source matrices; a revision key explicitly bypasses stale cache entries."""
    loader = XSMBDataLoader(url=_source_url(source_id), timeout=60)
    return loader.preprocess_and_build_matrices(loader.load_data())


@lru_cache(maxsize=4)
def _load_raw(source_id: str, source_revision: str | None = None) -> pd.DataFrame:
    """Load ordered prizes; a revision key explicitly bypasses stale cache entries."""
    loader = XSMBDataLoader(url=_source_url(source_id), timeout=60)
    return loader.load_data()


def _source_item(source_id: str, source_revision: str | None = None) -> dict[str, Any]:
    """Return validated source metadata using the requested cache revision."""
    source = _source_catalog()[source_id]
    item: dict[str, Any] = {key: source[key] for key in ("id", "label", "kind")}
    frequency, _ = _load_matrices(source_id, source_revision)
    item["first_date"] = frequency.index[0].strftime("%Y-%m-%d")
    item["last_date"] = frequency.index[-1].strftime("%Y-%m-%d")
    item["next_date"] = (frequency.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    item["days"] = int(len(frequency))
    return item


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
            "analytics": build_analytics_payload(result.evaluations, result.payouts),
        }
    )


def _prize_range_analytics(
    prizes: pd.DataFrame, results: list[dict[str, Any]], top_k: int,
) -> dict[str, object]:
    """Score settled Prize Poisson forecasts under the shared chart contract."""
    settled = [item for item in results if pd.Timestamp(item["target_date"]) in prizes.index]
    if not settled:
        return build_analytics_payload({}, {})

    frequency, _ = XSMBDataLoader().preprocess_and_build_matrices(prizes.reset_index())
    targets = pd.DatetimeIndex([item["target_date"] for item in settled])
    predictions = [
        [entry["number"] for entry in item["forecasts"]["poisson_prize_90d"]]
        for item in settled
    ]
    truth = [
        np.repeat(np.arange(100), frequency.loc[target].to_numpy(dtype=np.int64))
        for target in targets
    ]
    evaluation = evaluate_test_set(truth, predictions, k=top_k, dates=targets)
    selected = np.zeros((len(targets), 100), dtype=bool)
    for row, numbers in enumerate(predictions):
        selected[row, [int(number) for number in numbers]] = True
    payout = evaluate_payouts(
        frequency.loc[targets].to_numpy(dtype=np.int64), selected,
        payout_mode="per_occurrence", stake=27_000, gross_payout=99_000,
    )
    return build_analytics_payload(
        {"poisson_prize_90d": evaluation}, {"poisson_prize_90d": payout},
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
        & (labels >= target - pd.Timedelta(days=PRIZE_POISSON_HISTORY_DAYS))
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
    model = PrizeLGBMPredictor(
        "poisson",
        history_days=PRIZE_POISSON_HISTORY_DAYS,
        n_jobs=1,
        n_estimators=PRIZE_POISSON_N_ESTIMATORS,
    ).fit(train)
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
    source_id: str, top_k: int, target_date: str | None,
    target_end_date: str | None = None, source_revision: str | None = None,
) -> dict[str, Any]:
    """Forecast one date or an inclusive range with independent walk-forward refits."""
    raw = _load_raw(source_id, source_revision)
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
    analytics = _prize_range_analytics(prizes, results, top_k)
    common = {
        "source": source_id,
        "model": "poisson_prize_90d",
        "top_k": top_k,
        "test_days": PRIZE_POISSON_HISTORY_DAYS,
        "model_parameters": {
            "history_days": PRIZE_POISSON_HISTORY_DAYS,
            "n_estimators": PRIZE_POISSON_N_ESTIMATORS,
        },
        "analytics": analytics,
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
        try:
            item = _source_item(source["id"])
        except Exception as exc:  # source health should not hide other sources
            item = {key: source[key] for key in ("id", "label", "kind")}
            item["error"] = str(exc)
        items.append(item)
    return jsonify({"sources": items})


@app.post("/api/sources/refresh")
def refresh_source() -> Any:
    """Pull, validate, and expose the latest GitHub feed revision."""
    try:
        previous = _source_item("live")
        revision = datetime.now(timezone.utc).isoformat()
        current = _source_item("live", revision)
        return jsonify({
            "source": current,
            "source_revision": revision,
            "refresh": {
                "previous_last_date": previous["last_date"],
                "current_last_date": current["last_date"],
                "added_days": max(0, int(current["days"]) - int(previous["days"])),
                "changed": (
                    current["last_date"] != previous["last_date"]
                    or current["days"] != previous["days"]
                ),
            },
        })
    except Exception:
        app.logger.exception("GitHub source refresh failed")
        return jsonify({"error": "Could not refresh and validate the GitHub data feed."}), 502


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
    if not 30 <= test_days <= 366:
        return jsonify({"error": "test_days must be between 30 and 366."}), 400
    target_date = body.get("target_date") or None
    target_end_date = body.get("target_end_date") or None
    source_revision = body.get("source_revision") or None
    if source_revision is not None and (
        not isinstance(source_revision, str) or len(source_revision) > 100
    ):
        return jsonify({"error": "source_revision must be a short string."}), 400
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
            return jsonify(_prize_poisson_90d_payload(
                source_id, top_k, target_date, target_end_date, source_revision,
            ))
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
    return jsonify({
        "service": "xsmb-forecast-api",
        "health": "/api/health",
        "sources": "/api/sources",
        "refresh_source": "/api/sources/refresh",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
