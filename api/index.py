"""Vercel-compatible HTTP API for the forecasting project.

This module serves the browser dashboard locally and is the stateless
serverless entry point used by Vercel. It exposes only the
operations needed by the browser dashboard.
"""

from __future__ import annotations

from datetime import date as calendar_date, datetime, timezone
from functools import lru_cache
import hashlib
import json
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
CONFIRMED_MODEL = "poisson_prize_90d"
CONFIRMED_TOP_K = 3
MAX_CONFIRMED_FORECASTS = 366

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


def _confirmation_digest(record: dict[str, Any]) -> str:
    """Detect accidental edits to a browser-stored confirmation record."""
    unsigned = {key: value for key, value in record.items() if key != "confirmation_id"}
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_iso_date(value: Any, field: str) -> pd.Timestamp:
    try:
        parsed = calendar_date.fromisoformat(str(value))
        if parsed.isoformat() != str(value):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format.") from exc
    return pd.Timestamp(parsed)


def _confirmed_numbers(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != CONFIRMED_TOP_K:
        raise ValueError("A confirmed forecast must contain exactly three numbers.")
    numbers = [str(number) for number in value]
    if any(len(number) != 2 or not number.isdigit() for number in numbers):
        raise ValueError("Confirmed numbers must be two-digit strings from 00 to 99.")
    if len(set(numbers)) != CONFIRMED_TOP_K:
        raise ValueError("Confirmed forecast numbers must be distinct.")
    return numbers


def _validated_confirmation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Each confirmed forecast must be an object.")
    if value.get("record_version") != 1:
        raise ValueError("Unsupported confirmed forecast record version.")
    if value.get("source") != "live" or value.get("model_id") != CONFIRMED_MODEL:
        raise ValueError("Only live Poisson Prize 90d confirmations are supported.")
    if value.get("top_k") != CONFIRMED_TOP_K:
        raise ValueError("Confirmed forecast research is fixed to Top-3.")
    target = _parse_iso_date(value.get("target_date"), "target_date")
    cutoff = _parse_iso_date(value.get("data_through_date"), "data_through_date")
    if cutoff + pd.Timedelta(days=1) != target:
        raise ValueError("A confirmed forecast must use the immediately preceding draw.")
    numbers = _confirmed_numbers(value.get("numbers"))
    confirmed_at = value.get("confirmed_at_utc")
    if not isinstance(confirmed_at, str) or not confirmed_at or len(confirmed_at) > 64:
        raise ValueError("confirmed_at_utc is invalid.")
    record = {
        "record_version": 1,
        "source": "live",
        "model_id": CONFIRMED_MODEL,
        "target_date": target.strftime("%Y-%m-%d"),
        "data_through_date": cutoff.strftime("%Y-%m-%d"),
        "top_k": CONFIRMED_TOP_K,
        "numbers": numbers,
        "confirmed_at_utc": confirmed_at,
    }
    expected = _confirmation_digest(record)
    if value.get("confirmation_id") != expected:
        raise ValueError("Confirmed forecast integrity check failed.")
    return {**record, "confirmation_id": expected}


def _confirmed_analytics_payload(
    records: list[dict[str, Any]], frequency: pd.DataFrame,
) -> dict[str, Any]:
    """Settle only explicit browser confirmations and build chart-compatible rows."""
    latest = frequency.index[-1]
    status_rows: list[dict[str, Any]] = []
    settled_records: list[dict[str, Any]] = []
    predictions: list[list[str]] = []
    targets: list[pd.Timestamp] = []
    actual_rows: list[np.ndarray] = []

    for record in sorted(records, key=lambda item: item["target_date"]):
        target = pd.Timestamp(record["target_date"])
        status = "settled" if target in frequency.index else (
            "pending" if target > latest else "outcome_unavailable"
        )
        row: dict[str, Any] = {**record, "status": status}
        if status == "settled":
            actual = frequency.loc[target].to_numpy(dtype=np.int64)
            counts = [int(actual[int(number)]) for number in record["numbers"]]
            occurrences = int(sum(counts))
            receipts = occurrences * 99_000
            row.update({
                "matching_numbers": [
                    {"number": number, "count": count}
                    for number, count in zip(record["numbers"], counts, strict=True)
                    if count > 0
                ],
                "hits": int(sum(count > 0 for count in counts)),
                "paid_occurrences": occurrences,
                "stake": CONFIRMED_TOP_K * 27_000,
                "receipts": receipts,
                "profit": receipts - CONFIRMED_TOP_K * 27_000,
            })
            settled_records.append(record)
            predictions.append(record["numbers"])
            targets.append(target)
            actual_rows.append(actual)
        status_rows.append(row)

    if settled_records:
        target_index = pd.DatetimeIndex(targets)
        truth = [
            np.repeat(np.arange(100), actual).tolist()
            for actual in actual_rows
        ]
        evaluation = evaluate_test_set(
            truth, predictions, k=CONFIRMED_TOP_K, dates=target_index,
        )
        selected = np.zeros((len(settled_records), 100), dtype=bool)
        for row_index, numbers in enumerate(predictions):
            selected[row_index, [int(number) for number in numbers]] = True
        payout = evaluate_payouts(
            np.stack(actual_rows), selected, payout_mode="per_occurrence",
            stake=27_000, gross_payout=99_000,
        )
        analytics = build_analytics_payload(
            {CONFIRMED_MODEL: evaluation}, {CONFIRMED_MODEL: payout},
        )
        series_by_date = {
            item["date"]: item for item in analytics["models"][0]["series"]
        }
        for row in status_rows:
            series = series_by_date.get(row["target_date"])
            if series is not None:
                series.update({
                    "confirmed_at_utc": row["confirmed_at_utc"],
                    "numbers": row["numbers"],
                    "matching_numbers": row["matching_numbers"],
                    "confirmation_id": row["confirmation_id"],
                })
    else:
        analytics = build_analytics_payload({}, {})

    analytics.update({
        "basis": "user_confirmed_top3_forecasts",
        "confirmed_count": len(records),
        "settled_count": len(settled_records),
        "pending_count": sum(row["status"] == "pending" for row in status_rows),
        "unavailable_count": sum(
            row["status"] == "outcome_unavailable" for row in status_rows
        ),
    })
    return {
        "source": "live",
        "latest_draw_date": latest.strftime("%Y-%m-%d"),
        "records": status_rows,
        "analytics": analytics,
    }


def _forecast_payload(result: Any, source_id: str, model: str, top_k: int, test_days: int) -> dict[str, Any]:
    return _json_safe(
        {
            "target_date": result.prediction_date.strftime("%Y-%m-%d"),
            "data_through_date": (
                result.data_through_date.strftime("%Y-%m-%d")
                if result.data_through_date is not None else None
            ),
            "forecast_horizon_days": result.forecast_horizon_days,
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
    available_origins = pd.DatetimeIndex(origins.unique())
    prior_origins = available_origins[available_origins < target]
    if prior_origins.empty:
        raise ValueError(
            f"No completed draw is available before {target:%Y-%m-%d}."
        )
    origin = prior_origins.max()
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
        raise ValueError(f"Could not build inference rows through {origin:%Y-%m-%d}.")
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
        "data_through_date": origin.strftime("%Y-%m-%d"),
        "forecast_horizon_days": int((target - origin).days),
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


@app.post("/api/confirmed/confirm")
def confirm_forecast() -> Any:
    """Timestamp one prospective Poisson Prize Top-3 forecast for browser storage."""
    body = request.get_json(silent=True) or {}
    try:
        if body.get("source", "live") != "live" or body.get("model_id") != CONFIRMED_MODEL:
            raise ValueError("Only live Poisson Prize 90d confirmations are supported.")
        if body.get("top_k") != CONFIRMED_TOP_K:
            raise ValueError("Confirmed forecast research is fixed to Top-3.")
        target = _parse_iso_date(body.get("target_date"), "target_date")
        cutoff = _parse_iso_date(body.get("data_through_date"), "data_through_date")
        numbers = _confirmed_numbers(body.get("numbers"))

        revision = datetime.now(timezone.utc).isoformat()
        frequency, _ = _load_matrices("live", revision)
        latest = frequency.index[-1]
        if cutoff != latest or target != latest + pd.Timedelta(days=1):
            raise ValueError(
                f"Confirmation is allowed only for the next unresolved draw "
                f"{latest + pd.Timedelta(days=1):%Y-%m-%d}."
            )
        record = {
            "record_version": 1,
            "source": "live",
            "model_id": CONFIRMED_MODEL,
            "target_date": target.strftime("%Y-%m-%d"),
            "data_through_date": cutoff.strftime("%Y-%m-%d"),
            "top_k": CONFIRMED_TOP_K,
            "numbers": numbers,
            "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return jsonify({
            "record": {**record, "confirmation_id": _confirmation_digest(record)},
            "storage": "browser_local",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Forecast confirmation failed")
        return jsonify({"error": "Could not confirm this forecast against the live source."}), 500


@app.post("/api/confirmed/settle")
def settle_confirmed_forecasts() -> Any:
    """Score only previously confirmed Top-3 records whose draws are available."""
    body = request.get_json(silent=True) or {}
    values = body.get("forecasts", [])
    try:
        if body.get("source", "live") != "live":
            raise ValueError("Only the live GitHub source is supported.")
        if not isinstance(values, list):
            raise ValueError("forecasts must be a list.")
        if len(values) > MAX_CONFIRMED_FORECASTS:
            raise ValueError(
                f"At most {MAX_CONFIRMED_FORECASTS} confirmed forecasts may be settled at once."
            )
        records = [_validated_confirmation(value) for value in values]
        dates = [record["target_date"] for record in records]
        if len(set(dates)) != len(dates):
            raise ValueError("Only one confirmed forecast is allowed per target date.")
        revision = datetime.now(timezone.utc).isoformat()
        frequency, _ = _load_matrices("live", revision)
        return jsonify(_json_safe(_confirmed_analytics_payload(records, frequency)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Confirmed forecast settlement failed")
        return jsonify({"error": "Could not settle confirmed forecasts."}), 500


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
        "confirm_forecast": "/api/confirmed/confirm",
        "settle_confirmed_forecasts": "/api/confirmed/settle",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
