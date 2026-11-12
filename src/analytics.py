"""Stable daily analytics contract shared by the API and chart workspace."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .diagnostics import PayoutResult
from .evaluator import EvaluationResult


ANALYTICS_SCHEMA_VERSION = "1.0"
MODEL_LABELS = {
    "baseline": "Frequency baseline",
    "lightgbm": "Boosted trees",
    "regularized": "Regularized presence",
    "poisson": "Poisson count",
    "poisson_prize_90d": "Poisson Prize 90d",
}


def build_analytics_payload(
    evaluations: Mapping[str, EvaluationResult],
    payouts: Mapping[str, PayoutResult],
) -> dict[str, object]:
    """Serialize aligned daily evaluation and cash-flow results for charts.

    Field names are an API contract. Rates and ROI values are decimal ratios,
    currency values are VND, and ``hits`` always means distinct selected
    numbers present in that day's draw. Repeated winning occurrences are kept
    separately in ``paid_occurrences``.
    """
    if set(evaluations) != set(payouts):
        raise ValueError("Analytics evaluations and payouts must contain the same models.")

    models = []
    for model_id in evaluations:
        evaluation = evaluations[model_id]
        payout = payouts[model_id]
        daily_metrics = evaluation.per_day
        daily_money = payout.per_day
        if len(daily_metrics) != len(daily_money):
            raise ValueError(f"Analytics rows are misaligned for {model_id}.")

        cumulative_cost = daily_money["cost"].to_numpy(dtype=float).cumsum()
        cumulative_profit = daily_money["profit"].to_numpy(dtype=float).cumsum()
        daily_cost = daily_money["cost"].to_numpy(dtype=float)
        daily_profit = daily_money["profit"].to_numpy(dtype=float)
        daily_roi = np.divide(
            daily_profit, daily_cost,
            out=np.zeros_like(daily_profit), where=daily_cost != 0,
        )
        cumulative_roi = np.divide(
            cumulative_profit, cumulative_cost,
            out=np.zeros_like(cumulative_profit), where=cumulative_cost != 0,
        )

        series = []
        for position, (target_date, metrics) in enumerate(daily_metrics.iterrows()):
            money = daily_money.iloc[position]
            series.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "hits": int(metrics["hits"]),
                "hit": bool(metrics["hits"] > 0),
                "precision_at_k": float(metrics["precision_at_k"]),
                "recall_at_k": float(metrics["recall_at_k"]),
                "selections": int(money["selections"]),
                "paid_occurrences": int(money["paid_occurrences"]),
                "stake": float(money["cost"]),
                "receipts": float(money["receipts"]),
                "profit": float(money["profit"]),
                "roi": float(daily_roi[position]),
                "cumulative_profit": float(cumulative_profit[position]),
                "cumulative_roi": float(cumulative_roi[position]),
                "drawdown": float(money["drawdown"]),
            })

        evaluation_summary = evaluation.summary
        payout_summary = payout.summary
        models.append({
            "model_id": model_id,
            "model_name": MODEL_LABELS.get(model_id, model_id),
            "summary": {
                "days": int(evaluation_summary["n_days"]),
                "days_with_hit": int(evaluation_summary["days_with_hit"]),
                "hit_rate": float(evaluation_summary["hit_rate_at_k"]),
                "total_hits": int(evaluation_summary["total_hits"]),
                "mean_hits": float(evaluation_summary["mean_hits"]),
                "precision_at_k": float(evaluation_summary["precision_at_k"]),
                "recall_at_k": float(evaluation_summary["recall_at_k"]),
                "total_stake": float(payout_summary["cost"]),
                "total_receipts": float(payout_summary["receipts"]),
                "total_profit": float(payout_summary["profit"]),
                "roi": float(payout_summary["roi"]),
                "max_drawdown": float(payout_summary["max_drawdown"]),
            },
            "series": series,
        })

    dates = [row["date"] for model in models for row in model["series"]]
    return {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "granularity": "day",
        "currency": "VND",
        "period": {
            "start_date": min(dates) if dates else None,
            "end_date": max(dates) if dates else None,
        },
        "payout_assumptions": {
            "stake_per_selection": 27_000,
            "gross_payout_per_occurrence": 99_000,
            "payout_mode": "per_occurrence",
        },
        "models": models,
    }
