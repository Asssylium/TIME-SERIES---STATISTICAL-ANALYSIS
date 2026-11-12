"""Build reproducible Poisson/Binary reports after the feature-origin audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics import block_bootstrap_mean_interval


MODELS = ("poisson", "binary")
STAKE = 81_000
GROSS = 99_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def max_miss_streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.astype(bool):
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def load_1095(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"date": str, "prediction_top3": str})
    raw = raw.loc[raw.strategy.isin(["poisson_full_27", "binary_full_27"])].copy()
    raw["model"] = raw.strategy.str.replace("_full_27", "", regex=False)
    result = raw.rename(columns={
        "prediction_top3": "prediction",
        "full27_distinct_hits": "distinct_hits",
        "full27_occurrences": "occurrences",
        "stake_vnd": "cost_vnd",
        "receipts_full27_vnd": "receipts_vnd",
        "profit_full27_vnd": "profit_vnd",
    })[["date", "model", "prediction", "distinct_hits", "occurrences",
        "cost_vnd", "receipts_vnd", "profit_vnd"]]
    return validate_daily(result)


def load_90(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"date": str, "prediction": str})
    result = raw.loc[raw.model.isin(MODELS), [
        "date", "model", "prediction", "distinct_hits", "occurrences",
        "cost_vnd", "receipts_vnd", "profit_vnd",
    ]].copy()
    return validate_daily(result)


def validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["date", "model"]).reset_index(drop=True)
    numeric = ["distinct_hits", "occurrences", "cost_vnd", "receipts_vnd", "profit_vnd"]
    result[numeric] = result[numeric].apply(pd.to_numeric)
    result["hit_day"] = (result.distinct_hits > 0).astype(np.uint8)
    if result.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate date/model rows.")
    if set(result.model) != set(MODELS):
        raise ValueError("Expected exactly Poisson and Binary rows.")
    if not (result.cost_vnd == STAKE).all():
        raise ValueError("Unexpected daily stake.")
    if not (result.receipts_vnd == result.occurrences * GROSS).all():
        raise ValueError("Receipt arithmetic failed.")
    if not (result.profit_vnd == result.receipts_vnd - result.cost_vnd).all():
        raise ValueError("Profit arithmetic failed.")
    return result


def metric_row(rows: pd.DataFrame, **labels: object) -> dict[str, object]:
    hits = rows.hit_day.to_numpy(dtype=float)
    low, high = block_bootstrap_mean_interval(
        hits, block_length=7, n_resamples=10_000, seed=42
    )
    cost = int(rows.cost_vnd.sum())
    profit = int(rows.profit_vnd.sum())
    return {
        **labels,
        "days": len(rows),
        "hit_days": int(rows.hit_day.sum()),
        "hit_rate": float(rows.hit_day.mean()),
        "hit_rate_block95_low": low,
        "hit_rate_block95_high": high,
        "distinct_hits": int(rows.distinct_hits.sum()),
        "occurrences": int(rows.occurrences.sum()),
        "max_miss_streak": max_miss_streak(rows.hit_day),
        "stake_vnd": cost,
        "receipts_vnd": int(rows.receipts_vnd.sum()),
        "profit_vnd": profit,
        "roi": profit / cost,
    }


def paired_row(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str,
               **labels: object) -> dict[str, object]:
    merged = a.merge(b, on="date", suffixes=("_a", "_b"), validate="one_to_one")
    delta = merged.hit_day_a.to_numpy(float) - merged.hit_day_b.to_numpy(float)
    low, high = block_bootstrap_mean_interval(
        delta, block_length=7, n_resamples=10_000, seed=42
    )
    a_only = int(((merged.hit_day_a == 1) & (merged.hit_day_b == 0)).sum())
    b_only = int(((merged.hit_day_a == 0) & (merged.hit_day_b == 1)).sum())
    overlaps = [
        len(set(x.split()) & set(y.split()))
        for x, y in zip(merged.prediction_a, merged.prediction_b)
    ]
    return {
        **labels,
        "days": len(merged),
        f"{a_name}_hit_rate": float(merged.hit_day_a.mean()),
        f"{b_name}_hit_rate": float(merged.hit_day_b.mean()),
        f"hit_rate_delta_{a_name}_minus_{b_name}": float(delta.mean()),
        "delta_block95_low": low,
        "delta_block95_high": high,
        f"only_{a_name}_hit_days": a_only,
        f"only_{b_name}_hit_days": b_only,
        "both_hit_days": int(((merged.hit_day_a == 1) & (merged.hit_day_b == 1)).sum()),
        "both_miss_days": int(((merged.hit_day_a == 0) & (merged.hit_day_b == 0)).sum()),
        "mcnemar_exact_two_sided_p": mcnemar_exact(a_only, b_only),
        f"occurrence_delta_{a_name}_minus_{b_name}": int(
            merged.occurrences_a.sum() - merged.occurrences_b.sum()
        ),
        f"profit_delta_{a_name}_minus_{b_name}_vnd": int(
            merged.profit_vnd_a.sum() - merged.profit_vnd_b.sum()
        ),
        "same_top3_set_days": int(sum(value == 3 for value in overlaps)),
        "mean_top3_overlap_of_3": float(np.mean(overlaps)),
        "zero_overlap_days": int(sum(value == 0 for value in overlaps)),
    }


def money(value: int) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,}".replace(",", ".") + "đ"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%".replace(".", ",")


def build_clean(daily: pd.DataFrame, source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    for period in ("2025", "2026", "all"):
        period_rows = daily if period == "all" else daily.loc[daily.date.str.startswith(period)]
        for model in MODELS:
            summary_rows.append(metric_row(
                period_rows.loc[period_rows.model == model], period=period, model=model
            ))
    summary = pd.DataFrame(summary_rows)
    paired_rows = []
    for period in ("2025", "2026", "all"):
        rows = daily if period == "all" else daily.loc[daily.date.str.startswith(period)]
        paired_rows.append(paired_row(
            rows.loc[rows.model == "binary"], rows.loc[rows.model == "poisson"],
            "binary", "poisson", period=period,
        ))
    paired = pd.DataFrame(paired_rows)
    protocol = {
        "status": "corrected_after_feature_history_origin_audit",
        "evaluation_start": daily.date.min(),
        "evaluation_end": daily.date.max(),
        "models": list(MODELS),
        "candidate_numbers": "00-99",
        "top_k": 3,
        "evaluated_results_per_day": 27,
        "history_calendar_days": 1095,
        "refit_policy": "before each calendar quarter",
        "feature_history_policy": "complete fixed source history before trimming output rows",
        "input_daily": str(source.resolve()),
        "input_daily_sha256": sha256(source),
        "generator_sha256": sha256(Path(__file__)),
        "payout": {"stake_per_day_vnd": STAKE, "gross_per_occurrence_vnd": GROSS},
    }
    lines = [
        "# Báo cáo đã hiệu chỉnh: Top-3 Poisson và Binary, cửa sổ 1095 ngày",
        "",
        "> Báo cáo này thay thế bản trước. Đặc trưng nhân quả được dựng từ cùng một mốc lịch sử nguồn cố định, nên kết quả của một quý không đổi khi replay riêng quý đó.",
        "",
        "Phạm vi: chọn 3 số trong `00–99`, chấm trên đủ 27 kết quả mỗi ngày, giữ nháy lặp; tái huấn luyện trước mỗi quý. Không có mô hình 23 giải và không có Frequency 90d trong báo cáo này.",
        "",
        "| Giai đoạn | Mô hình | Ngày hit | Tỷ lệ hit (block 95%) | Lượt trúng | Lãi/lỗ | ROI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.model.capitalize()} | {row.hit_days}/{row.days} | "
            f"{pct(row.hit_rate)} ({pct(row.hit_rate_block95_low)}–{pct(row.hit_rate_block95_high)}) | "
            f"{row.occurrences} | {money(row.profit_vnd)} | {pct(row.roi)} |"
        )
    all_pair = paired.loc[paired.period == "all"].iloc[0]
    lines += [
        "",
        "## So sánh ghép cặp toàn kỳ",
        "",
        f"Binary − Poisson về tỷ lệ ngày hit: **{pct(all_pair.hit_rate_delta_binary_minus_poisson)}** "
        f"(block 95% {pct(all_pair.delta_block95_low)} đến {pct(all_pair.delta_block95_high)}; "
        f"McNemar p={all_pair.mcnemar_exact_two_sided_p:.3f}).",
        "",
        "[Dữ liệu ngày](./daily.csv) · [Tổng hợp](./summary.csv) · [So sánh ghép cặp](./paired_comparison.csv) · [Giao thức](./protocol.json)",
    ]
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(daily90: pd.DataFrame, daily1095: pd.DataFrame,
                     source90: Path, source1095: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    dates = sorted(set(daily90.date) & set(daily1095.date))
    left = daily90.loc[daily90.date.isin(dates)].copy()
    right = daily1095.loc[daily1095.date.isin(dates)].copy()
    if left.date.nunique() != len(dates) or right.date.nunique() != len(dates):
        raise ValueError("Incomplete paired dates.")
    summary_rows = []
    quarter_rows = []
    for protocol, frame in (("90d_daily", left), ("1095d_quarterly", right)):
        for model in MODELS:
            model_rows = frame.loc[frame.model == model]
            summary_rows.append(metric_row(model_rows, model=model, protocol=protocol))
            for quarter, rows in model_rows.groupby(pd.PeriodIndex(model_rows.date, freq="Q")):
                quarter_rows.append(metric_row(
                    rows, quarter=str(quarter), model=model, protocol=protocol
                ))
    summary = pd.DataFrame(summary_rows)
    quarterly = pd.DataFrame(quarter_rows)
    paired = pd.DataFrame([
        paired_row(
            left.loc[left.model == model], right.loc[right.model == model],
            "90d", "1095d", model=model,
        ) for model in MODELS
    ])
    protocol = {
        "status": "corrected_1095d_feature_history_origin",
        "evaluation_start": min(dates), "evaluation_end": max(dates),
        "paired_evaluation_days": len(dates), "models": list(MODELS),
        "90d_input_sha256": sha256(source90),
        "1095d_input_sha256": sha256(source1095),
        "generator_sha256": sha256(Path(__file__)),
        "1095d_feature_history_policy": "complete fixed source history before trimming output rows",
    }
    lines = [
        "# So sánh đã hiệu chỉnh: 90d daily và 1095d quarterly",
        "",
        "> Bản này thay thế so sánh trước; nhánh 1095d đã sửa lỗi phụ thuộc ngày bắt đầu đánh giá.",
        "",
        "| Mô hình | Giao thức | Ngày hit | Tỷ lệ hit | Lượt trúng | Lãi/lỗ | ROI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model.capitalize()} | {row.protocol} | {row.hit_days}/{row.days} | "
            f"{pct(row.hit_rate)} | {row.occurrences} | {money(row.profit_vnd)} | {pct(row.roi)} |"
        )
    lines += ["", "## Chênh lệch ghép cặp 90d − 1095d", ""]
    for row in paired.itertuples(index=False):
        lines.append(
            f"- **{row.model.capitalize()}**: hit-rate {pct(row.hit_rate_delta_90d_minus_1095d)} "
            f"(block 95% {pct(row.delta_block95_low)} đến {pct(row.delta_block95_high)}; "
            f"McNemar p={row.mcnemar_exact_two_sided_p:.3f}); "
            f"lượt trúng {row.occurrence_delta_90d_minus_1095d:+d}; "
            f"lãi/lỗ {money(row.profit_delta_90d_minus_1095d_vnd)}."
        )
    lines += [
        "",
        "[Tổng hợp](./summary.csv) · [So sánh ghép cặp](./paired_comparison.csv) · [Theo quý](./quarterly.csv) · [Giao thức](./protocol.json)",
    ]
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(output / "quarterly.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-1095", type=Path, required=True)
    parser.add_argument("--daily-90", type=Path, required=True)
    parser.add_argument("--clean-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    args = parser.parse_args()
    daily1095 = load_1095(args.daily_1095)
    daily90 = load_90(args.daily_90)
    build_clean(daily1095, args.daily_1095, args.clean_output)
    build_comparison(daily90, daily1095, args.daily_90, args.daily_1095, args.comparison_output)


if __name__ == "__main__":
    main()
