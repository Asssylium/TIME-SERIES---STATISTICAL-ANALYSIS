"""Compare daily-refit 90-day and full-history Poisson on one 2026 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.count_models import CountLGBMPredictor
from src.data_loader import NUMBER_COLUMNS, XSMBDataLoader
from src.prize_features import PRIZE_COLUMNS, normalize_prizes


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "processed" / "xsmb_prizes_2026-10-30.csv"
DEFAULT_OUTPUT = (
    ROOT / "reports" / "research" / "poisson_daily_refit_comparison_2026"
)
START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-10-29")
TOP_K = 3
STAKE_PER_SELECTION_VND = 27_000
GROSS_PAYOUT_PER_OCCURRENCE_VND = 99_000
WINDOW_DAYS = 90


def _forecast_one_day(
    model: CountLGBMPredictor,
    dataset: pd.DataFrame,
    origin: pd.Timestamp,
) -> np.ndarray:
    dates = pd.DatetimeIndex(dataset.index.get_level_values(0))
    rows = dataset.loc[dates == origin].drop(columns=COUNT_TARGET)
    scores = model.predict_expected_counts(rows)
    matrix = scores.droplevel(0).reindex(NUMBER_COLUMNS).to_numpy(dtype=float)
    if matrix.shape != (len(NUMBER_COLUMNS),) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid Poisson scores for feature date {origin:%Y-%m-%d}.")
    return np.argsort(-matrix, kind="stable")[:TOP_K]


def run_comparison(
    source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_csv(
        source, dtype="string", keep_default_na=False, encoding="utf-8-sig"
    )
    prizes = normalize_prizes(raw)
    loader = XSMBDataLoader(prize_columns=PRIZE_COLUMNS)
    frequency, binary = loader.preprocess_and_build_matrices(prizes.reset_index())
    dataset = build_count_feature_dataset(
        frequency, binary, include_unlabeled=True
    )
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    label_dates = origins + pd.Timedelta(days=1)
    available = set(frequency.index)
    candidates = pd.date_range(START_DATE, END_DATE, freq="D")
    target_dates = pd.DatetimeIndex([
        date for date in candidates
        if date in available and date - pd.Timedelta(days=1) in available
    ])
    if target_dates.empty:
        raise ValueError("No eligible target dates in the configured evaluation period.")

    daily_rows: list[dict[str, object]] = []
    for position, target in enumerate(target_dates, start=1):
        actual = frequency.loc[target, list(NUMBER_COLUMNS)].to_numpy(
            dtype=np.int64
        )
        origin = target - pd.Timedelta(days=1)
        date_string = target.strftime("%Y-%m-%d")
        for model_name, window_days in (
            ("poisson_90d_daily", WINDOW_DAYS),
            ("poisson_all_history_daily", None),
        ):
            training_mask = label_dates < target
            if window_days is not None:
                training_mask &= label_dates >= target - pd.Timedelta(
                    days=window_days
                )
            training = dataset.loc[training_mask]
            labeled_rows = training.loc[training[COUNT_TARGET].notna()]
            if labeled_rows.empty:
                raise ValueError(
                    f"No training labels available for {model_name} on {target}."
                )
            model = CountLGBMPredictor().fit(training)
            if model.trained_through_ is None or model.trained_through_ >= target:
                raise ValueError(
                    f"{model_name} training cutoff overlaps the forecast origin "
                    f"{origin:%Y-%m-%d} for {target:%Y-%m-%d}."
                )
            chosen = _forecast_one_day(model, dataset, origin)
            matched = actual[chosen]
            daily_rows.append({
                "date": date_string,
                "model": model_name,
                "prediction": " ".join(f"{n:02d}" for n in chosen),
                "hit": bool(np.any(matched > 0)),
                "occurrences": int(matched.sum()),
                "training_labels": int(len(labeled_rows)),
                "training_label_start": (
                    pd.Timestamp(labeled_rows.index.get_level_values(0).min())
                    + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d"),
                "training_label_end": model.trained_through_.strftime("%Y-%m-%d"),
            })
        if position % 10 == 0 or position == len(target_dates):
            print(
                f"Completed {position}/{len(target_dates)} paired Count Poisson fits.",
                flush=True,
            )

    excluded = pd.DataFrame([
        {
            "date": date.strftime("%Y-%m-%d"),
            "reason": (
                "actual outcome missing" if date not in available
                else "preceding calendar day missing"
            ),
        }
        for date in candidates if date not in set(target_dates)
    ])
    daily = pd.DataFrame(daily_rows)
    return (
        daily,
        excluded,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def summarize(daily: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in daily.groupby("model", sort=True):
        days = len(group)
        cost = days * TOP_K * STAKE_PER_SELECTION_VND
        occurrences = int(group.occurrences.sum())
        receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
        profit = receipts - cost
        rows.append({
            "model": model,
            "days": days,
            "hit_days": int(group.hit.sum()),
            "hit_rate": float(group.hit.mean()),
            "occurrences": occurrences,
            "cost_vnd": cost,
            "receipts_vnd": receipts,
            "profit_vnd": profit,
            "roi": profit / cost,
        })
    return pd.DataFrame(rows)


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if frame.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate daily model result.")
    pivot = frame.pivot(index="date", columns="model", values=["hit", "occurrences"])
    if pivot.isna().any().any():
        raise ValueError("Models must be compared on identical target dates.")
    report_end = pivot.index.max()
    rows = []
    for period, group in pivot.groupby(pivot.index.to_period("M"), sort=True):
        month_start = period.start_time
        calendar_days = int(period.days_in_month)
        days_in_scope = (
            min(calendar_days, report_end.day)
            if month_start.year == report_end.year
            and month_start.month == report_end.month
            else calendar_days
        )
        cost = len(group) * TOP_K * STAKE_PER_SELECTION_VND
        row: dict[str, int | float | str] = {
            "month": str(period),
            "calendar_days": calendar_days,
            "days_in_scope": days_in_scope,
            "evaluated_days": len(group),
            "excluded_days": days_in_scope - len(group),
            "stake_vnd": cost,
        }
        for model in ("poisson_90d_daily", "poisson_all_history_daily"):
            hits = int(group[("hit", model)].sum())
            occurrences = int(group[("occurrences", model)].sum())
            receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
            profit = receipts - cost
            row[f"{model}_hit_days"] = hits
            row[f"{model}_hit_rate"] = hits / len(group)
            row[f"{model}_occurrences"] = occurrences
            row[f"{model}_receipts_vnd"] = receipts
            row[f"{model}_profit_vnd"] = profit
            row[f"{model}_roi"] = profit / cost
        row["profit_delta_90d_minus_all_history_vnd"] = (
            int(row["poisson_90d_daily_profit_vnd"])
            - int(row["poisson_all_history_daily_profit_vnd"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_tables(markdown: str) -> None:
    group: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|") and line.endswith("|"):
            group.append(line)
        elif group:
            widths = [len(item.strip("|").split("|")) for item in group]
            if len(set(widths)) != 1:
                raise ValueError(f"Markdown table column mismatch: {widths}")
            group = []
    if group:
        widths = [len(item.strip("|").split("|")) for item in group]
        if len(set(widths)) != 1:
            raise ValueError(f"Markdown table column mismatch: {widths}")


def write_report(
    output: Path,
    daily: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    source: Path,
    source_sha256: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    monthly = aggregate_monthly(daily)
    summary = summarize(daily, excluded)
    daily.to_csv(output / "daily.csv", index=False)
    monthly.to_csv(output / "monthly.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    excluded.to_csv(output / "excluded_dates.csv", index=False)

    lines = [
        "# So sánh Poisson 90 ngày và Poisson mặc định — refit hằng ngày",
        "",
        f"**Kỳ đánh giá:** {START_DATE:%Y-%m-%d}–{END_DATE:%Y-%m-%d}; "
        f"{len(daily) // 2} ngày ghép cặp.",
        "",
        "**So sánh cùng giao thức:** hai biến thể dùng cùng `CountLGBMPredictor`, "
        "feature, Top-3, dữ liệu kết quả và cùng refit trước từng ngày dự báo. "
        "Poisson 90d chỉ dùng nhãn trong 90 ngày lịch trước ngày dự báo; Poisson "
        "mặc định dùng toàn bộ nhãn có trước ngày dự báo như khi chạy "
        "`main.py --model poisson --top-k 3` mỗi ngày. Không mô hình nào dùng nhãn "
        "của ngày dự báo hoặc tương lai.",
        "",
        "**Tính tiền:** 27.000đ/số/ngày; tổng cược 81.000đ/ngày; trả gộp 99.000đ "
        "cho mỗi lượt xuất hiện, kể cả lặp.",
        "",
        "## Tổng hợp",
        "",
        "| Model | Ngày | Ngày hit | Tỷ lệ hit | Lượt | Chi | Thu | Lãi/lỗ | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.days} | {row.hit_days} | {row.hit_rate:.2%} | "
            f"{row.occurrences} | {row.cost_vnd:,}đ | {row.receipts_vnd:,}đ | "
            f"{row.profit_vnd:+,}đ | {row.roi:.2%} |"
        )
    lines.extend([
        "",
        "## Chi tiết từng tháng",
        "",
        "| Tháng | Ngày cược / ngày trong kỳ | Bị loại | "
        "Poisson 90d: hit (tỷ lệ); lượt; lãi/lỗ (ROI) | "
        "Poisson mặc định: hit (tỷ lệ); lượt; lãi/lỗ (ROI) | "
        "Chênh lãi 90d − mặc định |",
        "|---|---:|---:|---|---|---:|",
    ])
    for row in monthly.itertuples():
        lines.append(
            f"| {row.month} ({row.calendar_days}) | "
            f"{row.evaluated_days}/{row.days_in_scope} | {row.excluded_days} | "
            f"{row.poisson_90d_daily_hit_days}/{row.evaluated_days} "
            f"({row.poisson_90d_daily_hit_rate:.2%}); "
            f"{row.poisson_90d_daily_occurrences}; "
            f"{row.poisson_90d_daily_profit_vnd:+,}đ "
            f"({row.poisson_90d_daily_roi:.2%}) | "
            f"{row.poisson_all_history_daily_hit_days}/{row.evaluated_days} "
            f"({row.poisson_all_history_daily_hit_rate:.2%}); "
            f"{row.poisson_all_history_daily_occurrences}; "
            f"{row.poisson_all_history_daily_profit_vnd:+,}đ "
            f"({row.poisson_all_history_daily_roi:.2%}) | "
            f"{row.profit_delta_90d_minus_all_history_vnd:+,}đ |"
        )
    profit_delta = int(
        summary.loc[summary.model.eq("poisson_90d_daily"), "profit_vnd"].iloc[0]
        - summary.loc[
            summary.model.eq("poisson_all_history_daily"), "profit_vnd"
        ].iloc[0]
    )
    lines.extend([
        "",
        f"**Chênh lãi cộng dồn 90d − mặc định: {profit_delta:+,}đ.** "
        "Đây là kết quả hồi cứu, không phải bảo đảm lợi thế tương lai.",
        "",
        f"SHA-256 nguồn: `{source_sha256}`. "
        "[Dữ liệu ngày](./daily.csv) · [Theo tháng](./monthly.csv) · "
        "[Tổng hợp](./summary.csv) · [Ngày bị loại](./excluded_dates.csv) · "
        "[Giao thức](./protocol.json).",
        "",
    ])
    markdown = "\n".join(lines)
    _validate_tables(markdown)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    protocol = {
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "evaluation_start": START_DATE.strftime("%Y-%m-%d"),
        "evaluation_end": END_DATE.strftime("%Y-%m-%d"),
        "paired_days": int(len(daily) // 2),
        "refit_policy": "Before each target day for both models",
        "models": {
            "poisson_90d_daily": (
                "CountLGBMPredictor; 90 calendar days of labels strictly before target"
            ),
            "poisson_all_history_daily": "All labels strictly before target",
        },
        "model_parameters": CountLGBMPredictor().params,
        "top_k": TOP_K,
        "stake_per_selection_vnd": STAKE_PER_SELECTION_VND,
        "gross_payout_per_occurrence_vnd": GROSS_PAYOUT_PER_OCCURRENCE_VND,
        "excluded_dates": excluded.to_dict(orient="records"),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")
    daily, excluded, source_sha256 = run_comparison(source)
    write_report(
        args.output.resolve(), daily, excluded,
        source=source, source_sha256=source_sha256,
    )
    print(summarize(daily, excluded).to_string(index=False))
    print(f"\nReport written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
