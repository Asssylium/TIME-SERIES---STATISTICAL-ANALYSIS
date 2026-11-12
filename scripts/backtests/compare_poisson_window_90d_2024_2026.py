"""Compare 90-day and expanding-history Poisson models on 2024-2026."""

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
DEFAULT_SOURCE = ROOT / "data" / "raw" / "xsmb_2026-11-05.csv"
DEFAULT_OUTPUT = (
    ROOT / "reports" / "research" / "poisson_window_90d_vs_expanding_2024_2026"
)
START_DATE = pd.Timestamp("2024-01-01")
TOP_K = 3
STAKE_PER_SELECTION_VND = 27_000
GROSS_PAYOUT_PER_OCCURRENCE_VND = 99_000
MODEL_PARAMS = {
    "objective": "poisson",
    "n_estimators": 100,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 200,
    "reg_lambda": 20.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def _prediction_rows(
    model: CountLGBMPredictor,
    dataset: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    origins = target_dates - pd.Timedelta(days=1)
    origin_dates = pd.DatetimeIndex(dataset.index.get_level_values(0))
    rows = dataset.loc[origin_dates.isin(origins)].drop(columns=COUNT_TARGET)
    scores = model.predict_expected_counts(rows)
    score_frame = scores.rename("score").reset_index().pivot(
        index=rows.index.names[0],
        columns=rows.index.names[1],
        values="score",
    )
    score_matrix = score_frame.loc[origins, list(NUMBER_COLUMNS)].to_numpy(
        dtype=float
    )
    if score_matrix.shape != (len(target_dates), len(NUMBER_COLUMNS)):
        raise ValueError("Poisson predictions do not cover all requested dates.")
    if not np.isfinite(score_matrix).all():
        raise ValueError("Poisson model returned non-finite scores.")

    order = np.argsort(-score_matrix, axis=1, kind="stable")[:, :TOP_K]
    return pd.DataFrame({
        "date": target_dates,
        "prediction": [
            " ".join(f"{number:02d}" for number in picks) for picks in order
        ],
        "number_indexes": list(order),
    })


def forecast_quarterly(
    frequency: pd.DataFrame,
    binary: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    *,
    window_days: int | None,
    model_label: str,
) -> pd.DataFrame:
    """Refit at quarter boundaries using labels strictly before that quarter."""
    dataset = build_count_feature_dataset(
        frequency, binary, include_unlabeled=True
    )
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    label_dates = origins + pd.Timedelta(days=1)
    rows: list[dict[str, object]] = []

    for quarter, quarter_targets in pd.Series(target_dates).groupby(
        target_dates.to_period("Q"), sort=True
    ):
        dates = pd.DatetimeIndex(quarter_targets.to_list())
        quarter_start = quarter.start_time
        eligible = label_dates < quarter_start
        if window_days is not None:
            eligible &= label_dates >= quarter_start - pd.Timedelta(
                days=window_days
            )
        training = dataset.loc[eligible]
        labeled_count = int(training[COUNT_TARGET].notna().sum())
        if labeled_count < 100:
            raise ValueError(
                f"Insufficient training labels for {model_label} in {quarter}: "
                f"{labeled_count}."
            )
        model = CountLGBMPredictor(**{
            key: value for key, value in MODEL_PARAMS.items()
            if key != "objective"
        }).fit(training)
        if model.trained_through_ is None or model.trained_through_ >= quarter_start:
            raise ValueError(f"{model_label} training overlaps {quarter}.")
        training_labels = training.loc[
            training[COUNT_TARGET].notna()
        ]
        first_training_label = (
            pd.Timestamp(training_labels.index.get_level_values(0).min())
            + pd.Timedelta(days=1)
        )

        forecast = _prediction_rows(model, dataset, dates)
        actual = frequency.loc[dates, list(NUMBER_COLUMNS)].to_numpy(
            dtype=np.int64
        )
        for date, prediction, indexes, counts in zip(
            forecast.date,
            forecast.prediction,
            forecast.number_indexes,
            actual,
        ):
            selected = counts[indexes]
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "model": model_label,
                "prediction": prediction,
                "hit_days": bool(np.any(selected > 0)),
                "occurrences": int(selected.sum()),
                "training_labels": labeled_count,
                "training_label_start": first_training_label.strftime("%Y-%m-%d"),
                "training_label_end": model.trained_through_.strftime(
                    "%Y-%m-%d"
                ),
            })
        print(
            f"{model_label} {quarter}: fit {labeled_count:,} labeled "
            f"number-rows; forecast {len(dates)} dates.",
            flush=True,
        )
    return pd.DataFrame(rows)


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired daily predictions into monthly P&L."""
    required = {
        "date", "model", "hit_days", "occurrences",
    }
    if not required.issubset(daily.columns):
        raise ValueError(f"Daily results missing: {sorted(required - set(daily.columns))}")
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    expected_models = {"poisson_90d", "poisson_all_history"}
    if set(frame.model) != expected_models:
        raise ValueError("Daily data must contain exactly both Poisson variants.")
    if frame.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate daily prediction for a model.")
    pivot = frame.pivot(index="date", columns="model", values=["hit_days", "occurrences"])
    if pivot.isna().any().any():
        raise ValueError("The Poisson variants must have identical target dates.")

    report_end = frame.date.max()
    rows: list[dict[str, int | float | str]] = []
    for period, group in pivot.groupby(pivot.index.to_period("M"), sort=True):
        month_start = period.start_time
        calendar_days = int(period.days_in_month)
        days_in_scope = (
            min(calendar_days, report_end.day)
            if month_start.year == report_end.year
            and month_start.month == report_end.month
            else calendar_days
        )
        evaluated = len(group)
        cost = evaluated * TOP_K * STAKE_PER_SELECTION_VND
        row: dict[str, int | float | str] = {
            "month": str(period),
            "calendar_days": calendar_days,
            "days_in_scope": days_in_scope,
            "evaluated_days": evaluated,
            "excluded_days": days_in_scope - evaluated,
            "stake_vnd": cost,
        }
        for model in sorted(expected_models):
            hits = int(group[("hit_days", model)].sum())
            occurrences = int(group[("occurrences", model)].sum())
            receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
            profit = receipts - cost
            row[f"{model}_hit_days"] = hits
            row[f"{model}_hit_rate"] = hits / evaluated
            row[f"{model}_occurrences"] = occurrences
            row[f"{model}_receipts_vnd"] = receipts
            row[f"{model}_profit_vnd"] = profit
            row[f"{model}_roi"] = profit / cost
        row["profit_delta_90d_minus_all_history_vnd"] = (
            int(row["poisson_90d_profit_vnd"])
            - int(row["poisson_all_history_profit_vnd"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in daily.groupby("model", sort=True):
        days = len(group)
        cost = days * TOP_K * STAKE_PER_SELECTION_VND
        occurrences = int(group.occurrences.sum())
        receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
        rows.append({
            "model": model,
            "days": days,
            "hit_days": int(group.hit_days.sum()),
            "hit_rate": float(group.hit_days.mean()),
            "occurrences": occurrences,
            "cost_vnd": cost,
            "receipts_vnd": receipts,
            "profit_vnd": receipts - cost,
            "roi": (receipts - cost) / cost,
        })
    return pd.DataFrame(rows)


def _yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame.date).dt.year
    rows = []
    for (year, model), group in frame.groupby(["year", "model"], sort=True):
        days = len(group)
        cost = days * TOP_K * STAKE_PER_SELECTION_VND
        occurrences = int(group.occurrences.sum())
        receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
        rows.append({
            "year": int(year),
            "model": model,
            "days": days,
            "hit_days": int(group.hit_days.sum()),
            "hit_rate": float(group.hit_days.mean()),
            "occurrences": occurrences,
            "cost_vnd": cost,
            "receipts_vnd": receipts,
            "profit_vnd": receipts - cost,
            "roi": (receipts - cost) / cost,
        })
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    daily: pd.DataFrame,
    *,
    source: Path,
    source_sha256: str,
    excluded_dates: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    monthly = build_monthly(daily)
    summary = _summary(daily)
    yearly = _yearly_summary(daily)
    daily.to_csv(output / "daily.csv", index=False)
    monthly.to_csv(output / "monthly.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    yearly.to_csv(output / "yearly.csv", index=False)
    excluded_dates.to_csv(output / "excluded_dates.csv", index=False)

    period_end = pd.Timestamp(daily.date.max())
    report = [
        "# So sánh Poisson cửa sổ 90 ngày và Poisson toàn lịch sử",
        "",
        f"**Kỳ đánh giá:** {START_DATE:%d/%m/%Y}–{period_end:%d/%m/%Y}; "
        f"{len(daily) // 2:,} ngày dự báo ghép cặp.",
        "",
        "**Thiết lập:** cùng `CountLGBMPredictor` (LightGBM, objective Poisson), "
        "Top-3, feature và quy tắc trả. Hai mô hình đều tái huấn luyện trước mỗi quý "
        "để so sánh công bằng: Poisson 90d dùng nhãn của 90 ngày lịch ngay trước quý; "
        "Poisson toàn lịch sử dùng tất cả nhãn có trước quý, tương ứng chính sách "
        "lịch sử đầy đủ của lệnh `main.py --model poisson --top-k 3`. "
        "Trong từng quý, dự báo không được cập nhật bằng nhãn của quý đó.",
        "",
        "**Lưu ý:** bản Poisson 90d trong báo cáo này được fit đầu mỗi quý rồi giữ "
        "cố định trong quý để cùng lịch refit với model toàn lịch sử; vì vậy không "
        "phải bản rolling-refit hằng ngày trong báo cáo `model_comparison_90d_2026`.",
        "",
        "**Tính tiền:** cược 27.000đ/số, 81.000đ/ngày; trả gộp 99.000đ cho mỗi "
        "lượt xuất hiện, kể cả lặp. ROI = (thu − chi) / chi.",
        "",
        "## Tổng hợp toàn kỳ",
        "",
        "| Mô hình | Ngày | Ngày hit | Tỷ lệ hit | Lượt trúng | Tổng chi | Tổng thu | Lãi/lỗ | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.model} | {row.days} | {row.hit_days} | "
            f"{row.hit_rate:.2%} | {row.occurrences} | {row.cost_vnd:,}đ | "
            f"{row.receipts_vnd:,}đ | {row.profit_vnd:+,}đ | {row.roi:.2%} |"
        )
    report.extend([
        "",
        "## Tổng hợp theo năm",
        "",
        "| Năm | Mô hình | Ngày | Ngày hit | Tỷ lệ hit | Lượt trúng | Tổng chi | Tổng thu | Lãi/lỗ | ROI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in yearly.itertuples():
        report.append(
            f"| {row.year} | {row.model} | {row.days} | {row.hit_days} | "
            f"{row.hit_rate:.2%} | {row.occurrences} | {row.cost_vnd:,}đ | "
            f"{row.receipts_vnd:,}đ | {row.profit_vnd:+,}đ | {row.roi:.2%} |"
        )
    report.extend([
        "",
        "## Chi tiết từng tháng",
        "",
        "| Tháng | Ngày cược / ngày trong kỳ | Bị loại | "
        "Poisson 90d hit | Lượt | Chi | Thu | Lãi/lỗ | ROI | "
        "Poisson toàn lịch sử hit | Lượt | Chi | Thu | Lãi/lỗ | ROI | "
        "Chênh lãi 90d − toàn lịch sử |",
        "|" + "|".join(["---:"] * 16) + "|",
    ])
    for row in monthly.itertuples():
        report.append(
            f"| {row.month} ({row.calendar_days}) | "
            f"{row.evaluated_days}/{row.days_in_scope} | {row.excluded_days} | "
            f"{row.poisson_90d_hit_days}/{row.evaluated_days} "
            f"({row.poisson_90d_hit_rate:.2%}) | {row.poisson_90d_occurrences} | "
            f"{row.stake_vnd:,}đ | {row.poisson_90d_receipts_vnd:,}đ | "
            f"{row.poisson_90d_profit_vnd:+,}đ | {row.poisson_90d_roi:.2%} | "
            f"{row.poisson_all_history_hit_days}/{row.evaluated_days} "
            f"({row.poisson_all_history_hit_rate:.2%}) | "
            f"{row.poisson_all_history_occurrences} | {row.stake_vnd:,}đ | "
            f"{row.poisson_all_history_receipts_vnd:,}đ | "
            f"{row.poisson_all_history_profit_vnd:+,}đ | "
            f"{row.poisson_all_history_roi:.2%} | "
            f"{row.profit_delta_90d_minus_all_history_vnd:+,}đ |"
        )
    profit_90d = int(monthly.poisson_90d_profit_vnd.sum())
    profit_all = int(monthly.poisson_all_history_profit_vnd.sum())
    report.extend([
        "",
        f"**Chênh lãi cộng dồn Poisson 90d − Poisson toàn lịch sử: "
        f"{profit_90d - profit_all:+,}đ.**",
        "",
        "Đây là backtest hồi cứu; kết quả tháng ngắn hoặc tháng có ngày bị loại "
        "biến động mạnh và không chứng minh lợi thế trong tương lai. Các ngày bị loại "
        "là ngày thiếu kết quả ngày đó hoặc kết quả ngày liền trước, không tính là ngày thua.",
        "",
        f"SHA-256 nguồn: `{source_sha256}`. "
        "[Dữ liệu ngày](./daily.csv) · [Theo tháng](./monthly.csv) · "
        "[Theo năm](./yearly.csv) · "
        "[Tổng hợp](./summary.csv) · [Ngày bị loại](./excluded_dates.csv) · "
        "[Giao thức](./protocol.json).",
        "",
    ])
    markdown = "\n".join(report)
    _validate_markdown_tables(markdown)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    protocol = {
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "evaluation_start": START_DATE.strftime("%Y-%m-%d"),
        "evaluation_end": period_end.strftime("%Y-%m-%d"),
        "paired_days": int(len(daily) // 2),
        "models": {
            "poisson_90d": (
                "CountLGBMPredictor; labels in 90 calendar days before quarter"
            ),
            "poisson_all_history": (
                "CountLGBMPredictor; all labels before quarter"
            ),
        },
        "refit_frequency": "Before each calendar quarter for both models",
        "target_k": TOP_K,
        "model_parameters": MODEL_PARAMS,
        "stake_per_selection_vnd": STAKE_PER_SELECTION_VND,
        "gross_payout_per_occurrence_vnd": GROSS_PAYOUT_PER_OCCURRENCE_VND,
        "tie_policy": "Descending score, ascending number for ties",
        "no_leakage_policy": "Training labels strictly before the first target of each quarter",
        "excluded_date_count": len(excluded_dates),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_markdown_tables(markdown: str) -> None:
    groups: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|") and line.endswith("|"):
            current.append(line)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    for table in groups:
        widths = [len(line.strip("|").split("|")) for line in table]
        if len(set(widths)) != 1 or len(widths) < 2:
            raise ValueError(f"Markdown table has inconsistent columns: {widths}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    raw = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    prizes = normalize_prizes(raw)
    loader = XSMBDataLoader(prize_columns=PRIZE_COLUMNS)
    frequency, binary = loader.preprocess_and_build_matrices(prizes.reset_index())
    available = frequency.index
    end_date = available.max()
    candidate_dates = pd.date_range(START_DATE, end_date, freq="D")
    date_set = set(available)
    target_dates = pd.DatetimeIndex([
        date for date in candidate_dates
        if date in date_set and date - pd.Timedelta(days=1) in date_set
    ])
    excluded_dates = pd.DataFrame({
        "date": [date.strftime("%Y-%m-%d") for date in candidate_dates
                 if date not in set(target_dates)],
        "reason": [
            "actual outcome missing" if date not in date_set
            else "preceding calendar day missing"
            for date in candidate_dates if date not in set(target_dates)
        ],
    })
    if target_dates.empty:
        raise ValueError("No eligible target dates in the requested period.")

    predictions = []
    for model_label, window in (
        ("poisson_90d", 90),
        ("poisson_all_history", None),
    ):
        predictions.append(forecast_quarterly(
            frequency, binary, target_dates,
            window_days=window, model_label=model_label,
        ))
    daily = pd.concat(predictions, ignore_index=True)
    if daily.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate paired predictions found.")
    write_report(
        args.output.resolve(), daily, source=source,
        source_sha256=source_sha256, excluded_dates=excluded_dates,
    )
    print(_summary(daily).to_string(index=False))
    print(f"\nReport written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
