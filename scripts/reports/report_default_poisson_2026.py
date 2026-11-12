"""Backtest the default full-history Poisson model on a frozen 2026 holdout."""

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
    ROOT / "reports" / "research" / "default_poisson_2026"
)
START_DATE = pd.Timestamp("2026-01-01")
TOP_K = 3
STAKE_PER_SELECTION_VND = 27_000
GROSS_PAYOUT_PER_OCCURRENCE_VND = 99_000


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "hit", "occurrences"}
    if not required.issubset(daily.columns) or daily.empty:
        raise ValueError(f"Daily results must contain {sorted(required)} and be nonempty.")
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if frame.date.duplicated().any() or not frame.date.dt.year.eq(2026).all():
        raise ValueError("Expected unique daily forecasts in 2026.")
    if frame.hit.isna().any():
        raise ValueError("Hit results must not be missing.")
    frame["hit"] = frame.hit.astype(bool)
    frame["occurrences"] = pd.to_numeric(frame.occurrences, errors="raise").astype(int)
    if (frame.occurrences < 0).any():
        raise ValueError("Occurrences must be nonnegative.")

    report_end = frame.date.max()
    rows = []
    for month, group in frame.groupby(frame.date.dt.strftime("%Y-%m"), sort=True):
        month_start = pd.Timestamp(f"{month}-01")
        calendar_days = int(group.date.dt.days_in_month.iloc[0])
        days_in_scope = (
            min(calendar_days, report_end.day)
            if month_start.year == report_end.year
            and month_start.month == report_end.month
            else calendar_days
        )
        cost = len(group) * TOP_K * STAKE_PER_SELECTION_VND
        occurrences = int(group.occurrences.sum())
        receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
        profit = receipts - cost
        rows.append({
            "month": month,
            "calendar_days": calendar_days,
            "days_in_scope": days_in_scope,
            "evaluated_days": len(group),
            "excluded_days": days_in_scope - len(group),
            "hit_days": int(group.hit.sum()),
            "hit_rate": float(group.hit.mean()),
            "occurrences": occurrences,
            "stake_vnd": cost,
            "receipts_vnd": receipts,
            "profit_vnd": profit,
            "roi": profit / cost,
        })
    return pd.DataFrame(rows)


def validate_markdown_tables(markdown: str) -> None:
    table: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|") and line.endswith("|"):
            table.append(line)
        elif table:
            widths = [len(row.strip("|").split("|")) for row in table]
            if len(set(widths)) != 1:
                raise ValueError(f"Markdown table has inconsistent columns: {widths}")
            table = []
    if table:
        widths = [len(row.strip("|").split("|")) for row in table]
        if len(set(widths)) != 1:
            raise ValueError(f"Markdown table has inconsistent columns: {widths}")


def run_backtest(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    prizes = normalize_prizes(raw)
    loader = XSMBDataLoader(prize_columns=PRIZE_COLUMNS)
    frequency, binary = loader.preprocess_and_build_matrices(prizes.reset_index())
    dataset = build_count_feature_dataset(
        frequency, binary, include_unlabeled=True
    )
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    label_dates = origins + pd.Timedelta(days=1)
    training = dataset.loc[label_dates < START_DATE]
    if training[COUNT_TARGET].notna().sum() < 100:
        raise ValueError("Insufficient full-history training data before 2026.")

    model = CountLGBMPredictor().fit(training)
    if model.trained_through_ != START_DATE - pd.Timedelta(days=1):
        raise ValueError(
            "Training labels must end on 2025-12-31 before evaluating 2026."
        )

    available = set(frequency.index)
    end_date = frequency.index.max()
    calendar_targets = pd.date_range(START_DATE, end_date, freq="D")
    target_dates = pd.DatetimeIndex([
        date for date in calendar_targets
        if date in available and date - pd.Timedelta(days=1) in available
    ])
    if target_dates.empty:
        raise ValueError("No eligible 2026 target dates found.")

    feature_dates = target_dates - pd.Timedelta(days=1)
    feature_origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    prediction_rows = dataset.loc[
        feature_origins.isin(feature_dates)
    ].drop(columns=COUNT_TARGET)
    scores = model.predict_expected_counts(prediction_rows)
    score_frame = scores.rename("score").reset_index().pivot(
        index=prediction_rows.index.names[0],
        columns=prediction_rows.index.names[1],
        values="score",
    )
    matrix = score_frame.loc[feature_dates, list(NUMBER_COLUMNS)].to_numpy(
        dtype=float
    )
    if matrix.shape != (len(target_dates), len(NUMBER_COLUMNS)) or not np.isfinite(matrix).all():
        raise ValueError("Default Poisson produced invalid forecast scores.")
    selections = np.argsort(-matrix, axis=1, kind="stable")[:, :TOP_K]
    actual = frequency.loc[target_dates, list(NUMBER_COLUMNS)].to_numpy(
        dtype=np.int64
    )
    selected_counts = np.take_along_axis(actual, selections, axis=1)
    daily = pd.DataFrame({
        "date": target_dates.strftime("%Y-%m-%d"),
        "prediction": [
            " ".join(f"{number:02d}" for number in picks)
            for picks in selections
        ],
        "hit": np.any(selected_counts > 0, axis=1),
        "occurrences": selected_counts.sum(axis=1),
        "stake_vnd": TOP_K * STAKE_PER_SELECTION_VND,
        "receipts_vnd": selected_counts.sum(axis=1)
        * GROSS_PAYOUT_PER_OCCURRENCE_VND,
    })
    daily["profit_vnd"] = daily.receipts_vnd - daily.stake_vnd
    daily["training_label_end"] = model.trained_through_.strftime("%Y-%m-%d")
    eligible_set = set(target_dates)
    excluded = pd.DataFrame([
        {
            "date": date.strftime("%Y-%m-%d"),
            "reason": (
                "actual outcome missing" if date not in available
                else "preceding calendar day missing"
            ),
        }
        for date in calendar_targets if date not in eligible_set
    ])
    return daily, excluded, hashlib.sha256(source.read_bytes()).hexdigest()


def write_report(
    output: Path,
    daily: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    source: Path,
    source_sha256: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    monthly = build_monthly(daily)
    daily.to_csv(output / "daily.csv", index=False)
    monthly.to_csv(output / "monthly.csv", index=False)
    excluded.to_csv(output / "excluded_dates.csv", index=False)
    total_days = len(daily)
    total_hits = int(daily.hit.sum())
    total_occurrences = int(daily.occurrences.sum())
    total_cost = int(daily.stake_vnd.sum())
    total_receipts = int(daily.receipts_vnd.sum())
    total_profit = total_receipts - total_cost

    lines = [
        "# Poisson toàn lịch sử cố định — holdout năm 2026",
        "",
        f"**Kỳ đánh giá:** 01/01/2026–{daily.date.max()}; "
        f"{total_days} ngày hợp lệ, {len(excluded)} ngày bị loại.",
        "",
        "**Giao thức:** `CountLGBMPredictor` mặc định của "
        "`main.py --model poisson --top-k 3`, fit đúng một lần bằng toàn bộ nhãn "
        "lịch sử có trước 01/01/2026 (nhãn huấn luyện đến 31/12/2025). "
        "Trong năm 2026, feature được cập nhật theo kết quả ngày liền trước, "
        "nhưng không huấn luyện lại model. Đây là holdout một năm để kiểm định "
        "ngoài mẫu; **không tương đương báo cáo Poisson cửa sổ 90 ngày refit hằng "
        "ngày, cũng không phải replay lệnh production chạy lại mỗi ngày.**",
        "",
        "**Tính tiền:** chọn 3 số/ngày; cược 27.000đ/số (81.000đ/ngày); "
        "trả gộp 99.000đ cho mỗi lần xuất hiện, kể cả lặp.",
        "",
        "## Tổng kết",
        "",
        "| Ngày | Ngày hit | Tỷ lệ hit | Lượt trúng | Tổng chi | Tổng thu | Lãi/lỗ | ROI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {total_days} | {total_hits} | {total_hits / total_days:.2%} | "
        f"{total_occurrences} | {total_cost:,}đ | {total_receipts:,}đ | "
        f"{total_profit:+,}đ | {total_profit / total_cost:.2%} |",
        "",
        "## Chi tiết từng tháng",
        "",
        "| Tháng | Ngày cược / ngày trong kỳ | Bị loại | Ngày hit | Tỷ lệ hit | "
        "Lượt trúng | Chi | Thu | Lãi/lỗ | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in monthly.itertuples():
        lines.append(
            f"| {row.month} ({row.calendar_days}) | "
            f"{row.evaluated_days}/{row.days_in_scope} | {row.excluded_days} | "
            f"{row.hit_days}/{row.evaluated_days} | {row.hit_rate:.2%} | "
            f"{row.occurrences} | {row.stake_vnd:,}đ | {row.receipts_vnd:,}đ | "
            f"{row.profit_vnd:+,}đ | {row.roi:.2%} |"
        )
    lines.extend([
        "",
        "Ngày thiếu kết quả thực tế hoặc thiếu kết quả ngày liền trước bị loại, "
        "không tính là ngày cược/thua. Đây là kết quả hồi cứu, không đảm bảo "
        "hiệu quả tương lai.",
        "",
        f"SHA-256 nguồn: `{source_sha256}`. "
        "[Dữ liệu ngày](./daily.csv) · [Theo tháng](./monthly.csv) · "
        "[Ngày bị loại](./excluded_dates.csv) · [Giao thức](./protocol.json).",
        "",
    ])
    markdown = "\n".join(lines)
    validate_markdown_tables(markdown)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    protocol = {
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "evaluation_start": START_DATE.strftime("%Y-%m-%d"),
        "evaluation_end": daily.date.max(),
        "training_policy": "All labeled history strictly before 2026-01-01",
        "training_label_end": "2025-12-31",
        "refit_policy": "Fit once before holdout; no refit during 2026",
        "features_policy": "Causal features updated using completed previous draws",
        "model": "src.count_models.CountLGBMPredictor defaults, fitted once before 2026",
        "top_k": TOP_K,
        "stake_per_selection_vnd": STAKE_PER_SELECTION_VND,
        "gross_payout_per_occurrence_vnd": GROSS_PAYOUT_PER_OCCURRENCE_VND,
        "tie_policy": "Descending score, ascending number for ties",
        "days": total_days,
        "excluded_days": len(excluded),
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
    daily, excluded, source_sha256 = run_backtest(source)
    write_report(
        args.output.resolve(), daily, excluded,
        source=source, source_sha256=source_sha256,
    )
    print(f"Days: {len(daily)}; hits: {int(daily.hit.sum())}; "
          f"occurrences: {int(daily.occurrences.sum())}; "
          f"profit: {int(daily.profit_vnd.sum()):+,} VND")
    print(f"Report written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
