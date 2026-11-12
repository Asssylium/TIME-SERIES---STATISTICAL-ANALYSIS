"""Build a standalone monthly 2026 report from paired model forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORY = (
    ROOT / "reports" / "research" / "short_horizon_persistence_2005_2026"
)
TOP_K = 3
STAKE_PER_NUMBER_VND = 27_000
GROSS_PAYOUT_PER_OCCURRENCE_VND = 99_000


def build_monthly_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired daily forecasts into monthly cash flows."""
    required = {
        "date",
        "frequency_prediction",
        "frequency_occurrences",
        "frequency_hit",
        "poisson_prediction",
        "poisson_occurrences",
        "poisson_hit",
    }
    if not isinstance(daily, pd.DataFrame) or not required.issubset(daily.columns):
        raise ValueError(f"daily data is missing columns: {sorted(required - set(daily.columns))}")
    frame = daily.loc[:, sorted(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    frame = frame.loc[frame.date.dt.year.eq(2026)].sort_values("date")
    if frame.empty or frame.date.duplicated().any():
        raise ValueError("Expected unique paired forecast dates in 2026.")

    for model in ("frequency", "poisson"):
        occurrence_column = f"{model}_occurrences"
        hit_column = f"{model}_hit"
        prediction_column = f"{model}_prediction"
        occurrences = pd.to_numeric(frame[occurrence_column], errors="raise")
        if (
            occurrences.isna().any()
            or (occurrences < 0).any()
            or not np.equal(occurrences, np.floor(occurrences)).all()
        ):
            raise ValueError(f"{occurrence_column} must contain nonnegative integers.")
        frame[occurrence_column] = occurrences.astype(np.int64)
        if frame[hit_column].isna().any():
            raise ValueError(f"{hit_column} must not contain missing values.")
        frame[hit_column] = frame[hit_column].astype(bool)
        picks = frame[prediction_column].astype("string").str.split()
        valid_picks = picks.map(
            lambda values: len(values) == TOP_K
            and len(set(values)) == TOP_K
            and all(len(value) == 2 and value.isdigit() for value in values)
        )
        if not valid_picks.all():
            raise ValueError(f"{prediction_column} must contain three unique 00-99 picks.")

    frame["month"] = frame.date.dt.strftime("%Y-%m")
    report_end = frame.date.max()
    rows = []
    for month, group in frame.groupby("month", sort=True):
        month_start = pd.Timestamp(f"{month}-01")
        calendar_days = int(group.date.dt.days_in_month.iloc[0])
        days_in_scope = (
            min(calendar_days, report_end.day)
            if month_start.year == report_end.year
            and month_start.month == report_end.month
            else calendar_days
        )
        row: dict[str, int | float | str] = {
            "month": month,
            "calendar_days": calendar_days,
            "days_in_scope": days_in_scope,
            "evaluated_days": int(len(group)),
            "excluded_days": int(days_in_scope - len(group)),
            "stake_vnd": int(
                len(group) * TOP_K * STAKE_PER_NUMBER_VND
            ),
        }
        for model in ("frequency", "poisson"):
            hits = int(group[f"{model}_hit"].sum())
            occurrences = int(group[f"{model}_occurrences"].sum())
            receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE_VND
            profit = receipts - int(row["stake_vnd"])
            row[f"{model}_hit_days"] = hits
            row[f"{model}_hit_rate"] = hits / len(group)
            row[f"{model}_occurrences"] = occurrences
            row[f"{model}_occurrences_per_day"] = occurrences / len(group)
            row[f"{model}_receipts_vnd"] = receipts
            row[f"{model}_profit_vnd"] = profit
            row[f"{model}_roi"] = profit / int(row["stake_vnd"])
        row["profit_delta_frequency_minus_poisson_vnd"] = (
            int(row["frequency_profit_vnd"]) - int(row["poisson_profit_vnd"])
        )
        row["occurrence_delta_frequency_minus_poisson"] = (
            int(row["frequency_occurrences"]) - int(row["poisson_occurrences"])
        )
        row["hit_rate_delta_frequency_minus_poisson"] = (
            float(row["frequency_hit_rate"]) - float(row["poisson_hit_rate"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _format_money(value: int) -> str:
    return f"{value:+,}đ" if value else "0đ"


def validate_markdown_tables(markdown: str) -> None:
    """Reject rendered Markdown tables whose rows have mismatched cell counts."""
    table_lines: list[str] = []

    def validate_group(lines: list[str]) -> None:
        if not lines:
            return
        widths = [len(line.strip().strip("|").split("|")) for line in lines]
        if len(set(widths)) != 1:
            raise ValueError(f"Markdown table rows have inconsistent column counts: {widths}")
        separators = [
            line for line in lines
            if all(
                cell.strip().replace(":", "").replace("-", "") == ""
                and cell.strip().count("-") >= 3
                for cell in line.strip().strip("|").split("|")
            )
        ]
        if len(separators) != 1 or lines.index(separators[0]) != 1:
            raise ValueError("Each Markdown table needs one separator directly after its header.")

    for line in markdown.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines.append(line.strip())
        else:
            validate_group(table_lines)
            table_lines = []
    validate_group(table_lines)


def write_report(daily_path: Path, output_directory: Path) -> tuple[Path, Path]:
    """Validate the source protocol and write standalone monthly CSV/Markdown."""
    protocol_path = daily_path.with_name("poisson_comparison_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("top_k") != TOP_K:
        raise ValueError("Paired source report does not use Top-3 predictions.")
    if (
        protocol.get("stake_per_number_vnd") != STAKE_PER_NUMBER_VND
        or protocol.get("gross_payout_per_occurrence_vnd")
        != GROSS_PAYOUT_PER_OCCURRENCE_VND
    ):
        raise ValueError("Paired source payout rules do not match this monthly report.")

    daily = pd.read_csv(daily_path)
    monthly = build_monthly_comparison(daily)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "monthly_comparison_2026.csv"
    report_path = output_directory / "monthly_comparison_2026.md"
    monthly.to_csv(csv_path, index=False)

    year_end = pd.Timestamp(f"{monthly.month.max()}-01") + pd.offsets.MonthEnd(0)
    as_of = min(year_end, pd.Timestamp(protocol["comparison_period_end"]))
    data_start = pd.Timestamp(f"{monthly.month.min()}-01")
    monthly_header = (
        "| Tháng (ngày lịch) | Ngày cược / ngày trong kỳ | Ngày bị loại | "
        "Frequency 90d: hit | Lượt | Chi | Thu | Lãi/lỗ | ROI | "
        "Poisson: hit | Lượt | Chi | Thu | Lãi/lỗ | ROI | "
        "Chênh lãi F90d − Poisson |"
    )
    monthly_separator = "|" + "|".join(
        ["---:"] * (monthly_header.count("|") - 1)
    ) + "|"
    lines = [
        "# So sánh lãi/lỗ theo tháng năm 2026",
        "",
        f"**Kỳ báo cáo:** {data_start:%d/%m/%Y}–{as_of:%d/%m/%Y}. "
        "Chỉ tính ngày có dự báo, kết quả thực tế và kết quả ngày liền trước hợp lệ; "
        "ngày bị loại không được tính là ngày cược hay ngày thua.",
        "",
        "**Quy tắc chung:** mỗi mô hình chọn 3 số/ngày; cược 27.000đ/số, "
        "tổng 81.000đ/ngày được đánh giá; trả gộp 99.000đ cho mỗi lần xuất hiện, "
        "kể cả lặp. ROI = (tiền thu − tiền cược) / tiền cược.",
        "",
        monthly_header,
        monthly_separator,
    ]
    for row in monthly.itertuples(index=False):
        lines.append(
            f"| {row.month} ({row.calendar_days}) "
            f"| {row.evaluated_days}/{row.days_in_scope} "
            f"| {row.excluded_days} "
            f"| {row.frequency_hit_days}/{row.evaluated_days} "
            f"({row.frequency_hit_rate:.2%}) | {row.frequency_occurrences} "
            f"| {row.stake_vnd:,}đ | {row.frequency_receipts_vnd:,}đ "
            f"| {_format_money(row.frequency_profit_vnd)} | {row.frequency_roi:.2%} "
            f"| {row.poisson_hit_days}/{row.evaluated_days} "
            f"({row.poisson_hit_rate:.2%}) | {row.poisson_occurrences} "
            f"| {row.stake_vnd:,}đ | {row.poisson_receipts_vnd:,}đ "
            f"| {_format_money(row.poisson_profit_vnd)} | {row.poisson_roi:.2%} "
            f"| {_format_money(row.profit_delta_frequency_minus_poisson_vnd)} |"
        )

    total_days = int(monthly.evaluated_days.sum())
    total_stake = int(monthly.stake_vnd.sum())
    lines.extend([
        "",
        "## Cộng dồn trong kỳ",
        "",
        "| Mô hình | Ngày cược | Ngày hit | Tỷ lệ hit | Lượt trúng | Tổng chi | Tổng thu | Lãi/lỗ | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    totals: dict[str, dict[str, int | float]] = {}
    for model, display in (
        ("frequency", "Frequency 90d"),
        ("poisson", "Poisson"),
    ):
        hits = int(monthly[f"{model}_hit_days"].sum())
        occurrences = int(monthly[f"{model}_occurrences"].sum())
        receipts = int(monthly[f"{model}_receipts_vnd"].sum())
        profit = receipts - total_stake
        roi = profit / total_stake
        totals[model] = {
            "hits": hits,
            "occurrences": occurrences,
            "receipts": receipts,
            "profit": profit,
            "roi": roi,
        }
        lines.append(
            f"| {display} | {total_days} | {hits} | {hits / total_days:.2%} "
            f"| {occurrences} | {total_stake:,}đ | {receipts:,}đ "
            f"| {_format_money(profit)} | {roi:.2%} |"
        )
    profit_delta = int(monthly.profit_delta_frequency_minus_poisson_vnd.sum())
    lines.extend([
        "",
        f"**Chênh lãi/lỗ cộng dồn Frequency 90d − Poisson: "
        f"{_format_money(profit_delta)}.** Frequency thắng ở "
        f"{int((monthly.profit_delta_frequency_minus_poisson_vnd > 0).sum())} tháng, "
        f"Poisson thắng ở {int((monthly.profit_delta_frequency_minus_poisson_vnd < 0).sum())} tháng; "
        f"hòa {int((monthly.profit_delta_frequency_minus_poisson_vnd == 0).sum())} tháng.",
        "",
        "## Cách đọc",
        "",
        "Đây là kết quả hồi cứu từ cùng các ngày dự báo ghép cặp, không phải bằng chứng "
        "mô hình nào sẽ thắng trong các tháng tới. Tháng có ít ngày, nhất là tháng 11 "
        "mới đến ngày 04/11, có P&L dao động lớn và không nên so trực tiếp ROI như thể "
        "đã đủ một tháng. Quy tắc dự báo và mức cược không thay đổi theo kết quả.",
        "",
        f"Nguồn dữ liệu dự báo: `{daily_path.name}`. Giao thức gốc: "
        f"`{protocol_path.name}`.",
        "",
        f"[CSV theo tháng](./{csv_path.name})",
    ])
    markdown = "\n".join(lines) + "\n"
    validate_markdown_tables(markdown)
    report_path.write_text(markdown, encoding="utf-8")
    return report_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report, csv_path = write_report(
        directory / "poisson_comparison_daily.csv", directory
    )
    print(f"Monthly report: {report}")
    print(f"Monthly CSV: {csv_path}")


if __name__ == "__main__":
    main()
