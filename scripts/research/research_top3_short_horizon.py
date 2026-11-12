"""Long-horizon research for short-term frequency in two-digit Top-3 picks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import XSMBDataLoader
from src.diagnostics import block_bootstrap_mean_interval
from src.prize_features import PRIZE_COLUMNS, normalize_prizes


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "raw" / "xsmb_2026-11-05.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "research" / "short_horizon_persistence_2005_2026"
TOP_K = 3
DRAW_COUNT = 27
HISTORY_DAYS = 1825
WEEKDAY_PRIOR_DAYS = 156
STAKE_PER_NUMBER = 27_000
GROSS_PAYOUT_PER_OCCURRENCE = 99_000
STRATEGIES = (
    "frequency_90d",
    "frequency_365d",
    "frequency_5y",
    "expanding",
    "weekday_5y",
    "weekday_shrink_3y",
)


def expected_top_k_metrics(
    scores: np.ndarray, actual: np.ndarray, top_k: int = TOP_K
) -> tuple[float, float, np.ndarray, int, bool, int]:
    """Return tie-neutral expected hits and the ascending-ID deterministic pick."""
    values = np.asarray(scores, dtype=float)
    counts = np.asarray(actual, dtype=np.int64)
    if (
        values.shape != (100,)
        or not np.isfinite(values).all()
        or counts.shape != (100,)
        or (counts < 0).any()
        or isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, (int, np.integer))
        or not 1 <= top_k <= 100
    ):
        raise ValueError("scores and actual must have 100 entries and valid top_k.")

    threshold = np.partition(values, len(values) - top_k)[len(values) - top_k]
    above = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)
    tie_slots = top_k - len(above)
    expected_occurrences = float(
        counts[above].sum() + tie_slots * counts[tied].sum() / len(tied)
    )
    if np.any(counts[above] > 0):
        expected_hit_probability = 1.0
    else:
        tied_hits = int(np.count_nonzero(counts[tied]))
        expected_hit_probability = 1.0 - (
            math.comb(len(tied) - tied_hits, tie_slots)
            / math.comb(len(tied), tie_slots)
        )

    order = np.argsort(-values, kind="stable")[:top_k]
    stable_occurrences = int(counts[order].sum())
    stable_hit = bool(np.any(counts[order] > 0))
    return (
        expected_occurrences,
        float(expected_hit_probability),
        order,
        stable_occurrences,
        stable_hit,
        len(tied),
    )


def _random_hit_probability(actual: np.ndarray) -> float:
    distinct = int(np.count_nonzero(actual))
    return 1.0 - math.comb(100 - distinct, TOP_K) / math.comb(100, TOP_K)


def build_daily_results(frequency: pd.DataFrame) -> pd.DataFrame:
    """Evaluate causal rolling and weekday-conditioned Top-3 strategies."""
    dates = frequency.index
    values = frequency.to_numpy(dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 100:
        raise ValueError("frequency must have exactly 100 number columns.")
    if (values < 0).any() or not np.equal(values.sum(axis=1), DRAW_COUNT).all():
        raise ValueError("Every frequency row must contain exactly 27 results.")

    prefix = np.zeros((len(values) + 1, 100), dtype=np.int64)
    np.cumsum(values, axis=0, dtype=np.int64, out=prefix[1:])
    rows: list[dict[str, object]] = []
    first_evaluation_date = dates[0] + pd.Timedelta(days=HISTORY_DAYS)

    for target_index in range(1, len(dates)):
        target_date = dates[target_index]
        if (
            target_date - dates[target_index - 1] != pd.Timedelta(days=1)
            or target_date < first_evaluation_date
        ):
            continue

        lower = {
            days: dates.searchsorted(target_date - pd.Timedelta(days=days), side="left")
            for days in (90, 365, HISTORY_DAYS)
        }
        history = prefix[target_index] - prefix[lower[HISTORY_DAYS]]
        pooled_rate = history / (target_index - lower[HISTORY_DAYS])
        history_90 = prefix[target_index] - prefix[lower[90]]
        history_365 = prefix[target_index] - prefix[lower[365]]

        weekday_mask = (
            (dates[:target_index].day_of_week == target_date.day_of_week)
            & (np.arange(target_index) >= lower[HISTORY_DAYS])
        )
        weekday_history = values[:target_index][weekday_mask]
        weekday_count = len(weekday_history)
        if (
            target_index == lower[90]
            or target_index == lower[365]
            or target_index == lower[HISTORY_DAYS]
            or weekday_count == 0
        ):
            raise ValueError(
                f"Insufficient history for a complete forecast on {target_date:%Y-%m-%d}."
            )
        weekday_totals = weekday_history.sum(axis=0, dtype=np.int64)
        scores = {
            "frequency_90d": history_90 / (target_index - lower[90]),
            "frequency_365d": history_365 / (target_index - lower[365]),
            "frequency_5y": pooled_rate,
            "expanding": prefix[target_index] / target_index,
            "weekday_5y": weekday_totals / weekday_count,
            "weekday_shrink_3y": (
                weekday_totals + WEEKDAY_PRIOR_DAYS * pooled_rate
            ) / (weekday_count + WEEKDAY_PRIOR_DAYS),
        }
        actual = values[target_index]
        random_hit = _random_hit_probability(actual)

        for strategy in STRATEGIES:
            (
                expected_occurrences,
                expected_hit,
                order,
                stable_occurrences,
                stable_hit,
                cutoff_tie_size,
            ) = expected_top_k_metrics(scores[strategy], actual)
            rows.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "strategy": strategy,
                "prediction_stable": " ".join(f"{number:02d}" for number in order),
                "training_days": target_index - lower[HISTORY_DAYS],
                "weekday_training_days": weekday_count,
                "cutoff_tie_size": cutoff_tie_size,
                "expected_occurrences_random_ties": expected_occurrences,
                "expected_hit_probability_random_ties": expected_hit,
                "stable_occurrences": stable_occurrences,
                "stable_hit": stable_hit,
                "random_hit_probability": random_hit,
            })

    if not rows:
        raise ValueError("No consecutive evaluation dates with sufficient history.")
    return pd.DataFrame(rows)


def build_summaries(
    daily: pd.DataFrame, *, bootstrap_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize strategies, calendar years, and pre-specified broad eras."""
    if bootstrap_resamples < 200:
        raise ValueError("bootstrap_resamples must be at least 200.")

    summary_rows = []
    for strategy in STRATEGIES:
        group = daily.loc[daily.strategy.eq(strategy)].sort_values("date")
        occurrence = group.expected_occurrences_random_ties.to_numpy(dtype=float)
        hit_delta = (
            group.expected_hit_probability_random_ties.to_numpy(dtype=float)
            - group.random_hit_probability.to_numpy(dtype=float)
        )
        occurrence_low, occurrence_high = block_bootstrap_mean_interval(
            occurrence - DRAW_COUNT * TOP_K / 100,
            block_length=7,
            n_resamples=bootstrap_resamples,
            seed=42,
        )
        hit_low, hit_high = block_bootstrap_mean_interval(
            hit_delta,
            block_length=7,
            n_resamples=bootstrap_resamples,
            seed=43,
        )
        stable_occurrence = group.stable_occurrences.to_numpy(dtype=float)
        mean_occurrence = float(occurrence.mean())
        expected_profit_per_day = (
            mean_occurrence * GROSS_PAYOUT_PER_OCCURRENCE
            - TOP_K * STAKE_PER_NUMBER
        )
        summary_rows.append({
            "strategy": strategy,
            "days": len(group),
            "expected_occurrences_per_day": mean_occurrence,
            "delta_occurrences_vs_uniform": mean_occurrence - DRAW_COUNT * TOP_K / 100,
            "occurrence_delta_block95_low_7d": occurrence_low,
            "occurrence_delta_block95_high_7d": occurrence_high,
            "expected_hit_rate": float(
                group.expected_hit_probability_random_ties.mean()
            ),
            "random_hit_rate_conditional": float(
                group.random_hit_probability.mean()
            ),
            "hit_rate_delta_block95_low_7d": hit_low,
            "hit_rate_delta_block95_high_7d": hit_high,
            "stable_tie_occurrences_per_day": float(stable_occurrence.mean()),
            "expected_profit_per_day": expected_profit_per_day,
            "expected_roi": expected_profit_per_day / (TOP_K * STAKE_PER_NUMBER),
        })

    summary = pd.DataFrame(summary_rows)
    ordered = daily.copy()
    ordered["year"] = pd.to_datetime(ordered.date).dt.year
    yearly = ordered.groupby(["year", "strategy"], sort=True).agg(
        days=("date", "size"),
        expected_occurrences_per_day=("expected_occurrences_random_ties", "mean"),
        expected_hit_rate=("expected_hit_probability_random_ties", "mean"),
        random_hit_rate_conditional=("random_hit_probability", "mean"),
        stable_tie_occurrences_per_day=("stable_occurrences", "mean"),
    ).reset_index()

    eras = []
    year_dates = pd.to_datetime(ordered.date)
    for era, start, end in (
        ("2010-2019", "2010-01-01", "2020-01-01"),
        ("2020-2022", "2020-01-01", "2023-01-01"),
        ("2023-2026", "2023-01-01", "2027-01-01"),
    ):
        subset = ordered.loc[(year_dates >= start) & (year_dates < end)]
        for strategy, group in subset.groupby("strategy", sort=False):
            eras.append({
                "era": era,
                "strategy": strategy,
                "days": len(group),
                "expected_occurrences_per_day": float(
                    group.expected_occurrences_random_ties.mean()
                ),
                "expected_hit_rate": float(
                    group.expected_hit_probability_random_ties.mean()
                ),
            })
    return summary, yearly, pd.DataFrame(eras)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    headers = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(formats[column].format(value) if column in formats else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    eras: pd.DataFrame,
    *,
    source: Path,
    bootstrap_resamples: int,
) -> None:
    """Write reproducible result tables, protocol, and the Vietnamese report."""
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    daily.to_csv(output / "daily.csv", index=False)
    yearly.to_csv(output / "yearly.csv", index=False)
    eras.to_csv(output / "eras.csv", index=False)

    primary = daily.loc[daily.strategy.eq("frequency_90d")].sort_values("date")
    occurrences = primary.expected_occurrences_random_ties.to_numpy(dtype=float)
    baseline = DRAW_COUNT * TOP_K / 100
    robust_intervals = {}
    for block_length in (7, 90):
        low, high = block_bootstrap_mean_interval(
            occurrences - baseline,
            block_length=block_length,
            n_resamples=bootstrap_resamples,
            seed=20261105 + block_length,
        )
        robust_intervals[str(block_length)] = {"low": low, "high": high}
    five_year = daily.loc[daily.strategy.eq("frequency_5y")].sort_values("date")
    paired = (
        primary.expected_occurrences_random_ties.to_numpy(dtype=float)
        - five_year.expected_occurrences_random_ties.to_numpy(dtype=float)
    )
    paired_low, paired_high = block_bootstrap_mean_interval(
        paired,
        block_length=90,
        n_resamples=bootstrap_resamples,
        seed=20261201,
    )

    metrics = summary.set_index("strategy")
    early = eras.loc[
        eras.strategy.eq("frequency_90d"),
        ["era", "days", "expected_occurrences_per_day"],
    ]
    report = [
        "# Kiểm định dài hạn Top-3 số hai chữ số: tín hiệu tần suất 90 ngày",
        "",
        "## Phát hiện sơ bộ",
        "",
        f"- Walk-forward có **{len(primary):,} ngày dự báo**, từ **{primary.date.min()}** đến "
        f"**{primary.date.max()}**; mỗi dự báo chỉ dùng kết quả trước ngày mục tiêu.",
        f"- Top-3 tần suất 90 ngày đạt **{metrics.loc['frequency_90d', 'expected_occurrences_per_day']:.4f} "
        f"lượt/ngày** khi hòa điểm được chia ngẫu nhiên công bằng; mốc chọn đều là **{baseline:.4f}** "
        f"(27 kết quả × 3/100). Chênh lệch: **{metrics.loc['frequency_90d', 'delta_occurrences_vs_uniform']:+.4f} "
        "lượt/ngày**.",
        f"- Khoảng block-bootstrap 95% cho chênh lệch với mốc đều: **{robust_intervals['7']['low']:+.4f} đến "
        f"{robust_intervals['7']['high']:+.4f}** (khối 7 ngày) và **{robust_intervals['90']['low']:+.4f} đến "
        f"{robust_intervals['90']['high']:+.4f}** (khối 90 ngày).",
        f"- Tỷ lệ ngày hit là **{metrics.loc['frequency_90d', 'expected_hit_rate']:.2%}** so với "
        f"**{metrics.loc['frequency_90d', 'random_hit_rate_conditional']:.2%}** của chọn ngẫu nhiên; "
        f"chênh lệch block-bootstrap 95% theo khối 7 ngày là "
        f"**{metrics.loc['frequency_90d', 'hit_rate_delta_block95_low_7d']:+.2%} đến "
        f"{metrics.loc['frequency_90d', 'hit_rate_delta_block95_high_7d']:+.2%}**, còn chứa 0.",
        f"- So với tần suất 5 năm, chênh lệch ghép cặp là **{paired.mean():+.4f} lượt/ngày**, "
        f"block-bootstrap 95% (khối 90 ngày) **{paired_low:+.4f} đến {paired_high:+.4f}**; "
        "khoảng này không chứa 0 ở mức bootstrap chưa hiệu chỉnh.",
        "",
        "Đây là **ứng viên cần xác nhận**, chưa phải đột phá đã được chứng minh. Mức tăng lợi nhuận suy ra "
        "từ giả định trả 99.000đ mỗi lần xuất hiện và chi 27.000đ mỗi số là xấp xỉ "
        f"**{metrics.loc['frequency_90d', 'expected_profit_per_day']:,.0f}đ/ngày** trong hồi cứu; "
        "không được xem là kỳ vọng tương lai đã xác nhận.",
        "",
        "## So sánh các chiến lược",
        "",
        _markdown_table(
            summary,
            [
                "strategy", "days", "expected_occurrences_per_day",
                "delta_occurrences_vs_uniform", "occurrence_delta_block95_low_7d",
                "occurrence_delta_block95_high_7d", "expected_hit_rate",
                "random_hit_rate_conditional", "stable_tie_occurrences_per_day",
            ],
            {
                "expected_occurrences_per_day": "{:.4f}",
                "delta_occurrences_vs_uniform": "{:+.4f}",
                "occurrence_delta_block95_low_7d": "{:+.4f}",
                "occurrence_delta_block95_high_7d": "{:+.4f}",
                "expected_hit_rate": "{:.4%}",
                "random_hit_rate_conditional": "{:.4%}",
                "stable_tie_occurrences_per_day": "{:.4f}",
            },
        ),
        "",
        "Tần suất 90 ngày tốt nhất trong nhóm so sánh, nhưng chiến lược và ngưỡng 90 ngày không được "
        "đăng ký trước lần xem dữ liệu này. Các khoảng tin cậy ở bảng là mô tả, chưa hiệu chỉnh cho "
        "thử nhiều chiến lược hoặc việc tái sử dụng dữ liệu.",
        "",
        "## Kiểm tra ổn định theo thời kỳ",
        "",
        _markdown_table(
            early,
            ["era", "days", "expected_occurrences_per_day"],
            {"expected_occurrences_per_day": "{:.4f}"},
        ),
        "",
        "Mức trung bình dương qua ba thời kỳ làm giả thuyết tần suất ngắn hạn đáng kiểm tra tiếp, nhưng "
        "không tạo ra ba tập xác nhận độc lập: cửa sổ huấn luyện chồng lấn và toàn bộ lịch sử đã từng "
        "được xem.",
        "",
        "## Thiết kế và giới hạn",
        "",
        f"- Nguồn: `{source}`; SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`; "
        f"{len(pd.unique(pd.to_datetime(daily.date))):,} ngày dự báo.",
        "- Dữ liệu là 27 kết quả hai chữ số mỗi ngày, giữ nguyên các lần lặp. Huấn luyện có tối thiểu "
        "1.825 ngày lịch; chỉ dự báo ngày liền sau một ngày có kết quả.",
        "- Đối chứng occurrences ngẫu nhiên chính xác là 0,81/ngày, không cần giả định 27 kết quả "
        "độc lập: mỗi ngày chọn 3 số đều nhau từ 100 số.",
        "- Mốc hit ngẫu nhiên được tính có điều kiện theo số lượng số khác nhau thực tế trong ngày, "
        "không dùng xấp xỉ nhị thức.",
        "- Chỉ số `expected_*_random_ties` tính kỳ vọng chính xác nếu chọn ngẫu nhiên trong nhóm hòa "
        "điểm ở ngưỡng Top-3. Cột `stable_tie_*` ghi quy tắc vận hành xác định: ưu tiên số nhỏ hơn.",
        "- Các khoảng bootstrap dùng khối tròn theo ngày dự báo; khoảng không xử lý việc chọn mô hình "
        "sau khi xem kết quả, cũng không chứng minh tính ngẫu nhiên hoặc lợi thế tương lai.",
        "- Kiểm toán trước đây đã thử nhiều giả thuyết tần suất/thứ trong tuần/tự tương quan trên cùng "
        "nguồn. Đây không phải holdout mới. Không thay production; cần khóa dự báo tiền cứu, paper-trial "
        "ít nhất 6–12 tháng và không tinh chỉnh trong thời gian đó.",
        "",
        "## Tệp tái lập",
        "",
        "- [Giao thức](./protocol.json) · [Dữ liệu theo ngày](./daily.csv) · "
        "[Tổng hợp](./summary.csv) · [Theo năm](./yearly.csv) · [Theo thời kỳ](./eras.csv).",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    script = Path(__file__)
    protocol = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_script": str(script.resolve()),
        "analysis_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "source_start": str(pd.read_csv(source, usecols=["date"], dtype="string").date.min()),
        "source_end": str(pd.read_csv(source, usecols=["date"], dtype="string").date.max()),
        "evaluation_start": str(primary.date.min()),
        "evaluation_end": str(primary.date.max()),
        "evaluation_days": int(len(primary)),
        "target_results_per_day": DRAW_COUNT,
        "candidate_numbers": 100,
        "top_k": TOP_K,
        "minimum_history_calendar_days": HISTORY_DAYS,
        "strategies": list(STRATEGIES),
        "weekday_prior_same_weekday_days": WEEKDAY_PRIOR_DAYS,
        "random_expected_occurrences_per_day": baseline,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_block_lengths": [7, 90],
        "tie_policy_primary": "Expected metrics average uniformly over candidates tied at Top-3 cutoff.",
        "tie_policy_operational": "Stable descending score, ascending number for ties.",
        "selection_bias": (
            "Retrospective exploratory result; historical source and candidate strategies have been "
            "examined previously. Intervals are not adjusted for multiple comparisons."
        ),
        "production_default_changed": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")

    raw = pd.read_csv(
        source, dtype="string", keep_default_na=False, encoding="utf-8-sig"
    )
    prizes = normalize_prizes(raw)
    frequency, _ = XSMBDataLoader(prize_columns=PRIZE_COLUMNS).preprocess_and_build_matrices(
        prizes.reset_index()
    )
    daily = build_daily_results(frequency)
    summary, yearly, eras = build_summaries(
        daily, bootstrap_resamples=args.bootstrap_resamples
    )
    write_report(
        args.output.resolve(),
        daily,
        summary,
        yearly,
        eras,
        source=source,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(summary.to_string(index=False))
    print(f"\nReport written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
