"""Paired walk-forward comparison of 90-day frequency and main Poisson model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from scripts.research.research_top3_short_horizon import (
    DEFAULT_OUTPUT as FREQUENCY_OUTPUT,
    GROSS_PAYOUT_PER_OCCURRENCE,
    STAKE_PER_NUMBER,
)
from src.count_features import COUNT_TARGET, build_count_feature_dataset
from src.count_models import CountLGBMPredictor
from src.data_loader import XSMBDataLoader
from src.diagnostics import block_bootstrap_mean_interval
from src.prize_features import PRIZE_COLUMNS, normalize_prizes


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "raw" / "xsmb_2026-11-05.csv"
TOP_K = 3
NUMBER_COLUMNS = tuple(f"{number:02d}" for number in range(100))
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


def forecast_poisson_quarterly(
    frequency: pd.DataFrame,
    binary: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Fit the main all-history Poisson model once before each test quarter."""
    dataset = build_count_feature_dataset(
        frequency, binary, include_unlabeled=True
    )
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    label_dates = origins + pd.Timedelta(days=1)
    forecast_rows: list[dict[str, object]] = []

    for quarter, quarter_targets in pd.Series(target_dates).groupby(
        target_dates.to_period("Q"), sort=True
    ):
        dates = pd.DatetimeIndex(quarter_targets.to_list())
        quarter_start = quarter.start_time
        training = dataset.loc[label_dates < quarter_start]
        if training.empty or training[COUNT_TARGET].notna().sum() < 100:
            raise ValueError(f"Insufficient pre-quarter training data for {quarter}.")
        print(
            f"Fitting Poisson for {quarter}: "
            f"{training.index.get_level_values(0).nunique():,} feature dates, "
            f"{len(dates)} forecasts.",
            flush=True,
        )

        model = CountLGBMPredictor(**{
            key: value for key, value in MODEL_PARAMS.items() if key != "objective"
        }).fit(training)
        if model.trained_through_ is None or model.trained_through_ >= quarter_start:
            raise ValueError(f"Poisson training labels overlap {quarter}.")
        prediction_origins = dates - pd.Timedelta(days=1)
        prediction_rows = dataset.loc[
            dataset.index.get_level_values(0).isin(prediction_origins)
        ].drop(columns=COUNT_TARGET)
        scores = model.predict_expected_counts(prediction_rows)
        score_frame = scores.rename("score").reset_index().pivot(
            index=prediction_rows.index.names[0],
            columns=prediction_rows.index.names[1],
            values="score",
        )
        score_matrix = score_frame.loc[
            prediction_origins, list(NUMBER_COLUMNS)
        ].to_numpy(dtype=float)
        if score_matrix.shape != (len(dates), 100) or not np.isfinite(score_matrix).all():
            raise ValueError(f"Invalid Poisson predictions for {quarter}.")

        order = np.argsort(-score_matrix, axis=1, kind="stable")[:, :TOP_K]
        actual = frequency.loc[dates, list(NUMBER_COLUMNS)].to_numpy(dtype=np.int64)
        selected_counts = np.take_along_axis(actual, order, axis=1)
        for date, chosen, counts in zip(dates, order, selected_counts):
            forecast_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "poisson_prediction": " ".join(
                    f"{number:02d}" for number in chosen
                ),
                "poisson_occurrences": int(counts.sum()),
                "poisson_hit": bool(np.any(counts > 0)),
                "training_label_end": model.trained_through_.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(forecast_rows).sort_values("date").reset_index(drop=True)


def summarize(
    daily: pd.DataFrame, bootstrap_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return strategy metrics and paired confidence intervals."""
    if bootstrap_resamples < 200:
        raise ValueError("bootstrap_resamples must be at least 200.")
    strategy_rows = []
    for model, occurrences_column, hit_column in (
        ("frequency_90d", "frequency_occurrences", "frequency_hit"),
        ("poisson", "poisson_occurrences", "poisson_hit"),
    ):
        occurrences = daily[occurrences_column].to_numpy(dtype=float)
        hits = daily[hit_column].to_numpy(dtype=bool)
        cost = len(daily) * TOP_K * STAKE_PER_NUMBER
        receipts = int(occurrences.sum()) * GROSS_PAYOUT_PER_OCCURRENCE
        profit = receipts - cost
        misses = ~hits
        run = longest = 0
        for miss in misses:
            run = run + 1 if miss else 0
            longest = max(longest, run)
        strategy_rows.append({
            "model": model,
            "days": len(daily),
            "hit_days": int(hits.sum()),
            "hit_rate": float(hits.mean()),
            "occurrences": int(occurrences.sum()),
            "occurrences_per_day": float(occurrences.mean()),
            "stake_vnd": cost,
            "receipts_vnd": receipts,
            "profit_vnd": profit,
            "roi": profit / cost,
            "max_miss_streak": longest,
        })

    hit_delta = (
        daily.frequency_hit.to_numpy(dtype=float)
        - daily.poisson_hit.to_numpy(dtype=float)
    )
    occurrence_delta = (
        daily.frequency_occurrences.to_numpy(dtype=float)
        - daily.poisson_occurrences.to_numpy(dtype=float)
    )
    profit_delta = occurrence_delta * GROSS_PAYOUT_PER_OCCURRENCE
    hit_low, hit_high = block_bootstrap_mean_interval(
        hit_delta, block_length=7, n_resamples=bootstrap_resamples, seed=9021
    )
    occurrence_low, occurrence_high = block_bootstrap_mean_interval(
        occurrence_delta, block_length=7, n_resamples=bootstrap_resamples, seed=9022
    )
    profit_low, profit_high = block_bootstrap_mean_interval(
        profit_delta, block_length=7, n_resamples=bootstrap_resamples, seed=9023
    )
    frequency_only = int(
        (daily.frequency_hit & ~daily.poisson_hit).sum()
    )
    poisson_only = int(
        (daily.poisson_hit & ~daily.frequency_hit).sum()
    )
    paired_rows = [{
        "days": len(daily),
        "hit_rate_delta_frequency_minus_poisson": float(hit_delta.mean()),
        "hit_rate_delta_block95_low_7d": hit_low,
        "hit_rate_delta_block95_high_7d": hit_high,
        "frequency_only_hit_days": frequency_only,
        "poisson_only_hit_days": poisson_only,
        "mcnemar_exact_two_sided_p": float(
            binomtest(
                frequency_only,
                frequency_only + poisson_only,
                p=0.5,
            ).pvalue
        ),
        "occurrence_delta_frequency_minus_poisson": float(
            occurrence_delta.sum()
        ),
        "occurrence_delta_per_day": float(occurrence_delta.mean()),
        "occurrence_delta_block95_low_7d": occurrence_low,
        "occurrence_delta_block95_high_7d": occurrence_high,
        "profit_delta_frequency_minus_poisson_vnd": int(profit_delta.sum()),
        "profit_delta_block95_low_7d_vnd": profit_low,
        "profit_delta_block95_high_7d_vnd": profit_high,
        "same_top3_set_days": int(
            (daily.frequency_prediction == daily.poisson_prediction).sum()
        ),
        "mean_top3_overlap": float(
            np.mean([
                len(set(left.split()) & set(right.split()))
                for left, right in zip(
                    daily.frequency_prediction, daily.poisson_prediction
                )
            ])
        ),
    }]
    year = pd.to_datetime(daily.date).dt.year
    era_rows = []
    for era, start, end in (
        ("2010-2019", 2010, 2020),
        ("2020-2022", 2020, 2023),
        ("2023-2026", 2023, 2027),
    ):
        era_mask = year.between(start, end - 1)
        group = daily.loc[era_mask]
        if group.empty:
            continue
        for model, occurrences_column, hit_column in (
            ("frequency_90d", "frequency_occurrences", "frequency_hit"),
            ("poisson", "poisson_occurrences", "poisson_hit"),
        ):
            occurrences = int(group[occurrences_column].sum())
            cost = len(group) * TOP_K * STAKE_PER_NUMBER
            receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE
            era_rows.append({
                "era": era,
                "model": model,
                "days": len(group),
                "hit_days": int(group[hit_column].sum()),
                "hit_rate": float(group[hit_column].mean()),
                "occurrences": occurrences,
                "occurrences_per_day": float(occurrences / len(group)),
                "profit_vnd": receipts - cost,
                "roi": (receipts - cost) / cost,
            })

    yearly_rows = []
    yearly_paired_rows = []
    year_values = pd.to_datetime(daily.date).dt.year
    for year_value, group in daily.groupby(year_values, sort=True):
        for model, occurrences_column, hit_column in (
            ("frequency_90d", "frequency_occurrences", "frequency_hit"),
            ("poisson", "poisson_occurrences", "poisson_hit"),
        ):
            occurrences = int(group[occurrences_column].sum())
            hit_days = int(group[hit_column].sum())
            cost = len(group) * TOP_K * STAKE_PER_NUMBER
            receipts = occurrences * GROSS_PAYOUT_PER_OCCURRENCE
            yearly_rows.append({
                "year": int(year_value),
                "model": model,
                "days": len(group),
                "hit_days": hit_days,
                "hit_rate": hit_days / len(group),
                "occurrences": occurrences,
                "occurrences_per_day": occurrences / len(group),
                "stake_vnd": cost,
                "receipts_vnd": receipts,
                "profit_vnd": receipts - cost,
                "roi": (receipts - cost) / cost,
            })

        occurrence_delta = (
            group.frequency_occurrences.to_numpy(dtype=float)
            - group.poisson_occurrences.to_numpy(dtype=float)
        )
        hit_delta = (
            group.frequency_hit.to_numpy(dtype=float)
            - group.poisson_hit.to_numpy(dtype=float)
        )
        occurrence_low, occurrence_high = block_bootstrap_mean_interval(
            occurrence_delta,
            block_length=7,
            n_resamples=bootstrap_resamples,
            seed=9024 + int(year_value),
        )
        hit_low, hit_high = block_bootstrap_mean_interval(
            hit_delta,
            block_length=7,
            n_resamples=bootstrap_resamples,
            seed=9026 + int(year_value),
        )
        frequency_only = int(
            (group.frequency_hit & ~group.poisson_hit).sum()
        )
        poisson_only = int(
            (group.poisson_hit & ~group.frequency_hit).sum()
        )
        yearly_paired_rows.append({
            "year": int(year_value),
            "days": len(group),
            "occurrence_delta_frequency_minus_poisson": int(
                occurrence_delta.sum()
            ),
            "occurrence_delta_per_day": float(occurrence_delta.mean()),
            "occurrence_delta_block95_low_7d": occurrence_low,
            "occurrence_delta_block95_high_7d": occurrence_high,
            "profit_delta_frequency_minus_poisson_vnd": int(
                occurrence_delta.sum() * GROSS_PAYOUT_PER_OCCURRENCE
            ),
            "hit_rate_delta_frequency_minus_poisson": float(hit_delta.mean()),
            "hit_rate_delta_block95_low_7d": hit_low,
            "hit_rate_delta_block95_high_7d": hit_high,
            "frequency_only_hit_days": frequency_only,
            "poisson_only_hit_days": poisson_only,
            "mcnemar_exact_two_sided_p": float(
                binomtest(
                    frequency_only,
                    frequency_only + poisson_only,
                    p=0.5,
                ).pvalue
            ),
        })
    return (
        pd.DataFrame(strategy_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(era_rows),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(yearly_paired_rows),
    )


def write_report(
    output: Path,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    eras: pd.DataFrame,
    yearly: pd.DataFrame,
    yearly_paired: pd.DataFrame,
    *,
    source: Path,
    source_sha256: str,
    bootstrap_resamples: int,
) -> None:
    """Persist paired data, protocol, and comparison report."""
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "poisson_comparison_daily.csv", index=False)
    summary.to_csv(output / "poisson_comparison_summary.csv", index=False)
    paired.to_csv(output / "poisson_paired_comparison.csv", index=False)
    eras.to_csv(output / "poisson_comparison_eras.csv", index=False)
    yearly.to_csv(output / "poisson_comparison_yearly.csv", index=False)
    yearly_paired.to_csv(
        output / "poisson_comparison_yearly_paired.csv", index=False
    )

    poisson = summary.loc[summary.model.eq("poisson")].iloc[0]
    frequency = summary.loc[summary.model.eq("frequency_90d")].iloc[0]
    difference = paired.iloc[0]
    report = [
        "# So sánh Top-3 tần suất 90 ngày với Poisson chính",
        "",
        f"- Cùng kỳ dự báo: **{daily.date.min():%Y-%m-%d} đến "
        f"{daily.date.max():%Y-%m-%d}**, "
        f"**{len(daily):,} ngày ghép cặp**.",
        "- Poisson là `CountLGBMPredictor` của `main.py` (LightGBM objective Poisson, "
        "đếm số lần xuất hiện và giữ kết quả lặp); tái huấn luyện trước mỗi quý bằng "
        "toàn bộ nhãn đã biết trước quý. Đặc trưng cập nhật hằng ngày.",
        "- Frequency 90d dùng tổng số lần xuất hiện trong 90 ngày lịch ngay trước ngày dự báo. "
        "Cả hai chọn 3 số; hòa điểm được xử lý ổn định theo số tăng dần.",
        "- Tiền cược/trả dùng đúng giao thức hiện tại: 27.000đ/số/ngày; trả gộp 99.000đ "
        "cho mỗi lượt xuất hiện.",
        "",
        "## Kết quả",
        "",
        "| Mô hình | Ngày hit | Tỷ lệ hit | Lượt trúng | Lượt/ngày | Lãi/lỗ | ROI | Miss tối đa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.model} | {row.hit_days}/{row.days} | {row.hit_rate:.2%} | "
            f"{row.occurrences} | {row.occurrences_per_day:.4f} | "
            f"{row.profit_vnd:+,}đ | {row.roi:.2%} | {row.max_miss_streak} |"
            for row in summary.itertuples()
        ],
        "",
        "## Chênh lệch ghép cặp: Frequency 90d − Poisson",
        "",
        f"- Chênh lệch hit-rate: **{difference.hit_rate_delta_frequency_minus_poisson * 100:+.2f} "
        "điểm phần trăm** "
        f"(block-bootstrap 95%, khối 7 ngày: "
        f"{difference.hit_rate_delta_block95_low_7d:+.2%} đến "
        f"{difference.hit_rate_delta_block95_high_7d:+.2%}; "
        f"McNemar exact p={difference.mcnemar_exact_two_sided_p:.3f}).",
        f"- Số lượt trúng: **{int(difference.occurrence_delta_frequency_minus_poisson):+,}** "
        f"({difference.occurrence_delta_per_day:+.4f}/ngày; block-bootstrap 95% "
        f"{difference.occurrence_delta_block95_low_7d:+.4f} đến "
        f"{difference.occurrence_delta_block95_high_7d:+.4f} lượt/ngày).",
        f"- Chênh lệch lãi ròng: **{difference.profit_delta_frequency_minus_poisson_vnd:+,}đ** "
        f"(block-bootstrap 95% theo ngày: "
        f"{difference.profit_delta_block95_low_7d_vnd:+,.0f} đến "
        f"{difference.profit_delta_block95_high_7d_vnd:+,.0f}đ/ngày).",
        f"- Hai mô hình chọn cùng Top-3 trong **{int(difference.same_top3_set_days)}** ngày; "
        f"số giao trung bình **{difference.mean_top3_overlap:.3f}/3**.",
        "",
        "## Kiểm tra theo thời kỳ",
        "",
        "| Giai đoạn | Mô hình | Ngày | Tỷ lệ hit | Lượt/ngày | Lãi/lỗ | ROI |",
        "|---|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row.era} | {row.model} | {row.days} | {row.hit_rate:.2%} | "
            f"{row.occurrences_per_day:.4f} | {row.profit_vnd:+,}đ | {row.roi:.2%} |"
            for row in eras.itertuples()
        ],
        "",
        "## Chi tiết theo năm",
        "",
        "| Năm | Mô hình | Ngày | Ngày hit | Tỷ lệ hit | Lượt | Lượt/ngày | Chi | Thu | Lãi/lỗ | ROI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.year} | {row.model} | {row.days} | {row.hit_days} | "
            f"{row.hit_rate:.2%} | {row.occurrences} | {row.occurrences_per_day:.4f} | "
            f"{row.stake_vnd:,}đ | {row.receipts_vnd:,}đ | "
            f"{row.profit_vnd:+,}đ | {row.roi:.2%} |"
            for row in yearly.itertuples()
        ],
        "",
        "| Năm | Chênh lượt F90d − Poisson | Chênh P&L | Chênh hit-rate | Bootstrap 95% hit | McNemar p |",
        "|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.year} | {row.occurrence_delta_frequency_minus_poisson:+d} "
            f"({row.occurrence_delta_per_day:+.4f}/ngày) | "
            f"{row.profit_delta_frequency_minus_poisson_vnd:+,}đ | "
            f"{row.hit_rate_delta_frequency_minus_poisson * 100:+.2f} điểm % | "
            f"{row.hit_rate_delta_block95_low_7d * 100:+.2f} đến "
            f"{row.hit_rate_delta_block95_high_7d * 100:+.2f} điểm % | "
            f"{row.mcnemar_exact_two_sided_p:.3f} |"
            for row in yearly_paired.itertuples()
        ],
        "",
        "## Diễn giải",
        "",
        (
            f"Trong kỳ này, Frequency 90d {'vượt' if frequency.profit_vnd > poisson.profit_vnd else 'không vượt'} "
            f"Poisson về P&L ({frequency.profit_vnd:+,}đ so với {poisson.profit_vnd:+,}đ). "
            "Khoảng chênh hit-rate và P&L cần được đọc cùng khoảng bất định; "
            "đây là so sánh hồi cứu, không phải xác nhận lợi thế."
        ),
        "Giao thức Poisson tái huấn luyện theo quý để khả thi và khớp cách đánh giá walk-forward "
        "đang có; production có thể huấn luyện lại mỗi lần chạy nên đây không phải replay "
        "từng lần khởi chạy production. Cả hai lựa chọn đã được xem trên lịch sử; không dùng "
        "kết quả này để tự động thay mô hình. Tiếp tục paper-trial với dự báo ghi trước và "
        "giữ nguyên cấu hình trong thời gian thử nghiệm.",
        "",
        f"SHA-256 nguồn: `{source_sha256}`. Bootstrap dùng {bootstrap_resamples:,} mẫu, "
        "khối tròn 7 ngày; khoảng mô tả chưa điều chỉnh việc lựa chọn mô hình.",
        "",
        "[Dữ liệu ghép cặp](./poisson_comparison_daily.csv) · "
        "[Tổng hợp](./poisson_comparison_summary.csv) · "
        "[Theo thời kỳ](./poisson_comparison_eras.csv) · "
        "[Theo năm](./poisson_comparison_yearly.csv) · "
        "[Chênh lệch ghép cặp theo năm](./poisson_comparison_yearly_paired.csv) · "
        "[Kiểm định chênh lệch](./poisson_paired_comparison.csv) · "
        "[Giao thức](./poisson_comparison_protocol.json).",
    ]
    (output / "poisson_comparison.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    protocol = {
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "comparison_period_start": daily.date.min().strftime("%Y-%m-%d"),
        "comparison_period_end": daily.date.max().strftime("%Y-%m-%d"),
        "paired_days": int(len(daily)),
        "top_k": TOP_K,
        "frequency_strategy": "Top-3 by 90-calendar-day occurrence counts",
        "poisson_model": "src.count_models.CountLGBMPredictor from main.py",
        "poisson_training": (
            "All labeled history strictly before the first target of each calendar quarter; "
            "features update daily."
        ),
        "refit_frequency": "Before each calendar quarter",
        "poisson_parameters": MODEL_PARAMS,
        "tie_policy": "Descending score, ascending number for ties",
        "stake_per_number_vnd": STAKE_PER_NUMBER,
        "gross_payout_per_occurrence_vnd": GROSS_PAYOUT_PER_OCCURRENCE,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_block_days": 7,
        "production_default_changed": False,
    }
    (output / "poisson_comparison_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=FREQUENCY_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    frequency_protocol = json.loads(
        (output / "protocol.json").read_text(encoding="utf-8")
    )
    if frequency_protocol["source_sha256"] != source_sha256:
        raise ValueError("Frequency report source does not match the requested source.")
    frequency_results = pd.read_csv(output / "daily.csv")
    frequency_results = frequency_results.loc[
        frequency_results.strategy.eq("frequency_90d")
    ].copy()
    frequency_results["date"] = pd.to_datetime(frequency_results.date)
    target_dates = pd.DatetimeIndex(frequency_results.date)

    raw = pd.read_csv(
        source, dtype="string", keep_default_na=False, encoding="utf-8-sig"
    )
    prizes = normalize_prizes(raw)
    loader = XSMBDataLoader(prize_columns=PRIZE_COLUMNS)
    frequency, binary = loader.preprocess_and_build_matrices(prizes.reset_index())
    poisson_daily = forecast_poisson_quarterly(frequency, binary, target_dates)
    frequency_daily = frequency_results.rename(columns={
        "prediction_stable": "frequency_prediction",
        "stable_occurrences": "frequency_occurrences",
        "stable_hit": "frequency_hit",
    })
    frequency_daily["date"] = pd.to_datetime(frequency_daily.date)
    poisson_daily["date"] = pd.to_datetime(poisson_daily.date)
    daily = frequency_daily[
        ["date", "frequency_prediction", "frequency_occurrences", "frequency_hit"]
    ].merge(poisson_daily, on="date", how="inner", validate="one_to_one")
    if len(daily) != len(target_dates):
        raise ValueError("Poisson and frequency predictions do not cover the same target dates.")
    daily["actual_occurrences"] = [
        int(frequency.loc[date, list(NUMBER_COLUMNS)].sum())
        for date in pd.DatetimeIndex(daily.date)
    ]
    if not np.equal(daily.actual_occurrences.to_numpy(), 27).all():
        raise ValueError("Unexpected target-day result counts.")

    if daily.training_label_end.ge(
        daily.date.dt.to_period("Q").dt.start_time.dt.strftime("%Y-%m-%d")
    ).any():
        raise ValueError("Poisson training label cutoff is not strictly before its quarter.")
    actual = frequency.loc[pd.DatetimeIndex(daily.date), list(NUMBER_COLUMNS)].to_numpy(
        dtype=np.int64
    )
    frequency_order = np.array([
        [int(number) for number in prediction.split()]
        for prediction in daily.frequency_prediction
    ])
    if frequency_order.shape != (len(daily), TOP_K):
        raise ValueError("Malformed frequency Top-3 predictions.")
    frequency_counts = np.take_along_axis(actual, frequency_order, axis=1)
    if not np.array_equal(
        frequency_counts.sum(axis=1),
        daily.frequency_occurrences.to_numpy(dtype=np.int64),
    ) or not np.array_equal(
        np.any(frequency_counts > 0, axis=1),
        daily.frequency_hit.to_numpy(dtype=bool),
    ):
        raise ValueError("Frequency summary does not match the source actual outcomes.")
    poisson_order = np.array([
        [int(number) for number in prediction.split()]
        for prediction in daily.poisson_prediction
    ])
    poisson_counts = np.take_along_axis(actual, poisson_order, axis=1)
    if not np.array_equal(
        poisson_counts.sum(axis=1),
        daily.poisson_occurrences.to_numpy(dtype=np.int64),
    ) or not np.array_equal(
        np.any(poisson_counts > 0, axis=1),
        daily.poisson_hit.to_numpy(dtype=bool),
    ):
        raise ValueError("Poisson summary does not match the source actual outcomes.")

    summary, paired, eras, yearly, yearly_paired = summarize(
        daily, args.bootstrap_resamples
    )
    write_report(
        output,
        daily,
        summary,
        paired,
        eras,
        yearly,
        yearly_paired,
        source=source,
        source_sha256=source_sha256,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(summary.to_string(index=False))
    print("\nPaired comparison")
    print(paired.to_string(index=False))
    print(f"\nReport written to {output / 'poisson_comparison.md'}")


if __name__ == "__main__":
    main()
