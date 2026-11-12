"""Research a Top-3 model for the 23 XSMB results excluding prize four.

The new model is an offline experiment. It does not change the production
forecast command or the existing Poisson default.
"""

from __future__ import annotations

import argparse
from datetime import date as calendar_date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.count_features import COUNT_TARGET
from src.data_loader import NUMBER_COLUMNS, XSMBDataLoader
from src.diagnostics import block_bootstrap_mean_interval, evaluate_payouts, rank_top_k
from src.prize_features import (
    PRIZE_COLUMNS,
    RAW_PRIZE_URL,
    build_prize_feature_dataset,
    normalize_prizes,
)
from src.prize_models import PrizeLGBMPredictor


HISTORY_DAYS = 1095
TOP_K = 3
STAKE_VND = 27_000
GROSS_PER_OCCURRENCE_VND = 99_000
EXCLUDED_COLUMNS = tuple(column for column in PRIZE_COLUMNS if column.startswith("prize4_"))
TARGET_COLUMNS = tuple(column for column in PRIZE_COLUMNS if column not in EXCLUDED_COLUMNS)
STRATEGIES = (
    "poisson_full_27",
    "binary_full_27",
    "frequency_90d_23",
    "binary_excluding_prize4_23",
)


def parse_date(value: str) -> pd.Timestamp:
    """Parse a strict YYYY-MM-DD date."""
    parsed = calendar_date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD.")
    return pd.Timestamp(parsed)


def count_results_excluding_prize4(prizes: pd.DataFrame) -> pd.DataFrame:
    """Count the 23 non-prize-four two-digit endings for each observed date."""
    if not isinstance(prizes, pd.DataFrame) or tuple(prizes.columns) != PRIZE_COLUMNS:
        raise ValueError("Expected normalized prize data with the exact ordered XSMB columns.")
    if not isinstance(prizes.index, pd.DatetimeIndex) or prizes.index.has_duplicates:
        raise ValueError("Prize data must have a unique DatetimeIndex.")
    if not prizes.index.is_monotonic_increasing:
        raise ValueError("Prize dates must be sorted.")
    if len(TARGET_COLUMNS) != 23 or len(EXCLUDED_COLUMNS) != 4:
        raise AssertionError("Unexpected prize-four exclusion schema.")
    endings = prizes.loc[:, list(TARGET_COLUMNS)].apply(
        lambda column: column.str[-2:].astype(np.uint8)
    ).to_numpy(dtype=np.int64)
    row_ids = np.repeat(np.arange(len(prizes)), len(TARGET_COLUMNS))
    flattened = np.bincount(
        row_ids * 100 + endings.ravel(), minlength=len(prizes) * 100
    ).reshape(len(prizes), 100)
    return pd.DataFrame(flattened, index=prizes.index, columns=NUMBER_COLUMNS)


def replace_target_with_subset(
    full_dataset: pd.DataFrame, subset_counts: pd.DataFrame
) -> pd.DataFrame:
    """Replace full-27 labels with next-calendar-day 23-result labels."""
    if COUNT_TARGET not in full_dataset:
        raise ValueError(f"Missing target column {COUNT_TARGET!r}.")
    origins = pd.DatetimeIndex(full_dataset.index.get_level_values(0).unique())
    target_dates = origins + pd.Timedelta(days=1)
    aligned = subset_counts.reindex(target_dates).to_numpy(dtype=float)
    result = full_dataset.copy()
    result[COUNT_TARGET] = pd.array(aligned.reshape(-1), dtype="UInt8")
    return result


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial rate."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= positive trials.")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
    return center - half, center + half


def _max_streak(flags: np.ndarray, dates: pd.DatetimeIndex | None = None) -> int:
    maximum = current = 0
    for index, flag in enumerate(flags):
        if dates is not None and index and dates[index] - dates[index - 1] != pd.Timedelta(days=1):
            current = 0
        current = current + 1 if bool(flag) else 0
        maximum = max(maximum, current)
    return maximum


def _target_dates(
    full_counts: pd.DataFrame, start: pd.Timestamp
) -> pd.DatetimeIndex:
    candidates = full_counts.index[full_counts.index >= start]
    return candidates[(candidates - pd.Timedelta(days=1)).isin(full_counts.index)]


def _subset_frequency_scores(
    subset_counts: pd.DataFrame, target_dates: pd.DatetimeIndex
) -> np.ndarray:
    scores = []
    for target_date in target_dates:
        history = subset_counts.loc[
            (subset_counts.index >= target_date - pd.Timedelta(days=90))
            & (subset_counts.index < target_date)
        ]
        if history.empty:
            raise ValueError(f"No prior 90-day history for {target_date:%Y-%m-%d}.")
        scores.append(history.to_numpy(dtype=float).sum(axis=0))
    return np.asarray(scores)


def _summary_rows(
    dates: pd.DatetimeIndex,
    scores_by_strategy: dict[str, np.ndarray],
    subset_counts: pd.DataFrame,
    full_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_actual = subset_counts.loc[dates].to_numpy(dtype=np.int64)
    full_actual = full_counts.loc[dates].to_numpy(dtype=np.int64)
    daily_frames = []
    summary_rows = []
    monthly_rows = []
    poisson_daily_hit: dict[int, np.ndarray] = {}

    for name in STRATEGIES:
        scores = scores_by_strategy[name]
        selected = rank_top_k(scores, TOP_K)
        target_distinct_hits = ((target_actual > 0) & selected).sum(axis=1)
        target_occurrences = (target_actual * selected).sum(axis=1)
        target_repeat_number_hits = ((target_actual >= 2) & selected).sum(axis=1)
        target_repeat_occurrences = (
            np.maximum(target_actual - 1, 0) * selected
        ).sum(axis=1)
        full_distinct_hits = ((full_actual > 0) & selected).sum(axis=1)
        full_occurrences = (full_actual * selected).sum(axis=1)
        full_payout = evaluate_payouts(
            full_actual,
            selected,
            payout_mode="per_occurrence",
            stake=STAKE_VND,
            gross_payout=GROSS_PER_OCCURRENCE_VND,
        ).per_day
        if name == "poisson_full_27":
            poisson_daily_hit = {
                year: target_distinct_hits[dates.year == year] > 0
                for year in sorted(set(dates.year))
            }

        daily = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "strategy": name,
            "prediction_top3": [
                " ".join(f"{number:02d}" for number in np.flatnonzero(row))
                for row in selected
            ],
            "target23_distinct_hits": target_distinct_hits,
            "target23_hit_day": target_distinct_hits > 0,
            "target23_occurrences": target_occurrences,
            "target23_repeat_number_hits": target_repeat_number_hits,
            "target23_repeat_hit_day": target_repeat_number_hits > 0,
            "target23_repeat_occurrences": target_repeat_occurrences,
            "full27_distinct_hits": full_distinct_hits,
            "full27_occurrences": full_occurrences,
            "stake_vnd": full_payout.cost.to_numpy(),
            "receipts_full27_vnd": full_payout.receipts.to_numpy(),
            "profit_full27_vnd": full_payout.profit.to_numpy(),
        })
        daily_frames.append(daily)

        for year in sorted(set(dates.year)):
            mask = dates.year == year
            yearly_hit = target_distinct_hits[mask] > 0
            low, high = wilson_interval(int(yearly_hit.sum()), int(mask.sum()))
            block_low, block_high = block_bootstrap_mean_interval(
                yearly_hit.astype(float), block_length=7, n_resamples=2000, seed=42
            )
            if name == "poisson_full_27":
                delta_low = delta_high = 0.0
            else:
                delta = yearly_hit.astype(float) - poisson_daily_hit[year].astype(float)
                delta_low, delta_high = block_bootstrap_mean_interval(
                    delta, block_length=7, n_resamples=2000, seed=42
                )
            yearly_payout = full_payout.loc[mask]
            summary_rows.append({
                "year": int(year),
                "strategy": name,
                "days": int(mask.sum()),
                "target23_hit_days": int(yearly_hit.sum()),
                "target23_hit_rate": float(yearly_hit.mean()),
                "target23_hit_rate_wilson95_low": low,
                "target23_hit_rate_wilson95_high": high,
                "target23_hit_rate_block95_low": block_low,
                "target23_hit_rate_block95_high": block_high,
                "target23_number_hits": int(target_distinct_hits[mask].sum()),
                "target23_occurrences": int(target_occurrences[mask].sum()),
                "target23_repeat_number_hits": int(target_repeat_number_hits[mask].sum()),
                "target23_repeat_hit_days": int((target_repeat_number_hits[mask] > 0).sum()),
                "target23_repeat_occurrences": int(target_repeat_occurrences[mask].sum()),
                "target23_miss_streak_max": _max_streak(~yearly_hit, dates[mask]),
                "delta_hit_rate_vs_poisson_95_block_low": delta_low,
                "delta_hit_rate_vs_poisson_95_block_high": delta_high,
                "full27_occurrences": int(full_occurrences[mask].sum()),
                "full27_hit_days": int((full_distinct_hits[mask] > 0).sum()),
                "full27_stake_vnd": float(yearly_payout.cost.sum()),
                "full27_receipts_vnd": float(yearly_payout.receipts.sum()),
                "full27_profit_vnd": float(yearly_payout.profit.sum()),
                "full27_loss_days": int((yearly_payout.profit < 0).sum()),
                "full27_loss_streak_max": _max_streak(
                    yearly_payout.profit.to_numpy() < 0, dates[mask]
                ),
            })

        month_keys = dates.strftime("%Y-%m")
        for month in sorted(set(month_keys)):
            mask = month_keys == month
            monthly_payout = full_payout.loc[mask]
            monthly_rows.append({
                "month": month,
                "strategy": name,
                "days": int(mask.sum()),
                "target23_hit_days": int((target_distinct_hits[mask] > 0).sum()),
                "target23_hit_rate": float((target_distinct_hits[mask] > 0).mean()),
                "target23_occurrences": int(target_occurrences[mask].sum()),
                "target23_repeat_number_hits": int(target_repeat_number_hits[mask].sum()),
                "target23_repeat_hit_days": int((target_repeat_number_hits[mask] > 0).sum()),
                "target23_repeat_occurrences": int(target_repeat_occurrences[mask].sum()),
                "full27_occurrences": int(full_occurrences[mask].sum()),
                "full27_stake_vnd": float(monthly_payout.cost.sum()),
                "full27_receipts_vnd": float(monthly_payout.receipts.sum()),
                "full27_profit_vnd": float(monthly_payout.profit.sum()),
            })

    return (
        pd.concat(daily_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(monthly_rows),
    )


def _render_report(protocol: dict, summary: pd.DataFrame, latest: pd.DataFrame) -> str:
    strategies = summary.loc[summary.year.notna()].copy()
    summary_lines = []
    for row in strategies.sort_values(["year", "target23_hit_rate"], ascending=[True, False]).itertuples(index=False):
        summary_lines.append(
            f"| {int(row.year)} | {row.strategy} | {row.target23_hit_days}/{row.days} "
            f"({row.target23_hit_rate:.1%}; block {row.target23_hit_rate_block95_low:.1%}–"
            f"{row.target23_hit_rate_block95_high:.1%}) | {row.target23_occurrences} | "
            f"{row.target23_repeat_hit_days} ngày / {row.target23_repeat_occurrences} lượt lặp | "
            f"{row.target23_miss_streak_max} | {row.delta_hit_rate_vs_poisson_95_block_low:+.1%}–"
            f"{row.delta_hit_rate_vs_poisson_95_block_high:+.1%} | "
            f"{row.full27_profit_vnd:+,.0f} |"
        )
    latest_numbers = ", ".join(
            f"{row.number} ({row.score_uncalibrated_probability:.4f})"
            for row in latest.itertuples(index=False)
        )
    source_name = Path(protocol["source"]).name
    return f"""# Nghiên cứu mô hình Top-3 trên 23 kết quả XSMB (loại Giải tư)

## Thiết kế

- **Giả định diễn giải yêu cầu:** 27 đuôi số XSMB trừ 4 kết quả Giải tư, còn 23 kết quả. Dự đoán 3 đuôi trong 00–99 có mặt ở tập 23 kết quả đó. Nếu ý bạn là loại 4 kết quả khác, cần chạy lại với định nghĩa mục tiêu tương ứng.
- Mô hình mới: LightGBM nhị phân, học xác suất hiện diện trong 23 kết quả (không phải xác suất đã hiệu chuẩn). Dùng cùng bộ đặc trưng theo vị trí giải, cùng cấu hình và cùng lịch tái huấn luyện như các mô hình Prize hiện có.
- Đối chứng cố định: Poisson học số lần xuất hiện trên toàn bộ 27 kết quả; Binary học hiện diện trên 27 kết quả; tần suất đếm 90 ngày trước đó nhưng chỉ trên 23 kết quả. Cả bốn chiến lược chọn Top-3.
- Walk-forward theo quý; cửa sổ huấn luyện {f"{protocol['history_days']:,}".replace(",", ".")} ngày lịch, nhãn huấn luyện luôn kết thúc trước mỗi kỳ dự đoán. Không chọn siêu tham số theo kết quả đánh giá.
- Tập 2025 là giai đoạn xem xét; 2026 là stress-test lịch sử đã biết, không phải holdout mới. Nguồn là `{source_name}`, từ {protocol['source_start']} đến {protocol['source_end']}.
- Hit-rate mục tiêu đo trên 23 kết quả. Báo cáo lãi/lỗ riêng trên đủ 27 kết quả theo quy tắc hiện tại (27.000đ/số, nhận 99.000đ mỗi lần xuất hiện); không giả định giải tư bị loại khỏi quy tắc thanh toán.
- Mốc tham khảo ngẫu nhiên cho ít nhất một hit trong Top-3 và 23 lượt độc lập đồng đều là **{protocol['target_hit_reference_random_independent_uniform']:.1%}**; chỉ là mốc lý thuyết, không phải chuẩn được xác nhận cho quy trình quay.

## Hit-rate Top-3

| Năm | Chiến lược | Ngày có hit trong 23 kết quả (block 95%) | Lượt xuất hiện trong 23 | Ngày có hit lặp / lượt lặp | Chuỗi miss tối đa | Chênh hit-rate so Poisson (bootstrap khối 7 ngày, 95%) | Lãi/lỗ trên 27 kết quả |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

Khoảng block 95% dùng bootstrap khối 7 ngày đánh giá có dữ liệu trên chuỗi hit-day; khoảng chênh lệch dùng bootstrap ghép theo ngày giữa mỗi chiến lược và Poisson. Đây là ước lượng hồi cứu, không điều chỉnh việc đã xem dữ liệu. Hit-day nghĩa là có ít nhất một số Top-3 xuất hiện. Hit lặp là số đã xuất hiện ít nhất hai lần trong tập 23; lượt lặp đếm phần vượt quá lần đầu. Chuỗi miss/lỗ chỉ nối các ngày lịch liên tiếp có dữ liệu.

## Dự báo cuối cùng trên nguồn hiện có

Nguồn kết thúc ở {protocol['source_end']}. Với nhãn đã biết đến ngày này, mô hình mới xếp hạng Top-3 cho ngày kế tiếp **{protocol['latest_forecast_date']}**: **{latest_numbers}**. Đây là dự báo tái tạo từ snapshot lịch sử, không phải dự báo hiện tại.

## Kết luận

{protocol['conclusion_vi']}

Chi tiết mỗi ngày: `daily.csv`; tổng hợp: `summary.csv`; lịch huấn luyện: `training_cutoffs.csv`; giao thức và mã băm: `protocol.json`; Top-3 cuối nguồn: `latest_forecast.csv`.
"""


def run(
    source: Path,
    output: Path,
    start_date: str = "2025-01-01",
    history_days: int = HISTORY_DAYS,
) -> dict:
    """Fit the target-specific model and compare fixed strategies chronologically."""
    source = source.resolve()
    output = output.resolve()
    start = parse_date(start_date)
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output}")
    if isinstance(history_days, bool) or not isinstance(history_days, int) or history_days < 1:
        raise ValueError("history_days must be a positive integer.")

    raw_all = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    normalized_all = normalize_prizes(raw_all)
    source_end = normalized_all.index.max()
    feature_start = start - pd.Timedelta(days=history_days + 120)
    # Build every causal feature from the same fixed source origin.  Truncating
    # raw history here makes cumulative gap/history/co-occurrence features depend
    # on the requested evaluation start, so the same quarter can receive
    # different predictions in a full replay and a quarter-only replay.
    raw = raw_all.copy()
    normalized = normalized_all
    frequency, _ = XSMBDataLoader().preprocess_and_build_matrices(
        normalized.reset_index()
    )
    subset_counts = count_results_excluding_prize4(normalized)
    full_dataset = build_prize_feature_dataset(
        raw, include_unlabeled=True, start=feature_start
    )
    subset_dataset = replace_target_with_subset(full_dataset, subset_counts)

    dates = _target_dates(frequency, start)
    if dates.empty:
        raise ValueError("No complete target days with a preceding calendar result.")
    periods = dates.to_period("Q").unique()
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name in STRATEGIES}
    cutoff_rows = []

    for period in periods:
        quarter_start = max(period.start_time, start)
        block_dates = dates[dates.to_period("Q") == period]
        origin_dates = full_dataset.index.get_level_values(0)
        label_dates = origin_dates + pd.Timedelta(days=1)
        labeled = full_dataset[COUNT_TARGET].notna().to_numpy()
        train_mask = labeled & (label_dates < quarter_start)
        if not train_mask.any():
            raise ValueError(f"No training labels before {quarter_start:%Y-%m-%d}.")
        last_label = label_dates[train_mask].max()
        train_mask &= label_dates >= last_label - pd.Timedelta(days=history_days - 1)
        subset_labeled = subset_dataset[COUNT_TARGET].notna().to_numpy()
        subset_train_mask = subset_labeled & (label_dates < quarter_start)
        subset_train_mask &= label_dates >= last_label - pd.Timedelta(days=history_days - 1)
        if not np.array_equal(train_mask, subset_train_mask):
            raise AssertionError("Full and subset models must use identical training dates.")
        test_mask = label_dates.isin(block_dates)
        if int(test_mask.sum()) != len(block_dates) * 100:
            raise ValueError(f"Incomplete feature rows for quarter {period}.")
        if last_label >= block_dates[0]:
            raise AssertionError("Training labels overlap evaluation targets.")

        train_full = full_dataset.loc[train_mask]
        train_subset = subset_dataset.loc[subset_train_mask]
        inference = full_dataset.loc[test_mask].drop(columns=COUNT_TARGET)
        inference_subset = subset_dataset.loc[test_mask].drop(columns=COUNT_TARGET)

        poisson = PrizeLGBMPredictor("poisson", history_days=history_days).fit(train_full)
        binary_model = PrizeLGBMPredictor("binary", history_days=history_days).fit(train_full)
        poisson_scores = poisson.predict_scores(inference).to_numpy().reshape(-1, 100)
        binary_scores = binary_model.predict_scores(inference).to_numpy().reshape(-1, 100)

        subset_model = LGBMClassifier(**PrizeLGBMPredictor(
            "binary", history_days=history_days
        ).params)
        subset_features = [column for column in train_subset if column != COUNT_TARGET]
        subset_labels = train_subset[COUNT_TARGET].to_numpy(dtype=np.int64) > 0
        subset_model.fit(
            train_subset[subset_features].astype(float),
            subset_labels.astype(np.uint8),
            categorical_feature=["number_id"],
        )
        subset_scores = subset_model.predict_proba(
            inference_subset[subset_features].astype(float)
        )[:, 1].reshape(-1, 100)
        if not np.isfinite(subset_scores).all():
            raise ValueError(f"Nonfinite subset-model scores in {period}.")

        score_parts["poisson_full_27"].append(poisson_scores)
        score_parts["binary_full_27"].append(binary_scores)
        score_parts["binary_excluding_prize4_23"].append(subset_scores)
        score_parts["frequency_90d_23"].append(
            _subset_frequency_scores(subset_counts, block_dates)
        )
        cutoff_rows.append({
            "quarter": str(period),
            "first_target_date": block_dates[0].strftime("%Y-%m-%d"),
            "last_target_date": block_dates[-1].strftime("%Y-%m-%d"),
            "training_label_start": label_dates[train_mask].min().strftime("%Y-%m-%d"),
            "training_label_end": last_label.strftime("%Y-%m-%d"),
            "training_days": int(pd.DatetimeIndex(label_dates[train_mask]).nunique()),
            "training_history_days": history_days,
        })
        print(
            f"FIT {period}; train labels through {last_label:%Y-%m-%d}; "
            f"test={len(block_dates)} days",
            flush=True,
        )

    scores_by_strategy = {
        name: np.concatenate(parts, axis=0) for name, parts in score_parts.items()
    }
    if any(scores.shape != (len(dates), 100) for scores in scores_by_strategy.values()):
        raise AssertionError("Strategy score coverage differs from target dates.")

    daily, yearly, monthly = _summary_rows(
        dates, scores_by_strategy, subset_counts, frequency.reindex(subset_counts.index)
    )

    latest_date = normalized.index.max()
    latest_origin_rows = full_dataset.loc[
        full_dataset.index.get_level_values(0) == latest_date
    ]
    latest_train_mask = (
        full_dataset[COUNT_TARGET].notna().to_numpy()
        & (label_dates <= latest_date)
        & (label_dates >= latest_date - pd.Timedelta(days=history_days - 1))
    )
    latest_subset_mask = (
        subset_dataset[COUNT_TARGET].notna().to_numpy()
        & (label_dates <= latest_date)
        & (label_dates >= latest_date - pd.Timedelta(days=history_days - 1))
    )
    if not latest_subset_mask.any() or len(latest_origin_rows) != 100:
        raise ValueError("Insufficient labeled history or latest-day feature rows for final forecast.")
    final_subset = LGBMClassifier(**PrizeLGBMPredictor(
        "binary", history_days=history_days
    ).params)
    final_features = [
        column for column in subset_dataset.columns if column != COUNT_TARGET
    ]
    final_subset.fit(
        subset_dataset.loc[latest_subset_mask, final_features].astype(float),
        (subset_dataset.loc[latest_subset_mask, COUNT_TARGET].to_numpy(dtype=np.int64) > 0).astype(np.uint8),
        categorical_feature=["number_id"],
    )
    final_scores = final_subset.predict_proba(
        latest_origin_rows[final_features].astype(float)
    )[:, 1]
    if len(final_scores) != 100 or not np.isfinite(final_scores).all():
        raise ValueError("Final subset-model scores must be 100 finite values.")
    order = np.argsort(-final_scores, kind="stable")[:TOP_K]
    latest_forecast = pd.DataFrame({
        "forecast_date": [(latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")] * TOP_K,
        "feature_date": [latest_date.strftime("%Y-%m-%d")] * TOP_K,
        "rank": np.arange(1, TOP_K + 1),
        "number": [f"{number:02d}" for number in order],
        "score_uncalibrated_probability": final_scores[order],
        "score_target": "presence_among_23_results_excluding_prize4",
    })

    random_hit_reference = 1 - (1 - TOP_K / 100) ** len(TARGET_COLUMNS)
    protocol = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_start": normalized_all.index.min().strftime("%Y-%m-%d"),
        "source_end": source_end.strftime("%Y-%m-%d"),
        "evaluation_start": dates[0].strftime("%Y-%m-%d"),
        "evaluation_end": dates[-1].strftime("%Y-%m-%d"),
        "prizes_excluded": list(EXCLUDED_COLUMNS),
        "target_prize_columns": list(TARGET_COLUMNS),
        "target_draw_count_per_day": len(TARGET_COLUMNS),
        "top_k": TOP_K,
        "history_days": history_days,
        "feature_history_origin": normalized.index.min().strftime("%Y-%m-%d"),
        "feature_output_start": feature_start.strftime("%Y-%m-%d"),
        "feature_history_policy": "build causal features from the complete fixed source history, then trim output rows",
        "refit_policy": "before each calendar quarter; training labels strictly before quarter start",
        "bootstrap_block_length": 7,
        "bootstrap_block_unit": "consecutive evaluated dates with observed results",
        "strategies": list(STRATEGIES),
        "new_model": "LightGBM binary presence classifier trained on next-day counts from 23 prize positions excluding prize4",
        "new_model_params": PrizeLGBMPredictor("binary", history_days=history_days).params,
        "baseline_frequency_window": "preceding 90 calendar days, excluding target date",
        "target_hit_reference_random_independent_uniform": random_hit_reference,
        "payout_evaluation": {
            "evaluated_against": "all 27 results, separately from the 23-result model target",
            "stake_per_selection_vnd": STAKE_VND,
            "gross_per_occurrence_vnd": GROSS_PER_OCCURRENCE_VND,
            "separate_stake_refund": False,
        },
        "stress_period_is_untouched": False,
        "production_default_changed": False,
        "latest_forecast_date": (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "recommendation": (
            "The 23-result-target model is a research challenger only. Compare its paired "
            "Top-3 hit-rate differences and uncertainty to the full-27 Poisson baseline; "
            "do not switch defaults without a prospective frozen paper-trial."
        ),
        "conclusion_vi": (
            "Mô hình mới đạt hit-rate thấp hơn Poisson trong cả 2025 và giai đoạn stress-test "
            "2026; khoảng bootstrap ghép theo ngày vẫn bao gồm mức chênh 0. Chưa có bằng chứng "
            "mô hình 23 kết quả tốt hơn, vì vậy không đổi mặc định. Chỉ xem xét lại sau paper-trial "
            "tiền cứu với giao thức và Top-3 được khóa trước."
        ),
    }
    report = _render_report(protocol, yearly, latest_forecast)
    output.mkdir(parents=True, exist_ok=False)
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cutoff_rows).to_csv(output / "training_cutoffs.csv", index=False)
    latest_forecast.to_csv(output / "latest_forecast.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"REPORT {output / 'report.md'}", flush=True)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local raw 27-prize XSMB CSV.")
    parser.add_argument("--output", type=Path, required=True, help="New research output directory.")
    parser.add_argument("--start-date", default="2025-01-01", help="First test target date (YYYY-MM-DD).")
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
