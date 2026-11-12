"""Chronological research for Top-3 three-digit XSMB endings (000-999)."""

from __future__ import annotations

import argparse
from datetime import date as calendar_date
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.diagnostics import block_bootstrap_mean_interval
from src.prize_features import PRIZE_COLUMNS, PRIZE_WIDTHS, normalize_prizes


HISTORY_DAYS = 1095
LONG_FREQUENCY_WINDOW_DAYS = 1825
SUFFIX_SHRINKAGE = 0.05
STAKE_PER_NUMBER = 23
GROSS_PAYOUT_PER_OCCURRENCE = 900
SOURCE_URL = (
    "https://raw.githubusercontent.com/khiemdoan/"
    "vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv"
)
TOP_K = 3
WINDOWS = (7, 30, 90, 365, 1095)
TARGET_COLUMNS = tuple(
    column for column in PRIZE_COLUMNS
    if not column.startswith("prize7_")
)
NUMBER_COLUMNS_3D = tuple(f"{number:03d}" for number in range(1000))
STRATEGIES = (
    "poisson_3digit_23",
    "binary_3digit_23",
    "frequency_90d_3digit_23",
    "frequency_5y_suffix_shrinkage_3digit_23",
)
MODEL_PARAMS = {
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


def _datetime_ns(values: pd.DatetimeIndex) -> np.ndarray:
    """Convert datetimes to nanoseconds explicitly; .asi8 uses index resolution."""
    return values.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def parse_date(value: str) -> pd.Timestamp:
    parsed = calendar_date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD.")
    return pd.Timestamp(parsed)


def count_three_digit_endings(prizes: pd.DataFrame) -> pd.DataFrame:
    """Count last-three-digit outcomes, preserving leading zeros and repeats."""
    if not isinstance(prizes, pd.DataFrame) or tuple(prizes.columns) != PRIZE_COLUMNS:
        raise ValueError("Expected normalized prize data with the exact ordered XSMB columns.")
    if not isinstance(prizes.index, pd.DatetimeIndex) or prizes.index.has_duplicates:
        raise ValueError("Prize data must have a unique DatetimeIndex.")
    if not prizes.index.is_monotonic_increasing:
        raise ValueError("Prize dates must be sorted.")
    if len(TARGET_COLUMNS) != 23:
        raise AssertionError("Expected 23 eligible positions after excluding the four two-digit prize-seven results.")
    endings = prizes.loc[:, list(TARGET_COLUMNS)].apply(
        lambda column: column.str[-3:].astype(np.uint16)
    ).to_numpy(dtype=np.int64)
    row_ids = np.repeat(np.arange(len(prizes)), len(TARGET_COLUMNS))
    counts = np.bincount(
        row_ids * 1000 + endings.ravel(), minlength=len(prizes) * 1000
    ).reshape(len(prizes), 1000)
    result = pd.DataFrame(
        counts.astype(np.uint8), index=prizes.index, columns=NUMBER_COLUMNS_3D
    )
    if not result.sum(axis=1).eq(23).all():
        raise AssertionError("Every target day must contain exactly 23 eligible prize results.")
    return result


class ThreeDigitFeatureBuilder:
    """Create past-only rolling features for each three-digit candidate."""

    def __init__(self, counts: pd.DataFrame) -> None:
        if (
            not isinstance(counts, pd.DataFrame)
            or tuple(counts.columns) != NUMBER_COLUMNS_3D
            or not isinstance(counts.index, pd.DatetimeIndex)
            or counts.index.has_duplicates
            or not counts.index.is_monotonic_increasing
        ):
            raise ValueError("Counts must be sorted daily 000..999 target counts.")
        values = counts.to_numpy(dtype=np.int32)
        if (values < 0).any()                 or not np.all(values.sum(axis=1) == 23):
            raise ValueError("Every target-count row must contain 23 nonnegative outcomes.")
        self.counts = counts
        self.dates = counts.index
        self.date_ns = _datetime_ns(self.dates)
        self.values = values
        self._prefixes: dict[str, np.ndarray] = {}
        self._make_prefix("three_count", values)
        self._make_prefix("three_hit", values > 0)
        two_count = values.reshape(len(values), 10, 100).sum(axis=1)
        two_hit = (values > 0).reshape(len(values), 10, 100).any(axis=1)
        one_count = two_count.reshape(len(values), 10, 10).sum(axis=1)
        one_hit = (two_count > 0).reshape(len(values), 10, 10).any(axis=1)
        self.two_count = two_count
        self.two_hit = two_hit.astype(np.int32)
        self.one_count = one_count
        self.one_hit = one_hit.astype(np.int32)
        self._make_prefix("two_count", self.two_count)
        self._make_prefix("two_hit", self.two_hit)
        self._make_prefix("one_count", self.one_count)
        self._make_prefix("one_hit", self.one_hit)

    def _make_prefix(self, name: str, values: np.ndarray) -> None:
        prefix = np.zeros((len(values) + 1, values.shape[1]), dtype=np.int32)
        np.cumsum(values, axis=0, dtype=np.int32, out=prefix[1:])
        self._prefixes[name] = prefix

    def transform(self, feature_dates: pd.DatetimeIndex) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return rows [date, candidate] and fixed causal feature names."""
        feature_dates = pd.DatetimeIndex(feature_dates)
        positions = self.dates.get_indexer(feature_dates)
        if (positions < 0).any():
            raise ValueError("Every feature date must have observed source results.")
        lower = {
            window: np.searchsorted(
                self.date_ns,
                _datetime_ns(feature_dates) - (window - 1) * 86_400_000_000_000,
                side="left",
            )
            for window in WINDOWS
        }
        feature_arrays: list[np.ndarray] = []
        names: list[str] = []

        def append_window(prefix: str, window: int, candidate_map: np.ndarray) -> None:
            cumulative = self._prefixes[prefix]
            sums = cumulative[positions + 1] - cumulative[lower[window]]
            feature_arrays.append(sums[:, candidate_map].astype(np.float32, copy=False))

        for window in WINDOWS:
            append_window("three_count", window, np.arange(1000))
            names.append(f"three_count_{window}d")
            append_window("three_hit", window, np.arange(1000))
            names.append(f"three_hit_days_{window}d")
        for window in (90, 365):
            append_window("two_count", window, np.arange(1000) % 100)
            names.append(f"two_digit_suffix_count_{window}d")
            append_window("two_hit", window, np.arange(1000) % 100)
            names.append(f"two_digit_suffix_hit_days_{window}d")
            append_window("one_count", window, np.arange(1000) % 10)
            names.append(f"last_digit_count_{window}d")
            append_window("one_hit", window, np.arange(1000) % 10)
            names.append(f"last_digit_hit_days_{window}d")
        candidates = np.arange(1000, dtype=np.uint16)
        digits = (
            np.broadcast_to(candidates // 100, (len(positions), 1000)),
            np.broadcast_to((candidates // 10) % 10, (len(positions), 1000)),
            np.broadcast_to(candidates % 10, (len(positions), 1000)),
        )
        for name, digit in zip(("hundreds_digit", "tens_digit", "units_digit"), digits):
            feature_arrays.append(digit.astype(np.float32, copy=False))
            names.append(name)
        tensor = np.stack(feature_arrays, axis=-1)
        return tensor.reshape(-1, tensor.shape[-1]), tuple(names)

    def labels(self, feature_dates: pd.DatetimeIndex) -> np.ndarray:
        target_dates = pd.DatetimeIndex(feature_dates) + pd.Timedelta(days=1)
        positions = self.dates.get_indexer(target_dates)
        if (positions < 0).any():
            raise ValueError("Every training target must have an observed next-calendar-day result.")
        return self.values[positions]


def _target_dates(counts: pd.DataFrame, start: pd.Timestamp) -> pd.DatetimeIndex:
    dates = counts.index[counts.index >= start]
    return dates[(dates - pd.Timedelta(days=1)).isin(counts.index)]


def _rolling_scores(
    builder: ThreeDigitFeatureBuilder, target_dates: pd.DatetimeIndex, window: int
) -> np.ndarray:
    return _rolling_mapped_scores(
        builder, target_dates, window, "three_count", np.arange(1000)
    )


def _rolling_mapped_scores(
    builder: ThreeDigitFeatureBuilder,
    target_dates: pd.DatetimeIndex,
    window: int,
    prefix_name: str,
    candidate_map: np.ndarray,
) -> np.ndarray:
    feature_dates = target_dates - pd.Timedelta(days=1)
    positions = builder.dates.get_indexer(feature_dates)
    if (positions < 0).any():
        raise ValueError("Every target date must have an observed preceding feature date.")
    lower = np.searchsorted(
        builder.date_ns,
        _datetime_ns(target_dates) - window * 86_400_000_000_000,
        side="left",
    )
    prefix = builder._prefixes[prefix_name]
    return (prefix[positions + 1] - prefix[lower])[:, candidate_map]


def _rolling_suffix_shrinkage_scores(
    builder: ThreeDigitFeatureBuilder,
    target_dates: pd.DatetimeIndex,
    window: int = LONG_FREQUENCY_WINDOW_DAYS,
    shrinkage: float = SUFFIX_SHRINKAGE,
) -> np.ndarray:
    three_digit_counts = _rolling_scores(builder, target_dates, window)
    suffix_counts = _rolling_mapped_scores(
        builder,
        target_dates,
        window,
        "two_count",
        np.arange(1000) % 100,
    )
    return three_digit_counts + shrinkage * suffix_counts


def _fit_pair(
    builder: ThreeDigitFeatureBuilder,
    train_dates: pd.DatetimeIndex,
    predict_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    train_features, feature_names = builder.transform(
        train_dates - pd.Timedelta(days=1)
    )
    test_features, test_names = builder.transform(
        predict_dates - pd.Timedelta(days=1)
    )
    if feature_names != test_names:
        raise AssertionError("Training and inference feature schemas differ.")
    train_labels = builder.labels(train_dates - pd.Timedelta(days=1)).reshape(-1)
    poisson = LGBMRegressor(objective="poisson", **MODEL_PARAMS)
    poisson.fit(train_features, train_labels)
    binary = LGBMClassifier(objective="binary", **MODEL_PARAMS)
    binary.fit(train_features, (train_labels > 0).astype(np.uint8))
    return (
        poisson.predict(test_features).reshape(-1, 1000),
        binary.predict_proba(test_features)[:, 1].reshape(-1, 1000),
    )


def _max_streak(flags: np.ndarray, dates: pd.DatetimeIndex) -> int:
    maximum = current = 0
    for index, flag in enumerate(flags):
        if index and dates[index] - dates[index - 1] != pd.Timedelta(days=1):
            current = 0
        current = current + 1 if bool(flag) else 0
        maximum = max(maximum, current)
    return maximum


def _rank_top_k(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != 1000 or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores for all 1,000 three-digit candidates.")
    selected = np.zeros(scores.shape, dtype=bool)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K]
    np.put_along_axis(selected, order, True, axis=1)
    return selected


def _summaries(
    dates: pd.DatetimeIndex,
    scores: dict[str, np.ndarray],
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actual = counts.loc[dates].to_numpy(dtype=np.int64)
    daily_frames = []
    yearly_rows = []
    monthly_rows = []
    poisson_hits: dict[int, np.ndarray] = {}
    poisson_month_hits: dict[str, np.ndarray] = {}
    candidates = np.arange(1000)

    for strategy in STRATEGIES:
        selected = _rank_top_k(scores[strategy])
        distinct_hits = ((actual > 0) & selected).sum(axis=1)
        occurrences = (actual * selected).sum(axis=1)
        repeat_numbers = ((actual >= 2) & selected).sum(axis=1)
        repeat_occurrences = (np.maximum(actual - 1, 0) * selected).sum(axis=1)
        daily_stake = selected.sum(axis=1) * STAKE_PER_NUMBER
        daily_gross_payout = occurrences * GROSS_PAYOUT_PER_OCCURRENCE
        daily_profit = daily_gross_payout - daily_stake
        if strategy == "poisson_3digit_23":
            poisson_hits = {
                year: distinct_hits[dates.year == year] > 0
                for year in sorted(set(dates.year))
            }
        daily_frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "prediction_top3": [
                " ".join(f"{number:03d}" for number in candidates[row])
                for row in np.argsort(-scores[strategy], axis=1, kind="stable")[:, :TOP_K]
            ],
            "hit_day": distinct_hits > 0,
            "distinct_hits": distinct_hits,
            "target_occurrences": occurrences,
            "repeat_number_hits": repeat_numbers,
            "repeat_hit_day": repeat_numbers > 0,
            "repeat_occurrences": repeat_occurrences,
            "stake_units": daily_stake,
            "gross_payout_units": daily_gross_payout,
            "net_profit_units": daily_profit,
        }))

        for year in sorted(set(dates.year)):
            mask = dates.year == year
            hit_days = distinct_hits[mask] > 0
            if strategy == "poisson_3digit_23":
                delta_low = delta_high = 0.0
            else:
                delta = hit_days.astype(float) - poisson_hits[year].astype(float)
                delta_low, delta_high = block_bootstrap_mean_interval(
                    delta, block_length=7, n_resamples=3000, seed=42
                )
            block_low, block_high = block_bootstrap_mean_interval(
                hit_days.astype(float),
                block_length=7,
                n_resamples=3000,
                seed=42,
            )
            yearly_rows.append({
                "year": int(year),
                "strategy": strategy,
                "days": int(mask.sum()),
                "hit_days": int(hit_days.sum()),
                "hit_rate": float(hit_days.mean()),
                "hit_rate_block95_low": block_low,
                "hit_rate_block95_high": block_high,
                "distinct_number_hits": int(distinct_hits[mask].sum()),
                "target_occurrences": int(occurrences[mask].sum()),
                "repeat_hit_days": int((repeat_numbers[mask] > 0).sum()),
                "repeat_number_hits": int(repeat_numbers[mask].sum()),
                "repeat_occurrences": int(repeat_occurrences[mask].sum()),
                "stake_units": int(daily_stake[mask].sum()),
                "gross_payout_units": int(daily_gross_payout[mask].sum()),
                "net_profit_units": int(daily_profit[mask].sum()),
                "profitable_days": int((daily_profit[mask] > 0).sum()),
                "losing_days": int((daily_profit[mask] < 0).sum()),
                "max_loss_streak": _max_streak(daily_profit[mask] < 0, dates[mask]),
                "miss_streak_max": _max_streak(~hit_days, dates[mask]),
                "delta_hit_rate_vs_poisson_block95_low": delta_low,
                "delta_hit_rate_vs_poisson_block95_high": delta_high,
            })

        months = dates.strftime("%Y-%m")
        if strategy == "poisson_3digit_23":
            poisson_month_hits = {
                month: distinct_hits[months == month] > 0 for month in sorted(set(months))
            }
        for month in sorted(set(months)):
            mask = months == month
            month_hits = distinct_hits[mask] > 0
            month_block_low, month_block_high = block_bootstrap_mean_interval(
                month_hits.astype(float),
                block_length=7,
                n_resamples=3000,
                seed=42,
            )
            if strategy == "poisson_3digit_23":
                month_delta_low = month_delta_high = 0.0
            else:
                month_delta_low, month_delta_high = block_bootstrap_mean_interval(
                    month_hits.astype(float) - poisson_month_hits[month].astype(float),
                    block_length=7,
                    n_resamples=3000,
                    seed=42,
                )
            monthly_rows.append({
                "month": month,
                "strategy": strategy,
                "days": int(mask.sum()),
                "hit_days": int(month_hits.sum()),
                "hit_rate": float(month_hits.mean()),
                "hit_rate_block95_low": month_block_low,
                "hit_rate_block95_high": month_block_high,
                "target_occurrences": int(occurrences[mask].sum()),
                "repeat_hit_days": int((repeat_numbers[mask] > 0).sum()),
                "repeat_occurrences": int(repeat_occurrences[mask].sum()),
                "stake_units": int(daily_stake[mask].sum()),
                "gross_payout_units": int(daily_gross_payout[mask].sum()),
                "net_profit_units": int(daily_profit[mask].sum()),
                "profitable_days": int((daily_profit[mask] > 0).sum()),
                "losing_days": int((daily_profit[mask] < 0).sum()),
                "delta_hit_rate_vs_poisson_block95_low": month_delta_low,
                "delta_hit_rate_vs_poisson_block95_high": month_delta_high,
            })
    return (
        pd.concat(daily_frames, ignore_index=True),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(monthly_rows),
    )


def _daily_detail(daily: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Make one auditable row per draw, with each model's exact hit numbers."""
    detail_rows = []
    strategies = list(STRATEGIES)
    for day, group in daily.groupby("date", sort=True):
        date = pd.Timestamp(day)
        actual = counts.loc[date]
        actual_list = [
            f"{int(number):03d}"
            for number in np.repeat(
                np.arange(1000), actual.to_numpy(dtype=np.int64)
            )
        ]
        if len(actual_list) != 23:
            raise AssertionError(f"{day} should contain 23 eligible three-digit endings.")
        row: dict[str, object] = {
            "date": day,
            "actual_23_endings_sorted": " ".join(actual_list),
        }
        by_strategy = group.set_index("strategy")
        for strategy in strategies:
            result = by_strategy.loc[strategy]
            picks = [int(value) for value in result.prediction_top3.split()]
            hit_numbers = [
                f"{number:03d}({int(actual.iloc[number])})"
                for number in picks
                if actual.iloc[number] > 0
            ]
            prefix = strategy.removesuffix("_3digit_23")
            row[f"{prefix}_top3"] = " ".join(f"{number:03d}" for number in picks)
            row[f"{prefix}_hit_numbers_count"] = "; ".join(hit_numbers) or "-"
            row[f"{prefix}_distinct_hits"] = int(result.distinct_hits)
            row[f"{prefix}_occurrences"] = int(result.target_occurrences)
            row[f"{prefix}_stake"] = int(result.stake_units)
            row[f"{prefix}_gross_payout"] = int(result.gross_payout_units)
            row[f"{prefix}_net_profit"] = int(result.net_profit_units)
        detail_rows.append(row)
    return pd.DataFrame(detail_rows)


def _report(
    protocol: dict,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    latest: pd.DataFrame,
) -> str:
    rows = []
    for row in yearly.sort_values(
        ["year", "hit_rate"], ascending=[True, False]
    ).itertuples(index=False):
        rows.append(
            f"| {row.year} | {row.strategy} | {row.hit_days}/{row.days} "
            f"({row.hit_rate:.1%}; {row.hit_rate_block95_low:.1%}–"
            f"{row.hit_rate_block95_high:.1%}) | {row.target_occurrences} | "
            f"{row.repeat_hit_days} / {row.repeat_occurrences} | "
            f"{row.miss_streak_max} | {row.delta_hit_rate_vs_poisson_block95_low:+.1%}–"
            f"{row.delta_hit_rate_vs_poisson_block95_high:+.1%} | "
            f"{row.stake_units:,} | {row.gross_payout_units:,} | {row.net_profit_units:+,} |"
        )
    numbers = ", ".join(
            f"{int(row.number):03d} ({row.score:.4f})"
            for row in latest.itertuples(index=False)
        )
    monthly_rows = []
    for row in monthly.sort_values(["month", "hit_rate"], ascending=[True, False]).itertuples(index=False):
        monthly_rows.append(
            f"| {row.month} | {row.strategy} | {row.hit_days}/{row.days} "
            f"({row.hit_rate:.1%}; {row.hit_rate_block95_low:.1%}–"
            f"{row.hit_rate_block95_high:.1%}) | {row.target_occurrences} | "
            f"{row.repeat_hit_days} / {row.repeat_occurrences} | "
            f"{row.delta_hit_rate_vs_poisson_block95_low:+.1%}–"
            f"{row.delta_hit_rate_vs_poisson_block95_high:+.1%} | "
            f"{row.stake_units:,} | {row.gross_payout_units:,} | {row.net_profit_units:+,} |"
        )
    return f"""# Nghiên cứu dự đoán Top-3 số 3 chữ số XSMB

## Định nghĩa

- Đây là nghiên cứu về **ba chữ số cuối (000–999)** của mỗi kết quả, không phải chọn ba số có hai chữ số.
- Lấy ba chữ số cuối của tất cả kết quả có từ ba chữ số trở lên, giữ các lần lặp và giữ số 0 đầu của kết quả ba chữ số. Chỉ loại bốn vị trí Giải bảy vì kết quả vốn chỉ có hai chữ số; giữ cả Giải tư (lấy ba chữ số cuối). Do đó có 23 kết quả đủ điều kiện mỗi ngày. Không biến kết quả hai chữ số thành 0xx.
- Walk-forward theo quý, huấn luyện chỉ trên kết quả đã công bố trước kỳ dự đoán, cửa sổ {protocol['history_days']} ngày lịch. So sánh LightGBM binary dự đoán hiện diện, LightGBM Poisson dự đoán số lần xuất hiện (cùng bộ đặc trưng lịch sử số 3 chữ số, hậu tố hai/chữ số, và chữ số vị trí), cùng baseline tần suất 90 ngày.
- Thử nghiệm thêm xếp hạng theo tần suất số 3 chữ số trong 5 năm cộng 0,05 lần tần suất hậu tố 2 chữ số cùng kỳ. Đây là ứng viên hồi cứu đã được thử sau khi xem kỳ đánh giá, không phải holdout độc lập hay thay đổi mặc định.
- Kỳ kiểm thử: **{protocol['evaluation_start']} đến {protocol['evaluation_end']}**. Nguồn: `{Path(protocol['source']).name}`, {protocol['source_start']}–{protocol['source_end']}. Đây là kiểm thử hồi cứu; nếu kỳ này đã được xem trước thì không phải holdout mới.
- Hit-rate là ngày có ít nhất một số Top-3 xuất hiện trong 23 kết quả đủ ba chữ số. Cược giả định: **23 đơn vị cho mỗi số mỗi ngày, trả gộp 900 đơn vị cho mỗi lần xuất hiện**. Ba số tốn 69 đơn vị/ngày; lãi ròng = 900 × số lần xuất hiện của Top-3 − 69. Số lặp được trả theo từng lần xuất hiện. Nếu “ăn 900” của bạn là lãi ròng chưa gồm tiền cược, số lãi cần cộng thêm phần cược trả lại theo quy tắc thực tế.
- Mốc tham khảo độc lập đồng đều: xác suất ít nhất một trong 23 kết quả trùng Top-3 là **{protocol['target_hit_reference_random_independent_uniform']:.1%}**; đây chỉ là mốc lý thuyết.

## Kết quả

| Năm | Chiến lược | Ngày có hit (block 95%) | Lượt xuất hiện | Ngày hit lặp / lượt lặp | Chuỗi miss tối đa | Chênh hit-rate so Poisson (95%) | Cược | Trả gộp | Lãi ròng |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Kết quả từng tháng

| Tháng | Chiến lược | Ngày có hit (block 95%) | Lượt trúng | Ngày hit lặp / lượt lặp | Chênh hit-rate so Poisson (95%) | Cược | Trả gộp | Lãi ròng |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(monthly_rows)}

Bootstrap dùng khối 7 kỳ có dữ liệu; khoảng chênh được ghép theo cùng ngày. Đây là bất định lấy mẫu hồi cứu, không hiệu chỉnh cho việc đã xem dữ liệu hay chọn mô hình. Hit lặp là số Top-3 xuất hiện ít nhất hai lần; lượt lặp đếm phần vượt quá lần đầu. Chuỗi miss chỉ nối những ngày lịch liên tiếp có dữ liệu.

## Top-3 của ngày kiểm thử cuối

Trước kết quả của ngày **{protocol['latest_forecast_date']}**, mô hình Poisson xếp hạng: **{numbers}**. Đây là dự báo cuối kỳ backtest, không phải dự báo hiện tại.

## Kết luận

{protocol['conclusion_vi']}

Không đổi lệnh dự đoán mặc định. Chi tiết từng ngày, số trúng và P&L nằm trong `daily_detail.csv`; dữ liệu theo từng mô hình ở `daily.csv`.
"""


def run(
    source: Path,
    output: Path,
    start_date: str = "2025-01-01",
    history_days: int = HISTORY_DAYS,
    end_date: str | None = None,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    start = parse_date(start_date)
    end = parse_date(end_date) if end_date is not None else None
    if end is not None and end < start:
        raise ValueError("end_date must be on or after start_date.")
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output}")
    if isinstance(history_days, bool) or not isinstance(history_days, int) or history_days < 1:
        raise ValueError("history_days must be a positive integer.")

    raw = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    normalized = normalize_prizes(raw)
    counts = count_three_digit_endings(normalized)
    builder = ThreeDigitFeatureBuilder(counts)
    dates = _target_dates(counts, start)
    if end is not None:
        dates = dates[dates <= end]
    if dates.empty:
        raise ValueError("No complete target days with a preceding calendar result.")

    score_parts: dict[str, list[np.ndarray]] = {strategy: [] for strategy in STRATEGIES}
    cutoff_rows = []
    for period in dates.to_period("Q").unique():
        block = dates[dates.to_period("Q") == period]
        quarter_start = max(period.start_time, start)
        eligible_labels = counts.index[
            (counts.index - pd.Timedelta(days=1)).isin(counts.index)
        ]
        prior_labels = eligible_labels[eligible_labels < quarter_start]
        if prior_labels.empty:
            raise ValueError(f"No training labels before {quarter_start:%Y-%m-%d}.")
        final_train_label = prior_labels[-1]
        first_train_label = final_train_label - pd.Timedelta(days=history_days - 1)
        train_labels = prior_labels[prior_labels >= first_train_label]
        train_origins = train_labels - pd.Timedelta(days=1)
        if not train_origins.isin(counts.index).all():
            raise ValueError(f"Missing preceding feature dates for training period {period}.")
        if (train_labels >= block[0]).any():
            raise AssertionError("Training labels overlap evaluation targets.")

        poisson_scores, binary_scores = _fit_pair(
            builder, train_labels, block
        )
        score_parts["poisson_3digit_23"].append(poisson_scores)
        score_parts["binary_3digit_23"].append(binary_scores)
        score_parts["frequency_90d_3digit_23"].append(
            _rolling_scores(builder, block, 90)
        )
        score_parts["frequency_5y_suffix_shrinkage_3digit_23"].append(
            _rolling_suffix_shrinkage_scores(builder, block)
        )
        cutoff_rows.append({
            "quarter": str(period),
            "first_target_date": block[0].strftime("%Y-%m-%d"),
            "last_target_date": block[-1].strftime("%Y-%m-%d"),
            "training_label_start": train_labels[0].strftime("%Y-%m-%d"),
            "training_label_end": final_train_label.strftime("%Y-%m-%d"),
            "training_days": len(train_labels),
            "history_calendar_days": history_days,
        })
        print(
            f"FIT {period}; train labels through {final_train_label:%Y-%m-%d}; "
            f"test={len(block)} draws",
            flush=True,
        )

    combined_scores = {
        name: np.concatenate(parts, axis=0)
        for name, parts in score_parts.items()
    }
    if any(array.shape != (len(dates), 1000) for array in combined_scores.values()):
        raise AssertionError("A strategy has missing or malformed prediction scores.")
    daily, yearly, monthly = _summaries(dates, combined_scores, counts)
    daily_detail = _daily_detail(daily, counts)

    latest = dates[-1]
    latest_scores = combined_scores["poisson_3digit_23"][-1]
    latest_order = np.argsort(-latest_scores, kind="stable")[:TOP_K]
    latest_forecast = pd.DataFrame({
        "forecast_date": [latest.strftime("%Y-%m-%d")] * TOP_K,
        "feature_date": [(latest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")] * TOP_K,
        "rank": np.arange(1, TOP_K + 1),
        "number": [f"{number:03d}" for number in latest_order],
        "score": latest_scores[latest_order],
        "score_target": "expected_occurrences_among_23_three_or_more_digit_endings",
    })

    random_hit_reference = 1 - (1 - TOP_K / 1000) ** 23
    binary_yearly = yearly.loc[yearly.strategy.eq("binary_3digit_23")].set_index("year")
    poisson_yearly = yearly.loc[yearly.strategy.eq("poisson_3digit_23")].set_index("year")
    comparison = pd.DataFrame({
        "hit_rate_binary": binary_yearly.hit_rate,
        "hit_rate_poisson": poisson_yearly.hit_rate,
        "delta_low": binary_yearly.delta_hit_rate_vs_poisson_block95_low,
        "delta_high": binary_yearly.delta_hit_rate_vs_poisson_block95_high,
    })
    if len(comparison) > 1 and (comparison.hit_rate_binary > comparison.hit_rate_poisson).all() and (comparison.delta_low > 0).all():
        conclusion = (
            "Mô hình LightGBM binary 3 chữ số thắng Poisson trên cả các giai đoạn, "
            "và khoảng chênh bootstrap không chứa 0. Đây vẫn là kết quả hồi cứu; "
            "chỉ nên thử paper-trial tiền cứu, chưa đổi mặc định."
        )
    elif len(comparison) > 1 and (comparison.hit_rate_binary < comparison.hit_rate_poisson).all():
        conclusion = (
            "Mô hình LightGBM binary 3 chữ số có hit-rate thấp hơn đối chứng Poisson "
            "trong mọi giai đoạn được đo. Chưa có cơ sở dùng thay Poisson; giữ nguyên "
            "mặc định và không kết luận có lợi thế dự đoán."
        )
    else:
        conclusion = (
            "Kết quả không ổn định giữa các giai đoạn và khoảng bootstrap ghép ngày "
            "cần được xem cùng số liệu từng năm. Chưa có cơ sở đổi mặc định; cần "
            "paper-trial tiền cứu với giao thức khóa trước."
        )
    totals = daily.groupby("strategy", sort=False).agg(
        days=("date", "nunique"),
        hit_days=("hit_day", "sum"),
        occurrences=("target_occurrences", "sum"),
        stake=("stake_units", "sum"),
        gross=("gross_payout_units", "sum"),
        net=("net_profit_units", "sum"),
    )
    total_lines = [
        f"{strategy.removesuffix('_3digit_23')}: {int(row.hit_days)}/{int(row.days)} "
        f"ngày hit, {int(row.occurrences)} lượt trúng, lãi ròng {int(row.net):+,} đơn vị"
        for strategy, row in totals.iterrows()
    ]
    annual_lines = []
    for year in sorted(set(dates.year)):
        year_frame = yearly.loc[yearly.year.eq(year)].set_index("strategy")
        annual_lines.append(
            f"{year}: "
            + "; ".join(
                f"{strategy.removesuffix('_3digit_23')} "
                f"{int(year_frame.loc[strategy, 'hit_days'])}/{int(year_frame.loc[strategy, 'days'])} hit, "
                f"{int(year_frame.loc[strategy, 'net_profit_units']):+,}"
                for strategy in STRATEGIES
            )
            + " đơn vị"
        )
    conclusion = (
        f"Kết quả toàn kỳ ({len(dates)} ngày): "
        + "; ".join(total_lines)
        + ". Theo năm: "
        + " | ".join(annual_lines)
        + ". Đây là P&L hồi cứu theo mức cược/trả đã nêu; so sánh lãi/lỗ không chứng minh "
        "mô hình có lợi thế ổn định."
    )

    protocol = {
        "source": str(source),
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_start": counts.index.min().strftime("%Y-%m-%d"),
        "source_end": counts.index.max().strftime("%Y-%m-%d"),
        "evaluation_start": dates[0].strftime("%Y-%m-%d"),
        "evaluation_end": dates[-1].strftime("%Y-%m-%d"),
        "target_digit_count": 3,
        "candidate_numbers": 1000,
        "target_draw_count_per_day": 23,
        "supersedes_report": "archived earlier two-digit interpretation",
        "excluded_prize_columns": [
            column for column in PRIZE_COLUMNS if column.startswith("prize7_")
        ],
        "leading_zero_padding": True,
        "top_k": TOP_K,
        "history_days": history_days,
        "refit_policy": "before each calendar quarter; training labels strictly before quarter start",
        "strategies": list(STRATEGIES),
        "exploratory_strategy": {
            "name": "frequency_5y_suffix_shrinkage_3digit_23",
            "history_calendar_days": LONG_FREQUENCY_WINDOW_DAYS,
            "suffix_count_weight": SUFFIX_SHRINKAGE,
            "selection_note": (
                "Exploratory candidate evaluated after inspecting the 2025-2026 "
                "period; results are not an independent holdout."
            ),
        },
        "model_params": MODEL_PARAMS,
        "bootstrap_block_length_observed_draws": 7,
        "target_hit_reference_random_independent_uniform": random_hit_reference,
        "payout_rule": {
            "stake_per_number_per_day": STAKE_PER_NUMBER,
            "gross_payout_per_matching_occurrence": GROSS_PAYOUT_PER_OCCURRENCE,
            "daily_top3_stake": TOP_K * STAKE_PER_NUMBER,
            "net_profit_formula": "gross payout per occurrence * selected occurrence count - daily Top-3 stake",
            "repeats_paid_per_occurrence": True,
            "currency_or_unit": "same unspecified unit as user's 23-to-900 payout quote",
        },
        "stress_period_is_untouched": False,
        "production_default_changed": False,
        "latest_forecast_date": latest.strftime("%Y-%m-%d"),
        "conclusion_vi": conclusion,
    }
    report = _report(protocol, yearly, monthly, latest_forecast)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / "source_snapshot.csv")
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    daily_detail.to_csv(output / "daily_detail.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cutoff_rows).to_csv(output / "training_cutoffs.csv", index=False)
    latest_forecast.to_csv(output / "latest_forecast.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"REPORT {output / 'report.md'}", flush=True)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local raw 27-prize XSMB CSV.")
    parser.add_argument("--output", type=Path, required=True, help="New research output directory.")
    parser.add_argument("--start-date", default="2025-01-01", help="First test target date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Last test target date inclusive (YYYY-MM-DD).")
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
