"""Paired walk-forward study of Top-3 picks restricted to 000-099."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from scripts.research.research_top3_3digit import (
    GROSS_PAYOUT_PER_OCCURRENCE,
    HISTORY_DAYS,
    MODEL_PARAMS,
    SOURCE_URL,
    STAKE_PER_NUMBER,
    TOP_K,
    ThreeDigitFeatureBuilder,
    _fit_pair,
    _max_streak,
    _rolling_scores,
    _target_dates,
    count_three_digit_endings,
    parse_date,
)
from src.diagnostics import block_bootstrap_mean_interval
from src.prize_features import normalize_prizes


RESTRICTED_CANDIDATES = np.arange(100, dtype=np.int64)
ALL_CANDIDATES = np.arange(1000, dtype=np.int64)
MODEL_NAMES = ("poisson", "binary", "frequency_90d")
SCOPES = {
    "all_000_999": ALL_CANDIDATES,
    "restricted_000_099": RESTRICTED_CANDIDATES,
}


def rank_candidates(
    scores: np.ndarray, candidates: np.ndarray, top_k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """Rank only the supplied candidates, preserving deterministic tie-breaking."""
    values = np.asarray(scores)
    candidate_ids = np.asarray(candidates)
    if values.ndim != 2 or values.shape[1] != 1000 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite (days, 1000) matrix.")
    if (
        candidate_ids.ndim != 1
        or len(candidate_ids) < top_k
        or not np.issubdtype(candidate_ids.dtype, np.integer)
        or (candidate_ids < 0).any()
        or (candidate_ids >= 1000).any()
        or len(np.unique(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("candidates must contain unique integer IDs in 0..999.")
    local_order = np.argsort(
        -values[:, candidate_ids], axis=1, kind="stable"
    )[:, :top_k]
    order = candidate_ids[local_order]
    selected = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(selected, order, True, axis=1)
    return selected, order


def random_pick_daily_hit_probabilities(
    actual_counts: np.ndarray, candidates: np.ndarray, top_k: int = TOP_K
) -> np.ndarray:
    """Per-day conditional hit chances for random picks without replacement."""
    actual = np.asarray(actual_counts)
    candidate_ids = np.asarray(candidates)
    if actual.ndim != 2 or actual.shape[1] != 1000 or (actual < 0).any():
        raise ValueError("actual_counts must be a nonnegative (days, 1000) matrix.")
    population = len(candidate_ids)
    denominator = math.comb(population, top_k)
    distinct = (actual[:, candidate_ids] > 0).sum(axis=1)
    no_hit = np.array([
        math.comb(population - int(count), top_k) / denominator
        if population - int(count) >= top_k else 0.0
        for count in distinct
    ])
    return 1.0 - no_hit


def random_pick_hit_probability(
    actual_counts: np.ndarray, candidates: np.ndarray, top_k: int = TOP_K
) -> float:
    """Mean conditional hit chance for random picks from a candidate set."""
    return float(np.mean(
        random_pick_daily_hit_probabilities(actual_counts, candidates, top_k)
    ))


def poisson_binomial_upper_tail(probabilities: np.ndarray, observed: int) -> float:
    """Exact P(X >= observed) for independent Bernoulli trials with varying p."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a nonempty finite vector.")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("probabilities must lie in [0, 1].")
    if isinstance(observed, bool) or not isinstance(observed, (int, np.integer)):
        raise ValueError("observed must be an integer.")
    if observed <= 0:
        return 1.0
    if observed > len(values):
        return 0.0
    distribution = np.zeros(len(values) + 1, dtype=float)
    distribution[0] = 1.0
    for index, probability in enumerate(values, 1):
        distribution[1:index + 1] = (
            distribution[1:index + 1] * (1 - probability)
            + distribution[:index] * probability
        )
        distribution[0] *= 1 - probability
    return float(distribution[observed:].sum())


def evaluate_scores(
    dates: pd.DatetimeIndex,
    counts: pd.DataFrame,
    score_sets: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate every model under unrestricted and 000-099-only ranking."""
    actual = counts.loc[dates].to_numpy(dtype=np.int64)
    rows: list[pd.DataFrame] = []
    for model in MODEL_NAMES:
        scores = score_sets[model]
        if scores.shape != (len(dates), 1000):
            raise ValueError(f"Malformed scores for {model}.")
        for scope, candidates in SCOPES.items():
            selected, order = rank_candidates(scores, candidates)
            distinct_hits = ((actual > 0) & selected).sum(axis=1)
            occurrences = (actual * selected).sum(axis=1)
            stake = selected.sum(axis=1) * STAKE_PER_NUMBER
            gross = occurrences * GROSS_PAYOUT_PER_OCCURRENCE
            rows.append(pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "model": model,
                "scope": scope,
                "strategy": f"{model}_{scope}",
                "prediction_top3": [
                    " ".join(f"{number:03d}" for number in row) for row in order
                ],
                "hit_day": distinct_hits > 0,
                "distinct_hits": distinct_hits,
                "target_occurrences": occurrences,
                "stake_units": stake,
                "gross_payout_units": gross,
                "net_profit_units": gross - stake,
            }))
    return pd.concat(rows, ignore_index=True)


def _aggregate(group: pd.DataFrame) -> dict[str, float | int]:
    hit = group.hit_day.to_numpy(dtype=float)
    low, high = block_bootstrap_mean_interval(
        hit, block_length=7, n_resamples=5000, seed=42
    )
    stake = int(group.stake_units.sum())
    profit = int(group.net_profit_units.sum())
    dates = pd.DatetimeIndex(group.date)
    misses = ~group.hit_day.to_numpy(dtype=bool)
    return {
        "days": len(group),
        "hit_days": int(hit.sum()),
        "hit_rate": float(hit.mean()),
        "hit_rate_block95_low": low,
        "hit_rate_block95_high": high,
        "distinct_hits": int(group.distinct_hits.sum()),
        "target_occurrences": int(group.target_occurrences.sum()),
        "stake_units": stake,
        "gross_payout_units": int(group.gross_payout_units.sum()),
        "net_profit_units": profit,
        "roi": float(profit / stake),
        "max_miss_streak": _max_streak(misses, dates),
    }


def build_summaries(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    yearly_rows = []
    for (model, scope), group in daily.groupby(["model", "scope"], sort=False):
        overall_rows.append({"model": model, "scope": scope, **_aggregate(group)})
        years = pd.DatetimeIndex(group.date).year
        for year in sorted(set(years)):
            yearly_rows.append({
                "year": int(year),
                "model": model,
                "scope": scope,
                **_aggregate(group.loc[years == year]),
            })
    return pd.DataFrame(overall_rows), pd.DataFrame(yearly_rows)


def build_comparisons(daily: pd.DataFrame) -> pd.DataFrame:
    """Paired restricted-minus-unrestricted differences by model."""
    rows = []
    for model in MODEL_NAMES:
        frame = daily.loc[daily.model.eq(model)].pivot(
            index="date", columns="scope",
            values=["hit_day", "target_occurrences", "net_profit_units"],
        ).sort_index()
        restricted_hits = frame[("hit_day", "restricted_000_099")].to_numpy(float)
        all_hits = frame[("hit_day", "all_000_999")].to_numpy(float)
        hit_delta = restricted_hits - all_hits
        occurrence_delta = (
            frame[("target_occurrences", "restricted_000_099")].to_numpy(float)
            - frame[("target_occurrences", "all_000_999")].to_numpy(float)
        )
        profit_delta = (
            frame[("net_profit_units", "restricted_000_099")].to_numpy(float)
            - frame[("net_profit_units", "all_000_999")].to_numpy(float)
        )
        hit_low, hit_high = block_bootstrap_mean_interval(
            hit_delta, block_length=7, n_resamples=5000, seed=42
        )
        occurrence_low, occurrence_high = block_bootstrap_mean_interval(
            occurrence_delta, block_length=7, n_resamples=5000, seed=42
        )
        profit_low, profit_high = block_bootstrap_mean_interval(
            profit_delta, block_length=7, n_resamples=5000, seed=42
        )
        rows.append({
            "model": model,
            "days": len(frame),
            "restricted_hit_days": int(restricted_hits.sum()),
            "unrestricted_hit_days": int(all_hits.sum()),
            "restricted_hit_rate": float(restricted_hits.mean()),
            "unrestricted_hit_rate": float(all_hits.mean()),
            "delta_hit_rate": float(hit_delta.mean()),
            "delta_hit_rate_block95_low": hit_low,
            "delta_hit_rate_block95_high": hit_high,
            "delta_occurrences_per_day": float(occurrence_delta.mean()),
            "delta_occurrences_per_day_block95_low": occurrence_low,
            "delta_occurrences_per_day_block95_high": occurrence_high,
            "delta_net_profit_units": int(profit_delta.sum()),
            "delta_profit_per_day_block95_low": profit_low,
            "delta_profit_per_day_block95_high": profit_high,
        })
    return pd.DataFrame(rows)


def _format_summary_rows(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"| {row.model} | {row.scope} | {row.hit_days}/{row.days} "
            f"({row.hit_rate:.2%}; {row.hit_rate_block95_low:.2%}–"
            f"{row.hit_rate_block95_high:.2%}) | {row.target_occurrences} | "
            f"{row.max_miss_streak} | {row.net_profit_units:+,} | {row.roi:+.2%} |"
        )
    return "\n".join(rows)


def _format_comparison_rows(comparisons: pd.DataFrame) -> str:
    rows = []
    for row in comparisons.itertuples(index=False):
        rows.append(
            f"| {row.model} | {row.restricted_hit_days}/{row.days} "
            f"({row.restricted_hit_rate:.2%}) | {row.unrestricted_hit_days}/{row.days} "
            f"({row.unrestricted_hit_rate:.2%}) | {row.delta_hit_rate:+.2%} "
            f"({row.delta_hit_rate_block95_low:+.2%}–"
            f"{row.delta_hit_rate_block95_high:+.2%}) | "
            f"{row.delta_net_profit_units:+,} |"
        )
    return "\n".join(rows)


def make_report(
    protocol: dict,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    comparisons: pd.DataFrame,
    latest: pd.DataFrame,
) -> str:
    poisson = comparisons.loc[comparisons.model.eq("poisson")].iloc[0]
    if poisson.delta_hit_rate_block95_low > 0:
        conclusion = "Giới hạn 000–099 làm tăng hit-rate Poisson với khoảng bootstrap không chứa 0."
    elif poisson.delta_hit_rate_block95_high < 0:
        conclusion = "Giới hạn 000–099 làm giảm hit-rate Poisson với khoảng bootstrap không chứa 0."
    else:
        conclusion = (
            "Chưa có bằng chứng giới hạn 000–099 làm tăng hit-rate Poisson; "
            "khoảng chênh lệch bootstrap vẫn chứa 0."
        )
    latest_rows = "\n".join(
        f"| {row.model} | {row.scope} | {row.prediction_top3} |"
        for row in latest.itertuples(index=False)
    )
    yearly_rows = []
    for row in yearly.itertuples(index=False):
        yearly_rows.append(
            f"| {row.year} | {row.model} | {row.scope} | {row.hit_days}/{row.days} "
            f"({row.hit_rate:.2%}) | {row.target_occurrences} | "
            f"{row.net_profit_units:+,} |"
        )
    return f"""# Kiểm định Top-3 chỉ trong 000–099

## Câu hỏi và protocol

- Mỗi ngày vẫn chọn đúng 3 số ba chữ số, nhưng chiến lược hạn chế chỉ được xếp hạng `000`–`099`.
- Đối chứng chọn Top-3 từ đủ `000`–`999` bằng chính cùng điểm số mô hình trên cùng ngày. Vì vậy phép so sánh ghép cặp đo riêng tác động của giới hạn không gian chọn số.
- Target gồm 23 đuôi ba chữ số mỗi ngày, loại bốn Giải bảy hai chữ số, giữ số 0 đầu và các lần lặp.
- Walk-forward theo quý, cửa sổ lịch sử {protocol['history_days']} ngày; nhãn huấn luyện cuối luôn trước ngày kiểm thử đầu của quý.
- Kỳ kiểm thử: **{protocol['evaluation_start']} đến {protocol['evaluation_end']}** ({protocol['evaluation_days']} ngày có đủ ngày liền trước).
- Nhóm `000–099` chiếm **{protocol['restricted_actual_occurrences']:,}/{protocol['all_actual_occurrences']:,} = {protocol['restricted_actual_share']:.2%}** lượt kết quả thực tế; tỷ lệ đồng đều kỳ vọng là 10%.
- Xác suất hit trung bình nếu chọn ngẫu nhiên 3 số, có điều kiện trên kết quả thực tế: toàn bộ `000–999` **{protocol['random_hit_reference_all']:.2%}**; chỉ `000–099` **{protocol['random_hit_reference_restricted']:.2%}**.
- P&L dùng cùng giả định: 23 đơn vị/số/ngày, trả gộp 900 đơn vị cho mỗi lượt xuất hiện.

## Kết quả toàn kỳ

| Mô hình | Không gian chọn | Ngày hit (block 95%) | Lượt trúng | Miss tối đa | Lãi ròng | ROI |
|---|---|---:|---:|---:|---:|---:|
{_format_summary_rows(summary)}

## So sánh ghép cặp: giới hạn trừ không giới hạn

| Mô hình | 000–099 | 000–999 | Chênh hit-rate (block 95%) | Chênh lãi ròng |
|---|---:|---:|---:|---:|
{_format_comparison_rows(comparisons)}

Khoảng tin cậy dùng circular block bootstrap 7 ngày, 5.000 lần lấy mẫu. Đây là bất định hồi cứu, không hiệu chỉnh cho việc thử nhiều ý tưởng.

## Kiểm định bổ sung

- Tỷ trọng thực tế của nhóm `0xx` so với mốc 10%: binomial hai phía **p={protocol['restricted_share_binomial_p_two_sided']:.3f}**.
- Poisson giới hạn so với chọn ngẫu nhiên 3 số trong `000–099`, có điều kiện trên từng ngày thực tế: Poisson-binomial một phía **p={protocol['poisson_restricted_vs_random_p_one_sided']:.3f}**.
- So sánh hit ghép cặp Poisson: giới hạn thắng riêng {protocol['poisson_restricted_only_hit_days']} ngày, không giới hạn thắng riêng {protocol['poisson_unrestricted_only_hit_days']} ngày; McNemar exact hai phía **p={protocol['poisson_paired_mcnemar_p_two_sided']:.3f}**.

## Theo năm

| Năm | Mô hình | Không gian chọn | Ngày hit | Lượt trúng | Lãi ròng |
|---:|---|---|---:|---:|---:|
{chr(10).join(yearly_rows)}

## Top-3 của ngày kiểm thử cuối

| Mô hình | Không gian chọn | Top-3 |
|---|---|---|
{latest_rows}

Đây là dự báo hồi cứu trước kết quả ngày {protocol['latest_backtest_date']}, không phải dự báo hiện tại.

## Kết luận

{conclusion} Kết luận chính phải dựa vào hàng Poisson ghép cặp và độ ổn định theo năm, không dựa vào một tháng hoặc một lần trúng riêng lẻ.
"""


def run(
    source: Path,
    output: Path,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    history_days: int = HISTORY_DAYS,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    start = parse_date(start_date)
    end = parse_date(end_date) if end_date else None
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    if end is not None and end < start:
        raise ValueError("end_date must not precede start_date.")
    raw = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    counts = count_three_digit_endings(normalize_prizes(raw))
    builder = ThreeDigitFeatureBuilder(counts)
    dates = _target_dates(counts, start)
    if end is not None:
        dates = dates[dates <= end]
    if dates.empty:
        raise ValueError("No complete evaluation dates.")

    score_parts = {model: [] for model in MODEL_NAMES}
    cutoff_rows = []
    eligible_labels = counts.index[(counts.index - pd.Timedelta(days=1)).isin(counts.index)]
    for period in dates.to_period("Q").unique():
        block = dates[dates.to_period("Q") == period]
        quarter_start = max(period.start_time, start)
        prior = eligible_labels[eligible_labels < quarter_start]
        if prior.empty:
            raise ValueError(f"No training labels before {quarter_start:%Y-%m-%d}.")
        final_train = prior[-1]
        first_train = final_train - pd.Timedelta(days=history_days - 1)
        train = prior[prior >= first_train]
        poisson, binary = _fit_pair(builder, train, block)
        score_parts["poisson"].append(poisson)
        score_parts["binary"].append(binary)
        score_parts["frequency_90d"].append(_rolling_scores(builder, block, 90))
        cutoff_rows.append({
            "quarter": str(period),
            "first_target_date": block[0].strftime("%Y-%m-%d"),
            "last_target_date": block[-1].strftime("%Y-%m-%d"),
            "training_label_start": train[0].strftime("%Y-%m-%d"),
            "training_label_end": final_train.strftime("%Y-%m-%d"),
            "training_days": len(train),
        })
        print(
            f"FIT {period}; train through {final_train:%Y-%m-%d}; test={len(block)}",
            flush=True,
        )
    score_sets = {
        model: np.concatenate(parts, axis=0) for model, parts in score_parts.items()
    }
    daily = evaluate_scores(dates, counts, score_sets)
    summary, yearly = build_summaries(daily)
    comparisons = build_comparisons(daily)
    latest_date = dates[-1].strftime("%Y-%m-%d")
    latest = daily.loc[daily.date.eq(latest_date), [
        "model", "scope", "prediction_top3"
    ]].copy()

    actual = counts.loc[dates].to_numpy(dtype=np.int64)
    restricted_occurrences = int(actual[:, :100].sum())
    all_occurrences = int(actual.sum())
    restricted_random_daily = random_pick_daily_hit_probabilities(
        actual, RESTRICTED_CANDIDATES
    )
    poisson_daily = daily.loc[daily.model.eq("poisson")].pivot(
        index="date", columns="scope", values="hit_day"
    ).sort_index()
    poisson_restricted_hits = poisson_daily["restricted_000_099"].to_numpy(bool)
    poisson_unrestricted_hits = poisson_daily["all_000_999"].to_numpy(bool)
    restricted_only = int((poisson_restricted_hits & ~poisson_unrestricted_hits).sum())
    unrestricted_only = int((poisson_unrestricted_hits & ~poisson_restricted_hits).sum())
    protocol = {
        "source": str(source),
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_start": counts.index.min().strftime("%Y-%m-%d"),
        "source_end": counts.index.max().strftime("%Y-%m-%d"),
        "evaluation_start": dates[0].strftime("%Y-%m-%d"),
        "evaluation_end": dates[-1].strftime("%Y-%m-%d"),
        "evaluation_days": len(dates),
        "history_days": history_days,
        "top_k": TOP_K,
        "restricted_candidates": "000-099",
        "control_candidates": "000-999",
        "target_outcomes_per_day": 23,
        "restricted_actual_occurrences": restricted_occurrences,
        "all_actual_occurrences": all_occurrences,
        "restricted_actual_share": restricted_occurrences / all_occurrences,
        "restricted_share_binomial_p_two_sided": float(
            binomtest(restricted_occurrences, all_occurrences, 0.1).pvalue
        ),
        "random_hit_reference_all": random_pick_hit_probability(actual, ALL_CANDIDATES),
        "random_hit_reference_restricted": float(restricted_random_daily.mean()),
        "poisson_restricted_vs_random_p_one_sided": poisson_binomial_upper_tail(
            restricted_random_daily, int(poisson_restricted_hits.sum())
        ),
        "poisson_restricted_only_hit_days": restricted_only,
        "poisson_unrestricted_only_hit_days": unrestricted_only,
        "poisson_paired_mcnemar_p_two_sided": float(
            binomtest(
                restricted_only,
                restricted_only + unrestricted_only,
                0.5,
            ).pvalue
        ),
        "model_params": MODEL_PARAMS,
        "latest_backtest_date": latest_date,
        "production_default_changed": False,
    }
    report = make_report(protocol, summary, yearly, comparisons, latest)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / "source_snapshot.csv")
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(output / "latest_backtest_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cutoff_rows).to_csv(output / "training_cutoffs.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"REPORT {output / 'report.md'}", flush=True)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
