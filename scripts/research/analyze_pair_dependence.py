"""Screen pairwise XSMB co-occurrence against a fair independent-draw null.

The analysis is retrospective and exploratory. It controls family-wise error
within each fixed chronological period but cannot establish a predictive edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.data_loader import NUMBER_COLUMNS, XSMBDataLoader


DRAW_COUNT = 27
NUMBER_COUNT = 100
DISCOVERY_END = pd.Timestamp("2022-12-31")
REPLICATION_START = pd.Timestamp("2023-01-01")
ALPHA = 0.05
PAIR_COUNT = NUMBER_COUNT * (NUMBER_COUNT - 1) // 2
FAIR_PAIR_PROBABILITY = (
    1 - 2 * (1 - 1 / NUMBER_COUNT) ** DRAW_COUNT
    + (1 - 2 / NUMBER_COUNT) ** DRAW_COUNT
)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm step-down adjusted p-values, valid under arbitrary dependence."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a nonempty one-dimensional array.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must be finite values in [0, 1].")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    multipliers = len(values) - np.arange(len(values))
    adjusted = np.minimum(1.0, np.maximum.accumulate(ordered * multipliers))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _wilson_interval(successes: np.ndarray, trials: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized two-sided 95% Wilson intervals for observed pair rates."""
    if trials < 1:
        raise ValueError("At least one observed date is required.")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width


def validate_frequency(frequency: pd.DataFrame) -> None:
    """Validate the canonical 00..99 daily frequency matrix."""
    if not isinstance(frequency, pd.DataFrame):
        raise TypeError("frequency must be a pandas DataFrame.")
    if not isinstance(frequency.index, pd.DatetimeIndex):
        raise ValueError("frequency must have a DatetimeIndex.")
    if frequency.empty or frequency.index.has_duplicates:
        raise ValueError("frequency must contain unique observed dates.")
    if not frequency.index.is_monotonic_increasing:
        raise ValueError("frequency dates must be sorted in ascending order.")
    if list(frequency.columns) != list(NUMBER_COLUMNS):
        raise ValueError("frequency columns must be the ordered number labels 00..99.")
    values = frequency.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values != np.floor(values)).any()
        or not np.all(values.sum(axis=1) == DRAW_COUNT)
    ):
        raise ValueError("Every date must contain nonnegative integer counts totaling 27.")


def split_periods(
    frequency: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the predeclared 2005–2022 discovery / 2023+ replication split."""
    validate_frequency(frequency)
    discovery = frequency.loc[:DISCOVERY_END]
    replication = frequency.loc[REPLICATION_START:]
    if len(discovery) < 365 or len(replication) < 365:
        raise ValueError("Discovery and replication periods must each contain >=365 dates.")
    return discovery, replication


def screen_pairs(frequency: pd.DataFrame) -> pd.DataFrame:
    """Test all 4,950 pairs against daily fair IID 27-draw co-occurrence rates."""
    validate_frequency(frequency)
    presence = frequency.to_numpy(dtype=np.int64) > 0
    left, right = np.triu_indices(NUMBER_COUNT, k=1)
    observed = np.count_nonzero(presence[:, left] & presence[:, right], axis=0)
    days = len(frequency)
    raw_p = np.fromiter(
        (binomtest(int(count), days, FAIR_PAIR_PROBABILITY).pvalue for count in observed),
        dtype=float,
        count=PAIR_COUNT,
    )
    adjusted_p = holm_adjust(raw_p)
    ci_low, ci_high = _wilson_interval(observed, days)
    rates = observed / days
    effects = rates - FAIR_PAIR_PROBABILITY
    result = pd.DataFrame({
        "number_a": np.asarray(NUMBER_COLUMNS)[left],
        "number_b": np.asarray(NUMBER_COLUMNS)[right],
        "days": days,
        "cooccurrence_days": observed,
        "expected_days_under_null": days * FAIR_PAIR_PROBABILITY,
        "observed_rate": rates,
        "null_rate": FAIR_PAIR_PROBABILITY,
        "effect_percentage_points": effects * 100,
        "observed_rate_ci95_low": ci_low,
        "observed_rate_ci95_high": ci_high,
        "effect_ci95_low_percentage_points": (ci_low - FAIR_PAIR_PROBABILITY) * 100,
        "effect_ci95_high_percentage_points": (ci_high - FAIR_PAIR_PROBABILITY) * 100,
        "p_value_exact_binomial": raw_p,
        "p_value_holm": adjusted_p,
    })
    return result.sort_values(
        ["p_value_holm", "p_value_exact_binomial", "number_a", "number_b"],
        kind="stable",
        ignore_index=True,
    )


def compare_periods(discovery: pd.DataFrame, replication: pd.DataFrame) -> pd.DataFrame:
    """Mark pairs significant in both periods with a consistent effect direction."""
    keys = ["number_a", "number_b"]
    comparison = discovery.merge(
        replication, on=keys, suffixes=("_discovery", "_replication"), validate="one_to_one"
    )
    same_direction = (
        comparison.effect_percentage_points_discovery
        * comparison.effect_percentage_points_replication
    ) > 0
    comparison["same_effect_direction"] = same_direction
    comparison["replicated_significant"] = (
        comparison.p_value_holm_discovery.le(ALPHA)
        & comparison.p_value_holm_replication.le(ALPHA)
        & same_direction
    )
    return comparison.sort_values(
        ["replicated_significant", "p_value_holm_discovery", "p_value_holm_replication"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )


def _period_details(frequency: pd.DataFrame) -> dict[str, object]:
    calendar = pd.date_range(frequency.index[0], frequency.index[-1], freq="D")
    return {
        "start": frequency.index[0].strftime("%Y-%m-%d"),
        "end": frequency.index[-1].strftime("%Y-%m-%d"),
        "observed_days": len(frequency),
        "missing_calendar_dates": len(calendar.difference(frequency.index)),
    }


def _recommendation(replicated_pair_count: int) -> str:
    if replicated_pair_count:
        return (
            "Một bất thường tỷ lệ cặp đã tái lập theo giao thức hồi cứu này. "
            "Đây chỉ là lý do để thiết kế paper-trial tiền cứu riêng, đăng ký trước; "
            "không phải khuyến nghị đổi dự đoán production."
        )
    return (
        "Không có cặp nào vượt hiệu chỉnh nhiều phép thử ở cả hai giai đoạn và "
        "cùng chiều hiệu ứng. Không thêm luật dự đoán theo cặp; giữ nguyên mặc "
        "định production hiện tại."
    )


def _report_text(summary: dict[str, object], discovery: pd.DataFrame,
                 replication: pd.DataFrame) -> str:
    top_rows = []
    for label, results in (("Khám phá", discovery), ("Đối chiếu", replication)):
        for row in results.head(5).itertuples(index=False):
            top_rows.append(
                f"| {label} | {str(row.number_a).zfill(2)}–{str(row.number_b).zfill(2)} | "
                f"{row.cooccurrence_days}/{row.days} | "
                f"{row.effect_percentage_points:+.3f} | {row.p_value_holm:.4g} |"
            )
    top_table = "\n".join(top_rows)
    discovery_info = summary["discovery"]
    replication_info = summary["replication"]
    return f"""# Nghiên cứu đồng xuất hiện theo cặp

## Giao thức

- Giai đoạn khám phá: {discovery_info['start']} đến {discovery_info['end']}.
- Giai đoạn đối chiếu: {replication_info['start']} đến {replication_info['end']}.
- Kiểm tra toàn bộ {PAIR_COUNT:,} cặp không thứ tự của các đuôi số khác nhau trong mỗi giai đoạn.
- Giả thuyết không: 27 kết quả mỗi ngày là các lượt rút độc lập, đồng xác suất từ 00–99. Theo giả thuyết này, xác suất một cặp cố định cùng xuất hiện ít nhất một lần là
  `1 - 2(0.99)^27 + (0.98)^27 = {FAIR_PAIR_PROBABILITY:.6%}` mỗi ngày có kết quả.
- Mỗi cặp dùng kiểm định nhị thức chính xác hai phía trên các ngày có kết quả. Hiệu chỉnh Holm kiểm soát sai số toàn cục trên cả 4.950 cặp trong từng giai đoạn, kể cả khi các phép thử phụ thuộc nhau.
- Hiệu ứng là tỷ lệ ngày cặp số cùng xuất hiện trừ tỷ lệ theo giả thuyết không. Khoảng Wilson 95% được tính cho tỷ lệ quan sát; trừ tỷ lệ giả thuyết không để ra khoảng hiệu ứng hiển thị.
- Dữ liệu lịch sử đã được xem trong nghiên cứu trước. Giai đoạn sau là đối chiếu hồi cứu, không phải holdout chưa xem hoặc xác nhận tiền cứu.

## Kết quả

- Giai đoạn khám phá: **{summary['discovery_significant_pairs']} / {PAIR_COUNT:,}** cặp có p đã hiệu chỉnh Holm ≤ 0,05.
- Giai đoạn đối chiếu: **{summary['replication_significant_pairs']} / {PAIR_COUNT:,}** cặp có p đã hiệu chỉnh Holm ≤ 0,05.
- Có ý nghĩa ở cả hai giai đoạn và cùng chiều hiệu ứng: **{summary['replicated_same_direction_pairs']}** cặp.
- Ngày lịch thiếu kết quả được loại ra, không nội suy. Mẫu số mỗi giai đoạn là số ngày có kết quả đã được xác thực.

| Giai đoạn | Cặp số | Số ngày cùng xuất hiện | Hiệu ứng so với giả thuyết không (điểm %) | p hiệu chỉnh Holm |
|---|---|---:|---:|---:|
{top_table}

Bảng liệt kê năm p hiệu chỉnh nhỏ nhất mỗi giai đoạn; đây không phải phát hiện có ý nghĩa nếu p đã hiệu chỉnh lớn hơn 0,05. Kết quả đầy đủ của mọi cặp và giai đoạn nằm trong `discovery.csv`, `replication.csv` và `stability.csv`.

## Diễn giải và giới hạn

Phép sàng lọc này kiểm tra liệu tỷ lệ đồng xuất hiện của một cặp có khác mô hình rút công bằng, độc lập và đồng đều hay không. Phép thử **không** tách riêng tương tác cặp khỏi sai lệch nhỏ trong tần suất biên của từng số: một tỷ lệ cặp bất thường có thể do tần suất của một hoặc cả hai số. Audit tần suất biên trước đó không tìm thấy tín hiệu sau hiệu chỉnh, nhưng điều đó không chứng minh phân phối hoàn toàn đồng đều.

Các p-value nhị thức chính xác giả định các ngày quay độc lập và 27 kết quả trong mỗi ngày độc lập, đồng đều. Hiệu chỉnh Holm xử lý sự phụ thuộc giữa 4.950 phép thử cặp số, không xử lý tự tương quan theo thời gian giữa các ngày. Ngày thiếu kết quả được bỏ qua. Ngay cả khi thấy bất thường thống kê, vẫn cần tái lập tiền cứu bằng dữ liệu mới và paper-trial với quy tắc đóng băng trước khi có thể dùng để dự đoán hoặc tuyên bố có lợi nhuận. Kết quả không có ý nghĩa thống kê không chứng minh tính ngẫu nhiên.

**Khuyến nghị:** {summary['recommendation']}

Không thay đổi hành vi dự đoán production hoặc mặc định Poisson.
"""


def run(source: Path, output: Path) -> dict[str, object]:
    """Run the fixed split analysis and write an isolated report bundle."""
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report directory: {output}")

    loader = XSMBDataLoader(url=source.as_uri())
    frequency, _ = loader.preprocess_and_build_matrices(loader.load_data())
    discovery_data, replication_data = split_periods(frequency)
    discovery_results = screen_pairs(discovery_data)
    replication_results = screen_pairs(replication_data)
    stability = compare_periods(discovery_results, replication_results)
    significant_discovery = discovery_results.p_value_holm.le(ALPHA)
    significant_replication = replication_results.p_value_holm.le(ALPHA)
    replicated = stability.replicated_significant

    recommendation = _recommendation(int(replicated.sum()))

    summary: dict[str, object] = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_loader_sha256": hashlib.sha256(
            Path(__file__).with_name("src").joinpath("data_loader.py").read_bytes()
        ).hexdigest(),
        "null_model": "27 independent uniform draws with replacement from 00..99 per observed date",
        "pairwise_null_probability": FAIR_PAIR_PROBABILITY,
        "pair_test_count_per_period": PAIR_COUNT,
        "test": "two-sided exact binomial",
        "multiplicity_correction": "Holm family-wise error correction, separately over all 4950 pairs in each period",
        "alpha": ALPHA,
        "discovery": _period_details(discovery_data),
        "replication": _period_details(replication_data),
        "discovery_significant_pairs": int(significant_discovery.sum()),
        "replication_significant_pairs": int(significant_replication.sum()),
        "replicated_same_direction_pairs": int(replicated.sum()),
        "replicated_pairs": [
            f"{row.number_a}-{row.number_b}"
            for row in stability.loc[replicated].itertuples(index=False)
        ],
        "production_default_changed": False,
        "recommendation": recommendation,
    }

    output.mkdir(parents=True, exist_ok=False)
    discovery_results.to_csv(output / "discovery.csv", index=False, encoding="utf-8-sig")
    replication_results.to_csv(output / "replication.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output / "stability.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "report.md").write_text(
        _report_text(summary, discovery_results, replication_results), encoding="utf-8"
    )
    print(f"REPORT {output / 'report.md'}", flush=True)
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Validated frequency or prize CSV.")
    parser.add_argument("--output", type=Path, required=True, help="New, not-yet-existing report directory.")
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
