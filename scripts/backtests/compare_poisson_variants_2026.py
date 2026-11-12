"""Paired 2026 walk-forward comparison of Poisson model families and windows."""

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
from src.prize_features import PRIZE_COLUMNS, build_prize_feature_dataset, normalize_prizes
from src.prize_models import PrizeLGBMPredictor


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "processed" / "xsmb_prizes_2026-11-06.csv"
DEFAULT_OUTPUT = (
    ROOT / "reports" / "research" / "poisson_variants_2026_to_2026-11-05"
)
DEFAULT_PRIZE_90D_CACHE = (
    ROOT / "reports" / "research"
    / "all_models_2026_90d_daily_refit_to_2026-11-05" / "daily.csv"
)
DEFAULT_COUNT_ALL_HISTORY_CACHE = (
    ROOT / "reports" / "research"
    / "poisson_daily_refit_comparison_2026" / "daily.csv"
)
START_DATE = pd.Timestamp("2026-01-01")
END_DATE = pd.Timestamp("2026-11-05")
TOP_K = 3
STAKE_PER_SELECTION_VND = 27_000
GROSS_PER_OCCURRENCE_VND = 99_000
WINDOWS = (30, 60, 90, 120)
MODEL_NAMES = (
    *(f"count_{window}d" for window in WINDOWS),
    "count_all_history",
    *(f"prize_{window}d" for window in WINDOWS),
    "prize_1095d",
)
MODEL_LABELS = {
    **{f"count_{window}d": f"Poisson Count {window}d" for window in WINDOWS},
    "count_all_history": "Poisson Count toàn lịch sử",
    **{f"prize_{window}d": f"Poisson Prize {window}d" for window in WINDOWS},
    "prize_1095d": "Poisson Prize 1095d (mặc định)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prize-90d-cache",
        type=Path,
        default=DEFAULT_PRIZE_90D_CACHE,
        help="Validated daily Top-3 Prize Poisson 90d predictions.",
    )
    parser.add_argument(
        "--count-all-history-cache",
        type=Path,
        default=DEFAULT_COUNT_ALL_HISTORY_CACHE,
        help="Daily Count Poisson expanding-history predictions.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Refit all Poisson variants instead of reusing validated historical runs.",
    )
    return parser.parse_args()


def _read_cache(path: Path | None, model: str) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Prediction cache does not exist: {resolved}")
    frame = pd.read_csv(resolved, dtype="string", keep_default_na=False)
    if not {"date", "model", "prediction"}.issubset(frame.columns):
        raise ValueError(f"Prediction cache is missing required columns: {resolved}")
    frame = frame.loc[frame.model.eq(model)]
    if frame.date.duplicated().any():
        raise ValueError(f"Duplicate cached dates for {model}: {resolved}")
    return {str(row["date"]): row.to_dict() for _, row in frame.iterrows()}


def _read_cache_protocol(path: Path, *, model: str) -> dict[str, object]:
    protocol_path = path.resolve().with_name("protocol.json")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Cache protocol does not exist: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("top_k") != TOP_K:
        raise ValueError(f"Cache is not a Top-{TOP_K} run: {protocol_path}")
    if protocol.get("evaluation_start", protocol.get("start")) != START_DATE.strftime("%Y-%m-%d"):
        raise ValueError(f"Cache evaluation start does not match: {protocol_path}")
    models = protocol.get("models")
    if isinstance(models, dict):
        if model not in models:
            raise ValueError(f"Cache protocol does not contain {model}: {protocol_path}")
    elif not isinstance(models, list) or model not in models:
        raise ValueError(f"Cache protocol does not contain {model}: {protocol_path}")
    return protocol


def _parse_picks(value: str, *, date: str, model: str) -> np.ndarray:
    pieces = value.split()
    if (
        len(pieces) != TOP_K
        or len(set(pieces)) != TOP_K
        or any(len(piece) != 2 or not piece.isdigit() for piece in pieces)
    ):
        raise ValueError(f"Invalid cached Top-{TOP_K} prediction for {model} on {date}.")
    return np.asarray([int(piece) for piece in pieces], dtype=np.int64)


def _record(
    date: pd.Timestamp,
    model: str,
    picks: np.ndarray,
    actual: np.ndarray,
    *,
    training_labels: int | None,
    training_label_start: str | None,
    training_label_end: str | None,
) -> dict[str, object]:
    if picks.shape != (TOP_K,) or len(set(picks.tolist())) != TOP_K:
        raise ValueError(f"{model} did not return {TOP_K} distinct predictions.")
    counts = actual[picks]
    distinct_hits = int(np.count_nonzero(counts))
    occurrences = int(counts.sum())
    cost = TOP_K * STAKE_PER_SELECTION_VND
    receipts = occurrences * GROSS_PER_OCCURRENCE_VND
    return {
        "date": date.strftime("%Y-%m-%d"),
        "model": model,
        "prediction": " ".join(f"{number:02d}" for number in picks),
        "hit_numbers_with_counts": " ".join(
            f"{number:02d}x{int(actual[number])}"
            for number in picks if actual[number] > 0
        ),
        "distinct_hits": distinct_hits,
        "hit": distinct_hits > 0,
        "occurrences": occurrences,
        "cost_vnd": cost,
        "receipts_vnd": receipts,
        "profit_vnd": receipts - cost,
        "training_labels": training_labels,
        "training_label_start": training_label_start,
        "training_label_end": training_label_end,
    }


def _training_mask(
    dataset: pd.DataFrame,
    target: pd.Timestamp,
    window_days: int | None,
) -> np.ndarray:
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    label_dates = origins + pd.Timedelta(days=1)
    mask = dataset[COUNT_TARGET].notna().to_numpy() & (label_dates < target)
    if window_days is not None:
        mask &= label_dates >= target - pd.Timedelta(days=window_days)
    return mask


def _fit_and_rank(
    model_name: str,
    dataset: pd.DataFrame,
    target: pd.Timestamp,
    actual: np.ndarray,
    *,
    family: str,
    window_days: int | None,
) -> dict[str, object]:
    training = dataset.loc[_training_mask(dataset, target, window_days)]
    labeled_dates = pd.DatetimeIndex(training.index.get_level_values(0).unique()) + pd.Timedelta(days=1)
    if training.empty or labeled_dates.max() >= target:
        raise ValueError(f"Invalid or empty training labels for {model_name} on {target:%Y-%m-%d}.")
    origin = target - pd.Timedelta(days=1)
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0))
    inference = dataset.loc[origins == origin].drop(columns=COUNT_TARGET)
    if len(inference) != len(NUMBER_COLUMNS):
        raise ValueError(f"Incomplete inference rows for {model_name} on {target:%Y-%m-%d}.")

    if family == "count":
        fitted = CountLGBMPredictor().fit(training)
        scores = fitted.predict_expected_counts(inference).droplevel(0).reindex(NUMBER_COLUMNS)
    elif family == "prize":
        if window_days is None:
            raise ValueError("Prize Poisson variants require a finite training window.")
        fitted = PrizeLGBMPredictor(
            objective="poisson", history_days=window_days
        ).fit(training)
        scores = fitted.predict_scores(inference).droplevel(0).reindex(NUMBER_COLUMNS)
    else:
        raise ValueError(f"Unknown Poisson family: {family}")

    values = scores.to_numpy(dtype=float)
    if values.shape != (len(NUMBER_COLUMNS),) or not np.isfinite(values).all():
        raise ValueError(f"Invalid Poisson scores for {model_name} on {target:%Y-%m-%d}.")
    picks = np.argsort(-values, kind="stable")[:TOP_K]
    return _record(
        target,
        model_name,
        picks,
        actual,
        training_labels=len(labeled_dates),
        training_label_start=labeled_dates.min().strftime("%Y-%m-%d"),
        training_label_end=labeled_dates.max().strftime("%Y-%m-%d"),
    )


def _validate_cached_row(
    row: dict[str, str],
    *,
    date: pd.Timestamp,
    model: str,
    actual: np.ndarray,
    require_training_end: bool,
) -> dict[str, object]:
    date_string = date.strftime("%Y-%m-%d")
    if row["date"] != date_string:
        raise ValueError(f"Cached date mismatch for {model} on {date_string}.")
    picks = _parse_picks(row["prediction"], date=date_string, model=model)
    if require_training_end:
        training_end = row.get("training_label_end")
        if not training_end or pd.Timestamp(training_end) >= date:
            raise ValueError(
                f"Cached {model} training cutoff is not strictly before {date_string}."
            )
    recorded_occurrences = int(row["occurrences"])
    recorded_hits = int(row["distinct_hits"]) if "distinct_hits" in row else None
    expected = _record(
        date,
        model,
        picks,
        actual,
        training_labels=None,
        training_label_start=None,
        training_label_end=None,
    )
    if (
        recorded_occurrences != expected["occurrences"]
        or (recorded_hits is not None and recorded_hits != expected["distinct_hits"])
        or ("hit" in row and row["hit"].lower() != str(expected["hit"]).lower())
    ):
        raise ValueError(f"Cached result does not match the current outcome for {model} on {date_string}.")
    return expected


def _daily_targets(
    frequency: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    candidates = pd.date_range(START_DATE, END_DATE, freq="D")
    available = set(frequency.index)
    targets = pd.DatetimeIndex([
        date for date in candidates
        if date in available and date - pd.Timedelta(days=1) in available
    ])
    target_set = set(targets)
    excluded = pd.DataFrame([
        {
            "date": date.strftime("%Y-%m-%d"),
            "reason": (
                "actual outcome missing" if date not in available
                else "preceding calendar day missing"
            ),
        }
        for date in candidates if date not in target_set
    ])
    if targets.empty:
        raise ValueError("No eligible target dates in the configured period.")
    return targets, excluded


def run(
    source: Path,
    *,
    prize_90d_cache: Path | None,
    count_all_history_cache: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")
    source_bytes = source.read_bytes()
    raw = pd.read_csv(source, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    prizes = normalize_prizes(raw)
    loader = XSMBDataLoader(prize_columns=PRIZE_COLUMNS)
    frequency, binary = loader.preprocess_and_build_matrices(prizes.reset_index())
    targets, excluded = _daily_targets(frequency)
    count_dataset = build_count_feature_dataset(frequency, binary, include_unlabeled=True)
    prize_dataset = build_prize_feature_dataset(
        raw,
        include_unlabeled=True,
        start=START_DATE - pd.Timedelta(days=1096),
    )

    prize_cache = _read_cache(prize_90d_cache, "poisson") if prize_90d_cache else {}
    count_cache = (
        _read_cache(count_all_history_cache, "poisson_all_history_daily")
        if count_all_history_cache else {}
    )
    cache_protocols: dict[str, object] = {}
    if prize_cache and prize_90d_cache is not None:
        prize_protocol = _read_cache_protocol(prize_90d_cache, model="poisson")
        if (
            prize_protocol.get("history_days") != 90
            or prize_protocol.get("end") != END_DATE.strftime("%Y-%m-%d")
        ):
            raise ValueError("Prize Poisson cache must be a daily 90-day run through 2026-11-05.")
        expected_source_hash = prize_protocol.get("source_snapshot_sha256_file")
        if expected_source_hash and expected_source_hash != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Prize Poisson cache source hash differs from the current snapshot.")
        cache_protocols["prize_90d"] = prize_protocol
    if count_cache and count_all_history_cache is not None:
        count_protocol = _read_cache_protocol(
            count_all_history_cache, model="poisson_all_history_daily"
        )
        if count_protocol.get("refit_policy") != "Before each target day for both models":
            raise ValueError("Count Poisson cache was not refit daily.")
        cache_protocols["count_all_history"] = count_protocol
    date_strings = {date.strftime("%Y-%m-%d") for date in targets}
    if prize_cache and set(prize_cache) != date_strings:
        raise ValueError("Cached Prize Poisson 90d predictions must cover every evaluation date.")
    if count_cache and not set(count_cache).issubset(date_strings):
        raise ValueError("Cached Count Poisson expanding predictions contain out-of-scope dates.")

    results: list[dict[str, object]] = []
    model_datasets = {
        **{f"count_{window}d": ("count", count_dataset, window) for window in WINDOWS},
        "count_all_history": ("count", count_dataset, None),
        **{f"prize_{window}d": ("prize", prize_dataset, window) for window in WINDOWS},
        "prize_1095d": ("prize", prize_dataset, 1095),
    }
    for position, target in enumerate(targets, start=1):
        actual = frequency.loc[target, list(NUMBER_COLUMNS)].to_numpy(dtype=np.int64)
        date_string = target.strftime("%Y-%m-%d")
        for model_name in MODEL_NAMES:
            if model_name == "prize_90d" and prize_cache:
                row = _validate_cached_row(
                    prize_cache[date_string],
                    date=target,
                    model=model_name,
                    actual=actual,
                    require_training_end=False,
                )
            elif model_name == "count_all_history" and date_string in count_cache:
                row = _validate_cached_row(
                    count_cache[date_string],
                    date=target,
                    model=model_name,
                    actual=actual,
                    require_training_end=True,
                )
            else:
                family, dataset, window = model_datasets[model_name]
                row = _fit_and_rank(
                    model_name,
                    dataset,
                    target,
                    actual,
                    family=family,
                    window_days=window,
                )
            results.append(row)
        if position % 10 == 0 or position == len(targets):
            print(f"Completed {position}/{len(targets)} target dates.", flush=True)

    daily = pd.DataFrame(results)
    validate_results(daily, excluded)
    protocol: dict[str, object] = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "evaluation_start": START_DATE.strftime("%Y-%m-%d"),
        "evaluation_end": END_DATE.strftime("%Y-%m-%d"),
        "paired_days": int(len(targets)),
        "excluded_dates": excluded.to_dict(orient="records"),
        "models": {
            name: {
                "family": "Count Poisson" if name.startswith("count_") else "Prize-feature Poisson",
                "training_window_days": (
                    None if name == "count_all_history"
                    else int(name.rsplit("_", 1)[1][:-1])
                    if name.rsplit("_", 1)[1].endswith("d")
                    else None
                ),
                "refit": "Daily, with labels strictly before target",
            }
            for name in MODEL_NAMES
        },
        "top_k": TOP_K,
        "stake_per_selection_vnd": STAKE_PER_SELECTION_VND,
        "gross_per_occurrence_vnd": GROSS_PER_OCCURRENCE_VND,
        "payout_mode": "Per occurrence including repeats; no separate stake refund",
        "cache_validation": {
            "prize_90d_source": str(prize_90d_cache.resolve()) if prize_cache else None,
            "prize_90d_prediction_dates_revalidated": len(prize_cache),
            "count_all_history_source": (
                str(count_all_history_cache.resolve()) if count_cache else None
            ),
            "count_all_history_prediction_dates_revalidated": len(count_cache),
            "all_cached_outcomes_revalidated_against_current_snapshot": True,
            "count_all_history_cached_training_cutoffs_checked": True,
            "cache_protocols": cache_protocols,
        },
        "notes": (
            "Retrospective paired comparison. Count and Prize feature families are distinct; "
            "all variants use the same target dates, Top-3, payout, and daily refit policy."
        ),
    }
    return daily, excluded, protocol


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        group = daily.loc[daily.model.eq(model)]
        days = len(group)
        cost = int(group.cost_vnd.sum())
        receipts = int(group.receipts_vnd.sum())
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "days": days,
            "hit_days": int(group.hit.sum()),
            "hit_rate": float(group.hit.mean()),
            "occurrences": int(group.occurrences.sum()),
            "cost_vnd": cost,
            "receipts_vnd": receipts,
            "profit_vnd": receipts - cost,
            "roi": (receipts - cost) / cost if cost else None,
        })
    return pd.DataFrame(rows)


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame.date.str[:7]
    rows = []
    for month, month_group in frame.groupby("month", sort=True):
        for model in MODEL_NAMES:
            group = month_group.loc[month_group.model.eq(model)]
            cost = int(group.cost_vnd.sum())
            receipts = int(group.receipts_vnd.sum())
            rows.append({
                "month": month,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "days": len(group),
                "hit_days": int(group.hit.sum()),
                "occurrences": int(group.occurrences.sum()),
                "cost_vnd": cost,
                "receipts_vnd": receipts,
                "profit_vnd": receipts - cost,
                "roi": (receipts - cost) / cost if cost else None,
            })
    return pd.DataFrame(rows)


def validate_results(daily: pd.DataFrame, excluded: pd.DataFrame) -> None:
    expected_days = 309 - len(excluded)
    expected_dates = set(daily.loc[daily.model.eq(MODEL_NAMES[0]), "date"])
    if len(expected_dates) != expected_days:
        raise ValueError(f"Expected {expected_days} shared target dates, got {len(expected_dates)}.")
    for model in MODEL_NAMES:
        rows = daily.loc[daily.model.eq(model)]
        if len(rows) != expected_days or set(rows.date) != expected_dates:
            raise ValueError(f"Unpaired or incomplete daily results for {model}.")
        if rows.date.duplicated().any():
            raise ValueError(f"Duplicate target dates for {model}.")
        if (rows.cost_vnd != TOP_K * STAKE_PER_SELECTION_VND).any():
            raise ValueError(f"Invalid daily stake for {model}.")
        if (rows.receipts_vnd != rows.occurrences * GROSS_PER_OCCURRENCE_VND).any():
            raise ValueError(f"Invalid daily payout for {model}.")
        if (rows.profit_vnd != rows.receipts_vnd - rows.cost_vnd).any():
            raise ValueError(f"Invalid daily profit for {model}.")

    monthly = aggregate_monthly(daily)
    summary = summarize(daily)
    for row in summary.itertuples():
        months = monthly.loc[monthly.model.eq(row.model)]
        if (
            int(months.days.sum()) != row.days
            or int(months.cost_vnd.sum()) != row.cost_vnd
            or int(months.receipts_vnd.sum()) != row.receipts_vnd
            or int(months.profit_vnd.sum()) != row.profit_vnd
            or int(months.hit_days.sum()) != row.hit_days
            or int(months.occurrences.sum()) != row.occurrences
        ):
            raise ValueError(f"Monthly totals do not reconcile for {row.model}.")


def _validate_markdown_tables(markdown: str) -> None:
    table: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|") and line.endswith("|"):
            table.append(line)
        elif table:
            widths = [len(row.strip("|").split("|")) for row in table]
            if len(set(widths)) != 1:
                raise ValueError(f"Inconsistent Markdown table columns: {widths}")
            table = []
    if table:
        widths = [len(row.strip("|").split("|")) for row in table]
        if len(set(widths)) != 1:
            raise ValueError(f"Inconsistent Markdown table columns: {widths}")


def write_report(
    output: Path,
    daily: pd.DataFrame,
    excluded: pd.DataFrame,
    protocol: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(daily)
    monthly = aggregate_monthly(daily)
    pivot = monthly.pivot(index="month", columns="model", values="profit_vnd").reindex(
        columns=MODEL_NAMES
    )
    daily.to_csv(output / "daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(output / "monthly_pivot.csv", encoding="utf-8-sig")
    excluded.to_csv(output / "excluded_dates.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# So sánh các biến thể Poisson — 2026",
        "",
        f"**Kỳ đánh giá:** {START_DATE:%d/%m/%Y}–{END_DATE:%d/%m/%Y}; "
        f"{len(daily) // len(MODEL_NAMES)} ngày ghép cặp.",
        "",
        "So sánh hai họ Poisson: **Count** dùng feature tần suất/số lần xuất hiện; "
        "**Prize** dùng feature từ 27 vị trí giải. Mỗi họ thử các cửa sổ huấn luyện "
        "30/60/90/120 ngày; Count còn có biến thể toàn lịch sử, Prize có cửa sổ "
        "1095 ngày (mặc định). Tất cả refit trước từng ngày dự báo, chỉ dùng nhãn "
        "đã có trước ngày đó.",
        "",
        "**Cùng điều kiện:** Top-3, cùng 304 ngày dự báo hợp lệ, 27.000đ/số/ngày "
        "(81.000đ/ngày/model); thu gộp 99.000đ cho mỗi lần xuất hiện trúng, tính "
        "cả số lặp. Tháng 2 có 23/28 ngày do thiếu dữ liệu 16–20/02; tháng 11 chỉ "
        "đến hết 05/11.",
        "",
        "## Tổng kết",
        "",
        "| Model | Ngày hit | Tỷ lệ hit | Lượt trúng | Tổng chi | Tổng thu | Lãi/lỗ | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {row.hit_days}/{row.days} | {row.hit_rate:.2%} | "
            f"{row.occurrences} | {row.cost_vnd:,}đ | {row.receipts_vnd:,}đ | "
            f"{row.profit_vnd:+,}đ | {row.roi:.2%} |"
        )
    lines.extend([
        "",
        "## Lãi/lỗ từng tháng",
        "",
        "Mỗi ô ghi **lãi/lỗ (ROI)**. Dấu `*` đánh dấu tháng không đủ ngày trong phạm vi.",
        "",
        "| Tháng | " + " | ".join(MODEL_LABELS[name] for name in MODEL_NAMES) + " |",
        "|" + "|".join(["---:"] * (len(MODEL_NAMES) + 1)) + "|",
    ])
    for month in pivot.index:
        evaluated_days = int(monthly.loc[monthly.month.eq(month), "days"].iloc[0])
        mark = "*" if evaluated_days != pd.Period(month).days_in_month else ""
        cells = []
        for model in MODEL_NAMES:
            record = monthly.loc[
                monthly.month.eq(month) & monthly.model.eq(model)
            ].iloc[0]
            cells.append(f"{int(record.profit_vnd):+,}đ ({record.roi:.1%})")
        lines.append(f"| {month}{mark} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "### Cách đọc các biến thể",
        "",
        "- `Poisson Count`: target là số lần mỗi số xuất hiện; `toàn lịch sử` là "
        "biến thể gần với `main.py --model poisson` khi đánh giá walk-forward.",
        "- `Poisson Prize`: cùng objective Poisson nhưng feature mô tả vị trí/thành "
        "phần giải; khác hẳn họ Count, không gộp hai họ thành một model.",
        "- `1095d` là cửa sổ 3 năm, không phải toàn bộ lịch sử.",
        "- Đây là backtest hồi cứu; lãi/lỗ không chứng minh lợi thế dự đoán trong tương lai.",
        "",
        "## Nguồn và dữ liệu chi tiết",
        "",
        f"Nguồn snapshot: `{protocol['source']}`; SHA-256 "
        f"`{protocol['source_sha256']}`. Bản 90d Prize và Count toàn lịch sử được "
        "tái sử dụng từ các cache trước đó sau khi đối chiếu dự đoán/điểm cắt huấn "
        "luyện và tính lại kết quả theo snapshot hiện tại. Các biến thể còn lại "
        "được fit lại theo từng ngày.",
        "",
        "[Dữ liệu từng ngày](./daily.csv) · [Theo tháng](./monthly.csv) · "
        "[Bảng lãi/lỗ theo tháng](./monthly_pivot.csv) · [Tổng hợp](./summary.csv) · "
        "[Ngày loại](./excluded_dates.csv) · [Giao thức](./protocol.json).",
        "",
    ])
    report = "\n".join(lines)
    _validate_markdown_tables(report)
    (output / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prize_cache = None if args.no_cache else args.prize_90d_cache
    count_cache = None if args.no_cache else args.count_all_history_cache
    daily, excluded, protocol = run(
        args.source,
        prize_90d_cache=prize_cache,
        count_all_history_cache=count_cache,
    )
    write_report(args.output.resolve(), daily, excluded, protocol)
    print(summarize(daily).drop(columns="model_label").to_string(index=False), flush=True)
    print(f"\nReport written to {args.output.resolve() / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
