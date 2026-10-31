"""Finite, reproducible chronological model research; never tune on final tests.

Run: python research.py
Outputs include a frozen protocol, development selection, held-out diagnostics,
cash flows for both payout rules, calibration tables, and a Vietnamese report.
Previously examined late-2026 dates are explicitly a retrospective audit only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import DEFAULT_XSMB_URL, XSMBDataLoader
from src.diagnostics import (
    block_bootstrap_mean_interval, evaluate_payouts, probability_diagnostics, rank_top_k,
)
from src.features import build_feature_dataset
from src.models import LGBMPredictor, TARGET


P0 = 1 - 0.99 ** 27
COUNT0 = 0.27
MODES = ("once_per_day", "per_occurrence")
WINDOWS = (7, 14, 30, 60, 90, 365)
ML_NAMES = ("logistic_3y", "lgbm_original", "lgbm_regularized_3y", "poisson_3y")
DEVELOPMENT = (
    ("dev_2024_h1", "2024-01-01", "2024-06-30"),
    ("dev_2024_h2", "2024-07-01", "2024-12-31"),
    ("dev_2025_h1", "2025-01-01", "2025-06-30"),
)


def simple_forecasts(frequency: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Causal presence probabilities and expected counts through each row.

    Rolling estimates have a fixed 30-observation uniform prior. Markov uses
    only adjacent calendar-day transitions already observed by each origin.
    Outputs at t predict t+1; none use an actual outcome after t.
    """
    binary = frequency.gt(0).astype(float)
    out = {}
    for window in WINDOWS:
        observations = pd.Series(1.0, index=frequency.index).rolling(f"{window}D").sum().to_numpy()[:, None]
        p = (binary.rolling(f"{window}D").sum().to_numpy() + 30 * P0) / (observations + 30)
        count = (frequency.rolling(f"{window}D").sum().to_numpy() + 30 * COUNT0) / (observations + 30)
        out[f"window_{window}"] = (p, count)
    n = np.arange(1, len(frequency) + 1)[:, None]
    out["historical"] = (
        (binary.cumsum().to_numpy() + 30 * P0) / (n + 30),
        (frequency.cumsum().to_numpy() + 30 * COUNT0) / (n + 30),
    )
    for span in [30, 90]:
        out[f"ewma_{span}"] = (
            binary.ewm(span=span, adjust=False).mean().to_numpy(),
            frequency.ewm(span=span, adjust=False).mean().to_numpy(),
        )
    b, c = binary.to_numpy(), frequency.to_numpy(dtype=float)
    adjacent = np.r_[False, np.diff(frequency.index.to_numpy(dtype="datetime64[D]")).astype(int) == 1]
    previous = np.vstack([np.zeros((1, 100)), b[:-1]])
    transitions = []
    for state in [0, 1]:
        observed = (previous == state) & adjacent[:, None]
        denominator = np.cumsum(observed, axis=0) + 100
        transitions.append((
            (np.cumsum(observed * b, axis=0) + 100 * P0) / denominator,
            (np.cumsum(observed * c, axis=0) + 100 * COUNT0) / denominator,
        ))
    out["markov_smoothed"] = (
        np.where(b > 0, transitions[1][0], transitions[0][0]),
        np.where(b > 0, transitions[1][1], transitions[0][1]),
    )
    return out


def normalized_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace growing cumulative quantities with stationary rates for new models."""
    x = frame.drop(columns=TARGET).astype(float).copy()
    x["co_occurrence_per_day"] = x.co_occurrence_total / x.history_days
    x = x.drop(columns=["co_occurrence_total", "history_days"])
    for window in [7, 14, 30, 60]:
        x[f"rolling_frequency_{window}d"] /= x[f"observed_days_{window}d"]
    return x


def fit_predict_ml(
    name: str, dataset: pd.DataFrame, target_dates: pd.DatetimeIndex,
    actual_counts: np.ndarray, first: pd.Timestamp, last: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit before a block; score its daily origins without refitting on the block."""
    train_days = target_dates < first
    if name != "lgbm_original":
        train_days &= target_dates >= first - pd.Timedelta(days=1095)
    test_days = (target_dates >= first) & (target_dates <= last)
    train = dataset.iloc[np.flatnonzero(np.repeat(train_days, 100))]
    test = dataset.iloc[np.flatnonzero(np.repeat(test_days, 100))]
    if train.empty or test.empty:
        raise ValueError("Empty chronological train or test block.")
    assert target_dates[train_days].max() < target_dates[test_days].min()
    y = train[TARGET].to_numpy(dtype=int)
    train_count = actual_counts[train_days]
    multiplicity = (train_count.sum(axis=0) + 100 * COUNT0 / P0) / ((train_count > 0).sum(axis=0) + 100)
    if name == "lgbm_original":
        model = LGBMPredictor().fit(train)
        p = model.predict_proba(test).to_numpy().reshape(-1, 100)
        count = p * multiplicity
    else:
        x, xt = normalized_features(train), normalized_features(test)
        params = dict(
            n_estimators=100, learning_rate=0.03, num_leaves=7, min_child_samples=200,
            reg_lambda=20.0, random_state=42, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        if name == "logistic_3y":
            numeric = [column for column in x if column != "number_id"]
            transform = ColumnTransformer([
                ("number", OneHotEncoder(handle_unknown="ignore"), ["number_id"]),
                ("numeric", StandardScaler(), numeric),
            ], sparse_threshold=1.0)
            model = make_pipeline(transform, LogisticRegression(C=1.0, max_iter=500, random_state=42))
            model.fit(x, y)
            p = model.predict_proba(xt)[:, 1].reshape(-1, 100)
            count = p * multiplicity
        elif name == "poisson_3y":
            model = LGBMRegressor(objective="poisson", **params)
            model.fit(x, train_count.ravel(), categorical_feature=["number_id"])
            count = np.maximum(model.predict(xt).reshape(-1, 100), 0)
            # A Poisson approximation; calibration is evaluated separately.
            p = -np.expm1(-count)
        else:
            model = LGBMClassifier(objective="binary", **params)
            model.fit(x, y, categorical_feature=["number_id"])
            p = model.predict_proba(xt)[:, 1].reshape(-1, 100)
            count = p * multiplicity
    return p, count, {
        "training_label_start": str(target_dates[train_days].min().date()),
        "training_label_end": str(target_dates[train_days].max().date()),
        "first_test_target": str(target_dates[test_days].min().date()),
        "last_test_target": str(target_dates[test_days].max().date()),
    }


def summarize(name, phase, dates, counts, p, expected, calibration=None):
    """Return metrics and day-level evidence for both explicitly defined payouts."""
    probability, reliability = probability_diagnostics(counts > 0, p)
    rows, daily = [], []
    for mode in MODES:
        selected = rank_top_k(expected if mode == "per_occurrence" else p, 5)
        result = evaluate_payouts(counts, selected, payout_mode=mode)
        frame = result.per_day.copy()
        frame.insert(0, "date", dates.strftime("%Y-%m-%d"))
        frame["model"], frame["phase"], frame["payout_mode"] = name, phase, mode
        frame["top5"] = [" ".join(f"{i:02d}" for i in np.flatnonzero(mask)) for mask in selected]
        frame["hit_numbers"] = [" ".join(f"{i:02d}" for i in np.flatnonzero(mask & (actual > 0))) for mask, actual in zip(selected, counts)]
        random_units = 5 * (counts.sum(axis=1) if mode == "per_occurrence" else (counts > 0).sum(axis=1)) / 100
        random_profit = random_units * 80_000 - 5 * 23_000
        lo, hi = block_bootstrap_mean_interval(frame.profit.to_numpy())
        delta_lo, delta_hi = block_bootstrap_mean_interval(frame.profit.to_numpy() - random_profit)
        row = {"model": name, "phase": phase, **result.summary, **probability,
               "profit_per_day_ci_low": lo, "profit_per_day_ci_high": hi,
               "random_expected_profit": float(random_profit.sum()),
               "advantage_per_day_ci_low": delta_lo, "advantage_per_day_ci_high": delta_hi}
        if calibration is not None:
            calibrated = apply_sigmoid(p, calibration)
            calibrated_metrics, _ = probability_diagnostics(counts > 0, calibrated)
            row.update({f"calibrated_{key}": value for key, value in calibrated_metrics.items()})
            row["top5_mean_calibrated_presence"] = float(calibrated[selected].mean())
        rows.append(row)
        daily.append(frame)
    reliability["model"], reliability["phase"] = name, phase
    return rows, daily, reliability


def apply_sigmoid(p: np.ndarray, configuration: dict) -> np.ndarray:
    """Apply only a calibration map chosen before the test block."""
    if not configuration.get("enabled", False):
        return p.copy()
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped))
    z = np.clip(configuration["slope"] * logits + configuration["intercept"], -40, 40)
    return 1 / (1 + np.exp(-z))


def run_research(output: Path, url: str = DEFAULT_XSMB_URL) -> None:
    """Execute one predeclared search; never iterate in response to holdout scores."""
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    loader = XSMBDataLoader(url=url)
    raw = loader.load_data()
    frequency, binary = loader.preprocess_and_build_matrices(raw)
    if frequency.index.max() < pd.Timestamp("2026-07-30"):
        raise ValueError("This frozen experiment requires data through 2026-07-30.")
    source_hash = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    names = [f"window_{w}" for w in WINDOWS] + ["historical", "ewma_30", "ewma_90", "markov_smoothed"] + list(ML_NAMES)
    protocol = {
        "version": 1, "source_url": url, "source_sha256": source_hash,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path(__file__).parent / "src/diagnostics.py",
                         Path(__file__).parent / "src/features.py", Path(__file__).parent / "src/models.py"]
        },
        "model_parameters": {
            "lgbm_original": LGBMPredictor().params,
            "new_lgbm": {"n_estimators": 100, "learning_rate": 0.03, "num_leaves": 7,
                         "min_child_samples": 200, "reg_lambda": 20, "random_state": 42},
            "logistic": {"C": 1, "max_iter": 500, "random_state": 42},
            "rolling_prior_observations": 30, "markov_prior_observations": 100,
            "multiplicity_prior_presence_events": 100,
        },
        "source_last_date": str(frequency.index.max().date()),
        "candidates": names, "top_k": 5, "stake": 23000, "gross_payout": 80000,
        "payout_modes": MODES, "development_blocks": DEVELOPMENT,
        "selection": "Maximum aggregate development payout units per selection, separately per payout rule; alphabetical tie break; freeze before later stages.",
        "calibration_fit": ["2025-07-01", "2025-09-30"],
        "calibration_check": ["2025-10-01", "2025-12-31"],
        "locked_test": ["2026-01-01", "2026-07-30"],
        "previously_examined_audit": ["2026-07-31", str(frequency.index.max().date())],
        "training": "Refit supervised models before each block; fixed within block; update causal features each day. Original model uses all history; other ML uses preceding 1095 calendar days.",
        "features": "Original unchanged; new ML drops cumulative history counters and normalizes rolling frequencies and co-occurrence by observed history.",
        "caution": "Chronological train/development/calibration/test, not fully nested CV. Historical dates were used as training in earlier exploratory work; this is a new locked scoring protocol, not genuinely unseen prospective data. No parameter changes based on locked-test scores.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Existing research output has a different protocol/source. Choose another output directory.")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    raw.to_csv(output / "source_snapshot.csv", index=False)
    print(f"Frozen protocol: {len(names)} candidates; {len(frequency)} observed days; output {output}", flush=True)
    dataset = build_feature_dataset(frequency, binary)
    origins = pd.DatetimeIndex(dataset.index.get_level_values(0).unique())
    targets = origins + pd.Timedelta(days=1)
    actual_counts = frequency.loc[targets].to_numpy(dtype=np.int64)
    simple = simple_forecasts(frequency)
    origin_positions = frequency.index.get_indexer(origins)
    all_rows, all_daily, all_reliability, audit_cutoffs = [], [], [], []

    def block(phase, start, end):
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        mask = (targets >= first) & (targets <= last)
        dates, counts = targets[mask], actual_counts[mask]
        if not len(dates):
            raise ValueError(f"Empty block: {phase}")
        predictions = {}
        for name in names:
            if name in simple:
                p, count = (array[origin_positions[mask]] for array in simple[name])
            else:
                path = cache / f"{phase}_{name}.npz"
                cutoff_path = path.with_suffix(".json")
                if path.exists() and cutoff_path.exists():
                    with np.load(path) as saved:
                        p, count = saved["p"], saved["count"]
                    cutoff = json.loads(cutoff_path.read_text())
                    print(f"{phase}: reuse {name}", flush=True)
                else:
                    print(f"{phase}: fitting {name} ({dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d})", flush=True)
                    p, count, cutoff = fit_predict_ml(name, dataset, targets, actual_counts, first, last)
                    np.savez_compressed(path, p=p, count=count)
                    cutoff_path.write_text(json.dumps(cutoff, indent=2))
                assert pd.Timestamp(cutoff["training_label_end"]) < dates.min()
                assert p.shape == counts.shape and count.shape == counts.shape
                audit_cutoffs.append({"phase": phase, "model": name, **cutoff})
            predictions[name] = (p, count)
        return dates, counts, predictions

    for phase, start, end in DEVELOPMENT:
        dates, counts, forecasts = block(phase, start, end)
        for name, (p, count) in forecasts.items():
            rows, daily, reliability = summarize(name, phase, dates, counts, p, count)
            all_rows.extend(rows); all_daily.extend(daily); all_reliability.append(reliability)

    development = pd.DataFrame(all_rows)
    selection = {}
    for mode in MODES:
        table = development.loc[development.payout_mode == mode].groupby("model", sort=True)[["payout_units", "selections"]].sum()
        table["units_per_selection"] = table.payout_units / table.selections
        selection[mode] = str(table.sort_values("units_per_selection", ascending=False, kind="stable").index[0])
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    development.to_csv(output / "development.csv", index=False)
    print("FROZEN SELECTION", selection, flush=True)

    dates, counts, forecasts = block("calibration", "2025-07-01", "2025-12-31")
    fit_mask = dates <= pd.Timestamp("2025-09-30")
    calibration = {}
    for name, (p, expected) in forecasts.items():
        fit_p = np.clip(p[fit_mask].ravel(), 1e-8, 1 - 1e-8)
        x = np.log(fit_p / (1 - fit_p)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, max_iter=500).fit(x, (counts[fit_mask] > 0).ravel().astype(int))
        settings = {"enabled": True, "slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])}
        raw_brier = probability_diagnostics(counts[~fit_mask] > 0, p[~fit_mask])[0]["brier"]
        calibrated_brier = probability_diagnostics(counts[~fit_mask] > 0, apply_sigmoid(p[~fit_mask], settings))[0]["brier"]
        settings.update(raw_check_brier=raw_brier, calibrated_check_brier=calibrated_brier)
        settings["enabled"] = settings["slope"] > 0 and calibrated_brier < raw_brier
        calibration[name] = settings
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2))

    for phase, start, end in [
        ("locked_test", "2026-01-01", "2026-07-30"),
        ("recent_audit", "2026-07-31", str(frequency.index.max().date())),
    ]:
        dates, counts, forecasts = block(phase, start, end)
        for name, (p, expected) in forecasts.items():
            rows, daily, reliability = summarize(name, phase, dates, counts, p, expected, calibration[name])
            all_rows.extend(rows); all_daily.extend(daily); all_reliability.append(reliability)
            calibrated = apply_sigmoid(p, calibration[name])
            _, calibrated_table = probability_diagnostics(counts > 0, calibrated)
            calibrated_table["model"], calibrated_table["phase"] = name + "_calibrated", phase
            all_reliability.append(calibrated_table)

    summary = pd.DataFrame(all_rows)
    daily = pd.concat(all_daily, ignore_index=True)
    summary.to_csv(output / "metrics.csv", index=False)
    daily.to_csv(output / "daily_results.csv", index=False)
    pd.concat(all_reliability, ignore_index=True).to_csv(output / "calibration_bins.csv", index=False)
    pd.DataFrame(audit_cutoffs).to_csv(output / "training_cutoffs.csv", index=False)
    # A fixed probability/expected-count reference is evaluated independently
    # of Top-K rankings, with no manufactured ranking among tied numbers.
    mask = (targets >= pd.Timestamp("2026-01-01")) & (targets <= pd.Timestamp("2026-07-30"))
    uniform_metrics, _ = probability_diagnostics(actual_counts[mask] > 0, np.full(actual_counts[mask].shape, P0))
    decisions = {}
    for mode, name in selection.items():
        row = summary.loc[(summary.phase == "locked_test") & (summary.model == name) & (summary.payout_mode == mode)].iloc[0]
        decisions[mode] = {
            "selected_model": name,
            "supported_positive_return": bool(row.profit_per_day_ci_low > 0 and row.advantage_per_day_ci_low > 0),
            "holdout_profit": float(row.profit), "holdout_roi": float(row.roi),
            "profit_per_day_95pct_interval": [float(row.profit_per_day_ci_low), float(row.profit_per_day_ci_high)],
        }
    (output / "decision.json").write_text(json.dumps({"decisions": decisions, "uniform_probability_metrics": uniform_metrics}, indent=2))
    write_report(output, protocol, selection, summary, calibration, decisions, uniform_metrics)
    print("DECISIONS", json.dumps(decisions), flush=True)
    print("UNIFORM PROBABILITY METRICS", uniform_metrics, flush=True)
    print("REPORT", output / "report.md", flush=True)


def write_report(output, protocol, selection, summary, calibration, decisions, uniform):
    """Write a complete result without relabeling test winners as selected models."""
    lines = [
        '# Kiểm tra cách tính và nghiên cứu mô hình Top 5', '',
        '## Kết luận theo cấu hình đã chọn trước khi xem tập kiểm tra', '',
        '| Trả thưởng | Cấu hình chọn trên phát triển | Lãi/lỗ kiểm tra | ROI | Khoảng 95% lãi/ngày | Có bằng chứng kỳ vọng dương? |',
        '|---|---|---:|---:|---|---|',
    ]
    for mode, decision in decisions.items():
        lo, hi = decision['profit_per_day_95pct_interval']
        lines.append(f"| {mode} | {decision['selected_model']} | {decision['holdout_profit']:,.0f}đ | {decision['holdout_roi']:.2%} | [{lo:,.0f}; {hi:,.0f}]đ | {'Có trong kiểm thử này, vẫn cần xác nhận tương lai' if decision['supported_positive_return'] else 'Chưa có'} |")
    lines += [
        '', '## Cách tính', '',
        '- Mỗi ngày chọn 5 số, chi 5 × 23.000 = 115.000đ. 80.000đ là tổng nhận, không cộng hoàn vốn.',
        '- once_per_day: số trúng khác nhau × 80.000 − tiền mua.',
        '- per_occurrence: tổng số lần xuất hiện của các số đã chọn × 80.000 − tiền mua.',
        '- Hòa vốn: 0,2875 đơn vị trả thưởng/lượt chọn. Với per_occurrence đây là kỳ vọng số lần, không phải xác suất xuất hiện ít nhất một lần.',
        '- Chọn ngẫu nhiên 5 số có kỳ vọng 1,35 lần xuất hiện/ngày (27 × 5/100), thu 108.000đ, lỗ kỳ vọng 7.000đ/ngày nếu trả mỗi lần. Không cần giả định 27 kết quả độc lập để tính kỳ vọng này khi chọn 5 số đều nhau.',
        '', '## Thiết kế nghiên cứu', '',
        '- 14 cấu hình cố định: 6 cửa sổ tần suất, lịch sử tích lũy, 2 EWMA, Markov làm trơn, Logistic Regression, LightGBM cũ, LightGBM điều chuẩn, Poisson.',
        '- Phát triển/chọn: 2024-01-01 đến 2025-06-30, chia 3 khối nửa năm. Chọn theo tổng đơn vị trả thưởng/lượt chọn trên phát triển; không dùng tập kiểm tra để chọn.',
        '- Hiệu chỉnh xác suất: khớp sigmoid ở quý 3/2025; chỉ giữ nếu Brier tốt hơn và độ dốc dương trên quý 4/2025.',
        '- Kiểm tra khóa: 2026-01-01 đến 2026-07-30. Giai đoạn 2026-07-31 trở đi là kiểm tra hồi cứu đã từng xem trước đó, không phải tập độc lập mới.',
        '- Mỗi mô hình ML huấn luyện trước từng khối, không huấn luyện lại bằng nhãn trong khối. Đặc trưng và baseline cập nhật sau mỗi kết quả đã biết. Đây là cách đánh giá theo khối, khác lượt chạy cũ huấn luyện ML lại hàng ngày.',
        '- Đây là train/development/calibration/test theo thời gian, không tuyên bố nested CV đầy đủ. Dữ liệu lịch sử từng được dùng huấn luyện trong các lượt trước nên vẫn cần kiểm chứng dự báo ghi trước kết quả thực tế.',
        '- Khoảng bất định dùng bootstrap theo khối 7 ngày, 2.000 mẫu, giữ nguyên cụm 100 số/ngày. Khoảng chỉ có tính xấp xỉ dưới giả định ổn định cục bộ; không bảo đảm cho tương lai.',
        '', '## Tập kiểm tra khóa: mọi cấu hình (chỉ mang tính chẩn đoán, không chọn lại theo bảng này)', '',
        '| Mô hình | Trả thưởng | Số ngày | Precision@5 | Lần xuất hiện/lượt chọn | Lãi/lỗ | ROI | Brier thô | Brier sau hiệu chỉnh |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, row in summary.loc[summary.phase == 'locked_test'].iterrows():
        lines.append(f"| {row.model} | {row.payout_mode} | {row.days:.0f} | {row.precision_at_k:.2%} | {row.occurrences_per_selection:.4f} | {row.profit:,.0f}đ | {row.roi:.2%} | {row.brier:.6f} | {row.calibrated_brier:.6f} |")
    lines += ['', f"Mốc xác suất đều nhau {P0:.4%}: Brier={uniform['brier']:.6f}, log loss={uniform['log_loss']:.6f}. Xác suất không phải điểm xếp hạng; sigmoid đơn điệu chủ yếu sửa độ tin cậy, không tạo thêm thông tin để chọn số.", '', '## Kiểm tra hồi cứu giai đoạn gần đây: cấu hình đã chọn', '', '| Trả thưởng | Mô hình | Số ngày | Lãi/lỗ | ROI |', '|---|---|---:|---:|---:|']
    for mode, name in selection.items():
        row = summary.loc[(summary.phase == 'recent_audit') & (summary.model == name) & (summary.payout_mode == mode)].iloc[0]
        lines.append(f"| {mode} | {name} | {row.days:.0f} | {row.profit:,.0f}đ | {row.roi:.2%} |")
    lines += [
        '', '## Tệp để kiểm tra và tái lập', '',
        '- [Giao thức khóa trước khi chạy](./protocol.json), [cấu hình chọn](./selection.json), [quyết định](./decision.json).',
        '- [Kết quả phát triển](./development.csv), [toàn bộ chỉ số](./metrics.csv), [từng ngày và dòng tiền](./daily_results.csv).',
        '- [Bảng hiệu chỉnh xác suất](./calibration_bins.csv), [ngày cuối nhãn huấn luyện](./training_cutoffs.csv), [quy tắc sigmoid](./calibration.json).',
        '- [Ảnh chụp dữ liệu nguồn](./source_snapshot.csv).',
        '', '## Tài liệu phương pháp', '',
        '- [scikit-learn: lựa chọn mô hình và sai lệch khi tái dùng dữ liệu kiểm tra](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).',
        '- [scikit-learn: hiệu chỉnh xác suất và reliability diagrams](https://scikit-learn.org/stable/modules/calibration.html).',
        '- [LightGBM: mục tiêu binary và Poisson](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective).',
        '', 'Không tự động thay mô hình sản xuất chỉ vì một cấu hình có ROI tốt nhất sau khi xem tập kiểm tra. Khi chưa có bằng chứng ổn định, kết luận đúng là chưa chứng minh được lợi thế, không phải tiếp tục tìm tham số cho đến khi báo cáo có lãi.', '',
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/research_2026-10-29'))
    parser.add_argument('--url', default=DEFAULT_XSMB_URL)
    args = parser.parse_args()
    run_research(args.output, args.url)
