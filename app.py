"""Interactive forecasting workspace for the XSMB time-series project.

Run with::

    streamlit run app.py

The web UI calls the same ``run_pipeline`` entry point used by the command-line
workflow. It adds a clear "today" eligibility check, model comparison, run
history, and exportable results without maintaining a second modelling path.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import date as calendar_date
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from main import run_pipeline
from src.count_features import COUNT_TARGET
from src.data_loader import DEFAULT_XSMB_URL, XSMBDataLoader
from src.prize_features import RAW_PRIZE_URL, build_prize_feature_dataset, normalize_prizes
from src.prize_models import PrizeLGBMPredictor


ROOT = Path(__file__).resolve().parent
REPORT_SUMMARY = ROOT / "reports" / "current" / "model_comparison_90d_2026" / "summary.csv"
REPORT_DAILY = ROOT / "reports" / "current" / "model_comparison_90d_2026" / "daily.csv"
DATA_DIRS = (ROOT / "data" / "processed", ROOT / "data" / "raw")
MODEL_LABELS = {
    "prize_poisson_90d": "Poisson Prize 90d",
    "baseline": "30-day frequency",
    "lightgbm": "LightGBM presence",
    "regularized": "Regularized presence",
    "poisson": "Poisson count",
}
MODEL_DESCRIPTIONS = {
    "prize_poisson_90d": "Primary model: ordered-prize features with a rolling 90-day Poisson window.",
    "baseline": "Ranks by trailing 30-day occurrence count.",
    "lightgbm": "Ranks by an uncalibrated next-day presence estimate.",
    "regularized": "A regularized presence model for comparison research.",
    "poisson": "Ranks by repeat-inclusive expected occurrence count.",
}


def _source_files() -> list[Path]:
    files: list[Path] = []
    for folder in DATA_DIRS:
        if folder.exists():
            files.extend(folder.glob("*.csv"))
    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)


def _source_options() -> dict[str, tuple[str, str]]:
    """Return stable source keys mapped to (display label, URL/path)."""
    options = {"live": ("Live raw prize feed", RAW_PRIZE_URL)}
    for path in _source_files():
        options[str(path)] = (f"{path.parent.name}/{path.name}", str(path))
    return options


def _source_url(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return source
    return Path(source).resolve().as_uri()


@st.cache_data(show_spinner=False)
def load_matrices(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a source through the production validation path."""
    loader = XSMBDataLoader(url=_source_url(source), timeout=60)
    return loader.preprocess_and_build_matrices(loader.load_data())


@st.cache_data(show_spinner=False)
def load_report(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def execute_prize_poisson_90d(
    source: str,
    top_k: int,
    target_date: str,
) -> dict[str, Any]:
    """Run the repository's ordered-prize Poisson model for one target date."""
    raw = XSMBDataLoader(url=_source_url(source), timeout=60).load_data()
    target = pd.Timestamp(target_date)
    raw_dates = pd.to_datetime(raw["date"], format="%Y-%m-%d")
    raw = raw.loc[raw_dates < target].copy()
    if raw.empty:
        raise ValueError(f"No completed draws are available before {target_date}.")
    prizes = normalize_prizes(raw)
    latest = prizes.index[-1]
    expected_latest = target - pd.Timedelta(days=1)
    if latest != expected_latest:
        raise ValueError(
            f"Poisson Prize 90d requires the complete draw on {expected_latest:%Y-%m-%d}; "
            f"source ends on {latest:%Y-%m-%d}."
        )
    feature_start = target - pd.Timedelta(days=91)
    data = build_prize_feature_dataset(raw, include_unlabeled=True, start=feature_start)
    origins = data.index.get_level_values(0)
    labels = origins + pd.Timedelta(days=1)
    training_mask = (
        data[COUNT_TARGET].notna().to_numpy()
        & (labels >= target - pd.Timedelta(days=90))
        & (labels < target)
    )
    train = data.loc[training_mask]
    inference = data.loc[origins == latest].drop(columns=COUNT_TARGET)
    if train.empty or len(inference) != 100:
        raise ValueError(f"Insufficient ordered-prize history for {target_date}.")
    model = PrizeLGBMPredictor("poisson", history_days=90, n_estimators=80, n_jobs=1).fit(train)
    scores = model.predict_scores(inference).droplevel(0)
    top = scores.sort_values(ascending=False, kind="stable").head(top_k)
    return {
        "target_date": target.strftime("%Y-%m-%d"),
        "source": source,
        "model": "prize_poisson_90d",
        "top_k": top_k,
        "test_days": 90,
        "forecasts": {
            "prize_poisson_90d": [
                {"number": str(number), "score": float(score)}
                for number, score in top.items()
            ]
        },
        "evaluations": {},
        "payouts": {},
        "benchmark_note": "Primary ordered-prize model; rolling 90-day training window.",
        "log": f"Poisson Prize 90d forecast for {target:%Y-%m-%d} using data through {latest:%Y-%m-%d}.",
        "warnings": "",
    }


@st.cache_data(show_spinner=False)
def execute_forecast(
    source: str,
    model: str,
    top_k: int,
    test_days: int,
    target_date: str,
) -> dict[str, Any]:
    """Run the validated pipeline and return a serializable UI payload."""
    if model == "prize_poisson_90d":
        return execute_prize_poisson_90d(source, top_k, target_date)
    output = StringIO()
    errors = StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        result = run_pipeline(
            url=_source_url(source),
            model=model,
            top_k=top_k,
            test_days=test_days,
            n_splits=5,
            n_estimators=80,
            n_jobs=1,
            timeout=60,
            forecast_date=target_date,
        )
    forecasts = {
        name: [{"number": number, "score": float(score)} for number, score in values]
        for name, values in result.forecasts.items()
    }
    evaluations = {
        name: dict(evaluation.summary) for name, evaluation in result.evaluations.items()
    }
    payouts = {name: dict(payout.summary) for name, payout in result.payouts.items()}
    return {
        "target_date": result.prediction_date.strftime("%Y-%m-%d"),
        "source": source,
        "model": model,
        "top_k": top_k,
        "test_days": test_days,
        "forecasts": forecasts,
        "evaluations": evaluations,
        "payouts": payouts,
        "log": output.getvalue(),
        "warnings": errors.getvalue(),
    }


def _filter_payload(payload: dict[str, Any], models: list[str]) -> dict[str, Any]:
    """Keep selected models while preserving the original all-model run metadata."""
    filtered = dict(payload)
    filtered["forecasts"] = {name: payload["forecasts"][name] for name in models if name in payload["forecasts"]}
    filtered["evaluations"] = {name: payload["evaluations"][name] for name in models if name in payload["evaluations"]}
    filtered["payouts"] = {name: payload["payouts"][name] for name in models if name in payload["payouts"]}
    filtered["model"] = ",".join(models)
    return filtered


def _merge_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    """Merge independent model runs into one comparison payload."""
    if not payloads:
        raise ValueError("At least one model payload is required.")
    merged = dict(payloads[0])
    for payload in payloads[1:]:
        merged["forecasts"] = {**merged["forecasts"], **payload["forecasts"]}
        merged["evaluations"] = {**merged["evaluations"], **payload["evaluations"]}
        merged["payouts"] = {**merged["payouts"], **payload["payouts"]}
    merged["model"] = ",".join(merged["forecasts"])
    return merged


def _format_vnd(value: float | int) -> str:
    return f"{value:,.0f} VND"


def _forecast_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, predictions in payload["forecasts"].items():
        for rank, prediction in enumerate(predictions, start=1):
            rows.append(
                {
                    "Target date": payload["target_date"],
                    "Model": MODEL_LABELS.get(model, model.title()),
                    "Rank": rank,
                    "Number": prediction["number"],
                    "Score": prediction["score"],
                }
            )
    return pd.DataFrame(rows)


def _comparison_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, summary in payload["evaluations"].items():
        payout = payload["payouts"].get(model, {})
        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model.title()),
                "Model key": model,
                "Holdout hit rate": summary.get("hit_rate_at_k", 0.0),
                "Precision@K": summary.get("precision_at_k", 0.0),
                "Recall@K": summary.get("recall_at_k", 0.0),
                "Mean hits/day": summary.get("mean_hits", 0.0),
                "Validation days": summary.get("n_days", 0),
                "Profit": payout.get("profit", 0),
                "ROI": payout.get("roi", 0.0),
                "Evidence": "Holdout + payout" if summary else "Primary forecast; see prize benchmark",
                "Top numbers": " ".join(item["number"] for item in payload["forecasts"].get(model, [])),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Holdout hit rate", "Precision@K", "ROI"], ascending=False, ignore_index=True
    )


def _consensus_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Compare model agreement by rank; score scales are intentionally not mixed."""
    rows: list[dict[str, Any]] = []
    model_names = list(payload["forecasts"])
    for number in [f"{value:02d}" for value in range(100)]:
        ranks: list[int] = []
        selected_by: list[str] = []
        for model in model_names:
            numbers = [item["number"] for item in payload["forecasts"].get(model, [])]
            if number in numbers:
                ranks.append(numbers.index(number) + 1)
                selected_by.append(MODEL_LABELS.get(model, model.title()))
        if selected_by:
            rows.append(
                {
                    "Number": number,
                    "Models selecting": len(selected_by),
                    "Average rank": sum(ranks) / len(ranks),
                    "Selected by": ", ".join(selected_by),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Number", "Models selecting", "Average rank", "Selected by"])
    return pd.DataFrame(rows).sort_values(
        ["Models selecting", "Average rank"], ascending=[False, True], ignore_index=True
    )


def _metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">TIME-SERIES RESEARCH CONSOLE</div>
          <h1>Signal Desk</h1>
          <p>Make a traceable next-draw forecast, compare model behavior, and keep the evidence beside the signal.</p>
          <div class="hero-meta"><span>1 · Choose data</span><span>2 · Forecast today</span><span>3 · Compare evidence</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
        :root { --ink:#172126; --muted:#6c7b80; --cream:#f5f3ed; --paper:#fffefa; --teal:#0d6b67; --orange:#e8754f; --line:#d9dfd9; }
        .stApp { background:var(--cream); color:var(--ink); font-family:Manrope, sans-serif; }
        [data-testid="stSidebar"] { background:#e9eee8; border-right:1px solid var(--line); }
        .hero { background:linear-gradient(135deg,#123d3b 0%,#0d6b67 65%,#2f8d7b 100%); color:#fffefa; border-radius:20px; padding:29px 34px 24px; margin-bottom:22px; box-shadow:0 12px 30px rgba(13,107,103,.16); }
        .hero h1 { font-size:44px; line-height:1; margin:8px 0 12px; letter-spacing:-.04em; }
        .hero p { max-width:700px; color:#dceee8; font-size:15px; margin:0; }
        .hero-meta { display:flex; gap:10px; flex-wrap:wrap; margin-top:20px; color:#b7e7d7; font:500 11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
        .hero-meta span { border:1px solid rgba(183,231,215,.35); border-radius:99px; padding:7px 10px; }
        .eyebrow,.metric-label,.section-kicker { font:500 11px 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }
        .eyebrow { color:#a9e1cf; }
        .metric-card { background:var(--paper); border:1px solid var(--line); border-radius:14px; padding:15px 17px; min-height:90px; }
        .metric-label { color:var(--muted); }
        .metric-value { color:var(--teal); font-size:24px; font-weight:800; margin-top:8px; letter-spacing:-.03em; }
        .metric-detail { color:var(--muted); font-size:12px; margin-top:4px; }
        .section-kicker { color:var(--orange); margin:8px 0 7px; }
        .panel { background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:18px 20px; }
        .target-banner { background:#edf7f3; border:1px solid #b9ddd0; border-radius:14px; padding:16px 18px; }
        .target-banner.stale { background:#fff2e8; border-color:#f5be9d; }
        .target-date { color:var(--teal); font:700 26px 'DM Mono',monospace; }
        .number-pill { display:inline-flex; align-items:center; justify-content:center; background:#e3f2ed; color:var(--teal); border:1px solid #b9ddd0; border-radius:10px; width:50px; height:45px; margin:4px 6px 4px 0; font:500 19px 'DM Mono',monospace; }
        .number-pill:first-child { background:#ffe7d8; border-color:#f5be9d; color:#9c452a; }
        .stButton>button { border-radius:10px; border:0; background:var(--orange); color:white; font-weight:700; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _target_context(last_date: pd.Timestamp) -> dict[str, Any]:
    today = pd.Timestamp.now().normalize()
    next_date = last_date + pd.Timedelta(days=1)
    today_eligible = last_date == today - pd.Timedelta(days=1)
    return {
        "today": today,
        "next_date": next_date,
        "today_eligible": today_eligible,
        "source_fresh": last_date >= today - pd.Timedelta(days=1),
    }


def _target_from_controls(
    choice: str, context: dict[str, Any], custom_date: calendar_date | None
) -> tuple[str | None, str | None]:
    if choice == "Today":
        if not context["today_eligible"]:
            return None, (
                f"Today is not forecastable from this source. It ends on "
                f"{context['next_date'] - pd.Timedelta(days=1):%Y-%m-%d}; "
                "update the live feed or use Next eligible draw."
            )
        return context["today"].strftime("%Y-%m-%d"), None
    if choice == "Next eligible draw":
        return context["next_date"].strftime("%Y-%m-%d"), None
    if custom_date is None:
        return None, "Choose a custom target date."
    return custom_date.isoformat(), None


def _record_history(payload: dict[str, Any], mode: str) -> None:
    entry = {
        "Run time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Mode": mode,
        "Target date": payload["target_date"],
        "Models": ", ".join(MODEL_LABELS.get(name, name) for name in payload["forecasts"]),
        "Top K": payload["top_k"],
        "Source": "Live raw prize feed" if payload["source"] == RAW_PRIZE_URL else Path(payload["source"]).name,
    }
    history = st.session_state.setdefault("run_history", [])
    history.insert(0, entry)
    st.session_state["run_history"] = history[:20]


def _render_forecast_results(payload: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Forecast output</div>', unsafe_allow_html=True)
    st.markdown(f"### Target date · `{payload['target_date']}`")
    st.caption("Scores preserve each model's native meaning; they are not comparable across model families.")
    for model, predictions in payload["forecasts"].items():
        evaluation = payload["evaluations"].get(model, {})
        payout = payload["payouts"].get(model, {})
        with st.container(border=True):
            title_col, numbers_col, stats_col = st.columns([1.15, 1.9, 1.25])
            with title_col:
                st.markdown(f"**{MODEL_LABELS.get(model, model.title())}**")
                st.caption(MODEL_DESCRIPTIONS.get(model, ""))
            with numbers_col:
                pills = "".join(
                    f'<span class="number-pill">{item["number"]}</span>' for item in predictions
                )
                st.markdown(pills, unsafe_allow_html=True)
                st.caption("  ·  ".join(f"#{rank} {item['score']:.4f}" for rank, item in enumerate(predictions, 1)))
            with stats_col:
                if evaluation:
                    st.metric("Holdout hit rate", f"{evaluation['hit_rate_at_k']:.1%}")
                    st.caption(f"{evaluation['n_days']} validation days · ROI {payout.get('roi', 0):.1%}")
                else:
                    st.metric("Model window", "90 days")
                    st.caption("Primary prize-feature model · score = expected occurrences")
    frame = _forecast_frame(payload)
    st.download_button(
        "Download this forecast",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{payload['target_date']}.csv",
        mime="text/csv",
        width="stretch",
    )


def _render_comparison(payload: dict[str, Any]) -> None:
    comparison = _comparison_frame(payload)
    if comparison.empty:
        st.info("Run a comparison to see model evidence here.")
        return
    st.markdown(f"### Model comparison · `{payload['target_date']}`")
    leader = comparison.iloc[0]
    roi_leader = comparison.sort_values("ROI", ascending=False).iloc[0]
    card_cols = st.columns(3)
    with card_cols[0]:
        _metric_card("Validation leader", str(leader["Model"]), f"hit rate {leader['Holdout hit rate']:.1%}")
    with card_cols[1]:
        _metric_card("ROI leader", str(roi_leader["Model"]), f"historical ROI {roi_leader['ROI']:.1%}")
    with card_cols[2]:
        _metric_card("Forecast target", payload["target_date"], f"Top {payload['top_k']} per model")
    st.caption("Leaders are descriptive holdout signals, not promises of future performance.")
    display = comparison.copy()
    for column in ["Holdout hit rate", "Precision@K", "Recall@K", "ROI"]:
        display[column] = display[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.1%}"
        )
    display["Profit"] = display["Profit"].map(_format_vnd)
    st.dataframe(
        display[["Model", "Top numbers", "Holdout hit rate", "Precision@K", "Recall@K", "Mean hits/day", "Validation days", "Profit", "ROI", "Evidence"]],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download comparison CSV",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name=f"model_comparison_{payload['target_date']}.csv",
        mime="text/csv",
        width="stretch",
    )
    st.markdown("#### Agreement view")
    consensus = _consensus_frame(payload)
    if consensus.empty:
        st.info("No model selected a candidate in the current comparison.")
    else:
        st.caption("Agreement uses rank only. Scores are not averaged because model score scales differ.")
        st.dataframe(consensus.head(20), width="stretch", hide_index=True)
        common = consensus.iloc[0]
        if common["Models selecting"] > 1:
            st.success(f"Strongest agreement: {common['Number']} selected by {common['Selected by']}.")
        else:
            st.info("The current models do not share a common Top-K candidate; treat the ranking as model disagreement.")


def main() -> None:
    st.set_page_config(
        page_title="Signal Desk · XSMB Research",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _render_styles()
    _render_header()

    source_options = _source_options()
    if not source_options:
        st.error("No CSV data source was found under data/processed or data/raw.")
        st.stop()
    with st.sidebar:
        st.markdown("### 1 · Choose data")
        source_key = st.selectbox(
            "Data source",
            options=list(source_options),
            format_func=lambda key: source_options[key][0],
            help="The live feed is fetched on demand. Local snapshots are read from this repository.",
        )
        source = source_options[source_key][1]
        refresh_source = st.button("Refresh source cache", width="stretch")
    if refresh_source:
        load_matrices.clear()
        execute_forecast.clear()
        st.rerun()

    with st.spinner("Reading and validating the selected source…"):
        try:
            frequency, _ = load_matrices(source)
        except Exception as exc:
            st.error(f"Source could not be loaded: {exc}")
            st.stop()
    last_date = frequency.index.max()
    context = _target_context(last_date)
    recent = frequency.tail(30).sum().sort_values(ascending=False)
    report_summary = load_report(str(REPORT_SUMMARY)) if REPORT_SUMMARY.exists() else pd.DataFrame()
    report_daily = load_report(str(REPORT_DAILY)) if REPORT_DAILY.exists() else pd.DataFrame()

    with st.sidebar:
        st.markdown("### 2 · Define the forecast")
        target_choice = st.radio(
            "Target date",
            options=["Today", "Next eligible draw", "Custom date"],
            index=0 if context["today_eligible"] else 1,
            help="Today requires yesterday's completed draw to be present in the selected source.",
        )
        custom_date = None
        if target_choice == "Custom date":
            custom_date = st.date_input("Custom target", value=context["next_date"].date())
        model = st.selectbox(
            "Forecast model",
            options=["prize_poisson_90d", "poisson", "baseline", "lightgbm", "regularized"],
            format_func=lambda value: MODEL_LABELS[value],
            index=0,
        )
        compare_models = st.multiselect(
            "Models to compare",
            options=list(MODEL_LABELS),
            default=list(MODEL_LABELS),
            format_func=lambda value: MODEL_LABELS[value],
        )
        top_k = st.slider("Numbers to rank", min_value=1, max_value=10, value=3)
        test_days = st.slider("Validation days", min_value=30, max_value=300, value=90, step=15)
        target_date, target_error = _target_from_controls(target_choice, context, custom_date)
        if target_choice == "Today" and not context["today_eligible"]:
            st.warning(
                f"Today is unavailable: the source ends {last_date:%Y-%m-%d}. "
                f"Next eligible target is {context['next_date']:%Y-%m-%d}."
            )
        forecast_clicked = st.button("Forecast selected model", type="primary", width="stretch")
        compare_clicked = st.button("Compare selected models", width="stretch")

    if forecast_clicked or compare_clicked:
        if target_error or target_date is None:
            st.error(target_error or "Choose a valid target date before running.")
        elif compare_clicked and not compare_models:
            st.error("Select at least one model to compare.")
        else:
            with st.spinner("Running the validated chronological pipeline…"):
                try:
                    if forecast_clicked:
                        payload = execute_forecast(source, model, top_k, test_days, target_date)
                        st.session_state["forecast_payload"] = payload
                        _record_history(payload, "Forecast")
                    if compare_clicked:
                        all_payload = execute_forecast(source, "all", top_k, test_days, target_date)
                        prize_payload = (
                            execute_forecast(source, "prize_poisson_90d", top_k, test_days, target_date)
                            if "prize_poisson_90d" in compare_models
                            else None
                        )
                        generic_models = [model_name for model_name in compare_models if model_name != "prize_poisson_90d"]
                        selected_payload = _filter_payload(all_payload, generic_models)
                        payload = _merge_payloads(
                            *([prize_payload] if prize_payload else []),
                            selected_payload,
                        )
                        st.session_state["compare_payload"] = payload
                        _record_history(payload, "Compare")
                except Exception as exc:
                    st.error(f"The run could not be completed: {exc}")

    st.markdown('<div class="section-kicker">Dataset pulse</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        _metric_card("Observed days", f"{len(frequency):,}", f"through {last_date:%d %b %Y}")
    with metric_cols[1]:
        _metric_card("Source state", "Ready" if context["source_fresh"] else "Stale", f"target basis {context['next_date']:%Y-%m-%d}")
    with metric_cols[2]:
        _metric_card("Draw positions", "27", "per completed day")
    with metric_cols[3]:
        best = report_summary.sort_values("roi", ascending=False).iloc[0] if not report_summary.empty else None
        _metric_card("Best reviewed ROI", f"{best.roi:.1%}" if best is not None else "—", best.model if best is not None else "no report loaded")

    st.markdown('<div class="section-kicker">Forecast readiness</div>', unsafe_allow_html=True)
    target_class = "target-banner" if context["today_eligible"] else "target-banner stale"
    if context["today_eligible"]:
        target_message = "The selected source contains yesterday's draw. You can forecast today without exposing a future outcome."
    else:
        target_message = "Today is not yet reproducible from this source. Use the live feed or run the next eligible target shown below."
    st.markdown(
        f'<div class="{target_class}"><div class="eyebrow">FORECAST READINESS</div>'
        f'<div class="target-date">{context["today"]:%Y-%m-%d} · Today</div><div>{target_message}</div>'
        f'<div style="margin-top:8px;color:#6c7b80">Last complete draw: <b>{last_date:%Y-%m-%d}</b> · Next eligible target: <b>{context["next_date"]:%Y-%m-%d}</b></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Recent context</div>', unsafe_allow_html=True)
    signal_cols = st.columns([1.35, 1, 1])
    with signal_cols[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"**Last complete draw**  \n`{last_date:%Y-%m-%d}`")
        st.caption("Most frequent endings in the trailing 30 observed days")
        pills = "".join(f'<span class="number-pill">{number}</span>' for number in recent.head(8).index)
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with signal_cols[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**30-day frequency leaders**")
        st.dataframe(recent.head(8).rename("occurrences").to_frame(), width="stretch", height=245)
        st.markdown('</div>', unsafe_allow_html=True)
    with signal_cols[2]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**How to use the workspace**")
        st.write("Choose a source, select a target, then run one model for a focused signal or compare several models for agreement and holdout evidence.")
        st.markdown('</div>', unsafe_allow_html=True)

    forecast_tab, compare_tab, history_tab = st.tabs(["Forecast today", "Compare models", "Run history & benchmark"])
    with forecast_tab:
        payload = st.session_state.get("forecast_payload")
        if payload:
            _render_forecast_results(payload)
        else:
            st.info("Choose a model and target in the sidebar, then click Forecast selected model.")
    with compare_tab:
        payload = st.session_state.get("compare_payload")
        if payload:
            _render_comparison(payload)
        else:
            st.info("Select two or more models in the sidebar, then click Compare selected models.")
    with history_tab:
        st.markdown("### Session runs")
        history = st.session_state.get("run_history", [])
        if history:
            st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
        else:
            st.caption("Your completed forecasts and comparisons will appear here during this session.")
        st.markdown("### Reviewed benchmark")
        if report_summary.empty:
            st.info("No curated comparison report is available yet.")
        else:
            comparison = report_summary.copy()
            comparison["model"] = comparison["model"].map(MODEL_LABELS).fillna(comparison["model"])
            comparison["ROI"] = comparison["roi"].map(lambda value: f"{value:.1%}")
            comparison["Profit"] = comparison["profit_vnd"].map(_format_vnd)
            st.dataframe(
                comparison[["model", "days", "top_k", "occurrences", "distinct_hits", "profitable_days", "Profit", "ROI"]]
                .rename(columns={"model": "Model", "days": "Days", "top_k": "Top K", "occurrences": "Occurrences", "distinct_hits": "Distinct hits", "profitable_days": "Profitable days"}),
                width="stretch",
                hide_index=True,
            )
            if not report_daily.empty:
                daily = report_daily.copy()
                daily["date"] = pd.to_datetime(daily["date"])
                pivot = daily.pivot_table(index="date", columns="model", values="profit_vnd", aggfunc="sum").sort_index()
                st.line_chart(pivot, y_label="Daily profit (VND)")
        with st.expander("Method and safety notes"):
            st.markdown(
                """
                - Forecasts are ranked candidates, not calibrated odds or betting advice.
                - Today is only enabled when yesterday's completed draw is present in the selected source.
                - The pipeline uses chronological splits and causal features; future outcomes are excluded from each forecast.
                - Baseline scores are trailing counts, LightGBM and regularized scores are uncalibrated presence estimates, and Poisson scores are expected occurrence counts.
                - Historical payout figures use the repository's documented paper-trading assumptions.
                """
            )


if __name__ == "__main__":
    main()
