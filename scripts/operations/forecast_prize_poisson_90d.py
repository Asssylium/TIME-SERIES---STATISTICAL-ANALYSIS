"""Forecast the next XSMB draw with the rolling 90-day Prize Poisson model."""

from __future__ import annotations

import argparse

import pandas as pd

from src.count_features import COUNT_TARGET
from src.data_loader import XSMBDataLoader
from src.prize_features import RAW_PRIZE_URL, build_prize_feature_dataset, normalize_prizes
from src.prize_models import PrizeLGBMPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3, help="Number of endings to print (default: 3).")
    parser.add_argument("--url", default=RAW_PRIZE_URL, help="Raw prize CSV URL.")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k must be in 1..100.")

    raw = XSMBDataLoader(url=args.url).load_data()
    prizes = normalize_prizes(raw)
    latest = prizes.index[-1]
    target = latest + pd.Timedelta(days=1)
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
        raise ValueError(f"Incomplete data for forecast {target:%Y-%m-%d}.")

    model = PrizeLGBMPredictor("poisson", history_days=90).fit(train)
    scores = model.predict_scores(inference).droplevel(0)
    top = scores.sort_values(ascending=False, kind="stable").head(args.top_k)

    print(f"Poisson Prize 90d — forecast for {target:%Y-%m-%d} (data through {latest:%Y-%m-%d})")
    for rank, (number, score) in enumerate(top.items(), start=1):
        print(f"{rank}. {number}: expected count {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
