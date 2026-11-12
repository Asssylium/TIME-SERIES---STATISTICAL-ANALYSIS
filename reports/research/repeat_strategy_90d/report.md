# Top-3 repeat strategy research

## Protocol

- Strategy selection period: selection_2025. The best strategy is selected by hit rate only.
- Previously reviewed stress-test period: known_stress_2026; it is descriptive, not an untouched holdout.
- All predictions use cached daily walk-forward scores. Frequency baselines use only results from the preceding 90 calendar days.
- Repeat filters exclude numbers selected by Poisson on the immediately preceding 1 or 3 calendar days, then fill from the current Poisson ranking. Repeat-pick diagnostics classify a selection as repeated if the same strategy selected it in the prior 3 calendar days; its hit rate is per selected number, not per day.
- A hit-day means at least one distinct Top-3 number appeared. Occurrences affect receipts, not hit-day count.
- Approximate random-selection hit-rate reference: 56.1%, assuming 27 independent, uniform prize endings; this is only a rough reference.
- Payout accounting: 27,000 VND per selection and 99,000 VND per occurrence; no stake refund.
- Poisson is trained on repeat-inclusive occurrence counts, while Binary is trained on next-day presence. Neither objective is guaranteed to maximize Top-3 hit-days; the repeat filters test whether removing recent picks helps rather than assuming it does.

## Selection-period results

| Strategy | Hit days | Hit rate (95% Wilson CI) | Δ vs Poisson (95% 7-observation block CI) | 2+ distinct | 3 distinct | 3+ occurrences | 4+ occurrences | Total occurrences | Repeat picks (hit rate; net) | Other picks net | Miss streak max | Loss days | Loss streak max | Net (VND) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:
| binary | 203/360 | 56.4% (51.2%–61.4%) | +0.3% [-6.1%, +6.9%] | 35 | 2 | 12 | 3 | 270 | 642 (23.1%; -1,197,000) | -1,233,000 | 6 | 157 | 6 | -2,430,000 |
| poisson | 202/360 | 56.1% (50.9%–61.1%) | +0.0% [+0.0%, +0.0%] | 41 | 5 | 13 | 1 | 284 | 650 (22.8%; -1,017,000) | -27,000 | 5 | 158 | 5 | -1,044,000 |
| poisson_no_repeat_3d | 201/360 | 55.8% (50.7%–60.9%) | -0.3% [-6.1%, +5.6%] | 48 | 6 | 15 | 5 | 295 | 369 (23.8%; -162,000) | +207,000 | 6 | 159 | 6 | +45,000 |
| frequency_90d | 199/360 | 55.3% (50.1%–60.3%) | -0.8% [-10.0%, +7.8%] | 45 | 5 | 14 | 3 | 277 | 952 (23.2%; -1,251,000) | -486,000 | 9 | 161 | 9 | -1,737,000 |
| poisson_no_repeat_1d | 198/360 | 55.0% (49.8%–60.1%) | -1.1% [-6.7%, +4.7%] | 44 | 6 | 19 | 1 | 287 | 420 (21.9%; -1,143,000) | +396,000 | 7 | 162 | 7 | -747,000 |

## Previously reviewed stress-test results

| Strategy | Hit days | Hit rate (95% Wilson CI) | Δ vs Poisson (95% 7-observation block CI) | 2+ distinct | 3 distinct | 3+ occurrences | 4+ occurrences | Total occurrences | Repeat picks (hit rate; net) | Other picks net | Miss streak max | Loss days | Loss streak max | Net (VND) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:
| poisson | 180/300 | 60.0% (54.4%–65.4%) | +0.0% [+0.0%, +0.0%] | 56 | 7 | 18 | 6 | 280 | 583 (27.8%; +2,970,000) | +450,000 | 6 | 120 | 6 | +3,420,000 |
| binary | 176/300 | 58.7% (53.0%–64.1%) | -1.3% [-6.3%, +3.3%] | 52 | 5 | 17 | 6 | 264 | 589 (25.8%; +1,125,000) | +711,000 | 5 | 124 | 5 | +1,836,000 |
| poisson_no_repeat_3d | 176/300 | 58.7% (53.0%–64.1%) | -1.3% [-8.3%, +5.3%] | 39 | 5 | 17 | 4 | 254 | 359 (24.0%; +108,000) | +738,000 | 9 | 124 | 9 | +846,000 |
| frequency_90d | 174/300 | 58.0% (52.3%–63.4%) | -2.0% [-9.3%, +5.0%] | 60 | 8 | 21 | 8 | 284 | 830 (26.5%; +3,231,000) | +585,000 | 5 | 126 | 5 | +3,816,000 |
| poisson_no_repeat_1d | 172/300 | 57.3% (51.7%–62.8%) | -2.7% [-9.3%, +3.3%] | 41 | 7 | 18 | 3 | 251 | 413 (24.9%; -63,000) | +612,000 | 8 | 128 | 8 | +549,000 |

## Interpretation

The selection-period winner by hit rate is **binary** (56.4%, 203/360). On the already-reviewed stress-test it reached 58.7% (176/300).
This is not evidence that the rule will outperform prospectively: the stress-test year has already been inspected, and the strategy set is small but still selected using historical outcomes.
Do not switch the production command based on these retrospective results. Freeze the selected rule and log forecasts before results for a prospective paper-trial.
Wilson intervals treat days as independent; paired moving-block intervals resample 7 evaluated observations to allow short-range dependence. These intervals are descriptive and not corrected for trying multiple strategies. Monthly results and calendar-aware daily streaks are in monthly.csv and daily.csv.

Files: [daily.csv](./daily.csv), [summary.csv](./summary.csv), [monthly.csv](./monthly.csv), [hit_rate_deltas.csv](./hit_rate_deltas.csv).
