# Ordered-prize model training and chronological comparison

**Bias audit:** historical periods have been repeatedly examined; positive quarters are not independent evidence of profitability. The Q3/2026 binary forecasts were exactly reproduced using training results only through June 30. See the [detailed audit and all monthly/quarterly results](./bias_audit/report.md).

Forecast date: 2026-10-30; training results through 2026-10-29. 206 input features including 27 ordered prize positions and all available leading digits.

Both experimental models were saved and reload predictions were checked for exact equality. Binary estimates appearance probability; Poisson estimates occurrence count, NOT probability. Raw ranking is retained even if calibration collapses scores to a constant.

Evaluation: chronological quarter-by-quarter refits with identical training cutoffs and 1095-day label windows. This is a different policy from daily-refit reports. 2025/2026 were already examined, so this is retrospective. No tuning, profit-driven model selection, or automatic replacement of the existing model.

| Year | Model | K | Days | Precision@K | Paid occurrences | Profit (VND) | ROI |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025 | regularized | 1 | 360 | 21.11% | 87 | -1,320,000 | -15.94% |
| 2025 | regularized | 5 | 360 | 22.17% | 455 | -5,000,000 | -12.08% |
| 2025 | regularized | 10 | 360 | 23.00% | 945 | -7,200,000 | -8.70% |
| 2025 | prize_binary | 1 | 360 | 20.83% | 86 | -1,400,000 | -16.91% |
| 2025 | prize_binary | 5 | 360 | 23.06% | 478 | -3,160,000 | -7.63% |
| 2025 | prize_binary | 10 | 360 | 23.06% | 948 | -6,960,000 | -8.41% |
| 2025 | prize_poisson | 1 | 360 | 21.39% | 95 | -680,000 | -8.21% |
| 2025 | prize_poisson | 5 | 360 | 23.22% | 481 | -2,920,000 | -7.05% |
| 2025 | prize_poisson | 10 | 360 | 23.06% | 950 | -6,800,000 | -8.21% |
| 2026 | regularized | 1 | 297 | 25.25% | 87 | +129,000 | 1.89% |
| 2026 | regularized | 5 | 297 | 25.19% | 431 | +325,000 | 0.95% |
| 2026 | regularized | 10 | 297 | 24.14% | 826 | -2,230,000 | -3.26% |
| 2026 | prize_binary | 1 | 297 | 21.89% | 80 | -431,000 | -6.31% |
| 2026 | prize_binary | 5 | 297 | 24.85% | 446 | +1,525,000 | 4.46% |
| 2026 | prize_binary | 10 | 297 | 24.11% | 836 | -1,430,000 | -2.09% |
| 2026 | prize_poisson | 1 | 297 | 26.26% | 91 | +449,000 | 6.57% |
| 2026 | prize_poisson | 5 | 297 | 23.97% | 420 | -555,000 | -1.62% |
| 2026 | prize_poisson | 10 | 297 | 23.74% | 823 | -2,470,000 | -3.62% |

Cash flow: 23,000 VND per selected number/day; gross 80,000 VND EACH occurrence, counting repeats. Missing target or preceding days are excluded, not imputed as losses/abstentions. Profit-difference intervals in ranking.csv use 2,000 seven-observation-day block bootstrap samples; they are approximate, cross calendar gaps, and are not adjusted for comparing multiple models/K.

Feature importances describe tree usage; they do not establish causality or a hidden physical rule. Temporal order follows the user's confirmation, not independent video/time verification.

[Ranking](./ranking.csv) · [Metrics](./metrics.csv) · [Daily results](./daily.csv) · [Forecast](./forecast_top_k.csv) · [Excluded dates](./excluded_dates.csv) · [Protocol](./protocol.json) · [Model manifest](./models/manifest.json)
