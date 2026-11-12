# Model comparison: 2026 Top 3, rolling 90-day training window

Results cover 2026-01-01 through 2026-11-01. 300/305 calendar days were evaluable; 5 dates were excluded.

Downloaded the current raw-prize feed on this run. Each daily fit uses up to 90 calendar days of labels strictly before the target. All models use the same target dates, Top 3, 27,000 VND stake per selection and 99,000 VND gross receipt per matching occurrence; there is no separate stake refund.

| Model | Days | Occurrences | Profitable days | Cost (VND) | Receipts (VND) | Profit (VND) | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| regularized | 300 | 258 | 173 | 24,300,000 | 25,542,000 | +1,242,000 | 5.11% |
| poisson | 300 | 280 | 180 | 24,300,000 | 27,720,000 | +3,420,000 | 14.07% |
| binary | 300 | 264 | 176 | 24,300,000 | 26,136,000 | +1,836,000 | 7.56% |
| lightgbm | 300 | 264 | 169 | 24,300,000 | 26,136,000 | +1,836,000 | 7.56% |

## Monthly profit/loss

| Month | Model | Days | Cost (VND) | Receipts (VND) | Profit (VND) |
|---|---|---:|---:|---:|---:|
| 2026-01 | regularized | 31 | 2,511,000 | 2,772,000 | +261,000 |
| 2026-02 | regularized | 23 | 1,863,000 | 1,485,000 | -378,000 |
| 2026-03 | regularized | 31 | 2,511,000 | 2,673,000 | +162,000 |
| 2026-04 | regularized | 30 | 2,430,000 | 2,871,000 | +441,000 |
| 2026-05 | regularized | 31 | 2,511,000 | 2,277,000 | -234,000 |
| 2026-06 | regularized | 30 | 2,430,000 | 2,475,000 | +45,000 |
| 2026-07 | regularized | 31 | 2,511,000 | 2,376,000 | -135,000 |
| 2026-08 | regularized | 31 | 2,511,000 | 2,673,000 | +162,000 |
| 2026-09 | regularized | 30 | 2,430,000 | 2,673,000 | +243,000 |
| 2026-10 | regularized | 31 | 2,511,000 | 3,069,000 | +558,000 |
| 2026-11 | regularized | 1 | 81,000 | 198,000 | +117,000 |
| 2026-01 | poisson | 31 | 2,511,000 | 3,069,000 | +558,000 |
| 2026-02 | poisson | 23 | 1,863,000 | 1,188,000 | -675,000 |
| 2026-03 | poisson | 31 | 2,511,000 | 3,069,000 | +558,000 |
| 2026-04 | poisson | 30 | 2,430,000 | 3,861,000 | +1,431,000 |
| 2026-05 | poisson | 31 | 2,511,000 | 2,772,000 | +261,000 |
| 2026-06 | poisson | 30 | 2,430,000 | 2,277,000 | -153,000 |
| 2026-07 | poisson | 31 | 2,511,000 | 3,069,000 | +558,000 |
| 2026-08 | poisson | 31 | 2,511,000 | 2,376,000 | -135,000 |
| 2026-09 | poisson | 30 | 2,430,000 | 2,871,000 | +441,000 |
| 2026-10 | poisson | 31 | 2,511,000 | 3,168,000 | +657,000 |
| 2026-11 | poisson | 1 | 81,000 | 0 | -81,000 |
| 2026-01 | binary | 31 | 2,511,000 | 2,673,000 | +162,000 |
| 2026-02 | binary | 23 | 1,863,000 | 1,584,000 | -279,000 |
| 2026-03 | binary | 31 | 2,511,000 | 2,475,000 | -36,000 |
| 2026-04 | binary | 30 | 2,430,000 | 3,366,000 | +936,000 |
| 2026-05 | binary | 31 | 2,511,000 | 2,574,000 | +63,000 |
| 2026-06 | binary | 30 | 2,430,000 | 2,772,000 | +342,000 |
| 2026-07 | binary | 31 | 2,511,000 | 2,871,000 | +360,000 |
| 2026-08 | binary | 31 | 2,511,000 | 2,574,000 | +63,000 |
| 2026-09 | binary | 30 | 2,430,000 | 2,277,000 | -153,000 |
| 2026-10 | binary | 31 | 2,511,000 | 2,970,000 | +459,000 |
| 2026-11 | binary | 1 | 81,000 | 0 | -81,000 |
| 2026-01 | lightgbm | 31 | 2,511,000 | 2,673,000 | +162,000 |
| 2026-02 | lightgbm | 23 | 1,863,000 | 1,980,000 | +117,000 |
| 2026-03 | lightgbm | 31 | 2,511,000 | 1,485,000 | -1,026,000 |
| 2026-04 | lightgbm | 30 | 2,430,000 | 2,277,000 | -153,000 |
| 2026-05 | lightgbm | 31 | 2,511,000 | 2,970,000 | +459,000 |
| 2026-06 | lightgbm | 30 | 2,430,000 | 2,376,000 | -54,000 |
| 2026-07 | lightgbm | 31 | 2,511,000 | 2,772,000 | +261,000 |
| 2026-08 | lightgbm | 31 | 2,511,000 | 3,366,000 | +855,000 |
| 2026-09 | lightgbm | 30 | 2,430,000 | 2,871,000 | +441,000 |
| 2026-10 | lightgbm | 31 | 2,511,000 | 3,366,000 | +855,000 |
| 2026-11 | lightgbm | 1 | 81,000 | 0 | -81,000 |

Retrospective backtest results do not prove future advantage. The Poisson score estimates occurrence counts; binary scores estimate presence and are uncalibrated. Do not interpret these model scores as reliable real-world odds or select a future model solely from this comparison.

The source currently ends on 2026-11-01; 2026 is not yet a complete year. Dates without the target draw or its preceding calendar draw are excluded, not counted as losses.

Files: [daily.csv](./daily.csv), [summary.csv](./summary.csv), [monthly.csv](./monthly.csv), [protocol.json](./protocol.json), [excluded_dates.csv](./excluded_dates.csv).
