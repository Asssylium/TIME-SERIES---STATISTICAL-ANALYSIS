# Regularized Top 10: 2025-01-01 to 2025-12-31

Daily refit of the current RegularizedLGBMPredictor with unchanged default parameters. Select the highest presence probabilities, not expected occurrence counts. One unit per selected number per day: cost 23,000 VND; gross receipt 80,000 VND for EACH occurrence. No separate stake refund, compounding, threshold, or adaptive Top K.

Evaluated 360/365 calendar days; 5 excluded. Missing outcomes or missing preceding days have UNKNOWN cash flows, not zero losses or assumed abstention. Totals and cumulative profit describe evaluable dates only.

| Month | Evaluated days | Excluded | Occurrences | Cost (VND) | Receipts (VND) | Profit (VND) | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-01 | 27 | 4 | 73 | 6,210,000 | 5,840,000 | -370,000 | -5.96% |
| 2025-02 | 27 | 1 | 63 | 6,210,000 | 5,040,000 | -1,170,000 | -18.84% |
| 2025-03 | 31 | 0 | 90 | 7,130,000 | 7,200,000 | +70,000 | 0.98% |
| 2025-04 | 30 | 0 | 84 | 6,900,000 | 6,720,000 | -180,000 | -2.61% |
| 2025-05 | 31 | 0 | 86 | 7,130,000 | 6,880,000 | -250,000 | -3.51% |
| 2025-06 | 30 | 0 | 87 | 6,900,000 | 6,960,000 | +60,000 | 0.87% |
| 2025-07 | 31 | 0 | 69 | 7,130,000 | 5,520,000 | -1,610,000 | -22.58% |
| 2025-08 | 31 | 0 | 82 | 7,130,000 | 6,560,000 | -570,000 | -7.99% |
| 2025-09 | 30 | 0 | 85 | 6,900,000 | 6,800,000 | -100,000 | -1.45% |
| 2025-10 | 31 | 0 | 89 | 7,130,000 | 7,120,000 | -10,000 | -0.14% |
| 2025-11 | 30 | 0 | 71 | 6,900,000 | 5,680,000 | -1,220,000 | -17.68% |
| 2025-12 | 31 | 0 | 88 | 7,130,000 | 7,040,000 | -90,000 | -1.26% |
| TOTAL | 360 | 5 | 967 | 82,800,000 | 77,360,000 | -5,440,000 | -6.57% |

Profitable days: 176; losing days: 184. Maximum drawdown over evaluated days: 6,130,000 VND.

Training labels for each target end before that target. The model retains the preceding 1095 calendar days of labels relative to the latest known training label. All-history causal features are computed once and sliced by date; prediction rows contain no target column. Missing dates are never shifted to a different calendar day.

This is a retrospective reconstruction using today's model. The 2025 period was previously used in model research and calibration; it is not an untouched prospective test. No hyperparameters or Top K were selected during this run.

[Daily results](./daily.csv) · [Monthly results](./monthly.csv) · [Summary](./summary.csv) · [Excluded dates](./excluded_dates.csv) · [Protocol](./protocol.json)
