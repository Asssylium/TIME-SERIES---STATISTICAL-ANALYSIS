# Daily Lottery Draw Forecasting

A Python-based research project for forecasting daily lottery draw outcomes using time-series features and machine learning. The project treats each number from 00 to 99 as a separate time series and builds causal features from historical daily results to rank the most likely numbers for the next draw.

This repository is designed for experimentation, backtesting, and research analysis. It is not a production gambling system and does not place bets or execute live transactions.

## Project goal

The objective is to model the probability or expected frequency of each number appearing on the next day, then select a Top-K list of the most promising numbers based on predicted scores.

The system combines:
- historical daily draw data
- robust validation with time-based splits
- causal feature engineering
- baseline and LightGBM models
- evaluation of predictive quality and payout-style metrics

## What the project does

- Loads daily draw CSV data and validates integrity
- Converts raw prize draws into 100-number frequency and binary matrices
- Builds date-indexed features such as:
  - lagged frequency
  - presence/absence indicators
  - rolling-window counts
  - gap days since last appearance
  - co-occurrence summaries
- Fits forecast models for next-day ranking
- Evaluates models with chronological train/test splits
- Prints summary reports and Top-K forecasts
- Supports research and paper-trading simulations using companion scripts and reporting docs

## Technology stack

- Python 3.10+
- NumPy
- pandas
- scikit-learn
- LightGBM
- unittest for automated validation

## Repository layout

```text
.
├── main.py                     # Stable CLI pipeline entry point
├── public/                     # Browser dashboard served locally and on Vercel
├── api/                        # Vercel serverless API
├── scripts/
│   ├── research/               # Research experiments
│   ├── reports/                # Report generation
│   ├── backtests/              # Backtests and model comparisons
│   └── operations/             # Forecasting, training, paper trading
├── src/                        # Reusable forecasting library
├── tests/                      # Automated tests
├── data/                       # Raw and processed datasets
├── reports/                    # Curated and generated outputs
├── requirements.txt
├── RESEARCH.md
└── PAPER_TRADING.md
```
## Core pipeline

The main execution path is handled in `main.py`.

The workflow is roughly:

1. Load source CSV data
2. Validate date integrity and draw counts
3. Convert to 100-column frequency and binary matrices
4. Build causal features using only information available through each feature date
5. Create chronological holdout splits
6. Train models on earlier data
7. Forecast next-day scores for the test period
8. Evaluate hits, precision, payout-style summaries, and final Top-K recommendations

## Model types

### Baseline predictor

Uses rolling historical frequencies over a fixed window (default 30 days) to rank numbers by count.

### LightGBM predictor

Fits a binary LightGBM classifier on per-number rows. The label indicates whether a number appears at least once on the next day. Probabilities are uncalibrated model outputs, not final betting odds.

### Regularized model

An experimental alternative that uses normalized count-based features and regularized training dynamics. It is designed for research comparison rather than being treated as the default production model.

### Poisson count model

Fits a Poisson LightGBM regressor to next-day occurrence counts, including repeats. Its ranked scores estimate expected occurrence counts, not appearance probabilities. It uses normalized count features and is available alongside the other models in the common command.

## Validation strategy

The project uses time-series validation instead of random cross-validation:
- training and test periods are ordered chronologically
- the final held-out fold is selected as the most recent valid split
- gaps are enforced so training labels do not overlap with future feature periods
- all 100 numbers for each day move together as a group

This avoids leakage and reflects the constraints of real forecasting.

## Quick start (Windows PowerShell)

### Web application

Run the Flask API locally to use the same browser dashboard deployed on Vercel.

```powershell
Set-Location 'D:\TIME-SERIES & STATISTICAL ANALYSIS'
uv run --with-requirements requirements.txt python -m flask --app api.index run --port 8000
```

Open `http://localhost:8000`. Press Ctrl+C in the terminal to stop the server.

### Vercel deployment

The project uses a serverless Flask API and a browser dashboard. The Vercel entry point is
`api/index.py`. The API and dashboard use the validated live GitHub prize feed
as their only data source; local CSV snapshots remain research fixtures and are
not bundled into the deployment.

Install the Vercel CLI, link the repository once, and start a local Vercel
environment:

```powershell
npm install --global vercel
vercel login
vercel link
vercel dev
```

Open `http://localhost:3000`. To publish a preview or production deployment:

```powershell
vercel
vercel --prod
```

The deployment uses `vercel.json` and exposes `/api/health`, `/api/sources`, and
`POST /api/forecast` for the browser UI or other clients.

Every settled forecast response includes an `analytics` object using schema
version `1.0`. Its `models[].series[]` rows use stable names for `date`, `hits`,
`paid_occurrences`, `stake`, `receipts`, `profit`, `roi`, `cumulative_profit`,
`cumulative_roi`, and `drawdown`. Rates are decimal ratios and currency values
are VND. The Chart analysis page can run selected models over 30 days, 90 days,
12 months, or a custom 30–366 day range and can export these rows as CSV.

`poisson_prize_90d` accepts a historical `target_date` as long as the previous
calendar day's completed draw exists. Add `target_end_date` to run an inclusive
walk-forward range; every date is independently refitted with labels strictly
before its target. The dashboard automatically splits long selections into
31-day requests and combines the daily results. Direct API clients should do
the same so each Vercel invocation stays within its execution window.

```json
{
  "source": "live",
  "model": "poisson_prize_90d",
  "top_k": 3,
  "target_date": "2026-10-01",
  "target_end_date": "2026-10-31"
}
```

Python is pinned to 3.12 in `.python-version`. The empty `installCommand` in
`vercel.json` skips the static site's automatic Python dependency installation;
the Python API builder still installs `requirements.txt`. The OpenMP build
step uses only the standard library. Do not add a second pip/uv install to the
static build: it creates `.vercel_python_packages`, duplicating the API's
dependencies and potentially exceeding the function bundle limit. This folder
is explicitly excluded, including when restored from an older build cache.

After changing these deployment settings, redeploy without the build cache.

Open PowerShell in the repository folder:

```powershell
Set-Location 'D:\TIME-SERIES & STATISTICAL ANALYSIS'
$python = 'C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe'
function py { & $python -c "import os,runpy,sys; sys.path[:0]=[r'D:\xsmb-model-check\deps',os.getcwd()]; a=sys.argv[1:]; m=a[0]=='-m'; t=a[1] if m else a[0]; sys.argv=a[1:] if m else a; (runpy.run_module if m else runpy.run_path)(t,run_name='__main__')" @args }
```

Run this short setup once in each new PowerShell window. Then `py main.py ...` and `py -m unittest ...` use the configured interpreter and dependencies. These paths are specific to this machine; change them if your Python or dependency folder is elsewhere.

On a different machine with Python on `PATH`, install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Run the default chronological evaluation and forecast:

```powershell
py main.py
```

The default forecast date is the day after the latest available draw in the source data. The command also evaluates a recent chronological holdout; it is not an automated daily schedule.

## Commands and when to run them

There is no required weekday for the command-line forecasting or research scripts. Run them when you need an evaluation or a forecast. A forecast for date **D** requires the draw for **D − 1** to be present in the input data; run it after that result is published. If you omit `--forecast-date`, `main.py` uses the latest observed date plus one day.

| Workflow | When to run |
|---|---|
| `main.py` forecast | On demand, after the previous day's draw is available; no fixed weekday |
| Paper journal forecast | Every day by 18:00 UTC+7 |
| Paper journal settlement | Every day after 19:00 UTC+7; scheduled run is 19:30 |
| Research/backtest/diagnostics | On demand for a planned study; not daily |
| Tests | After code changes |

### Forecast and evaluate

To run all models at once and print a separate Top 3 for each, run this after the previous day's draw has been published:

```powershell
py main.py --model all --top-k 3
```

The output contains separate sections for `BASELINE TOP 3`, `LIGHTGBM TOP 3`, `REGULARIZED TOP 3`, and `POISSON TOP 3`. The scores are not merged: baseline gives 30-day counts, LightGBM and regularized give uncalibrated appearance estimates, and Poisson gives expected counts including repeats. To run Poisson by itself, use `--model poisson`. To forecast a specific date, specify it explicitly; the source must contain the preceding day's draw:

```powershell
py main.py --model all --top-k 3 --forecast-date 2026-10-31
```

```powershell
py main.py --model poisson --top-k 3
```

`main.py --model poisson` is the **Poisson Count** model (count/frequency features and all available history); it is not the Prize-feature model. To forecast the next draw with **Poisson Prize 90d** (ordered features from all 27 prize positions, rolling 90-day training window), run:

```powershell
uv run --with-requirements requirements.txt python forecast_prize_poisson_90d.py
```

The forecast command downloads the raw prize feed and prints the Top 3 for the day after its latest complete draw. Use `--top-k 5` to print five endings, or `--url <CSV_URL>` to supply another raw prize feed. `compare_models_2026.py --model poisson --history-days 90` is for historical evaluation/backtesting, not forecasting an outcome that has not been published.

`main.py` downloads its default live frequency feed each time it runs; it does not reuse the previous run's data. For a refreshed 2026 profit/loss comparison of Regularized, Poisson, Binary, and LightGBM using a rolling 90-day training window, run:

```powershell
py compare_models_2026.py
```

This downloads the latest raw-prize CSV, evaluates all models on the same available dates and Top 3 using the current payout rule, and writes a dated run under `work/reports/`. Run it again after the source publishes a new draw to refresh the comparison. Promote only reviewed results to the curated `reports/` tree. The source may lag, and missing dates are excluded rather than counted as losses; this command is on demand, not a scheduled background job.

To see each day's Top 3 hit/miss and cash flow for Poisson in October:

```powershell
py compare_models_2026.py --start 2026-10-01 --end 2026-10-31 --model poisson
```

The report includes the winning numbers and occurrence counts, daily cost, gross receipts, and net profit/loss. It also downloads the current feed each time; `--end` limits scored target dates even when the feed has newer draws.

To research whether a different Top-3 ranking or repeat filter improves hit rate without changing the forecast command, first generate daily walk-forward scores for a selection period, then compare them with the previously reviewed 2026 stress-test cache:

```powershell
py compare_models_2026.py --start 2025-01-01 --end 2025-12-31 --history-days 90 --top-k 3 --model all --output work\reports\repeat_selection_2025
py compare_models_2026.py --start 2026-01-01 --end 2026-10-31 --history-days 90 --top-k 3 --model all --output work\reports\repeat_stress_2026
py analyze_repeat_strategies.py --selection-cache work\reports\repeat_selection_2025\cache --stress-cache work\reports\repeat_stress_2026\cache --output work\reports\repeat_strategy_analysis
```

The research report compares Poisson, binary presence ranking, trailing 90-day frequency, and Poisson selections that exclude numbers chosen in the preceding one or three calendar days. It selects by hit rate on 2025 and reports 2026 only as a previously reviewed stress test; it does not change `main.py` or establish future profitability.

```powershell
# Evaluate the default LightGBM model and forecast the next date available in the data
py main.py

# Compare baseline and LightGBM on a 90-day chronological holdout
py main.py --model both --test-days 90 --top-k 5

# Forecast a specific date; the previous calendar day's result must be available
py main.py --model lightgbm --top-k 5 --forecast-date 2026-11-02

# Use the experimental regularized model
py main.py --model regularized --top-k 5 --forecast-date 2026-11-02

# Use a custom CSV source
py main.py --model baseline --url 'https://example.com/draws.csv'

# Show available options
py main.py --help
```

Available `--model` values are `baseline`, `lightgbm`, `regularized`, `poisson`, `both` (baseline + LightGBM), and `all` (baseline and all three machine-learning models). Other useful options include `--n-splits`, `--gap`, `--n-estimators`, `--n-jobs`, `--timeout`, and `--inspect`.

### Paper journal: daily run times

The paper journal is the only workflow in this repository with a defined daily schedule. It is simulation only; it does not place bets or send predictions externally.

Initialize once, preferably before **18:00 UTC+7**:

```powershell
.\run_paper.ps1 init
```

The trial lasts 90 calendar days. Initialization before 18:00 starts the trial today; initialization after 18:00 starts it tomorrow. Re-running `init` preserves the original trial.

Run the daily journal:

```powershell
# Forecast checkpoint: run by 18:00 UTC+7
.\run_paper.ps1 run

# Settlement/report checkpoint: run after 19:00 UTC+7 (scheduled at 19:30)
.\run_paper.ps1 run
```

The first run records a forecast if today's draw is not yet in the source. The later run settles eligible earlier forecasts once the source contains all 27 results. If a day is missed or results are not yet available, run again later; the journal will not backdate a forecast.

Inspect or verify the journal at any time:

```powershell
.\run_paper.ps1 status
.\run_paper.ps1 verify
```

To install the Windows scheduled task, initialize the journal first, then run:

```powershell
.\install_paper_schedule.ps1
Get-ScheduledTask -TaskName 'XSMB-Paper-90Day-Trial'
```

The task runs every day at **17:00 and 19:30 UTC+7** until the trial end date. The computer must be awake and the same Windows account must be logged in. The schedule requires a Windows timezone set to UTC+07:00 without daylight-saving time. To pause or resume it:

```powershell
Disable-ScheduledTask -TaskName 'XSMB-Paper-90Day-Trial'
Enable-ScheduledTask -TaskName 'XSMB-Paper-90Day-Trial'
```

The local `run_paper.ps1` wrapper uses machine-specific Python/dependency paths. On another machine, configure those paths first, or use the Python entry point directly:

```powershell
py -m scripts.operations.paper_trade init
py -m scripts.operations.paper_trade run
py -m scripts.operations.paper_trade status
py -m scripts.operations.paper_trade verify
```

### Research and historical backtests

These are on-demand, potentially lengthy experiments, not daily commands. Choose a new output directory to avoid replacing an earlier run.

```powershell
# Full research workflow
py research.py --output work/reports/research_run_01

# Period-based backtest over a chosen historical date range
py -m scripts.backtests.backtest_periods --start 2026-01-01 --end 2026-10-29 --period-days 90 --workers 2 --output work/reports/backtest_run_01

# Daily-refit regularized-model backtest
py backtest_regularized.py --year 2025 --end 2025-12-31 --top-k 10 --workers 2 --output work/reports/regularized_run_01

# Train and compare ordered-prize models; choose the forecast date explicitly
py -m scripts.operations.train_prize_model --forecast-date 2026-11-02 --top-k 5 --output work/reports/prize_model_run_01

# Run signal diagnostics against a local CSV source
py -m scripts.operations.diagnose_signal --source data/processed/xsmb_prizes_2026-10-30.csv --output work/reports/signal_diagnostics_run_01 --permutations 499

# Test pairwise co-occurrence against a fair independent-draw null
py analyze_pair_dependence.py --source data/processed/xsmb_prizes_2026-10-30.csv --output work/reports/pair_dependence_research_run_01

# Research Top-3 predictions over three-digit endings (000-999)
py -m scripts.research.research_top3_3digit --source reports/prize_model_2026-10-30/source_snapshot.csv --output work/reports/top3_3digit_research_run_01

# Restrict evaluation to the May-July 2026 draw dates
py -m scripts.research.research_top3_3digit --source reports/prize_model_2026-10-30/source_snapshot.csv --start-date 2026-05-01 --end-date 2026-07-31 --output work/reports/top3_3digit_research_2026_may_july

# Evaluate August-October 2026 and apply the 23-to-900 per-occurrence payout
py -m scripts.research.research_top3_3digit --source reports\prize_model_2026-10-30\source_snapshot.csv --start-date 2026-08-01 --end-date 2026-10-31 --output work\reports\top3_3digit_research_2026_aug_oct

# Historical two-digit-target research (not the three-digit experiment)
py -m scripts.research.research_top3_excluding_prize4 --source reports/prize_model_2026-10-30/source_snapshot.csv --output work/reports/top3_excluding_prize4_research_run_01
```

`analyze_pair_dependence.py` tests all 4,950 number pairs on a fixed 2005–2022 discovery and 2023+ replication split. It uses exact binomial tests with Holm correction separately in each period. Its findings are retrospective diagnostics, not proof of a forecast edge; the command does not change production defaults.

`research_top3_3digit.py` evaluates Top-3 predictions among 1,000 three-digit endings (000–999). It uses the last three digits of each prize result except the four two-digit Giải bảy results. Giải tư and Giải sáu results remain eligible, including leading-zero three-digit endings such as 007. Thus there are 23 eligible outcomes per draw, with no artificial 0xx conversions from two-digit results. It compares LightGBM Poisson, binary-presence, and recent-frequency rankings with chronological quarterly refits. It reports hit rates, repeats, miss streaks, and uncertainty. Profit is computed using the configured per-selection payout assumption. This is separate from the earlier two-digit-target experiment in `research_top3_excluding_prize4.py`; the latter does not answer a three-digit prediction request. Earlier three-digit reports with a different target definition are retained for audit and superseded by reruns using this corrected target definition.

Both experiments are isolated from the default forecast command and refuse to overwrite an existing report directory.

`audit_experiment.py` is a fixed, retrospective audit that reads a particular existing snapshot under `reports/` and writes to its preset audit directory. Run it only when intentionally reproducing that audit:

```powershell
py audit_experiment.py
```

### Tests

Run the full unittest suite after code changes:

```powershell
py -m unittest discover -s tests -v
```

## Data handling

The project expects daily CSV data with either:
- a complete 00..99 frequency schema, or
- raw prize-value columns that are parsed into the last two digits of each prize result

The data loader checks for:
- valid dates
- duplicate dates
- missing or malformed rows
- daily total counts of exactly 27 result entries
- consistent 100-number representation

Missing calendar dates are preserved as gaps instead of being imputed.

## Output and reporting

Disposable commands generate runs under `work/reports/`. The curated `reports/` directory keeps only reviewed outputs, organized by purpose:

- `reports/current/` for the current model comparisons
- `reports/research/` for selected long-lived experiments
- `reports/audits/` for integrity and reproducibility checks
- `reports/paper_trial/` and the retained prize-model snapshot for operational state

See `reports/README.md` for the report index. Generated outputs can include:
- metrics summaries
- daily evaluation files
- forecast snapshots
- comparative ranking outputs
- paper-trial or audit-style experiment notes

The companion research and paper-trading documents explain how results should be interpreted and how trials are structured over time.

## Important notes

- This project is research-focused and experimental.
- Results are not guarantees of future performance.
- Forecast probabilities are model estimates and may require calibration before being treated as trading odds.
- The project is intentionally designed around historical validation and transparent methodology, not black-box automation.
- The code is organized to preserve causal timing rules: features are built using only information available up to the forecast date.

## Research docs

For deeper context, read:
- `RESEARCH.md`
- `PAPER_TRADING.md`

These documents explain the study protocol, validation assumptions, payout logic, and the rules for how experiments and paper-trading simulations are conducted.

## Disclaimer

This repository is for analytical and academic use. It does not provide financial advice, guarantee returns, or automate real-world betting or market transactions.
