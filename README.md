# Trader Lifecycle & Capital Allocation Engine

A data science pipeline that models how retail investors evolve over time — from casual depositors to high-value active traders. The core inversion from standard churn modeling: instead of asking *who leaves*, we ask *who becomes valuable, and what drives that transition*.

---

## What it does

Ingests portfolio behavior, engineers financially-meaningful features, and fits a sequence of models:

| Stage | Model | Output |
|---|---|---|
| Segmentation | Gaussian Mixture Model | Archetype probabilities per user |
| Survival | Kaplan-Meier + Cox PH | Churn risk at 30 / 90 days |
| Lifecycle | Markov chain + multinomial logit | Monthly transition probabilities |
| LTV | LightGBM + discounted cash flow | 12 / 24-month revenue forecast |
| Uplift | T-learner (LightGBM) | CATE per intervention type |

Predictions are served via a FastAPI endpoint and tracked in MLflow.

---

## Quickstart

```bash
# Install dependencies (requires Python 3.11+, uv, and libomp on macOS)
brew install libomp   # macOS only — required for LightGBM
uv sync

# Run pipeline stages
make ingest     # generate synthetic data + pull market data
make features   # engineer feature matrix
make train      # fit all models
make serve      # start FastAPI server

make test       # run test suite
```

---

## Configuration

Copy `.env.example` to `.env` and override any defaults:

```bash
cp .env.example .env
```

All paths, hyperparameters, and financial constants are set in `src/config.py` via `pydantic-settings`. The defaults work out of the box — nothing in `.env` is required to run the pipeline.

Key constants:

| Key | Default | Meaning |
|---|---|---|
| `ANNUAL_MGMT_FEE` | `0.0025` | 25 bps AUM fee used in LTV calculation |
| `MONTHLY_DISCOUNT_RATE` | `0.005` | DCF discount rate (0.5%/month) |
| `RISK_FREE_ANNUAL_RATE` | `0.045` | T-bill proxy for Sharpe/Sortino |
| `N_USERS` | `5000` | Synthetic cohort size |
| `N_DAYS` | `730` | Simulation window (2 years) |

---

## Project layout

```
src/
├── config.py           # pydantic-settings, all constants and paths
├── ingestion/          # synthetic data generation + market data pull
├── features/           # HHI, Sharpe, drawdown, behavioral signals
├── models/             # GMM, Cox PH, Markov, LightGBM LTV, uplift
├── evaluation/         # metrics + SHAP explainability
└── serving/            # FastAPI scoring endpoint
data/
├── raw/                # immutable source files (gitignored)
├── processed/          # cleaned and joined outputs (gitignored)
└── features/           # feature matrix parquet (gitignored)
models/artifacts/       # serialized model files (gitignored)
```

---

## Build status

| Step | Status |
|---|---|
| 1 — Project init | done |
| 2 — Synthetic data | done |
| 3 — Market data | done |
| 4 — Feature engineering | done |
| 4 — Feature engineering | - |
| 5 — Segmentation (GMM) | - |
| 6 — Survival analysis | - |
| 7 — Markov transitions | - |
| 8 — LTV forecasting | - |
| 9 — Uplift modeling | - |
| 10 — Evaluation + SHAP | - |
| 11 — Serving (FastAPI) | - |
| 12 — MLflow tracking | - |
