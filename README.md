# MacroPulse

**Probabilistic macro regime intelligence platform.**

MacroPulse is a full-stack system that ingests macroeconomic and financial data, computes a Net Liquidity Proxy, extracts latent factors via PCA, and classifies the current macro regime using a Gaussian Hidden Markov Model. The result is a daily regime signal with full probability breakdowns, served through a REST API, streamed via WebSocket, and visualised in a real-time dashboard.

---

## Output

```json
{
  "timestamp": "2026-03-11",
  "macro_regime": "tightening",
  "risk_score": -34,
  "probabilities": {
    "expansion": 0.18,
    "tightening": 0.54,
    "risk_off": 0.21,
    "recovery": 0.07
  },
  "volatility_state": "normal",
  "model_version": "v1"
}
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data Sources         │  Pipeline              │  Output    │
│                       │                        │            │
│  FRED API ──────┐     │  Feature Engineering   │  REST API  │
│  Yahoo Finance ─┤     │  → PCA (4 factors)     │  WebSocket │
│                 ├────►│  → HMM (4 regimes)     │──► Dashboard│
│  (daily cron)   │     │  → Risk Score          │  Alerts    │
│                 │     │  → Drift Monitoring     │  DB Store  │
└─────────────────┴─────┴────────────────────────┴────────────┘
```

## Quick Start

### 1. Start infrastructure

```bash
docker compose up timescaledb -d
```

### 2. Configure

```bash
cp .env.example .env
# Add your FRED_API_KEY (free at https://fred.stlouisfed.org/docs/api/api_key.html)
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Train models (first time)

```bash
python scripts/retrain_models.py
```

### 5. Run pipeline

```bash
python scripts/run_daily_pipeline.py
```

### 6. Start API

```bash
uvicorn api.main:app --reload
```

### 7. Start dashboard (dev)

```bash
cd frontend && npm install && npm run dev
```

### Full production stack

```bash
cd frontend && npm run build && cd ..
docker compose up --build
```

This starts TimescaleDB, the API (with built-in scheduler), and nginx serving the frontend on port 80.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/regime/current` | Latest macro regime signal |
| `GET` | `/v1/regime/history` | Historical signals (date filters + limit) |
| `GET` | `/v1/liquidity` | Net liquidity proxy time series |
| `GET` | `/v1/factors` | PCA latent factors time series |
| `GET` | `/v1/drift` | Model drift metrics |
| `POST` | `/v1/backtest` | Run historical regime backtest |
| `WS` | `/ws/regime` | Real-time regime stream |
| `GET` | `/health` | Liveness probe |

Interactive docs: `http://localhost:8000/docs`

## Repository Structure

```
macropulse/
├── api/                        # FastAPI application
│   ├── main.py                 # Entry point with lifespan + scheduler
│   ├── auth.py                 # API key authentication
│   ├── routes/
│   │   ├── regime.py           # REST endpoints
│   │   ├── backtest.py         # Backtest endpoint
│   │   └── websocket.py        # Real-time WebSocket
│   └── schemas/
│       └── responses.py        # Pydantic models
├── data/
│   ├── ingestion/
│   │   ├── fred_client.py      # FRED API client
│   │   └── market_client.py    # Yahoo Finance client
│   ├── processing/
│   │   └── feature_engineering.py  # Stationary transforms
│   └── pipelines/
│       └── daily_pipeline.py   # Orchestrator (13 steps)
├── models/
│   ├── artifacts/              # Serialized models (gitignored)
│   ├── pca_model.py            # PCA + scaler wrapper
│   ├── hmm_model.py            # Gaussian HMM wrapper
│   └── regime_classifier.py    # State → label mapping
├── database/
│   ├── connection.py           # psycopg2 helpers
│   ├── schema.sql              # TimescaleDB DDL (6 tables)
│   └── queries.py              # Parameterised SQL (upserts + reads)
├── services/
│   ├── inference.py            # Stateless inference service
│   ├── drift_monitor.py        # PCA / persistence / distribution drift
│   ├── alerting.py             # Email + webhook notifications
│   ├── backtest.py             # Historical regime replay
│   ├── validation.py           # Data validation guards
│   └── scheduler.py            # APScheduler cron runner
├── config/
│   └── settings.py             # pydantic-settings (all env vars)
├── scripts/
│   ├── run_daily_pipeline.py   # CLI pipeline runner
│   ├── retrain_models.py       # Model training script
│   └── init_db.py              # Schema initialisation
├── frontend/                   # React dashboard (Vite + Tailwind)
│   ├── src/
│   │   ├── App.jsx             # Dashboard layout
│   │   ├── components/         # RegimeCard, Timeline, Charts, Drift
│   │   ├── hooks/              # useFetch, useRegimeSocket
│   │   └── lib/                # API client, utilities
│   └── public/
│       └── landing.html        # Marketing / landing page
├── nginx/
│   └── nginx.conf              # Reverse proxy config
├── Dockerfile                  # Python API image
├── docker-compose.yml          # Full stack (DB + API + nginx)
├── requirements.txt
├── pyproject.toml              # ruff + mypy config
└── README.md
```

## Key Design Decisions

**Frozen Model Pattern** — Models train once and serialize via joblib. Daily pipeline only runs inference. Deterministic, fast, auditable.

**13-Step Pipeline** — Fetch → validate → engineer → validate → store → lag-guard → PCA → HMM → store → alert → drift → broadcast → log. Each step is independently testable.

**Data Lag Guard** — FRED publishes irregularly. If data >3 days stale, pipeline logs `data_lag=true`, skips inference, API returns last confirmed state.

**Idempotent Writes** — `INSERT … ON CONFLICT DO UPDATE` everywhere. Pipeline is safe to re-run.

**Regime Change Alerts** — Email (SMTP) and webhook (Slack/Discord/Teams) notifications fire on macro regime transitions and drift threshold breaches.

**WebSocket Streaming** — Connected dashboards receive regime updates the instant the pipeline completes. Auto-reconnect with exponential backoff.

**API Key Auth** — Header (`X-API-Key`) or query param. Dev mode (no keys configured) bypasses auth for local development.

**Validation Layer** — Raw data and features are validated before model inference. Sanity checks on value ranges, NaN ratios, staleness, and z-score outliers.

## Configuration

All config via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `FRED_API_KEY` | — | FRED API key (required) |
| `DB_HOST` | localhost | TimescaleDB host |
| `DB_PORT` | 5432 | TimescaleDB port |
| `DB_USER` | macropulse | Database user |
| `DB_PASSWORD` | macropulse | Database password |
| `API_KEYS` | [] | Comma-separated API keys (empty = dev mode) |
| `WEBHOOK_URL` | — | Slack/Discord webhook for alerts |
| `SMTP_HOST` | — | SMTP server for email alerts |
| `PIPELINE_CRON_HOUR` | 18 | Daily pipeline hour (UTC) |
| `PIPELINE_CRON_MINUTE` | 30 | Daily pipeline minute |
| `DEFAULT_MODEL_VERSION` | v1 | Model artifact version |

## Retraining

When drift metrics indicate degradation:

```bash
python scripts/retrain_models.py --version v2
# Update DEFAULT_MODEL_VERSION=v2 in .env
```

## License

Proprietary. Internal use only.
