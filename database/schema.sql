-- ============================================================
-- MacroPulse – TimescaleDB Schema
-- ============================================================
-- Run this file against a PostgreSQL database with the
-- TimescaleDB extension installed:
--
--   psql -U macropulse -d macropulse -f database/schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Pipeline metadata ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    status          TEXT NOT NULL,          -- success | partial | failed
    data_lag        BOOLEAN NOT NULL DEFAULT FALSE,
    duration_sec    DOUBLE PRECISION,
    error_message   TEXT,
    model_version   TEXT
);

-- ── Model version registry ──────────────────────────────────

CREATE TABLE IF NOT EXISTS model_versions (
    version         TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    pca_variance    DOUBLE PRECISION,
    hmm_n_regimes   INTEGER,
    notes           TEXT
);

-- ── Macro features (stationary series) ──────────────────────

CREATE TABLE IF NOT EXISTS macro_features (
    time            TIMESTAMPTZ NOT NULL,
    net_liquidity   DOUBLE PRECISION,
    d_liquidity     DOUBLE PRECISION,
    d_sp500         DOUBLE PRECISION,
    d_vix           DOUBLE PRECISION,
    d_dxy           DOUBLE PRECISION,
    d_hy_spread     DOUBLE PRECISION,
    d_yield_curve   DOUBLE PRECISION,
    d_10y           DOUBLE PRECISION,
    d_2y            DOUBLE PRECISION,
    PRIMARY KEY (time)
);

SELECT create_hypertable('macro_features', 'time', if_not_exists => TRUE);

-- ── PCA latent factors ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS macro_factors (
    time            TIMESTAMPTZ NOT NULL,
    factor_1        DOUBLE PRECISION,
    factor_2        DOUBLE PRECISION,
    factor_3        DOUBLE PRECISION,
    factor_4        DOUBLE PRECISION,
    model_version   TEXT,
    PRIMARY KEY (time)
);

SELECT create_hypertable('macro_factors', 'time', if_not_exists => TRUE);

-- ── Regime probabilities ────────────────────────────────────

CREATE TABLE IF NOT EXISTS macro_regimes (
    time              TIMESTAMPTZ NOT NULL,
    regime            TEXT,
    prob_expansion    DOUBLE PRECISION,
    prob_tightening   DOUBLE PRECISION,
    prob_risk_off     DOUBLE PRECISION,
    prob_recovery     DOUBLE PRECISION,
    risk_score        DOUBLE PRECISION,
    volatility_state  TEXT,
    model_version     TEXT,
    PRIMARY KEY (time)
);

SELECT create_hypertable('macro_regimes', 'time', if_not_exists => TRUE);

-- ── Drift monitoring ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_drift_metrics (
    time                    TIMESTAMPTZ NOT NULL,
    pca_explained_variance  DOUBLE PRECISION,
    regime_persistence      DOUBLE PRECISION,
    feature_mean_shift      DOUBLE PRECISION,
    feature_std_shift       DOUBLE PRECISION,
    model_version           TEXT,
    PRIMARY KEY (time)
);

SELECT create_hypertable('model_drift_metrics', 'time', if_not_exists => TRUE);

-- ── User management ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id                   BIGSERIAL PRIMARY KEY,
    email                TEXT NOT NULL UNIQUE,
    name                 TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    paddle_customer_id   TEXT,
    paddle_subscription_id TEXT
);

-- Tiers: free (50 req/day) | starter (500 req/day) | pro (unlimited)
CREATE TABLE IF NOT EXISTS api_keys (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash        TEXT NOT NULL UNIQUE,       -- SHA-256 of plaintext key
    key_prefix      TEXT NOT NULL,              -- first 12 chars for display
    tier            TEXT NOT NULL DEFAULT 'free',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at    TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash   ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user   ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;
