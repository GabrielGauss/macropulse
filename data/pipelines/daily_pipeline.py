"""
MacroPulse daily pipeline.

Orchestrates the end-to-end flow:
  1. Fetch FRED + market data
  2. Validate raw data
  3. Compute engineered features
  4. Validate features
  5. Store features → TimescaleDB
  6. Data-lag guard
  7. Load frozen models → PCA transform
  8. HMM inference → regime probabilities
  9. Store regime results → TimescaleDB
  10. Detect regime change → fire alerts
  11. Compute & store drift metrics
  12. Broadcast via WebSocket
  13. Log pipeline run
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from data.ingestion.fred_client import fetch_all_fred
from data.ingestion.market_client import fetch_market_data
from data.processing.feature_engineering import (
    MODEL_FEATURE_COLS,
    MODEL_FEATURE_COLS_V1,
    build_features,
)
from database import queries
from models.garch_model import GARCHModel
from models.hmm_model import HMMModel
from models.pca_model import PCAModel
from models.regime_classifier import RegimeClassifier
from services.alerting import alert_drift_warning, alert_regime_change
from services.drift_monitor import (
    compute_feature_shift,
    compute_pca_variance_drift,
    compute_regime_persistence,
)
from services.validation import validate_features, validate_market_data, validate_raw_fred

logger = logging.getLogger(__name__)

# Drift thresholds
_DRIFT_VARIANCE_WARN = 0.10
_DRIFT_PERSISTENCE_WARN = 0.97
_DRIFT_FEATURE_SHIFT_WARN = 1.5


def _log_run(
    status: str,
    data_lag: bool,
    duration: float,
    error: str | None = None,
    model_version: str | None = None,
) -> None:
    queries.insert_pipeline_run(
        {
            "run_ts": dt.datetime.now(dt.timezone.utc),
            "status": status,
            "data_lag": data_lag,
            "duration_sec": round(duration, 2),
            "error_message": error,
            "model_version": model_version,
        }
    )


def run_daily_pipeline(
    target_date: dt.date | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """
    Execute the daily macro regime pipeline.

    Parameters
    ----------
    target_date   : override for the pipeline date (default: today).
    model_version : which frozen model artifacts to use.

    Returns
    -------
    dict – The regime output for the target date, or a partial result
           if data lag is detected.
    """
    settings = get_settings()
    version = model_version or settings.default_model_version
    today = target_date or dt.date.today()
    start = today - dt.timedelta(days=settings.data_lookback_days)
    t0 = time.monotonic()
    data_lag = False

    logger.info("═══ MacroPulse pipeline start (%s) ═══", today)

    # ── 1. Fetch data ────────────────────────────────────────────
    try:
        fred_df = fetch_all_fred(start=start, end=today)
        market_df = fetch_market_data(start=start, end=today)
    except Exception as exc:
        duration = time.monotonic() - t0
        logger.error("Data fetch failed: %s", exc)
        _log_run("failed", data_lag=False, duration=duration, error=str(exc), model_version=version)
        raise

    # ── 2. Validate raw data ─────────────────────────────────────
    fred_report = validate_raw_fred(fred_df)
    market_report = validate_market_data(market_df)

    if not fred_report.passed or not market_report.passed:
        errors = fred_report.errors + market_report.errors
        duration = time.monotonic() - t0
        msg = f"Raw data validation failed: {errors}"
        _log_run("failed", data_lag=False, duration=duration, error=msg, model_version=version)
        raise RuntimeError(msg)

    # ── 3. Feature engineering ───────────────────────────────────
    features = build_features(fred_df, market_df)

    # ── 4. Validate features ─────────────────────────────────────
    feat_report = validate_features(features)
    if not feat_report.passed:
        duration = time.monotonic() - t0
        msg = f"Feature validation failed: {feat_report.errors}"
        _log_run("failed", data_lag=False, duration=duration, error=msg, model_version=version)
        raise RuntimeError(msg)

    # ── 5. Store features ────────────────────────────────────────
    latest_row = features.iloc[-1]
    queries.upsert_macro_features(
        {
            "time": features.index[-1].isoformat(),
            **{col: float(latest_row[col]) if pd.notna(latest_row[col]) else None for col in features.columns},
        }
    )

    # ── 6. Data-lag guard ────────────────────────────────────────
    latest_fred_date = fred_df.index.max().date() if len(fred_df) else None
    if latest_fred_date and (today - latest_fred_date).days > 3:
        data_lag = True
        logger.warning("FRED data lag: latest=%s, target=%s", latest_fred_date, today)
        duration = time.monotonic() - t0
        _log_run("partial", data_lag=True, duration=duration, model_version=version)
        return {"status": "data_lag", "timestamp": today.isoformat()}

    # ── 7. Load frozen models + PCA ──────────────────────────────
    pca_model = PCAModel.load(version)
    hmm_model = HMMModel.load(version)
    classifier = RegimeClassifier.load(version)

    # v1 artifacts were trained on 8 features; v2 uses all 10.
    feature_cols = MODEL_FEATURE_COLS_V1 if version == "v1" else MODEL_FEATURE_COLS
    X = features[feature_cols].values
    factors = pca_model.transform(X)

    # ── 7a. GARCH volatility state (replaces simple VIX threshold) ────
    garch_vol_state: str | None = None
    try:
        garch_model = GARCHModel.load(series_name="d_sp500", version=version)
        cond_vol = garch_model.forecast_vol(features["d_sp500"])
        garch_vol_state = garch_model.classify_vol_state(cond_vol)
        logger.info("GARCH vol state: %s (cond_vol=%.4f)", garch_vol_state, cond_vol)
    except Exception as exc:
        logger.warning(
            "GARCH model not found for version=%s; falling back to VIX threshold. (%s)",
            version, exc,
        )

    # Store latest factor row
    lf = factors[-1]
    queries.upsert_macro_factors(
        {
            "time": features.index[-1].isoformat(),
            "factor_1": float(lf[0]),
            "factor_2": float(lf[1]),
            "factor_3": float(lf[2]) if len(lf) > 2 else None,
            "factor_4": float(lf[3]) if len(lf) > 3 else None,
            "model_version": version,
        }
    )

    # ── 8. HMM inference ─────────────────────────────────────────
    state_probs = hmm_model.predict_proba(factors)
    latest_probs = state_probs[-1]

    # Use GARCH vol state when available; fall back to VIX-diff threshold.
    if garch_vol_state is not None:
        # Override the volatility_state returned by classify() with the
        # GARCH-derived classification.
        result = classifier.classify(latest_probs, vix_diff=None)
        result["volatility_state"] = garch_vol_state
    else:
        vix_diff = float(latest_row["d_vix"]) if pd.notna(latest_row.get("d_vix")) else None
        result = classifier.classify(latest_probs, vix_diff=vix_diff)

    # ── 9. Store regime ──────────────────────────────────────────
    ts_iso = features.index[-1].isoformat()
    regime_row = {
        "time": ts_iso,
        "regime": result["regime"],
        "prob_expansion": result["probabilities"].get("expansion", 0),
        "prob_tightening": result["probabilities"].get("tightening", 0),
        "prob_risk_off": result["probabilities"].get("risk_off", 0),
        "prob_recovery": result["probabilities"].get("recovery", 0),
        "risk_score": result["risk_score"],
        "volatility_state": result["volatility_state"],
        "model_version": version,
    }
    queries.upsert_macro_regime(regime_row)

    # ── 10. Regime change detection + alerting ───────────────────
    prev_regime_row = queries.fetch_regime_history(limit=2)
    if len(prev_regime_row) >= 2:
        prev_regime = prev_regime_row[1]["regime"]
        if prev_regime != result["regime"]:
            alert_regime_change(
                previous=prev_regime,
                current=result["regime"],
                risk_score=result["risk_score"],
                probabilities=result["probabilities"],
                timestamp=ts_iso,
            )

    # ── 11. Drift metrics ────────────────────────────────────────
    # X is already filtered to the correct feature_cols above.
    pca_drift = compute_pca_variance_drift(pca_model, X[-60:])
    regimes_seq = hmm_model.predict(factors[-60:])
    persistence = compute_regime_persistence(regimes_seq)
    mean_shift, std_shift = compute_feature_shift(X[:-60], X[-60:])

    queries.upsert_drift_metrics(
        {
            "time": ts_iso,
            "pca_explained_variance": float(pca_drift),
            "regime_persistence": float(persistence),
            "feature_mean_shift": float(mean_shift),
            "feature_std_shift": float(std_shift),
            "model_version": version,
        }
    )

    # Fire drift alerts if thresholds exceeded
    if pca_drift > _DRIFT_VARIANCE_WARN:
        alert_drift_warning("pca_variance_drift", pca_drift, _DRIFT_VARIANCE_WARN, ts_iso)
    if persistence > _DRIFT_PERSISTENCE_WARN:
        alert_drift_warning("regime_persistence", persistence, _DRIFT_PERSISTENCE_WARN, ts_iso)
    if mean_shift > _DRIFT_FEATURE_SHIFT_WARN:
        alert_drift_warning("feature_mean_shift", mean_shift, _DRIFT_FEATURE_SHIFT_WARN, ts_iso)

    # ── 12. WebSocket broadcast ──────────────────────────────────
    output = {
        "timestamp": ts_iso,
        "macro_regime": result["regime"],
        "risk_score": result["risk_score"],
        "probabilities": result["probabilities"],
        "volatility_state": result["volatility_state"],
        "model_version": version,
    }
    try:
        from api.routes.websocket import notify_regime_update
        notify_regime_update(output)
    except Exception:
        pass  # CLI mode — no event loop

    # ── 13. Log success ──────────────────────────────────────────
    duration = time.monotonic() - t0
    _log_run("success", data_lag=False, duration=duration, model_version=version)

    logger.info("═══ Pipeline complete in %.1fs ═══  %s", duration, output)
    return output
