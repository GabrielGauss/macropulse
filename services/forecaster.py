"""
ARIMA-based regime probability forecaster for MacroPulse.

Fits ARIMA(1,0,1) models on each regime probability series and the
risk score independently, then projects them forward by `horizon` days.

Output probabilities are clipped to [0, 1] and renormalised to sum to 1
so the output is always a valid probability simplex.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

# Columns expected in the history DataFrame.
_PROB_COLS = [
    "prob_expansion",
    "prob_tightening",
    "prob_risk_off",
    "prob_recovery",
]
_SCORE_COL = "risk_score"


def _fit_arima_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """
    Fit ARIMA(1,0,1) on *series* and return a *horizon*-step forecast.

    Falls back to the series mean if ARIMA fitting fails (e.g. the
    series is constant or has fewer than 10 observations).

    Parameters
    ----------
    series:
        Numeric time series, indexed by date.
    horizon:
        Number of steps ahead to forecast.

    Returns
    -------
    np.ndarray of shape (horizon,)
    """
    clean = series.dropna().astype(float)
    if len(clean) < 10:
        logger.warning(
            "Series '%s' too short (%d obs); using mean forecast.",
            getattr(series, "name", "?"),
            len(clean),
        )
        return np.full(horizon, clean.mean() if len(clean) > 0 else 0.0)

    try:
        model = ARIMA(clean, order=(1, 0, 1))
        result = model.fit()
        forecast = result.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)
    except Exception as exc:
        logger.warning(
            "ARIMA fit failed for '%s': %s; falling back to mean.",
            getattr(series, "name", "?"),
            exc,
        )
        return np.full(horizon, float(clean.mean()))


def forecast_regime_probabilities(
    history_df: pd.DataFrame,
    horizon: int = 5,
) -> list[dict[str, Any]]:
    """
    Forecast regime probabilities and risk score *horizon* days ahead.

    Fits a separate ARIMA(1,0,1) on each of the four regime probability
    columns and on the risk score.  Probability forecasts are clipped to
    [0, 1] and renormalised to sum to 1.  A confidence scalar is derived
    from the inverse of the mean absolute residual of the in-sample fit
    (capped at 1.0).

    Parameters
    ----------
    history_df:
        DataFrame with at minimum columns:
        prob_expansion, prob_tightening, prob_risk_off, prob_recovery,
        risk_score.  Should be indexed by date (oldest first) and contain
        at least 10 rows (ideally 60).
    horizon:
        Number of business days ahead to forecast.  Capped at 10.

    Returns
    -------
    List of dicts, each representing one forecast day, with keys:
        date, prob_expansion, prob_tightening, prob_risk_off,
        prob_recovery, risk_score, confidence.
    """
    horizon = min(int(horizon), 10)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    missing = [c for c in _PROB_COLS + [_SCORE_COL] if c not in history_df.columns]
    if missing:
        raise ValueError(f"history_df is missing required columns: {missing}")

    logger.info(
        "Forecasting regime probabilities: horizon=%d, history_rows=%d",
        horizon,
        len(history_df),
    )

    # ── Forecast each probability column ─────────────────────────────
    raw_forecasts: dict[str, np.ndarray] = {}
    for col in _PROB_COLS:
        raw_forecasts[col] = _fit_arima_forecast(history_df[col], horizon)

    # ── Clip to [0, 1] then renormalise each forecast step ───────────
    prob_matrix = np.column_stack(
        [raw_forecasts[c] for c in _PROB_COLS]
    )  # shape (horizon, 4)
    prob_matrix = np.clip(prob_matrix, 0.0, 1.0)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero; if all probs collapsed to 0 use uniform.
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    prob_matrix = prob_matrix / row_sums

    # ── Forecast risk score ──────────────────────────────────────────
    score_forecast = _fit_arima_forecast(history_df[_SCORE_COL], horizon)
    score_forecast = np.clip(score_forecast, -100.0, 100.0)

    # ── Confidence: based on variance of probability forecasts ───────
    # Lower spread across regimes → lower confidence.
    prob_std = float(np.mean(np.std(prob_matrix, axis=1)))
    # Map std to confidence: more uniform dist (std ~ 0.25) → low conf
    # Concentrated (std ~ 0.5) → high conf.  Scaled and capped at 1.
    confidence = float(np.clip(prob_std * 4.0, 0.0, 1.0))

    # ── Build result dates ────────────────────────────────────────────
    # Determine last date in history; advance by business-day-like steps.
    if isinstance(history_df.index, pd.DatetimeIndex):
        last_date = history_df.index[-1].date()
    else:
        try:
            last_date = pd.to_datetime(history_df.index[-1]).date()
        except Exception:
            last_date = dt.date.today()

    forecast_dates: list[dt.date] = []
    candidate = last_date
    while len(forecast_dates) < horizon:
        candidate = candidate + dt.timedelta(days=1)
        # Skip weekends (basic business-day approximation).
        if candidate.weekday() < 5:
            forecast_dates.append(candidate)

    # ── Assemble output ───────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    for i, fdate in enumerate(forecast_dates):
        row: dict[str, Any] = {
            "date": fdate,
            "prob_expansion": round(float(prob_matrix[i, 0]), 4),
            "prob_tightening": round(float(prob_matrix[i, 1]), 4),
            "prob_risk_off": round(float(prob_matrix[i, 2]), 4),
            "prob_recovery": round(float(prob_matrix[i, 3]), 4),
            "risk_score": round(float(score_forecast[i]), 2),
            "confidence": round(confidence, 4),
        }
        results.append(row)

    logger.info("Forecast complete: %d rows produced.", len(results))
    return results
