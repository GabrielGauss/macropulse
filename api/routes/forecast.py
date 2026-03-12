"""
5-day ahead regime probability forecast endpoint.

GET /v1/forecast
    Returns ARIMA(1,0,1)-based forecasts of each macro regime probability
    and the risk score for the next `horizon` business days (max 10).
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas.responses import ForecastResponse, ForecastRow
from database import queries
from services.forecaster import forecast_regime_probabilities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Forecast"])

# Minimum number of historical rows required to run the forecaster.
_MIN_HISTORY_ROWS = 10


@router.get("/forecast", response_model=ForecastResponse, summary="5-day regime forecast")
def get_forecast(
    horizon: int = Query(
        default=5,
        ge=1,
        le=10,
        description="Number of business days ahead to forecast (1–10).",
    ),
) -> ForecastResponse:
    """
    Return an ARIMA-based forward projection of macro regime probabilities.

    Fetches the last 60 trading days of regime history from the database,
    fits a separate ARIMA(1,0,1) on each probability column and on the
    risk score, and returns a day-by-day forecast for the requested horizon.

    Probabilities are clipped to [0, 1] and renormalised to sum to 1.
    The `confidence` field (0–1) reflects agreement across the probability
    forecasts: a highly concentrated forecast yields higher confidence.
    """
    logger.info("GET /v1/forecast horizon=%d", horizon)

    # ── Fetch history ─────────────────────────────────────────────────
    raw_history = queries.fetch_regime_history(limit=60)
    if not raw_history:
        raise HTTPException(
            status_code=503,
            detail="No regime history available. Run the daily pipeline first.",
        )

    if len(raw_history) < _MIN_HISTORY_ROWS:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Insufficient history: need at least {_MIN_HISTORY_ROWS} rows, "
                f"found {len(raw_history)}. Run the pipeline for more days."
            ),
        )

    # DB returns rows newest-first; reverse so index is chronological.
    raw_history = list(reversed(raw_history))

    history_df = pd.DataFrame(raw_history)
    history_df = history_df.set_index("time").sort_index()

    required_cols = [
        "prob_expansion",
        "prob_tightening",
        "prob_risk_off",
        "prob_recovery",
        "risk_score",
    ]
    missing = [c for c in required_cols if c not in history_df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Regime history is missing columns: {missing}",
        )

    # ── Run forecaster ────────────────────────────────────────────────
    try:
        forecast_rows = forecast_regime_probabilities(history_df, horizon=horizon)
    except Exception as exc:
        logger.exception("Forecaster failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Forecast computation failed: {exc}") from exc

    # ── Build response ────────────────────────────────────────────────
    rows = [ForecastRow(**row) for row in forecast_rows]

    return ForecastResponse(
        horizon=horizon,
        generated_at=dt.datetime.now(dt.timezone.utc),
        forecast=rows,
    )
