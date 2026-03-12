"""
Backtest API endpoint.

Allows users to trigger a historical regime replay and retrieve results.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Backtest"])


class BacktestRequest(BaseModel):
    start: dt.date | None = None
    end: dt.date | None = None
    model_version: str | None = None
    window: int | None = None


class BacktestSummary(BaseModel):
    total_days: int
    transitions: int
    avg_persistence_days: float
    regime_distribution: dict[str, float]
    mean_risk_score: float
    risk_score_std: float
    model_version: str | None = None


class BacktestResponse(BaseModel):
    summary: BacktestSummary
    timeline: list[dict[str, Any]]


@router.post("/backtest", response_model=BacktestResponse)
def run_backtest_endpoint(
    req: BacktestRequest,
    _key: str = Depends(require_api_key),
) -> BacktestResponse:
    """Run a historical regime backtest."""
    try:
        from config.settings import get_settings
        from data.ingestion.fred_client import fetch_all_fred
        from data.ingestion.market_client import fetch_market_data
        from data.processing.feature_engineering import build_features
        from services.backtest import run_backtest

        settings = get_settings()
        end = req.end or dt.date.today()
        start = req.start or (end - dt.timedelta(days=settings.data_lookback_days))

        fred_df = fetch_all_fred(start=start, end=end)
        market_df = fetch_market_data(start=start, end=end)
        features = build_features(fred_df, market_df)

        result = run_backtest(
            features,
            model_version=req.model_version,
            window=req.window,
        )

        timeline = [
            {"date": d, "regime": r, "risk_score": s, "probabilities": p}
            for d, r, s, p in zip(
                result.dates, result.regimes, result.risk_scores, result.probabilities
            )
        ]

        return BacktestResponse(
            summary=BacktestSummary(**result.summary),
            timeline=timeline,
        )
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")
