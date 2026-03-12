"""
FastAPI route definitions for MacroPulse v1 endpoints.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.responses import (
    DriftResponse,
    DriftRow,
    FactorRow,
    FactorsResponse,
    LiquidityResponse,
    LiquidityRow,
    RegimeProbabilities,
    RegimeResponse,
)
from database import queries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["MacroPulse"])


@router.get("/regime/current", response_model=RegimeResponse)
def get_current_regime() -> RegimeResponse:
    """Return the most recent macro regime signal."""
    row = queries.fetch_current_regime()
    if row is None:
        raise HTTPException(status_code=404, detail="No regime data available.")
    return RegimeResponse(
        timestamp=row["time"],
        macro_regime=row["regime"],
        risk_score=row["risk_score"],
        probabilities=RegimeProbabilities(
            expansion=row["prob_expansion"],
            tightening=row["prob_tightening"],
            risk_off=row["prob_risk_off"],
            recovery=row["prob_recovery"],
        ),
        volatility_state=row.get("volatility_state"),
        model_version=row.get("model_version"),
    )


@router.get("/regime/history", response_model=list[RegimeResponse])
def get_regime_history(
    start: Optional[dt.date] = Query(None, description="Start date (inclusive)"),
    end: Optional[dt.date] = Query(None, description="End date (inclusive)"),
    limit: int = Query(90, ge=1, le=1000),
) -> list[RegimeResponse]:
    """Return historical regime signals."""
    rows = queries.fetch_regime_history(start=start, end=end, limit=limit)
    return [
        RegimeResponse(
            timestamp=r["time"],
            macro_regime=r["regime"],
            risk_score=r["risk_score"],
            probabilities=RegimeProbabilities(
                expansion=r["prob_expansion"],
                tightening=r["prob_tightening"],
                risk_off=r["prob_risk_off"],
                recovery=r["prob_recovery"],
            ),
            volatility_state=r.get("volatility_state"),
            model_version=r.get("model_version"),
        )
        for r in rows
    ]


@router.get("/liquidity", response_model=LiquidityResponse)
def get_liquidity(
    limit: int = Query(30, ge=1, le=500),
) -> LiquidityResponse:
    """Return recent net liquidity values."""
    rows = queries.fetch_latest_liquidity(limit=limit)
    return LiquidityResponse(
        data=[LiquidityRow(**r) for r in rows]
    )


@router.get("/factors", response_model=FactorsResponse)
def get_factors(
    limit: int = Query(30, ge=1, le=500),
) -> FactorsResponse:
    """Return recent PCA latent factors."""
    rows = queries.fetch_latest_factors(limit=limit)
    return FactorsResponse(
        data=[FactorRow(**r) for r in rows]
    )


@router.get("/drift", response_model=DriftResponse)
def get_drift(
    limit: int = Query(30, ge=1, le=500),
) -> DriftResponse:
    """Return recent model drift metrics."""
    rows = queries.fetch_latest_drift(limit=limit)
    return DriftResponse(
        data=[DriftRow(**r) for r in rows]
    )
