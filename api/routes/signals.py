"""
Unified signal endpoints for MacroPulse.

Implements the /v1/signals surface defined in the Master Build & Launch Manual §5.2:
  GET /v1/signals/latest          — most recent signal package
  GET /v1/signals/{date}          — historical lookup by date (YYYY-MM-DD)
  GET /v1/signals/range           — time series (query: start, end; max 365 days)
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.auth import require_api_key
from api.schemas.responses import SignalPackageResponse
from services.signals import build_signal_package, build_signal_range

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/signals", tags=["Signals"])


@router.get(
    "/latest",
    response_model=SignalPackageResponse,
    summary="Latest unified signal package",
)
def get_latest_signal(
    _key: str = Depends(require_api_key),
) -> SignalPackageResponse:
    """
    Return the most recent unified macro signal package.

    Combines HMM regime, net liquidity state, PCA factors, and model metadata
    into a single response.  This is the primary integration endpoint for clients.
    """
    pkg = build_signal_package()
    if pkg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No regime data available. Run the daily pipeline first.",
        )
    return SignalPackageResponse(**pkg)


@router.get(
    "/range",
    response_model=list[SignalPackageResponse],
    summary="Signal time series over a date range",
)
def get_signal_range(
    start: dt.date = Query(..., description="Start date (YYYY-MM-DD)"),
    end: dt.date = Query(..., description="End date (YYYY-MM-DD)"),
    _key: str = Depends(require_api_key),
) -> list[SignalPackageResponse]:
    """
    Return signal packages for every trading day in [start, end].

    Capped at 365 days.  Returns an empty list if no data exists for the range.
    """
    if end < start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end must be >= start.",
        )
    results = build_signal_range(start=start, end=end)
    return [SignalPackageResponse(**r) for r in results]


@router.get(
    "/{date}",
    response_model=SignalPackageResponse,
    summary="Signal package for a specific date",
)
def get_signal_by_date(
    date: dt.date,
    _key: str = Depends(require_api_key),
) -> SignalPackageResponse:
    """
    Return the signal package for a specific date (YYYY-MM-DD).

    Returns 404 if no regime data exists for that date.
    """
    pkg = build_signal_package(target_date=date)
    if pkg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No signal data found for {date}.",
        )
    return SignalPackageResponse(**pkg)
