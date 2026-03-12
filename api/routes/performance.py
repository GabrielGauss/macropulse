"""
Performance attribution endpoint for MacroPulse.

GET /v1/performance?lookback=252

Returns regime-level return statistics and a regime-following strategy
equity curve vs buy-and-hold S&P 500.  This is the primary proof-of-value
endpoint — it answers "does this model actually generate alpha?".
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from services.performance import compute_regime_performance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Performance"])


@router.get(
    "/performance",
    summary="Regime performance attribution",
    description=(
        "Computes return statistics per macro regime and a regime-following "
        "strategy equity curve vs buy-and-hold S&P 500.  "
        "Use `lookback` to control the number of trading days in the sample "
        "(default 252 ≈ 1 year, max 756 ≈ 3 years)."
    ),
)
def get_performance(
    lookback: int = Query(
        default=252,
        ge=30,
        le=756,
        description="Number of trading days to include (30–756).",
    ),
) -> JSONResponse:
    result = compute_regime_performance(lookback_days=lookback)
    if "error" in result:
        return JSONResponse(status_code=503, content=result)
    return JSONResponse(content=result)
