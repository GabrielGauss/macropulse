"""
Composite macro analysis endpoint.

GET /v1/analysis/composite
    Runs the rule-based multi-domain analyst orchestrator on the
    latest DB data and returns a structured composite view.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, HTTPException

from api.schemas.responses import CompositeAnalysisResponse, DomainSignal
from database import queries
from services.orchestrator import composite_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/analysis", tags=["Analysis"])


@router.get(
    "/composite",
    response_model=CompositeAnalysisResponse,
    summary="Multi-domain composite macro analysis",
)
def get_composite_analysis() -> CompositeAnalysisResponse:
    """
    Return a rule-based composite macro analysis across four domains:
    equity, rates, credit, and liquidity.

    Data is fetched from the database at request time.  No LLM or
    external calls are made — all analysis is deterministic.

    The response includes:
    - **composite_signal**: overall risk disposition ("risk_on" / "neutral" / "risk_off")
    - **composite_score**: numeric score from -100 (full risk-off) to +100 (full risk-on)
    - **conviction**: analyst agreement ("high" / "medium" / "low")
    - **domain_signals**: individual analyst outputs for equity, rates, credit, liquidity
    - **regime_alignment**: whether the composite agrees with the current HMM regime
    """
    logger.info("GET /v1/analysis/composite")

    # ── Fetch data ────────────────────────────────────────────────────
    regime_row = queries.fetch_current_regime()
    if regime_row is None:
        raise HTTPException(
            status_code=503,
            detail="No regime data available. Run the daily pipeline first.",
        )

    history = queries.fetch_regime_history(limit=60)
    features = queries.fetch_latest_liquidity(limit=60)  # includes d_liquidity
    # For rates and credit analysis we need d_yield_curve, d_10y, d_hy_spread.
    # fetch_latest_liquidity returns macro_features rows which contain all feature cols.
    liquidity = queries.fetch_latest_liquidity(limit=30)

    # ── Run orchestrator ──────────────────────────────────────────────
    try:
        result = composite_analysis(
            regime_row=regime_row,
            history=history,
            features=features,
            liquidity=liquidity,
        )
    except Exception as exc:
        logger.exception("Composite analysis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis computation failed: {exc}",
        ) from exc

    # ── Serialise domain signals ──────────────────────────────────────
    domain_signals = {
        name: DomainSignal(**sig)
        for name, sig in result["domain_signals"].items()
    }

    return CompositeAnalysisResponse(
        generated_at=dt.datetime.now(dt.timezone.utc),
        composite_signal=result["composite_signal"],
        composite_score=result["composite_score"],
        conviction=result["conviction"],
        regime_alignment=result["regime_alignment"],
        domain_signals=domain_signals,
    )
