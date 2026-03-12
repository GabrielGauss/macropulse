"""
MacroPulse – FastAPI application entry point.

Run with:
    uvicorn api.main:app --reload

Serves:
  • REST API (regime, liquidity, factors, drift, backtest)
  • WebSocket stream (/ws/regime)
  • Built-in pipeline scheduler
  • Static frontend (production builds of the dashboard)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.middleware.rate_limit import RateLimitMiddleware
from api.routes.auth import router as auth_router
from api.routes.billing import router as billing_router
from api.routes.analysis import router as analysis_router
from api.routes.backtest import router as backtest_router
from api.routes.commentary import router as commentary_router
from api.routes.dashboard import router as dashboard_router
from api.routes.forecast import router as forecast_router
from api.routes.performance import router as performance_router
from api.routes.regime import router as regime_router
from api.routes.signals import router as signals_router
from api.routes.websocket import router as ws_router
from api.schemas.responses import HealthResponse
from config.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start / stop the background scheduler with the app lifecycle."""
    from services.scheduler import start_scheduler, stop_scheduler

    logger.info("Starting MacroPulse API v%s", settings.app_version)
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("MacroPulse API shut down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Macro regime intelligence API. "
        "Produces daily probabilistic macro regime signals derived from "
        "PCA latent factors, Hidden Markov Models, GARCH volatility, "
        "ARIMA forecasts, and rule-based multi-domain analysis."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, limit_per_day=settings.rate_limit_per_day)

# ── Routes ───────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(regime_router)
app.include_router(commentary_router)
app.include_router(dashboard_router)
app.include_router(backtest_router)
app.include_router(ws_router)
app.include_router(forecast_router)
app.include_router(analysis_router)
app.include_router(performance_router)
app.include_router(signals_router)

# ── Static frontend (served in production) ───────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
    logger.info("Serving frontend from %s", _frontend_dist)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=settings.app_version)
