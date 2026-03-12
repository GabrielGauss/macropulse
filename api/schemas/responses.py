"""
Pydantic response models for the MacroPulse REST API.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class RegimeProbabilities(BaseModel):
    expansion: float = Field(..., ge=0, le=1)
    tightening: float = Field(..., ge=0, le=1)
    risk_off: float = Field(..., ge=0, le=1)
    recovery: float = Field(..., ge=0, le=1)


class RegimeResponse(BaseModel):
    timestamp: dt.datetime
    macro_regime: str
    risk_score: float
    probabilities: RegimeProbabilities
    volatility_state: str | None = None
    model_version: str | None = None


class LiquidityRow(BaseModel):
    time: dt.datetime
    net_liquidity: float | None = None
    d_liquidity: float | None = None


class LiquidityResponse(BaseModel):
    data: list[LiquidityRow]


class FactorRow(BaseModel):
    time: dt.datetime
    factor_1: float | None = None
    factor_2: float | None = None
    factor_3: float | None = None
    factor_4: float | None = None
    model_version: str | None = None


class FactorsResponse(BaseModel):
    data: list[FactorRow]


class DriftRow(BaseModel):
    time: dt.datetime
    pca_explained_variance: float | None = None
    regime_persistence: float | None = None
    feature_mean_shift: float | None = None
    feature_std_shift: float | None = None
    model_version: str | None = None


class DriftResponse(BaseModel):
    data: list[DriftRow]


class CommentaryResponse(BaseModel):
    timestamp: dt.datetime
    macro_regime: str
    risk_score: float
    headline: str
    narrative: str
    key_signals: list[str] = []
    watch_for: str = ""
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Forecast ──────────────────────────────────────────────────────────


class ForecastRow(BaseModel):
    """A single day in the 5-day regime probability forecast."""

    date: dt.date
    prob_expansion: float = Field(..., ge=0, le=1)
    prob_tightening: float = Field(..., ge=0, le=1)
    prob_risk_off: float = Field(..., ge=0, le=1)
    prob_recovery: float = Field(..., ge=0, le=1)
    risk_score: float = Field(..., ge=-100, le=100)
    confidence: float = Field(..., ge=0, le=1)


class ForecastResponse(BaseModel):
    """ARIMA-based n-day ahead regime probability forecast."""

    horizon: int
    generated_at: dt.datetime
    forecast: list[ForecastRow]


# ── Composite Analysis ────────────────────────────────────────────────


class DomainSignal(BaseModel):
    """Single-domain analyst output."""

    signal: str  # "bullish" | "neutral" | "bearish" (equity/rates/credit)
    # or "risk_on" | "neutral" | "risk_off" (liquidity/composite)
    score: float
    rationale: str


class CompositeAnalysisResponse(BaseModel):
    """
    Multi-domain rule-based macro analysis.

    composite_signal is one of: "risk_on", "neutral", "risk_off".
    composite_score ranges from -100 (full risk-off) to +100 (full risk-on).
    conviction is determined by analyst agreement: "high", "medium", "low".
    regime_alignment is True when the composite agrees with the HMM regime.
    """

    generated_at: dt.datetime
    composite_signal: str
    composite_score: float = Field(..., ge=-100, le=100)
    conviction: str
    regime_alignment: bool
    domain_signals: dict[str, DomainSignal]


# ── Unified Signal Package (§5.3) ─────────────────────────────────────


class NetLiquiditySignal(BaseModel):
    """Fed net-liquidity component of the signal package."""

    level_bn: float | None = None
    change_4w_bn: float | None = None
    zscore: float | None = None
    trend: str  # "EXPANDING" | "CONTRACTING" | "STABLE" | "UNKNOWN"


class PCAFactorsSignal(BaseModel):
    """PCA factor scores + per-component variance."""

    pc1: float | None = None
    pc2: float | None = None
    pc3: float | None = None
    pc4: float | None = None
    variance_explained_pct: list[float] = []


class RegimeSignal(BaseModel):
    """HMM regime component of the signal package."""

    most_likely: str
    probabilities: dict[str, float]
    confidence: str  # "HIGH" | "MODERATE" | "LOW"
    persistence_days: int
    expected_duration_remaining_days: int
    risk_score: float


class ModelMetadata(BaseModel):
    """Artifact provenance for the signal package."""

    model_version: str
    pca_fit_date: str | None = None
    hmm_fit_date: str | None = None
    data_vintage: str


class SignalPackageResponse(BaseModel):
    """
    Unified signal package — primary commercial endpoint (§5.3).

    Combines regime classification, net liquidity state, PCA factors,
    and model provenance into a single response.
    """

    date: str
    regime: RegimeSignal
    net_liquidity: NetLiquiditySignal
    pca_factors: PCAFactorsSignal
    model_metadata: ModelMetadata


# ── User / Auth ───────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str
    name: str | None = None


class RegisterResponse(BaseModel):
    """
    Returned once on registration.  The plaintext api_key is shown ONLY here;
    it cannot be recovered later.
    """

    user_id: int
    email: str
    api_key: str          # plaintext — shown once, store securely
    key_prefix: str       # first 12 chars, safe to display
    tier: str
    daily_limit: int      # 0 = unlimited


class KeyInfoResponse(BaseModel):
    """Safe representation of a key (no plaintext)."""

    user_id: int
    email: str
    key_prefix: str
    tier: str
    daily_limit: int
    created_at: dt.datetime
    last_used_at: dt.datetime | None = None


class UsageResponse(BaseModel):
    tier: str
    daily_limit: int      # 0 = unlimited
    used_today: int
    remaining: int        # -1 = unlimited
    reset_at: int         # Unix timestamp midnight UTC


class RotateKeyResponse(BaseModel):
    """New key issued after rotation.  Previous key is immediately revoked."""

    api_key: str          # plaintext — shown once
    key_prefix: str
    tier: str
    daily_limit: int
