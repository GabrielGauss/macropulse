"""
Multi-domain macro analyst orchestrator for MacroPulse.

Pure-Python, rule-based analysis across four macro domains:
  - Equity  : SPX momentum / VIX regime
  - Rates   : yield curve slope trend and 10-year direction
  - Credit  : HY spread trend (tightening vs widening)
  - Liquidity: net-liquidity 30-day trend

The `composite_analysis` function aggregates all four domain signals
into a single composite view.

No LLM or external calls are made here; all logic is deterministic
and based on rolling statistics of the input DataFrames.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Domain thresholds ─────────────────────────────────────────────────

# Minimum rows needed for a rolling trend computation.
_MIN_ROWS = 5

# Regime probability threshold below which a regime is "dominant".
_DOMINANT_PROB = 0.50

# Score range: each analyst returns a float in [-100, +100].
_SCORE_MAX = 100.0
_SCORE_MIN = -100.0


def _safe_slope(series: pd.Series) -> float:
    """
    Return the ordinary-least-squares slope of *series* vs. time index.

    Returns 0.0 if there are fewer than 2 non-NaN values.
    """
    clean = series.dropna()
    n = len(clean)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, clean.values.astype(float), 1)[0])
    return slope


def _clip_score(score: float) -> float:
    """Clamp a domain score to [-100, +100]."""
    return float(np.clip(score, _SCORE_MIN, _SCORE_MAX))


# ── Domain analysts ───────────────────────────────────────────────────


def analyse_equity(
    regime_row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Equity domain analysis based on regime probabilities and VIX state.

    Logic
    -----
    - Recovery + expansion probability trend (30d slope): positive → bullish
    - Risk-off probability level: high (>0.5) → bearish
    - VIX volatility state from regime_row: "low"/"normal" → +, "elevated"/"crisis" → -

    Parameters
    ----------
    regime_row:
        The most recent row from macro_regimes (dict with keys:
        prob_expansion, prob_tightening, prob_risk_off, prob_recovery,
        risk_score, volatility_state, regime).
    history:
        Recent regime history rows (newest-first from DB), used to
        compute probability trends.

    Returns
    -------
    dict with keys: signal ("bullish"|"neutral"|"bearish"), score (float), rationale (str).
    """
    # Convert history to DataFrame (chronological).
    hist_df = pd.DataFrame(list(reversed(history))) if history else pd.DataFrame()

    score = 0.0
    rationale_parts: list[str] = []

    # ── Risk-off probability ──────────────────────────────────────────
    risk_off_prob = float(regime_row.get("prob_risk_off", 0))
    if risk_off_prob > _DOMINANT_PROB:
        score -= 50.0
        rationale_parts.append(f"risk-off probability elevated ({risk_off_prob:.0%})")
    elif risk_off_prob < 0.20:
        score += 20.0
        rationale_parts.append(f"risk-off probability suppressed ({risk_off_prob:.0%})")

    # ── Expansion probability level ───────────────────────────────────
    exp_prob = float(regime_row.get("prob_expansion", 0))
    rec_prob = float(regime_row.get("prob_recovery", 0))
    growth_prob = exp_prob + rec_prob
    if growth_prob > 0.6:
        score += 40.0
        rationale_parts.append(f"growth regime probability high ({growth_prob:.0%})")
    elif growth_prob < 0.3:
        score -= 20.0
        rationale_parts.append(f"growth regime probability low ({growth_prob:.0%})")

    # ── Probability trend over recent history ────────────────────────
    if len(hist_df) >= _MIN_ROWS and "prob_recovery" in hist_df.columns:
        growth_trend = _safe_slope(
            hist_df["prob_expansion"].fillna(0) + hist_df["prob_recovery"].fillna(0)
        )
        if growth_trend > 0.002:
            score += 20.0
            rationale_parts.append("growth probabilities trending up")
        elif growth_trend < -0.002:
            score -= 20.0
            rationale_parts.append("growth probabilities trending down")

    # ── VIX volatility state ──────────────────────────────────────────
    vol_state = str(regime_row.get("volatility_state", "normal")).lower()
    if vol_state in ("low", "normal"):
        score += 10.0
        rationale_parts.append(f"VIX state benign ({vol_state})")
    elif vol_state == "elevated":
        score -= 15.0
        rationale_parts.append("VIX elevated")
    elif vol_state == "crisis":
        score -= 30.0
        rationale_parts.append("VIX in crisis state")

    score = _clip_score(score)

    if score >= 25:
        signal = "bullish"
    elif score <= -25:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "score": round(score, 1),
        "rationale": "; ".join(rationale_parts) if rationale_parts else "no dominant signal",
    }


def analyse_rates(features_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Rates domain analysis using yield curve slope trend and 10-year direction.

    Logic
    -----
    - Yield curve slope (d_yield_curve cumulative): positive / steepening → bullish
    - 10-year yield direction (d_10y trend): falling → bullish (accommodative),
      rising sharply → bearish (restrictive)

    Parameters
    ----------
    features_rows:
        Recent macro_features rows (newest-first from DB), with at minimum
        d_yield_curve and d_10y columns.

    Returns
    -------
    dict with keys: signal, score, rationale.
    """
    feat_df = pd.DataFrame(list(reversed(features_rows))) if features_rows else pd.DataFrame()

    score = 0.0
    rationale_parts: list[str] = []

    if feat_df.empty or len(feat_df) < _MIN_ROWS:
        return {
            "signal": "neutral",
            "score": 0.0,
            "rationale": "insufficient rates data",
        }

    # ── Yield curve trend (steepening vs flattening) ──────────────────
    if "d_yield_curve" in feat_df.columns:
        curve_slope = _safe_slope(feat_df["d_yield_curve"].fillna(0))
        # Cumulative yield curve level (sum of differences) as proxy for level
        curve_level = float(feat_df["d_yield_curve"].fillna(0).sum())
        if curve_slope > 0.0005:
            score += 25.0
            rationale_parts.append("yield curve steepening (trend)")
        elif curve_slope < -0.0005:
            score -= 25.0
            rationale_parts.append("yield curve flattening / inverting (trend)")

        if curve_level > 0:
            score += 10.0
            rationale_parts.append("net yield curve steepening over window")
        else:
            score -= 10.0
            rationale_parts.append("net yield curve flattening over window")

    # ── 10-year yield direction ───────────────────────────────────────
    if "d_10y" in feat_df.columns:
        ten_yr_slope = _safe_slope(feat_df["d_10y"].fillna(0))
        if ten_yr_slope < -0.001:
            score += 20.0
            rationale_parts.append("10Y yield falling (accommodative)")
        elif ten_yr_slope > 0.002:
            score -= 30.0
            rationale_parts.append("10Y yield rising sharply (restrictive)")
        elif ten_yr_slope > 0.001:
            score -= 10.0
            rationale_parts.append("10Y yield rising modestly")

    score = _clip_score(score)

    if score >= 25:
        signal = "bullish"
    elif score <= -25:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "score": round(score, 1),
        "rationale": "; ".join(rationale_parts) if rationale_parts else "no dominant rates signal",
    }


def analyse_credit(features_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Credit domain analysis using HY spread trend.

    Logic
    -----
    - d_hy_spread trend: negative (tightening) → bullish, positive (widening) → bearish
    - Cumulative net change over window as confirmation

    Parameters
    ----------
    features_rows:
        Recent macro_features rows (newest-first from DB).

    Returns
    -------
    dict with keys: signal, score, rationale.
    """
    feat_df = pd.DataFrame(list(reversed(features_rows))) if features_rows else pd.DataFrame()

    score = 0.0
    rationale_parts: list[str] = []

    if feat_df.empty or len(feat_df) < _MIN_ROWS or "d_hy_spread" not in feat_df.columns:
        return {
            "signal": "neutral",
            "score": 0.0,
            "rationale": "insufficient credit data",
        }

    spread_series = feat_df["d_hy_spread"].fillna(0)
    spread_slope = _safe_slope(spread_series)
    spread_net = float(spread_series.sum())
    spread_std = float(spread_series.std()) if len(spread_series) > 1 else 1.0
    if spread_std == 0:
        spread_std = 1.0

    # ── Trend direction ───────────────────────────────────────────────
    normalised_slope = spread_slope / max(spread_std, 1e-6)
    if normalised_slope < -0.1:
        score += 40.0
        rationale_parts.append("HY spreads tightening (trend)")
    elif normalised_slope < 0:
        score += 15.0
        rationale_parts.append("HY spreads mildly tightening")
    elif normalised_slope > 0.1:
        score -= 40.0
        rationale_parts.append("HY spreads widening (trend)")
    elif normalised_slope > 0:
        score -= 15.0
        rationale_parts.append("HY spreads mildly widening")

    # ── Net level over window ─────────────────────────────────────────
    if spread_net < -0.2:
        score += 20.0
        rationale_parts.append("net spread compression over window")
    elif spread_net > 0.2:
        score -= 20.0
        rationale_parts.append("net spread expansion over window")

    score = _clip_score(score)

    if score >= 25:
        signal = "bullish"
    elif score <= -25:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "score": round(score, 1),
        "rationale": "; ".join(rationale_parts) if rationale_parts else "no dominant credit signal",
    }


def analyse_liquidity(liquidity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Liquidity domain analysis using the Fed net-liquidity proxy.

    Logic
    -----
    - 30-day trend direction of net_liquidity: positive → risk_on
    - Magnitude (slope normalised by level): large positive → strong conviction

    Parameters
    ----------
    liquidity_rows:
        Recent macro_features rows with net_liquidity column
        (newest-first from DB).

    Returns
    -------
    dict with keys: signal ("risk_on"|"neutral"|"risk_off"), score, rationale.
    """
    liq_df = pd.DataFrame(list(reversed(liquidity_rows))) if liquidity_rows else pd.DataFrame()

    score = 0.0
    rationale_parts: list[str] = []

    if liq_df.empty or "net_liquidity" not in liq_df.columns:
        return {
            "signal": "neutral",
            "score": 0.0,
            "rationale": "insufficient liquidity data",
        }

    # Use last 30 rows (≈30 trading days).
    liq_series = liq_df["net_liquidity"].dropna().tail(30)

    if len(liq_series) < _MIN_ROWS:
        return {
            "signal": "neutral",
            "score": 0.0,
            "rationale": "insufficient liquidity data",
        }

    slope = _safe_slope(liq_series)
    level = float(liq_series.iloc[-1]) if len(liq_series) > 0 else 0.0
    start_level = float(liq_series.iloc[0]) if len(liq_series) > 0 else 1.0
    if start_level == 0:
        start_level = 1.0

    # Normalise slope as fraction of the starting level.
    normalised_slope = slope / abs(start_level) * 1_000  # per-thousand per day

    if normalised_slope > 0.5:
        score += 60.0
        rationale_parts.append(
            f"net liquidity expanding strongly (slope={slope:+.0f}M/day)"
        )
    elif normalised_slope > 0.1:
        score += 30.0
        rationale_parts.append("net liquidity expanding")
    elif normalised_slope < -0.5:
        score -= 60.0
        rationale_parts.append(
            f"net liquidity contracting strongly (slope={slope:+.0f}M/day)"
        )
    elif normalised_slope < -0.1:
        score -= 30.0
        rationale_parts.append("net liquidity contracting")
    else:
        rationale_parts.append("net liquidity trend flat")

    # ── Absolute level context ────────────────────────────────────────
    # Use d_liquidity if available for a finer recent change signal.
    if "d_liquidity" in liq_df.columns:
        d_liq = liq_df["d_liquidity"].dropna().tail(5)
        recent_flow = float(d_liq.mean()) if len(d_liq) > 0 else 0.0
        if recent_flow > 0:
            score += 10.0
            rationale_parts.append("recent 5-day liquidity flow positive")
        elif recent_flow < 0:
            score -= 10.0
            rationale_parts.append("recent 5-day liquidity flow negative")

    score = _clip_score(score)

    if score >= 25:
        signal = "risk_on"
    elif score <= -25:
        signal = "risk_off"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "score": round(score, 1),
        "rationale": "; ".join(rationale_parts) if rationale_parts else "no dominant liquidity signal",
    }


# ── Composite ─────────────────────────────────────────────────────────


def composite_analysis(
    regime_row: dict[str, Any],
    history: list[dict[str, Any]],
    features: list[dict[str, Any]],
    liquidity: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run all four domain analysts and aggregate into a composite view.

    The composite score is the arithmetic mean of the four domain
    scores (each already in [-100, +100]).  Conviction is derived from
    the standard deviation of domain scores: low dispersion across
    analysts implies high conviction; high dispersion implies low
    conviction.

    Parameters
    ----------
    regime_row:
        Most recent macro_regimes DB row.
    history:
        Recent regime history rows (newest-first), ~30–60 rows.
    features:
        Recent macro_features rows (newest-first), ~30–60 rows.
    liquidity:
        Recent liquidity rows (newest-first), ~30 rows.

    Returns
    -------
    dict with keys:
        composite_signal  : "risk_on" | "neutral" | "risk_off"
        composite_score   : float  (-100 to +100)
        conviction        : "high" | "medium" | "low"
        domain_signals    : dict[str → {signal, score, rationale}]
        regime_alignment  : bool
    """
    equity_out = analyse_equity(regime_row, history)
    rates_out = analyse_rates(features)
    credit_out = analyse_credit(features)
    liquidity_out = analyse_liquidity(liquidity)

    domain_signals = {
        "equity": equity_out,
        "rates": rates_out,
        "credit": credit_out,
        "liquidity": liquidity_out,
    }

    scores = np.array(
        [equity_out["score"], rates_out["score"], credit_out["score"], liquidity_out["score"]],
        dtype=float,
    )
    composite_score = float(np.clip(np.mean(scores), _SCORE_MIN, _SCORE_MAX))

    # ── Conviction from analyst agreement ────────────────────────────
    score_std = float(np.std(scores))
    # Low std → analysts agree → high conviction.
    if score_std < 20.0:
        conviction = "high"
    elif score_std < 45.0:
        conviction = "medium"
    else:
        conviction = "low"

    # ── Composite signal ──────────────────────────────────────────────
    if composite_score >= 20:
        composite_signal = "risk_on"
    elif composite_score <= -20:
        composite_signal = "risk_off"
    else:
        composite_signal = "neutral"

    # ── Regime alignment ──────────────────────────────────────────────
    current_regime = str(regime_row.get("regime", "")).lower()
    # HMM regime → expected composite signal mapping
    regime_to_expected: dict[str, str] = {
        "expansion": "risk_on",
        "recovery": "risk_on",
        "tightening": "neutral",
        "risk_off": "risk_off",
    }
    expected_signal = regime_to_expected.get(current_regime, "neutral")
    regime_alignment = composite_signal == expected_signal

    logger.info(
        "Composite analysis: signal=%s score=%.1f conviction=%s alignment=%s",
        composite_signal,
        composite_score,
        conviction,
        regime_alignment,
    )

    return {
        "composite_signal": composite_signal,
        "composite_score": round(composite_score, 1),
        "conviction": conviction,
        "domain_signals": domain_signals,
        "regime_alignment": regime_alignment,
    }
