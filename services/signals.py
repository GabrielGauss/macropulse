"""
Unified signal package assembly for MacroPulse.

Assembles the canonical /v1/signals response from the database and
frozen model artifacts.  This is the primary commercial API surface
defined in the MacroPulse Master Build & Launch Manual §5.3.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import get_settings
from database import queries
from models.hmm_model import HMMModel
from models.pca_model import PCAModel
from models.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def _artifact_mtime(filename: str) -> str | None:
    """Return ISO-8601 mtime of an artifact file, or None if missing."""
    settings = get_settings()
    path = Path(settings.model_artifacts_dir) / filename
    if path.exists():
        return dt.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    return None


def _persistence_and_confidence(
    regime_rows: list[dict[str, Any]],
    current_regime: str,
    max_prob: float,
) -> tuple[int, str]:
    """
    Compute:
      persistence_days – how many consecutive trailing days share the same regime label
      confidence       – HIGH / MODERATE / LOW from max_prob
    """
    streak = 0
    for row in regime_rows:  # rows ordered DESC (newest first)
        if row.get("regime") == current_regime:
            streak += 1
        else:
            break

    if max_prob >= 0.70:
        confidence = "HIGH"
    elif max_prob >= 0.50:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return streak, confidence


def _expected_duration_remaining(
    transmat: np.ndarray,
    state_idx: int,
    persistence_days: int,
) -> int:
    """
    Expected remaining days in the current regime.

    Based on the geometric distribution of the self-transition probability:
        expected_total = 1 / (1 - P_stay)
        remaining      = max(1, expected_total - persistence_days)
    """
    p_stay = float(transmat[state_idx, state_idx])
    if p_stay >= 1.0:
        return 999  # absorbing state — stay forever
    expected_total = 1.0 / (1.0 - p_stay)
    remaining = max(1, int(round(expected_total - persistence_days)))
    return remaining


def _net_liquidity_signals(
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Derive net_liquidity sub-signals from recent feature rows (DESC order).

    Returns:
        level_bn           – latest net_liquidity in billions
        change_4w_bn       – current minus 20-rows-back, divided by 1000
        zscore             – z-score vs trailing 504-day (2-year) window
        trend              – "EXPANDING" / "CONTRACTING" / "STABLE"
    """
    if not feature_rows:
        return {
            "level_bn": None,
            "change_4w_bn": None,
            "zscore": None,
            "trend": "UNKNOWN",
        }

    levels = [
        float(r["net_liquidity"])
        for r in feature_rows
        if r.get("net_liquidity") is not None
    ]

    if not levels:
        return {
            "level_bn": None,
            "change_4w_bn": None,
            "zscore": None,
            "trend": "UNKNOWN",
        }

    current_level = levels[0]

    # 4-week change (20 trading days back)
    level_20ago = levels[20] if len(levels) > 20 else levels[-1]
    change_4w_bn = round((current_level - level_20ago) / 1000.0, 3)

    # Z-score vs 2-year window (up to 504 rows)
    window = levels[:504]
    if len(window) > 1:
        mean_w = float(np.mean(window))
        std_w = float(np.std(window))
        zscore = round((current_level - mean_w) / std_w, 3) if std_w > 0 else 0.0
    else:
        zscore = 0.0

    # Trend from d_liquidity sign over last 4 weeks
    d_liq_recent = [
        float(r["d_liquidity"])
        for r in feature_rows[:20]
        if r.get("d_liquidity") is not None
    ]
    if d_liq_recent:
        pos = sum(1 for v in d_liq_recent if v > 0)
        neg = sum(1 for v in d_liq_recent if v < 0)
        if pos >= 12:
            trend = "EXPANDING"
        elif neg >= 12:
            trend = "CONTRACTING"
        else:
            trend = "STABLE"
    else:
        trend = "STABLE"

    return {
        "level_bn": round(current_level / 1000.0, 1),
        "change_4w_bn": change_4w_bn,
        "zscore": zscore,
        "trend": trend,
    }


# ── Public API ───────────────────────────────────────────────────────


def build_signal_package(
    target_date: dt.date | None = None,
    version: str | None = None,
) -> dict[str, Any] | None:
    """
    Assemble the full unified signal package for a given date.

    If target_date is None, uses the latest available regime row.
    Returns None if no data is found for the requested date.
    """
    settings = get_settings()
    version = version or settings.default_model_version

    # ── Load regime data ──────────────────────────────────────────
    if target_date is None:
        regime_row = queries.fetch_current_regime()
        if regime_row is None:
            return None
        target_date = (
            regime_row["time"].date()
            if hasattr(regime_row["time"], "date")
            else regime_row["time"]
        )
    else:
        # Look for an exact date match in history
        rows = queries.fetch_regime_history(
            start=target_date,
            end=target_date + dt.timedelta(days=1),
            limit=1,
        )
        if not rows:
            return None
        regime_row = rows[0]

    # ── Load trailing regime history for persistence ─────────────
    trailing_regimes = queries.fetch_regime_history(limit=200)

    # ── Load trailing features for liquidity signals ─────────────
    feature_rows = queries.fetch_latest_features(limit=504)

    # ── Load frozen models ────────────────────────────────────────
    try:
        pca_model = PCAModel.load(version)
        hmm_model = HMMModel.load(version)
        classifier = RegimeClassifier.load(version)
    except Exception as exc:
        logger.error("Failed to load model artifacts (version=%s): %s", version, exc)
        return None

    # ── Regime signal ─────────────────────────────────────────────
    current_regime: str = regime_row["regime"]
    probabilities: dict[str, float] = {
        "expansion": float(regime_row.get("prob_expansion") or 0),
        "tightening": float(regime_row.get("prob_tightening") or 0),
        "risk_off": float(regime_row.get("prob_risk_off") or 0),
        "recovery": float(regime_row.get("prob_recovery") or 0),
    }
    max_prob = max(probabilities.values())

    persistence_days, confidence = _persistence_and_confidence(
        trailing_regimes, current_regime, max_prob
    )

    # Map regime label → HMM state index
    reverse_map: dict[str, int] = {v: k for k, v in classifier.label_map.items()}
    state_idx = reverse_map.get(current_regime, 0)

    expected_duration_remaining = _expected_duration_remaining(
        hmm_model.transition_matrix, state_idx, persistence_days
    )

    regime_signal: dict[str, Any] = {
        "most_likely": current_regime,
        "probabilities": probabilities,
        "confidence": confidence,
        "persistence_days": persistence_days,
        "expected_duration_remaining_days": expected_duration_remaining,
        "risk_score": float(regime_row.get("risk_score") or 0),
    }

    # ── PCA factors signal ────────────────────────────────────────
    factor_rows = queries.fetch_latest_factors(limit=1)
    if factor_rows:
        fr = factor_rows[0]
        pca_factors: dict[str, Any] = {
            "pc1": round(float(fr.get("factor_1") or 0), 4),
            "pc2": round(float(fr.get("factor_2") or 0), 4),
            "pc3": round(float(fr.get("factor_3") or 0), 4) if fr.get("factor_3") is not None else None,
            "pc4": round(float(fr.get("factor_4") or 0), 4) if fr.get("factor_4") is not None else None,
            "variance_explained_pct": [
                round(v * 100, 2) for v in pca_model.explained_variance_ratio
            ],
        }
    else:
        pca_factors = {
            "pc1": None, "pc2": None, "pc3": None, "pc4": None,
            "variance_explained_pct": [],
        }

    # ── Net liquidity signal ──────────────────────────────────────
    net_liquidity = _net_liquidity_signals(feature_rows)

    # ── Model metadata ────────────────────────────────────────────
    model_metadata: dict[str, Any] = {
        "model_version": version,
        "pca_fit_date": _artifact_mtime(f"pca_{version}.pkl"),
        "hmm_fit_date": _artifact_mtime(f"hmm_{version}.pkl"),
        "data_vintage": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    return {
        "date": target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date),
        "regime": regime_signal,
        "net_liquidity": net_liquidity,
        "pca_factors": pca_factors,
        "model_metadata": model_metadata,
    }


def build_signal_range(
    start: dt.date,
    end: dt.date,
    version: str | None = None,
) -> list[dict[str, Any]]:
    """
    Assemble signal packages for every date in [start, end] that has a DB row.

    Caps at 365 days. Reuses the same loaded models for all dates.
    """
    settings = get_settings()
    version = version or settings.default_model_version

    # ── Load models once ──────────────────────────────────────────
    try:
        pca_model = PCAModel.load(version)
        hmm_model = HMMModel.load(version)
        classifier = RegimeClassifier.load(version)
    except Exception as exc:
        logger.error("Failed to load model artifacts (version=%s): %s", version, exc)
        return []

    # ── Fetch regime history for the date range (up to 365 days) ─
    delta = (end - start).days
    if delta > 365:
        start = end - dt.timedelta(days=365)

    rows = queries.fetch_regime_history(start=start, end=end, limit=400)
    if not rows:
        return []

    rows_by_date: dict[str, dict] = {}
    for r in rows:
        d = r["time"].date() if hasattr(r["time"], "date") else r["time"]
        rows_by_date[d.isoformat()] = r

    # ── Load trailing features and factors once ───────────────────
    feature_rows = queries.fetch_latest_features(limit=504)
    factor_rows_all = queries.fetch_latest_factors(limit=400)
    factor_by_date: dict[str, dict] = {}
    for fr in factor_rows_all:
        d = fr["time"].date() if hasattr(fr["time"], "date") else fr["time"]
        factor_by_date[d.isoformat()] = fr

    net_liquidity = _net_liquidity_signals(feature_rows)
    variance_pct = [round(v * 100, 2) for v in pca_model.explained_variance_ratio]
    metadata: dict[str, Any] = {
        "model_version": version,
        "pca_fit_date": _artifact_mtime(f"pca_{version}.pkl"),
        "hmm_fit_date": _artifact_mtime(f"hmm_{version}.pkl"),
        "data_vintage": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    reverse_map: dict[str, int] = {v: k for k, v in classifier.label_map.items()}

    results: list[dict[str, Any]] = []
    sorted_dates = sorted(rows_by_date.keys())

    for date_str in sorted_dates:
        rr = rows_by_date[date_str]
        current_regime: str = rr["regime"]
        probabilities: dict[str, float] = {
            "expansion": float(rr.get("prob_expansion") or 0),
            "tightening": float(rr.get("prob_tightening") or 0),
            "risk_off": float(rr.get("prob_risk_off") or 0),
            "recovery": float(rr.get("prob_recovery") or 0),
        }
        max_prob = max(probabilities.values())

        # Persistence: count matching consecutive rows before this date
        streak = 0
        for other_date in reversed(sorted_dates[: sorted_dates.index(date_str) + 1]):
            if rows_by_date[other_date]["regime"] == current_regime:
                streak += 1
            else:
                break

        if max_prob >= 0.70:
            confidence = "HIGH"
        elif max_prob >= 0.50:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        state_idx = reverse_map.get(current_regime, 0)
        expected_remaining = _expected_duration_remaining(
            hmm_model.transition_matrix, state_idx, streak
        )

        fr = factor_by_date.get(date_str, {})
        pca_factors: dict[str, Any] = {
            "pc1": round(float(fr["factor_1"]), 4) if fr.get("factor_1") is not None else None,
            "pc2": round(float(fr["factor_2"]), 4) if fr.get("factor_2") is not None else None,
            "pc3": round(float(fr["factor_3"]), 4) if fr.get("factor_3") is not None else None,
            "pc4": round(float(fr["factor_4"]), 4) if fr.get("factor_4") is not None else None,
            "variance_explained_pct": variance_pct,
        }

        results.append({
            "date": date_str,
            "regime": {
                "most_likely": current_regime,
                "probabilities": probabilities,
                "confidence": confidence,
                "persistence_days": streak,
                "expected_duration_remaining_days": expected_remaining,
                "risk_score": float(rr.get("risk_score") or 0),
            },
            "net_liquidity": net_liquidity,
            "pca_factors": pca_factors,
            "model_metadata": metadata,
        })

    return results
