"""
Backtesting module for MacroPulse regime signals.

Replays historical data through the frozen model and produces
performance analytics: regime accuracy, transition timing,
and signal stability metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.processing.feature_engineering import MODEL_FEATURE_COLS
from models.regime_classifier import RegimeClassifier, compute_risk_score
from services.inference import RegimeInferenceService

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    dates: list[str]
    regimes: list[str]
    risk_scores: list[float]
    probabilities: list[dict[str, float]]
    transitions: int
    regime_distribution: dict[str, float]
    avg_persistence_days: float
    summary: dict[str, Any]


def run_backtest(
    features: pd.DataFrame,
    model_version: str | None = None,
    window: int | None = None,
) -> BacktestResult:
    """
    Replay historical features through frozen models.

    Parameters
    ----------
    features : DataFrame with MODEL_FEATURE_COLS + index as dates.
    model_version : which artifact version to use.
    window : optional rolling window size for walk-forward replay.
             If None, runs inference on the full matrix at once.

    Returns
    -------
    BacktestResult with full regime timeline and summary statistics.
    """
    svc = RegimeInferenceService(model_version=model_version)
    X = features[MODEL_FEATURE_COLS].values
    dates_raw = features.index

    if window and window < len(X):
        # Walk-forward: run inference on expanding window
        all_regimes: list[str] = []
        all_scores: list[float] = []
        all_probs: list[dict[str, float]] = []

        for i in range(window, len(X) + 1):
            chunk = X[:i]
            result = svc.infer(chunk)
            all_regimes.append(result["regime"])
            all_scores.append(result["risk_score"])
            all_probs.append(result["probabilities"])

        dates_out = [str(d.date()) if hasattr(d, "date") else str(d) for d in dates_raw[window - 1:]]
    else:
        # Full-matrix inference
        factors = svc.pca.transform(X)
        state_probs = svc.hmm.predict_proba(factors)
        classifier = svc.classifier

        all_regimes = []
        all_scores = []
        all_probs = []
        for i in range(len(state_probs)):
            res = classifier.classify(state_probs[i])
            all_regimes.append(res["regime"])
            all_scores.append(res["risk_score"])
            all_probs.append(res["probabilities"])

        dates_out = [str(d.date()) if hasattr(d, "date") else str(d) for d in dates_raw]

    # ── Compute summary statistics ───────────────────────────────
    regime_series = np.array(all_regimes)
    transitions = int(np.sum(regime_series[1:] != regime_series[:-1]))

    unique, counts = np.unique(regime_series, return_counts=True)
    total = len(regime_series)
    regime_dist = {str(r): round(float(c / total), 4) for r, c in zip(unique, counts)}

    # Average persistence: mean run length
    if transitions > 0:
        avg_persist = float(len(regime_series) / (transitions + 1))
    else:
        avg_persist = float(len(regime_series))

    summary = {
        "total_days": len(all_regimes),
        "transitions": transitions,
        "avg_persistence_days": round(avg_persist, 1),
        "regime_distribution": regime_dist,
        "mean_risk_score": round(float(np.mean(all_scores)), 2),
        "risk_score_std": round(float(np.std(all_scores)), 2),
        "model_version": svc.version,
    }

    logger.info(
        "Backtest complete: %d days, %d transitions, dist=%s",
        len(all_regimes),
        transitions,
        regime_dist,
    )

    return BacktestResult(
        dates=dates_out,
        regimes=all_regimes,
        risk_scores=all_scores,
        probabilities=all_probs,
        transitions=transitions,
        regime_distribution=regime_dist,
        avg_persistence_days=round(avg_persist, 1),
        summary=summary,
    )
