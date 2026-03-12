"""
Regime classification layer.

Maps raw HMM state indices to interpretable macro regime labels
using the mean characteristics of each state (e.g. high liquidity growth
→ expansion, negative liquidity + high VIX → risk-off).

The label mapping is determined once at training time and persisted
alongside the model artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)

REGIME_LABELS: list[str] = ["expansion", "tightening", "risk_off", "recovery"]


def _assign_regime_labels(
    hmm_means: np.ndarray,
    feature_names: list[str],
) -> dict[int, str]:
    """
    Assign regime labels to HMM states using a 2-factor heuristic.

    Factor 1 broadly captures the liquidity / monetary-conditions direction.
    Factor 2 broadly captures risk appetite.

    Algorithm
    ---------
    1.  Detect "crisis" states: upper-tail F1 outlier (> Q3 + IQR) AND
        negative risk appetite (F2 below median).  These are crisis-liquidity-
        injection events (e.g. SVB bailout) — labelled "risk_off".
    2.  Rank remaining states by F1 (ascending) and assign:
        low F1 → tightening (tight money)
        mid F1 → recovery  (transitional)
        high F1 → expansion (loose money, risk-on)

    This correctly handles pathological cases where an extreme F1 outlier
    coincides with collapsing risk appetite (a macro crisis, not expansion).
    """
    n_states = hmm_means.shape[0]
    f1 = hmm_means[:, 0]
    f2 = hmm_means[:, 1] if hmm_means.shape[1] > 1 else np.zeros(n_states)

    # Robust outlier fence (1× IQR above Q3)
    q1, q3 = np.percentile(f1, [25, 75])
    iqr     = q3 - q1
    upper_fence = q3 + iqr  # 1× IQR — catches strong outliers without being too tight

    f2_median = float(np.median(f2))

    label_map: dict[int, str] = {}

    # Step 1: tag crisis states
    for i in range(n_states):
        if float(f1[i]) > upper_fence and float(f2[i]) < f2_median:
            label_map[i] = "risk_off"

    # Step 2: assign remaining by F1 rank
    remaining = sorted(
        [i for i in range(n_states) if i not in label_map],
        key=lambda i: float(f1[i]),
    )

    non_crisis_seq: list[str]
    if len(remaining) == 1:
        non_crisis_seq = ["expansion"]
    elif len(remaining) == 2:
        non_crisis_seq = ["tightening", "expansion"]
    elif len(remaining) == 3:
        non_crisis_seq = ["tightening", "recovery", "expansion"]
    else:  # 4 — no crisis states were found, use standard ordering
        non_crisis_seq = ["risk_off", "tightening", "recovery", "expansion"]

    for rank, state in enumerate(remaining):
        label_map[state] = non_crisis_seq[rank]

    logger.info("Regime label mapping: %s", label_map)
    return label_map


def compute_risk_score(probabilities: dict[str, float]) -> float:
    """
    Derive a single risk score in [-100, +100].

    Positive values indicate expansionary conditions;
    negative values indicate contraction / risk-off.
    """
    score = (
        probabilities.get("expansion", 0) * 100
        + probabilities.get("recovery", 0) * 50
        - probabilities.get("tightening", 0) * 50
        - probabilities.get("risk_off", 0) * 100
    )
    return round(float(np.clip(score, -100, 100)), 1)


def classify_volatility(vix_diff: float | None) -> str:
    """Simple volatility-state label from VIX change."""
    if vix_diff is None:
        return "unknown"
    if vix_diff > 2.0:
        return "elevated"
    if vix_diff < -2.0:
        return "compressed"
    return "normal"


class RegimeClassifier:
    """End-to-end regime classification from HMM probabilities."""

    def __init__(self, label_map: dict[int, str] | None = None) -> None:
        self.label_map = label_map or {}

    def fit_labels(
        self,
        hmm_means: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Derive the label mapping from HMM state means."""
        self.label_map = _assign_regime_labels(
            hmm_means, feature_names or []
        )

    def classify(
        self,
        state_probs: np.ndarray,
        vix_diff: float | None = None,
    ) -> dict[str, Any]:
        """
        Convert a single-row probability vector into a regime dict.

        Parameters
        ----------
        state_probs : ndarray of shape (n_states,)
        vix_diff    : optional VIX first-difference for vol labelling

        Returns
        -------
        dict with keys: regime, probabilities, risk_score, volatility_state
        """
        prob_dict: dict[str, float] = {}
        for state_idx, label in self.label_map.items():
            prob_dict[label] = round(float(state_probs[state_idx]), 4)

        regime = max(prob_dict, key=prob_dict.get)  # type: ignore[arg-type]
        risk = compute_risk_score(prob_dict)

        return {
            "regime": regime,
            "probabilities": prob_dict,
            "risk_score": risk,
            "volatility_state": classify_volatility(vix_diff),
        }

    # ── Persistence ──────────────────────────────────────────────

    def save(self, version: str | None = None) -> Path:
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"regime_classifier_{version}.pkl"
        joblib.dump(self.label_map, path)
        logger.info("Saved regime classifier to %s", path)
        return artifacts

    @classmethod
    def load(cls, version: str | None = None) -> "RegimeClassifier":
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)
        label_map = joblib.load(artifacts / f"regime_classifier_{version}.pkl")
        instance = cls(label_map=label_map)
        logger.info("Loaded regime classifier (version=%s)", version)
        return instance
