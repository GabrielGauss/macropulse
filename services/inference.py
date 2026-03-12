"""
Regime inference service.

Provides a high-level interface that loads frozen models once and
exposes a stateless ``infer()`` method for use by both the daily
pipeline and any ad-hoc analysis scripts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config.settings import get_settings
from models.hmm_model import HMMModel
from models.pca_model import PCAModel
from models.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)


class RegimeInferenceService:
    """Loads frozen artifacts and runs PCA → HMM → classification."""

    def __init__(self, model_version: str | None = None) -> None:
        settings = get_settings()
        self.version = model_version or settings.default_model_version
        self._pca: PCAModel | None = None
        self._hmm: HMMModel | None = None
        self._classifier: RegimeClassifier | None = None

    # ── Lazy loading ─────────────────────────────────────────────

    @property
    def pca(self) -> PCAModel:
        if self._pca is None:
            self._pca = PCAModel.load(self.version)
        return self._pca

    @property
    def hmm(self) -> HMMModel:
        if self._hmm is None:
            self._hmm = HMMModel.load(self.version)
        return self._hmm

    @property
    def classifier(self) -> RegimeClassifier:
        if self._classifier is None:
            self._classifier = RegimeClassifier.load(self.version)
        return self._classifier

    # ── Public API ───────────────────────────────────────────────

    def infer(
        self,
        feature_matrix: np.ndarray,
        vix_diff: float | None = None,
    ) -> dict[str, Any]:
        """
        Run the full inference chain on a feature matrix.

        Parameters
        ----------
        feature_matrix : ndarray of shape (T, n_features)
            Stationary feature matrix (MODEL_FEATURE_COLS order).
        vix_diff : optional latest VIX first-difference for vol label.

        Returns
        -------
        dict with keys:
            factors        – ndarray (T, n_components)
            regime         – str
            probabilities  – dict[str, float]
            risk_score     – float
            volatility_state – str
            all_probs      – ndarray (T, n_regimes)
        """
        factors = self.pca.transform(feature_matrix)
        all_probs = self.hmm.predict_proba(factors)
        latest_probs = all_probs[-1]

        result = self.classifier.classify(latest_probs, vix_diff=vix_diff)
        result["factors"] = factors
        result["all_probs"] = all_probs

        logger.info(
            "Inference complete: regime=%s  risk=%s  version=%s",
            result["regime"],
            result["risk_score"],
            self.version,
        )
        return result

    def predict_sequence(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return the Viterbi state sequence for drift / back-testing."""
        factors = self.pca.transform(feature_matrix)
        return self.hmm.predict(factors)
