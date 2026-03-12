"""
Hidden Markov Model for macro regime detection.

Uses hmmlearn's GaussianHMM to identify latent economic regimes
from PCA factor time-series.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM

from config.settings import get_settings

logger = logging.getLogger(__name__)


class HMMModel:
    """Gaussian HMM wrapper with frozen-model semantics."""

    def __init__(
        self,
        n_regimes: int | None = None,
        n_iter: int | None = None,
        covariance_type: str | None = None,
    ) -> None:
        settings = get_settings()
        self.n_regimes = n_regimes or settings.hmm_n_regimes
        self.n_iter = n_iter or settings.hmm_n_iter
        self.covariance_type = covariance_type or settings.hmm_covariance_type

        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
        )

    # ── Training ─────────────────────────────────────────────────

    def fit(self, factors: np.ndarray) -> "HMMModel":
        """Fit the HMM on the PCA factor matrix (T × K)."""
        self.hmm.fit(factors)
        logger.info(
            "HMM fitted – %d regimes, converged=%s",
            self.n_regimes,
            self.hmm.monitor_.converged,
        )
        return self

    # ── Inference ────────────────────────────────────────────────

    def predict_proba(self, factors: np.ndarray) -> np.ndarray:
        """
        Return posterior state probabilities for each timestep.

        Shape: (T, n_regimes)
        """
        return self.hmm.predict_proba(factors)

    def predict(self, factors: np.ndarray) -> np.ndarray:
        """Return the most likely state sequence via Viterbi."""
        return self.hmm.predict(factors)

    def score(self, factors: np.ndarray) -> float:
        """Log-likelihood of the observation sequence."""
        return float(self.hmm.score(factors))

    @property
    def transition_matrix(self) -> np.ndarray:
        """HMM state transition probability matrix (n_regimes × n_regimes)."""
        return self.hmm.transmat_

    # ── Persistence ──────────────────────────────────────────────

    def save(self, version: str | None = None) -> Path:
        """Serialize the HMM to the artifacts directory."""
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)

        path = artifacts / f"hmm_{version}.pkl"
        joblib.dump(self.hmm, path)
        logger.info("Saved HMM artifact to %s", path)
        return artifacts

    @classmethod
    def load(cls, version: str | None = None) -> "HMMModel":
        """Load a frozen HMM from disk."""
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)

        instance = cls.__new__(cls)
        instance.hmm = joblib.load(artifacts / f"hmm_{version}.pkl")
        instance.n_regimes = instance.hmm.n_components
        instance.covariance_type = instance.hmm.covariance_type
        instance.n_iter = instance.hmm.n_iter
        logger.info("Loaded HMM model (version=%s)", version)
        return instance
