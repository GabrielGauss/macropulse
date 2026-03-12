"""
PCA factor model for MacroPulse.

Wraps sklearn PCA and StandardScaler with train / save / load / infer
semantics that follow the frozen-model pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import get_settings

logger = logging.getLogger(__name__)


class PCAModel:
    """Dimensionality-reduction wrapper around StandardScaler + PCA."""

    def __init__(
        self,
        n_components: int | None = None,
        variance_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.n_components = n_components or settings.pca_n_components
        self.variance_threshold = variance_threshold or settings.pca_variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)

    # ── Training ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "PCAModel":
        """Fit scaler and PCA on the training matrix."""
        scaled = self.scaler.fit_transform(X)
        self.pca.fit(scaled)
        explained = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info(
            "PCA fitted – %d components explain %.1f%% variance",
            self.n_components,
            explained * 100,
        )
        return self

    # ── Inference ────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale and project new observations onto the latent factors."""
        scaled = self.scaler.transform(X)
        return self.pca.transform(scaled)

    @property
    def explained_variance(self) -> float:
        """Total explained variance ratio of the fitted model."""
        return float(np.sum(self.pca.explained_variance_ratio_))

    @property
    def explained_variance_ratio(self) -> list[float]:
        """Per-component explained variance ratios."""
        return [float(v) for v in self.pca.explained_variance_ratio_]

    # ── Persistence ──────────────────────────────────────────────

    def save(self, version: str | None = None) -> Path:
        """Serialize scaler and PCA to the artifacts directory."""
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)

        scaler_path = artifacts / f"scaler_{version}.pkl"
        pca_path = artifacts / f"pca_{version}.pkl"

        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.pca, pca_path)
        logger.info("Saved PCA artifacts to %s (version=%s)", artifacts, version)
        return artifacts

    @classmethod
    def load(cls, version: str | None = None) -> "PCAModel":
        """Deserialize a frozen PCA model from disk."""
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)

        instance = cls.__new__(cls)
        instance.scaler = joblib.load(artifacts / f"scaler_{version}.pkl")
        instance.pca = joblib.load(artifacts / f"pca_{version}.pkl")
        instance.n_components = instance.pca.n_components
        instance.variance_threshold = settings.pca_variance_threshold
        logger.info("Loaded PCA model (version=%s)", version)
        return instance
