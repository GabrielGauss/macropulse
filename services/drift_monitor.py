"""
Model drift monitoring for MacroPulse.

Computes lightweight drift metrics after each pipeline run:
- PCA variance stability (current vs training)
- Regime persistence (how often the regime stays the same)
- Feature distribution shift (z-score of recent mean/std vs training)
"""

from __future__ import annotations

import logging

import numpy as np

from models.pca_model import PCAModel

logger = logging.getLogger(__name__)


def compute_pca_variance_drift(
    pca_model: PCAModel,
    X_new: np.ndarray,
) -> float:
    """
    Compare explained variance on new data against training-time variance.

    Returns the absolute difference (0 = no drift).
    """
    scaled = pca_model.scaler.transform(X_new)
    transformed = pca_model.pca.transform(scaled)
    reconstruction = pca_model.pca.inverse_transform(transformed)
    total_var = np.var(scaled, axis=0).sum()
    residual_var = np.var(scaled - reconstruction, axis=0).sum()
    current_explained = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0
    training_explained = pca_model.explained_variance
    drift = abs(current_explained - training_explained)
    logger.info(
        "PCA variance drift: training=%.3f, current=%.3f, delta=%.4f",
        training_explained,
        current_explained,
        drift,
    )
    return round(drift, 6)


def compute_regime_persistence(regimes: np.ndarray) -> float:
    """
    Fraction of consecutive days where the regime did not change.

    High persistence (>0.9) may indicate a stuck model.
    """
    if len(regimes) < 2:
        return 1.0
    same = np.sum(regimes[1:] == regimes[:-1])
    persistence = float(same / (len(regimes) - 1))
    return round(persistence, 4)


def compute_feature_shift(
    X_train: np.ndarray,
    X_new: np.ndarray,
) -> tuple[float, float]:
    """
    Compare mean and std of new features vs training features.

    Returns (mean_shift, std_shift) averaged across all features.
    """
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0) + 1e-8
    new_mean = np.mean(X_new, axis=0)
    new_std = np.std(X_new, axis=0) + 1e-8

    mean_shift = float(np.mean(np.abs(new_mean - train_mean) / train_std))
    std_shift = float(np.mean(np.abs(new_std - train_std) / train_std))
    return round(mean_shift, 6), round(std_shift, 6)
