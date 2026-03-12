#!/usr/bin/env python3
"""
Train (or retrain) the MacroPulse PCA + HMM + GARCH pipeline.

This script:
  1. Fetches historical FRED + market data.
  2. Builds the feature matrix (10 features for v2).
  3. Fits StandardScaler + PCA.
  4. Fits a Gaussian HMM on the PCA factors.
  5. Derives regime label mappings.
  6. Fits GARCH(1,1) on d_sp500 and d_vix returns.
  7. Serializes all artifacts under models/artifacts/.

Usage:
    python scripts/retrain_models.py
    python scripts/retrain_models.py --version v2 --lookback 1260

Note: Pass --version v2 to train the full 10-feature set including
      gold and oil.  The default version (v1) still uses the original
      8-feature set for backward compatibility.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from data.ingestion.fred_client import fetch_all_fred  # noqa: E402
from data.ingestion.market_client import fetch_market_data  # noqa: E402
from data.processing.feature_engineering import (  # noqa: E402
    MODEL_FEATURE_COLS,
    MODEL_FEATURE_COLS_V1,
    build_features,
)
from models.garch_model import GARCHModel  # noqa: E402
from models.hmm_model import HMMModel  # noqa: E402
from models.pca_model import PCAModel  # noqa: E402
from models.regime_classifier import RegimeClassifier  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="MacroPulse model trainer")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Artifact version label (default from settings).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Number of calendar days of history to use for training.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    settings = get_settings()
    version = args.version or settings.default_model_version
    lookback = args.lookback or settings.data_lookback_days
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback)

    logger.info("Fetching training data from %s to %s …", start, today)
    fred_df = fetch_all_fred(start=start, end=today)
    market_df = fetch_market_data(start=start, end=today)

    logger.info("Building features …")
    features = build_features(fred_df, market_df)

    # v1 uses the legacy 8-feature set; v2 uses all 10 features.
    feature_cols = MODEL_FEATURE_COLS_V1 if version == "v1" else MODEL_FEATURE_COLS
    X = features[feature_cols].values
    logger.info(
        "Training matrix shape: %s  (version=%s, features=%d)",
        X.shape, version, len(feature_cols),
    )

    # ── PCA ──────────────────────────────────────────────────────
    pca_model = PCAModel()
    pca_model.fit(X)
    pca_model.save(version)

    factors = pca_model.transform(X)
    logger.info("PCA factors shape: %s", factors.shape)

    # ── HMM ──────────────────────────────────────────────────────
    hmm_model = HMMModel()
    hmm_model.fit(factors)
    hmm_model.save(version)

    # ── Regime label mapping ─────────────────────────────────────
    classifier = RegimeClassifier()
    classifier.fit_labels(hmm_model.hmm.means_, feature_names=feature_cols)
    classifier.save(version)

    # ── GARCH volatility models ───────────────────────────────────
    garch_sp500 = GARCHModel(series_name="d_sp500")
    garch_sp500.fit(features["d_sp500"])
    garch_sp500.save(version)

    garch_vix = GARCHModel(series_name="d_vix")
    garch_vix.fit(features["d_vix"])
    garch_vix.save(version)

    # ── Summary ──────────────────────────────────────────────────
    explained = pca_model.explained_variance
    regimes = hmm_model.predict(factors)
    unique, counts = np.unique(regimes, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))

    # Sample GARCH vol state on the most recent observation.
    latest_sp500_vol = garch_sp500.forecast_vol(features["d_sp500"])
    garch_vol_state = garch_sp500.classify_vol_state(latest_sp500_vol)

    logger.info("══════════════════════════════════════════════")
    logger.info("Training complete (version=%s)", version)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    logger.info("PCA explained variance: %.2f%%", explained * 100)
    logger.info("HMM state distribution: %s", dist)
    logger.info("Label map: %s", classifier.label_map)
    logger.info(
        "GARCH (d_sp500) latest vol state: %s  (cond_vol=%.4f)",
        garch_vol_state, latest_sp500_vol,
    )
    logger.info("Artifacts saved to: %s", settings.model_artifacts_dir)
    logger.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
