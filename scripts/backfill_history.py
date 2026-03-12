#!/usr/bin/env python3
"""
MacroPulse historical backfill script.

Fetches FRED + market data ONCE for the full period, then loops through every
trading date and runs PCA → HMM inference, writing regime + factor + feature
rows to the database.  Much faster than calling run_daily_pipeline per day.

Usage:
    python scripts/backfill_history.py
    python scripts/backfill_history.py --start 2023-01-01 --version v2
    python scripts/backfill_history.py --start 2022-01-01 --end 2024-12-31 --version v2

Notes:
  - Skips dates that already have a regime row (safe to re-run).
  - Pass --overwrite to force re-computation of all dates.
  - Requires frozen model artifacts (run retrain_models.py first).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from data.ingestion.fred_client import fetch_all_fred  # noqa: E402
from data.ingestion.market_client import fetch_market_data  # noqa: E402
from data.processing.feature_engineering import (  # noqa: E402
    MODEL_FEATURE_COLS,
    MODEL_FEATURE_COLS_V1,
    build_features,
)
from database import queries  # noqa: E402
from models.hmm_model import HMMModel  # noqa: E402
from models.pca_model import PCAModel  # noqa: E402
from models.regime_classifier import RegimeClassifier  # noqa: E402

logger = logging.getLogger(__name__)


def _existing_dates(version: str) -> set[dt.date]:
    """Return the set of dates already stored for this model version."""
    rows = queries.fetch_regime_history(limit=5000)
    return {
        r["time"].date() if hasattr(r["time"], "date") else r["time"]
        for r in rows
        if r.get("model_version") == version
    }


def run_backfill(
    start: dt.date,
    end: dt.date,
    version: str,
    overwrite: bool = False,
    min_warmup: int = 60,
) -> None:
    """
    Backfill macro regime history from `start` to `end`.

    Parameters
    ----------
    start       : first date to backfill (needs `min_warmup` prior days of data)
    end         : last date to backfill (inclusive)
    version     : model artifact version ("v1" | "v2")
    overwrite   : if False, skip dates already in the DB
    min_warmup  : minimum rows before the target date needed for stable inference
    """
    logger.info("═══ MacroPulse backfill  %s → %s  (version=%s) ═══", start, end, version)

    # ── 1. Fetch all data once ────────────────────────────────────────────
    # Pull extra history before `start` for warm-up
    fetch_start = start - dt.timedelta(days=max(min_warmup * 2, 180))
    logger.info("Fetching FRED data from %s …", fetch_start)
    fred_df = fetch_all_fred(start=fetch_start, end=end)

    logger.info("Fetching market data from %s …", fetch_start)
    market_df = fetch_market_data(start=fetch_start, end=end)

    # ── 2. Build full feature matrix once ────────────────────────────────
    logger.info("Building feature matrix …")
    features = build_features(fred_df, market_df)
    logger.info("Feature matrix: %s", features.shape)

    if features.empty:
        logger.error("Feature matrix is empty — check FRED / market data.")
        return

    # ── 3. Load frozen models ─────────────────────────────────────────────
    pca_model  = PCAModel.load(version)
    hmm_model  = HMMModel.load(version)
    classifier = RegimeClassifier.load(version)
    feature_cols = MODEL_FEATURE_COLS_V1 if version == "v1" else MODEL_FEATURE_COLS

    # ── 4. Determine dates to process ────────────────────────────────────
    all_dates = [
        idx.date() if hasattr(idx, "date") else idx
        for idx in features.index
    ]
    target_dates = [d for d in all_dates if start <= d <= end]

    if not overwrite:
        existing = _existing_dates(version)
        target_dates = [d for d in target_dates if d not in existing]
        logger.info(
            "Dates to process: %d  (skipping %d already stored)",
            len(target_dates), len(all_dates) - len(target_dates),
        )
    else:
        logger.info("Dates to process: %d  (overwrite=True)", len(target_dates))

    if not target_dates:
        logger.info("Nothing to do — all dates already stored.")
        return

    # ── 5. Loop through target dates ──────────────────────────────────────
    ok = 0
    skipped = 0
    errors = 0

    for target_date in target_dates:
        try:
            # Slice features up to (and including) target_date for inference
            mask = features.index <= pd.Timestamp(target_date)
            feat_slice = features[mask]

            if len(feat_slice) < min_warmup:
                logger.debug("Skipping %s — only %d rows (need %d)", target_date, len(feat_slice), min_warmup)
                skipped += 1
                continue

            # Missing required feature columns?
            missing = [c for c in feature_cols if c not in feat_slice.columns]
            if missing:
                logger.debug("Skipping %s — missing columns: %s", target_date, missing)
                skipped += 1
                continue

            X_slice = feat_slice[feature_cols].values
            if np.isnan(X_slice).any():
                # Drop rows with NaN and retry
                X_slice = X_slice[~np.isnan(X_slice).any(axis=1)]
                if len(X_slice) < min_warmup:
                    skipped += 1
                    continue

            # PCA transform
            factors = pca_model.transform(X_slice)

            # HMM inference
            state_probs = hmm_model.predict_proba(factors)
            latest_probs = state_probs[-1]

            # Regime classification (no VIX diff; use neutral)
            latest_row = feat_slice.iloc[-1]
            vix_diff = float(latest_row["d_vix"]) if pd.notna(latest_row.get("d_vix")) else None
            result = classifier.classify(latest_probs, vix_diff=vix_diff)

            ts_iso = feat_slice.index[-1].isoformat()

            # ── Store regime ──────────────────────────────────────────
            queries.upsert_macro_regime({
                "time": ts_iso,
                "regime": result["regime"],
                "prob_expansion":  result["probabilities"].get("expansion", 0),
                "prob_tightening": result["probabilities"].get("tightening", 0),
                "prob_risk_off":   result["probabilities"].get("risk_off", 0),
                "prob_recovery":   result["probabilities"].get("recovery", 0),
                "risk_score":      result["risk_score"],
                "volatility_state": result["volatility_state"],
                "model_version":   version,
            })

            # ── Store PCA factors ─────────────────────────────────────
            lf = factors[-1]
            queries.upsert_macro_factors({
                "time":          ts_iso,
                "factor_1":      float(lf[0]),
                "factor_2":      float(lf[1]),
                "factor_3":      float(lf[2]) if len(lf) > 2 else None,
                "factor_4":      float(lf[3]) if len(lf) > 3 else None,
                "model_version": version,
            })

            # ── Store features ────────────────────────────────────────
            row_data: dict = {"time": ts_iso}
            for col in feat_slice.columns:
                val = latest_row.get(col)
                row_data[col] = float(val) if val is not None and pd.notna(val) else None

            queries.upsert_macro_features(row_data)

            ok += 1
            if ok % 50 == 0:
                logger.info("  … %d/%d dates stored", ok, len(target_dates))

        except Exception as exc:
            logger.warning("Error on %s: %s", target_date, exc)
            errors += 1

    logger.info(
        "═══ Backfill complete: %d stored  %d skipped  %d errors ═══",
        ok, skipped, errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MacroPulse historical backfill")
    parser.add_argument(
        "--start",
        type=lambda s: dt.date.fromisoformat(s),
        default=dt.date.today() - dt.timedelta(days=730),  # 2 years back
        help="Start date (YYYY-MM-DD).  Default: 2 years ago.",
    )
    parser.add_argument(
        "--end",
        type=lambda s: dt.date.fromisoformat(s),
        default=dt.date.today(),
        help="End date (YYYY-MM-DD).  Default: today.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model artifact version (e.g. v2).  Default: from settings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-compute and overwrite dates already in the database.",
    )
    parser.add_argument(
        "--min-warmup",
        type=int,
        default=60,
        help="Minimum prior rows required before running inference (default 60).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    settings = get_settings()
    version = args.version or settings.default_model_version

    run_backfill(
        start=args.start,
        end=args.end,
        version=version,
        overwrite=args.overwrite,
        min_warmup=args.min_warmup,
    )


if __name__ == "__main__":
    main()
