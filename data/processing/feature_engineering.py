"""
Feature engineering for MacroPulse.

Transforms raw macro and market series into stationary features
suitable for PCA / HMM consumption.

Feature set (10 total):
    d_liquidity, d_sp500, d_vix, d_dxy, d_hy_spread,
    d_yield_curve, d_10y, d_2y, d_gold, d_oil
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_net_liquidity(fred_df: pd.DataFrame) -> pd.Series:
    """
    Compute the Net Liquidity Proxy:

        NetLiquidity = WALCL − WTREGEN − RRPONTSYD

    All inputs are in millions of USD (FRED native units).
    """
    required = {"WALCL", "WTREGEN", "RRPONTSYD"}
    missing = required - set(fred_df.columns)
    if missing:
        raise KeyError(f"Missing FRED columns for liquidity calc: {missing}")

    net_liq = fred_df["WALCL"] - fred_df["WTREGEN"] - fred_df["RRPONTSYD"]
    net_liq.name = "net_liquidity"
    return net_liq


def build_features(
    fred_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine FRED and market data, then produce stationary features.

    Returns a DataFrame with columns:
        net_liquidity, d_liquidity, d_sp500, d_vix, d_dxy,
        d_hy_spread, d_yield_curve, d_10y, d_2y, d_gold, d_oil

    Gold and oil are optional — missing columns are silently filled
    with zeros so that downstream code operates unchanged on v1
    model pipelines that were trained without these features.
    """
    # ── Normalise both indices to tz-naive date-only DatetimeIndex ───
    fred_df = fred_df.copy()
    fred_df.index = pd.to_datetime(fred_df.index.date if hasattr(fred_df.index, "date") else fred_df.index)

    market_df = market_df.copy()
    market_df.index = pd.to_datetime(market_df.index)

    # ── Merge on date ────────────────────────────────────────────
    combined = fred_df.join(market_df, how="outer").sort_index().ffill()

    # ── Net Liquidity ────────────────────────────────────────────
    combined["net_liquidity"] = compute_net_liquidity(combined)

    # ── Yield curve spread ───────────────────────────────────────
    combined["yield_curve"] = combined["DGS10"] - combined["DGS2"]

    # ── Stationary transformations ───────────────────────────────
    features = pd.DataFrame(index=combined.index)
    features["net_liquidity"] = combined["net_liquidity"]

    # Log returns for price-like series
    features["d_sp500"] = np.log(combined["sp500"]).diff()

    # First differences for level / spread series
    features["d_liquidity"] = combined["net_liquidity"].diff()
    features["d_vix"] = combined["vix"].diff() if "vix" in combined.columns else pd.Series(0.0, index=combined.index)
    features["d_dxy"] = combined["dxy"].diff()
    features["d_hy_spread"] = combined["BAMLH0A0HYM2"].diff()
    features["d_yield_curve"] = combined["yield_curve"].diff()
    features["d_10y"] = combined["DGS10"].diff()
    features["d_2y"] = combined["DGS2"].diff()

    # Optional commodity features — graceful fallback when tickers are absent.
    if "gold" in combined.columns and combined["gold"].notna().any():
        features["d_gold"] = np.log(combined["gold"].replace(0, np.nan)).diff()
    else:
        logger.warning("Gold data unavailable; d_gold set to 0.")
        features["d_gold"] = pd.Series(0.0, index=combined.index)

    if "oil" in combined.columns and combined["oil"].notna().any():
        features["d_oil"] = np.log(combined["oil"].replace(0, np.nan)).diff()
    else:
        logger.warning("Oil data unavailable; d_oil set to 0.")
        features["d_oil"] = pd.Series(0.0, index=combined.index)

    # Fill any remaining NaNs in optional columns with 0 before dropna
    # so that gaps in commodity data do not purge otherwise-complete rows.
    features["d_gold"] = features["d_gold"].fillna(0.0)
    features["d_oil"] = features["d_oil"].fillna(0.0)

    features = features.dropna()
    logger.info("Built feature matrix: %s", features.shape)
    return features


# Feature columns used for model input (excludes raw net_liquidity).
# v1 models were trained on the first 8 features only; v2 models use
# all 10 (including d_gold and d_oil).
MODEL_FEATURE_COLS: list[str] = [
    "d_liquidity",
    "d_sp500",
    "d_vix",
    "d_dxy",
    "d_hy_spread",
    "d_yield_curve",
    "d_10y",
    "d_2y",
    "d_gold",
    "d_oil",
]

# Legacy 8-feature set for backward compatibility with v1 artifacts.
MODEL_FEATURE_COLS_V1: list[str] = [
    "d_liquidity",
    "d_sp500",
    "d_vix",
    "d_dxy",
    "d_hy_spread",
    "d_yield_curve",
    "d_10y",
    "d_2y",
]
