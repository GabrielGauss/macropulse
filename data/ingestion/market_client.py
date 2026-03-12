"""
Market data ingestion via Yahoo Finance.

Pulls S&P 500, VIX, DXY, Gold, and Crude Oil daily data.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Tickers used for market signal features.
_MARKET_TICKERS: dict[str, str] = {
    "^GSPC": "sp500",
    "^VIX": "vix",
    "DX-Y.NYB": "dxy",
    "GC=F": "gold",
    "CL=F": "oil",
}


def fetch_market_data(
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for market signal tickers.

    Returns a DataFrame indexed by date with columns:
    sp500, vix, dxy, gold, oil.
    Gold (GC=F) and oil (CL=F) may have gaps on non-trading days;
    forward-fill is applied after download.
    """
    tickers = list(_MARKET_TICKERS.keys())
    raw: pd.DataFrame = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()

    close = close.rename(columns=_MARKET_TICKERS)

    # yfinance returns a tz-aware DatetimeIndex (UTC). Strip tz so it aligns
    # cleanly with FRED's tz-naive index when joining.
    if close.index.tz is not None:
        close.index = close.index.tz_convert("UTC").tz_localize(None)

    # Normalize to date-only (no time component) to match FRED index granularity.
    close.index = pd.to_datetime(close.index.date)
    close.index.name = "date"
    close = close.sort_index().ffill()

    # Log missing tickers (e.g. VIX cache lock) but don't abort.
    for col in _MARKET_TICKERS.values():
        if col not in close.columns or close[col].isna().all():
            logger.warning("Market column '%s' is missing or all-NaN after download.", col)

    logger.info("Fetched market data (%d rows)", len(close))
    return close
