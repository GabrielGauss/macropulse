"""
Regime performance attribution for MacroPulse.

Computes return statistics per macro regime and a regime-following equity
curve vs buy-and-hold.  This is the primary sales/proof-of-value module.

Strategy logic:
  - expansion  → 100% long  (full risk-on)
  - recovery   → 75% long   (cautious long)
  - tightening → 25% long   (defensive)
  - risk_off   → 0% long    (cash/flat)

S&P 500 daily returns are fetched directly from yfinance so the attribution
is independent of any pipeline execution quirks.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import yfinance as yf

from database import queries

logger = logging.getLogger(__name__)

# Regime → equity exposure fraction
_EXPOSURE: dict[str, float] = {
    "expansion":  1.00,
    "recovery":   0.75,
    "tightening": 0.25,
    "risk_off":   0.00,
}

_ALL_REGIMES = ["expansion", "recovery", "tightening", "risk_off"]


def _fetch_spx_returns(start: dt.date, end: dt.date) -> pd.Series:
    """
    Fetch SPX daily close-to-close percentage returns from yfinance.
    Returns a Series indexed by date (tz-naive).
    """
    ticker = yf.Ticker("^GSPC")
    raw = ticker.history(
        start=start - dt.timedelta(days=5),  # buffer for return calc
        end=end + dt.timedelta(days=1),
        auto_adjust=True,
    )
    if raw.empty:
        return pd.Series(dtype=float)

    close = raw["Close"]
    # Strip timezone to date-only index
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index = pd.to_datetime(close.index.date)

    returns = close.pct_change().dropna()
    return returns


def _fetch_aligned(lookback_days: int) -> pd.DataFrame | None:
    """
    Join regime history with live SPX returns on date index.
    Returns None if insufficient data.
    """
    history = queries.fetch_regime_history(limit=lookback_days)
    if not history:
        return None

    reg_df = pd.DataFrame(list(reversed(history))).set_index("time").sort_index()
    reg_df.index = pd.to_datetime(
        reg_df.index.date if hasattr(reg_df.index, "date") else reg_df.index
    )

    start_date = reg_df.index[0].date() if hasattr(reg_df.index[0], "date") else reg_df.index[0]
    end_date   = reg_df.index[-1].date() if hasattr(reg_df.index[-1], "date") else reg_df.index[-1]

    spx_rets = _fetch_spx_returns(start_date, end_date)
    if spx_rets.empty:
        return None

    combined = reg_df[["regime", "risk_score"]].join(
        spx_rets.rename("d_sp500"), how="inner"
    )
    combined = combined.dropna(subset=["d_sp500"])

    if len(combined) < 10:
        return None

    return combined


def compute_regime_performance(lookback_days: int = 252) -> dict[str, Any]:
    """
    Compute full regime performance attribution.

    Returns
    -------
    dict with:
      regime_stats     – per-regime return/risk breakdown
      strategy         – aggregate regime-following strategy stats
      buy_and_hold     – passive SPX stats for comparison
      equity_curve     – dates + cumulative returns for charting
      observation_days – total trading days in sample
    """
    combined = _fetch_aligned(lookback_days)
    if combined is None:
        return {"error": "Insufficient data for performance attribution"}

    # ── Per-regime statistics ─────────────────────────────────────
    regime_stats: dict[str, Any] = {}
    for regime in _ALL_REGIMES:
        mask = combined["regime"] == regime
        rets = combined.loc[mask, "d_sp500"]
        n = int(mask.sum())

        if n < 5:
            regime_stats[regime] = {
                "observation_days": n,
                "avg_daily_return_pct": None,
                "annualized_return_pct": None,
                "sharpe_ratio": None,
                "win_rate_pct": None,
            }
            continue

        avg_daily = float(rets.mean())
        ann_ret   = avg_daily * 252
        ann_vol   = float(rets.std()) * (252 ** 0.5)
        sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
        win_rate  = float((rets > 0).mean())

        regime_stats[regime] = {
            "observation_days": n,
            "avg_daily_return_pct": round(avg_daily * 100, 4),
            "annualized_return_pct": round(ann_ret * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_pct": round(win_rate * 100, 1),
        }

    # ── Strategy vs buy-and-hold ──────────────────────────────────
    combined["exposure"]  = combined["regime"].map(_EXPOSURE).fillna(0.5)
    combined["strat_ret"] = combined["exposure"] * combined["d_sp500"]
    combined["bnh_ret"]   = combined["d_sp500"]

    # Cumulative return index (starts at 1.0)
    combined["strat_cum"] = (1 + combined["strat_ret"]).cumprod()
    combined["bnh_cum"]   = (1 + combined["bnh_ret"]).cumprod()

    def _stats(ret_col: str, cum_col: str) -> dict[str, float]:
        rets    = combined[ret_col]
        cum     = combined[cum_col]
        ann_ret = float(rets.mean() * 252)
        ann_vol = float(rets.std() * (252 ** 0.5))
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
        roll_max = cum.cummax()
        max_dd   = float(((cum - roll_max) / roll_max).min())
        total_r  = float(cum.iloc[-1] - 1)
        return {
            "annualized_return_pct":     round(ann_ret * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio":              round(sharpe, 3),
            "max_drawdown_pct":          round(max_dd * 100, 2),
            "total_return_pct":          round(total_r * 100, 2),
        }

    strategy_stats = _stats("strat_ret", "strat_cum")
    bnh_stats      = _stats("bnh_ret",   "bnh_cum")

    alpha = strategy_stats["annualized_return_pct"] - bnh_stats["annualized_return_pct"]
    strategy_stats["alpha_vs_bnh_pct"] = round(alpha, 2)

    # ── Equity curve for charting ─────────────────────────────────
    dates = [
        v.date().isoformat() if hasattr(v, "date") else str(v)
        for v in combined.index
    ]

    return {
        "lookback_days":    lookback_days,
        "observation_days": len(combined),
        "regime_stats":     regime_stats,
        "strategy":         strategy_stats,
        "buy_and_hold":     bnh_stats,
        "equity_curve": {
            "dates":        dates,
            "strategy":     [round(float(v), 4) for v in combined["strat_cum"]],
            "buy_and_hold": [round(float(v), 4) for v in combined["bnh_cum"]],
        },
    }
