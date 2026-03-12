"""
FRED data ingestion for MacroPulse.

Pulls macroeconomic series from the Federal Reserve Economic Data API.
Adds TTL caching (1-hour) and exponential-backoff retry so the pipeline
is robust to transient API failures and doesn't hammer the FRED endpoint.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd
from fredapi import Fred

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ── In-process TTL cache ─────────────────────────────────────────────
# Maps cache_key → (epoch_timestamp, pd.Series).
_series_cache: dict[str, tuple[float, pd.Series]] = {}
_CACHE_TTL: int = 3_600   # seconds (1 hour)
_MAX_RETRIES: int = 3


def _get_fred_client() -> Fred:
    settings = get_settings()
    if not settings.fred_api_key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. "
            "Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=settings.fred_api_key)


def fetch_fred_series(
    series_id: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch a single FRED series with TTL caching and exponential-backoff retry.

    Parameters
    ----------
    series_id  : FRED series identifier (e.g. 'WALCL').
    start / end: Optional date range window.
    use_cache  : Set False to force a live fetch (e.g. during retraining).
    """
    cache_key = f"{series_id}:{start}:{end}"

    if use_cache and cache_key in _series_cache:
        cached_at, cached_data = _series_cache[cache_key]
        if time.time() - cached_at < _CACHE_TTL:
            logger.debug("Cache hit for FRED series %s", series_id)
            return cached_data
        del _series_cache[cache_key]

    fred = _get_fred_client()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            data: pd.Series = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            data.name = series_id
            data.index.name = "date"
            _series_cache[cache_key] = (time.time(), data)
            logger.info("Fetched FRED series %s  (%d observations)", series_id, len(data))
            return data
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "FRED fetch attempt %d/%d for %s failed (%s). Retrying in %ds…",
                attempt + 1, _MAX_RETRIES, series_id, exc, wait,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch FRED series {series_id} after {_MAX_RETRIES} attempts: {last_exc}"
    )


def invalidate_cache(series_id: str | None = None) -> None:
    """Clear the in-process TTL cache (all series or a specific one)."""
    if series_id is None:
        _series_cache.clear()
        logger.info("FRED cache cleared (all series).")
    else:
        removed = [k for k in list(_series_cache) if k.startswith(f"{series_id}:")]
        for k in removed:
            del _series_cache[k]
        logger.info("FRED cache cleared for %s (%d keys).", series_id, len(removed))


def fetch_all_fred(
    start: dt.date | None = None,
    end: dt.date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch all required FRED series and return a single merged DataFrame.

    Columns: WALCL, RRPONTSYD, WTREGEN, DGS10, DGS2, BAMLH0A0HYM2
    Missing rows are forward-filled (FRED publishes on different schedules).
    """
    settings = get_settings()
    frames: list[pd.Series] = []
    for sid in settings.fred_series:
        try:
            s = fetch_fred_series(sid, start=start, end=end, use_cache=use_cache)
            frames.append(s)
        except Exception:
            logger.warning("Could not fetch FRED series %s", sid, exc_info=True)

    if not frames:
        raise RuntimeError("No FRED data could be fetched.")

    df = pd.concat(frames, axis=1)
    df = df.sort_index().ffill()
    return df
