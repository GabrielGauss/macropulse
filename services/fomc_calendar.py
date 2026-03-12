"""
FOMC meeting calendar for MacroPulse.

Provides hardcoded 2025 and 2026 FOMC meeting dates (last day of each
meeting, i.e. the date the policy decision is announced) along with
utility functions for querying proximity to meetings.

Source: Federal Reserve public calendar
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hardcoded FOMC decision dates ─────────────────────────────────────
# These are the *announcement* dates (last day of the two-day meeting).
# 2025 dates: 8 scheduled meetings.
# 2026 dates: 8 scheduled meetings.

_FOMC_DATES_2025: list[dt.date] = [
    dt.date(2025, 1, 29),   # Jan 28–29
    dt.date(2025, 3, 19),   # Mar 18–19
    dt.date(2025, 5, 7),    # May 6–7
    dt.date(2025, 6, 18),   # Jun 17–18
    dt.date(2025, 7, 30),   # Jul 29–30
    dt.date(2025, 9, 17),   # Sep 16–17
    dt.date(2025, 10, 29),  # Oct 28–29
    dt.date(2025, 12, 10),  # Dec 9–10
]

_FOMC_DATES_2026: list[dt.date] = [
    dt.date(2026, 1, 28),   # Jan 27–28
    dt.date(2026, 3, 18),   # Mar 17–18
    dt.date(2026, 4, 29),   # Apr 28–29
    dt.date(2026, 6, 17),   # Jun 16–17
    dt.date(2026, 7, 29),   # Jul 28–29
    dt.date(2026, 9, 16),   # Sep 15–16
    dt.date(2026, 10, 28),  # Oct 27–28
    dt.date(2026, 12, 9),   # Dec 8–9
]

# Combined sorted list of all known FOMC dates.
_ALL_FOMC_DATES: list[dt.date] = sorted(_FOMC_DATES_2025 + _FOMC_DATES_2026)

# Window (in calendar days) used by is_fomc_week.
_FOMC_PROXIMITY_DAYS: int = 3


def get_fomc_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    """
    Return all FOMC meeting dates within the inclusive [start, end] range.

    Parameters
    ----------
    start:
        Inclusive range start date.
    end:
        Inclusive range end date.

    Returns
    -------
    List of dates, sorted ascending.
    """
    result = [d for d in _ALL_FOMC_DATES if start <= d <= end]
    logger.info(
        "get_fomc_dates(%s, %s): %d meeting(s) found.", start, end, len(result)
    )
    return result


def is_fomc_week(date: dt.date) -> bool:
    """
    Return True if *date* falls within 3 calendar days of an FOMC meeting.

    The window is symmetric: 3 days before or after the announcement date
    are all considered "FOMC week".

    Parameters
    ----------
    date:
        The date to test.

    Returns
    -------
    bool
    """
    for fomc_date in _ALL_FOMC_DATES:
        if abs((date - fomc_date).days) <= _FOMC_PROXIMITY_DAYS:
            return True
    return False


def next_fomc(date: dt.date) -> Optional[dt.date]:
    """
    Return the next FOMC meeting date strictly after *date*.

    Returns None if no future meeting is in the hardcoded calendar.

    Parameters
    ----------
    date:
        Reference date.

    Returns
    -------
    dt.date or None
    """
    for fomc_date in _ALL_FOMC_DATES:
        if fomc_date > date:
            logger.info("next_fomc(%s) -> %s", date, fomc_date)
            return fomc_date
    logger.info("next_fomc(%s) -> None (past end of calendar)", date)
    return None
