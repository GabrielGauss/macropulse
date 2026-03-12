"""
Per-API-key tier-aware rate limiting middleware for MacroPulse.

Daily limits by tier:
  free     →  50  requests / day
  starter  →  500 requests / day
  pro      →  0   (unlimited)

Limits are looked up from the database by key hash and cached in memory
for the duration of the calendar day.  The cache is cheap to rebuild —
on restart or tier upgrade, the new limit takes effect immediately.

Set RATE_LIMIT_PER_DAY=0 in .env to bypass globally (dev / internal use).

Response headers (RFC 6585 / de-facto standard):
  X-RateLimit-Limit      daily cap for this key
  X-RateLimit-Remaining  requests left today
  X-RateLimit-Reset      Unix timestamp of next window reset (midnight UTC)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections import defaultdict

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths that are never rate-limited (health, public demo, docs, auth)
_EXEMPT_PATHS = {
    "/health", "/docs", "/openapi.json", "/redoc", "/dashboard",
    "/v1/auth/register",
}

# Tier → daily request cap  (0 = unlimited)
TIER_LIMITS: dict[str, int] = {
    "free":    50,
    "starter": 500,
    "pro":     0,
}

# In-memory store: client_id → (date_string, request_count)
_counters: dict[str, tuple[str, int]] = defaultdict(lambda: ("", 0))

# Per-key limit cache: key_hash → (date_string, daily_limit)
# Refreshed once per day so tier upgrades take effect within 24h.
_limit_cache: dict[str, tuple[str, int]] = {}


def _reset_ts() -> int:
    """Unix timestamp of next midnight UTC."""
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    return int(dt.datetime.combine(tomorrow, dt.time(), tzinfo=dt.timezone.utc).timestamp())


def _resolve_limit(raw_key: str | None, default_limit: int) -> int:
    """
    Return the daily limit for a given raw API key.

    Looks up the tier from the DB (cached per calendar day).
    Falls back to `default_limit` when the key is absent or on any error.
    """
    if not raw_key:
        return default_limit

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    today = dt.date.today().isoformat()
    cached = _limit_cache.get(key_hash)
    if cached and cached[0] == today:
        return cached[1]

    try:
        from database.queries import get_api_key_by_hash
        record = get_api_key_by_hash(key_hash)
        if record:
            tier = record.get("tier", "free")
            limit = TIER_LIMITS.get(tier, default_limit)
        else:
            limit = default_limit
    except Exception:
        limit = default_limit

    _limit_cache[key_hash] = (today, limit)
    return limit


def get_usage_today(client_id: str) -> int:
    """Return the request count for a client_id for today (used by /auth/usage)."""
    today = dt.date.today().isoformat()
    date_str, count = _counters[client_id]
    return count if date_str == today else 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-API-key tier-aware daily request cap."""

    def __init__(self, app, limit_per_day: int = 0) -> None:
        super().__init__(app)
        self.default_limit = limit_per_day  # 0 = global bypass (dev mode)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Path-based exemptions
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        raw_key = (
            request.headers.get("X-MacroPulse-Key")
            or request.query_params.get("api_key")
        )
        client_id = raw_key or (request.client.host if request.client else "unknown")

        # Resolve this key's daily limit (tier-aware)
        limit = _resolve_limit(raw_key, self.default_limit)

        # 0 = unlimited (pro tier or global bypass)
        if limit == 0:
            return await call_next(request)

        today = dt.date.today().isoformat()
        date_str, count = _counters[client_id]
        if date_str != today:
            count = 0

        reset = _reset_ts()

        if count >= limit:
            logger.warning(
                "Rate limit exceeded: client=%s count=%d limit=%d", client_id, count, limit
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": (
                        f"Daily limit of {limit} requests reached. "
                        "Upgrade your plan or wait until midnight UTC."
                    ),
                    "upgrade_url": "https://macropulse.io/pricing",
                    "reset_at": reset,
                },
                headers={
                    "X-RateLimit-Limit":     str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset":     str(reset),
                    "Retry-After": str(reset - int(
                        dt.datetime.now(dt.timezone.utc).timestamp()
                    )),
                },
            )

        count += 1
        _counters[client_id] = (today, count)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - count)
        response.headers["X-RateLimit-Reset"]     = str(reset)
        return response
