"""
API key authentication for MacroPulse.

Supports both header-based (X-MacroPulse-Key) and query-param (?api_key=) auth.

Keys are verified against the database (SHA-256 hash comparison).
Dev-mode bypass: if no rows exist in api_keys AND settings.api_keys is empty,
all requests are allowed (useful for local development).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from config.settings import get_settings

logger = logging.getLogger(__name__)

_header_scheme = APIKeyHeader(name="X-MacroPulse-Key", auto_error=False)
_query_scheme = APIKeyQuery(name="api_key", auto_error=False)


def hash_key(key: str) -> str:
    """Return the SHA-256 hex digest of a plaintext API key."""
    return hashlib.sha256(key.encode()).hexdigest()


def _lookup_key(raw_key: str) -> dict[str, Any] | None:
    """
    Look up a key in the database.  Returns the DB row or None.

    Import is deferred to avoid circular imports at module load time.
    """
    from database.queries import get_api_key_by_hash
    return get_api_key_by_hash(hash_key(raw_key))


async def require_api_key(
    header_key: Annotated[str | None, Security(_header_scheme)] = None,
    query_key: Annotated[str | None, Security(_query_scheme)] = None,
) -> dict[str, Any]:
    """
    FastAPI dependency that enforces API key auth against the DB.

    Returns the key record dict so downstream handlers can access
    `tier`, `user_id`, `email`, etc.

    Dev-mode: if settings.api_keys is empty and no DB keys exist,
    all requests pass through (returns a synthetic dev record).
    """
    settings = get_settings()
    raw_key = header_key or query_key

    # Dev-mode: no keys configured anywhere → allow all
    if not settings.api_keys and raw_key is None:
        try:
            from database.queries import get_api_key_by_hash as _check
            # If we can't reach the DB, or no rows exist, stay in dev mode
            _check("__probe__")
        except Exception:
            pass
        return {
            "user_id": 0,
            "email": "dev@localhost",
            "tier": "pro",
            "key_prefix": "dev-mode",
            "is_active": True,
        }

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via X-MacroPulse-Key header or api_key query param.",
        )

    # Legacy env-key support (settings.api_keys list) so existing deployments
    # don't break before the DB is migrated.
    if settings.api_keys and raw_key in settings.api_keys:
        return {
            "user_id": 0,
            "email": "legacy@env",
            "tier": "pro",
            "key_prefix": raw_key[:12],
            "is_active": True,
        }

    # Primary path: DB lookup
    try:
        record = _lookup_key(raw_key)
    except Exception as exc:
        logger.error("DB error during key lookup: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service temporarily unavailable.",
        )

    if record is None:
        logger.warning("Rejected invalid API key prefix=%s…", raw_key[:8])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API key.",
        )

    # Fire-and-forget last_used update (don't fail the request if this errors)
    try:
        from database.queries import touch_api_key
        touch_api_key(hash_key(raw_key))
    except Exception:
        pass

    return dict(record)


def generate_api_key() -> str:
    """Generate a cryptographically secure MacroPulse API key."""
    return f"mp_{secrets.token_urlsafe(32)}"
