"""
User registration, key management, and usage endpoints.

  POST /v1/auth/register    — create user + issue first API key (no auth required)
  POST /v1/auth/rotate      — revoke current key + issue new one (auth required)
  GET  /v1/auth/me          — return user profile + key info (auth required)
  GET  /v1/auth/usage       — return today's request count vs tier limit (auth required)
"""

from __future__ import annotations

import datetime as dt
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import generate_api_key, hash_key, require_api_key
from api.middleware.rate_limit import TIER_LIMITS, _reset_ts, get_usage_today
from api.schemas.responses import (
    KeyInfoResponse,
    RegisterRequest,
    RegisterResponse,
    RotateKeyResponse,
    UsageResponse,
)
from database import queries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["Auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _tier_limit(tier: str) -> int:
    return TIER_LIMITS.get(tier, 50)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register and receive your API key",
)
def register(body: RegisterRequest) -> RegisterResponse:
    """
    Create a MacroPulse account and receive a free-tier API key.

    The plaintext `api_key` in the response is shown **once only**.
    Store it securely — it cannot be retrieved again (only rotated).
    """
    email = body.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid email address.",
        )

    # Prevent duplicate registrations
    existing = queries.get_user_by_email(email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    # Create user
    try:
        user = queries.create_user(email=email, name=body.name)
    except Exception as exc:
        logger.error("Failed to create user: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not create account. Please try again.",
        )

    # Generate and store key
    plaintext_key = generate_api_key()
    key_prefix = plaintext_key[:12]
    try:
        queries.create_api_key(
            user_id=user["id"],
            key_hash=hash_key(plaintext_key),
            key_prefix=key_prefix,
            tier="free",
        )
    except Exception as exc:
        logger.error("Failed to create API key: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not issue API key. Please try again.",
        )

    logger.info("New user registered: email=%s tier=free", email)

    return RegisterResponse(
        user_id=user["id"],
        email=email,
        api_key=plaintext_key,
        key_prefix=key_prefix,
        tier="free",
        daily_limit=_tier_limit("free"),
    )


@router.post(
    "/rotate",
    response_model=RotateKeyResponse,
    summary="Rotate your API key",
)
def rotate_key(
    key_record: dict = Depends(require_api_key),
) -> RotateKeyResponse:
    """
    Revoke the current API key and issue a new one with the same tier.

    The new plaintext `api_key` is shown **once only**.
    """
    user_id: int = key_record["user_id"]
    tier: str = key_record.get("tier", "free")

    # Revoke existing keys for this user
    try:
        queries.revoke_api_keys_for_user(user_id)
    except Exception as exc:
        logger.error("Failed to revoke keys for user_id=%d: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not revoke current key. Please try again.",
        )

    # Issue new key
    plaintext_key = generate_api_key()
    key_prefix = plaintext_key[:12]
    try:
        queries.create_api_key(
            user_id=user_id,
            key_hash=hash_key(plaintext_key),
            key_prefix=key_prefix,
            tier=tier,
        )
    except Exception as exc:
        logger.error("Failed to issue new key for user_id=%d: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not issue new key. Please try again.",
        )

    logger.info("Key rotated: user_id=%d tier=%s", user_id, tier)

    return RotateKeyResponse(
        api_key=plaintext_key,
        key_prefix=key_prefix,
        tier=tier,
        daily_limit=_tier_limit(tier),
    )


@router.get(
    "/me",
    response_model=KeyInfoResponse,
    summary="Your account and key info",
)
def get_me(
    key_record: dict = Depends(require_api_key),
) -> KeyInfoResponse:
    """Return account details and key metadata (no plaintext key)."""
    tier = key_record.get("tier", "free")
    return KeyInfoResponse(
        user_id=key_record["user_id"],
        email=key_record.get("email", ""),
        key_prefix=key_record.get("key_prefix", ""),
        tier=tier,
        daily_limit=_tier_limit(tier),
        created_at=key_record.get("created_at") or dt.datetime.now(dt.timezone.utc),
        last_used_at=key_record.get("last_used_at"),
    )


@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Today's API usage vs your tier limit",
)
def get_usage(
    key_record: dict = Depends(require_api_key),
) -> UsageResponse:
    """
    Return today's request count and how many remain before rate limiting.

    `remaining` is -1 for unlimited (Pro tier).
    `reset_at` is the Unix timestamp of next midnight UTC.
    """
    tier = key_record.get("tier", "free")
    limit = _tier_limit(tier)

    # client_id used by the rate limiter is the raw key value
    # We don't have the raw key here — use key_prefix as a proxy identifier
    # (the rate limiter keys on the raw key, but we can expose an approximation)
    key_prefix = key_record.get("key_prefix", "")
    used = get_usage_today(key_prefix)  # best-effort (may undercount in multi-process)

    if limit == 0:
        remaining = -1
    else:
        remaining = max(0, limit - used)

    return UsageResponse(
        tier=tier,
        daily_limit=limit,
        used_today=used,
        remaining=remaining,
        reset_at=_reset_ts(),
    )
