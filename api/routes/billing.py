"""
Paddle billing endpoints for MacroPulse.

  POST /v1/billing/checkout          — create a Paddle checkout session (auth required)
  POST /v1/billing/portal            — get Paddle customer portal URL (auth required)
  POST /v1/billing/webhook           — Paddle webhook receiver (no auth, signature verified)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from api.auth import require_api_key
from config.settings import get_settings
from database import queries
from services.paddle import (
    create_checkout_url,
    create_portal_url,
    handle_webhook_event,
    verify_webhook,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["Billing"])


class CheckoutRequest(BaseModel):
    tier: str  # "starter" | "pro"


class CheckoutResponse(BaseModel):
    checkout_url: str
    tier: str


class PortalResponse(BaseModel):
    portal_url: str


def _price_id_for_tier(tier: str) -> str:
    settings = get_settings()
    mapping = {
        "starter": settings.paddle_starter_price_id,
        "pro":     settings.paddle_pro_price_id,
    }
    price_id = mapping.get(tier, "")
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Price ID for tier '{tier}' is not configured. Contact support.",
        )
    return price_id


@router.post(
    "/checkout",
    response_model=CheckoutResponse,
    summary="Create a Paddle checkout session",
)
def create_checkout(
    body: CheckoutRequest,
    key_record: dict = Depends(require_api_key),
) -> CheckoutResponse:
    """
    Returns a Paddle hosted checkout URL for upgrading to Starter or Pro.

    Redirect the user (or open in browser) to `checkout_url`.
    After payment, Paddle fires a webhook that upgrades the tier automatically.
    """
    tier = body.tier.lower()
    if tier not in ("starter", "pro"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tier must be 'starter' or 'pro'.",
        )

    current_tier = key_record.get("tier", "free")
    if current_tier == tier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"You are already on the {tier} plan.",
        )

    user_id: int = key_record["user_id"]
    email: str = key_record.get("email", "")
    price_id = _price_id_for_tier(tier)

    try:
        checkout_url = create_checkout_url(
            price_id=price_id,
            user_id=user_id,
            email=email,
            tier=tier,
        )
    except Exception as exc:
        logger.error("Paddle checkout error for user_id=%d: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not create checkout session. Please try again.",
        )

    return CheckoutResponse(checkout_url=checkout_url, tier=tier)


@router.post(
    "/portal",
    response_model=PortalResponse,
    summary="Get Paddle customer portal URL",
)
def get_portal(
    key_record: dict = Depends(require_api_key),
) -> PortalResponse:
    """
    Returns the Paddle customer portal URL so the user can manage or cancel
    their subscription directly.
    """
    user_id: int = key_record["user_id"]
    user = queries.get_user_by_id(user_id)

    if not user or not user.get("paddle_customer_id"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active Paddle subscription found for this account.",
        )

    try:
        portal_url = create_portal_url(user["paddle_customer_id"])
    except Exception as exc:
        logger.error("Paddle portal error for user_id=%d: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate portal URL. Please try again.",
        )

    return PortalResponse(portal_url=portal_url)


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="Paddle webhook receiver",
    include_in_schema=False,   # hide from public docs
)
async def paddle_webhook(
    request: Request,
    paddle_signature: str | None = Header(None, alias="Paddle-Signature"),
) -> dict:
    """
    Receives and processes Paddle billing events.

    Verifies the HMAC-SHA256 signature before processing.
    Always returns 200 so Paddle doesn't retry valid deliveries.
    """
    raw_body = await request.body()

    if paddle_signature is None:
        logger.warning("Paddle webhook received without signature header")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Paddle-Signature header.",
        )

    if not verify_webhook(raw_body, paddle_signature):
        logger.warning("Paddle webhook signature verification failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature.",
        )

    try:
        import json
        event = json.loads(raw_body)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body.",
        )

    try:
        result = handle_webhook_event(event)
        logger.info("Webhook processed: %s → %s", event.get("event_type"), result)
    except Exception as exc:
        # Log but return 200 — don't let Paddle retry on our DB errors
        logger.error("Webhook handler error: %s", exc, exc_info=True)

    return {"ok": True}
