"""
Alerting service for MacroPulse.

Sends notifications when:
  • The macro regime changes (expansion → tightening, etc.)
  • Risk score crosses a threshold
  • Drift metrics exceed warning levels

Supports SMTP email and generic webhook (Slack / Discord / Teams).
"""

from __future__ import annotations

import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from urllib.request import Request, urlopen

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _send_email(subject: str, body_html: str) -> None:
    """Send an HTML email via SMTP."""
    settings = get_settings()
    if not settings.smtp_host or not settings.alert_recipients:
        logger.debug("SMTP not configured; skipping email alert.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_user
    msg["To"] = ", ".join(settings.alert_recipients)
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            if settings.smtp_user and settings.smtp_password:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(
                settings.smtp_user, settings.alert_recipients, msg.as_string()
            )
        logger.info("Email alert sent: %s", subject)
    except Exception:
        logger.error("Failed to send email alert", exc_info=True)


def _send_webhook(payload: dict[str, Any]) -> None:
    """POST a JSON payload to the configured webhook URL."""
    settings = get_settings()
    if not settings.webhook_url:
        logger.debug("Webhook not configured; skipping.")
        return

    data = json.dumps(payload).encode()
    req = Request(
        settings.webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            logger.info("Webhook sent (%d): %s", resp.status, payload.get("text", ""))
    except Exception:
        logger.error("Webhook delivery failed", exc_info=True)


# ── Public alert triggers ────────────────────────────────────────────


def alert_regime_change(
    previous: str,
    current: str,
    risk_score: float,
    probabilities: dict[str, float],
    timestamp: str,
) -> None:
    """Fire alerts when the macro regime transitions."""
    subject = f"[MacroPulse] Regime Change: {previous} → {current}"

    probs_str = " | ".join(
        f"{k}: {v:.0%}" for k, v in sorted(probabilities.items())
    )
    body = f"""
    <div style="font-family: monospace; max-width:600px;">
        <h2 style="color:#e63946;">Macro Regime Change Detected</h2>
        <table style="border-collapse:collapse; width:100%;">
            <tr><td style="padding:6px; font-weight:bold;">Timestamp</td>
                <td style="padding:6px;">{timestamp}</td></tr>
            <tr><td style="padding:6px; font-weight:bold;">Previous</td>
                <td style="padding:6px;">{previous}</td></tr>
            <tr><td style="padding:6px; font-weight:bold;">Current</td>
                <td style="padding:6px;"><strong>{current}</strong></td></tr>
            <tr><td style="padding:6px; font-weight:bold;">Risk Score</td>
                <td style="padding:6px;">{risk_score}</td></tr>
            <tr><td style="padding:6px; font-weight:bold;">Probabilities</td>
                <td style="padding:6px;">{probs_str}</td></tr>
        </table>
    </div>
    """
    _send_email(subject, body)

    _send_webhook({
        "text": f"🔄 *MacroPulse Regime Change*\n"
                f"`{previous}` → `{current}` | Risk: {risk_score}\n"
                f"{probs_str}",
        "regime": current,
        "risk_score": risk_score,
        "timestamp": timestamp,
    })
    logger.info("Regime change alert dispatched: %s → %s", previous, current)


def alert_drift_warning(
    metric_name: str,
    value: float,
    threshold: float,
    timestamp: str,
) -> None:
    """Fire alerts when drift metrics exceed warning thresholds."""
    subject = f"[MacroPulse] Drift Warning: {metric_name} = {value:.4f}"
    body = f"""
    <div style="font-family: monospace;">
        <h2 style="color:#f77f00;">Model Drift Warning</h2>
        <p><strong>{metric_name}</strong> has exceeded the threshold.</p>
        <p>Value: {value:.4f} | Threshold: {threshold:.4f}</p>
        <p>Timestamp: {timestamp}</p>
        <p>Consider retraining the model.</p>
    </div>
    """
    _send_email(subject, body)
    _send_webhook({
        "text": f"⚠️ *MacroPulse Drift Warning*\n"
                f"`{metric_name}` = {value:.4f} (threshold: {threshold:.4f})",
        "timestamp": timestamp,
    })
