"""
WebSocket endpoint for real-time regime signal streaming.

Clients connect to /ws/regime and receive JSON messages whenever
the pipeline produces a new regime result. The broadcast is
triggered by the pipeline via ``notify_regime_update()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])

# Simple in-process connection manager.  For horizontal scaling,
# replace with Redis pub/sub or a proper message broker.
_connections: set[WebSocket] = set()


@router.websocket("/ws/regime")
async def regime_stream(ws: WebSocket) -> None:
    """Accept a WebSocket and stream regime updates until disconnect."""
    await ws.accept()
    _connections.add(ws)
    logger.info("WS client connected (%d total)", len(_connections))
    try:
        while True:
            # Keep connection alive; client can send pings or requests.
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(ws)
        logger.info("WS client disconnected (%d remaining)", len(_connections))


async def broadcast_regime(payload: dict[str, Any]) -> None:
    """Push a regime update to every connected WebSocket client."""
    if not _connections:
        return
    message = json.dumps(payload, default=str)
    stale: list[WebSocket] = []
    for ws in _connections:
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _connections.discard(ws)


def notify_regime_update(payload: dict[str, Any]) -> None:
    """
    Synchronous wrapper for the pipeline to call after storing results.

    If an event loop is running (e.g. inside the API process), it
    schedules the broadcast. Otherwise it's a no-op (CLI pipeline mode).
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast_regime(payload))
    except RuntimeError:
        # No event loop – running from the CLI pipeline; skip WS broadcast.
        pass
