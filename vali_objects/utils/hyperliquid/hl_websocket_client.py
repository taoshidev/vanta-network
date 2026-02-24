"""
Hyperliquid WebSocket Client — Subscribes to userFills for a single HL wallet.

Returns an async context that connects to the HL WebSocket API and invokes
a callback for each incoming fill.
"""
import asyncio
import json
import logging

import websockets

from vali_objects.vali_config import ValiConfig

logger = logging.getLogger(__name__)


async def subscribe_to_user_fills(
    hl_wallet_address: str,
    on_fill,
    *,
    ws_uri: str = None,
    shutdown_event: asyncio.Event = None,
):
    """
    Connect to the Hyperliquid WebSocket and subscribe to userFills.

    Args:
        hl_wallet_address: The HL wallet address to subscribe to.
        on_fill: Async or sync callable(fill_dict) invoked for each fill.
        ws_uri: Override the WebSocket URI (defaults to ValiConfig.HL_WEBSOCKET_URI).
        shutdown_event: An asyncio.Event that, when set, causes the connection to close.
    """
    uri = ws_uri or ValiConfig.HL_WEBSOCKET_URI
    subscribe_msg = json.dumps({
        "method": "subscribe",
        "subscription": {
            "type": "userFills",
            "user": hl_wallet_address,
        },
    })

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                await ws.send(subscribe_msg)
                logger.info(f"[HL_WS] Subscribed to userFills for {hl_wallet_address[:10]}...")

                async for raw_msg in ws:
                    if shutdown_event and shutdown_event.is_set():
                        logger.info(f"[HL_WS] Shutdown signalled for {hl_wallet_address[:10]}...")
                        return

                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning(f"[HL_WS] Non-JSON message: {raw_msg[:200]}")
                        continue

                    channel = msg.get("channel")
                    if channel != "userFills":
                        continue

                    fills = msg.get("data", [])
                    for fill in fills:
                        try:
                            if asyncio.iscoroutinefunction(on_fill):
                                await on_fill(fill)
                            else:
                                on_fill(fill)
                        except Exception:
                            logger.exception(f"[HL_WS] Error in on_fill callback for {hl_wallet_address[:10]}...")

        except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
            logger.warning(f"[HL_WS] Connection lost for {hl_wallet_address[:10]}...: {e}. Reconnecting in 5s...")
        except Exception:
            logger.exception(f"[HL_WS] Unexpected error for {hl_wallet_address[:10]}... Reconnecting in 10s...")
            await asyncio.sleep(5)  # Extra 5s on top of the normal 5s below

        if shutdown_event and shutdown_event.is_set():
            return
        await asyncio.sleep(5)
