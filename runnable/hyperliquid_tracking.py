#!/usr/bin/env python3
"""
Subscribe to *all* Hyperliquid perp trade streams (one subscription per coin)
and print trades as they arrive.

Docs:
- WebSocket URL + basic example: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket  :contentReference[oaicite:0]{index=0}
- Subscription message format + trades subscription: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions :contentReference[oaicite:1]{index=1}
- Heartbeats: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/timeouts-and-heartbeats :contentReference[oaicite:2]{index=2}
- WebSocket limits: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits :contentReference[oaicite:3]{index=3}

Install:
  pip install websockets requests
Run:
  python hl_all_trades.py
  python hl_all_trades.py --testnet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import requests
import websockets
from websockets.client import WebSocketClientProtocol

MAINNET_WS = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"
MAINNET_INFO = "https://api.hyperliquid.xyz/info"
TESTNET_INFO = "https://api.hyperliquid-testnet.xyz/info"


def fetch_perp_universe(info_url: str, timeout_s: float = 10.0) -> List[str]:
    """
    Fetch perp coin symbols from the /info meta response.

    Note: Hyperliquid docs indicate perp `coin` values are the names returned in the `meta` response. :contentReference[oaicite:4]{index=4}
    The `meta` request is commonly done via POST /info with {"type":"meta"}.
    """
    r = requests.post(info_url, json={"type": "meta"}, timeout=timeout_s)
    r.raise_for_status()
    meta = r.json()
    # Expected shape: {"universe":[{"name":"BTC", ...}, ...], ...}
    universe = meta.get("universe", [])
    coins: List[str] = []
    for a in universe:
        name = a.get("name")
        if isinstance(name, str) and name:
            coins.append(name)
    return coins


async def heartbeat(ws: WebSocketClientProtocol, interval_s: float = 30.0) -> None:
    """
    Send ping messages to keep the connection alive.
    Server closes idle connections if it hasn't sent a message to it in ~60s;
    client can send {"method":"ping"} and receive {"channel":"pong"}. :contentReference[oaicite:5]{index=5}
    """
    while True:
        await asyncio.sleep(interval_s)
        try:
            await ws.send(json.dumps({"method": "ping"}))
        except Exception:
            return


async def subscribe_trades(ws: WebSocketClientProtocol, coins: List[str]) -> None:
    """
    Subscribe to trades for each coin:
      {"method":"subscribe","subscription":{"type":"trades","coin":"<coin_symbol>"}} :contentReference[oaicite:6]{index=6}
    """
    for coin in coins:
        msg = {"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}
        await ws.send(json.dumps(msg))

    # Reminder: there is a maximum of 1000 websocket subscriptions per IP. :contentReference[oaicite:7]{index=7}


async def run_stream(ws_url: str, info_url: str) -> None:
    coins = fetch_perp_universe(info_url)
    if not coins:
        raise RuntimeError("No coins returned from /info meta; cannot subscribe.")

    print(f"Will subscribe to trades for {len(coins)} perp coins.")
    # Simple reconnect loop with backoff
    backoff_s = 1.0

    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=None) as ws:
                print(f"Connected: {ws_url}")
                backoff_s = 1.0

                hb_task = asyncio.create_task(heartbeat(ws))
                await subscribe_trades(ws, coins)

                async for raw in ws:
                    # Messages are JSON text frames in examples. :contentReference[oaicite:8]{index=8}
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        print(raw)
                        continue

                    # Print everything; trades typically come with channel/data.
                    print(json.dumps(msg, separators=(",", ":"), ensure_ascii=False))

                hb_task.cancel()

        except Exception as e:
            print(f"Disconnected/error: {e!r}")
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)

ws_url = TESTNET_WS
info_url = TESTNET_INFO
await run_stream(ws_url, info_url)

