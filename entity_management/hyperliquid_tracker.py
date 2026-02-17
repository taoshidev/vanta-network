# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
HyperliquidTracker - Daemon service that tracks Hyperliquid trader fills
and forwards them as Vanta signals through the existing pipeline.

Runs as a daemon thread in the validator process, maintaining a single
WebSocket connection to Hyperliquid mainnet and subscribing to userFills
for each registered HL subaccount address (max 10 per HL WS limits).

Architecture:
- Own asyncio event loop in a daemon thread
- Single WebSocket connection with heartbeat and reconnection
- Periodic refresh of subscribed addresses (every 60s)
- Fill dedup via bounded hash set
- Converts fills to market orders via OrderProcessor.process_order()
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import bittensor as bt

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None
    WebSocketClientProtocol = None

from entity_management.entity_client import EntityClient
from shared_objects.rate_limiter import RateLimiter
from time_util.time_util import TimeUtil
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from vali_objects.vali_config import ValiConfig, TradePair, TRADE_PAIR_ID_TO_TRADE_PAIR


class HyperliquidTracker:
    """
    Tracks Hyperliquid trader fills via WebSocket and forwards them as Vanta signals.

    Runs in a daemon thread with its own asyncio event loop.
    """

    # Max fill hashes to track for dedup (bounded to prevent memory growth)
    MAX_DEDUP_HASHES = 50_000
    # How often to refresh the list of subscribed addresses (seconds)
    ADDRESS_REFRESH_INTERVAL_S = 60.0

    def __init__(
        self,
        entity_client: EntityClient,
        elimination_client,
        price_fetcher_client,
        asset_selection_client,
        market_order_manager,
        limit_order_client,
        uuid_tracker,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self._entity_client = entity_client
        self._elimination_client = elimination_client
        self._price_fetcher_client = price_fetcher_client
        self._asset_selection_client = asset_selection_client
        self._market_order_manager = market_order_manager
        self._limit_order_client = limit_order_client
        self._uuid_tracker = uuid_tracker
        self._rate_limiter = rate_limiter or RateLimiter()

        # State
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

        # Currently subscribed HL addresses (synced with entity manager)
        self._subscribed_addresses: Set[str] = set()

        # Dedup: ordered dict of fill_hash -> True (bounded, oldest evicted first)
        self._processed_hashes: OrderedDict[str, bool] = OrderedDict()

        # Metrics
        self._connected = False
        self._fills_processed = 0
        self._last_fill_time: Optional[float] = None

    # ==================== Lifecycle ====================

    def start(self):
        """Start the tracker in a daemon thread."""
        if websockets is None:
            bt.logging.warning("[HL_TRACKER] websockets library not installed - HL tracking disabled")
            return

        if self._thread and self._thread.is_alive():
            bt.logging.warning("[HL_TRACKER] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="hl-tracker")
        self._thread.start()
        bt.logging.info("[HL_TRACKER] Started daemon thread")

    def stop(self):
        """Signal the tracker to stop."""
        self._stop_event.set()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        bt.logging.info("[HL_TRACKER] Stopped")

    def get_status(self) -> dict:
        """Get tracker status for health monitoring."""
        return {
            "connected": self._connected,
            "subscribed_addresses": len(self._subscribed_addresses),
            "fills_processed": self._fills_processed,
            "last_fill_time": self._last_fill_time,
        }

    # ==================== Thread Entry ====================

    def _run_loop(self):
        """Entry point for the daemon thread - runs asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_stream())
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Event loop crashed: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self._loop.close()

    # ==================== WebSocket Stream ====================

    async def _run_stream(self):
        """Main WebSocket loop with reconnection and exponential backoff."""
        backoff_s = 1.0

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    ValiConfig.HL_MAINNET_WS, ping_interval=None
                ) as ws:
                    bt.logging.info(f"[HL_TRACKER] Connected to {ValiConfig.HL_MAINNET_WS}")
                    self._connected = True
                    backoff_s = 1.0

                    # Start heartbeat and periodic refresh tasks
                    hb_task = asyncio.create_task(self._heartbeat(ws))
                    refresh_task = asyncio.create_task(self._periodic_refresh(ws))

                    # Subscribe to all current HL addresses
                    await self._subscribe_all(ws)

                    # Process messages
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self._handle_message(msg)

                    hb_task.cancel()
                    refresh_task.cancel()

            except Exception as e:
                bt.logging.warning(f"[HL_TRACKER] Disconnected/error: {e!r}")
            finally:
                self._connected = False

            if self._stop_event.is_set():
                break

            bt.logging.info(f"[HL_TRACKER] Reconnecting in {backoff_s:.1f}s...")
            await asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, ValiConfig.HL_WS_RECONNECT_BACKOFF_MAX_S)

    async def _heartbeat(self, ws: WebSocketClientProtocol):
        """Send ping messages to keep the connection alive."""
        while True:
            await asyncio.sleep(ValiConfig.HL_WS_HEARTBEAT_INTERVAL_S)
            try:
                await ws.send(json.dumps({"method": "ping"}))
            except Exception:
                return

    async def _subscribe_all(self, ws: WebSocketClientProtocol):
        """Subscribe to userFills for all active HL addresses."""
        try:
            hl_subaccounts = self._entity_client.get_all_active_hl_subaccounts()
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Failed to get HL subaccounts: {e}")
            return

        new_addresses = set()
        for hl_address, _info in hl_subaccounts:
            new_addresses.add(hl_address)
            if hl_address not in self._subscribed_addresses:
                msg = {
                    "method": "subscribe",
                    "subscription": {"type": "userFills", "user": hl_address}
                }
                try:
                    await ws.send(json.dumps(msg))
                    bt.logging.info(f"[HL_TRACKER] Subscribed to userFills for {hl_address}")
                except Exception as e:
                    bt.logging.error(f"[HL_TRACKER] Failed to subscribe for {hl_address}: {e}")

        # Unsubscribe from removed addresses
        for old_addr in self._subscribed_addresses - new_addresses:
            msg = {
                "method": "unsubscribe",
                "subscription": {"type": "userFills", "user": old_addr}
            }
            try:
                await ws.send(json.dumps(msg))
                bt.logging.info(f"[HL_TRACKER] Unsubscribed from userFills for {old_addr}")
            except Exception as e:
                bt.logging.warning(f"[HL_TRACKER] Failed to unsubscribe for {old_addr}: {e}")

        self._subscribed_addresses = new_addresses
        bt.logging.info(f"[HL_TRACKER] Subscribed to {len(self._subscribed_addresses)} HL addresses")

    async def _periodic_refresh(self, ws: WebSocketClientProtocol):
        """Periodically refresh subscriptions for new/removed HL addresses."""
        while True:
            await asyncio.sleep(self.ADDRESS_REFRESH_INTERVAL_S)
            try:
                await self._subscribe_all(ws)
            except Exception as e:
                bt.logging.error(f"[HL_TRACKER] Periodic refresh error: {e}")

    # ==================== Message Handling ====================

    def _handle_message(self, msg: dict):
        """Route incoming WebSocket messages."""
        channel = msg.get("channel")

        if channel == "pong":
            return

        if channel == "userFills":
            self._handle_user_fills(msg)

    def _handle_user_fills(self, msg: dict):
        """Handle userFills channel messages."""
        data = msg.get("data", {})
        is_snapshot = data.get("isSnapshot", False)
        user = data.get("user")
        fills = data.get("fills", [])

        if not user or not fills:
            return

        for fill in fills:
            fill_hash = fill.get("hash") or fill.get("tid")
            if not fill_hash:
                continue

            # Record hash for dedup (even for snapshots)
            if fill_hash in self._processed_hashes:
                continue
            self._record_hash(fill_hash)

            # Skip snapshot fills (historical data on reconnect)
            if is_snapshot:
                continue

            # Process new fill
            try:
                self._process_fill(user, fill)
            except Exception as e:
                bt.logging.error(f"[HL_TRACKER] Error processing fill for {user}: {e}")
                bt.logging.error(traceback.format_exc())

    def _record_hash(self, fill_hash: str):
        """Record a fill hash in the bounded dedup set."""
        self._processed_hashes[fill_hash] = True
        # Evict oldest entries if over limit
        while len(self._processed_hashes) > self.MAX_DEDUP_HASHES:
            self._processed_hashes.popitem(last=False)

    # ==================== Fill Processing ====================

    def _process_fill(self, hl_address: str, fill: dict):
        """
        Convert a Hyperliquid fill to a Vanta signal and process it.

        Steps:
        1. Map coin to Vanta TradePair
        2. Resolve synthetic hotkey
        3. Run should_fail_early-equivalent checks
        4. Build signal and process via OrderProcessor
        """
        coin = fill.get("coin")
        if not coin:
            return

        # Map coin to trade pair ID
        trade_pair_id = ValiConfig.HL_COIN_TO_TRADE_PAIR.get(coin)
        if not trade_pair_id:
            bt.logging.debug(f"[HL_TRACKER] Unsupported coin: {coin}")
            return

        trade_pair = TRADE_PAIR_ID_TO_TRADE_PAIR.get(trade_pair_id)
        if not trade_pair:
            bt.logging.warning(f"[HL_TRACKER] Trade pair not found: {trade_pair_id}")
            return

        # Resolve synthetic hotkey
        synthetic_hotkey = self._entity_client.get_synthetic_hotkey_for_hl_address(hl_address)
        if not synthetic_hotkey:
            bt.logging.warning(f"[HL_TRACKER] No synthetic hotkey for HL address {hl_address}")
            return

        # Get subaccount info for account_size
        subaccount_info = self._entity_client.get_subaccount_info_for_synthetic(synthetic_hotkey)
        if not subaccount_info:
            bt.logging.warning(f"[HL_TRACKER] No subaccount info for {synthetic_hotkey}")
            return

        account_size = subaccount_info.get("account_size", 0)
        if account_size <= 0:
            bt.logging.warning(f"[HL_TRACKER] Invalid account size for {synthetic_hotkey}")
            return

        now_ms = TimeUtil.now_in_millis()

        # === Fail-early checks (mirrors validator.py should_fail_early) ===

        # Rate limiting
        allowed, wait_time = self._rate_limiter.is_allowed(synthetic_hotkey)
        if not allowed:
            bt.logging.debug(f"[HL_TRACKER] Rate limited: {synthetic_hotkey}, wait {wait_time:.1f}s")
            return

        # Elimination check
        elimination_info = self._elimination_client.get_elimination_local_cache(synthetic_hotkey)
        if elimination_info:
            bt.logging.debug(f"[HL_TRACKER] Eliminated miner: {synthetic_hotkey}")
            return

        # Subaccount status check
        validation = self._entity_client.validate_hotkey_for_orders(synthetic_hotkey)
        if not validation.get("is_valid"):
            bt.logging.debug(f"[HL_TRACKER] Invalid hotkey: {synthetic_hotkey} - {validation.get('error_message')}")
            return

        # Trade pair blocked check
        if trade_pair.is_blocked:
            bt.logging.debug(f"[HL_TRACKER] Blocked trade pair: {trade_pair_id}")
            return

        # Market hours check (only for market orders)
        is_market_open = self._price_fetcher_client.is_market_open(trade_pair, now_ms)
        if not is_market_open:
            bt.logging.debug(f"[HL_TRACKER] Market closed for {trade_pair_id}")
            return

        # === Build signal ===
        side = fill.get("side", "")
        fill_sz = float(fill.get("sz", 0))
        fill_px = float(fill.get("px", 0))

        if fill_sz <= 0 or fill_px <= 0:
            return

        # Determine order type from side
        # HL side: "B" = buy (LONG), "A" = sell (SHORT)
        if side == "B":
            order_type = "LONG"
        elif side == "A":
            order_type = "SHORT"
        else:
            bt.logging.warning(f"[HL_TRACKER] Unknown fill side: {side}")
            return

        # Calculate leverage: position notional / account_size, clamped to crypto limits
        raw_leverage = (fill_sz * fill_px) / account_size
        leverage = max(ValiConfig.CRYPTO_MIN_LEVERAGE, min(raw_leverage, ValiConfig.CRYPTO_MAX_LEVERAGE))

        signal = {
            "order_type": order_type,
            "leverage": leverage,
            "trade_pair": {"trade_pair_id": trade_pair_id},
            "execution_type": "MARKET",
        }

        miner_order_uuid = str(uuid.uuid4())

        # === Process order ===
        try:
            result = OrderProcessor.process_order(
                signal=signal,
                miner_order_uuid=miner_order_uuid,
                now_ms=now_ms,
                miner_hotkey=synthetic_hotkey,
                miner_repo_version="hl_tracker",
                limit_order_client=self._limit_order_client,
                market_order_manager=self._market_order_manager,
            )

            # Track UUID
            if result.should_track_uuid:
                self._uuid_tracker.add(miner_order_uuid)

            self._fills_processed += 1
            self._last_fill_time = time.time()

            bt.logging.info(
                f"[HL_TRACKER] Processed fill: {coin} {side} {fill_sz}@{fill_px} -> "
                f"{synthetic_hotkey} {order_type} leverage={leverage:.4f}"
            )

        except SignalException as e:
            bt.logging.warning(f"[HL_TRACKER] Signal rejected for {synthetic_hotkey}: {e}")
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Order processing error for {synthetic_hotkey}: {e}")
            bt.logging.error(traceback.format_exc())
