"""
HyperliquidTrackerServer - RPC server for tracking Hyperliquid perp trades.

This server:
- Maintains miner_hotkey -> hyperliquid_address registrations (persisted to JSON)
- Runs an async WebSocket connection to Hyperliquid's API
- Subscribes to userFills for each registered address
- Converts fills to signals and forwards them via OrderProcessor.process_order()

Follows the AssetSelectionServer pattern.

Usage:
    # Validator spawns the server at startup via ServerOrchestrator
    # Other processes connect via HyperliquidTrackerClient
"""

import os
import re
import json
import uuid
import asyncio
import threading
import websockets
import bittensor as bt
from typing import Dict, Optional

from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, TradePair, RPCConnectionMode


# Build coin -> TradePair lookup for crypto pairs
# Maps Hyperliquid coin names (e.g., "BTC") to TradePair enum values (e.g., TradePair.BTCUSD)
COIN_TO_TRADE_PAIR: Dict[str, TradePair] = {}
for tp in TradePair:
    if tp.is_crypto:
        coin = tp.trade_pair_id.replace("USD", "")
        COIN_TO_TRADE_PAIR[coin] = tp

# Valid Ethereum address pattern
ETH_ADDRESS_PATTERN = re.compile(r'^0x[0-9a-fA-F]{40}$')

HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"

# Persistence file path (relative to validation directory)
REGISTRATIONS_FILENAME = "hyperliquid_registrations.json"


def _get_registrations_filepath(running_unit_tests: bool = False) -> str:
    suffix = "/tests" if running_unit_tests else ""
    return os.path.join(
        ValiConfig.BASE_DIR,
        f"{suffix}/validation/{REGISTRATIONS_FILENAME}"
    )


def is_valid_eth_address(address: str) -> bool:
    return bool(ETH_ADDRESS_PATTERN.match(address))


def fill_to_signal(fill: dict, trade_pair: TradePair) -> Optional[dict]:
    """
    Convert a Hyperliquid fill to a signal dict compatible with OrderProcessor.

    Args:
        fill: Hyperliquid fill data with keys: coin, px, sz, side, dir, etc.
        trade_pair: The TradePair enum for this coin

    Returns:
        Signal dict or None if the fill direction is unrecognized
    """
    direction = fill.get("dir", "")

    if direction == "Open Long":
        order_type = "LONG"
    elif direction == "Open Short":
        order_type = "SHORT"
    elif direction in ("Close Long", "Close Short"):
        order_type = "FLAT"
    else:
        bt.logging.warning(f"[HL_TRACKER] Unknown fill direction: {direction}")
        return None

    px = float(fill.get("px", 0))
    sz = float(fill.get("sz", 0))

    return {
        "trade_pair": {"trade_pair_id": trade_pair.trade_pair_id},
        "order_type": order_type,
        "leverage": None,
        "value": sz * px,
        "quantity": None,
        "execution_type": "MARKET",
    }


class HyperliquidTrackerServer(RPCServerBase):
    """
    RPC server for Hyperliquid trade tracking.

    Maintains registrations and runs a WebSocket tracker daemon that
    forwards Hyperliquid fills as signals through OrderProcessor.
    """

    service_name = ValiConfig.RPC_HYPERLIQUID_TRACKER_SERVICE_NAME
    service_port = ValiConfig.RPC_HYPERLIQUID_TRACKER_PORT

    def __init__(
        self,
        config=None,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        **kwargs,
    ):
        self._config = config
        self.running_unit_tests = running_unit_tests
        self._connection_mode = connection_mode

        # Registration storage: miner_hotkey -> {address, registered_ms}
        self._registrations: Dict[str, dict] = {}
        self._registrations_lock = threading.Lock()
        self._load_registrations()

        # WebSocket state
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_connection = None
        self._ws_running = False
        self._subscribed_addresses: set = set()

        # OrderProcessor dependencies (created lazily on first daemon iteration)
        self._limit_order_client = None
        self._market_order_manager = None

        bt.logging.success("[HL_TRACKER] HyperliquidTrackerServer registrations loaded")

        super().__init__(
            service_name=ValiConfig.RPC_HYPERLIQUID_TRACKER_SERVICE_NAME,
            port=ValiConfig.RPC_HYPERLIQUID_TRACKER_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=15.0,
            hang_timeout_s=120.0,
            connection_mode=connection_mode,
        )

        bt.logging.success("[HL_TRACKER] HyperliquidTrackerServer initialized")

    # ==================== Persistence ====================

    def _get_filepath(self) -> str:
        return _get_registrations_filepath(self.running_unit_tests)

    def _load_registrations(self) -> None:
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    self._registrations = json.load(f)
                bt.logging.info(
                    f"[HL_TRACKER] Loaded {len(self._registrations)} registrations from disk"
                )
            except Exception as e:
                bt.logging.error(f"[HL_TRACKER] Failed to load registrations: {e}")
                self._registrations = {}
        else:
            self._registrations = {}

    def _save_registrations(self) -> None:
        filepath = self._get_filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, "w") as f:
                json.dump(self._registrations, f, indent=2)
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Failed to save registrations: {e}")

    # ==================== RPC Methods ====================

    def register_address_rpc(self, miner_hotkey: str, address: str) -> dict:
        if not is_valid_eth_address(address):
            return {"success": False, "error": "Invalid Ethereum address"}

        with self._registrations_lock:
            self._registrations[miner_hotkey] = {
                "address": address,
                "registered_ms": TimeUtil.now_in_millis(),
            }
            self._save_registrations()

        bt.logging.info(
            f"[HL_TRACKER] Registered {miner_hotkey} -> {address}"
        )
        return {"success": True, "message": f"Registered address {address}"}

    def unregister_address_rpc(self, miner_hotkey: str) -> dict:
        with self._registrations_lock:
            if miner_hotkey not in self._registrations:
                return {"success": False, "error": "No registration found for this hotkey"}
            del self._registrations[miner_hotkey]
            self._save_registrations()

        bt.logging.info(f"[HL_TRACKER] Unregistered {miner_hotkey}")
        return {"success": True, "message": "Registration removed"}

    def get_registrations_rpc(self) -> dict:
        with self._registrations_lock:
            return dict(self._registrations)

    def get_registration_rpc(self, miner_hotkey: str) -> Optional[dict]:
        with self._registrations_lock:
            return self._registrations.get(miner_hotkey)

    def get_health_check_details(self) -> dict:
        return {
            "total_registrations": len(self._registrations),
            "ws_connected": self._ws_connection is not None and self._ws_running,
            "subscribed_addresses": len(self._subscribed_addresses),
        }

    # ==================== Daemon ====================

    def _ensure_order_processor_deps(self) -> None:
        """Lazily create OrderProcessor dependencies."""
        if self._limit_order_client is None:
            from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
            self._limit_order_client = LimitOrderClient(
                connection_mode=self._connection_mode
            )
        if self._market_order_manager is None:
            from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager
            self._market_order_manager = MarketOrderManager(
                serve=False,
                running_unit_tests=self.running_unit_tests,
                connection_mode=self._connection_mode,
            )

    def run_daemon_iteration(self) -> None:
        """
        Daemon iteration: ensure WebSocket is running and healthy.

        If registrations exist and WS is not running, start it.
        If registrations changed, update subscriptions.
        If WS died, restart it.
        """
        with self._registrations_lock:
            current_addresses = {
                reg["address"] for reg in self._registrations.values()
            }
            has_registrations = len(current_addresses) > 0

        if not has_registrations:
            if self._ws_running:
                self._stop_ws()
            return

        # Ensure OrderProcessor dependencies exist
        self._ensure_order_processor_deps()

        if not self._ws_running:
            self._start_ws()
        elif current_addresses != self._subscribed_addresses:
            # Registrations changed; restart WS to update subscriptions
            bt.logging.info("[HL_TRACKER] Registrations changed, restarting WebSocket")
            self._stop_ws()
            self._start_ws()

    # ==================== WebSocket Management ====================

    def _start_ws(self) -> None:
        if self._ws_running:
            return

        self._ws_running = True
        self._ws_thread = threading.Thread(
            target=self._ws_thread_main,
            name="hl_tracker_ws",
            daemon=True,
        )
        self._ws_thread.start()
        bt.logging.info("[HL_TRACKER] WebSocket thread started")

    def _stop_ws(self) -> None:
        self._ws_running = False
        if self._ws_loop is not None:
            self._ws_loop.call_soon_threadsafe(self._ws_loop.stop)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=10)
            self._ws_thread = None
        self._ws_connection = None
        self._ws_loop = None
        self._subscribed_addresses.clear()
        bt.logging.info("[HL_TRACKER] WebSocket thread stopped")

    def _ws_thread_main(self) -> None:
        """Main function for the WebSocket thread. Runs an asyncio event loop."""
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)
        try:
            self._ws_loop.run_until_complete(self._ws_run_loop())
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] WebSocket thread error: {e}")
        finally:
            self._ws_loop.close()
            self._ws_running = False

    async def _ws_run_loop(self) -> None:
        """Async WebSocket connection loop with automatic reconnection."""
        backoff = 1
        max_backoff = 60

        while self._ws_running:
            try:
                async with websockets.connect(HYPERLIQUID_WS_URL) as ws:
                    self._ws_connection = ws
                    backoff = 1  # Reset on successful connection
                    bt.logging.success("[HL_TRACKER] WebSocket connected")

                    # Subscribe to all registered addresses
                    await self._subscribe_all(ws)

                    # Listen for messages
                    async for message in ws:
                        if not self._ws_running:
                            break
                        try:
                            data = json.loads(message)
                            self._handle_ws_message(data)
                        except json.JSONDecodeError:
                            bt.logging.warning(
                                f"[HL_TRACKER] Non-JSON message: {message[:200]}"
                            )
                        except Exception as e:
                            bt.logging.error(
                                f"[HL_TRACKER] Error handling message: {e}"
                            )

            except Exception as e:
                if not self._ws_running:
                    break
                bt.logging.warning(
                    f"[HL_TRACKER] WebSocket disconnected: {e}. "
                    f"Reconnecting in {backoff}s..."
                )
                self._ws_connection = None
                self._subscribed_addresses.clear()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def _subscribe_all(self, ws) -> None:
        """Subscribe to userFills for all registered addresses."""
        with self._registrations_lock:
            addresses = {reg["address"] for reg in self._registrations.values()}

        self._subscribed_addresses = set()
        for address in addresses:
            try:
                subscribe_msg = json.dumps({
                    "method": "subscribe",
                    "subscription": {
                        "type": "userFills",
                        "user": address,
                    },
                })
                await ws.send(subscribe_msg)
                self._subscribed_addresses.add(address)
                bt.logging.info(f"[HL_TRACKER] Subscribed to fills for {address}")
            except Exception as e:
                bt.logging.error(
                    f"[HL_TRACKER] Failed to subscribe for {address}: {e}"
                )

    def _handle_ws_message(self, data: dict) -> None:
        """Handle an incoming WebSocket message."""
        channel = data.get("channel")
        if channel != "userFills":
            return

        msg_data = data.get("data", {})
        user_address = msg_data.get("user", "")
        fills = msg_data.get("fills", [])

        if not fills:
            return

        # Find the miner_hotkey for this address
        miner_hotkey = None
        with self._registrations_lock:
            for hotkey, reg in self._registrations.items():
                if reg["address"].lower() == user_address.lower():
                    miner_hotkey = hotkey
                    break

        if miner_hotkey is None:
            bt.logging.warning(
                f"[HL_TRACKER] Received fills for unknown address: {user_address}"
            )
            return

        for fill in fills:
            self._process_fill(fill, miner_hotkey)

    def _process_fill(self, fill: dict, miner_hotkey: str) -> None:
        """Convert a Hyperliquid fill to a signal and process it."""
        coin = fill.get("coin", "")
        trade_pair = COIN_TO_TRADE_PAIR.get(coin)

        if trade_pair is None:
            bt.logging.debug(
                f"[HL_TRACKER] Skipping unsupported coin: {coin}"
            )
            return

        signal = fill_to_signal(fill, trade_pair)
        if signal is None:
            return

        try:
            from vali_objects.utils.limit_order.order_processor import OrderProcessor

            now_ms = TimeUtil.now_in_millis()
            result = OrderProcessor.process_order(
                signal=signal,
                miner_order_uuid=str(uuid.uuid4()),
                now_ms=now_ms,
                miner_hotkey=miner_hotkey,
                miner_repo_version="hyperliquid_tracker",
                limit_order_client=self._limit_order_client,
                market_order_manager=self._market_order_manager,
            )

            bt.logging.info(
                f"[HL_TRACKER] Processed fill for {miner_hotkey}: "
                f"{coin} {fill.get('dir')} {fill.get('sz')} @ {fill.get('px')} "
                f"-> success={result.success}"
            )
        except Exception as e:
            bt.logging.error(
                f"[HL_TRACKER] Error processing fill for {miner_hotkey}: {e}"
            )

    # ==================== Test Helpers ====================

    def clear_registrations_for_test_rpc(self) -> None:
        """Clear all registrations (TEST ONLY)."""
        if not self.running_unit_tests:
            raise RuntimeError("clear_registrations_for_test is only available in test mode")
        with self._registrations_lock:
            self._registrations.clear()
            self._save_registrations()
