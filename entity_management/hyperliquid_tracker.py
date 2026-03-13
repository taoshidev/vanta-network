# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
HyperliquidTracker - Daemon service that tracks Hyperliquid trader fills
and forwards them as Vanta signals through the existing pipeline.

Runs as a daemon thread in the validator process. Supports sharding across
multiple Decodo SOCKS5 proxy IPs to scale beyond the 10-address-per-IP
Hyperliquid WebSocket limit.

Architecture:
- Own asyncio event loop in a daemon thread
- One _WebSocketShard per proxy IP (or one direct shard if no proxy configured)
- Each shard manages up to 10 addresses, its own heartbeat, reconnect w/ backoff
- Shared fill dedup via bounded hash set across all shards
- Converts fills to market orders via OrderProcessor.process_order()
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import bittensor as bt
import requests

import ssl

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None
    WebSocketClientProtocol = None

try:
    from python_socks.async_.asyncio import Proxy as SocksProxy
except ImportError:
    SocksProxy = None

from entity_management.entity_client import EntityClient
from entity_management.hl_orderbook_utils import simulate_fill
from shared_objects.rate_limiter import RateLimiter
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair, TRADE_PAIR_ID_TO_TRADE_PAIR, RPCConnectionMode
from vanta_api.websocket_notifier import WebSocketNotifierClient


class HyperliquidTracker:
    """
    Tracks Hyperliquid trader fills via WebSocket and forwards them as Vanta signals.

    Supports sharding across multiple proxy IPs. Without proxy config, behaves
    identically to the original single-connection implementation.
    """

    # Max fill hashes to track for dedup (bounded to prevent memory growth)
    MAX_DEDUP_HASHES = 50_000
    # How often to refresh the list of subscribed addresses (seconds)
    ADDRESS_REFRESH_INTERVAL_S = 60.0
    # Cache TTL for dynamic HL coin discovery used by L2 subscriptions.
    L2_COIN_CACHE_TTL_S = 300.0

    # ==================== Inner class: _WebSocketShard ====================

    class _WebSocketShard:
        """
        Encapsulates a single WebSocket connection through a specific proxy port (= IP).
        Manages up to HL_MAX_TRACKED_ADDRESSES_PER_IP addresses, its own heartbeat,
        subscribe/unsubscribe, and reconnect with backoff.
        """

        def __init__(self, shard_id: int, proxy_url: Optional[str], tracker: 'HyperliquidTracker'):
            self.shard_id = shard_id
            self.proxy_url = proxy_url  # None = direct connection
            self.tracker = tracker
            self.addresses: Set[str] = set()
            self.subscribed_addresses: Set[str] = set()
            self.healthy = True
            self.connected = False
            self.task: Optional[asyncio.Task] = None
            self._consecutive_failures = 0

        @property
        def port(self) -> Optional[int]:
            """Extract port from proxy URL for logging."""
            if not self.proxy_url:
                return None
            try:
                return int(self.proxy_url.rsplit(":", 1)[-1])
            except (ValueError, IndexError):
                return None

        @property
        def capacity(self) -> int:
            """Remaining address capacity for this shard."""
            return ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP - len(self.addresses)

        @property
        def label(self) -> str:
            port = self.port
            return f"SHARD-{self.shard_id}" + (f"/port={port}" if port else "/direct")

        async def _open_ws_connection(self):
            """Open a WebSocket connection, optionally through a SOCKS5 proxy."""
            if self.proxy_url and SocksProxy:
                proxy = SocksProxy.from_url(self.proxy_url)
                hl_host = ValiConfig.hl_host()
                # HL WS is wss:// so we connect to port 443
                sock = await asyncio.wait_for(
                    proxy.connect(dest_host=hl_host, dest_port=443),
                    timeout=15,
                )
                ssl_ctx = ssl.create_default_context()
                return websockets.connect(
                    ValiConfig.hl_ws_url(),
                    sock=sock,
                    ssl=ssl_ctx,
                    server_hostname=hl_host,
                    ping_interval=None,
                )
            else:
                return websockets.connect(
                    ValiConfig.hl_ws_url(),
                    ping_interval=None,
                )

        async def run(self):
            """Main loop: connect, subscribe, process messages, reconnect on failure."""
            backoff_s = 1.0

            while not self.tracker._stop_event.is_set():
                ws: Optional[WebSocketClientProtocol] = None
                try:
                    ws_ctx = await self._open_ws_connection()
                    ws = await ws_ctx

                    bt.logging.info(
                        f"[HL_{self.label}] Connected to {ValiConfig.hl_ws_url()}"
                        + (f" via {self.proxy_url}" if self.proxy_url else " (direct)")
                    )
                    self._consecutive_failures = 0
                    self.healthy = True
                    self.connected = True
                    backoff_s = 1.0

                    # New socket starts with no server-side subscriptions. Clear local
                    # cache so _sync_subscriptions replays all address subscriptions.
                    self.subscribed_addresses.clear()

                    # Subscribe to current addresses
                    await self._sync_subscriptions(ws)

                    # Start heartbeat + periodic refresh
                    hb_task = asyncio.create_task(self._heartbeat(ws))
                    refresh_task = asyncio.create_task(self._periodic_refresh(ws))

                    try:
                        while not self.tracker._stop_event.is_set():
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            except asyncio.TimeoutError:
                                continue
                            except websockets.exceptions.ConnectionClosed as e:
                                bt.logging.warning(
                                    f"[HL_{self.label}] WebSocket closed: code={getattr(e, 'code', None)} "
                                    f"reason={getattr(e, 'reason', '')!r}"
                                )
                                break
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            self.tracker._handle_message(msg, self.shard_id)
                    finally:
                        hb_task.cancel()
                        refresh_task.cancel()

                except Exception as e:
                    self._consecutive_failures += 1
                    bt.logging.warning(
                        f"[HL_{self.label}] Connection failed ({self._consecutive_failures}x): {e!r}"
                    )
                    if self._consecutive_failures >= ValiConfig.HL_SHARD_MAX_CONSECUTIVE_FAILURES:
                        bt.logging.error(
                            f"[HL_{self.label}] Marked UNHEALTHY after "
                            f"{self._consecutive_failures} consecutive failures"
                        )
                        self.healthy = False
                        self.connected = False
                        return  # Stop this shard; orchestrator will redistribute
                finally:
                    self.connected = False
                    if ws is not None:
                        try:
                            # Proxied sockets can stall on close handshake; bound close time.
                            loop = asyncio.get_running_loop()
                            if not loop.is_closed():
                                await asyncio.wait_for(ws.close(), timeout=2.0)
                        except RuntimeError:
                            # Event loop is already closing/closed.
                            pass
                        except Exception:
                            transport = getattr(ws, "transport", None)
                            if transport is not None:
                                try:
                                    loop = asyncio.get_running_loop()
                                    if not loop.is_closed():
                                        transport.abort()
                                except RuntimeError:
                                    pass

                if self.tracker._stop_event.is_set():
                    break

                bt.logging.info(f"[HL_{self.label}] Reconnecting in {backoff_s:.1f}s...")
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, ValiConfig.HL_WS_RECONNECT_BACKOFF_MAX_S)

        async def _heartbeat(self, ws):
            """Send ping messages to keep the connection alive."""
            while True:
                await asyncio.sleep(ValiConfig.HL_WS_HEARTBEAT_INTERVAL_S)
                try:
                    await ws.send(json.dumps({"method": "ping"}))
                except Exception:
                    return

        async def _sync_subscriptions(self, ws):
            """Subscribe/unsubscribe to match self.addresses."""
            new_addresses = set(self.addresses)

            # Subscribe to new
            for addr in new_addresses - self.subscribed_addresses:
                msg = {
                    "method": "subscribe",
                    "subscription": {"type": "userFills", "user": addr.lower()},
                }
                try:
                    await ws.send(json.dumps(msg))
                    bt.logging.info(f"[HL_{self.label}] Subscribed to userFills for {addr}")
                except Exception as e:
                    bt.logging.error(f"[HL_{self.label}] Failed to subscribe for {addr}: {e}")

            # Unsubscribe from removed
            for addr in self.subscribed_addresses - new_addresses:
                msg = {
                    "method": "unsubscribe",
                    "subscription": {"type": "userFills", "user": addr},
                }
                try:
                    await ws.send(json.dumps(msg))
                    bt.logging.info(f"[HL_{self.label}] Unsubscribed from userFills for {addr}")
                except Exception as e:
                    bt.logging.warning(f"[HL_{self.label}] Failed to unsubscribe for {addr}: {e}")

            self.subscribed_addresses = new_addresses

            # Shard 0 subscribes to fine-grained L2 book (nSigFigs=5) for precise
            # near-spread slippage. Shard 1 subscribes to coarse L2 book (nSigFigs=2)
            # for deep coverage on large orders. Both are combined during slippage calc.
            if self.shard_id == 0:
                n_sig = ValiConfig.HL_L2_FINE_SIG_FIGS
            elif self.shard_id == 1:
                n_sig = ValiConfig.HL_L2_COARSE_SIG_FIGS
            else:
                n_sig = None

            if n_sig is not None:
                for coin in self.tracker._get_l2_subscription_coins():
                    try:
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": coin,
                                             "nSigFigs": n_sig}
                        }))
                    except Exception as e:
                        bt.logging.warning(
                            f"[HL_{self.label}] Failed to subscribe l2Book "
                            f"(nSigFigs={n_sig}) for {coin}: {e}"
                        )

        async def _periodic_refresh(self, ws):
            """Periodically sync subscriptions for address changes."""
            while True:
                await asyncio.sleep(HyperliquidTracker.ADDRESS_REFRESH_INTERVAL_S)
                try:
                    await self._sync_subscriptions(ws)
                except Exception as e:
                    bt.logging.error(f"[HL_{self.label}] Periodic refresh error: {e}")

    # ==================== HyperliquidTracker ====================

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
        ws_notifier_client: Optional[WebSocketNotifierClient] = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
    ):
        self._entity_client = entity_client
        self._elimination_client = elimination_client
        self._price_fetcher_client = price_fetcher_client
        self._asset_selection_client = asset_selection_client
        self._market_order_manager = market_order_manager
        self._limit_order_client = limit_order_client
        self._uuid_tracker = uuid_tracker
        self._rate_limiter = rate_limiter or RateLimiter()
        self._ws_notifier_client = ws_notifier_client

        # Position client for querying current Vanta positions (weight delta calculation)
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # L2 orderbook snapshots per coin at two resolutions (updated via WebSocket).
        # Fine (nSigFigs=5): precise near-spread pricing, subscribed on shard 0.
        # Coarse (nSigFigs=2): deep coverage for large orders, subscribed on shard 1.
        self._orderbooks_fine: Dict[str, dict] = {}
        self._orderbooks_coarse: Dict[str, dict] = {}

        # State
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

        # Dedup: ordered dict of fill_hash -> True (bounded, oldest evicted first)
        self._processed_hashes: OrderedDict[str, bool] = OrderedDict()

        # Shard state
        self._shards: Dict[int, HyperliquidTracker._WebSocketShard] = {}
        self._address_to_shard: Dict[str, int] = {}
        self._next_shard_id = 0

        # Proxy config (populated in _load_proxy_config)
        self._proxy_base_url: Optional[str] = None  # e.g. "socks5://user:pass@host"
        self._available_ports: List[int] = []
        self._unhealthy_ports: Set[int] = set()

        # Metrics
        self._fills_processed = 0
        self._last_fill_time: Optional[float] = None
        self._l2_coin_cache: Optional[Set[str]] = None
        self._l2_coin_cache_ts: float = 0.0

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
        # Let _run_stream exit naturally to allow shard tasks to clean up.
        # Forcing loop.stop() can interrupt websocket teardown and produce
        # "Event loop is closed" warnings on shutdown.
        if self._thread:
            self._thread.join(timeout=5.0)
        bt.logging.info("[HL_TRACKER] Stopped")

    def get_status(self) -> dict:
        """Get tracker status for health monitoring."""
        shard_statuses = []
        for sid, shard in self._shards.items():
            shard_statuses.append({
                "shard_id": sid,
                "port": shard.port,
                "healthy": shard.healthy,
                "connected": shard.connected,
                "address_count": len(shard.addresses),
            })
        return {
            "shards": shard_statuses,
            "total_connected": sum(1 for s in self._shards.values() if s.connected),
            "total_subscribed_addresses": len(self._address_to_shard),
            "fills_processed": self._fills_processed,
            "last_fill_time": self._last_fill_time,
            "proxy_configured": self._proxy_base_url is not None,
            "available_ports": len(self._available_ports),
            "unhealthy_ports": len(self._unhealthy_ports),
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
            pending = asyncio.all_tasks(self._loop)
            if pending:
                for task in pending:
                    task.cancel()
                try:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass
            self._loop.close()

    def _get_l2_subscription_coins(self) -> List[str]:
        """
        Return HL coins that should be used for l2Book subscriptions.

        We dynamically intersect configured coins with HL `allMids` to avoid sending
        unsupported testnet subscriptions that can cause abrupt socket closes.
        """
        configured_coins = set(ValiConfig.HL_COIN_TO_TRADE_PAIR.keys())
        now = time.time()
        if (
            self._l2_coin_cache is not None
            and (now - self._l2_coin_cache_ts) < self.L2_COIN_CACHE_TTL_S
        ):
            return sorted(self._l2_coin_cache)

        try:
            resp = requests.post(
                ValiConfig.hl_info_url(),
                json={"type": "allMids"},
                timeout=5,
            )
            mids = resp.json()
            if isinstance(mids, dict):
                supported = configured_coins.intersection(mids.keys())
            else:
                supported = configured_coins

            if not supported:
                supported = configured_coins
            elif supported != configured_coins:
                skipped = sorted(configured_coins - supported)
                bt.logging.info(
                    f"[HL_TRACKER] Skipping unsupported l2Book coins on current HL env: {skipped}"
                )

            self._l2_coin_cache = supported
            self._l2_coin_cache_ts = now
            return sorted(supported)
        except Exception as e:
            bt.logging.warning(
                f"[HL_TRACKER] Failed to fetch dynamic l2Book coin list: {e}. "
                "Falling back to configured coins."
            )
            if self._l2_coin_cache:
                return sorted(self._l2_coin_cache)
            return sorted(configured_coins)

    # ==================== Proxy Config ====================

    def _load_proxy_config(self):
        """Load proxy configuration from secrets.json. No-op if not configured."""
        try:
            secrets = ValiUtils.get_secrets()
        except Exception:
            secrets = {}

        proxy_url = secrets.get(ValiConfig.HL_PROXY_SECRET_KEY)
        ports_str = secrets.get(ValiConfig.HL_PROXY_PORTS_SECRET_KEY)

        if not proxy_url or not ports_str:
            bt.logging.info("[HL_TRACKER] No proxy config found - using direct connection (max 10 addresses)")
            self._proxy_base_url = None
            self._available_ports = []
            return

        if SocksProxy is None:
            bt.logging.error(
                "[HL_TRACKER] Proxy config found but python-socks is not installed! "
                "Run: pip install python-socks  — falling back to direct connection"
            )
            self._proxy_base_url = None
            self._available_ports = []
            return

        self._proxy_base_url = proxy_url.rstrip("/")
        self._available_ports = self._parse_ports(ports_str)

        # Cap to safety limit
        if len(self._available_ports) > ValiConfig.HL_MAX_PROXY_SHARDS:
            bt.logging.warning(
                f"[HL_TRACKER] Capping proxy ports from {len(self._available_ports)} to {ValiConfig.HL_MAX_PROXY_SHARDS}"
            )
            self._available_ports = self._available_ports[:ValiConfig.HL_MAX_PROXY_SHARDS]

        bt.logging.info(
            f"[HL_TRACKER] Proxy configured: {len(self._available_ports)} ports available "
            f"(max {len(self._available_ports) * ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP} addresses)"
        )

    @staticmethod
    def _parse_ports(ports_str: str) -> List[int]:
        """Parse port string like '10001-10010' or '10001,10002,10005' into list of ints."""
        ports = []
        for part in ports_str.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    ports.extend(range(int(start.strip()), int(end.strip()) + 1))
                except ValueError:
                    bt.logging.warning(f"[HL_TRACKER] Invalid port range: {part}")
            else:
                try:
                    ports.append(int(part))
                except ValueError:
                    bt.logging.warning(f"[HL_TRACKER] Invalid port: {part}")
        return ports

    def _make_shard_proxy_url(self, port: int) -> str:
        """Build full proxy URL for a specific port."""
        return f"{self._proxy_base_url}:{port}"

    def get_max_tracked_addresses(self) -> int:
        """Return the max number of HL addresses we can track given proxy config."""
        if self._proxy_base_url and self._available_ports:
            # All ports (available + already in use by shards)
            total_ports = len(self._available_ports) + len(self._shards)
            return total_ports * ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP
        return ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP

    # ==================== Shard Orchestration ====================

    async def _run_stream(self):
        """Orchestrator: loads proxy config, then loops assigning addresses and managing shards."""
        self._load_proxy_config()

        while not self._stop_event.is_set():
            try:
                self._assign_addresses_to_shards()
                self._ensure_shard_tasks()
            except Exception as e:
                bt.logging.error(f"[HL_TRACKER] Orchestrator error: {e}")
                bt.logging.error(traceback.format_exc())

            # Wait before next refresh cycle
            for _ in range(int(self.ADDRESS_REFRESH_INTERVAL_S)):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(1.0)

    def _assign_addresses_to_shards(self):
        """
        Assign active HL addresses to shards.
        1. Remove addresses no longer active
        2. Redistribute addresses from unhealthy shards
        3. Assign new addresses to shard with most capacity
        4. Create new shards if needed and ports available
        5. Tear down empty shards
        """
        # Get current active addresses
        try:
            hl_subaccounts = self._entity_client.get_all_active_hl_subaccounts()
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Failed to get HL subaccounts: {e}")
            return

        active_addresses = {addr for addr, _info in hl_subaccounts}

        # 1. Remove addresses no longer active
        stale = set(self._address_to_shard.keys()) - active_addresses
        for addr in stale:
            sid = self._address_to_shard.pop(addr, None)
            if sid is not None and sid in self._shards:
                self._shards[sid].addresses.discard(addr)

        # 2. Collect addresses from unhealthy shards for redistribution
        orphaned: Set[str] = set()
        unhealthy_shard_ids = [sid for sid, s in self._shards.items() if not s.healthy]
        for sid in unhealthy_shard_ids:
            shard = self._shards[sid]
            orphaned.update(shard.addresses & active_addresses)
            # Return port to unhealthy set
            port = shard.port
            if port is not None:
                self._unhealthy_ports.add(port)
            # Clean up shard
            for addr in shard.addresses:
                self._address_to_shard.pop(addr, None)
            shard.addresses.clear()
            if shard.task and not shard.task.done():
                shard.task.cancel()
            del self._shards[sid]
            bt.logging.warning(f"[HL_TRACKER] Removed unhealthy shard {shard.label}, {len(orphaned)} addresses to redistribute")

        # 3. Addresses that need assignment (new + orphaned)
        already_assigned = set(self._address_to_shard.keys())
        to_assign = (active_addresses - already_assigned) | orphaned

        if not to_assign:
            # 5. Tear down empty shards
            self._teardown_empty_shards()
            return

        for addr in to_assign:
            assigned = False

            # Find healthy shard with most capacity
            best_shard = None
            best_capacity = 0
            for sid, shard in self._shards.items():
                if shard.healthy and shard.capacity > best_capacity:
                    best_shard = shard
                    best_capacity = shard.capacity

            if best_shard and best_capacity > 0:
                best_shard.addresses.add(addr)
                self._address_to_shard[addr] = best_shard.shard_id
                assigned = True
            else:
                # 4. Need a new shard
                new_shard = self._create_new_shard()
                if new_shard:
                    new_shard.addresses.add(addr)
                    self._address_to_shard[addr] = new_shard.shard_id
                    assigned = True

            if not assigned:
                bt.logging.warning(
                    f"[HL_TRACKER] Cannot assign address {addr} - all ports exhausted or unhealthy"
                )

        # 5. Tear down empty shards
        self._teardown_empty_shards()

        # Log summary
        total = len(self._address_to_shard)
        bt.logging.info(
            f"[HL_TRACKER] Address assignment: {total} addresses across {len(self._shards)} shard(s)"
        )

    def _create_new_shard(self) -> Optional['HyperliquidTracker._WebSocketShard']:
        """Create a new shard. Uses a proxy port if available, or direct if no proxy configured."""
        if self._proxy_base_url:
            if not self._available_ports:
                return None
            port = self._available_ports.pop(0)
            proxy_url = self._make_shard_proxy_url(port)
        else:
            # Direct (no proxy) - only one direct shard allowed
            if self._shards:
                return None  # Already have the single direct shard
            proxy_url = None

        sid = self._next_shard_id
        self._next_shard_id += 1
        shard = HyperliquidTracker._WebSocketShard(sid, proxy_url, self)
        self._shards[sid] = shard
        bt.logging.info(f"[HL_TRACKER] Created {shard.label}")
        return shard

    def _teardown_empty_shards(self):
        """Remove shards with no assigned addresses and return their ports."""
        empty_ids = [sid for sid, s in self._shards.items() if not s.addresses]
        for sid in empty_ids:
            shard = self._shards.pop(sid)
            if shard.task and not shard.task.done():
                shard.task.cancel()
            # Return port to available pool (if proxy and port was healthy)
            port = shard.port
            if port is not None and port not in self._unhealthy_ports:
                self._available_ports.append(port)
            bt.logging.info(f"[HL_TRACKER] Tore down empty {shard.label}")

    def _ensure_shard_tasks(self):
        """Ensure all shards with addresses have a running asyncio task."""
        for sid, shard in self._shards.items():
            if shard.addresses and (shard.task is None or shard.task.done()):
                shard.task = asyncio.ensure_future(shard.run())

    # ==================== Message Handling (shared across all shards) ====================

    def _handle_message(self, msg: dict, shard_id: int = 0):
        """Route incoming WebSocket messages."""
        channel = msg.get("channel")

        if channel == "pong":
            return

        if channel == "userFills":
            self._handle_user_fills(msg)
        elif channel == "l2Book":
            book = msg.get("data", {})
            coin = book.get("coin")
            if coin:
                if shard_id == 0:
                    self._orderbooks_fine[coin] = book
                elif shard_id == 1:
                    self._orderbooks_coarse[coin] = book

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

    # ==================== Rejection Broadcast ====================

    def _broadcast_rejection(self, synthetic_hotkey: str, error_msg: str) -> None:
        """Broadcast a rejection/error message to WebSocket subscribers for a subaccount."""
        if not self._ws_notifier_client:
            return
        try:
            self._ws_notifier_client.broadcast_subaccount_dashboard(
                synthetic_hotkey, {"error_msg": error_msg}
            )
        except Exception as e:
            bt.logging.debug(f"[HL_TRACKER] Rejection broadcast failed for {synthetic_hotkey}: {e}")

    def _broadcast_accepted_fill(
        self,
        synthetic_hotkey: str,
        trade_pair: str,
        order_type: str,
        fill_hash: str = "",
    ) -> None:
        """Broadcast an accepted fill event to WebSocket subscribers for a subaccount."""
        if not self._ws_notifier_client:
            return
        try:
            self._ws_notifier_client.broadcast_subaccount_dashboard(
                synthetic_hotkey,
                {
                    "order_event": {
                        "status": "accepted",
                        "trade_pair": trade_pair,
                        "order_type": order_type,
                        "fill_hash": fill_hash or "",
                    }
                },
            )
        except Exception as e:
            bt.logging.debug(f"[HL_TRACKER] Accepted event broadcast failed for {synthetic_hotkey}: {e}")

    # ==================== HL Account State ====================

    def _fetch_hl_account_state(self, hl_address: str) -> Optional[dict]:
        """
        Fetch HL account state via REST and compute portfolio weight per position.

        Returns dict with:
          - total_portfolio_value: perp + spot available (avoiding double-counting)
          - positions: {coin: {"szi": float, "positionValue": float, "weight": float}}
        """
        api_url = ValiConfig.hl_info_url()
        try:
            perp = requests.post(api_url, json={"type": "clearinghouseState", "user": hl_address}, timeout=10).json()
            spot = requests.post(api_url, json={"type": "spotClearinghouseState", "user": hl_address}, timeout=10).json()
            all_mids = requests.post(api_url, json={"type": "allMids"}, timeout=10).json()
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] REST error fetching account state for {hl_address}: {e}")
            return None

        margin = perp.get("crossMarginSummary", perp.get("marginSummary", {}))
        perp_value = float(margin.get("accountValue", 0))

        # Spot: sum USD value of all holdings, subtract amount locked as perp margin
        spot_value, spot_hold = 0.0, 0.0
        for b in spot.get("balances", []):
            coin = b.get("coin", "")
            total_qty = float(b.get("total", 0))
            hold_qty = float(b.get("hold", 0))
            if coin == "USDC":
                usd_val, hold_val = total_qty, hold_qty
            else:
                mid_price = float(all_mids.get(coin, 0))
                usd_val, hold_val = total_qty * mid_price, hold_qty * mid_price
            spot_value += usd_val
            spot_hold += hold_val

        spot_available = spot_value - spot_hold
        total_portfolio_value = perp_value + spot_available

        # Collect per-coin position weights
        positions = {}
        for p in perp.get("assetPositions", []):
            pos = p.get("position", {})
            coin = pos.get("coin", "")
            szi = float(pos.get("szi", 0))
            pos_value_abs = float(pos.get("positionValue", 0))
            sign = 1 if szi >= 0 else -1
            pos_value = sign * pos_value_abs
            weight = pos_value / total_portfolio_value if total_portfolio_value > 0 else 0
            positions[coin] = {"szi": szi, "positionValue": pos_value_abs, "weight": weight}

        return {"total_portfolio_value": total_portfolio_value, "positions": positions}

    # ==================== Fill Processing ====================

    def _process_fill(self, hl_address: str, fill: dict):
        """
        Convert a Hyperliquid fill to a Vanta signal and process it.

        Uses portfolio-weight-to-delta approach:
        1. Fetch HL account state -> compute target position weight
        2. Query current Vanta position -> compute current signed leverage
        3. Delta = target - current -> build incremental Vanta signal
        4. Calculate L2 orderbook slippage for taker fills
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
        if not synthetic_hotkey and isinstance(hl_address, str):
            synthetic_hotkey = self._entity_client.get_synthetic_hotkey_for_hl_address(
                hl_address.lower()
            )
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
            self._broadcast_rejection(synthetic_hotkey, f"Rate limited. Please wait {wait_time:.0f}s.")
            return

        # Elimination check
        elimination_info = self._elimination_client.get_elimination_local_cache(synthetic_hotkey)
        if elimination_info:
            bt.logging.debug(f"[HL_TRACKER] Eliminated miner: {synthetic_hotkey}")
            self._broadcast_rejection(synthetic_hotkey, f"Miner {synthetic_hotkey} has been eliminated.")
            return

        # Subaccount status check
        validation = self._entity_client.validate_hotkey_for_orders(synthetic_hotkey)
        if not validation.get("is_valid"):
            error_message = validation.get('error_message', 'Subaccount validation failed')
            bt.logging.debug(f"[HL_TRACKER] Invalid hotkey: {synthetic_hotkey} - {error_message}")
            self._broadcast_rejection(synthetic_hotkey, error_message)
            return

        # Trade pair blocked check
        if trade_pair.is_blocked:
            bt.logging.debug(f"[HL_TRACKER] Blocked trade pair: {trade_pair_id}")
            self._broadcast_rejection(synthetic_hotkey, f"Trade pair {trade_pair_id} is no longer supported.")
            return

        # Market hours check (only for market orders)
        is_market_open = self._price_fetcher_client.is_market_open(trade_pair, now_ms)
        if not is_market_open:
            bt.logging.debug(f"[HL_TRACKER] Market closed for {trade_pair_id}")
            self._broadcast_rejection(synthetic_hotkey, f"Market is closed for {trade_pair_id}.")
            return

        # === Step 1: Fetch HL account state -> compute target weight ===
        account_state = self._fetch_hl_account_state(hl_address)
        if not account_state or account_state["total_portfolio_value"] <= 0:
            bt.logging.warning(f"[HL_TRACKER] Zero/missing portfolio value for {hl_address}")
            return

        pos_info = account_state["positions"].get(coin)

        # Step 2: Compute target signed weight (+ = long, - = short)
        if pos_info:
            target_signed_weight = pos_info["weight"]
        else:
            target_signed_weight = 0.0  # position closed on HL side

        # Clip to Vanta limits (signed)
        max_lev = ValiConfig.CRYPTO_MAX_LEVERAGE
        min_lev = ValiConfig.CRYPTO_MIN_LEVERAGE
        if abs(target_signed_weight) < min_lev:
            target_signed_weight = 0.0  # below minimum -> treat as flat
        elif abs(target_signed_weight) > max_lev:
            sign = 1 if target_signed_weight > 0 else -1
            target_signed_weight = sign * max_lev

        # Step 3: Get current Vanta position -> compute current signed leverage
        current_position = self._position_client.get_open_position_for_trade_pair(
            synthetic_hotkey, trade_pair_id
        )

        if current_position and not current_position.is_closed_position:
            if current_position.position_type == OrderType.LONG:
                current_signed_lev = current_position.net_leverage
            elif current_position.position_type == OrderType.SHORT:
                current_signed_lev = -current_position.net_leverage
            else:
                current_signed_lev = 0.0
        else:
            current_signed_lev = 0.0

        # Step 4: Compute delta order
        delta = target_signed_weight - current_signed_lev

        if abs(delta) < min_lev and target_signed_weight != 0.0:
            bt.logging.debug(f"[HL_TRACKER] Delta {delta:.4f} below min leverage, skipping")
            return

        # Step 5: Convert delta to order_type + leverage
        if target_signed_weight == 0.0:
            order_type = "FLAT"
            leverage = 0.0
        elif delta > 0:
            order_type = "LONG"
            leverage = delta
        else:
            order_type = "SHORT"
            leverage = abs(delta)

        # === Step 6: Calculate L2 orderbook slippage (multi-resolution) ===
        # Walk the fine-grained book (nSigFigs=5) first for precise near-spread
        # pricing, then continue with the coarse book (nSigFigs=2) for remaining
        # depth if the order penetrates all fine levels.
        is_taker = fill.get("crossed", True)
        slippage_pct = 0.0

        fine_book = self._orderbooks_fine.get(coin, {})
        coarse_book = self._orderbooks_coarse.get(coin, {})
        has_book = fine_book or coarse_book

        if is_taker and has_book:
            # Use fine book for mid-price if available, else coarse
            primary = fine_book or coarse_book
            bids = primary.get("levels", [[], []])[0]
            asks = primary.get("levels", [[], []])[1]
            if asks and bids:
                mid = (float(asks[0]["px"]) + float(bids[0]["px"])) / 2
                if mid > 0:
                    if order_type == "FLAT":
                        # Closing a position: trade size is the current position's leverage,
                        # direction is opposite to position type (closing LONG = sell = eat bids,
                        # closing SHORT = buy = eat asks)
                        translated_size_usd = abs(current_signed_lev) * account_size
                        is_buying = current_signed_lev < 0  # closing SHORT = buying
                    else:
                        translated_size_usd = leverage * account_size
                        is_buying = order_type == "LONG"

                    side_idx = 1 if is_buying else 0  # asks if buying, bids if selling
                    fine_levels = fine_book.get("levels", [[], []])[side_idx]
                    coarse_levels = coarse_book.get("levels", [[], []])[side_idx]

                    # Phase 1: walk fine-grained levels
                    fills_result, remaining = simulate_fill(fine_levels, translated_size_usd, "usd")

                    # Phase 2: if order penetrated all fine levels, continue with
                    # coarse levels beyond the fine book's price coverage
                    if remaining > 0 and coarse_levels and fine_levels:
                        last_fine_px = float(fine_levels[-1]["px"])
                        if is_buying:
                            deeper = [l for l in coarse_levels if float(l["px"]) > last_fine_px]
                        else:
                            deeper = [l for l in coarse_levels if float(l["px"]) < last_fine_px]
                        coarse_fills, remaining = simulate_fill(deeper, remaining, "usd")
                        fills_result.extend(coarse_fills)
                    elif remaining > 0 and coarse_levels and not fine_levels:
                        # No fine book available, use coarse only
                        coarse_fills, remaining = simulate_fill(coarse_levels, remaining, "usd")
                        fills_result.extend(coarse_fills)

                    total_coins = sum(f[1] for f in fills_result)
                    total_usd = sum(f[2] for f in fills_result)
                    avg_price = total_usd / total_coins if total_coins > 0 else mid
                    if is_buying:
                        slippage_pct = (avg_price - mid) / mid
                    else:
                        slippage_pct = (mid - avg_price) / mid
                    slippage_pct = max(0.0, min(slippage_pct, 0.03))  # clip to 3% max

        # === Build signal ===
        signal = {
            "order_type": order_type,
            "leverage": leverage,
            "trade_pair": {"trade_pair_id": trade_pair_id},
            "execution_type": "MARKET",
            "is_hl": True,
            "is_hl_taker": is_taker,
            "hl_slippage": slippage_pct,
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
            self._broadcast_accepted_fill(
                synthetic_hotkey=synthetic_hotkey,
                trade_pair=trade_pair_id,
                order_type=order_type,
                fill_hash=fill.get("hash") or fill.get("tid") or "",
            )

            bt.logging.info(
                f"[HL_TRACKER] Processed fill: {coin} target_weight={target_signed_weight:+.4f} "
                f"current_lev={current_signed_lev:+.4f} delta={delta:+.4f} -> "
                f"{synthetic_hotkey} {order_type} leverage={leverage:.4f} slippage={slippage_pct:.6f}"
            )

        except SignalException as e:
            bt.logging.warning(f"[HL_TRACKER] Signal rejected for {synthetic_hotkey}: {e}")
            self._broadcast_rejection(synthetic_hotkey, f"Order rejected: {e}")
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Order processing error for {synthetic_hotkey}: {e}")
            self._broadcast_rejection(synthetic_hotkey, f"Order rejected: {e}")
            bt.logging.error(traceback.format_exc())