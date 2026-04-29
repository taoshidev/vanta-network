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
from shared_objects.rate_limiter import RateLimiter
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode, HL_COIN_TO_DYNAMIC_TRADE_PAIR
from vanta_api.websocket_notifier import WebSocketNotifierClient


class _PortHealthRecord:
    """Tracks health state for a single proxy port with exponential backoff probing."""

    __slots__ = (
        "port",
        "healthy",
        "unhealthy_since",
        "last_probe_time",
        "consecutive_probe_failures",
        "rest_consecutive_failures",
    )

    def __init__(self, port: int):
        self.port = port
        self.healthy = True
        self.unhealthy_since: Optional[float] = None
        self.last_probe_time: Optional[float] = None
        self.consecutive_probe_failures = 0
        self.rest_consecutive_failures = 0

    def mark_unhealthy(self):
        if self.healthy:
            self.unhealthy_since = time.time()
            self.consecutive_probe_failures = 0
        self.healthy = False

    def mark_healthy(self):
        self.healthy = True
        self.unhealthy_since = None
        self.last_probe_time = None
        self.consecutive_probe_failures = 0
        self.rest_consecutive_failures = 0

    def cooldown_seconds(self) -> float:
        base = ValiConfig.HL_PORT_HEALTH_PROBE_INTERVAL_S
        cap = ValiConfig.HL_PORT_HEALTH_MAX_COOLDOWN_S
        return min(base * (2 ** self.consecutive_probe_failures), cap)

    def is_probe_due(self) -> bool:
        if self.healthy:
            return False
        ref_time = self.last_probe_time or self.unhealthy_since or 0
        return (time.time() - ref_time) >= self.cooldown_seconds()


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
                    "subscription": {"type": "userFills", "user": addr.lower(), "aggregateByTime": True},
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
        self._port_health: Dict[int, _PortHealthRecord] = {}

        # Metrics
        self._fills_processed = 0
        self._last_fill_time: Optional[float] = None
        # Backup REST poll state
        self._last_poll_time_ms: Dict[str, int] = {}  # hl_address -> watermark (epoch ms)
        self._backup_poll_task: Optional[asyncio.Task] = None
        self._backup_fills_caught = 0
        self._backup_polls_total = 0
        self._proxy_index_rest = 0
        self._hl_universe: dict = dict(HL_COIN_TO_DYNAMIC_TRADE_PAIR)  # seed from persisted registry; replaced on first refresh
        self._last_universe_refresh: float = 0.0
        # (hl_address_lower) -> {coin: last_observed_szi_float}
        # Used by reconciliation to detect real HL position-size changes vs
        # PnL-driven weight drift. Populated from _fetch_hl_account_state results.
        self._last_observed_szi: Dict[str, Dict[str, float]] = {}
        self._load_backup_poll_watermarks()
        self._load_observed_szi()

    @property
    def _unhealthy_ports(self) -> Set[int]:
        return {p for p, rec in self._port_health.items() if not rec.healthy}

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
        port_health_list = []
        now = time.time()
        for port, rec in sorted(self._port_health.items()):
            entry = {
                "port": port,
                "healthy": rec.healthy,
                "rest_failures": rec.rest_consecutive_failures,
                "probe_failures": rec.consecutive_probe_failures,
            }
            if not rec.healthy:
                ref_time = rec.last_probe_time or rec.unhealthy_since or 0
                next_probe_in = max(0, ref_time + rec.cooldown_seconds() - now)
                entry["next_probe_in_s"] = round(next_probe_in, 1)
            port_health_list.append(entry)
        return {
            "shards": shard_statuses,
            "total_connected": sum(1 for s in self._shards.values() if s.connected),
            "total_subscribed_addresses": len(self._address_to_shard),
            "fills_processed": self._fills_processed,
            "last_fill_time": self._last_fill_time,
            "proxy_configured": self._proxy_base_url is not None,
            "available_ports": len(self._available_ports),
            "unhealthy_ports": len(self._unhealthy_ports),
            "port_health": port_health_list,
            "backup_poll": {
                "fills_caught": self._backup_fills_caught,
                "total_polls": self._backup_polls_total,
                "tracked_addresses": len(self._last_poll_time_ms),
            },
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

        self._port_health = {port: _PortHealthRecord(port) for port in self._available_ports}

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
        # Kick off universe refresh in a thread so WebSocket shards can start immediately.
        self._last_universe_refresh = time.time()
        loop = asyncio.get_running_loop()
        asyncio.ensure_future(loop.run_in_executor(None, self._refresh_hl_universe))
        self._backup_poll_task = asyncio.ensure_future(self._backup_poll_cycle())

        try:
            while not self._stop_event.is_set():
                try:
                    self._assign_addresses_to_shards()
                    self._ensure_shard_tasks()
                    self._probe_unhealthy_ports()
                except Exception as e:
                    bt.logging.error(f"[HL_TRACKER] Orchestrator error: {e}")
                    bt.logging.error(traceback.format_exc())

                # Wait before next refresh cycle
                for _ in range(int(self.ADDRESS_REFRESH_INTERVAL_S)):
                    if self._stop_event.is_set():
                        return
                    await asyncio.sleep(1.0)
        finally:
            if self._backup_poll_task and not self._backup_poll_task.done():
                self._backup_poll_task.cancel()
                try:
                    await self._backup_poll_task
                except asyncio.CancelledError:
                    pass
            self._save_backup_poll_watermarks()

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
            # Mark port unhealthy for probe-based recovery.
            port = shard.port
            if port is not None and port in self._port_health:
                self._port_health[port].mark_unhealthy()
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
        new_addresses = active_addresses - already_assigned
        to_assign = new_addresses | orphaned

        if not to_assign:
            # 5. Tear down empty shards
            self._teardown_empty_shards()
            return

        # For addresses without persisted watermark, start with restart lookback.
        # This allows catching fills that occurred while validator was down.
        now_ms = int(time.time() * 1000)
        watermark_changed = False
        for addr in new_addresses:
            if addr not in self._last_poll_time_ms:
                self._last_poll_time_ms[addr] = max(
                    0,
                    now_ms - ValiConfig.HL_BACKUP_RESTART_LOOKBACK_MS,
                )
                watermark_changed = True

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
        if watermark_changed:
            self._save_backup_poll_watermarks()

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

    def _probe_unhealthy_ports(self):
        """Probe unhealthy ports and restore them to rotation on success."""
        if not self._proxy_base_url:
            return
        for port, rec in self._port_health.items():
            if rec.healthy or not rec.is_probe_due():
                continue

            rec.last_probe_time = time.time()
            proxy_url = self._make_shard_proxy_url(port)
            try:
                session = requests.Session()
                session.proxies = {"http": proxy_url, "https": proxy_url}
                resp = session.post(ValiConfig.hl_info_url(), json={"type": "meta"}, timeout=10)
                resp.raise_for_status()
                session.close()
                rec.mark_healthy()
                if port not in self._available_ports:
                    self._available_ports.append(port)
                bt.logging.info(f"[HL_TRACKER] Port {port} probe succeeded, restored")
            except Exception as e:
                rec.consecutive_probe_failures += 1
                bt.logging.debug(
                    f"[HL_TRACKER] Port {port} probe failed "
                    f"(attempt={rec.consecutive_probe_failures}, cooldown={rec.cooldown_seconds():.0f}s): {e}"
                )

    # ==================== Message Handling (shared across all shards) ====================

    def _handle_message(self, msg: dict, shard_id: int = 0):
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

    # ==================== Rejection Broadcast ====================

    def _broadcast_rejection(self, synthetic_hotkey: str, error_msg: str) -> None:
        """Broadcast a rejection/error message to WebSocket subscribers for a subaccount."""
        bt.logging.info(f"[HL_TRACKER] Broadcasting rejection for {synthetic_hotkey}: {error_msg}")
        if not self._ws_notifier_client:
            bt.logging.debug(f"[HL_TRACKER] No WS notifier client, skipping rejection broadcast")
            return
        try:
            self._ws_notifier_client.broadcast_subaccount_dashboard(synthetic_hotkey)
            bt.logging.debug(f"[HL_TRACKER] Rejection broadcast sent for {synthetic_hotkey}")
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
            self._ws_notifier_client.broadcast_subaccount_dashboard(synthetic_hotkey)
        except Exception as e:
            bt.logging.debug(f"[HL_TRACKER] Accepted event broadcast failed for {synthetic_hotkey}: {e}")

    # ==================== HL Account State ====================

    def _make_proxied_session(self) -> requests.Session:
        """
        Create a requests session that rotates across healthy proxy ports.
        Falls back to direct session when no healthy ports are available.
        """
        session = requests.Session()
        session._hl_proxy_port = None  # type: ignore[attr-defined]

        if not self._proxy_base_url:
            return session

        all_ports = list(self._available_ports)
        for shard in self._shards.values():
            if shard.healthy and shard.port is not None:
                all_ports.append(shard.port)
        all_ports = sorted(set(all_ports))
        healthy_ports = [p for p in all_ports if p not in self._unhealthy_ports]
        if not healthy_ports:
            if all_ports:
                bt.logging.warning("[HL_BACKUP] All proxy ports unhealthy; falling back to direct REST")
            return session

        port = healthy_ports[self._proxy_index_rest % len(healthy_ports)]
        self._proxy_index_rest += 1
        proxy_url = self._make_shard_proxy_url(port)
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session._hl_proxy_port = port  # type: ignore[attr-defined]
        return session

    def _report_rest_proxy_success(self, port: Optional[int]):
        if port is not None and port in self._port_health:
            self._port_health[port].rest_consecutive_failures = 0

    def _report_rest_proxy_failure(self, port: Optional[int]):
        if port is None or port not in self._port_health:
            return
        record = self._port_health[port]
        record.rest_consecutive_failures += 1
        if record.rest_consecutive_failures >= ValiConfig.HL_PORT_REST_FAILURE_THRESHOLD:
            record.mark_unhealthy()
            bt.logging.warning(
                f"[HL_BACKUP] Port {port} marked unhealthy after "
                f"{record.rest_consecutive_failures} REST failures"
            )

    def _fetch_hl_account_state(self, hl_address: str) -> Optional[dict]:
        """
        Fetch HL account state via REST and compute portfolio weight per position.

        Returns dict with:
          - total_portfolio_value: perp + spot available (avoiding double-counting)
          - positions: {coin: {"szi": float, "positionValue": float, "weight": float}}
        """
        api_url = ValiConfig.hl_info_url()
        session = self._make_proxied_session()
        proxy_port = getattr(session, "_hl_proxy_port", None)
        try:
            perp = session.post(api_url, json={"type": "clearinghouseState", "user": hl_address}, timeout=10).json()
            spot = session.post(api_url, json={"type": "spotClearinghouseState", "user": hl_address}, timeout=10).json()
            all_mids = session.post(api_url, json={"type": "allMids"}, timeout=10).json()
            self._report_rest_proxy_success(proxy_port)
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] REST error fetching account state for {hl_address}: {e}")
            self._report_rest_proxy_failure(proxy_port)
            return None
        finally:
            session.close()

        if not isinstance(perp, dict):
            bt.logging.info(f"[HL_TRACKER] No perp account for {hl_address}")
            return None
        if not isinstance(spot, dict):
            spot = {}
        if not isinstance(all_mids, dict):
            all_mids = {}

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

        result = {"total_portfolio_value": total_portfolio_value, "positions": positions}
        self._remember_hl_szi(hl_address, result)
        return result

    def _remember_hl_szi(self, hl_address: str, account_state: dict) -> None:
        """Cache per-coin szi from the latest HL account state for reconcile gating.

        Persists on every update so a validator restart sees the same szi map
        the pre-restart process had — otherwise the first reconcile cycle
        post-restart would recompute fresh weights against an empty cache and
        emit one drift order per open HL position.
        """
        key = hl_address.lower() if isinstance(hl_address, str) else hl_address
        new_snapshot = {
            coin: float(info.get("szi", 0.0))
            for coin, info in account_state.get("positions", {}).items()
        }
        if self._last_observed_szi.get(key) == new_snapshot:
            return
        self._last_observed_szi[key] = new_snapshot
        self._save_observed_szi()

    # ==================== Backup REST Poll ====================

    def _load_backup_poll_watermarks(self):
        """Load persisted HL backup watermarks for restart continuity."""
        try:
            try:
                data = json.loads(ValiBkpUtils.get_file(ValiBkpUtils.get_hl_backup_watermarks_path()))
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}
            if not isinstance(data, dict):
                return
            loaded: Dict[str, int] = {}
            for addr, ts in data.items():
                if not isinstance(addr, str):
                    continue
                try:
                    loaded[addr] = int(ts)
                except (TypeError, ValueError):
                    continue
            self._last_poll_time_ms = loaded
            if loaded:
                bt.logging.info(
                    f"[HL_BACKUP] Loaded {len(loaded)} persisted watermark(s)"
                )
        except Exception as e:
            bt.logging.warning(f"[HL_BACKUP] Failed to load watermarks: {e}")

    def _fetch_30d_avg_volume(self, coin: str) -> float:
        """Return 30-day mean daily USD volume for *coin* using HL candleSnapshot.

        Uses complete days only (endTime = today midnight UTC).
        Retries up to 3 times on failure.  Returns 0.0 on persistent error.
        """
        today_midnight_ms = (int(time.time()) // 86400) * 86400 * 1000
        start_ms = today_midnight_ms - ValiConfig.HL_LIQUIDITY_LOOKBACK_DAYS * 86400 * 1000
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": today_midnight_ms,
            },
        }
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                resp = requests.post(ValiConfig.hl_info_url(), json=payload, timeout=15)
                if resp.status_code == 429:
                    bt.logging.warning(f"[HL_TRACKER] 429 on {coin}, waiting 60s...")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                candles = resp.json()
                if not candles:
                    return 0.0
                daily_vols = [float(c["v"]) * float(c["c"]) for c in candles]
                return sum(daily_vols) / len(daily_vols)
            except Exception as e:
                if attempt == 2:
                    bt.logging.warning(f"[HL_TRACKER] candleSnapshot failed for {coin}: {e}")
        return 0.0

    def _fetch_dex_collateral_map(self, dex_names: List[Optional[str]]) -> Dict[str, str]:
        """Return {dex_name: collateral_token} for all named dexes.

        Empty string key represents the default crypto dex, which is always USDC.
        Fetches spotMeta once for token index resolution, then queries each named dex.
        """
        result: Dict[str, str] = {"": "USDC"}  # default crypto dex is always USDC
        token_index_map: Dict[int, str] = {}
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                resp = requests.post(ValiConfig.hl_info_url(), json={"type": "spotMeta"}, timeout=10)
                if resp.status_code == 429:
                    bt.logging.warning(f"[HL_TRACKER] 429 fetching spotMeta, waiting 60s...")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                token_index_map = {t["index"]: t["name"] for t in resp.json().get("tokens", [])}
                break
            except Exception as e:
                if attempt == 2:
                    bt.logging.warning(f"[HL_TRACKER] Failed to fetch spotMeta: {e}")

        for dex in dex_names:
            if dex is None:
                continue
            time.sleep(1)
            try:
                resp = requests.post(ValiConfig.hl_info_url(), json={"type": "meta", "dex": dex}, timeout=10)
                resp.raise_for_status()
                idx = resp.json().get("collateralToken", 0)
                result[dex] = token_index_map.get(idx, "USDC")
            except Exception as e:
                bt.logging.warning(f"[HL_TRACKER] Failed to fetch collateral token for dex={dex}: {e}")
                result[dex] = "USDC"

        return result

    def _fetch_candidates_for_dex(self, dex: Optional[str]) -> List[tuple]:
        """Return [(coin, maxLeverage), ...] for *dex* (None = default crypto dex)."""
        payload: dict = {"type": "metaAndAssetCtxs"}
        if dex is not None:
            payload["dex"] = dex
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                resp = requests.post(ValiConfig.hl_info_url(), json=payload, timeout=10)
                if resp.status_code == 429:
                    bt.logging.warning(f"[HL_TRACKER] 429 fetching candidates for dex={dex}, waiting 60s...")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                meta, _ = resp.json()
                return [(asset["name"], asset["maxLeverage"]) for asset in meta["universe"]]
            except Exception as e:
                if attempt == 2:
                    bt.logging.warning(f"[HL_TRACKER] Failed to fetch candidates for dex={dex}: {e}")
        return []

    def _refresh_hl_universe(self):
        """Discover all dexes, apply 30-day liquidity filter, update _hl_universe + HL_DYNAMIC_REGISTRY."""
        from vali_objects.vali_config import HL_DYNAMIC_REGISTRY, DynamicTradePair

        # 1. Discover all dex names; None represents the default crypto dex.
        # perpDexs returns a list where the first element is None (default dex) and the rest
        # are dicts like {"name": "xyz", ...} — extract the name strings from those dicts.
        named_dexes: List[str] = []
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                resp = requests.post(ValiConfig.hl_info_url(), json={"type": "perpDexs"}, timeout=10)
                if resp.status_code == 429:
                    bt.logging.warning(f"[HL_TRACKER] 429 fetching perpDexs, waiting 60s...")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                named_dexes = [d["name"] for d in resp.json() if isinstance(d, dict) and d.get("name")]
                break
            except Exception as e:
                if attempt == 2:
                    bt.logging.warning(f"[HL_TRACKER] Failed to fetch perpDexs: {e} — using default dex only")
        dex_names: List[Optional[str]] = [None] + named_dexes

        # 2. Fetch collateral token per dex (one spotMeta call + one meta call per named dex)
        dex_to_collateral = self._fetch_dex_collateral_map(dex_names)

        # 3. Collect (coin, maxLeverage) across all dexes (1s sleep between dexes to stay under rate limit)
        all_candidates: List[tuple] = []
        for dex in dex_names:
            for coin, max_lev in self._fetch_candidates_for_dex(dex):
                all_candidates.append((coin, max_lev))
            time.sleep(1)

        if not all_candidates:
            bt.logging.warning("[HL_TRACKER] No candidates found — keeping existing registry")
            return

        # 4. Fetch 30-day avg USD volume sequentially (1s sleep → ≤60 req/min, under 1200 weight/min limit)
        avg_volumes: Dict[str, float] = {}
        for coin, _ in all_candidates:
            avg_volumes[coin] = self._fetch_30d_avg_volume(coin)
            time.sleep(1)

        # 5. Filter and build universe
        new_universe = {}
        for coin, max_lev in all_candidates:
            if coin.split(":")[-1] in ValiConfig.HL_EXCLUDED_ASSETS:
                continue
            if avg_volumes.get(coin, 0.0) < ValiConfig.HL_MIN_LIQUIDITY_USD:
                continue
            dex_name = coin.split(":")[0] if ":" in coin else ""
            collateral = dex_to_collateral.get(dex_name, "USDC")
            # hs_max_leverage = (ValiConfig.HS_HIGH_TIER_MAX_LEVERAGE
            #                    if max_lev >= ValiConfig.HL_HIGH_TIER_THRESHOLD
            #                    else ValiConfig.HS_MAX_LEVERAGE)
            hs_max_leverage = ValiConfig.HS_MAX_LEVERAGE
            new_universe[coin] = DynamicTradePair(
                trade_pair_id=f"{coin}{collateral}",
                trade_pair=f"{coin}/{collateral}",
                hl_coin=coin,
                max_leverage=hs_max_leverage,
            )

        self._hl_universe = new_universe
        from vali_objects.vali_config import HL_COIN_TO_DYNAMIC_TRADE_PAIR
        for dtp in new_universe.values():
            HL_DYNAMIC_REGISTRY[dtp.trade_pair_id] = dtp
            HL_COIN_TO_DYNAMIC_TRADE_PAIR[dtp.hl_coin] = dtp
        self._persist_hl_dynamic_registry()
        self._last_universe_refresh = time.time()
        bt.logging.info(
            f"[HL_TRACKER] Universe: {len(new_universe)} active across {len(named_dexes) + 1} dex(es) "
            f"(30d avg vol ≥ ${ValiConfig.HL_MIN_LIQUIDITY_USD:,}), {len(HL_DYNAMIC_REGISTRY)} total"
        )

    def _persist_hl_dynamic_registry(self):
        """Write HL_DYNAMIC_REGISTRY to disk so other processes can load it."""
        from vali_objects.vali_config import HL_DYNAMIC_REGISTRY, _HL_REGISTRY_PATH
        data = {
            tid: {
                "trade_pair_id":    dtp.trade_pair_id,
                "trade_pair":       dtp.trade_pair,
                "hl_coin":          dtp.hl_coin,
                "max_leverage":     dtp.max_leverage,
                "min_leverage":     dtp.min_leverage,
                "fees":             dtp.fees,
                "trade_pair_category": dtp.trade_pair_category.value,
            }
            for tid, dtp in HL_DYNAMIC_REGISTRY.items()
        }
        try:
            with open(_HL_REGISTRY_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            bt.logging.warning(f"[HL_TRACKER] Failed to persist HL_DYNAMIC_REGISTRY: {e}")

    def _save_backup_poll_watermarks(self):
        """Persist HL backup watermarks for restart continuity."""
        try:
            serializable = {k: int(v) for k, v in self._last_poll_time_ms.items()}
            ValiBkpUtils.write_file(ValiBkpUtils.get_hl_backup_watermarks_path(), serializable)
        except Exception as e:
            bt.logging.warning(f"[HL_BACKUP] Failed to persist watermarks: {e}")

    def _load_observed_szi(self):
        """Load persisted per-address szi snapshot for restart continuity."""
        try:
            try:
                data = json.loads(ValiBkpUtils.get_file(ValiBkpUtils.get_hl_observed_szi_path()))
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}
            if not isinstance(data, dict):
                return
            loaded: Dict[str, Dict[str, float]] = {}
            for addr, coin_map in data.items():
                if not isinstance(addr, str) or not isinstance(coin_map, dict):
                    continue
                parsed: Dict[str, float] = {}
                for coin, szi in coin_map.items():
                    try:
                        parsed[str(coin)] = float(szi)
                    except (TypeError, ValueError):
                        continue
                loaded[addr] = parsed
            self._last_observed_szi = loaded
            if loaded:
                bt.logging.info(
                    f"[HL_BACKUP] Loaded szi snapshot for {len(loaded)} address(es)"
                )
        except Exception as e:
            bt.logging.warning(f"[HL_BACKUP] Failed to load observed szi: {e}")

    def _save_observed_szi(self):
        """Persist per-address szi snapshot so reconcile gate survives restart."""
        try:
            serializable = {
                addr: {coin: float(szi) for coin, szi in coin_map.items()}
                for addr, coin_map in self._last_observed_szi.items()
            }
            ValiBkpUtils.write_file(ValiBkpUtils.get_hl_observed_szi_path(), serializable)
        except Exception as e:
            bt.logging.warning(f"[HL_BACKUP] Failed to persist observed szi: {e}")

    async def _fetch_fills_by_time(self, hl_address: str, start_time_ms: int) -> Optional[List[dict]]:
        """Fetch fills for a tracked address via userFillsByTime REST endpoint."""
        api_url = ValiConfig.hl_info_url()
        payload = {"type": "userFillsByTime", "user": hl_address, "startTime": start_time_ms}
        loop = asyncio.get_event_loop()
        session = self._make_proxied_session()
        proxy_port = getattr(session, "_hl_proxy_port", None)

        try:
            def _do_request():
                try:
                    resp = session.post(api_url, json=payload, timeout=10)
                    resp.raise_for_status()
                    try:
                        return resp.json()
                    except Exception:
                        body_preview = (resp.text or "")[:400]
                        bt.logging.warning(
                            f"[HL_BACKUP] Non-JSON response for address={hl_address} "
                            f"status={resp.status_code} body_preview={body_preview!r}"
                        )
                        raise
                finally:
                    session.close()

            result = await loop.run_in_executor(None, _do_request)
            self._backup_polls_total += 1
            self._report_rest_proxy_success(proxy_port)
            if isinstance(result, list):
                return result

            bt.logging.warning(
                f"[HL_BACKUP] Unexpected non-list userFillsByTime response "
                f"address={hl_address} type={type(result).__name__} "
                f"response_preview={json.dumps(result, default=str)[:400] if result is not None else 'None'}"
            )
            return []
        except Exception as e:
            bt.logging.warning(
                f"[HL_BACKUP] userFillsByTime failed address={hl_address} "
                f"start_ms={start_time_ms} url={api_url} "
                f"hl_use_testnet={ValiConfig.HL_USE_TESTNET} proxy_port={proxy_port} error={e}"
            )
            self._backup_polls_total += 1
            self._report_rest_proxy_failure(proxy_port)
            return None

    async def _backup_poll_cycle(self):
        """Periodic background poll for fills missed by websocket downtime."""
        await asyncio.sleep(10.0)
        bt.logging.info("[HL_BACKUP] Backup REST polling started")

        while not self._stop_event.is_set():
            try:
                if time.time() - self._last_universe_refresh > ValiConfig.HL_UNIVERSE_REFRESH_INTERVAL_S:
                    await asyncio.get_running_loop().run_in_executor(None, self._refresh_hl_universe)
                tracked_addresses = list(self._address_to_shard.keys())
                if not tracked_addresses:
                    await asyncio.sleep(ValiConfig.HL_BACKUP_POLL_INTERVAL_S)
                    continue

                min_delay_s = 60.0 / max(1, ValiConfig.HL_BACKUP_POLL_RATE_BUDGET)
                catches_this_cycle = 0
                now_ms = int(time.time() * 1000)

                for hl_address in tracked_addresses:
                    if self._stop_event.is_set():
                        break

                    start_ms = self._last_poll_time_ms.get(
                        hl_address,
                        now_ms - ValiConfig.HL_BACKUP_POLL_LOOKBACK_MS,
                    )
                    fills = await self._fetch_fills_by_time(hl_address, start_ms)

                    if fills is not None:
                        for fill in fills:
                            fill_hash = fill.get("hash") or fill.get("tid")
                            if not fill_hash or fill_hash in self._processed_hashes:
                                continue
                            self._record_hash(fill_hash)
                            try:
                                self._process_fill(hl_address, fill)
                                catches_this_cycle += 1
                                self._backup_fills_caught += 1
                                bt.logging.info(
                                    f"[HL_BACKUP] Caught missed fill: {fill_hash} "
                                    f"address={hl_address} coin={fill.get('coin')}"
                                )
                            except Exception as e:
                                bt.logging.error(f"[HL_BACKUP] Error processing fill {fill_hash}: {e}")
                                bt.logging.error(traceback.format_exc())

                        # Advance only on successful fetch to avoid skipping gaps.
                        self._last_poll_time_ms[hl_address] = int(time.time() * 1000)
                        self._save_backup_poll_watermarks()
                        if len(fills) == 0:
                            # Reconcile from current account state when no new fills are returned.
                            # This catches offline drift even if a historical fill was missed.
                            self._reconcile_address_positions(hl_address)

                    await asyncio.sleep(min_delay_s)

                # Cleanup stale watermarks.
                active_set = set(self._address_to_shard.keys())
                stale = [addr for addr in self._last_poll_time_ms if addr not in active_set]
                for addr in stale:
                    del self._last_poll_time_ms[addr]
                if stale:
                    self._save_backup_poll_watermarks()

                if catches_this_cycle > 0:
                    bt.logging.info(
                        f"[HL_BACKUP] Cycle caught {catches_this_cycle} missed fill(s). "
                        f"total={self._backup_fills_caught}"
                    )
            except Exception as e:
                bt.logging.error(f"[HL_BACKUP] Poll cycle error: {e}")
                bt.logging.error(traceback.format_exc())

            await asyncio.sleep(ValiConfig.HL_BACKUP_POLL_INTERVAL_S)

    def _reconcile_address_positions(self, hl_address: str):
        """Reconcile only coins whose szi changed since last observation.

        PnL- or funding-driven changes to total_portfolio_value (which shifts
        weight but not szi) no longer trigger delta orders.
        """
        key = hl_address.lower() if isinstance(hl_address, str) else hl_address
        prev_szi = dict(self._last_observed_szi.get(key, {}))  # snapshot before fetch

        account_state = self._fetch_hl_account_state(hl_address)
        if not account_state:
            return

        synthetic_hotkey = self._entity_client.get_synthetic_hotkey_for_hl_address(hl_address)
        if not synthetic_hotkey and isinstance(hl_address, str):
            synthetic_hotkey = self._entity_client.get_synthetic_hotkey_for_hl_address(hl_address.lower())
        if not synthetic_hotkey:
            return

        current_positions = account_state.get("positions", {})

        # coins seen now (real open or newly flat)
        current_szi = {c: float(info.get("szi", 0.0)) for c, info in current_positions.items()}
        # coins that were open before but vanished in current state (closed outside of our ws feed)
        for coin in prev_szi:
            current_szi.setdefault(coin, 0.0)

        # Include any Vanta-open coins the HL account no longer lists, so we can
        # drive them to FLAT if HL closed behind our back.
        from vali_objects.vali_config import HL_DYNAMIC_REGISTRY
        trade_pair_to_coin = {
            dtp.trade_pair_id: dtp.hl_coin
            for dtp in list(HL_DYNAMIC_REGISTRY.values())
        }
        try:
            open_positions = self._position_client.get_positions_for_one_hotkey(
                synthetic_hotkey, only_open_positions=True
            )
        except Exception as e:
            bt.logging.debug(f"[HL_BACKUP] Failed to fetch open positions for reconcile {synthetic_hotkey}: {e}")
            open_positions = []
        # Coins Vanta has open but HL no longer lists must always be reconciled,
        # even when prev_szi is empty (e.g., first cycle after restart), so the
        # szi-unchanged-at-zero compare can't mask a stale Vanta open.
        force_reconcile: Set[str] = set()
        for pos in open_positions or []:
            if pos and not pos.is_closed_position:
                coin = trade_pair_to_coin.get(pos.trade_pair.trade_pair_id)
                if coin and coin not in current_szi:
                    current_szi[coin] = 0.0
                    force_reconcile.add(coin)

        # Only act on coins whose szi changed (creation, close, or any delta)
        # or on Vanta-open coins with no current HL exposure.
        coins_to_reconcile = [
            coin for coin, new_sz in current_szi.items()
            if coin in force_reconcile or new_sz != prev_szi.get(coin, 0.0)
        ]

        if not coins_to_reconcile:
            return  # pure PnL / funding drift — nothing to do

        bt.logging.info(
            f"[HL_BACKUP] Reconciling {len(coins_to_reconcile)} coin(s) for {hl_address} "
            f"with szi changes: {coins_to_reconcile}"
        )
        for coin in sorted(coins_to_reconcile):
            try:
                self._process_fill(
                    hl_address,
                    {"coin": coin, "crossed": False},
                    account_state=account_state,
                )
            except Exception as e:
                bt.logging.debug(f"[HL_BACKUP] Reconcile failed address={hl_address} coin={coin}: {e}")

    # ==================== Fill Processing ====================

    def _process_fill(self, hl_address: str, fill: dict, account_state: Optional[dict] = None):
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

        # Map coin to trade pair via dynamic registry
        from vali_objects.vali_config import HL_COIN_TO_DYNAMIC_TRADE_PAIR
        trade_pair = self._hl_universe.get(coin)
        below_threshold = False
        if not trade_pair:
            # Coin is below liquidity threshold or not yet refreshed — check full registry
            # to handle close/reduce fills for previously tracked coins.
            trade_pair = HL_COIN_TO_DYNAMIC_TRADE_PAIR.get(coin)
            if not trade_pair:
                bt.logging.debug(f"[HL_TRACKER] Unknown coin: {coin}")
                return
            below_threshold = True
        trade_pair_id = trade_pair.trade_pair_id

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
            bt.logging.info(f"[HL_TRACKER] Rate limited: {synthetic_hotkey} {coin}, wait {wait_time:.1f}s")
            self._broadcast_rejection(synthetic_hotkey, f"Rate limited. Please wait {wait_time:.0f}s.")
            return

        # Elimination check
        elimination_info = self._elimination_client.get_elimination_local_cache(synthetic_hotkey)
        if elimination_info:
            bt.logging.info(f"[HL_TRACKER] Eliminated miner: {synthetic_hotkey} {coin}")
            self._broadcast_rejection(synthetic_hotkey, f"Miner {synthetic_hotkey} has been eliminated.")
            return

        # Subaccount status check
        validation = self._entity_client.validate_hotkey_for_orders(synthetic_hotkey)
        if not validation.get("is_valid"):
            error_message = validation.get('error_message', 'Subaccount validation failed')
            bt.logging.info(f"[HL_TRACKER] Invalid hotkey: {synthetic_hotkey} {coin} - {error_message}")
            self._broadcast_rejection(synthetic_hotkey, error_message)
            return

        # Trade pair blocked check
        if trade_pair.is_blocked:
            bt.logging.info(f"[HL_TRACKER] Blocked trade pair: {trade_pair_id} ({synthetic_hotkey})")
            self._broadcast_rejection(synthetic_hotkey, f"Trade pair {trade_pair_id} is no longer supported.")
            return

        # Market hours check (only for market orders)
        # DynamicTradePairs are HL instruments — HL trades 24/7 regardless of asset type,
        # so skip the market-hours check entirely for them.
        from vali_objects.vali_config import DynamicTradePair
        if isinstance(trade_pair, DynamicTradePair):
            is_market_open = True
        else:
            is_market_open = self._price_fetcher_client.is_market_open(trade_pair, now_ms)
        if not is_market_open:
            bt.logging.info(f"[HL_TRACKER] Market closed for {trade_pair_id} ({synthetic_hotkey})")
            self._broadcast_rejection(synthetic_hotkey, f"Market is closed for {trade_pair_id}.")
            return

        # === Step 1: Fetch HL account state -> compute target weight ===
        if account_state is None:
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

        # Clip to per-asset Vanta limits (signed)
        max_lev = trade_pair.max_leverage
        min_lev = trade_pair.min_leverage
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
            bt.logging.info(
                f"[HL_TRACKER] Skipping fill: {coin} delta={delta:+.4f} below min leverage {min_lev} "
                f"target={target_signed_weight:+.4f} current={current_signed_lev:+.4f} -> {synthetic_hotkey}"
            )
            return

        # Block position increases for coins below the liquidity threshold.
        if below_threshold and delta != 0:
            would_increase = (current_signed_lev >= 0 and delta > 0) or \
                             (current_signed_lev <= 0 and delta < 0)
            if would_increase:
                bt.logging.info(
                    f"[HL_TRACKER] Skipping fill: {coin} delta={delta:+.4f} blocked (below liquidity threshold) "
                    f"current={current_signed_lev:+.4f} -> {synthetic_hotkey}"
                )
                return
            # delta reduces or closes — allow through

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

        # === Step 6: Determine fill price from HL data ===
        # Taker (market order): simulate avg fill price by walking the local L2 orderbook
        #   (fine book first for near-spread precision, coarse book for deeper levels).
        #   Fallback to actual HL fill price if orderbook data is unavailable.
        # Maker (limit order): use the actual HL fill price directly.
        # hl_slippage is always 0.0 — the price passed already reflects true execution
        # quality, so applying slippage on top would double-count it (position.py multiplies
        # entry/exit price by (1 + slippage)).
        is_taker = fill.get("crossed", True)
        hl_fill_price = None

        raw_px = fill.get("px")
        raw_fill_price = None
        if raw_px is not None:
            try:
                raw_fill_price = float(raw_px)
            except (ValueError, TypeError):
                pass

        if is_taker:
            if order_type == "FLAT":
                translated_size_usd = abs(current_signed_lev) * account_size
                is_buying = current_signed_lev < 0  # closing SHORT = buying
            else:
                translated_size_usd = leverage * account_size
                is_buying = order_type == "LONG"

            hl_fill_price = self._price_fetcher_client.simulate_avg_fill_price(
                trade_pair, translated_size_usd, is_buying
            )
            # Fallback: use actual HL fill price if orderbook simulation produced no result
            if hl_fill_price is None:
                hl_fill_price = raw_fill_price
        else:
            # Maker (limit order): actual HL fill price is exact, no adjustment needed
            hl_fill_price = raw_fill_price

        # === Build signal ===
        signal = {
            "order_type": order_type,
            "leverage": leverage,
            "trade_pair": {"trade_pair_id": trade_pair_id},
            "execution_type": "MARKET",
            "is_hl": True,
            "is_hl_taker": is_taker,
            "hl_slippage": 0.0,  # price already reflects execution quality; applying slippage on top would double-count it
        }
        if hl_fill_price:
            signal["price"] = hl_fill_price

        miner_order_uuid = str(uuid.uuid4())

        bt.logging.info(
            f"[HL_TRACKER] Attempting order: {coin} {order_type} leverage={leverage:.4f} "
            f"target_weight={target_signed_weight:+.4f} current_lev={current_signed_lev:+.4f} "
            f"delta={delta:+.4f} fill_px={hl_fill_price} is_taker={is_taker} -> {synthetic_hotkey}"
        )

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
                f"{synthetic_hotkey} {order_type} leverage={leverage:.4f} "
                f"fill_px={hl_fill_price} is_taker={is_taker}"
            )

        except SignalException as e:
            bt.logging.warning(f"[HL_TRACKER] Signal rejected for {synthetic_hotkey}: {e}")
            self._broadcast_rejection(synthetic_hotkey, f"Order rejected: {e}")
        except Exception as e:
            bt.logging.error(f"[HL_TRACKER] Order processing error for {synthetic_hotkey}: {e}")
            self._broadcast_rejection(synthetic_hotkey, f"Order rejected: {e}")
            bt.logging.error(traceback.format_exc())
