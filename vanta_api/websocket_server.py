import argparse
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from multiprocessing import Manager
import os
from queue import Queue
from setproctitle import setproctitle
from threading import Lock, Thread
import time
import traceback
from typing import Dict, Any, Optional, Deque
import websockets
from websockets import CloseCode
from websockets.legacy.server import WebSocketServerProtocol

import bittensor as bt

from entity_management.entity_client import EntityClient
from entity_management.entity_utils import (
    create_subaccount_dashboard,
    is_synthetic_hotkey,
    parse_synthetic_hotkey,
)
from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account import MinerAccountClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.statistics.miner_statistics_client import MinerStatisticsClient
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import TradePair, ValiConfig, RPCConnectionMode
from vanta_api.api_key_refresh import APIKeyMixin

# Maximum number of websocket connections allowed per API key for tiers below
# subaccount access
MAX_N_WS_PER_API_KEY = 5

# How often to refresh subaccount dashboard data to subscribers
SUBACCOUNT_REFRESH_INTERVAL_S = 0.1
SUBACCOUNT_REFRESH_EXPIRATION_MS = 15 * 1000


@dataclass
class DashboardSubscription:
    positions_time_ms: int = 0
    limit_orders_time_ms: int = 0
    checkpoints_time_ms: int = 0
    daily_returns_time_ms: int = 0
    last_update_time_ms: int = 0
    lock: Lock = field(default_factory=Lock)

    @staticmethod
    def _get_subkey(branch: dict, subkeys: list[str], default=None):
        if not branch:
            return default
        for key in subkeys:
            branch = branch.get(key)
            if branch is None:
                return default
        return branch

    def update_times(self, dashboard) -> None:
        self.positions_time_ms = self._get_subkey(
            dashboard,
            ["positions", "positions_time_ms"],
            self.positions_time_ms,
        )
        self.limit_orders_time_ms = self._get_subkey(
            dashboard,
            ["limit_orders", "limit_orders_time_ms"],
            self.limit_orders_time_ms,
        )
        self.checkpoints_time_ms = self._get_subkey(
            dashboard,
            ["ledger", "checkpoints_time_ms"],
            self.checkpoints_time_ms,
        )
        self.daily_returns_time_ms = self._get_subkey(
            dashboard,
            ["statistics", "daily_returns_time_ms"],
            self.daily_returns_time_ms,
        )
        self.last_update_time_ms = TimeUtil.now_in_millis()

@dataclass
class WebSocketServerClient:
    client_id: int
    websocket: WebSocketServerProtocol
    api_key: str
    tier: int = 0
    sequence_number: int = 0
    subscribe_broadcasts: bool = False
    dashboard_subscriptions: dict[str, DashboardSubscription] = field(default_factory=dict)

    async def send(self, message:dict) -> None:
        serialized_message = json.dumps(
            {
                "sequence": self.sequence_number,
                "timestamp": TimeUtil.now_in_millis(),
                "data": message
            },
            cls=CustomEncoder,
            separators=(",", ":")
        )
        self.sequence_number += 1
        await self.websocket.send(serialized_message)


class WebSocketServer(APIKeyMixin, RPCServerBase):
    """
    WebSocket server with RPC interface for position broadcasting.

    Inherits from:
    - APIKeyMixin: Provides API key authentication and refresh
    - RPCServerBase: Provides RPC server lifecycle management

    The server runs a WebSocket server on the specified port (default 8765) and
    also exposes RPC methods on ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT (50014)
    for other processes to queue position updates for broadcasting.
    """

    service_name = ValiConfig.RPC_WEBSOCKET_NOTIFIER_SERVICE_NAME
    service_port = ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT

    def __init__(self,
                 api_keys_file: str,
                 reconnect_interval: int = 3,
                 max_reconnect_attempts: int = 10,
                 refresh_interval: int = 15,
                 send_test_positions: bool = False,
                 test_position_interval: int = 5,
                 start_server: bool = True,
                 running_unit_tests: bool = False,
                 connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
                 websocket_host: Optional[str] = None,
                 websocket_port: Optional[int] = None):
        """Initialize the WebSocket server.

        The server runs on configurable endpoints (defaults from ValiConfig):
        - WebSocket: websocket_host:websocket_port (default: ValiConfig.VANTA_WEBSOCKET_HOST:VANTA_WEBSOCKET_PORT)
        - RPC health: ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT (50014)

        Args:
            api_keys_file: Path to the API keys file
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum number of reconnection attempts (0=infinite)
            refresh_interval: How often to check for API key changes (seconds)
            send_test_positions: Whether to periodically send test orders (for testing only)
            test_position_interval: How often to send test orders (seconds)
            start_server: Whether to start the RPC server immediately
            running_unit_tests: Whether running in unit test mode
            connection_mode: RPC connection mode (RPC or LOCAL)
            websocket_host: Host address for WebSocket server (default: ValiConfig.VANTA_WEBSOCKET_HOST)
            websocket_port: Port for WebSocket server (default: ValiConfig.VANTA_WEBSOCKET_PORT)
        """
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)

        # Store for later use
        self.running_unit_tests = running_unit_tests

        # WebSocket server configuration - use provided host/port or fall back to ValiConfig defaults
        self.host = websocket_host if websocket_host is not None else ValiConfig.VANTA_WEBSOCKET_HOST
        self.port = websocket_port if websocket_port is not None else ValiConfig.VANTA_WEBSOCKET_PORT
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.server = None
        self.shutdown_event = None

        # IMPORTANT: Save WebSocket port to separate attribute BEFORE RPCServerBase.__init__
        # RPCServerBase.__init__ will overwrite self.port to the RPC port (50014),
        # but we need to preserve the WebSocket port (8765) for cleanup and binding
        self.websocket_port = self.port

        # Client tracking
        self._clients: dict[int, WebSocketServerClient] = {}

        # API key tracking - maintain a FIFO queue for each API key
        self._api_key_client_ids: Dict[str, Deque[int]] = defaultdict(deque)

        # Message queueing and processing
        self._broadcast_message_queue = asyncio.Queue()
        self._dashboard_update_queue = Queue()

        # Tasks and threads
        self._event_loop = None
        self._broadcast_message_queue_task = None
        self._thread_pool = ThreadPoolExecutor(max_workers=8)

        self.test_positions_task = None

        self._entity_client = EntityClient(connection_mode=connection_mode)
        self._position_client = PositionManagerClient(connection_mode=connection_mode)
        self._debt_ledger_client = DebtLedgerClient(connection_mode=connection_mode)
        self._limit_order_client = LimitOrderClient(connection_mode=connection_mode)
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)
        self._statistics_client = MinerStatisticsClient(connection_mode=connection_mode)
        self._challenge_period_client = ChallengePeriodClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests
        )
        self._elimination_client = EliminationClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Test order configuration
        self.send_test_positions = send_test_positions
        self.test_positions_interval = test_position_interval

        # Start API key refresh thread
        self.start_refresh_thread()

        bt.logging.info(f"WebSocketServer: Initialized with {len(self.accessible_api_keys)} API keys")
        if self.send_test_positions:
            bt.logging.info(f"WebSocketServer: Test orders will be sent every {self.test_positions_interval} seconds")

        # Initialize RPCServerBase (provides RPC server for other processes to queue messages)
        # This will set self.port = ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT (50014)
        # Note: _cleanup_stale_server() override will be called during this __init__
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_WEBSOCKET_NOTIFIER_SERVICE_NAME,
            port=ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT,
            connection_mode=connection_mode,
            start_server=start_server,
            start_daemon=False  # WebSocket server doesn't need a daemon loop
        )

        # Restore WebSocket port for convenience (some methods expect self.port = websocket port)
        self.port = self.websocket_port

        bt.logging.success(f"WebSocketServer: RPC server initialized on port {ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT}")

    def _cleanup_stale_server(self):
        """
        Override RPCServerBase._cleanup_stale_server() to clean up BOTH ports.

        WebSocketServer uniquely uses two ports:
        - RPC port (self.port during parent __init__): 50014
        - WebSocket port (self.websocket_port): 8765

        Parent's _cleanup_stale_server() only cleans the RPC port, so we override
        to clean both ports before binding.

        Note: This is called during RPCServerBase.__init__(), so we use self.websocket_port
        which was saved before calling parent's __init__.
        """
        # Clean up RPC port using parent's logic (cleans self.port = 50014)
        super()._cleanup_stale_server()

        # Now clean up WebSocket port using self.websocket_port (8765)
        from shared_objects.rpc.port_manager import PortManager
        if not PortManager.is_port_free(self.websocket_port):
            bt.logging.warning(f"WebSocketServer: WebSocket port {self.websocket_port} in use, forcing cleanup...")
            PortManager.force_kill_port(self.websocket_port)

            # Wait for OS to release the port after killing process
            if not PortManager.wait_for_port_release(self.websocket_port, timeout=2.0):
                bt.logging.warning(
                    f"WebSocketServer: WebSocket port {self.websocket_port} still not free after cleanup. "
                    f"Will attempt to bind anyway (reuse_port may work)"
                )

    async def _generate_test_ws_positions(self) -> None:
        """Periodically generate test positions and put them in the queue."""
        if not self.send_test_positions:
            return

        while True:
            try:
                # Create a test order
                current_time = TimeUtil.now_in_millis()
                # Queue it for processing
                await self._broadcast_message_queue.put(self._create_test_position(current_time))

                bt.logging.info(f"WebSocketServer: Generated test position")

                # Wait before generating the next order
                await asyncio.sleep(self.test_positions_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error generating test order: {e}")
                await asyncio.sleep(self.test_positions_interval)

    def _create_test_position(self, timestamp: int) -> Dict[str, Any]:
        """Create a test order with randomized parameters."""
        p = Position(**
        {'miner_hotkey': '5EWKUhycaBQHiHnfE3i2suZ1BvxAAE3HcsFsp8TaR6mu3JrJ',
         'position_uuid': 'bbc676e5-9ab3-4c11-8be4-5f2022ae9208', 'open_ms': timestamp,
         'trade_pair': TradePair.BTCUSD,
         'orders': [{'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 82872.75,
                     'bid': 82880.06, 'ask': 82880.07, 'slippage': 1e-05, 'processed_ms': timestamp,
                     'order_uuid': 'bbc676e5-9ab3-4c11-8be4-5f2022ae9208', 'price_sources': [
                         {'ask': 82880.07, 'bid': 82880.06, 'close': 82872.75, 'high': 82872.75, 'lag_ms': 1348,
                          'low': 82872.75,
                          'open': 82872.75, 'source': 'Tiingo_gdax_rest', 'start_ms': 1742357760395, 'timespan_ms': 0,
                          'vwap': 82872.75, 'websocket': True},
                         {'ask': 0.0, 'bid': 0.0, 'close': 82869.24, 'high': 82880.07, 'lag_ms': 2048, 'low': 82869.24,
                          'open': 82880.07, 'source': 'Polygon_rest', 'start_ms': 1742357756000, 'timespan_ms': 1000,
                          'vwap': 82875.8692, 'websocket': False},
                         {'ask': 0.0, 'bid': 0.0, 'close': 82047.16, 'high': 82047.16, 'lag_ms': 26043047, 'low': 82047.16,
                          'open': 82047.16, 'source': 'Polygon_ws', 'start_ms': 1742331716000, 'timespan_ms': 0,
                          'vwap': 82047.16,
                          'websocket': True}], 'src': 0},
                    {'trade_pair': TradePair.BTCUSD, 'order_type': OrderType.FLAT, 'leverage': 0.5, 'price': 83202.0,
                     'bid': 83202.0, 'ask': 83202.0, 'slippage': 1e-05, 'processed_ms': timestamp + 15000,
                     'order_uuid': 'd6e1e768-9024-4cc5-84d7-d66691d82061', 'price_sources': [], 'src': 0}],
         'current_return': 0.9980034808990859, 'close_ms': 1742362860000, 'net_leverage': 0.0,
         'return_at_close': 0.9980034808990859, 'average_entry_price': 82871.92127250001,
         'position_type': OrderType.FLAT,
         'is_closed_position': True})
        return p.to_websocket_dict()

    def _remove_client(self, client_id: int) -> None:
        client = self._clients.pop(client_id, None)
        if client is not None:
            # Remove from API key map if present
            client_ids = self._api_key_client_ids[client.api_key]
            if client_ids:
                if client_id in client_ids:
                    client_ids.remove(client_id)
                    api_key_alias = self.api_key_to_alias.get(client.api_key, "Unknown")
                    bt.logging.info(
                        f"WebSocketServer: Removed client {client_id} from API key {api_key_alias}")

            if client.websocket.state == websockets.protocol.OPEN:
                client.websocket.fail_connection(code=CloseCode.ABNORMAL_CLOSURE)

            bt.logging.info(f"WebSocketServer: Client {client_id} removed")

    async def _send_message(self, client, message) -> None:
        client_id = client.client_id

        if not self.can_access_tier(client.api_key, client.tier):
            bt.logging.warning(f"WebSocketServer: Client {client_id} tier changed")
            self._remove_client(client_id)
            return

        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            bt.logging.info(f"WebSocketServer: Client {client_id} disconnected while sending")
            self._remove_client(client_id)
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error sending to client {client_id}: {e}")
            self._remove_client(client_id)

    async def _process_broadcast_message_queue(self) -> None:
        while True:
            try:
                message = await self._broadcast_message_queue.get()

                send_tasks = []
                for client in list(self._clients.values()):
                    if client.subscribe_broadcasts:
                        send_tasks.append(asyncio.create_task(self._send_message(client, message)))

                if send_tasks:
                    await asyncio.gather(*send_tasks, return_exceptions=False)

                self._broadcast_message_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error processing message from queue: {e}")
                bt.logging.error(traceback.format_exc())

    def _queue_broadcast_message(self, message) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_message_queue.put(message),
                self._event_loop
            )

        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error queueing message: {e}")
            bt.logging.error(traceback.format_exc())

    def _send_dashboard_update(
        self,
        synthetic_hotkey: str,
        client: WebSocketServerClient,
        subscription: DashboardSubscription
    ) -> None:
        try:
            with subscription.lock:
                subaccount_dashboard = self._entity_client.get_subaccount_dashboard(synthetic_hotkey)
                if subaccount_dashboard is None:
                    bt.logging.warning(f"WebSocketServer: No dashboard found for synthetic hotkey {synthetic_hotkey}")
                else:
                    dashboard = create_subaccount_dashboard(
                        synthetic_hotkey,
                        subaccount_dashboard,
                        self._challenge_period_client,
                        self._elimination_client,
                        self._miner_account_client,
                        self._position_client,
                        self._limit_order_client,
                        self._debt_ledger_client,
                        self._statistics_client,
                        subscription.positions_time_ms,
                        subscription.limit_orders_time_ms,
                        subscription.checkpoints_time_ms,
                        subscription.daily_returns_time_ms,
                    )

                    subscription.update_times(dashboard)

                    asyncio.run_coroutine_threadsafe(
                        self._send_message(client, {"dashboard": dashboard}),
                        self._event_loop
                    )

        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error processing dashboard update: {e}")
            bt.logging.error(traceback.format_exc())

    def _process_dashboard_update_queue(self) -> None:
        while True:
            try:
                synthetic_hotkey = self._dashboard_update_queue.get()
                for client in list(self._clients.values()):
                    subscription = client.dashboard_subscriptions.get(synthetic_hotkey)
                    if subscription is not None:
                        self._thread_pool.submit(self._send_dashboard_update, synthetic_hotkey, client, subscription)
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error processing dashboard update: {e}")
                bt.logging.error(traceback.format_exc())

    def _process_dashboard_refresh(self) -> None:
        while True:
            try:
                time.sleep(SUBACCOUNT_REFRESH_INTERVAL_S)
                expiration_ms = TimeUtil.now_in_millis() - SUBACCOUNT_REFRESH_EXPIRATION_MS
                for client in list(self._clients.values()):
                    subscriptions = dict(client.dashboard_subscriptions)
                    for synthetic_hotkey, subscription in subscriptions.items():
                        if subscription.last_update_time_ms < expiration_ms:
                            self._send_dashboard_update(synthetic_hotkey, client, subscription)

            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error processing dashboard refresh: {e}")
                bt.logging.error(traceback.format_exc())

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work.

        Note: WebSocketServer doesn't need a daemon loop - all work is done
        asynchronously in the WebSocket event loop. This is a no-op.
        """
        pass

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "connected_clients": len(self._clients),
            "queue_size": self._broadcast_message_queue.qsize(),
            "dashboard_update_queue_size": self._dashboard_update_queue.qsize(),
        }

    def broadcast_position_update_rpc(self, position: Position, miner_repo_version: str = None) -> None:
        """
        RPC method to broadcast a position update to all subscribed WebSocket clients.

        This method is called via RPC from other processes (MarketOrderManager,
        PositionManager, EliminationServer) to notify WebSocket clients of position changes.

        Args:
            position: Position object to broadcast
            miner_repo_version: Optional miner repository version for the websocket dict
        """
        try:
            position_dict = position.to_websocket_dict(miner_repo_version=miner_repo_version)
            self._queue_broadcast_message(position_dict)

            if is_synthetic_hotkey(position.miner_hotkey):
                self.broadcast_subaccount_dashboard_rpc(position.miner_hotkey)

        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error broadcasting position update: {e}")
            bt.logging.error(traceback.format_exc())

    def broadcast_subaccount_dashboard_rpc(self, synthetic_hotkey: str) -> None:
        """
        RPC method to broadcast subaccount dashboard to subscribed clients.

        Args:
            synthetic_hotkey: The synthetic hotkey to broadcast dashboard for
        """
        update_queue = self._dashboard_update_queue
        # Attempt to avoid some duplicates when possible
        if synthetic_hotkey not in update_queue.queue:
            update_queue.put(synthetic_hotkey)

    async def _handle_client(self, websocket) -> None:
        """Handle client connection with authentication and subscriptions.

        API key is read from the Authorization: Bearer header set during the
        WebSocket upgrade handshake. No auth message exchange is required.

        Args:
            websocket: WebSocket connection
        """
        client_id = id(websocket)
        bt.logging.info(f"WebSocketServer: New client connected (ID: {client_id})")

        try:
            # Read API key from Authorization: Bearer header (websockets 15+: websocket.request.headers)
            auth_header = websocket.request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Authentication required. Provide API key in Authorization: Bearer header."
                }))
                return

            api_key = auth_header[7:]

            # Validate API key
            try:
                is_valid = self.is_valid_api_key(api_key)
            except Exception as e:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"Error validating API key: {e}"
                }))
                return

            if not is_valid:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid API key. Authentication failed."
                }))
                return

            api_key_tier = self.get_api_key_tier(api_key)

            if api_key_tier < 100:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "WebSocket connections require tier 100 access. Please upgrade your API key.",
                    "code": "INSUFFICIENT_TIER"
                }))
                return

            if api_key_tier < ValiConfig.SUBACCOUNT_SUBSCRIPTION_TIER:
                # Enforce per-key connection limit for non-entity clients
                if len(self._api_key_client_ids[api_key]) >= MAX_N_WS_PER_API_KEY:
                    oldest_client_id = self._api_key_client_ids[api_key].popleft()
                    oldest_client = self._clients.get(oldest_client_id)
                    if oldest_client is not None:
                        try:
                            await oldest_client.websocket.send(json.dumps({
                                "status": "error",
                                "message": "Disconnected due to too many connections for this API key",
                                "code": "MAX_CONNECTIONS_EXCEEDED"
                            }))
                            await oldest_client.websocket.close()
                        except Exception as e:
                            bt.logging.error(f"WebSocketServer: Error closing oldest connection {oldest_client_id}: {e}")
                    self._remove_client(oldest_client_id)
                    api_key_alias = self.api_key_to_alias.get(api_key, "Unknown")
                    bt.logging.info(f"WebSocketServer: Dropped oldest client {oldest_client_id} for API key "
                          f"{api_key_alias} to make room for new client {client_id}")

            # Register client
            client = WebSocketServerClient(client_id=client_id, websocket=websocket, api_key=api_key, tier=api_key_tier)
            self._clients[client_id] = client
            self._api_key_client_ids[api_key].append(client_id)

            bt.logging.info(f"WebSocketServer: Client {client_id} authenticated successfully with tier {api_key_tier}")

            # For entity clients (tier >= 200), auto-subscribe to all active subaccounts
            hl_mappings = {}
            subscribed_subaccounts = 0

            if api_key_tier >= ValiConfig.SUBACCOUNT_SUBSCRIPTION_TIER:
                entity_hotkey = self.api_key_to_alias.get(api_key)
                if entity_hotkey:
                    try:
                        loop = asyncio.get_running_loop()
                        entity_data = await loop.run_in_executor(
                            self._thread_pool,
                            self._entity_client.get_entity_data,
                            entity_hotkey
                        )
                        if entity_data:
                            subaccounts = entity_data.get('subaccounts', {})
                            for sub_id, sub_info in subaccounts.items():
                                if sub_info.get('status') not in ('active', 'admin'):
                                    continue
                                synthetic_hotkey = sub_info.get('synthetic_hotkey') or f"{entity_hotkey}_{sub_id}"
                                client.dashboard_subscriptions[synthetic_hotkey] = DashboardSubscription()
                                subscribed_subaccounts += 1
                                hl_addr = sub_info.get('hl_address')
                                if hl_addr:
                                    hl_mappings[hl_addr] = synthetic_hotkey
                            if subscribed_subaccounts > 0:
                                bt.logging.info(
                                    f"WebSocketServer: Auto-subscribed client {client_id} to "
                                    f"{subscribed_subaccounts} subaccounts for entity {entity_hotkey}"
                                )
                    except Exception as e:
                        bt.logging.error(f"WebSocketServer: Error auto-subscribing entity {entity_hotkey}: {e}")
                        bt.logging.error(traceback.format_exc())

            await websocket.send(json.dumps({
                "status": "success",
                "message": "Authentication successful.",
                "current_sequence": 0,
                "tier": api_key_tier,
                "subscribed_subaccounts": subscribed_subaccounts,
                "hl_mappings": hl_mappings
            }))

            # Process client messages (subscriptions, etc.)
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    message_type = data.get("type", "")

                    if message_type == "ping":
                        client_timestamp = data.get("timestamp", 0)
                        server_timestamp = TimeUtil.now_in_millis()
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "client_timestamp": client_timestamp,
                            "server_timestamp": server_timestamp,
                            "timestamp": server_timestamp
                        }))

                    elif message_type == "subscribe":
                        # Handle subscription to all data
                        if data.get("all", False):
                            client.subscribe_broadcasts = True
                            bt.logging.info(f"WebSocketServer: Client {client_id} subscribed to all data")
                            await websocket.send(json.dumps({
                                "type": "subscription_status",
                                "status": "success",
                                "all": True,
                                "action": "subscribe",
                                "sender_timestamp": data.get("sender_timestamp", 0),
                                "received_timestamp": TimeUtil.now_in_millis()
                            }))

                    elif message_type == "unsubscribe":
                        # Handle unsubscription
                        if data.get("all", False):
                            client.subscribe_broadcasts = False
                            bt.logging.info(f"WebSocketServer: Client {client_id} unsubscribed from all data")
                            await websocket.send(json.dumps({
                                "type": "subscription_status",
                                "status": "success",
                                "all": True,
                                "action": "unsubscribe"
                            }))

                    elif message_type == "subscribe_subaccount":
                        synthetic_hotkey = data.get("synthetic_hotkey")
                        positions_time_ms = data.get("positions_time_ms", 0)
                        limit_orders_time_ms = data.get("limit_orders_time_ms", 0)
                        checkpoints_time_ms = data.get("checkpoints_time_ms", 0)
                        daily_returns_time_ms = data.get("daily_returns_time_ms", 0)

                        if not synthetic_hotkey:
                            await websocket.send(json.dumps({
                                "type": "subscription_status",
                                "status": "error",
                                "action": "subscribe_subaccount",
                                "message": "Missing synthetic_hotkey parameter"
                            }))
                        else:
                            # Ownership check: entity clients may only subscribe to their own subaccounts
                            api_key_alias = self.api_key_to_alias.get(client.api_key)
                            try:
                                parsed_entity, _ = parse_synthetic_hotkey(synthetic_hotkey)
                            except Exception:
                                parsed_entity = None

                            if api_key_alias and parsed_entity and api_key_alias == parsed_entity:
                                # Entity subscribing to own subaccount — no tier check needed
                                client.dashboard_subscriptions[synthetic_hotkey] = DashboardSubscription(
                                    positions_time_ms=positions_time_ms,
                                    limit_orders_time_ms=limit_orders_time_ms,
                                    checkpoints_time_ms=checkpoints_time_ms,
                                    daily_returns_time_ms=daily_returns_time_ms,
                                )
                                bt.logging.info(f"WebSocketServer: Client {client_id} subscribed to subaccount {synthetic_hotkey}")
                                await websocket.send(json.dumps({
                                    "type": "subscription_status",
                                    "status": "success",
                                    "action": "subscribe_subaccount",
                                    "subscribed_to": synthetic_hotkey
                                }))
                            elif client.tier < ValiConfig.SUBACCOUNT_SUBSCRIPTION_TIER:
                                await websocket.send(json.dumps({
                                    "type": "subscription_status",
                                    "status": "error",
                                    "action": "subscribe_subaccount",
                                    "synthetic_hotkey": synthetic_hotkey,
                                    "message": "Subaccount subscriptions require tier 200 access.",
                                    "code": "INSUFFICIENT_TIER"
                                }))
                            else:
                                client.dashboard_subscriptions[synthetic_hotkey] = DashboardSubscription(
                                    positions_time_ms=positions_time_ms,
                                    limit_orders_time_ms=limit_orders_time_ms,
                                    checkpoints_time_ms=checkpoints_time_ms,
                                    daily_returns_time_ms=daily_returns_time_ms,
                                )
                                bt.logging.info(f"WebSocketServer: Client {client_id} subscribed to subaccount {synthetic_hotkey}")
                                await websocket.send(json.dumps({
                                    "type": "subscription_status",
                                    "status": "success",
                                    "action": "subscribe_subaccount",
                                    "subscribed_to": synthetic_hotkey
                                }))

                    elif message_type == "unsubscribe_subaccount":
                        synthetic_hotkey = data.get("synthetic_hotkey")
                        client.dashboard_subscriptions.pop(synthetic_hotkey, None)
                        bt.logging.info(f"WebSocketServer: Client {client_id} unsubscribed from subaccount {synthetic_hotkey}")
                        await websocket.send(json.dumps({
                            "type": "subscription_status",
                            "status": "success",
                            "action": "unsubscribe_subaccount",
                            "synthetic_hotkey": synthetic_hotkey
                        }))

                except websockets.exceptions.ConnectionClosed:
                    break
                except json.JSONDecodeError:
                    bt.logging.warning(f"WebSocketServer: Received invalid JSON from client {client_id}")
                except Exception as e:
                    bt.logging.error(f"WebSocketServer: Error processing message from client {client_id}: {e}")
                    bt.logging.error(traceback.format_exc())

        except websockets.exceptions.ConnectionClosed:
            bt.logging.info(f"WebSocketServer: Client {client_id} disconnected")
        except json.JSONDecodeError:
            bt.logging.warning(f"WebSocketServer: Received invalid JSON data from client {client_id}")
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error handling client {client_id}: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self._remove_client(client_id)

    def notify_new_subaccount_rpc(self, entity_hotkey: str, synthetic_hotkey: str) -> bool:
        """
        RPC method to auto-subscribe connected entity clients to a newly created subaccount.

        Called by EntityManager when a new subaccount is created while the entity is connected.

        Args:
            entity_hotkey: The entity hotkey that owns the new subaccount
            synthetic_hotkey: The synthetic hotkey of the new subaccount

        Returns:
            bool: True if notification was sent (or no clients to notify)
        """
        try:
            # Reverse-lookup entity_hotkey → api_key
            api_key = next(
                (k for k, v in self.api_key_to_alias.items() if v == entity_hotkey),
                None
            )
            if api_key is None:
                return True  # Entity has no API key yet

            client_ids = list(self._api_key_client_ids.get(api_key, []))
            if not client_ids:
                return True  # No connected clients

            for client_id in client_ids:
                client = self._clients.get(client_id)
                if client is not None:
                    client.dashboard_subscriptions[synthetic_hotkey] = DashboardSubscription()

            self.broadcast_subaccount_dashboard_rpc(synthetic_hotkey)

            message = {
                "type": "subscription_status",
                "action": "new_subaccount_subscribed",
                "synthetic_hotkey": synthetic_hotkey,
                "entity_hotkey": entity_hotkey
            }
            if self._event_loop is not None:
                for client_id in client_ids:
                    client = self._clients.get(client_id)
                    if client is not None:
                        asyncio.run_coroutine_threadsafe(
                            self._send_message(client, message),
                            self._event_loop
                        )

            bt.logging.info(
                f"WebSocketServer: Auto-subscribed {len(client_ids)} entity client(s) to new subaccount {synthetic_hotkey}")
            return True
        except Exception as e:
            bt.logging.error(f"WebSocketServer: Error notifying new subaccount {synthetic_hotkey}: {e}")
            return False

    async def start(self) -> None:
        # Store the event loop reference for thread-safe operations
        self._event_loop = asyncio.get_running_loop()

        # Suppress EOFError logs from websockets library (caused by health monitor TCP checks)
        # These errors occur when the health monitor creates a raw TCP connection to check if the port is open,
        # but doesn't complete the WebSocket handshake. This is expected behavior and not an actual error.

        # Add a filter to suppress the specific EOF errors during handshake
        class HandshakeEOFFilter(logging.Filter):
            def filter(self, record):
                # Get the full message including exception info
                msg = record.getMessage()
                if record.exc_info:
                    import io
                    sio = io.StringIO()
                    traceback.print_exception(*record.exc_info, file=sio)
                    msg += '\n' + sio.getvalue()

                # Suppress "connection closed while reading HTTP request line" errors
                if 'connection closed while reading HTTP request line' in msg:
                    return False
                # Suppress "opening handshake failed" errors
                if 'opening handshake failed' in msg:
                    return False
                # Suppress "stream ends after 0 bytes" errors (health check connections)
                if 'stream ends after 0 bytes, before end of line' in msg:
                    return False
                # Suppress EOFError from health checks
                if 'EOFError' in msg and 'handshake' in msg.lower():
                    return False
                return True

        # Apply filter to websockets loggers
        for logger_name in ['websockets.server', 'websockets', 'websockets.protocol']:
            logger = logging.getLogger(logger_name)
            logger.addFilter(HandshakeEOFFilter())

        self._broadcast_message_queue_task = asyncio.create_task(self._process_broadcast_message_queue())

        # Start test order generator if enabled
        if self.send_test_positions:
            bt.logging.info(f"WebSocketServer: Starting test order generator")
            self.test_positions_task = asyncio.create_task(self._generate_test_ws_positions())

        attempts = 0
        while attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0:
            try:
                # Create the server with appropriate handler
                bt.logging.info(f"WebSocketServer: Attempting to bind WebSocket server to {self.host}:{self.port} (attempt {attempts + 1})...")

                try:
                    self.server = await websockets.serve(
                        self._handle_client,
                        self.host,
                        self.port,
                        compression="deflate",
                        reuse_address=True,  # Allow reuse of the address
                        reuse_port=True  # Allow reuse of the port (on platforms that support it)
                    )
                    bt.logging.info(f"WebSocketServer: websockets.serve() completed successfully")
                except Exception as serve_error:
                    bt.logging.error(f"WebSocketServer: ERROR: websockets.serve() raised exception: {type(serve_error).__name__}: {serve_error}")
                    raise

                bt.logging.info(f"WebSocketServer: WebSocket server started at ws://{self.host}:{self.port}")

                # Keep the server running indefinitely
                bt.logging.info(f"WebSocketServer: Entering main event loop (await asyncio.Future())...")
                await asyncio.Future()

            except OSError as e:
                attempts += 1
                bt.logging.error(
                    f"WebSocketServer: Failed to start WebSocket server (attempt {attempts}/{self.max_reconnect_attempts}): OSError {e.errno}: {e}")
                bt.logging.error(f"WebSocketServer: Error details - errno: {e.errno}, strerror: {e.strerror}")

                if attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0:
                    bt.logging.info(f"WebSocketServer: Retrying in {self.reconnect_interval} seconds...")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    bt.logging.error(f"WebSocketServer: Maximum retry attempts reached. Giving up.")
                    raise
            except asyncio.CancelledError:
                if self._broadcast_message_queue_task:
                    self._broadcast_message_queue_task.cancel()
                    try:
                        await self._broadcast_message_queue_task
                    except asyncio.CancelledError:
                        pass

                if self.test_positions_task:
                    self.test_positions_task.cancel()
                    try:
                        await self.test_positions_task
                    except asyncio.CancelledError:
                        pass
                raise

            except Exception as e:
                bt.logging.error(f"WebSocketServer: Unexpected error starting WebSocket server: {type(e).__name__}: {e}")
                bt.logging.error(f"WebSocketServer: Full traceback:")
                bt.logging.error(f"WebSocketServer: {traceback.format_exc()}")
                raise

    async def shutdown(self) -> None:
        """Gracefully shut down the WebSocket server."""
        bt.logging.info(f"WebSocketServer: Shutting down WebSocket server...")

        # Signal the shutdown to all tasks
        if self.shutdown_event:
            self.shutdown_event.set()

        self._thread_pool.shutdown(wait=False, cancel_futures=True)

        # Close the server
        if self.server:
            await self.server.close()
            try:
                await self.server.wait_closed()
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error while waiting for server to close: {e}")

        connected_clients = list(self._clients.values())
        self._clients = {}

        # Close all client connections
        for client in connected_clients:
            try:
                await client.websocket.close()
            except Exception as e:
                bt.logging.error(f"WebSocketServer: Error closing client {client.client_id}: {e}")

        # Wait a bit for connections to close
        await asyncio.sleep(0.5)

        # Cancel all tasks with proper handling
        tasks_to_cancel = []

        if self._broadcast_message_queue_task and not self._broadcast_message_queue_task.done():
            self._broadcast_message_queue_task.cancel()
            tasks_to_cancel.append(self._broadcast_message_queue_task)

        if self.test_positions_task and not self.test_positions_task.done():
            self.test_positions_task.cancel()
            tasks_to_cancel.append(self.test_positions_task)

        # Wait for all tasks to complete cancellation with exception handling
        if tasks_to_cancel:
            bt.logging.info(f"WebSocketServer: Waiting for {len(tasks_to_cancel)} tasks to cancel...")
            for task in tasks_to_cancel:
                try:
                    # Use wait_for with a timeout to avoid hanging
                    await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    # This is expected
                    pass
                except Exception as e:
                    bt.logging.error(f"WebSocketServer: Error cancelling task: {e}")

        bt.logging.info(f"WebSocketServer: WebSocket server shutdown complete")

    @classmethod
    def entry_point_start_server(cls, **kwargs):
        """
        Entry point for WebSocket server process.

        Overrides RPCServerBase.entry_point_start_server() because WebSocketServer
        needs to run an async event loop via run(), not just block.
        """

        assert cls.service_name, f"{cls.__name__} must set service_name class attribute"
        assert cls.service_port, f"{cls.__name__} must set service_port class attribute"

        # Set process title
        setproctitle(f"vali_{cls.service_name}")

        # Extract ServerProcessHandle-specific parameters
        server_ready = kwargs.pop('server_ready', None)
        kwargs.pop('health_check_interval_s', None)
        kwargs.pop('enable_auto_restart', None)

        # Add required parameters
        kwargs['start_server'] = True
        kwargs['connection_mode'] = RPCConnectionMode.RPC

        # Filter kwargs to only include valid parameters
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        # Log filtered parameters
        filtered_out = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out:
            bt.logging.debug(f"[{cls.service_name}] Filtered out parameters: {filtered_out}")

        # Create server instance (starts RPC server)
        bt.logging.info(f"[{cls.service_name}] Creating server instance...")
        server_instance = cls(**filtered_kwargs)

        bt.logging.success(f"[{cls.service_name}] RPC server ready on port {cls.service_port}")

        # Signal ready BEFORE starting async loop (so clients can connect to RPC)
        if server_ready:
            server_ready.set()
            bt.logging.info(f"[{cls.service_name}] Server ready event signaled")

        # Now start the WebSocket async event loop (this blocks)
        bt.logging.info(f"[{cls.service_name}] Starting WebSocket async event loop...")
        try:
            server_instance.run()
        except Exception as e:
            bt.logging.error(f"[{cls.service_name}] WebSocket loop error: {e}")
            bt.logging.error(traceback.format_exc())
            raise

        bt.logging.info(f"[{cls.service_name}] process exiting")

    def run(self):
        """Start the server in the current process."""
        bt.logging.info(f"WebSocketServer: Starting WebSocket server...")
        setproctitle(f"vali_{self.__class__.__name__}")
        try:
            bt.logging.info(f"WebSocketServer: Creating new event loop...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            self.shutdown_event = asyncio.Event()

            bt.logging.info(f"WebSocketServer: Creating main task for start()...")
            main_task = loop.create_task(self.start())

            bt.logging.info(f"WebSocketServer: Creating thread for dashboard updates...")
            dashboard_update_thread = Thread(target=self._process_dashboard_update_queue, daemon=True)
            dashboard_update_thread.start()

            bt.logging.info(f"WebSocketServer: Creating thread for dashboard refreshes...")
            dashboard_refresh_thread = Thread(target=self._process_dashboard_refresh, daemon=True)
            dashboard_refresh_thread.start()

            # Run the loop until keyboard interrupt
            bt.logging.info(f"WebSocketServer: Running event loop with run_until_complete()...")
            try:
                loop.run_until_complete(main_task)
                bt.logging.info(f"WebSocketServer: Event loop completed (this shouldn't happen unless shutting down)")
            except KeyboardInterrupt:
                bt.logging.info(f"WebSocketServer: Keyboard interrupt detected! Shutting down...")
                # Set shutdown event - this will signal all tasks to stop
                self.shutdown_event.set()

                # Create and run shutdown task
                shutdown_task = loop.create_task(self.shutdown())
                loop.run_until_complete(shutdown_task)

                # Cancel any remaining tasks
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        task.cancel()

                # Run until all tasks complete cancellation
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                bt.logging.info(f"WebSocketServer: All tasks have been stopped.")
            finally:
                # Close the loop
                loop.close()

        except Exception as e:
            bt.logging.error(f"WebSocketServer: FATAL ERROR in WebSocket server run(): {type(e).__name__}: {e}")
            bt.logging.error(f"WebSocketServer: Full traceback:")
            bt.logging.error(f"{traceback.format_exc()}")
            bt.logging.error(f"WebSocketServer: WebSocket server process exiting due to error")
            raise


# For standalone testing
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WebSocket Server for PTN Data API')
    parser.add_argument('--api-keys-file', type=str, help='Path to the API keys file', default="api_keys.json")
    parser.add_argument('--test-positions', action='store_true', help='Enable periodic test positions', default=True)
    parser.add_argument('--test-position-interval', type=int, help='Interval in seconds between test positions', default=5)
    parser.set_defaults(test_positions=True)

    args = parser.parse_args()

    # Create a test API keys file if it doesn't exist
    if not os.path.exists(args.api_keys_file):
        with open(args.api_keys_file, "w") as f:
            json.dump({"test_user": "test_key", "client": "abc"}, f)
        bt.logging.info(f"WebSocketServer: Created test API keys file at {args.api_keys_file}")

    # Create a manager instance for testing
    mp_manager = Manager()
    test_queue = mp_manager.Queue()

    bt.logging.info(f"WebSocketServer: Starting WebSocket server on {ValiConfig.VANTA_WEBSOCKET_HOST}:{ValiConfig.VANTA_WEBSOCKET_PORT} (hardcoded in ValiConfig)")
    bt.logging.info(f"WebSocketServer: Test positions: {'Enabled' if args.test_positions else 'Disabled'}")
    if args.test_positions:
        bt.logging.info(f"WebSocketServer: Test position interval: {args.test_position_interval} seconds")

    # Create and run the server (host/port read from ValiConfig)
    server = WebSocketServer(
        api_keys_file=args.api_keys_file,
        send_test_positions=args.test_positions,
        test_position_interval=args.test_position_interval
    )

    # Run the server
    server.run()
