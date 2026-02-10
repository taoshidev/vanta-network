# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
LimitOrderServer - RPC server for limit order management.

This server runs in its own process and exposes limit order management via RPC.
Clients connect using LimitOrderClient.

"""

from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


# ==================== Server Implementation ====================

class LimitOrderServer(RPCServerBase):
    """
    RPC server for limit order management.

    Inherits from:
    - RPCServerBase: Provides RPC server lifecycle, daemon management

    All public methods ending in _rpc are exposed via RPC to LimitOrderClient.

    PROCESS BOUNDARY: Runs in SEPARATE process from validator.

    Architecture:
    - Internal data: {TradePair: {hotkey: [Order]}} - regular Python dicts (NO IPC)
    - RPC methods: Called from LimitOrderClient (validator process)
    - Daemon: Background thread checks/fills orders every 15 seconds
    - File persistence: Orders saved to disk for crash recovery

    Responsibilities:
    - Store and manage limit order lifecycle
    - Check order trigger conditions against live prices
    - Fill orders when limit price is reached
    - Persist orders to disk

    NOT responsible for:
    - Protocol/synapse handling (validator's job)
    - UUID tracking (validator's job - separate process)
    - Understanding miner signals (validator's job)
    """
    service_name = ValiConfig.RPC_LIMITORDERMANAGER_SERVICE_NAME
    service_port = ValiConfig.RPC_LIMITORDERMANAGER_PORT

    def __init__(
        self,
        running_unit_tests=False,
        slack_notifier=None,
        start_server=True,
        start_daemon=True,
        serve=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize LimitOrderServer.

        Server creates its own clients internally (forward compatibility - no parameter passing):
        - CommonDataClient (for shutdown_dict)
        - LivePriceFetcherClient
        - PositionManagerClient
        - EliminationClient
        - MarketOrderManager (for filling orders)

        Args:
            running_unit_tests: Whether running in test mode
            slack_notifier: Optional SlackNotifier for health check alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            serve: Whether MarketOrderManager should start its own RPC servers (True in production, False in tests)
        """
        self.running_unit_tests = running_unit_tests
        self._common_data_client = CommonDataClient(connect_immediately=False)

        # Create the manager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        from vali_objects.utils.limit_order.limit_order_manager import LimitOrderManager
        self._manager = LimitOrderManager(
            running_unit_tests=running_unit_tests,
            serve=serve,
            connection_mode=connection_mode
        )

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_LIMITORDERMANAGER_SERVICE_NAME,
            port=ValiConfig.RPC_LIMITORDERMANAGER_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=ValiConfig.LIMIT_ORDER_CHECK_REFRESH_MS / 1000.0,  # 10 seconds
            hang_timeout_s=120.0
        )

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.
        Checks and fills limit orders.
        """
        self._manager.check_and_fill_limit_orders()

    # ==================== Properties ====================

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return self._manager.health_check_rpc()

    def process_limit_order_rpc(self, miner_hotkey, order, is_edit=False):
        """
        RPC method to process a limit order or bracket order.
        Args:
            miner_hotkey: The miner's hotkey
            order: Order object (pickled automatically by RPC)
            is_edit: If True, this is an edit operation (replaces existing order)
        Returns:
            dict with status and order_uuid
        """
        return self._manager.process_limit_order(miner_hotkey, order, is_edit)

    def cancel_limit_order_rpc(self, miner_hotkey, trade_pair_id, order_uuid, now_ms):
        """
        RPC method to cancel limit order(s).
        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: Trade pair ID string
            order_uuid: UUID of specific order to cancel, or None/empty for all
            now_ms: Current timestamp
        Returns:
            dict with cancellation details
        """
        return self._manager.cancel_limit_order(miner_hotkey, trade_pair_id, order_uuid, now_ms)

    def get_limit_order_by_uuid_rpc(self, miner_hotkey, order_uuid):
        """
        RPC method to get an unfilled limit order by UUID.
        Args:
            miner_hotkey: The miner's hotkey
            order_uuid: UUID of the order to find
        Returns:
            Order dict if found, None if not found
        """
        return self._manager.get_limit_order_by_uuid(miner_hotkey, order_uuid)

    def get_limit_orders_for_hotkey_rpc(self, miner_hotkey):
        """
        RPC method to get all limit orders for a hotkey.
        Returns:
            List of order dicts
        """
        return self._manager.get_limit_orders_for_hotkey_rpc(miner_hotkey)

    def get_limit_orders_for_trade_pair_rpc(self, trade_pair_id):
        """
        RPC method to get all limit orders for a trade pair.
        Returns:
            Dict of {hotkey: [order_dicts]}
        """
        return self._manager.get_limit_orders_for_trade_pair_rpc(trade_pair_id)

    def to_dashboard_dict_rpc(self, miner_hotkey, status_filter=None):
        """
        RPC method to get dashboard representation of limit orders.

        Args:
            miner_hotkey: The miner's hotkey
            status_filter: Optional list of status strings ['unfilled', 'filled', 'cancelled']

        Returns:
            If status_filter is None: list of order dicts (backward compatible)
            If status_filter provided: dict of {status: [order dicts]}
        """
        return self._manager.to_dashboard_dict_rpc(miner_hotkey, status_filter)

    def get_all_limit_orders_rpc(self):
        """
        RPC method to get all limit orders across all trade pairs and hotkeys.

        Returns:
            Dict of {trade_pair_id: {hotkey: [order_dicts]}}
        """
        return self._manager.get_all_limit_orders_rpc()

    def delete_all_limit_orders_for_hotkey_rpc(self, miner_hotkey):
        """
        RPC method to delete all limit orders (both in-memory and on-disk) for a hotkey.

        This is called when a miner is eliminated to clean up their limit order data.

        Args:
            miner_hotkey: The miner's hotkey

        Returns:
            dict with deletion details
        """
        return self._manager.delete_all_limit_orders_for_hotkey_rpc(miner_hotkey)

    def sync_limit_orders_rpc(self, sync_data):
        """
        RPC method to sync limit orders from external source.
        """
        return self._manager.sync_limit_orders(sync_data)

    def clear_limit_orders_rpc(self):
        """
        RPC method to clear all limit orders from memory.

        This is primarily used for testing and development.
        Does NOT delete orders from disk.
        """
        if not self.running_unit_tests:
            raise Exception('clear_limit_orders_rpc can only be called in unit test mode')
        return self._manager.clear_limit_orders()

    def check_and_fill_limit_orders_rpc(self, call_id=None):
        """
        RPC method to manually trigger limit order check and fill (daemon method).

        This is primarily used for testing to trigger fills without waiting for daemon.

        Args:
            call_id: Optional unique identifier to prevent RPC caching. Pass a unique value
                    (like timestamp) in tests to ensure each call executes.

        Returns:
            dict: Execution stats with {'checked': int, 'filled': int, 'timestamp_ms': int}
        """
        if not self.running_unit_tests:
            raise Exception('check_and_fill_limit_orders_rpc can only be called in unit test mode')
        return self._manager.check_and_fill_limit_orders(call_id)

    def get_limit_orders_dict_rpc(self):
        """
        RPC method to get internal _limit_orders dict for test verification.

        Returns: Dict[TradePair, Dict[str, List[Order]]] serialized to dicts
        """
        if not self.running_unit_tests:
            raise Exception('get_limit_orders_dict_rpc can only be called in unit test mode')

        result = {}
        for trade_pair, hotkey_dict in self._manager._limit_orders.items():
            result[trade_pair.trade_pair_id] = {}
            for hotkey, orders in hotkey_dict.items():
                result[trade_pair.trade_pair_id][hotkey] = [order.to_python_dict() for order in orders]
        return result

    def set_limit_orders_dict_rpc(self, orders_dict):
        """
        RPC method to set internal _limit_orders dict for testing.

        Args:
            orders_dict: Dict[str, Dict[str, List[dict]]] - trade_pair_id -> hotkey -> [order_dicts]
        """
        if not self.running_unit_tests:
            raise Exception('set_limit_orders_dict_rpc can only be called in unit test mode')

        from vali_objects.vali_config import TradePair
        from vali_objects.vali_dataclasses.order import Order

        self._manager._limit_orders.clear()
        for trade_pair_id, hotkey_dict in orders_dict.items():
            trade_pair = TradePair.from_trade_pair_id(trade_pair_id)
            self._manager._limit_orders[trade_pair] = {}
            for hotkey, order_dicts in hotkey_dict.items():
                self._manager._limit_orders[trade_pair][hotkey] = [
                    Order.from_dict(order_dict) for order_dict in order_dicts
                ]

    def get_last_fill_time_rpc(self):
        """
        RPC method to get internal _last_fill_time dict for test verification.

        Returns: Dict[TradePair, Dict[str, int]] serialized with trade_pair_id keys
        """
        if not self.running_unit_tests:
            raise Exception('get_last_fill_time_rpc can only be called in unit test mode')

        result = {}
        for trade_pair, hotkey_dict in self._manager._last_fill_time.items():
            result[trade_pair.trade_pair_id] = dict(hotkey_dict)
        return result

    def set_last_fill_time_rpc(self, trade_pair_id, hotkey, fill_time):
        """
        RPC method to set _last_fill_time for testing.

        Args:
            trade_pair_id: Trade pair ID string
            hotkey: Miner hotkey
            fill_time: Timestamp in milliseconds
        """
        if not self.running_unit_tests:
            raise Exception('set_last_fill_time_rpc can only be called in unit test mode')

        from vali_objects.vali_config import TradePair
        trade_pair = TradePair.from_trade_pair_id(trade_pair_id)

        if trade_pair not in self._manager._last_fill_time:
            self._manager._last_fill_time[trade_pair] = {}
        self._manager._last_fill_time[trade_pair][hotkey] = fill_time

    def evaluate_limit_trigger_price_rpc(self, order_type, position, price_source, limit_price):
        """
        RPC method to test limit trigger price evaluation.

        Args:
            order_type: OrderType enum (auto-pickled)
            position: Position object or None (auto-pickled)
            price_source: PriceSource object (auto-pickled)
            limit_price: Limit price to check

        Returns:
            Trigger price if triggered, None otherwise
        """
        if not self.running_unit_tests:
            raise Exception('evaluate_limit_trigger_price_rpc can only be called in unit test mode')

        return self._manager._evaluate_limit_trigger_price(order_type, position, price_source, limit_price)

    def fill_limit_order_with_price_source_rpc(self, miner_hotkey, order, price_source, fill_price, enforce_market_cooldown=False):
        """
        RPC method to test filling a limit order with a specific price source.

        Args:
            miner_hotkey: Miner's hotkey
            order: Order object (auto-pickled)
            price_source: PriceSource object (auto-pickled)
            fill_price: Price to fill at
            enforce_market_cooldown: Whether to enforce market cooldown

        Returns:
            Error message on failure, None on success
        """
        if not self.running_unit_tests:
            raise Exception('fill_limit_order_with_price_source_rpc can only be called in unit test mode')

        return self._manager._fill_limit_order_with_price_source(
            miner_hotkey, order, price_source, fill_price, enforce_market_cooldown
        )

    def count_unfilled_orders_for_hotkey_rpc(self, miner_hotkey):
        """
        RPC method to count unfilled orders for a hotkey.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            Count of unfilled orders
        """
        if not self.running_unit_tests:
            raise Exception('count_unfilled_orders_for_hotkey_rpc can only be called in unit test mode')

        return self._manager._count_unfilled_orders_for_hotkey(miner_hotkey)

    def get_position_for_rpc(self, hotkey, order):
        """
        RPC method to get position for hotkey/trade pair.

        Args:
            hotkey: Miner's hotkey
            order: Order object (auto-pickled)

        Returns:
            Position object or None (auto-pickled)
        """
        if not self.running_unit_tests:
            raise Exception('get_position_for_rpc can only be called in unit test mode')

        return self._manager._get_open_position(hotkey, order)

    def create_sltp_order_rpc(self, miner_hotkey, parent_order):
        """
        RPC method to create SL/TP bracket order from a filled order.
        Args:
            miner_hotkey: Miner's hotkey
            parent_order: Parent order object (auto-pickled) - the filled order

        Returns:
            None
        """
        return self._manager.create_sltp_order(miner_hotkey, parent_order)

    def evaluate_bracket_trigger_price_rpc(self, order, position, price_source):
        """
        RPC method to test bracket order trigger price evaluation.

        Args:
            order: Bracket order object (auto-pickled)
            position: Position object or None (auto-pickled)
            price_source: PriceSource object (auto-pickled)

        Returns:
            Trigger price if triggered, None otherwise
        """
        if not self.running_unit_tests:
            raise Exception('evaluate_bracket_trigger_price_rpc can only be called in unit test mode')

        return self._manager._evaluate_bracket_trigger_price(order, position, price_source)

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def process_limit_order(self, miner_hotkey, order, is_edit=False):
        """Process a limit order (direct call for tests)."""
        return self._manager.process_limit_order(miner_hotkey, order, is_edit)

    def cancel_limit_order(self, miner_hotkey, trade_pair_id, order_uuid, now_ms):
        """Cancel limit order(s) (direct call for tests)."""
        return self._manager.cancel_limit_order(miner_hotkey, trade_pair_id, order_uuid, now_ms)

    def get_limit_order_by_uuid(self, miner_hotkey, order_uuid):
        """Get an unfilled limit order by UUID (direct call for tests)."""
        return self._manager.get_limit_order_by_uuid(miner_hotkey, order_uuid)

    def get_limit_orders(self, miner_hotkey):
        """Get all limit orders for a hotkey (direct call for tests)."""
        return self._manager.get_limit_orders_for_hotkey_rpc(miner_hotkey)

    def get_all_limit_orders(self):
        """Get all limit orders (direct call for tests)."""
        return self._manager.get_all_limit_orders_rpc()

    def delete_all_limit_orders_for_hotkey(self, miner_hotkey):
        """Delete all limit orders for a hotkey (direct call for tests)."""
        return self._manager.delete_all_limit_orders_for_hotkey_rpc(miner_hotkey)

    def to_dashboard_dict(self, miner_hotkey, status_filter=None):
        """Get dashboard representation (direct call for tests)."""
        return self._manager.to_dashboard_dict_rpc(miner_hotkey, status_filter)

    def clear_limit_orders(self):
        """Clear all limit orders (direct call for tests)."""
        return self._manager.clear_limit_orders()

