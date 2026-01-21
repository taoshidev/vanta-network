from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class LimitOrderClient(RPCClientBase):
    """
    Lightweight RPC client for LimitOrderServer.

    Can be created in ANY process. No server ownership.
    No pickle complexity - just pass the port to child processes.

    Usage:
        # In any process that needs limit order data
        client = LimitOrderClient()

        client.process_limit_order(miner_hotkey, order)

    For child processes:
        # Parent passes port number (not manager object!)
        Process(target=child_func, args=(limit_order_port,))

        # Child creates its own client
        def child_func(limit_order_port):
            client = LimitOrderClient(port=limit_order_port)
            client.process_limit_order(miner_hotkey, order)
    """

    def __init__(self, port: int = None, connect_immediately: bool = False, running_unit_tests=False,
                 connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        """
        Initialize limit order client.

        Args:
            port: Port number of the limit order server (default: ValiConfig.RPC_LIMITORDERMANAGER_PORT)
            connect_immediately: If True, connect in __init__. If False, call connect() later.
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_LIMITORDERMANAGER_SERVICE_NAME,
            port=port or ValiConfig.RPC_LIMITORDERMANAGER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Order Processing Methods ====================

    def process_limit_order(self, miner_hotkey: str, order: Order, is_edit: bool = False) -> dict:
        """
        Process a limit order via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            order: Order object to save
            is_edit: If True, this is an edit operation (replaces existing order)

        Returns:
            dict with status and order_uuid

        Raises:
            SignalException: Validation errors (pickled from server)
            Exception: RPC or server errors
        """
        return self._server.process_limit_order_rpc(miner_hotkey, order, is_edit)

    def cancel_limit_order(self, miner_hotkey: str, trade_pair_id: str,
                          order_uuid: str, now_ms: int) -> dict:
        """
        Cancel limit order(s) via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID string
            order_uuid: UUID of order to cancel
            now_ms: Current timestamp

        Returns:
            dict with cancellation details

        Raises:
            SignalException: Order not found (pickled from server)
            Exception: RPC or server errors
        """
        return self._server.cancel_limit_order(miner_hotkey, trade_pair_id, order_uuid, now_ms)

    # ==================== Query Methods ====================

    def get_limit_order_by_uuid(self, miner_hotkey: str, order_uuid: str) -> dict:
        """
        Get an unfilled limit order by UUID via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            order_uuid: UUID of the order to find

        Returns:
            Order dict if found, None if not found
        """
        return self._server.get_limit_order_by_uuid_rpc(miner_hotkey, order_uuid)

    def get_limit_orders(self, miner_hotkey: str) -> list:
        """
        Get all limit orders for a hotkey via RPC.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            List of order dicts
        """
        return self._server.get_limit_orders_for_hotkey_rpc(miner_hotkey)

    def get_limit_orders_for_trade_pair(self, trade_pair_id: str) -> dict:
        """
        Get all limit orders for a trade pair via RPC.

        Args:
            trade_pair_id: Trade pair ID string

        Returns:
            Dict of {hotkey: [order_dicts]}
        """
        return self._server.get_limit_orders_for_trade_pair_rpc(trade_pair_id)

    def get_all_limit_orders(self) -> dict:
        """
        Get all limit orders via RPC.

        Returns:
            Dict of {trade_pair_id: {hotkey: [order_dicts]}}
        """
        return self._server.get_all_limit_orders_rpc()

    def to_dashboard_dict(self, miner_hotkey: str, status_filter: list = None):
        """
        Get dashboard representation via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            status_filter: Optional list of status strings ['unfilled', 'filled', 'cancelled']

        Returns:
            If status_filter is None: list of order dicts (backward compatible)
            If status_filter provided: dict of {status: [order dicts]}
        """
        return self._server.to_dashboard_dict_rpc(miner_hotkey, status_filter)

    # ==================== Mutation Methods ====================

    def delete_all_limit_orders_for_hotkey(self, miner_hotkey: str) -> dict:
        """
        Delete all limit orders for a hotkey via RPC.

        This is called when a miner is eliminated to clean up their limit order data.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            dict with deletion details

        Raises:
            Exception: RPC or server errors
        """
        return self._server.delete_all_limit_orders_for_hotkey_rpc(miner_hotkey)

    def sync_limit_orders(self, sync_data: dict) -> None:
        """
        Sync limit orders from external source via RPC.

        Args:
            sync_data: Dict of {miner_hotkey: [order_dicts]}
        """
        return self._server.sync_limit_orders_rpc(sync_data)

    def clear_limit_orders(self) -> None:
        """
        Clear all limit orders from memory via RPC.

        This is primarily used for testing and development.
        Does NOT delete orders from disk.
        """
        return self._server.clear_limit_orders_rpc()

    # ==================== Test-Only Methods ====================

    def check_and_fill_limit_orders(self, call_id=None) -> dict:
        """
        Manually trigger limit order check and fill (daemon method) via RPC.

        This is primarily used for testing to trigger fills without waiting for daemon.

        Args:
            call_id: Optional unique identifier to prevent RPC caching. Pass a unique value
                    (like timestamp) in tests to ensure each call executes.

        Returns:
            dict: Execution stats with {'checked': int, 'filled': int, 'timestamp_ms': int}
        """
        return self._server.check_and_fill_limit_orders_rpc(call_id)

    def get_limit_orders_dict(self) -> dict:
        """
        Get internal _limit_orders dict for test verification via RPC.

        Returns: Dict[str, Dict[str, List[dict]]] - trade_pair_id -> hotkey -> [order_dicts]
        """
        return self._server.get_limit_orders_dict_rpc()

    def set_limit_orders_dict(self, orders_dict: dict) -> None:
        """
        Set internal _limit_orders dict for testing via RPC.

        Args:
            orders_dict: Dict[str, Dict[str, List[dict]]] - trade_pair_id -> hotkey -> [order_dicts]
        """
        return self._server.set_limit_orders_dict_rpc(orders_dict)

    def get_last_fill_time(self) -> dict:
        """
        Get internal _last_fill_time dict for test verification via RPC.

        Returns: Dict[str, Dict[str, int]] - trade_pair_id -> hotkey -> timestamp_ms
        """
        return self._server.get_last_fill_time_rpc()

    def set_last_fill_time(self, trade_pair_id: str, hotkey: str, fill_time: int) -> None:
        """
        Set _last_fill_time for testing via RPC.

        Args:
            trade_pair_id: Trade pair ID string
            hotkey: Miner hotkey
            fill_time: Timestamp in milliseconds
        """
        return self._server.set_last_fill_time_rpc(trade_pair_id, hotkey, fill_time)

    def evaluate_limit_trigger_price(self, order_type, position, price_source, limit_price: float):
        """
        Test limit trigger price evaluation via RPC.

        Args:
            order_type: OrderType enum
            position: Position object or None
            price_source: PriceSource object
            limit_price: Limit price to check

        Returns:
            Trigger price if triggered, None otherwise
        """
        return self._server.evaluate_limit_trigger_price_rpc(
            order_type, position, price_source, limit_price
        )

    def fill_limit_order_with_price_source(self, miner_hotkey: str, order,
                                          price_source, fill_price: float,
                                          enforce_market_cooldown: bool = False):
        """
        Test filling a limit order with a specific price source via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            order: Order object
            price_source: PriceSource object
            fill_price: Price to fill at
            enforce_market_cooldown: Whether to enforce market cooldown

        Returns:
            Error message on failure, None on success
        """
        return self._server.fill_limit_order_with_price_source_rpc(
            miner_hotkey, order, price_source, fill_price, enforce_market_cooldown
        )

    def count_unfilled_orders_for_hotkey(self, miner_hotkey: str) -> int:
        """
        Count unfilled orders for a hotkey via RPC.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            Count of unfilled orders
        """
        return self._server.count_unfilled_orders_for_hotkey_rpc(miner_hotkey)

    def get_position_for(self, hotkey: str, order):
        """
        Get position for hotkey/trade pair via RPC.

        Args:
            hotkey: Miner's hotkey
            order: Order object

        Returns:
            Position object or None
        """
        return self._server.get_position_for_rpc(hotkey, order)

    def create_sltp_order(self, miner_hotkey: str, parent_order):
        """
        Create SL/TP bracket order from a filled market order via RPC.

        Called by OrderProcessor after a market order with stop_loss/take_profit
        fields is successfully filled. Creates a bracket order that will trigger
        when price hits SL or TP levels.

        Args:
            miner_hotkey: Miner's hotkey
            parent_order: Parent order object - the filled market order with SL/TP fields

        Returns:
            None
        """
        return self._server.create_sltp_order_rpc(miner_hotkey, parent_order)

    def evaluate_bracket_trigger_price(self, order, position, price_source):
        """
        Test bracket order trigger price evaluation via RPC.

        Args:
            order: Bracket order object
            position: Position object or None
            price_source: PriceSource object

        Returns:
            Trigger price if triggered, None otherwise
        """
        return self._server.evaluate_bracket_trigger_price_rpc(
            order, position, price_source
        )
