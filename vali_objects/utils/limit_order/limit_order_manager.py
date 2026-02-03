import os
import traceback

import bittensor as bt

from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.exceptions.bracket_order_exception import BracketOrderException
from shared_objects.locks.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource


class LimitOrderManager(CacheController):
    """
    Server-side limit order manager.

    PROCESS BOUNDARY: Runs in SEPARATE process from validator.

    Architecture:
    - Internal data: {TradePair: {hotkey: [Order]}} - regular Python dicts (NO IPC)
    - RPC methods: Called from LimitOrderManagerClient (validator process)
    - Daemon: Background thread checks/fills orders every 60 seconds
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

    def __init__(self, running_unit_tests=False, serve=True, connection_mode: RPCConnectionMode=RPCConnectionMode.RPC):
        super().__init__(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        # Create own MarketOrderManager (forward compatibility - no parameter passing)
        from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager
        self._market_order_manager = MarketOrderManager(
            serve=serve,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )
        # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
        from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests,
                                                         connection_mode=connection_mode)

        # Create own RPC clients (forward compatibility - no parameter passing)
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        from vali_objects.utils.elimination.elimination_client import EliminationClient
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )
        self._elimination_client = EliminationClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self.running_unit_tests = running_unit_tests

        # Internal data structure: {TradePair: {hotkey: [Order]}}
        # Regular Python dict - NO IPC!
        self._limit_orders = {}
        self._closed_orders = {}
        self._last_fill_time = {}
        self._last_print_time_ms = 0

        self._read_limit_orders_from_disk()

        # Create dedicated locks for protecting self._limit_orders dictionary
        # Convert limit orders structure to format expected by PositionLocks
        hotkey_to_orders = {}
        for trade_pair, hotkey_dict in self._limit_orders.items():
            for hotkey, orders in hotkey_dict.items():
                if hotkey not in hotkey_to_orders:
                    hotkey_to_orders[hotkey] = []
                hotkey_to_orders[hotkey].extend(orders)

        # limit_order_locks: protects _limit_orders dictionary operations
        self.limit_order_locks = PositionLocks(
            hotkey_to_positions=hotkey_to_orders,
            is_backtesting=running_unit_tests,
            running_unit_tests=running_unit_tests,
            mode='local'
        )

    # ============================================================================
    # RPC Methods (called from client)
    # ============================================================================

    @property
    def live_price_fetcher(self):
        """Get live price fetcher client."""
        return self._live_price_client

    @property
    def position_manager(self):
        """Get position manager client."""
        return self._position_client

    @property
    def elimination_manager(self):
        """Get elimination manager client."""
        return self._elimination_client

    @property
    def market_order_manager(self):
        """Get market order manager."""
        return self._market_order_manager

    # ==================== Public API Methods ====================
    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring"""
        total_orders = sum(
            len(orders)
            for hotkey_dict in self._limit_orders.values()
            for orders in hotkey_dict.values()
        )
        unfilled_count = sum(
            1 for hotkey_dict in self._limit_orders.values()
            for orders in hotkey_dict.values()
            for order in orders
            if order.src in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]
        )

        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "total_orders": total_orders,
            "unfilled_orders": unfilled_count,
            "num_trade_pairs": len(self._limit_orders)
        }

    # ==================== Validation Helper Methods ====================

    def _validate_sltp_against_price(self, order_type, stop_loss, take_profit, reference_price, order_uuid=None):
        """
        Validate stop loss and take profit values against a reference price.

        Args:
            order_type: OrderType.LONG or OrderType.SHORT
            stop_loss: Stop loss price (or None)
            take_profit: Take profit price (or None)
            reference_price: The price to validate against (fill price or limit price)
            order_uuid: Optional order UUID for error messages

        Raises:
            SignalException: If validation fails
        """
        order_id = f"[{order_uuid}]" if order_uuid else ""

        if order_type == OrderType.LONG:
            # For LONG: SL must be below reference, TP must be above reference
            if stop_loss is not None and stop_loss >= reference_price:
                raise SignalException(
                    f"Invalid LONG bracket order {order_id}: "
                    f"stop_loss ({stop_loss}) must be < reference_price ({reference_price})"
                )
            if take_profit is not None and take_profit <= reference_price:
                raise SignalException(
                    f"Invalid LONG bracket order {order_id}: "
                    f"take_profit ({take_profit}) must be > reference_price ({reference_price})"
                )
        elif order_type == OrderType.SHORT:
            # For SHORT: SL must be above reference, TP must be below reference
            if stop_loss is not None and stop_loss <= reference_price:
                raise SignalException(
                    f"Invalid SHORT bracket order {order_id}: "
                    f"stop_loss ({stop_loss}) must be > reference_price ({reference_price})"
                )
            if take_profit is not None and take_profit >= reference_price:
                raise SignalException(
                    f"Invalid SHORT bracket order {order_id}: "
                    f"take_profit ({take_profit}) must be < reference_price ({reference_price})"
                )
        else:
            raise SignalException(
                f"Invalid order type for bracket order {order_id}: {order_type}. Must be LONG or SHORT"
            )

    def _validate_bracket_order(self, order, open_position, unfilled_orders, reference_price=None):
        """
        Validate a BRACKET order and apply position-derived values.

        Args:
            order: Order object to validate (will be modified in place)
            open_position: Position object (optional)
            unfilled_orders: List of unfilled limit orders (optional)
            reference_price: Optional price to validate SL/TP against (e.g., limit_price, fill_price)

        Raises:
            SignalException: If validation fails
        """
        if not open_position and not unfilled_orders:
            raise SignalException(
                f"Invalid bracket order: no open position or previous unfilled limit orders found for {order.trade_pair.trade_pair_id}"
            )

        # Validate that at least one of SL or TP is set
        if order.stop_loss is None and order.take_profit is None:
            raise SignalException(
                f"BRACKET orders must have at least one of stop_loss or take_profit set"
            )

        # Set order type based on open position, skip validation if there is no position.
        if open_position:
            order.order_type = open_position.position_type
        else:
            return

        # Validate SL/TP against reference price if provided
        if reference_price is not None:
            self._validate_sltp_against_price(
                order.order_type, order.stop_loss, order.take_profit, reference_price, order.order_uuid
            )

        # Use position quantity if not specified
        if open_position and order.leverage is None and order.value is None and order.quantity is None:
            order.quantity = open_position.net_quantity

    def _validate_limit_order(self, order):
        """
        Validate a LIMIT order.

        Args:
            order: Order object to validate

        Raises:
            SignalException: If validation fails
        """
        if order.limit_price is None or order.limit_price <= 0:
            raise SignalException(
                f"LIMIT orders must have a valid limit_price > 0 (got {order.limit_price})"
            )

        if order.order_type == OrderType.FLAT:
            raise SignalException(f"FLAT order is not supported for LIMIT orders")

        # Validate bracket_orders if provided
        if order.bracket_orders:
            for i, bracket in enumerate(order.bracket_orders):
                stop_loss = bracket.get('stop_loss')
                take_profit = bracket.get('take_profit')

                # Validate SL/TP are positive if provided
                if stop_loss is not None and stop_loss <= 0:
                    raise SignalException(f"bracket_orders[{i}]: stop_loss must be greater than 0")
                if take_profit is not None and take_profit <= 0:
                    raise SignalException(f"bracket_orders[{i}]: take_profit must be greater than 0")

                # Validate SL/TP against limit price
                self._validate_sltp_against_price(
                    order.order_type, stop_loss, take_profit, order.limit_price, f"{order.order_uuid}-bracket-{i}"
                )

    # ==================== Public API Methods ====================

    def get_limit_order_by_uuid(self, miner_hotkey, order_uuid):
        """
        Get an unfilled limit order by UUID.

        Args:
            miner_hotkey: The miner's hotkey
            order_uuid: UUID of the order to find

        Returns:
            Order dict if found, None if not found
        """
        for trade_pair, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for order in hotkey_dict[miner_hotkey]:
                    if order.order_uuid == order_uuid:
                        if order.src in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                            return order.to_python_dict()
        return None

    def process_limit_order(self, miner_hotkey, order, is_edit=False):
        """
        RPC method to process a limit order or bracket order.
        Handles both new orders and edits (replacing existing order with same UUID).

        Validation responsibilities:
        - OrderProcessor (for edits): Order exists, is unfilled, trade pair matches
        - LimitOrderManager: Business rules (SL/TP relationships), max orders, immediate fill

        Args:
            miner_hotkey: The miner's hotkey
            order: Order object (pickled automatically by RPC)
                   For edits: fully-formed Order with execution_type/src already set
            is_edit: If True, this is an edit operation (replaces existing order)

        Returns:
            dict with status and order_uuid
        """
        trade_pair = order.trade_pair
        order_uuid = order.order_uuid

        # Variables to track whether to fill immediately
        should_fill_immediately = False
        trigger_price = None
        price_sources = None

        with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
            # Ensure trade_pair exists in structure
            if trade_pair not in self._limit_orders:
                self._limit_orders[trade_pair] = {}
                self._last_fill_time[trade_pair] = {}

            if miner_hotkey not in self._limit_orders[trade_pair]:
                self._limit_orders[trade_pair][miner_hotkey] = []
                self._last_fill_time[trade_pair][miner_hotkey] = 0

            if is_edit:
                # EDIT PATH: OrderProcessor already validated existence, unfilled status, and trade pair match.
                # Re-verify under lock for race condition protection.
                existing_order = self._find_existing_order_under_lock(miner_hotkey, order_uuid)
                if not existing_order:
                    raise SignalException(f"Cannot edit order {order_uuid}: order not found (race condition)")
                if existing_order.src not in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                    raise SignalException(f"Cannot edit order {order_uuid}: order is no longer unfilled (race condition)")
            else:
                # NEW ORDER PATH: Check max unfilled orders limit
                total_unfilled = self._count_unfilled_orders_for_hotkey(miner_hotkey)
                if total_unfilled >= ValiConfig.MAX_UNFILLED_LIMIT_ORDERS:
                    raise SignalException(
                        f"miner has too many unfilled limit orders "
                        f"[{total_unfilled}] >= [{ValiConfig.MAX_UNFILLED_LIMIT_ORDERS}]"
                    )

            # Get position for validation
            open_position = self._get_open_position(miner_hotkey, order)
            unfilled_orders = self._get_unfilled_orders(miner_hotkey, trade_pair)

            # Validate order using shared validation logic (business rules)
            if order.execution_type == ExecutionType.BRACKET:
                self._validate_bracket_order(order, open_position, unfilled_orders)
            elif order.execution_type == ExecutionType.LIMIT:
                self._validate_limit_order(order)

            bt.logging.info(
                f"{'EDIT' if is_edit else 'INCOMING'} {order.execution_type} ORDER | {trade_pair.trade_pair_id} | "
                f"{order.order_type.name} | bracket_orders={order.bracket_orders}"
            )

            # Check if order can be filled immediately (only if market is open)
            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, order.processed_ms)
            if price_sources and self.live_price_fetcher.is_market_open(trade_pair, order.processed_ms):
                trigger_price = self._evaluate_trigger_price(order, open_position, price_sources[0])

                if trigger_price:
                    should_fill_immediately = True

        # Fill outside the lock to avoid reentrant lock issue
        # Treat order that fills immediately as market order
        if should_fill_immediately:
            # If replacing, remove the old order first
            if is_edit:
                orders_list = self._limit_orders[trade_pair][miner_hotkey]
                for i, o in enumerate(orders_list):
                    if o.order_uuid == order_uuid:
                        orders_list.pop(i)
                        break

            order.execution_type = ExecutionType.MARKET
            order.src = OrderSource.ORGANIC
            fill_error = self._fill_limit_order_with_price_source(miner_hotkey, order, price_sources[0], None, enforce_market_cooldown=True)
            if fill_error:
                raise SignalException(fill_error)
            bt.logging.info(f"Filled order {order_uuid} @ market price {price_sources[0].close}")

        else:
            self._write_to_disk(miner_hotkey, order)
            if is_edit:
                # Replace existing order in list
                orders_list = self._limit_orders[trade_pair][miner_hotkey]
                for i, o in enumerate(orders_list):
                    if o.order_uuid == order_uuid:
                        orders_list[i] = order
                        break
                # Update bracket order on position for edits
                if order.execution_type == ExecutionType.BRACKET:
                    self.position_manager.remove_bracket_order_from_position(
                        miner_hotkey, trade_pair.trade_pair_id, order_uuid
                    )
                    self.position_manager.attach_bracket_order_to_position(
                        miner_hotkey, trade_pair.trade_pair_id, order.to_python_dict()
                    )
            else:
                # Append new order
                self._limit_orders[trade_pair][miner_hotkey].append(order)
                # Attach bracket order to position for new orders
                if order.execution_type == ExecutionType.BRACKET:
                    self.position_manager.attach_bracket_order_to_position(
                        miner_hotkey, trade_pair.trade_pair_id, order.to_python_dict()
                    )

        return {"status": "success", "order_uuid": order_uuid}

    def _find_existing_order_under_lock(self, miner_hotkey, order_uuid):
        """
        Find an existing order by UUID. Must be called while holding the lock.

        Returns:
            Order if found, None otherwise
        """
        for tp, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for o in hotkey_dict[miner_hotkey]:
                    if o.order_uuid == order_uuid:
                        return o
        return None


    def cancel_limit_order(self, miner_hotkey, trade_pair_id, order_uuid, now_ms):
        """
        RPC method to cancel limit order(s).
        Args:
            miner_hotkey: The miner's hotkey
            order_uuid: UUID of specific order to cancel, comma-separated for multiple, or None/empty for all
            now_ms: Current timestamp
        Returns:
            dict with cancellation details
        """
        # TODO support cancel by trade pair in v2
        try:
            # Parse trade_pair only if trade_pair_id is provided
            # trade_pair = TradePair.from_trade_pair_id(trade_pair_id) if trade_pair_id else None

            # Split order_uuid by commas to support multiple cancellations
            order_uuids = [uuid.strip() for uuid in order_uuid.split(',')] if order_uuid else []

            # Find orders for each UUID
            orders_to_cancel = []
            for uuid in order_uuids:
                orders_to_cancel.extend(self._find_orders_to_cancel_by_uuid(miner_hotkey, uuid))

            # Only cancel one order at a time with order_uuid
            # if not orders_to_cancel and trade_pair:
            #     orders_to_cancel = self._find_orders_to_cancel_by_trade_pair(miner_hotkey, trade_pair)

            if not orders_to_cancel:
                raise SignalException(
                    f"No unfilled limit orders found for {miner_hotkey} (uuid={order_uuid})"
                )

            for order in orders_to_cancel:
                cancel_src = OrderSource.get_cancel(order.src)
                self._close_limit_order(miner_hotkey, order, cancel_src, now_ms)

            return {
                "status": "cancelled",
                "order_uuid": order_uuid if order_uuid else "all",
                "miner_hotkey": miner_hotkey,
                "cancelled_ms": now_ms,
                "num_cancelled": len(orders_to_cancel)
            }

        except Exception as e:
            bt.logging.error(f"Error cancelling limit order: {e}")
            bt.logging.error(traceback.format_exc())
            raise

    def get_limit_orders_for_hotkey_rpc(self, miner_hotkey):
        """
        RPC method to get all limit orders for a hotkey.
        Returns:
            List of order dicts
        """
        try:
            orders = []
            for trade_pair, hotkey_dict in self._limit_orders.items():
                if miner_hotkey in hotkey_dict:
                    for order in hotkey_dict[miner_hotkey]:
                        orders.append(order.to_python_dict())
            return orders
        except Exception as e:
            bt.logging.error(f"Error getting limit orders: {e}")
            return []

    def get_limit_orders_for_trade_pair_rpc(self, trade_pair_id):
        """
        RPC method to get all limit orders for a trade pair.
        Returns:
            Dict of {hotkey: [order_dicts]}
        """
        try:
            trade_pair = TradePair.from_trade_pair_id(trade_pair_id)
            if trade_pair not in self._limit_orders:
                return {}

            result = {}
            for hotkey, orders in self._limit_orders[trade_pair].items():
                result[hotkey] = [order.to_python_dict() for order in orders]
            return result
        except Exception as e:
            bt.logging.error(f"Error getting limit orders for trade pair: {e}")
            return {}

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
        try:
            filtered_orders = []

            if not status_filter or "unfilled" in status_filter:
                # Get unfilled from memory
                for _, hotkey_dict in self._limit_orders.items():
                    if miner_hotkey in hotkey_dict:
                        for order in hotkey_dict[miner_hotkey]:
                            filtered_orders.append(order)

            # No filter - return flat list (backward compatible)
            if not status_filter:
                return_data = [self._order_to_dict(o) for o in filtered_orders]
                return return_data if return_data else None

            # Get closed from cache (only when filtering)
            for closed_order in self._closed_orders.get(miner_hotkey, []):
                filtered_orders.append(closed_order)

            # With filter - return dict grouped by status
            status_set = set(s.upper() for s in status_filter)
            result = {s.lower(): [] for s in status_set}

            for order in filtered_orders:
                status = OrderSource.status(order.src)  # "UNFILLED", "FILLED", "CANCELLED"
                if status in status_set:
                    result[status.lower()].append(self._order_to_dict(order))

            return result if any(result.values()) else None

        except Exception as e:
            bt.logging.error(f"Error creating dashboard dict: {e}")
            return None

    def _order_to_dict(self, order):
        """Convert order to dict for dashboard response."""
        return order.to_python_dict()

    def get_all_limit_orders_rpc(self):
        """
        RPC method to get all limit orders across all trade pairs and hotkeys.

        Returns:
            Dict of {trade_pair_id: {hotkey: [order_dicts]}}
        """
        try:
            result = {}
            for trade_pair, hotkey_dict in self._limit_orders.items():
                trade_pair_id = trade_pair.trade_pair_id
                result[trade_pair_id] = {}
                for hotkey, orders in hotkey_dict.items():
                    result[trade_pair_id][hotkey] = [order.to_python_dict() for order in orders]
            return result
        except Exception as e:
            bt.logging.error(f"Error getting all limit orders: {e}")
            return {}

    def delete_all_limit_orders_for_hotkey_rpc(self, miner_hotkey):
        """
        RPC method to delete all limit orders (both in-memory and on-disk) for a hotkey.

        This is called when a miner is eliminated to clean up their limit order data.

        Args:
            miner_hotkey: The miner's hotkey

        Returns:
            dict with deletion details
        """
        try:
            deleted_count = 0

            # Delete from memory and disk for each trade pair
            for trade_pair in list(self._limit_orders.keys()):
                # Acquire lock for this specific (hotkey, trade_pair) combination
                with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                    if miner_hotkey in self._limit_orders[trade_pair]:
                        orders = self._limit_orders[trade_pair][miner_hotkey]
                        deleted_count += len(orders)

                        # Delete disk files for each order
                        for order in orders:
                            self._delete_from_disk(miner_hotkey, order)

                        # Remove from memory
                        del self._limit_orders[trade_pair][miner_hotkey]

                        # Clean up _last_fill_time for this hotkey
                        if trade_pair in self._last_fill_time and miner_hotkey in self._last_fill_time[trade_pair]:
                            del self._last_fill_time[trade_pair][miner_hotkey]

                        # Clean up empty trade_pair entries
                        if not self._limit_orders[trade_pair]:
                            del self._limit_orders[trade_pair]
                            # Also remove from _last_fill_time to prevent memory leak
                            if trade_pair in self._last_fill_time:
                                del self._last_fill_time[trade_pair]

            bt.logging.info(f"Deleted {deleted_count} limit orders for eliminated miner [{miner_hotkey}]")

            return {
                "status": "deleted",
                "miner_hotkey": miner_hotkey,
                "deleted_count": deleted_count
            }

        except Exception as e:
            bt.logging.error(f"Error deleting limit orders for hotkey {miner_hotkey}: {e}")
            bt.logging.error(traceback.format_exc())
            raise

    # ============================================================================
    # Daemon Method (runs in separate process)
    # ============================================================================


    def check_and_fill_limit_orders(self, call_id=None):
        """
        Iterate through all trade pairs and attempt to fill unfilled limit orders.

        Args:
            call_id: Optional unique identifier for this call. Used to prevent RPC caching.
                    In production (daemon), this is not needed. In tests, pass a unique value
                    (like timestamp) to ensure each call executes.

        Returns:
            dict: Execution stats with {
                'checked': int,      # Orders checked
                'filled': int,       # Orders filled
                'timestamp_ms': int  # Execution timestamp
            }
        """
        now_ms = TimeUtil.now_in_millis()
        total_checked = 0
        total_filled = 0

        if self.running_unit_tests:
            print(f"[CHECK_AND_FILL_CALLED] check_and_fill_limit_orders(call_id={call_id}) called, {len(self._limit_orders)} trade pairs")

        if now_ms - self._last_print_time_ms > 60 * 1000:
            total_orders = sum(len(orders) for hotkey_dict in self._limit_orders.values() for orders in hotkey_dict.values())
            bt.logging.info(f"Checking {total_orders} limit orders across {len(self._limit_orders)} trade pairs")
            self._last_print_time_ms = now_ms

        for trade_pair, hotkey_dict in self._limit_orders.items():
            # Check if market is open
            if not self.live_price_fetcher.is_market_open(trade_pair, now_ms):
                if self.running_unit_tests:
                    print(f"[CHECK_ORDERS DEBUG] Market closed for {trade_pair.trade_pair_id}")
                bt.logging.debug(f"Market closed for {trade_pair.trade_pair_id}, skipping")
                continue

            # Get price sources for this trade pair
            # price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, now_ms)
            price_sources = self._get_best_price_source(trade_pair, now_ms)
            if not price_sources:
                if self.running_unit_tests:
                    print(f"[CHECK_ORDERS DEBUG] No price sources for {trade_pair.trade_pair_id}")
                bt.logging.debug(f"No price sources for {trade_pair.trade_pair_id}, skipping")
                continue

            # Iterate through all hotkeys for this trade pair
            for miner_hotkey, orders in hotkey_dict.items():
                last_fill_time = self._last_fill_time.get(trade_pair, {}).get(miner_hotkey, 0)
                time_since_last_fill = now_ms - last_fill_time

                if time_since_last_fill < ValiConfig.LIMIT_ORDER_FILL_INTERVAL_MS:
                    bt.logging.info(f"Skipping {trade_pair.trade_pair_id} for {miner_hotkey}: {time_since_last_fill}ms since last fill")
                    continue

                for order in orders:
                    # Check both regular limit orders and SL/TP Bracket orders
                    if order.src not in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                        continue

                    total_checked += 1

                    # Cancel bracket orders with no position and no unfilled limit orders created before it
                    if order.src == OrderSource.BRACKET_UNFILLED:
                        position = self._get_open_position(miner_hotkey, order)
                        unfilled_orders = self._get_unfilled_orders(miner_hotkey, trade_pair, before_ms=order.processed_ms)
                        if not position and not unfilled_orders:
                            bt.logging.info(f"[BRACKET CANCELLED] No position found for bracket order {order.order_uuid}, cancelling")
                            self._close_limit_order(miner_hotkey, order, OrderSource.BRACKET_CANCELLED, now_ms)
                            continue

                    # Attempt to fill
                    if self._attempt_fill_limit_order(miner_hotkey, order, price_sources, now_ms):
                        total_filled += 1
                        # DESIGN: Break after first fill to enforce LIMIT_ORDER_FILL_INTERVAL_MS
                        # Only one order per trade pair per hotkey can fill within the interval.
                        # This prevents rapid sequential fills and enforces rate limiting.
                        break

        if total_filled > 0:
            bt.logging.info(f"Limit order check complete: checked={total_checked}, filled={total_filled}")

        return {
            'checked': total_checked,
            'filled': total_filled,
            'timestamp_ms': now_ms
        }

    # ============================================================================
    # Internal Helper Methods
    # ============================================================================

    def _get_unfilled_orders(self, miner_hotkey: str, trade_pair: TradePair, before_ms: int = None) -> list:
        """
        Get unfilled limit orders for a miner and trade pair.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair: The trade pair to filter by
            before_ms: If provided, only return orders created before this timestamp

        Returns:
            List of unfilled limit orders
        """
        if trade_pair not in self._limit_orders:
            return []

        if miner_hotkey not in self._limit_orders[trade_pair]:
            return []

        orders = [
            order for order in self._limit_orders[trade_pair][miner_hotkey]
            if order.src == OrderSource.LIMIT_UNFILLED
        ]

        if before_ms is not None:
            orders = [order for order in orders if order.processed_ms < before_ms]

        return orders


    def _count_unfilled_orders_for_hotkey(self, miner_hotkey):
        """Count total unfilled orders across all trade pairs for a hotkey."""
        count = 0
        for trade_pair, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for order in hotkey_dict[miner_hotkey]:
                    # Count both regular limit orders and bracket orders
                    if order.src in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                        count += 1
        return count

    def _find_orders_to_cancel_by_uuid(self, miner_hotkey, order_uuid):
        """
        Find orders to cancel by UUID across all trade pairs.

        DESIGN: Supports partial UUID matching for bracket orders.
        When a limit order with SL/TP fills, it creates a bracket order with UUID format:
        "{parent_order_uuid}-bracket"

        This allows miners to cancel the resulting bracket order by providing the parent
        order's UUID. Example:
        - Parent limit order UUID: "abc123"
        - Created bracket order UUID: "abc123-bracket"
        - Miner can cancel bracket by providing "abc123" (startswith matching)
        """
        orders_to_cancel = []
        for trade_pair, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for order in hotkey_dict[miner_hotkey]:
                    # Exact match for regular limit orders
                    if order.order_uuid == order_uuid and order.src == OrderSource.LIMIT_UNFILLED:
                        orders_to_cancel.append(order)
                    # Prefix match for bracket orders (allows canceling via parent UUID)
                    elif order.src == OrderSource.BRACKET_UNFILLED and order.order_uuid.startswith(order_uuid):
                        orders_to_cancel.append(order)

        return orders_to_cancel

    def _find_order_by_uuid(self, miner_hotkey, order_uuid):
        """
        Find a single unfilled order by UUID across all trade pairs.

        Args:
            miner_hotkey: The miner's hotkey
            order_uuid: UUID of the order to find

        Returns:
            Tuple of (order, trade_pair) if found, raises SignalException if not found
        """
        for trade_pair, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for order in hotkey_dict[miner_hotkey]:
                    if order.order_uuid == order_uuid:
                        return order, trade_pair

        raise SignalException(
            f"No unfilled limit order found for {miner_hotkey} with uuid={order_uuid}"
        )

    def _find_orders_to_cancel_by_trade_pair(self, miner_hotkey, trade_pair):
        """Find all unfilled orders for a specific trade pair."""
        orders_to_cancel = []
        if trade_pair in self._limit_orders and miner_hotkey in self._limit_orders[trade_pair]:
            for order in self._limit_orders[trade_pair][miner_hotkey]:
                if order.src in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                    orders_to_cancel.append(order)
        return orders_to_cancel

    def _get_best_price_source(self, trade_pair, now_ms):
        """
        Get the best price source for a trade pair at a given time.
        Uses the median price source to avoid outliers.

        Args:
            trade_pair: TradePair to get price for
            now_ms: Current timestamp in milliseconds

        Returns:
            The median price source, or None if no price sources available
        """
        end_ms = now_ms
        start_ms = now_ms - ValiConfig.LIMIT_ORDER_PRICE_BUFFER_MS
        price_sources = self.live_price_fetcher.get_ws_price_sources_in_window(trade_pair, start_ms, end_ms)

        if not price_sources:
            return None

        # Sort price sources by close price and return median
        sorted_sources = sorted(price_sources, key=lambda ps: ps.close)
        median_index = len(sorted_sources) // 2
        return [sorted_sources[median_index]]


    def _attempt_fill_limit_order(self, miner_hotkey, order, price_sources, now_ms):
        """
        Attempt to fill a limit order. Returns True if filled, False otherwise.

        IMPORTANT: This method checks trigger conditions under lock, but releases the lock
        before calling _fill_limit_order_with_price_source to avoid deadlock (since that
        method calls _close_limit_order which also acquires a lock).
        """
        trade_pair = order.trade_pair
        should_fill = False
        best_price_source = None
        trigger_price = None

        try:
            # Check if order should be filled (under limit_order_locks)
            with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                # Verify order still unfilled (either regular limit or SL/TP)
                if order.src not in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                    return False

                # Check if limit price triggered
                best_price_source = price_sources[0]
                position = self._get_open_position(miner_hotkey, order)
                unfilled_orders = self._get_unfilled_orders(miner_hotkey, trade_pair)
                trigger_price = self._evaluate_trigger_price(order, position, best_price_source)

                if trigger_price is not None:
                    should_fill = True

            # Fill OUTSIDE the lock to avoid deadlock with _close_limit_order
            # Note: There's a small window where order could be cancelled between check and fill,
            # but _fill_limit_order_with_price_source handles this gracefully
            if should_fill:
                self._fill_limit_order_with_price_source(miner_hotkey, order, best_price_source, trigger_price)
                return True

            return False

        except Exception as e:
            bt.logging.error(f"Error attempting to fill limit order {order.order_uuid}: {e}")
            bt.logging.error(traceback.format_exc())
            return False

    def _fill_limit_order_with_price_source(self, miner_hotkey, order, price_source, fill_price, enforce_market_cooldown=False):
        """Fill a limit order and update position. Returns error message on failure, None on success."""
        trade_pair = order.trade_pair
        fill_time = price_source.start_ms
        error_msg = None

        new_src = OrderSource.get_fill(order.src)

        try:
            order_dict = Order.to_python_dict(order)
            order_dict['price'] = fill_price

            # Reverse order direction when exeucting BRACKET orders
            if order.execution_type == ExecutionType.BRACKET:
                # Get the closing order type (opposite direction)
                closing_order_type = OrderType.opposite_order_type(order.order_type)
                if closing_order_type:
                    order_dict['order_type'] = closing_order_type.name
                    order_dict['leverage'] = -order.leverage if order.leverage else None
                    order_dict['value'] = -order.value if order.value else None
                    order_dict['quantity'] = -order.quantity if order.quantity else None
                else:
                    raise ValueError("Bracket Order type was not LONG or SHORT")

            err_msg, updated_position, created_order = self.market_order_manager._process_market_order(
                order.order_uuid,
                "limit_order",
                trade_pair,
                fill_time,
                order_dict,
                miner_hotkey,
                [price_source],
                enforce_market_cooldown
            )

            # Issue 2: Check if err_msg is set - treat as failure
            if err_msg:
                raise ValueError(err_msg)

            # Issue 5: updated_position being None is an error case, not fallback
            if not updated_position:
                raise ValueError("No position returned from market order processing")

            # Issue 4: Copy values TO original order object rather than reassigning variable
            filled_order = updated_position.orders[-1]
            order.leverage = filled_order.leverage
            order.value = filled_order.value
            order.quantity = filled_order.quantity
            order.price_sources = filled_order.price_sources
            order.price = fill_price if fill_price else filled_order.price
            order.bid = filled_order.bid
            order.ask = filled_order.ask
            order.slippage = filled_order.slippage
            order.processed_ms = filled_order.processed_ms

            # Issue 3: Log success only after successful update
            bt.logging.success(f"Filled limit order {order.order_uuid} at {order.price}")

            if trade_pair not in self._last_fill_time:
                self._last_fill_time[trade_pair] = {}
            self._last_fill_time[trade_pair][miner_hotkey] = fill_time


            if order.execution_type == ExecutionType.LIMIT:
                if updated_position.is_open_position and len(updated_position.orders) == 1:
                    # New position created - sync any pending bracket orders with the position
                    self._sync_pending_bracket_orders(miner_hotkey, updated_position)

                if order.bracket_orders is not None:
                    self.create_sltp_order(miner_hotkey, order)

        except BracketOrderException as e:
            error_msg = f"Limit order [{order.order_uuid}] filled successfully, but bracket order creation failed: {e}"
            bt.logging.warning(error_msg)

        except Exception as e:
            error_msg = f"Could not fill limit order [{order.order_uuid}]: {e}. Cancelling order"
            bt.logging.error(error_msg)
            new_src = OrderSource.get_cancel(order.src)

        finally:
            self._close_limit_order(miner_hotkey, order, new_src, fill_time)

        return error_msg

    def _close_limit_order(self, miner_hotkey, order, src, time_ms):
        """Mark order as closed and update disk."""
        order_uuid = order.order_uuid
        trade_pair = order.trade_pair
        trade_pair_id = trade_pair.trade_pair_id

        # Orders that were filled immediately are not stored on disk or in memory
        if src == OrderSource.ORGANIC:
            order.src = src
            order.processed_ms = time_ms
            bt.logging.info(f"Closed ORGANIC limit order [{order_uuid}] [{trade_pair_id}] for [{miner_hotkey}] - no disk cleanup")
            return

        with self.limit_order_locks.get_lock(miner_hotkey, trade_pair_id):
            unfilled_dir = ValiBkpUtils.get_limit_orders_dir(miner_hotkey, trade_pair_id, "unfilled", self.running_unit_tests)
            closed_filename = unfilled_dir + order_uuid

            if os.path.exists(closed_filename):
                os.remove(closed_filename)
            else:
                bt.logging.warning(f"Closed unfilled limit order not found on disk [{order_uuid}]")

            order.src = src
            order.processed_ms = time_ms
            self._write_to_disk(miner_hotkey, order)

            # Remove closed orders from memory to prevent memory leak
            # Closed orders are persisted to disk and don't need to stay in memory
            if trade_pair in self._limit_orders and miner_hotkey in self._limit_orders[trade_pair]:
                orders = self._limit_orders[trade_pair][miner_hotkey]
                # Remove the order from the list instead of updating it
                self._limit_orders[trade_pair][miner_hotkey] = [
                    o for o in orders if o.order_uuid != order_uuid
                ]

            # Remove from position if bracket order
            if order.execution_type == ExecutionType.BRACKET:
                self.position_manager.remove_bracket_order_from_position(
                    miner_hotkey, trade_pair_id, order_uuid
                )

            if miner_hotkey not in self._closed_orders:
                self._closed_orders[miner_hotkey] = []
            self._closed_orders[miner_hotkey].append(order)

            bt.logging.info(f"Successfully closed limit order [{order_uuid}] [{trade_pair_id}] for [{miner_hotkey}]")

    def create_sltp_order(self, miner_hotkey, parent_order):
        """
        Create bracket order(s) from parent_order.bracket_orders list.

        Note: Order's normalize_bracket_orders validator converts stop_loss/take_profit
        to bracket_orders format, so this method only needs to process bracket_orders.

        DESIGN: Bracket order UUID format is "{parent_uuid}-bracket-{i}"
        This allows miners to cancel bracket orders by providing the parent order UUID.
        See _find_orders_to_cancel_by_uuid() for the cancellation logic.
        """
        trade_pair = parent_order.trade_pair
        now_ms = TimeUtil.now_in_millis()

        # Validate fill price exists
        fill_price = parent_order.price
        if not fill_price:
            raise BracketOrderException(f"Unexpected: no fill price from order [{parent_order.order_uuid}]")

        if not parent_order.bracket_orders:
            raise SignalException(f"No bracket_orders specified for order [{parent_order.order_uuid}]")

        # Build brackets to create
        brackets_to_create = []
        for i, bracket in enumerate(parent_order.bracket_orders):
            stop_loss = float(bracket['stop_loss']) if bracket.get('stop_loss') is not None else None
            take_profit = float(bracket['take_profit']) if bracket.get('take_profit') is not None else None
            leverage = float(bracket['leverage']) if bracket.get('leverage') is not None else None
            value = float(bracket['value']) if bracket.get('value') is not None else None
            quantity = float(bracket['quantity']) if bracket.get('quantity') is not None else None

            bracket_uuid = f"{parent_order.order_uuid}-bracket-{i}"

            self._validate_sltp_against_price(
                parent_order.order_type, stop_loss, take_profit,
                fill_price, bracket_uuid
            )

            brackets_to_create.append({
                'uuid': bracket_uuid,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': leverage,
                'value': value,
                'quantity': quantity,
            })

        try:
            with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                if trade_pair not in self._limit_orders:
                    self._limit_orders[trade_pair] = {}
                    self._last_fill_time[trade_pair] = {}
                if miner_hotkey not in self._limit_orders[trade_pair]:
                    self._limit_orders[trade_pair][miner_hotkey] = []
                    self._last_fill_time[trade_pair][miner_hotkey] = 0

                for bracket_data in brackets_to_create:
                    bracket_order = Order(
                        trade_pair=trade_pair,
                        order_uuid=bracket_data['uuid'],
                        processed_ms=now_ms,
                        price=0.0,
                        order_type=parent_order.order_type,
                        leverage=bracket_data['leverage'],
                        value=bracket_data['value'],
                        quantity=bracket_data['quantity'],
                        execution_type=ExecutionType.BRACKET,
                        limit_price=None,
                        stop_loss=bracket_data['stop_loss'],
                        take_profit=bracket_data['take_profit'],
                        src=OrderSource.BRACKET_UNFILLED
                    )

                    self._write_to_disk(miner_hotkey, bracket_order)
                    self._limit_orders[trade_pair][miner_hotkey].append(bracket_order)

                    # Attach bracket order to position via RPC
                    self.position_manager.attach_bracket_order_to_position(
                        miner_hotkey, trade_pair.trade_pair_id, bracket_order.to_python_dict()
                    )

                    bt.logging.success(
                        f"Created bracket order [{bracket_order.order_uuid}] "
                        f"with SL={bracket_data['stop_loss']}, TP={bracket_data['take_profit']}"
                    )

        except Exception as e:
            bt.logging.error(f"Error creating bracket order: {e}")
            bt.logging.error(traceback.format_exc())
            raise BracketOrderException(f"Error creating bracket order: {e}")

    def _get_open_position(self, hotkey, order):
        """Get open position for hotkey and trade pair."""
        trade_pair_id = order.trade_pair.trade_pair_id
        return self.position_manager.get_open_position_for_trade_pair(hotkey, trade_pair_id)

    def _evaluate_trigger_price(self, order, position, ps):
        if order.execution_type == ExecutionType.LIMIT:
            return self._evaluate_limit_trigger_price(order.order_type, position, ps, order.limit_price)

        elif order.execution_type == ExecutionType.BRACKET:
            return self._evaluate_bracket_trigger_price(order, position, ps)

        return None


    def _evaluate_limit_trigger_price(self, order_type, position, ps, limit_price):
        """Check if limit price is triggered. Returns the limit_price if triggered, None otherwise."""
        bid_price = ps.bid if ps.bid > 0 else ps.open
        ask_price = ps.ask if ps.ask > 0 else ps.open

        if order_type == OrderType.LONG:
            return limit_price if ask_price <= limit_price else None
        elif order_type == OrderType.SHORT:
            return limit_price if bid_price >= limit_price else None
        else:
            return None

    def _evaluate_bracket_trigger_price(self, order, position, ps):
        """
        Evaluate trigger price for bracket orders (SLTP combined).
        Checks both stop_loss and take_profit boundaries.
        Returns trigger price when either boundary is hit.

        The bracket order has the SAME type as the parent order.

        Trigger logic based on order type:
        - LONG order: SL triggers when price < SL, TP triggers when price > TP
        - SHORT order: SL triggers when price > SL, TP triggers when price < TP
        """
        if not position:
            return None

        bid_price = ps.bid if ps.bid > 0 else ps.open
        ask_price = ps.ask if ps.ask > 0 else ps.open

        order_type = position.position_type
        order.order_type = order_type

        # For LONG orders:
        # - Stop loss: triggers when market price < SL (use bid for selling)
        # - Take profit: triggers when market price > TP (use bid for selling)
        if order_type == OrderType.LONG:
            # Check stop loss first (higher priority on losses)
            if order.stop_loss is not None and bid_price < order.stop_loss:
                bt.logging.info(f"Bracket order stop loss triggered: bid={bid_price} < SL={order.stop_loss}")
                return order.stop_loss
            # Check take profit
            if order.take_profit is not None and bid_price > order.take_profit:
                bt.logging.info(f"Bracket order take profit triggered: bid={bid_price} > TP={order.take_profit}")
                return order.take_profit

        # For SHORT orders:
        # - Stop loss: triggers when market price > SL (use ask for buying)
        # - Take profit: triggers when market price < TP (use ask for buying)
        elif order_type == OrderType.SHORT:
            # Check stop loss first (higher priority on losses)
            if order.stop_loss is not None and ask_price > order.stop_loss:
                bt.logging.info(f"Bracket order stop loss triggered: ask={ask_price} > SL={order.stop_loss}")
                return order.stop_loss
            # Check take profit
            if order.take_profit is not None and ask_price < order.take_profit:
                bt.logging.info(f"Bracket order take profit triggered: ask={ask_price} < TP={order.take_profit}")
                return order.take_profit

        return None

    def _read_limit_orders_from_disk(self, hotkeys=None):
        """Read limit orders from disk and populate internal structure."""
        if not hotkeys:
            hotkeys = ValiBkpUtils.get_directories_in_dir(
                ValiBkpUtils.get_miner_dir(self.running_unit_tests)
            )

        eliminated_hotkeys = self.elimination_manager.get_eliminated_hotkeys()

        total_orders_read = 0
        total_bracket_orders = 0
        total_bracket_attached = 0

        bt.logging.info(f"[LIMIT ORDER DISK] Reading limit orders from disk for {len(hotkeys)} hotkeys...")

        for hotkey in hotkeys:
            if hotkey in eliminated_hotkeys:
                continue

            miner_order_dicts = ValiBkpUtils.get_limit_orders(hotkey, False, running_unit_tests=self.running_unit_tests)
            for order_dict in miner_order_dicts:
                try:
                    order = Order.from_dict(order_dict)
                    trade_pair = order.trade_pair

                    # Initialize nested structure
                    if trade_pair not in self._limit_orders:
                        self._limit_orders[trade_pair] = {}
                        self._last_fill_time[trade_pair] = {}
                    if hotkey not in self._limit_orders[trade_pair]:
                        self._limit_orders[trade_pair][hotkey] = []

                    if OrderSource.is_open(order.src):
                        self._limit_orders[trade_pair][hotkey].append(order)
                        total_orders_read += 1
                        # Attach bracket orders to position
                        if order.src == OrderSource.BRACKET_UNFILLED:
                            total_bracket_orders += 1
                            attached = self.position_manager.attach_bracket_order_to_position(
                                hotkey, trade_pair.trade_pair_id, order.to_python_dict()
                            )
                            if attached:
                                total_bracket_attached += 1
                    else:
                        if hotkey not in self._closed_orders:
                            self._closed_orders[hotkey] = []
                        self._closed_orders[hotkey].append(order)
                    self._last_fill_time[trade_pair][hotkey] = 0

                except Exception as e:
                    bt.logging.error(f"Error reading limit order from disk: {e}")
                    continue

        # Sort orders by processed_ms for each (trade_pair, hotkey)
        for trade_pair in self._limit_orders:
            for hotkey in self._limit_orders[trade_pair]:
                self._limit_orders[trade_pair][hotkey].sort(key=lambda o: o.processed_ms)

        bt.logging.info(f"[LIMIT ORDER DISK] Finished reading limit orders: {total_orders_read} open orders, {total_bracket_orders} bracket orders, {total_bracket_attached} attached to positions")

    def _sync_pending_bracket_orders(self, miner_hotkey: str, position):
        """
        Sync pending bracket orders with a position.
        Sets the bracket order's order_type to match the position's position_type
        and attaches the bracket order to the position.

        Args:
            miner_hotkey: The miner's hotkey
            position: The position to sync bracket orders with
        """
        if not position:
            return

        trade_pair = position.trade_pair
        orders_to_sync = []

        # Collect bracket orders that need syncing for this position
        if trade_pair in self._limit_orders and miner_hotkey in self._limit_orders[trade_pair]:
            for order in self._limit_orders[trade_pair][miner_hotkey]:
                if order.src == OrderSource.BRACKET_UNFILLED:
                    orders_to_sync.append(order)

        now_ms = TimeUtil.now_in_millis()
        for order in orders_to_sync:
            # Check if already attached (idempotent)
            existing_uuids = {o.get('order_uuid') for o in position.unfilled_orders}
            if order.order_uuid in existing_uuids:
                continue

            # Validate bracket order now that we have a position
            # This handles bracket orders submitted on pending limit orders (no position at submission)
            try:
                self._validate_bracket_order(order, position, [], reference_price=position.initial_entry_price)
            except SignalException as e:
                bt.logging.warning(f"[BRACKET SYNC] Validation failed for {order.order_uuid}: {e}")
                self._close_limit_order(miner_hotkey, order, OrderSource.BRACKET_CANCELLED, now_ms)
                continue

            # Persist updated order to disk (order_type was set by _validate_bracket_order)
            self._write_to_disk(miner_hotkey, order)

            # Attach to position
            attached = self.position_manager.attach_bracket_order_to_position(
                miner_hotkey, trade_pair.trade_pair_id, order.to_python_dict()
            )
            if attached:
                bt.logging.info(
                    f"[BRACKET SYNC] Synced bracket order {order.order_uuid} "
                    f"with position {miner_hotkey}/{trade_pair.trade_pair_id} "
                    f"(order_type={order.order_type})"
                )

    def _write_to_disk(self, miner_hotkey, order):
        """Write order to disk."""
        if not order:
            return
        try:
            trade_pair_id = order.trade_pair.trade_pair_id
            if order.src in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                status = "unfilled"
            else:
                status = "closed"

            order_dir = ValiBkpUtils.get_limit_orders_dir(miner_hotkey, trade_pair_id, status, self.running_unit_tests)
            os.makedirs(order_dir, exist_ok=True)

            filepath = order_dir + order.order_uuid
            ValiBkpUtils.write_file(filepath, order)
        except Exception as e:
            bt.logging.error(f"Error writing limit order to disk: {e}")

    def _delete_from_disk(self, miner_hotkey, order):
        """Delete order file from disk (both unfilled and closed directories)."""
        if not order:
            return
        try:
            trade_pair_id = order.trade_pair.trade_pair_id
            order_uuid = order.order_uuid

            # Try both unfilled and closed directories
            for status in ["unfilled", "closed"]:
                order_dir = ValiBkpUtils.get_limit_orders_dir(miner_hotkey, trade_pair_id, status, self.running_unit_tests)
                filepath = order_dir + order_uuid

                if os.path.exists(filepath):
                    os.remove(filepath)
                    bt.logging.debug(f"Deleted limit order file: {filepath}")

        except Exception as e:
            bt.logging.error(f"Error deleting limit order from disk: {e}")

    def sync_limit_orders(self, sync_data):
        """Sync limit orders from external source."""
        if not sync_data:
            return

        for miner_hotkey, orders_data in sync_data.items():
            if not orders_data:
                continue

            try:
                for data in orders_data:
                    order = Order.from_dict(data)
                    self._write_to_disk(miner_hotkey, order)
            except Exception as e:
                bt.logging.error(f"Could not sync limit orders: {e}")

        self._read_limit_orders_from_disk()

    def clear_limit_orders(self):
        """
        Clear all limit orders from memory.

        This is primarily used for testing and development.
        Does NOT delete orders from disk.
        """
        self._limit_orders.clear()
        self._last_fill_time.clear()
        # Also clear market order manager's cooldown cache
        self.market_order_manager.clear_order_cooldown_cache()
        bt.logging.info("Cleared all limit orders from memory")
