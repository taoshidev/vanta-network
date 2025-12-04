import os
import traceback

import bittensor as bt

from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.order import OrderSource, Order


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
        from vali_objects.utils.market_order_manager import MarketOrderManager
        self._market_order_manager = MarketOrderManager(
            serve=serve,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )
        # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
        from vali_objects.utils.live_price_server import LivePriceFetcherClient
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests,
                                                         connection_mode=connection_mode)

        # Create own RPC clients (forward compatibility - no parameter passing)
        from vali_objects.utils.position_manager_client import PositionManagerClient
        from vali_objects.utils.elimination_client import EliminationClient
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
        self._last_fill_time = {}

        self._read_limit_orders_from_disk()
        self._reset_counters()

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

    def process_limit_order(self, miner_hotkey, order):
        """
        RPC method to process a limit order or bracket order.
        Args:
            miner_hotkey: The miner's hotkey
            order: Order object (pickled automatically by RPC)
        Returns:
            dict with status and order_uuid
        """
        trade_pair = order.trade_pair

        # Variables to track whether to fill immediately
        should_fill_immediately = False
        trigger_price = None
        price_sources = None

        with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
            order_uuid = order.order_uuid
            # Ensure trade_pair exists in structure
            if trade_pair not in self._limit_orders:
                self._limit_orders[trade_pair] = {}
                self._last_fill_time[trade_pair] = {}

            if miner_hotkey not in self._limit_orders[trade_pair]:
                self._limit_orders[trade_pair][miner_hotkey] = []
                self._last_fill_time[trade_pair][miner_hotkey] = 0

            # Check max unfilled orders for this miner across ALL trade pairs
            total_unfilled = self._count_unfilled_orders_for_hotkey(miner_hotkey)
            if total_unfilled >= ValiConfig.MAX_UNFILLED_LIMIT_ORDERS:
                raise SignalException(
                    f"miner has too many unfilled limit orders "
                    f"[{total_unfilled}] >= [{ValiConfig.MAX_UNFILLED_LIMIT_ORDERS}]"
                )

            # Get position for validation
            position = self._get_position_for(miner_hotkey, order)

            # Special handling for BRACKET orders
            if order.execution_type == ExecutionType.BRACKET:
                if not position:
                    raise SignalException(
                        f"Cannot create bracket order: no open position found for {trade_pair.trade_pair_id}"
                    )

                # Validate that at least one of SL or TP is set
                if order.stop_loss is None and order.take_profit is None:
                    raise SignalException(
                        f"BRACKET orders must have at least one of stop_loss or take_profit set"
                    )

                order.order_type = position.position_type

                # Use miner-provided leverage if specified, otherwise use position leverage
                if order.leverage is None and order.value is None and order.quantity is None:
                    order.leverage = position.net_leverage
                    order.value = position.net_value
                    order.quantity = position.net_quantity

            # Validation for LIMIT orders
            if order.execution_type == ExecutionType.LIMIT:
                if order.limit_price is None or order.limit_price <= 0:
                    raise SignalException(
                        f"LIMIT orders must have a valid limit_price > 0 (got {order.limit_price})"
                    )

            # Validation for FLAT orders
            if order.order_type == OrderType.FLAT:
                raise SignalException(f"FLAT order is not supported for LIMIT orders")

            if order.execution_type == ExecutionType.BRACKET:
                bt.logging.info(
                    f"INCOMING BRACKET ORDER | {trade_pair.trade_pair_id} | "
                    f"{order.order_type.name} | SL={order.stop_loss} TP={order.take_profit}"
                )
            else:
                bt.logging.info(
                    f"INCOMING LIMIT ORDER | {trade_pair.trade_pair_id} | "
                    f"{order.order_type.name} @ {order.limit_price}"
                )

            self._write_to_disk(miner_hotkey, order)
            self._limit_orders[trade_pair][miner_hotkey].append(order)

            # Check if order can be filled immediately
            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, order.processed_ms)
            if price_sources:
                trigger_price = self._evaluate_trigger_price(order, position, price_sources[0])

                if trigger_price:
                    should_fill_immediately = True

        # Fill outside the lock to avoid reentrant lock issue
        if should_fill_immediately:
            fill_error = self._fill_limit_order_with_price_source(miner_hotkey, order, price_sources[0], None, enforce_market_cooldown=True)
            if fill_error:
                raise SignalException(fill_error)

            bt.logging.info(f"Filled order {order_uuid} @ market price {price_sources[0].close}")

        return {"status": "success", "order_uuid": order_uuid}


    def cancel_limit_order(self, miner_hotkey, trade_pair_id, order_uuid, now_ms):
        """
        RPC method to cancel limit order(s).
        Args:
            miner_hotkey: The miner's hotkey
            order_uuid: UUID of specific order to cancel, or None/empty for all
            now_ms: Current timestamp
        Returns:
            dict with cancellation details
        """
        # TODO support cancel by trade pair in v2
        try:
            # Parse trade_pair only if trade_pair_id is provided
            # trade_pair = TradePair.from_trade_pair_id(trade_pair_id) if trade_pair_id else None

            # Try to find orders by UUID first
            orders_to_cancel = self._find_orders_to_cancel_by_uuid(miner_hotkey, order_uuid)

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

    def to_dashboard_dict_rpc(self, miner_hotkey):
        """
        RPC method to get dashboard representation of limit orders.
        """
        try:
            order_list = []
            for trade_pair, hotkey_dict in self._limit_orders.items():
                if miner_hotkey in hotkey_dict:
                    for order in hotkey_dict[miner_hotkey]:
                        data = {
                            "trade_pair": [order.trade_pair.trade_pair_id, order.trade_pair.trade_pair],
                            "order_type": str(order.order_type),
                            "processed_ms": order.processed_ms,
                            "limit_price": order.limit_price,
                            "price": order.price,
                            "leverage": order.leverage,
                            'value': order.value,
                            'quantity': order.quantity,
                            "src": order.src,
                            "execution_type": order.execution_type.name,
                            "order_uuid": order.order_uuid,
                            "stop_loss": order.stop_loss,
                            "take_profit": order.take_profit
                        }
                        order_list.append(data)
            return order_list if order_list else None
        except Exception as e:
            bt.logging.error(f"Error creating dashboard dict: {e}")
            return None

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

        bt.logging.info(f"Checking limit orders across {len(self._limit_orders)} trade pairs")

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
                    if self.running_unit_tests:
                        print(f"[CHECK_ORDERS DEBUG] Fill interval not met: {time_since_last_fill}ms < {ValiConfig.LIMIT_ORDER_FILL_INTERVAL_MS}ms")
                    bt.logging.debug(f"Skipping {trade_pair.trade_pair_id} for {miner_hotkey}: {time_since_last_fill}ms since last fill")
                    continue

                if self.running_unit_tests:
                    print(f"[CHECK_ORDERS DEBUG] Checking {len(orders)} orders for {miner_hotkey}")

                for order in orders:
                    # Check both regular limit orders and SL/TP Bracket orders
                    if order.src not in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                        if self.running_unit_tests:
                            print(f"[CHECK_ORDERS DEBUG] Skipping order {order.order_uuid} with src={order.src}")
                        continue

                    if self.running_unit_tests:
                        print(f"[CHECK_ORDERS DEBUG] Attempting to fill order {order.order_uuid} type={order.execution_type}")

                    total_checked += 1

                    # Attempt to fill
                    if self._attempt_fill_limit_order(miner_hotkey, order, price_sources, now_ms):
                        total_filled += 1
                        # DESIGN: Break after first fill to enforce LIMIT_ORDER_FILL_INTERVAL_MS
                        # Only one order per trade pair per hotkey can fill within the interval.
                        # This prevents rapid sequential fills and enforces rate limiting.
                        break

        bt.logging.info(f"Limit order check complete: checked={total_checked}, filled={total_filled}")

        return {
            'checked': total_checked,
            'filled': total_filled,
            'timestamp_ms': now_ms
        }

    # ============================================================================
    # Internal Helper Methods
    # ============================================================================

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
                position = self._get_position_for(miner_hotkey, order)
                trigger_price = self._evaluate_trigger_price(order, position, best_price_source)

                if self.running_unit_tests and order.execution_type == ExecutionType.BRACKET:
                    print(f"[BRACKET DEBUG] position={position is not None}, trigger_price={trigger_price}, ps.bid={best_price_source.bid}, ps.ask={best_price_source.ask}, order={order.order_uuid}")

                if trigger_price is not None:
                    should_fill = True

            if order.execution_type == ExecutionType.BRACKET and not position:
                print(f"[BRACKET CANCELLED] No position found for bracket order {order.order_uuid}, cancelling")
                self._close_limit_order(miner_hotkey, order, OrderSource.BRACKET_CANCELLED, now_ms)
                return False

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
                    order_dict['leverage'] = abs(order.leverage) if order.leverage else None
                    order_dict['value'] = abs(order.value) if order.value else None
                    order_dict['quantity'] = abs(order.quantity) if order.quantity else None
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

            if order.execution_type == ExecutionType.LIMIT and (order.stop_loss is not None or order.take_profit is not None):
                self._create_sltp_orders(miner_hotkey, order)

        except Exception as e:
            error_msg = f"Could not fill limit order [{order.order_uuid}]: {e}. Cancelling order"
            bt.logging.info(error_msg)
            new_src = OrderSource.get_cancel(order.src)

        finally:
            self._close_limit_order(miner_hotkey, order, new_src, fill_time)

        return error_msg

    def _close_limit_order(self, miner_hotkey, order, src, time_ms):
        """Mark order as closed and update disk."""
        order_uuid = order.order_uuid
        trade_pair = order.trade_pair
        trade_pair_id = trade_pair.trade_pair_id
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

            bt.logging.info(f"Successfully closed limit order [{order_uuid}] [{trade_pair_id}] for [{miner_hotkey}]")

    def _create_sltp_orders(self, miner_hotkey, parent_order):
        """
        Create a single bracket order with both stop loss and take profit.
        Replaces the previous two-order SLTP system.

        DESIGN: Bracket order UUID format is "{parent_uuid}-bracket"
        This allows miners to cancel the bracket order by providing the parent order UUID.
        See _find_orders_to_cancel_by_uuid() for the cancellation logic.
        """
        trade_pair = parent_order.trade_pair
        now_ms = TimeUtil.now_in_millis()

        # Require at least one of SL or TP to be set
        if parent_order.stop_loss is None and parent_order.take_profit is None:
            bt.logging.debug(f"No SL/TP specified for order [{parent_order.order_uuid}], skipping bracket creation")
            return

        # Validate SL/TP against fill price before creating bracket order
        fill_price = parent_order.price
        order_type = parent_order.order_type

        # Validate stop loss and take profit based on order type
        if order_type == OrderType.LONG:
            # For LONG positions:
            # - Stop loss must be BELOW fill price (selling at a loss)
            # - Take profit must be ABOVE fill price (selling at a gain)
            if parent_order.stop_loss is not None and parent_order.stop_loss >= fill_price:
                bt.logging.warning(
                    f"Invalid LONG bracket order [{parent_order.order_uuid}]: "
                    f"stop_loss ({parent_order.stop_loss}) must be < fill_price ({fill_price}). "
                    f"Skipping bracket creation"
                )
                return

            if parent_order.take_profit is not None and parent_order.take_profit <= fill_price:
                bt.logging.warning(
                    f"Invalid LONG bracket order [{parent_order.order_uuid}]: "
                    f"take_profit ({parent_order.take_profit}) must be > fill_price ({fill_price}). "
                    f"Skipping bracket creation"
                )
                return

        elif order_type == OrderType.SHORT:
            # For SHORT positions:
            # - Stop loss must be ABOVE fill price (buying back at a loss)
            # - Take profit must be BELOW fill price (buying back at a gain)
            if parent_order.stop_loss is not None and parent_order.stop_loss <= fill_price:
                bt.logging.warning(
                    f"Invalid SHORT bracket order [{parent_order.order_uuid}]: "
                    f"stop_loss ({parent_order.stop_loss}) must be > fill_price ({fill_price}). "
                    f"Skipping bracket creation"
                )
                return

            if parent_order.take_profit is not None and parent_order.take_profit >= fill_price:
                bt.logging.warning(
                    f"Invalid SHORT bracket order [{parent_order.order_uuid}]: "
                    f"take_profit ({parent_order.take_profit}) must be < fill_price ({fill_price}). "
                    f"Skipping bracket creation"
                )
                return
        else:
            bt.logging.error(
                f"Invalid order type for bracket order [{parent_order.order_uuid}]: {order_type}. "
                f"Must be LONG or SHORT"
            )
            return

        try:
            # Create single bracket order with both SL and TP
            # UUID format: "{parent_uuid}-bracket" enables cancellation via parent UUID
            bracket_order = Order(
                trade_pair=trade_pair,
                order_uuid=f"{parent_order.order_uuid}-bracket",
                processed_ms=now_ms,
                price=0.0,
                order_type=parent_order.order_type,
                leverage=None,
                value=None,
                quantity=parent_order.quantity,  # Unify to quantity
                execution_type=ExecutionType.BRACKET,
                limit_price=None,  # Not used for bracket orders
                stop_loss=parent_order.stop_loss,
                take_profit=parent_order.take_profit,
                src=OrderSource.BRACKET_UNFILLED
            )

            with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                if trade_pair not in self._limit_orders:
                    self._limit_orders[trade_pair] = {}
                    self._last_fill_time[trade_pair] = {}
                if miner_hotkey not in self._limit_orders[trade_pair]:
                    self._limit_orders[trade_pair][miner_hotkey] = []
                    self._last_fill_time[trade_pair][miner_hotkey] = 0

                self._write_to_disk(miner_hotkey, bracket_order)
                self._limit_orders[trade_pair][miner_hotkey].append(bracket_order)

                bt.logging.success(
                    f"Created bracket order [{bracket_order.order_uuid}] "
                    f"with SL={parent_order.stop_loss}, TP={parent_order.take_profit}"
                )

        except Exception as e:
            bt.logging.error(f"Error creating bracket order: {e}")
            bt.logging.error(traceback.format_exc())

    def _get_position_for(self, hotkey, order):
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

        position_type = position.position_type if position else None

        buy_type = order_type == OrderType.LONG or (order_type == OrderType.FLAT and position_type == OrderType.SHORT)
        sell_type = order_type == OrderType.SHORT or (order_type == OrderType.FLAT and position_type == OrderType.LONG)

        if buy_type:
            return limit_price if ask_price <= limit_price else None
        elif sell_type:
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
        bid_price = ps.bid if ps.bid > 0 else ps.open
        ask_price = ps.ask if ps.ask > 0 else ps.open

        order_type = order.order_type

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

        for hotkey in hotkeys:
            if hotkey in eliminated_hotkeys:
                continue

            miner_order_dicts = ValiBkpUtils.get_limit_orders(hotkey, True, running_unit_tests=self.running_unit_tests)
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

                    self._limit_orders[trade_pair][hotkey].append(order)
                    self._last_fill_time[trade_pair][hotkey] = 0

                except Exception as e:
                    bt.logging.error(f"Error reading limit order from disk: {e}")
                    continue

        # Sort orders by processed_ms for each (trade_pair, hotkey)
        for trade_pair in self._limit_orders:
            for hotkey in self._limit_orders[trade_pair]:
                self._limit_orders[trade_pair][hotkey].sort(key=lambda o: o.processed_ms)

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

    def _reset_counters(self):
        """Reset evaluation counters."""
        self._limit_orders_evaluated = 0
        self._limit_orders_filled = 0

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
