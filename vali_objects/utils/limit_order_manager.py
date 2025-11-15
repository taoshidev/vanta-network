import json
import os
import time
import traceback
from multiprocessing import Process
from multiprocessing.managers import BaseManager

import bittensor as bt

from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, TradePair
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

    def __init__(self, position_manager, live_price_fetcher, market_order_manager,
                 shutdown_dict=None, running_unit_tests=False):
        super().__init__(running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.elimination_manager = position_manager.elimination_manager
        self.live_price_fetcher = live_price_fetcher
        self.market_order_manager = market_order_manager  # For filling orders
        self.shutdown_dict = shutdown_dict or {}
        self.running_unit_tests = running_unit_tests

        # Internal data structure: {TradePair: {hotkey: [Order]}}
        # Regular Python dict - NO IPC!
        self._limit_orders = {}

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
            use_ipc=False
        )

    # ============================================================================
    # RPC Methods (called from client)
    # ============================================================================

    def process_limit_order_rpc(self, miner_hotkey, order_dict):
        """
        RPC method to process a limit order.
        Args:
            miner_hotkey: The miner's hotkey
            order_dict: Order serialized as dict
        Returns:
            dict with status and order_uuid
        """
        order = Order.from_dict(order_dict)
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

            if miner_hotkey not in self._limit_orders[trade_pair]:
                self._limit_orders[trade_pair][miner_hotkey] = []

            # Check max unfilled orders for this miner across ALL trade pairs
            total_unfilled = self._count_unfilled_orders_for_hotkey(miner_hotkey)
            if total_unfilled >= ValiConfig.MAX_UNFILLED_LIMIT_ORDERS:
                raise SignalException(
                    f"miner has too many unfilled limit orders "
                    f"[{total_unfilled}] >= [{ValiConfig.MAX_UNFILLED_LIMIT_ORDERS}]"
                )

            # Don't need realtime position
            position = self._get_position_for(miner_hotkey, order)
            if not position and order.order_type == OrderType.FLAT:
                raise SignalException(f"No position found for FLAT order")

            self._write_to_disk(miner_hotkey, order)
            self._limit_orders[trade_pair][miner_hotkey].append(order)

            bt.logging.info(f"Saved [{miner_hotkey}] limit order [{order_uuid}]")

            # Check if order can be filled immediately
            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, order.processed_ms)
            if price_sources:
                trigger_price = self._evaluate_trigger_price(order.order_type, position, price_sources[0], order.limit_price)
                if trigger_price:
                    should_fill_immediately = True

        # Fill outside the lock to avoid reentrant lock issue
        if should_fill_immediately:
            self._fill_limit_order_with_price_source(miner_hotkey, order, price_sources[0], trigger_price)

        return {"status": "success", "order_uuid": order_uuid}


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
        try:
            trade_pair = TradePair.from_trade_pair_id(trade_pair_id)

            orders_to_cancel = []

            if order_uuid:
                # Cancel specific order
                orders_to_cancel = self._find_orders_to_cancel_by_uuid(miner_hotkey, order_uuid)
            else:
                # Cancel all unfilled orders for this trade pair
                orders_to_cancel = self._find_orders_to_cancel_by_trade_pair(miner_hotkey, trade_pair)

            if not orders_to_cancel:
                raise SignalException(
                    f"No unfilled limit orders found for {miner_hotkey} "
                    f"(uuid={order_uuid}, trade_pair={trade_pair_id})"
                )

            for order in orders_to_cancel:
                self._close_limit_order(miner_hotkey, order, OrderSource.ORDER_SRC_LIMIT_CANCELLED, now_ms)

            return {
                "status": "cancelled",
                "order_uuid": order_uuid if order_uuid else "all",
                "miner_hotkey": miner_hotkey,
                "trade_pair_id": trade_pair_id,
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
                            "src": order.src
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

                        # Clean up empty trade_pair entries
                        if not self._limit_orders[trade_pair]:
                            del self._limit_orders[trade_pair]

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

    def run_limit_order_daemon(self):
        """
        Daemon process that checks and attempts to fill limit orders every minute.
        """
        bt.logging.info("Limit order daemon started")

        while not self.shutdown_dict:
            try:
                self.check_and_fill_limit_orders()
                time.sleep(60)
            except Exception as e:
                bt.logging.error(f"Error in limit order daemon: {e}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)

        bt.logging.info("Limit order daemon shutting down")

    def check_and_fill_limit_orders(self):
        """
        Iterate through all trade pairs and attempt to fill unfilled limit orders.
        """
        now_ms = TimeUtil.now_in_millis()
        total_checked = 0
        total_filled = 0

        bt.logging.info(f"Checking limit orders across {len(self._limit_orders)} trade pairs")

        for trade_pair, hotkey_dict in self._limit_orders.items():
            # Check if market is open
            if not self.live_price_fetcher.is_market_open(trade_pair, now_ms):
                bt.logging.debug(f"Market closed for {trade_pair.trade_pair_id}, skipping")
                continue

            # Get price sources for this trade pair
            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, now_ms)
            if not price_sources:
                bt.logging.debug(f"No price sources for {trade_pair.trade_pair_id}, skipping")
                continue

            # Iterate through all hotkeys for this trade pair
            for miner_hotkey, orders in hotkey_dict.items():
                for order in orders:
                    if order.src != OrderSource.ORDER_SRC_LIMIT_UNFILLED:
                        continue

                    total_checked += 1

                    # Attempt to fill
                    if self._attempt_fill_limit_order(miner_hotkey, order, price_sources, now_ms):
                        total_filled += 1

        bt.logging.info(f"Limit order check complete: checked={total_checked}, filled={total_filled}")

    # ============================================================================
    # Internal Helper Methods
    # ============================================================================

    def _count_unfilled_orders_for_hotkey(self, miner_hotkey):
        """Count total unfilled orders across all trade pairs for a hotkey."""
        count = 0
        for trade_pair, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for order in hotkey_dict[miner_hotkey]:
                    if order.src == OrderSource.ORDER_SRC_LIMIT_UNFILLED:
                        count += 1
        return count

    def _find_orders_to_cancel_by_uuid(self, miner_hotkey, order_uuid):
        """Find orders to cancel by UUID across all trade pairs."""
        orders_to_cancel = []
        for trade_pair, hotkey_dict in self._limit_orders.items():
            if miner_hotkey in hotkey_dict:
                for order in hotkey_dict[miner_hotkey]:
                    if order.order_uuid == order_uuid and order.src == OrderSource.ORDER_SRC_LIMIT_UNFILLED:
                        orders_to_cancel.append(order)
        return orders_to_cancel

    def _find_orders_to_cancel_by_trade_pair(self, miner_hotkey, trade_pair):
        """Find all unfilled orders for a specific trade pair."""
        orders_to_cancel = []
        if trade_pair in self._limit_orders and miner_hotkey in self._limit_orders[trade_pair]:
            for order in self._limit_orders[trade_pair][miner_hotkey]:
                if order.src == OrderSource.ORDER_SRC_LIMIT_UNFILLED:
                    orders_to_cancel.append(order)
        return orders_to_cancel

    def _attempt_fill_limit_order(self, miner_hotkey, order, price_sources, now_ms):
        """
        Attempt to fill a limit order. Returns True if filled, False otherwise.

        Uses limit_order_locks to check trigger condition, then position_locks to fill.
        """
        trade_pair = order.trade_pair

        try:
            # Check if order should be filled (under limit_order_locks)
            with self.limit_order_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                # Verify order still unfilled
                if order.src != OrderSource.ORDER_SRC_LIMIT_UNFILLED:
                    return False

                # Check if limit price triggered
                best_price_source = price_sources[0]
                position = self._get_position_for(miner_hotkey, order)
                trigger_price = self._evaluate_trigger_price(
                    order.order_type,
                    position,
                    best_price_source,
                    order.limit_price
                )

                if trigger_price is None:
                    return False

            # Re-check order still unfilled (race condition protection)
            if order.src != OrderSource.ORDER_SRC_LIMIT_UNFILLED:
                return False

            # Fill the order using the triggered price_source
            self._fill_limit_order_with_price_source(miner_hotkey, order, best_price_source, trigger_price)
            return True

        except Exception as e:
            bt.logging.error(f"Error attempting to fill limit order {order.order_uuid}: {e}")
            bt.logging.error(traceback.format_exc())
            return False

    def _fill_limit_order_with_price_source(self, miner_hotkey, order, price_source, fill_price):
        """Fill a limit order and update position."""
        trade_pair = order.trade_pair
        fill_time = price_source.start_ms
        new_src = OrderSource.ORDER_SRC_LIMIT_FILLED

        try:
            err_msg, updated_position = self.market_order_manager._process_market_order(
                order.order_uuid,
                "limit_order",
                trade_pair,
                fill_time,
                Order.to_python_dict(order),
                miner_hotkey,
                [price_source]
            )

            # Issue 2: Check if err_msg is set - treat as failure
            if err_msg:
                raise ValueError(err_msg)

            # Issue 5: updated_position being None is an error case, not fallback
            if not updated_position:
                raise ValueError("No position returned from market order processing")

            # Issue 4: Copy values TO original order object rather than reassigning variable
            filled_order = updated_position.orders[-1]
            order.price_sources = filled_order.price_sources
            order.price = filled_order.price
            order.bid = filled_order.bid
            order.ask = filled_order.ask
            order.slippage = filled_order.slippage
            order.processed_ms = filled_order.processed_ms

            # Issue 3: Log success only after successful update
            bt.logging.success(f"Filled limit order {order.order_uuid} at {order.price}")

        except Exception as e:
            bt.logging.info(f"Could not fill limit order [{order.order_uuid}]: {e}. Cancelling order")
            new_src = OrderSource.ORDER_SRC_LIMIT_CANCELLED
        finally:
            self._close_limit_order(miner_hotkey, order, new_src, fill_time)

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

            # Update internal structure
            if trade_pair in self._limit_orders and miner_hotkey in self._limit_orders[trade_pair]:
                orders = self._limit_orders[trade_pair][miner_hotkey]
                for i, o in enumerate(orders):
                    if o.order_uuid == order_uuid:
                        orders[i] = order
                        break

            bt.logging.info(f"Successfully closed limit order [{order_uuid}] [{trade_pair_id}] for [{miner_hotkey}]")

    def _get_position_for(self, hotkey, order):
        """Get open position for hotkey and trade pair."""
        trade_pair_id = order.trade_pair.trade_pair_id
        return self.position_manager.get_open_position_for_a_miner_trade_pair(hotkey, trade_pair_id)

    def _evaluate_trigger_price(self, order_type, position, ps, limit_price):
        """Check if limit price is triggered."""
        bid_price = ps.bid if ps.bid > 0 else ps.open
        ask_price = ps.ask if ps.ask > 0 else ps.open

        position_type = position.position_type if position else None

        buy_type = order_type == OrderType.LONG or (order_type == OrderType.FLAT and position_type == OrderType.SHORT)
        sell_type = order_type == OrderType.SHORT or (order_type == OrderType.FLAT and position_type == OrderType.LONG)

        if buy_type:
            return ask_price if ask_price <= limit_price else None
        elif sell_type:
            return bid_price if bid_price >= limit_price else None
        else:
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

            miner_order_dicts = ValiBkpUtils.get_limit_orders(hotkey, self.running_unit_tests)
            for order_dict in miner_order_dicts:
                try:
                    order = Order.from_dict(order_dict)
                    trade_pair = order.trade_pair

                    # Initialize nested structure
                    if trade_pair not in self._limit_orders:
                        self._limit_orders[trade_pair] = {}
                    if hotkey not in self._limit_orders[trade_pair]:
                        self._limit_orders[trade_pair][hotkey] = []

                    self._limit_orders[trade_pair][hotkey].append(order)

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
            if order.src == OrderSource.ORDER_SRC_LIMIT_UNFILLED:
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


# ============================================================================
# RPC Client
# ============================================================================

class LimitOrderManagerClient:
    """
    RPC client for LimitOrderManager.

    PROCESS BOUNDARY: This client runs in the VALIDATOR process.
    All methods make RPC calls to the LimitOrderManager server process.

    Design Principles:
    - Client handles NO business logic - just RPC wrapper
    - Client passes only serializable data (no synapse objects)
    - Exceptions from server are pickled and re-raised in validator
    - Validator owns all protocol/synapse handling logic
    """

    class ClientManager(BaseManager):
        pass

    ClientManager.register('LimitOrderManager')

    def __init__(self, position_manager, live_price_fetcher, market_order_manager,
                 shutdown_dict=None, running_unit_tests=False,
                 address=('localhost', 50001)):
        """
        Initialize client and start the server process.

        Args:
            position_manager: PositionManager instance
            live_price_fetcher: LivePriceFetcher instance
            market_order_manager: MarketOrderManager instance (provides position_locks)
            shutdown_dict: Shutdown dictionary
            running_unit_tests: Whether running unit tests
            address: (host, port) for RPC server

        Note: uuid_tracker is NOT passed here because this runs in a separate process.
              All UUID tracking must happen in the validator process.
        """
        self.address = address
        self.running_unit_tests = running_unit_tests
        self.shutdown_dict = shutdown_dict or {}
        self.market_order_manager = market_order_manager

        if not running_unit_tests:
            # Start server process
            self.server_process = Process(
                target=self._start_server,
                args=(position_manager, live_price_fetcher, market_order_manager, shutdown_dict, address),
                daemon=True
            )
            self.server_process.start()
            time.sleep(1)  # Give server time to start

            # Connect client
            self.manager = self.ClientManager(address=address, authkey=b'limit_order_manager')
            self.manager.connect()
            self.limit_order_manager = self.manager.LimitOrderManager()

            bt.logging.info(f"LimitOrderManagerClient connected to server at {address}")
        else:
            # For unit tests, use direct instance
            self.limit_order_manager = LimitOrderManager(
                position_manager, live_price_fetcher, market_order_manager,
                shutdown_dict, running_unit_tests
            )

    @staticmethod
    def _start_server(position_manager, live_price_fetcher, market_order_manager, shutdown_dict, address):
        """Start the RPC server and daemon in a separate process."""
        import threading

        class ServerManager(BaseManager):
            pass

        # Create manager instance
        manager = LimitOrderManager(position_manager, live_price_fetcher, market_order_manager,
                                    shutdown_dict, False)

        # Start daemon thread in this process
        daemon_thread = threading.Thread(target=manager.run_limit_order_daemon, daemon=True)
        daemon_thread.start()
        bt.logging.info("Limit order daemon thread started in server process")

        # Register with server
        ServerManager.register('LimitOrderManager', callable=lambda: manager)

        # Start server (blocks forever)
        server = ServerManager(address=address, authkey=b'limit_order_manager')
        bt.logging.info(f"LimitOrderManager RPC server starting at {address}")
        server.get_server().serve_forever()

    # ============================================================================
    # Client API Methods
    # ============================================================================

    def process_limit_order(self, miner_hotkey: str, limit_order: Order) -> dict:
        """
        Process a limit order via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            limit_order: Order object to save

        Returns:
            dict with status and order_uuid

        Raises:
            SignalException: Validation errors (pickled from server)
            Exception: RPC or server errors
        """
        order_dict = limit_order.to_python_dict()
        return self.limit_order_manager.process_limit_order_rpc(miner_hotkey, order_dict)

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
        return self.limit_order_manager.cancel_limit_order_rpc(miner_hotkey, trade_pair_id, order_uuid, now_ms)

    def get_all_limit_orders(self) -> dict:
        """
        Get all limit orders via RPC.

        Returns:
            Dict of {trade_pair_id: {hotkey: [order_dicts]}}
        """
        return self.limit_order_manager.get_all_limit_orders_rpc()

    def get_limit_orders(self, miner_hotkey: str) -> list:
        """
        Get all limit orders for a hotkey via RPC.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            List of order dicts
        """
        return self.limit_order_manager.get_limit_orders_for_hotkey_rpc(miner_hotkey)

    def get_limit_orders_for_trade_pair(self, trade_pair_id: str) -> dict:
        """
        Get all limit orders for a trade pair via RPC.

        Args:
            trade_pair_id: Trade pair ID string

        Returns:
            Dict of {hotkey: [order_dicts]}
        """
        return self.limit_order_manager.get_limit_orders_for_trade_pair_rpc(trade_pair_id)

    def to_dashboard_dict(self, miner_hotkey: str):
        """
        Get dashboard representation via RPC.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            List of order data for dashboard or None
        """
        return self.limit_order_manager.to_dashboard_dict_rpc(miner_hotkey)

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
        return self.limit_order_manager.delete_all_limit_orders_for_hotkey_rpc(miner_hotkey)

