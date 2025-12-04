import unittest
from unittest.mock import Mock, patch
from copy import deepcopy
import threading
import time

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position import Position
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestLimitOrders(TestBase):
    """
    Integration tests for limit order management using server/client architecture.
    Uses class-level server setup for efficiency - servers start once and are shared.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    limit_order_client = None
    limit_order_handle = None  # Keep handle for direct access to server in tests

    # Class-level constants
    DEFAULT_MINER_HOTKEY = "test_miner"

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.limit_order_client = cls.orchestrator.get_client('limit_order')

        # Get limit order server handle for direct access in tests
        cls.limit_order_handle = cls.orchestrator._servers.get('limit_order')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # Clear all data for test isolation (includes price sources and market open)
        self.orchestrator.clear_all_test_data()

        # Set up metagraph with test miner
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY])

        # Create fresh test data
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def create_test_limit_order(self, order_type: OrderType = OrderType.LONG, limit_price=49000.0,
                               trade_pair=None, leverage=0.5, order_uuid=None,
                               stop_loss=None, take_profit=None, quantity=None):
        """Helper to create test limit orders"""
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if order_uuid is None:
            order_uuid = f"test_limit_order_{TimeUtil.now_in_millis()}"

        return Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=order_type,
            leverage=leverage,
            quantity=quantity,
            execution_type=ExecutionType.LIMIT,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            src=OrderSource.LIMIT_UNFILLED
        )

    def create_test_price_source(self, price, bid=None, ask=None, start_ms=None):
        """Helper to create a single price source"""
        if start_ms is None:
            start_ms = TimeUtil.now_in_millis()
        if bid is None:
            bid = price
        if ask is None:
            ask = price

        return PriceSource(
            source='test',
            timespan_ms=0,
            open=price,
            close=price,
            vwap=None,
            high=price,
            low=price,
            start_ms=start_ms,
            websocket=True,
            lag_ms=100,
            bid=bid,
            ask=ask
        )

    def create_test_position(self, trade_pair=None, miner_hotkey=None, position_type=None):
        """Helper to create test positions"""
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if miner_hotkey is None:
            miner_hotkey = self.DEFAULT_MINER_HOTKEY

        position = Position(
            miner_hotkey=miner_hotkey,
            position_uuid=f"pos_{TimeUtil.now_in_millis()}",
            open_ms=TimeUtil.now_in_millis(),
            trade_pair=trade_pair,
            account_size=1000.0  # Required for position validation
        )
        if position_type:
            position.position_type = position_type
        return position

    def get_orders_from_server(self, miner_hotkey, trade_pair):
        """Helper to get orders from server via client"""
        orders_for_trade_pair = self.limit_order_client.get_limit_orders_for_trade_pair(trade_pair.trade_pair_id)
        if miner_hotkey in orders_for_trade_pair:
            # Convert dicts back to Order objects for compatibility
            from vali_objects.vali_dataclasses.order import Order
            return [Order.from_dict(o) if isinstance(o, dict) else o for o in orders_for_trade_pair[miner_hotkey]]
        return []

    def count_orders_in_server(self, miner_hotkey):
        """Helper to count all orders for a hotkey across all trade pairs"""
        orders = self.limit_order_client.get_limit_orders(miner_hotkey)
        return len(orders)

    # ============================================================================
    # Test RPC Methods: process_limit_order_rpc
    # ============================================================================

    def test_process_limit_order_rpc_basic(self):
        """Test basic limit order placement via RPC"""
        limit_order = self.create_test_limit_order()

        result = self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            limit_order
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["order_uuid"], limit_order.order_uuid)

        # Verify stored in server
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_uuid, limit_order.order_uuid)
        self.assertEqual(orders[0].src, OrderSource.LIMIT_UNFILLED)

    def test_process_limit_order_rpc_exceeds_maximum(self):
        """Test limit order rejection when exceeding maximum unfilled orders"""
        # Fill up to the maximum
        for i in range(ValiConfig.MAX_UNFILLED_LIMIT_ORDERS):
            limit_order = self.create_test_limit_order(
                order_uuid=f"test_order_{i}",
                trade_pair=TradePair.BTCUSD if i % 2 == 0 else TradePair.ETHUSD
            )
            self.limit_order_client.process_limit_order(
                self.DEFAULT_MINER_HOTKEY,
                limit_order
            )

        # Attempt to add one more
        excess_order = self.create_test_limit_order(order_uuid="excess_order")

        with self.assertRaises(SignalException) as context:
            self.limit_order_client.process_limit_order(
                self.DEFAULT_MINER_HOTKEY,
                excess_order
            )
        self.assertIn("too many unfilled limit orders", str(context.exception))

    def test_process_limit_order_rpc_flat_no_position(self):
        """Test FLAT limit order rejection when no position exists"""
        flat_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=51000.0
        )

        with self.assertRaises(SignalException) as context:
            self.limit_order_client.process_limit_order(
                self.DEFAULT_MINER_HOTKEY,
                flat_order
            )
        self.assertIn("FLAT order is not supported for LIMIT orders", str(context.exception))

    def test_process_limit_order_rpc_flat_with_position(self):
        """Test FLAT limit order rejection even when position exists"""
        position = self.create_test_position(position_type=OrderType.LONG)
        self.position_client.save_miner_position(position)

        flat_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=51000.0
        )

        with self.assertRaises(SignalException) as context:
            self.limit_order_client.process_limit_order(
                self.DEFAULT_MINER_HOTKEY,
                flat_order
            )
        self.assertIn("FLAT order is not supported for LIMIT orders", str(context.exception))

    def test_process_limit_order_rpc_immediate_fill(self):
        """Test limit order is filled immediately when price already triggered"""
        # Setup position for the order
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        # Set test price via IPC (replaces patch/mock approach)
        trigger_price_source = self.create_test_price_source(48500.0, bid=48500.0, ask=48500.0)
        self.live_price_fetcher_client.set_test_price_source(TradePair.BTCUSD, trigger_price_source)

        # Create LONG order with limit price 49000 - should trigger at ask=48500
        limit_order = self.create_test_limit_order(
            order_type=OrderType.LONG,
            limit_price=49000.0
        )

        result = self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            limit_order
        )

        self.assertEqual(result["status"], "success")

        # Verify order was filled by checking position was updated
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        # Should have original position plus the new fill creates a new position
        # (or updates existing depending on logic)
        self.assertGreaterEqual(len(positions), 1, "At least one position should exist after fill")

    def test_process_limit_order_multiple_trade_pairs(self):
        """Test storing limit orders across multiple trade pairs"""
        btc_order = self.create_test_limit_order(
            trade_pair=TradePair.BTCUSD,
            order_uuid="btc_order"
        )
        eth_order = self.create_test_limit_order(
            trade_pair=TradePair.ETHUSD,
            order_uuid="eth_order"
        )

        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            btc_order
        )
        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            eth_order
        )

        # Verify structure
        btc_orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, TradePair.BTCUSD)
        eth_orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, TradePair.ETHUSD)

        self.assertEqual(len(btc_orders), 1)
        self.assertEqual(len(eth_orders), 1)
        self.assertEqual(btc_orders[0].order_uuid, "btc_order")
        self.assertEqual(eth_orders[0].order_uuid, "eth_order")

    # ============================================================================
    # Test RPC Methods: cancel_limit_order_rpc
    # ============================================================================

    def test_cancel_limit_order_rpc_specific_order(self):
        """Test cancelling a specific limit order by UUID"""
        order1 = self.create_test_limit_order(order_uuid="order1")
        order2 = self.create_test_limit_order(order_uuid="order2")

        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order1
        )
        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order2
        )

        # Cancel order1
        result = self.limit_order_client.cancel_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "order1",
            TimeUtil.now_in_millis()
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["num_cancelled"], 1)

        # Verify order1 removed from memory (Issue 8 fix), order2 still unfilled
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)

        # order1 should be removed
        order1_exists = any(o.order_uuid == "order1" for o in orders)
        self.assertFalse(order1_exists, "Cancelled order should be removed from memory")

        # order2 should still be unfilled
        order2_in_list = next((o for o in orders if o.order_uuid == "order2"), None)
        self.assertIsNotNone(order2_in_list)
        self.assertEqual(order2_in_list.src, OrderSource.LIMIT_UNFILLED)

    # TODO support cancel by trade pair in v2
    # def test_cancel_limit_order_rpc_all_for_trade_pair(self):
    #     """Test cancelling all limit orders for a trade pair"""
    #     for i in range(3):
    #         order = self.create_test_limit_order(order_uuid=f"order{i}")
    #         self.limit_order_client.process_limit_order(
    #             self.DEFAULT_MINER_HOTKEY,
    #             order
    #         )

    #     # Cancel all (empty order_uuid)
    #     result = self.limit_order_client.cancel_limit_order(
    #         self.DEFAULT_MINER_HOTKEY,
    #         self.DEFAULT_TRADE_PAIR.trade_pair_id,
    #         "",
    #         TimeUtil.now_in_millis()
    #     )

    #     self.assertEqual(result["status"], "cancelled")
    #     self.assertEqual(result["num_cancelled"], 3)

    #     # Verify all cancelled orders removed from memory (Issue 8 fix)
    #     orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
    #     self.assertEqual(len(orders), 0, "All cancelled orders should be removed from memory")

    def test_cancel_limit_order_rpc_nonexistent(self):
        """Test cancelling non-existent order raises exception"""
        with self.assertRaises(SignalException) as context:
            self.limit_order_client.cancel_limit_order(
                self.DEFAULT_MINER_HOTKEY,
                self.DEFAULT_TRADE_PAIR.trade_pair_id,
                "nonexistent_uuid",
                TimeUtil.now_in_millis()
            )
        self.assertIn("No unfilled limit orders found", str(context.exception))

    # ============================================================================
    # Test RPC Methods: delete_all_limit_orders_for_hotkey_rpc
    # ============================================================================

    def test_delete_all_limit_orders_for_hotkey_rpc(self):
        """Test deleting all limit orders for eliminated miner"""
        # Create orders across multiple trade pairs
        btc_order = self.create_test_limit_order(trade_pair=TradePair.BTCUSD, order_uuid="btc1")
        eth_order = self.create_test_limit_order(trade_pair=TradePair.ETHUSD, order_uuid="eth1")

        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            btc_order
        )
        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            eth_order
        )

        # Delete all
        result = self.limit_order_client.delete_all_limit_orders_for_hotkey(
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["deleted_count"], 2)

        # Verify all deleted from memory
        total_orders = self.count_orders_in_server(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(total_orders, 0)

    def test_delete_all_limit_orders_multiple_miners(self):
        """Test deletion only affects target miner"""
        miner2 = "miner2"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, miner2])

        order1 = self.create_test_limit_order(order_uuid="miner1_order")
        order2 = self.create_test_limit_order(order_uuid="miner2_order")

        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order1
        )
        self.limit_order_client.process_limit_order(
            miner2,
            order2
        )

        # Delete only miner1
        result = self.limit_order_client.delete_all_limit_orders_for_hotkey(
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertEqual(result["deleted_count"], 1)

        # Verify miner2's orders still exist
        miner2_orders = self.get_orders_from_server(miner2, self.DEFAULT_TRADE_PAIR)
        miner1_orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)

        self.assertEqual(len(miner2_orders), 1)
        self.assertEqual(len(miner1_orders), 0)

    # ============================================================================
    # Test Trigger Price Evaluation
    # ============================================================================

    def test_evaluate_trigger_price_long_order(self):
        """Test LONG order trigger evaluation"""
        # LONG order: triggers when ask <= limit_price
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # ask=50100 > limit=50000 -> no trigger
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # ask=50000 = limit=50000 -> trigger at ask
        price_source.ask = 50000.0
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

        # ask=49900 < limit=50000 -> trigger at limit_price
        price_source.ask = 49900.0
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

    def test_evaluate_trigger_price_short_order(self):
        """Test SHORT order trigger evaluation"""
        # SHORT order: triggers when bid >= limit_price
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # bid=49900 < limit=50000 -> no trigger
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.SHORT,
            None,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # bid=50000 = limit=50000 -> trigger at bid
        price_source.bid = 50000.0
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.SHORT,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

        # bid=50100 > limit=50000 -> trigger at limit_price
        price_source.bid = 50100.0
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.SHORT,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

    def test_evaluate_trigger_price_flat_long_position(self):
        """Test FLAT order trigger for LONG position (sells at bid)"""
        position = self.create_test_position(position_type=OrderType.LONG)

        # FLAT for LONG position: triggers when bid >= limit_price (selling)
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # bid=49900 < limit=50000 -> no trigger
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # bid=50100 > limit=50000 -> trigger at limit_price
        price_source.bid = 50100.0
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

    def test_evaluate_trigger_price_flat_short_position(self):
        """Test FLAT order trigger for SHORT position (buys at ask)"""
        position = self.create_test_position(position_type=OrderType.SHORT)

        # FLAT for SHORT position: triggers when ask <= limit_price (buying)
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # ask=50100 > limit=50000 -> no trigger
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # ask=49900 < limit=50000 -> trigger at limit_price
        price_source.ask = 49900.0
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

    def test_evaluate_trigger_price_fallback_to_open(self):
        """Test fallback to open price when bid/ask is 0"""
        price_source = self.create_test_price_source(50000.0, bid=0, ask=0)

        # LONG uses ask (0) -> falls back to open=50000
        trigger = self.limit_order_client.evaluate_limit_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50100.0
        )
        self.assertEqual(trigger, 50100.0)  # Returns limit_price when triggered (open <= limit)

    # ============================================================================
    # Test Fill Logic with Market Order Manager Integration
    # ============================================================================

    def test_fill_limit_order_success(self):
        """Test successful limit order fill creates position via market_order_manager"""
        order = self.create_test_limit_order(limit_price=50000.0)
        price_source = self.create_test_price_source(49000.0, bid=49000.0, ask=49000.0)

        # Setup initial position
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        # Explicitly inject None to prevent immediate fill during order processing
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)

        # Store order first via RPC (no price source available, so it won't trigger immediately)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Now register test price source for manual fill
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, price_source)

        # Fill it manually
        # For LONG order with limit_price=50000 and ask=49000, the order fills at limit_price (50000)
        # This is because when ask <= limit_price, the order gets the limit price
        self.limit_order_client.fill_limit_order_with_price_source(
            self.DEFAULT_MINER_HOTKEY,
            order,
            price_source,
            50000.0  # Fill at limit price, not ask price
        )

        # Verify filled order removed from memory (Issue 8 fix)
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 0, "Filled orders should be removed from memory")

        # Verify position was created/updated
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertGreaterEqual(len(positions), 1, "Position should exist after fill")

        # Verify the filled order has correct attributes
        position = positions[0]
        self.assertEqual(len(position.orders), 1, f"Position should have exactly one order")
        filled_order = position.orders[0]  # The filled limit order
        # The filled order should have exact values from the fill
        # For a LONG limit order, when ask <= limit_price, the order fills at limit_price (50000)
        self.assertEqual(filled_order.price, 50000.0, "Filled order should have correct price (limit price)")
        self.assertIsNotNone(filled_order.slippage, "Filled order should have slippage calculated")
        self.assertGreaterEqual(filled_order.slippage, 0, "Filled order slippage should be >= 0")
        self.assertEqual(filled_order.src, OrderSource.LIMIT_FILLED, "Order should be marked as LIMIT_FILLED")

    def test_fill_limit_order_error_cancels(self):
        """Test limit order is cancelled when fill fails due to missing position"""
        order = self.create_test_limit_order(limit_price=50000.0)
        fill_price_source = self.create_test_price_source(49000.0)

        # Set a non-triggering price to prevent immediate fill (ask > limit_price)
        non_trigger_price_source = self.create_test_price_source(51000.0, bid=51000.0, ask=51000.0)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, non_trigger_price_source)

        # Store order (should not fill immediately because ask=51000 > limit_price=50000)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Verify order is stored before fill attempt
        orders_before = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders_before), 1, "Order should be in memory before fill")

        # Register fill price source for USD conversions
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, fill_price_source)

        # Attempt fill WITHOUT creating a position first
        # This should fail because market_order_manager can't find a position to update
        self.limit_order_client.fill_limit_order_with_price_source(
            self.DEFAULT_MINER_HOTKEY,
            order,
            fill_price_source,
            49000.0
        )

        # Verify order was cancelled and removed from memory (Issue 8 fix)
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 0, "Cancelled orders should be removed from memory when fill fails")

    def test_fill_limit_order_exception_cancels(self):
        """Test limit order handling when position doesn't exist (error scenario)"""
        order = self.create_test_limit_order(limit_price=50000.0)
        fill_price_source = self.create_test_price_source(49000.0)

        # Set a non-triggering price to prevent immediate fill (ask > limit_price)
        non_trigger_price_source = self.create_test_price_source(51000.0, bid=51000.0, ask=51000.0)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, non_trigger_price_source)

        # Store order (should not fill immediately because ask=51000 > limit_price=50000)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Verify order exists before fill attempt
        orders_before = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders_before), 1, "Order should be in memory")

        # Register fill price source for USD conversions
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, fill_price_source)

        # Attempt fill WITHOUT a position (similar to test_fill_limit_order_error_cancels)
        # This tests exception handling path
        self.limit_order_client.fill_limit_order_with_price_source(
            self.DEFAULT_MINER_HOTKEY,
            order,
            fill_price_source,
            49000.0
        )

        # Verify order was removed from memory after error
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 0, "Orders should be removed from memory after error")

    # ============================================================================
    # Test Daemon: check_and_fill_limit_orders
    # ============================================================================

    def test_check_and_fill_limit_orders_no_orders(self):
        """Test daemon runs without errors when no orders exist"""
        self.limit_order_client.check_and_fill_limit_orders()
        # Should complete without errors

    def test_check_and_fill_limit_orders_market_closed(self):
        """Test daemon skips orders when market is closed"""
        order = self.create_test_limit_order()
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Set market to closed for testing
        self.live_price_fetcher_client.set_test_market_open(False)
        self.limit_order_client.check_and_fill_limit_orders()

        # Order should remain unfilled
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(orders[0].src, OrderSource.LIMIT_UNFILLED)

    def test_check_and_fill_limit_orders_no_price_sources(self):
        """Test daemon skips when no price sources available"""
        order = self.create_test_limit_order()
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Set market open but don't provide price sources (no test price source set = no data available)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.limit_order_client.check_and_fill_limit_orders()

        # Order should remain unfilled
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(orders[0].src, OrderSource.LIMIT_UNFILLED)

    def test_check_and_fill_limit_orders_triggers_and_fills(self):
        """
        INTEGRATION TEST: Test full daemon code flow including Issue 8 fix.

        This tests the complete production path:
        1. check_and_fill_limit_orders() iterates through orders
        2. Checks market status and price sources
        3. _attempt_fill_limit_order() evaluates trigger conditions
        4. _fill_limit_order_with_price_source() processes the fill
        5. _close_limit_order() removes filled orders from memory (Issue 8 fix)

        Uses real market_order_manager for true integration testing.
        """
        # Setup position FIRST (required if order fills immediately during process_limit_order)
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        # Create order with limit price that WON'T trigger immediately
        # Use a price below current market (~50k for BTC) for LONG order
        # This ensures even if price data exists, the order won't fill during processing
        order = self.create_test_limit_order(
            order_type=OrderType.LONG,
            limit_price=30000.0  # Well below current BTC price, won't trigger on LONG
        )

        # Process the limit order (won't fill immediately with price below market)
        result = self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)
        self.assertEqual(result["status"], "success", f"Order processing failed: {result}")

        # Verify order is in memory before daemon runs
        orders_before = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders_before), 1, "Order should be in memory before fill")
        self.assertEqual(orders_before[0].src, OrderSource.LIMIT_UNFILLED)

        # Set up test environment: market OPEN and price source that WILL trigger the order
        # For LONG order with limit 30k, price of 29k (bid) will trigger
        trigger_price_source = self.create_test_price_source(29000.0, bid=29000.0, ask=29000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price_source)

        # Run the FULL daemon code flow
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify the complete integration:
        # 1. Market was checked as open
        # 2. Price sources were fetched
        # 3. Order was evaluated and filled
        # 4. Order was removed from memory (Issue 8 fix)
        orders_after = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders_after), 0, "Filled orders should be removed from memory (Issue 8 fix)")

        # Verify fill happened by checking position was created
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(positions), 1, "Position should be created after fill")
        position = positions[0]
        self.assertEqual(len(position.orders), 1, "Position should have one order")
        self.assertEqual(position.orders[0].src, OrderSource.LIMIT_FILLED, "Order should be marked as LIMIT_FILLED")

        # Verify fill time was tracked
        fill_times = self.limit_order_client.get_last_fill_time()
        last_fill_time = fill_times.get(self.DEFAULT_TRADE_PAIR.trade_pair_id, {}).get(self.DEFAULT_MINER_HOTKEY, 0)
        self.assertGreater(last_fill_time, 0)

    def test_check_and_fill_limit_orders_skips_filled_orders(self):
        """Test daemon skips already filled orders"""
        order = self.create_test_limit_order()
        order.src = OrderSource.LIMIT_FILLED

        # Manually add filled order to server state (shouldn't happen in practice)
        self.limit_order_client.set_limit_orders_dict({
            self.DEFAULT_TRADE_PAIR.trade_pair_id: {
                self.DEFAULT_MINER_HOTKEY: [order.to_python_dict()]
            }
        })

        # Set up test environment: market open with triggering price
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(
            self.DEFAULT_TRADE_PAIR,
            self.create_test_price_source(40000.0)
        )
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify no position was created (order was skipped)
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(positions), 0, "No position should be created for already-filled orders")

    # ============================================================================
    # Test Helper Methods
    # ============================================================================

    def test_count_unfilled_orders_for_hotkey(self):
        """Test counting unfilled orders across trade pairs"""
        # Add unfilled orders across different trade pairs
        for trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD]:
            for i in range(2):
                order = self.create_test_limit_order(
                    trade_pair=trade_pair,
                    order_uuid=f"{trade_pair.trade_pair_id}_{i}"
                )
                self.limit_order_client.process_limit_order(
                    self.DEFAULT_MINER_HOTKEY,
                    order
                )

        count = self.limit_order_client.count_unfilled_orders_for_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(count, 4)

        # Fill one order - need to update server state
        orders_dict = self.limit_order_client.get_limit_orders_dict()
        # Modify the first BTC order to be filled (use integer value from IntEnum)
        orders_dict[TradePair.BTCUSD.trade_pair_id][self.DEFAULT_MINER_HOTKEY][0]['src'] = OrderSource.LIMIT_FILLED.value
        # Send updated dict back to server
        self.limit_order_client.set_limit_orders_dict(orders_dict)

        count = self.limit_order_client.count_unfilled_orders_for_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(count, 3)

    def test_get_position_for(self):
        """Test getting position for limit order"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        order = self.create_test_limit_order()

        retrieved_position = self.limit_order_client.get_position_for(
            self.DEFAULT_MINER_HOTKEY,
            order
        )

        self.assertIsNotNone(retrieved_position)
        self.assertEqual(retrieved_position.position_uuid, position.position_uuid)

    # ============================================================================
    # Test Data Structure and Persistence
    # ============================================================================

    def test_data_structure_nested_by_trade_pair(self):
        """Test limit orders are stored in nested structure {TradePair: {hotkey: [Order]}}"""
        order = self.create_test_limit_order()
        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order
        )

        # Verify structure
        all_orders = self.limit_order_client.get_limit_orders_dict()
        self.assertIsInstance(all_orders, dict)
        self.assertIn(self.DEFAULT_TRADE_PAIR.trade_pair_id, all_orders)
        self.assertIsInstance(all_orders[self.DEFAULT_TRADE_PAIR.trade_pair_id], dict)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, all_orders[self.DEFAULT_TRADE_PAIR.trade_pair_id])
        self.assertIsInstance(all_orders[self.DEFAULT_TRADE_PAIR.trade_pair_id][self.DEFAULT_MINER_HOTKEY], list)

    def test_multiple_miners_isolation(self):
        """Test limit orders are isolated by miner"""
        miner2 = "miner2"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, miner2])

        order1 = self.create_test_limit_order(order_uuid="miner1_order")
        order2 = self.create_test_limit_order(order_uuid="miner2_order")

        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order1
        )
        self.limit_order_client.process_limit_order(
            miner2,
            order2
        )

        miner1_orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        miner2_orders = self.get_orders_from_server(miner2, self.DEFAULT_TRADE_PAIR)

        self.assertEqual(len(miner1_orders), 1)
        self.assertEqual(len(miner2_orders), 1)
        self.assertEqual(miner1_orders[0].order_uuid, "miner1_order")
        self.assertEqual(miner2_orders[0].order_uuid, "miner2_order")

    def test_read_limit_orders_from_disk_skips_eliminated(self):
        """Test that eliminated miners' orders are not loaded from disk"""
        # Add order
        order = self.create_test_limit_order()
        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order
        )

        # Eliminate miner - use proper API method
        from vali_objects.utils.elimination_manager import EliminationReason
        self.elimination_client.append_elimination_row(
            self.DEFAULT_MINER_HOTKEY,
            TimeUtil.now_in_millis(),
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )

        # Trigger cleanup for eliminated miner (deletes limit orders)
        self.elimination_client.handle_eliminated_miner(
            self.DEFAULT_MINER_HOTKEY,
            trade_pair_to_price_source_dict={},
            iteration_epoch=None
        )

        # Verify eliminated miner's orders are not accessible via orchestrator client
        orders_dict = self.limit_order_client.get_all_limit_orders()
        orders = orders_dict.get(self.DEFAULT_TRADE_PAIR.trade_pair_id, {}).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 0, "Eliminated miner's orders should not be returned")

    def test_create_bracket_order_with_both_sltp(self):
        """Test creating a bracket order with both stop loss and take profit"""
        # Create parent limit order with SL and TP using quantity
        parent_order = self.create_test_limit_order(
            limit_price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            leverage=None,  # Use quantity instead
            quantity=0.5  # 0.5 BTC
        )

        # Manually call _create_sltp_orders as it's called after fill
        self.limit_order_client.create_sltp_orders(self.DEFAULT_MINER_HOTKEY, parent_order)

        # Verify only ONE bracket order was created
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        bracket_orders = [o for o in orders if o.order_uuid.endswith('-bracket')]
        self.assertEqual(len(bracket_orders), 1, "Should create exactly one bracket order")

        # Verify bracket order properties
        bracket_order = bracket_orders[0]
        self.assertEqual(bracket_order.execution_type, ExecutionType.BRACKET)
        self.assertEqual(bracket_order.stop_loss, 49000.0)
        self.assertEqual(bracket_order.take_profit, 51000.0)
        self.assertEqual(bracket_order.src, OrderSource.BRACKET_UNFILLED)
        self.assertEqual(bracket_order.order_type, OrderType.LONG)  # Same as parent
        self.assertEqual(bracket_order.quantity, parent_order.quantity)  # Same quantity (0.5 BTC)
        self.assertIsNone(bracket_order.leverage)  # Bracket orders have None leverage

    def test_create_bracket_order_with_only_sl(self):
        """Test creating a bracket order with only stop loss"""
        parent_order = self.create_test_limit_order(
            limit_price=50000.0,
            stop_loss=49000.0,
            take_profit=None
        )

        self.limit_order_client.create_sltp_orders(self.DEFAULT_MINER_HOTKEY, parent_order)

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        bracket_orders = [o for o in orders if o.order_uuid.endswith('-bracket')]
        self.assertEqual(len(bracket_orders), 1)

        bracket_order = bracket_orders[0]
        self.assertEqual(bracket_order.stop_loss, 49000.0)
        self.assertIsNone(bracket_order.take_profit)

    def test_create_bracket_order_with_only_tp(self):
        """Test creating a bracket order with only take profit"""
        parent_order = self.create_test_limit_order(
            limit_price=50000.0,
            stop_loss=None,
            take_profit=51000.0
        )

        self.limit_order_client.create_sltp_orders(self.DEFAULT_MINER_HOTKEY, parent_order)

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        bracket_orders = [o for o in orders if o.order_uuid.endswith('-bracket')]
        self.assertEqual(len(bracket_orders), 1)

        bracket_order = bracket_orders[0]
        self.assertIsNone(bracket_order.stop_loss)
        self.assertEqual(bracket_order.take_profit, 51000.0)

    def test_evaluate_bracket_trigger_price_long_stop_loss(self):
        """Test bracket order trigger for LONG bracket hitting stop loss"""
        # LONG bracket order (same type as parent LONG)
        # SL triggers when bid < SL (price fell)
        bracket_order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test-bracket",
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=OrderType.LONG,  # Same as parent
            leverage=1.0,
            execution_type=ExecutionType.BRACKET,
            stop_loss=48000.0,  # SL below entry
            take_profit=52000.0,  # TP above entry
            src=OrderSource.BRACKET_UNFILLED
        )

        # Create mock position (LONG position being protected by bracket)
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[],
            position_type=OrderType.LONG,
            current_return=0.0,
        )

        # Price falls BELOW stop loss (bid < 48000)
        price_source = PriceSource(
            source="test",
            timespan_ms=1000,
            open=50000.0,
            close=47500.0,
            high=50000.0,
            low=47500.0,
            bid=47500.0,  # bid < 48000 triggers stop loss
            ask=47600.0,
            start_ms=TimeUtil.now_in_millis(),
            websocket=False,
            lag_ms=0
        )

        trigger_price = self.limit_order_client.evaluate_bracket_trigger_price(
            bracket_order,
            position,
            price_source
        )

        self.assertIsNotNone(trigger_price)
        self.assertEqual(trigger_price, 48000.0)  # Returns the stop_loss price

    def test_evaluate_bracket_trigger_price_long_take_profit(self):
        """Test bracket order trigger for LONG bracket hitting take profit"""
        # LONG bracket order (same type as parent LONG)
        # TP triggers when bid > TP (price rose)
        bracket_order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test-bracket",
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=OrderType.LONG,  # Same as parent
            leverage=1.0,
            execution_type=ExecutionType.BRACKET,
            stop_loss=48000.0,
            take_profit=52000.0,  # TP above entry
            src=OrderSource.BRACKET_UNFILLED
        )

        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[],
            position_type=OrderType.LONG,
            current_return=0.0,
        )

        # Price rises ABOVE take profit (bid > 52000)
        price_source = PriceSource(
            source="test",
            timespan_ms=1000,
            open=50000.0,
            close=52500.0,
            high=52500.0,
            low=50000.0,
            bid=52500.0,  # bid > 52000 triggers take profit
            ask=52600.0,
            start_ms=TimeUtil.now_in_millis(),
            websocket=False,
            lag_ms=0
        )

        trigger_price = self.limit_order_client.evaluate_bracket_trigger_price(
            bracket_order,
            position,
            price_source
        )

        self.assertIsNotNone(trigger_price)
        self.assertEqual(trigger_price, 52000.0)  # Returns the take_profit price

    def test_evaluate_bracket_trigger_price_short_stop_loss(self):
        """Test bracket order trigger for SHORT bracket hitting stop loss"""
        # SHORT bracket order (same type as parent SHORT)
        # SL triggers when ask > SL (price rose)
        bracket_order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test-bracket",
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=OrderType.SHORT,  # Same as parent
            leverage=-1.0,
            execution_type=ExecutionType.BRACKET,
            stop_loss=52000.0,  # SL above entry
            take_profit=48000.0,  # TP below entry
            src=OrderSource.BRACKET_UNFILLED
        )

        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[],
            position_type=OrderType.SHORT,
            current_return=0.0,
        )

        # Price rises ABOVE stop loss (ask > 52000)
        price_source = PriceSource(
            source="test",
            timespan_ms=1000,
            open=50000.0,
            close=52500.0,
            high=52500.0,
            low=50000.0,
            bid=52400.0,
            ask=52500.0,  # ask > 52000 triggers stop loss
            start_ms=TimeUtil.now_in_millis(),
            websocket=False,
            lag_ms=0
        )

        trigger_price = self.limit_order_client.evaluate_bracket_trigger_price(
            bracket_order,
            position,
            price_source
        )

        self.assertIsNotNone(trigger_price)
        self.assertEqual(trigger_price, 52000.0)  # Returns the stop_loss price

    def test_evaluate_bracket_trigger_price_short_take_profit(self):
        """Test bracket order trigger for SHORT bracket hitting take profit"""
        # SHORT bracket order (same type as parent SHORT)
        # TP triggers when ask < TP (price fell)
        bracket_order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test-bracket",
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=OrderType.SHORT,  # Same as parent
            leverage=-1.0,
            execution_type=ExecutionType.BRACKET,
            stop_loss=52000.0,  # SL above entry
            take_profit=48000.0,  # TP below entry
            src=OrderSource.BRACKET_UNFILLED
        )

        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[],
            position_type=OrderType.SHORT,
            current_return=0.0,
        )

        # Price falls BELOW take profit (ask < 48000)
        price_source = PriceSource(
            source="test",
            timespan_ms=1000,
            open=50000.0,
            close=47500.0,
            high=50000.0,
            low=47500.0,
            bid=47400.0,
            ask=47500.0,  # ask < 48000 triggers take profit
            start_ms=TimeUtil.now_in_millis(),
            websocket=False,
            lag_ms=0
        )

        trigger_price = self.limit_order_client.evaluate_bracket_trigger_price(
            bracket_order,
            position,
            price_source
        )

        self.assertIsNotNone(trigger_price)
        self.assertEqual(trigger_price, 48000.0)  # Returns the take_profit price

    def test_evaluate_bracket_trigger_price_no_trigger(self):
        """Test bracket order when price doesn't hit either boundary"""
        # LONG bracket order - same type as parent
        # SL below entry, TP above entry
        bracket_order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test-bracket",
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=OrderType.LONG,
            leverage=1.0,
            execution_type=ExecutionType.BRACKET,
            stop_loss=48000.0,  # Loss if bid < 48000
            take_profit=52000.0,  # Profit if bid > 52000
            src=OrderSource.BRACKET_UNFILLED
        )

        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[],
            position_type=OrderType.LONG,
            current_return=0.0,
        )

        # Price stays between SL and TP: 48000 < 50000 < 52000
        # For LONG bracket: triggers when bid < SL OR bid > TP
        # bid=50000: not < 48000 (no SL), not > 52000 (no TP) → no trigger
        price_source = PriceSource(
            source="test",
            timespan_ms=1000,
            open=50000.0,
            close=50000.0,
            high=50500.0,
            low=49500.0,
            bid=50000.0,  # 48000 < 50000 < 52000, so no trigger
            ask=50100.0,
            start_ms=TimeUtil.now_in_millis(),
            websocket=False,
            lag_ms=0
        )

        trigger_price = self.limit_order_client.evaluate_bracket_trigger_price(
            bracket_order,
            position,
            price_source
        )

        self.assertIsNone(trigger_price)

    # ============================================================================
    # Test Design Behavior: Fill Interval Enforcement
    # ============================================================================

    def test_fill_interval_enforcement_only_one_order_per_interval(self):
        """
        Test DESIGN BEHAVIOR: Only one order per trade pair per hotkey can fill within the interval.

        This enforces LIMIT_ORDER_FILL_INTERVAL_MS rate limiting by breaking after the first fill.
        Even if multiple orders are triggered, only the first one fills, and subsequent orders
        must wait for the next interval.

        Note: This test bypasses process_limit_order() which has its own immediate fill logic
        and interval enforcement. Instead, it directly injects orders into the server to test
        ONLY the check_and_fill_limit_orders() daemon's interval enforcement.
        """
        # Setup position first (required for market_order_manager)
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        # Create multiple orders
        order1 = self.create_test_limit_order(
            order_uuid="order1",
            order_type=OrderType.LONG,
            limit_price=100000.0
        )
        order2 = self.create_test_limit_order(
            order_uuid="order2",
            order_type=OrderType.LONG,
            limit_price=100000.0
        )
        order3 = self.create_test_limit_order(
            order_uuid="order3",
            order_type=OrderType.LONG,
            limit_price=100000.0
        )

        # Directly inject orders into the server to bypass process_limit_order's immediate fill logic
        # This allows us to test ONLY the check_and_fill_limit_orders daemon's interval enforcement
        orders_dict = {
            self.DEFAULT_TRADE_PAIR.trade_pair_id: {
                self.DEFAULT_MINER_HOTKEY: [
                    order1.to_python_dict(),
                    order2.to_python_dict(),
                    order3.to_python_dict()
                ]
            }
        }
        self.limit_order_client.set_limit_orders_dict(orders_dict)

        # Set up test environment: market open and trigger price available
        # Only set ONE price source to avoid median calculation issues
        trigger_price_source = self.create_test_price_source(49000.0, bid=49000.0, ask=49000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price_source)

        # Run FULL daemon code flow - this is where interval enforcement happens
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify ONLY the first order was filled (and removed from memory due to Issue 8 fix)
        # The other two should remain unfilled
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)

        # After Issue 8 fix: filled order is removed, so only 2 unfilled orders remain
        self.assertEqual(len(orders), 2, "Two unfilled orders should remain (filled order removed)")

        # Verify both remaining orders are unfilled
        for order in orders:
            self.assertEqual(order.src, OrderSource.LIMIT_UNFILLED,
                           "Remaining orders should be unfilled")

        # Verify exactly one position was created (from the one fill)
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(positions), 1, "Exactly one position should be created from the one fill")

    def test_fill_interval_enforcement_multiple_miners_independent(self):
        """
        Test that fill interval enforcement is per (trade_pair, hotkey) pair.
        Multiple miners can fill on the same trade pair in the same interval.

        Note: This test bypasses process_limit_order() which has its own immediate fill logic
        and interval enforcement. Instead, it directly injects orders into the server to test
        ONLY the check_and_fill_limit_orders() daemon's interval enforcement.
        """
        miner2 = "miner2"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, miner2])

        # Setup positions for both miners (required for market_order_manager)
        position1 = self.create_test_position(miner_hotkey=self.DEFAULT_MINER_HOTKEY)
        position2 = self.create_test_position(miner_hotkey=miner2)
        self.position_client.save_miner_position(position1)
        self.position_client.save_miner_position(position2)

        # Create orders for two different miners
        order1 = self.create_test_limit_order(
            order_uuid="miner1_order",
            order_type=OrderType.LONG,
            limit_price=50000.0
        )
        order2 = self.create_test_limit_order(
            order_uuid="miner2_order",
            order_type=OrderType.LONG,
            limit_price=50000.0
        )

        # Directly inject orders into the server to bypass process_limit_order's immediate fill logic
        # This allows us to test ONLY the check_and_fill_limit_orders daemon's interval enforcement
        orders_dict = {
            self.DEFAULT_TRADE_PAIR.trade_pair_id: {
                self.DEFAULT_MINER_HOTKEY: [order1.to_python_dict()],
                miner2: [order2.to_python_dict()]
            }
        }
        self.limit_order_client.set_limit_orders_dict(orders_dict)

        # Set up test environment: market open and price source available
        trigger_price_source = self.create_test_price_source(49000.0, bid=49000.0, ask=49000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price_source)

        # Run FULL daemon code flow - this is where interval enforcement happens
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify BOTH miners' orders were filled and removed from memory (Issue 8 fix)
        # Different hotkeys = independent intervals, so both can fill in same daemon run
        miner1_orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        miner2_orders = self.get_orders_from_server(miner2, self.DEFAULT_TRADE_PAIR)

        self.assertEqual(len(miner1_orders), 0, "Miner1's filled order should be removed from memory")
        self.assertEqual(len(miner2_orders), 0, "Miner2's filled order should be removed from memory")

        # Verify both miners got positions (actual fills happened)
        miner1_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        miner2_positions = self.position_client.get_positions_for_one_hotkey(miner2)
        self.assertEqual(len(miner1_positions), 1, "Miner1 should have one position")
        self.assertEqual(len(miner2_positions), 1, "Miner2 should have one position")

    # ============================================================================
    # Test Design Behavior: Partial UUID Matching for Bracket Orders
    # ============================================================================

    def test_cancel_bracket_order_using_parent_uuid(self):
        """
        Test DESIGN BEHAVIOR: Bracket orders can be cancelled using parent order UUID.

        When a limit order with SL/TP fills, it creates a bracket order with UUID:
        "{parent_uuid}-bracket"

        Miners can cancel this bracket order by providing just the parent UUID,
        which uses startswith() matching.
        """
        # Create parent limit order with SL/TP
        parent_order = self.create_test_limit_order(
            order_uuid="parent123",
            limit_price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            leverage=0.1
        )

        # Manually create bracket order (as would happen after fill)
        self.limit_order_client.create_sltp_orders(self.DEFAULT_MINER_HOTKEY, parent_order)

        # Verify bracket order exists with correct UUID
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        bracket_orders = [o for o in orders if o.execution_type == ExecutionType.BRACKET]
        self.assertEqual(len(bracket_orders), 1)
        self.assertEqual(bracket_orders[0].order_uuid, "parent123-bracket")

        # Cancel using PARENT UUID (not the full bracket UUID)
        result = self.limit_order_client.cancel_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "parent123",  # Using parent UUID, not "parent123-bracket"
            TimeUtil.now_in_millis()
        )

        # Verify bracket order was cancelled
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["num_cancelled"], 1)

        # Verify the bracket order has been removed from memory (Issue 8 fix)
        # Cancelled orders are persisted to disk but removed from active memory
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        bracket_orders = [o for o in orders if o.order_uuid == "parent123-bracket"]
        self.assertEqual(len(bracket_orders), 0, "Cancelled bracket order should be removed from memory")

    def test_cancel_bracket_order_using_full_uuid(self):
        """
        Test that bracket orders can also be cancelled using the full UUID.
        """
        # Create bracket order
        parent_order = self.create_test_limit_order(
            order_uuid="parent456",
            limit_price=50000.0,
            stop_loss=49000.0
        )

        self.limit_order_client.create_sltp_orders(self.DEFAULT_MINER_HOTKEY, parent_order)

        # Cancel using FULL bracket UUID
        result = self.limit_order_client.cancel_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "parent456-bracket",  # Using full bracket UUID
            TimeUtil.now_in_millis()
        )

        # Verify cancellation succeeded
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["num_cancelled"], 1)

    def test_cancel_parent_uuid_does_not_affect_regular_limit_orders(self):
        """
        Test that partial UUID matching only applies to BRACKET orders.
        Regular limit orders require exact UUID match.
        """
        # Create two regular limit orders with similar UUIDs
        order1 = self.create_test_limit_order(order_uuid="order123")
        order2 = self.create_test_limit_order(order_uuid="order123-extra")

        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order1
        )
        self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            order2
        )

        # Try to cancel using partial UUID "order123"
        result = self.limit_order_client.cancel_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "order123",
            TimeUtil.now_in_millis()
        )

        # Should only cancel the exact match, not the one with prefix
        self.assertEqual(result["num_cancelled"], 1)

        # Verify only order1 was cancelled (removed from memory), order2 remains unfilled
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)

        # order1 should be removed from memory (cancelled orders are cleaned up)
        order1_exists = any(o.order_uuid == "order123" for o in orders)
        self.assertFalse(order1_exists, "Cancelled order should be removed from memory")

        # order2 should still be unfilled
        order2_in_list = next((o for o in orders if o.order_uuid == "order123-extra"), None)
        self.assertIsNotNone(order2_in_list, "Unfilled order should remain in memory")
        self.assertEqual(order2_in_list.src, OrderSource.LIMIT_UNFILLED)

    def test_bracket_order_uuid_format(self):
        """
        Test DESIGN BEHAVIOR: Bracket order UUID format is always "{parent_uuid}-bracket".

        This consistent format enables the partial UUID matching for cancellation.
        """
        test_cases = [
            ("abc123", "abc123-bracket"),
            ("order-xyz-789", "order-xyz-789-bracket"),
            ("simple", "simple-bracket"),
        ]

        for parent_uuid, expected_bracket_uuid in test_cases:
            with self.subTest(parent_uuid=parent_uuid):
                # Clear previous orders
                self.limit_order_client.clear_limit_orders()

                # Create parent order
                parent_order = self.create_test_limit_order(
                    order_uuid=parent_uuid,
                    limit_price=50000.0,
                    stop_loss=49000.0,
                    leverage=0.1
                )

                # Create bracket order
                self.limit_order_client.create_sltp_orders(self.DEFAULT_MINER_HOTKEY, parent_order)

                # Verify bracket UUID format
                orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
                bracket_orders = [o for o in orders if o.execution_type == ExecutionType.BRACKET]

                self.assertEqual(len(bracket_orders), 1)
                self.assertEqual(bracket_orders[0].order_uuid, expected_bracket_uuid)


if __name__ == '__main__':
    unittest.main()
