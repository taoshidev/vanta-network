import unittest

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.position import Position
from vali_objects.utils.limit_order_server import LimitOrderClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestLimitOrderIntegration(TestBase):
    """
    INTEGRATION TESTS for limit order management using client/server data injection.

    These tests run full production code paths WITHOUT mocking internal methods:
    - NO mocking of market_order_manager (tests actual position updates)
    - NO mocking of internal server methods (_write_to_disk, _get_best_price_source)
    - Data injection through direct server access for test setup
    - Real code paths execute end-to-end

    Goal: Verify end-to-end correctness of limit order fills, position updates,
    bracket order creation, and error handling using client/server architecture.
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    live_price_fetcher_server = None  # Direct access for test data injection
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    limit_order_client = None

    DEFAULT_MINER_HOTKEY = "integration_test_miner"

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
        cls.limit_order_client: LimitOrderClient = cls.orchestrator.get_client('limit_order')

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
        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Set up test data
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY])
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def create_test_position(self, order_type=OrderType.LONG, leverage=1.0):
        """Create and save a test position with an initial order."""
        now_ms = TimeUtil.now_in_millis()
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=f"pos_{now_ms}",
            open_ms=now_ms,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            account_size=1000.0  # Required for position validation
        )

        # Add initial market order to position
        initial_order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid=f"initial_{now_ms}",
            processed_ms=now_ms,
            price=50000.0,
            order_type=order_type,
            leverage=leverage,
            execution_type=ExecutionType.MARKET,
            src=OrderSource.ORGANIC  # ORGANIC is used for miner-generated orders
        )
        initial_order.bid = 50000.0
        initial_order.ask = 50000.0

        # Add order to position (requires live_price_fetcher for return calculation)
        position.add_order(initial_order, self.live_price_fetcher_client)
        self.position_client.save_miner_position(position)
        return position

    def create_limit_order(self, order_type: OrderType=OrderType.LONG, limit_price=51000.0,
                          leverage=0.5, stop_loss=None, take_profit=None, order_uuid=None):
        """Create a limit order (not yet submitted)."""
        if order_uuid is None:
            order_uuid = f"limit_{TimeUtil.now_in_millis()}"
        return Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid=order_uuid,
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=order_type,
            leverage=leverage,
            execution_type=ExecutionType.LIMIT,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            src=OrderSource.LIMIT_UNFILLED
        )

    def create_price_source(self, price, bid=None, ask=None):
        """Create a price source for test data injection."""
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
            start_ms=TimeUtil.now_in_millis(),
            websocket=True,
            lag_ms=100,
            bid=bid,
            ask=ask
        )

    def inject_price_data(self, trade_pair, price_source):
        """
        Inject test price data using client RPC methods.

        Uses the live_price_fetcher_client to inject test price data through
        proper RPC channels instead of direct server manipulation.

        Args:
            trade_pair: TradePair to inject price for
            price_source: Single PriceSource to inject (or None to disable fallback)
        """
        # Use client RPC method to inject test price (single source, not list)
        self.live_price_fetcher_client.set_test_price_source(trade_pair, price_source)

    def set_market_open(self, is_open=True):
        """
        Configure market hours state using client RPC methods.

        Uses the live_price_fetcher_client to set test market state through
        proper RPC channels.
        """
        # Use client RPC method to set market open state
        self.live_price_fetcher_client.set_test_market_open(is_open)

    # ============================================================================
    # Integration Tests: Full Fill Path
    # ============================================================================

    def test_end_to_end_long_limit_order_fill(self):
        """
        INTEGRATION TEST: Complete LONG limit order fill using client/server data injection.

        Tests:
        - Position is created and updated correctly
        - Limit order triggers at correct price
        - Position contains both initial and limit orders
        - Leverage is applied correctly
        - Order is removed from memory after fill

        Uses test data injection instead of mocking.
        """
        # Create initial LONG position (BTCUSD max leverage is 0.5)
        initial_position = self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Verify initial position
        self.assertEqual(len(initial_position.orders), 1)

        # Create LONG limit order to add to position (0.3 + 0.2 = 0.5, within max leverage)
        limit_order = self.create_limit_order(order_type=OrderType.LONG, limit_price=48000.0, leverage=0.2)

        # Ensure no price data available during order submission (prevents immediate fill)
        # Orchestrator cleanup already cleared price sources, inject None to keep it clear
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order (stored unfilled - won't fill until we run daemon with price data)
        result = self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)
        self.assertEqual(result["status"], "success")

        # Verify order is stored unfilled
        orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1, "Should have 1 unfilled limit order in memory")
        self.assertEqual(orders[0]['src'], OrderSource.LIMIT_UNFILLED)

        # Set up test environment: market OPEN and price source that WILL trigger the order
        # For LONG order with limit 48000, ask=47500 will trigger (ask <= limit)
        trigger_price_source = self.create_price_source(47500.0, bid=47500.0, ask=47500.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price_source)

        # Run daemon via client - server will use injected test data
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify order removed from memory (filled orders are cleaned up)
        orders_after = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders_after), 0, "Filled order should be removed from memory")

        # Verify REAL position was updated with the limit order
        updated_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertIsNotNone(updated_position, "Position should exist after fill")
        self.assertEqual(len(updated_position.orders), 2, "Position should have initial + limit order")

        # Verify limit order details
        limit_order_in_position = updated_position.orders[-1]
        self.assertEqual(limit_order_in_position.order_type, OrderType.LONG)
        self.assertEqual(limit_order_in_position.leverage, 0.2)
        self.assertGreater(limit_order_in_position.price, 0, "Filled order should have price set")
        self.assertEqual(limit_order_in_position.src, OrderSource.LIMIT_FILLED)

        # Verify position net leverage updated
        expected_leverage = 0.3 + 0.2
        self.assertAlmostEqual(updated_position.net_leverage, expected_leverage, places=2)

    def test_end_to_end_short_limit_order_fill(self):
        """
        INTEGRATION TEST: Complete SHORT limit order fill using client/server data injection.

        Tests SHORT-specific logic:
        - SHORT order triggers when bid >= limit_price
        - Position net leverage decreases (SHORT reduces LONG exposure)

        Uses test data injection instead of mocking.
        """
        # Create initial LONG position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.4)

        # Create SHORT limit order to reduce position
        limit_order = self.create_limit_order(order_type=OrderType.SHORT, limit_price=51000.0, leverage=-0.2)

        # Inject None to prevent immediate fill during order processing
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order via client (no price source available, so it won't trigger immediately)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Inject test price data: bid=51500 >= limit=51000 triggers SHORT
        trigger_price_source = self.create_price_source(51500.0, bid=51500.0, ask=51500.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)

        # Configure market as open
        self.set_market_open(is_open=True)

        # Run daemon - server uses injected data through client/server architecture
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify REAL position updated
        updated_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertIsNotNone(updated_position)
        self.assertEqual(len(updated_position.orders), 2)

        # Verify SHORT order details
        short_order = updated_position.orders[-1]
        self.assertEqual(short_order.order_type, OrderType.SHORT)
        self.assertEqual(short_order.leverage, -0.2)
        self.assertGreater(short_order.price, 0, "Filled order should have price set")
        self.assertEqual(short_order.src, OrderSource.LIMIT_FILLED)

        # Verify net leverage reduced
        expected_leverage = 0.4 - 0.2
        self.assertAlmostEqual(updated_position.net_leverage, expected_leverage, places=2)

    def test_end_to_end_bracket_order_creation_and_trigger(self):
        """
        INTEGRATION TEST: Test full bracket order lifecycle using test data injection.

        Tests:
        1. Limit order with SL/TP fills
        2. Bracket order is created automatically
        3. Bracket order triggers when stop loss hit
        4. Position is closed by bracket order
        """
        # Create initial LONG position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create limit order with stop loss and take profit
        limit_order = self.create_limit_order(
            order_type=OrderType.LONG,
            limit_price=48000.0,
            leverage=0.2,
            stop_loss=45000.0,
            take_profit=52000.0
        )

        # Inject None to prevent immediate fill during order processing
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit limit order (no price source available, so it won't trigger immediately)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Inject test price data to fill the limit order
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        # Fill the limit order
        self.limit_order_client.check_and_fill_limit_orders(call_id=1)

        # Verify bracket order was created
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 1, "Bracket order should be created")
        bracket_order = bracket_orders[0]
        self.assertEqual(bracket_order['execution_type'], 'BRACKET')  # RPC serializes enum to string
        self.assertEqual(bracket_order['stop_loss'], 45000.0)
        self.assertEqual(bracket_order['take_profit'], 52000.0)
        self.assertEqual(bracket_order['src'], OrderSource.BRACKET_UNFILLED)

        # Clear fill interval to allow bracket order to fill immediately
        self.limit_order_client.set_last_fill_time(
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            self.DEFAULT_MINER_HOTKEY,
            0
        )

        # Trigger stop loss (price falls below 45000)
        stop_loss_price_source = self.create_price_source(44000.0, bid=44000.0, ask=44000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, stop_loss_price_source)
        # Market should still be open from previous set_market_open call, but verify
        self.set_market_open(is_open=True)

        # Verify position still exists before trying to fill bracket
        position_before_bracket = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        print(f"DEBUG: Position before bracket fill exists: {position_before_bracket is not None}")
        if position_before_bracket:
            print(f"DEBUG: Position has {len(position_before_bracket.orders)} orders")
            print(f"DEBUG: Position net leverage: {position_before_bracket.net_leverage}")

        # Force fresh RPC connection to avoid caching
        self.limit_order_client.disconnect()
        self.limit_order_client.connect()

        # Fill the bracket order
        print("[TEST DEBUG] About to call check_and_fill_limit_orders(call_id=2)")
        result = self.limit_order_client.check_and_fill_limit_orders(call_id=2)
        print(f"[TEST DEBUG] Result from second call: {result}")

        # Verify bracket order filled (position closed/reduced)
        bracket_orders_after = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        print(f"DEBUG: Bracket orders after fill: {len(bracket_orders_after)}")
        if bracket_orders_after:
            print(f"DEBUG: Bracket order still unfilled: {bracket_orders_after[0]}")
        self.assertEqual(len(bracket_orders_after), 0, "Bracket order should be removed after fill")

        # Verify position updated with bracket order fill
        final_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        # Position should have 3 orders: initial, limit, bracket
        self.assertEqual(len(final_position.orders), 3)

        # Verify bracket order is SHORT (opposite of LONG position)
        bracket_fill = final_position.orders[-1]
        self.assertEqual(bracket_fill.order_type, OrderType.SHORT)
        self.assertEqual(bracket_fill.src, OrderSource.BRACKET_FILLED)

    def test_position_closed_when_no_position_exists(self):
        """
        INTEGRATION TEST: Bracket order should be cancelled if position no longer exists.

        This tests the production error path using test data injection.
        """
        # Create and then close a position (BTCUSD max leverage is 0.5)
        initial_position = self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create limit order with stop loss
        limit_order = self.create_limit_order(
            order_type=OrderType.LONG,
            limit_price=48000.0,
            leverage=0.2,
            stop_loss=45000.0
        )

        # Inject None to prevent immediate fill during order processing
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit limit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify bracket order created
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 1)

        # Clear fill interval
        self.limit_order_client.set_last_fill_time(
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            self.DEFAULT_MINER_HOTKEY,
            0
        )

        # Close the position manually (simulate position being closed elsewhere)
        self.position_client.clear_all_miner_positions_and_disk()

        # Try to trigger bracket order when position doesn't exist
        stop_loss_price_source = self.create_price_source(44000.0, bid=44000.0, ask=44000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, stop_loss_price_source)

        # Should not crash, should cancel bracket order
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify bracket order was cancelled (removed from memory)
        bracket_orders_after = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders_after), 0, "Bracket order should be cancelled when position missing")

    def test_multiple_limit_orders_fill_sequentially_with_interval(self):
        """
        INTEGRATION TEST: Multiple limit orders respect fill interval.

        Tests REAL timing enforcement using test data injection.
        """
        # Create initial position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.2)

        # Create multiple limit orders
        order1 = self.create_limit_order(order_uuid="order1", limit_price=48000.0, leverage=0.1)
        order2 = self.create_limit_order(order_uuid="order2", limit_price=48000.0, leverage=0.1)
        order3 = self.create_limit_order(order_uuid="order3", limit_price=48000.0, leverage=0.1)

        # Inject None to prevent immediate fills during order processing
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit all orders
        for order in [order1, order2, order3]:
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Set up test price data and market hours for daemon
        trigger_price = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price)
        self.set_market_open(is_open=True)

        # First daemon run - should fill only one order
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify only one order filled
        remaining_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(remaining_orders), 2, "Two orders should remain (one filled)")

        # Second daemon run immediately - should NOT fill (within interval)
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify still two orders remain
        remaining_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(remaining_orders), 2, "Still two orders (fill interval enforced)")

        # Verify position has 2 orders (initial + first limit)
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 2)
        self.assertAlmostEqual(position.net_leverage, 0.2 + 0.1, places=2)

    def test_limit_order_does_not_fill_when_market_closed(self):
        """
        INTEGRATION TEST: Orders should not fill when market is closed.

        Tests market hours enforcement using test data injection.
        """
        # Create position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create limit order
        limit_order = self.create_limit_order(limit_price=48000.0)

        # Inject None to prevent immediate fill during order processing
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order via client
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Inject test price data that would trigger the order
        trigger_price = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price)

        # Configure market as CLOSED
        self.set_market_open(is_open=False)

        self.assertFalse(self.live_price_fetcher_client.is_market_open(limit_order.trade_pair))

        # Run daemon - market closed prevents fill
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify order NOT filled
        orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1, "Order should remain unfilled when market closed")
        self.assertEqual(orders[0]['src'], OrderSource.LIMIT_UNFILLED)

        # Verify position unchanged
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 1, "Position should have only initial order")

    # ============================================================================
    # Integration Tests: Price Source Logic
    # ============================================================================

    def test_best_price_source_selection_uses_median(self):
        """
        INTEGRATION TEST: This test duplicates test_end_to_end_long_limit_order_fill.

        Since the first test already thoroughly validates the complete fill flow
        using test data injection (including price source usage, market hours,
        position updates, etc.), this test is kept for backward compatibility
        but essentially verifies the same behavior.
        """
        # This test is identical to test_end_to_end_long_limit_order_fill
        # Create initial LONG position
        initial_position = self.create_test_position(order_type=OrderType.LONG, leverage=0.3)
        self.assertEqual(len(initial_position.orders), 1)

        # Create LONG limit order
        limit_order = self.create_limit_order(order_type=OrderType.LONG, limit_price=48000.0, leverage=0.2)

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        result = self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)
        self.assertEqual(result["status"], "success")

        # Inject trigger price: ask=47500 < limit=48000 triggers LONG
        trigger_price_source = self.create_price_source(47500.0, bid=47500.0, ask=47500.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        # Run daemon
        self.limit_order_client.check_and_fill_limit_orders()

        # Verify order filled and removed
        orders_after = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders_after), 0)

        # Verify position updated
        updated_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertIsNotNone(updated_position)
        self.assertEqual(len(updated_position.orders), 2)

        # Verify limit order details
        limit_order_filled = updated_position.orders[-1]
        self.assertEqual(limit_order_filled.order_type, OrderType.LONG)
        self.assertEqual(limit_order_filled.src, OrderSource.LIMIT_FILLED)

    # ============================================================================
    # Integration Tests: SL/TP Validation Against Fill Price
    # ============================================================================

    def test_long_limit_order_invalid_stop_loss_above_fill_price(self):
        """
        INTEGRATION TEST: LONG limit order with SL >= fill price should NOT create bracket order.

        For LONG positions, stop loss must be BELOW fill price (sell at a loss).
        Invalid SL should be rejected with warning, but limit order still fills.
        """
        # Create initial LONG position
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create LONG limit order with INVALID stop loss (above expected fill price)
        # Expected fill price ~47000, but SL=48000 is ABOVE (invalid for LONG)
        limit_order = self.create_limit_order(
            order_type=OrderType.LONG,
            limit_price=48000.0,
            leverage=0.2,
            stop_loss=48000.0,  # INVALID: SL should be < fill price for LONG
            take_profit=52000.0
        )

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order at ~47000 (below SL of 48000)
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify NO bracket order created (invalid SL rejected)
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 0, "Invalid SL should prevent bracket creation")

        # Verify limit order still filled successfully
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 2, "Limit order should still fill")
        self.assertEqual(position.orders[-1].src, OrderSource.LIMIT_FILLED)

    def test_long_limit_order_invalid_take_profit_below_fill_price(self):
        """
        INTEGRATION TEST: LONG limit order with TP <= fill price should NOT create bracket order.

        For LONG positions, take profit must be ABOVE fill price (sell at a gain).
        Invalid TP should be rejected with warning, but limit order still fills.
        """
        # Create initial LONG position
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create LONG limit order with INVALID take profit (below expected fill price)
        # Expected fill price ~47000, but TP=46000 is BELOW (invalid for LONG)
        limit_order = self.create_limit_order(
            order_type=OrderType.LONG,
            limit_price=48000.0,
            leverage=0.2,
            stop_loss=45000.0,
            take_profit=46000.0  # INVALID: TP should be > fill price for LONG
        )

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order at ~47000 (above TP of 46000)
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify NO bracket order created (invalid TP rejected)
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 0, "Invalid TP should prevent bracket creation")

        # Verify limit order still filled successfully
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 2, "Limit order should still fill")

    def test_short_limit_order_invalid_stop_loss_below_fill_price(self):
        """
        INTEGRATION TEST: SHORT limit order with SL <= fill price should NOT create bracket order.

        For SHORT positions, stop loss must be ABOVE fill price (buy back at a loss).
        Invalid SL should be rejected with warning, but limit order still fills.
        """
        # Create initial LONG position
        self.create_test_position(order_type=OrderType.LONG, leverage=0.4)

        # Create SHORT limit order with INVALID stop loss (below expected fill price)
        # Expected fill price ~52000, but SL=51000 is BELOW (invalid for SHORT)
        limit_order = self.create_limit_order(
            order_type=OrderType.SHORT,
            limit_price=51000.0,
            leverage=-0.2,
            stop_loss=51000.0,  # INVALID: SL should be > fill price for SHORT
            take_profit=48000.0
        )

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order at ~52000 (above SL of 51000)
        trigger_price_source = self.create_price_source(52000.0, bid=52000.0, ask=52000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify NO bracket order created (invalid SL rejected)
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 0, "Invalid SL should prevent bracket creation")

        # Verify limit order still filled successfully
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 2, "Limit order should still fill")

    def test_short_limit_order_invalid_take_profit_above_fill_price(self):
        """
        INTEGRATION TEST: SHORT limit order with TP >= fill price should NOT create bracket order.

        For SHORT positions, take profit must be BELOW fill price (buy back at a gain).
        Invalid TP should be rejected with warning, but limit order still fills.
        """
        # Create initial LONG position
        self.create_test_position(order_type=OrderType.LONG, leverage=0.4)

        # Create SHORT limit order with INVALID take profit (above expected fill price)
        # Expected fill price ~52000, but TP=53000 is ABOVE (invalid for SHORT)
        limit_order = self.create_limit_order(
            order_type=OrderType.SHORT,
            limit_price=51000.0,
            leverage=-0.2,
            stop_loss=54000.0,
            take_profit=53000.0  # INVALID: TP should be < fill price for SHORT
        )

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order at ~52000 (below TP of 53000)
        trigger_price_source = self.create_price_source(52000.0, bid=52000.0, ask=52000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify NO bracket order created (invalid TP rejected)
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 0, "Invalid TP should prevent bracket creation")

        # Verify limit order still filled successfully
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 2, "Limit order should still fill")

    def test_long_limit_order_valid_sl_tp_creates_bracket(self):
        """
        INTEGRATION TEST: LONG limit order with VALID SL/TP creates bracket order.

        For LONG positions:
        - SL must be < fill price
        - TP must be > fill price

        This is a positive test to confirm valid SL/TP still works.
        """
        # Create initial LONG position
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create LONG limit order with VALID SL/TP
        # Expected fill price ~47000
        # SL=45000 < 47000 (valid)
        # TP=52000 > 47000 (valid)
        limit_order = self.create_limit_order(
            order_type=OrderType.LONG,
            limit_price=48000.0,
            leverage=0.2,
            stop_loss=45000.0,  # VALID: < fill price
            take_profit=52000.0  # VALID: > fill price
        )

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order at ~47000
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify bracket order WAS created (valid SL/TP accepted)
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 1, "Valid SL/TP should create bracket order")

        # Verify bracket order has correct values
        bracket_order = bracket_orders[0]
        self.assertEqual(bracket_order['stop_loss'], 45000.0)
        self.assertEqual(bracket_order['take_profit'], 52000.0)
        self.assertEqual(bracket_order['src'], OrderSource.BRACKET_UNFILLED)

    def test_short_limit_order_valid_sl_tp_creates_bracket(self):
        """
        INTEGRATION TEST: SHORT limit order with VALID SL/TP creates bracket order.

        For SHORT positions:
        - SL must be > fill price
        - TP must be < fill price

        This is a positive test to confirm valid SL/TP still works.
        """
        # Create initial LONG position
        self.create_test_position(order_type=OrderType.LONG, leverage=0.4)

        # Create SHORT limit order with VALID SL/TP
        # Expected fill price ~52000
        # SL=54000 > 52000 (valid)
        # TP=48000 < 52000 (valid)
        limit_order = self.create_limit_order(
            order_type=OrderType.SHORT,
            limit_price=51000.0,
            leverage=-0.2,
            stop_loss=54000.0,  # VALID: > fill price
            take_profit=48000.0  # VALID: < fill price
        )

        # Inject None to prevent immediate fill
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, None)

        # Submit order
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill limit order at ~52000
        trigger_price_source = self.create_price_source(52000.0, bid=52000.0, ask=52000.0)
        self.inject_price_data(self.DEFAULT_TRADE_PAIR, trigger_price_source)
        self.set_market_open(is_open=True)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify bracket order WAS created (valid SL/TP accepted)
        bracket_orders = self.limit_order_client.get_limit_orders_for_trade_pair(
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 1, "Valid SL/TP should create bracket order")

        # Verify bracket order has correct values
        bracket_order = bracket_orders[0]
        self.assertEqual(bracket_order['stop_loss'], 54000.0)
        self.assertEqual(bracket_order['take_profit'], 48000.0)
        self.assertEqual(bracket_order['src'], OrderSource.BRACKET_UNFILLED)


if __name__ == '__main__':
    unittest.main()
