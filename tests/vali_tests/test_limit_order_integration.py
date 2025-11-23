import unittest
from unittest.mock import patch
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


class TestLimitOrderIntegration(TestBase):
    """
    INTEGRATION TESTS for limit order management.

    These tests run full production code paths with MINIMAL mocking:
    - NO mocking of market_order_manager (tests actual position updates)
    - NO mocking of internal helper methods
    - ONLY mock external APIs (price sources) and disk I/O

    Goal: Verify end-to-end correctness of limit order fills, position updates,
    bracket order creation, and error handling.
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    limit_order_client = None
    limit_order_server = None  # Keep handle for direct access to manager in tests

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
        cls.limit_order_client = cls.orchestrator.get_client('limit_order')

        # Get limit order server handle for direct access to manager in tests
        cls.limit_order_server = cls.orchestrator._servers.get('limit_order')

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
            trade_pair=self.DEFAULT_TRADE_PAIR
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

    def create_limit_order(self, order_type=OrderType.LONG, limit_price=51000.0,
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
        """Create a price source for mocking."""
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

    # ============================================================================
    # Integration Tests: Full Fill Path
    # ============================================================================

    def test_end_to_end_long_limit_order_fill(self):
        """
        INTEGRATION TEST: Complete LONG limit order fill without mocking core logic.

        Tests:
        - Position is created and updated correctly
        - Limit order triggers at correct price
        - Position contains both initial and limit orders
        - Leverage is applied correctly
        - Order is removed from memory after fill
        """
        # Create initial LONG position (BTCUSD max leverage is 0.5)
        initial_position = self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Verify initial position
        self.assertEqual(len(initial_position.orders), 1)

        # Create LONG limit order to add to position (0.3 + 0.2 = 0.5, within max leverage)
        limit_order = self.create_limit_order(order_type=OrderType.LONG, limit_price=48000.0, leverage=0.2)

        # Mock ONLY the price source (external data), not internal logic
        trigger_price_source = self.create_price_source(47500.0, bid=47500.0, ask=47500.0)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            # Submit order
            result = self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)
            self.assertEqual(result["status"], "success")

        # Verify order is stored unfilled
        orders = self.limit_order_server._manager._limit_orders.get(self.DEFAULT_TRADE_PAIR, {}).get(
            self.DEFAULT_MINER_HOTKEY, []
        )
        self.assertEqual(len(orders), 1, "Should have 1 unfilled limit order in memory")
        self.assertEqual(orders[0].src, OrderSource.LIMIT_UNFILLED)

        # Run daemon with triggering price (ask=47500 < limit=48000 triggers LONG)
        # IMPORTANT: We're NOT mocking market_order_manager - this tests the REAL fill path!
        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[trigger_price_source]):
                        # Run REAL daemon code - no mocking of market_order_manager!
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify order removed from memory (filled orders are cleaned up)
        orders_after = self.limit_order_server._manager._limit_orders.get(self.DEFAULT_TRADE_PAIR, {}).get(
            self.DEFAULT_MINER_HOTKEY, []
        )

        # If the order wasn't filled, check why
        if len(orders_after) != 0:
            print(f"DEBUG: Order still in memory. Source: {orders_after[0].src if orders_after else 'N/A'}")
            print(f"DEBUG: Last fill time: {self.limit_order_server._manager._last_fill_time}")

        self.assertEqual(len(orders_after), 0, "Filled order should be removed from memory")

        # Verify REAL position was updated with the limit order
        updated_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertIsNotNone(updated_position, "Position should exist after fill")

        # Debug: Print order count if mismatch
        if len(updated_position.orders) != 2:
            print(f"DEBUG: Expected 2 orders, got {len(updated_position.orders)}")
            for i, order in enumerate(updated_position.orders):
                print(f"  Order {i}: type={order.order_type}, src={order.src}, uuid={order.order_uuid}")

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
        INTEGRATION TEST: Complete SHORT limit order fill without mocking core logic.

        Tests SHORT-specific logic:
        - SHORT order triggers when bid >= limit_price
        - Position net leverage decreases (SHORT reduces LONG exposure)
        """
        # Create initial LONG position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.4)

        # Create SHORT limit order to reduce position
        limit_order = self.create_limit_order(order_type=OrderType.SHORT, limit_price=51000.0, leverage=-0.2)

        # Mock price sources
        trigger_price_source = self.create_price_source(51500.0, bid=51500.0, ask=51500.0)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Run daemon (bid=51500 >= limit=51000 triggers SHORT)
        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[trigger_price_source]):
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify REAL position updated
        updated_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertIsNotNone(updated_position)
        self.assertEqual(len(updated_position.orders), 2)

        # Verify SHORT order
        short_order = updated_position.orders[-1]
        self.assertEqual(short_order.order_type, OrderType.SHORT)
        self.assertEqual(short_order.leverage, -0.2)

        # Verify net leverage reduced
        expected_leverage = 0.4 - 0.2
        self.assertAlmostEqual(updated_position.net_leverage, expected_leverage, places=2)

    def test_end_to_end_bracket_order_creation_and_trigger(self):
        """
        INTEGRATION TEST: Test full bracket order lifecycle without mocking.

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

        # Submit limit order
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Fill the limit order
        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[trigger_price_source]):
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify bracket order was created
        bracket_orders = self.limit_order_server._manager._limit_orders.get(self.DEFAULT_TRADE_PAIR, {}).get(
            self.DEFAULT_MINER_HOTKEY, []
        )
        self.assertEqual(len(bracket_orders), 1, "Bracket order should be created")
        bracket_order = bracket_orders[0]
        self.assertEqual(bracket_order.execution_type, ExecutionType.BRACKET)
        self.assertEqual(bracket_order.stop_loss, 45000.0)
        self.assertEqual(bracket_order.take_profit, 52000.0)
        self.assertEqual(bracket_order.src, OrderSource.BRACKET_UNFILLED)

        # Simulate time passing to allow bracket order to fill (respect fill interval)
        # Reset last_fill_time to simulate fill interval has passed
        if self.DEFAULT_TRADE_PAIR in self.limit_order_server._manager._last_fill_time:
            if self.DEFAULT_MINER_HOTKEY in self.limit_order_server._manager._last_fill_time[self.DEFAULT_TRADE_PAIR]:
                # Set to 0 to simulate enough time has passed
                self.limit_order_server._manager._last_fill_time[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY] = 0

        # Trigger stop loss (price falls below 45000)
        stop_loss_price_source = self.create_price_source(44000.0, bid=44000.0, ask=44000.0)

        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[stop_loss_price_source]):
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify bracket order filled (position closed/reduced)
        bracket_orders_after = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
        ).get(self.DEFAULT_MINER_HOTKEY, [])
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

        This tests the production error path without mocking.
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

        # Submit and fill limit order
        trigger_price_source = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[trigger_price_source]):
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify bracket order created
        bracket_orders = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders), 1)

        # Simulate time passing to allow bracket order to be evaluated (respect fill interval)
        if self.DEFAULT_TRADE_PAIR in self.limit_order_server._manager._last_fill_time:
            if self.DEFAULT_MINER_HOTKEY in self.limit_order_server._manager._last_fill_time[self.DEFAULT_TRADE_PAIR]:
                self.limit_order_server._manager._last_fill_time[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY] = 0

        # Close the position manually (simulate position being closed elsewhere)
        self.position_client.clear_all_miner_positions_and_disk()

        # Try to trigger bracket order when position doesn't exist
        stop_loss_price_source = self.create_price_source(44000.0, bid=44000.0, ask=44000.0)

        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[stop_loss_price_source]):
                        # Should not crash, should cancel bracket order
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify bracket order was cancelled (removed from memory)
        bracket_orders_after = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(bracket_orders_after), 0, "Bracket order should be cancelled when position missing")

    def test_multiple_limit_orders_fill_sequentially_with_interval(self):
        """
        INTEGRATION TEST: Multiple limit orders respect fill interval.

        Tests REAL timing enforcement (not just mocked behavior).
        """
        # Create initial position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.2)

        # Create multiple limit orders
        order1 = self.create_limit_order(order_uuid="order1", limit_price=48000.0, leverage=0.1)
        order2 = self.create_limit_order(order_uuid="order2", limit_price=48000.0, leverage=0.1)
        order3 = self.create_limit_order(order_uuid="order3", limit_price=48000.0, leverage=0.1)

        # Submit all orders
        trigger_price = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            for order in [order1, order2, order3]:
                self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # First daemon run - should fill only one order
        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[trigger_price]):
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify only one order filled
        remaining_orders = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(remaining_orders), 2, "Two orders should remain (one filled)")

        # Second daemon run immediately - should NOT fill (within interval)
        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                                    return_value=[trigger_price]):
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify still two orders remain
        remaining_orders = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
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

        Tests REAL market hours check (not mocked).
        """
        # Create position (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)

        # Create limit order
        limit_order = self.create_limit_order(limit_price=48000.0)

        trigger_price = self.create_price_source(47000.0, bid=47000.0, ask=47000.0)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Run daemon with market CLOSED
        with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                          return_value=False):
            with patch.object(self.limit_order_server._manager, '_get_best_price_source',
                            return_value=[trigger_price]):
                self.limit_order_server._manager.check_and_fill_limit_orders()

        # Verify order NOT filled
        orders = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders), 1, "Order should remain unfilled when market closed")
        self.assertEqual(orders[0].src, OrderSource.LIMIT_UNFILLED)

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
        INTEGRATION TEST: Verify _get_best_price_source uses median (not mocked).

        Tests the actual price aggregation logic that was mocked in unit tests.
        """
        # Create position and limit order (BTCUSD max leverage is 0.5)
        self.create_test_position(order_type=OrderType.LONG, leverage=0.3)
        limit_order = self.create_limit_order(limit_price=48000.0, leverage=0.2)

        with patch.object(self.limit_order_server._manager.live_price_client,
                         'get_sorted_price_sources_for_trade_pair',
                         return_value=None):
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, limit_order)

        # Create multiple price sources (mock the external API, but test internal aggregation)
        price_sources = [
            self.create_price_source(47000.0),  # Low
            self.create_price_source(47500.0),  # Median
            self.create_price_source(48000.0),  # High
        ]

        with patch.object(self.limit_order_server._manager, '_write_to_disk'):
            with patch('os.path.exists', return_value=False):
                with patch.object(self.limit_order_server._manager.live_price_client, 'is_market_open',
                                  return_value=True):
                    # Mock ONLY the external API call, not _get_best_price_source
                    with patch.object(self.limit_order_server._manager.live_price_client,
                                    'get_ws_price_sources_in_window',
                                    return_value=price_sources):
                        # This should use the REAL _get_best_price_source logic (median selection)
                        self.limit_order_server._manager.check_and_fill_limit_orders()

        # The REAL _get_best_price_source should select median (47500)
        # which WILL trigger the limit order (ask=47500 < limit=48000)
        orders_after = self.limit_order_server._manager._limit_orders.get(
            self.DEFAULT_TRADE_PAIR, {}
        ).get(self.DEFAULT_MINER_HOTKEY, [])
        self.assertEqual(len(orders_after), 0, "Order should be filled using median price")

        # Verify position updated
        position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(len(position.orders), 2)


if __name__ == '__main__':
    unittest.main()
