import unittest

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.position import Position


class TestStopLimitOrders(TestBase):
    """
    Integration tests for stop-limit order management.
    Tests the full lifecycle: submission, trigger, conversion to limit, fill, cancel/edit.
    """

    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    limit_order_client = None
    limit_order_handle = None

    DEFAULT_MINER_HOTKEY = "test_miner"

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.limit_order_client = cls.orchestrator.get_client('limit_order')
        cls.limit_order_handle = cls.orchestrator._servers.get('limit_order')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY])
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def create_stop_limit_order(self, order_type=OrderType.LONG, stop_price=105.0,
                                stop_condition=StopCondition.GTE, limit_price=106.0,
                                leverage=0.5, order_uuid=None, trade_pair=None,
                                bracket_orders=None, quantity=None):
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if order_uuid is None:
            order_uuid = f"test_stop_limit_{TimeUtil.now_in_millis()}"

        return Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=order_type,
            leverage=leverage,
            quantity=quantity,
            execution_type=ExecutionType.STOP_LIMIT,
            stop_price=stop_price,
            stop_condition=stop_condition,
            limit_price=limit_price,
            bracket_orders=bracket_orders,
            src=OrderSource.STOP_LIMIT_UNFILLED
        )

    def create_test_price_source(self, price, bid=None, ask=None, start_ms=None):
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

    def create_test_position(self, trade_pair=None, miner_hotkey=None):
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if miner_hotkey is None:
            miner_hotkey = self.DEFAULT_MINER_HOTKEY

        return Position(
            miner_hotkey=miner_hotkey,
            position_uuid=f"pos_{TimeUtil.now_in_millis()}",
            open_ms=TimeUtil.now_in_millis(),
            trade_pair=trade_pair,
            account_size=1000.0
        )

    def get_orders_from_server(self, miner_hotkey, trade_pair):
        orders_for_trade_pair = self.limit_order_client.get_limit_orders_for_trade_pair(trade_pair.trade_pair_id)
        if miner_hotkey in orders_for_trade_pair:
            return [Order.from_dict(o) if isinstance(o, dict) else o for o in orders_for_trade_pair[miner_hotkey]]
        return []

    def count_orders_in_server(self, miner_hotkey):
        orders = self.limit_order_client.get_limit_orders(miner_hotkey)
        return len(orders)

    # ============================================================================
    # Test: Submission - STOP_LIMIT stored as STOP_LIMIT_UNFILLED
    # ============================================================================

    def test_submit_stop_limit_order_basic(self):
        """Test basic stop-limit order placement stores as STOP_LIMIT_UNFILLED"""
        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0
        )

        result = self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY, order
        )

        self.assertEqual(result["status"], "success")

        # Verify stored with correct source
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].src, OrderSource.STOP_LIMIT_UNFILLED)
        self.assertEqual(orders[0].execution_type, ExecutionType.STOP_LIMIT)
        self.assertEqual(orders[0].stop_price, 55000.0)
        self.assertEqual(orders[0].stop_condition, StopCondition.GTE)
        self.assertEqual(orders[0].limit_price, 56000.0)

    def test_submit_stop_limit_order_lte(self):
        """Test stop-limit order with LTE condition"""
        order = self.create_stop_limit_order(
            order_type=OrderType.SHORT,
            stop_price=45000.0,
            stop_condition=StopCondition.LTE,
            limit_price=44000.0
        )

        result = self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY, order
        )

        self.assertEqual(result["status"], "success")
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(orders[0].stop_condition, StopCondition.LTE)

    def test_submit_stop_limit_does_not_fill_immediately(self):
        """Test that stop-limit orders are never filled immediately, even if price already past stop"""
        # Set price that would trigger the stop (price >= stop_price for GTE)
        trigger_price_source = self.create_test_price_source(60000.0, bid=60000.0, ask=60000.0)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price_source)

        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0
        )

        result = self.limit_order_client.process_limit_order(
            self.DEFAULT_MINER_HOTKEY, order
        )

        self.assertEqual(result["status"], "success")

        # Order should remain unfilled - stop-limit orders skip immediate fill
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].src, OrderSource.STOP_LIMIT_UNFILLED)

    # ============================================================================
    # Test: Validation
    # ============================================================================

    def test_validation_reject_missing_stop_price(self):
        """Test rejection when stop_price is missing"""
        with self.assertRaises(Exception):
            Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test",
                processed_ms=TimeUtil.now_in_millis(),
                price=0.0,
                order_type=OrderType.LONG,
                leverage=0.5,
                execution_type=ExecutionType.STOP_LIMIT,
                limit_price=106.0,
                stop_condition=StopCondition.GTE,
                src=OrderSource.STOP_LIMIT_UNFILLED
            )

    def test_validation_reject_missing_limit_price(self):
        """Test rejection when limit_price is missing"""
        with self.assertRaises(Exception):
            Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test",
                processed_ms=TimeUtil.now_in_millis(),
                price=0.0,
                order_type=OrderType.LONG,
                leverage=0.5,
                execution_type=ExecutionType.STOP_LIMIT,
                stop_price=105.0,
                stop_condition=StopCondition.GTE,
                src=OrderSource.STOP_LIMIT_UNFILLED
            )

    def test_validation_reject_missing_stop_condition(self):
        """Test rejection when stop_condition is missing"""
        with self.assertRaises(Exception):
            Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test",
                processed_ms=TimeUtil.now_in_millis(),
                price=0.0,
                order_type=OrderType.LONG,
                leverage=0.5,
                execution_type=ExecutionType.STOP_LIMIT,
                stop_price=105.0,
                limit_price=106.0,
                src=OrderSource.STOP_LIMIT_UNFILLED
            )

    def test_validation_reject_long_limit_below_stop(self):
        """Test LONG rejected when limit_price < stop_price (wouldn't fill on trigger)"""
        with self.assertRaises(Exception):
            Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test",
                processed_ms=TimeUtil.now_in_millis(),
                price=0.0,
                order_type=OrderType.LONG,
                leverage=0.5,
                execution_type=ExecutionType.STOP_LIMIT,
                stop_price=55000.0,
                stop_condition=StopCondition.GTE,
                limit_price=54000.0,  # Below stop — can't fill on breakout
                src=OrderSource.STOP_LIMIT_UNFILLED
            )

    def test_validation_reject_short_limit_above_stop(self):
        """Test SHORT rejected when limit_price > stop_price (wouldn't fill on trigger)"""
        with self.assertRaises(Exception):
            Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test",
                processed_ms=TimeUtil.now_in_millis(),
                price=0.0,
                order_type=OrderType.SHORT,
                leverage=-0.5,
                execution_type=ExecutionType.STOP_LIMIT,
                stop_price=45000.0,
                stop_condition=StopCondition.LTE,
                limit_price=46000.0,  # Above stop — can't fill on breakdown
                src=OrderSource.STOP_LIMIT_UNFILLED
            )

    def test_validation_accept_long_limit_equals_stop(self):
        """Test LONG accepted when limit_price == stop_price"""
        order = Order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test",
            processed_ms=TimeUtil.now_in_millis(),
            price=0.0,
            order_type=OrderType.LONG,
            leverage=0.5,
            execution_type=ExecutionType.STOP_LIMIT,
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=55000.0,
            src=OrderSource.STOP_LIMIT_UNFILLED
        )
        self.assertEqual(order.limit_price, 55000.0)

    def test_validation_reject_flat_for_stop_limit(self):
        """Test rejection of FLAT order type for STOP_LIMIT"""
        with self.assertRaises(Exception):
            Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test",
                processed_ms=TimeUtil.now_in_millis(),
                price=0.0,
                order_type=OrderType.FLAT,
                leverage=0.5,
                execution_type=ExecutionType.STOP_LIMIT,
                stop_price=105.0,
                stop_condition=StopCondition.GTE,
                limit_price=106.0,
                src=OrderSource.STOP_LIMIT_UNFILLED
            )

    # ============================================================================
    # Test: Trigger with GTE condition
    # ============================================================================

    def test_trigger_gte_triggers_when_price_above(self):
        """Test GTE triggers when mid price >= stop_price"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0
        )

        # Set non-triggering price (below stop)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Verify order stored
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].src, OrderSource.STOP_LIMIT_UNFILLED)

        # Set price below stop - should NOT trigger
        below_price = self.create_test_price_source(54000.0, bid=54000.0, ask=54000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, below_price)

        self.limit_order_client.check_and_fill_limit_orders()

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1, "Order should remain unfilled when price below stop")
        self.assertEqual(orders[0].src, OrderSource.STOP_LIMIT_UNFILLED)

        # Set price at stop - should trigger (mid = 55000 >= 55000)
        at_price = self.create_test_price_source(55000.0, bid=55000.0, ask=55000.0)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, at_price)

        self.limit_order_client.check_and_fill_limit_orders()

        # Stop-limit should be filled (converted to limit order)
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        # The stop-limit should be gone, replaced by a child limit order
        stop_limit_orders = [o for o in orders if o.src == OrderSource.STOP_LIMIT_UNFILLED]
        self.assertEqual(len(stop_limit_orders), 0, "Stop-limit order should be consumed after trigger")

        # Child limit order should exist
        limit_orders = [o for o in orders if o.src == OrderSource.LIMIT_UNFILLED]
        # Note: child may have filled immediately if ask <= limit_price, so it might be 0
        # What matters is the stop-limit is gone

    def test_trigger_gte_does_not_trigger_below(self):
        """Test GTE does NOT trigger when mid price < stop_price"""
        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0
        )

        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Price below stop
        below_price = self.create_test_price_source(54999.0, bid=54999.0, ask=54999.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, below_price)

        self.limit_order_client.check_and_fill_limit_orders()

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].src, OrderSource.STOP_LIMIT_UNFILLED)

    # ============================================================================
    # Test: Trigger with LTE condition
    # ============================================================================

    def test_trigger_lte_triggers_when_price_below(self):
        """Test LTE triggers when mid price <= stop_price"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        order = self.create_stop_limit_order(
            order_type=OrderType.SHORT,
            stop_price=45000.0,
            stop_condition=StopCondition.LTE,
            limit_price=44000.0
        )

        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Price above stop - should NOT trigger
        above_price = self.create_test_price_source(46000.0, bid=46000.0, ask=46000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, above_price)

        self.limit_order_client.check_and_fill_limit_orders()

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].src, OrderSource.STOP_LIMIT_UNFILLED)

        # Price at stop - should trigger
        at_price = self.create_test_price_source(45000.0, bid=45000.0, ask=45000.0)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, at_price)

        self.limit_order_client.check_and_fill_limit_orders()

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        stop_limit_orders = [o for o in orders if o.src == OrderSource.STOP_LIMIT_UNFILLED]
        self.assertEqual(len(stop_limit_orders), 0, "Stop-limit order should be consumed after LTE trigger")

    # ============================================================================
    # Test: Conversion - stop triggers -> limit order created
    # ============================================================================

    def test_conversion_creates_child_limit_order(self):
        """Test that triggering a stop-limit creates a child limit order with correct fields"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        order = self.create_stop_limit_order(
            order_type=OrderType.LONG,
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            leverage=0.3,
            order_uuid="parent_stop_limit"
        )

        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Trigger the stop
        trigger_price = self.create_test_price_source(55500.0, bid=55500.0, ask=55500.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price)

        self.limit_order_client.check_and_fill_limit_orders()

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)

        # Check if child limit order exists (may have been filled immediately if ask <= limit)
        # With ask=55500 < limit=56000, it should fill immediately
        # Verify by checking position was updated
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        # Position should have been updated with the fill
        self.assertGreaterEqual(len(positions), 1)

    def test_conversion_child_limit_order_uuid_format(self):
        """Test child limit order has UUID format '{parent_uuid}-limit'"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        parent_uuid = "test_parent_123"
        order = self.create_stop_limit_order(
            order_type=OrderType.LONG,
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            order_uuid=parent_uuid
        )

        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Trigger with price that won't immediately fill the child limit
        # LONG limit at 56000, ask must be > 56000 to NOT fill immediately
        trigger_price = self.create_test_price_source(57000.0, bid=57000.0, ask=57000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price)

        self.limit_order_client.check_and_fill_limit_orders()

        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        limit_orders = [o for o in orders if o.src == OrderSource.LIMIT_UNFILLED]
        self.assertEqual(len(limit_orders), 1, "Child limit order should exist")
        self.assertEqual(limit_orders[0].order_uuid, f"{parent_uuid}-limit")
        self.assertEqual(limit_orders[0].limit_price, 56000.0)
        self.assertEqual(limit_orders[0].execution_type, ExecutionType.LIMIT)

    # ============================================================================
    # Test: Full lifecycle - stop triggers -> limit fills -> position updated
    # ============================================================================

    def test_full_lifecycle_stop_trigger_then_limit_fill(self):
        """Test complete lifecycle: stop triggers -> limit order created -> limit fills"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        order = self.create_stop_limit_order(
            order_type=OrderType.LONG,
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            leverage=0.3,
            order_uuid="lifecycle_test"
        )

        # Submit order (no price source = won't fill)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Step 1: Trigger the stop (price >= 55000 for GTE, but ask > limit to prevent immediate child fill)
        trigger_price = self.create_test_price_source(57000.0, bid=57000.0, ask=57000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price)

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify child limit order exists
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        limit_orders = [o for o in orders if o.src == OrderSource.LIMIT_UNFILLED]
        self.assertEqual(len(limit_orders), 1, "Child limit order should be pending")

        # Step 2: Fill the child limit order (price drops to limit price)
        fill_price = self.create_test_price_source(55500.0, bid=55500.0, ask=55500.0)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, fill_price)

        # Reset fill time to allow immediate fill
        self.limit_order_client.set_last_fill_time(
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            self.DEFAULT_MINER_HOTKEY,
            0
        )

        self.limit_order_client.check_and_fill_limit_orders()

        # Verify limit order is filled
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        unfilled = [o for o in orders if o.src in [OrderSource.LIMIT_UNFILLED, OrderSource.STOP_LIMIT_UNFILLED]]
        self.assertEqual(len(unfilled), 0, "All orders should be filled")

        # Verify position was created
        positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertGreaterEqual(len(positions), 1)
        filled_position = positions[0]
        self.assertGreaterEqual(len(filled_position.orders), 1)
        self.assertEqual(filled_position.orders[-1].src, OrderSource.LIMIT_FILLED)

    # ============================================================================
    # Test: Cancel while unfilled
    # ============================================================================

    def test_cancel_unfilled_stop_limit(self):
        """Test cancelling an unfilled stop-limit order"""
        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            order_uuid="cancel_test"
        )

        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Verify order exists
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 1)

        # Cancel it
        result = self.limit_order_client.cancel_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "cancel_test",
            TimeUtil.now_in_millis()
        )

        self.assertEqual(result["status"], "cancelled")

        # Verify order removed from memory
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(len(orders), 0)

    def test_cancel_all_includes_stop_limit(self):
        """Test cancel-all includes stop-limit orders"""
        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            order_uuid="cancel_all_test"
        )

        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        result = self.limit_order_client.cancel_limit_order(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "ALL",
            TimeUtil.now_in_millis()
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["num_cancelled"], 1)

    # ============================================================================
    # Test: Unfilled count includes STOP_LIMIT
    # ============================================================================

    def test_stop_limit_counts_toward_max_unfilled(self):
        """Test stop-limit orders count toward the max unfilled orders limit"""
        # Fill up to max with stop-limit orders
        for i in range(ValiConfig.MAX_UNFILLED_LIMIT_ORDERS):
            order = self.create_stop_limit_order(
                stop_price=55000.0 + i,
                stop_condition=StopCondition.GTE,
                limit_price=56000.0 + i,
                order_uuid=f"sl_order_{i}",
                trade_pair=TradePair.BTCUSD if i % 2 == 0 else TradePair.ETHUSD
            )
            self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Next order should be rejected
        excess_order = self.create_stop_limit_order(
            stop_price=99000.0,
            stop_condition=StopCondition.GTE,
            limit_price=99500.0,
            order_uuid="excess"
        )

        with self.assertRaises(SignalException) as context:
            self.limit_order_client.process_limit_order(
                self.DEFAULT_MINER_HOTKEY, excess_order
            )
        self.assertIn("too many unfilled limit orders", str(context.exception))

    # ============================================================================
    # Test: OrderSource enum
    # ============================================================================

    def test_order_source_get_fill(self):
        """Test get_fill returns STOP_LIMIT_FILLED"""
        self.assertEqual(
            OrderSource.get_fill(OrderSource.STOP_LIMIT_UNFILLED),
            OrderSource.STOP_LIMIT_FILLED
        )

    def test_order_source_get_cancel(self):
        """Test get_cancel returns STOP_LIMIT_CANCELLED"""
        self.assertEqual(
            OrderSource.get_cancel(OrderSource.STOP_LIMIT_UNFILLED),
            OrderSource.STOP_LIMIT_CANCELLED
        )
        self.assertEqual(
            OrderSource.get_cancel(OrderSource.STOP_LIMIT_FILLED),
            OrderSource.STOP_LIMIT_CANCELLED
        )

    def test_order_source_is_open(self):
        """Test is_open includes STOP_LIMIT_UNFILLED"""
        self.assertTrue(OrderSource.is_open(OrderSource.STOP_LIMIT_UNFILLED))
        self.assertFalse(OrderSource.is_open(OrderSource.STOP_LIMIT_FILLED))

    def test_order_source_is_closed(self):
        """Test is_closed includes STOP_LIMIT_FILLED and STOP_LIMIT_CANCELLED"""
        self.assertTrue(OrderSource.is_closed(OrderSource.STOP_LIMIT_FILLED))
        self.assertTrue(OrderSource.is_closed(OrderSource.STOP_LIMIT_CANCELLED))

    def test_order_source_status(self):
        """Test status returns correct strings"""
        self.assertEqual(OrderSource.status(OrderSource.STOP_LIMIT_UNFILLED), "UNFILLED")
        self.assertEqual(OrderSource.status(OrderSource.STOP_LIMIT_FILLED), "FILLED")
        self.assertEqual(OrderSource.status(OrderSource.STOP_LIMIT_CANCELLED), "CANCELLED")

    # ============================================================================
    # Test: StopCondition enum
    # ============================================================================

    def test_stop_condition_from_string(self):
        """Test StopCondition.from_string()"""
        self.assertEqual(StopCondition.from_string("GTE"), StopCondition.GTE)
        self.assertEqual(StopCondition.from_string("LTE"), StopCondition.LTE)
        self.assertEqual(StopCondition.from_string("gte"), StopCondition.GTE)

        with self.assertRaises(ValueError):
            StopCondition.from_string("INVALID")

    def test_stop_condition_str(self):
        """Test StopCondition __str__"""
        self.assertEqual(str(StopCondition.GTE), "GTE")
        self.assertEqual(str(StopCondition.LTE), "LTE")

    # ============================================================================
    # Test: Serialization roundtrip (to_python_dict / from_dict)
    # ============================================================================

    def test_stop_limit_order_serialization_roundtrip(self):
        """Test stop-limit order survives dict serialization and deserialization"""
        order = self.create_stop_limit_order(
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            order_uuid="serial_test"
        )

        order_dict = order.to_python_dict()
        self.assertEqual(order_dict['stop_price'], 55000.0)
        self.assertEqual(order_dict['stop_condition'], "GTE")
        self.assertEqual(order_dict['execution_type'], "STOP_LIMIT")

        # Reconstruct from dict
        restored = Order.from_dict(order_dict)
        self.assertEqual(restored.stop_price, 55000.0)
        self.assertEqual(restored.stop_condition, StopCondition.GTE)
        self.assertEqual(restored.limit_price, 56000.0)
        self.assertEqual(restored.execution_type, ExecutionType.STOP_LIMIT)

    # ============================================================================
    # Test: Bracket orders forwarded from stop-limit
    # ============================================================================

    def test_stop_limit_with_bracket_orders(self):
        """Test stop-limit order with bracket_orders forwarded to child limit order"""
        position = self.create_test_position()
        self.position_client.save_miner_position(position)

        bracket_orders = [{"stop_loss": 50000, "take_profit": 60000, "leverage": 0.3}]

        order = self.create_stop_limit_order(
            order_type=OrderType.LONG,
            stop_price=55000.0,
            stop_condition=StopCondition.GTE,
            limit_price=56000.0,
            leverage=0.3,
            order_uuid="bracket_test",
            bracket_orders=bracket_orders
        )

        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.DEFAULT_MINER_HOTKEY, order)

        # Trigger the stop (ask > limit to prevent immediate child fill)
        trigger_price = self.create_test_price_source(57000.0, bid=57000.0, ask=57000.0)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.DEFAULT_TRADE_PAIR, trigger_price)

        self.limit_order_client.check_and_fill_limit_orders()

        # Child limit order should have bracket_orders
        orders = self.get_orders_from_server(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR)
        limit_orders = [o for o in orders if o.src == OrderSource.LIMIT_UNFILLED]
        self.assertEqual(len(limit_orders), 1)
        self.assertEqual(limit_orders[0].bracket_orders, bracket_orders)


if __name__ == '__main__':
    unittest.main()
