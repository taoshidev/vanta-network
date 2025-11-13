import unittest
from unittest.mock import Mock, MagicMock, patch
from copy import deepcopy

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.mock_classes import MockLivePriceFetcher
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.limit_order_manager import LimitOrderManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestLimitOrders(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(None, None, self.mock_metagraph)
        self.perf_ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)

        self.position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=self.elimination_manager,
            running_unit_tests=True
        )

        self.position_locks = PositionLocks({}, use_ipc=False)

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)

        # Create mock market_order_manager with required methods
        self.mock_market_order_manager = Mock()
        self.mock_market_order_manager.position_locks = self.position_locks
        self.mock_market_order_manager._process_market_order = Mock()

        self.limit_order_manager = LimitOrderManager(
            position_manager=self.position_manager,
            live_price_fetcher=self.live_price_fetcher,
            market_order_manager=self.mock_market_order_manager,
            shutdown_dict=None,
            running_unit_tests=True
        )

        self.position_manager.clear_all_miner_positions()
        self.limit_order_manager._limit_orders.clear()

        # Mock price fetcher to return None by default (no immediate fills unless explicitly mocked)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(return_value=None)

    def create_test_limit_order(self, order_type=OrderType.LONG, limit_price=49000.0,
                               trade_pair=None, leverage=0.5, order_uuid=None):
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
            execution_type=ExecutionType.LIMIT,
            limit_price=limit_price,
            src=OrderSource.ORDER_SRC_LIMIT_UNFILLED
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
            trade_pair=trade_pair
        )
        if position_type:
            position.position_type = position_type
        return position

    # ============================================================================
    # Test RPC Methods: process_limit_order_rpc
    # ============================================================================

    def test_process_limit_order_rpc_basic(self):
        """Test basic limit order placement via RPC"""
        limit_order = self.create_test_limit_order()

        result = self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            limit_order.to_python_dict()
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["order_uuid"], limit_order.order_uuid)

        # Verify stored in correct structure
        self.assertIn(self.DEFAULT_TRADE_PAIR, self.limit_order_manager._limit_orders)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR])

        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_uuid, limit_order.order_uuid)
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_UNFILLED)

    def test_process_limit_order_rpc_exceeds_maximum(self):
        """Test limit order rejection when exceeding maximum unfilled orders"""
        # Fill up to the maximum
        for i in range(ValiConfig.MAX_UNFILLED_LIMIT_ORDERS):
            limit_order = self.create_test_limit_order(
                order_uuid=f"test_order_{i}",
                trade_pair=TradePair.BTCUSD if i % 2 == 0 else TradePair.ETHUSD
            )
            self.limit_order_manager.process_limit_order_rpc(
                self.DEFAULT_MINER_HOTKEY,
                limit_order.to_python_dict()
            )

        # Attempt to add one more
        excess_order = self.create_test_limit_order(order_uuid="excess_order")

        with self.assertRaises(SignalException) as context:
            self.limit_order_manager.process_limit_order_rpc(
                self.DEFAULT_MINER_HOTKEY,
                excess_order.to_python_dict()
            )
        self.assertIn("too many unfilled limit orders", str(context.exception))

    def test_process_limit_order_rpc_flat_no_position(self):
        """Test FLAT limit order rejection when no position exists"""
        flat_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=51000.0
        )

        with self.assertRaises(SignalException) as context:
            self.limit_order_manager.process_limit_order_rpc(
                self.DEFAULT_MINER_HOTKEY,
                flat_order.to_python_dict()
            )
        self.assertIn("No position found for FLAT order", str(context.exception))

    def test_process_limit_order_rpc_flat_with_position(self):
        """Test FLAT limit order acceptance when position exists"""
        position = self.create_test_position(position_type=OrderType.LONG)
        self.position_manager.save_miner_position(position)

        flat_order = self.create_test_limit_order(
            order_type=OrderType.FLAT,
            limit_price=51000.0
        )

        result = self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            flat_order.to_python_dict()
        )

        self.assertEqual(result["status"], "success")
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].order_type, OrderType.FLAT)

    def test_process_limit_order_rpc_immediate_fill(self):
        """Test limit order is filled immediately when price already triggered"""
        # Setup position for the order
        position = self.create_test_position()
        self.position_manager.save_miner_position(position)

        # Mock market_order_manager to return successful fill
        filled_order = self.create_test_limit_order(limit_price=50000.0)
        filled_order.price = 49000.0
        filled_order.src = OrderSource.ORDER_SRC_LIMIT_FILLED

        mock_position = self.create_test_position()
        mock_position.orders = [filled_order]

        self.mock_market_order_manager._process_market_order.return_value = (None, mock_position)

        # Mock live_price_fetcher to return triggering price
        trigger_price_source = self.create_test_price_source(48500.0, bid=48500.0, ask=48500.0)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(
            return_value=[trigger_price_source]
        )

        # Create LONG order with limit price 49000 - should trigger at ask=48500
        limit_order = self.create_test_limit_order(
            order_type=OrderType.LONG,
            limit_price=49000.0
        )

        result = self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            limit_order.to_python_dict()
        )

        self.assertEqual(result["status"], "success")

        # Verify _process_market_order was called
        self.mock_market_order_manager._process_market_order.assert_called_once()

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

        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            btc_order.to_python_dict()
        )
        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            eth_order.to_python_dict()
        )

        # Verify structure
        self.assertIn(TradePair.BTCUSD, self.limit_order_manager._limit_orders)
        self.assertIn(TradePair.ETHUSD, self.limit_order_manager._limit_orders)

        btc_orders = self.limit_order_manager._limit_orders[TradePair.BTCUSD][self.DEFAULT_MINER_HOTKEY]
        eth_orders = self.limit_order_manager._limit_orders[TradePair.ETHUSD][self.DEFAULT_MINER_HOTKEY]

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

        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            order1.to_python_dict()
        )
        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            order2.to_python_dict()
        )

        # Cancel order1
        result = self.limit_order_manager.cancel_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "order1",
            TimeUtil.now_in_millis()
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["num_cancelled"], 1)

        # Verify order1 is cancelled, order2 still unfilled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        order1_in_list = next(o for o in orders if o.order_uuid == "order1")
        order2_in_list = next(o for o in orders if o.order_uuid == "order2")

        self.assertEqual(order1_in_list.src, OrderSource.ORDER_SRC_LIMIT_CANCELLED)
        self.assertEqual(order2_in_list.src, OrderSource.ORDER_SRC_LIMIT_UNFILLED)

    def test_cancel_limit_order_rpc_all_for_trade_pair(self):
        """Test cancelling all limit orders for a trade pair"""
        for i in range(3):
            order = self.create_test_limit_order(order_uuid=f"order{i}")
            self.limit_order_manager.process_limit_order_rpc(
                self.DEFAULT_MINER_HOTKEY,
                order.to_python_dict()
            )

        # Cancel all (empty order_uuid)
        result = self.limit_order_manager.cancel_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "",
            TimeUtil.now_in_millis()
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["num_cancelled"], 3)

        # Verify all cancelled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        for order in orders:
            self.assertEqual(order.src, OrderSource.ORDER_SRC_LIMIT_CANCELLED)

    def test_cancel_limit_order_rpc_nonexistent(self):
        """Test cancelling non-existent order raises exception"""
        with self.assertRaises(SignalException) as context:
            self.limit_order_manager.cancel_limit_order_rpc(
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

        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            btc_order.to_python_dict()
        )
        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            eth_order.to_python_dict()
        )

        # Delete all
        result = self.limit_order_manager.delete_all_limit_orders_for_hotkey_rpc(
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["deleted_count"], 2)

        # Verify all deleted from memory
        for trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD]:
            if trade_pair in self.limit_order_manager._limit_orders:
                self.assertNotIn(self.DEFAULT_MINER_HOTKEY,
                               self.limit_order_manager._limit_orders[trade_pair])

    def test_delete_all_limit_orders_multiple_miners(self):
        """Test deletion only affects target miner"""
        miner2 = "miner2"
        self.mock_metagraph.hotkeys.append(miner2)

        order1 = self.create_test_limit_order(order_uuid="miner1_order")
        order2 = self.create_test_limit_order(order_uuid="miner2_order")

        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            order1.to_python_dict()
        )
        self.limit_order_manager.process_limit_order_rpc(
            miner2,
            order2.to_python_dict()
        )

        # Delete only miner1
        result = self.limit_order_manager.delete_all_limit_orders_for_hotkey_rpc(
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertEqual(result["deleted_count"], 1)

        # Verify miner2's orders still exist
        self.assertIn(miner2, self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR])
        self.assertNotIn(self.DEFAULT_MINER_HOTKEY,
                        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR])

    # ============================================================================
    # Test Trigger Price Evaluation
    # ============================================================================

    def test_evaluate_trigger_price_long_order(self):
        """Test LONG order trigger evaluation"""
        # LONG order: triggers when ask <= limit_price
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # ask=50100 > limit=50000 -> no trigger
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # ask=50000 = limit=50000 -> trigger at ask
        price_source.ask = 50000.0
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

        # ask=49900 < limit=50000 -> trigger at ask
        price_source.ask = 49900.0
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 49900.0)

    def test_evaluate_trigger_price_short_order(self):
        """Test SHORT order trigger evaluation"""
        # SHORT order: triggers when bid >= limit_price
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # bid=49900 < limit=50000 -> no trigger
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.SHORT,
            None,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # bid=50000 = limit=50000 -> trigger at bid
        price_source.bid = 50000.0
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.SHORT,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50000.0)

        # bid=50100 > limit=50000 -> trigger at bid
        price_source.bid = 50100.0
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.SHORT,
            None,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50100.0)

    def test_evaluate_trigger_price_flat_long_position(self):
        """Test FLAT order trigger for LONG position (sells at bid)"""
        position = self.create_test_position(position_type=OrderType.LONG)

        # FLAT for LONG position: triggers when bid >= limit_price (selling)
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # bid=49900 < limit=50000 -> no trigger
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # bid=50100 > limit=50000 -> trigger at bid
        price_source.bid = 50100.0
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 50100.0)

    def test_evaluate_trigger_price_flat_short_position(self):
        """Test FLAT order trigger for SHORT position (buys at ask)"""
        position = self.create_test_position(position_type=OrderType.SHORT)

        # FLAT for SHORT position: triggers when ask <= limit_price (buying)
        price_source = self.create_test_price_source(50000.0, bid=49900.0, ask=50100.0)

        # ask=50100 > limit=50000 -> no trigger
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertIsNone(trigger)

        # ask=49900 < limit=50000 -> trigger at ask
        price_source.ask = 49900.0
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.FLAT,
            position,
            price_source,
            50000.0
        )
        self.assertEqual(trigger, 49900.0)

    def test_evaluate_trigger_price_fallback_to_open(self):
        """Test fallback to open price when bid/ask is 0"""
        price_source = self.create_test_price_source(50000.0, bid=0, ask=0)

        # LONG uses ask (0) -> falls back to open=50000
        trigger = self.limit_order_manager._evaluate_trigger_price(
            OrderType.LONG,
            None,
            price_source,
            50100.0
        )
        self.assertEqual(trigger, 50000.0)  # open <= limit

    # ============================================================================
    # Test Fill Logic with Market Order Manager Integration
    # ============================================================================

    def test_fill_limit_order_success(self):
        """Test successful limit order fill delegates to market_order_manager"""
        order = self.create_test_limit_order(limit_price=50000.0)
        price_source = self.create_test_price_source(49000.0, bid=49000.0, ask=49000.0)

        # Mock successful fill
        filled_order = deepcopy(order)
        filled_order.price = 49000.0
        filled_order.bid = 49000.0
        filled_order.ask = 49000.0
        filled_order.slippage = 10.0
        filled_order.processed_ms = price_source.start_ms
        filled_order.src = OrderSource.ORDER_SRC_LIMIT_FILLED

        mock_position = self.create_test_position()
        mock_position.orders = [filled_order]

        self.mock_market_order_manager._process_market_order.return_value = (None, mock_position)

        # Store order first
        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Fill it
        self.limit_order_manager._fill_limit_order_with_price_source(
            self.DEFAULT_MINER_HOTKEY,
            order,
            price_source,
            49000.0
        )

        # Verify market_order_manager was called
        self.mock_market_order_manager._process_market_order.assert_called_once()
        call_args = self.mock_market_order_manager._process_market_order.call_args[0]

        self.assertEqual(call_args[0], order.order_uuid)  # order_uuid
        self.assertEqual(call_args[1], "limit_order")     # miner_repo_version
        self.assertEqual(call_args[2], self.DEFAULT_TRADE_PAIR)  # trade_pair

        # Verify order was updated with filled values
        self.assertEqual(order.price, 49000.0)
        self.assertEqual(order.bid, 49000.0)
        self.assertEqual(order.ask, 49000.0)
        self.assertEqual(order.slippage, 10.0)

        # Verify _close_limit_order was called (order should be FILLED)
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_FILLED)

    def test_fill_limit_order_error_cancels(self):
        """Test limit order is cancelled when fill fails"""
        order = self.create_test_limit_order(limit_price=50000.0)
        price_source = self.create_test_price_source(49000.0)

        # Mock fill error
        self.mock_market_order_manager._process_market_order.return_value = (
            "Error: position not found",
            None
        )

        # Store order
        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Attempt fill
        self.limit_order_manager._fill_limit_order_with_price_source(
            self.DEFAULT_MINER_HOTKEY,
            order,
            price_source,
            49000.0
        )

        # Verify order was cancelled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_CANCELLED)

    def test_fill_limit_order_exception_cancels(self):
        """Test limit order is cancelled when exception occurs"""
        order = self.create_test_limit_order(limit_price=50000.0)
        price_source = self.create_test_price_source(49000.0)

        # Mock exception
        self.mock_market_order_manager._process_market_order.side_effect = Exception("Network error")

        # Store order
        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Attempt fill
        self.limit_order_manager._fill_limit_order_with_price_source(
            self.DEFAULT_MINER_HOTKEY,
            order,
            price_source,
            49000.0
        )

        # Verify order was cancelled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_CANCELLED)

    # ============================================================================
    # Test Daemon: check_and_fill_limit_orders
    # ============================================================================

    def test_check_and_fill_limit_orders_no_orders(self):
        """Test daemon runs without errors when no orders exist"""
        self.limit_order_manager.check_and_fill_limit_orders()
        # Should complete without errors

    def test_check_and_fill_limit_orders_market_closed(self):
        """Test daemon skips orders when market is closed"""
        order = self.create_test_limit_order()
        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Mock market closed
        self.live_price_fetcher.is_market_open = Mock(return_value=False)

        self.limit_order_manager.check_and_fill_limit_orders()

        # Order should remain unfilled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_UNFILLED)

    def test_check_and_fill_limit_orders_no_price_sources(self):
        """Test daemon skips when no price sources available"""
        order = self.create_test_limit_order()
        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Mock market open but no prices
        self.live_price_fetcher.is_market_open = Mock(return_value=True)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(return_value=None)

        self.limit_order_manager.check_and_fill_limit_orders()

        # Order should remain unfilled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_UNFILLED)

    def test_check_and_fill_limit_orders_triggers_and_fills(self):
        """Test daemon successfully checks and fills triggered orders"""
        order = self.create_test_limit_order(
            order_type=OrderType.LONG,
            limit_price=50000.0
        )

        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Mock market open with triggering price
        self.live_price_fetcher.is_market_open = Mock(return_value=True)
        trigger_price_source = self.create_test_price_source(49000.0, bid=49000.0, ask=49000.0)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(
            return_value=[trigger_price_source]
        )

        # Mock successful fill
        filled_order = deepcopy(order)
        filled_order.price = 49000.0
        mock_position = self.create_test_position()
        mock_position.orders = [filled_order]
        self.mock_market_order_manager._process_market_order.return_value = (None, mock_position)

        self.limit_order_manager.check_and_fill_limit_orders()

        # Verify order was filled
        orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(orders[0].src, OrderSource.ORDER_SRC_LIMIT_FILLED)

    def test_check_and_fill_limit_orders_skips_filled_orders(self):
        """Test daemon skips already filled orders"""
        order = self.create_test_limit_order()
        order.src = OrderSource.ORDER_SRC_LIMIT_FILLED

        self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR] = {
            self.DEFAULT_MINER_HOTKEY: [order]
        }

        # Mock market open with triggering price
        self.live_price_fetcher.is_market_open = Mock(return_value=True)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(
            return_value=[self.create_test_price_source(40000.0)]
        )

        self.limit_order_manager.check_and_fill_limit_orders()

        # Verify _process_market_order was NOT called
        self.mock_market_order_manager._process_market_order.assert_not_called()

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
                self.limit_order_manager.process_limit_order_rpc(
                    self.DEFAULT_MINER_HOTKEY,
                    order.to_python_dict()
                )

        count = self.limit_order_manager._count_unfilled_orders_for_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(count, 4)

        # Fill one order
        btc_orders = self.limit_order_manager._limit_orders[TradePair.BTCUSD][self.DEFAULT_MINER_HOTKEY]
        btc_orders[0].src = OrderSource.ORDER_SRC_LIMIT_FILLED

        count = self.limit_order_manager._count_unfilled_orders_for_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(count, 3)

    def test_get_position_for(self):
        """Test getting position for limit order"""
        position = self.create_test_position()
        self.position_manager.save_miner_position(position)

        order = self.create_test_limit_order()

        retrieved_position = self.limit_order_manager._get_position_for(
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
        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            order.to_python_dict()
        )

        # Verify structure
        self.assertIsInstance(self.limit_order_manager._limit_orders, dict)
        self.assertIn(self.DEFAULT_TRADE_PAIR, self.limit_order_manager._limit_orders)
        self.assertIsInstance(self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR], dict)
        self.assertIn(self.DEFAULT_MINER_HOTKEY,
                     self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR])
        self.assertIsInstance(
            self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY],
            list
        )

    def test_multiple_miners_isolation(self):
        """Test limit orders are isolated by miner"""
        miner2 = "miner2"
        self.mock_metagraph.hotkeys.append(miner2)

        order1 = self.create_test_limit_order(order_uuid="miner1_order")
        order2 = self.create_test_limit_order(order_uuid="miner2_order")

        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            order1.to_python_dict()
        )
        self.limit_order_manager.process_limit_order_rpc(
            miner2,
            order2.to_python_dict()
        )

        miner1_orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][self.DEFAULT_MINER_HOTKEY]
        miner2_orders = self.limit_order_manager._limit_orders[self.DEFAULT_TRADE_PAIR][miner2]

        self.assertEqual(len(miner1_orders), 1)
        self.assertEqual(len(miner2_orders), 1)
        self.assertEqual(miner1_orders[0].order_uuid, "miner1_order")
        self.assertEqual(miner2_orders[0].order_uuid, "miner2_order")

    def test_read_limit_orders_from_disk_skips_eliminated(self):
        """Test that eliminated miners' orders are not loaded from disk"""
        # Add order
        order = self.create_test_limit_order()
        self.limit_order_manager.process_limit_order_rpc(
            self.DEFAULT_MINER_HOTKEY,
            order.to_python_dict()
        )

        # Eliminate miner - add to eliminations dict
        self.elimination_manager.eliminations[self.DEFAULT_MINER_HOTKEY] = TimeUtil.now_in_millis()

        # Create new manager instance (simulates restart)
        new_manager = LimitOrderManager(
            position_manager=self.position_manager,
            live_price_fetcher=self.live_price_fetcher,
            market_order_manager=self.mock_market_order_manager,
            shutdown_dict=None,
            running_unit_tests=True
        )

        # Verify eliminated miner's orders not loaded
        self.assertNotIn(self.DEFAULT_MINER_HOTKEY,
                        new_manager._limit_orders.get(self.DEFAULT_TRADE_PAIR, {}))


if __name__ == '__main__':
    unittest.main()
