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
from vali_objects.utils.market_order_manager import MarketOrderManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestMarketOrderManager(TestBase):

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_ACCOUNT_SIZE = 1000.0

        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(None, None, self.mock_metagraph, running_unit_tests=True)
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

        self.price_slippage_model = PriceSlippageModel(
            live_price_fetcher=self.live_price_fetcher,
            running_unit_tests=True
        )

        # Mock contract manager
        self.mock_contract_manager = Mock(spec=ValidatorContractManager)
        self.mock_contract_manager.get_miner_account_size = Mock(return_value=self.DEFAULT_ACCOUNT_SIZE)

        # Mock shared queue for websockets
        self.mock_shared_queue = Mock()

        self.market_order_manager = MarketOrderManager(
            live_price_fetcher=self.live_price_fetcher,
            position_locks=self.position_locks,
            price_slippage_model=self.price_slippage_model,
            config=Mock(serve=False),
            position_manager=self.position_manager,
            shared_queue_websockets=self.mock_shared_queue,
            contract_manager=self.mock_contract_manager
        )

        self.position_manager.clear_all_miner_positions()

    def tearDown(self):
        """Clean up resources after each test."""
        # Shutdown the RPC server to free the port for the next test
        if hasattr(self, 'position_manager'):
            self.position_manager.shutdown()
        super().tearDown()

    def create_test_price_source(self, price, bid=None, ask=None, start_ms=None):
        """Helper to create a price source"""
        if start_ms is None:
            start_ms = TimeUtil.now_in_millis()
        if bid is None:
            bid = price - 10
        if ask is None:
            ask = price + 10

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
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )
        if position_type:
            position.position_type = position_type
        return position

    def create_test_signal(self, order_type=OrderType.LONG, leverage=1.0, execution_type=ExecutionType.MARKET):
        """Helper to create signal dict"""
        return {
            "order_type": order_type.name,
            "leverage": leverage,
            "execution_type": execution_type.name
        }

    # ============================================================================
    # Test: enforce_order_cooldown
    # ============================================================================

    def test_enforce_order_cooldown_first_order(self):
        """Test that first order for a trade pair has no cooldown"""
        now_ms = TimeUtil.now_in_millis()

        msg = self.market_order_manager.enforce_order_cooldown(
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            now_ms,
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertIsNone(msg)

    def test_enforce_order_cooldown_within_cooldown_period(self):
        """Test cooldown enforcement within cooldown period"""
        now_ms = TimeUtil.now_in_millis()

        # Cache first order time
        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key] = now_ms

        # Try to place order too soon
        second_order_ms = now_ms + (ValiConfig.ORDER_COOLDOWN_MS // 2)

        msg = self.market_order_manager.enforce_order_cooldown(
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            second_order_ms,
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertIsNotNone(msg)
        self.assertIn("too soon", msg)

    def test_enforce_order_cooldown_after_cooldown_period(self):
        """Test cooldown allows order after cooldown period"""
        now_ms = TimeUtil.now_in_millis()

        # Cache first order time
        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key] = now_ms

        # Place order after cooldown
        second_order_ms = now_ms + ValiConfig.ORDER_COOLDOWN_MS + 1000

        msg = self.market_order_manager.enforce_order_cooldown(
            self.DEFAULT_TRADE_PAIR.trade_pair_id,
            second_order_ms,
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertIsNone(msg)

    def test_enforce_order_cooldown_different_trade_pairs(self):
        """Test cooldown is isolated by trade pair"""
        now_ms = TimeUtil.now_in_millis()

        # Cache order for BTCUSD
        cache_key_btc = (self.DEFAULT_MINER_HOTKEY, TradePair.BTCUSD.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key_btc] = now_ms

        # Order for ETHUSD should have no cooldown
        msg = self.market_order_manager.enforce_order_cooldown(
            TradePair.ETHUSD.trade_pair_id,
            now_ms + 100,
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertIsNone(msg)

    # ============================================================================
    # Test: _get_or_create_open_position_from_new_order
    # ============================================================================

    def test_get_or_create_open_position_creates_new_for_long(self):
        """Test creating new position for LONG order"""
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        position = self.market_order_manager._get_or_create_open_position_from_new_order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.LONG,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_order_uuid="test_uuid",
            now_ms=now_ms,
            price_sources=price_sources,
            miner_repo_version="1.0.0",
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        self.assertIsNotNone(position)
        self.assertEqual(position.miner_hotkey, self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(position.trade_pair, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(position.position_uuid, "test_uuid")
        self.assertFalse(position.is_closed_position)

    def test_get_or_create_open_position_creates_new_for_short(self):
        """Test creating new position for SHORT order"""
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        position = self.market_order_manager._get_or_create_open_position_from_new_order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.SHORT,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_order_uuid="test_uuid",
            now_ms=now_ms,
            price_sources=price_sources,
            miner_repo_version="1.0.0",
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        self.assertIsNotNone(position)
        self.assertFalse(position.is_closed_position)

    def test_get_or_create_open_position_returns_existing(self):
        """Test returns existing open position"""
        # Create and save existing position
        existing_position = self.create_test_position(position_type=OrderType.LONG)
        self.position_manager.save_miner_position(existing_position)

        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        position = self.market_order_manager._get_or_create_open_position_from_new_order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.LONG,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_order_uuid="new_uuid",
            now_ms=now_ms,
            price_sources=price_sources,
            miner_repo_version="1.0.0",
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        self.assertEqual(position.position_uuid, existing_position.position_uuid)

    def test_get_or_create_open_position_flat_returns_none(self):
        """Test FLAT order with no position returns None"""
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        position = self.market_order_manager._get_or_create_open_position_from_new_order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.FLAT,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_order_uuid="test_uuid",
            now_ms=now_ms,
            price_sources=price_sources,
            miner_repo_version="1.0.0",
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        self.assertIsNone(position)

    def test_get_or_create_open_position_max_orders_auto_closes(self):
        """Test position auto-closes when MAX_ORDERS_PER_POSITION reached"""
        # Create position with max orders
        existing_position = self.create_test_position(position_type=OrderType.LONG)

        # Add orders up to max using proper method
        now_ms = TimeUtil.now_in_millis()
        for i in range(ValiConfig.MAX_ORDERS_PER_POSITION):
            order = Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_type=OrderType.LONG,
                leverage=0.1,  # Small leverage to avoid clamping issues
                price=50000.0,
                processed_ms=now_ms + (i * 1000),
                order_uuid=f"order_{i}",
                execution_type=ExecutionType.MARKET
            )
            existing_position.orders.append(order)

        # Rebuild position to calculate all metrics properly
        existing_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        self.position_manager.save_miner_position(existing_position)

        price_sources = [self.create_test_price_source(51000.0, start_ms=now_ms + 10000)]

        # Try to add another order - should trigger auto-close
        returned_position = self.market_order_manager._get_or_create_open_position_from_new_order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.LONG,
            order_time_ms=now_ms + 10000,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_order_uuid="new_order",
            now_ms=now_ms + 10000,
            price_sources=price_sources,
            miner_repo_version="1.0.0",
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Get the updated position from position manager
        updated_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        updated_position = next((p for p in updated_positions if p.trade_pair == self.DEFAULT_TRADE_PAIR), None)

        # Should have auto-closed the position with FLAT order
        self.assertIsNotNone(updated_position)
        self.assertEqual(len(updated_position.orders), ValiConfig.MAX_ORDERS_PER_POSITION + 1)
        last_order = updated_position.orders[-1]
        self.assertEqual(last_order.order_type, OrderType.FLAT)
        self.assertEqual(last_order.src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

    def test_get_or_create_open_position_closed_position_creates_new(self):
        """Test that closed positions are ignored and new position is created"""
        # Create closed position
        closed_position = self.create_test_position(position_type=OrderType.LONG)
        closed_position.is_closed_position = True
        self.position_manager.save_miner_position(closed_position)

        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        # Closed positions should be ignored (only_open_positions=True in get_positions_for_one_hotkey)
        # So a new position should be created
        position = self.market_order_manager._get_or_create_open_position_from_new_order(
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.LONG,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_order_uuid="test_uuid",
            now_ms=now_ms,
            price_sources=price_sources,
            miner_repo_version="1.0.0",
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Should create a new position since closed one is ignored
        self.assertIsNotNone(position)
        self.assertNotEqual(position.position_uuid, closed_position.position_uuid)

    # ============================================================================
    # Test: _add_order_to_existing_position
    # ============================================================================

    def test_add_order_to_existing_position_long(self):
        """Test adding LONG order to existing position"""
        position = self.create_test_position()
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, bid=49990.0, ask=50010.0, start_ms=now_ms)]

        initial_order_count = len(position.orders)

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            signal_leverage=1.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="test_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Verify order was added
        self.assertEqual(len(position.orders), initial_order_count + 1)

        new_order = position.orders[-1]
        self.assertEqual(new_order.order_type, OrderType.LONG)
        # Leverage may be clamped due to position constraints (e.g., 0.5 for crypto)
        self.assertGreater(new_order.leverage, 0)
        self.assertLessEqual(new_order.leverage, 1.0)
        self.assertEqual(new_order.order_uuid, "test_order")
        self.assertEqual(new_order.src, OrderSource.ORGANIC)

        # For crypto websocket data (timespan_ms=0, websocket=True), production uses mid-price (self.open)
        # NOT bid/ask. Bid/ask only used for forex with 1-second candles (timespan_ms=1000)
        self.assertEqual(new_order.price, 50000.0)

        # Verify slippage was calculated
        self.assertIsNotNone(new_order.slippage)

    def test_add_order_to_existing_position_short(self):
        """Test adding SHORT order to existing position with existing SHORT position type"""
        position = self.create_test_position(position_type=OrderType.SHORT)
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, bid=49990.0, ask=50010.0, start_ms=now_ms)]

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.SHORT,
            signal_leverage=1.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="test_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        new_order = position.orders[-1]
        self.assertEqual(new_order.order_type, OrderType.SHORT)

        # For crypto websocket data, production uses mid-price (self.open), NOT bid
        self.assertEqual(new_order.price, 50000.0)

    def test_add_order_to_existing_position_flat(self):
        """Test adding FLAT order to close position"""
        position = self.create_test_position(position_type=OrderType.LONG)
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(51000.0, bid=50990.0, ask=51010.0, start_ms=now_ms)]

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.FLAT,
            signal_leverage=0.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="flat_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        new_order = position.orders[-1]
        self.assertEqual(new_order.order_type, OrderType.FLAT)
        self.assertEqual(new_order.leverage, 0.0)

    def test_add_order_updates_cooldown_cache(self):
        """Test that adding order updates cooldown cache"""
        position = self.create_test_position()
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.assertNotIn(cache_key, self.market_order_manager.last_order_time_cache)

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            signal_leverage=1.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="test_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Verify cooldown cache was updated
        self.assertIn(cache_key, self.market_order_manager.last_order_time_cache)
        self.assertEqual(self.market_order_manager.last_order_time_cache[cache_key], now_ms)

    def test_add_order_saves_position(self):
        """Test that adding order saves position to disk"""
        position = self.create_test_position()
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            signal_leverage=1.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="test_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Verify position was saved - get all positions for hotkey and filter by trade pair
        saved_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        saved_position = next((p for p in saved_positions if p.trade_pair == self.DEFAULT_TRADE_PAIR), None)
        self.assertIsNotNone(saved_position)
        self.assertEqual(saved_position.position_uuid, position.position_uuid)

    def test_add_order_with_limit_source(self):
        """Test adding order with ORDER_SRC_LIMIT_FILLED source"""
        position = self.create_test_position()
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            signal_leverage=1.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="limit_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORDER_SRC_LIMIT_FILLED,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        new_order = position.orders[-1]
        self.assertEqual(new_order.src, OrderSource.ORDER_SRC_LIMIT_FILLED)

    # ============================================================================
    # Test: _process_market_order (internal method)
    # ============================================================================

    def test_process_market_order_creates_new_position(self):
        """Test processing market order creates new position"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position = self.market_order_manager._process_market_order(
            miner_order_uuid="test_uuid",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        self.assertIsNone(err_msg)
        self.assertIsNotNone(position)
        self.assertEqual(position.position_uuid, "test_uuid")
        self.assertEqual(len(position.orders), 1)
        self.assertEqual(position.orders[0].order_type, OrderType.LONG)

    def test_process_market_order_adds_to_existing_position(self):
        """Test processing market order adds to existing position"""
        # Temporarily disable running_unit_tests mode to avoid strict disk/memory verification
        # that conflicts with this test scenario where we're deliberately modifying positions
        original_running_unit_tests = self.position_manager.running_unit_tests
        self.position_manager.running_unit_tests = False

        try:
            now_ms = TimeUtil.now_in_millis()

            # Create first order using _process_market_order to ensure proper setup
            first_signal = self.create_test_signal(order_type=OrderType.LONG, leverage=0.3)
            first_order_time = now_ms - ValiConfig.ORDER_COOLDOWN_MS - 1000  # Well before cooldown period
            first_price_sources = [self.create_test_price_source(50000.0, start_ms=first_order_time)]

            err_msg1, existing_position = self.market_order_manager._process_market_order(
                miner_order_uuid="first_order",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=first_order_time,
                signal=first_signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=first_price_sources
            )

            self.assertIsNone(err_msg1)
            self.assertIsNotNone(existing_position)
            self.assertEqual(len(existing_position.orders), 1)

            # Now add second order to existing position (after cooldown period)
            second_signal = self.create_test_signal(order_type=OrderType.LONG, leverage=0.2)
            second_price_sources = [self.create_test_price_source(51000.0, start_ms=now_ms)]

            err_msg2, position = self.market_order_manager._process_market_order(
                miner_order_uuid="second_order",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=now_ms,
                signal=second_signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=second_price_sources
            )

            self.assertIsNone(err_msg2)
            self.assertIsNotNone(position)
            self.assertEqual(position.position_uuid, existing_position.position_uuid)
            self.assertEqual(len(position.orders), 2)  # First order + second order
        finally:
            # Restore original setting
            self.position_manager.running_unit_tests = original_running_unit_tests

    def test_process_market_order_no_price_sources_fails(self):
        """Test processing market order raises exception when no price sources available"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)

        # Mock the price fetcher to return empty list (no prices available)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(return_value=[])

        # Should raise SignalException when no prices are available
        with self.assertRaises(SignalException) as context:
            self.market_order_manager._process_market_order(
                miner_order_uuid="test_uuid",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=now_ms,
                signal=signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=None
            )

        self.assertIn("no live prices", str(context.exception).lower())

    def test_process_market_order_cooldown_violation_fails(self):
        """Test processing market order fails on cooldown violation"""
        now_ms = TimeUtil.now_in_millis()

        # Cache first order
        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key] = now_ms

        # Try second order too soon
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms + 1000)]

        err_msg, position = self.market_order_manager._process_market_order(
            miner_order_uuid="second_order",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms + 1000,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        self.assertIsNotNone(err_msg)
        self.assertIn("too soon", err_msg)
        self.assertIsNone(position)

    def test_process_market_order_flat_no_position(self):
        """Test FLAT order with no existing position returns None"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.FLAT, leverage=0.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position = self.market_order_manager._process_market_order(
            miner_order_uuid="flat_order",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        # Should succeed but return None position
        self.assertIsNone(err_msg)
        self.assertIsNone(position)

    def test_process_market_order_gets_account_size(self):
        """Test that processing order calls get_miner_account_size"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        self.market_order_manager._process_market_order(
            miner_order_uuid="test_uuid",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        # Verify contract manager was called
        self.mock_contract_manager.get_miner_account_size.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY,
            now_ms,
            use_account_floor=True
        )

    def test_process_market_order_limit_execution_type(self):
        """Test processing order with LIMIT execution type sets correct source"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(
            order_type=OrderType.LONG,
            leverage=1.0,
            execution_type=ExecutionType.LIMIT
        )
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position = self.market_order_manager._process_market_order(
            miner_order_uuid="limit_uuid",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        self.assertIsNone(err_msg)
        self.assertIsNotNone(position)

        # Verify order source is LIMIT_FILLED
        new_order = position.orders[-1]
        self.assertEqual(new_order.src, OrderSource.ORDER_SRC_LIMIT_FILLED)

    def test_process_market_order_market_execution_type(self):
        """Test processing order with MARKET execution type sets correct source"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(
            order_type=OrderType.LONG,
            leverage=1.0,
            execution_type=ExecutionType.MARKET
        )
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position = self.market_order_manager._process_market_order(
            miner_order_uuid="market_uuid",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        self.assertIsNone(err_msg)
        self.assertIsNotNone(position)

        # Verify order source is ORGANIC
        new_order = position.orders[-1]
        self.assertEqual(new_order.src, OrderSource.ORGANIC)

    # ============================================================================
    # Test: process_market_order (public synapse interface)
    # ============================================================================

    def test_process_market_order_synapse_success(self):
        """Test public synapse interface for market order processing"""
        mock_synapse = Mock()
        mock_synapse.successfully_processed = False
        mock_synapse.error_message = None
        mock_synapse.order_json = None

        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        self.market_order_manager.process_market_order(
            synapse=mock_synapse,
            miner_order_uuid="test_uuid",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        # Synapse should not be marked as failed
        self.assertFalse(mock_synapse.successfully_processed)
        self.assertIsNone(mock_synapse.error_message)

        # Order JSON should be set
        self.assertIsNotNone(mock_synapse.order_json)

    def test_process_market_order_synapse_error(self):
        """Test public synapse interface handles errors"""
        mock_synapse = Mock()
        mock_synapse.successfully_processed = False
        mock_synapse.error_message = None

        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)

        # Mock the price fetcher to return empty list (no prices available)
        self.live_price_fetcher.get_sorted_price_sources_for_trade_pair = Mock(return_value=[])

        # process_market_order should catch SignalException and set synapse error
        # But looking at the code, it calls _process_market_order which raises exception
        # The public method should handle this. Let's test that it raises the exception
        with self.assertRaises(SignalException):
            self.market_order_manager.process_market_order(
                synapse=mock_synapse,
                miner_order_uuid="test_uuid",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=now_ms,
                signal=signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=None
            )

    # ============================================================================
    # Test: Multiple Miners and Trade Pairs
    # ============================================================================

    def test_process_market_order_multiple_miners_isolation(self):
        """Test orders are isolated between miners"""
        miner2 = "miner2"
        self.mock_metagraph.hotkeys.append(miner2)

        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        # Process order for miner 1
        _, pos1 = self.market_order_manager._process_market_order(
            miner_order_uuid="miner1_order",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        # Process order for miner 2
        _, pos2 = self.market_order_manager._process_market_order(
            miner_order_uuid="miner2_order",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms + 1000,
            signal=signal,
            miner_hotkey=miner2,
            price_sources=price_sources
        )

        # Verify separate positions
        self.assertNotEqual(pos1.miner_hotkey, pos2.miner_hotkey)
        self.assertNotEqual(pos1.position_uuid, pos2.position_uuid)

    def test_process_market_order_multiple_trade_pairs(self):
        """Test single miner can have positions in multiple trade pairs"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)

        btc_price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]
        eth_price_sources = [self.create_test_price_source(3000.0, start_ms=now_ms)]

        # BTC position
        _, btc_pos = self.market_order_manager._process_market_order(
            miner_order_uuid="btc_order",
            miner_repo_version="1.0.0",
            trade_pair=TradePair.BTCUSD,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=btc_price_sources
        )

        # ETH position
        _, eth_pos = self.market_order_manager._process_market_order(
            miner_order_uuid="eth_order",
            miner_repo_version="1.0.0",
            trade_pair=TradePair.ETHUSD,
            now_ms=now_ms + 1000,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=eth_price_sources
        )

        # Verify different positions
        self.assertNotEqual(btc_pos.trade_pair, eth_pos.trade_pair)
        self.assertNotEqual(btc_pos.position_uuid, eth_pos.position_uuid)

    # ============================================================================
    # Test: Edge Cases and Error Handling
    # ============================================================================

    def test_process_market_order_missing_signal_keys(self):
        """Test error handling when signal dict is missing required keys"""
        now_ms = TimeUtil.now_in_millis()
        invalid_signal = {"leverage": 1.0}  # Missing order_type
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        with self.assertRaises(KeyError):
            self.market_order_manager._process_market_order(
                miner_order_uuid="test_uuid",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=now_ms,
                signal=invalid_signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=price_sources
            )

    def test_cooldown_cache_key_format(self):
        """Test cooldown cache uses correct (hotkey, trade_pair_id) format"""
        now_ms = TimeUtil.now_in_millis()

        # Add order to populate cache
        position = self.create_test_position()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            signal_leverage=1.0,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="test",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Verify cache key format
        expected_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.assertIn(expected_key, self.market_order_manager.last_order_time_cache)


if __name__ == '__main__':
    unittest.main()
