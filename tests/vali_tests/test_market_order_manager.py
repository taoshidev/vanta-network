# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Market order manager tests using modern client/server architecture.
Tests all market order functionality with proper server/client separation.
"""
from unittest.mock import Mock

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.position import Position
from vali_objects.utils.market_order_manager import MarketOrderManager
from vali_objects.utils.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestMarketOrderManager(TestBase):
    """
    Integration tests for Market Order Manager using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    contract_client = None
    market_order_manager = None

    # Test constants
    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_ACCOUNT_SIZE = 1000.0

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
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.contract_client = cls.orchestrator.get_client('contract')

        # Get market order manager instance from orchestrator
        cls.market_order_manager = MarketOrderManager(False, running_unit_tests=True)

        # Initialize metagraph with test miners
        cls.metagraph_client.set_hotkeys([cls.DEFAULT_MINER_HOTKEY])

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
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Clear market order manager cache
        self.market_order_manager.last_order_time_cache.clear()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()
        self.market_order_manager.last_order_time_cache.clear()

    # ============================================================================
    # Helper Methods
    # ============================================================================

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

    @staticmethod
    def create_test_signal(order_type:OrderType=OrderType.LONG, leverage=1.0, execution_type:ExecutionType=ExecutionType.MARKET):
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
        self.position_client.save_miner_position(existing_position)

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

        # Add orders up to max
        now_ms = TimeUtil.now_in_millis()
        for i in range(ValiConfig.MAX_ORDERS_PER_POSITION):
            order = Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_type=OrderType.LONG,
                leverage=0.1,
                price=50000.0,
                processed_ms=now_ms + (i * 1000),
                order_uuid=f"order_{i}",
                execution_type=ExecutionType.MARKET
            )
            existing_position.orders.append(order)

        # Rebuild position
        existing_position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
        self.position_client.save_miner_position(existing_position)

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

        # Get updated position
        updated_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        updated_position = next((p for p in updated_positions if p.trade_pair == self.DEFAULT_TRADE_PAIR), None)

        # Should have auto-closed
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
        self.position_client.save_miner_position(closed_position)

        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        # Should create new position (closed ones ignored)
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

        # Calculate order size from leverage
        signal = {"leverage": 1.0}
        quantity, leverage, value = self.market_order_manager.parse_order_size(
            signal, 1.0, self.DEFAULT_TRADE_PAIR, self.DEFAULT_ACCOUNT_SIZE
        )

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            quantity=quantity,
            leverage=leverage,
            value=value,
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
        self.assertGreater(new_order.leverage, 0)
        self.assertLessEqual(new_order.leverage, 1.0)
        self.assertEqual(new_order.order_uuid, "test_order")
        self.assertEqual(new_order.src, OrderSource.ORGANIC)
        self.assertEqual(new_order.price, 50000.0)
        self.assertIsNotNone(new_order.slippage)

    def test_add_order_to_existing_position_short(self):
        """Test adding SHORT order to existing position"""
        position = self.create_test_position(position_type=OrderType.SHORT)
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, bid=49990.0, ask=50010.0, start_ms=now_ms)]

        # Calculate order size from leverage
        signal = {"leverage": 1.0}
        quantity, leverage, value = self.market_order_manager.parse_order_size(
            signal, 1.0, self.DEFAULT_TRADE_PAIR, self.DEFAULT_ACCOUNT_SIZE
        )

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.SHORT,
            quantity=quantity,
            leverage=leverage,
            value=value,
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
        self.assertEqual(new_order.price, 50000.0)

    def test_add_order_to_existing_position_flat(self):
        """Test adding FLAT order to close position"""
        position = self.create_test_position(position_type=OrderType.LONG)
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(51000.0, bid=50990.0, ask=51010.0, start_ms=now_ms)]

        # FLAT orders use 0.0 for all values
        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.FLAT,
            quantity=0.0,
            leverage=0.0,
            value=0.0,
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

        # Calculate order size from leverage
        signal = {"leverage": 1.0}
        quantity, leverage, value = self.market_order_manager.parse_order_size(
            signal, 1.0, self.DEFAULT_TRADE_PAIR, self.DEFAULT_ACCOUNT_SIZE
        )

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            quantity=quantity,
            leverage=leverage,
            value=value,
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

        # Calculate order size from leverage
        signal = {"leverage": 1.0}
        quantity, leverage, value = self.market_order_manager.parse_order_size(
            signal, 1.0, self.DEFAULT_TRADE_PAIR, self.DEFAULT_ACCOUNT_SIZE
        )

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            quantity=quantity,
            leverage=leverage,
            value=value,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="test_order",
            miner_repo_version="1.0.0",
            src=OrderSource.ORGANIC,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        # Verify position was saved
        saved_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        saved_position = next((p for p in saved_positions if p.trade_pair == self.DEFAULT_TRADE_PAIR), None)
        self.assertIsNotNone(saved_position)
        self.assertEqual(saved_position.position_uuid, position.position_uuid)

    def test_add_order_with_limit_source(self):
        """Test adding order with ORDER_SRC_LIMIT_FILLED source"""
        position = self.create_test_position()
        now_ms = TimeUtil.now_in_millis()
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        # Calculate order size from leverage
        signal = {"leverage": 1.0}
        quantity, leverage, value = self.market_order_manager.parse_order_size(
            signal, 1.0, self.DEFAULT_TRADE_PAIR, self.DEFAULT_ACCOUNT_SIZE
        )

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            quantity=quantity,
            leverage=leverage,
            value=value,
            order_time_ms=now_ms,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources,
            miner_order_uuid="limit_order",
            miner_repo_version="1.0.0",
            src=OrderSource.LIMIT_FILLED,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        new_order = position.orders[-1]
        self.assertEqual(new_order.src, OrderSource.LIMIT_FILLED)

    # ============================================================================
    # Test: _process_market_order (internal method)
    # ============================================================================

    def test_process_market_order_creates_new_position(self):
        """Test processing market order creates new position"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position, created_order = self.market_order_manager._process_market_order(
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
        self.assertIsNotNone(created_order)
        self.assertEqual(position.position_uuid, "test_uuid")
        self.assertEqual(len(position.orders), 1)
        self.assertEqual(position.orders[0].order_type, OrderType.LONG)

    def test_process_market_order_adds_to_existing_position(self):
        """Test processing market order adds to existing position"""
        now_ms = TimeUtil.now_in_millis()

        # Create first order
        first_signal = self.create_test_signal(order_type=OrderType.LONG, leverage=0.3)
        first_order_time = now_ms - ValiConfig.ORDER_COOLDOWN_MS - 1000
        first_price_sources = [self.create_test_price_source(50000.0, start_ms=first_order_time)]

        err_msg1, existing_position, _ = self.market_order_manager._process_market_order(
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

        # Add second order
        second_signal = self.create_test_signal(order_type=OrderType.LONG, leverage=0.2)
        second_price_sources = [self.create_test_price_source(51000.0, start_ms=now_ms)]

        err_msg2, position, _ = self.market_order_manager._process_market_order(
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
        self.assertEqual(len(position.orders), 2)

    def test_process_market_order_no_price_sources_fails(self):
        """Test processing market order fails when no price sources available"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)

        # Pass empty list (not None) to simulate no prices available
        # None would cause the code to fetch prices from live_price_fetcher
        with self.assertRaises(SignalException) as context:
            self.market_order_manager._process_market_order(
                miner_order_uuid="test_uuid",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=now_ms,
                signal=signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=[]  # Empty list, not None
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

        err_msg, position, created_order = self.market_order_manager._process_market_order(
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
        self.assertIsNone(created_order)

    def test_process_market_order_flat_no_position(self):
        """Test FLAT order with no existing position returns None"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.FLAT, leverage=0.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position, created_order = self.market_order_manager._process_market_order(
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
        self.assertIsNone(created_order)

    def test_process_market_order_gets_account_size(self):
        """Test that processing order retrieves account size"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        # Should not raise any errors (contract_client handles account size)
        err_msg, position, _ = self.market_order_manager._process_market_order(
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

    def test_process_market_order_limit_execution_type(self):
        """Test processing order with LIMIT execution type sets correct source"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(
            order_type=OrderType.LONG,
            leverage=1.0,
            execution_type=ExecutionType.LIMIT
        )
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position, created_order = self.market_order_manager._process_market_order(
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
        self.assertEqual(new_order.src, OrderSource.LIMIT_FILLED)

    def test_process_market_order_market_execution_type(self):
        """Test processing order with MARKET execution type sets correct source"""
        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(
            order_type=OrderType.LONG,
            leverage=1.0,
            execution_type=ExecutionType.MARKET
        )
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        err_msg, position, created_order = self.market_order_manager._process_market_order(
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

        created_order = self.market_order_manager.process_market_order(
            synapse=mock_synapse,
            miner_order_uuid="test_uuid",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        # Verify order was created
        self.assertIsNotNone(created_order)
        self.assertEqual(created_order.order_type, OrderType.LONG)

    def test_process_market_order_synapse_error(self):
        """Test public synapse interface handles errors"""
        mock_synapse = Mock()
        mock_synapse.successfully_processed = False
        mock_synapse.error_message = None

        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)

        # Pass empty list for price_sources to trigger error
        # process_market_order should raise SignalException
        with self.assertRaises(SignalException):
            self.market_order_manager.process_market_order(
                synapse=mock_synapse,
                miner_order_uuid="test_uuid",
                miner_repo_version="1.0.0",
                trade_pair=self.DEFAULT_TRADE_PAIR,
                now_ms=now_ms,
                signal=signal,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                price_sources=[]  # Empty list, not None
            )

    # ============================================================================
    # Test: Multiple Miners and Trade Pairs
    # ============================================================================

    def test_process_market_order_multiple_miners_isolation(self):
        """Test orders are isolated between miners"""
        miner2 = "miner2"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, miner2])

        now_ms = TimeUtil.now_in_millis()
        signal = self.create_test_signal(order_type=OrderType.LONG, leverage=1.0)
        price_sources = [self.create_test_price_source(50000.0, start_ms=now_ms)]

        # Process order for miner 1
        _, pos1, _ = self.market_order_manager._process_market_order(
            miner_order_uuid="miner1_order",
            miner_repo_version="1.0.0",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=price_sources
        )

        # Process order for miner 2
        _, pos2, _ = self.market_order_manager._process_market_order(
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
        _, btc_pos, _ = self.market_order_manager._process_market_order(
            miner_order_uuid="btc_order",
            miner_repo_version="1.0.0",
            trade_pair=TradePair.BTCUSD,
            now_ms=now_ms,
            signal=signal,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            price_sources=btc_price_sources
        )

        # ETH position
        _, eth_pos, _ = self.market_order_manager._process_market_order(
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

        # Calculate order size from leverage
        signal = {"leverage": 1.0}
        quantity, leverage, value = self.market_order_manager.parse_order_size(
            signal, 1.0, self.DEFAULT_TRADE_PAIR, self.DEFAULT_ACCOUNT_SIZE
        )

        self.market_order_manager._add_order_to_existing_position(
            existing_position=position,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            signal_order_type=OrderType.LONG,
            quantity=quantity,
            leverage=leverage,
            value=value,
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
