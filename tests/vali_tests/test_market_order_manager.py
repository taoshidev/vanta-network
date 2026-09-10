# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
"""
Market order manager tests using the client/server architecture.

MarketOrderManager's current surface is execute_order() (with internal
_apply_order()), close_positions(), enforce_order_cooldown(),
clear_order_cooldown_cache(), and the static _is_effective_close(). This
covers all of them: pure-logic unit tests for cooldown/effective-close,
and ServerOrchestrator-backed integration tests for order execution and
position closing.
"""
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.market_order.market_order_manager import MarketOrderManager
from vali_objects.utils.limit_order.order_utils import OrderSize
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestMarketOrderManager(TestBase):
    """
    Integration tests for Market Order Manager using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across
    all test methods/classes; per-test isolation comes from clearing data
    state rather than restarting servers.
    """

    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    market_order_manager = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    # BTCUSD is a blocked/deprecated native pair; MarketOrderManager itself doesn't
    # enforce is_blocked (that lives in OrderProcessor.validate), but BTCUSDC keeps
    # these tests aligned with the currently-tradable pair.
    DEFAULT_TRADE_PAIR = TradePair.BTCUSDC

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
        cls.position_client = cls.orchestrator.get_client('position_manager')

        cls.market_order_manager = MarketOrderManager(False, running_unit_tests=True)

        cls.metagraph_client.set_hotkeys([cls.DEFAULT_MINER_HOTKEY])

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.market_order_manager.last_order_time_cache.clear()
        self.live_price_fetcher_client.clear_test_market_open()
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY])

    def tearDown(self):
        self.orchestrator.clear_all_test_data()
        self.market_order_manager.last_order_time_cache.clear()
        self.live_price_fetcher_client.clear_test_market_open()

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def create_test_price_source(self, price, bid=None, ask=None, start_ms=None):
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

    def create_test_position(self, trade_pair=None, miner_hotkey=None, position_type=OrderType.LONG):
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if miner_hotkey is None:
            miner_hotkey = self.DEFAULT_MINER_HOTKEY

        return Position(
            miner_hotkey=miner_hotkey,
            position_uuid=f"pos_{TimeUtil.now_in_millis()}",
            open_ms=TimeUtil.now_in_millis(),
            trade_pair=trade_pair,
            position_type=position_type,
            account_size=1000.0,
        )

    def execute(self, hotkey, order_uuid, order_type, price, value=None, leverage=None, quantity=None,
                bracket_pct=None, now_ms=None, trade_pair=None, enforce_cooldown=False, **kwargs):
        """Convenience wrapper: pass an explicit fill_price + price_sources so the
        manager never needs to reach into the live price fetcher for a fill."""
        if now_ms is None:
            now_ms = TimeUtil.now_in_millis()
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        price_source = self.create_test_price_source(price, start_ms=now_ms)
        order_size = OrderSize(value=value, leverage=leverage, quantity=quantity, bracket_pct=bracket_pct)
        return self.market_order_manager.execute_order(
            hotkey, order_uuid, trade_pair, ExecutionType.MARKET, order_type, order_size,
            fill_price=price, price_sources=[price_source], now_ms=now_ms,
            enforce_cooldown=enforce_cooldown, **kwargs,
        )

    # ============================================================================
    # Test: enforce_order_cooldown
    # ============================================================================

    def test_enforce_order_cooldown_first_order(self):
        now_ms = TimeUtil.now_in_millis()
        msg = self.market_order_manager.enforce_order_cooldown(
            self.DEFAULT_TRADE_PAIR.trade_pair_id, now_ms, self.DEFAULT_MINER_HOTKEY
        )
        self.assertIsNone(msg)

    def test_enforce_order_cooldown_within_cooldown_period(self):
        now_ms = TimeUtil.now_in_millis()
        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key] = now_ms

        second_order_ms = now_ms + (ValiConfig.ORDER_COOLDOWN_MS // 2)
        msg = self.market_order_manager.enforce_order_cooldown(
            self.DEFAULT_TRADE_PAIR.trade_pair_id, second_order_ms, self.DEFAULT_MINER_HOTKEY
        )

        self.assertIsNotNone(msg)
        self.assertIn("too soon", msg)

    def test_enforce_order_cooldown_after_cooldown_period(self):
        now_ms = TimeUtil.now_in_millis()
        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key] = now_ms

        second_order_ms = now_ms + ValiConfig.ORDER_COOLDOWN_MS + 1000
        msg = self.market_order_manager.enforce_order_cooldown(
            self.DEFAULT_TRADE_PAIR.trade_pair_id, second_order_ms, self.DEFAULT_MINER_HOTKEY
        )
        self.assertIsNone(msg)

    def test_enforce_order_cooldown_different_trade_pairs(self):
        now_ms = TimeUtil.now_in_millis()
        cache_key_btc = (self.DEFAULT_MINER_HOTKEY, TradePair.BTCUSDC.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key_btc] = now_ms

        msg = self.market_order_manager.enforce_order_cooldown(
            TradePair.ETHUSDC.trade_pair_id, now_ms + 100, self.DEFAULT_MINER_HOTKEY
        )
        self.assertIsNone(msg)

    def test_clear_order_cooldown_cache_requires_test_mode(self):
        production_manager = MarketOrderManager(False, running_unit_tests=False)
        with self.assertRaises(Exception):
            production_manager.clear_order_cooldown_cache()

    def test_clear_order_cooldown_cache_clears_entries(self):
        cache_key = (self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id)
        self.market_order_manager.last_order_time_cache[cache_key] = TimeUtil.now_in_millis()
        self.market_order_manager.clear_order_cooldown_cache()
        self.assertEqual(len(self.market_order_manager.last_order_time_cache), 0)

    # ============================================================================
    # Test: _is_effective_close (static)
    # ============================================================================

    def test_is_effective_close_flat_order_type_always_closes(self):
        position = self.create_test_position(position_type=OrderType.LONG)
        self.assertTrue(
            MarketOrderManager._is_effective_close(position, OrderType.FLAT, -1.0, -100.0)
        )

    def test_is_effective_close_no_existing_direction_is_never_close(self):
        position = self.create_test_position(position_type=OrderType.LONG)
        position.position_type = None
        self.assertFalse(
            MarketOrderManager._is_effective_close(position, OrderType.LONG, 1.0, 100.0)
        )

    def test_is_effective_close_same_direction_is_never_close(self):
        position = self.create_test_position(position_type=OrderType.LONG)
        self.assertFalse(
            MarketOrderManager._is_effective_close(position, OrderType.LONG, 1.0, 100.0)
        )

    def test_is_effective_close_long_position_flips_to_flat_on_quantity_cross(self):
        position = self.create_test_position(position_type=OrderType.LONG)
        position.net_quantity = 1.0
        position.net_value = 1000.0
        position.unrealized_pnl = 0.0
        # SHORT order that fully offsets net_quantity.
        self.assertTrue(
            MarketOrderManager._is_effective_close(position, OrderType.SHORT, -1.0, -1000.0)
        )

    def test_is_effective_close_long_position_partial_reduce_is_not_close(self):
        position = self.create_test_position(position_type=OrderType.LONG)
        position.net_quantity = 2.0
        position.net_value = 2000.0
        position.unrealized_pnl = 0.0
        # SHORT order that only offsets part of net_quantity/net_value.
        self.assertFalse(
            MarketOrderManager._is_effective_close(position, OrderType.SHORT, -0.5, -500.0)
        )

    def test_is_effective_close_short_position_flips_to_flat_on_quantity_cross(self):
        position = self.create_test_position(position_type=OrderType.SHORT)
        position.net_quantity = -1.0
        position.net_value = -1000.0
        position.unrealized_pnl = 0.0
        self.assertTrue(
            MarketOrderManager._is_effective_close(position, OrderType.LONG, 1.0, 1000.0)
        )

    def test_is_effective_close_short_position_partial_reduce_is_not_close(self):
        position = self.create_test_position(position_type=OrderType.SHORT)
        position.net_quantity = -2.0
        position.net_value = -2000.0
        position.unrealized_pnl = 0.0
        self.assertFalse(
            MarketOrderManager._is_effective_close(position, OrderType.LONG, 0.5, 500.0)
        )

    # ============================================================================
    # Test: execute_order
    # ============================================================================

    def test_execute_order_creates_new_position_for_long(self):
        order, position = self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_long", OrderType.LONG, 50000.0, value=500.0)

        self.assertIsNotNone(order)
        self.assertIsNotNone(position)
        self.assertEqual(position.miner_hotkey, self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(position.trade_pair, self.DEFAULT_TRADE_PAIR)
        self.assertEqual(position.position_uuid, "uuid_long")
        self.assertEqual(position.position_type, OrderType.LONG)
        self.assertFalse(position.is_closed_position)
        self.assertEqual(order.order_type, OrderType.LONG)
        self.assertEqual(order.price, 50000.0)

    def test_execute_order_creates_new_position_for_short(self):
        # SHORT orders arrive with a pre-negated size (Signal validation negates it
        # upstream before MarketOrderManager ever sees it); OrderSize sign, not
        # order_type, determines buy/sell direction inside _apply_order.
        order, position = self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_short", OrderType.SHORT, 50000.0, value=-500.0)

        self.assertIsNotNone(position)
        self.assertEqual(position.position_type, OrderType.SHORT)
        self.assertFalse(position.is_closed_position)

    def test_execute_order_flat_with_no_position_returns_none(self):
        result = self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_flat", OrderType.FLAT, 50000.0, quantity=0.0)
        self.assertIsNone(result)

    def test_execute_order_adds_to_existing_position(self):
        _, position = self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_1", OrderType.LONG, 50000.0, value=500.0)
        now_ms2 = TimeUtil.now_in_millis() + 1
        _, position2 = self.execute(
            self.DEFAULT_MINER_HOTKEY, "uuid_2", OrderType.LONG, 51000.0, value=300.0, now_ms=now_ms2
        )

        self.assertEqual(position2.position_uuid, position.position_uuid)
        self.assertEqual(len(position2.orders), 2)

    def test_execute_order_partial_reduce_keeps_position_open(self):
        _, position = self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_1", OrderType.LONG, 50000.0, value=800.0)
        now_ms2 = TimeUtil.now_in_millis() + 1
        _, position2 = self.execute(
            self.DEFAULT_MINER_HOTKEY, "uuid_2", OrderType.SHORT, 50000.0, value=-200.0, now_ms=now_ms2
        )

        self.assertEqual(position2.position_uuid, position.position_uuid)
        self.assertFalse(position2.is_closed_position)
        self.assertEqual(position2.position_type, OrderType.LONG)

    def test_execute_order_full_close_via_bracket_pct(self):
        _, position = self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_1", OrderType.LONG, 50000.0, value=500.0)
        now_ms2 = TimeUtil.now_in_millis() + 1

        order2, position2 = self.execute(
            self.DEFAULT_MINER_HOTKEY, "uuid_close", OrderType.FLAT, 50000.0, bracket_pct=1.0, now_ms=now_ms2
        )

        self.assertEqual(order2.order_type, OrderType.FLAT)
        self.assertTrue(position2.is_closed_position)
        self.assertEqual(position2.position_uuid, position.position_uuid)

    def test_execute_order_enforces_cooldown(self):
        self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_1", OrderType.LONG, 50000.0, value=500.0, enforce_cooldown=True)

        now_ms2 = TimeUtil.now_in_millis() + 100
        with self.assertRaises(SignalException) as ctx:
            self.execute(
                self.DEFAULT_MINER_HOTKEY, "uuid_2", OrderType.LONG, 50000.0, value=500.0,
                now_ms=now_ms2, enforce_cooldown=True,
            )
        self.assertIn("too soon", str(ctx.exception))

    def test_execute_order_bypasses_cooldown_when_disabled(self):
        self.execute(self.DEFAULT_MINER_HOTKEY, "uuid_1", OrderType.LONG, 50000.0, value=500.0, enforce_cooldown=True)

        now_ms2 = TimeUtil.now_in_millis() + 100
        # Should not raise, since enforce_cooldown=False bypasses the check.
        order, position = self.execute(
            self.DEFAULT_MINER_HOTKEY, "uuid_2", OrderType.LONG, 50000.0, value=500.0,
            now_ms=now_ms2, enforce_cooldown=False,
        )
        self.assertIsNotNone(order)

    def test_execute_order_market_closed_raises(self):
        self.live_price_fetcher_client.set_test_market_open(False)
        with self.assertRaises(SignalException) as ctx:
            self.execute(
                self.DEFAULT_MINER_HOTKEY, "uuid_1", OrderType.LONG, 50000.0, value=500.0,
                trade_pair=TradePair.EURUSD,
            )
        self.assertIn("currently closed", str(ctx.exception))

    def test_execute_order_max_orders_per_position_auto_closes(self):
        existing_position = self.create_test_position(position_type=OrderType.LONG)
        now_ms = TimeUtil.now_in_millis()
        for i in range(ValiConfig.MAX_ORDERS_PER_POSITION):
            order = Order(
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_type=OrderType.LONG,
                leverage=0.01,
                price=50000.0,
                processed_ms=now_ms + i,
                order_uuid=f"order_{i}",
                execution_type=ExecutionType.MARKET,
            )
            existing_position.orders.append(order)
        existing_position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
        self.position_client.save_miner_position(existing_position)

        now_ms2 = now_ms + ValiConfig.MAX_ORDERS_PER_POSITION + 1000
        new_order, new_position = self.execute(
            self.DEFAULT_MINER_HOTKEY, "new_order", OrderType.LONG, 51000.0, value=500.0, now_ms=now_ms2
        )

        self.assertEqual(new_position.position_uuid, "new_order")
        self.assertEqual(len(new_position.orders), 1)

        all_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        closed = [p for p in all_positions if p.position_uuid == existing_position.position_uuid]
        self.assertEqual(len(closed), 1)
        self.assertTrue(closed[0].is_closed_position)
        self.assertEqual(closed[0].orders[-1].order_type, OrderType.FLAT)
        self.assertEqual(closed[0].orders[-1].src, OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)

    def test_execute_order_multiple_miners_isolated(self):
        miner2 = "miner2"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, miner2])

        _, pos1 = self.execute(self.DEFAULT_MINER_HOTKEY, "m1_order", OrderType.LONG, 50000.0, value=500.0)
        _, pos2 = self.execute(miner2, "m2_order", OrderType.LONG, 50000.0, value=500.0)

        self.assertNotEqual(pos1.miner_hotkey, pos2.miner_hotkey)
        self.assertNotEqual(pos1.position_uuid, pos2.position_uuid)

    def test_execute_order_multiple_trade_pairs(self):
        _, btc_pos = self.execute(
            self.DEFAULT_MINER_HOTKEY, "btc_order", OrderType.LONG, 50000.0, value=500.0,
            trade_pair=TradePair.BTCUSDC,
        )
        now_ms2 = TimeUtil.now_in_millis() + 1
        _, eth_pos = self.execute(
            self.DEFAULT_MINER_HOTKEY, "eth_order", OrderType.LONG, 3000.0, value=500.0,
            trade_pair=TradePair.ETHUSDC, now_ms=now_ms2,
        )

        self.assertNotEqual(btc_pos.trade_pair, eth_pos.trade_pair)
        self.assertNotEqual(btc_pos.position_uuid, eth_pos.position_uuid)

    # ============================================================================
    # Test: close_positions
    # ============================================================================

    def test_close_positions_closes_all(self):
        trade_pairs = [TradePair.BTCUSDC, TradePair.ETHUSDC, TradePair.SOLUSDC]
        now_ms = TimeUtil.now_in_millis()

        for i, trade_pair in enumerate(trade_pairs):
            self.execute(
                self.DEFAULT_MINER_HOTKEY, f"position_{i}", OrderType.LONG, 1000.0 + i * 10, value=500.0,
                trade_pair=trade_pair, now_ms=now_ms + i,
            )

        open_positions_before = self.position_client.get_positions_for_hotkeys(
            [self.DEFAULT_MINER_HOTKEY], only_open_positions=True
        ).get(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(open_positions_before), 3)

        self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000
        )

        all_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(all_positions), 3)
        for position in all_positions:
            self.assertTrue(position.is_closed_position)
            last_order = position.orders[-1]
            self.assertEqual(last_order.order_type, OrderType.FLAT)
            # close_positions() doesn't pass order_src, so execute_order's default (ORGANIC) applies.
            self.assertEqual(last_order.src, OrderSource.ORGANIC)

    def test_close_positions_closes_only_matching_uuids(self):
        now_ms = TimeUtil.now_in_millis()
        _, pos1 = self.execute(
            self.DEFAULT_MINER_HOTKEY, "keep_open", OrderType.LONG, 1000.0, value=500.0,
            trade_pair=TradePair.BTCUSDC, now_ms=now_ms,
        )
        _, pos2 = self.execute(
            self.DEFAULT_MINER_HOTKEY, "to_close", OrderType.LONG, 2000.0, value=500.0,
            trade_pair=TradePair.ETHUSDC, now_ms=now_ms + 1,
        )

        self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[pos2.position_uuid], now_ms=now_ms + 10000
        )

        all_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        by_uuid = {p.position_uuid: p for p in all_positions}
        self.assertFalse(by_uuid[pos1.position_uuid].is_closed_position)
        self.assertTrue(by_uuid[pos2.position_uuid].is_closed_position)

    def test_close_positions_no_open_positions_is_noop(self):
        # Should not raise even though the miner has no positions at all.
        self.market_order_manager.close_positions(hotkey="miner_with_no_positions", close_all=True)

    def test_close_positions_no_matching_uuids_is_noop(self):
        now_ms = TimeUtil.now_in_millis()
        _, pos = self.execute(self.DEFAULT_MINER_HOTKEY, "order_1", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms)

        self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=["not_a_real_uuid"], now_ms=now_ms + 10000
        )

        refreshed = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertFalse(next(p for p in refreshed if p.position_uuid == pos.position_uuid).is_closed_position)
