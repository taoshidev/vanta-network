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
from unittest.mock import patch

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
from data_generator.polygon_nasdaq_basic_data_service import NASDAQ_BASIC_SOURCE
from vali_objects.vali_dataclasses.price_source import NASDAQ_BASIC_TRADE_SOURCE


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

    def test_execute_order_max_orders_per_position_rejects(self):
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
        with self.assertRaises(SignalException) as ctx:
            self.execute(
                self.DEFAULT_MINER_HOTKEY, "new_order", OrderType.LONG, 51000.0, value=500.0, now_ms=now_ms2
            )
        self.assertIn("order limit", str(ctx.exception))

        # The position is left untouched: no new position, no auto-close order.
        all_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(all_positions), 1)
        self.assertEqual(all_positions[0].position_uuid, existing_position.position_uuid)
        self.assertFalse(all_positions[0].is_closed_position)
        self.assertEqual(len(all_positions[0].orders), ValiConfig.MAX_ORDERS_PER_POSITION)

    def test_execute_order_multiple_miners_isolated(self):
        miner2 = "miner2"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, miner2])

        _, pos1 = self.execute(self.DEFAULT_MINER_HOTKEY, "m1_order", OrderType.LONG, 50000.0, value=500.0)
        _, pos2 = self.execute(miner2, "m2_order", OrderType.LONG, 50000.0, value=500.0)

        self.assertNotEqual(pos1.miner_hotkey, pos2.miner_hotkey)
        self.assertNotEqual(pos1.position_uuid, pos2.position_uuid)

    def test_execute_order_trigger_price_caps_unfavorable_slippage(self):
        # Slippage would push the LONG fill from 100 to 110, past the trigger (limit/TP/SL)
        # boundary of 105 - the fill must be capped at the boundary with slippage zeroed out.
        order, _ = self.execute(
            self.DEFAULT_MINER_HOTKEY, "uuid_capped", OrderType.LONG, 100.0, value=1000.0,
            slippage=0.10, trigger_price=105.0,
        )
        self.assertEqual(order.price, 105.0)
        self.assertEqual(order.slippage, 0)

        # Same slippage, but the trigger boundary (200) is beyond the natural fill (110) -
        # the more favorable natural market price passes through untouched.
        order2, _ = self.execute(
            self.DEFAULT_MINER_HOTKEY, "uuid_uncapped", OrderType.LONG, 100.0, value=1000.0,
            slippage=0.10, trigger_price=200.0, trade_pair=TradePair.ETHUSDC,
        )
        self.assertEqual(order2.price, 100.0)
        self.assertEqual(order2.slippage, 0.10)

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

        closed = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000
        )

        # The closed pairs come back as plain trade_pair_id strings (they cross RPC)
        self.assertEqual(sorted(closed), sorted(tp.trade_pair_id for tp in trade_pairs))
        self.assertTrue(all(type(trade_pair_id) is str for trade_pair_id in closed))

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

        closed = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[pos2.position_uuid], now_ms=now_ms + 10000
        )

        self.assertEqual(closed, [TradePair.ETHUSDC.trade_pair_id])
        all_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        by_uuid = {p.position_uuid: p for p in all_positions}
        self.assertFalse(by_uuid[pos1.position_uuid].is_closed_position)
        self.assertTrue(by_uuid[pos2.position_uuid].is_closed_position)

    def open_btc_eth_sol(self, now_ms):
        for i, trade_pair in enumerate((TradePair.BTCUSDC, TradePair.ETHUSDC, TradePair.SOLUSDC)):
            self.execute(
                self.DEFAULT_MINER_HOTKEY, f"position_{i}", OrderType.LONG, 1000.0 + i * 10, value=500.0,
                trade_pair=trade_pair, now_ms=now_ms + i,
            )

    def open_trade_pair_ids(self):
        positions = self.position_client.get_positions_for_hotkeys(
            [self.DEFAULT_MINER_HOTKEY], only_open_positions=True
        ).get(self.DEFAULT_MINER_HOTKEY) or []
        return sorted(p.trade_pair.trade_pair_id for p in positions)

    def test_close_positions_leaves_out_a_pair_whose_close_failed(self):
        now_ms = TimeUtil.now_in_millis()
        self.open_btc_eth_sol(now_ms)
        execute_order = self.market_order_manager.execute_order

        def fail_eth(*args, **kwargs):
            if kwargs["trade_pair"] == TradePair.ETHUSDC:
                raise SignalException("Order Rejected: no live prices being found for ETHUSDC. Please try again.")
            return execute_order(*args, **kwargs)

        with patch.object(self.market_order_manager, "execute_order", side_effect=fail_eth):
            closed = self.market_order_manager.close_positions(
                hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000
            )

        # ETH is still open, so it must not be reported: the caller cancels brackets on what is listed
        self.assertEqual(sorted(closed), [TradePair.BTCUSDC.trade_pair_id, TradePair.SOLUSDC.trade_pair_id])
        self.assertEqual(self.open_trade_pair_ids(), [TradePair.ETHUSDC.trade_pair_id])

    def test_close_positions_leaves_out_a_position_that_was_no_longer_open(self):
        now_ms = TimeUtil.now_in_millis()
        self.open_btc_eth_sol(now_ms)
        execute_order = self.market_order_manager.execute_order

        def eth_already_closed(*args, **kwargs):
            if kwargs["trade_pair"] == TradePair.ETHUSDC:
                # Something else (a stop loss, another FLAT) closes ETH between the listing and its
                # turn, so the FLAT finds no open position and returns None
                execute_order(
                    self.DEFAULT_MINER_HOTKEY, "eth_closed_elsewhere", TradePair.ETHUSDC, ExecutionType.MARKET,
                    OrderType.FLAT, OrderSize(), now_ms=now_ms + 5000, enforce_cooldown=False,
                )
            return execute_order(*args, **kwargs)

        with patch.object(self.market_order_manager, "execute_order", side_effect=eth_already_closed):
            closed = self.market_order_manager.close_positions(
                hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000
            )

        self.assertEqual(sorted(closed), [TradePair.BTCUSDC.trade_pair_id, TradePair.SOLUSDC.trade_pair_id])
        self.assertEqual(self.open_trade_pair_ids(), [])

    def test_close_positions_leaves_out_a_position_its_order_did_not_close(self):
        now_ms = TimeUtil.now_in_millis()
        _, position = self.execute(self.DEFAULT_MINER_HOTKEY, "order_1", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms)
        # The position already holds an order under the uuid close_positions uses, so execute_order
        # answers with that order as a replay and closes nothing. Not raising is not closing.
        self.execute(
            self.DEFAULT_MINER_HOTKEY, f"{position.position_uuid}-flat-all", OrderType.LONG, 1000.0, value=100.0,
            now_ms=now_ms + 1,
        )

        closed = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000
        )

        self.assertEqual(closed, [])
        self.assertEqual(self.open_trade_pair_ids(), [self.DEFAULT_TRADE_PAIR.trade_pair_id])

    # A FLAT_ALL resent with the same position uuids: its first attempt closed the positions, but the
    # caller never got to cancel their bracket orders (it timed out or died in between)

    def test_close_positions_reports_the_pair_of_a_named_position_already_closed(self):
        now_ms = TimeUtil.now_in_millis()
        self.open_btc_eth_sol(now_ms)
        by_pair = {p.trade_pair.trade_pair_id: p.position_uuid for p in self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)}
        eth, sol = TradePair.ETHUSDC.trade_pair_id, TradePair.SOLUSDC.trade_pair_id
        first = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[by_pair[eth]], now_ms=now_ms + 10000
        )
        self.assertEqual(first, [eth])

        # The resend names ETH again, and SOL, which is still open
        resent = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[by_pair[eth], by_pair[sol], by_pair[eth]], now_ms=now_ms + 11000
        )

        self.assertEqual(sorted(resent), [eth, sol])
        self.assertEqual(self.open_trade_pair_ids(), [TradePair.BTCUSDC.trade_pair_id])
        # With nothing left open under the named uuids, the pairs are reported all the same
        again = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[by_pair[sol], by_pair[eth]], now_ms=now_ms + 12000
        )
        self.assertEqual(again, [sol, eth])
        self.assertEqual(self.open_trade_pair_ids(), [TradePair.BTCUSDC.trade_pair_id])

    def test_close_positions_leaves_out_a_closed_named_position_whose_pair_was_reopened(self):
        now_ms = TimeUtil.now_in_millis()
        _, old = self.execute(self.DEFAULT_MINER_HOTKEY, "old_order", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms)
        self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[old.position_uuid], now_ms=now_ms + 1000
        )
        _, new = self.execute(self.DEFAULT_MINER_HOTKEY, "new_order", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms + 2000)
        self.assertNotEqual(new.position_uuid, old.position_uuid)

        # The pair has an open position again: its brackets are that position's, so the pair is not
        # reported, and the position named is the old one, so the new one is not closed either
        resent = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[old.position_uuid], now_ms=now_ms + 3000
        )

        self.assertEqual(resent, [])
        open_position = self.position_client.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id
        )
        self.assertEqual(open_position.position_uuid, new.position_uuid)

    def test_close_positions_leaves_out_a_pair_opened_since_the_listing(self):
        now_ms = TimeUtil.now_in_millis()
        _, old = self.execute(self.DEFAULT_MINER_HOTKEY, "old_order", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms)
        self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[old.position_uuid], now_ms=now_ms + 1000
        )
        position_client = self.market_order_manager._position_client
        get_position = position_client.get_position

        def reopen_then_get(hotkey, position_uuid):
            # The pair is opened again after the request listed the open positions (none) and
            # before it looks the named position up
            self.execute(self.DEFAULT_MINER_HOTKEY, "new_order", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms + 2000)
            return get_position(hotkey, position_uuid)

        with patch.object(position_client, "get_position", side_effect=reopen_then_get):
            resent = self.market_order_manager.close_positions(
                hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[old.position_uuid], now_ms=now_ms + 3000
            )

        self.assertEqual(resent, [])
        self.assertEqual(self.open_trade_pair_ids(), [self.DEFAULT_TRADE_PAIR.trade_pair_id])

    def test_close_positions_ignores_a_uuid_of_another_hotkey_or_of_no_position(self):
        other_hotkey = "other_miner"
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY, other_hotkey])
        now_ms = TimeUtil.now_in_millis()
        _, theirs = self.execute(other_hotkey, "their_order", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms)
        self.market_order_manager.close_positions(hotkey=other_hotkey, position_uuids=[theirs.position_uuid], now_ms=now_ms + 1000)
        named = [theirs.position_uuid, "not_a_real_uuid", ""]

        closed = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=named, now_ms=now_ms + 2000
        )

        self.assertEqual(closed, [])
        # The same closed position counts for the hotkey it belongs to
        self.assertEqual(
            self.market_order_manager.close_positions(hotkey=other_hotkey, position_uuids=named, now_ms=now_ms + 2000),
            [self.DEFAULT_TRADE_PAIR.trade_pair_id],
        )

    def test_close_positions_close_all_reports_nothing_once_everything_is_closed(self):
        # A resent "ALL" names no position: what its first attempt closed is not reported again
        now_ms = TimeUtil.now_in_millis()
        self.open_btc_eth_sol(now_ms)
        first = self.market_order_manager.close_positions(hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000)
        self.assertEqual(len(first), 3)

        resent = self.market_order_manager.close_positions(hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 11000)

        self.assertEqual(resent, [])

    def test_close_positions_looks_up_no_closed_position_when_every_named_one_is_open(self):
        now_ms = TimeUtil.now_in_millis()
        self.open_btc_eth_sol(now_ms)
        uuids = [p.position_uuid for p in self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)]
        position_client = self.market_order_manager._position_client

        with patch.object(position_client, "get_position", wraps=position_client.get_position) as get_position:
            closed = self.market_order_manager.close_positions(
                hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=uuids, now_ms=now_ms + 10000
            )
            self.assertEqual(len(closed), 3)
            get_position.assert_not_called()
            self.market_order_manager.close_positions(hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 11000)
            get_position.assert_not_called()

    def test_close_positions_caps_the_closed_position_lookups(self):
        now_ms = TimeUtil.now_in_millis()
        self.open_btc_eth_sol(now_ms)
        uuids = [p.position_uuid for p in self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)]
        self.market_order_manager.close_positions(hotkey=self.DEFAULT_MINER_HOTKEY, close_all=True, now_ms=now_ms + 10000)
        position_client = self.market_order_manager._position_client

        with patch.object(self.market_order_manager, "MAX_CLOSED_POSITION_LOOKUPS", 2), \
                patch.object(position_client, "get_position", wraps=position_client.get_position) as get_position:
            resent = self.market_order_manager.close_positions(
                hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=["unknown_1"] + uuids, now_ms=now_ms + 11000
            )

        # Two lookups: the unknown uuid and the first closed position. The rest is left to the daemon
        self.assertEqual(get_position.call_count, 2)
        self.assertEqual(len(resent), 1)

    def test_close_positions_no_open_positions_is_noop(self):
        # Should not raise even though the miner has no positions at all.
        closed = self.market_order_manager.close_positions(hotkey="miner_with_no_positions", close_all=True)
        self.assertEqual(closed, [])

    def test_close_positions_no_matching_uuids_is_noop(self):
        now_ms = TimeUtil.now_in_millis()
        _, pos = self.execute(self.DEFAULT_MINER_HOTKEY, "order_1", OrderType.LONG, 1000.0, value=500.0, now_ms=now_ms)

        closed = self.market_order_manager.close_positions(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=["not_a_real_uuid"], now_ms=now_ms + 10000
        )

        self.assertEqual(closed, [])
        refreshed = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertFalse(next(p for p in refreshed if p.position_uuid == pos.position_uuid).is_closed_position)

    # ============================================================================
    # Test: equity fills from Nasdaq Basic quotes
    # ============================================================================

    def inject_equity_prices(self, now_ms, nasdaq_bid=None, nasdaq_ask=None):
        """Business FMV at 100.0, plus a Nasdaq Basic quote when bid/ask are given."""
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(TradePair.AAPL, PriceSource(
            source='Polygon_ws', open=100.0, close=100.0, high=100.0, low=100.0, vwap=100.0,
            start_ms=now_ms - 50, websocket=True, bid=100.0, ask=100.0))
        if nasdaq_bid is not None:
            mid = (nasdaq_bid + nasdaq_ask) / 2
            self.live_price_fetcher_client.set_test_price_source(TradePair.AAPL, PriceSource(
                source=NASDAQ_BASIC_SOURCE, open=mid, close=mid, high=mid, low=mid, vwap=mid,
                start_ms=now_ms - 100, websocket=True, bid=nasdaq_bid, ask=nasdaq_ask))

    def execute_live_equity(self, order_uuid, order_type, now_ms, value=None, bracket_pct=None):
        """No fill_price/price_sources, so the manager prices the order from the live price fetcher."""
        return self.market_order_manager.execute_order(
            self.DEFAULT_MINER_HOTKEY, order_uuid, TradePair.AAPL, ExecutionType.MARKET, order_type,
            OrderSize(value=value, bracket_pct=bracket_pct), now_ms=now_ms, enforce_cooldown=False,
        )

    def test_equity_long_fills_at_nasdaq_ask(self):
        now_ms = TimeUtil.now_in_millis()
        self.inject_equity_prices(now_ms, 99.99, 100.01)
        order, _ = self.execute_live_equity("aapl_long", OrderType.LONG, now_ms, value=500.0)
        self.assertEqual(order.price, 100.01)
        self.assertEqual((order.bid, order.ask), (99.99, 100.01))
        self.assertEqual(order.price_sources[0].source, NASDAQ_BASIC_SOURCE)

    def test_equity_short_fills_at_nasdaq_bid(self):
        now_ms = TimeUtil.now_in_millis()
        self.inject_equity_prices(now_ms, 99.99, 100.01)
        order, _ = self.execute_live_equity("aapl_short", OrderType.SHORT, now_ms, value=-500.0)
        self.assertEqual(order.price, 99.99)

    def test_equity_close_of_long_fills_at_nasdaq_bid(self):
        now_ms = TimeUtil.now_in_millis()
        self.inject_equity_prices(now_ms, 99.99, 100.01)
        self.execute_live_equity("aapl_open", OrderType.LONG, now_ms, value=500.0)
        close_order, position = self.execute_live_equity("aapl_close", OrderType.FLAT, now_ms + 1, bracket_pct=1.0)
        self.assertTrue(position.is_closed_position)
        self.assertEqual(close_order.price, 99.99)

    def test_equity_invalid_quote_falls_back_to_fmv(self):
        now_ms = TimeUtil.now_in_millis()
        self.inject_equity_prices(now_ms, 99.0, 101.0)  # 200 bps spread exceeds the cap
        order, _ = self.execute_live_equity("aapl_wide", OrderType.LONG, now_ms, value=500.0)
        self.assertEqual(order.price, 100.0)
        self.assertEqual(order.price_sources[0].source, 'Polygon_ws')

    def test_equity_without_nasdaq_basic_fills_at_fmv(self):
        now_ms = TimeUtil.now_in_millis()
        self.inject_equity_prices(now_ms)
        order, _ = self.execute_live_equity("aapl_fmv", OrderType.LONG, now_ms, value=500.0)
        self.assertEqual(order.price, 100.0)
        self.assertEqual(order.price_sources[0].source, 'Polygon_ws')

    # ============================================================================
    # Test: pre-market / after-hours
    # ============================================================================

    def set_session(self, session):
        self.live_price_fetcher_client.set_test_market_open(session == 'regular')
        self.live_price_fetcher_client.set_test_equity_session(session)

    def test_market_order_rejected_in_extended_hours(self):
        for session in ('pre', 'post'):
            self.set_session(session)
            with self.assertRaises(SignalException) as ctx:
                self.execute_live_equity(f"aapl_{session}", OrderType.LONG, TimeUtil.now_in_millis(), value=500.0)
            self.assertIn("Only limit orders are accepted", str(ctx.exception))

    def test_orders_rejected_when_session_closed(self):
        self.set_session('closed')
        with self.assertRaises(SignalException) as ctx:
            self.execute_live_equity("aapl_closed", OrderType.LONG, TimeUtil.now_in_millis(), value=500.0)
        self.assertIn("currently closed", str(ctx.exception))

    def test_limit_fill_allowed_in_extended_hours_at_fill_price(self):
        self.set_session('post')
        now_ms = TimeUtil.now_in_millis()
        trade = PriceSource(source=NASDAQ_BASIC_TRADE_SOURCE, open=99.5, close=99.5, high=99.5, low=99.5, vwap=99.5,
                            start_ms=now_ms, websocket=True, bid=99.5, ask=99.5)
        order, position = self.market_order_manager.execute_order(
            self.DEFAULT_MINER_HOTKEY, "aapl_limit_post", TradePair.AAPL, ExecutionType.LIMIT, OrderType.LONG,
            OrderSize(value=500.0), fill_price=100.0, trigger_price=100.0, price_sources=[trade], slippage=0,
            order_src=OrderSource.LIMIT_FILLED, now_ms=now_ms, enforce_cooldown=False,
        )
        self.assertEqual(order.price, 100.0)
        self.assertFalse(position.is_closed_position)
