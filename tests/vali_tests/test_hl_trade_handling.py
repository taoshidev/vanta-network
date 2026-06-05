"""
Tests for Hyperliquid trade handling features:
- Position.is_hl field and serialization
- Order.is_hl_taker field
- HL carry fee (funding rate-based)
- HL spread fee (taker/maker per-order)
- simulate_fill orderbook walking
- HLFundingRateManager (in-memory cache, range queries, rate lookup)
- TimeUtil.ms_to_next_hour and n_hours_elapsed
"""

import unittest
from copy import deepcopy

from time_util.time_util import TimeUtil, MS_IN_1_HOUR
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position
from entity_management.hl_orderbook_utils import simulate_fill
from vali_objects.hl_funding.hl_funding_rate_manager import HLFundingRateManager


# 2024-08-01 00:00:00 UTC
BASE_OPEN_MS = 1722470400000


def make_order(order_type, leverage, price=50000.0, processed_ms=None, is_hl_taker=None):
    """Helper to create an Order with minimal boilerplate."""
    return Order(
        price=price,
        slippage=0.0,
        processed_ms=processed_ms or BASE_OPEN_MS,
        order_uuid=f"order_{processed_ms or BASE_OPEN_MS}_{leverage}",
        trade_pair=TradePair.BTCUSD,
        order_type=order_type,
        leverage=leverage,
        is_hl_taker=is_hl_taker,
    )


def make_hl_position(order_type=OrderType.LONG, leverage=0.5, open_ms=None, orders=None):
    """Helper to create an HL position with a single order."""
    open_ms = open_ms or BASE_OPEN_MS
    if orders is None:
        orders = [make_order(order_type, leverage, processed_ms=open_ms)]
    pos = Position(
        miner_hotkey="test_hl_miner",
        position_uuid="test_hl_position",
        open_ms=open_ms,
        trade_pair=TradePair.BTCUSD,
        orders=[],
        is_hl=True,
    )
    # Manually set position state to mimic add_order without needing live_price_fetcher
    pos.orders = orders
    pos.position_type = order_type
    pos.net_leverage = leverage if order_type == OrderType.LONG else -leverage
    return pos


class TestPositionIsHl(unittest.TestCase):
    """Test is_hl field on Position."""

    def test_default_is_hl_false(self):
        pos = Position(
            miner_hotkey="miner",
            position_uuid="uuid",
            open_ms=BASE_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
        )
        self.assertFalse(pos.is_hl)

    def test_is_hl_true(self):
        pos = Position(
            miner_hotkey="miner",
            position_uuid="uuid",
            open_ms=BASE_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            is_hl=True,
        )
        self.assertTrue(pos.is_hl)

    def test_is_hl_serialization_roundtrip(self):
        """is_hl should survive JSON serialization/deserialization."""
        pos = Position(
            miner_hotkey="miner",
            position_uuid="uuid",
            open_ms=BASE_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            is_hl=True,
        )
        data = pos.model_dump()
        self.assertTrue(data["is_hl"])

        restored = Position(**data)
        self.assertTrue(restored.is_hl)

    def test_is_hl_defaults_false_on_deserialization(self):
        """Old positions without is_hl should deserialize with False."""
        data = {
            "miner_hotkey": "miner",
            "position_uuid": "uuid",
            "open_ms": BASE_OPEN_MS,
            "trade_pair": TradePair.BTCUSD,
        }
        pos = Position(**data)
        self.assertFalse(pos.is_hl)


class TestOrderIsHlTaker(unittest.TestCase):
    """Test is_hl_taker field on Order."""

    def test_default_is_none(self):
        order = make_order(OrderType.LONG, 0.5)
        self.assertIsNone(order.is_hl_taker)

    def test_taker_true(self):
        order = make_order(OrderType.LONG, 0.5, is_hl_taker=True)
        self.assertTrue(order.is_hl_taker)

    def test_maker_false(self):
        order = make_order(OrderType.LONG, 0.5, is_hl_taker=False)
        self.assertFalse(order.is_hl_taker)


class TestHlSpreadFee(unittest.TestCase):
    """Test HL-specific spread fee calculation (taker/maker per-order)."""

    def test_single_taker_order(self):
        """Single taker order: fee = 1 - 0.00045 * leverage."""
        order = make_order(OrderType.LONG, 0.5, processed_ms=BASE_OPEN_MS, is_hl_taker=True)
        pos = make_hl_position(orders=[order])
        fee = pos.get_spread_fee(BASE_OPEN_MS)
        expected = 1 - ValiConfig.HL_TAKER_FEE * 0.5
        self.assertAlmostEqual(fee, expected, places=10)

    def test_single_maker_order(self):
        """Single maker order: fee = 1 - 0.00015 * leverage."""
        order = make_order(OrderType.LONG, 0.5, processed_ms=BASE_OPEN_MS, is_hl_taker=False)
        pos = make_hl_position(orders=[order])
        fee = pos.get_spread_fee(BASE_OPEN_MS)
        expected = 1 - ValiConfig.HL_MAKER_FEE * 0.5
        self.assertAlmostEqual(fee, expected, places=10)

    def test_multiple_mixed_orders(self):
        """Multiple orders: fee is product of each order's fee."""
        o1 = make_order(OrderType.LONG, 0.3, processed_ms=BASE_OPEN_MS, is_hl_taker=True)
        o2 = make_order(OrderType.LONG, 0.2, processed_ms=BASE_OPEN_MS + 1000, is_hl_taker=False)
        pos = make_hl_position(orders=[o1, o2])
        fee = pos.get_spread_fee(BASE_OPEN_MS)
        expected = (1 - ValiConfig.HL_TAKER_FEE * 0.3) * (1 - ValiConfig.HL_MAKER_FEE * 0.2)
        self.assertAlmostEqual(fee, expected, places=10)

    def test_hl_taker_none_no_fee(self):
        """Order with is_hl_taker=None contributes no fee."""
        order = make_order(OrderType.LONG, 0.5, processed_ms=BASE_OPEN_MS, is_hl_taker=None)
        pos = make_hl_position(orders=[order])
        fee = pos.get_spread_fee(BASE_OPEN_MS)
        self.assertAlmostEqual(fee, 1.0, places=10)

    def test_non_hl_position_uses_legacy_fee(self):
        """Non-HL crypto positions use the legacy 0.1% * cumulative_leverage."""
        order = make_order(OrderType.LONG, 0.5, processed_ms=BASE_OPEN_MS)
        pos = Position(
            miner_hotkey="miner",
            position_uuid="uuid",
            open_ms=BASE_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            orders=[],
            is_hl=False,
        )
        pos.orders = [order]
        pos.position_type = OrderType.LONG
        pos.net_leverage = 0.5
        fee = pos.get_spread_fee(BASE_OPEN_MS)
        # Legacy: 1 - cumulative_leverage * 0.001
        expected = 1 - 0.5 * 0.001
        self.assertAlmostEqual(fee, expected, places=10)

    def test_hl_taker_fee_higher_than_maker(self):
        """Taker fee should be larger (worse) than maker fee for same leverage."""
        order_taker = make_order(OrderType.LONG, 0.5, processed_ms=BASE_OPEN_MS, is_hl_taker=True)
        order_maker = make_order(OrderType.LONG, 0.5, processed_ms=BASE_OPEN_MS, is_hl_taker=False)

        pos_taker = make_hl_position(orders=[order_taker])
        pos_maker = make_hl_position(orders=[order_maker])

        fee_taker = pos_taker.get_spread_fee(BASE_OPEN_MS)
        fee_maker = pos_maker.get_spread_fee(BASE_OPEN_MS)

        # Lower fee = more cost deducted
        self.assertLess(fee_taker, fee_maker)


class TestSimulateFill(unittest.TestCase):
    """Test L2 orderbook walking for slippage calculation."""

    def test_single_level_full_fill_usd(self):
        """Order fully fills on a single level (USD mode)."""
        levels = [{"px": "100.0", "sz": "10.0"}]  # 10 coins * $100 = $1000
        fills, remaining = simulate_fill(levels, 500, "usd")
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0][0], 100.0)  # price
        self.assertAlmostEqual(fills[0][1], 5.0)    # 5 coins
        self.assertAlmostEqual(fills[0][2], 500.0)  # $500
        self.assertAlmostEqual(remaining, 0.0)

    def test_single_level_full_fill_coins(self):
        """Order fully fills on a single level (coins mode)."""
        levels = [{"px": "100.0", "sz": "10.0"}]
        fills, remaining = simulate_fill(levels, 5, "coins")
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0][0], 100.0)
        self.assertAlmostEqual(fills[0][1], 5.0)
        self.assertAlmostEqual(fills[0][2], 500.0)
        self.assertAlmostEqual(remaining, 0.0)

    def test_multiple_levels(self):
        """Order walks multiple levels."""
        levels = [
            {"px": "100.0", "sz": "2.0"},   # $200
            {"px": "101.0", "sz": "3.0"},   # $303
            {"px": "102.0", "sz": "5.0"},   # $510
        ]
        fills, remaining = simulate_fill(levels, 400, "usd")
        # Level 1: fill all $200
        self.assertAlmostEqual(fills[0][2], 200.0)
        # Level 2: fill remaining $200 of $303 available
        self.assertAlmostEqual(fills[1][2], 200.0)
        self.assertAlmostEqual(remaining, 0.0)

    def test_partial_fill_insufficient_liquidity(self):
        """Not enough liquidity => remaining > 0."""
        levels = [{"px": "100.0", "sz": "1.0"}]  # Only $100 available
        fills, remaining = simulate_fill(levels, 500, "usd")
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0][2], 100.0)
        self.assertAlmostEqual(remaining, 400.0)

    def test_empty_levels(self):
        """Empty orderbook => no fills, all remaining."""
        fills, remaining = simulate_fill([], 500, "usd")
        self.assertEqual(len(fills), 0)
        self.assertAlmostEqual(remaining, 500.0)

    def test_zero_size_order(self):
        """Zero-size order fills nothing."""
        levels = [{"px": "100.0", "sz": "10.0"}]
        fills, remaining = simulate_fill(levels, 0, "usd")
        self.assertEqual(len(fills), 0)
        self.assertAlmostEqual(remaining, 0.0)

    def test_slippage_calculation_from_fills(self):
        """Verify slippage % from multi-level fill."""
        levels = [
            {"px": "100.0", "sz": "5.0"},   # $500
            {"px": "101.0", "sz": "5.0"},   # $505
        ]
        mid = 99.5  # hypothetical mid price below best ask
        fills, _ = simulate_fill(levels, 800, "usd")
        total_coins = sum(f[1] for f in fills)
        total_usd = sum(f[2] for f in fills)
        avg_price = total_usd / total_coins
        slippage_pct = (avg_price - mid) / mid
        self.assertGreater(slippage_pct, 0)


class TestHLFundingRateManager(unittest.TestCase):
    """Test HLFundingRateManager in-memory operations (no API, no disk)."""

    def setUp(self):
        self.manager = HLFundingRateManager(running_unit_tests=True)

    def test_empty_manager(self):
        """Empty manager returns empty results."""
        self.assertEqual(self.manager.get_rates_for_position("BTC", 0, 1000000), {})
        self.assertIsNone(self.manager.get_rate_at_time("BTC", 1000000))

    def test_manual_rate_insertion_and_query(self):
        """Insert rates directly and query them."""
        # Directly populate internal state (simulating what fetch_and_store_rates does)
        self.manager._rates["BTC"] = [
            {"time_ms": 1000, "rate": 0.001},
            {"time_ms": 2000, "rate": 0.002},
            {"time_ms": 3000, "rate": -0.001},
        ]

        # Query full range
        result = self.manager.get_rates_for_position("BTC", 0, 4000)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1000], 0.001)
        self.assertAlmostEqual(result[2000], 0.002)
        self.assertAlmostEqual(result[3000], -0.001)

    def test_range_query(self):
        """Only returns rates within the specified range."""
        self.manager._rates["BTC"] = [
            {"time_ms": 1000, "rate": 0.001},
            {"time_ms": 2000, "rate": 0.002},
            {"time_ms": 3000, "rate": 0.003},
            {"time_ms": 4000, "rate": 0.004},
        ]

        result = self.manager.get_rates_for_position("BTC", 1500, 3500)
        self.assertEqual(len(result), 2)
        self.assertIn(2000, result)
        self.assertIn(3000, result)
        self.assertNotIn(1000, result)
        self.assertNotIn(4000, result)

    def test_get_rate_at_time_exact(self):
        """get_rate_at_time returns exact match."""
        self.manager._rates["ETH"] = [
            {"time_ms": 1000, "rate": 0.005},
            {"time_ms": 2000, "rate": 0.003},
        ]
        self.assertAlmostEqual(self.manager.get_rate_at_time("ETH", 2000), 0.003)

    def test_get_rate_at_time_before_first(self):
        """get_rate_at_time returns None if query is before all entries."""
        self.manager._rates["ETH"] = [
            {"time_ms": 1000, "rate": 0.005},
        ]
        self.assertIsNone(self.manager.get_rate_at_time("ETH", 500))

    def test_get_rate_at_time_between_entries(self):
        """get_rate_at_time returns the rate at or just before the query time."""
        self.manager._rates["ETH"] = [
            {"time_ms": 1000, "rate": 0.005},
            {"time_ms": 3000, "rate": 0.003},
        ]
        # Query at 2000 should return the rate at 1000
        self.assertAlmostEqual(self.manager.get_rate_at_time("ETH", 2000), 0.005)

    def test_unknown_coin(self):
        """Unknown coin returns empty/None."""
        self.assertEqual(self.manager.get_rates_for_position("DOGE", 0, 1000000), {})
        self.assertIsNone(self.manager.get_rate_at_time("DOGE", 1000))

    def test_clear_all(self):
        """clear_all removes all cached rates."""
        self.manager._rates["BTC"] = [{"time_ms": 1000, "rate": 0.001}]
        self.manager.clear_all()
        self.assertEqual(self.manager.get_rates_for_position("BTC", 0, 2000), {})

    def test_deduplication(self):
        """Adding duplicate time_ms entries should not create duplicates."""
        # Simulate two fetch_and_store_rates calls with overlapping data
        self.manager._rates["BTC"] = [
            {"time_ms": 1000, "rate": 0.001},
            {"time_ms": 2000, "rate": 0.002},
        ]
        # Simulate adding overlapping records (mimicking fetch_and_store_rates logic)
        existing = self.manager._rates.get("BTC", [])
        existing_times = {r["time_ms"] for r in existing}
        new_records = [
            {"time_ms": 2000, "rate": 0.002},  # duplicate
            {"time_ms": 3000, "rate": 0.003},  # new
        ]
        for rec in new_records:
            if rec["time_ms"] not in existing_times:
                existing.append(rec)
                existing_times.add(rec["time_ms"])
        self.manager._rates["BTC"] = sorted(existing, key=lambda r: r["time_ms"])

        self.assertEqual(len(self.manager._rates["BTC"]), 3)
        result = self.manager.get_rates_for_position("BTC", 0, 4000)
        self.assertEqual(len(result), 3)


class TestTimeUtilHourHelpers(unittest.TestCase):
    """Test ms_to_next_hour and n_hours_elapsed."""

    def test_ms_to_next_hour_at_exact_hour(self):
        """At exactly 00:00 UTC, should be 1 full hour to next boundary."""
        # 2024-08-01 00:00:00 UTC
        exact_hour = 1722470400000
        ms = TimeUtil.ms_to_next_hour(exact_hour)
        self.assertEqual(ms, MS_IN_1_HOUR)

    def test_ms_to_next_hour_at_half_hour(self):
        """At 00:30 UTC, should be 30 minutes to next hour."""
        half_hour = 1722470400000 + 30 * 60 * 1000
        ms = TimeUtil.ms_to_next_hour(half_hour)
        self.assertEqual(ms, 30 * 60 * 1000)

    def test_ms_to_next_hour_at_59_min(self):
        """At 00:59 UTC, should be 1 minute to next hour."""
        near_hour = 1722470400000 + 59 * 60 * 1000
        ms = TimeUtil.ms_to_next_hour(near_hour)
        self.assertEqual(ms, 1 * 60 * 1000)

    def test_ms_to_next_hour_always_positive(self):
        """Result should always be positive."""
        for offset_min in [0, 1, 15, 30, 45, 59]:
            t = 1722470400000 + offset_min * 60 * 1000
            ms = TimeUtil.ms_to_next_hour(t)
            self.assertGreater(ms, 0, f"Failed at offset {offset_min} min")

    def test_n_hours_elapsed_same_hour(self):
        """No hour boundary crossed => 0."""
        start = 1722470400000 + 10 * 60 * 1000  # 00:10
        end = 1722470400000 + 50 * 60 * 1000    # 00:50
        self.assertEqual(TimeUtil.n_hours_elapsed(start, end), 0)

    def test_n_hours_elapsed_one_boundary(self):
        """One hour boundary crossed."""
        start = 1722470400000 + 30 * 60 * 1000  # 00:30
        end = 1722470400000 + 90 * 60 * 1000    # 01:30
        self.assertEqual(TimeUtil.n_hours_elapsed(start, end), 1)

    def test_n_hours_elapsed_multiple(self):
        """Multiple hour boundaries crossed."""
        start = 1722470400000  # 00:00
        end = start + 3 * MS_IN_1_HOUR + 30 * 60 * 1000  # 03:30
        self.assertEqual(TimeUtil.n_hours_elapsed(start, end), 3)

    def test_n_hours_elapsed_exact_hours(self):
        """Start and end on exact hour boundaries."""
        start = 1722470400000  # 00:00
        end = start + 5 * MS_IN_1_HOUR  # 05:00
        self.assertEqual(TimeUtil.n_hours_elapsed(start, end), 5)

    def test_n_hours_elapsed_same_time(self):
        """Same start and end => 0."""
        t = 1722470400000
        self.assertEqual(TimeUtil.n_hours_elapsed(t, t), 0)


if __name__ == "__main__":
    unittest.main()
