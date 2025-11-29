import copy
import unittest
from copy import deepcopy

import numpy as np
from tests.shared_objects.mock_classes import MockLivePriceFetcher

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.risk_profiling import RiskProfiling
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils

class TestRiskProfile(TestBase):
    """
    This class tests the risk profiling functionality
    """

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_ORDER_DIRECTION = OrderType.LONG
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_ORDER_MS = 1000
        self.DEFAULT_PRICE = 1000
        self.DEFAULT_LEVERAGE = 1.0
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_OPEN = False
        self.DEFAULT_ACCOUNT_SIZE = 100_000

        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            is_closed_position=self.DEFAULT_OPEN,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.default_order = Order(
            order_uuid=self.DEFAULT_ORDER_UUID,
            order_type=self.DEFAULT_ORDER_DIRECTION,
            leverage=self.DEFAULT_LEVERAGE,
            price=self.DEFAULT_PRICE,
            processed_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True)
        self.position_manager.clear_all_miner_positions()

    def tearDown(self):
        super().tearDown()

    def check_write_position(self, position: Position):
        position_trade_pair = position.trade_pair
        position_hotkey = position.hotkey

        self.position_manager.save_miner_position(position)
        self.position_manager.get_open_position_for_a_miner_trade_pair(position_hotkey, position_trade_pair)
        self.assertEqual(len(self.position_manager.get_miner_positions()), 1, "Position should be saved to disk")

    def test_monotonic_positions_one(self):
        """Test the monotonically increasing leverage detection with various edge cases"""
        # Test with empty position (no orders)
        position1 = deepcopy(self.default_position)
        result = RiskProfiling.monotonic_positions(position1)
        self.assertEqual(len(result.orders), 0, "Empty position should result in empty monotonic position")

        # Test with single order
        order1 = copy.deepcopy(self.default_order)
        order1.order_uuid = "order1"
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = 1000
        position1.add_order(order1, self.live_price_fetcher)
        result = RiskProfiling.monotonic_positions(position1)
        self.assertEqual(len(result.orders), 0, "Position with single order should result in empty monotonic position")


    def test_mono_positions_winning_standard(self):
        # Test with winning positions only (should not flag)
        position2 = deepcopy(self.default_position)
        order2 = copy.deepcopy(self.default_order)
        order2.order_uuid = "order2"
        order2.leverage = 0.1
        order2.price = 150
        order2.processed_ms = 1000 + (1000 * 60 * 60 * 24)
        position2.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_uuid = "order3"
        order3.leverage = 0.1
        order3.price = 200
        order3.processed_ms = 1000 + (1000 * 60 * 60 * 24 * 2)
        position2.add_order(order3, self.live_price_fetcher)

        order4 = copy.deepcopy(self.default_order)
        order4.order_uuid = "order4"
        order4.leverage = 0.1
        order4.price = 250
        order4.processed_ms = 1000 + (1000 * 60 * 60 * 24 * 3)
        position2.add_order(order4, self.live_price_fetcher)

        result = RiskProfiling.monotonic_positions(position2)
        self.assertEqual(len(result.orders), 0, "Winning positions should not be flagged")


    def test_mono_positions_losing_increasing_slowly(self):
        # Test with losing positions but decreasing additional leverage
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)

        order1.leverage = 0.3
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.2
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 2, "Losing positions with increasing total leverage should be flagged")


    def test_mono_positions_losing_decreasing_standard(self):
        # Test with losing positions and incrementally decreasing leverage (negative increments)
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.3
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = -0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = -0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Losing positions with decreasing leverage should not be flagged")

    def test_mono_positions_winning_increasing_standard(self):
        # Test SHORT positions with increasing leverage on losing trade
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT

        order1 = copy.deepcopy(self.default_order)
        order1.order_type = OrderType.SHORT
        order1.leverage = -0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.order_type = OrderType.SHORT
        order2.leverage = -0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_type = OrderType.SHORT
        order3.leverage = -0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 0, "Winning SHORT positions should not be flagged")

    def test_mono_positions_losing_increasing_standard_short(self):
        # Test SHORT positions with increasing leverage on losing trade
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT

        order1 = copy.deepcopy(self.default_order)
        order1.order_type = OrderType.SHORT
        order1.leverage = -0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.order_type = OrderType.SHORT
        order2.leverage = -0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_type = OrderType.SHORT
        order3.leverage = -0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.monotonic_positions(position)
        self.assertEqual(len(result.orders), 2, "Losing SHORT positions with increasing leverage should be flagged")

    def test_mono_positions_closed_position_increasing_rapidly(self):
        # Test with closed position (final order should be ignored)
        position = deepcopy(self.default_position)

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.15
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.5
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        order4 = copy.deepcopy(self.default_order)
        order4.order_type = OrderType.FLAT
        order4.leverage = 0.0
        order4.price = 70
        order4.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 4)
        position.add_order(order4, self.live_price_fetcher)
        position.is_closed_position = True

        result = RiskProfiling.monotonic_positions(position)

        self.assertEqual([x.leverage for x in result.orders], [0.25, 0.75])
        self.assertEqual(len(result.orders), 2, "Should only count orders before closing order")

    def test_risk_assessment_steps_utilization(self):
        """Test the steps utilization function"""
        # Test with empty position (no orders)
        position = deepcopy(self.default_position)
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Empty position should have 0 steps")

        # Test with single order
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Position with single order should have 0 steps")

    def test_risk_assessment_steps_utilization_positive(self):
        # Test with winning positions only
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Winning positions should have 0 steps")

    def test_risk_assessment_steps_utilization_negative(self):
        # Test with losing positions and decreasing leverage (negative increments)
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.3
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = -0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = -0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 0, "Losing positions with decreasing leverage should have 0 steps")

    def test_risk_assessment_steps_utilization_positive_increasing(self):
        # Test with losing positions and increasing leverage
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 2, "Losing positions with increasing leverage should have 2 steps")

    def test_risk_assessment_steps_utilization_negative_increasing(self):
        # Test with SHORT positions on losing trade
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT

        order1 = copy.deepcopy(self.default_order)
        order1.order_type = OrderType.SHORT
        order1.leverage = -0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.order_type = OrderType.SHORT
        order2.leverage = -0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_type = OrderType.SHORT
        order3.leverage = -0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        self.assertEqual(len(position.orders), 3, "Position should have 3 orders")
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 2, "Losing SHORT positions with increasing leverage should have 2 steps")

    def test_risk_assessment_steps_utilization_closed_position(self):
        # Test with closed position
        position = deepcopy(self.default_position)
        position.position_type = OrderType.LONG

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        order4 = copy.deepcopy(self.default_order)
        order4.order_type = OrderType.FLAT
        order4.leverage = 0.0
        order4.price = 70
        order4.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 3)
        position.add_order(order4, self.live_price_fetcher)

        position.is_closed_position = True
        result = RiskProfiling.risk_assessment_steps_utilization(position)
        self.assertEqual(result, 2, "Should only count steps before closing order")

    def test_risk_assessment_monotonic_utilization(self):
        """Test the monotome utilization function"""
        # Test with empty position
        position = deepcopy(self.default_position)
        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 0, "Empty position should have 0 monotonic utilization")

        # Test with single order
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 0, "Position with single order should have 0 monotonic utilization")

    def test_risk_assessment_monotonic_utilization_positive(self):
        # Test with winning positions only
        position = deepcopy(self.default_position)
        position.position_type = OrderType.LONG

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.11
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.12
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 0, "Winning positions should have 0 monotonic utilization")

    def test_risk_assessment_monotonic_utilization_losing(self):
        # Test with losing positions and monotonically increasing leverage
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.2
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_monotonic_utilization(position)
        self.assertEqual(result, 2, "Losing positions with increasing leverage should have 2 monotonic utilization")

    def test_risk_assessment_margin_utilization(self):
        """Test the margin utilization function"""
        # Test with position having low margin utilization
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.01
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.01
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.01
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_margin_utilization(position)
        self.assertLess(result, 0.1, "Low leverage should result in low margin utilization")

    def test_margin_utilization_increasing_slowly_winning(self):
        # Test with position having high margin utilization
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.4
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.05
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.02
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_margin_utilization(position)
        self.assertGreater(result, 0.8, "High leverage should result in high margin utilization")

    def test_margin_utilization_increasing_slowly_winning_short(self):
        # Test with SHORT position
        position = deepcopy(self.default_position)
        position.position_type = OrderType.SHORT

        order1 = copy.deepcopy(self.default_order)
        order1.order_type = OrderType.SHORT
        order1.leverage = -0.4
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.order_type = OrderType.SHORT
        order2.leverage = -0.05
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_type = OrderType.SHORT
        order3.leverage = -0.02
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_margin_utilization(position)
        self.assertGreater(result, 0.8, "High leverage SHORT position should result in high margin utilization")

    def test_risk_assessment_leverage_advancement_utilization(self):
        """Test the leverage advancement utilization function"""
        # Test with position having low leverage advancement
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.01
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.01
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        self.assertLess(result, 1.5, "Position with small leverage advancement should have low utilization")

    def test_risk_assessment_leverage_advancement_utilization_positive(self):
        # Test with position having zero initial leverage (edge case)
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.05
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.15
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        self.assertGreaterEqual(result, 1.0, "Position with zero initial leverage should return at least 1.0")

    def test_risk_assessment_leverage_advancement_utilization_high(self):
        # Test with position having high leverage advancement
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.2
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.3
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_leverage_advancement_utilization(position)
        self.assertGreaterEqual(result, 6.0, "Position with large leverage advancement should have high utilization")

    def test_risk_assessment_time_utilization(self):
        """Test the time utilization function"""
        # Test with position having fewer than 3 orders
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Position with fewer than 3 orders should have 0 time utilization")

    def test_time_utilization_even_intervals(self):
        # Test with empty position
        position = deepcopy(self.default_position)
        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Empty position should have 0 time utilization")

        # Test with position having perfectly even time intervals
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Position with even time intervals should have 0 time utilization")

    def test_time_utilization_even_intervals_shorter(self):
        # Test with position having uneven time intervals
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 12) # Half a day
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertGreater(result, 0.0, "Position with uneven time intervals should have positive time utilization")

    def test_time_utilization_zero_intervals(self):
        # Test with position having orders with zero time interval
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_assessment_time_utilization(position)
        self.assertEqual(result, 0.0, "Position with zero time intervals should handle the edge case")

    def test_risk_profile_single(self):
        """Test the risk profile single function"""
        # Create a position that should trigger risk flags
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.2
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.3
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        order4 = copy.deepcopy(self.default_order)
        order4.leverage = 0.4
        order4.price = 70
        order4.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 3)
        position.add_order(order4, self.live_price_fetcher)

        position.return_at_close = 0.9  # 10% loss

        result = RiskProfiling.risk_profile_single(position)

        # Verify the keys in the result
        expected_keys = [
            "position_return", "relative_weighting_strength", "overall_flag",
            "steps_utilization", "steps_criteria",
            "monotonic_utilization", "monotonic_criteria",
            "margin_utilization", "margin_criteria",
            "leverage_advancement_utilization", "leverage_advancement_criteria",
            "time_utilization", "time_criteria",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Key {key} missing from risk profile result")

        # Verify types of values
        self.assertIsInstance(result["position_return"], float)
        self.assertIsInstance(result["relative_weighting_strength"], float)
        self.assertIsInstance(result["overall_flag"], int)
        self.assertIsInstance(result["steps_utilization"], int)
        self.assertIsInstance(result["steps_criteria"], int)

    def test_risk_profile_reporting(self):
        """Test the risk profile reporting function"""
        # Test with empty list
        result = RiskProfiling.risk_profile_reporting([])
        self.assertEqual(result, {}, "Empty position list should return empty dict")

        # Test with single position
        position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_profile_reporting([position])
        self.assertEqual(len(result), 1, "Should contain one entry for one position")
        self.assertIn(position.position_uuid, result, "Position UUID should be in result")

        # Test with multiple positions
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position_2"
        position2.position_type = OrderType.SHORT

        order1 = copy.deepcopy(self.default_order)
        order1.order_type = OrderType.SHORT
        order1.leverage = -0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        position2.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.order_type = OrderType.SHORT
        order2.leverage = -0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        position2.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_type = OrderType.SHORT
        order3.leverage = -0.1
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        position2.add_order(order3, self.live_price_fetcher)

        result = RiskProfiling.risk_profile_reporting([position, position2])
        self.assertEqual(len(result), 2, "Should contain two entries for two positions")
        self.assertIn(position.position_uuid, result, "First position UUID should be in result")
        self.assertIn(position2.position_uuid, result, "Second position UUID should be in result")

    def test_risk_profile_score_list(self):
        """Test the risk profile score list function with numerical stability cases"""
        # Test with empty list
        result = RiskProfiling.risk_profile_score_list([])
        self.assertEqual(result, 0.0, "Empty position list should return 0.0")

        # Test with extremely small return values (to check numerical stability)
        small_return_position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        small_return_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        small_return_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        small_return_position.add_order(order3, self.live_price_fetcher)

        # Set a very small return value to test numerical stability
        small_return_position.return_at_close = 0.0001  # 99.99% loss

        # Configure for guaranteed risk flagging
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 1
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.0

        # The function should handle this edge case without numerical errors
        result = RiskProfiling.risk_profile_score_list([small_return_position])
        self.assertGreaterEqual(result, 0.0, "Function should handle extremely small return values")
        self.assertLessEqual(result, 1.0, "Function should handle extremely small return values")

        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

        # Test with single non-risky position
        safe_position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.05
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        safe_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.03
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        safe_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.01
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        safe_position.add_order(order3, self.live_price_fetcher)

        safe_position.return_at_close = 1.1  # 10% gain

        # Save original thresholds to ensure position is safe
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA

        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.2
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = 2.0
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.0

        result = RiskProfiling.risk_profile_score_list([safe_position])
        self.assertEqual(result, 0.0, "Non-risky position should have score 0.0")

        # Test with single risky position
        risky_position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        risky_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 12) # Half a day
        risky_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.2
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        risky_position.add_order(order3, self.live_price_fetcher)

        risky_position.return_at_close = 0.9  # 10% loss

        result = RiskProfiling.risk_profile_score_list([risky_position])
        self.assertEqual(result, 1.0, "Risky position should have score 1.0")

        # Test with mix of risky and non-risky positions
        result = RiskProfiling.risk_profile_score_list([safe_position, risky_position])

        # The weighted average should consider return_at_close with the RISK_PROFILING_SCOPING_MECHANIC power
        # Safe position has 10% gain, risky position has 10% loss
        # Weight for safe position: 1.1^100
        # Weight for risky position: 0.9^100
        # Risky has score 1, safe has score 0
        # Should be heavily weighted towards safe position's score (0)
        self.assertLess(result, 0.01, "Weighted score should be closer to safe position's score")

        # Restore original thresholds
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

    def test_risk_profile_score(self):
        """Test the risk profile score function with multiple miners"""
        # Create positions for two miners
        safe_position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.05
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        safe_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.03
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        safe_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.01
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        safe_position.add_order(order3, self.live_price_fetcher)

        safe_position.return_at_close = 1.1  # 10% gain

        risky_position = deepcopy(self.default_position)
        risky_position.miner_hotkey = "risky_miner"

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        risky_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 12) # Half a day
        risky_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.2
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        risky_position.add_order(order3, self.live_price_fetcher)

        risky_position.return_at_close = 0.9  # 10% loss

        # Make the risky position actually trigger risk criteria
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA

        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.2
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.1

        miner_positions = {
            "test_miner": [safe_position],
            "risky_miner": [risky_position],
        }

        result = RiskProfiling.risk_profile_score(miner_positions)

        self.assertEqual(len(result), 2, "Should have scores for two miners")
        self.assertIn("test_miner", result, "test_miner should be in results")
        self.assertIn("risky_miner", result, "risky_miner should be in results")

        self.assertEqual(result["test_miner"], 0.0, "Safe miner should have score 0.0")
        self.assertEqual(result["risky_miner"], 1.0, "Risky miner should have score 1.0")

        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

    def test_risk_profile_penalty(self):
        """Test the risk profile penalty function"""
        # Test with empty dict
        result = RiskProfiling.risk_profile_penalty({})
        self.assertEqual(result, {}, "Empty dict should return empty dict")

        # Create positions for two miners
        safe_position = deepcopy(self.default_position)
        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.05
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        safe_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.03
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        safe_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.01
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        safe_position.add_order(order3, self.live_price_fetcher)

        safe_position.return_at_close = 1.1  # 10% gain

        risky_position = deepcopy(self.default_position)
        risky_position.miner_hotkey = "risky_miner"

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        risky_position.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 12) # Half a day
        risky_position.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.2
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        risky_position.add_order(order3, self.live_price_fetcher)

        risky_position.return_at_close = 0.9  # 10% loss

        # Make the risky position actually trigger risk criteria
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA

        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.2
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.1

        miner_positions = {
            "test_miner": [safe_position],
            "risky_miner": [risky_position],
        }

        result = RiskProfiling.risk_profile_penalty(miner_positions)

        self.assertEqual(len(result), 2, "Should have penalties for two miners")
        self.assertIn("test_miner", result, "test_miner should be in results")
        self.assertIn("risky_miner", result, "risky_miner should be in results")

        # Safe miner should have penalty close to 1 (no penalty)
        self.assertGreater(result["test_miner"], 0.98, "Safe miner should have minimal penalty")

        # Risky miner should have significant penalty (less than 0.5)
        self.assertLess(result["risky_miner"], 0.75, "Risky miner should have significant penalty")

        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time

    def test_integration_complete_risk_assessment(self):
        """Integration test for full risk assessment workflow"""
        # Create a diverse set of positions
        positions = []

        # Position 1: Safe position with good returns
        pos1 = deepcopy(self.default_position)
        pos1.position_uuid = "pos1"

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.05
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        pos1.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.03
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        pos1.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.01
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        pos1.add_order(order3, self.live_price_fetcher)

        pos1.return_at_close = 1.2  # 20% gain
        positions.append(pos1)

        # Position 2: Risky position with increasing leverage on losing trade
        pos2 = deepcopy(self.default_position)
        pos2.position_uuid = "pos2"

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        pos2.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 90
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 12) # Half a day
        pos2.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.2
        order3.price = 80
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        pos2.add_order(order3, self.live_price_fetcher)

        pos2.return_at_close = 0.8  # 20% loss
        positions.append(pos2)

        # Position 3: Risky SHORT position with high leverage
        pos3 = deepcopy(self.default_position)
        pos3.position_uuid = "pos3"
        pos3.position_type = OrderType.SHORT

        order1 = copy.deepcopy(self.default_order)
        order1.order_type = OrderType.SHORT
        order1.leverage = -0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        pos3.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.order_type = OrderType.SHORT
        order2.leverage = -0.1
        order2.price = 110
        order2.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24)
        pos3.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.order_type = OrderType.SHORT
        order3.leverage = -0.3
        order3.price = 120
        order3.processed_ms = self.DEFAULT_ORDER_MS + (1000 * 60 * 60 * 24 * 2)
        pos3.add_order(order3, self.live_price_fetcher)

        pos3.return_at_close = 0.9  # 10% loss
        positions.append(pos3)

        # Position 4: Zero-variance position with all orders at same time
        pos4 = deepcopy(self.default_position)
        pos4.position_uuid = "pos4"

        order1 = copy.deepcopy(self.default_order)
        order1.leverage = 0.1
        order1.price = 100
        order1.processed_ms = self.DEFAULT_ORDER_MS
        pos4.add_order(order1, self.live_price_fetcher)

        order2 = copy.deepcopy(self.default_order)
        order2.leverage = 0.1
        order2.price = 100
        order2.processed_ms = self.DEFAULT_ORDER_MS
        pos4.add_order(order2, self.live_price_fetcher)

        order3 = copy.deepcopy(self.default_order)
        order3.leverage = 0.1
        order3.price = 100
        order3.processed_ms = self.DEFAULT_ORDER_MS
        pos4.add_order(order3, self.live_price_fetcher)

        pos4.return_at_close = 1.0  # 0% gain/loss
        positions.append(pos4)

        # Make sure the risky positions actually trigger risk criteria
        original_steps = ValiConfig.RISK_PROFILING_STEPS_CRITERIA
        original_monotonic = ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA
        original_margin = ValiConfig.RISK_PROFILING_MARGIN_CRITERIA
        original_leverage = ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE
        original_time = ValiConfig.RISK_PROFILING_TIME_CRITERIA

        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = 2
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = 0.4
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = 3.0
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = 0.1

        # Test individual profiles
        for pos in positions:
            profile = RiskProfiling.risk_profile_single(pos)
            self.assertIsInstance(profile, dict, f"Profile for {pos.position_uuid} should be a dict")
            self.assertIn("overall_flag", profile, f"Profile for {pos.position_uuid} missing overall_flag")

            # Verify flag states match our expectations
            if pos.position_uuid == "pos1":
                self.assertEqual(profile["overall_flag"], 0, "Safe position should not be flagged")
            elif pos.position_uuid == "pos2":
                self.assertEqual(profile["overall_flag"], 1, "Risky position should be flagged")

        # Test full risk profile report
        report = RiskProfiling.risk_profile_reporting(positions)
        self.assertEqual(len(report), 4, "Report should contain 4 positions")

        # Test with multiple miners
        miner_positions = {
            "safe_miner": [pos1, pos4],
            "risky_miner": [pos2, pos3],
        }

        # Test scores
        scores = RiskProfiling.risk_profile_score(miner_positions)
        self.assertEqual(len(scores), 2, "Should have scores for 2 miners")
        self.assertLess(scores["safe_miner"], scores["risky_miner"], "Safe miner should have lower risk score")

        # Test penalties
        penalties = RiskProfiling.risk_profile_penalty(miner_positions)
        self.assertEqual(len(penalties), 2, "Should have penalties for 2 miners")
        self.assertGreater(penalties["safe_miner"], penalties["risky_miner"], "Safe miner should have higher penalty multiplier")

        # Test edge case with empty miner
        empty_miner_positions = {
            "empty_miner": [],
            "risky_miner": [pos2, pos3],
        }

        scores = RiskProfiling.risk_profile_score(empty_miner_positions)
        self.assertEqual(scores["empty_miner"], 0.0, "Empty miner should have 0.0 score")

        penalties = RiskProfiling.risk_profile_penalty(empty_miner_positions)
        self.assertGreater(penalties["empty_miner"], 0.98, "Empty miner should have minimal penalty")

        # Test random positions don't trigger penalties
        # Generate 10 "random" positions with reasonable patterns
        np.random.seed(42)  # For reproducibility
        random_positions = []

        for i in range(10):
            pos = deepcopy(self.default_position)
            pos.position_uuid = f"random_{i}"

            # Reasonable number of orders (2-5)
            n_orders = np.random.randint(2, 6)

            # Order type randomly chosen
            order_type = np.random.choice([OrderType.LONG, OrderType.SHORT])
            pos.position_type = order_type

            # For short positions, make leverages negative
            leverage_sign = -1 if order_type == OrderType.SHORT else 1

            # Generate orders with random parameters
            for j in range(n_orders):
                # Generate reasonable leverage (0.01-0.2)
                leverage = leverage_sign * np.random.uniform(0.01, 0.2)

                # Generate reasonable price with some trend and noise
                base_price = 100
                trend = np.random.uniform(-0.2, 0.2)  # -20% to +20% trend
                price = base_price * (1 + trend * j/n_orders + np.random.uniform(-0.05, 0.05))

                # Generate evenly spaced timestamp with some noise
                timestamp = self.DEFAULT_ORDER_MS + (j * 1000 * 60 * 60 * 24) + np.random.randint(-1000, 1000)

                order = copy.deepcopy(self.default_order)
                order.order_type = order_type
                order.leverage = leverage
                order.price = price
                order.processed_ms = timestamp
                order.order_uuid = f"{pos.position_uuid}_{j}"
                pos.add_order(order, self.live_price_fetcher)

            # Set a reasonable return
            pos.return_at_close = 1.0 + np.random.uniform(-0.1, 0.2)  # -10% to +20%

            random_positions.append(pos)

        # Check if any random positions are flagged as risky
        risky_count = 0
        for pos in random_positions:
            if RiskProfiling.risk_profile_full_criteria(pos):
                risky_count += 1

        # We expect most randomly generated positions to not be risky
        self.assertLess(risky_count, 3, "Less than 30% of random positions should be flagged as risky")

        # Restore original values
        ValiConfig.RISK_PROFILING_STEPS_CRITERIA = original_steps
        ValiConfig.RISK_PROFILING_MONOTONIC_CRITERIA = original_monotonic
        ValiConfig.RISK_PROFILING_MARGIN_CRITERIA = original_margin
        ValiConfig.RISK_PROFILING_LEVERAGE_ADVANCE = original_leverage
        ValiConfig.RISK_PROFILING_TIME_CRITERIA = original_time


if __name__ == "__main__":
    unittest.main()
