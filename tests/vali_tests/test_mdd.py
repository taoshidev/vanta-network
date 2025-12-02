# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
MDD (Maximum Drawdown) Checker tests using modern RPC infrastructure.

Tests MDDCheckerServer functionality with proper server/client setup.
"""
from unittest.mock import patch

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position

from vali_objects.utils.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestMDDChecker(TestBase):
    """
    Integration tests for MDD Checker using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    position_client = None
    elimination_client = None
    metagraph_client = None
    mdd_checker_client = None

    # Mock patch for price data
    data_patch = None
    mock_fetch_prices = None

    MINER_HOTKEY = "test_miner"

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator and set up price data mock."""
        # Set up price data mock (kept from original test)
        cls.data_patch = patch('vali_objects.utils.live_price_fetcher.LivePriceFetcher.get_tp_to_sorted_price_sources')
        cls.mock_fetch_prices = cls.data_patch.start()
        cls.mock_fetch_prices.return_value = {
            TradePair.BTCUSD: [
                PriceSource(source='Tiingo_rest', timespan_ms=60000, open=64751.73, close=64771.04, vwap=None,
                           high=64813.66, low=64749.99, start_ms=1721937480000, websocket=False, lag_ms=29041),
                PriceSource(source='Tiingo_ws', timespan_ms=0, open=64681.6, close=64681.6, vwap=None,
                           high=64681.6, low=64681.6, start_ms=1721937625000, websocket=True, lag_ms=174041),
                PriceSource(source='Polygon_ws', timespan_ms=0, open=64693.52, close=64693.52, vwap=64693.7546,
                           high=64696.22, low=64693.52, start_ms=1721937626000, websocket=True, lag_ms=175041),
                PriceSource(source='Polygon_rest', timespan_ms=1000, open=64695.87, close=64681.9, vwap=64682.2898,
                           high=64695.87, low=64681.9, start_ms=1721937628000, websocket=False, lag_ms=177041)
            ],
            TradePair.ETHUSD: [
                PriceSource(source='Polygon_ws', timespan_ms=0, open=3267.8, close=3267.8, vwap=3267.8, high=3267.8,
                           low=3267.8, start_ms=1722390426999, websocket=True, lag_ms=2470),
                PriceSource(source='Polygon_rest', timespan_ms=1000, open=3267.8, close=3267.8, vwap=3267.8,
                           high=3267.8, low=3267.8, start_ms=1722390426000, websocket=False, lag_ms=2470),
                PriceSource(source='Tiingo_ws', timespan_ms=0, open=3267.9, close=3267.9, vwap=None, high=3267.9,
                           low=3267.9, start_ms=1722390422000, websocket=True, lag_ms=7469),
                PriceSource(source='Tiingo_rest', timespan_ms=60000, open=3271.26001, close=3268.6001, vwap=None,
                           high=3271.26001, low=3268.1001, start_ms=1722389640000, websocket=False, lag_ms=729470)
            ],
        }

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
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.mdd_checker_client = cls.orchestrator.get_client('mdd_checker')

        # Initialize metagraph with test hotkey
        cls.metagraph_client.set_hotkeys([cls.MINER_HOTKEY])

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: Clean up mocks only.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        cls.data_patch.stop()

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        self.MINER_HOTKEY = "test_miner"

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Create fresh test data
        self._create_test_data()

        # Reset MDD checker state via client
        self.mdd_checker_client.reset_debug_counters()
        self.mdd_checker_client.price_correction_enabled = False  # Disabled by default, enable per test

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        self.trade_pair_to_default_position = {x: Position(
            miner_hotkey=self.MINER_HOTKEY,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + str(x.trade_pair_id),
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=x,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        ) for x in TradePair}


        # Create default positions for all trade pairs
        self.trade_pair_to_default_position = {
            x: Position(
                miner_hotkey=self.MINER_HOTKEY,
                position_uuid=self.DEFAULT_TEST_POSITION_UUID + str(x.trade_pair_id),
                open_ms=self.DEFAULT_OPEN_MS,
                trade_pair=x,
            ) for x in TradePair
        }

    def verify_elimination_data_in_memory_and_disk(self, expected_eliminations):
        """Verify elimination data matches expectations."""
        expected_eliminated_hotkeys = [x['hotkey'] for x in expected_eliminations]
        eliminated_hotkeys = [x['hotkey'] for x in self.elimination_client.get_eliminations_from_memory()]

        self.assertEqual(
            len(eliminated_hotkeys),
            len(expected_eliminated_hotkeys),
            f"Eliminated hotkeys in memory/disk do not match expected. "
            f"eliminated_hotkeys: {eliminated_hotkeys} "
            f"expected_eliminated_hotkeys: {expected_eliminated_hotkeys}"
        )
        self.assertEqual(set(eliminated_hotkeys), set(expected_eliminated_hotkeys))

        for v1, v2 in zip(expected_eliminations, self.elimination_client.get_eliminations_from_memory()):
            self.assertEqual(v1['hotkey'], v2['hotkey'])
            self.assertEqual(v1['reason'], v2['reason'])
            self.assertAlmostEqual(
                v1['elimination_initiated_time_ms'] / 1000.0,
                v2['elimination_initiated_time_ms'] / 1000.0,
                places=1
            )
            self.assertAlmostEqual(v1['dd'], v2['dd'], places=2)

    def verify_positions_on_disk(self, in_memory_positions, assert_all_closed=None, assert_all_open=None,
                                 verify_positions_same=True, assert_price_changes=False):
        """Verify positions on disk match in-memory positions."""
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(
            len(positions_from_disk),
            len(in_memory_positions),
            f"Mismatched number of positions. Positions on disk: {positions_from_disk} "
            f"Positions in memory: {in_memory_positions}"
        )

        for position in in_memory_positions:
            matching_disk_position = next(
                (x for x in positions_from_disk if x.position_uuid == position.position_uuid),
                None
            )
            self.assertIsNotNone(matching_disk_position)
            # Use static method for comparison
            if verify_positions_same:
                success, reason = PositionManagerClient.positions_are_the_same(position, matching_disk_position)
                self.assertTrue(success, reason)

            if assert_price_changes:
                self.assertNotEqual(
                    position.orders[-1].price,
                    matching_disk_position.orders[-1].price,
                    f"Expected price change for position {position.position_uuid} "
                    f"but found same price on disk."
                )
                self.assertNotEqual(position.average_entry_price, matching_disk_position.average_entry_price)

            if assert_all_closed:
                self.assertTrue(
                    matching_disk_position.is_closed_position,
                    f"Position in memory: {position} Position on disk: {matching_disk_position}"
                )
            if assert_all_open:
                self.assertFalse(matching_disk_position.is_closed_position)

    def add_order_to_position_and_save_to_disk(self, position, order):
        """Add order to position and save to disk."""
        position.add_order(order, self.live_price_fetcher_client)
        self.position_client.save_miner_position(position)

    def test_get_live_prices(self):
        """Test that live price fetching works with mocked data."""
        live_price, price_sources = self.live_price_fetcher_client.get_latest_price(
            trade_pair=TradePair.BTCUSD,
            time_ms=TimeUtil.now_in_millis() - 1000 * 180
        )
        self.assertTrue(live_price > 0)
        self.assertTrue(price_sources)
        self.assertTrue(all([x.close > 0 for x in price_sources]))

    def test_mdd_price_correction(self):
        self.mdd_checker_client.price_correction_enabled = True
        self.verify_elimination_data_in_memory_and_disk([])
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=TimeUtil.now_in_millis(),
                order_uuid="1000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker_client.last_price_fetch_time_ms = TimeUtil.now_in_millis() - 1000 * 30
        self.mdd_checker_client.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.assertFalse(relevant_position.is_closed_position)
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)
        self.mdd_checker_client.last_price_fetch_time_ms = TimeUtil.now_in_millis() - 1000 * 30
        self.mdd_checker_client.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True, verify_positions_same=False, assert_price_changes=True)

    def test_no_mdd_failures(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_price, _ = self.live_price_fetcher_client.get_latest_price(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        o2 = Order(order_type=OrderType.LONG,
                leverage=.5,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")

        self.mdd_checker_client.last_price_fetch_time_ms = TimeUtil.now_in_millis()

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker_client.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.mdd_checker_client.mdd_check()
        self.assertEqual(relevant_position.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True, verify_positions_same=False, assert_price_changes=True)

        self.add_order_to_position_and_save_to_disk(relevant_position, o2)
        self.assertEqual(relevant_position.is_closed_position, False)
        self.mdd_checker_client.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)

    def test_no_mdd_failures_high_leverage_one_order(self):
        """Test that high leverage positions with small losses don't trigger MDD."""
        self.verify_elimination_data_in_memory_and_disk([])
        position_btc = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_btc_price, _ = self.live_price_fetcher_client.get_latest_price(trade_pair=TradePair.BTCUSD)

        o1 = Order(
            order_type=OrderType.LONG,
            leverage=20.0,
            price=live_btc_price * 1.001,  # Down 0.1%
            trade_pair=TradePair.BTCUSD,
            processed_ms=1000,
            order_uuid="1000"
        )

        self.mdd_checker_client.last_price_fetch_time_ms = TimeUtil.now_in_millis()

        # Running mdd_check with no positions should not cause any eliminations
        self.mdd_checker_client.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(position_btc, o1)
        self.mdd_checker_client.mdd_check()
        self.assertEqual(position_btc.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])

        # Reload position from disk after mdd_check (prices may have been corrected)
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), 1)
        btc_position_from_disk = positions_from_disk[0]
        self.assertFalse(btc_position_from_disk.is_closed_position)
        self.assertIsNotNone(btc_position_from_disk)

        # Add ETH position
        position_eth = self.trade_pair_to_default_position[TradePair.ETHUSD]
        live_eth_price, _ = self.live_price_fetcher_client.get_latest_price(trade_pair=TradePair.ETHUSD)

        o2 = Order(
            order_type=OrderType.LONG,
            leverage=20.0,
            price=live_eth_price * 1.001,  # Down 0.1%
            trade_pair=TradePair.ETHUSD,
            processed_ms=2000,
            order_uuid="2000"
        )

        self.add_order_to_position_and_save_to_disk(position_eth, o2)
        self.mdd_checker_client.mdd_check()
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), 2)


if __name__ == '__main__':
    import unittest
    unittest.main()
