# developer: jbonilla
# Copyright © 2024 Taoshi Inc
from unittest.mock import patch

from tests.shared_objects.mock_classes import MockMDDChecker
from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestMDDChecker(TestBase):
    @classmethod
    def setUpClass(cls):
        cls.data_patch = patch('vali_objects.utils.live_price_fetcher.LivePriceFetcher.get_tp_to_sorted_price_sources')
        cls.mock_fetch_prices = cls.data_patch.start()
        cls.mock_fetch_prices.return_value = {TradePair.BTCUSD:
            [PriceSource(source='Tiingo_rest', timespan_ms=60000, open=64751.73, close=64771.04, vwap=None,
                         high=64813.66, low=64749.99, start_ms=1721937480000, websocket=False, lag_ms=29041,
                         volume=None),
             PriceSource(source='Tiingo_ws', timespan_ms=0, open=64681.6, close=64681.6, vwap=None,
                         high=64681.6, low=64681.6, start_ms=1721937625000, websocket=True, lag_ms=174041,
                         volume=None),
             PriceSource(source='Polygon_ws', timespan_ms=0, open=64693.52, close=64693.52, vwap=64693.7546,
                         high=64696.22, low=64693.52, start_ms=1721937626000, websocket=True, lag_ms=175041,
                         volume=0.00023784),
             PriceSource(source='Polygon_rest', timespan_ms=1000, open=64695.87, close=64681.9, vwap=64682.2898,
                         high=64695.87, low=64681.9, start_ms=1721937628000, websocket=False, lag_ms=177041,
                         volume=0.05812185)],
            TradePair.ETHUSD: [PriceSource(source='Polygon_ws', timespan_ms=0, open=3267.8, close=3267.8, vwap=3267.8, high=3267.8,
                             low=3267.8, start_ms=1722390426999, websocket=True, lag_ms=2470, volume=0.00697151),
                 PriceSource(source='Polygon_rest', timespan_ms=1000, open=3267.8, close=3267.8, vwap=3267.8,
                             high=3267.8, low=3267.8, start_ms=1722390426000, websocket=False, lag_ms=2470,
                             volume=0.00697151),
                 PriceSource(source='Tiingo_ws', timespan_ms=0, open=3267.9, close=3267.9, vwap=None, high=3267.9,
                             low=3267.9, start_ms=1722390422000, websocket=True, lag_ms=7469, volume=None),
                 PriceSource(source='Tiingo_rest', timespan_ms=60000, open=3271.26001, close=3268.6001, vwap=None,
                             high=3271.26001, low=3268.1001, start_ms=1722389640000, websocket=False, lag_ms=729470,
                             volume=None)],
        }
        cls.position_locks = PositionLocks()





    @classmethod
    def tearDownClass(cls):
        cls.data_patch.stop()

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.MINER_HOTKEY = "test_miner"
        self.mock_metagraph = MockMetagraph([self.MINER_HOTKEY])
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        self.perf_ledger_manager = PerfLedgerManager(metagraph=self.mock_metagraph,
                                                     live_price_fetcher=self.live_price_fetcher,
                                                     running_unit_tests=True)
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True,
                                                perf_ledger_manager=self.perf_ledger_manager, elimination_manager=self.elimination_manager)
        self.elimination_manager.position_manager = self.position_manager

        self.mdd_checker = MockMDDChecker(self.mock_metagraph, self.position_manager, self.live_price_fetcher)
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

        self.mdd_checker.elimination_manager.clear_eliminations()
        self.position_manager.clear_all_miner_positions()
        self.mdd_checker.price_correction_enabled = False

    def verify_elimination_data_in_memory_and_disk(self, expected_eliminations):
        #self.mdd_checker.elimination_manager._refresh_eliminations_in_memory_and_disk()
        expected_eliminated_hotkeys = [x['hotkey'] for x in expected_eliminations]

        eliminated_hotkeys = [x['hotkey'] for x in self.mdd_checker.elimination_manager.get_eliminations_from_memory()]
        self.assertEqual(len(eliminated_hotkeys),
                         len(expected_eliminated_hotkeys),
                         "Eliminated hotkeys in memory/disk do not match expected. eliminated_hotkeys: "
                         + str(eliminated_hotkeys) + " expected_eliminated_hotkeys: " + str(expected_eliminated_hotkeys))
        self.assertEqual(set(eliminated_hotkeys), set(expected_eliminated_hotkeys))
        for v1, v2 in zip(expected_eliminations, self.mdd_checker.elimination_manager.get_eliminations_from_memory()):
            self.assertEqual(v1['hotkey'], v2['hotkey'])
            self.assertEqual(v1['reason'], v2['reason'])
            self.assertAlmostEquals(v1['elimination_initiated_time_ms'] / 1000.0, v2['elimination_initiated_time_ms'] / 1000.0, places=1)
            self.assertAlmostEquals(v1['dd'], v2['dd'], places=2)

    def verify_positions_on_disk(self, in_memory_positions, assert_all_closed=None, assert_all_open=None):
        positions_from_disk = self.position_manager.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), len(in_memory_positions),
                         f"Mismatched number of positions. Positions on disk: {positions_from_disk}"
                         f" Positions in memory: {in_memory_positions}")
        for position in in_memory_positions:
            matching_disk_position = next((x for x in positions_from_disk if x.position_uuid == position.position_uuid), None)
            self.position_manager.positions_are_the_same(position, matching_disk_position)
            if assert_all_closed:
                self.assertTrue(matching_disk_position.is_closed_position, f"Position in memory: {position} Position on disk: {matching_disk_position}")
            if assert_all_open:
                self.assertFalse(matching_disk_position.is_closed_position)


    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

    def test_get_live_prices(self):
        live_price, price_sources = self.live_price_fetcher.get_latest_price(trade_pair=TradePair.BTCUSD, time_ms=TimeUtil.now_in_millis() - 1000 * 180)
        for i in range(len(price_sources)):
            print('%%%%', price_sources[i], '%%%%')
        self.assertTrue(live_price > 0)
        self.assertTrue(price_sources)
        self.assertTrue(all([x.close > 0 for x in price_sources]))

    def test_mdd_price_correction(self):
        self.mdd_checker.price_correction_enabled = True
        self.verify_elimination_data_in_memory_and_disk([])
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=1000,
                trade_pair=TradePair.BTCUSD,
                processed_ms=TimeUtil.now_in_millis(),
                order_uuid="1000")

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker.last_price_fetch_time_ms = TimeUtil.now_in_millis() - 1000 * 30
        self.mdd_checker.mdd_check(self.position_locks)
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.assertFalse(relevant_position.is_closed_position)
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)
        self.mdd_checker.last_price_fetch_time_ms = TimeUtil.now_in_millis() - 1000 * 30
        self.mdd_checker.mdd_check(self.position_locks)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)

    def test_no_mdd_failures(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_price, _ = self.live_price_fetcher.get_latest_price(trade_pair=TradePair.BTCUSD)
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

        self.mdd_checker.last_price_fetch_time_ms = TimeUtil.now_in_millis()

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker.mdd_check(self.position_locks)
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        self.mdd_checker.mdd_check(self.position_locks)
        self.assertEqual(relevant_position.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)

        self.add_order_to_position_and_save_to_disk(relevant_position, o2)
        self.assertEqual(relevant_position.is_closed_position, False)
        self.mdd_checker.mdd_check(self.position_locks)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([relevant_position], assert_all_open=True)


    def test_no_mdd_failures_high_leverage_one_order(self):
        self.verify_elimination_data_in_memory_and_disk([])
        position_btc = self.trade_pair_to_default_position[TradePair.BTCUSD]
        live_btc_price, _ = self.live_price_fetcher.get_latest_price(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.LONG,
                leverage=20.0,
                price=live_btc_price *1.001, # Down 0.1%
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        self.mdd_checker.last_price_fetch_time_ms = TimeUtil.now_in_millis()

        self.mdd_checker.mdd_check(self.position_locks)
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(position_btc, o1)
        self.mdd_checker.mdd_check(self.position_locks)
        self.assertEqual(position_btc.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])
        self.verify_positions_on_disk([position_btc], assert_all_open=True)

        btc_position_from_disk = self.position_manager.get_positions_for_one_hotkey(self.MINER_HOTKEY, from_disk=True)[0]
        btc_position_from_memory = self.position_manager.get_positions_for_one_hotkey(self.MINER_HOTKEY, from_disk=False)[0]
        assert self.position_manager.positions_are_the_same(btc_position_from_disk, btc_position_from_memory)
        print("Position return on BTC after mdd_check:", btc_position_from_disk.current_return)
        # print("Max MDD for closed positions:", self.mdd_checker.portfolio_max_dd_closed_positions)
        # print("Max MDD for all positions:", self.mdd_checker.portfolio_max_dd_all_positions)


        print("Adding ETH position")
        position_eth = self.trade_pair_to_default_position[TradePair.ETHUSD]
        live_eth_price, price_sources = self.live_price_fetcher.get_latest_price(trade_pair=TradePair.ETHUSD)
        o2 = Order(order_type=OrderType.LONG,
                   leverage=20.0,
                   price=live_eth_price * 1.001,  # Down 0.1%
                   trade_pair=TradePair.ETHUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        self.add_order_to_position_and_save_to_disk(position_eth, o2)
        self.mdd_checker.mdd_check(self.position_locks)
        positions_from_disk = self.position_manager.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        for p in positions_from_disk:
            print('individual position return', p.trade_pair, p.current_return)
        # print("Max MDD for closed positions:", self.mdd_checker.portfolio_max_dd_closed_positions)
        # print("Max MDD for all position:", self.mdd_checker.portfolio_max_dd_all_positions)


if __name__ == '__main__':
    import unittest
    unittest.main()
