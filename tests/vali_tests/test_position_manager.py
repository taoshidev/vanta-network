# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import random
from copy import deepcopy

from shared_objects.mock_metagraph import MockMetagraph
from tests.shared_objects.mock_classes import MockLivePriceFetcher
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.exceptions.vali_records_misalignment_exception import (
    ValiRecordsMisalignmentException,
)
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair


class TestPositionManager(TestBase):
    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True, live_price_fetcher=self.live_price_fetcher)
        self.position_manager.clear_all_miner_positions()

    def _find_disk_position_from_memory_position(self, position):
        for disk_position in self.position_manager.get_positions_for_one_hotkey(position.miner_hotkey):
            if disk_position.position_uuid == position.position_uuid:
                return disk_position
        raise ValueError(f"Could not find position {position.position_uuid} in disk")

    def validate_positions(self, in_memory_position, expected_state):
        disk_position = self._find_disk_position_from_memory_position(in_memory_position)
        success, reason = PositionManager.positions_are_the_same(in_memory_position, expected_state)
        self.assertTrue(success, "In memory position is not as expected. " + reason)
        success, reason = PositionManager.positions_are_the_same(disk_position, expected_state)
        self.assertTrue(success, "Disc position is not as expected. " + reason)

    def test_creating_closing_and_fetching_multiple_positions(self):
        n_trade_pairs = len(TradePair)
        idx_to_position = {}
        # Create 6 orders per trade pair
        for i in range(n_trade_pairs):
            trade_pair = list(TradePair)[i]
            # Create 5 closed positions
            for j in range(5):
                position = deepcopy(self.default_position)
                position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_{j}"
                position.open_ms = self.DEFAULT_OPEN_MS + 100 * i + j
                position.trade_pair = trade_pair
                position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                position.close_out_position(position.open_ms + 1)
                idx_to_position[(i, j)] = position
                self.position_manager.save_miner_position(position)
            # Create 1 open position
            j = 5
            position = deepcopy(self.default_position)
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_{j}"
            position.open_ms = self.DEFAULT_OPEN_MS + 100 * i + j
            position.trade_pair = trade_pair
            position.rebuild_position_with_updated_orders(self.live_price_fetcher)
            idx_to_position[(i, j)] = position
            self.position_manager.save_miner_position(position)

        all_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY, sort_positions=True)
        self.assertEqual(len(all_disk_positions), n_trade_pairs * 6)
        # Ensure the positions in all_disk_positions are sorted by close_ms.
        t0 = all_disk_positions[0].close_ms
        for i in range(n_trade_pairs):
            for j in range(6):
                n = i * 6 + j
                t1 = all_disk_positions[n].close_ms or float('inf')
                # Ensure the timestamp is increasing.
                self.assertTrue(t0 <= t1, 'timestamps not increasing or a valid timestamp came after a None timestamp')
                t0 = t1


        # Fetch all positions and verify that they are the same as the ones we created
        for i in range(n_trade_pairs):
            for j in range(6):
                expected_position = idx_to_position[(i, j)]
                disk_position = self._find_disk_position_from_memory_position(expected_position)
                self.validate_positions(expected_position, disk_position)

    def test_sorting_and_fetching_positions_with_several_open_positions_for_the_same_trade_pair(self):
        num_positions = 100
        open_time_start = 1000
        open_time_end = 2000
        positions = []

        # Generate and save positions
        for i in range(num_positions):
            open_ms = random.randint(open_time_start, open_time_end)
            close_ms = open_ms + random.randint(1, 1000)
            position = deepcopy(self.default_position)
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}"
            position.open_ms = open_ms
            position.close_out_position(close_ms)
            self.position_manager.save_miner_position(position)
            positions.append(position)

        # Add two open positions
        for i in range(2):
            position = deepcopy(self.default_position)
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_open_{i}"
            position.open_ms = random.randint(open_time_start, open_time_end)
            if i == 1:
                with self.assertRaises(ValiRecordsMisalignmentException):
                    self.position_manager.save_miner_position(position)
            else:
                self.position_manager.save_miner_position(position)

        all_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(all_disk_positions), num_positions + 1)

        open_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY, only_open_positions=True)
        self.assertEqual(len(open_disk_positions), 1)


    def test_sorting_and_fetching_positions_with_random_close_times_all_closed_positions(self):
            num_positions = 100
            open_time_start = 1000
            open_time_end = 2000
            positions = []

            # Generate and save positions
            for i in range(num_positions):
                open_ms = random.randint(open_time_start, open_time_end)
                close_ms = open_ms + random.randint(1, 1000)
                position = deepcopy(self.default_position)
                # Get a random trade pair
                position.trade_pair = random.choice(list(TradePair))
                position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}"
                position.open_ms = open_ms
                if close_ms:
                    position.close_out_position(close_ms)
                self.position_manager.save_miner_position(position)
                positions.append(position)

            # Fetch and sort positions from disk
            all_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                    sort_positions=True)

            # Verify the number of positions fetched matches expectations
            self.assertEqual(len(all_disk_positions), num_positions)

            # Verify that positions are sorted correctly by close_ms, treating None as infinity
            for i in range(1, num_positions):
                prev_close_ms = all_disk_positions[i - 1].close_ms or float('inf')
                curr_close_ms = all_disk_positions[i].close_ms or float('inf')
                self.assertTrue(prev_close_ms <= curr_close_ms, "Positions are not sorted correctly by close_ms")


            # Ensure no open positions are fetched
            all_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                    sort_positions=True, only_open_positions=True)
            self.assertEqual(len(all_disk_positions), 0)

    def test_one_close_and_one_open_order_per_position(self):
        open_time_start = 1000
        open_time_end = 2000
        positions = []

        # Generate and save positions
        for i in range(len(TradePair)):
            open_ms = random.randint(open_time_start, open_time_end)
            close_ms = open_ms + random.randint(1, 1000)
            position = deepcopy(self.default_position)
            # Get a random trade pair
            position.trade_pair = list(TradePair)[i]
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_closed"
            position.open_ms = open_ms
            if close_ms:
                position.close_out_position(close_ms)
            self.position_manager.save_miner_position(position)
            positions.append(position)

        for i in range(len(TradePair)):
            open_ms = random.randint(open_time_start, open_time_end)
            close_ms = None
            position = deepcopy(self.default_position)
            # Get a random trade pair
            position.trade_pair = list(TradePair)[i]
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_open"
            position.open_ms = open_ms

            self.position_manager.save_miner_position(position)
            positions.append(position)

        # Fetch and sort positions from disk
        all_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                sort_positions=True)

        # Verify the number of positions fetched matches expectations
        self.assertEqual(len(all_disk_positions), 2 * len(TradePair))

        # Verify that positions are sorted correctly by close_ms, treating None as infinity
        for i in range(1, 2 * len(TradePair)):
            prev_close_ms = all_disk_positions[i - 1].close_ms or float('inf')
            curr_close_ms = all_disk_positions[i].close_ms or float('inf')
            self.assertTrue(prev_close_ms <= curr_close_ms, "Positions are not sorted correctly by close_ms")

        # Ensure all open positions are fetched
        open_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                 sort_positions=True,
                                                                                 only_open_positions=True)
        self.assertEqual(len(open_disk_positions), len(TradePair))

        all_disk_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                sort_positions=True)
        self.assertEqual(len(all_disk_positions), 2 * len(TradePair))

    def test_compute_realtime_drawdown_with_various_drawdowns(self):
        """Test compute_realtime_drawdown across range of drawdown percentages"""
        from unittest.mock import MagicMock

        max_portfolio_value = 2.0

        # Test cases: (current_value, expected_drawdown_ratio, description)
        test_cases = [
            (2.0, 1.0, "0% drawdown"),
            (1.98, 0.99, "1% drawdown"),
            (1.96, 0.98, "2% drawdown"),
            (1.90, 0.95, "5% drawdown"),
            (1.80, 0.90, "10% drawdown"),
            (1.70, 0.85, "15% drawdown"),
            (1.50, 0.75, "25% drawdown"),
        ]

        for current_value, expected_ratio, description in test_cases:
            with self.subTest(description=description):
                # Mock perf ledger
                mock_ledger = MagicMock()
                mock_ledger.cps = [MagicMock()]
                mock_ledger.max_return = max_portfolio_value
                mock_ledger.init_max_portfolio_value = MagicMock()

                mock_bundle = {'portfolio': mock_ledger}

                # Mock perf_ledger_manager
                mock_perf_ledger_manager = MagicMock()
                mock_perf_ledger_manager.get_perf_ledgers = MagicMock(
                    return_value={self.DEFAULT_MINER_HOTKEY: mock_bundle}
                )
                self.position_manager.perf_ledger_manager = mock_perf_ledger_manager

                # Mock current portfolio value
                self.position_manager._calculate_current_portfolio_value = MagicMock(return_value=current_value)

                # Compute drawdown
                drawdown = self.position_manager.compute_realtime_drawdown(self.DEFAULT_MINER_HOTKEY)

                # Assert
                self.assertAlmostEqual(drawdown, expected_ratio, places=4,
                                      msg=f"{description}: Expected {expected_ratio}, got {drawdown}")

    """
    def test_retroactive_eliminations(self):
        position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=False, perform_price_adjustment=True)

        hotkey_positions_with_filter = position_manager.get_all_disk_positions_for_all_miners(
            sort_positions=True,
            only_open_positions=False,
        )
        n_positions_total_with_filter = 0
        for hotkey, positions in hotkey_positions_with_filter.items():
            n_positions_total_with_filter += len(positions)
    """


if __name__ == '__main__':
    import unittest

    unittest.main()
