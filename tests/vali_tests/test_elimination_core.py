# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Consolidated core elimination tests combining basic and comprehensive elimination manager functionality.
Tests all elimination types, persistence, and core operations.
"""
import os
from unittest.mock import patch, MagicMock

from tests.shared_objects.mock_classes import MockPositionManager
from shared_objects.mock_metagraph import MockMetagraph
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager, EliminationReason
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils

class TestEliminationCore(TestBase):
    """Core elimination manager functionality combining basic and comprehensive tests"""
    
    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )


        # Create diverse set of test miners
        self.MDD_MINER = "miner_mdd"
        self.REGULAR_MINER = "miner_regular"
        self.ZOMBIE_MINER = "miner_zombie"
        self.PLAGIARIST_MINER = "miner_plagiarist"
        self.CHALLENGE_FAIL_MINER = "miner_challenge_fail"
        self.LIQUIDATED_MINER = "miner_liquidated"
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        
        # Initialize system components with all miners
        self.all_miners = [
            self.MDD_MINER, 
            self.REGULAR_MINER, 
            self.ZOMBIE_MINER,
            self.PLAGIARIST_MINER,
            self.CHALLENGE_FAIL_MINER,
            self.LIQUIDATED_MINER
        ]
        self.mock_metagraph = MockMetagraph(self.all_miners)
        
        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        
        # Create perf ledger manager
        self.ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)

        # Create elimination manager
        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.elimination_manager = EliminationManager(
            self.mock_metagraph, 
            self.live_price_fetcher,
            None,  # challengeperiod_manager set later
            running_unit_tests=True,
            contract_manager=self.contract_manager
        )
        
        # Create position manager
        self.position_manager = MockPositionManager(
            self.mock_metagraph,
            perf_ledger_manager=self.ledger_manager,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Set up circular references
        self.elimination_manager.position_manager = self.position_manager
        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.plagiarism_manager = PlagiarismManager(slack_notifier=None, running_unit_tests=True)

        # Create challenge period manager
        self.challengeperiod_manager = ChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.ledger_manager,
            plagiarism_manager=self.plagiarism_manager,
            running_unit_tests=True
        )
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        
        # Create position locks
        self.position_locks = PositionLocks()
        
        # Clear all previous data
        self.position_manager.clear_all_miner_positions()
        self.elimination_manager.clear_eliminations()
        self.ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        
        # Set up initial positions for all miners
        self._setup_initial_positions()
        
        # Set up challenge period status
        self._setup_challenge_period_status()
        
        # Set up performance ledgers
        self._setup_perf_ledgers()

    def tearDown(self):
        super().tearDown()
        # Cleanup
        self.position_manager.clear_all_miner_positions()
        self.ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

    def _setup_initial_positions(self):
        """Create initial positions for all miners"""
        base_time = TimeUtil.now_in_millis() - MS_IN_8_HOURS * 10
        
        for miner in self.all_miners:
            position = Position(
                miner_hotkey=miner,
                position_uuid=f"{miner}_position",
                open_ms=base_time,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[Order(
                    price=60000,
                    processed_ms=base_time,
                    order_uuid=f"order_{miner}",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=0.5
                )]
            )
            self.position_manager.save_miner_position(position)

    def _setup_challenge_period_status(self):
        """Set up challenge period status for miners"""
        # Most miners in main competition
        for miner in [self.MDD_MINER, self.REGULAR_MINER, self.ZOMBIE_MINER, 
                      self.PLAGIARIST_MINER, self.LIQUIDATED_MINER]:
            self.challengeperiod_manager.active_miners[miner] = (MinerBucket.MAINCOMP, 0, None, None)
        
        # Challenge fail miner in challenge period
        self.challengeperiod_manager.active_miners[self.CHALLENGE_FAIL_MINER] = (
            MinerBucket.CHALLENGE,
            TimeUtil.now_in_millis() - (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS * 24 * 60 * 60 * 1000) - MS_IN_8_HOURS,
            None,
            None
        )

    def _setup_perf_ledgers(self):
        """Set up performance ledgers for testing"""
        ledgers = {}
        
        # MDD miner - will be eliminated
        ledgers[self.MDD_MINER] = generate_losing_ledger(
            0, 
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )
        
        # Regular miners - good performance
        for miner in [self.REGULAR_MINER, self.ZOMBIE_MINER, 
                      self.PLAGIARIST_MINER, self.LIQUIDATED_MINER]:
            ledgers[miner] = generate_winning_ledger(
                0,
                ValiConfig.TARGET_LEDGER_WINDOW_MS
            )
        
        # Challenge fail miner - poor performance
        ledgers[self.CHALLENGE_FAIL_MINER] = generate_losing_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )
        
        self.ledger_manager.save_perf_ledgers(ledgers)

    # ========== Basic Elimination Tests (from test_elimination_manager.py) ==========
    
    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_basic_mdd_elimination(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test basic MDD elimination functionality"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Initially no eliminations
        self.assertEqual(len(self.challengeperiod_manager.get_success_miners()), 5)
        
        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Check MDD miner was eliminated
        eliminations = self.elimination_manager.get_eliminations_from_disk()
        self.assertEqual(len(eliminations), 1)
        self.assertEqual(eliminations[0]["hotkey"], self.MDD_MINER)
        self.assertEqual(eliminations[0]["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_zombie_elimination_basic(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test basic zombie elimination when miner leaves metagraph"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Process initial eliminations
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Remove all miners from metagraph
        self.mock_metagraph.hotkeys = []
        
        # Process eliminations again
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Check all miners are now eliminated
        eliminations = self.elimination_manager.get_eliminations_from_disk()
        eliminated_hotkeys = [e["hotkey"] for e in eliminations]
        
        for miner in self.all_miners:
            self.assertIn(miner, eliminated_hotkeys)
        
        # Verify reasons
        for elimination in eliminations:
            if elimination["hotkey"] == self.MDD_MINER:
                # MDD miner keeps original reason
                self.assertEqual(elimination["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)
            else:
                # Others become zombies
                self.assertEqual(elimination["reason"], EliminationReason.ZOMBIE.value)

    # ========== Comprehensive Elimination Tests (from test_elimination_manager_comprehensive.py) ==========
    
    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_mdd_elimination_comprehensive(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test comprehensive MDD elimination with position closure"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Process MDD eliminations
        self.elimination_manager.handle_mdd_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Verify elimination
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        mdd_elim = next((e for e in eliminations if e["hotkey"] == self.MDD_MINER), None)
        self.assertIsNotNone(mdd_elim)
        self.assertEqual(mdd_elim["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)
        self.assertIn("dd", mdd_elim)
        self.assertIn("elimination_initiated_time_ms", mdd_elim)
        
        # Verify positions were closed
        positions = self.position_manager.get_positions_for_one_hotkey(self.MDD_MINER)
        for pos in positions:
            self.assertTrue(pos.is_closed_position)
            self.assertEqual(pos.orders[-1].order_type, OrderType.FLAT)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_challenge_period_elimination(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test elimination for miners failing challenge period"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Set up challenge period failure
        self.challengeperiod_manager.eliminations_with_reasons = {
            self.CHALLENGE_FAIL_MINER: (
                EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
                0.08
            )
        }
        
        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Verify elimination
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        challenge_elim = next((e for e in eliminations if e["hotkey"] == self.CHALLENGE_FAIL_MINER), None)
        self.assertIsNotNone(challenge_elim)
        self.assertEqual(challenge_elim["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value)
        self.assertEqual(challenge_elim["dd"], 0.08)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_perf_ledger_elimination(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test elimination triggered by perf ledger manager"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Create a perf ledger elimination
        pl_elimination = {
            'hotkey': self.LIQUIDATED_MINER,
            'reason': EliminationReason.LIQUIDATED.value,
            'dd': 0.15,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {
                str(TradePair.BTCUSD): 55000,
                str(TradePair.ETHUSD): 2800
            }
        }
        
        # Add to perf ledger eliminations
        self.ledger_manager.pl_elimination_rows.append(pl_elimination)
        
        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Check that liquidated miner was eliminated
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        liquidated_elim = next((e for e in eliminations if e["hotkey"] == self.LIQUIDATED_MINER), None)
        self.assertIsNotNone(liquidated_elim)
        self.assertEqual(liquidated_elim["reason"], EliminationReason.LIQUIDATED.value)
        
        # Verify positions were closed for elimination
        positions = self.position_manager.get_positions_for_one_hotkey(self.LIQUIDATED_MINER)
        for pos in positions:
            self.assertTrue(pos.is_closed_position)
            # Verify flat order was added
            self.assertEqual(pos.orders[-1].order_type, OrderType.FLAT)

    def test_elimination_persistence(self):
        """Test that eliminations are persisted to disk correctly"""
        # Create eliminations
        test_elimination = {
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        }
        
        self.elimination_manager.eliminations.append(test_elimination)
        
        # Force write to disk
        self.elimination_manager.write_eliminations_to_disk(self.elimination_manager.eliminations)
        
        # Clear memory and reload
        self.elimination_manager.eliminations = []
        loaded_eliminations = self.elimination_manager.get_eliminations_from_disk()
        
        # Verify persistence
        self.assertEqual(len(loaded_eliminations), 1)
        self.assertEqual(loaded_eliminations[0]['hotkey'], self.MDD_MINER)
        self.assertEqual(loaded_eliminations[0]['reason'], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

    def test_elimination_row_generation(self):
        """Test elimination row data structure generation"""
        test_dd = 0.15
        test_reason = EliminationReason.MAX_TOTAL_DRAWDOWN.value
        test_time = TimeUtil.now_in_millis()
        
        row = self.elimination_manager.generate_elimination_row(
            self.MDD_MINER,
            test_dd,
            test_reason,
            t_ms=test_time
        )
        
        # Verify structure
        self.assertEqual(row['hotkey'], self.MDD_MINER)
        self.assertEqual(row['dd'], test_dd)
        self.assertEqual(row['reason'], test_reason)
        self.assertEqual(row['elimination_initiated_time_ms'], test_time)

    def test_elimination_sync(self):
        """Test elimination synchronization between validators"""
        # Create test elimination
        test_elim = {
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.15,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        }
        
        # Simulate receiving elimination from another validator
        self.elimination_manager.sync_eliminations([test_elim])
        
        # Verify it was added
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        self.assertEqual(len(eliminations), 1)
        self.assertEqual(eliminations[0]['hotkey'], self.MDD_MINER)

    def test_is_zombie_hotkey(self):
        """Test zombie hotkey detection"""
        # Get all hotkeys set
        all_hotkeys_set = set(self.mock_metagraph.hotkeys)
        
        # Initially not zombie
        self.assertFalse(self.elimination_manager.is_zombie_hotkey(self.ZOMBIE_MINER, all_hotkeys_set))
        
        # Remove from metagraph and update set
        self.mock_metagraph.hotkeys = [hk for hk in self.mock_metagraph.hotkeys if hk != self.ZOMBIE_MINER]
        all_hotkeys_set = set(self.mock_metagraph.hotkeys)
        
        # Now should be zombie
        self.assertTrue(self.elimination_manager.is_zombie_hotkey(self.ZOMBIE_MINER, all_hotkeys_set))

    def test_hotkey_in_eliminations(self):
        """Test checking if hotkey is in eliminations"""
        # Add elimination
        self.elimination_manager.eliminations.append({
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        })
        
        # Test existing elimination
        result = self.elimination_manager.hotkey_in_eliminations(self.MDD_MINER)
        self.assertIsNotNone(result)
        self.assertEqual(result['reason'], EliminationReason.MAX_TOTAL_DRAWDOWN.value)
        
        # Test non-existing elimination
        result = self.elimination_manager.hotkey_in_eliminations('non_existent')
        self.assertIsNone(result)

    def test_elimination_cache_controller_functionality(self):
        """Test that elimination manager properly inherits from CacheController"""
        # Test that cache controller methods are available
        # First call refresh_allowed to initialize attempted_start_time_ms
        result = self.elimination_manager.refresh_allowed(100)
        # In unit tests, refresh_allowed always returns True
        self.assertTrue(result)
        
        # Test set_last_update_time doesn't raise errors
        self.elimination_manager.set_last_update_time(skip_message=True)
        
        # Test get_last_update_time_ms
        last_update = self.elimination_manager.get_last_update_time_ms()
        self.assertIsInstance(last_update, int)
        self.assertGreater(last_update, 0)

    def test_elimination_with_ipc_manager(self):
        """Test elimination manager with IPC manager for multiprocessing"""
        # Mock IPC manager
        mock_ipc_manager = MagicMock()
        mock_ipc_manager.list.return_value = []
        mock_ipc_manager.dict.return_value = {}
        
        # Create elimination manager with IPC
        ipc_elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            ipc_manager=mock_ipc_manager
        )
        
        # Verify IPC list was created
        mock_ipc_manager.list.assert_called()
        
        # Test adding elimination
        test_elim = ipc_elimination_manager.generate_elimination_row(
            self.MDD_MINER,
            0.12,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )
        ipc_elimination_manager.eliminations.append(test_elim)
        
        # Verify it works with IPC manager
        self.assertEqual(len(ipc_elimination_manager.eliminations), 1)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multiple_eliminations_same_miner(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test that a miner can only be eliminated once"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # First elimination
        self.elimination_manager.eliminations.append({
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        })
        
        # Try to add another elimination for same miner
        # Process eliminations should not duplicate
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Should still have only one elimination for this miner
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        mdd_eliminations = [e for e in eliminations if e['hotkey'] == self.MDD_MINER]
        self.assertEqual(len(mdd_eliminations), 1)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_elimination_deletion_after_timeout(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test that old eliminations are cleaned up after timeout"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Create an old elimination
        old_time = TimeUtil.now_in_millis() - ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS - MS_IN_8_HOURS
        
        old_elim = self.elimination_manager.generate_elimination_row(
            'old_miner',
            0.15,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            t_ms=old_time
        )
        self.elimination_manager.eliminations.append(old_elim)
        
        # Remove from metagraph
        self.mock_metagraph.hotkeys = [hk for hk in self.mock_metagraph.hotkeys if hk != 'old_miner']
        
        # Create miner directory
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=True) + 'old_miner'
        os.makedirs(miner_dir, exist_ok=True)
        
        # Process eliminations (should clean up)
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Verify cleanup
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        old_miner_elim = next((e for e in eliminations if e['hotkey'] == 'old_miner'), None)
        self.assertIsNone(old_miner_elim)
        self.assertFalse(os.path.exists(miner_dir))

    def test_elimination_with_no_positions(self):
        """Test elimination handling when miner has no positions"""
        # Clear positions for MDD miner
        self.position_manager.clear_all_miner_positions(target_hotkey=self.MDD_MINER)
        
        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Should still be eliminated even without positions
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        mdd_elim = next((e for e in eliminations if e['hotkey'] == self.MDD_MINER), None)
        self.assertIsNotNone(mdd_elim)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_elimination_first_refresh_handling(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test first refresh behavior after validator start"""
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Create new elimination manager
        new_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            contract_manager=self.contract_manager
        )
        
        # First refresh should have special handling
        self.assertFalse(new_manager.first_refresh_ran)
        
        # Process eliminations
        new_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Flag should be set
        self.assertTrue(new_manager.first_refresh_ran)
