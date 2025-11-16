# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
from unittest.mock import MagicMock, Mock, patch
from tests.vali_tests.mock_utils import (
    EnhancedMockMetagraph,
    EnhancedMockChallengePeriodManager,
    EnhancedMockPositionManager,
    EnhancedMockPerfLedgerManager,
    MockLedgerFactory,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import (
    EliminationManager,
    DEPARTED_HOTKEYS_KEY
)
from shared_objects.metagraph_utils import (
    ANOMALY_DETECTION_MIN_LOST,
    ANOMALY_DETECTION_PERCENT_THRESHOLD
)
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
import template

class TestReregistration(TestBase):
    """Integration tests for re-registration tracking and rejection"""

    def setUp(self):
        super().setUp()

        # Create test miners
        self.NORMAL_MINER = "normal_miner"
        self.DEREGISTERED_MINER = "deregistered_miner"
        self.REREGISTERED_MINER = "reregistered_miner"
        self.FUTURE_REREG_MINER = "future_rereg_miner"

        self.all_miners = [
            self.NORMAL_MINER,
            self.DEREGISTERED_MINER,
            self.REREGISTERED_MINER,
            self.FUTURE_REREG_MINER
        ]

        # Initialize components
        self.mock_metagraph = EnhancedMockMetagraph(self.all_miners)

        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)

        self.position_locks = PositionLocks()

        # Create managers
        self.perf_ledger_manager = EnhancedMockPerfLedgerManager(
            self.mock_metagraph,
            running_unit_tests=True,
            perf_ledger_hks_to_invalidate={}
        )

        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.plagiarism_manager = PlagiarismManager(slack_notifier=None, running_unit_tests=True)

        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            None,  # position_manager set later
            None,  # challengeperiod_manager set later
            running_unit_tests=True,
            use_ipc=False,
            contract_manager=self.contract_manager
        )

        self.position_manager = EnhancedMockPositionManager(
            self.mock_metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )

        self.challengeperiod_manager = EnhancedMockChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.perf_ledger_manager,
            contract_manager=self.contract_manager,
            plagiarism_manager=self.plagiarism_manager,
            running_unit_tests=True
        )

        # Set circular references (auto-synced to server via property setters)
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

        # Clear all data
        self.clear_all_data()

        # Set up initial state
        self._setup_test_environment()

    def tearDown(self):
        super().tearDown()
        self.clear_all_data()

    def clear_all_data(self):
        """Clear all test data"""
        self.position_manager.clear_all_miner_positions()
        self.perf_ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

        # Clear departed hotkeys file
        departed_file = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=True)
        if os.path.exists(departed_file):
            os.remove(departed_file)

    def _setup_test_environment(self):
        """Set up basic test environment"""
        # Create positions for all miners
        base_time = TimeUtil.now_in_millis() - MS_IN_24_HOURS * 5

        for miner in self.all_miners:
            position = Position(
                miner_hotkey=miner,
                position_uuid=f"{miner}_BTCUSD",
                open_ms=base_time,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                orders=[Order(
                    price=60000,
                    processed_ms=base_time,
                    order_uuid=f"order_{miner}_BTCUSD",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=0.5
                )]
            )
            self.position_manager.save_miner_position(position)

        # Set all miners to main competition
        for miner in self.all_miners:
            self.challengeperiod_manager.set_miner_bucket(miner, MinerBucket.MAINCOMP, 0)

        # Create basic performance ledgers
        ledgers = {}
        for miner in self.all_miners:
            ledgers[miner] = MockLedgerFactory.create_winning_ledger(final_return=1.05)
        self.perf_ledger_manager.save_perf_ledgers(ledgers)

    def _setup_polygon_mocks(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Helper to set up Polygon API mocks"""
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_departed_hotkey_tracking_on_deregistration(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test that departed hotkeys are tracked when miners leave the metagraph"""
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)

        # Initial state - no departed hotkeys
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 0)

        # Remove a miner from metagraph (simulate de-registration)
        self.mock_metagraph.remove_hotkey(self.DEREGISTERED_MINER)

        # Process eliminations to trigger departed hotkey tracking
        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify the departed hotkey was tracked
        self.assertIn(self.DEREGISTERED_MINER, self.elimination_manager.get_departed_hotkeys())
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 1)

        # Verify it was persisted to disk
        departed_file = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(departed_file))

        # Load from disk and verify
        departed_data = ValiUtils.get_vali_json_file(departed_file, DEPARTED_HOTKEYS_KEY)
        self.assertIn(self.DEREGISTERED_MINER, departed_data)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multiple_departures_tracked(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        """Test tracking multiple miners leaving the metagraph"""
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)

        # Remove multiple miners
        self.mock_metagraph.remove_hotkey(self.DEREGISTERED_MINER)
        self.mock_metagraph.remove_hotkey(self.FUTURE_REREG_MINER)

        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify both were tracked
        self.assertIn(self.DEREGISTERED_MINER, self.elimination_manager.get_departed_hotkeys())
        self.assertIn(self.FUTURE_REREG_MINER, self.elimination_manager.get_departed_hotkeys())
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 2)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_anomalous_departure_ignored(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that anomalous mass departures are ignored to avoid false positives"""
        # Create a large number of miners
        large_miner_set = [f"miner_{i}" for i in range(50)]
        self.mock_metagraph = EnhancedMockMetagraph(large_miner_set)

        # Reinitialize elimination manager with new metagraph
        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            use_ipc=False,
            contract_manager=self.contract_manager
        )

        # Process once to set previous_metagraph_hotkeys
        self.elimination_manager.process_eliminations(self.position_locks)

        # Remove 30% of miners (should trigger anomaly detection: >10 hotkeys AND >=25%)
        miners_to_remove = large_miner_set[:15]  # 15 out of 50 = 30%
        for miner in miners_to_remove:
            self.mock_metagraph.remove_hotkey(miner)

        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify departed hotkeys were NOT tracked (anomaly detected)
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 0)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_normal_departure_below_anomaly_threshold(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that normal departures below threshold are tracked"""
        # Create miners
        miner_set = [f"miner_{i}" for i in range(50)]
        self.mock_metagraph = EnhancedMockMetagraph(miner_set)

        # Reinitialize elimination manager
        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            use_ipc=False,
            contract_manager=self.contract_manager
        )

        # Process once to set baseline
        self.elimination_manager.process_eliminations(self.position_locks)

        # Remove only 5 miners (5 out of 50 = 10%, below 25% threshold)
        miners_to_remove = miner_set[:5]
        for miner in miners_to_remove:
            self.mock_metagraph.remove_hotkey(miner)

        # Process eliminations
        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify departed hotkeys WERE tracked (not anomalous)
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 5)
        for miner in miners_to_remove:
            self.assertIn(miner, self.elimination_manager.get_departed_hotkeys())

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_reregistration_detection(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test detection when a departed miner re-registers"""
        # Remove miner from metagraph
        self.mock_metagraph.remove_hotkey(self.REREGISTERED_MINER)

        # Process to track departure
        self.elimination_manager.process_eliminations(self.position_locks)
        self.assertIn(self.REREGISTERED_MINER, self.elimination_manager.get_departed_hotkeys())

        # Re-add miner to metagraph (simulate re-registration)
        self.mock_metagraph.add_hotkey(self.REREGISTERED_MINER)

        # Process eliminations again
        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify re-registration was detected (check via is_hotkey_re_registered)
        self.assertTrue(self.elimination_manager.is_hotkey_re_registered(self.REREGISTERED_MINER))

        # Verify the hotkey is still in departed list (permanent record)
        self.assertIn(self.REREGISTERED_MINER, self.elimination_manager.get_departed_hotkeys())

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_is_hotkey_re_registered_method(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test the is_hotkey_re_registered() lookup method"""
        # Normal miner - should return False
        self.assertFalse(self.elimination_manager.is_hotkey_re_registered(self.NORMAL_MINER))

        # Miner that has never been in metagraph - should return False
        self.assertFalse(self.elimination_manager.is_hotkey_re_registered("unknown_miner"))

        # Set up re-registered miner
        self.mock_metagraph.remove_hotkey(self.REREGISTERED_MINER)
        self.elimination_manager.process_eliminations(self.position_locks)

        # While departed - should return False (not currently in metagraph)
        self.assertFalse(self.elimination_manager.is_hotkey_re_registered(self.REREGISTERED_MINER))

        # Re-add to metagraph
        self.mock_metagraph.add_hotkey(self.REREGISTERED_MINER)

        # Now should return True (in metagraph AND in departed list)
        self.assertTrue(self.elimination_manager.is_hotkey_re_registered(self.REREGISTERED_MINER))

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_departed_hotkeys_persistence_across_restart(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that departed hotkeys persist across elimination manager restart"""
        # Track some departed miners
        self.mock_metagraph.remove_hotkey(self.DEREGISTERED_MINER)
        self.mock_metagraph.remove_hotkey(self.FUTURE_REREG_MINER)
        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify they were tracked
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 2)

        # Create new elimination manager (simulate restart)
        new_elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            contract_manager=self.contract_manager
        )

        # Verify departed hotkeys were loaded from disk
        self.assertEqual(len(new_elimination_manager.get_departed_hotkeys()), 2)
        self.assertIn(self.DEREGISTERED_MINER, new_elimination_manager.get_departed_hotkeys())
        self.assertIn(self.FUTURE_REREG_MINER, new_elimination_manager.get_departed_hotkeys())

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_validator_rejects_reregistered_orders(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that validator's should_fail_early rejects re-registered miners"""
        # Import validator components
        from neurons.validator import Validator

        # Create mock synapse for signal
        mock_synapse = Mock(spec=template.protocol.SendSignal)
        mock_synapse.dendrite = Mock()
        mock_synapse.dendrite.hotkey = self.REREGISTERED_MINER
        mock_synapse.miner_order_uuid = "test_uuid"
        mock_synapse.successfully_processed = True
        mock_synapse.error_message = ""

        # Create mock signal
        mock_signal = {
            "trade_pair": {
                "trade_pair_id": "BTCUSD"
            },
            "order_type": "LONG",
            "leverage": 0.5
        }

        # Set up re-registered miner
        self.mock_metagraph.remove_hotkey(self.REREGISTERED_MINER)
        self.elimination_manager.process_eliminations(self.position_locks)
        self.mock_metagraph.add_hotkey(self.REREGISTERED_MINER)

        # Verify re-registration detected
        self.assertTrue(self.elimination_manager.is_hotkey_re_registered(self.REREGISTERED_MINER))

        # Test rejection logic directly (simulating should_fail_early check)
        if self.elimination_manager.is_hotkey_re_registered(mock_synapse.dendrite.hotkey):
            mock_synapse.successfully_processed = False
            mock_synapse.error_message = (
                f"This miner hotkey {mock_synapse.dendrite.hotkey} was previously de-registered "
                f"and is not allowed to re-register. Re-registration is not permitted on this subnet."
            )

        # Verify the order was rejected
        self.assertFalse(mock_synapse.successfully_processed)
        self.assertIn("previously de-registered", mock_synapse.error_message)
        self.assertIn("not allowed to re-register", mock_synapse.error_message)

    def test_normal_miner_not_rejected(self):
        """Test that normal miners (never departed) are not rejected"""
        # Create mock synapse
        mock_synapse = Mock(spec=template.protocol.SendSignal)
        mock_synapse.dendrite = Mock()
        mock_synapse.dendrite.hotkey = self.NORMAL_MINER
        mock_synapse.successfully_processed = True
        mock_synapse.error_message = ""

        # Normal miner should not be flagged as re-registered
        self.assertFalse(self.elimination_manager.is_hotkey_re_registered(self.NORMAL_MINER))

        # Simulate the check (should pass)
        if self.elimination_manager.is_hotkey_re_registered(mock_synapse.dendrite.hotkey):
            mock_synapse.successfully_processed = False
            mock_synapse.error_message = "Should not reach here"

        # Verify order was NOT rejected
        self.assertTrue(mock_synapse.successfully_processed)
        self.assertEqual(mock_synapse.error_message, "")

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_departed_miner_not_yet_reregistered(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that departed miners (not yet re-registered) are handled correctly"""
        # Create mock synapse
        mock_synapse = Mock(spec=template.protocol.SendSignal)
        mock_synapse.dendrite = Mock()
        mock_synapse.dendrite.hotkey = self.DEREGISTERED_MINER

        # De-register the miner
        self.mock_metagraph.remove_hotkey(self.DEREGISTERED_MINER)
        self.elimination_manager.process_eliminations(self.position_locks)

        # Departed but not re-registered should return False (not in metagraph)
        self.assertFalse(self.elimination_manager.is_hotkey_re_registered(self.DEREGISTERED_MINER))

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multiple_reregistrations_tracked(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test tracking multiple re-registrations"""
        # Set up multiple re-registered miners
        miners_to_rereg = [self.REREGISTERED_MINER, self.FUTURE_REREG_MINER]

        for miner in miners_to_rereg:
            # De-register
            self.mock_metagraph.remove_hotkey(miner)

        self.elimination_manager.process_eliminations(self.position_locks)

        # Verify both tracked as departed
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 2)

        # Re-register both
        for miner in miners_to_rereg:
            self.mock_metagraph.add_hotkey(miner)

        # Both should be detected as re-registered
        for miner in miners_to_rereg:
            self.assertTrue(self.elimination_manager.is_hotkey_re_registered(miner))

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_departed_file_format(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that the departed hotkeys file has correct format"""
        # Track some departures
        self.mock_metagraph.remove_hotkey(self.DEREGISTERED_MINER)
        self.elimination_manager.process_eliminations(self.position_locks)

        # Read file directly
        departed_file = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=True)
        with open(departed_file, 'r') as f:
            import json
            data = json.load(f)

        # Verify structure - should be a dict with metadata
        self.assertIn(DEPARTED_HOTKEYS_KEY, data)
        self.assertIsInstance(data[DEPARTED_HOTKEYS_KEY], dict)
        self.assertIn(self.DEREGISTERED_MINER, data[DEPARTED_HOTKEYS_KEY])
        # Verify metadata is present
        metadata = data[DEPARTED_HOTKEYS_KEY][self.DEREGISTERED_MINER]
        self.assertIn("detected_ms", metadata)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_no_duplicate_departed_tracking(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test that the same miner isn't added to departed list multiple times"""
        # Remove miner
        self.mock_metagraph.remove_hotkey(self.DEREGISTERED_MINER)
        self.elimination_manager.process_eliminations(self.position_locks)

        # Process multiple times
        self.elimination_manager.process_eliminations(self.position_locks)
        self.elimination_manager.process_eliminations(self.position_locks)

        # Should only appear once (dict keys are unique by definition)
        self.assertIn(self.DEREGISTERED_MINER, self.elimination_manager.get_departed_hotkeys())
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 1)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_anomaly_threshold_boundary(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test anomaly detection at exact boundary conditions"""
        # Create exactly 40 miners (to test 10 miner / 25% boundary)
        miner_set = [f"miner_{i}" for i in range(40)]
        self.mock_metagraph = EnhancedMockMetagraph(miner_set)

        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            use_ipc=False,
            contract_manager=self.contract_manager
        )

        self.elimination_manager.process_eliminations(self.position_locks)

        # Remove exactly 10 miners = 25% (boundary case: should NOT trigger anomaly, needs >10)
        miners_to_remove = miner_set[:10]
        for miner in miners_to_remove:
            self.mock_metagraph.remove_hotkey(miner)

        self.elimination_manager.process_eliminations(self.position_locks)

        # At boundary (exactly 10 miners AND 25%), should NOT trigger anomaly (needs > 10)
        # So departed hotkeys should be tracked
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 10)

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_below_anomaly_threshold_boundary(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        self._setup_polygon_mocks(mock_candle_fetcher, mock_get_candles, mock_market_close)
        """Test tracking just below anomaly threshold"""
        # Create 41 miners
        miner_set = [f"miner_{i}" for i in range(41)]
        self.mock_metagraph = EnhancedMockMetagraph(miner_set)

        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True,
            use_ipc=False,
            contract_manager=self.contract_manager
        )

        self.elimination_manager.process_eliminations(self.position_locks)

        # Remove 10 miners = 24.4% (just below 25% threshold, should NOT trigger anomaly)
        miners_to_remove = miner_set[:10]
        for miner in miners_to_remove:
            self.mock_metagraph.remove_hotkey(miner)

        self.elimination_manager.process_eliminations(self.position_locks)

        # Just below threshold, should track
        self.assertEqual(len(self.elimination_manager.get_departed_hotkeys()), 10)
