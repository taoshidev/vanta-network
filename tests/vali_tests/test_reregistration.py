# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Integration tests for re-registration tracking and rejection using client/server architecture.
Tests departed hotkey tracking, re-registration detection, and anomaly protection.
"""
import os

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import generate_winning_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.utils.elimination.elimination_manager import DEPARTED_HOTKEYS_KEY
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class TestReregistration(TestBase):
    """
    Integration tests for re-registration tracking and rejection.
    Uses class-level server setup for efficiency.
    Server infrastructure starts once in setUpClass and is shared across all tests.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    challenge_period_client = None
    plagiarism_client = None
    asset_selection_client = None

    # Test miner constants
    NORMAL_MINER = "normal_miner"
    DEREGISTERED_MINER = "deregistered_miner"
    REREGISTERED_MINER = "reregistered_miner"
    FUTURE_REREG_MINER = "future_rereg_miner"

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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')

        # Define test miners and initialize metagraph
        cls.all_test_miners = [
            cls.NORMAL_MINER,
            cls.DEREGISTERED_MINER,
            cls.REREGISTERED_MINER,
            cls.FUTURE_REREG_MINER
        ]
        cls.metagraph_client.set_hotkeys(cls.all_test_miners)

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
        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Create fresh test data
        self._create_test_data()

        # Clear departed hotkeys AFTER setting test hotkeys to avoid tracking previous test's miners as departed
        self.elimination_client.clear_departed_hotkeys()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()
        self.elimination_client.clear_departed_hotkeys()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        # Define all test miners
        self.all_miners = [
            self.NORMAL_MINER,
            self.DEREGISTERED_MINER,
            self.REREGISTERED_MINER,
            self.FUTURE_REREG_MINER
        ]

        # Set up metagraph with all miner names
        self.metagraph_client.set_hotkeys(self.all_miners)

        # Set up initial positions for all miners
        self._setup_initial_positions()

        # Set up challenge period status
        self._setup_challenge_period_status()

        # Set up performance ledgers
        self._setup_perf_ledgers()

    def _setup_initial_positions(self):
        """Create initial positions for all miners"""
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
            self.position_client.save_miner_position(position)

    def _setup_challenge_period_status(self):
        """Set up challenge period status for miners"""
        # Build miners dict - all miners in main competition for reregistration tests
        miners = {}
        for miner in self.all_miners:
            miners[miner] = (MinerBucket.MAINCOMP, 0, None, None)

        # Update using client API
        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners(miners)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

    def _setup_perf_ledgers(self):
        """Set up performance ledgers for testing"""
        ledgers = {}

        # All miners have good performance for reregistration tests
        for miner in self.all_miners:
            ledgers[miner] = generate_winning_ledger(
                0,
                ValiConfig.TARGET_LEDGER_WINDOW_MS
            )

        self.perf_ledger_client.save_perf_ledgers(ledgers)
        self.perf_ledger_client.re_init_perf_ledger_data()

    # ========== Departed Hotkey Tracking Tests ==========

    def test_departed_hotkey_tracking_on_deregistration(self):
        """Test that departed hotkeys are tracked when miners leave the metagraph"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Initial state - no departed hotkeys
        self.assertEqual(len(self.elimination_client.get_departed_hotkeys()), 0)

        # Remove a miner from metagraph (simulate de-registration)
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.DEREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations to trigger departed hotkey tracking
        self.elimination_client.process_eliminations()

        # Verify the departed hotkey was tracked
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertIn(self.DEREGISTERED_MINER, departed)
        self.assertEqual(len(departed), 1)

        # Verify it was persisted to disk
        departed_file = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(departed_file))

        # Load from disk and verify
        departed_data = ValiUtils.get_vali_json_file(departed_file, DEPARTED_HOTKEYS_KEY)
        self.assertIn(self.DEREGISTERED_MINER, departed_data)

    def test_multiple_departures_tracked(self):
        """Test tracking multiple miners leaving the metagraph"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Remove multiple miners
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys
                      if hk not in [self.DEREGISTERED_MINER, self.FUTURE_REREG_MINER]]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Verify both were tracked
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertIn(self.DEREGISTERED_MINER, departed)
        self.assertIn(self.FUTURE_REREG_MINER, departed)
        self.assertEqual(len(departed), 2)

    # ========== Anomaly Detection Tests ==========

    def test_anomalous_departure_ignored(self):
        """Test that anomalous mass departures are ignored to avoid false positives"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Create a large number of miners
        large_miner_set = [f"miner_{i}" for i in range(50)]
        self.metagraph_client.set_hotkeys(large_miner_set)

        # Clear departed hotkeys after changing metagraph (setUp tracked test miners as departed)
        self.elimination_client.clear_departed_hotkeys()

        # Process once to set previous_metagraph_hotkeys
        self.elimination_client.process_eliminations()

        # Remove 30% of miners (should trigger anomaly detection: >10 hotkeys AND >=25%)
        miners_to_remove = large_miner_set[:15]  # 15 out of 50 = 30%
        new_hotkeys = [hk for hk in large_miner_set if hk not in miners_to_remove]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Verify departed hotkeys were NOT tracked (anomaly detected)
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 0)

    def test_normal_departure_below_anomaly_threshold(self):
        """Test that normal departures below threshold are tracked"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Create miners
        miner_set = [f"miner_{i}" for i in range(50)]
        self.metagraph_client.set_hotkeys(miner_set)

        # Clear departed hotkeys after changing metagraph (setUp tracked test miners as departed)
        self.elimination_client.clear_departed_hotkeys()

        # Process once to set baseline
        self.elimination_client.process_eliminations()

        # Remove only 5 miners (5 out of 50 = 10%, below 25% threshold)
        miners_to_remove = miner_set[:5]
        new_hotkeys = [hk for hk in miner_set if hk not in miners_to_remove]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Verify departed hotkeys WERE tracked (not anomalous)
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 5)
        for miner in miners_to_remove:
            self.assertIn(miner, departed)

    def test_anomaly_threshold_boundary(self):
        """Test anomaly detection at exact boundary conditions"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Create exactly 40 miners (to test 10 miner / 25% boundary)
        miner_set = [f"miner_{i}" for i in range(40)]
        self.metagraph_client.set_hotkeys(miner_set)

        # Clear departed hotkeys after changing metagraph (setUp tracked test miners as departed)
        self.elimination_client.clear_departed_hotkeys()

        self.elimination_client.process_eliminations()

        # Remove exactly 10 miners = 25% (boundary case: should NOT trigger anomaly, needs >10)
        miners_to_remove = miner_set[:10]
        new_hotkeys = [hk for hk in miner_set if hk not in miners_to_remove]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        self.elimination_client.process_eliminations()

        # At boundary (exactly 10 miners AND 25%), should NOT trigger anomaly (needs > 10)
        # So departed hotkeys should be tracked
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 10)

    def test_below_anomaly_threshold_boundary(self):
        """Test tracking just below anomaly threshold"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Create 41 miners
        miner_set = [f"miner_{i}" for i in range(41)]
        self.metagraph_client.set_hotkeys(miner_set)

        # Clear departed hotkeys after changing metagraph (setUp tracked test miners as departed)
        self.elimination_client.clear_departed_hotkeys()

        self.elimination_client.process_eliminations()

        # Remove 10 miners = 24.4% (just below 25% threshold, should NOT trigger anomaly)
        miners_to_remove = miner_set[:10]
        new_hotkeys = [hk for hk in miner_set if hk not in miners_to_remove]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        self.elimination_client.process_eliminations()

        # Just below threshold, should track
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 10)

    # ========== Re-registration Detection Tests ==========

    def test_reregistration_detection(self):
        """Test detection when a departed miner re-registers"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Remove miner from metagraph
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.REREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process to track departure
        self.elimination_client.process_eliminations()
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertIn(self.REREGISTERED_MINER, departed)

        # Re-add miner to metagraph (simulate re-registration)
        new_hotkeys.append(self.REREGISTERED_MINER)
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations again
        self.elimination_client.process_eliminations()

        # Verify re-registration was detected (check via is_hotkey_re_registered)
        self.assertTrue(self.elimination_client.is_hotkey_re_registered(self.REREGISTERED_MINER))

        # Verify the hotkey is still in departed list (permanent record)
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertIn(self.REREGISTERED_MINER, departed)

    def test_is_hotkey_re_registered_method(self):
        """Test the is_hotkey_re_registered() lookup method"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Normal miner - should return False
        self.assertFalse(self.elimination_client.is_hotkey_re_registered(self.NORMAL_MINER))

        # Miner that has never been in metagraph - should return False
        self.assertFalse(self.elimination_client.is_hotkey_re_registered("unknown_miner"))

        # Set up re-registered miner
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.REREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        self.elimination_client.process_eliminations()

        # While departed - should return False (not currently in metagraph)
        self.assertFalse(self.elimination_client.is_hotkey_re_registered(self.REREGISTERED_MINER))

        # Re-add to metagraph
        new_hotkeys.append(self.REREGISTERED_MINER)
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Now should return True (in metagraph AND in departed list)
        self.assertTrue(self.elimination_client.is_hotkey_re_registered(self.REREGISTERED_MINER))

    def test_multiple_reregistrations_tracked(self):
        """Test tracking multiple re-registrations"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Set up multiple re-registered miners
        miners_to_rereg = [self.REREGISTERED_MINER, self.FUTURE_REREG_MINER]

        # De-register both
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk not in miners_to_rereg]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        self.elimination_client.process_eliminations()

        # Verify both tracked as departed
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 2)

        # Re-register both
        new_hotkeys.extend(miners_to_rereg)
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Both should be detected as re-registered
        for miner in miners_to_rereg:
            self.assertTrue(self.elimination_client.is_hotkey_re_registered(miner))

    # ========== Persistence Tests ==========

    def test_departed_hotkeys_persistence_across_restart(self):
        """Test that departed hotkeys persist across elimination manager restart"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Track some departed miners
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys
                      if hk not in [self.DEREGISTERED_MINER, self.FUTURE_REREG_MINER]]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        self.elimination_client.process_eliminations()

        # Verify they were tracked
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 2)

        # Create new elimination client (simulate restart - connects to same server)
        new_elimination_client = EliminationClient()

        # Verify departed hotkeys were loaded from disk
        departed = new_elimination_client.get_departed_hotkeys()
        self.assertEqual(len(departed), 2)
        self.assertIn(self.DEREGISTERED_MINER, departed)
        self.assertIn(self.FUTURE_REREG_MINER, departed)

    def test_departed_file_format(self):
        """Test that the departed hotkeys file has correct format"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Track some departures
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.DEREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        self.elimination_client.process_eliminations()

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

    def test_no_duplicate_departed_tracking(self):
        """Test that the same miner isn't added to departed list multiple times"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Remove miner
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.DEREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        self.elimination_client.process_eliminations()

        # Process multiple times
        self.elimination_client.process_eliminations()
        self.elimination_client.process_eliminations()

        # Should only appear once (dict keys are unique by definition)
        departed = self.elimination_client.get_departed_hotkeys()
        self.assertIn(self.DEREGISTERED_MINER, departed)
        self.assertEqual(len(departed), 1)

    # ========== Validator Rejection Tests ==========

    def test_validator_rejects_reregistered_orders(self):
        """Test that validator's should_fail_early logic would reject re-registered miners"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Import for type checking
        from unittest.mock import Mock
        import template

        # Create mock synapse for signal
        mock_synapse = Mock(spec=template.protocol.SendSignal)
        mock_synapse.dendrite = Mock()
        mock_synapse.dendrite.hotkey = self.REREGISTERED_MINER
        mock_synapse.miner_order_uuid = "test_uuid"
        mock_synapse.successfully_processed = True
        mock_synapse.error_message = ""

        # Set up re-registered miner
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.REREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        self.elimination_client.process_eliminations()
        new_hotkeys.append(self.REREGISTERED_MINER)
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Verify re-registration detected
        self.assertTrue(self.elimination_client.is_hotkey_re_registered(self.REREGISTERED_MINER))

        # Test rejection logic directly (simulating should_fail_early check)
        if self.elimination_client.is_hotkey_re_registered(mock_synapse.dendrite.hotkey):
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
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        from unittest.mock import Mock
        import template

        # Create mock synapse
        mock_synapse = Mock(spec=template.protocol.SendSignal)
        mock_synapse.dendrite = Mock()
        mock_synapse.dendrite.hotkey = self.NORMAL_MINER
        mock_synapse.successfully_processed = True
        mock_synapse.error_message = ""

        # Normal miner should not be flagged as re-registered
        self.assertFalse(self.elimination_client.is_hotkey_re_registered(self.NORMAL_MINER))

        # Simulate the check (should pass)
        if self.elimination_client.is_hotkey_re_registered(mock_synapse.dendrite.hotkey):
            mock_synapse.successfully_processed = False
            mock_synapse.error_message = "Should not reach here"

        # Verify order was NOT rejected
        self.assertTrue(mock_synapse.successfully_processed)
        self.assertEqual(mock_synapse.error_message, "")

    def test_departed_miner_not_yet_reregistered(self):
        """Test that departed miners (not yet re-registered) are handled correctly"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        from unittest.mock import Mock
        import template

        # Create mock synapse
        mock_synapse = Mock(spec=template.protocol.SendSignal)
        mock_synapse.dendrite = Mock()
        mock_synapse.dendrite.hotkey = self.DEREGISTERED_MINER

        # De-register the miner
        current_hotkeys = self.metagraph_client.get_hotkeys()
        new_hotkeys = [hk for hk in current_hotkeys if hk != self.DEREGISTERED_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        self.elimination_client.process_eliminations()

        # Departed but not re-registered should return False (not in metagraph)
        self.assertFalse(self.elimination_client.is_hotkey_re_registered(self.DEREGISTERED_MINER))
