# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Consolidated core elimination tests combining basic and comprehensive elimination manager functionality.
Tests all elimination types, persistence, and core operations.
"""
import os

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.challenge_period.challengeperiod_manager import MinerBucketState
from shared_objects.locks.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class TestEliminationCore(TestBase):
    """Core elimination manager functionality combining basic and comprehensive tests"""

    """
    Core elimination manager functionality combining basic and comprehensive tests.
    Uses ServerOrchestrator singleton for shared server infrastructure across all test classes.
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

    # Test miner constants
    MDD_MINER = "miner_mdd"
    REGULAR_MINER = "miner_regular"
    ZOMBIE_MINER = "miner_zombie"
    PLAGIARIST_MINER = "miner_plagiarist"
    CHALLENGE_FAIL_MINER = "miner_challenge_fail"
    DEFAULT_ACCOUNT_SIZE = 100_000

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

        # Define test miners BEFORE creating test data to avoid re-registration warnings
        cls.all_test_miners = [
            "miner_mdd",
            "miner_regular",
            "miner_zombie",
            "miner_plagiarist",
            "miner_challenge_fail",
        ]
        # Initialize metagraph with test miners
        cls.metagraph_client.set_hotkeys(cls.all_test_miners)

        # Create position locks instance
        cls.position_locks = PositionLocks()

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

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        # Define all test miners
        self.all_miners = [
            self.MDD_MINER,
            self.REGULAR_MINER,
            self.ZOMBIE_MINER,
            self.PLAGIARIST_MINER,
            self.CHALLENGE_FAIL_MINER,
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
            self.position_client.save_miner_position(position)

    def _setup_challenge_period_status(self):
        """Set up challenge period status for miners"""
        # Build miners dict
        miners = {}

        # Most miners in main competition
        for miner in [self.MDD_MINER, self.REGULAR_MINER, self.ZOMBIE_MINER, self.PLAGIARIST_MINER]:
            miners[miner] = (MinerBucket.MAINCOMP, 0, None, None)

        # Challenge fail miner in challenge period
        miners[self.CHALLENGE_FAIL_MINER] = (
            MinerBucket.CHALLENGE,
            TimeUtil.now_in_millis() - (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS * 24 * 60 * 60 * 1000) - MS_IN_8_HOURS,
            None,
            None
        )

        # Update using client API
        miner_states_data = {
            hotkey: MinerBucketState(hotkey, [BucketEntry(bucket, start_time)]).to_json()
            for hotkey, (bucket, start_time, _, _) in miners.items()
        }
        self.challenge_period_client.sync_challenge_period_data(miner_states_data)

    def _setup_perf_ledgers(self):
        """Set up performance ledgers for testing"""
        ledgers = {}

        # MDD miner - will be eliminated
        ledgers[self.MDD_MINER] = generate_losing_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        # Regular miners - good performance
        for miner in [self.REGULAR_MINER, self.ZOMBIE_MINER, self.PLAGIARIST_MINER]:
            ledgers[miner] = generate_winning_ledger(
                0,
                ValiConfig.TARGET_LEDGER_WINDOW_MS
            )

        # Challenge fail miner - poor performance
        ledgers[self.CHALLENGE_FAIL_MINER] = generate_losing_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        self.perf_ledger_client.save_perf_ledgers(ledgers)
        self.perf_ledger_client.re_init_perf_ledger_data()

    # ========== Basic Elimination Tests (from test_elimination_manager.py) ==========

    def test_basic_mdd_elimination(self):
        """Test basic MDD elimination functionality"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Initially no eliminations
        self.assertEqual(len(self.challenge_period_client.get_success_miners()), 4)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Check MDD miner was eliminated
        eliminations = self.elimination_client.get_eliminations_from_disk()
        self.assertEqual(len(eliminations), 1)
        self.assertEqual(eliminations[0]["hotkey"], self.MDD_MINER)
        self.assertEqual(eliminations[0]["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

    def test_zombie_elimination_basic(self):
        """Test basic zombie elimination when miner leaves metagraph"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data
        for miner in self.all_miners:
            self.assertTrue(self.metagraph_client.has_hotkey(miner))

        # Process initial eliminations
        self.elimination_client.process_eliminations()

        # Remove all miners from metagraph
        self.metagraph_client.set_hotkeys([])

        for miner in self.all_miners:
            self.assertFalse(self.metagraph_client.has_hotkey(miner))

        # Process eliminations again
        self.elimination_client.process_eliminations()

        # Check all miners are now eliminated
        eliminations = self.elimination_client.get_eliminations_from_disk()
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

    def test_challenge_period_elimination(self):
        """Test elimination for miners failing challenge period"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Set up challenge period failure
        self.elimination_client.append_elimination_row(
            self.CHALLENGE_FAIL_MINER,
            EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
            elimination_drawdown_pct=0.08
        )

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Verify elimination
        eliminations = self.elimination_client.get_eliminations_from_memory()
        challenge_elim = next((e for e in eliminations if e["hotkey"] == self.CHALLENGE_FAIL_MINER), None)
        self.assertIsNotNone(challenge_elim)
        self.assertEqual(challenge_elim["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value)
        self.assertEqual(challenge_elim["dd"], 0.08)


    def test_elimination_persistence(self):
        """Test that eliminations are persisted to disk correctly"""
        # Add elimination using append_elimination_row which saves to disk
        test_dd = 0.12
        test_reason = EliminationReason.MAX_TOTAL_DRAWDOWN.value
        test_time = TimeUtil.now_in_millis()

        self.elimination_client.append_elimination_row(
            self.MDD_MINER,
            test_reason,
            elimination_drawdown_pct=test_dd,
            elimination_time_ms=test_time
        )

        # Verify it's in memory
        eliminations_in_memory = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(eliminations_in_memory), 1)
        self.assertEqual(eliminations_in_memory[0]['hotkey'], self.MDD_MINER)

        # Load from disk to verify persistence
        loaded_eliminations = self.elimination_client.get_eliminations_from_disk()

        # Verify persistence
        self.assertEqual(len(loaded_eliminations), 1)
        self.assertEqual(loaded_eliminations[0]['hotkey'], self.MDD_MINER)
        self.assertEqual(loaded_eliminations[0]['reason'], test_reason)
        self.assertEqual(loaded_eliminations[0]['dd'], test_dd)

    def test_elimination_row_generation(self):
        """Test elimination row data structure generation"""
        test_dd = 0.15
        test_reason = EliminationReason.MAX_TOTAL_DRAWDOWN.value
        test_time = TimeUtil.now_in_millis()

        self.elimination_client.append_elimination_row(
            self.MDD_MINER,
            test_reason,
            elimination_drawdown_pct=test_dd,
            elimination_time_ms=test_time
        )
        row = self.elimination_client.get_elimination(self.MDD_MINER)

        # Verify structure
        self.assertEqual(row['hotkey'], self.MDD_MINER)
        self.assertEqual(row['elimination_drawdown_pct'], test_dd)
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
        self.elimination_client.sync_eliminations([test_elim])

        # Verify it was added
        eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(eliminations), 1)
        self.assertEqual(eliminations[0]['hotkey'], self.MDD_MINER)

    def test_is_zombie_hotkey(self):
        """Test zombie hotkey detection"""
        # Get all hotkeys set
        all_hotkeys_set = set(self.metagraph_client.get_hotkeys())

        # Initially not zombie
        self.assertFalse(self.elimination_client.is_zombie_hotkey(self.ZOMBIE_MINER, all_hotkeys_set))

        # Remove from metagraph and update set
        new_hotkeys = [hk for hk in self.metagraph_client.get_hotkeys() if hk != self.ZOMBIE_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        all_hotkeys_set = set(self.metagraph_client.get_hotkeys())

        # Now should be zombie
        self.assertTrue(self.elimination_client.is_zombie_hotkey(self.ZOMBIE_MINER, all_hotkeys_set))

    def test_hotkey_in_eliminations(self):
        """Test checking if hotkey is in eliminations"""
        # Add elimination
        self.elimination_client.append_elimination_row(
            self.MDD_MINER,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            elimination_drawdown_pct=0.12
        )

        # Test existing elimination
        result = self.elimination_client.hotkey_in_eliminations(self.MDD_MINER)
        self.assertIsNotNone(result)
        self.assertEqual(result['reason'], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

        # Test non-existing elimination
        result = self.elimination_client.hotkey_in_eliminations('non_existent')
        self.assertIsNone(result)

    def test_elimination_with_ipc_manager(self):
        """Test elimination manager with RPC client/server pattern"""
        # Clear any existing eliminations
        self.elimination_client.clear_eliminations()

        # Test adding elimination via RPC
        self.elimination_client.append_elimination_row(
            self.MDD_MINER,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            elimination_drawdown_pct=0.12
        )

        # Verify it works with RPC
        eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(eliminations), 1)

    def test_multiple_eliminations_same_miner(self):
        """Test that a miner can only be eliminated once"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # First elimination
        self.elimination_client.append_elimination_row(
            self.MDD_MINER,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            elimination_drawdown_pct=0.12
        )

        # Try to add another elimination for same miner
        # Process eliminations should not duplicate
        self.elimination_client.process_eliminations()

        # Should still have only one elimination for this miner
        eliminations = self.elimination_client.get_eliminations_from_memory()
        mdd_eliminations = [e for e in eliminations if e['hotkey'] == self.MDD_MINER]
        self.assertEqual(len(mdd_eliminations), 1)

    def test_elimination_deletion_after_timeout(self):
        """Test that old eliminations are cleaned up after timeout"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Create an old elimination
        old_time = TimeUtil.now_in_millis() - ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS - MS_IN_8_HOURS

        self.elimination_client.append_elimination_row(
            'old_miner',
            EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            elimination_drawdown_pct=0.15,
            elimination_time_ms=old_time
        )

        # Remove from metagraph
        new_hotkeys = [hk for hk in self.metagraph_client.get_hotkeys() if hk != 'old_miner']
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Create miner directory
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=True) + 'old_miner'
        os.makedirs(miner_dir, exist_ok=True)

        # Process eliminations (should purge files but retain elimination record)
        self.elimination_client.process_eliminations()

        # Verify miner directory was purged
        self.assertFalse(os.path.exists(miner_dir))

        # Elimination record is retained in state after purge
        eliminations = self.elimination_client.get_eliminations_from_memory()
        old_miner_elim = next((e for e in eliminations if e['hotkey'] == 'old_miner'), None)
        self.assertIsNotNone(old_miner_elim)

    def test_elimination_with_no_positions(self):
        """Test elimination handling when miner has no positions"""
        # Clear positions for MDD miner
        self.position_client.clear_all_miner_positions_and_disk(hotkey=self.MDD_MINER)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Should still be eliminated even without positions
        eliminations = self.elimination_client.get_eliminations_from_memory()
        mdd_elim = next((e for e in eliminations if e['hotkey'] == self.MDD_MINER), None)
        self.assertIsNotNone(mdd_elim)

    def test_elimination_first_refresh_handling(self):
        """Test first refresh behavior after validator start"""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Reset first_refresh_ran flag via client
        self.elimination_client.set_first_refresh_ran(False)
        self.elimination_client.clear_eliminations()

        # First refresh should have special handling
        self.assertFalse(self.elimination_client.get_first_refresh_ran())

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Flag should be set
        self.assertTrue(self.elimination_client.get_first_refresh_ran())
