# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Integration tests for the complete elimination flow using server/client architecture.
Tests end-to-end elimination scenarios with real server infrastructure.
"""
import os

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class TestEliminationIntegration(TestBase):
    """
    Integration tests for complete elimination flow using server/client architecture.
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
    HEALTHY_MINER = "healthy_miner"
    MDD_MINER = "mdd_miner"
    PLAGIARIST_MINER = "plagiarist_miner"
    CHALLENGE_FAIL_MINER = "challenge_fail_miner"
    ZOMBIE_MINER = "zombie_miner"
    LIQUIDATED_MINER = "liquidated_miner"
    NEW_MINER = "new_miner"
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

        # Define test miners BEFORE creating test data
        cls.all_test_miners = [
            "healthy_miner",
            "mdd_miner",
            "plagiarist_miner",
            "challenge_fail_miner",
            "zombie_miner",
            "liquidated_miner",
            "new_miner"
        ]
        # Initialize metagraph with test miners
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

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        # Define all test miners
        self.all_miners = [
            self.HEALTHY_MINER,
            self.MDD_MINER,
            self.PLAGIARIST_MINER,
            self.CHALLENGE_FAIL_MINER,
            self.ZOMBIE_MINER,
            self.LIQUIDATED_MINER,
            self.NEW_MINER
        ]

        # Set up metagraph with all miner names
        self.metagraph_client.set_hotkeys(self.all_miners)

        # Set up initial positions for all miners
        self._setup_positions()

        # Set up challenge period status
        self._setup_challenge_period_status()

        # Set up performance ledgers
        self._setup_perf_ledgers()

    def _setup_positions(self):
        """Create positions for all miners with diverse trade pairs"""
        base_time = TimeUtil.now_in_millis() - MS_IN_24_HOURS * 10

        for miner in self.all_miners:
            # Create multiple positions per miner with different trade pairs
            for i, trade_pair in enumerate([TradePair.BTCUSD, TradePair.ETHUSD, TradePair.GBPUSD]):
                position = Position(
                    miner_hotkey=miner,
                    position_uuid=f"{miner}_{trade_pair.trade_pair_id}_{i}",
                    open_ms=base_time + (i * MS_IN_8_HOURS),
                    trade_pair=trade_pair,
                    is_closed_position=False,
                    account_size=self.DEFAULT_ACCOUNT_SIZE,
                    orders=[Order(
                        price=60000 if trade_pair == TradePair.BTCUSD else (
                            3000 if trade_pair == TradePair.ETHUSD else 1.25
                        ),
                        processed_ms=base_time + (i * MS_IN_8_HOURS),
                        order_uuid=f"order_{miner}_{trade_pair.trade_pair_id}_{i}",
                        trade_pair=trade_pair,
                        order_type=OrderType.LONG if i % 2 == 0 else OrderType.SHORT,
                        leverage=0.5 + (i * 0.1)
                    )]
                )
                self.position_client.save_miner_position(position)

    def _setup_challenge_period_status(self):
        """Set up challenge period status for miners"""
        # Build miners dict
        miners = {}

        # Main competition miners
        for miner in [self.HEALTHY_MINER, self.MDD_MINER, self.PLAGIARIST_MINER,
                      self.ZOMBIE_MINER, self.LIQUIDATED_MINER]:
            miners[miner] = (MinerBucket.MAINCOMP, 0, None, None)

        # Challenge period miner - past minimum days
        miners[self.CHALLENGE_FAIL_MINER] = (
            MinerBucket.CHALLENGE,
            TimeUtil.now_in_millis() - (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS * 24 * 60 * 60 * 1000) - MS_IN_24_HOURS,
            None,
            None
        )

        # New miner in challenge period
        miners[self.NEW_MINER] = (
            MinerBucket.CHALLENGE,
            TimeUtil.now_in_millis() - MS_IN_24_HOURS,
            None,
            None
        )

        # Update using client API
        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners(miners)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

    def _setup_perf_ledgers(self):
        """Set up performance ledgers for testing"""
        ledgers = {}

        # Healthy miner - good performance
        ledgers[self.HEALTHY_MINER] = generate_winning_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        # MDD miner - will be eliminated (>10% drawdown)
        ledgers[self.MDD_MINER] = generate_losing_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        # Plagiarist - good performance (plagiarism detection is separate)
        ledgers[self.PLAGIARIST_MINER] = generate_winning_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        # Challenge fail miner - poor performance
        ledgers[self.CHALLENGE_FAIL_MINER] = generate_losing_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        # Zombie, liquidated, and new miners - normal performance
        for miner in [self.ZOMBIE_MINER, self.LIQUIDATED_MINER, self.NEW_MINER]:
            ledgers[miner] = generate_winning_ledger(
                0,
                ValiConfig.TARGET_LEDGER_WINDOW_MS
            )

        self.perf_ledger_client.save_perf_ledgers(ledgers)
        self.perf_ledger_client.re_init_perf_ledger_data()

    def test_complete_elimination_flow(self):
        """Test the complete elimination flow from detection to persistence"""
        # Step 1: Initial state verification
        initial_eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(initial_eliminations), 0)

        # Verify all miners have open positions
        for miner in self.all_miners:
            positions = self.position_client.get_positions_for_one_hotkey(
                miner, only_open_positions=True
            )
            self.assertGreater(len(positions), 0)

        # Step 2: Initial processing to detect MDD eliminations
        self.elimination_client.process_eliminations()

        # Verify MDD miner was eliminated
        eliminations = self.elimination_client.get_eliminations_from_memory()
        mdd_elim = next((e for e in eliminations if e['hotkey'] == self.MDD_MINER), None)
        self.assertIsNotNone(mdd_elim)
        self.assertEqual(mdd_elim['reason'], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

        # Step 3: Simulate zombie miner (remove from metagraph)
        new_hotkeys = [hk for hk in self.metagraph_client.get_hotkeys() if hk != self.ZOMBIE_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations (should detect zombie)
        self.elimination_client.process_eliminations()

        # Verify zombie was eliminated
        eliminations = self.elimination_client.get_eliminations_from_memory()
        zombie_elim = next((e for e in eliminations if e['hotkey'] == self.ZOMBIE_MINER), None)
        self.assertIsNotNone(zombie_elim)
        self.assertEqual(zombie_elim['reason'], EliminationReason.ZOMBIE.value)

        # Step 4: Challenge period failure
        self.challenge_period_client.update_elimination_reasons({
            self.CHALLENGE_FAIL_MINER: (
                EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
                0.08
            )
        })

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Verify challenge fail elimination
        eliminations = self.elimination_client.get_eliminations_from_memory()
        challenge_elim = next(
            (e for e in eliminations if e['hotkey'] == self.CHALLENGE_FAIL_MINER), None
        )
        self.assertIsNotNone(challenge_elim)
        self.assertEqual(
            challenge_elim['reason'],
            EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value
        )

        # Step 5: Perf ledger elimination (liquidation)
        pl_elim = {
            'hotkey': self.LIQUIDATED_MINER,
            'reason': EliminationReason.LIQUIDATED.value,
            'dd': 0.20,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {
                str(TradePair.BTCUSD): 45000,
                str(TradePair.ETHUSD): 2200
            }
        }
        self.perf_ledger_client.add_elimination_row(pl_elim)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Step 6: Verify all eliminations
        final_eliminations = self.elimination_client.get_eliminations_from_memory()
        eliminated_hotkeys = [e['hotkey'] for e in final_eliminations]

        self.assertIn(self.MDD_MINER, eliminated_hotkeys)
        self.assertIn(self.ZOMBIE_MINER, eliminated_hotkeys)
        self.assertIn(self.CHALLENGE_FAIL_MINER, eliminated_hotkeys)
        self.assertIn(self.LIQUIDATED_MINER, eliminated_hotkeys)

        # Step 7: Verify positions were closed for eliminated miners
        for eliminated_miner in [self.MDD_MINER, self.CHALLENGE_FAIL_MINER, self.LIQUIDATED_MINER]:
            positions = self.position_client.get_positions_for_one_hotkey(eliminated_miner)
            for pos in positions:
                self.assertTrue(pos.is_closed_position)
                # Verify flat order was added
                self.assertEqual(pos.orders[-1].order_type, OrderType.FLAT)

        # Step 8: Verify healthy miners still have open positions
        healthy_positions = self.position_client.get_positions_for_one_hotkey(
            self.HEALTHY_MINER, only_open_positions=True
        )
        self.assertGreater(len(healthy_positions), 0)

        # Step 9: Test persistence
        elimination_file = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(elimination_file))

        # Verify eliminations persisted to disk
        persisted_eliminations = self.elimination_client.get_eliminations_from_disk()
        persisted_hotkeys = [e['hotkey'] for e in persisted_eliminations]

        for eliminated_miner in [self.MDD_MINER, self.CHALLENGE_FAIL_MINER]:
            self.assertIn(eliminated_miner, persisted_hotkeys)

    def test_concurrent_elimination_scenarios(self):
        """Test handling of multiple concurrent elimination scenarios"""
        # Set up multiple elimination conditions simultaneously

        # 1. Challenge period failure
        self.challenge_period_client.update_elimination_reasons({
            self.CHALLENGE_FAIL_MINER: (
                EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value,
                None
            )
        })

        # 2. Perf ledger liquidation
        self.perf_ledger_client.add_elimination_row({
            'hotkey': self.LIQUIDATED_MINER,
            'reason': EliminationReason.LIQUIDATED.value,
            'dd': 0.25,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {}
        })

        # Process all eliminations at once
        self.elimination_client.process_eliminations()

        # Verify all eliminations occurred
        eliminations = self.elimination_client.get_eliminations_from_memory()
        eliminated_hotkeys = [e['hotkey'] for e in eliminations]

        # Check each elimination type
        self.assertIn(self.MDD_MINER, eliminated_hotkeys)
        self.assertIn(self.CHALLENGE_FAIL_MINER, eliminated_hotkeys)
        self.assertIn(self.LIQUIDATED_MINER, eliminated_hotkeys)

        # Verify correct reasons
        for elim in eliminations:
            if elim['hotkey'] == self.MDD_MINER:
                self.assertEqual(elim['reason'], EliminationReason.MAX_TOTAL_DRAWDOWN.value)
            elif elim['hotkey'] == self.CHALLENGE_FAIL_MINER:
                self.assertEqual(elim['reason'], EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value)
            elif elim['hotkey'] == self.LIQUIDATED_MINER:
                self.assertEqual(elim['reason'], EliminationReason.LIQUIDATED.value)

    def test_elimination_recovery_flow(self):
        """Test that miners below MDD threshold are not eliminated"""
        # Initially process eliminations (MDD_MINER should be eliminated)
        self.elimination_client.process_eliminations()

        # Verify healthy miner is NOT eliminated (has good performance)
        eliminations = self.elimination_client.get_eliminations_from_memory()
        healthy_elim = next(
            (e for e in eliminations if e['hotkey'] == self.HEALTHY_MINER), None
        )
        self.assertIsNone(healthy_elim)

        # Update healthy miner with better performance
        improved_ledger = generate_winning_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )
        self.perf_ledger_client.save_perf_ledgers({
            self.HEALTHY_MINER: improved_ledger
        })
        self.perf_ledger_client.re_init_perf_ledger_data()

        # Process again
        self.elimination_client.process_eliminations()

        # Still not eliminated
        eliminations = self.elimination_client.get_eliminations_from_memory()
        healthy_elim = next(
            (e for e in eliminations if e['hotkey'] == self.HEALTHY_MINER), None
        )
        self.assertIsNone(healthy_elim)

        # Verify healthy miner still has open positions
        positions = self.position_client.get_positions_for_one_hotkey(
            self.HEALTHY_MINER, only_open_positions=True
        )
        self.assertGreater(len(positions), 0)

    def test_elimination_timing_and_delays(self):
        """Test elimination timing, delays, and cleanup"""
        # Create an old elimination
        old_elimination_time = TimeUtil.now_in_millis() - ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS - MS_IN_24_HOURS

        # Add old elimination directly
        old_elim = self.elimination_client.generate_elimination_row(
            'old_eliminated_miner',
            0.15,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            t_ms=old_elimination_time
        )
        self.elimination_client.add_elimination('old_eliminated_miner', old_elim)

        # Remove from metagraph (deregistered)
        new_hotkeys = [hk for hk in self.metagraph_client.get_hotkeys() if hk != 'old_eliminated_miner']
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Create miner directory
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=True) + 'old_eliminated_miner'
        os.makedirs(miner_dir, exist_ok=True)

        # Process eliminations (should clean up old elimination)
        self.elimination_client.process_eliminations()

        # Verify old elimination was removed
        current_eliminations = self.elimination_client.get_eliminations_from_memory()
        old_miner_elim = next(
            (e for e in current_eliminations if e['hotkey'] == 'old_eliminated_miner'), None
        )
        self.assertIsNone(old_miner_elim)

        # Verify directory was cleaned up
        self.assertFalse(os.path.exists(miner_dir))


    def test_multiple_eliminations_same_miner(self):
        """Test that a miner can only be eliminated once"""
        # First elimination
        self.elimination_client.add_elimination(self.MDD_MINER, {
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        })

        # Try to process eliminations again (should not duplicate)
        self.elimination_client.process_eliminations()

        # Should still have only one elimination for this miner
        eliminations = self.elimination_client.get_eliminations_from_memory()
        mdd_eliminations = [e for e in eliminations if e['hotkey'] == self.MDD_MINER]
        self.assertEqual(len(mdd_eliminations), 1)

    def test_elimination_with_no_positions(self):
        """Test elimination handling when miner has no positions"""
        # Clear positions for MDD miner
        self.position_client.clear_all_miner_positions_and_disk(hotkey=self.MDD_MINER)

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Should still be eliminated even without positions (based on perf ledger)
        eliminations = self.elimination_client.get_eliminations_from_memory()
        mdd_elim = next((e for e in eliminations if e['hotkey'] == self.MDD_MINER), None)
        self.assertIsNotNone(mdd_elim)

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
        self.assertFalse(
            self.elimination_client.is_zombie_hotkey(self.ZOMBIE_MINER, all_hotkeys_set)
        )

        # Remove from metagraph and update set
        new_hotkeys = [hk for hk in self.metagraph_client.get_hotkeys() if hk != self.ZOMBIE_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)
        all_hotkeys_set = set(self.metagraph_client.get_hotkeys())

        # Now should be zombie
        self.assertTrue(
            self.elimination_client.is_zombie_hotkey(self.ZOMBIE_MINER, all_hotkeys_set)
        )

    def test_hotkey_in_eliminations(self):
        """Test checking if hotkey is in eliminations"""
        # Add elimination
        self.elimination_client.add_elimination(self.MDD_MINER, {
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis()
        })

        # Test existing elimination
        result = self.elimination_client.hotkey_in_eliminations(self.MDD_MINER)
        self.assertIsNotNone(result)
        self.assertEqual(result['reason'], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

        # Test non-existing elimination
        result = self.elimination_client.hotkey_in_eliminations('non_existent')
        self.assertIsNone(result)

    def test_elimination_first_refresh_handling(self):
        """Test first refresh behavior after validator start"""
        # Reset first_refresh_ran flag via client
        self.elimination_client.set_first_refresh_ran(False)
        self.elimination_client.clear_eliminations()

        # First refresh should have special handling
        self.assertFalse(self.elimination_client.get_first_refresh_ran())

        # Process eliminations
        self.elimination_client.process_eliminations()

        # Flag should be set
        self.assertTrue(self.elimination_client.get_first_refresh_ran())
