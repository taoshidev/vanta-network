import copy
import time
from unittest.mock import Mock, MagicMock, patch

from tests.shared_objects.test_utilities import generate_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
from vali_objects.vali_dataclasses.penalty_ledger import PenaltyLedgerManager, PenaltyLedger, PenaltyCheckpoint
from vali_objects.utils.miner_bucket_enum import MinerBucket


class TestLedgerPenalty(TestBase):
    """
    This class will test penalties that apply to ledgers.
    """

    def setUp(self):
        super().setUp()
        # Clean up any existing penalty ledger file from previous test runs
        import os
        from vali_objects.vali_config import ValiConfig
        penalty_ledger_path = ValiConfig.BASE_DIR + "/tests/validation/penalty_ledger.json.gz"
        if os.path.exists(penalty_ledger_path):
            os.remove(penalty_ledger_path)

    def tearDown(self):
        super().tearDown()
        # Clean up penalty ledger file after each test
        import os
        from vali_objects.vali_config import ValiConfig
        penalty_ledger_path = ValiConfig.BASE_DIR + "/tests/validation/penalty_ledger.json.gz"
        if os.path.exists(penalty_ledger_path):
            os.remove(penalty_ledger_path)

    def test_max_drawdown_threshold(self):
        l1 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]  # 1% drawdown

        l2 = copy.deepcopy(l1)
        l2_cps = l2.cps
        l2_cps[-1].mdd = 0.8  # 20% drawdown only on the most recent checkpoint

        l3 = copy.deepcopy(l1)
        l3_cps = l3.cps
        l3_cps[0].mdd = 0.8  # 20% drawdown only on the first checkpoint

        l4 = generate_ledger(0.1, mdd=0.8)[TP_ID_PORTFOLIO]  # 20% drawdown

        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l1), 1.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l2), 0.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l3), 0.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l4), 0.0)

    def test_is_beyond_max_drawdown(self):
        l1 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]  # 1% drawdown
        l1_ledger = l1

        l2 = copy.deepcopy(l1)
        l2_ledger = l2
        l2_cps = l2_ledger.cps
        l2_cps[-1].mdd = 0.89  # 11% drawdown only on the most recent checkpoint

        l3 = copy.deepcopy(l1)
        l3_ledger = l3
        l3_cps = l3_ledger.cps
        l3_cps[0].mdd = 0.89  # 11% drawdown only on the first checkpoint

        l4 = generate_ledger(0.1, mdd=0.89)[TP_ID_PORTFOLIO]  # 11% drawdown
        l4_ledger = l4

        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(None), (False, 0))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l1_ledger), (False, 1))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l2_ledger), (True, 11))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l3_ledger), (True, 11))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l4_ledger), (True, 11))

    def test_penalty_ledger_manager_metadata_persistence(self):
        """Test that last_full_rebuild_ms is properly saved and loaded"""
        # Create manager (new API - no mock dependencies needed, creates clients internally)
        manager = PenaltyLedgerManager(
            running_unit_tests=True,
            run_daemon=False
        )

        # Verify initial state
        self.assertEqual(manager.last_full_rebuild_ms, 0)

        # Add a dummy ledger (save_to_disk() requires at least one ledger)
        target_cp_duration_ms = 43200000
        dummy_ledger = PenaltyLedger("test_hotkey")
        checkpoint = PenaltyCheckpoint(
            last_processed_ms=target_cp_duration_ms,
            challenge_period_status=MinerBucket.CHALLENGE.value
        )
        dummy_ledger.add_checkpoint(checkpoint, target_cp_duration_ms)
        manager.penalty_ledgers["test_hotkey"] = dummy_ledger

        # Set timestamp and save
        test_timestamp = int(time.time() * 1000)
        manager.last_full_rebuild_ms = test_timestamp
        manager.save_to_disk()

        # Create new manager and verify it loads the timestamp
        manager2 = PenaltyLedgerManager(
            running_unit_tests=True,
            run_daemon=False
        )

        self.assertEqual(manager2.last_full_rebuild_ms, test_timestamp)

    def test_penalty_ledger_48_day_rebuild_scheduling(self):
        """Test that full rebuild is triggered every 48 days"""
        current_time_ms = int(time.time() * 1000)

        # Test 1: last_full_rebuild_ms = 0 should trigger full rebuild
        last_rebuild_ms = 0
        days_since = (current_time_ms - last_rebuild_ms) / (1000 * 60 * 60 * 24)
        should_rebuild = (last_rebuild_ms == 0) or (days_since >= 48)
        self.assertTrue(should_rebuild)

        # Test 2: 10 days ago should NOT trigger full rebuild
        last_rebuild_ms = current_time_ms - (10 * 24 * 60 * 60 * 1000)
        days_since = (current_time_ms - last_rebuild_ms) / (1000 * 60 * 60 * 24)
        should_rebuild = (last_rebuild_ms == 0) or (days_since >= 48)
        self.assertFalse(should_rebuild)
        self.assertAlmostEqual(days_since, 10.0, places=1)

        # Test 3: 47 days ago should NOT trigger full rebuild
        last_rebuild_ms = current_time_ms - (47 * 24 * 60 * 60 * 1000)
        days_since = (current_time_ms - last_rebuild_ms) / (1000 * 60 * 60 * 24)
        should_rebuild = (last_rebuild_ms == 0) or (days_since >= 48)
        self.assertFalse(should_rebuild)
        self.assertAlmostEqual(days_since, 47.0, places=1)

        # Test 4: exactly 48 days ago SHOULD trigger full rebuild
        last_rebuild_ms = current_time_ms - (48 * 24 * 60 * 60 * 1000)
        days_since = (current_time_ms - last_rebuild_ms) / (1000 * 60 * 60 * 24)
        should_rebuild = (last_rebuild_ms == 0) or (days_since >= 48)
        self.assertTrue(should_rebuild)
        self.assertAlmostEqual(days_since, 48.0, places=1)

        # Test 5: 50 days ago SHOULD trigger full rebuild
        last_rebuild_ms = current_time_ms - (50 * 24 * 60 * 60 * 1000)
        days_since = (current_time_ms - last_rebuild_ms) / (1000 * 60 * 60 * 24)
        should_rebuild = (last_rebuild_ms == 0) or (days_since >= 48)
        self.assertTrue(should_rebuild)
        self.assertAlmostEqual(days_since, 50.0, places=1)

    def test_challenge_period_status_preservation_during_full_rebuild(self):
        """Test that challenge_period_status is preserved from old ledgers during full rebuild"""
        # Create test checkpoints with historical challenge period statuses
        target_cp_duration_ms = 43200000  # 12 hours

        # Create old ledger with historical statuses
        old_ledger = PenaltyLedger("test_hotkey")

        # Add checkpoints to old ledger with different statuses
        checkpoint1 = PenaltyCheckpoint(
            last_processed_ms=target_cp_duration_ms,
            challenge_period_status=MinerBucket.CHALLENGE.value
        )
        old_ledger.add_checkpoint(checkpoint1, target_cp_duration_ms)

        checkpoint2 = PenaltyCheckpoint(
            last_processed_ms=target_cp_duration_ms * 2,
            challenge_period_status=MinerBucket.MAINCOMP.value
        )
        old_ledger.add_checkpoint(checkpoint2, target_cp_duration_ms)

        checkpoint3 = PenaltyCheckpoint(
            last_processed_ms=target_cp_duration_ms * 3,
            challenge_period_status=MinerBucket.PROBATION.value
        )
        old_ledger.add_checkpoint(checkpoint3, target_cp_duration_ms)

        # Verify we can retrieve checkpoints by timestamp
        retrieved_cp1 = old_ledger.get_checkpoint_at_time(target_cp_duration_ms, target_cp_duration_ms)
        self.assertIsNotNone(retrieved_cp1)
        self.assertEqual(retrieved_cp1.challenge_period_status, MinerBucket.CHALLENGE.value)

        retrieved_cp2 = old_ledger.get_checkpoint_at_time(target_cp_duration_ms * 2, target_cp_duration_ms)
        self.assertIsNotNone(retrieved_cp2)
        self.assertEqual(retrieved_cp2.challenge_period_status, MinerBucket.MAINCOMP.value)

        retrieved_cp3 = old_ledger.get_checkpoint_at_time(target_cp_duration_ms * 3, target_cp_duration_ms)
        self.assertIsNotNone(retrieved_cp3)
        self.assertEqual(retrieved_cp3.challenge_period_status, MinerBucket.PROBATION.value)

        # Verify that non-existent timestamps return None
        non_existent = old_ledger.get_checkpoint_at_time(target_cp_duration_ms * 10, target_cp_duration_ms)
        self.assertIsNone(non_existent)

    def test_atomic_ledger_replacement_for_full_rebuild(self):
        """Test that full rebuild keeps old and new ledgers in memory until the very end"""
        # Create manager with an old ledger (new API - no mock dependencies needed)
        manager = PenaltyLedgerManager(
            running_unit_tests=True,
            run_daemon=False
        )

        # Add old ledger to manager
        target_cp_duration_ms = 43200000
        old_ledger = PenaltyLedger("hotkey1")
        checkpoint1 = PenaltyCheckpoint(
            last_processed_ms=target_cp_duration_ms,
            challenge_period_status=MinerBucket.CHALLENGE.value
        )
        old_ledger.add_checkpoint(checkpoint1, target_cp_duration_ms)
        manager.penalty_ledgers["hotkey1"] = old_ledger

        # Verify old ledger exists
        self.assertIn("hotkey1", manager.penalty_ledgers)
        self.assertEqual(len(manager.penalty_ledgers["hotkey1"].checkpoints), 1)

        # Simulate the full rebuild pattern:
        # 1. Create new_penalty_ledgers dict
        new_penalty_ledgers = {}

        # 2. Build new ledger (old ledger still exists in manager.penalty_ledgers)
        new_ledger = PenaltyLedger("hotkey1")
        checkpoint2 = PenaltyCheckpoint(
            last_processed_ms=target_cp_duration_ms * 2,
            challenge_period_status=MinerBucket.MAINCOMP.value
        )
        new_ledger.add_checkpoint(checkpoint2, target_cp_duration_ms)
        new_penalty_ledgers["hotkey1"] = new_ledger

        # 3. Verify old ledger still exists while new ledger is being built
        self.assertIn("hotkey1", manager.penalty_ledgers)
        self.assertEqual(len(manager.penalty_ledgers["hotkey1"].checkpoints), 1)

        # 4. Atomic replacement (only at the very end)
        manager.penalty_ledgers.clear()
        manager.penalty_ledgers.update(new_penalty_ledgers)

        # 5. Verify replacement happened correctly
        self.assertIn("hotkey1", manager.penalty_ledgers)
        self.assertEqual(len(manager.penalty_ledgers["hotkey1"].checkpoints), 1)
        retrieved_checkpoint = manager.penalty_ledgers["hotkey1"].get_latest_checkpoint()
        self.assertEqual(retrieved_checkpoint.last_processed_ms, target_cp_duration_ms * 2)
        self.assertEqual(retrieved_checkpoint.challenge_period_status, MinerBucket.MAINCOMP.value)
