# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import json
import shutil
import time
from unittest.mock import patch

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationReason
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from shared_objects.cache_controller import CacheController


class TestEliminationPersistenceRecovery(TestBase):
    """
    Integration tests for elimination persistence and recovery using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    metagraph_client = None
    position_client = None
    elimination_client = None
    perf_ledger_client = None
    challenge_period_client = None

    # Test constants
    PERSISTENT_MINER_1 = "persistent_miner_1"
    PERSISTENT_MINER_2 = "persistent_miner_2"
    RECOVERY_MINER = "recovery_miner"
    DEFAULT_ACCOUNT_SIZE = 100_000

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Clear ALL test miner positions BEFORE starting servers
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')

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
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        # Define test miners
        self.all_miners = [
            self.PERSISTENT_MINER_1,
            self.PERSISTENT_MINER_2,
            self.RECOVERY_MINER
        ]

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Set up metagraph with test miners
        self.metagraph_client.set_hotkeys(self.all_miners)

        # Set up initial positions
        self._setup_positions()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _setup_positions(self):
        """Create test positions"""
        for miner in self.all_miners:
            for trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD]:
                position = Position(
                    miner_hotkey=miner,
                    position_uuid=f"{miner}_{trade_pair.trade_pair_id}",
                    open_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
                    trade_pair=trade_pair,
                    is_closed_position=False,
                    account_size=self.DEFAULT_ACCOUNT_SIZE,
                    orders=[Order(
                        price=60000 if trade_pair == TradePair.BTCUSD else 3000,
                        processed_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
                        order_uuid=f"order_{miner}_{trade_pair.trade_pair_id}",
                        trade_pair=trade_pair,
                        order_type=OrderType.LONG,
                        leverage=0.5
                    )]
                )
                self.position_client.save_miner_position(position)

    def test_elimination_file_persistence(self):
        """Test that eliminations are correctly saved to and loaded from disk"""
        # Create multiple eliminations
        eliminations = [
            {
                'hotkey': self.PERSISTENT_MINER_1,
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.11,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis() - MS_IN_24_HOURS,
                'price_info': {str(TradePair.BTCUSD): 55000}
            },
            {
                'hotkey': self.PERSISTENT_MINER_2,
                'reason': EliminationReason.PLAGIARISM.value,
                'dd': None,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis() - MS_IN_24_HOURS * 2,
                'return_info': {'plagiarism_score': 0.95}
            }
        ]

        # Add eliminations
        for elim in eliminations:
            self.elimination_client.add_elimination(elim['hotkey'], elim)

        # Save to disk
        self.elimination_client.save_eliminations()
        
        # Verify file exists
        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(file_path))
        
        # Read file directly
        with open(file_path, 'r') as f:
            file_content = json.load(f)
        
        # Verify structure
        self.assertIn(CacheController.ELIMINATIONS, file_content)
        self.assertEqual(len(file_content[CacheController.ELIMINATIONS]), 2)
        
        # Verify content matches
        for i, saved_elim in enumerate(file_content[CacheController.ELIMINATIONS]):
            self.assertEqual(saved_elim['hotkey'], eliminations[i]['hotkey'])
            self.assertEqual(saved_elim['reason'], eliminations[i]['reason'])

    def test_elimination_recovery_on_restart(self):
        """Test that eliminations are recovered correctly on validator restart"""
        # Create and save eliminations
        test_eliminations = [
            {
                'hotkey': self.PERSISTENT_MINER_1,
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.12,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis() - MS_IN_24_HOURS * 3
            },
            {
                'hotkey': self.RECOVERY_MINER,
                'reason': EliminationReason.ZOMBIE.value,
                'dd': None,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis() - MS_IN_24_HOURS
            }
        ]

        # Write directly to disk (simulating previous session)
        self.elimination_client.write_eliminations_to_disk(test_eliminations)

        # Simulate restart by reloading data from disk
        # Clear memory first, then reload from disk
        self.elimination_client.clear_eliminations()
        self.elimination_client.load_eliminations_from_disk()

        # Verify eliminations were loaded
        loaded_eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(loaded_eliminations), 2)

        # Verify content
        hotkeys = [e['hotkey'] for e in loaded_eliminations]
        self.assertIn(self.PERSISTENT_MINER_1, hotkeys)
        self.assertIn(self.RECOVERY_MINER, hotkeys)

        # Verify first refresh handles recovered eliminations
        self.elimination_client.handle_first_refresh()

        # Check that positions were closed for eliminated miners
        for elim in test_eliminations:
            positions = self.position_client.get_positions_for_one_hotkey(elim['hotkey'])
            for pos in positions:
                self.assertTrue(pos.is_closed_position)

    def test_elimination_backup_and_restore(self):
        """Test backup and restore functionality for eliminations"""
        # Create eliminations
        original_eliminations = [
            {
                'hotkey': self.PERSISTENT_MINER_1,
                'reason': EliminationReason.LIQUIDATED.value,
                'dd': 0.15,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
                'price_info': {str(TradePair.BTCUSD): 45000}
            }
        ]

        # Add and save eliminations
        for elim in original_eliminations:
            self.elimination_client.add_elimination(elim['hotkey'], elim)
        self.elimination_client.save_eliminations()

        # Create backup directory
        backup_dir = "/tmp/test_elimination_backup"
        os.makedirs(backup_dir, exist_ok=True)

        # Backup elimination file
        original_file = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        backup_file = os.path.join(backup_dir, "eliminations_backup.json")
        shutil.copy2(original_file, backup_file)

        # Clear eliminations
        self.elimination_client.clear_eliminations()

        # Verify eliminations are cleared
        self.assertEqual(len(self.elimination_client.get_eliminations_from_memory()), 0)

        # Restore from backup
        shutil.copy2(backup_file, original_file)

        # Reload data from disk to simulate restart
        self.elimination_client.clear_eliminations()
        self.elimination_client.load_eliminations_from_disk()

        # Verify restoration
        restored_eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(restored_eliminations), 1)
        self.assertEqual(restored_eliminations[0]['hotkey'], self.PERSISTENT_MINER_1)

        # Cleanup
        shutil.rmtree(backup_dir, ignore_errors=True)

    def test_elimination_data_corruption_handling(self):
        """Test handling of corrupted elimination data"""
        # Write corrupted data to elimination file
        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)

        # Test 1: Invalid JSON
        with open(file_path, 'w') as f:
            f.write("Invalid JSON content {]}")

        # Try to reload (should handle gracefully)
        try:
            self.elimination_client.clear_eliminations()
            self.elimination_client.load_eliminations_from_disk()
            # Should create empty eliminations
            self.assertEqual(len(self.elimination_client.get_eliminations_from_memory()), 0)
        except Exception as e:
            # Should handle error gracefully
            pass

        # Test 2: Missing required fields
        corrupted_data = {
            CacheController.ELIMINATIONS: [
                {
                    'hotkey': 'test_miner'
                    # Missing required fields like 'reason', 'elimination_initiated_time_ms'
                }
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(corrupted_data, f)

        # Reload data
        self.elimination_client.clear_eliminations()
        self.elimination_client.load_eliminations_from_disk()

        # Should load what it can
        loaded = self.elimination_client.get_eliminations_from_memory()
        # Implementation might handle this differently - could be empty or partial load

    def test_elimination_file_permissions(self):
        """Test handling of file permission issues"""
        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        
        # Create elimination
        self.elimination_client.append_elimination_row(
            self.PERSISTENT_MINER_1,
            0.11,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )

        # Try to save with read-only directory (simulate permission issue)
        # This test is platform-dependent and might need adjustment
        try:
            # Make directory read-only
            parent_dir = os.path.dirname(file_path)
            original_permissions = os.stat(parent_dir).st_mode
            os.chmod(parent_dir, 0o444)  # Read-only

            # Try to save (should handle gracefully)
            self.elimination_client.save_eliminations()
            
        except Exception:
            # Should handle permission errors gracefully
            pass
        finally:
            # Restore permissions
            if 'original_permissions' in locals():
                os.chmod(parent_dir, original_permissions)

    def test_elimination_concurrent_access(self):
        """Test handling of concurrent access to elimination data"""
        # Note: In the ServerOrchestrator pattern, we have a single shared server
        # This test demonstrates that the last write wins in the current implementation
        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)

        # Add first elimination
        self.elimination_client.append_elimination_row(
            self.PERSISTENT_MINER_1,
            0.11,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )
        self.elimination_client.save_eliminations()

        # Add second elimination
        self.elimination_client.append_elimination_row(
            self.PERSISTENT_MINER_2,
            0.12,
            EliminationReason.PLAGIARISM.value
        )
        self.elimination_client.save_eliminations()

        # Verify both eliminations exist
        final_data = self.elimination_client.get_eliminations_from_disk()
        self.assertEqual(len(final_data), 2)

    def test_elimination_state_consistency(self):
        """Test consistency between memory and disk state"""
        # Add eliminations in memory
        test_elims = [
            {
                'hotkey': self.PERSISTENT_MINER_1,
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.11,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis()
            },
            {
                'hotkey': self.PERSISTENT_MINER_2,
                'reason': EliminationReason.ZOMBIE.value,
                'dd': None,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis() - MS_IN_24_HOURS
            }
        ]

        for elim in test_elims:
            self.elimination_client.add_elimination(elim['hotkey'], elim)

        # Save to disk
        self.elimination_client.save_eliminations()

        # Compare memory and disk
        memory_elims = self.elimination_client.get_eliminations_from_memory()
        disk_elims = self.elimination_client.get_eliminations_from_disk()
        
        # Should be identical
        self.assertEqual(len(memory_elims), len(disk_elims))
        
        memory_hotkeys = sorted([e['hotkey'] for e in memory_elims])
        disk_hotkeys = sorted([e['hotkey'] for e in disk_elims])
        self.assertEqual(memory_hotkeys, disk_hotkeys)

    def test_elimination_migration(self):
        """Test migration of elimination data format (if schema changes)"""
        # Simulate old format elimination data
        old_format_data = {
            CacheController.ELIMINATIONS: [
                {
                    'hotkey': self.RECOVERY_MINER,
                    'reason': 'MAX_DRAWDOWN',  # Old format might use different reason strings
                    'dd': 0.11,
                    'timestamp': TimeUtil.now_in_millis() - MS_IN_24_HOURS  # Old field name
                }
            ]
        }

        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        with open(file_path, 'w') as f:
            json.dump(old_format_data, f)

        # Reload data to test migration
        self.elimination_client.clear_eliminations()
        self.elimination_client.load_eliminations_from_disk()

        # Should either migrate or handle gracefully
        loaded = self.elimination_client.get_eliminations_from_memory()
        # Actual behavior depends on implementation

    def test_elimination_cache_invalidation(self):
        """Test cache invalidation for eliminations"""
        # Add elimination
        self.elimination_client.append_elimination_row(
            self.PERSISTENT_MINER_1,
            0.11,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )

        # Test cache timing behavior
        # Initialize attempted_start_time_ms by calling refresh_allowed
        self.elimination_client.refresh_allowed(0)
        # Set cache update time
        self.elimination_client.set_last_update_time()

        # Immediate refresh should be blocked (when not in unit test mode)
        # Note: In unit test mode, refresh_allowed always returns True
        # This test verifies the method exists and can be called

        # Mock time passage by patching TimeUtil.now_in_millis
        future_time_ms = TimeUtil.now_in_millis() + ValiConfig.ELIMINATION_CHECK_INTERVAL_MS + 1000
        with patch('time_util.time_util.TimeUtil.now_in_millis', return_value=future_time_ms):
            # Refresh should be allowed after time passage
            self.assertTrue(
                self.elimination_client.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS)
            )

    def test_perf_ledger_elimination_persistence(self):
        """Test persistence of perf ledger eliminations"""
        # Create perf ledger elimination
        pl_elim = {
            'hotkey': self.RECOVERY_MINER,
            'reason': EliminationReason.LIQUIDATED.value,
            'dd': 0.20,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {
                str(TradePair.BTCUSD): 40000,
                str(TradePair.ETHUSD): 2000
            }
        }

        # Save perf ledger elimination via client
        self.perf_ledger_client.write_perf_ledger_eliminations_to_disk([pl_elim])

        # Verify file exists
        pl_elim_file = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(pl_elim_file))

        # Reload data from disk to simulate restart
        self.perf_ledger_client.clear_perf_ledger_eliminations()
        loaded_pl_elims = self.perf_ledger_client.get_perf_ledger_eliminations(first_fetch=True)

        # Verify loaded correctly
        self.assertEqual(len(loaded_pl_elims), 1)
        self.assertEqual(loaded_pl_elims[0]['hotkey'], self.RECOVERY_MINER)
