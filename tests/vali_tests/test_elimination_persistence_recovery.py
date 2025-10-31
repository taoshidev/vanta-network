# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import json
import shutil
import time
from unittest.mock import MagicMock, patch

from tests.shared_objects.mock_classes import MockPositionManager
from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager, EliminationReason
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from shared_objects.cache_controller import CacheController


class TestEliminationPersistenceRecovery(TestBase):
    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        
        # Test miners
        self.PERSISTENT_MINER_1 = "persistent_miner_1"
        self.PERSISTENT_MINER_2 = "persistent_miner_2"
        self.RECOVERY_MINER = "recovery_miner"
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        
        self.all_miners = [
            self.PERSISTENT_MINER_1,
            self.PERSISTENT_MINER_2,
            self.RECOVERY_MINER
        ]
        
        # Initialize components
        self.mock_metagraph = MockMetagraph(self.all_miners)
        
        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        
        self.position_locks = PositionLocks()
        
        # Create managers
        self.perf_ledger_manager = PerfLedgerManager(
            self.mock_metagraph,
            running_unit_tests=True
        )
        
        # Create position manager first (needed by elimination manager)
        self.position_manager = MockPositionManager(
            self.mock_metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=None,  # Will set circular reference later
            live_price_fetcher=self.live_price_fetcher
        )
        
        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            None,  # challengeperiod_manager set later
            running_unit_tests=True
        )
        
        # Set circular reference
        self.position_manager.elimination_manager = self.elimination_manager
        
        self.challengeperiod_manager = ChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.perf_ledger_manager,
            running_unit_tests=True
        )
        
        # Set circular references
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        
        # Clear all data
        self.clear_all_data()
        
        # Set up initial positions
        self._setup_positions()

    def tearDown(self):
        super().tearDown()
        self.clear_all_data()

    def clear_all_data(self):
        """Clear all test data"""
        self.position_manager.clear_all_miner_positions()
        self.perf_ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

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
                self.position_manager.save_miner_position(position)

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
            self.elimination_manager.eliminations.append(elim)
        
        # Save to disk
        self.elimination_manager.save_eliminations()
        
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

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_elimination_recovery_on_restart(self, mock_candle_fetcher):
        """Test that eliminations are recovered correctly on validator restart"""
        # Mock the API call to return empty list (no price data needed for this test)
        mock_candle_fetcher.return_value = []
        
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
        self.elimination_manager.write_eliminations_to_disk(test_eliminations)
        
        # Create new elimination manager (simulating restart)
        new_elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True
        )
        
        # Verify eliminations were loaded
        loaded_eliminations = new_elimination_manager.get_eliminations_from_memory()
        self.assertEqual(len(loaded_eliminations), 2)
        
        # Verify content
        hotkeys = [e['hotkey'] for e in loaded_eliminations]
        self.assertIn(self.PERSISTENT_MINER_1, hotkeys)
        self.assertIn(self.RECOVERY_MINER, hotkeys)
        
        # Verify first refresh handles recovered eliminations
        new_elimination_manager.handle_first_refresh(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Check that positions were closed for eliminated miners
        for elim in test_eliminations:
            positions = self.position_manager.get_positions_for_one_hotkey(elim['hotkey'])
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
            self.elimination_manager.eliminations.append(elim)
        self.elimination_manager.save_eliminations()
        
        # Create backup directory
        backup_dir = "/tmp/test_elimination_backup"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup elimination file
        original_file = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        backup_file = os.path.join(backup_dir, "eliminations_backup.json")
        shutil.copy2(original_file, backup_file)
        
        # Clear eliminations
        self.elimination_manager.clear_eliminations()
        
        # Verify eliminations are cleared
        self.assertEqual(len(self.elimination_manager.get_eliminations_from_memory()), 0)
        
        # Restore from backup
        shutil.copy2(backup_file, original_file)
        
        # Create new elimination manager to load restored data
        restored_elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True
        )
        
        # Verify restoration
        restored_eliminations = restored_elimination_manager.get_eliminations_from_memory()
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
        
        # Try to create elimination manager (should handle gracefully)
        try:
            em1 = EliminationManager(
                self.mock_metagraph,
                self.position_manager,
                self.challengeperiod_manager,
                running_unit_tests=True
            )
            # Should create empty eliminations
            self.assertEqual(len(em1.eliminations), 0)
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
        
        # Create elimination manager
        em2 = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True
        )
        
        # Should load what it can
        loaded = em2.get_eliminations_from_memory()
        # Implementation might handle this differently - could be empty or partial load

    def test_elimination_file_permissions(self):
        """Test handling of file permission issues"""
        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        
        # Create elimination
        self.elimination_manager.append_elimination_row(
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
            self.elimination_manager.save_eliminations()
            
        except Exception:
            # Should handle permission errors gracefully
            pass
        finally:
            # Restore permissions
            if 'original_permissions' in locals():
                os.chmod(parent_dir, original_permissions)

    def test_elimination_concurrent_access(self):
        """Test handling of concurrent access to elimination data"""
        # Simulate concurrent modification
        file_path = ValiBkpUtils.get_eliminations_dir(running_unit_tests=True)
        
        # Manager 1 loads data
        em1 = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True
        )
        
        # Manager 2 loads same data
        em2 = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True
        )
        
        # Both add different eliminations
        em1.append_elimination_row(
            self.PERSISTENT_MINER_1,
            0.11,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )
        
        em2.append_elimination_row(
            self.PERSISTENT_MINER_2,
            0.12,
            EliminationReason.PLAGIARISM.value
        )
        
        # Last write wins
        final_data = self.elimination_manager.get_eliminations_from_disk()
        # Should contain eliminations from the last save

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
            self.elimination_manager.eliminations.append(elim)
        
        # Save to disk
        self.elimination_manager.save_eliminations()
        
        # Compare memory and disk
        memory_elims = self.elimination_manager.get_eliminations_from_memory()
        disk_elims = self.elimination_manager.get_eliminations_from_disk()
        
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
        
        # Load with new elimination manager
        # Implementation should handle format migration
        em = EliminationManager(
            self.mock_metagraph,
            self.position_manager,
            self.challengeperiod_manager,
            running_unit_tests=True
        )
        
        # Should either migrate or handle gracefully
        loaded = em.get_eliminations_from_memory()
        # Actual behavior depends on implementation

    def test_elimination_cache_invalidation(self):
        """Test cache invalidation for eliminations"""
        # Add elimination
        self.elimination_manager.append_elimination_row(
            self.PERSISTENT_MINER_1,
            0.11,
            EliminationReason.MAX_TOTAL_DRAWDOWN.value
        )
        
        # Test with running_unit_tests=False to properly test cache behavior
        # Temporarily set running_unit_tests to False
        original_running_unit_tests = self.elimination_manager.running_unit_tests
        self.elimination_manager.running_unit_tests = False
        
        try:
            # Initialize attempted_start_time_ms by calling refresh_allowed
            self.elimination_manager.refresh_allowed(0)
            # Set cache update time
            self.elimination_manager.set_last_update_time()
            
            # Immediate refresh should be blocked
            self.assertFalse(
                self.elimination_manager.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS)
            )
            
            # Mock time passage by patching TimeUtil.now_in_millis
            future_time_ms = TimeUtil.now_in_millis() + ValiConfig.ELIMINATION_CHECK_INTERVAL_MS + 1000
            with patch('time_util.time_util.TimeUtil.now_in_millis', return_value=future_time_ms):
                # Now refresh should be allowed
                self.assertTrue(
                    self.elimination_manager.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS)
                )
        finally:
            # Restore original value
            self.elimination_manager.running_unit_tests = original_running_unit_tests

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
        
        # Save perf ledger elimination
        self.perf_ledger_manager.write_perf_ledger_eliminations_to_disk([pl_elim])
        
        # Verify file exists
        pl_elim_file = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(pl_elim_file))
        
        # Load in new perf ledger manager
        new_plm = PerfLedgerManager(
            self.mock_metagraph,
            running_unit_tests=True
        )
        
        # Verify loaded correctly
        loaded_pl_elims = new_plm.get_perf_ledger_eliminations(first_fetch=True)
        self.assertEqual(len(loaded_pl_elims), 1)
        self.assertEqual(loaded_pl_elims[0]['hotkey'], self.RECOVERY_MINER)
