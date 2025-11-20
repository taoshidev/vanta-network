import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from collateral_sdk import Network

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.validator_contract_manager import ValidatorContractManager, CollateralRecord
from vali_objects.vali_config import ValiConfig


class TestValidatorContractManager(TestBase):
    def setUp(self):
        super().setUp()
        
        # Test miners
        self.MINER_1 = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        self.MINER_2 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        self.DAY_MS = 1000 * 60 * 60 * 24
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the CollateralManager to avoid actual contract calls
        with patch('vali_objects.utils.validator_contract_manager.CollateralManager') as mock_collateral_manager:
            self.mock_collateral_manager_instance = MagicMock()
            mock_collateral_manager.return_value = self.mock_collateral_manager_instance
            
            # Create a mock config and metagraph
            mock_config = MagicMock()
            mock_config.subtensor.network = "test"
            mock_metagraph = MagicMock()
            
            # Initialize ValidatorContractManager with test setup
            with patch('vali_objects.utils.validator_contract_manager.ValidatorContractManager._save_miner_account_sizes_to_disk'):
                self.contract_manager = ValidatorContractManager(
                    config=mock_config,
                    running_unit_tests=True
                )
        
        # Clear any existing data to ensure test isolation
        self.contract_manager.miner_account_sizes.clear()
        
        # Set up mock collateral balances
        self.mock_balances = {
            self.MINER_1: 1000000,  # 1M theta
            self.MINER_2: 500000    # 500K theta
        }
        
        self.mock_collateral_manager_instance.balance_of.side_effect = lambda hotkey: self.mock_balances.get(hotkey, 0)
    
    def tearDown(self):
        super().tearDown()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collateral_record_creation(self):
        """Test CollateralRecord creation and properties"""
        timestamp_ms = int(time.time() * 1000)
        account_size = 10000.0
        account_size_theta = 10000.0 / ValiConfig.COST_PER_THETA
        
        record = CollateralRecord(account_size, account_size_theta, timestamp_ms)
        
        self.assertEqual(record.account_size, account_size)
        self.assertEqual(record.update_time_ms, timestamp_ms)
        self.assertIsInstance(record.valid_date_timestamp, int)
        self.assertIsInstance(record.valid_date_str, str)
        
        # Test date string format
        self.assertRegex(record.valid_date_str, r'^\d{4}-\d{2}-\d{2}$')

    def test_set_and_get_miner_account_size(self):
        """Test setting and getting miner account sizes"""
        current_time = int(time.time() * 1000)
        day_after_current_time = self.DAY_MS + current_time
        
        # Initially should return None for non-existent miner
        self.assertIsNone(self.contract_manager.get_miner_account_size(self.MINER_1))
        
        # Mock the collateral balance and set account size (ValidatorContractManager calculates account size from collateral)
        with patch.object(self.contract_manager, '_save_miner_account_sizes_to_disk'):
            self.mock_collateral_manager_instance.balance_of.return_value = 1000000  # 1M rao
            self.contract_manager.set_miner_account_size(self.MINER_1, current_time)
            
            # Verify retrieval - should return the calculated account size
            account_size = self.contract_manager.get_miner_account_size(self.MINER_1, day_after_current_time)
            self.assertIsNotNone(account_size)
            
            # Set for second miner
            self.mock_collateral_manager_instance.balance_of.return_value = 500000  # 500K rao
            self.contract_manager.set_miner_account_size(self.MINER_2, current_time)
            account_size_2 = self.contract_manager.get_miner_account_size(self.MINER_2, day_after_current_time)
            self.assertIsNotNone(account_size_2)

    def test_account_size_persistence(self):
        """Test that account sizes are saved to and loaded from disk"""
        current_time = int(time.time() * 1000)
        day_after_current_time = self.DAY_MS + current_time
        
        # Mock collateral balance and set account size
        with patch.object(self.contract_manager, '_save_miner_account_sizes_to_disk'):
            self.mock_collateral_manager_instance.balance_of.return_value = 1000000  # 1M rao
            self.contract_manager.set_miner_account_size(self.MINER_1, current_time)
            
            # Verify it was set
            account_size = self.contract_manager.get_miner_account_size(self.MINER_1, day_after_current_time)
            self.assertIsNotNone(account_size)
            
            # Test the disk persistence by checking the internal data structure
            self.assertIn(self.MINER_1, self.contract_manager.miner_account_sizes)
            self.assertEqual(len(self.contract_manager.miner_account_sizes[self.MINER_1]), 1)

    def test_multiple_account_size_records(self):
        """Test that multiple records are stored and sorted correctly"""
        base_time = int(time.time() * 1000)
        
        # Mock collateral balance for consistent account size calculation
        with patch.object(self.contract_manager, '_save_miner_account_sizes_to_disk'):
            self.mock_collateral_manager_instance.balance_of.side_effect = [
                1_000_000,  # First call
                2_000_000,  # Second call
                3_000_000,  # Third call
            ]
            
            # Add multiple records with different timestamps
            self.contract_manager.set_miner_account_size(self.MINER_1, base_time)
            self.contract_manager.set_miner_account_size(self.MINER_1, base_time + 1000)
            self.contract_manager.set_miner_account_size(self.MINER_1, base_time + 2000)
            
            # Verify records are stored
            records = self.contract_manager.miner_account_sizes[self.MINER_1]
            self.assertEqual(len(records), 3)
            
            # Verify records are sorted by update_time_ms
            for i in range(1, len(records)):
                self.assertGreaterEqual(records[i].update_time_ms, records[i-1].update_time_ms)

    def test_no_duplicate_account_size_records(self):
        """Test that duplicate records are ignored"""
        base_time = int(time.time() * 1000)

        # Mock collateral balance for consistent account size calculation
        with patch.object(self.contract_manager, '_save_miner_account_sizes_to_disk'):
            self.mock_collateral_manager_instance.balance_of.return_value = 1000000  # 1M rao

            # Add multiple records with different timestamps
            self.contract_manager.set_miner_account_size(self.MINER_1, base_time)
            self.contract_manager.set_miner_account_size(self.MINER_1, base_time + 1000)
            self.contract_manager.set_miner_account_size(self.MINER_1, base_time + 2000)

            # Verify records are stored
            records = self.contract_manager.miner_account_sizes[self.MINER_1]
            self.assertEqual(len(records), 1)

    def test_sync_miner_account_sizes_data(self):
        """Test syncing miner account sizes from external data"""
        # Create test data in the format expected by sync method
        test_data = {
            self.MINER_1: [
                {
                    "account_size": 15000.0,
                    "account_size_theta": 1000.0,
                    "update_time_ms": int(time.time() * 1000) - 1000,
                    "valid_date_timestamp": CollateralRecord.valid_from_ms(int(time.time() * 1000) - 1000)
                }
            ],
            self.MINER_2: [
                {
                    "account_size": 25000.0,
                    "account_size_theta": 2000.0,
                    "update_time_ms": int(time.time() * 1000),
                    "valid_date_timestamp": CollateralRecord.valid_from_ms(int(time.time() * 1000))
                }
            ]
        }
        
        # Sync the data
        self.contract_manager.sync_miner_account_sizes_data(test_data)
        
        # Verify data was synced correctly
        self.assertIn(self.MINER_1, self.contract_manager.miner_account_sizes)
        self.assertIn(self.MINER_2, self.contract_manager.miner_account_sizes)
        
        # Check the records
        miner1_records = self.contract_manager.miner_account_sizes[self.MINER_1]
        miner2_records = self.contract_manager.miner_account_sizes[self.MINER_2]
        
        self.assertEqual(len(miner1_records), 1)
        self.assertEqual(len(miner2_records), 1)
        self.assertEqual(miner1_records[0].account_size, 15000.0)
        self.assertEqual(miner2_records[0].account_size, 25000.0)

    def test_to_checkpoint_dict(self):
        """Test converting account sizes to checkpoint dictionary format"""
        current_time = int(time.time() * 1000)
        
        # Mock collateral balance and set account sizes
        with patch.object(self.contract_manager, '_save_miner_account_sizes_to_disk'):
            self.mock_collateral_manager_instance.balance_of.return_value = 1000000  # 1M rao
            self.contract_manager.set_miner_account_size(self.MINER_1, current_time)
            
            self.mock_collateral_manager_instance.balance_of.return_value = 500000  # 500K rao  
            self.contract_manager.set_miner_account_size(self.MINER_2, current_time)
            
            # Get checkpoint dict
            checkpoint_dict = self.contract_manager.miner_account_sizes_dict()
            
            # Verify structure
            self.assertIsInstance(checkpoint_dict, dict)
            self.assertIn(self.MINER_1, checkpoint_dict)
            self.assertIn(self.MINER_2, checkpoint_dict)
            
            # Verify record structure
            for hotkey, records in checkpoint_dict.items():
                self.assertIsInstance(records, list)
                for record in records:
                    self.assertIn('account_size', record)
                    self.assertIn('update_time_ms', record)
                    self.assertIn('valid_date_timestamp', record)

    def test_collateral_balance_retrieval(self):
        """Test getting collateral balance for miners"""
        # Mock different balances
        self.mock_collateral_manager_instance.balance_of.return_value = 1500000  # 1.5M rao

        # Get balance
        balance = self.contract_manager.get_miner_collateral_balance(self.MINER_1)
        self.assertIsNotNone(balance)

        # Verify the mock was called
        self.mock_collateral_manager_instance.balance_of.assert_called_with(self.MINER_1)

    def test_compute_slash_amount_formula_accuracy(self):
        """Test the exact formula for slash calculation across range of drawdowns"""
        balance_theta = 1000.0
        self.contract_manager.get_miner_collateral_balance = MagicMock(return_value=balance_theta)

        # Test cases: (drawdown, expected_drawdown_percentage, expected_slash_proportion)
        test_cases = [
            (1.0, 0.0, 0.0),      # 0% drawdown -> 0% slash
            (0.99, 1.0, 0.1),     # 1% drawdown -> 10% slash
            (0.98, 2.0, 0.2),     # 2% drawdown -> 20% slash
            (0.97, 3.0, 0.3),     # 3% drawdown -> 30% slash
            (0.96, 4.0, 0.4),     # 4% drawdown -> 40% slash
            (0.95, 5.0, 0.5),     # 5% drawdown -> 50% slash
            (0.94, 6.0, 0.6),     # 6% drawdown -> 60% slash
            (0.93, 7.0, 0.7),     # 7% drawdown -> 70% slash
            (0.92, 8.0, 0.8),     # 8% drawdown -> 80% slash
            (0.91, 9.0, 0.9),     # 9% drawdown -> 90% slash
            (0.90, 10.0, 1.0),    # 10% drawdown -> 100% slash (elimination)
        ]

        for drawdown, dd_pct, expected_proportion in test_cases:
            with self.subTest(drawdown_pct=dd_pct):
                slash_amount = self.contract_manager.compute_slash_amount(self.MINER_1, drawdown=drawdown)
                expected_slash = balance_theta * expected_proportion

                self.assertAlmostEqual(slash_amount, expected_slash, places=1,
                                       msg=f"{dd_pct}% drawdown should slash {expected_proportion*100}% of balance. "
                                           f"Expected {expected_slash}, got {slash_amount}")
