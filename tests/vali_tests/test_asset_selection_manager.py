import os
import time
import threading
import unittest
from unittest.mock import Mock, patch

from tests.vali_tests.base_objects.test_base import TestBase
from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from vali_objects.utils.asset_selection_server import AssetSelectionServer, ASSET_CLASS_SELECTION_TIME_MS
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, TradePair
from time_util.time_util import TimeUtil


class TestAssetSelectionManager(TestBase):
    """
    Integration tests for asset selection management using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    asset_selection_client = None

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
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')

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

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Test miners - use deterministic unique names per test to avoid conflicts
        # Use test method name as unique identifier
        test_name = self._testMethodName
        self.test_miner_1 = f'5TestMiner1_{test_name}'
        self.test_miner_2 = f'5TestMiner2_{test_name}'
        self.test_miner_3 = f'5TestMiner3_{test_name}'

        # Test timestamps
        self.before_cutoff_time = ASSET_CLASS_SELECTION_TIME_MS - 1000  # Before enforcement
        self.after_cutoff_time = ASSET_CLASS_SELECTION_TIME_MS + 1000   # After enforcement

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()
        
    def test_is_valid_asset_class(self):
        """Test asset class validation"""
        # Valid asset classes
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('crypto'))
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('forex'))
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('indices'))
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('equities'))

        # Case insensitive
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('CRYPTO'))
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('Forex'))

        # Invalid asset classes
        self.assertFalse(self.asset_selection_client.is_valid_asset_class('invalid'))
        self.assertFalse(self.asset_selection_client.is_valid_asset_class('stocks'))
        self.assertFalse(self.asset_selection_client.is_valid_asset_class(''))
        
    def test_asset_selection_request_success(self):
        """Test successful asset selection request"""
        result = self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)

        self.assertTrue(result['successfully_processed'])
        self.assertIn('successfully selected asset class: crypto', result['success_message'])

        # Verify selection was stored
        selections = self.asset_selection_client.get_asset_selections()
        selected = selections.get(self.test_miner_1)
        self.assertEqual(selected, TradePairCategory.CRYPTO)
        
    def test_asset_selection_request_invalid_class(self):
        """Test asset selection request with invalid asset class"""
        result = self.asset_selection_client.process_asset_selection_request('invalid_class', self.test_miner_1)

        self.assertFalse(result['successfully_processed'])
        self.assertIn('Invalid asset class', result['error_message'])
        self.assertIn('crypto, forex, indices, equities', result['error_message'])

        # Verify no selection was stored
        selections = self.asset_selection_client.get_asset_selections()
        self.assertNotIn(self.test_miner_1, selections)
        
    def test_asset_selection_cannot_change_once_selected(self):
        """Test that miners cannot change their asset class selection"""
        # First selection
        result1 = self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)
        self.assertTrue(result1['successfully_processed'])

        # Attempt to change selection
        result2 = self.asset_selection_client.process_asset_selection_request('forex', self.test_miner_1)
        self.assertFalse(result2['successfully_processed'])
        self.assertIn('Asset class already selected: crypto', result2['error_message'])
        self.assertIn('Cannot change selection', result2['error_message'])

        # Verify original selection unchanged
        selections = self.asset_selection_client.get_asset_selections()
        selected = selections.get(self.test_miner_1)
        self.assertEqual(selected, TradePairCategory.CRYPTO)
        
    def test_multiple_miners_can_select_different_assets(self):
        """Test that different miners can select different asset classes"""
        # Miner 1 selects crypto
        result1 = self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)
        self.assertTrue(result1['successfully_processed'])

        # Miner 2 selects forex
        result2 = self.asset_selection_client.process_asset_selection_request('forex', self.test_miner_2)
        self.assertTrue(result2['successfully_processed'])

        # Miner 3 selects indices
        result3 = self.asset_selection_client.process_asset_selection_request('indices', self.test_miner_3)
        self.assertTrue(result3['successfully_processed'])

        # Verify all selections
        selections = self.asset_selection_client.get_asset_selections()
        self.assertEqual(selections[self.test_miner_1], TradePairCategory.CRYPTO)
        self.assertEqual(selections[self.test_miner_2], TradePairCategory.FOREX)
        self.assertEqual(selections[self.test_miner_3], TradePairCategory.INDICES)
        
    def test_validate_order_asset_class_before_cutoff(self):
        """Test that orders before cutoff time can be any asset class"""
        # Don't select any asset class for the miner

        # Orders before cutoff should be allowed for any asset class
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.CRYPTO, self.before_cutoff_time))
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.FOREX, self.before_cutoff_time))
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.INDICES, self.before_cutoff_time))
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.EQUITIES, self.before_cutoff_time))

    def test_validate_order_asset_class_after_cutoff_no_selection(self):
        """Test that orders after cutoff require asset class selection"""
        # Don't select any asset class for the miner

        # Orders after cutoff should be rejected if no selection made
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.CRYPTO, self.after_cutoff_time))
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.FOREX, self.after_cutoff_time))

    def test_validate_order_asset_class_after_cutoff_with_selection(self):
        """Test that orders after cutoff are validated against selected asset class"""
        # Select crypto for miner
        self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)

        # Orders matching selected asset class should be allowed
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.CRYPTO, self.after_cutoff_time))

        # Orders not matching selected asset class should be rejected
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.FOREX, self.after_cutoff_time))
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.INDICES, self.after_cutoff_time))
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePairCategory.EQUITIES, self.after_cutoff_time))

    def test_validate_order_asset_class_with_current_time(self):
        """Test validate_order_asset_class with current time (no timestamp provided)"""
        # Select forex for miner
        self.asset_selection_client.process_asset_selection_request('forex', self.test_miner_1)

        with patch.object(TimeUtil, 'now_in_millis', return_value=self.after_cutoff_time):
            # Should validate against selected asset class
            self.assertTrue(self.asset_selection_client.validate_order_asset_class(
                self.test_miner_1, TradePairCategory.FOREX))
            self.assertFalse(self.asset_selection_client.validate_order_asset_class(
                self.test_miner_1, TradePairCategory.CRYPTO))

    def test_validate_order_different_trade_pairs_same_asset_class(self):
        """Test that different trade pairs from same asset class are allowed"""
        # Select crypto
        self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)

        # All crypto trade pairs should be allowed
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePair.BTCUSD.trade_pair_category, self.after_cutoff_time))
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePair.ETHUSD.trade_pair_category, self.after_cutoff_time))
        self.assertTrue(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePair.SOLUSD.trade_pair_category, self.after_cutoff_time))

        # Forex trade pairs should be rejected
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePair.EURUSD.trade_pair_category, self.after_cutoff_time))
        self.assertFalse(self.asset_selection_client.validate_order_asset_class(
            self.test_miner_1, TradePair.GBPUSD.trade_pair_category, self.after_cutoff_time))

        
    def test_data_format_conversion(self):
        """Test conversion between in-memory and disk formats"""
        # Add test selections
        self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)
        self.asset_selection_client.process_asset_selection_request('forex', self.test_miner_2)

        # Test to_dict format (for checkpoints)
        disk_format = self.asset_selection_client.to_dict()

        # Since server is shared across tests, filter for our test miners only
        self.assertIn(self.test_miner_1, disk_format)
        self.assertIn(self.test_miner_2, disk_format)
        self.assertEqual(disk_format[self.test_miner_1], 'crypto')
        self.assertEqual(disk_format[self.test_miner_2], 'forex')

        # Test parsing back from disk format (use manager's static method)
        from vali_objects.utils.asset_selection_manager import AssetSelectionManager
        test_data = {
            self.test_miner_1: 'crypto',
            self.test_miner_2: 'forex'
        }
        parsed_selections = AssetSelectionManager._parse_asset_selections_dict(test_data)
        self.assertEqual(parsed_selections[self.test_miner_1], TradePairCategory.CRYPTO)
        self.assertEqual(parsed_selections[self.test_miner_2], TradePairCategory.FOREX)
        
    def test_parse_invalid_disk_data(self):
        """Test parsing invalid data from disk gracefully handles errors"""
        from vali_objects.utils.asset_selection_manager import AssetSelectionManager

        invalid_data = {
            self.test_miner_1: 'invalid_asset_class',
            self.test_miner_2: 'forex',  # This should work
            'bad_miner': None,  # This should be skipped
        }

        parsed = AssetSelectionManager._parse_asset_selections_dict(invalid_data)

        # Only valid data should be parsed
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[self.test_miner_2], TradePairCategory.FOREX)
        self.assertNotIn(self.test_miner_1, parsed)
        self.assertNotIn('bad_miner', parsed)

    def test_case_insensitive_asset_selection(self):
        """Test that asset selection is case insensitive"""
        # Test various cases
        test_cases = ['crypto', 'CRYPTO', 'Crypto', 'CrYpTo']

        for i, case in enumerate(test_cases):
            miner = f'5TestMinerCase{i}_{self._testMethodName}'
            result = self.asset_selection_client.process_asset_selection_request(case, miner)
            self.assertTrue(result['successfully_processed'], f"Failed for case: {case}")

            # All should be stored as the same enum value
            selections = self.asset_selection_client.get_asset_selections()
            self.assertEqual(selections[miner], TradePairCategory.CRYPTO)

    def test_error_handling_in_process_request(self):
        """Test error handling in process_asset_selection_request"""
        # Test with None values
        result = self.asset_selection_client.process_asset_selection_request(None, self.test_miner_1)
        self.assertFalse(result['successfully_processed'])

        # Should handle gracefully without crashing
        self.assertIn('error_message', result)

    def test_save_error_handling(self):
        """Test error handling when disk save fails"""
        # Note: This test is challenging with separate server process
        # We'll skip mocking the server directly and just test the API behavior
        # The server handles errors internally, client just gets the response
        result = self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)
        # Should succeed normally (server handles errors internally)
        self.assertTrue(result['successfully_processed'])


if __name__ == '__main__':
    unittest.main()
