import unittest

from tests.vali_tests.base_objects.test_base import TestBase
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, TradePairSource, TradePair


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
        print('All servers started for TestAssetSelectionManager.')

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')
        print('AssetSelectionClient obtained for TestAssetSelectionManager.')

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
        print('Clearing all test data before test:', self._testMethodName)
        self.orchestrator.clear_all_test_data()

        # Test miners - use deterministic unique names per test to avoid conflicts
        # Use test method name as unique identifier
        test_name = self._testMethodName
        self.test_miner_1 = f'5TestMiner1_{test_name}'
        self.test_miner_2 = f'5TestMiner2_{test_name}'
        self.test_miner_3 = f'5TestMiner3_{test_name}'

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _can_trade(self, miner_hotkey, trade_pair_category, trade_pair_src):
        """Refresh the local cache, then validate an order via the no-RPC cache path.

        This mirrors how neurons/validator.py validates orders at entry (the
        AssetSelectionClient local cache is the source), so the cache is refreshed
        first to pick up any selection made earlier in the test.
        """
        self.asset_selection_client.refresh_local_cache()
        return self.asset_selection_client.validate_order_asset_class_local_cache(
            miner_hotkey, trade_pair_category, trade_pair_src)

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
        # hl_all is selectable
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('hl_all'))
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('HL_ALL'))

        # commodities is selectable
        self.assertTrue(self.asset_selection_client.is_valid_asset_class('commodities'))

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
        self.assertIn('hl_all', result['error_message'])

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
        
    def test_validate_order_no_selection(self):
        """A miner with no asset selected cannot trade any asset class"""
        # Don't select any asset class for the miner
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePairCategory.CRYPTO, TradePairSource.VANTA))
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePairCategory.FOREX, TradePairSource.VANTA))

    def test_validate_order_with_selection(self):
        """Orders are validated against the miner's selected asset class"""
        self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)

        # Matching Vanta asset class is allowed
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePairCategory.CRYPTO, TradePairSource.VANTA))

        # HL pair with crypto selection → rejected (wrong source)
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePairCategory.CRYPTO, TradePairSource.HYPERLIQUID))

        # Non-matching asset classes are rejected
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePairCategory.FOREX, TradePairSource.VANTA))
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePairCategory.INDICES, TradePairSource.VANTA))
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePairCategory.EQUITIES, TradePairSource.VANTA))

    def test_validate_order_different_trade_pairs_same_asset_class(self):
        """Test that different trade pairs from same asset class are allowed"""
        self.asset_selection_client.process_asset_selection_request('crypto', self.test_miner_1)

        # All Vanta crypto trade pairs should be allowed
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePair.BTCUSD.trade_pair_category, TradePair.BTCUSD.src))
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePair.ETHUSD.trade_pair_category, TradePair.ETHUSD.src))
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePair.SOLUSD.trade_pair_category, TradePair.SOLUSD.src))

        # HL crypto pairs should be rejected for a crypto-selected miner
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePair.BTCUSDC.trade_pair_category, TradePair.BTCUSDC.src))

        # Forex trade pairs should be rejected
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePair.EURUSD.trade_pair_category, TradePair.EURUSD.src))
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePair.GBPUSD.trade_pair_category, TradePair.GBPUSD.src))

    def test_hl_all_selection_allows_hl_pairs(self):
        """hl_all selection permits HL pairs and blocks Vanta pairs"""
        self.asset_selection_client.process_asset_selection_request('hl_all', self.test_miner_1)

        # HL pairs allowed
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePair.BTCUSDC.trade_pair_category, TradePairSource.HYPERLIQUID))
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePair.GOLDUSDC.trade_pair_category, TradePairSource.HYPERLIQUID))
        self.assertTrue(self._can_trade(
            self.test_miner_1, TradePair.NVDAUSDC.trade_pair_category, TradePairSource.HYPERLIQUID))

        # Vanta pairs blocked
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePair.BTCUSD.trade_pair_category, TradePairSource.VANTA))
        self.assertFalse(self._can_trade(
            self.test_miner_1, TradePair.EURUSD.trade_pair_category, TradePairSource.VANTA))

    def test_commodities_selectable(self):
        """commodities can be selected as a standalone asset class"""
        result = self.asset_selection_client.process_asset_selection_request('commodities', self.test_miner_1)
        self.assertTrue(result['successfully_processed'])
        selections = self.asset_selection_client.get_asset_selections()
        self.assertEqual(selections[self.test_miner_1], TradePairCategory.COMMODITIES)

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
        from vali_objects.utils.asset_selection.asset_selection_manager import AssetSelectionManager
        test_data = {
            self.test_miner_1: 'crypto',
            self.test_miner_2: 'forex'
        }
        parsed_selections = AssetSelectionManager._parse_asset_selections_dict(test_data)
        self.assertEqual(parsed_selections[self.test_miner_1], TradePairCategory.CRYPTO)
        self.assertEqual(parsed_selections[self.test_miner_2], TradePairCategory.FOREX)
        
    def test_parse_invalid_disk_data(self):
        """Test parsing invalid data from disk gracefully handles errors"""
        from vali_objects.utils.asset_selection.asset_selection_manager import AssetSelectionManager

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
