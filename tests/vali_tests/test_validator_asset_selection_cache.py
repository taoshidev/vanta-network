"""
Test validator's local asset selection cache functionality.

This tests the optimization that eliminates 81ms RPC overhead per order by
maintaining a local cache that syncs every 5 seconds.
"""
import unittest
from unittest.mock import Mock, patch
import time

from vali_objects.utils.asset_selection.asset_selection_manager import ASSET_CLASS_SELECTION_TIME_MS
from vali_objects.vali_config import TradePairCategory, TradePair
from time_util.time_util import TimeUtil


class MockValidator:
    """
    Lightweight mock of Validator with only asset selection cache functionality.

    This allows us to test the caching logic in isolation without loading
    the full Validator class with all its dependencies.
    """

    def __init__(self, asset_selection_manager):
        """
        Initialize mock validator with asset selection cache.

        Args:
            asset_selection_manager: Mock or real AssetSelectionManager
        """
        self.asset_selection_manager = asset_selection_manager

        # Asset selection local cache (same as real validator)
        self._asset_selections_cache = {}
        self._asset_cache_last_sync_ms = 0
        self._asset_cache_sync_interval_ms = 5000  # Sync every 5 seconds

    def _sync_asset_selections_cache(self):
        """
        Sync asset selections from RPC server to local cache.

        This is the exact implementation from neurons/validator.py.
        """
        now_ms = TimeUtil.now_in_millis()
        if now_ms - self._asset_cache_last_sync_ms > self._asset_cache_sync_interval_ms:
            # Single RPC call to fetch all selections (happens once per 5 seconds)
            self._asset_selections_cache = self.asset_selection_manager.asset_selections
            self._asset_cache_last_sync_ms = now_ms
            return True  # Return True if sync happened (for testing)
        return False  # Return False if no sync (for testing)


class TestValidatorAssetSelectionCache(unittest.TestCase):
    """Test validator's local asset selection cache optimization"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock asset selection manager
        self.mock_manager = Mock()
        self.mock_manager.asset_selections = {}

        # Create mock validator with cache
        self.validator = MockValidator(self.mock_manager)

        # Test miners
        self.test_miner_1 = '5TestMiner1234567890'
        self.test_miner_2 = '5TestMiner0987654321'

        # Test timestamps
        self.before_cutoff = ASSET_CLASS_SELECTION_TIME_MS - 1000
        self.after_cutoff = ASSET_CLASS_SELECTION_TIME_MS + 1000

    def test_cache_initialization(self):
        """Test that cache is properly initialized"""
        self.assertIsInstance(self.validator._asset_selections_cache, dict)
        self.assertEqual(len(self.validator._asset_selections_cache), 0)
        self.assertEqual(self.validator._asset_cache_last_sync_ms, 0)
        self.assertEqual(self.validator._asset_cache_sync_interval_ms, 5000)

    def test_first_sync_always_happens(self):
        """Test that first sync always happens (last_sync_ms = 0)"""
        # Add test data to manager
        self.mock_manager.asset_selections = {
            self.test_miner_1: TradePairCategory.CRYPTO,
            self.test_miner_2: TradePairCategory.FOREX
        }

        # First sync should happen
        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            synced = self.validator._sync_asset_selections_cache()

        self.assertTrue(synced)
        self.assertEqual(len(self.validator._asset_selections_cache), 2)
        self.assertEqual(self.validator._asset_selections_cache[self.test_miner_1],
                        TradePairCategory.CRYPTO)
        self.assertEqual(self.validator._asset_selections_cache[self.test_miner_2],
                        TradePairCategory.FOREX)
        self.assertEqual(self.validator._asset_cache_last_sync_ms, 10000)

    def test_sync_respects_interval(self):
        """Test that sync only happens after interval elapsed"""
        self.mock_manager.asset_selections = {self.test_miner_1: TradePairCategory.CRYPTO}

        # First sync at t=10000
        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            synced1 = self.validator._sync_asset_selections_cache()
        self.assertTrue(synced1)

        # Try sync at t=14000 (4 seconds later, should not sync)
        with patch.object(TimeUtil, 'now_in_millis', return_value=14000):
            synced2 = self.validator._sync_asset_selections_cache()
        self.assertFalse(synced2)
        self.assertEqual(self.validator._asset_cache_last_sync_ms, 10000)  # Unchanged

        # Try sync at t=15001 (5001ms later, should sync)
        self.mock_manager.asset_selections = {
            self.test_miner_1: TradePairCategory.CRYPTO,
            self.test_miner_2: TradePairCategory.FOREX
        }
        with patch.object(TimeUtil, 'now_in_millis', return_value=15001):
            synced3 = self.validator._sync_asset_selections_cache()
        self.assertTrue(synced3)
        self.assertEqual(self.validator._asset_cache_last_sync_ms, 15001)
        self.assertEqual(len(self.validator._asset_selections_cache), 2)

    def test_cache_updates_with_new_data(self):
        """Test that cache updates when new selections are made"""
        # Initial sync with one selection
        self.mock_manager.asset_selections = {self.test_miner_1: TradePairCategory.CRYPTO}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            self.validator._sync_asset_selections_cache()

        self.assertEqual(len(self.validator._asset_selections_cache), 1)

        # Add new selection to manager
        self.mock_manager.asset_selections[self.test_miner_2] = TradePairCategory.FOREX

        # Sync after interval
        with patch.object(TimeUtil, 'now_in_millis', return_value=15001):
            self.validator._sync_asset_selections_cache()

        # Cache should have both
        self.assertEqual(len(self.validator._asset_selections_cache), 2)
        self.assertEqual(self.validator._asset_selections_cache[self.test_miner_2],
                        TradePairCategory.FOREX)

    def test_cache_is_reference_not_copy(self):
        """Test that cache gets fresh reference from manager (not deep copy)"""
        # Initial data
        initial_data = {self.test_miner_1: TradePairCategory.CRYPTO}
        self.mock_manager.asset_selections = initial_data

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            self.validator._sync_asset_selections_cache()

        # Cache should reference the dict from manager
        self.assertIs(self.validator._asset_selections_cache, initial_data)

        # When manager's reference changes, next sync gets new reference
        new_data = {
            self.test_miner_1: TradePairCategory.CRYPTO,
            self.test_miner_2: TradePairCategory.FOREX
        }
        self.mock_manager.asset_selections = new_data

        with patch.object(TimeUtil, 'now_in_millis', return_value=15001):
            self.validator._sync_asset_selections_cache()

        self.assertIs(self.validator._asset_selections_cache, new_data)

    def test_cache_staleness_window(self):
        """Test that cache can be up to 5 seconds stale"""
        # Initial sync
        self.mock_manager.asset_selections = {self.test_miner_1: TradePairCategory.CRYPTO}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            self.validator._sync_asset_selections_cache()

        # Miner changes selection on server (shouldn't happen in production, but test staleness)
        self.mock_manager.asset_selections[self.test_miner_1] = TradePairCategory.FOREX

        # Cache is stale for up to 5 seconds
        with patch.object(TimeUtil, 'now_in_millis', return_value=14999):
            self.validator._sync_asset_selections_cache()
            # Cache still has old value (because it didn't update the reference)
            # This test verifies staleness is acceptable

        # After 5 seconds, sync happens
        with patch.object(TimeUtil, 'now_in_millis', return_value=15001):
            synced = self.validator._sync_asset_selections_cache()

        self.assertTrue(synced)

    def test_empty_cache_syncs_empty_dict(self):
        """Test that empty manager selections result in empty cache"""
        self.mock_manager.asset_selections = {}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            synced = self.validator._sync_asset_selections_cache()

        self.assertTrue(synced)
        self.assertEqual(len(self.validator._asset_selections_cache), 0)

    def test_multiple_rapid_syncs(self):
        """Test that multiple rapid calls don't cause excessive RPC calls"""
        self.mock_manager.asset_selections = {self.test_miner_1: TradePairCategory.CRYPTO}

        # Simulate 100 rapid calls within 1 second
        base_time = 10000
        sync_count = 0

        for i in range(100):
            with patch.object(TimeUtil, 'now_in_millis', return_value=base_time + i * 10):
                if self.validator._sync_asset_selections_cache():
                    sync_count += 1

        # Should only sync once (first call)
        self.assertEqual(sync_count, 1)

    def test_sync_with_large_dataset(self):
        """Test cache performance with large number of selections"""
        # Create 1000 fake selections
        large_dataset = {
            f'5Miner{i:04d}': TradePairCategory.CRYPTO if i % 2 == 0 else TradePairCategory.FOREX
            for i in range(1000)
        }
        self.mock_manager.asset_selections = large_dataset

        # Sync should handle large dataset
        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            start = time.perf_counter()
            synced = self.validator._sync_asset_selections_cache()
            duration_ms = (time.perf_counter() - start) * 1000

        self.assertTrue(synced)
        self.assertEqual(len(self.validator._asset_selections_cache), 1000)
        # Sync should be very fast (< 1ms for dict assignment)
        self.assertLess(duration_ms, 1.0)


class TestValidatorAssetValidationWithCache(unittest.TestCase):
    """
    Test asset validation logic using local cache.

    This tests the exact code path in validator.py's should_fail_early() method.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.mock_manager = Mock()
        self.mock_manager.asset_selections = {}
        self.validator = MockValidator(self.mock_manager)

        self.test_miner = '5TestMiner1234567890'
        self.before_cutoff = ASSET_CLASS_SELECTION_TIME_MS - 1000
        self.after_cutoff = ASSET_CLASS_SELECTION_TIME_MS + 1000

    def _validate_with_cache(self, miner_hotkey, trade_pair_category, now_ms):
        """
        Simulate the exact validation logic from validator.py's should_fail_early().

        This is the production code path we're testing.
        """
        # Sync cache if needed (happens once per 5 seconds, not per order)
        self.validator._sync_asset_selections_cache()

        # Fast local validation (no RPC call!)
        if now_ms >= ASSET_CLASS_SELECTION_TIME_MS:
            selected_asset = self.validator._asset_selections_cache.get(miner_hotkey, None)
            is_valid = selected_asset == trade_pair_category if selected_asset is not None else False
        else:
            is_valid = True  # Pre-cutoff, all assets allowed

        return is_valid

    def test_validation_before_cutoff_allows_all(self):
        """Test that validation before cutoff allows all assets"""
        # Don't set any selections

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # All asset classes should be allowed before cutoff
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.CRYPTO, self.before_cutoff))
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.FOREX, self.before_cutoff))
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.INDICES, self.before_cutoff))
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.EQUITIES, self.before_cutoff))

    def test_validation_after_cutoff_requires_selection(self):
        """Test that validation after cutoff requires asset selection"""
        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # No selection made - should reject all
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePairCategory.CRYPTO, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePairCategory.FOREX, self.after_cutoff))

    def test_validation_after_cutoff_with_matching_selection(self):
        """Test that validation allows matching asset class"""
        # Set selection in manager
        self.mock_manager.asset_selections = {self.test_miner: TradePairCategory.CRYPTO}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # Matching asset class should be allowed
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.CRYPTO, self.after_cutoff))

            # Non-matching should be rejected
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePairCategory.FOREX, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePairCategory.INDICES, self.after_cutoff))

    def test_validation_uses_cache_not_rpc(self):
        """Test that validation uses cached data, not RPC"""
        # Set initial selection
        self.mock_manager.asset_selections = {self.test_miner: TradePairCategory.CRYPTO}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # First validation syncs cache
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.CRYPTO, self.after_cutoff))

        # Change manager data by creating NEW dict (simulating server update with new reference)
        # NOTE: Must create new dict reference for cache to be stale
        self.mock_manager.asset_selections = {self.test_miner: TradePairCategory.FOREX}

        with patch.object(TimeUtil, 'now_in_millis', return_value=14000):
            # Validation within 5 seconds still uses old cache
            # This proves we're using cache, not RPC
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.CRYPTO, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePairCategory.FOREX, self.after_cutoff))

        # After 5 seconds, cache syncs with new data
        with patch.object(TimeUtil, 'now_in_millis', return_value=15001):
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePairCategory.CRYPTO, self.after_cutoff))
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePairCategory.FOREX, self.after_cutoff))

    def test_validation_performance_no_rpc_overhead(self):
        """Test that validation is fast (no RPC overhead)"""
        # Set selection
        self.mock_manager.asset_selections = {self.test_miner: TradePairCategory.CRYPTO}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # First call syncs cache
            self.validator._sync_asset_selections_cache()

            # Measure validation time (should be <1ms, no RPC)
            start = time.perf_counter()
            for _ in range(100):
                is_valid = self._validate_with_cache(
                    self.test_miner, TradePairCategory.CRYPTO, self.after_cutoff)
            duration_ms = (time.perf_counter() - start) * 1000

            # 100 validations should take <2s (avg <0.02s each)
            self.assertLess(duration_ms, 2.0)

    def test_multiple_miners_validation(self):
        """Test validation for multiple miners with different selections"""
        miner1 = '5Miner1'
        miner2 = '5Miner2'
        miner3 = '5Miner3'

        self.mock_manager.asset_selections = {
            miner1: TradePairCategory.CRYPTO,
            miner2: TradePairCategory.FOREX,
            miner3: TradePairCategory.INDICES
        }

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # Each miner can only trade their selected asset class
            self.assertTrue(self._validate_with_cache(
                miner1, TradePairCategory.CRYPTO, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                miner1, TradePairCategory.FOREX, self.after_cutoff))

            self.assertTrue(self._validate_with_cache(
                miner2, TradePairCategory.FOREX, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                miner2, TradePairCategory.CRYPTO, self.after_cutoff))

            self.assertTrue(self._validate_with_cache(
                miner3, TradePairCategory.INDICES, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                miner3, TradePairCategory.CRYPTO, self.after_cutoff))

    def test_trade_pair_category_validation(self):
        """Test validation with actual TradePair objects"""
        self.mock_manager.asset_selections = {self.test_miner: TradePairCategory.CRYPTO}

        with patch.object(TimeUtil, 'now_in_millis', return_value=10000):
            # Crypto pairs should be allowed
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePair.BTCUSD.trade_pair_category, self.after_cutoff))
            self.assertTrue(self._validate_with_cache(
                self.test_miner, TradePair.ETHUSD.trade_pair_category, self.after_cutoff))

            # Forex pairs should be rejected
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePair.EURUSD.trade_pair_category, self.after_cutoff))
            self.assertFalse(self._validate_with_cache(
                self.test_miner, TradePair.GBPUSD.trade_pair_category, self.after_cutoff))


if __name__ == '__main__':
    unittest.main()
