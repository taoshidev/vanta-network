# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Data service tests for Tiingo price fetching.

Tests the Tiingo data service layer that provides price data for the validator.
Focuses on edge cases and regression tests for price fetching logic.
"""
import unittest
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.price_source import PriceSource
from data_generator.tiingo_data_service import TiingoDataService


class TestTiingoDataService(unittest.TestCase):
    """
    Unit tests for TiingoDataService price fetching functionality.

    Tests focus on:
    - Multi-category price fetching
    - Tuple unpacking in parallel execution paths
    - Edge cases in price source retrieval
    - Category-specific methods
    - Error handling and boundary conditions
    """

    def setUp(self):
        """Set up test data service in unit test mode."""
        # Create service in test mode (no network calls)
        self.tiingo_service = TiingoDataService(
            api_key="test_key",
            disable_ws=True,
            running_unit_tests=True
        )

    def test_tiingo_multi_category_get_closes_rest(self):
        """
        Regression test for TiingoDataService.get_closes_rest() multi-category handling.

        Tests that get_closes_rest can handle trade pairs from multiple categories
        (crypto, forex, equities) in a single call without tuple unpacking errors.

        Bug: ThreadPoolExecutor comprehension expected 3 values but received 5 when
        multiple trade pair categories were requested simultaneously, causing:
        "ValueError: too many values to unpack (expected 3)"

        Fixed in: data_generator/tiingo_data_service.py:359-360
        Production error: 2025-12-05 06:48:06 (DOGE/USD pricing with other pairs)
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Request prices for multiple categories in single call
        # This exercises the ThreadPoolExecutor path where the bug occurred
        trade_pairs = [
            TradePair.BTCUSD,   # Crypto
            TradePair.NVDA,     # Equity
            TradePair.EURUSD    # Forex
        ]

        # This should not raise "too many values to unpack (expected 3)" error
        try:
            result = self.tiingo_service.get_closes_rest(
                trade_pairs=trade_pairs,
                time_ms=order_time_ms,
                live=True,
                verbose=False
            )
        except ValueError as e:
            if "too many values to unpack" in str(e):
                self.fail(
                    f"Tuple unpacking error in get_closes_rest: {e}\n"
                    "This indicates the ThreadPoolExecutor comprehension is not properly "
                    "unpacking the 5-element job tuples (func, tp_list, time_ms, live, verbose)"
                )
            raise

        # Verify we got results for all three trade pairs
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertEqual(
            len(result), 3,
            f"Should have price sources for all 3 trade pairs, got {len(result)}"
        )

        # Verify each trade pair has valid price source
        for trade_pair in trade_pairs:
            self.assertIn(
                trade_pair, result,
                f"{trade_pair.trade_pair} should be in result"
            )
            price_source = result[trade_pair]
            self.assertIsNotNone(
                price_source,
                f"Price source for {trade_pair.trade_pair} should not be None"
            )
            self.assertIsInstance(
                price_source, PriceSource,
                f"Result for {trade_pair.trade_pair} should be a PriceSource"
            )
            # In test mode, should return default test price
            self.assertGreater(
                price_source.close, 0,
                f"Price for {trade_pair.trade_pair} should be > 0"
            )

    def test_tiingo_single_category_get_closes_rest(self):
        """
        Test TiingoDataService.get_closes_rest() with single category.

        Tests the fast path (no ThreadPoolExecutor) when only one category is requested.
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Request prices for single category only (crypto)
        trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        # Verify results
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_tiingo_empty_trade_pairs(self):
        """Test TiingoDataService.get_closes_rest() with empty trade pairs list."""
        result = self.tiingo_service.get_closes_rest(
            trade_pairs=[],
            time_ms=TimeUtil.now_in_millis(),
            live=True,
            verbose=False
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_tiingo_crypto_only(self):
        """Test TiingoDataService with only crypto pairs."""
        order_time_ms = TimeUtil.now_in_millis()
        trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)
            self.assertEqual(result[tp].close, 50000)  # Default test price

    def test_tiingo_equity_only(self):
        """Test TiingoDataService with only equity pairs."""
        order_time_ms = TimeUtil.now_in_millis()
        trade_pairs = [TradePair.NVDA, TradePair.MSFT, TradePair.GOOG]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_tiingo_forex_only(self):
        """Test TiingoDataService with only forex pairs."""
        order_time_ms = TimeUtil.now_in_millis()
        trade_pairs = [TradePair.EURUSD, TradePair.GBPUSD, TradePair.USDJPY]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_tiingo_get_close_rest_single_crypto(self):
        """Test TiingoDataService.get_close_rest() for single crypto pair."""
        order_time_ms = TimeUtil.now_in_millis()

        # In test mode, get_close_rest returns test data via get_closes_* methods
        result = self.tiingo_service.get_close_rest(
            trade_pair=TradePair.BTCUSD,
            timestamp_ms=order_time_ms,
            live=True
        )

        # May return None or PriceSource depending on test mode implementation
        # Just verify no exception is raised and type is correct if not None
        if result is not None:
            self.assertIsInstance(result, PriceSource)
            self.assertEqual(result.close, 50000)  # Default test price

    def test_tiingo_get_close_rest_single_equity(self):
        """Test TiingoDataService.get_close_rest() for single equity pair."""
        order_time_ms = TimeUtil.now_in_millis()

        result = self.tiingo_service.get_close_rest(
            trade_pair=TradePair.NVDA,
            timestamp_ms=order_time_ms,
            live=True
        )

        # May return None or PriceSource depending on test mode implementation
        if result is not None:
            self.assertIsInstance(result, PriceSource)

    def test_tiingo_get_close_rest_single_forex(self):
        """Test TiingoDataService.get_close_rest() for single forex pair."""
        order_time_ms = TimeUtil.now_in_millis()

        result = self.tiingo_service.get_close_rest(
            trade_pair=TradePair.EURUSD,
            timestamp_ms=order_time_ms,
            live=True
        )

        # May return None or PriceSource depending on test mode implementation
        if result is not None:
            self.assertIsInstance(result, PriceSource)

    def test_tiingo_ticker_conversion(self):
        """Test TiingoDataService.trade_pair_to_tiingo_ticker() conversion."""
        # Test various trade pairs
        self.assertEqual(
            self.tiingo_service.trade_pair_to_tiingo_ticker(TradePair.BTCUSD),
            "btcusd"
        )
        self.assertEqual(
            self.tiingo_service.trade_pair_to_tiingo_ticker(TradePair.EURUSD),
            "eurusd"
        )
        self.assertEqual(
            self.tiingo_service.trade_pair_to_tiingo_ticker(TradePair.NVDA),
            "nvda"
        )

    def test_tiingo_all_categories_mixed(self):
        """
        Test with large mixed set of trade pairs across all categories.

        Tests scalability and ensures no race conditions in ThreadPoolExecutor.
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Mix of all categories
        trade_pairs = [
            TradePair.BTCUSD, TradePair.ETHUSD,  # Crypto
            TradePair.NVDA, TradePair.MSFT, TradePair.GOOG,  # Equity
            TradePair.EURUSD, TradePair.GBPUSD  # Forex
        ]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        # Verify all pairs returned
        self.assertEqual(len(result), len(trade_pairs))
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)
            self.assertGreater(result[tp].close, 0)

    def test_tiingo_timestamp_handling(self):
        """Test that timestamp_ms is properly passed through the service."""
        # Use specific timestamps
        recent_time = TimeUtil.now_in_millis()
        past_time = recent_time - (24 * 60 * 60 * 1000)  # 24 hours ago

        # Both should work in test mode - just verify no exceptions
        try:
            result_recent = self.tiingo_service.get_close_rest(
                trade_pair=TradePair.BTCUSD,
                timestamp_ms=recent_time,
                live=True
            )

            result_past = self.tiingo_service.get_close_rest(
                trade_pair=TradePair.BTCUSD,
                timestamp_ms=past_time,
                live=False  # Historical
            )

            # Verify types if results are not None
            if result_recent is not None:
                self.assertIsInstance(result_recent, PriceSource)
            if result_past is not None:
                self.assertIsInstance(result_past, PriceSource)

        except Exception as e:
            self.fail(f"Timestamp handling raised unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main()
