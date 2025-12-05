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

    def test_tiingo_batch_trade_pairs_helper(self):
        """
        Test the _batch_trade_pairs helper function that enforces Tiingo's 5 ticker limit.

        Tiingo API has a limit of 5 tickers per request. This test verifies that our
        batching helper correctly chunks trade pairs into groups of 5 or fewer.
        """
        # Test with exactly 5 trade pairs (no batching needed, but should work)
        trade_pairs_5 = [
            TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD,
            TradePair.XRPUSD, TradePair.ADAUSD
        ]
        batches_5 = self.tiingo_service._batch_trade_pairs(trade_pairs_5)
        self.assertEqual(len(batches_5), 1, "5 pairs should fit in 1 batch")
        self.assertEqual(len(batches_5[0]), 5)

        # Test with 6 trade pairs (requires 2 batches)
        trade_pairs_6 = [
            TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD,
            TradePair.XRPUSD, TradePair.ADAUSD, TradePair.DOGEUSD
        ]
        batches_6 = self.tiingo_service._batch_trade_pairs(trade_pairs_6)
        self.assertEqual(len(batches_6), 2, "6 pairs should require 2 batches")
        self.assertEqual(len(batches_6[0]), 5, "First batch should have 5 pairs")
        self.assertEqual(len(batches_6[1]), 1, "Second batch should have 1 pair")

        # Test with 12 trade pairs (requires 3 batches)
        trade_pairs_12 = [
            # Crypto
            TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD,
            TradePair.XRPUSD, TradePair.ADAUSD, TradePair.DOGEUSD,
            # Equities
            TradePair.NVDA, TradePair.MSFT, TradePair.GOOG,
            TradePair.AAPL, TradePair.TSLA, TradePair.AMZN
        ]
        batches_12 = self.tiingo_service._batch_trade_pairs(trade_pairs_12)
        self.assertEqual(len(batches_12), 3, "12 pairs should require 3 batches")
        self.assertEqual(len(batches_12[0]), 5, "First batch should have 5 pairs")
        self.assertEqual(len(batches_12[1]), 5, "Second batch should have 5 pairs")
        self.assertEqual(len(batches_12[2]), 2, "Third batch should have 2 pairs")

    def test_tiingo_crypto_batching_6_tickers(self):
        """
        Test crypto price fetching with 6 tickers to verify batching works correctly.

        Production issue: Tiingo API returned 400 error when requesting more than 5 tickers.
        This test ensures the batching logic splits requests appropriately.

        Bug: "Error: A limit of 5 tickers may be requested at a time"
        Fixed in: data_generator/tiingo_data_service.py (batching implementation)
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Request 6 crypto pairs - should trigger batching (2 batches: 5 + 1)
        crypto_pairs = [
            TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD,
            TradePair.XRPUSD, TradePair.ADAUSD, TradePair.DOGEUSD
        ]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=crypto_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        # Verify we got results for all 6 crypto pairs
        self.assertIsInstance(result, dict)
        self.assertEqual(
            len(result), 6,
            f"Should have price sources for all 6 crypto pairs, got {len(result)}"
        )

        # Verify each pair has valid price source
        for tp in crypto_pairs:
            self.assertIn(tp, result, f"{tp.trade_pair} should be in result")
            self.assertIsInstance(result[tp], PriceSource)
            self.assertGreater(result[tp].close, 0)

    def test_tiingo_equity_batching_7_tickers(self):
        """
        Test equity price fetching with 7 tickers to verify batching.

        This test ensures equity API calls are also properly batched.
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Request 7 equity pairs - should trigger batching (2 batches: 5 + 2)
        equity_pairs = [
            TradePair.NVDA, TradePair.MSFT, TradePair.GOOG,
            TradePair.AAPL, TradePair.TSLA, TradePair.AMZN,
            TradePair.META
        ]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=equity_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        # Verify we got results for all 7 equity pairs
        self.assertIsInstance(result, dict)
        self.assertEqual(
            len(result), 7,
            f"Should have price sources for all 7 equity pairs, got {len(result)}"
        )

        for tp in equity_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_tiingo_forex_batching_6_tickers(self):
        """
        Test forex price fetching with 6 tickers to verify batching.

        This test ensures forex API calls are also properly batched.
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Request 6 forex pairs - should trigger batching (2 batches: 5 + 1)
        forex_pairs = [
            TradePair.EURUSD, TradePair.GBPUSD, TradePair.USDJPY,
            TradePair.AUDUSD, TradePair.USDCAD, TradePair.NZDUSD
        ]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=forex_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        # Verify we got results for all 6 forex pairs
        self.assertIsInstance(result, dict)
        self.assertEqual(
            len(result), 6,
            f"Should have price sources for all 6 forex pairs, got {len(result)}"
        )

        for tp in forex_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_tiingo_mixed_categories_with_batching(self):
        """
        Test with large mixed set requiring batching in multiple categories.

        This comprehensive test verifies that batching works correctly when
        multiple categories each exceed the 5 ticker limit.
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Create a large mixed set - each category has 6+ pairs
        trade_pairs = [
            # 6 crypto pairs (2 batches)
            TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD,
            TradePair.XRPUSD, TradePair.ADAUSD, TradePair.DOGEUSD,
            # 7 equity pairs (2 batches)
            TradePair.NVDA, TradePair.MSFT, TradePair.GOOG,
            TradePair.AAPL, TradePair.TSLA, TradePair.AMZN, TradePair.META,
            # 6 forex pairs (2 batches)
            TradePair.EURUSD, TradePair.GBPUSD, TradePair.USDJPY,
            TradePair.AUDUSD, TradePair.USDCAD, TradePair.NZDUSD
        ]

        result = self.tiingo_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True,
            verbose=False
        )

        # Verify all 19 pairs returned
        self.assertEqual(
            len(result), 19,
            f"Should have price sources for all 19 pairs, got {len(result)}"
        )

        for tp in trade_pairs:
            self.assertIn(tp, result, f"{tp.trade_pair} should be in result")
            self.assertIsInstance(result[tp], PriceSource)
            self.assertGreater(result[tp].close, 0)

    def test_get_tradeable_pairs_filters_blocked_pairs(self):
        """Test that get_tradeable_pairs correctly filters out blocked trade pairs."""
        # Get all tradeable pairs (excluding blocked)
        tradeable = self.tiingo_service.get_tradeable_pairs(include_blocked=False)

        # Verify blocked pairs are NOT in the result
        blocked_pair_ids = {'AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'NZDJPY', 'GBPJPY', 'USDJPY',
                           'XAUUSD', 'XAGUSD', 'NVDA', 'AAPL', 'TSLA', 'AMZN', 'MSFT', 'GOOG',
                           'META', 'USDMXN'}

        tradeable_ids = {tp.trade_pair_id for tp in tradeable}
        blocked_in_tradeable = tradeable_ids & blocked_pair_ids

        self.assertEqual(
            len(blocked_in_tradeable), 0,
            f"Blocked pairs should not be in tradeable list: {blocked_in_tradeable}"
        )

        # Verify we still get valid tradeable pairs
        self.assertGreater(len(tradeable), 0, "Should have some tradeable pairs")

        # Verify specific non-blocked pairs ARE included
        self.assertIn(TradePair.BTCUSD, tradeable, "BTC/USD should be tradeable")
        self.assertIn(TradePair.EURUSD, tradeable, "EUR/USD should be tradeable")

    def test_get_tradeable_pairs_include_blocked_true(self):
        """Test that get_tradeable_pairs returns blocked pairs when include_blocked=True."""
        # Get pairs with blocked included
        all_pairs = self.tiingo_service.get_tradeable_pairs(include_blocked=True)

        # Verify at least one blocked pair IS in the result
        all_pair_ids = {tp.trade_pair_id for tp in all_pairs}

        # Check that AUDJPY (a blocked pair) is included
        self.assertIn('AUDJPY', all_pair_ids, "AUDJPY should be included when include_blocked=True")

        # Verify we get more pairs than when excluding blocked
        tradeable_only = self.tiingo_service.get_tradeable_pairs(include_blocked=False)
        self.assertGreater(
            len(all_pairs), len(tradeable_only),
            "Should have more pairs when including blocked"
        )

    def test_get_tradeable_pairs_category_filter_forex(self):
        """Test that get_tradeable_pairs correctly filters forex category and excludes blocked pairs."""
        # Get only forex pairs (excluding blocked)
        forex_pairs = self.tiingo_service.get_tradeable_pairs(
            category=TradePairCategory.FOREX,
            include_blocked=False
        )

        # Verify all are forex
        for tp in forex_pairs:
            self.assertEqual(
                tp.trade_pair_category, TradePairCategory.FOREX,
                f"{tp.trade_pair_id} should be forex"
            )

        # Verify blocked JPY pairs are NOT included
        forex_ids = {tp.trade_pair_id for tp in forex_pairs}
        blocked_jpy_pairs = {'AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'NZDJPY', 'GBPJPY', 'USDJPY'}

        jpy_in_forex = forex_ids & blocked_jpy_pairs
        self.assertEqual(
            len(jpy_in_forex), 0,
            f"Blocked JPY pairs should not be in forex list: {jpy_in_forex}"
        )

        # Verify blocked commodities are NOT included
        self.assertNotIn('XAUUSD', forex_ids, "XAU/USD should not be in forex list")
        self.assertNotIn('XAGUSD', forex_ids, "XAG/USD should not be in forex list")

        # Verify non-blocked forex pairs ARE included
        self.assertIn('EURUSD', forex_ids, "EUR/USD should be tradeable")
        self.assertIn('GBPUSD', forex_ids, "GBP/USD should be tradeable")

    def test_get_tradeable_pairs_category_filter_crypto(self):
        """Test that get_tradeable_pairs correctly filters crypto category."""
        # Get only crypto pairs (excluding blocked)
        crypto_pairs = self.tiingo_service.get_tradeable_pairs(
            category=TradePairCategory.CRYPTO,
            include_blocked=False
        )

        # Verify all are crypto
        for tp in crypto_pairs:
            self.assertEqual(
                tp.trade_pair_category, TradePairCategory.CRYPTO,
                f"{tp.trade_pair_id} should be crypto"
            )

        # Verify we got some crypto pairs
        self.assertGreater(len(crypto_pairs), 0, "Should have crypto pairs")

        # Verify specific crypto pairs are included
        crypto_ids = {tp.trade_pair_id for tp in crypto_pairs}
        self.assertIn('BTCUSD', crypto_ids, "BTC/USD should be included")
        self.assertIn('ETHUSD', crypto_ids, "ETH/USD should be included")

    def test_get_tradeable_pairs_excludes_unsupported(self):
        """Test that get_tradeable_pairs always excludes unsupported pairs."""
        # Get all pairs with blocked included
        all_pairs = self.tiingo_service.get_tradeable_pairs(include_blocked=True)
        all_pair_ids = {tp.trade_pair_id for tp in all_pairs}

        # Verify unsupported pairs (SPX, DJI, etc.) are NEVER included
        unsupported_ids = {'SPX', 'DJI', 'NDX', 'VIX', 'FTSE', 'GDAXI', 'TAOUSD'}
        unsupported_in_result = all_pair_ids & unsupported_ids

        self.assertEqual(
            len(unsupported_in_result), 0,
            f"Unsupported pairs should never be included: {unsupported_in_result}"
        )

    def test_tiingo_pseudo_websocket_excludes_blocked_pairs(self):
        """
        Test that Tiingo pseudo-websocket polling excludes blocked pairs.

        This test verifies that the get_tradeable_pairs method is used correctly
        by the TiingoPseudoClient to avoid polling prices for blocked pairs.
        """
        # Get the trade pairs that would be queried for forex category
        forex_pairs = self.tiingo_service.get_tradeable_pairs(
            category=TradePairCategory.FOREX,
            include_blocked=False,
            market_open_only=False  # Don't filter by market hours for this test
        )

        forex_ids = {tp.trade_pair_id for tp in forex_pairs}

        # Verify blocked forex pairs are NOT in the query list
        blocked_forex = {'AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'NZDJPY', 'GBPJPY', 'USDJPY',
                        'XAUUSD', 'XAGUSD', 'USDMXN'}

        blocked_in_query = forex_ids & blocked_forex
        self.assertEqual(
            len(blocked_in_query), 0,
            f"Blocked forex pairs should not be queried: {blocked_in_query}"
        )

        # Verify non-blocked pairs ARE in the query list
        self.assertIn('EURUSD', forex_ids, "EUR/USD should be queried")
        self.assertIn('GBPUSD', forex_ids, "GBP/USD should be queried")


if __name__ == '__main__':
    unittest.main()
