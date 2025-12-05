# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
PolygonDataService tests.

Tests the Polygon data service layer that provides price data for the validator.
Focuses on:
- Price source injection and retrieval
- Multi-category price fetching
- Test data management
- Quote fetching
- Candle data
"""
import unittest
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.price_source import PriceSource
from data_generator.polygon_data_service import PolygonDataService


class TestPolygonDataService(unittest.TestCase):
    """
    Unit tests for PolygonDataService functionality.

    Tests focus on:
    - Test price source injection and retrieval
    - Multi-category price fetching with ThreadPoolExecutor
    - Quote (bid/ask) retrieval
    - Candle data fetching
    - Edge cases and error handling
    """

    def setUp(self):
        """Set up test data service in unit test mode."""
        self.polygon_service = PolygonDataService(
            api_key="test_key",
            disable_ws=True,
            running_unit_tests=True
        )

    def test_polygon_set_and_get_test_price_source(self):
        """Test setting and retrieving test price sources."""
        order_time_ms = TimeUtil.now_in_millis()
        test_price = 65000.0

        # Create and inject test price source
        price_source = PriceSource(
            source='test',
            timespan_ms=1000,
            open=test_price,
            close=test_price,
            vwap=test_price,
            high=test_price,
            low=test_price,
            start_ms=order_time_ms,
            websocket=False,
            lag_ms=0,
            bid=test_price - 10,
            ask=test_price + 10
        )

        self.polygon_service.set_test_price_source(TradePair.BTCUSD, price_source)

        # Retrieve via get_close_rest
        result = self.polygon_service.get_close_rest(
            trade_pair=TradePair.BTCUSD,
            timestamp_ms=order_time_ms
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.close, test_price)
        self.assertEqual(result.bid, test_price - 10)
        self.assertEqual(result.ask, test_price + 10)

    def test_polygon_clear_test_price_sources(self):
        """Test clearing test price sources."""
        order_time_ms = TimeUtil.now_in_millis()
        test_price = 65000.0

        # Inject test price
        price_source = PriceSource(
            source='test',
            timespan_ms=1000,
            open=test_price,
            close=test_price,
            vwap=test_price,
            high=test_price,
            low=test_price,
            start_ms=order_time_ms,
            websocket=False,
            lag_ms=0
        )

        self.polygon_service.set_test_price_source(TradePair.BTCUSD, price_source)

        # Clear and verify
        self.polygon_service.clear_test_price_sources()

        # Should now return default fallback
        result = self.polygon_service.get_close_rest(
            trade_pair=TradePair.BTCUSD,
            timestamp_ms=order_time_ms
        )

        self.assertIsNotNone(result)
        # Should be default fallback price (50000)
        self.assertEqual(result.close, 50000)

    def test_polygon_multi_category_get_closes_rest(self):
        """
        Test PolygonDataService.get_closes_rest() with multiple categories.

        Ensures ThreadPoolExecutor handles multiple trade pair categories correctly.
        """
        order_time_ms = TimeUtil.now_in_millis()

        # Request prices for multiple categories
        trade_pairs = [
            TradePair.BTCUSD,   # Crypto
            TradePair.NVDA,     # Equity
            TradePair.EURUSD    # Forex
        ]

        result = self.polygon_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True
        )

        # Verify results
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)

        for trade_pair in trade_pairs:
            self.assertIn(trade_pair, result)
            price_source = result[trade_pair]
            self.assertIsNotNone(price_source)
            self.assertIsInstance(price_source, PriceSource)
            self.assertGreater(price_source.close, 0)

    def test_polygon_single_category_crypto(self):
        """Test Polygon with crypto pairs only."""
        order_time_ms = TimeUtil.now_in_millis()
        trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD]

        result = self.polygon_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True
        )

        self.assertEqual(len(result), 2)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_polygon_single_category_equity(self):
        """Test Polygon with equity pairs only."""
        order_time_ms = TimeUtil.now_in_millis()
        trade_pairs = [TradePair.NVDA, TradePair.MSFT]

        result = self.polygon_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True
        )

        self.assertEqual(len(result), 2)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_polygon_single_category_forex(self):
        """Test Polygon with forex pairs only."""
        order_time_ms = TimeUtil.now_in_millis()
        trade_pairs = [TradePair.EURUSD, TradePair.GBPUSD]

        result = self.polygon_service.get_closes_rest(
            trade_pairs=trade_pairs,
            time_ms=order_time_ms,
            live=True
        )

        self.assertEqual(len(result), 2)
        for tp in trade_pairs:
            self.assertIn(tp, result)
            self.assertIsInstance(result[tp], PriceSource)

    def test_polygon_get_close_rest_single_pair(self):
        """Test get_close_rest for single trade pair."""
        order_time_ms = TimeUtil.now_in_millis()

        result = self.polygon_service.get_close_rest(
            trade_pair=TradePair.BTCUSD,
            timestamp_ms=order_time_ms
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, PriceSource)
        self.assertGreater(result.close, 0)

    def test_polygon_empty_trade_pairs(self):
        """Test with empty trade pairs list."""
        result = self.polygon_service.get_closes_rest(
            trade_pairs=[],
            time_ms=TimeUtil.now_in_millis(),
            live=True
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_polygon_test_candle_data(self):
        """Test setting and retrieving candle data."""
        start_ms = TimeUtil.now_in_millis()
        end_ms = start_ms + 60000  # 1 minute later

        # Create test candle data
        candles = [
            PriceSource(
                source='test',
                timespan_ms=1000,
                open=65000.0,
                close=65100.0,
                vwap=65050.0,
                high=65200.0,
                low=64900.0,
                start_ms=start_ms,
                websocket=False,
                lag_ms=0
            )
        ]

        # Set test candle data
        self.polygon_service.set_test_candle_data(
            trade_pair=TradePair.BTCUSD,
            start_ms=start_ms,
            end_ms=end_ms,
            candles=candles
        )

        # Retrieve via unified_candle_fetcher
        result = self.polygon_service.unified_candle_fetcher(
            trade_pair=TradePair.BTCUSD,
            start_timestamp_ms=start_ms,
            end_timestamp_ms=end_ms,
            timespan='second'
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_polygon_clear_test_candle_data(self):
        """Test clearing test candle data."""
        start_ms = TimeUtil.now_in_millis()
        end_ms = start_ms + 60000

        candles = [
            PriceSource(
                source='test',
                timespan_ms=1000,
                open=65000.0,
                close=65100.0,
                vwap=65050.0,
                high=65200.0,
                low=64900.0,
                start_ms=start_ms,
                websocket=False,
                lag_ms=0
            )
        ]

        self.polygon_service.set_test_candle_data(
            TradePair.BTCUSD, start_ms, end_ms, candles
        )

        # Clear
        self.polygon_service.clear_test_candle_data()

        # Should return empty list in test mode after clearing
        result = self.polygon_service.unified_candle_fetcher(
            trade_pair=TradePair.BTCUSD,
            start_timestamp_ms=start_ms,
            end_timestamp_ms=end_ms,
            timespan='second'
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_polygon_default_test_fallback(self):
        """Test that default test fallback is used when no specific price is set."""
        order_time_ms = TimeUtil.now_in_millis()

        # Don't set any specific price - should use default fallback
        result = self.polygon_service.get_close_rest(
            trade_pair=TradePair.BTCUSD,
            timestamp_ms=order_time_ms
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, PriceSource)
        # Default fallback price is 50000
        self.assertEqual(result.close, 50000)

    def test_polygon_explicit_none_price_source(self):
        """Test setting explicit None for a trade pair (no price available)."""
        order_time_ms = TimeUtil.now_in_millis()

        # Explicitly set None (no price source for this pair)
        self.polygon_service.set_test_price_source(TradePair.BTCUSD, None)

        result = self.polygon_service.get_close_rest(
            trade_pair=TradePair.BTCUSD,
            timestamp_ms=order_time_ms
        )

        # Should return None (explicitly no price)
        self.assertIsNone(result)

    def test_polygon_multiple_pairs_different_prices(self):
        """Test multiple trade pairs with different injected prices."""
        order_time_ms = TimeUtil.now_in_millis()

        # Set different prices for different pairs
        btc_price = PriceSource(
            source='test', timespan_ms=1000, open=65000.0, close=65000.0,
            vwap=65000.0, high=65000.0, low=65000.0, start_ms=order_time_ms,
            websocket=False, lag_ms=0
        )
        eth_price = PriceSource(
            source='test', timespan_ms=1000, open=3200.0, close=3200.0,
            vwap=3200.0, high=3200.0, low=3200.0, start_ms=order_time_ms,
            websocket=False, lag_ms=0
        )

        self.polygon_service.set_test_price_source(TradePair.BTCUSD, btc_price)
        self.polygon_service.set_test_price_source(TradePair.ETHUSD, eth_price)

        # Retrieve both
        btc_result = self.polygon_service.get_close_rest(TradePair.BTCUSD, order_time_ms)
        eth_result = self.polygon_service.get_close_rest(TradePair.ETHUSD, order_time_ms)

        self.assertEqual(btc_result.close, 65000.0)
        self.assertEqual(eth_result.close, 3200.0)

    def test_polygon_set_test_price_source_requires_test_mode(self):
        """Test that set_test_price_source raises error in production mode."""
        # Create service in production mode (running_unit_tests=False)
        prod_service = PolygonDataService(
            api_key="test_key",
            disable_ws=True,
            running_unit_tests=False
        )

        price_source = PriceSource(
            source='test', timespan_ms=1000, open=65000.0, close=65000.0,
            vwap=65000.0, high=65000.0, low=65000.0, start_ms=TimeUtil.now_in_millis(),
            websocket=False, lag_ms=0
        )

        with self.assertRaises(RuntimeError) as context:
            prod_service.set_test_price_source(TradePair.BTCUSD, price_source)

        self.assertIn("can only be used in unit test mode", str(context.exception))

    def test_polygon_clear_test_price_sources_requires_test_mode(self):
        """Test that clear_test_price_sources raises error in production mode."""
        # Create service in production mode (running_unit_tests=False)
        prod_service = PolygonDataService(
            api_key="test_key",
            disable_ws=True,
            running_unit_tests=False
        )

        with self.assertRaises(RuntimeError) as context:
            prod_service.clear_test_price_sources()

        self.assertIn("can only be used in unit test mode", str(context.exception))

    def test_polygon_set_test_candle_data_requires_test_mode(self):
        """Test that set_test_candle_data raises error in production mode."""
        # Create service in production mode (running_unit_tests=False)
        prod_service = PolygonDataService(
            api_key="test_key",
            disable_ws=True,
            running_unit_tests=False
        )

        start_ms = TimeUtil.now_in_millis()
        end_ms = start_ms + 60000

        candles = [
            PriceSource(
                source='test', timespan_ms=1000, open=65000.0, close=65100.0,
                vwap=65050.0, high=65200.0, low=64900.0, start_ms=start_ms,
                websocket=False, lag_ms=0
            )
        ]

        with self.assertRaises(RuntimeError) as context:
            prod_service.set_test_candle_data(TradePair.BTCUSD, start_ms, end_ms, candles)

        self.assertIn("can only be used in unit test mode", str(context.exception))

    def test_polygon_clear_test_candle_data_requires_test_mode(self):
        """Test that clear_test_candle_data raises error in production mode."""
        # Create service in production mode (running_unit_tests=False)
        prod_service = PolygonDataService(
            api_key="test_key",
            disable_ws=True,
            running_unit_tests=False
        )

        with self.assertRaises(RuntimeError) as context:
            prod_service.clear_test_candle_data()

        self.assertIn("can only be used in unit test mode", str(context.exception))


if __name__ == '__main__':
    unittest.main()
