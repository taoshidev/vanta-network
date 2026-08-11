import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock

import requests

from data_generator.hyperliquid_data_service import HyperliquidDataService, HYPERLIQUID_PROVIDER_NAME
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory


class TestHyperliquidDataService(unittest.TestCase):

    def setUp(self):
        self.service = HyperliquidDataService(
            disable_ws=True,
            running_unit_tests=True
        )

    def _make_l2book_msg(self, coin="BTC", bid="30000.0", ask="30001.0", time_ms=None):
        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        return json.dumps({
            "channel": "l2Book",
            "data": {
                "coin": coin,
                "time": time_ms,
                "levels": [
                    [{"px": bid, "sz": "1.5", "n": 5}, {"px": str(float(bid) - 1), "sz": "2.0", "n": 3}],
                    [{"px": ask, "sz": "0.8", "n": 3}, {"px": str(float(ask) + 1), "sz": "1.2", "n": 2}]
                ]
            }
        })

    # -- Coin mapping tests --

    def test_coin_mapping_contains_all_crypto(self):
        expected_coins = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA",
                          "TAO", "HYPE", "ZEC", "BCH", "LINK", "XMR", "LTC"}
        actual_coins = set(self.service._coin_to_trade_pair.keys())
        self.assertEqual(expected_coins, actual_coins)

    def test_coin_mapping_excludes_blocked(self):
        for tp in TradePair:
            if tp.is_blocked and tp.is_crypto:
                self.assertNotIn(tp.base, self.service._coin_to_trade_pair)

    def test_coin_mapping_values(self):
        self.assertEqual(self.service._coin_to_trade_pair["BTC"], TradePair.BTCUSD)
        self.assertEqual(self.service._coin_to_trade_pair["ETH"], TradePair.ETHUSD)
        self.assertEqual(self.service._coin_to_trade_pair["SOL"], TradePair.SOLUSD)

    # -- Enabled categories --

    def test_enabled_categories_crypto_only(self):
        self.assertEqual(self.service.enabled_websocket_categories, {TradePairCategory.CRYPTO})

    # -- handle_msg_full tests (price feed + fine orderbook) --

    def test_handle_msg_full_valid_l2book(self):
        now_ms = TimeUtil.now_in_millis()
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30002.0", time_ms=now_ms)

        asyncio.run(self.service.handle_msg_full(msg))

        symbol = TradePair.BTCUSD.trade_pair  # "BTC/USD"
        self.assertIn(symbol, self.service.latest_websocket_events)

        ps = self.service.latest_websocket_events[symbol]
        self.assertEqual(ps.bid, 30000.0)
        self.assertEqual(ps.ask, 30002.0)
        self.assertEqual(ps.close, 30001.0)  # mid price
        self.assertEqual(ps.source, f"{HYPERLIQUID_PROVIDER_NAME}_ws")
        self.assertTrue(ps.websocket)
        self.assertEqual(ps.timespan_ms, 0)

    def test_handle_msg_full_updates_fine_orderbook(self):
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30002.0")
        asyncio.run(self.service.handle_msg_full(msg))
        self.assertIn("BTC", self.service._orderbooks_full)
        self.assertNotIn("BTC", self.service._orderbooks_coarse)

    def test_handle_msg_coarse_updates_coarse_orderbook_only(self):
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30002.0")
        asyncio.run(self.service.handle_msg_coarse(msg))
        self.assertIn("BTC", self.service._orderbooks_coarse)
        self.assertNotIn("BTC", self.service._orderbooks_full)
        # coarse handler must NOT update the price feed
        self.assertNotIn(TradePair.BTCUSD.trade_pair, self.service.latest_websocket_events)

    def test_handle_msg_full_stores_in_recent_events(self):
        msg = self._make_l2book_msg(coin="ETH", bid="2000.0", ask="2001.0")
        asyncio.run(self.service.handle_msg_full(msg))

        symbol = TradePair.ETHUSD.trade_pair
        self.assertIn(symbol, self.service.trade_pair_to_recent_events)

        tracker = self.service.trade_pair_to_recent_events[symbol]
        self.assertTrue(len(tracker.events) > 0)

    def test_handle_msg_full_ignores_non_l2book(self):
        # Subscription confirmation message
        msg = json.dumps({"channel": "subscriptionResponse", "data": {"method": "subscribe"}})
        asyncio.run(self.service.handle_msg_full(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_full_ignores_unknown_coin(self):
        msg = self._make_l2book_msg(coin="UNKNOWN", bid="100.0", ask="101.0")
        asyncio.run(self.service.handle_msg_full(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_full_handles_empty_levels(self):
        msg = json.dumps({
            "channel": "l2Book",
            "data": {"coin": "BTC", "time": TimeUtil.now_in_millis(), "levels": []}
        })
        asyncio.run(self.service.handle_msg_full(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_full_handles_empty_bids(self):
        msg = json.dumps({
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": TimeUtil.now_in_millis(),
                "levels": [[], [{"px": "30001.0", "sz": "0.8", "n": 3}]]
            }
        })
        asyncio.run(self.service.handle_msg_full(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_full_increments_event_counter(self):
        initial_count = self.service.tpc_to_n_events[TradePairCategory.CRYPTO]
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30001.0")
        asyncio.run(self.service.handle_msg_full(msg))
        self.assertEqual(self.service.tpc_to_n_events[TradePairCategory.CRYPTO], initial_count + 1)

    # -- get_closes_websocket tests --

    def test_get_closes_websocket_returns_injected_data(self):
        now_ms = TimeUtil.now_in_millis()
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30002.0", time_ms=now_ms)
        asyncio.run(self.service.handle_msg_full(msg))

        results = self.service.get_closes_websocket([TradePair.BTCUSD], now_ms)
        self.assertIn(TradePair.BTCUSD, results)
        self.assertEqual(results[TradePair.BTCUSD].close, 30001.0)

    def test_get_closes_websocket_empty_for_no_data(self):
        results = self.service.get_closes_websocket([TradePair.BTCUSD], TimeUtil.now_in_millis())
        self.assertNotIn(TradePair.BTCUSD, results)

    def test_multiple_coins(self):
        now_ms = TimeUtil.now_in_millis()
        for coin, bid, ask in [("BTC", "30000", "30002"), ("ETH", "2000", "2001"), ("SOL", "100", "100.5")]:
            msg = self._make_l2book_msg(coin=coin, bid=bid, ask=ask, time_ms=now_ms)
            asyncio.run(self.service.handle_msg_full(msg))

        results = self.service.get_closes_websocket(
            [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD], now_ms
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[TradePair.BTCUSD].close, 30001.0)
        self.assertEqual(results[TradePair.ETHUSD].close, 2000.5)
        self.assertEqual(results[TradePair.SOLUSD].close, 100.25)


    # -- simulate_slippage tests --

    def _inject_books(self, coin, fine_bids, fine_asks, coarse_bids, coarse_asks):
        """Directly populate both orderbook caches."""
        if fine_bids is not None:
            self.service._orderbooks_full[coin] = {"bids": fine_bids, "asks": fine_asks}
        if coarse_bids is not None:
            self.service._orderbooks_coarse[coin] = {"bids": coarse_bids, "asks": coarse_asks}

    def test_simulate_slippage_returns_none_with_no_data(self):
        self.assertIsNone(self.service.simulate_slippage(TradePair.BTCUSD, 1000.0, True))

    def test_simulate_slippage_falls_back_to_coarse_only(self):
        # No fine book; coarse only
        self._inject_books(
            "BTC",
            fine_bids=None, fine_asks=None,
            coarse_bids=[{"px": "29999.0", "sz": "1.0"}, {"px": "29998.0", "sz": "2.0"}],
            coarse_asks=[{"px": "30001.0", "sz": "1.0"}, {"px": "30002.0", "sz": "2.0"}],
        )
        result = self.service.simulate_slippage(TradePair.BTCUSD, 1000.0, True)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.0)

    def test_simulate_slippage_uses_fine_book_for_small_order(self):
        # Order fits entirely in fine book
        self._inject_books(
            "BTC",
            fine_bids=[{"px": "29999.0", "sz": "10.0"}],
            fine_asks=[{"px": "30001.0", "sz": "10.0"}],
            coarse_bids=[{"px": "29990.0", "sz": "100.0"}],
            coarse_asks=[{"px": "30010.0", "sz": "100.0"}],
        )
        # Buy $100 — fits in fine ask level at 30001, mid = 30000
        result = self.service.simulate_slippage(TradePair.BTCUSD, 100.0, True)
        self.assertIsNotNone(result)
        # slippage = (30001 - 30000) / 30000 ≈ 0.0000333
        self.assertAlmostEqual(result, 1.0 / 30000.0, places=6)

    def test_simulate_slippage_extends_into_coarse_for_large_order(self):
        # Fine book has 1 BTC at ask 30001 ($30001 total capacity)
        # Large order of $60000 must spill into coarse
        self._inject_books(
            "BTC",
            fine_bids=[{"px": "29999.0", "sz": "1.0"}],
            fine_asks=[{"px": "30001.0", "sz": "1.0"}],
            coarse_bids=[{"px": "29990.0", "sz": "100.0"}],
            coarse_asks=[{"px": "30002.0", "sz": "100.0"}, {"px": "30010.0", "sz": "100.0"}],
        )
        result_small = self.service.simulate_slippage(TradePair.BTCUSD, 1000.0, True)
        result_large = self.service.simulate_slippage(TradePair.BTCUSD, 60000.0, True)
        # Larger order should have equal or greater slippage
        self.assertGreaterEqual(result_large, result_small)

    def test_simulate_slippage_buy_vs_sell_symmetry(self):
        # Symmetric book: spread is 2 units
        self._inject_books(
            "BTC",
            fine_bids=[{"px": "29999.0", "sz": "10.0"}],
            fine_asks=[{"px": "30001.0", "sz": "10.0"}],
            coarse_bids=None, coarse_asks=None,
        )
        buy_slip = self.service.simulate_slippage(TradePair.BTCUSD, 100.0, True)
        sell_slip = self.service.simulate_slippage(TradePair.BTCUSD, 100.0, False)
        self.assertIsNotNone(buy_slip)
        self.assertIsNotNone(sell_slip)
        self.assertAlmostEqual(buy_slip, sell_slip, places=8)

    # -- REST fallback tests --

    def test_get_price_rest_unit_test_mode(self):
        """In unit test mode, get_price_rest returns default fallback price sources."""
        results = self.service.get_price_rest([TradePair.BTCUSD, TradePair.ETHUSD], TimeUtil.now_in_millis())
        self.assertEqual(len(results), 2)
        self.assertIn(TradePair.BTCUSD, results)
        self.assertIn(TradePair.ETHUSD, results)

    def test_get_price_rest_ignores_non_crypto(self):
        """Non-crypto pairs should be ignored."""
        results = self.service.get_price_rest([TradePair.EURUSD], TimeUtil.now_in_millis())
        # EURUSD is forex, not crypto — should not be in results (unit test mode returns for all passed pairs,
        # but the method filters to crypto only before the unit test shortcut)
        # Since running_unit_tests returns early for all trade_pairs before filtering, let's
        # test with a non-unit-test service instead
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        with patch("data_generator.hyperliquid_data_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
            mock_post.return_value.raise_for_status = MagicMock()
            results = svc.get_price_rest([TradePair.EURUSD], TimeUtil.now_in_millis())
        self.assertEqual(len(results), 0)

    def test_get_price_rest_single_pair(self):
        """get_price_rest should return a PriceSource for a single pair."""
        result = self.service.get_price_rest(TradePair.BTCUSD, TimeUtil.now_in_millis())
        self.assertIsNotNone(result)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_get_price_rest_uses_all_mids(self, mock_post):
        """REST fallback should use allMids endpoint and produce correct PriceSources."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"BTC": "67500.5", "ETH": "3400.25", "SOL": "145.0"}
        mock_post.return_value = mock_response

        results = svc.get_price_rest(
            [TradePair.BTCUSD, TradePair.ETHUSD], TimeUtil.now_in_millis()
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[TradePair.BTCUSD].close, 67500.5)
        self.assertAlmostEqual(results[TradePair.ETHUSD].close, 3400.25)
        self.assertEqual(results[TradePair.BTCUSD].source, f"{HYPERLIQUID_PROVIDER_NAME}_rest")
        self.assertFalse(results[TradePair.BTCUSD].websocket)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_get_price_rest_falls_back_to_l2book(self, mock_post):
        """If allMids is missing a coin, should fall back to l2Book for that coin."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)

        def side_effect(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if json and json.get("type") == "allMids":
                # BTC missing from allMids
                resp.json.return_value = {"ETH": "3400.0"}
            elif json and json.get("type") == "l2Book":
                resp.json.return_value = {
                    "levels": [
                        [{"px": "67000.0", "sz": "1.0", "n": 1}],
                        [{"px": "67002.0", "sz": "1.0", "n": 1}],
                    ]
                }
            return resp

        mock_post.side_effect = side_effect

        results = svc.get_price_rest(
            [TradePair.BTCUSD, TradePair.ETHUSD], TimeUtil.now_in_millis()
        )

        self.assertEqual(len(results), 2)
        # BTC came from l2Book
        self.assertAlmostEqual(results[TradePair.BTCUSD].close, 67001.0)
        self.assertEqual(results[TradePair.BTCUSD].bid, 67000.0)
        self.assertEqual(results[TradePair.BTCUSD].ask, 67002.0)
        # ETH came from allMids
        self.assertAlmostEqual(results[TradePair.ETHUSD].close, 3400.0)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_get_price_rest_handles_api_failure(self, mock_post):
        """If the REST API fails entirely, should return empty dict."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        mock_post.side_effect = Exception("Connection refused")

        results = svc.get_price_rest([TradePair.BTCUSD], TimeUtil.now_in_millis())
        self.assertEqual(len(results), 0)

    # -- fetch_candle_range tests --

    def _candle(self, t_ms, close):
        return {"t": t_ms, "T": t_ms, "o": close, "h": close, "l": close, "c": close, "v": "0"}

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_fetch_candle_range_short_window_uses_1m_single_request(self, mock_post):
        """A window well under 5000 minutes should resolve to '1m' and make exactly one request."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        start_ms = 0
        end_ms = 30 * 60 * 1000  # 30 minutes

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = [self._candle(start_ms + i * 60_000, 100.0 + i) for i in range(30)]
        mock_post.return_value = resp

        candles = svc.fetch_candle_range(TradePair.BTCUSDC, start_ms, end_ms)

        self.assertEqual(mock_post.call_count, 1)
        sent_req = mock_post.call_args.kwargs["json"]["req"]
        self.assertEqual(sent_req["interval"], "1m")
        self.assertEqual(len(candles), 30)
        self.assertEqual(candles[0].span_ms, 60_000)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_fetch_candle_range_respects_min_interval_floor(self, mock_post):
        """A short window with a 12h min_interval_span_ms floor should resolve to '12h',
        not '1m', so short live-daemon gaps don't need a fresh HL request every time."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        start_ms = 0
        end_ms = 30 * 60 * 1000  # 30 minutes - would resolve to '1m' with no floor

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = [self._candle(start_ms, 100.0)]
        mock_post.return_value = resp

        twelve_hours_ms = 12 * 60 * 60 * 1000
        candles = svc.fetch_candle_range(TradePair.BTCUSDC, start_ms, end_ms,
                                          min_interval_span_ms=twelve_hours_ms)

        sent_req = mock_post.call_args.kwargs["json"]["req"]
        self.assertEqual(sent_req["interval"], "12h")
        self.assertEqual(candles[0].span_ms, twelve_hours_ms)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_fetch_candle_range_long_window_uses_coarser_interval(self, mock_post):
        """A 34-day window would need ~49000 1m candles; should pick a coarser interval
        that fits in one request instead of exceeding HL's per-request cap."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        interval_span_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
                             "1h": 3_600_000, "12h": 43_200_000, "1d": 86_400_000}
        start_ms = 0
        end_ms = 34 * 24 * 60 * 60 * 1000  # ~34 days

        def side_effect(url, json=None, timeout=None):
            req = json["req"]
            span_ms = interval_span_ms[req["interval"]]
            n = (req["endTime"] - req["startTime"]) // span_ms
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [self._candle(req["startTime"] + i * span_ms, 100.0) for i in range(max(n, 1))]
            return resp

        mock_post.side_effect = side_effect

        candles = svc.fetch_candle_range(TradePair.BTCUSDC, start_ms, end_ms)

        self.assertEqual(mock_post.call_count, 1)
        sent_req = mock_post.call_args.kwargs["json"]["req"]
        self.assertNotEqual(sent_req["interval"], "1m")
        self.assertGreater(len(candles), 0)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_fetch_candle_range_paginates_when_even_coarsest_interval_overflows(self, mock_post):
        """A window so long that even '1d' candles exceed the per-request cap should page
        across multiple sequential requests instead of silently truncating."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        day_ms = 24 * 60 * 60 * 1000
        start_ms = 0
        end_ms = 6000 * day_ms  # 6000 days of '1d' candles > 5000 per-request cap

        call_windows = []

        def side_effect(url, json=None, timeout=None):
            req = json["req"]
            call_windows.append((req["startTime"], req["endTime"]))
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            chunk_start, chunk_end = req["startTime"], req["endTime"]
            n = (chunk_end - chunk_start) // day_ms
            resp.json.return_value = [self._candle(chunk_start + i * day_ms, 100.0) for i in range(max(n, 1))]
            return resp

        mock_post.side_effect = side_effect

        candles = svc.fetch_candle_range(TradePair.BTCUSDC, start_ms, end_ms)

        self.assertGreater(mock_post.call_count, 1)
        self.assertGreater(len(candles), 5000)
        # Chunks should advance forward, not repeat/overlap.
        for (s1, _), (s2, _) in zip(call_windows, call_windows[1:]):
            self.assertGreater(s2, s1)

    @patch("data_generator.hyperliquid_data_service.time.sleep")
    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_fetch_candle_range_retries_on_failure_then_succeeds(self, mock_post, mock_sleep):
        """A transient failure should be retried before giving up."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        start_ms = 0
        end_ms = 10 * 60 * 1000

        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        # Cover the full requested window so the pagination loop stops after this one chunk.
        ok_resp.json.return_value = [self._candle(start_ms + i * 60_000, 100.0) for i in range(10)]

        mock_post.side_effect = [requests.exceptions.ConnectionError("timeout"),
                                  requests.exceptions.ConnectionError("timeout"), ok_resp]

        candles = svc.fetch_candle_range(TradePair.BTCUSDC, start_ms, end_ms)

        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(len(candles), 10)

    @patch("data_generator.hyperliquid_data_service.time.sleep")
    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_fetch_candle_range_returns_partial_data_on_persistent_failure(self, mock_post, mock_sleep):
        """After retries are exhausted, should return an empty list for that chunk rather
        than raising, so the caller can treat the gap as not-yet-covered."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        start_ms = 0
        end_ms = 10 * 60 * 1000

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        candles = svc.fetch_candle_range(TradePair.BTCUSDC, start_ms, end_ms)

        self.assertEqual(candles, [])
        self.assertEqual(mock_post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
