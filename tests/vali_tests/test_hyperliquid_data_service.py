import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock

from data_generator.hyperliquid_data_service import HyperliquidDataService, HYPERLIQUID_PROVIDER_NAME
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory, ValiConfig


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

    def test_coin_mapping_excludes_unsupported(self):
        for tp in ValiConfig.UNSUPPORTED_TRADE_PAIRS:
            if tp.is_crypto:
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

    def test_get_closes_rest_unit_test_mode(self):
        """In unit test mode, get_closes_rest returns default fallback price sources."""
        results = self.service.get_closes_rest([TradePair.BTCUSD, TradePair.ETHUSD], TimeUtil.now_in_millis())
        self.assertEqual(len(results), 2)
        self.assertIn(TradePair.BTCUSD, results)
        self.assertIn(TradePair.ETHUSD, results)

    def test_get_closes_rest_ignores_non_crypto(self):
        """Non-crypto pairs should be ignored."""
        results = self.service.get_closes_rest([TradePair.EURUSD], TimeUtil.now_in_millis())
        # EURUSD is forex, not crypto — should not be in results (unit test mode returns for all passed pairs,
        # but the method filters to crypto only before the unit test shortcut)
        # Since running_unit_tests returns early for all trade_pairs before filtering, let's
        # test with a non-unit-test service instead
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        with patch("data_generator.hyperliquid_data_service.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {})
            mock_post.return_value.raise_for_status = MagicMock()
            results = svc.get_closes_rest([TradePair.EURUSD], TimeUtil.now_in_millis())
        self.assertEqual(len(results), 0)

    def test_get_close_rest_single_pair(self):
        """get_close_rest should return a PriceSource for a single pair."""
        result = self.service.get_close_rest(TradePair.BTCUSD, TimeUtil.now_in_millis())
        self.assertIsNotNone(result)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_get_closes_rest_uses_all_mids(self, mock_post):
        """REST fallback should use allMids endpoint and produce correct PriceSources."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"BTC": "67500.5", "ETH": "3400.25", "SOL": "145.0"}
        mock_post.return_value = mock_response

        results = svc.get_closes_rest(
            [TradePair.BTCUSD, TradePair.ETHUSD], TimeUtil.now_in_millis()
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[TradePair.BTCUSD].close, 67500.5)
        self.assertAlmostEqual(results[TradePair.ETHUSD].close, 3400.25)
        self.assertEqual(results[TradePair.BTCUSD].source, f"{HYPERLIQUID_PROVIDER_NAME}_rest")
        self.assertFalse(results[TradePair.BTCUSD].websocket)

    @patch("data_generator.hyperliquid_data_service.requests.post")
    def test_get_closes_rest_falls_back_to_l2book(self, mock_post):
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

        results = svc.get_closes_rest(
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
    def test_get_closes_rest_handles_api_failure(self, mock_post):
        """If the REST API fails entirely, should return empty dict."""
        svc = HyperliquidDataService(disable_ws=True, running_unit_tests=False)
        mock_post.side_effect = Exception("Connection refused")

        results = svc.get_closes_rest([TradePair.BTCUSD], TimeUtil.now_in_millis())
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
