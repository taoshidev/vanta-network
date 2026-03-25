import asyncio
import json
import unittest

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
        expected_coins = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"}
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

    # -- handle_msg tests --

    def test_handle_msg_valid_l2book(self):
        now_ms = TimeUtil.now_in_millis()
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30002.0", time_ms=now_ms)

        asyncio.run(self.service.handle_msg(msg))

        symbol = TradePair.BTCUSD.trade_pair  # "BTC/USD"
        self.assertIn(symbol, self.service.latest_websocket_events)

        ps = self.service.latest_websocket_events[symbol]
        self.assertEqual(ps.bid, 30000.0)
        self.assertEqual(ps.ask, 30002.0)
        self.assertEqual(ps.close, 30001.0)  # mid price
        self.assertEqual(ps.source, f"{HYPERLIQUID_PROVIDER_NAME}_ws")
        self.assertTrue(ps.websocket)
        self.assertEqual(ps.timespan_ms, 0)

    def test_handle_msg_stores_in_recent_events(self):
        msg = self._make_l2book_msg(coin="ETH", bid="2000.0", ask="2001.0")
        asyncio.run(self.service.handle_msg(msg))

        symbol = TradePair.ETHUSD.trade_pair
        self.assertIn(symbol, self.service.trade_pair_to_recent_events)

        tracker = self.service.trade_pair_to_recent_events[symbol]
        self.assertTrue(len(tracker.events) > 0)

    def test_handle_msg_ignores_non_l2book(self):
        # Subscription confirmation message
        msg = json.dumps({"channel": "subscriptionResponse", "data": {"method": "subscribe"}})
        asyncio.run(self.service.handle_msg(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_ignores_unknown_coin(self):
        msg = self._make_l2book_msg(coin="UNKNOWN", bid="100.0", ask="101.0")
        asyncio.run(self.service.handle_msg(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_handles_empty_levels(self):
        msg = json.dumps({
            "channel": "l2Book",
            "data": {"coin": "BTC", "time": TimeUtil.now_in_millis(), "levels": []}
        })
        asyncio.run(self.service.handle_msg(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_handles_empty_bids(self):
        msg = json.dumps({
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "time": TimeUtil.now_in_millis(),
                "levels": [[], [{"px": "30001.0", "sz": "0.8", "n": 3}]]
            }
        })
        asyncio.run(self.service.handle_msg(msg))
        self.assertEqual(len(self.service.latest_websocket_events), 0)

    def test_handle_msg_increments_event_counter(self):
        initial_count = self.service.tpc_to_n_events[TradePairCategory.CRYPTO]
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30001.0")
        asyncio.run(self.service.handle_msg(msg))
        self.assertEqual(self.service.tpc_to_n_events[TradePairCategory.CRYPTO], initial_count + 1)

    # -- get_closes_websocket tests --

    def test_get_closes_websocket_returns_injected_data(self):
        now_ms = TimeUtil.now_in_millis()
        msg = self._make_l2book_msg(coin="BTC", bid="30000.0", ask="30002.0", time_ms=now_ms)
        asyncio.run(self.service.handle_msg(msg))

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
            asyncio.run(self.service.handle_msg(msg))

        results = self.service.get_closes_websocket(
            [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD], now_ms
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[TradePair.BTCUSD].close, 30001.0)
        self.assertEqual(results[TradePair.ETHUSD].close, 2000.5)
        self.assertEqual(results[TradePair.SOLUSD].close, 100.25)


if __name__ == "__main__":
    unittest.main()
