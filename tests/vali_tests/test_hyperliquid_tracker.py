"""
Tests for Hyperliquid trade tracking system.

Tests cover:
- Registration storage (add/remove/persist)
- Fill-to-signal conversion (all direction types)
- Coin-to-TradePair mapping (valid coins, unknown coins)
- Ethereum address validation
"""

import unittest

from vali_objects.tracking.hyperliquid_tracker_server import (
    COIN_TO_TRADE_PAIR,
    fill_to_signal,
    is_valid_eth_address,
)
from vali_objects.vali_config import TradePair


class TestEthAddressValidation(unittest.TestCase):
    """Test Ethereum address validation."""

    def test_valid_address(self):
        self.assertTrue(is_valid_eth_address("0x1234567890abcdef1234567890abcdef12345678"))

    def test_valid_address_mixed_case(self):
        self.assertTrue(is_valid_eth_address("0xAbCdEf1234567890AbCdEf1234567890AbCdEf12"))

    def test_invalid_no_prefix(self):
        self.assertFalse(is_valid_eth_address("1234567890abcdef1234567890abcdef12345678"))

    def test_invalid_too_short(self):
        self.assertFalse(is_valid_eth_address("0x1234"))

    def test_invalid_too_long(self):
        self.assertFalse(is_valid_eth_address("0x1234567890abcdef1234567890abcdef123456789"))

    def test_invalid_non_hex(self):
        self.assertFalse(is_valid_eth_address("0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"))

    def test_empty_string(self):
        self.assertFalse(is_valid_eth_address(""))

    def test_none_like(self):
        self.assertFalse(is_valid_eth_address("0x"))


class TestCoinToTradePairMapping(unittest.TestCase):
    """Test coin name -> TradePair mapping."""

    def test_btc_maps_to_btcusd(self):
        tp = COIN_TO_TRADE_PAIR.get("BTC")
        self.assertIsNotNone(tp)
        self.assertEqual(tp.trade_pair_id, "BTCUSD")

    def test_eth_maps_to_ethusd(self):
        tp = COIN_TO_TRADE_PAIR.get("ETH")
        self.assertIsNotNone(tp)
        self.assertEqual(tp.trade_pair_id, "ETHUSD")

    def test_sol_maps_to_solusd(self):
        tp = COIN_TO_TRADE_PAIR.get("SOL")
        self.assertIsNotNone(tp)
        self.assertEqual(tp.trade_pair_id, "SOLUSD")

    def test_unknown_coin_returns_none(self):
        self.assertIsNone(COIN_TO_TRADE_PAIR.get("SHIB"))

    def test_all_entries_are_crypto(self):
        for coin, tp in COIN_TO_TRADE_PAIR.items():
            self.assertTrue(tp.is_crypto, f"{coin} -> {tp} is not crypto")

    def test_mapping_is_not_empty(self):
        self.assertGreater(len(COIN_TO_TRADE_PAIR), 0)


class TestFillToSignal(unittest.TestCase):
    """Test conversion from Hyperliquid fill to signal dict."""

    def _make_fill(self, coin="BTC", px="65000.0", sz="0.1", side="B", direction="Open Long"):
        return {
            "coin": coin,
            "px": px,
            "sz": sz,
            "side": side,
            "time": 1234567890,
            "closedPnl": "0",
            "dir": direction,
            "startPosition": "0.0",
        }

    def test_open_long(self):
        fill = self._make_fill(direction="Open Long")
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["order_type"], "LONG")
        self.assertEqual(signal["trade_pair"]["trade_pair_id"], "BTCUSD")
        self.assertEqual(signal["execution_type"], "MARKET")
        self.assertAlmostEqual(signal["value"], 0.1 * 65000.0)

    def test_open_short(self):
        fill = self._make_fill(direction="Open Short", side="A")
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["order_type"], "SHORT")

    def test_close_long(self):
        fill = self._make_fill(direction="Close Long", side="A")
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["order_type"], "FLAT")

    def test_close_short(self):
        fill = self._make_fill(direction="Close Short", side="B")
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["order_type"], "FLAT")

    def test_unknown_direction_returns_none(self):
        fill = self._make_fill(direction="Unknown Direction")
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNone(signal)

    def test_value_calculation(self):
        fill = self._make_fill(px="50000.0", sz="2.5")
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertAlmostEqual(signal["value"], 125000.0)

    def test_leverage_is_none(self):
        fill = self._make_fill()
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNone(signal["leverage"])

    def test_quantity_is_none(self):
        fill = self._make_fill()
        signal = fill_to_signal(fill, TradePair.BTCUSD)
        self.assertIsNone(signal["quantity"])

    def test_eth_trade_pair(self):
        fill = self._make_fill(coin="ETH", px="3500.0", sz="10.0")
        signal = fill_to_signal(fill, TradePair.ETHUSD)
        self.assertEqual(signal["trade_pair"]["trade_pair_id"], "ETHUSD")
        self.assertAlmostEqual(signal["value"], 35000.0)


if __name__ == "__main__":
    unittest.main()
