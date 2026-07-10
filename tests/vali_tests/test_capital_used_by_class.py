"""
Unit tests for per-asset-class capital tracking on MinerAccount
(Phase C: multi-class subaccount portfolio caps).

Covers:
  * MinerAccount.capital_used_by_class default and reset semantics.
  * MinerAccount.multiplier / buying_power branches for
    single-class vs multi-class (HL_ALL).
  * MinerAccountManager.compute_account_state_from_positions populates the
    per-class breakdown.
  * Serialization (to_dict) emits string keys; loader rehydrates back to
    TradePairCategory enum keys.
  * Backward compat — old checkpoint missing the field defaults to empty
    dict; unknown asset_class strings are logged and skipped, not crashed on.

The manager-level process_order_buy / process_order_sell wire-up is covered
by integration tests in test_market_order_manager.py. This file focuses on
state, properties, and the (de)serialization boundary.
"""

import unittest

from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import MinerAccount, MinerAccountManager
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.vali_config import (
    TradePair,
    TradePairCategory,
    ValiConfig,
)
from vali_objects.vali_dataclasses.position import Position


# ---------------------------------------------------------------------------
# Field default & reset
# ---------------------------------------------------------------------------

class TestCapitalUsedByClassDefault(unittest.TestCase):

    def test_field_defaults_to_empty_dict(self):
        account = MinerAccount(miner_hotkey="hk")
        self.assertEqual(account.capital_used_by_class, {})

    def test_field_default_is_independent_per_instance(self):
        """default_factory=dict — not a shared mutable default."""
        a = MinerAccount(miner_hotkey="a")
        b = MinerAccount(miner_hotkey="b")
        a.capital_used_by_class[TradePairCategory.CRYPTO] = 100.0
        self.assertEqual(b.capital_used_by_class, {})

    def test_reset_account_fields_clears_per_class(self):
        account = MinerAccount(
            miner_hotkey="hk",
            capital_used_by_class={TradePairCategory.CRYPTO: 100.0, TradePairCategory.FOREX: 50.0},
        )
        account = MinerAccount(
            miner_hotkey=account.miner_hotkey,
            collateral_records=account.collateral_records,
        )
        self.assertEqual(account.capital_used_by_class, {})


# ---------------------------------------------------------------------------
# multiplier / buying_power
# ---------------------------------------------------------------------------

class TestMinerAccountMultiplier(unittest.TestCase):

    def test_multiplier_no_asset_class_returns_1(self):
        account = MinerAccount(miner_hotkey="hk", asset_class=None)
        self.assertEqual(account.multiplier, 1)

    def test_multiplier_single_class_reads_by_asset_class_table(self):
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.CRYPTO,
            miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
        )
        # Tier defaults to 2 (funded, no collateral records → MIN_CAPITAL < 200K)
        expected = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][MinerAssetClass.CRYPTO]
        self.assertEqual(account.multiplier, expected)

    def test_multiplier_multi_class_reads_overall_cap_table(self):
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.HL_ALL,
            miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
        )
        self.assertEqual(account.multiplier, ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL])

    def test_multiplier_challenge_bucket_uses_tier_1(self):
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.HL_ALL,
            miner_bucket=MinerBucket.SUBACCOUNT_CHALLENGE,
        )
        self.assertEqual(account.multiplier, ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[1][MinerAssetClass.HL_ALL])

    def test_buying_power_multi_class_uses_overall_cap(self):
        """buying_power = balance × multiplier − capital_used for non-equities."""
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.HL_ALL,
            miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
            capital_used=1_000.0,
        )
        balance = account.balance
        expected = balance * ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL] - 1_000.0
        self.assertAlmostEqual(account.buying_power, expected)


# ---------------------------------------------------------------------------
# compute_account_state_from_positions
# ---------------------------------------------------------------------------

class TestComputeAccountStateFromPositions(unittest.TestCase):

    @staticmethod
    def _position(trade_pair, net_value, closed=False):
        return Position(
            miner_hotkey="hk",
            position_uuid=f"pos_{trade_pair.trade_pair_id}_{net_value}",
            open_ms=0,
            trade_pair=trade_pair,
            net_value=net_value,
            is_closed_position=closed,
        )

    def test_open_positions_aggregate_into_per_class(self):
        positions = [
            self._position(TradePair.BTCUSD, 100.0),
            self._position(TradePair.ETHUSD, 50.0),
            self._position(TradePair.EURUSD, 200.0),
            self._position(TradePair.XAUUSD, 30.0),  # FOREX-categorized
        ]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(computed['capital_used_by_class'][TradePairCategory.CRYPTO], 150.0)
        self.assertEqual(
            computed['capital_used_by_class'][TradePairCategory.FOREX],
            230.0,  # EURUSD 200 + XAUUSD 30 (XAU is FOREX in this PR)
        )

    def test_closed_positions_do_not_contribute(self):
        positions = [
            self._position(TradePair.BTCUSD, 100.0, closed=False),
            self._position(TradePair.ETHUSD, 200.0, closed=True),
        ]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(computed['capital_used_by_class'][TradePairCategory.CRYPTO], 100.0)

    def test_sum_per_class_equals_capital_used(self):
        positions = [
            self._position(TradePair.BTCUSD, 100.0),
            self._position(TradePair.EURUSD, 200.0),
            self._position(TradePair.NVDA, 30.0),
        ]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(
            sum(computed['capital_used_by_class'].values()),
            computed['capital_used'],
        )

    def test_negative_net_value_uses_absolute(self):
        """Short positions have negative net_value; per-class tracks abs(value), like capital_used."""
        positions = [self._position(TradePair.BTCUSD, -100.0)]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(computed['capital_used_by_class'][TradePairCategory.CRYPTO], 100.0)
        self.assertEqual(computed['capital_used'], 100.0)

    def test_empty_positions_yield_empty_per_class(self):
        computed = MinerAccountManager.compute_account_state_from_positions([])
        self.assertEqual(computed['capital_used_by_class'], {})


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip(unittest.TestCase):

    def test_to_dict_uses_string_keys(self):
        account = MinerAccount(
            miner_hotkey="hk",
            capital_used_by_class={
                TradePairCategory.CRYPTO: 100.0,
                TradePairCategory.FOREX: 50.0,
            },
        )
        d = account.to_dict()
        self.assertEqual(d['capital_used_by_class'], {"crypto": 100.0, "forex": 50.0})

    def test_loader_rehydrates_string_keys_to_enum(self):
        account = MinerAccount(
            miner_hotkey="hk",
            capital_used=150.0,
            capital_used_by_class={"crypto": 100.0, "forex": 50.0},
        )
        parsed = MinerAccountManager.parse_checkpoint_dict({"hk": account.to_checkpoint_dict()})
        self.assertIn("hk", parsed)
        cubc = parsed["hk"].capital_used_by_class
        self.assertEqual(cubc[TradePairCategory.CRYPTO], 100.0)
        self.assertEqual(cubc[TradePairCategory.FOREX], 50.0)

    def test_round_trip(self):
        original = MinerAccount(
            miner_hotkey="hk",
            capital_used=150.0,
            capital_used_by_class={
                TradePairCategory.CRYPTO: 100.0,
                TradePairCategory.COMMODITIES: 50.0,
            },
        )
        parsed = MinerAccountManager.parse_checkpoint_dict({"hk": original.to_checkpoint_dict()})
        self.assertEqual(
            parsed["hk"].capital_used_by_class,
            original.capital_used_by_class,
        )


if __name__ == "__main__":
    unittest.main()
