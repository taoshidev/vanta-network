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
        account.reset_account_fields()
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
            capital_used_by_class={TradePairCategory.CRYPTO: 1_000.0},
        )
        balance = account.balance
        expected = balance * ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL] - 1_000.0
        self.assertAlmostEqual(account.buying_power, expected)


# ---------------------------------------------------------------------------
# FX carve-out: FX and non-FX draw on separate portfolio pools (subaccounts)
# ---------------------------------------------------------------------------

class TestFxCarveOut(unittest.TestCase):

    def _subaccount(self, asset_class=MinerAssetClass.ALL_MARKETS, used=None):
        used = used or {}
        return MinerAccount(
            miner_hotkey="hk",
            asset_class=asset_class,
            miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
            capital_used=sum(used.values()),
            capital_used_by_class=dict(used),
        )

    def test_maxed_fx_leaves_non_fx_room_untouched(self):
        account = self._subaccount()
        fx_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.FOREX]
        maxed = self._subaccount(used={TradePairCategory.FOREX: account.balance * fx_cap})

        self.assertAlmostEqual(maxed.portfolio_cap(TradePairCategory.FOREX)[0], 0.0)
        self.assertAlmostEqual(
            maxed.portfolio_cap(TradePairCategory.CRYPTO)[0],
            account.portfolio_cap(TradePairCategory.CRYPTO)[0],
        )

    def test_maxed_non_fx_leaves_fx_room_untouched(self):
        account = self._subaccount()
        overall = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.ALL_MARKETS]
        maxed = self._subaccount(used={TradePairCategory.CRYPTO: account.balance * overall})

        self.assertAlmostEqual(maxed.portfolio_cap(TradePairCategory.CRYPTO)[0], 0.0)
        self.assertAlmostEqual(
            maxed.portfolio_cap(TradePairCategory.FOREX)[0],
            account.portfolio_cap(TradePairCategory.FOREX)[0],
        )

    def test_fx_only_subaccount_room_comes_from_the_fx_pool(self):
        fx_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.FOREX]
        probe = self._subaccount(asset_class=MinerAssetClass.FOREX)
        used = probe.balance * 2
        account = self._subaccount(
            asset_class=MinerAssetClass.FOREX, used={TradePairCategory.FOREX: used}
        )
        self.assertAlmostEqual(account.buying_power_fx, account.balance * fx_cap - used)
        self.assertAlmostEqual(account.portfolio_cap(TradePairCategory.FOREX)[0],
                               account.buying_power_fx)
        # No non-FX access, so it has no cross-asset pool to report.
        self.assertEqual(account.buying_power_non_fx, 0.0)

    def test_legacy_buying_power_keeps_the_pre_carve_out_formula(self):
        """buying_power must stay value-compatible for existing clients."""
        account = self._subaccount(used={
            TradePairCategory.FOREX: 3_000.0,
            TradePairCategory.CRYPTO: 2_000.0,
        })
        self.assertAlmostEqual(
            account.buying_power,
            account.balance * account.multiplier - account.capital_used,
        )
        # ...and is therefore NOT either live pool.
        self.assertNotAlmostEqual(account.buying_power, account.buying_power_fx)
        self.assertNotAlmostEqual(account.buying_power, account.buying_power_non_fx)

    def test_regular_miner_keeps_single_cross_asset_pool(self):
        """Only subaccounts carve FX out; main-comp miners keep one pool over all exposure."""
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.FOREX,
            miner_bucket=MinerBucket.MAINCOMP,
            capital_used=1_000.0,
            capital_used_by_class={TradePairCategory.FOREX: 1_000.0},
        )
        self.assertFalse(account.fx_carved_out)
        expected = account.balance * account.multiplier - 1_000.0
        self.assertAlmostEqual(account.portfolio_cap(TradePairCategory.FOREX)[0], expected)

    def test_hl_all_has_no_fx_pool(self):
        """Hyperliquid lists no forex pairs, so Hyperscaled keeps one cross-asset pool."""
        account = self._subaccount(asset_class=MinerAssetClass.HL_ALL,
                                   used={TradePairCategory.CRYPTO: 1_000.0})
        self.assertFalse(account.fx_has_access)
        self.assertFalse(account.fx_carved_out)
        self.assertEqual(account.buying_power_fx, 0.0)
        self.assertEqual(account.capital_used_fx, 0.0)
        self.assertAlmostEqual(account.buying_power, account.balance * account.multiplier - 1_000.0)

    def test_fx_access_tolerates_trade_pair_category_asset_class(self):
        """asset_class holds a TradePairCategory at some call sites; both str enums share values."""
        account = self._subaccount(asset_class=TradePairCategory.FOREX)
        self.assertTrue(account.fx_has_access)
        self.assertTrue(account.fx_carved_out)

        account = self._subaccount(asset_class=TradePairCategory.EQUITIES)
        self.assertFalse(account.fx_has_access)
        self.assertEqual(account.buying_power_fx, 0.0)

    def test_single_class_non_fx_pools_are_zero_on_fx_side(self):
        for ac in (MinerAssetClass.CRYPTO, MinerAssetClass.COMMODITIES, MinerAssetClass.EQUITIES):
            with self.subTest(asset_class=ac):
                account = self._subaccount(asset_class=ac)
                self.assertEqual(account.buying_power_fx, 0.0)
                self.assertEqual(account.capital_used_fx, 0.0)

    def test_reported_pools_sum_to_total_capital_used(self):
        account = self._subaccount(used={
            TradePairCategory.FOREX: 3_000.0,
            TradePairCategory.CRYPTO: 2_000.0,
        })
        self.assertAlmostEqual(account.capital_used_fx, 3_000.0)
        self.assertAlmostEqual(account.capital_used_non_fx, 2_000.0)
        self.assertAlmostEqual(
            account.capital_used_fx + account.capital_used_non_fx, account.capital_used
        )

    def test_dashboard_and_dict_expose_both_pools(self):
        account = self._subaccount(used={TradePairCategory.FOREX: 3_000.0})
        for payload in (account.to_dict(), account.to_dashboard()):
            for key in ('capital_used_fx', 'capital_used_non_fx',
                        'buying_power_fx', 'buying_power_non_fx'):
                self.assertIn(key, payload)
            self.assertAlmostEqual(payload['capital_used_fx'], 3_000.0)
            self.assertAlmostEqual(payload['buying_power_fx'], account.buying_power_fx)
            self.assertAlmostEqual(payload['buying_power_non_fx'], account.buying_power_non_fx)

    def test_legacy_checkpoint_without_per_class_fails_closed_in_both_pools(self):
        """Untracked exposure must cost room in BOTH pools, never grant a fresh allowance."""
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.ALL_MARKETS,
            miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
            capital_used=1_000.0,
        )
        self.assertAlmostEqual(account.capital_used_untracked, 1_000.0)
        self.assertAlmostEqual(account.capital_used_non_fx, 1_000.0)

        # Non-FX pool: charged, as before.
        self.assertAlmostEqual(
            account.portfolio_cap(TradePairCategory.CRYPTO)[0],
            account.balance * account.multiplier - 1_000.0,
        )
        # FX pool: must also be charged, or an account holding untracked FX would keep a
        # full FX allowance on top of it.
        self.assertAlmostEqual(
            account.portfolio_cap(TradePairCategory.FOREX)[0],
            account.balance * account.fx_multiplier - 1_000.0,
        )

    def test_untracked_fx_exposure_cannot_exceed_the_fx_cap(self):
        """An all-FX book with no per-class breakdown must not reach 2x the FX cap."""
        probe = self._subaccount()
        fx_cap = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.FOREX]
        already_open = probe.balance * 10
        account = MinerAccount(
            miner_hotkey="hk",
            asset_class=MinerAssetClass.ALL_MARKETS,
            miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
            capital_used=already_open,
        )
        reachable = already_open + account.portfolio_cap(TradePairCategory.FOREX)[0]
        self.assertLessEqual(reachable, account.balance * fx_cap + 1e-6)

    def test_unknown_category_falls_back_to_one_pool(self):
        """process_order_buy without a category must not reject FX-only subaccounts."""
        for ac in (MinerAssetClass.FOREX, MinerAssetClass.ALL_MARKETS, MinerAssetClass.EQUITIES):
            with self.subTest(asset_class=ac):
                account = self._subaccount(asset_class=ac,
                                           used={TradePairCategory.FOREX: 1_000.0}
                                           if ac != MinerAssetClass.EQUITIES else {})
                room, _ = account.portfolio_cap(None)
                self.assertGreater(room, 0.0)


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
        self.assertEqual(computed.capital_used_by_class[TradePairCategory.CRYPTO], 150.0)
        self.assertEqual(
            computed.capital_used_by_class[TradePairCategory.FOREX],
            230.0,  # EURUSD 200 + XAUUSD 30 (XAU is FOREX in this PR)
        )

    def test_closed_positions_do_not_contribute(self):
        positions = [
            self._position(TradePair.BTCUSD, 100.0, closed=False),
            self._position(TradePair.ETHUSD, 200.0, closed=True),
        ]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(computed.capital_used_by_class[TradePairCategory.CRYPTO], 100.0)

    def test_sum_per_class_equals_capital_used(self):
        positions = [
            self._position(TradePair.BTCUSD, 100.0),
            self._position(TradePair.EURUSD, 200.0),
            self._position(TradePair.NVDA, 30.0),
        ]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(
            sum(computed.capital_used_by_class.values()),
            computed.capital_used,
        )

    def test_negative_net_value_uses_absolute(self):
        """Short positions have negative net_value; per-class tracks abs(value), like capital_used."""
        positions = [self._position(TradePair.BTCUSD, -100.0)]
        computed = MinerAccountManager.compute_account_state_from_positions(positions)
        self.assertEqual(computed.capital_used_by_class[TradePairCategory.CRYPTO], 100.0)
        self.assertEqual(computed.capital_used, 100.0)

    def test_empty_positions_yield_empty_per_class(self):
        computed = MinerAccountManager.compute_account_state_from_positions([])
        self.assertEqual(computed.capital_used_by_class, {})


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
        # Simulate the on-disk dict format: {hotkey: [last_record_dict]}
        disk_data = {
            "hk": [{
                "account_size": 50_000,
                "update_time_ms": 0,
                "capital_used": 150.0,
                "capital_used_by_class": {"crypto": 100.0, "forex": 50.0},
            }]
        }
        parsed = MinerAccountManager._parse_accounts_dict(disk_data)
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
        serialized = original.to_dict()
        # Mimic the on-disk format: list of records, last record carries account state
        disk_data = {"hk": [{
            "account_size": 50_000,
            "update_time_ms": 0,
            **serialized,
        }]}
        parsed = MinerAccountManager._parse_accounts_dict(disk_data)
        self.assertEqual(
            parsed["hk"].capital_used_by_class,
            original.capital_used_by_class,
        )


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

class TestBackwardCompat(unittest.TestCase):

    def test_old_checkpoint_missing_field_loads_empty_dict(self):
        """Pre-redesign checkpoint has no capital_used_by_class key — defaults to empty."""
        disk_data = {
            "hk": [{
                "account_size": 50_000,
                "update_time_ms": 0,
                "capital_used": 150.0,
                # NO capital_used_by_class field
            }]
        }
        parsed = MinerAccountManager._parse_accounts_dict(disk_data)
        self.assertEqual(parsed["hk"].capital_used_by_class, {})

    def test_old_checkpoint_null_field_loads_empty_dict(self):
        """A null/None value for the field (not just missing) also defaults to empty."""
        disk_data = {
            "hk": [{
                "account_size": 50_000,
                "update_time_ms": 0,
                "capital_used": 0.0,
                "capital_used_by_class": None,
            }]
        }
        parsed = MinerAccountManager._parse_accounts_dict(disk_data)
        self.assertEqual(parsed["hk"].capital_used_by_class, {})

    def test_unknown_asset_class_string_is_skipped(self):
        """Forward compat: an unknown asset_class string in the dict is skipped (not crashed on)."""
        disk_data = {
            "hk": [{
                "account_size": 50_000,
                "update_time_ms": 0,
                "capital_used": 100.0,
                "capital_used_by_class": {
                    "crypto": 100.0,
                    "future_unknown_class": 999.0,
                },
            }]
        }
        parsed = MinerAccountManager._parse_accounts_dict(disk_data)
        cubc = parsed["hk"].capital_used_by_class
        self.assertIn(TradePairCategory.CRYPTO, cubc)
        self.assertEqual(cubc[TradePairCategory.CRYPTO], 100.0)
        # Unknown key dropped silently (with a warning log, not an exception)
        self.assertEqual(len(cubc), 1)

    def test_floats_in_string_form_are_coerced(self):
        """Defensive: amounts may arrive as strings if some serializer along the way string-ified them."""
        disk_data = {
            "hk": [{
                "account_size": 50_000,
                "update_time_ms": 0,
                "capital_used": 100.0,
                "capital_used_by_class": {"crypto": "100.0"},
            }]
        }
        parsed = MinerAccountManager._parse_accounts_dict(disk_data)
        self.assertEqual(parsed["hk"].capital_used_by_class[TradePairCategory.CRYPTO], 100.0)


if __name__ == "__main__":
    unittest.main()
