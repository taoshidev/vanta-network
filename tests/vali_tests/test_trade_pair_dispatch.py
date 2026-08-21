"""
Unit tests for trade pair classification + leverage dispatch reshape
(Phase A: COMMODITIES enum migration; Phase B: InstrumentType field, per-pair
Tier-1 base × tier dispatch, portfolio leverage table split).

Covers:
  * Config completeness — every TradePair has the new fields; every dict has
    the right keys; HL_ALL is absent from the single-class portfolio table.
  * get_tier_positional_leverage — base × SUBACCOUNT_TIER_LEVERAGE_MULTIPLIER[tier],
    XAU/XAG mini-dict bypass, Reg-T cap on EQUITIES SPOT.
  * get_portfolio_caps — multi-class returns (per-class, overall); single-class
    returns the same value twice.
  * TradePair property accessors are position-independent (type-scan).
"""

import types
import unittest

from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.miner_account_manager import MinerAccount
from vali_objects.utils.leverage_utils import (
    REG_T_OVERNIGHT_EQUITY_SPOT_CAP,
    _LEGACY_XAU_XAG_TIER_POSITIONAL,
    get_max_order_size,
    get_portfolio_caps,
    get_tier_positional_leverage,
)
from vali_objects.vali_dataclasses.position import Position
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.trade_pair import (
    InstrumentType,
    SubaccountTierBaseLeverage,
    TradePair,
    TradePairCategory,
    TradePairSource,
)
from vali_objects.vali_config import ValiConfig



# ---------------------------------------------------------------------------
# Config completeness
# ---------------------------------------------------------------------------

class TestConfigCompleteness(unittest.TestCase):

    ALL_TIERS = (1, 2, 3, 4)
    ALL_REAL_CATEGORIES = (
        TradePairCategory.CRYPTO,
        TradePairCategory.FOREX,
        TradePairCategory.EQUITIES,
        TradePairCategory.INDICES,
        TradePairCategory.COMMODITIES,
    )
    ALL_INSTRUMENT_TYPES = (InstrumentType.SPOT, InstrumentType.PERP)

    def test_every_trade_pair_has_instrument_type(self):
        for tp in TradePair:
            with self.subTest(pair=tp.trade_pair_id):
                self.assertIsInstance(tp.instrument_type, InstrumentType)

    def test_every_trade_pair_has_subaccount_tier_base(self):
        for tp in TradePair:
            with self.subTest(pair=tp.trade_pair_id):
                base = tp.subaccount_tier_base_leverage
                self.assertIsInstance(base, float)
                self.assertGreater(base, 0)

    def test_vanta_pairs_are_spot_hl_pairs_are_perp(self):
        for tp in TradePair:
            with self.subTest(pair=tp.trade_pair_id):
                if tp.src == TradePairSource.HYPERLIQUID:
                    self.assertEqual(tp.instrument_type, InstrumentType.PERP)
                else:
                    self.assertEqual(tp.instrument_type, InstrumentType.SPOT)

    def test_subaccount_challenge_returns_threshold_has_commodities(self):
        self.assertIn(
            TradePairCategory.COMMODITIES,
            ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD,
        )

    def test_tier_portfolio_leverage_by_pair_full_matrix(self):
        for tier in self.ALL_TIERS:
            for cat in self.ALL_REAL_CATEGORIES:
                for it in self.ALL_INSTRUMENT_TYPES:
                    with self.subTest(tier=tier, cat=cat, it=it):
                        self.assertIn((cat, it), ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_PAIR[tier])

    def test_tier_portfolio_leverage_by_category_has_no_hl_all(self):
        for tier in self.ALL_TIERS:
            with self.subTest(tier=tier):
                self.assertNotIn(
                    TradePairCategory.HL_ALL,
                    ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[tier],
                )

    def test_tier_portfolio_leverage_by_category_full_matrix(self):
        for tier in self.ALL_TIERS:
            for cat in self.ALL_REAL_CATEGORIES:
                with self.subTest(tier=tier, cat=cat):
                    self.assertIn(cat, ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[tier])

    def test_tier_portfolio_leverage_by_asset_class_has_multi_class_entries(self):
        for table in (ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS,
                      ValiConfig.SUBACCOUNT_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS):
            for tier in self.ALL_TIERS:
                for ac in (MinerAssetClass.HL_ALL, MinerAssetClass.ALL_MARKETS):
                    self.assertIn(ac, table[tier])
                    self.assertGreater(table[tier][ac], 0)

    def test_portfolio_leverage_cap_full_matrix(self):
        for cat in self.ALL_REAL_CATEGORIES:
            for it in self.ALL_INSTRUMENT_TYPES:
                with self.subTest(cat=cat, it=it):
                    self.assertIn((cat, it), ValiConfig.PORTFOLIO_LEVERAGE_CAP)


# ---------------------------------------------------------------------------
# get_tier_positional_leverage
# ---------------------------------------------------------------------------

class TestGetTierPositionalLeverage(unittest.TestCase):

    def test_returns_base_times_tier_multiplier_for_regular_pair(self):
        # BTCUSD has subaccount_tier_base_leverage = 0.5 (CRYPTO SPOT placeholder)
        base = TradePair.BTCUSD.subaccount_tier_base_leverage
        for tier in (1, 2, 3, 4):
            with self.subTest(tier=tier):
                self.assertEqual(
                    get_tier_positional_leverage(tier, TradePair.BTCUSD),
                    base * ValiConfig.SUBACCOUNT_TIER_LEVERAGE_MULTIPLIER[tier],
                )

    def test_tier_2_matches_tier_1(self):
        # Funded (Tier 2) and challenge (Tier 1) share the same per-pair limits.
        for tp in (TradePair.BTCUSDC, TradePair.EURUSD, TradePair.GOLDUSDC, TradePair.SP500USDC):
            with self.subTest(pair=tp.trade_pair_id):
                self.assertEqual(
                    get_tier_positional_leverage(2, tp),
                    get_tier_positional_leverage(1, tp),
                )

    def test_forex_bases(self):
        # G1 majors: base = 20.0; crosses (G2-G5): base = 10.0
        self.assertEqual(get_tier_positional_leverage(1, TradePair.EURUSD), 20.0)
        self.assertEqual(get_tier_positional_leverage(1, TradePair.AUDJPY), 10.0)

    def test_xau_xag_bypass_uses_mini_dict(self):
        for pair in (TradePair.XAUUSD, TradePair.XAGUSD):
            for tier in (1, 2, 3, 4):
                with self.subTest(pair=pair.trade_pair_id, tier=tier):
                    self.assertEqual(
                        get_tier_positional_leverage(tier, pair),
                        _LEGACY_XAU_XAG_TIER_POSITIONAL[tier],
                    )

    def test_xau_xag_bypass_ignores_base_field(self):
        """Even though XAU/XAG have base=2.5, the mini-dict (1/1/1.5/2) wins."""
        self.assertEqual(TradePair.XAUUSD.subaccount_tier_base_leverage, 2.5)
        self.assertNotEqual(
            get_tier_positional_leverage(1, TradePair.XAUUSD),
            2.5 * 1,
        )

    def test_equity_spot_never_exceeds_reg_t(self):
        # Equity SPOT must stay within the Reg-T overnight margin at every tier. The guard
        # is dormant while base × multiplier lands below it; this pins the invariant.
        for tier in (1, 2, 3, 4):
            with self.subTest(tier=tier):
                self.assertLessEqual(
                    get_tier_positional_leverage(tier, TradePair.NVDA),
                    REG_T_OVERNIGHT_EQUITY_SPOT_CAP,
                )

    def test_equity_perp_not_capped_by_reg_t(self):
        # HL equity perps are PERP, not SPOT — Reg-T should NOT apply.
        # Use the live config value as-is; assert base × tier multiplier with no clip.
        base = TradePair.NVDAUSDC.subaccount_tier_base_leverage
        for tier in (1, 2, 3, 4):
            with self.subTest(tier=tier):
                self.assertEqual(
                    get_tier_positional_leverage(tier, TradePair.NVDAUSDC),
                    base * ValiConfig.SUBACCOUNT_TIER_LEVERAGE_MULTIPLIER[tier],
                )

    def test_reg_t_cap_actually_clips_when_base_exceeds(self):
        """Inject a synthetic EQUITIES SPOT pair with a base that would breach Reg-T at tier 4.

        Confirms the guard is reachable, not dead code that happens to be a no-op only
        because today's placeholder base = 0.5 lands tier 4 at exactly 2.0.
        """
        synthetic = types.SimpleNamespace(
            trade_pair_category=TradePairCategory.EQUITIES,
            instrument_type=InstrumentType.SPOT,
            subaccount_tier_base_leverage=1.0,  # tier 3 would give 3.0; Reg-T clips to 2.0
        )
        # tier 1: 1.0 × 1 = 1.0  (below cap, unchanged)
        # tier 4: 1.0 × 4 = 4.0  (clipped to 2.0)
        self.assertEqual(get_tier_positional_leverage(1, synthetic), 1.0)
        self.assertEqual(get_tier_positional_leverage(4, synthetic), REG_T_OVERNIGHT_EQUITY_SPOT_CAP)


# ---------------------------------------------------------------------------
# get_portfolio_caps
# ---------------------------------------------------------------------------

class TestGetPortfolioCaps(unittest.TestCase):

    BUCKET = MinerBucket.SUBACCOUNT_FUNDED
    ACCT = 50_000.0  # tier 2

    def test_single_class_returns_same_value_twice(self):
        per_class, overall = get_portfolio_caps(
            MinerAssetClass.CRYPTO, self.BUCKET, self.ACCT, TradePairCategory.CRYPTO,
        )
        self.assertEqual(per_class, overall)
        self.assertEqual(
            per_class,
            ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.CRYPTO],
        )

    def test_multi_class_overall_from_dedicated_table(self):
        _, overall = get_portfolio_caps(
            MinerAssetClass.HL_ALL, self.BUCKET, self.ACCT, TradePairCategory.CRYPTO,
        )
        self.assertEqual(overall, ValiConfig.SUBACCOUNT_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL])

    def test_multi_class_per_class_keyed_on_order_category(self):
        for cat in (
            TradePairCategory.CRYPTO,
            TradePairCategory.FOREX,
            TradePairCategory.EQUITIES,
            TradePairCategory.INDICES,
            TradePairCategory.COMMODITIES,
        ):
            with self.subTest(cat=cat):
                per_class, _ = get_portfolio_caps(
                    MinerAssetClass.HL_ALL, self.BUCKET, self.ACCT, cat,
                )
                expected = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][cat]
                self.assertEqual(per_class, expected)

    def test_none_subaccount_class_uses_defensive_default(self):
        per_class, overall = get_portfolio_caps(
            None, self.BUCKET, self.ACCT, TradePairCategory.CRYPTO,
        )
        # asset_class=None goes to single-class branch keyed on trade_pair_category.
        expected = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.CRYPTO]
        self.assertEqual(per_class, expected)
        self.assertEqual(overall, expected)

    def test_challenge_bucket_uses_tier_1(self):
        per_class, _ = get_portfolio_caps(
            MinerAssetClass.CRYPTO,
            MinerBucket.SUBACCOUNT_CHALLENGE,
            self.ACCT,
            TradePairCategory.CRYPTO,
        )
        self.assertEqual(
            per_class,
            ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[1][TradePairCategory.CRYPTO],
        )


# ---------------------------------------------------------------------------
# get_max_order_size — every class draws on the single cross-asset portfolio pool
# ---------------------------------------------------------------------------

class TestMaxOrderSizeSinglePool(unittest.TestCase):

    def _account(self, asset_class=MinerAssetClass.ALL_MARKETS,
                 bucket=MinerBucket.SUBACCOUNT_FUNDED, used=None):
        used = used or {}
        return MinerAccount(
            miner_hotkey="hk",
            asset_class=asset_class,
            miner_bucket=bucket,
            capital_used=sum(used.values()),
            capital_used_by_class=dict(used),
        )

    @staticmethod
    def _position(trade_pair, net_value=0.0):
        position = Position(
            miner_hotkey="hk", position_uuid="u", open_ms=0,
            trade_pair=trade_pair, position_type=OrderType.LONG,
        )
        position.net_value = net_value
        return position

    def test_fx_major_reaches_its_per_pair_cap(self):
        """20x EURUSD must be reachable — the 30x portfolio cap must not bind it."""
        account = self._account()
        max_value, _ = get_max_order_size(account, self._position(TradePair.EURUSD))
        expected = account.balance * get_tier_positional_leverage(2, TradePair.EURUSD)
        self.assertAlmostEqual(max_value, expected)

    def test_pure_fx_subaccount_reaches_the_full_30x_class_cap(self):
        """A FOREX subaccount's portfolio cap equals its 30x class cap: with 20x of majors
        open, a cross still gets its full 10x per-pair room, totalling 30x."""
        overall = ValiConfig.SUBACCOUNT_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.FOREX]
        empty = self._account(asset_class=MinerAssetClass.FOREX)
        loaded = self._account(
            asset_class=MinerAssetClass.FOREX,
            used={TradePairCategory.FOREX: empty.balance * (overall - 10.0)},
        )
        max_value, _ = get_max_order_size(loaded, self._position(TradePair.AUDJPY))
        self.assertAlmostEqual(max_value, loaded.balance * 10.0)

    def test_maxed_fx_consumes_the_shared_pool(self):
        """FX at the full 30x portfolio cap leaves no room in any class, FX included."""
        cap = ValiConfig.SUBACCOUNT_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.ALL_MARKETS]
        empty = self._account()
        maxed_fx = self._account(used={TradePairCategory.FOREX: empty.balance * cap})

        for pair in (TradePair.BTCUSDC, TradePair.SP500USDC, TradePair.GOLDUSDC, TradePair.EURUSD):
            with self.subTest(pair=pair.trade_pair_id):
                room, _ = get_max_order_size(maxed_fx, self._position(pair))
                self.assertAlmostEqual(room, 0.0)

    def test_non_fx_exposure_shrinks_fx_room(self):
        """Non-FX exposure draws on the same pool: with 20x of non-FX open, an FX order is
        clipped to the remaining 10x of portfolio room instead of its 20x per-pair cap."""
        empty = self._account()
        balance = empty.balance
        loaded = self._account(used={
            TradePairCategory.CRYPTO: balance * 5.0,
            TradePairCategory.INDICES: balance * 10.0,
            TradePairCategory.COMMODITIES: balance * 3.0,
            TradePairCategory.EQUITIES: balance * 2.0,
        })
        room, binding = get_max_order_size(loaded, self._position(TradePair.EURUSD))
        self.assertAlmostEqual(room, balance * 10.0)
        self.assertIn("overall portfolio cap", binding)

    def test_regular_miner_fx_consumes_the_single_pool(self):
        """Main-comp miners keep one pool at the regular-miner (main) cap values: a FOREX
        miner with 9x of its 10x pool used has 1x of room left."""
        overall = ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.FOREX]
        empty = self._account(asset_class=MinerAssetClass.FOREX, bucket=MinerBucket.MAINCOMP)
        loaded = self._account(
            asset_class=MinerAssetClass.FOREX, bucket=MinerBucket.MAINCOMP,
            used={TradePairCategory.FOREX: empty.balance * (overall - 1.0)},
        )
        room, binding = get_max_order_size(loaded, self._position(TradePair.EURUSD))
        self.assertAlmostEqual(room, loaded.balance * 1.0)
        self.assertIn("overall portfolio cap", binding)


# ---------------------------------------------------------------------------
# TradePair property accessors are position-independent
# ---------------------------------------------------------------------------

class TestTradePairPropertyAccessors(unittest.TestCase):

    def test_instrument_type_via_type_scan(self):
        """Both Vanta (no subcategory, src) and HL (with subcategory=None, src) shapes resolve."""
        # Vanta crypto: [id, name, fee, min, max, category, subcategory, instrument, base]
        self.assertEqual(TradePair.BTCUSD.instrument_type, InstrumentType.SPOT)
        # HL crypto: [id, name, fee, min, max, category, None, src, instrument, base]
        self.assertEqual(TradePair.BTCUSDC.instrument_type, InstrumentType.PERP)
        # HL commodity: [id, name, fee, min, max, category, None, src, coin, instrument, base]
        self.assertEqual(TradePair.GOLDUSDC.instrument_type, InstrumentType.PERP)
        # Equities (no subcategory): [id, name, fee, min, max, category, instrument, base]
        self.assertEqual(TradePair.NVDA.instrument_type, InstrumentType.SPOT)

    def test_subaccount_tier_base_via_named_tuple_scan(self):
        self.assertEqual(TradePair.BTCUSD.subaccount_tier_base_leverage, 0.5)
        self.assertEqual(TradePair.EURUSD.subaccount_tier_base_leverage, 20.0)
        self.assertEqual(TradePair.GOLDUSDC.subaccount_tier_base_leverage, 3.0)
        self.assertEqual(TradePair.NVDA.subaccount_tier_base_leverage, 1.0)

    def test_subaccount_tier_base_wrapper_isolates_from_floats(self):
        """The SubaccountTierBaseLeverage wrapper is distinct from raw float fields."""
        wrapper = SubaccountTierBaseLeverage(0.5)
        self.assertFalse(isinstance(wrapper, float))
        self.assertEqual(wrapper.value, 0.5)

    def test_src_property_still_works_after_field_extension(self):
        self.assertEqual(TradePair.BTCUSD.src, TradePairSource.VANTA)
        self.assertEqual(TradePair.BTCUSDC.src, TradePairSource.HYPERLIQUID)

    def test_hl_coin_property_still_works(self):
        # GOLDUSDC has hl_coin="xyz:GOLD"; BTCUSD has no hl_coin → falls back to base name
        self.assertEqual(TradePair.GOLDUSDC.hl_coin, "xyz:GOLD")


if __name__ == "__main__":
    unittest.main()
