"""
Unit tests for trade pair classification + leverage dispatch reshape
(Phase A: COMMODITIES enum migration; Phase B: InstrumentType field, per-pair
Tier-1 base × tier dispatch, portfolio leverage table split).

Covers:
  * Config completeness — every TradePair has the new fields; every dict has
    the right keys; HL_ALL is absent from the single-class portfolio table.
  * Value preservation — the new "base × tier" formula reproduces main's
    per-(category, instrument_type) values for all 171 pairs × 4 tiers.
  * get_legacy_tier_positional_leverage — base × tier.
  * get_legacy_leverage_tier — standard subaccounts pinned to one tier; HL-linked
    subaccounts, pro subaccounts and regular miners keep the legacy bucket/size curve.
  * get_legacy_portfolio_caps — multi-class returns (per-class, overall); single-class
    returns the same value twice.
  * TradePair property accessors are position-independent (type-scan).
"""

import unittest

from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.leverage_utils import (
    get_legacy_leverage_tier,
    get_legacy_portfolio_caps,
    get_legacy_tier_positional_leverage,
)
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.trade_pair import (
    HL_COIN_TO_TRADE_PAIR,
    ExposureGroup,
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

    def test_legacy_tier_portfolio_leverage_by_category_has_no_hl_all(self):
        for tier in self.ALL_TIERS:
            with self.subTest(tier=tier):
                self.assertNotIn(
                    MinerAssetClass.HL_ALL,
                    ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[tier],
                )

    def test_legacy_tier_portfolio_leverage_by_category_full_matrix(self):
        for tier in self.ALL_TIERS:
            for cat in self.ALL_REAL_CATEGORIES:
                with self.subTest(tier=tier, cat=cat):
                    self.assertIn(cat, ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[tier])

    def test_legacy_tier_portfolio_leverage_by_asset_class_has_multi_class_entries(self):
        for tier in self.ALL_TIERS:
            for ac in (MinerAssetClass.HL_ALL, MinerAssetClass.ALL_MARKETS):
                self.assertIn(ac, ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier])
                self.assertGreater(ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier][ac], 0)


# ---------------------------------------------------------------------------
# get_legacy_tier_positional_leverage
# ---------------------------------------------------------------------------

class TestGetLegacyTierPositionalLeverage(unittest.TestCase):

    def test_returns_base_times_tier_for_regular_pair(self):
        # BTCUSD has subaccount_tier_base_leverage = 0.5 (CRYPTO SPOT placeholder)
        base = TradePair.BTCUSD.subaccount_tier_base_leverage
        for tier in (1, 2, 3, 4):
            with self.subTest(tier=tier):
                self.assertEqual(get_legacy_tier_positional_leverage(tier, TradePair.BTCUSD), base * tier)

    def test_forex_uses_2_5_base(self):
        # EURUSD: FOREX SPOT base = 2.5
        self.assertEqual(get_legacy_tier_positional_leverage(1, TradePair.EURUSD), 2.5)
        self.assertEqual(get_legacy_tier_positional_leverage(4, TradePair.EURUSD), 10.0)

    def test_equity_spot_and_perp_are_base_times_tier(self):
        # No Reg T clip any more: equity bases are 0.5, so tier 4 lands at 2.0 for SPOT and PERP alike.
        for pair in (TradePair.NVDA, TradePair.NVDAUSDC):
            base = pair.subaccount_tier_base_leverage
            for tier in (1, 2, 3, 4):
                with self.subTest(pair=pair.trade_pair_id, tier=tier):
                    self.assertEqual(get_legacy_tier_positional_leverage(tier, pair), base * tier)


# ---------------------------------------------------------------------------
# get_legacy_portfolio_caps
# ---------------------------------------------------------------------------

class TestGetLegacyPortfolioCaps(unittest.TestCase):

    BUCKET = MinerBucket.SUBACCOUNT_FUNDED
    ACCT = 50_000.0  # tier 2
    HL_ADDRESS = "0x" + "a" * 40

    def test_single_class_returns_same_value_twice(self):
        per_class, overall = get_legacy_portfolio_caps(
            MinerAssetClass.CRYPTO, self.BUCKET, self.ACCT, TradePairCategory.CRYPTO, None,
        )
        self.assertEqual(per_class, overall)
        self.assertEqual(
            per_class,
            ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.CRYPTO],
        )

    def test_multi_class_overall_from_dedicated_table(self):
        _, overall = get_legacy_portfolio_caps(
            MinerAssetClass.HL_ALL, self.BUCKET, self.ACCT, TradePairCategory.CRYPTO, self.HL_ADDRESS,
        )
        self.assertEqual(overall, ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[2][MinerAssetClass.HL_ALL])

    def test_multi_class_per_class_keyed_on_order_category(self):
        for cat in (
            TradePairCategory.CRYPTO,
            TradePairCategory.FOREX,
            TradePairCategory.EQUITIES,
            TradePairCategory.INDICES,
            TradePairCategory.COMMODITIES,
        ):
            with self.subTest(cat=cat):
                per_class, _ = get_legacy_portfolio_caps(
                    MinerAssetClass.HL_ALL, self.BUCKET, self.ACCT, cat, self.HL_ADDRESS,
                )
                expected = ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][cat]
                self.assertEqual(per_class, expected)

    def test_none_subaccount_class_uses_defensive_default(self):
        per_class, overall = get_legacy_portfolio_caps(
            None, self.BUCKET, self.ACCT, TradePairCategory.CRYPTO, None,
        )
        # per-class still comes from the category table; the overall cap falls back to 1.0
        # because asset_class=None has no entry in the asset-class table.
        expected = ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[2][TradePairCategory.CRYPTO]
        self.assertEqual(per_class, expected)
        self.assertEqual(overall, 1.0)

    def test_standard_challenge_bucket_uses_standard_tier(self):
        per_class, _ = get_legacy_portfolio_caps(
            MinerAssetClass.CRYPTO,
            MinerBucket.SUBACCOUNT_CHALLENGE,
            self.ACCT,
            TradePairCategory.CRYPTO,
            None,
        )
        self.assertEqual(
            per_class,
            ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[ValiConfig.LEGACY_STANDARD_SUBACCOUNT_LEVERAGE_TIER][TradePairCategory.CRYPTO],
        )

    def test_hl_challenge_bucket_uses_tier_1(self):
        per_class, overall = get_legacy_portfolio_caps(
            MinerAssetClass.HL_ALL,
            MinerBucket.SUBACCOUNT_CHALLENGE,
            self.ACCT,
            TradePairCategory.CRYPTO,
            self.HL_ADDRESS,
        )
        self.assertEqual(per_class, ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[1][TradePairCategory.CRYPTO])
        self.assertEqual(overall, ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[1][MinerAssetClass.HL_ALL])


# ---------------------------------------------------------------------------
# get_legacy_leverage_tier
# ---------------------------------------------------------------------------

class TestGetLegacyLeverageTier(unittest.TestCase):

    HL_ADDRESS = "0x" + "a" * 40
    SIZES = (5_000.0, 50_000.0, 100_000.0, 200_000.0, 1_000_000.0)

    def test_standard_subaccount_pinned_regardless_of_bucket_and_size(self):
        expected = ValiConfig.LEGACY_STANDARD_SUBACCOUNT_LEVERAGE_TIER
        for bucket in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            for size in self.SIZES:
                with self.subTest(bucket=bucket, size=size):
                    self.assertEqual(get_legacy_leverage_tier(bucket, size, None), expected)
                    self.assertEqual(get_legacy_leverage_tier(bucket, size, ""), expected)

    def test_hl_subaccount_keeps_legacy_curve(self):
        cases = (
            (MinerBucket.SUBACCOUNT_CHALLENGE, 50_000.0, 1),
            (MinerBucket.SUBACCOUNT_CHALLENGE, 2_000_000.0, 1),
            (MinerBucket.SUBACCOUNT_FUNDED, 50_000.0, 2),
            (MinerBucket.SUBACCOUNT_FUNDED, 199_999.0, 2),
            (MinerBucket.SUBACCOUNT_FUNDED, 200_000.0, 3),
            (MinerBucket.SUBACCOUNT_FUNDED, 1_000_000.0, 4),
        )
        for bucket, size, expected in cases:
            with self.subTest(bucket=bucket, size=size):
                self.assertEqual(get_legacy_leverage_tier(bucket, size, self.HL_ADDRESS), expected)

    def test_regular_miner_keeps_size_curve(self):
        for bucket in (MinerBucket.CHALLENGE, MinerBucket.MAINCOMP, MinerBucket.PROBATION, None):
            with self.subTest(bucket=bucket):
                self.assertEqual(get_legacy_leverage_tier(bucket, 50_000.0, None), 2)
                self.assertEqual(get_legacy_leverage_tier(bucket, 200_000.0, None), 3)
                self.assertEqual(get_legacy_leverage_tier(bucket, 1_000_000.0, None), 4)

    def test_pro_subaccount_keeps_legacy_curve(self):
        cases = (
            (MinerBucket.PRO_CHALLENGE_DIRECT, 50_000.0, 1),
            (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 50_000.0, 1),
            (MinerBucket.PRO_FUNDED, 50_000.0, 2),
            (MinerBucket.PRO_FUNDED, 200_000.0, 3),
            (MinerBucket.PRO_FUNDED, 1_000_000.0, 4),
        )
        for bucket, size, expected in cases:
            with self.subTest(bucket=bucket, size=size):
                self.assertEqual(get_legacy_leverage_tier(bucket, size, None), expected)

    def test_pro_transition_bucket_is_pinned_like_standard(self):
        # Transition week is still on the standard account, so it keeps the standard pinned tier.
        for size in self.SIZES:
            with self.subTest(size=size):
                self.assertEqual(
                    get_legacy_leverage_tier(MinerBucket.PRO_CHALLENGE_TRANSITION, size, None),
                    ValiConfig.LEGACY_STANDARD_SUBACCOUNT_LEVERAGE_TIER,
                )


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
        # Equities (no subcategory): [id, name, fee, min, max, category, instrument, base, exposure_group]
        self.assertEqual(TradePair.NVDA.instrument_type, InstrumentType.SPOT)

    def test_subaccount_tier_base_via_named_tuple_scan(self):
        self.assertEqual(TradePair.BTCUSD.subaccount_tier_base_leverage, 0.5)
        self.assertEqual(TradePair.EURUSD.subaccount_tier_base_leverage, 2.5)
        self.assertEqual(TradePair.GOLDUSDC.subaccount_tier_base_leverage, 1.0)
        self.assertEqual(TradePair.NVDA.subaccount_tier_base_leverage, 0.5)

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

    def test_hl_coin_falls_back_to_base_for_non_hl_pairs(self):
        """Equities and forex sit at len(value) > 8, so the index-8 slot must not be read."""
        self.assertEqual(TradePair.NVDA.hl_coin, "NVDA")
        self.assertEqual(TradePair.EURUSD.hl_coin, "EUR")
        self.assertEqual(TradePair.NVDAUSDC.hl_coin, "xyz:NVDA")
        self.assertIs(HL_COIN_TO_TRADE_PAIR["xyz:NVDA"], TradePair.NVDAUSDC)

    def test_exposure_group_is_equities_only(self):
        self.assertEqual(TradePair.NVDA.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertIsNone(TradePair.EURUSD.exposure_group)
        self.assertIsNone(TradePair.BTCUSD.exposure_group)
        self.assertIsNone(TradePair.SP500USDC.exposure_group)


class TestExposureGroups(unittest.TestCase):
    """Sectors are literals on each equity member; nothing is read from the CSV at runtime."""

    def test_every_group_is_used(self):
        in_use = {tp.exposure_group for tp in TradePair if tp.exposure_group is not None}
        self.assertEqual(in_use, set(ExposureGroup))

    def test_hand_assigned_sectors(self):
        # These deliberately disagree with russell1000.csv (Industrials / Information Technology).
        self.assertEqual(TradePair.UBER.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.APP.exposure_group, ExposureGroup.COMMUNICATION)
        # Sector ETFs, which the CSV does not cover at all.
        self.assertEqual(TradePair.XLK.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.VGT.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.VNQ.exposure_group, ExposureGroup.REAL_ESTATE)

    def test_values_are_the_csv_labels_verbatim(self):
        """The generator derives member names from these, so they must not be reformatted."""
        self.assertEqual(ExposureGroup.HEALTH_CARE.value, "Health Care")
        self.assertEqual(ExposureGroup.INFORMATION_TECHNOLOGY.value, "Information Technology")
        for group in ExposureGroup:
            self.assertEqual(group.value.upper().replace(" ", "_"), group.name)


# ---------------------------------------------------------------------------
# Pro account trade pair gating
# ---------------------------------------------------------------------------
class TestProTradePairGating(unittest.TestCase):
    """MinerAssetClass.can_trade(..., is_pro=True) must honor TradePair.is_pro."""

    def test_every_trade_pair_declares_is_pro(self):
        for tp in TradePair:
            self.assertIsInstance(tp.is_pro, bool, tp.trade_pair_id)

    def test_pro_cannot_trade_pairs_flagged_not_pro(self):
        for tp in TradePair:
            if tp.is_pro:
                continue
            for asset_class in MinerAssetClass:
                self.assertFalse(
                    asset_class.can_trade(tp, is_pro=True),
                    f"{asset_class.value} let a pro account trade {tp.trade_pair_id}",
                )

    def test_pro_universe_is_a_subset_of_the_standard_universe(self):
        for tp in TradePair:
            for asset_class in MinerAssetClass:
                if asset_class.can_trade(tp, is_pro=True):
                    self.assertTrue(asset_class.can_trade(tp, is_pro=False), tp.trade_pair_id)

    def test_representative_pairs(self):
        # HL crypto is open to pro; the Vanta-native equivalent is not.
        self.assertTrue(MinerAssetClass.CRYPTO.can_trade(TradePair.BTCUSDC, is_pro=True))
        self.assertFalse(MinerAssetClass.CRYPTO.can_trade(TradePair.BTCUSD, is_pro=True))
        # HL crypto below the capacity floor is excluded.
        self.assertFalse(MinerAssetClass.CRYPTO.can_trade(TradePair.AAVEUSDC, is_pro=True))
        self.assertTrue(MinerAssetClass.CRYPTO.can_trade(TradePair.AAVEUSDC, is_pro=False))
        # Equities: NVDA clears the ADV floor, VGT is a duplicate of the SPDR suite.
        self.assertTrue(MinerAssetClass.EQUITIES.can_trade(TradePair.NVDA, is_pro=True))
        self.assertFalse(MinerAssetClass.EQUITIES.can_trade(TradePair.VGT, is_pro=True))
        self.assertTrue(MinerAssetClass.EQUITIES.can_trade(TradePair.VGT, is_pro=False))
        # Forex carries no pro-specific exclusions.
        self.assertTrue(MinerAssetClass.FOREX.can_trade(TradePair.EURUSD, is_pro=True))

    def test_blocked_pairs_are_never_pro(self):
        for tp in TradePair:
            if tp.is_blocked:
                self.assertFalse(tp.is_pro, tp.trade_pair_id)


if __name__ == "__main__":
    unittest.main()
