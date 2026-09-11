"""
Unit tests for the standard subaccount leverage tiers (1 = Base, 2 = Boost I, 3 = Boost II).

Covers:
  * Table completeness — every tier has a value for every group, asset class and
    portfolio asset class; HL_ALL has no row.
  * Group resolution — every tradable pair maps to exactly one group; the explicit
    coin / id sets exist and sit in the expected category.
  * Values — match the Pro Launch spec §2a row by row and never decrease with tier.
  * Single-class portfolio cap equals the class cap.
  * Order path — get_max_order_size and MinerAccount.multiplier use the standard tables only
    for a non-HL, non-pro subaccount with a leverage_tier; everything else keeps the legacy curve.
"""

import unittest

from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount
from vali_objects.trade_pair import (
    BLOCKED_TRADE_PAIR_IDS,
    StandardLeverageGroup,
    TradePair,
    TradePairCategory,
)
from vali_objects.utils.leverage_utils import (
    get_effective_leverage_tier,
    get_legacy_leverage_tier,
    get_legacy_tier_positional_leverage,
    get_max_order_size,
    get_standard_class_leverage,
    get_standard_leverage_group,
    get_standard_portfolio_leverage,
    get_standard_positional_leverage,
    is_standard_tiered,
)
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.position import Position


TIERS = ValiConfig.STANDARD_LEVERAGE_TIERS
CATEGORIES = (
    TradePairCategory.CRYPTO,
    TradePairCategory.FOREX,
    TradePairCategory.EQUITIES,
    TradePairCategory.INDICES,
    TradePairCategory.COMMODITIES,
)
SINGLE_CLASS_ASSET_CLASSES = (
    (MinerAssetClass.CRYPTO, TradePairCategory.CRYPTO),
    (MinerAssetClass.FOREX, TradePairCategory.FOREX),
    (MinerAssetClass.EQUITIES, TradePairCategory.EQUITIES),
    (MinerAssetClass.COMMODITIES, TradePairCategory.COMMODITIES),
)


def _tradable_pairs():
    return [tp for tp in TradePair if tp.trade_pair_id not in BLOCKED_TRADE_PAIR_IDS]


class TestTableCompleteness(unittest.TestCase):

    def test_tiers_and_default(self):
        self.assertEqual(TIERS, (1, 2, 3))
        self.assertIn(ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT, TIERS)

    def test_positional_table_covers_every_group_for_every_tier(self):
        self.assertEqual(set(ValiConfig.STANDARD_POSITIONAL_LEVERAGE_BY_TIER), set(TIERS))
        for tier in TIERS:
            self.assertEqual(set(ValiConfig.STANDARD_POSITIONAL_LEVERAGE_BY_TIER[tier]), set(StandardLeverageGroup))

    def test_class_table_covers_every_category_for_every_tier(self):
        self.assertEqual(set(ValiConfig.STANDARD_CLASS_LEVERAGE_BY_TIER), set(TIERS))
        for tier in TIERS:
            self.assertEqual(set(ValiConfig.STANDARD_CLASS_LEVERAGE_BY_TIER[tier]), set(CATEGORIES))

    def test_portfolio_table_covers_standard_asset_classes_only(self):
        self.assertEqual(set(ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER), set(TIERS))
        expected = {ac for ac, _ in SINGLE_CLASS_ASSET_CLASSES} | {MinerAssetClass.ALL_MARKETS}
        for tier in TIERS:
            self.assertEqual(set(ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER[tier]), expected)
            self.assertNotIn(MinerAssetClass.HL_ALL, ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER[tier])

    def test_all_values_positive(self):
        for table in (
            ValiConfig.STANDARD_POSITIONAL_LEVERAGE_BY_TIER,
            ValiConfig.STANDARD_CLASS_LEVERAGE_BY_TIER,
            ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER,
        ):
            for tier, row in table.items():
                for key, value in row.items():
                    with self.subTest(tier=tier, key=key):
                        self.assertGreater(value, 0)


class TestGroupResolution(unittest.TestCase):

    def test_every_tradable_pair_resolves_and_has_a_value_at_every_tier(self):
        for tp in _tradable_pairs():
            with self.subTest(pair=tp.trade_pair_id):
                group = get_standard_leverage_group(tp)
                self.assertIsInstance(group, StandardLeverageGroup)
                for tier in TIERS:
                    self.assertGreater(get_standard_positional_leverage(tier, tp), 0)

    def test_every_group_is_used_by_some_tradable_pair(self):
        used = {get_standard_leverage_group(tp) for tp in _tradable_pairs()}
        self.assertEqual(used, set(StandardLeverageGroup))

    def test_crypto_majors_by_coin(self):
        majors = [tp for tp in _tradable_pairs() if get_standard_leverage_group(tp) == StandardLeverageGroup.CRYPTO_MAJORS]
        self.assertEqual({tp.base for tp in majors}, ValiConfig.STANDARD_CRYPTO_MAJOR_COINS)
        for tp in majors:
            self.assertEqual(tp.trade_pair_category, TradePairCategory.CRYPTO)
        self.assertEqual(get_standard_leverage_group(TradePair.ADAUSDC), StandardLeverageGroup.CRYPTO_OTHER)

    def test_nzd_crosses_are_forex_pairs_and_nzdusd_is_not_one(self):
        ids = {tp.trade_pair_id for tp in TradePair}
        for pair_id in ValiConfig.STANDARD_FX_NZD_CROSS_IDS:
            with self.subTest(pair=pair_id):
                self.assertIn(pair_id, ids)
                tp = TradePair.from_trade_pair_id(pair_id)
                self.assertEqual(tp.trade_pair_category, TradePairCategory.FOREX)
                self.assertEqual(get_standard_leverage_group(tp), StandardLeverageGroup.FX_NZD_CROSSES)
        self.assertEqual(len(ValiConfig.STANDARD_FX_NZD_CROSS_IDS), 6)
        self.assertEqual(get_standard_leverage_group(TradePair.NZDUSD), StandardLeverageGroup.FX)
        self.assertEqual(get_standard_leverage_group(TradePair.EURUSD), StandardLeverageGroup.FX)

    def test_indices_split(self):
        self.assertEqual(get_standard_leverage_group(TradePair.SP500USDC), StandardLeverageGroup.INDICES_US)
        self.assertEqual(get_standard_leverage_group(TradePair.XYZ100USDC), StandardLeverageGroup.INDICES_US)
        self.assertEqual(get_standard_leverage_group(TradePair.EWYUSDC), StandardLeverageGroup.INDICES_OTHER)

    def test_commodities_and_equities_are_whole_classes(self):
        for tp in _tradable_pairs():
            if tp.trade_pair_category == TradePairCategory.COMMODITIES:
                self.assertEqual(get_standard_leverage_group(tp), StandardLeverageGroup.COMMODITIES)
            if tp.trade_pair_category == TradePairCategory.EQUITIES:
                self.assertEqual(get_standard_leverage_group(tp), StandardLeverageGroup.EQUITIES)
        # HL equity perps use the equities row, same as Vanta equities.
        self.assertEqual(get_standard_leverage_group(TradePair.NVDAUSDC), StandardLeverageGroup.EQUITIES)


class TestValuesMatchSpec(unittest.TestCase):
    """Rows of the Pro Launch spec §2a, (tier 1, tier 2, tier 3)."""

    POSITIONAL = {
        TradePair.BTCUSDC:    (1.5, 2.0, 2.5),   # crypto majors
        TradePair.ADAUSDC:    (0.5, 0.75, 1.0),  # all other coins
        TradePair.EURUSD:     (10.0, 15.0, 20.0),
        TradePair.NZDUSD:     (10.0, 15.0, 20.0),
        TradePair.EURNZD:     (5.0, 7.5, 10.0),
        TradePair.SP500USDC:  (2.5, 4.0, 5.0),
        TradePair.EWYUSDC:    (1.0, 1.5, 2.0),
        TradePair.GOLDUSDC:   (1.5, 2.0, 3.0),
        TradePair.NVDA:       (0.5, 1.0, 1.5),
        TradePair.NVDAUSDC:   (0.5, 1.0, 1.5),
    }
    CLASS = {
        TradePairCategory.CRYPTO:      (1.5, 2.0, 2.5),
        TradePairCategory.FOREX:       (10.0, 15.0, 20.0),
        TradePairCategory.COMMODITIES: (1.5, 2.0, 3.0),
        TradePairCategory.INDICES:     (2.5, 4.0, 5.0),
        TradePairCategory.EQUITIES:    (1.0, 2.0, 3.0),
    }
    PORTFOLIO_ALL_MARKETS = (15.0, 20.0, 25.0)

    def test_positional_values(self):
        for tp, expected in self.POSITIONAL.items():
            for tier, value in zip(TIERS, expected):
                with self.subTest(pair=tp.trade_pair_id, tier=tier):
                    self.assertEqual(get_standard_positional_leverage(tier, tp), value)

    def test_class_values(self):
        for cat, expected in self.CLASS.items():
            for tier, value in zip(TIERS, expected):
                with self.subTest(category=cat, tier=tier):
                    self.assertEqual(get_standard_class_leverage(tier, cat), value)

    def test_all_markets_portfolio_values(self):
        for tier, value in zip(TIERS, self.PORTFOLIO_ALL_MARKETS):
            with self.subTest(tier=tier):
                self.assertEqual(get_standard_portfolio_leverage(tier, MinerAssetClass.ALL_MARKETS), value)

    def test_single_class_portfolio_cap_equals_class_cap(self):
        for asset_class, category in SINGLE_CLASS_ASSET_CLASSES:
            for tier in TIERS:
                with self.subTest(asset_class=asset_class, tier=tier):
                    self.assertEqual(
                        get_standard_portfolio_leverage(tier, asset_class),
                        get_standard_class_leverage(tier, category),
                    )

    def test_values_never_decrease_with_tier(self):
        for tp in _tradable_pairs():
            values = [get_standard_positional_leverage(tier, tp) for tier in TIERS]
            with self.subTest(pair=tp.trade_pair_id):
                self.assertEqual(values, sorted(values))
        for cat in CATEGORIES:
            values = [get_standard_class_leverage(tier, cat) for tier in TIERS]
            self.assertEqual(values, sorted(values))
        for asset_class in ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER[1]:
            values = [get_standard_portfolio_leverage(tier, asset_class) for tier in TIERS]
            self.assertEqual(values, sorted(values))

    def test_pair_cap_never_exceeds_its_class_cap(self):
        for tp in _tradable_pairs():
            for tier in TIERS:
                with self.subTest(pair=tp.trade_pair_id, tier=tier):
                    self.assertLessEqual(
                        get_standard_positional_leverage(tier, tp),
                        get_standard_class_leverage(tier, tp.trade_pair_category),
                    )

    def test_portfolio_lookup_falls_back_to_1x_for_unknown_asset_class(self):
        self.assertEqual(get_standard_portfolio_leverage(1, MinerAssetClass.HL_ALL), 1.0)


class TestStandardTierOrderPath(unittest.TestCase):
    """get_max_order_size and MinerAccount.multiplier switch tables on is_standard_tiered."""

    HL_ADDRESS = "0x" + "a" * 40
    SIZE = 100_000.0
    STANDARD_BUCKETS = (
        MinerBucket.SUBACCOUNT_CHALLENGE,
        MinerBucket.SUBACCOUNT_FUNDED,
        MinerBucket.PRO_CHALLENGE_TRANSITION,
    )

    def _account(self, bucket, asset_class, leverage_tier=None, hl_address=None, capital_used=0.0,
                 capital_used_by_class=None):
        account = MinerAccount(
            miner_hotkey="ent_0", asset_class=asset_class, miner_bucket=bucket,
            hl_address=hl_address, leverage_tier=leverage_tier, capital_used=capital_used,
            capital_used_by_class=capital_used_by_class or {},
        )
        account.add_collateral_record(CollateralRecord(self.SIZE, self.SIZE / 5000, 0, is_first_record=True))
        return account

    def _position(self, trade_pair):
        return Position(
            miner_hotkey="ent_0", position_uuid="u", open_ms=0, trade_pair=trade_pair,
            position_type=OrderType.LONG, account_size=self.SIZE,
        )

    # ---- is_standard_tiered ----

    def test_is_standard_tiered(self):
        for bucket in self.STANDARD_BUCKETS:
            self.assertTrue(is_standard_tiered(self._account(bucket, MinerAssetClass.CRYPTO, leverage_tier=1)))
            # no stored tier still counts as standard (default tier)
            self.assertTrue(is_standard_tiered(self._account(bucket, MinerAssetClass.CRYPTO)))
        self.assertTrue(is_standard_tiered(self._account(None, MinerAssetClass.CRYPTO, leverage_tier=1)))
        # no bucket and no tier: nothing says it is a subaccount, so legacy
        self.assertFalse(is_standard_tiered(self._account(None, MinerAssetClass.CRYPTO)))
        # HL-linked, pro, regular -> legacy, with or without a stored tier
        for tier in (None, 1):
            self.assertFalse(is_standard_tiered(
                self._account(MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.HL_ALL, leverage_tier=tier, hl_address=self.HL_ADDRESS)
            ))
            for bucket in (MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED):
                self.assertFalse(is_standard_tiered(self._account(bucket, MinerAssetClass.CRYPTO, leverage_tier=tier)))
        self.assertFalse(is_standard_tiered(self._account(MinerBucket.MAINCOMP, MinerAssetClass.CRYPTO)))

    def test_hl_all_account_without_hl_address_stays_legacy(self):
        # MinerAccounts older than the hl_address field: HL_ALL alone must keep the legacy curve.
        for bucket in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED):
            with self.subTest(bucket=bucket):
                account = self._account(bucket, MinerAssetClass.HL_ALL)
                self.assertFalse(is_standard_tiered(account))
                legacy_tier = get_legacy_leverage_tier(bucket, self.SIZE)
                self.assertEqual(
                    account.multiplier,
                    ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[legacy_tier][MinerAssetClass.HL_ALL],
                )

    def test_effective_tier_defaults_when_not_stored(self):
        account = self._account(MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.CRYPTO)
        self.assertEqual(get_effective_leverage_tier(account), ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT)
        account = self._account(MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.CRYPTO, leverage_tier=3)
        self.assertEqual(get_effective_leverage_tier(account), 3)

    def test_standard_subaccount_without_tier_uses_default_tier(self):
        default = ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT
        for bucket in self.STANDARD_BUCKETS:
            with self.subTest(bucket=bucket):
                account = self._account(bucket, MinerAssetClass.ALL_MARKETS)
                self.assertEqual(account.multiplier, get_standard_portfolio_leverage(default, MinerAssetClass.ALL_MARKETS))
                max_value, label = get_max_order_size(account, self._position(TradePair.BTCUSDC))
                self.assertAlmostEqual(max_value, self.SIZE * get_standard_positional_leverage(default, TradePair.BTCUSDC), places=2)
                self.assertIn("per pair cap", label)

    # ---- standard tiers ----

    def test_per_pair_cap_binds_at_every_tier_and_bucket(self):
        for tier in TIERS:
            for bucket in self.STANDARD_BUCKETS:
                account = self._account(bucket, MinerAssetClass.ALL_MARKETS, leverage_tier=tier)
                for pair in (TradePair.BTCUSDC, TradePair.ADAUSDC, TradePair.EURUSD, TradePair.EURNZD,
                             TradePair.EWYUSDC, TradePair.GOLDUSDC, TradePair.NVDA):
                    with self.subTest(tier=tier, bucket=bucket, pair=pair.trade_pair_id):
                        max_value, label = get_max_order_size(account, self._position(pair))
                        expected = self.SIZE * get_standard_positional_leverage(tier, pair)
                        self.assertAlmostEqual(max_value, expected, places=2)
                        self.assertIn("per pair cap", label)

    def test_multiplier_uses_standard_portfolio_table(self):
        for tier in TIERS:
            for asset_class in (MinerAssetClass.ALL_MARKETS, MinerAssetClass.CRYPTO, MinerAssetClass.FOREX,
                                MinerAssetClass.EQUITIES, MinerAssetClass.COMMODITIES):
                with self.subTest(tier=tier, asset_class=asset_class):
                    account = self._account(MinerBucket.SUBACCOUNT_FUNDED, asset_class, leverage_tier=tier)
                    self.assertEqual(account.multiplier, get_standard_portfolio_leverage(tier, asset_class))

    def test_class_cap_binds_when_class_exposure_is_high(self):
        # tier 1 all_markets: crypto class cap 1.5x = 150K; 120K already in crypto leaves 30K,
        # tighter than BTC's per-pair 150K and the 1.5M portfolio cap.
        account = self._account(
            MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.ALL_MARKETS, leverage_tier=1,
            capital_used=120_000.0, capital_used_by_class={TradePairCategory.CRYPTO: 120_000.0},
        )
        max_value, label = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        self.assertAlmostEqual(max_value, 30_000.0, places=2)
        self.assertIn("per class cap crypto 1.5x", label)

    def test_portfolio_cap_binds_when_total_exposure_is_high(self):
        # tier 1 all_markets: portfolio cap 15x = 1.5M; 1.49M used leaves 10K.
        account = self._account(
            MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.ALL_MARKETS, leverage_tier=1,
            capital_used=1_490_000.0, capital_used_by_class={TradePairCategory.FOREX: 1_490_000.0},
        )
        max_value, label = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        self.assertLessEqual(max_value, 10_000.0)
        self.assertIn("overall portfolio cap 15.0x", label)

    # ---- legacy curve untouched ----

    def test_hl_and_pro_accounts_keep_legacy_values(self):
        cases = (
            (MinerBucket.SUBACCOUNT_CHALLENGE, MinerAssetClass.HL_ALL, self.HL_ADDRESS),
            (MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.HL_ALL, self.HL_ADDRESS),
            (MinerBucket.PRO_CHALLENGE_DIRECT, MinerAssetClass.ALL_MARKETS, None),
            (MinerBucket.PRO_FUNDED, MinerAssetClass.ALL_MARKETS, None),
        )
        for bucket, asset_class, hl in cases:
            with self.subTest(bucket=bucket, asset_class=asset_class):
                account = self._account(bucket, asset_class, hl_address=hl)
                legacy_tier = get_legacy_leverage_tier(bucket, self.SIZE)
                self.assertEqual(
                    account.multiplier,
                    ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[legacy_tier][asset_class],
                )
                max_value, _ = get_max_order_size(account, self._position(TradePair.BTCUSDC))
                expected = self.SIZE * get_legacy_tier_positional_leverage(legacy_tier, TradePair.BTCUSDC)
                self.assertAlmostEqual(max_value, expected, places=2)

    def test_pro_promoted_account_ignores_its_old_tier(self):
        account = self._account(MinerBucket.PRO_FUNDED, MinerAssetClass.CRYPTO, leverage_tier=3)
        legacy_tier = get_legacy_leverage_tier(MinerBucket.PRO_FUNDED, self.SIZE)
        self.assertEqual(
            account.multiplier,
            ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[legacy_tier][MinerAssetClass.CRYPTO],
        )
        max_value, _ = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        expected = self.SIZE * get_legacy_tier_positional_leverage(legacy_tier, TradePair.BTCUSDC)
        self.assertAlmostEqual(max_value, expected, places=2)

    def test_regular_miner_unchanged(self):
        account = self._account(MinerBucket.MAINCOMP, MinerAssetClass.CRYPTO)
        max_value, label = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        self.assertAlmostEqual(max_value, min(self.SIZE * TradePair.BTCUSDC.max_leverage, account.buying_power), places=2)


if __name__ == "__main__":
    unittest.main()
