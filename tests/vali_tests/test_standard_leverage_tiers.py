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
  * Tier 0 — a standard subaccount with no stored tier gets max(its legacy value, Base) on every
    per-pair, class and portfolio limit; the legacy side follows its bucket and account size.
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
    get_grandfathered_class_leverage,
    get_grandfathered_portfolio_leverage,
    get_grandfathered_positional_leverage,
    get_grandfathered_tier_key,
    get_legacy_leverage_tier,
    get_legacy_tier_positional_leverage,
    get_max_order_size,
    get_max_position_leverage,
    get_per_class_leverage_cap,
    get_standard_account_tier_key,
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
        self.assertEqual(ValiConfig.STANDARD_LEVERAGE_TIER_BASE, 1)
        self.assertIn(ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT, TIERS)
        # tier 0 exists only as the pre-tier floor; no endpoint may select it
        self.assertEqual(ValiConfig.STANDARD_LEVERAGE_TIER_GRANDFATHERED, 0)
        self.assertNotIn(ValiConfig.STANDARD_LEVERAGE_TIER_GRANDFATHERED, TIERS)
        self.assertFalse(ValiConfig.is_valid_standard_leverage_tier(ValiConfig.STANDARD_LEVERAGE_TIER_GRANDFATHERED))

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
        TradePairCategory.INDICES:     (3.0, 6.0, 8.0),  # raised from the spec's 2.5 / 4 / 5 on 2026-09-17
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


class TestTier0Floor(unittest.TestCase):
    """get_grandfathered_*: max(legacy value at the account's legacy tier, Base), row by row."""

    LEGACY_TIERS = (1, 2, 3, 4)
    BASE = ValiConfig.STANDARD_LEVERAGE_TIER_BASE

    def test_per_pair_floor(self):
        for legacy_tier in self.LEGACY_TIERS:
            for pair in _tradable_pairs():
                with self.subTest(legacy_tier=legacy_tier, pair=pair.trade_pair_id):
                    legacy = get_legacy_tier_positional_leverage(legacy_tier, pair)
                    base = get_standard_positional_leverage(self.BASE, pair)
                    self.assertEqual(get_grandfathered_positional_leverage(legacy_tier, pair), max(legacy, base))

    def test_class_and_portfolio_floor(self):
        for legacy_tier in self.LEGACY_TIERS:
            for category in CATEGORIES:
                with self.subTest(legacy_tier=legacy_tier, category=category):
                    legacy = ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[legacy_tier][category]
                    base = get_standard_class_leverage(self.BASE, category)
                    self.assertEqual(get_grandfathered_class_leverage(legacy_tier, category), max(legacy, base))
            for asset_class in ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER[self.BASE]:
                with self.subTest(legacy_tier=legacy_tier, asset_class=asset_class):
                    legacy = ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[legacy_tier][asset_class]
                    base = get_standard_portfolio_leverage(self.BASE, asset_class)
                    self.assertEqual(get_grandfathered_portfolio_leverage(legacy_tier, asset_class), max(legacy, base))

    def test_floor_pair_cap_never_exceeds_floor_class_cap(self):
        for legacy_tier in self.LEGACY_TIERS:
            for pair in _tradable_pairs():
                self.assertLessEqual(
                    get_grandfathered_positional_leverage(legacy_tier, pair),
                    get_grandfathered_class_leverage(legacy_tier, pair.trade_pair_category),
                    f"legacy tier {legacy_tier} {pair.trade_pair_id}",
                )

    def test_floor_class_cap_never_exceeds_floor_portfolio_cap(self):
        for legacy_tier in self.LEGACY_TIERS:
            for asset_class, category in SINGLE_CLASS_ASSET_CLASSES:
                self.assertLessEqual(get_grandfathered_class_leverage(legacy_tier, category),
                                     get_grandfathered_portfolio_leverage(legacy_tier, asset_class))
            for category in CATEGORIES:
                self.assertLessEqual(get_grandfathered_class_leverage(legacy_tier, category),
                                     get_grandfathered_portfolio_leverage(legacy_tier, MinerAssetClass.ALL_MARKETS))

    def test_tier_key_is_minus_the_legacy_tier_and_never_a_selectable_tier(self):
        for legacy_tier in self.LEGACY_TIERS:
            key = get_grandfathered_tier_key(legacy_tier)
            self.assertEqual(key, -legacy_tier)
            self.assertNotIn(key, ValiConfig.STANDARD_LEVERAGE_TIERS)
            self.assertFalse(ValiConfig.is_valid_standard_leverage_tier(key))
        self.assertEqual(ValiConfig.LEGACY_LEVERAGE_TIERS, self.LEGACY_TIERS)

    def test_rows_the_rollout_would_have_lowered(self):
        # funded < $200K (legacy tier 2): Base cut these, the floor keeps them ...
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.NVDA), 1.0)
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.GOLDUSDC), 2.0)
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.SP500USDC), 3.0)
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.EWYUSDC), 3.0)
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.ADAUSDC), 1.0)
        self.assertEqual(get_grandfathered_class_leverage(2, TradePairCategory.EQUITIES), 1.5)
        self.assertEqual(get_grandfathered_class_leverage(2, TradePairCategory.INDICES), 6.0)
        # ... and Base still lifts the rows it raised
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.EURUSD), 10.0)
        self.assertEqual(get_grandfathered_positional_leverage(2, TradePair.BTCUSDC), 1.5)
        self.assertEqual(get_grandfathered_portfolio_leverage(2, MinerAssetClass.ALL_MARKETS), 15.0)
        # challenge (legacy tier 1) is Base except EWY and the crypto / commodities class caps
        self.assertEqual(get_grandfathered_positional_leverage(1, TradePair.NVDA), 0.5)
        self.assertEqual(get_grandfathered_positional_leverage(1, TradePair.EWYUSDC), 1.5)
        self.assertEqual(get_grandfathered_class_leverage(1, TradePairCategory.CRYPTO), 2.0)
        self.assertEqual(get_grandfathered_class_leverage(1, TradePairCategory.INDICES), 3.0)
        self.assertEqual(get_grandfathered_class_leverage(1, TradePairCategory.COMMODITIES), 2.0)


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
                 capital_used_by_class=None, size=None):
        size = self.SIZE if size is None else size
        account = MinerAccount(
            miner_hotkey="ent_0", asset_class=asset_class, miner_bucket=bucket,
            hl_address=hl_address, leverage_tier=leverage_tier, capital_used=capital_used,
            capital_used_by_class=capital_used_by_class or {},
        )
        account.add_collateral_record(CollateralRecord(size, size / 5000, 0, is_first_record=True))
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
            # no stored tier still counts as standard (tier 0)
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

    def test_effective_tier_is_0_when_not_stored_and_reported_as_minus_legacy_tier(self):
        account = self._account(MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.CRYPTO)  # $100K funded: legacy tier 2
        self.assertEqual(get_effective_leverage_tier(account), ValiConfig.STANDARD_LEVERAGE_TIER_GRANDFATHERED)
        self.assertEqual(get_standard_account_tier_key(account), -2)
        self.assertEqual(account.leverage_limits(), {
            "is_pro": False, "tier_curve": "standard", "tier": -2, "portfolio_multiplier": account.multiplier,
        })
        account = self._account(MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.CRYPTO, leverage_tier=3)
        self.assertEqual(get_effective_leverage_tier(account), 3)
        self.assertEqual(get_standard_account_tier_key(account), 3)
        self.assertEqual(account.leverage_limits()["tier"], 3)

    # (bucket, account size) -> the legacy tier the tier 0 floor is taken against
    TIER_0_CASES = (
        (MinerBucket.SUBACCOUNT_CHALLENGE, 100_000.0, 1),
        (MinerBucket.SUBACCOUNT_FUNDED, 100_000.0, 2),
        (MinerBucket.PRO_CHALLENGE_TRANSITION, 100_000.0, 2),
        (MinerBucket.SUBACCOUNT_FUNDED, 250_000.0, 3),
        (MinerBucket.SUBACCOUNT_FUNDED, 1_500_000.0, 4),
    )
    TIER_0_PAIRS = (TradePair.BTCUSDC, TradePair.ADAUSDC, TradePair.EURUSD, TradePair.EURNZD, TradePair.SP500USDC,
                    TradePair.EWYUSDC, TradePair.GOLDUSDC, TradePair.WTIOILUSDC, TradePair.NVDA)

    def test_standard_subaccount_without_tier_trades_the_tier_0_floor(self):
        base = ValiConfig.STANDARD_LEVERAGE_TIER_BASE
        for bucket, size, legacy_tier in self.TIER_0_CASES:
            account = self._account(bucket, MinerAssetClass.ALL_MARKETS, size=size)
            self.assertEqual(get_legacy_leverage_tier(bucket, size), legacy_tier)
            with self.subTest(bucket=bucket, size=size, limit="tier key"):
                self.assertEqual(account.leverage_limits()["tier"], -legacy_tier)
            with self.subTest(bucket=bucket, size=size, limit="portfolio"):
                expected = max(ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[legacy_tier][MinerAssetClass.ALL_MARKETS],
                               get_standard_portfolio_leverage(base, MinerAssetClass.ALL_MARKETS))
                self.assertEqual(account.multiplier, expected)
                self.assertEqual(account.multiplier, get_grandfathered_portfolio_leverage(legacy_tier, MinerAssetClass.ALL_MARKETS))
            for category in CATEGORIES:
                with self.subTest(bucket=bucket, size=size, limit=f"class {category.value}"):
                    expected = max(ValiConfig.LEGACY_TIER_PORTFOLIO_LEVERAGE_BY_CATEGORY[legacy_tier][category],
                                   get_standard_class_leverage(base, category))
                    self.assertEqual(get_per_class_leverage_cap(account, category), expected)
            for pair in self.TIER_0_PAIRS:
                with self.subTest(bucket=bucket, size=size, pair=pair.trade_pair_id):
                    expected = max(get_legacy_tier_positional_leverage(legacy_tier, pair),
                                   get_standard_positional_leverage(base, pair))
                    self.assertEqual(get_max_position_leverage(account, pair), expected)
                    max_value, label = get_max_order_size(account, self._position(pair))
                    self.assertAlmostEqual(max_value, size * expected, places=2)
                    self.assertIn("per pair cap", label)

    def test_tier_0_never_lowers_a_limit_and_is_never_below_base(self):
        base = ValiConfig.STANDARD_LEVERAGE_TIER_BASE
        for bucket, size, legacy_tier in self.TIER_0_CASES:
            account = self._account(bucket, MinerAssetClass.ALL_MARKETS, size=size)
            for pair in _tradable_pairs():
                floor = get_max_position_leverage(account, pair)
                self.assertGreaterEqual(floor, get_legacy_tier_positional_leverage(legacy_tier, pair), pair.trade_pair_id)
                self.assertGreaterEqual(floor, get_standard_positional_leverage(base, pair), pair.trade_pair_id)

    def test_tier_0_class_cap_binds(self):
        # funded < $200K, all_markets: crypto class floor is max(legacy 2.0, Base 1.5) = 2.0x = 200K;
        # 170K already in crypto leaves 30K, tighter than BTC's per-pair max(1.0, 1.5) = 150K.
        account = self._account(
            MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.ALL_MARKETS,
            capital_used=170_000.0, capital_used_by_class={TradePairCategory.CRYPTO: 170_000.0},
        )
        max_value, label = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        self.assertAlmostEqual(max_value, 30_000.0, places=2)
        self.assertIn("per class cap crypto 2.0x", label)

    def test_tier_0_floor_follows_the_same_account_through_promotion_and_size(self):
        account = self._account(MinerBucket.SUBACCOUNT_CHALLENGE, MinerAssetClass.ALL_MARKETS)
        # challenge, legacy tier 1: NVDA is Base 0.5, the indices class cap is legacy 3.0
        self.assertEqual(get_max_position_leverage(account, TradePair.NVDA), 0.5)
        self.assertEqual(get_per_class_leverage_cap(account, TradePairCategory.INDICES), 3.0)
        self.assertEqual(account.multiplier, 15.0)
        # promoted, legacy tier 2: the same account's floor rises
        account.miner_bucket = MinerBucket.SUBACCOUNT_FUNDED
        self.assertEqual(get_max_position_leverage(account, TradePair.NVDA), 1.0)
        self.assertEqual(get_per_class_leverage_cap(account, TradePairCategory.INDICES), 6.0)
        self.assertEqual(account.multiplier, 15.0)
        # scaled to $250K, legacy tier 3
        account.add_collateral_record(CollateralRecord(250_000.0, 250_000.0 / 5000, 1))
        self.assertEqual(get_max_position_leverage(account, TradePair.NVDA), 1.5)
        self.assertEqual(get_per_class_leverage_cap(account, TradePairCategory.INDICES), 8.0)
        self.assertEqual(account.multiplier, 18.0)

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

    def test_hl_accounts_keep_legacy_values(self):
        cases = (
            (MinerBucket.SUBACCOUNT_CHALLENGE, MinerAssetClass.HL_ALL, self.HL_ADDRESS),
            (MinerBucket.SUBACCOUNT_FUNDED, MinerAssetClass.HL_ALL, self.HL_ADDRESS),
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

    def test_pro_accounts_use_the_flat_pro_tables(self):
        """Pro runs neither curve in this file: its own flat table, with no tier dimension."""
        from vali_objects.utils.leverage_utils import get_pro_positional_leverage

        for bucket in (MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED):
            with self.subTest(bucket=bucket):
                account = self._account(bucket, MinerAssetClass.ALL_MARKETS)
                self.assertEqual(account.multiplier, ValiConfig.PRO_PORTFOLIO_LEVERAGE)
                max_value, _ = get_max_order_size(account, self._position(TradePair.BTCUSDC))
                expected = self.SIZE * get_pro_positional_leverage(TradePair.BTCUSDC)
                self.assertAlmostEqual(max_value, expected, places=2)

    def test_pro_promoted_account_ignores_its_old_tier(self):
        """A subaccount promoted to pro keeps a stored leverage_tier that must not be applied."""
        from vali_objects.utils.leverage_utils import get_pro_positional_leverage

        account = self._account(MinerBucket.PRO_FUNDED, MinerAssetClass.CRYPTO, leverage_tier=3)
        self.assertEqual(account.multiplier, ValiConfig.PRO_PORTFOLIO_LEVERAGE)
        max_value, _ = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        expected = self.SIZE * get_pro_positional_leverage(TradePair.BTCUSDC)
        self.assertAlmostEqual(max_value, expected, places=2)

    def test_regular_miner_unchanged(self):
        account = self._account(MinerBucket.MAINCOMP, MinerAssetClass.CRYPTO)
        max_value, label = get_max_order_size(account, self._position(TradePair.BTCUSDC))
        self.assertAlmostEqual(max_value, min(self.SIZE * TradePair.BTCUSDC.max_leverage, account.buying_power), places=2)


if __name__ == "__main__":
    unittest.main()
