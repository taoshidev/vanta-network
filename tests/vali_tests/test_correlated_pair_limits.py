"""
Unit tests for correlated-exposure limits (pro accounts only).

Covers TradePair.exposure_group resolution, the correlation-leg decomposition in
leverage_utils, and the resulting order-size caps for currency, sector, and US index groups.
"""

import unittest

from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.miner_account_manager import MinerAccount
from vali_objects.trade_pair import ExposureGroup, TradePair
from vali_objects.utils.leverage_utils import (
    compute_correlated_exposures,
    get_correlation_legs,
    get_max_correlated_order_size,
    get_max_order_size,
)
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.position import Position

# Matches MinerAccount.balance for an account with no collateral records, so positions sized
# off BALANCE line up with the account the caps are computed against.
BALANCE = ValiConfig.MIN_CAPITAL

# Broad-market and country ETFs deliberately belong to no sector.
NO_SECTOR_EQUITY_IDS = {
    "SPY", "QQQ", "DIA", "IWM", "EWU", "EWG", "EWJ", "EWH",
    "EWA", "EWQ", "EFA", "IEMG", "INDA", "VT",
}


def make_position(trade_pair: TradePair, leverage: float) -> Position:
    """Open position whose net value is `leverage` x BALANCE (negative leverage = short)."""
    net_value = leverage * BALANCE
    return Position(
        miner_hotkey="hk",
        position_uuid=f"{trade_pair.trade_pair_id}-{leverage}",
        open_ms=0,
        trade_pair=trade_pair,
        position_type=OrderType.LONG if leverage >= 0 else OrderType.SHORT,
        net_value=net_value,
        account_size=BALANCE,
    )


def make_account(bucket: MinerBucket) -> MinerAccount:
    return MinerAccount(
        miner_hotkey="hk",
        asset_class=MinerAssetClass.ALL_MARKETS,
        miner_bucket=bucket,
    )


class TestExposureGroupResolution(unittest.TestCase):

    def test_csv_derived_sectors(self):
        self.assertEqual(TradePair.AA.exposure_group, ExposureGroup.MATERIALS)
        self.assertEqual(TradePair.XOM.exposure_group, ExposureGroup.ENERGY)
        self.assertEqual(TradePair.PFE.exposure_group, ExposureGroup.HEALTH_CARE)

    def test_overrides_beat_the_csv(self):
        # The Russell export files these as Industrials / Information Technology.
        self.assertEqual(TradePair.UBER.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.APP.exposure_group, ExposureGroup.COMMUNICATION)

    def test_symbols_absent_from_the_csv(self):
        self.assertEqual(TradePair.TSM.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.BABA.exposure_group, ExposureGroup.CONSUMER_DISCRETIONARY)
        self.assertEqual(TradePair.SPCX.exposure_group, ExposureGroup.COMMUNICATION)

    def test_ticker_whose_display_name_differs_from_its_id(self):
        self.assertEqual(TradePair.BRK_B.trade_pair, "BRK.B")
        self.assertEqual(TradePair.BRK_B.exposure_group, ExposureGroup.FINANCIALS)

    def test_sector_etfs(self):
        self.assertEqual(TradePair.XLK.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.VGT.exposure_group, ExposureGroup.INFORMATION_TECHNOLOGY)
        self.assertEqual(TradePair.VNQ.exposure_group, ExposureGroup.REAL_ESTATE)
        self.assertEqual(TradePair.XLE.exposure_group, ExposureGroup.ENERGY)

    def test_broad_market_etfs_have_no_sector(self):
        for trade_pair_id in NO_SECTOR_EQUITY_IDS:
            with self.subTest(trade_pair_id=trade_pair_id):
                self.assertIsNone(TradePair[trade_pair_id].exposure_group)

    def test_hl_perp_matches_its_spot_twin(self):
        self.assertEqual(TradePair.NVDAUSDC.exposure_group, TradePair.NVDA.exposure_group)
        self.assertEqual(TradePair.COINUSDC.exposure_group, ExposureGroup.FINANCIALS)

    def test_non_equities_have_no_group(self):
        for trade_pair in (TradePair.EURUSD, TradePair.BTCUSD, TradePair.SP500USDC, TradePair.GOLDUSDC):
            with self.subTest(trade_pair=trade_pair):
                self.assertIsNone(trade_pair.exposure_group)

    def test_every_equity_is_grouped_or_known_ungrouped(self):
        ungrouped = {tp.trade_pair_id for tp in TradePair if tp.is_equities and tp.exposure_group is None}
        self.assertEqual(ungrouped, NO_SECTOR_EQUITY_IDS)


class TestCorrelationLegs(unittest.TestCase):

    def test_forex_splits_into_two_currency_legs(self):
        self.assertEqual(
            get_correlation_legs(TradePair.EURUSD),
            (("currency:EUR", 1.0), ("currency:USD", -1.0)),
        )
        self.assertEqual(
            get_correlation_legs(TradePair.EURJPY),
            (("currency:EUR", 1.0), ("currency:JPY", -1.0)),
        )

    def test_unlisted_currency_legs_are_skipped(self):
        self.assertEqual(get_correlation_legs(TradePair.USDMXN), (("currency:USD", 1.0),))
        self.assertEqual(get_correlation_legs(TradePair.XAUUSD), (("currency:USD", -1.0),))

    def test_equities_produce_a_sector_leg(self):
        self.assertEqual(get_correlation_legs(TradePair.NVDA), (("sector:Information Technology", 1.0),))
        self.assertEqual(get_correlation_legs(TradePair.XLK), (("sector:Information Technology", 1.0),))

    def test_us_index_group(self):
        for trade_pair in (TradePair.SPY, TradePair.QQQ, TradePair.IWM,
                           TradePair.DIA, TradePair.SP500USDC, TradePair.XYZ100USDC):
            with self.subTest(trade_pair=trade_pair):
                self.assertEqual(get_correlation_legs(trade_pair), (("index:us", 1.0),))

    def test_pairs_outside_every_group(self):
        for trade_pair in (TradePair.EWYUSDC, TradePair.BTCUSD, TradePair.EFA, TradePair.VT):
            with self.subTest(trade_pair=trade_pair):
                self.assertEqual(get_correlation_legs(trade_pair), ())

    def test_exposures_net_across_positions(self):
        exposures = compute_correlated_exposures([
            make_position(TradePair.EURUSD, 2.0),
            make_position(TradePair.EURGBP, -1.0),
        ])
        self.assertAlmostEqual(exposures["currency:EUR"], 1.0 * BALANCE)
        self.assertAlmostEqual(exposures["currency:USD"], -2.0 * BALANCE)
        self.assertAlmostEqual(exposures["currency:GBP"], 1.0 * BALANCE)


class TestCorrelatedOrderSize(unittest.TestCase):
    """Room left for an order, in USD, given the portfolio's existing exposure."""

    def room(self, trade_pair, open_positions, value_sign=1.0):
        return get_max_correlated_order_size(trade_pair, open_positions, BALANCE, value_sign)[0]

    def test_stacking_the_same_currency_is_capped(self):
        # Spec example 1: long EURUSD 20x + long EURJPY 15x is 35x net EUR, above the 30x limit.
        positions = [make_position(TradePair.EURUSD, 20.0)]
        self.assertAlmostEqual(self.room(TradePair.EURJPY, positions), 10.0 * BALANCE)

    def test_offsetting_the_same_currency_is_allowed(self):
        # Spec example 2: long EURUSD 20x + short EURGBP 10x is 10x net EUR.
        positions = [make_position(TradePair.EURUSD, 20.0)]
        # A short EURGBP reduces EUR, so EUR is not the binding leg; GBP is (30x from flat).
        self.assertAlmostEqual(self.room(TradePair.EURGBP, positions, value_sign=-1.0), 30.0 * BALANCE)

    def test_nzd_has_a_tighter_limit(self):
        positions = [make_position(TradePair.NZDCAD, 5.0)]
        # NZD room 20x - 5x = 15x; CAD room 30x + 5x = 35x.
        self.assertAlmostEqual(self.room(TradePair.NZDJPY, positions), 15.0 * BALANCE)

    def test_sector_exposure_is_capped(self):
        # Spec example: long NVDA 2x + long XLK 2x is 4x Information Technology, above 3x.
        positions = [make_position(TradePair.NVDA, 2.0)]
        self.assertAlmostEqual(self.room(TradePair.XLK, positions), 1.0 * BALANCE)

    def test_sectors_are_independent(self):
        positions = [make_position(TradePair.NVDA, 3.0)]
        self.assertAlmostEqual(self.room(TradePair.XLE, positions), 3.0 * BALANCE)

    def test_us_index_instruments_share_one_limit(self):
        positions = [
            make_position(TradePair.SPY, 10.0),
            make_position(TradePair.QQQ, 5.0),
            make_position(TradePair.SP500USDC, 5.0),
        ]
        self.assertAlmostEqual(self.room(TradePair.IWM, positions), 5.0 * BALANCE)

    def test_ewy_is_outside_the_index_group(self):
        positions = [make_position(TradePair.SPY, 25.0)]
        self.assertEqual(self.room(TradePair.EWYUSDC, positions), float("inf"))

    def test_reducing_order_that_grows_a_currency_is_capped(self):
        # Long EURUSD + long USDJPY nets USD -5x. Selling EURUSD pushes USD positive.
        positions = [
            make_position(TradePair.EURUSD, 30.0),
            make_position(TradePair.USDJPY, 25.0),
        ]
        # USD nets -30x + 25x = -5x and a sell moves it up, so USD binds at 30x + 5x = 35x
        # (the EUR leg would allow 60x: 30x of room down to its own -30x bound).
        self.assertAlmostEqual(self.room(TradePair.EURUSD, positions, value_sign=-1.0), 35.0 * BALANCE)

    def test_room_never_goes_negative(self):
        positions = [make_position(TradePair.NVDA, 10.0)]
        self.assertEqual(self.room(TradePair.XLK, positions), 0.0)


class TestGetMaxOrderSizeGating(unittest.TestCase):
    """Only pro accounts may be constrained by correlated exposure."""

    BREACHING = [
        make_position(TradePair.EURUSD, 30.0),
        make_position(TradePair.NVDA, 5.0),
        make_position(TradePair.SPY, 30.0),
    ]

    def max_size(self, bucket, trade_pair, open_positions, is_buy=True, value_sign=1.0):
        position = make_position(trade_pair, 0.0)
        return get_max_order_size(
            make_account(bucket), position,
            open_positions=open_positions, is_buy=is_buy, value_sign=value_sign,
        )[0]

    def test_non_pro_buckets_ignore_correlated_exposure(self):
        for bucket in (MinerBucket.MAINCOMP, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            with self.subTest(bucket=bucket):
                self.assertEqual(
                    self.max_size(bucket, TradePair.EURJPY, self.BREACHING),
                    self.max_size(bucket, TradePair.EURJPY, None),
                )

    def test_pro_buckets_apply_correlated_exposure(self):
        for bucket in (MinerBucket.SUBACCOUNT_PRO_CHALLENGE, MinerBucket.SUBACCOUNT_PRO_FUNDED):
            with self.subTest(bucket=bucket):
                self.assertLess(
                    self.max_size(bucket, TradePair.EURJPY, self.BREACHING),
                    self.max_size(bucket, TradePair.EURJPY, None),
                )

    def test_reducing_order_is_uncapped_without_correlation(self):
        # Non-buy on a non-pro account has no applicable cap at all.
        self.assertEqual(
            self.max_size(MinerBucket.SUBACCOUNT_FUNDED, TradePair.EURUSD, self.BREACHING, is_buy=False),
            float("inf"),
        )

    def test_reducing_order_on_pro_account_uses_correlated_cap_only(self):
        max_value = self.max_size(
            MinerBucket.SUBACCOUNT_PRO_FUNDED, TradePair.NVDA, self.BREACHING,
            is_buy=False, value_sign=-1.0,
        )
        # Technology is 5x long, and a sell moves it down, so there is 3x + 5x of room.
        self.assertAlmostEqual(max_value, (ValiConfig.PRO_SECTOR_EXPOSURE_LIMIT + 5.0) * BALANCE)


if __name__ == "__main__":
    unittest.main()
