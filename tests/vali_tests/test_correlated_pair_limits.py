"""
Unit tests for correlated-exposure limits (pro accounts only).

Covers TradePair.exposure_group resolution, the correlation-leg decomposition in leverage_utils,
and the resulting order-size caps for currency, sector, and US index groups. Each group caps its
gross long and gross short exposure separately, and only orders that open or grow a position are
checked at all.
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

    def test_exposures_track_each_side_separately(self):
        # Long EURUSD 2x is +2x EUR / -2x USD; short EURGBP 1x is -1x EUR / +1x GBP. EUR therefore
        # carries 2x on its long side and 1x on its short side rather than netting to 1x.
        exposures = compute_correlated_exposures([
            make_position(TradePair.EURUSD, 2.0),
            make_position(TradePair.EURGBP, -1.0),
        ])
        eur_long, eur_short = exposures["currency:EUR"]
        self.assertAlmostEqual(eur_long, 2.0 * BALANCE)
        self.assertAlmostEqual(eur_short, 1.0 * BALANCE)

        usd_long, usd_short = exposures["currency:USD"]
        self.assertAlmostEqual(usd_long, 0.0)
        self.assertAlmostEqual(usd_short, 2.0 * BALANCE)

        gbp_long, gbp_short = exposures["currency:GBP"]
        self.assertAlmostEqual(gbp_long, 1.0 * BALANCE)
        self.assertAlmostEqual(gbp_short, 0.0)


class TestCorrelatedOrderSize(unittest.TestCase):
    """Room left for an order, in USD, given the portfolio's existing exposure."""

    def room(self, trade_pair, open_positions, position_type=OrderType.LONG):
        """Room for an order that opens or grows a `position_type` position in `trade_pair`."""
        return get_max_correlated_order_size(trade_pair, open_positions, BALANCE, position_type)[0]

    def test_stacking_the_same_currency_is_capped(self):
        # Spec example 1: long EURUSD 20x + long EURJPY 15x is 35x gross long EUR, above the 30x limit.
        positions = [make_position(TradePair.EURUSD, 20.0)]
        self.assertAlmostEqual(self.room(TradePair.EURJPY, positions), 10.0 * BALANCE)

    def test_long_and_short_sides_both_get_a_full_allowance(self):
        positions = [make_position(TradePair.EURUSD, 20.0)]
        # A short EURGBP lands on EUR's short side, which is empty, and on GBP's long side, which
        # is also empty. The 20x already sitting on EUR's long side does not enter into it.
        self.assertAlmostEqual(
            self.room(TradePair.EURGBP, positions, position_type=OrderType.SHORT), 30.0 * BALANCE
        )

    def test_nzd_has_a_tighter_limit(self):
        positions = [make_position(TradePair.NZDCAD, 5.0)]
        # NZDJPY legs are NZD and JPY: NZD long side has 5x of 20x used, leaving 15x; JPY's short
        # side is empty, leaving its full 30x.
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

    def test_opposing_sides_of_a_group_do_not_share_room(self):
        # Long EURUSD 30x maxes out USD's *short* side. That buys no extra room on the long side:
        # a long-USD order still gets exactly the 30x allowance, not 60x. USDMXN has a single USD
        # leg (MXN is unlisted), so USD is unambiguously the binding group.
        positions = [make_position(TradePair.EURUSD, 30.0)]
        self.assertAlmostEqual(self.room(TradePair.USDMXN, positions), 30.0 * BALANCE)

    def test_each_side_of_a_group_gets_its_own_limit(self):
        # USD ends up 30x short (from long EURUSD) and 30x long (from long USDMXN) at the same
        # time, which is allowed. Both sides are now full, so neither direction has room left.
        positions = [
            make_position(TradePair.EURUSD, 30.0),
            make_position(TradePair.USDMXN, 30.0),
        ]
        self.assertAlmostEqual(self.room(TradePair.USDMXN, positions), 0.0)  # adds to USD long
        self.assertAlmostEqual(self.room(TradePair.XAUUSD, positions), 0.0)  # long XAUUSD is short USD

    def test_short_positions_fill_the_short_side_of_a_sector(self):
        # Short NVDA 3x fills the short side of Information Technology and leaves the long side
        # untouched, so a long XLK still has the full 3x.
        positions = [make_position(TradePair.NVDA, -3.0)]
        self.assertAlmostEqual(self.room(TradePair.XLK, positions, position_type=OrderType.SHORT), 0.0)
        self.assertAlmostEqual(self.room(TradePair.XLK, positions), 3.0 * BALANCE)

    def test_stacking_the_same_currency_short_is_capped(self):
        # Mirror of test_stacking_the_same_currency_is_capped. A short EURUSD is short EUR, so a
        # further short EURJPY stacks the same bet and binds on EUR's short side, not its long one.
        positions = [make_position(TradePair.EURUSD, -20.0)]
        self.assertAlmostEqual(
            self.room(TradePair.EURJPY, positions, position_type=OrderType.SHORT), 10.0 * BALANCE
        )

    def test_short_position_fills_the_long_side_of_the_quote_currency(self):
        # A short EURUSD is long USD. USDMXN has a single USD leg (MXN is unlisted), so a long
        # USDMXN order binds against USD's long side, which is already full at 30x.
        positions = [make_position(TradePair.EURUSD, -30.0)]
        self.assertAlmostEqual(self.room(TradePair.USDMXN, positions), 0.0)

    def test_short_order_on_a_single_leg_pair_uses_the_long_side(self):
        # A long XAUUSD is short USD, so a short XAUUSD is long USD. The held long EURUSD only
        # fills USD's short side, leaving the long side's full allowance for this order.
        positions = [make_position(TradePair.EURUSD, 30.0)]
        self.assertAlmostEqual(
            self.room(TradePair.XAUUSD, positions, position_type=OrderType.SHORT), 30.0 * BALANCE
        )

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

    def max_size(self, bucket, trade_pair, open_positions):
        position = make_position(trade_pair, 0.0)
        return get_max_order_size(make_account(bucket), position, open_positions=open_positions)[0]

    def binding_cap(self, bucket, trade_pair, open_positions):
        position = make_position(trade_pair, 0.0)
        return get_max_order_size(make_account(bucket), position, open_positions=open_positions)[1]

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

    def test_correlated_cap_cannot_bind_without_open_positions(self):
        # market_order_manager fetches open_positions only for orders that open or grow a
        # position, so a reducing order reaches this function as None. The correlated cap then
        # cannot bind, even on a pro account whose groups are already breaching.
        self.assertNotIn(
            "exposure cap",
            self.binding_cap(MinerBucket.SUBACCOUNT_PRO_FUNDED, TradePair.NVDA, None),
        )

    def test_correlated_cap_binds_when_open_positions_are_supplied(self):
        # Counterpart to the above: BREACHING holds NVDA 5x long against a 3x sector limit, so
        # the correlated cap is the binding one once the positions are passed in.
        self.assertIn(
            "exposure cap",
            self.binding_cap(MinerBucket.SUBACCOUNT_PRO_FUNDED, TradePair.NVDA, self.BREACHING),
        )


if __name__ == "__main__":
    unittest.main()
