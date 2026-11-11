"""
The fast track of a standard miner, and of a subaccount created from ValiConfig.SUBACCOUNT_PROMOTION_RULES_START_MS on:
3% in the first week from the first order as a funded account, 5% in the second, then 10%, and 3% in any week once
the account's leverage use has reached ValiConfig.AUTO_PROMOTION_FAST_TRACK_LEVERAGE_USE. Leverage use is the larger
of the open positions' entry value over the portfolio cap and the highest asset class's over that class's cap, each
on the balance; its peak is taken after every fill and fee and starts over with every new account.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from entity_management.entity_client import EntityClient
from entity_management.entity_server import EntityServer
from tests.vali_tests.test_auto_promotion import _holds, _standard_miner
from tests.vali_tests.test_auto_promotion_stats import _account, _seed
from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager is a fixture
    ACCOUNT_SIZE,
    DAILY_MS,
    HOTKEY,
    NOW_MS,
    manager,
)
from tests.vali_tests.test_miner_account_lock_order import _MinerAccountFilesInTempDir
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import AutoPromotionStats
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount, MinerAccountManager
from vali_objects.vali_config import RPCConnectionMode, TradePairCategory, ValiConfig

SUBACCOUNT = "entity_hotkey_0"
SIZE = 100_000.0
CREATED_MS = NOW_MS - 30 * DAILY_MS
WEEK_THREE_MS = NOW_MS - 20 * DAILY_MS  # a first funded order twenty days ago: the 10% line
HALF = ValiConfig.AUTO_PROMOTION_FAST_TRACK_LEVERAGE_USE


def _funded(asset_class: MinerAssetClass, tier: int, by_class: dict | None = None,
            realized_pnl: float = 0.0) -> MinerAccount:
    record = CollateralRecord(account_size=SIZE, account_size_theta=SIZE / 1_500, update_time_ms=NOW_MS,
                              is_first_record=True)
    by_class = by_class or {}
    return MinerAccount(miner_hotkey=SUBACCOUNT, collateral_records=[record], asset_class=asset_class,
                        miner_bucket=MinerBucket.SUBACCOUNT_FUNDED, leverage_tier=tier,
                        capital_used=sum(by_class.values()), capital_used_by_class=dict(by_class),
                        total_realized_pnl=realized_pnl)


# ── How much of its leverage an account uses ──────────────────────────────────

@pytest.mark.parametrize("by_class,use", [
    ({}, 0.0),
    # All markets at Boost II: crypto alone at half its 10x class cap, an eighth of the 40x portfolio cap
    ({TradePairCategory.CRYPTO: 500_000.0}, 0.5),
    # Four classes at 0.4 of their caps add up to 21.6x, more than half of the 40x portfolio cap
    ({TradePairCategory.CRYPTO: 400_000.0, TradePairCategory.FOREX: 1_200_000.0,
      TradePairCategory.INDICES: 320_000.0, TradePairCategory.COMMODITIES: 240_000.0}, 2_160_000.0 / 4_000_000.0),
])
def test_leverage_use_is_the_larger_of_the_whole_account_and_its_most_used_class(by_class, use):
    assert _funded(MinerAssetClass.ALL_MARKETS, 3, by_class).leverage_use() == pytest.approx(use)


def test_leverage_use_follows_the_accounts_own_tier_and_class():
    # Crypto alone at Base: a 1.5x class and portfolio cap
    assert _funded(MinerAssetClass.CRYPTO, 1, {TradePairCategory.CRYPTO: 75_000.0}).leverage_use() == pytest.approx(0.5)


def test_leverage_use_is_measured_on_the_balance():
    account = _funded(MinerAssetClass.ALL_MARKETS, 3, {TradePairCategory.CRYPTO: 400_000.0}, realized_pnl=-20_000.0)
    assert account.leverage_use() == pytest.approx(400_000.0 / (80_000.0 * 10))
    assert _funded(MinerAssetClass.ALL_MARKETS, 3, {TradePairCategory.CRYPTO: 1.0},
                   realized_pnl=-SIZE).leverage_use() == 0.0


# ── Its peak, after every fill and fee ────────────────────────────────────────

class TestPeakLeverageUse(_MinerAccountFilesInTempDir):

    def setUp(self):
        super().setUp()
        self.accounts = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        self.accounts.set_miner_account_size(SUBACCOUNT, collateral_balance_theta=SIZE / 1_500,
                                             timestamp_ms=TimeUtil.now_in_millis(), account_size=SIZE)
        account = self.accounts.accounts[SUBACCOUNT]
        account.asset_class, account.miner_bucket, account.leverage_tier = (
            MinerAssetClass.CRYPTO, MinerBucket.SUBACCOUNT_FUNDED, 1)

    def _peak(self) -> float:
        return self.accounts.accounts[SUBACCOUNT].peak_leverage_use

    def _buy(self, value: float):
        self.accounts.process_order_buy(SUBACCOUNT, value, 0.0, 0.0, TradePairCategory.CRYPTO)

    def _sell(self, entry_value: float, realized_pnl: float):
        self.accounts.process_order_sell(SUBACCOUNT, entry_value, realized_pnl, 0.0, 0.0, TradePairCategory.CRYPTO)

    def test_a_fill_raises_the_peak_and_a_close_never_lowers_it(self):
        self._buy(60_000.0)  # 60K of a 150K crypto cap
        self.assertAlmostEqual(self._peak(), 0.4)
        self._sell(60_000.0, 0.0)
        self.assertAlmostEqual(self._peak(), 0.4)
        self._buy(30_000.0)
        self.assertAlmostEqual(self._peak(), 0.4)

    def test_a_realized_loss_raises_the_use_of_what_stays_open(self):
        self._buy(60_000.0)
        self._sell(6_000.0, -30_000.0)  # 54K open on a 70K balance
        self.assertAlmostEqual(self._peak(), 54_000.0 / (70_000.0 * 1.5))

    def test_a_fee_raises_the_use_of_what_is_open(self):
        self._buy(74_900.0)
        self.accounts.process_fees({SUBACCOUNT: 200.0})
        self.assertAlmostEqual(self._peak(), 74_900.0 / (99_800.0 * 1.5))
        self.assertGreaterEqual(self._peak(), HALF)

    def test_the_peak_survives_saves_reloads_and_rebuilds_and_a_new_account_starts_over(self):
        self._buy(75_000.0)
        self.accounts.rebuild_account_state_from_positions(SUBACCOUNT, [])
        self.assertAlmostEqual(self._peak(), 0.5)
        reloaded = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        self.assertAlmostEqual(reloaded.accounts[SUBACCOUNT].peak_leverage_use, 0.5)
        self.assertAlmostEqual(self.accounts.accounts[SUBACCOUNT].to_dashboard()["peak_leverage_use"], 0.5)
        self.accounts.reset_account(SUBACCOUNT, MinerBucket.SUBACCOUNT_FUNDED)
        self.assertEqual(self._peak(), 0.0)

    def test_a_failure_to_measure_never_fails_the_fill(self):
        with patch.object(MinerAccount, "leverage_use", side_effect=RuntimeError("no caps")):
            self._buy(30_000.0)
        self.assertEqual(self.accounts.accounts[SUBACCOUNT].capital_used, 30_000.0)
        self.assertEqual(self._peak(), 0.0)


# ── The fast track's line ─────────────────────────────────────────────────────

@pytest.fixture
def on_the_rules(manager):
    """A subaccount created after the switch, on a mainnet validator."""
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "created_at_ms": CREATED_MS}
    manager._entity_client.is_testnet.return_value = False
    with patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS", CREATED_MS):
        yield


def _with_peak(account: MinerAccount, peak: float) -> MinerAccount:
    account.peak_leverage_use = peak
    return account


@pytest.mark.parametrize("make_account", [_standard_miner, _account], ids=["standard miner", "new subaccount"])
def test_half_its_leverage_lowers_the_line_to_three_percent_in_any_week(manager, on_the_rules, make_account):
    _seed(manager)
    assert not _holds(manager, _with_peak(make_account(), HALF - 0.0001), 1.0999, WEEK_THREE_MS)
    assert _holds(manager, _with_peak(make_account(), HALF), 1.03 + 1e-9, WEEK_THREE_MS)
    assert not _holds(manager, _with_peak(make_account(), 1.0), 1.0299, WEEK_THREE_MS)


@pytest.mark.parametrize("days,line", [(None, 0.10), (0, 0.03), (6.99, 0.03), (7, 0.05), (13.99, 0.05), (14, 0.10)])
def test_a_new_subaccount_takes_the_weekly_lines(manager, on_the_rules, days, line):
    _seed(manager)
    first_order_ms = None if days is None else NOW_MS - int(days * DAILY_MS)
    assert not _holds(manager, _account(), 1.0 + line - 0.0001, first_order_ms)
    assert _holds(manager, _account(), 1.0 + line + 1e-9, first_order_ms)


@pytest.mark.parametrize("start_ms", [None, CREATED_MS + 1])
def test_an_older_subaccount_keeps_ten_percent_whatever_its_leverage(manager, start_ms):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "created_at_ms": CREATED_MS}
    manager._entity_client.is_testnet.return_value = False
    _seed(manager)
    with patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS", start_ms):
        assert not _holds(manager, _with_peak(_account(), 1.0), 1.0999, NOW_MS - DAILY_MS)
        assert _holds(manager, _with_peak(_account(), 1.0), 1.10, NOW_MS - DAILY_MS)


def test_a_subaccount_record_without_its_creation_time_keeps_ten_percent(manager):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {"initial_bucket": None}
    manager._entity_client.is_testnet.return_value = False
    _seed(manager)
    with patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS", CREATED_MS):
        assert not _holds(manager, _with_peak(_account(), 1.0), 1.0999, NOW_MS - DAILY_MS)


def test_a_standard_miner_is_a_candidate_without_asking_the_network(manager):
    manager._entity_client.is_testnet.side_effect = RuntimeError("entity server not up yet")
    _seed(manager)
    assert manager._is_auto_promotion_candidate(HOTKEY, _standard_miner())
    assert _holds(manager, _with_peak(_standard_miner(), HALF), 1.04, WEEK_THREE_MS)
    manager._entity_client.is_testnet.assert_not_called()


def test_a_testnet_validator_reads_the_testnet_switch_and_asks_its_network_once(manager):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "created_at_ms": CREATED_MS}
    manager._entity_client.is_testnet.return_value = True
    _seed(manager)
    with patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS", CREATED_MS):
        assert not _holds(manager, _account(), 1.04, NOW_MS - DAILY_MS)
    manager._promotion_rules.clear()
    with patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS_TESTNET", CREATED_MS):
        assert _holds(manager, _account(), 1.04, NOW_MS - DAILY_MS, now_ms=NOW_MS + 30_000)
    manager._entity_client.is_testnet.assert_called_once()


def test_a_hold_begun_under_ten_percent_carries_on_once_the_leverage_line_applies(manager):
    _seed(manager)
    account = _standard_miner()
    assert _holds(manager, account, 1.10, WEEK_THREE_MS)
    assert _holds(manager, _with_peak(account, HALF), 1.04, WEEK_THREE_MS, now_ms=NOW_MS + 60_000)
    assert manager._fast_track_since_ms[HOTKEY] == NOW_MS


def test_the_auto_promotion_records_the_peak(manager, on_the_rules):
    _seed(manager)
    manager._entity_client.is_mothership.return_value = True
    manager._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
    stats = AutoPromotionStats(calmar=0.0, daily_consistency=0.5, max_drawdown=0.97, trading_days=10)
    with patch.object(manager, "_sync_buckets_to_accounts"):
        assert manager._move_to_auto_promotion(HOTKEY, _with_peak(_account(), 0.62), stats, ("fast_track", 3), NOW_MS)
    assert manager.miner_states[HOTKEY].auto_promotion.peak_leverage_use == 0.62
    assert ACCOUNT_SIZE * 3 == manager.miner_states[HOTKEY].auto_promotion.pro_account_size


def test_the_network_reaches_the_challenge_period_through_the_entity_client():
    server = EntityServer.__new__(EntityServer)  # only its forwarding is under test
    server._manager = SimpleNamespace(is_testnet=True)
    client = EntityClient(connection_mode=RPCConnectionMode.LOCAL, running_unit_tests=True)
    client.set_direct_server(server)
    assert client.is_testnet() is True
