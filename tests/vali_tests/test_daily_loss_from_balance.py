"""
Standard subaccounts in SUBACCOUNT_CHALLENGE, SUBACCOUNT_FUNDED and PRO_CHALLENGE_TRANSITION measure the
daily loss limit (intraday drawdown) from the day-open balance instead of the day-open equity, under both
drawdown criteria. HL-linked and Instant Funded subaccounts, pro buckets and regular miners keep the
day-open equity.
"""
from types import SimpleNamespace

import pytest

from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager is a pytest fixture
    ACCOUNT_SIZE,
    DAILY_MS,
    HOTKEY,
    MIDNIGHT_MS,
    NOW_MS,
    _elimination_kwargs,
    _run_refresh,
    manager,
)
from vali_objects.challenge_period.challengeperiod_manager import DrawdownStats, MinerBucketState
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.miner_account.account_snapshot import AccountSnapshot
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

BALANCE_BUCKETS = (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED,
                   MinerBucket.PRO_CHALLENGE_TRANSITION)
EQUITY_BUCKETS = (MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                  MinerBucket.PRO_FUNDED, MinerBucket.CHALLENGE, MinerBucket.MAINCOMP)
CRITERIA = (DrawdownCriteria.TRAILING, DrawdownCriteria.STATIC)
INTRADAY_REASON = {
    MinerBucket.SUBACCOUNT_CHALLENGE: EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN,
    MinerBucket.SUBACCOUNT_FUNDED: EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN,
    MinerBucket.PRO_CHALLENGE_TRANSITION: EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN,
}
LIMIT = 0.05
HL_ADDRESS = "0x" + "c" * 40
NOW = NOW_MS + 6 * 60 * 60 * 1000  # mid-morning UTC, same day as MIDNIGHT_MS

# Up on the starting balance so the static rule never binds. The overnight position is 2% under water
# at the open, so the balance floor sits above the equity floor.
OPEN_BALANCE = 1.10
OPEN_EQUITY_LOSS = 1.08
OPEN_EQUITY_GAIN = 1.12


def _account(equity: float, open_equity: float, open_balance: float = OPEN_BALANCE, hl_address: str | None = None,
             asset_class: MinerAssetClass | None = None, snapshot_ms: int | None = MIDNIGHT_MS) -> MinerAccount:
    """An account whose balance is still the day-open balance and whose live equity is `equity`."""
    record = CollateralRecord(account_size=ACCOUNT_SIZE, account_size_theta=ACCOUNT_SIZE / ValiConfig.COST_PER_THETA,
                              update_time_ms=NOW_MS - 30 * DAILY_MS, is_first_record=True)
    account = MinerAccount(miner_hotkey=HOTKEY, collateral_records=[record], hl_address=hl_address,
                           asset_class=asset_class)
    account.total_realized_pnl = (open_balance - 1.0) * ACCOUNT_SIZE
    account.unrealized_pnl = (equity - open_balance) * ACCOUNT_SIZE
    if snapshot_ms is not None:
        account.daily_open_snapshot = AccountSnapshot(snapshot_ms=snapshot_ms, account_size=ACCOUNT_SIZE,
                                                      balance=open_balance * ACCOUNT_SIZE,
                                                      equity=open_equity * ACCOUNT_SIZE)
    return account


@pytest.fixture(autouse=True)
def _standard_subaccount(manager):
    """The entity record of a subaccount created through SUBACCOUNT_CHALLENGE."""
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {"initial_bucket": None}


def _seed(manager, bucket: MinerBucket, criteria: DrawdownCriteria = DrawdownCriteria.TRAILING) -> None:
    manager.set_miner_bucket(HOTKEY, bucket, NOW_MS - DAILY_MS, drawdown_criteria=criteria,
                             intraday_drawdown_threshold=LIMIT)


def _refresh_cache(manager, account: MinerAccount, ledger: PerfLedger | None = None, now_ms: int = NOW) -> DrawdownStats:
    manager._refresh_drawdown_cache([HOTKEY], {HOTKEY: account}, {HOTKEY: ledger} if ledger else {},
                                    {HOTKEY: [SimpleNamespace()]}, now_ms)
    return manager.miner_states[HOTKEY].drawdown


# ── The day-open reference ────────────────────────────────────────────────────

def test_intraday_drawdown_reads_the_day_open_balance_when_set():
    stats = DrawdownStats(current_equity=1.0395, daily_open_equity=OPEN_EQUITY_LOSS)
    assert stats.intraday_drawdown_pct == pytest.approx((1 - 1.0395 / OPEN_EQUITY_LOSS) * 100)
    stats.daily_open_balance = OPEN_BALANCE
    assert stats.intraday_drawdown_pct == pytest.approx(5.5)


@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", BALANCE_BUCKETS)
def test_standard_buckets_track_the_day_open_balance(manager, bucket, criteria):
    _seed(manager, bucket, criteria)
    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS))
    assert stats.daily_open_balance == pytest.approx(OPEN_BALANCE)
    assert stats.daily_open_equity == pytest.approx(OPEN_EQUITY_LOSS)
    assert stats.intraday_drawdown_pct == pytest.approx(5.5)


@pytest.mark.parametrize("bucket", EQUITY_BUCKETS)
def test_pro_buckets_and_regular_miners_keep_the_day_open_equity(manager, bucket):
    _seed(manager, bucket)
    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS))
    assert stats.daily_open_balance is None
    assert stats.intraday_drawdown_pct == pytest.approx((1 - 1.0395 / OPEN_EQUITY_LOSS) * 100)


@pytest.mark.parametrize("account_kwargs", [{"hl_address": HL_ADDRESS}, {"asset_class": MinerAssetClass.HL_ALL}])
@pytest.mark.parametrize("bucket", BALANCE_BUCKETS)
def test_hl_linked_subaccounts_keep_the_day_open_equity(manager, bucket, account_kwargs):
    _seed(manager, bucket)
    assert _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS, **account_kwargs)).daily_open_balance is None


@pytest.mark.parametrize("record,from_balance", [
    ({"initial_bucket": MinerBucket.SUBACCOUNT_CHALLENGE.value}, True),
    ({"initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value}, False),               # Instant Funded
    ({"initial_bucket": MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value}, False),     # Instant Funded
    (None, False),                                                                  # no entity record
])
def test_only_subaccounts_created_through_the_challenge_track_the_balance(manager, record, from_balance):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = record
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS))
    assert (stats.daily_open_balance is not None) == from_balance


@pytest.mark.parametrize("on_balance", [True, False])
def test_a_failed_entity_lookup_keeps_the_current_reference(manager, on_balance):
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS))  # today's open is locked in
    if not on_balance:
        manager.miner_states[HOTKEY].drawdown.daily_open_balance = None
    manager._entity_client.get_subaccount_info_for_synthetic.side_effect = RuntimeError("entity server down")
    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS, snapshot_ms=MIDNIGHT_MS + DAILY_MS),
                           now_ms=NOW + DAILY_MS)
    assert (stats.daily_open_balance is not None) == on_balance


@pytest.mark.parametrize("snapshot_ms", [None, MIDNIGHT_MS - DAILY_MS])
def test_without_todays_snapshot_the_day_runs_on_the_day_open_equity(manager, snapshot_ms):
    """No recorded day-open balance: the ledger supplies the day-open equity and the balance stays unset."""
    ledger = PerfLedger(cps=[PerfCheckpoint(last_update_ms=MIDNIGHT_MS, prev_portfolio_ret=1.0, equity_ret=OPEN_EQUITY_LOSS,
                                            unrealized_pnl=(OPEN_EQUITY_LOSS - OPEN_BALANCE) * ACCOUNT_SIZE)])
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    manager.miner_states[HOTKEY].drawdown.daily_open_balance = 1.20  # yesterday's
    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS, snapshot_ms=snapshot_ms), ledger)
    assert stats.daily_open_equity == pytest.approx(OPEN_EQUITY_LOSS)
    assert stats.daily_open_balance is None
    assert stats.intraday_drawdown_pct == pytest.approx((1 - 1.0395 / OPEN_EQUITY_LOSS) * 100)


def test_a_day_open_locked_before_the_balance_was_tracked_waits_for_the_next_day_open(manager):
    """A checkpoint written before this rule has today's open locked in with no balance: the rest of
    the day runs on the day-open equity, and the next day open brings in the balance."""
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    manager.miner_states[HOTKEY].drawdown = DrawdownStats(
        daily_open_equity=OPEN_EQUITY_LOSS, eod_hwm=OPEN_EQUITY_LOSS, last_eod_equity=OPEN_EQUITY_LOSS,
        last_eod_checked_ms=MIDNIGHT_MS)
    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS))
    assert stats.daily_open_balance is None
    assert stats.intraday_drawdown_pct == pytest.approx((1 - 1.0395 / OPEN_EQUITY_LOSS) * 100)

    stats = _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS, snapshot_ms=MIDNIGHT_MS + DAILY_MS),
                           now_ms=NOW + DAILY_MS)
    assert stats.daily_open_balance == pytest.approx(OPEN_BALANCE)
    assert stats.intraday_drawdown_pct == pytest.approx(5.5)


def test_leaving_the_standard_buckets_drops_the_balance(manager):
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    account = _account(1.0395, OPEN_EQUITY_LOSS)
    assert _refresh_cache(manager, account).daily_open_balance == pytest.approx(OPEN_BALANCE)
    # PRO_FUNDED keeps the account, so nothing resets the drawdown stats on the way in
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, NOW)
    assert _refresh_cache(manager, account).daily_open_balance is None


# ── Elimination ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", BALANCE_BUCKETS)
def test_an_overnight_loss_tightens_the_day(manager, bucket, criteria):
    """5.5% below the open balance eliminates, though only 3.75% below the open equity."""
    _seed(manager, bucket, criteria)
    _refresh_cache(manager, _account(OPEN_BALANCE * (1 - 0.055), OPEN_EQUITY_LOSS))
    _run_refresh(manager, HOTKEY, now_ms=NOW)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.ELIMINATED
    kwargs = _elimination_kwargs(manager)
    assert kwargs["reason"] == INTRADAY_REASON[bucket]
    assert kwargs["intraday_drawdown_pct"] == pytest.approx(5.5)


@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", BALANCE_BUCKETS)
def test_an_overnight_gain_loosens_the_day(manager, bucket, criteria):
    """4% below the open balance survives, though 5.71% below the open equity."""
    _seed(manager, bucket, criteria)
    _refresh_cache(manager, _account(OPEN_BALANCE * (1 - 0.04), OPEN_EQUITY_GAIN))
    _run_refresh(manager, HOTKEY, now_ms=NOW)
    assert manager.get_miner_bucket(HOTKEY) == bucket


def test_hl_linked_survives_the_same_overnight_loss(manager):
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    _refresh_cache(manager, _account(OPEN_BALANCE * (1 - 0.055), OPEN_EQUITY_LOSS, hl_address=HL_ADDRESS))
    _run_refresh(manager, HOTKEY, now_ms=NOW)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.SUBACCOUNT_FUNDED


# ── Checkpoint and admin override ─────────────────────────────────────────────

def test_the_day_open_balance_round_trips_through_the_checkpoint():
    state = MinerBucketState(HOTKEY, [BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, NOW_MS)],
                             drawdown=DrawdownStats(daily_open_balance=OPEN_BALANCE))
    assert MinerBucketState.from_checkpoint_dict(HOTKEY, state.to_checkpoint_dict()).drawdown.daily_open_balance == OPEN_BALANCE

    data = state.to_checkpoint_dict()
    data["drawdown"].pop("daily_open_balance")
    assert MinerBucketState.from_checkpoint_dict(HOTKEY, data).drawdown.daily_open_balance is None


def test_admin_can_set_the_day_open_balance_only_where_it_applies(manager):
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED)
    _refresh_cache(manager, _account(1.0395, OPEN_EQUITY_LOSS))
    ok, _, stats = manager.set_miner_drawdown_stats(HOTKEY, {"daily_open_balance": 1.09}, now_ms=NOW)
    assert ok and stats["daily_open_balance"] == 1.09

    manager.miner_states[HOTKEY].drawdown.daily_open_balance = None
    ok, message, _ = manager.set_miner_drawdown_stats(HOTKEY, {"daily_open_balance": 1.09}, now_ms=NOW)
    assert not ok and "daily_open_equity" in message
