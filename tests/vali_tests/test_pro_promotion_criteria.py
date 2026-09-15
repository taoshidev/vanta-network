"""
Unit tests for the pro challenge promotion criteria and the 1000-day perf ledger window.

PRO_CHALLENGE_DIRECT and PRO_CHALLENGE_FROM_STANDARD promote to PRO_FUNDED only after clearing
all four gates:
  1. 90 full calendar days since the account's first trade
  2. 6% return on the account
  3. all-time calmar of at least 1.75, whose denominator is the largest drop of live equity below
     the end-of-day equity high-water mark
  4. daily return consistency of at most 20%, computed after capping each day's return at 1.5%

Failing either drawdown rule demotes them back onto a fresh standard account instead.
No RPC connections, no disk I/O, no daemon simulation.
"""
import contextlib
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vali_objects.challenge_period.challengeperiod_manager import (
    ChallengePeriodManager,
    DrawdownStats,
    MinerBucketState,
    ProStats,
)
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

# ── Constants ─────────────────────────────────────────────────────────────────

DAILY_MS = ValiConfig.DAILY_MS
CP_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
ACCOUNT_SIZE = 100_000.0

MIN_DAYS = ValiConfig.PRO_CHALLENGE_MINIMUM_DAYS                    # 90
RETURNS_THRESHOLD = ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT  # 0.06
CALMAR_THRESHOLD = ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD        # 1.75
CONSISTENCY_THRESHOLD = ValiConfig.PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD  # 0.20
DAILY_CAP = ValiConfig.PRO_DAILY_RETURN_CAP                         # 0.015

INTRADAY_THRESHOLD = ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD  # 0.05
EOD_THRESHOLD = ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD            # 0.08

# The two buckets the spec names. Both promote to PRO_FUNDED and both demote on a breach.
PRO_CHALLENGE_BUCKETS = (MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_FROM_STANDARD)
DEMOTION_TARGET = {
    MinerBucket.PRO_CHALLENGE_DIRECT: MinerBucket.SUBACCOUNT_CHALLENGE,
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD: MinerBucket.SUBACCOUNT_FUNDED,
}

LEDGER_START_MS = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC, a midnight boundary
NOW_MS = LEDGER_START_MS + 400 * DAILY_MS
MIDNIGHT_MS = (NOW_MS // DAILY_MS) * DAILY_MS

_CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager.PerfLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.PositionManagerClient",
    "vali_objects.challenge_period.challengeperiod_manager.LimitOrderClient",
    "vali_objects.challenge_period.challengeperiod_manager.EliminationClient",
    "vali_objects.challenge_period.challengeperiod_manager.PlagiarismClient",
    "vali_objects.challenge_period.challengeperiod_manager.MinerAccountClient",
    "vali_objects.challenge_period.challengeperiod_manager.CommonDataClient",
    "vali_objects.challenge_period.challengeperiod_manager.AssetSelectionClient",
    "vali_objects.challenge_period.challengeperiod_manager.DebtLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.EntityClient",
]


# ── Fixtures & helpers ────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    with contextlib.ExitStack() as stack:
        for path in _CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        mgr._entity_client.apply_bucket_account_size.return_value = (True, "")
        yield mgr


def _ledger(daily_returns: list[float], realized_pnl_usd: float = 0.0, fees_usd: float = 0.0,
            start_ms: int = LEDGER_START_MS, partial_last_day: bool = False) -> PerfLedger:
    """A portfolio ledger holding one full UTC day per entry in daily_returns.

    LedgerUtils counts a day only when both of its 12h checkpoints are full cells, and reads the
    day's return as the sum of gain + loss across them, so each day's return is split in half.
    """
    cps = []
    for day_idx, daily_log_return in enumerate(daily_returns):
        day_start = start_ms + day_idx * DAILY_MS
        for cp_idx in range(int(ValiConfig.DAILY_CHECKPOINTS)):
            cps.append(PerfCheckpoint(
                last_update_ms=day_start + (cp_idx + 1) * CP_MS,
                prev_portfolio_ret=1.0,
                accum_ms=CP_MS,
                gain=max(daily_log_return, 0.0) / 2,
                loss=min(daily_log_return, 0.0) / 2,
                mdd=1.0,
            ))
    if cps:
        if partial_last_day:
            cps[-1].accum_ms = CP_MS // 2
        cps[-1].prev_portfolio_realized_pnl = realized_pnl_usd
        cps[-1].cumulative_fees_usd = fees_usd
    return PerfLedger(initialization_time_ms=start_ms, cps=cps)


def _even_ledger(n_days: int, realized_pnl_usd: float = 0.0, **kwargs) -> PerfLedger:
    """n_days of identical small gains, so return consistency is 1/n_days and never the binding gate."""
    return _ledger([math.log(1.001)] * n_days, realized_pnl_usd=realized_pnl_usd, **kwargs)


def _state(bucket: MinerBucket, start_ms: int = NOW_MS - DAILY_MS) -> MinerBucketState:
    return MinerBucketState("pro_hk", [BucketEntry(bucket, start_ms)])


def _healthy_drawdown(equity: float = 1.07) -> DrawdownStats:
    """Clear of both drawdown rules, with `equity` as both the live and the latched EOD value."""
    return DrawdownStats(current_equity=equity, current_balance=equity, daily_open_equity=equity,
                         eod_hwm=equity, last_eod_equity=equity, last_eod_checked_ms=MIDNIGHT_MS)


def _passing_pro_stats() -> ProStats:
    return ProStats(calmar=CALMAR_THRESHOLD, daily_consistency=CONSISTENCY_THRESHOLD,
                    max_drawdown=0.96, trading_days=MIN_DAYS)


def _promotable_state(bucket: MinerBucket) -> MinerBucketState:
    """A state clearing all four gates, so a single tweak isolates the gate under test."""
    state = _state(bucket)
    state.drawdown = _healthy_drawdown(1.0 + RETURNS_THRESHOLD + 0.01)
    state.pro_stats = _passing_pro_stats()
    return state


def _seed(manager, hk: str, bucket: MinerBucket, drawdown: DrawdownStats) -> None:
    manager.miner_states[hk] = MinerBucketState(hk, [BucketEntry(bucket, NOW_MS - DAILY_MS)])
    manager.miner_states[hk].drawdown = drawdown


def _run_refresh(manager, hk: str, ledger: PerfLedger | None = None,
                 account_size: float | None = ACCOUNT_SIZE, now_ms: int = NOW_MS) -> None:
    """One refresh() pass with the drawdown cache pinned, so pro stats and routing run for real."""
    manager._position_client.get_all_hotkeys.return_value = [hk]
    manager._position_client.filtered_positions_for_scoring.return_value = ({hk: []}, {})
    manager._position_client.get_positions_for_hotkeys.return_value = {hk: []}
    manager._elimination_client.get_eliminated_hotkeys.return_value = []
    manager._plagiarism_client.get_plagiarism_miners.return_value = []
    manager._miner_account_client.get_accounts.return_value = (
        {hk: SimpleNamespace(account_size=account_size)} if account_size else {})
    manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = ({hk: ledger} if ledger else {})
    manager._asset_selection_client.get_asset_selections.return_value = {hk: MinerAssetClass.CRYPTO}
    with (
        patch.object(manager, "_refresh_drawdown_cache"),
        patch.object(manager, "_refresh_rank_cache"),
        patch.object(manager, "_save_to_disk"),
        patch.object(manager, "_sync_buckets_to_accounts"),
    ):
        manager.refresh(current_time_ms=now_ms)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Gate 1: 90 full calendar days since the first trade
# ═══════════════════════════════════════════════════════════════════════════════

def test_trading_days_counts_one_per_full_calendar_day(manager):
    """The ledger starts at the first order, so its full days are calendar days on the account."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())

    manager._refresh_pro_stats([hk], {hk: _even_ledger(MIN_DAYS)},
                               {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})
    assert manager.miner_states[hk].pro_stats.trading_days == MIN_DAYS


def test_a_half_covered_day_is_not_a_full_calendar_day(manager):
    """A day the ledger only half covered does not count, so the 90 are 90 *full* days."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())

    manager._refresh_pro_stats([hk], {hk: _even_ledger(MIN_DAYS, partial_last_day=True)},
                               {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})
    assert manager.miner_states[hk].pro_stats.trading_days == MIN_DAYS - 1


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_blocked_one_day_short_of_ninety(bucket):
    state = _promotable_state(bucket)
    state.pro_stats.trading_days = MIN_DAYS - 1
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_allowed_at_ninety_days(bucket):
    state = _promotable_state(bucket)
    state.pro_stats.trading_days = MIN_DAYS
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is True


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_days_count_from_the_first_trade_not_from_entering_the_bucket(bucket):
    """A year parked in the bucket without trading still fails; 90 traded days pass on day one."""
    idle = _promotable_state(bucket)
    idle.entries[-1].start_time_ms = NOW_MS - 365 * DAILY_MS
    idle.pro_stats.trading_days = MIN_DAYS - 1
    assert ChallengePeriodManager._check_promotion(idle, RETURNS_THRESHOLD, NOW_MS) is False

    traded = _promotable_state(bucket)
    traded.entries[-1].start_time_ms = NOW_MS
    assert ChallengePeriodManager._check_promotion(traded, RETURNS_THRESHOLD, NOW_MS) is True


def test_minimum_days_is_ninety_for_both_pro_challenge_buckets():
    for bucket in PRO_CHALLENGE_BUCKETS:
        assert bucket.minimum_trading_days == 90


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Gate 2: 6% return
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
@pytest.mark.parametrize("asset_class", list(MinerAssetClass))
def test_returns_threshold_is_six_percent_for_every_asset_class(bucket, asset_class):
    assert bucket.returns_threshold(asset_class) == 0.06


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_blocked_just_under_six_percent(bucket):
    state = _promotable_state(bucket)
    state.drawdown = _healthy_drawdown(1.0 + RETURNS_THRESHOLD - 0.001)
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_allowed_just_over_six_percent(bucket):
    state = _promotable_state(bucket)
    state.drawdown = _healthy_drawdown(1.0 + RETURNS_THRESHOLD + 0.001)
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is True


def test_return_is_the_lesser_of_equity_and_balance():
    """Unrealized gains do not count toward the 6%: the balance caps the return."""
    state = _promotable_state(MinerBucket.PRO_CHALLENGE_DIRECT)
    state.drawdown.current_equity = 1.20
    state.drawdown.current_balance = 1.05
    assert state.drawdown.current_return == pytest.approx(0.05)
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Gate 3: all-time calmar of 1.75 over the EOD high-water-mark drawdown
# ═══════════════════════════════════════════════════════════════════════════════

def test_calmar_denominator_is_live_equity_against_the_eod_high_water_mark(manager):
    """7% realized over a 4% drop below the midnight mark is exactly the 1.75 threshold."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=0.96,
                        eod_hwm=1.0, last_eod_equity=0.96))

    manager._refresh_pro_stats([hk], {hk: _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)},
                               {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})

    stats = manager.miner_states[hk].pro_stats
    assert stats.max_drawdown == pytest.approx(0.96)
    assert stats.calmar == pytest.approx(CALMAR_THRESHOLD)


def test_calmar_denominator_ignores_an_intraday_peak_above_the_mark(manager):
    """The mark only moves at UTC midnight, so a peak earlier today cannot deepen the drawdown."""
    hk = "pro_hk"
    # Live equity is 0.96 and the account traded up to 1.30 earlier today, but the last midnight
    # latch is 1.00, so the drawdown is 4%, not the 26% an intraday peak would imply.
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=1.0,
                        eod_hwm=1.0, last_eod_equity=1.0))

    manager._refresh_pro_stats([hk], {hk: _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)},
                               {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})
    assert manager.miner_states[hk].pro_stats.max_drawdown == pytest.approx(0.96)


def test_only_midnight_checkpoints_set_the_high_water_mark():
    """_parse_eod_checkpoints reads UTC midnights only, so a midday high never becomes the mark."""
    ledger = _even_ledger(3)
    for cp in ledger.cps:
        cp.equity_ret = 1.0
    midday_cps = [cp for cp in ledger.cps if cp.last_update_ms % DAILY_MS != 0]
    assert midday_cps, "fixture should contain midday checkpoints"
    for cp in midday_cps:
        cp.equity_ret = 1.40  # a spike that never survives to a midnight latch

    _, _, eod_hwm, _ = ChallengePeriodManager._parse_eod_checkpoints(
        ledger, LEDGER_START_MS + 3 * DAILY_MS)
    assert eod_hwm == pytest.approx(1.0)


def test_a_recovered_dip_still_blocks_promotion(manager):
    """The ratchet is what makes the drawdown all-time: recovering does not hand the denominator back."""
    hk = "pro_hk"
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)
    accounts = {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)}

    # Day one: equity 5% under the midnight mark
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.95, current_balance=0.95, daily_open_equity=0.97,
                        eod_hwm=1.0, last_eod_equity=0.97))
    manager._refresh_pro_stats([hk], {hk: ledger}, accounts)
    assert manager.miner_states[hk].pro_stats.max_drawdown == pytest.approx(0.95)

    # Later: fully recovered and 7% up, but the 5% drawdown still divides the calmar
    manager.miner_states[hk].drawdown = _healthy_drawdown(1.07)
    manager._refresh_pro_stats([hk], {hk: ledger}, accounts)

    stats = manager.miner_states[hk].pro_stats
    assert stats.max_drawdown == pytest.approx(0.95)
    assert stats.calmar == pytest.approx(0.07 / 0.05)
    assert stats.calmar < CALMAR_THRESHOLD

    state = manager.miner_states[hk]
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_blocked_just_under_the_calmar_threshold(bucket):
    state = _promotable_state(bucket)
    state.pro_stats.calmar = CALMAR_THRESHOLD - 0.01
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_allowed_at_the_calmar_threshold(bucket):
    state = _promotable_state(bucket)
    state.pro_stats.calmar = CALMAR_THRESHOLD
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is True


def test_calmar_threshold_is_one_seventy_five_for_both_pro_challenge_buckets():
    for bucket in PRO_CHALLENGE_BUCKETS:
        assert bucket.calmar_threshold == 1.75


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Gate 4: 20% return consistency on daily returns capped at 1.5%
# ═══════════════════════════════════════════════════════════════════════════════

def test_consistency_caps_each_day_at_one_and_a_half_percent(manager):
    """A 10% day counts as 1.5%, so six 1% days carry it to exactly the 20% limit."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())
    ledger = _ledger([math.log(1.10)] + [math.log(1.01)] * 6)

    manager._refresh_pro_stats([hk], {hk: ledger}, {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})

    # capped total = 0.015 + 6 x 0.01 = 0.075, best day = 0.015
    assert manager.miner_states[hk].pro_stats.daily_consistency == pytest.approx(DAILY_CAP / 0.075)
    assert manager.miner_states[hk].pro_stats.daily_consistency == pytest.approx(CONSISTENCY_THRESHOLD)


def test_consistency_breaches_when_the_best_day_carries_the_account(manager):
    """The same 10% day against only three 1% days is 33% of the capped total."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())
    ledger = _ledger([math.log(1.10)] + [math.log(1.01)] * 3)

    manager._refresh_pro_stats([hk], {hk: ledger}, {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})

    consistency = manager.miner_states[hk].pro_stats.daily_consistency
    assert consistency == pytest.approx(DAILY_CAP / 0.045)
    assert consistency > CONSISTENCY_THRESHOLD


def test_consistency_counts_losing_days_in_full(manager):
    """Losses are not capped, so they shrink the total and push the best day's share up."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())
    winners_only = _ledger([math.log(1.01)] * 8)
    with_a_loss = _ledger([math.log(1.01)] * 8 + [math.log(0.96)])
    accounts = {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)}

    manager._refresh_pro_stats([hk], {hk: winners_only}, accounts)
    clean = manager.miner_states[hk].pro_stats.daily_consistency

    manager._refresh_pro_stats([hk], {hk: with_a_loss}, accounts)
    after_loss = manager.miner_states[hk].pro_stats.daily_consistency

    assert clean == pytest.approx(0.125)  # 0.01 / 0.08
    assert after_loss == pytest.approx(0.01 / 0.04)
    assert after_loss > clean


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_blocked_just_over_the_consistency_limit(bucket):
    state = _promotable_state(bucket)
    state.pro_stats.daily_consistency = CONSISTENCY_THRESHOLD + 0.01
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_promotion_allowed_at_the_consistency_limit(bucket):
    state = _promotable_state(bucket)
    state.pro_stats.daily_consistency = CONSISTENCY_THRESHOLD
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is True


def test_consistency_threshold_is_twenty_percent_for_both_pro_challenge_buckets():
    for bucket in PRO_CHALLENGE_BUCKETS:
        assert bucket.daily_consistency_threshold == 0.20


def test_daily_return_cap_is_one_and_a_half_percent():
    assert ValiConfig.PRO_DAILY_RETURN_CAP == 0.015


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — all four gates through refresh()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_refresh_promotes_to_pro_funded_when_every_gate_clears(manager, bucket):
    """90 even days, 8% realized, a 4% ratcheted drawdown and 8% live equity: calmar is 2.0."""
    hk = "pro_hk"
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=8_000.0)

    # The 4% drawdown happened earlier in the account's life and has since recovered
    _seed(manager, hk, bucket,
          DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=0.97,
                        eod_hwm=1.0, last_eod_equity=0.97))
    manager._refresh_pro_stats([hk], {hk: ledger}, {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})
    assert manager.miner_states[hk].pro_stats.max_drawdown == pytest.approx(0.96)
    manager.miner_states[hk].drawdown = _healthy_drawdown(1.08)

    _run_refresh(manager, hk, ledger=ledger)

    # Promotion runs the account switch, which resets the stats it promoted on
    assert manager.get_miner_bucket(hk) == MinerBucket.PRO_FUNDED
    assert manager.miner_states[hk].pro_stats == ProStats()


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_refresh_holds_in_bucket_when_the_ledger_is_a_day_short(manager, bucket):
    hk = "pro_hk"
    ledger = _even_ledger(MIN_DAYS - 1, realized_pnl_usd=7_000.0)
    _seed(manager, hk, bucket, _healthy_drawdown(1.07))

    _run_refresh(manager, hk, ledger=ledger)

    assert manager.miner_states[hk].pro_stats.trading_days == MIN_DAYS - 1
    assert manager.get_miner_bucket(hk) == bucket


def test_refresh_holds_in_bucket_when_the_ratcheted_drawdown_sinks_calmar(manager):
    """Everything else clears, but a 10% drawdown earlier in the account's life leaves calmar at 0.7."""
    hk = "pro_hk"
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)

    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.90, current_balance=0.90, daily_open_equity=0.94,
                        eod_hwm=1.0, last_eod_equity=0.94))
    manager._refresh_pro_stats([hk], {hk: ledger}, {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})
    manager.miner_states[hk].drawdown = _healthy_drawdown(1.07)

    _run_refresh(manager, hk, ledger=ledger)

    assert manager.miner_states[hk].pro_stats.calmar == pytest.approx(0.70)
    assert manager.get_miner_bucket(hk) == MinerBucket.PRO_CHALLENGE_DIRECT


def test_refresh_holds_in_bucket_when_one_day_carries_the_return(manager):
    """90 days and 7% realized, but a single uncapped spike day fails the consistency gate."""
    hk = "pro_hk"
    # 0.05% filler days keep the capped total thin enough that the one spike owns 25% of it
    ledger = _ledger([math.log(1.10)] + [math.log(1.0005)] * (MIN_DAYS - 1), realized_pnl_usd=7_000.0)
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown(1.07))

    _run_refresh(manager, hk, ledger=ledger)

    stats = manager.miner_states[hk].pro_stats
    assert stats.trading_days == MIN_DAYS
    assert stats.daily_consistency > CONSISTENCY_THRESHOLD
    assert manager.get_miner_bucket(hk) == MinerBucket.PRO_CHALLENGE_DIRECT


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — the two drawdown rules demote onto a fresh standard account
# ═══════════════════════════════════════════════════════════════════════════════

def _intraday_breach() -> DrawdownStats:
    """5.1% below the day's open, with the EOD mark intact so only the daily rule binds."""
    return DrawdownStats(current_equity=0.949, current_balance=0.949, daily_open_equity=1.0,
                         eod_hwm=1.0, last_eod_equity=1.0, last_eod_checked_ms=MIDNIGHT_MS)


def _eod_breach() -> DrawdownStats:
    """Latched EOD equity 8.1% below the mark, only 1% below the day's open."""
    return DrawdownStats(current_equity=0.919, current_balance=0.919, daily_open_equity=0.928,
                         eod_hwm=1.0, last_eod_equity=0.919, last_eod_checked_ms=MIDNIGHT_MS)


def _assert_fresh_account(manager, hk: str) -> None:
    """The demotion must restart the account rather than carry the pro one across."""
    manager._position_client.close_all_positions.assert_called_once()
    manager._position_client.archive_positions_for_hotkey.assert_called_once()
    manager._limit_order_client.cancel_limit_order.assert_called_once()
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_called_once_with([hk])
    manager._debt_ledger_client.delete_debt_ledger.assert_called_once_with(hk)
    manager._miner_account_client.reset_account.assert_called_once()
    assert manager.miner_states[hk].pro_stats == ProStats()
    assert manager.miner_states[hk].drawdown == DrawdownStats()


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_daily_loss_limit_demotes_onto_a_fresh_standard_account(manager, bucket):
    hk = "pro_hk"
    _seed(manager, hk, bucket, _intraday_breach())
    manager.miner_states[hk].pro_stats = _passing_pro_stats()

    _run_refresh(manager, hk, ledger=_even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0))

    assert manager.get_miner_bucket(hk) == DEMOTION_TARGET[bucket]
    manager._elimination_client.append_elimination_row.assert_not_called()
    _assert_fresh_account(manager, hk)


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_trailing_loss_limit_demotes_onto_a_fresh_standard_account(manager, bucket):
    hk = "pro_hk"
    _seed(manager, hk, bucket, _eod_breach())
    manager.miner_states[hk].pro_stats = _passing_pro_stats()

    _run_refresh(manager, hk, ledger=_even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0))

    assert manager.get_miner_bucket(hk) == DEMOTION_TARGET[bucket]
    manager._elimination_client.append_elimination_row.assert_not_called()
    _assert_fresh_account(manager, hk)


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_a_breach_beats_a_promotion_in_the_same_pass(manager, bucket):
    """Every gate clears, but the day's loss limit is gone: the miner demotes, never promotes."""
    hk = "pro_hk"
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)
    _seed(manager, hk, bucket, _intraday_breach())
    manager.miner_states[hk].pro_stats = _passing_pro_stats()

    _run_refresh(manager, hk, ledger=ledger)

    assert manager.get_miner_bucket(hk) == DEMOTION_TARGET[bucket]


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_just_inside_the_daily_loss_limit_keeps_trading(manager, bucket):
    hk = "pro_hk"
    just_inside = 1.0 - INTRADAY_THRESHOLD + 0.001
    _seed(manager, hk, bucket,
          DrawdownStats(current_equity=just_inside, current_balance=just_inside,
                        daily_open_equity=1.0, eod_hwm=1.0, last_eod_equity=1.0,
                        last_eod_checked_ms=MIDNIGHT_MS))

    _run_refresh(manager, hk, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(hk) == bucket
    manager._position_client.close_all_positions.assert_not_called()


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_just_inside_the_trailing_loss_limit_keeps_trading(manager, bucket):
    hk = "pro_hk"
    just_inside = 1.0 - EOD_THRESHOLD + 0.001
    _seed(manager, hk, bucket,
          DrawdownStats(current_equity=just_inside, current_balance=just_inside,
                        daily_open_equity=just_inside, eod_hwm=1.0, last_eod_equity=just_inside,
                        last_eod_checked_ms=MIDNIGHT_MS))

    _run_refresh(manager, hk, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(hk) == bucket
    manager._position_client.close_all_positions.assert_not_called()


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_the_trailing_limit_is_not_real_time(manager, bucket):
    """Live equity 12% under the mark breaches nothing while the latched EOD equity sits at it."""
    hk = "pro_hk"
    _seed(manager, hk, bucket,
          DrawdownStats(current_equity=0.88, current_balance=0.88, daily_open_equity=0.92,
                        eod_hwm=1.0, last_eod_equity=1.0, last_eod_checked_ms=MIDNIGHT_MS))

    _run_refresh(manager, hk, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(hk) == bucket
    manager._position_client.close_all_positions.assert_not_called()


def test_demotion_resizes_a_synthetic_subaccount_before_wiping_it(manager):
    """A real subaccount is pointed at the standard size first, so the fresh account is sized right."""
    hk = "entity_hk_1"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _intraday_breach())

    _run_refresh(manager, hk, ledger=_even_ledger(10))

    manager._entity_client.apply_bucket_account_size.assert_called_once_with(
        hk, MinerBucket.SUBACCOUNT_CHALLENGE)
    assert manager.get_miner_bucket(hk) == MinerBucket.SUBACCOUNT_CHALLENGE


def test_demotion_is_abandoned_when_the_account_cannot_be_resized(manager):
    """A failed resize leaves the miner on the pro account rather than stranding them mid-switch."""
    hk = "entity_hk_1"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _intraday_breach())
    manager._entity_client.apply_bucket_account_size.return_value = (False, "no standard size on record")

    _run_refresh(manager, hk, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(hk) == MinerBucket.PRO_CHALLENGE_DIRECT
    manager._position_client.close_all_positions.assert_not_called()
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()


def test_pro_funded_is_eliminated_rather_than_demoted(manager):
    """Only the two challenge buckets fall back to a standard account; PRO_FUNDED has nowhere to go."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_FUNDED, _intraday_breach())

    _run_refresh(manager, hk, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(hk) == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN


def test_pro_buckets_never_run_the_static_drawdown_rule():
    """The pro track has no static elimination reason, so the static check must decline pro buckets."""
    for bucket in (*PRO_CHALLENGE_BUCKETS, MinerBucket.PRO_FUNDED):
        state = _state(bucket)
        state.drawdown = DrawdownStats(current_equity=0.5)  # far past any static threshold
        assert ChallengePeriodManager._check_static_drawdown(state) is None

    for name in ("FAILED_PRO_CHALLENGE_PERIOD_STATIC_DRAWDOWN",
                 "FAILED_PRO_FUNDED_PERIOD_STATIC_DRAWDOWN",
                 "FAILED_PRO_CHALLENGE_PERIOD_STATIC_EOD_DRAWDOWN",
                 "FAILED_PRO_FUNDED_PERIOD_STATIC_EOD_DRAWDOWN"):
        assert name not in EliminationReason.__members__


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — the 1000-day perf ledger window
# ═══════════════════════════════════════════════════════════════════════════════

def test_target_ledger_window_is_one_thousand_days():
    assert ValiConfig.TARGET_LEDGER_WINDOW_DAYS == 1000
    assert ValiConfig.TARGET_LEDGER_WINDOW_MS == 1000 * DAILY_MS


def test_a_loaded_ledger_adopts_the_configured_window():
    """Ledgers serialized under the old 180-day window must not keep purging at 180."""
    stale = _even_ledger(5).to_dict()
    stale["target_ledger_window_ms"] = 180 * DAILY_MS

    restored = PerfLedger.from_dict(stale)

    assert restored.target_ledger_window_ms == ValiConfig.TARGET_LEDGER_WINDOW_MS


def test_a_new_ledger_defaults_to_the_configured_window():
    assert PerfLedger().target_ledger_window_ms == ValiConfig.TARGET_LEDGER_WINDOW_MS


def test_purge_keeps_everything_inside_the_window():
    ledger = _even_ledger(999)
    n_cps = len(ledger.cps)

    ledger.purge_old_cps()

    assert len(ledger.cps) == n_cps
    assert ledger.get_total_ledger_duration_ms() == 999 * DAILY_MS


def test_purge_trims_back_to_the_window():
    ledger = _even_ledger(1010)

    ledger.purge_old_cps()

    assert ledger.get_total_ledger_duration_ms() == ValiConfig.TARGET_LEDGER_WINDOW_MS


def test_pro_metrics_read_the_whole_window_not_the_old_one_hundred_and_eighty_days(manager):
    """Every metric behind the pro gates comes off the ledger, so all of them now see 1000 days."""
    hk = "pro_hk"
    _seed(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())
    ledger = _even_ledger(400, realized_pnl_usd=7_000.0)
    ledger.purge_old_cps()

    manager._refresh_pro_stats([hk], {hk: ledger}, {hk: SimpleNamespace(account_size=ACCOUNT_SIZE)})

    stats = manager.miner_states[hk].pro_stats
    assert stats.trading_days == 400
    assert stats.daily_consistency == pytest.approx(1 / 400, rel=1e-3)


def test_realized_return_is_all_time_and_not_windowed():
    """The calmar numerator is the running realized total on the last checkpoint, never a window sum."""
    ledger = _even_ledger(1010, realized_pnl_usd=7_000.0)
    before = LedgerUtils.realized_return(ledger, ACCOUNT_SIZE)

    ledger.purge_old_cps()

    assert before == pytest.approx(0.07)
    assert LedgerUtils.realized_return(ledger, ACCOUNT_SIZE) == pytest.approx(0.07)
    assert Metrics.all_time_calmar(before, 0.96) == pytest.approx(CALMAR_THRESHOLD)
