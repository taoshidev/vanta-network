"""
The pro account track, end to end: how a subaccount gets onto it, what it has to clear to be
funded, and what takes it off.

  * Bucket taxonomy and the thresholds each pro bucket resolves.
  * The four promotion gates PRO_CHALLENGE_DIRECT and PRO_CHALLENGE_FROM_STANDARD clear to reach
    PRO_FUNDED:
      1. 90 full calendar days since the account's first trade
      2. 6% return on the account
      3. all-time calmar of at least 1.75, whose denominator is the largest drop of live equity
         below the end-of-day equity high-water mark
      4. daily return consistency of at most 20%, computed after capping each day at 1.5%
  * The two drawdown rules. Rule 1 is the daily loss limit against the day's opening equity;
    Rule 2 is the loss limit on the latched end-of-day equity against the end-of-day high-water
    mark, checked once per UTC day. Failing either is a hard breach: the subaccount is eliminated,
    on the challenge track as well as the funded one, with no fall back to the standard track.
  * The daily soft-breach latch that withholds a payout week, and what an elimination revert
    restores.
  * ChallengePeriodManager.promote_subaccount: the three hops an entity can ask for, the account
    switch that closes positions / cancels limit orders / restarts the ledgers, the sizing through
    a real EntityManager, and the rollback of a failed hop.
  * PRO_CHALLENGE_TRANSITION's effect on resting orders: the miner keeps trading the standard
    account and winds it down themselves, so only resting *entries* are swept.

The HTTP surface (promote, limits, pro trade pairs) lives in test_pro_endpoints.py, and the
account-sizing rules behind it in test_pro_account_size.py.
No RPC connections, no disk I/O, no daemon simulation.
"""
import contextlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.vali_tests.test_pro_account_size import (
    GRANTED_SIZE,
    REQUIRED,
    STANDARD_SIZE,
    _add_demoted,
    _add_standard as _add_standard_subaccount,
    _bare_manager as _bare_entity_manager,
)
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import (
    ChallengePeriodManager,
    DrawdownStats,
    MinerBucketState,
    ProStats,
)
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.limit_order_manager import LimitOrderManager
from vali_objects.utils.market_order.market_order_manager import OrderExecution
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_dataclasses.price_source import PriceSource

# ── Constants ─────────────────────────────────────────────────────────────────

DAILY_MS = ValiConfig.DAILY_MS
CP_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
ACCOUNT_SIZE = 100_000.0
HOTKEY = "pro_hk"

MIN_DAYS = ValiConfig.PRO_CHALLENGE_MINIMUM_DAYS                             # 90
RETURNS_THRESHOLD = ValiConfig.PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT       # 0.06
CALMAR_THRESHOLD = ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD                 # 1.75
CONSISTENCY_THRESHOLD = ValiConfig.PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD  # 0.20
DAILY_CAP = ValiConfig.PRO_DAILY_RETURN_CAP                                  # 0.015

INTRADAY_THRESHOLD = ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD    # 0.05
EOD_THRESHOLD = ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD              # 0.08

# The two buckets that promote to PRO_FUNDED. Both eliminate on a breach.
PRO_CHALLENGE_BUCKETS = (MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_FROM_STANDARD)
ALL_PRO_BUCKETS = (*PRO_CHALLENGE_BUCKETS, MinerBucket.PRO_FUNDED)
CHALLENGE_REASON = {
    "INTRADAY": EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN,
    "EOD": EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_EOD_DRAWDOWN,
}
FUNDED_REASON = {
    "INTRADAY": EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN,
    "EOD": EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN,
}

# The only bucket moves an entity can ask for, source -> target.
HOPS = (
    (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.PRO_CHALLENGE_DIRECT),
    (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_CHALLENGE_TRANSITION),
    (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD),
)
# Buckets with nowhere to promote to: the end of the pro track, the standard buckets off it, and
# the regular (non-subaccount) miner buckets.
NO_PROMOTION = (
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
    MinerBucket.PRO_CHALLENGE_DIRECT,
    MinerBucket.PRO_FUNDED,
    MinerBucket.SUBACCOUNT_ALPHA,
    MinerBucket.ELIMINATED,
    MinerBucket.MAINCOMP,
    MinerBucket.CHALLENGE,
)

LEDGER_START_MS = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC, a midnight boundary
NOW_MS = LEDGER_START_MS + 400 * DAILY_MS                # 2026-02-05 00:00 UTC, a Thursday
MIDNIGHT_MS = (NOW_MS // DAILY_MS) * DAILY_MS
MONDAY_MS = TimeUtil.ms_at_start_of_week(NOW_MS)         # the Monday 00:00 UTC that opened this week
NEXT_MONDAY_MS = MONDAY_MS + 7 * DAILY_MS

_CP_CLIENT_PATHS = [
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

_LIMIT_ORDER_CLIENT_PATHS = [
    "vali_objects.utils.market_order.market_order_client.MarketOrderClient",
    "vali_objects.price_fetcher.live_price_client.LivePriceFetcherClient",
    "vali_objects.position_management.position_manager_client.PositionManagerClient",
    "vali_objects.miner_account.miner_account_client.MinerAccountClient",
]


# ── Fixtures & helpers ────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    """A ChallengePeriodManager with every RPC client mocked and sizing calls accepted."""
    with contextlib.ExitStack() as stack:
        for path in _CP_CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        mgr._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
        yield mgr


@pytest.fixture
def hop_manager(manager):
    """The same manager with the account sync stubbed, for tests that move buckets directly."""
    with patch.object(manager, "_sync_buckets_to_accounts"):
        yield manager


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
    return MinerBucketState(HOTKEY, [BucketEntry(bucket, start_ms)])


def _healthy_drawdown(equity: float = 1.07) -> DrawdownStats:
    """Clear of both drawdown rules, with `equity` as both the live and the latched EOD value."""
    return DrawdownStats(current_equity=equity, current_balance=equity, daily_open_equity=equity,
                         eod_hwm=equity, last_eod_equity=equity, last_eod_checked_ms=MIDNIGHT_MS)


def _intraday_breach() -> DrawdownStats:
    """5.1% below the day's open, with the EOD mark intact so only the daily rule binds."""
    return DrawdownStats(current_equity=0.949, current_balance=0.949, daily_open_equity=1.0,
                         eod_hwm=1.0, last_eod_equity=1.0, last_eod_checked_ms=MIDNIGHT_MS)


def _eod_breach() -> DrawdownStats:
    """Latched EOD equity 8.1% below the mark, only 1% below the day's open."""
    return DrawdownStats(current_equity=0.919, current_balance=0.919, daily_open_equity=0.928,
                         eod_hwm=1.0, last_eod_equity=0.919, last_eod_checked_ms=MIDNIGHT_MS)


def _live_dip_only() -> DrawdownStats:
    """Live equity 8.57% below the EOD high-water mark, but the latched EOD equity sits at the
    mark and the day's open is only 3% away. Breaches nothing now that Rule 2 is EOD-only."""
    return DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=0.99,
                         eod_hwm=1.05, last_eod_equity=1.05, last_eod_checked_ms=MIDNIGHT_MS)


def _breach(rule: str) -> DrawdownStats:
    return _intraday_breach() if rule == "INTRADAY" else _eod_breach()


def _expected_reason(bucket: MinerBucket, rule: str) -> EliminationReason:
    return FUNDED_REASON[rule] if bucket == MinerBucket.PRO_FUNDED else CHALLENGE_REASON[rule]


def _passing_pro_stats() -> ProStats:
    return ProStats(calmar=CALMAR_THRESHOLD, daily_consistency=CONSISTENCY_THRESHOLD,
                    max_drawdown=0.96, trading_days=MIN_DAYS)


def _breaching_pro_stats() -> ProStats:
    """Below the calmar line, so `soft_breach` is true in PRO_FUNDED."""
    return ProStats(calmar=CALMAR_THRESHOLD - 0.5, daily_consistency=CONSISTENCY_THRESHOLD,
                    max_drawdown=0.96, trading_days=MIN_DAYS)


def _promotable_state(bucket: MinerBucket) -> MinerBucketState:
    """A state clearing all four gates, so a single tweak isolates the gate under test."""
    state = _state(bucket)
    state.drawdown = _healthy_drawdown(1.0 + RETURNS_THRESHOLD + 0.01)
    state.pro_stats = _passing_pro_stats()
    return state


def _seed(manager, hk: str, bucket: MinerBucket, drawdown: DrawdownStats,
          criteria: DrawdownCriteria = DrawdownCriteria.TRAILING) -> MinerBucketState:
    manager.set_miner_bucket(hk, bucket, NOW_MS - DAILY_MS, drawdown_criteria=criteria)
    manager.miner_states[hk].drawdown = drawdown
    return manager.miner_states[hk]


def _accounts(hk: str, account_size: float | None = ACCOUNT_SIZE) -> dict:
    return {hk: SimpleNamespace(account_size=account_size)} if account_size else {}


def _refresh_pro_stats(manager, hk: str, ledger: PerfLedger | None = None,
                       account_size: float | None = ACCOUNT_SIZE) -> ProStats:
    manager._refresh_pro_stats([hk], {hk: ledger} if ledger else {}, _accounts(hk, account_size))
    return manager.miner_states[hk].pro_stats


def _run_refresh(manager, hk: str, ledger: PerfLedger | None = None,
                 account_size: float | None = ACCOUNT_SIZE, now_ms: int = NOW_MS) -> None:
    """One refresh() pass with the drawdown cache pinned, so pro stats and routing run for real."""
    manager._position_client.get_all_hotkeys.return_value = [hk]
    manager._position_client.filtered_positions_for_scoring.return_value = ({hk: []}, {})
    manager._position_client.get_positions_for_hotkeys.return_value = {hk: []}
    manager._elimination_client.get_eliminated_hotkeys.return_value = []
    manager._plagiarism_client.get_plagiarism_miners.return_value = []
    manager._miner_account_client.get_accounts.return_value = _accounts(hk, account_size)
    manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = ({hk: ledger} if ledger else {})
    manager._asset_selection_client.get_asset_selections.return_value = {hk: MinerAssetClass.CRYPTO}
    with (
        patch.object(manager, "_refresh_drawdown_cache"),
        patch.object(manager, "_refresh_rank_cache"),
        patch.object(manager, "_save_to_disk"),
        patch.object(manager, "_sync_buckets_to_accounts"),
    ):
        manager.refresh(current_time_ms=now_ms)


def _elimination_kwargs(manager) -> dict:
    return manager._elimination_client.append_elimination_row.call_args.kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Bucket taxonomy and thresholds
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bucket", ALL_PRO_BUCKETS)
def test_pro_bucket_taxonomy(bucket):
    """Pro buckets are subaccounts on their own track: active, never rank-based, never timed out."""
    assert bucket.is_pro is True
    assert bucket.is_pro_track is True
    assert bucket.is_subaccount is True
    assert bucket.is_active is True
    assert bucket.is_rank_based is False
    assert bucket.max_time_ms is None


def test_the_pro_track_ends_at_pro_funded():
    for bucket in PRO_CHALLENGE_BUCKETS:
        assert bucket.next_bucket is MinerBucket.PRO_FUNDED
    assert MinerBucket.PRO_FUNDED.next_bucket is None
    # PRO_FUNDED is deliberately absent from switches_account: passing the challenge keeps the
    # same account, so the ratio the miner passed with carries over
    assert MinerBucket.PRO_FUNDED.switches_account is False
    assert all(b.switches_account for b in PRO_CHALLENGE_BUCKETS)


def test_only_the_direct_challenge_earns_nothing():
    """A subaccount that came up through the transition keeps earning (at the scaled payout) while
    it sits in the challenge; one that went straight onto a pro account does not."""
    assert MinerBucket.PRO_CHALLENGE_DIRECT.is_subaccount_earning is False
    assert MinerBucket.PRO_CHALLENGE_FROM_STANDARD.is_subaccount_earning is True
    assert MinerBucket.PRO_CHALLENGE_FROM_STANDARD.payout_scale_applies is True
    assert MinerBucket.PRO_FUNDED.is_subaccount_earning is True
    assert MinerBucket.PRO_FUNDED.payout_scale_applies is False
    # Only PRO_FUNDED withholds a week on a soft breach
    assert MinerBucket.PRO_FUNDED.soft_breach_applies is True
    assert not any(b.soft_breach_applies for b in PRO_CHALLENGE_BUCKETS)


def test_the_transition_is_on_the_track_but_trades_the_standard_account():
    transition = MinerBucket.PRO_CHALLENGE_TRANSITION
    assert transition.is_pro_track is True
    assert transition.is_pro is False  # so it keeps the standard curve and the standard rules
    assert transition.next_bucket is MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    # It leaves on the week boundary, not a time limit of its own
    assert transition.max_time_ms is None


def test_promotion_gate_thresholds():
    """The four gates, read off the bucket rather than hard-coded at each call site."""
    for bucket in PRO_CHALLENGE_BUCKETS:
        assert bucket.minimum_trading_days == 90
        assert bucket.calmar_threshold == 1.75
        assert bucket.daily_consistency_threshold == 0.20
        for asset_class in MinerAssetClass:
            assert bucket.returns_threshold(asset_class) == 0.06
    assert ValiConfig.PRO_DAILY_RETURN_CAP == 0.015


def test_drawdown_thresholds_are_configured_independently_of_the_standard_track():
    """Pro values come from their own config keys, so the standard ones can move on their own."""
    for bucket in PRO_CHALLENGE_BUCKETS:
        assert bucket.intraday_drawdown_threshold() == ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD
        assert bucket.eod_drawdown_threshold() == ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD
    assert MinerBucket.PRO_FUNDED.intraday_drawdown_threshold() == ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD
    assert MinerBucket.PRO_FUNDED.eod_drawdown_threshold() == ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD
    for bucket in ALL_PRO_BUCKETS:
        assert bucket.intraday_drawdown_threshold() == 0.05
        assert bucket.eod_drawdown_threshold() == 0.08


def test_pro_thresholds_only_resolve_for_pro_buckets():
    assert MinerBucket.SUBACCOUNT_CHALLENGE.calmar_threshold is None
    assert MinerBucket.SUBACCOUNT_CHALLENGE.daily_consistency_threshold is None
    assert MinerBucket.SUBACCOUNT_CHALLENGE.minimum_trading_days is None


def test_account_type_validation():
    assert AccountType.is_valid("standard") is True
    assert AccountType.is_valid("pro") is True
    assert AccountType.is_valid("nonsense") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — State persistence
# ═══════════════════════════════════════════════════════════════════════════════

def test_pro_state_round_trips_through_the_checkpoint():
    """The penalty ledger reads the pro stats and the latch out of the challenge period
    checkpoint, so both have to survive serialization and a validator restart."""
    state = _state(MinerBucket.PRO_FUNDED)
    state.pro_stats = ProStats(calmar=1.5, daily_consistency=0.25, max_drawdown=0.94,
                               trading_days=42, soft_breach_days=[MIDNIGHT_MS - DAILY_MS, MIDNIGHT_MS])

    restored = MinerBucketState.from_checkpoint_dict(HOTKEY, state.to_checkpoint_dict())

    assert restored.current_bucket is MinerBucket.PRO_FUNDED
    assert restored.pro_stats == state.pro_stats


def test_pro_stats_default_when_missing_from_the_checkpoint():
    data = _state(MinerBucket.PRO_FUNDED).to_checkpoint_dict()
    del data["pro_stats"]

    assert MinerBucketState.from_checkpoint_dict(HOTKEY, data).pro_stats == ProStats()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Gate 1: 90 full calendar days since the first trade
# ═══════════════════════════════════════════════════════════════════════════════

def test_trading_days_counts_one_full_calendar_day_at_a_time(manager):
    """The ledger starts at the first order, so its full days are calendar days on the account.
    A day it only half covered does not count, so the 90 are 90 *full* days."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())

    assert _refresh_pro_stats(manager, HOTKEY, _even_ledger(MIN_DAYS)).trading_days == MIN_DAYS

    partial = _even_ledger(MIN_DAYS, partial_last_day=True)
    assert _refresh_pro_stats(manager, HOTKEY, partial).trading_days == MIN_DAYS - 1


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_days_count_from_the_first_trade_not_from_time_in_the_bucket(bucket):
    """A year parked in the bucket without trading still fails; 90 traded days pass on day one."""
    idle = _promotable_state(bucket)
    idle.entries[-1].start_time_ms = NOW_MS - 365 * DAILY_MS
    idle.pro_stats.trading_days = MIN_DAYS - 1
    assert ChallengePeriodManager._check_promotion(idle, RETURNS_THRESHOLD, NOW_MS) is False

    traded = _promotable_state(bucket)
    traded.entries[-1].start_time_ms = NOW_MS
    assert ChallengePeriodManager._check_promotion(traded, RETURNS_THRESHOLD, NOW_MS) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Gate 2: 6% return
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_return_is_the_lesser_of_equity_and_balance():
    """Unrealized gains do not count toward the 6%: the balance caps the return."""
    state = _promotable_state(MinerBucket.PRO_CHALLENGE_DIRECT)
    state.drawdown.current_equity = 1.20
    state.drawdown.current_balance = 1.05

    assert state.drawdown.current_return == pytest.approx(0.05)
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Gate 3: all-time calmar over the EOD high-water-mark drawdown
# ═══════════════════════════════════════════════════════════════════════════════

def test_calmar_denominator_is_live_equity_against_the_eod_high_water_mark(manager):
    """7% realized over a 4% drop below the midnight mark is exactly the 1.75 threshold."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=0.96,
                        eod_hwm=1.0, last_eod_equity=0.96))

    stats = _refresh_pro_stats(manager, HOTKEY, _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0))

    assert stats.max_drawdown == pytest.approx(0.96)
    assert stats.calmar == pytest.approx(CALMAR_THRESHOLD)


def test_the_mark_only_moves_at_utc_midnight(manager):
    """A peak earlier today cannot deepen the drawdown, and the ledger's own mdd never sets it."""
    # Live equity is 0.96 and the account traded up to 1.30 earlier today, but the last midnight
    # latch is 1.00, so the drawdown is 4%, not the 26% an intraday peak would imply.
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=1.0,
                        eod_hwm=1.0, last_eod_equity=1.0))
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)
    for cp in ledger.cps:
        cp.mdd = 0.50  # a foil: the ratchet reads the drawdown cache, not the ledger

    assert _refresh_pro_stats(manager, HOTKEY, ledger).max_drawdown == pytest.approx(0.96)


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


def test_the_ratchet_never_hands_the_denominator_back(manager):
    """What makes the drawdown all-time: recovering does not give the calmar denominator back."""
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)

    # Day one: equity 5% under the midnight mark
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.95, current_balance=0.95, daily_open_equity=0.97,
                        eod_hwm=1.0, last_eod_equity=0.97))
    assert _refresh_pro_stats(manager, HOTKEY, ledger).max_drawdown == pytest.approx(0.95)

    # Later: fully recovered and 7% up, but the 5% drawdown still divides the calmar
    manager.miner_states[HOTKEY].drawdown = _healthy_drawdown(1.07)
    stats = _refresh_pro_stats(manager, HOTKEY, ledger)

    assert stats.max_drawdown == pytest.approx(0.95)
    assert stats.calmar == pytest.approx(0.07 / 0.05)
    assert stats.calmar < CALMAR_THRESHOLD
    assert ChallengePeriodManager._check_promotion(
        manager.miner_states[HOTKEY], RETURNS_THRESHOLD, NOW_MS) is False


def test_the_ratchet_clamps_at_par(manager):
    """Equity above the midnight mark is not a negative drawdown."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=1.50, daily_open_equity=1.50, eod_hwm=1.20,
                        last_eod_equity=1.50))

    assert _refresh_pro_stats(manager, HOTKEY, _even_ledger(5)).max_drawdown == 1.0


def test_the_ratchet_holds_on_a_corrupt_high_water_mark(manager):
    """A non-positive mark holds the ratchet rather than handing back a fresh denominator."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.5, eod_hwm=0.0))
    manager.miner_states[HOTKEY].pro_stats = ProStats(max_drawdown=0.9)

    assert _refresh_pro_stats(manager, HOTKEY, _even_ledger(5)).max_drawdown == pytest.approx(0.9)


def test_the_ratchet_survives_a_missing_ledger(manager):
    """A missing ledger skips calmar, but must not lose a drawdown that already happened."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.9, daily_open_equity=0.9, eod_hwm=1.0, last_eod_equity=0.9))

    stats = _refresh_pro_stats(manager, HOTKEY, ledger=None, account_size=None)

    assert stats.max_drawdown == pytest.approx(0.9)
    assert stats.calmar == 0.0
    assert stats.trading_days == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Gate 4: 20% return consistency on daily returns capped at 1.5%
# ═══════════════════════════════════════════════════════════════════════════════

def test_consistency_caps_each_day_at_one_and_a_half_percent(manager):
    """A 10% day counts as 1.5%, so six 1% days carry it to exactly the 20% limit."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())

    stats = _refresh_pro_stats(manager, HOTKEY, _ledger([math.log(1.10)] + [math.log(1.01)] * 6))

    # capped total = 0.015 + 6 x 0.01 = 0.075, best day = 0.015
    assert stats.daily_consistency == pytest.approx(DAILY_CAP / 0.075)
    assert stats.daily_consistency == pytest.approx(CONSISTENCY_THRESHOLD)


def test_consistency_breaches_when_the_best_day_carries_the_account(manager):
    """The same 10% day against only three 1% days is 33% of the capped total."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())

    stats = _refresh_pro_stats(manager, HOTKEY, _ledger([math.log(1.10)] + [math.log(1.01)] * 3))

    assert stats.daily_consistency == pytest.approx(DAILY_CAP / 0.045)
    assert stats.daily_consistency > CONSISTENCY_THRESHOLD


def test_consistency_counts_losing_days_in_full(manager):
    """Losses are not capped, so they shrink the total and push the best day's share up."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())

    clean = _refresh_pro_stats(manager, HOTKEY, _ledger([math.log(1.01)] * 8)).daily_consistency
    after_loss = _refresh_pro_stats(
        manager, HOTKEY, _ledger([math.log(1.01)] * 8 + [math.log(0.96)])).daily_consistency

    assert clean == pytest.approx(0.125)  # 0.01 / 0.08
    assert after_loss == pytest.approx(0.01 / 0.04)
    assert after_loss > clean


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — every gate is a hard boundary
# ═══════════════════════════════════════════════════════════════════════════════

def _set_gate(state: MinerBucketState, gate: str, *, passing: bool) -> None:
    """Move `gate` to the value just inside or just outside the line, leaving the others clear."""
    if gate == "trading_days":
        state.pro_stats.trading_days = MIN_DAYS if passing else MIN_DAYS - 1
    elif gate == "return":
        state.drawdown = _healthy_drawdown(1.0 + RETURNS_THRESHOLD + (0.001 if passing else -0.001))
    elif gate == "calmar":
        state.pro_stats.calmar = CALMAR_THRESHOLD if passing else CALMAR_THRESHOLD - 0.01
    elif gate == "consistency":
        state.pro_stats.daily_consistency = (CONSISTENCY_THRESHOLD if passing
                                             else CONSISTENCY_THRESHOLD + 0.01)
    else:
        raise AssertionError(f"unknown gate {gate}")


@pytest.mark.parametrize("gate", ["trading_days", "return", "calmar", "consistency"])
@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_each_gate_blocks_just_under_the_line_and_passes_at_it(bucket, gate):
    blocked = _promotable_state(bucket)
    _set_gate(blocked, gate, passing=False)
    assert ChallengePeriodManager._check_promotion(blocked, RETURNS_THRESHOLD, NOW_MS) is False

    allowed = _promotable_state(bucket)
    _set_gate(allowed, gate, passing=True)
    assert ChallengePeriodManager._check_promotion(allowed, RETURNS_THRESHOLD, NOW_MS) is True


def test_a_pro_miner_with_no_computed_stats_cannot_promote():
    """Default ProStats has no tracked days, so an unscored account never slips through."""
    state = _promotable_state(MinerBucket.PRO_CHALLENGE_DIRECT)
    state.pro_stats = ProStats()

    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


def test_non_pro_buckets_ignore_pro_stats():
    """The gates hang off the bucket, so a standard subaccount promotes on its returns alone even
    with pro metrics that would fail every pro gate."""
    standard_threshold = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE)
    state.drawdown = _healthy_drawdown(1.0 + standard_threshold + 0.01)
    state.pro_stats = ProStats(calmar=-100.0, daily_consistency=1.0)

    assert ChallengePeriodManager._check_promotion(state, standard_threshold, NOW_MS) is True


def test_the_transition_bucket_never_promotes_on_returns():
    """The transition runs to the week boundary: returns cannot shorten it, however good they are.
    Blocking the returns path leaves that boundary as the only automatic way out."""
    state = _promotable_state(MinerBucket.PRO_CHALLENGE_TRANSITION)
    state.drawdown = DrawdownStats(current_equity=2.0, current_balance=2.0)
    assert ChallengePeriodManager._check_promotion(state, RETURNS_THRESHOLD, NOW_MS) is False


def test_the_transition_ends_at_the_next_monday_however_short_that_is():
    """Entering mid-week does not buy a full week: the wind-down closes at the first Monday 00:00
    UTC after the miner entered, so a Saturday entry has two days of it."""
    state = _state(MinerBucket.PRO_CHALLENGE_TRANSITION, MONDAY_MS - 2 * DAILY_MS)  # the Saturday before

    assert ChallengePeriodManager._check_transition_expiry(state, MONDAY_MS - DAILY_MS) is False
    assert ChallengePeriodManager._check_transition_expiry(state, MONDAY_MS - 1) is False
    # The boundary itself, and every refresh after it: whichever run comes first promotes them
    assert ChallengePeriodManager._check_transition_expiry(state, MONDAY_MS) is True
    assert ChallengePeriodManager._check_transition_expiry(state, NOW_MS) is True


def test_a_transition_entered_on_the_boundary_keeps_the_whole_week():
    """The Monday has to be crossed, so a miner moved in at the seam is not promoted straight back
    out on the same refresh."""
    state = _state(MinerBucket.PRO_CHALLENGE_TRANSITION, MONDAY_MS)

    assert ChallengePeriodManager._check_transition_expiry(state, MONDAY_MS) is False
    assert ChallengePeriodManager._check_transition_expiry(state, NEXT_MONDAY_MS - 1) is False
    assert ChallengePeriodManager._check_transition_expiry(state, NEXT_MONDAY_MS) is True


@pytest.mark.parametrize("bucket", [b for b in MinerBucket if b != MinerBucket.PRO_CHALLENGE_TRANSITION])
def test_no_other_bucket_leaves_on_the_week_boundary(bucket):
    """Only the transition is advanced by the calendar; every other bucket promotes on its own
    gates however many Mondays it has sat through."""
    state = _state(bucket, NOW_MS - 400 * DAILY_MS)
    assert ChallengePeriodManager._check_transition_expiry(state, NOW_MS) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — all four gates through refresh()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_refresh_promotes_to_pro_funded_when_every_gate_clears(manager, bucket):
    """90 even days, 8% realized, a 4% ratcheted drawdown and 8% live equity: calmar is 2.0."""
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=8_000.0)

    # The 4% drawdown happened earlier in the account's life and has since recovered
    _seed(manager, HOTKEY, bucket,
          DrawdownStats(current_equity=0.96, current_balance=0.96, daily_open_equity=0.97,
                        eod_hwm=1.0, last_eod_equity=0.97))
    assert _refresh_pro_stats(manager, HOTKEY, ledger).max_drawdown == pytest.approx(0.96)
    manager.miner_states[HOTKEY].drawdown = _healthy_drawdown(1.08)

    _run_refresh(manager, HOTKEY, ledger=ledger)

    # The PRO_FUNDED hop keeps the account, so the funded miner carries the ratio it passed with
    # rather than restarting at calmar 0 (which would read as an immediate soft breach)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_FUNDED
    stats = manager.miner_states[HOTKEY].pro_stats
    assert stats.calmar == pytest.approx(2.0)
    assert stats.max_drawdown == pytest.approx(0.96)
    assert stats.trading_days == MIN_DAYS


def test_refresh_holds_in_bucket_when_the_ledger_is_a_day_short(manager):
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown(1.07))

    _run_refresh(manager, HOTKEY, ledger=_even_ledger(MIN_DAYS - 1, realized_pnl_usd=7_000.0))

    assert manager.miner_states[HOTKEY].pro_stats.trading_days == MIN_DAYS - 1
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_CHALLENGE_DIRECT


def test_refresh_holds_in_bucket_when_the_ratcheted_drawdown_sinks_calmar(manager):
    """Everything else clears, but a 10% drawdown earlier in the account's life leaves calmar at 0.7."""
    ledger = _even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0)
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT,
          DrawdownStats(current_equity=0.90, current_balance=0.90, daily_open_equity=0.94,
                        eod_hwm=1.0, last_eod_equity=0.94))
    _refresh_pro_stats(manager, HOTKEY, ledger)
    manager.miner_states[HOTKEY].drawdown = _healthy_drawdown(1.07)

    _run_refresh(manager, HOTKEY, ledger=ledger)

    assert manager.miner_states[HOTKEY].pro_stats.calmar == pytest.approx(0.70)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_CHALLENGE_DIRECT


def test_refresh_holds_in_bucket_when_one_day_carries_the_return(manager):
    """90 days and 7% realized, but a single uncapped spike day fails the consistency gate."""
    # 0.05% filler days keep the capped total thin enough that the one spike owns 25% of it
    ledger = _ledger([math.log(1.10)] + [math.log(1.0005)] * (MIN_DAYS - 1), realized_pnl_usd=7_000.0)
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown(1.07))

    _run_refresh(manager, HOTKEY, ledger=ledger)

    stats = manager.miner_states[HOTKEY].pro_stats
    assert stats.trading_days == MIN_DAYS
    assert stats.daily_consistency > CONSISTENCY_THRESHOLD
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_CHALLENGE_DIRECT


def test_refresh_moves_the_transition_onto_the_pro_account_once_the_week_turns(manager):
    """End to end: a miner that entered the wind-down on Sunday is still in it on Sunday night and
    is moved on by the first refresh past Monday 00:00 UTC, with nothing but the calendar changing."""
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, MONDAY_MS - DAILY_MS)
    manager.miner_states[HOTKEY].drawdown = _healthy_drawdown(1.0)

    _run_refresh(manager, HOTKEY, now_ms=MONDAY_MS - 1)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_CHALLENGE_TRANSITION

    _run_refresh(manager, HOTKEY, now_ms=MONDAY_MS)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_CHALLENGE_FROM_STANDARD


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9 — the 1000-day perf ledger window the pro metrics read
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_configured_ledger_window_is_one_thousand_days():
    assert ValiConfig.TARGET_LEDGER_WINDOW_DAYS == 1000
    assert ValiConfig.TARGET_LEDGER_WINDOW_MS == 1000 * DAILY_MS
    assert PerfLedger().target_ledger_window_ms == ValiConfig.TARGET_LEDGER_WINDOW_MS

    # Ledgers serialized under the old 180-day window must not keep purging at 180
    stale = _even_ledger(5).to_dict()
    stale["target_ledger_window_ms"] = 180 * DAILY_MS
    assert PerfLedger.from_dict(stale).target_ledger_window_ms == ValiConfig.TARGET_LEDGER_WINDOW_MS


def test_purge_keeps_the_window_and_trims_past_it():
    inside = _even_ledger(999)
    n_cps = len(inside.cps)
    inside.purge_old_cps()
    assert len(inside.cps) == n_cps
    assert inside.get_total_ledger_duration_ms() == 999 * DAILY_MS

    over = _even_ledger(1010)
    over.purge_old_cps()
    assert over.get_total_ledger_duration_ms() == ValiConfig.TARGET_LEDGER_WINDOW_MS


def test_pro_metrics_read_the_whole_window_not_the_old_one_hundred_and_eighty_days(manager):
    """Every metric behind the pro gates comes off the ledger, so all of them now see 1000 days."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())
    ledger = _even_ledger(400, realized_pnl_usd=7_000.0)
    ledger.purge_old_cps()

    stats = _refresh_pro_stats(manager, HOTKEY, ledger)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10 — the two drawdown rules eliminate
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("rule", ["INTRADAY", "EOD"])
@pytest.mark.parametrize("bucket", ALL_PRO_BUCKETS)
def test_a_drawdown_breach_eliminates_with_the_buckets_own_reason(manager, bucket, rule):
    """Every promotion gate clears in this pass, so the breach is what decides: a subaccount that
    would otherwise have been promoted is eliminated instead."""
    _seed(manager, HOTKEY, bucket, _breach(rule))
    manager.miner_states[HOTKEY].pro_stats = _passing_pro_stats()

    _run_refresh(manager, HOTKEY, ledger=_even_ledger(MIN_DAYS, realized_pnl_usd=7_000.0))

    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.ELIMINATED
    assert _elimination_kwargs(manager)["reason"] == _expected_reason(bucket, rule)


def test_the_elimination_row_carries_both_drawdown_numbers(manager):
    """An 8.1% EOD breach is reported as such, next to the 1% the day itself was down."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _eod_breach())

    _run_refresh(manager, HOTKEY)

    kwargs = _elimination_kwargs(manager)
    assert kwargs["elimination_drawdown_pct"] == pytest.approx(8.1, abs=0.05)
    assert kwargs["eod_drawdown_pct"] == pytest.approx(8.1, abs=0.05)
    assert kwargs["intraday_drawdown_pct"] == pytest.approx(0.97, abs=0.05)


def test_the_elimination_row_backdates_to_the_last_eod_latch(manager):
    """The breach happened at the midnight latch, not on the refresh pass that noticed it."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _eod_breach())
    _run_refresh(manager, HOTKEY)
    assert _elimination_kwargs(manager)["elimination_time_ms"] == MIDNIGHT_MS

    no_latch = _eod_breach()
    no_latch.last_eod_checked_ms = None
    _seed(manager, "other_hk", MinerBucket.PRO_FUNDED, no_latch)
    _run_refresh(manager, "other_hk")
    assert _elimination_kwargs(manager)["elimination_time_ms"] == NOW_MS


@pytest.mark.parametrize("rule", ["INTRADAY", "EOD"])
def test_a_pro_subaccount_created_static_still_runs_the_pro_rules(manager, rule):
    """Pro buckets run the pro rules whatever drawdown_criteria the subaccount was created with,
    so the row carries the pro number rather than static equity-vs-starting-balance."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _breach(rule), criteria=DrawdownCriteria.STATIC)

    _run_refresh(manager, HOTKEY)

    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.ELIMINATED
    assert _elimination_kwargs(manager)["reason"] == FUNDED_REASON[rule]


def test_the_transition_stays_on_the_standard_static_rules(manager):
    """PRO_CHALLENGE_TRANSITION still trades the standard account, so a static breach binds it."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION,
          DrawdownStats(current_equity=1.0 - ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD - 0.001),
          criteria=DrawdownCriteria.STATIC)

    _run_refresh(manager, HOTKEY)

    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.ELIMINATED
    assert _elimination_kwargs(manager)["reason"] == EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN


def test_a_standard_subaccount_keeps_the_standard_daily_loss_limit(manager):
    """The foil for the pro routing: equity is up on the starting balance, so the static rule is
    clear, but it is more than the standard threshold below the day's open, so the daily rule binds."""
    day_open = 1.10
    breach = day_open * (1 - ValiConfig.SUBACCOUNT_STATIC_INTRADAY_DRAWDOWN_THRESHOLD) - 0.001
    drawdown = DrawdownStats(current_equity=breach, daily_open_equity=day_open,
                             eod_hwm=day_open, last_eod_equity=day_open)
    clear_of_the_static_rule = _state(MinerBucket.SUBACCOUNT_FUNDED)
    clear_of_the_static_rule.drawdown = drawdown
    assert ChallengePeriodManager._check_static_drawdown(clear_of_the_static_rule) is None

    _seed(manager, "static_hk", MinerBucket.SUBACCOUNT_FUNDED, drawdown,
          criteria=DrawdownCriteria.STATIC)
    _run_refresh(manager, "static_hk")

    assert manager.get_miner_bucket("static_hk") == MinerBucket.ELIMINATED
    assert _elimination_kwargs(manager)["reason"] == EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN


@pytest.mark.parametrize("rule", ["INTRADAY", "EOD"])
@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_just_inside_either_limit_keeps_trading(manager, bucket, rule):
    just_inside = 1.0 - (INTRADAY_THRESHOLD if rule == "INTRADAY" else EOD_THRESHOLD) + 0.001
    open_equity = 1.0 if rule == "INTRADAY" else just_inside
    _seed(manager, HOTKEY, bucket,
          DrawdownStats(current_equity=just_inside, current_balance=just_inside,
                        daily_open_equity=open_equity, eod_hwm=1.0,
                        last_eod_equity=just_inside if rule == "EOD" else 1.0,
                        last_eod_checked_ms=MIDNIGHT_MS))

    _run_refresh(manager, HOTKEY, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(HOTKEY) == bucket
    manager._position_client.close_all_positions.assert_not_called()


def test_a_live_dip_below_the_mark_is_not_a_breach_but_still_deepens_calmar(manager):
    """The spec's two halves in one scenario: Rule 2 is EOD-only, so live equity 8.57% under the
    midnight mark neither eliminates nor demotes -- but it is exactly what the calmar denominator
    measures, so the miner keeps trading and still cannot promote."""
    drawdown = _live_dip_only()
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, drawdown)

    _run_refresh(manager, HOTKEY, ledger=_even_ledger(10))

    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.PRO_CHALLENGE_DIRECT
    manager._elimination_client.append_elimination_row.assert_not_called()

    manager.miner_states[HOTKEY].drawdown = drawdown
    stats = _refresh_pro_stats(manager, HOTKEY, _even_ledger(5, realized_pnl_usd=6_000.0))

    assert stats.max_drawdown == pytest.approx(0.96 / 1.05)  # 8.57% under the mark
    assert stats.calmar == pytest.approx(0.70)               # 0.06 / 0.0857
    assert stats.calmar < CALMAR_THRESHOLD                   # so promotion is blocked


@pytest.mark.parametrize("bucket", PRO_CHALLENGE_BUCKETS)
def test_a_breach_never_falls_back_to_the_standard_track(manager, bucket):
    """No demotion path: nothing resizes the subaccount back onto a standard account."""
    _seed(manager, "entity_hk_1", bucket, _intraday_breach())

    _run_refresh(manager, "entity_hk_1", ledger=_even_ledger(10))

    assert manager.get_miner_bucket("entity_hk_1") == MinerBucket.ELIMINATED
    manager._entity_client.apply_bucket_account_size.assert_not_called()
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()


def test_pro_buckets_never_run_the_static_drawdown_rule():
    """The pro track has no static elimination reason, so the static check must decline pro buckets."""
    for bucket in ALL_PRO_BUCKETS:
        state = _state(bucket)
        state.drawdown = DrawdownStats(current_equity=0.5)  # far past any static threshold
        assert ChallengePeriodManager._check_static_drawdown(state) is None

    for name in ("FAILED_PRO_CHALLENGE_PERIOD_STATIC_DRAWDOWN",
                 "FAILED_PRO_FUNDED_PERIOD_STATIC_DRAWDOWN",
                 "FAILED_PRO_CHALLENGE_PERIOD_STATIC_EOD_DRAWDOWN",
                 "FAILED_PRO_FUNDED_PERIOD_STATIC_EOD_DRAWDOWN"):
        assert name not in EliminationReason.__members__


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11 — the daily soft-breach latch
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_latch_records_the_utc_day_a_pro_rule_broke(manager):
    """The pro metrics move continuously but only land on 12h checkpoints, so a breach that heals
    inside a checkpoint would never withhold the week unless the day is latched. The refresh loop
    runs every 30s, so a day is recorded once, not once per tick."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = _breaching_pro_stats()

    for offset_ms in (0, 30_000, 60_000):
        manager._latch_soft_breaches([HOTKEY], NOW_MS + offset_ms)

    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == [MIDNIGHT_MS]
    assert manager.miner_states[HOTKEY].is_soft_breach_latched(MIDNIGHT_MS)


def test_a_breach_that_heals_still_leaves_the_day_latched(manager):
    """The whole point of latching: recovering before the next 12h sample does not un-withhold
    the week."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = _breaching_pro_stats()
    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    manager.miner_states[HOTKEY].pro_stats.calmar = CALMAR_THRESHOLD
    manager._latch_soft_breaches([HOTKEY], NOW_MS + 60_000)

    assert manager.miner_states[HOTKEY].soft_breach is False
    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == [MIDNIGHT_MS]


def test_refreshing_pro_stats_carries_the_latch(manager):
    """_refresh_pro_stats rebuilds ProStats from the ledger on every pass. The metrics are derived
    and should be recomputed, but the latch is a history - losing it would erase the days holding
    this week's payout twice a minute."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = _breaching_pro_stats()
    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    stats = _refresh_pro_stats(manager, HOTKEY, _even_ledger(MIN_DAYS, realized_pnl_usd=8_000.0))

    assert stats.trading_days == MIN_DAYS  # metrics did recompute
    assert stats.soft_breach_days == [MIDNIGHT_MS]


def test_a_clean_day_is_never_latched(manager):
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = _passing_pro_stats()

    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == []


def test_a_consistency_breach_alone_is_never_latched(manager):
    """Daily consistency only gates promotion into PRO_FUNDED. Once funded, a lumpy week is not a
    soft breach and the payout stands - calmar is the only rule that withholds it."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = ProStats(
        calmar=CALMAR_THRESHOLD, daily_consistency=CONSISTENCY_THRESHOLD + 0.5,
        max_drawdown=0.96, trading_days=MIN_DAYS)

    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].soft_breach is False
    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == []


@pytest.mark.parametrize("bucket", [MinerBucket.SUBACCOUNT_FUNDED,
                                    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                                    MinerBucket.PRO_CHALLENGE_TRANSITION])
def test_buckets_without_soft_breaches_are_never_latched(manager, bucket):
    """Standard subaccounts and the two buckets that keep earning through a breach are untouched."""
    _seed(manager, HOTKEY, bucket, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = _breaching_pro_stats()

    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == []


def test_latched_days_are_pruned_to_the_retention_window(manager):
    """Bounded so the challenge period checkpoint cannot grow without limit."""
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = _breaching_pro_stats()
    retention = ValiConfig.PRO_SOFT_BREACH_LATCH_RETENTION_DAYS
    stale_day = MIDNIGHT_MS - (retention + 5) * DAILY_MS
    manager.miner_states[HOTKEY].pro_stats.soft_breach_days = [stale_day]

    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == [MIDNIGHT_MS]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12 — reverting an elimination restores the account, it does not reset it
# ═══════════════════════════════════════════════════════════════════════════════

def _eliminated_state(manager, hk: str, bucket=MinerBucket.PRO_FUNDED) -> MinerBucketState:
    """An account carrying pro history that was then eliminated out of `bucket`."""
    state = MinerBucketState(hk, [BucketEntry(bucket, NOW_MS - 10 * DAILY_MS)])
    state.drawdown = _healthy_drawdown(1.07)
    state.pro_stats = _passing_pro_stats()
    state.pro_stats.soft_breach_days = [MIDNIGHT_MS - DAILY_MS]
    manager.miner_states[hk] = state
    state.add_bucket_entry(MinerBucket.ELIMINATED, NOW_MS)
    return state


@contextlib.contextmanager
def _quiet(manager):
    with patch.object(manager, "_save_to_disk"), patch.object(manager, "_sync_buckets_to_accounts"):
        yield


def test_a_reverted_elimination_is_kept_as_a_record_but_never_governs_a_timestamp(manager):
    """The elimination did happen and the audit trail keeps it. But bucket history is replayed
    point-in-time when the ledgers rebuild, and a reverted ELIMINATED span that still governed its
    window would reclassify every checkpoint inside it as non-earning and off the soft-breach
    track, forfeiting escrow for an elimination that was undone."""
    state = _eliminated_state(manager, HOTKEY)
    eliminated_at_ms = state.entries[-1].start_time_ms

    with _quiet(manager):
        assert manager.revert_elimination(HOTKEY)

    assert state.current_bucket == MinerBucket.PRO_FUNDED
    assert [e.bucket for e in state.entries] == [
        MinerBucket.PRO_FUNDED, MinerBucket.ELIMINATED, MinerBucket.PRO_FUNDED
    ]
    assert state.entries[1].is_reverted
    assert state.bucket(eliminated_at_ms) == MinerBucket.PRO_FUNDED
    assert state.bucket(NOW_MS - DAILY_MS) == MinerBucket.PRO_FUNDED


def test_the_penalty_ledger_skips_a_reverted_span_when_stamping_checkpoints(manager):
    """The penalty ledger replays the serialized history, so the revert has to survive the round
    trip into the challenge period checkpoint."""
    from vali_objects.vali_dataclasses.ledger.penalty.penalty_ledger import PenaltyLedgerManager

    state = _eliminated_state(manager, HOTKEY)
    eliminated_at_ms = state.entries[-1].start_time_ms

    with _quiet(manager):
        assert manager.revert_elimination(HOTKEY)

    status = PenaltyLedgerManager._get_status_for_checkpoint(
        PenaltyLedgerManager, eliminated_at_ms + 1, state.to_checkpoint_dict()
    )
    assert status == MinerBucket.PRO_FUNDED.value


def test_a_pro_revert_keeps_the_stats_the_latch_and_the_trailing_loss_limit(manager):
    """A week that was breached before the elimination stays breached after the revert, and
    eod_hwm (the denominator of the 8% trailing rule) is kept - clearing it would hand the miner a
    fresh 8% of room for free."""
    state = _eliminated_state(manager, HOTKEY)

    with _quiet(manager):
        assert manager.revert_elimination(HOTKEY)

    expected = _passing_pro_stats()
    expected.soft_breach_days = [MIDNIGHT_MS - DAILY_MS]
    assert state.pro_stats == expected
    assert state.drawdown.eod_hwm == pytest.approx(1.07)
    assert state.drawdown.last_eod_equity == pytest.approx(1.07)
    assert state.drawdown.last_eod_checked_ms == MIDNIGHT_MS


def test_only_pro_funded_gets_both_restorations(manager):
    """PRO_FUNDED has finished the pro challenge, so a reverted elimination leaves it as though
    the elimination never happened - the span earns again and the trailing loss limit is kept."""
    state = _eliminated_state(manager, HOTKEY, MinerBucket.PRO_FUNDED)

    with _quiet(manager):
        assert manager.revert_elimination(HOTKEY)

    assert state.entries[1].is_reverted
    assert state.drawdown.eod_hwm == pytest.approx(1.07)


@pytest.mark.parametrize("bucket", (*PRO_CHALLENGE_BUCKETS, MinerBucket.PRO_CHALLENGE_TRANSITION))
def test_a_pro_challenge_revert_does_not_pay_for_the_eliminated_span(manager, bucket):
    """Marking the span reverted makes it earn again, so it is withheld from every bucket that has
    not finished the pro challenge: an account let back in restarts from the elimination rather
    than being paid for the time it spent there."""
    state = _eliminated_state(manager, HOTKEY, bucket)
    eliminated_at_ms = state.entries[-1].start_time_ms

    with _quiet(manager):
        assert manager.revert_elimination(HOTKEY)

    assert state.current_bucket == bucket
    assert not state.entries[1].is_reverted
    assert state.bucket(eliminated_at_ms) == MinerBucket.ELIMINATED
    assert state.drawdown == DrawdownStats()


@pytest.mark.parametrize("bucket", [MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_CHALLENGE,
                                    MinerBucket.MAINCOMP])
def test_a_standard_revert_is_unchanged_from_before_the_pro_track(manager, bucket):
    """Neither restoration applies off the pro track: a standard subaccount or a regular miner
    reverts exactly the way it did before pro accounts existed - the eliminated span still governs
    its window, and the drawdown cache is cleared outright."""
    state = _eliminated_state(manager, "std_hk", bucket)
    eliminated_at_ms = state.entries[-1].start_time_ms

    with _quiet(manager):
        assert manager.revert_elimination("std_hk")

    assert state.current_bucket == bucket
    assert not state.entries[1].is_reverted
    assert state.bucket(eliminated_at_ms) == MinerBucket.ELIMINATED
    assert state.drawdown == DrawdownStats()


def test_revert_elimination_is_a_no_op_for_a_live_miner(manager):
    _seed(manager, HOTKEY, MinerBucket.PRO_FUNDED, _healthy_drawdown())

    assert manager.revert_elimination(HOTKEY) is False


def test_switching_account_clears_the_pro_stats_and_the_latch(manager):
    """A hop that really does start a new account gets a new breach history, unlike PRO_FUNDED:
    the ratchet must not survive onto the fresh account."""
    _seed(manager, HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, _healthy_drawdown())
    manager.miner_states[HOTKEY].pro_stats = ProStats(calmar=2.0, daily_consistency=0.1,
                                                      max_drawdown=0.8, trading_days=120,
                                                      soft_breach_days=[MIDNIGHT_MS])

    assert manager._switch_account(HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, NOW_MS) is True

    assert manager.miner_states[HOTKEY].pro_stats == ProStats()


def test_admin_move_to_pro_funded_sizes_the_account(manager):
    """PRO_FUNDED no longer runs the account switch, so the sizing it used to do has to happen
    explicitly - an admin can drop a subaccount straight in from a standard bucket."""
    _seed(manager, HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, _healthy_drawdown())

    with _quiet(manager), patch(
            "vali_objects.challenge_period.challengeperiod_manager.is_synthetic_hotkey",
            return_value=True):
        success, _ = manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_FUNDED, NOW_MS)

    assert success
    manager._entity_client.apply_bucket_account_size.assert_called_once_with(HOTKEY, MinerBucket.PRO_FUNDED)
    # ...and it is a sizing call only: the account itself is untouched
    manager._position_client.close_all_positions.assert_not_called()
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()


def test_admin_move_to_pro_funded_aborts_when_sizing_fails(manager):
    """A subaccount with no granted pro size must not end up in a pro bucket on a standard size."""
    _seed(manager, HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, _healthy_drawdown())
    manager._entity_client.apply_bucket_account_size.return_value = (False, REQUIRED)

    with _quiet(manager), patch(
            "vali_objects.challenge_period.challengeperiod_manager.is_synthetic_hotkey",
            return_value=True):
        success, message = manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_FUNDED, NOW_MS)

    assert not success
    assert REQUIRED in message
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13 — ChallengePeriodManager.promote_subaccount, the entity-initiated hops
# ═══════════════════════════════════════════════════════════════════════════════

def _in_bucket(manager, bucket, hotkey=HOTKEY):
    manager.set_miner_bucket(hotkey, bucket, NOW_MS)
    return manager


@pytest.mark.parametrize("source,target", HOPS)
def test_each_hop_moves_to_its_own_target(hop_manager, source, target):
    _in_bucket(hop_manager, source)

    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)

    assert success, message
    assert hop_manager.miner_states[HOTKEY].current_bucket == target
    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(HOTKEY, target, GRANTED_SIZE)


def test_an_omitted_size_is_passed_through_as_none(hop_manager):
    """A hop within the track may leave the size out; the entity manager falls back to the recorded one."""
    _in_bucket(hop_manager, MinerBucket.PRO_CHALLENGE_TRANSITION)

    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS)[0]

    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, None
    )


@pytest.mark.parametrize("source,target", [h for h in HOPS if h[1] != MinerBucket.PRO_CHALLENGE_TRANSITION])
def test_only_the_hops_onto_a_pro_account_wind_the_standard_account_down(hop_manager, source, target):
    _in_bucket(hop_manager, source)

    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)[0]

    assert target.switches_account
    hop_manager._position_client.close_all_positions.assert_called_once_with(
        hotkey=HOTKEY, close_time_ms=NOW_MS, order_source=OrderSource.SUBACCOUNT_PROMOTION
    )
    hop_manager._position_client.archive_positions_for_hotkey.assert_called_once_with(HOTKEY, archive_all=True)
    hop_manager._limit_order_client.cancel_limit_order.assert_called_once_with(HOTKEY, None, "ALL", NOW_MS)
    hop_manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_called_once_with([HOTKEY])
    hop_manager._debt_ledger_client.delete_debt_ledger.assert_called_once_with(HOTKEY)
    hop_manager._miner_account_client.reset_account.assert_called_once_with(HOTKEY, target)
    hop_manager._limit_order_client.cancel_entry_orders.assert_not_called()


def test_entering_the_transition_wipes_nothing_and_only_sweeps_entry_orders(hop_manager):
    """PRO_CHALLENGE_TRANSITION is a wind-down week on the standard account, so this hop keeps the
    subaccount's positions, limit orders and ledgers. Only the resting orders that could open or
    increase a position are cancelled; closes and reductions stay for the week."""
    _in_bucket(hop_manager, MinerBucket.SUBACCOUNT_FUNDED)

    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)[0]

    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    hop_manager._position_client.close_all_positions.assert_not_called()
    hop_manager._position_client.archive_positions_for_hotkey.assert_not_called()
    hop_manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()
    hop_manager._debt_ledger_client.delete_debt_ledger.assert_not_called()
    hop_manager._miner_account_client.reset_account.assert_not_called()
    # the blanket "cancel everything" sweep belongs to an account switch; this hop is not one
    hop_manager._limit_order_client.cancel_limit_order.assert_not_called()
    hop_manager._limit_order_client.cancel_entry_orders.assert_called_once_with(
        HOTKEY, NOW_MS, OrderSource.PRO_TRANSITION_CANCELLED
    )


@pytest.mark.parametrize("challenge_bucket", PRO_CHALLENGE_BUCKETS)
def test_pro_funded_keeps_the_account_it_passed_on(hop_manager, challenge_bucket):
    """Passing the pro challenge keeps the same account.

    Balance, equity, positions and the ledgers all carry over, so all-time calmar keeps the ratio
    the miner passed with. Challenge-period gains are kept out of the payout by the payout paths
    reading each checkpoint's own bucket, not by wiping the history.
    """
    _in_bucket(hop_manager, challenge_bucket)

    assert hop_manager.promote_hotkeys([HOTKEY], NOW_MS)

    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_FUNDED
    hop_manager._position_client.close_all_positions.assert_not_called()
    hop_manager._position_client.archive_positions_for_hotkey.assert_not_called()
    hop_manager._limit_order_client.cancel_limit_order.assert_not_called()
    hop_manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()
    hop_manager._debt_ledger_client.delete_debt_ledger.assert_not_called()
    hop_manager._miner_account_client.reset_account.assert_not_called()


def test_entity_rejection_blocks_the_promotion(hop_manager):
    """When the entity manager cannot size the pro account, nothing is wound down."""
    _in_bucket(hop_manager, MinerBucket.PRO_CHALLENGE_TRANSITION)
    hop_manager._entity_client.apply_bucket_account_size.return_value = (False, REQUIRED)

    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)

    assert not success
    assert REQUIRED in message
    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    hop_manager._position_client.close_all_positions.assert_not_called()


@pytest.mark.parametrize("bucket", NO_PROMOTION)
def test_buckets_off_the_promotion_path_are_refused(hop_manager, bucket):
    """Each hop is single use, so the target of the last one has no promotion of its own."""
    _in_bucket(hop_manager, bucket)

    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)

    assert not success
    assert bucket.value in message
    assert hop_manager.miner_states[HOTKEY].current_bucket == bucket
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()
    hop_manager._position_client.close_all_positions.assert_not_called()


def test_unknown_hotkey_is_rejected(hop_manager):
    success, message = hop_manager.promote_subaccount("not_a_miner_0", NOW_MS, GRANTED_SIZE)

    assert not success
    assert "not found" in message
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()


# ==================== sizing through a real EntityManager ====================

def _with_real_entity_manager(manager, bucket=MinerBucket.SUBACCOUNT_FUNDED,
                              hotkey_factory=_add_standard_subaccount):
    """Swap the mocked entity client for a real in-memory EntityManager holding one standard
    subaccount. Returns (entity_manager, synthetic_hotkey)."""
    entity_manager = _bare_entity_manager()
    hotkey = hotkey_factory(entity_manager)
    manager._entity_client = entity_manager
    manager.set_miner_bucket(hotkey, bucket, NOW_MS)
    return entity_manager, hotkey


def test_the_transition_records_the_size_then_the_next_hop_trades_it(hop_manager):
    """Entering the transition records the pro size but keeps trading the standard account; the
    hop out of it is what puts the subaccount on the pro account and rescales its payout."""
    entity_manager, hotkey = _with_real_entity_manager(hop_manager)

    assert hop_manager.promote_subaccount(hotkey, NOW_MS, 250_000)[0]

    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 250_000
    assert info.standard_account_size == STANDARD_SIZE
    assert info.account_size == STANDARD_SIZE
    # PRO_CHALLENGE_TRANSITION still trades the standard account, so it keeps its own class
    assert info.asset_class == "forex"

    success, message = hop_manager.promote_subaccount(hotkey, NOW_MS + 1000)

    assert success, message
    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 250_000
    assert info.account_size == 250_000
    assert info.standard_account_size == STANDARD_SIZE
    assert entity_manager.get_payout_scale(hotkey) == pytest.approx(
        ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * STANDARD_SIZE / 250_000)


def test_the_direct_hop_trades_the_size_immediately_and_widens_the_asset_class(hop_manager):
    """The hop off the standard challenge goes straight onto the pro account, whose universe spans
    every asset class."""
    entity_manager, hotkey = _with_real_entity_manager(
        hop_manager, bucket=MinerBucket.SUBACCOUNT_CHALLENGE)
    assert entity_manager.get_subaccount_info_for_synthetic(hotkey).asset_class == "forex"

    assert hop_manager.promote_subaccount(hotkey, NOW_MS, 450_000)[0]

    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_DIRECT
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 450_000
    assert info.account_size == 450_000
    assert info.asset_class == "all_markets"


def test_entering_the_track_without_a_size_is_rejected(hop_manager):
    """A subaccount that never had a size set (nothing to fall back on) is not promoted onto a pro
    account of unknown size; nor is one re-entering the track on the size of an earlier journey."""
    entity_manager, hotkey = _with_real_entity_manager(hop_manager)

    success, message = hop_manager.promote_subaccount(hotkey, NOW_MS)

    assert not success
    assert REQUIRED in message
    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.SUBACCOUNT_FUNDED
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size is None
    assert info.account_size == STANDARD_SIZE
    hop_manager._position_client.close_all_positions.assert_not_called()

    demoted_manager, demoted = _with_real_entity_manager(hop_manager, hotkey_factory=_add_demoted)
    assert demoted_manager.get_subaccount_info_for_synthetic(demoted).pro_account_size == GRANTED_SIZE
    assert not hop_manager.promote_subaccount(demoted, NOW_MS)[0]


def test_organic_promotion_keeps_the_recorded_size(hop_manager):
    """The end-of-week auto promotion and the pro funded promotion send no size either."""
    entity_manager, hotkey = _with_real_entity_manager(hop_manager)
    assert hop_manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]

    assert hop_manager.promote_hotkeys([hotkey], NOW_MS)
    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    assert entity_manager.get_subaccount_info_for_synthetic(hotkey).account_size == GRANTED_SIZE

    assert hop_manager.promote_hotkeys([hotkey], NOW_MS + 1000)
    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_FUNDED
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == GRANTED_SIZE
    assert info.account_size == GRANTED_SIZE


def test_an_unaffordable_promotion_fee_is_refused_before_anything_is_written(hop_manager):
    """The fee is the one predictable failure that lives inside the committing call, so it has to be
    checked before the write, not compensated after it."""
    entity_manager, hotkey = _with_real_entity_manager(
        hop_manager, bucket=MinerBucket.SUBACCOUNT_CHALLENGE)
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    info.reg_fee_theta = 1.0  # collateral-exempt subaccounts (0.0) never pay the promotion fee
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.5

    success, message = hop_manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)

    assert not success
    assert "Insufficient collateral" in message
    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.SUBACCOUNT_CHALLENGE
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_type == "standard"
    assert info.pro_account_size is None
    assert info.standard_account_size is None
    assert info.account_size == STANDARD_SIZE
    assert info.pro_fee_theta == 0.0
    assert info.pro_fee_theta_pending == 0.0
    # neither the live account nor the collateral reservation was touched
    entity_manager._miner_account_client.set_miner_account_size.assert_not_called()
    entity_manager._entity_collateral_client.offset_collateral_cache.assert_not_called()
    hop_manager._position_client.close_all_positions.assert_not_called()


# ==================== rollback of a failed hop ====================

@pytest.mark.parametrize("failure", ["rejected", "crashed"])
def test_a_failed_hop_onto_the_pro_account_rolls_the_sizing_back(hop_manager, failure):
    """The pro account is sized before the bucket moves, so a failed move must put the sizing back:
    PRO_CHALLENGE_FROM_STANDARD trades the pro size, and a subaccount left holding it while still in
    PRO_CHALLENGE_TRANSITION is supposed to be trading the standard account."""
    entity_manager, hotkey = _with_real_entity_manager(hop_manager)
    assert hop_manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]
    before = entity_manager.get_subaccount_info_for_synthetic(hotkey).model_copy()
    assert before.account_size == STANDARD_SIZE

    if failure == "rejected":
        with patch.object(hop_manager, "admin_set_bucket", return_value=(False, "bucket move failed")):
            success, message = hop_manager.promote_subaccount(hotkey, NOW_MS + 1000)
        assert not success
        assert "bucket move failed" in message
    else:
        # _switch_account runs before the bucket entry, so a crash there means no move happened
        with patch.object(hop_manager, "_switch_account", side_effect=RuntimeError("rpc down")):
            with pytest.raises(RuntimeError):
                hop_manager.promote_subaccount(hotkey, NOW_MS + 1000)

    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_size == STANDARD_SIZE
    assert info.standard_account_size == before.standard_account_size
    assert info.pro_account_size == GRANTED_SIZE
    assert info.account_type == before.account_type
    resized_to = [c.kwargs['account_size']
                  for c in entity_manager._miner_account_client.set_miner_account_size.call_args_list]
    assert resized_to[-1] == STANDARD_SIZE


def test_a_crash_after_the_bucket_moves_keeps_the_sizing(hop_manager):
    """The disk write and the entry-order sweep run after the bucket entry has landed. Rolling the
    sizing back there would leave a pro bucket trading the standard size."""
    entity_manager, hotkey = _with_real_entity_manager(hop_manager)
    assert hop_manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]

    with patch.object(hop_manager, "_save_to_disk", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError):
            hop_manager.promote_subaccount(hotkey, NOW_MS + 1000)

    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_type == "pro"
    assert info.account_size == GRANTED_SIZE
    assert info.pro_account_size == GRANTED_SIZE


def test_a_failed_first_hop_leaves_no_pro_marking(hop_manager):
    """A subaccount left marked pro would trade the pro size in a standard bucket, and the next
    size-less attempt would take the "recorded" branch instead of demanding a size."""
    entity_manager, hotkey = _with_real_entity_manager(
        hop_manager, bucket=MinerBucket.SUBACCOUNT_CHALLENGE)

    with patch.object(hop_manager, "admin_set_bucket", return_value=(False, "bucket move failed")):
        assert not hop_manager.promote_subaccount(hotkey, NOW_MS, 500_000)[0]

    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_type == "standard"
    assert info.pro_account_size is None
    assert info.standard_account_size is None
    assert info.account_size == STANDARD_SIZE
    # and the widened asset class goes back with the sizing
    assert info.asset_class == "forex"

    # behavioural backstop: the promotion did not happen, so a size is still required
    success, message = hop_manager.promote_subaccount(hotkey, NOW_MS)
    assert not success
    assert REQUIRED in message


# ═══════════════════════════════════════════════════════════════════════════════
# Section 14 — resting orders during PRO_CHALLENGE_TRANSITION
#
# The transition is not an account switch: the miner keeps trading the standard account and has to
# wind it down themselves before their pro account starts. They may no longer open a position or
# add to one, but they still need their brackets and their resting exits to close out with. So the
# orders that have to go are exactly the resting entries, and nothing else.
# ═══════════════════════════════════════════════════════════════════════════════

ORDER_NOW_MS = 1_748_000_000_000
ORDER_HOTKEY = "test_miner"


def _order(uuid, order_type, execution_type=ExecutionType.LIMIT, src=None, trade_pair=TradePair.BTCUSD):
    if src is None:
        src = {
            ExecutionType.LIMIT: OrderSource.LIMIT_UNFILLED,
            ExecutionType.BRACKET: OrderSource.BRACKET_UNFILLED,
            ExecutionType.STOP_LIMIT: OrderSource.STOP_LIMIT_UNFILLED,
        }[execution_type]
    return Order(
        trade_pair=trade_pair,
        order_uuid=uuid,
        processed_ms=ORDER_NOW_MS - 60_000,
        price=0.0,
        order_type=order_type,
        execution_type=execution_type,
        limit_price=50_000,
        stop_price=50_000 if execution_type == ExecutionType.STOP_LIMIT else None,
        leverage=0.1,
        src=src,
    )


def _position(position_type, trade_pair=TradePair.BTCUSD):
    return Position(
        miner_hotkey=ORDER_HOTKEY,
        position_uuid=f"pos_{trade_pair.trade_pair_id}",
        open_ms=ORDER_NOW_MS - 120_000,
        trade_pair=trade_pair,
        position_type=position_type,
        account_size=100_000.0,
    )


class ProTransitionRestingOrdersTest(unittest.TestCase):
    """The limit order manager side: which orders the transition takes, and when."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for path in _LIMIT_ORDER_CLIENT_PATHS:
            stack.enter_context(patch(path))
        stack.enter_context(patch.object(LimitOrderManager, "_read_limit_orders_from_disk"))

        self.manager = LimitOrderManager(running_unit_tests=True, serve=False)
        # Disk is not what these tests are about; record the writes instead of making them.
        self.written = []
        stack.enter_context(
            patch.object(LimitOrderManager, "_write_to_disk",
                         side_effect=lambda hk, order: self.written.append((order.order_uuid, order.src)))
        )
        self._set_bucket(MinerBucket.PRO_CHALLENGE_TRANSITION)
        self._set_open_positions([])

    # ---- fixtures -------------------------------------------------------------------

    def _set_bucket(self, bucket):
        self.manager._miner_account_client.get_account.return_value = MagicMock(miner_bucket=bucket)

    def _set_open_positions(self, positions):
        by_trade_pair = {p.trade_pair.trade_pair_id: p for p in positions}
        self.manager._position_client.get_positions_for_one_hotkey.return_value = positions
        self.manager._position_client.get_open_position_for_trade_pair.side_effect = (
            lambda hotkey, trade_pair_id: by_trade_pair.get(trade_pair_id)
        )

    def _rest(self, *orders):
        for order in orders:
            self.manager._limit_orders.setdefault(order.trade_pair, {}).setdefault(
                ORDER_HOTKEY, []).append(order)

    def _resting_uuids(self):
        return {o.order_uuid
                for hotkey_dict in self.manager._limit_orders.values()
                for o in hotkey_dict.get(ORDER_HOTKEY, [])}

    def _fill(self, order):
        price_source = PriceSource(source="test", start_ms=ORDER_NOW_MS, open=50_000, close=50_000,
                                   high=50_000, low=50_000, bid=49_999, ask=50_001)
        return self.manager._fill_limit_order_with_price_source(ORDER_HOTKEY, order, price_source, 50_000)

    def _expect_fill(self, order, position):
        filled = _order(order.order_uuid, order.order_type)
        filled.price = 50_000
        self.manager._market_order_client.execute_order.return_value = OrderExecution(filled, position)

    # ---- cancel_entry_orders --------------------------------------------------------

    def test_sweep_takes_entries_and_leaves_everything_else(self):
        opener = _order("opener", OrderType.LONG)
        adder = _order("adder", OrderType.LONG, trade_pair=TradePair.ETHUSD)
        reducer = _order("reducer", OrderType.SHORT, trade_pair=TradePair.ETHUSD)
        bracket = _order("bracket", OrderType.LONG, ExecutionType.BRACKET, trade_pair=TradePair.ETHUSD)
        stop_entry = _order("stop_entry", OrderType.SHORT, ExecutionType.STOP_LIMIT, trade_pair=TradePair.SOLUSD)
        self._rest(opener, adder, reducer, bracket, stop_entry)
        self._set_open_positions([_position(OrderType.LONG, TradePair.ETHUSD)])

        result = self.manager.cancel_entry_orders(ORDER_HOTKEY, ORDER_NOW_MS,
                                                  OrderSource.PRO_TRANSITION_CANCELLED)

        self.assertEqual(result["num_cancelled"], 3)
        # The brackets and the resting exit are what the miner winds the account down with.
        self.assertEqual(self._resting_uuids(), {"reducer", "bracket"})
        self.assertEqual(
            dict(self.written),
            {"opener": OrderSource.PRO_TRANSITION_CANCELLED,
             "adder": OrderSource.PRO_TRANSITION_CANCELLED,
             "stop_entry": OrderSource.PRO_TRANSITION_CANCELLED},
        )

    def test_sweep_derives_a_cancel_src_when_none_is_given(self):
        self._rest(_order("opener", OrderType.LONG),
                   _order("stop_entry", OrderType.LONG, ExecutionType.STOP_LIMIT, trade_pair=TradePair.ETHUSD))

        self.manager.cancel_entry_orders(ORDER_HOTKEY, ORDER_NOW_MS)

        self.assertEqual(
            dict(self.written),
            {"opener": OrderSource.LIMIT_CANCELLED, "stop_entry": OrderSource.STOP_LIMIT_CANCELLED},
        )

    def test_sweep_ignores_orders_that_are_no_longer_resting(self):
        self._rest(_order("already_filled", OrderType.LONG, src=OrderSource.LIMIT_FILLED),
                   _order("already_cancelled", OrderType.LONG, src=OrderSource.LIMIT_CANCELLED,
                          trade_pair=TradePair.ETHUSD))

        result = self.manager.cancel_entry_orders(ORDER_HOTKEY, ORDER_NOW_MS,
                                                  OrderSource.PRO_TRANSITION_CANCELLED)

        self.assertEqual(result["num_cancelled"], 0)
        self.assertEqual(self.written, [])

    # ---- the fill path --------------------------------------------------------------

    def test_a_triggered_entry_order_is_cancelled_with_a_reason(self):
        order = _order("opener", OrderType.LONG)
        self._rest(order)

        error_msg = self._fill(order)

        self.manager._market_order_client.execute_order.assert_not_called()
        self.assertIn("transitioning to a Pro Account", error_msg)
        self.assertEqual(self.written, [("opener", OrderSource.PRO_TRANSITION_CANCELLED)])
        self.assertEqual(self._resting_uuids(), set())

    def test_an_order_that_became_an_entry_order_is_caught_at_fill_time(self):
        """A resting exit that outlives its position is an opener by the time it triggers, so the
        sweep at the start of the transition cannot have caught it."""
        order = _order("was_an_exit", OrderType.SHORT)
        self._rest(order)
        self._set_open_positions([])  # the LONG it was closing is already gone

        self._fill(order)

        self.manager._market_order_client.execute_order.assert_not_called()
        self.assertEqual(self.written, [("was_an_exit", OrderSource.PRO_TRANSITION_CANCELLED)])

    def test_reducing_orders_and_brackets_still_fill_during_the_transition(self):
        for uuid, order_type, execution_type in (("reducer", OrderType.SHORT, ExecutionType.LIMIT),
                                                 ("bracket", OrderType.LONG, ExecutionType.BRACKET)):
            with self.subTest(order=uuid):
                self.manager._limit_orders.clear()
                self.manager._market_order_client.execute_order.reset_mock()
                order = _order(uuid, order_type, execution_type)
                position = _position(OrderType.LONG)
                self._rest(order)
                self._set_open_positions([position])
                self._expect_fill(order, position)

                self.assertIsNone(self._fill(order))
                self.manager._market_order_client.execute_order.assert_called_once()

    def test_an_entry_order_fills_normally_outside_the_transition(self):
        self._set_bucket(MinerBucket.SUBACCOUNT_FUNDED)
        order = _order("opener", OrderType.LONG)
        position = _position(OrderType.LONG)
        self._rest(order)
        self._expect_fill(order, position)

        self.assertIsNone(self._fill(order))
        self.manager._market_order_client.execute_order.assert_called_once()

    # ---- stop-limit conversion ------------------------------------------------------

    def test_a_rejected_stop_limit_conversion_records_the_parent_as_cancelled(self):
        parent = _order("stop_entry", OrderType.LONG, ExecutionType.STOP_LIMIT)
        self._rest(parent)

        with patch.object(self.manager, "process_limit_order",
                          side_effect=SignalException("transitioning to a Pro Account")):
            self.manager._convert_stop_limit_to_limit_order(ORDER_HOTKEY, parent, ORDER_NOW_MS)

        # Without the cancellation record the parent leaves no trace: _close_limit_order removes the
        # unfilled file and only persists orders whose src is a cancellation.
        self.assertEqual(self.written, [("stop_entry", OrderSource.STOP_LIMIT_CANCELLED)])
        self.assertEqual(parent.src, OrderSource.STOP_LIMIT_CANCELLED)

    def test_a_successful_stop_limit_conversion_still_closes_the_parent_as_filled(self):
        parent = _order("stop_entry", OrderType.LONG, ExecutionType.STOP_LIMIT)
        self._rest(parent)

        with patch.object(self.manager, "process_limit_order") as process:
            self.manager._convert_stop_limit_to_limit_order(ORDER_HOTKEY, parent, ORDER_NOW_MS)

        process.assert_called_once()
        self.assertEqual(parent.src, OrderSource.STOP_LIMIT_FILLED)
        self.assertEqual(self.written, [])  # filled orders are not persisted


class AdminSetBucketCancelsEntryOrdersTest(unittest.TestCase):
    """The challenge period side: the sweep is wired into the move into the transition."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for path in _CP_CLIENT_PATHS:
            stack.enter_context(patch(path))
        self.manager = ChallengePeriodManager(is_backtesting=True)
        self.manager._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
        stack.enter_context(patch.object(self.manager, "_sync_buckets_to_accounts"))
        stack.enter_context(patch.object(self.manager, "_save_to_disk"))
        self.manager.set_miner_bucket(ORDER_HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, ORDER_NOW_MS)

    def test_entering_the_transition_sweeps_entry_orders_only(self):
        success, message = self.manager.admin_set_bucket(
            ORDER_HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, ORDER_NOW_MS)

        self.assertTrue(success, message)
        self.manager._limit_order_client.cancel_entry_orders.assert_called_once_with(
            ORDER_HOTKEY, ORDER_NOW_MS, OrderSource.PRO_TRANSITION_CANCELLED
        )
        # The transition is not an account switch, so the brackets and the resting exits survive.
        self.manager._limit_order_client.cancel_limit_order.assert_not_called()
        self.manager._position_client.close_all_positions.assert_not_called()

    def test_an_account_switch_still_cancels_everything(self):
        self.manager.set_miner_bucket(ORDER_HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, ORDER_NOW_MS)

        success, message = self.manager.admin_set_bucket(
            ORDER_HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, ORDER_NOW_MS)

        self.assertTrue(success, message)
        self.manager._limit_order_client.cancel_limit_order.assert_called_once_with(
            ORDER_HOTKEY, None, "ALL", ORDER_NOW_MS)
        self.manager._limit_order_client.cancel_entry_orders.assert_not_called()

    def test_a_standard_bucket_move_touches_no_orders(self):
        success, message = self.manager.admin_set_bucket(
            ORDER_HOTKEY, MinerBucket.SUBACCOUNT_ALPHA, ORDER_NOW_MS)

        self.assertTrue(success, message)
        self.manager._limit_order_client.cancel_entry_orders.assert_not_called()
        self.manager._limit_order_client.cancel_limit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
