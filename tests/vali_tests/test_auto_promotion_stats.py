"""
Auto-promotion candidates are standard subaccounts in SUBACCOUNT_FUNDED (created through SUBACCOUNT_CHALLENGE,
not HL-linked). From the first refresh that sees one, its drawdown is tracked live in DrawdownStats, ratcheted
as the pro challenge ratchets its own; the time before is replayed from the perf ledger
(LedgerUtils.eod_hwm_drawdown). Calmar, daily consistency and trading days are measured as in the pro challenge
into AutoPromotionStats. The pro track and every other account are unaffected.

What each refresh decided for a candidate is what its dashboard shows (get_auto_promotion); no other account
has that section.
"""
import json
from unittest.mock import patch

import pytest

from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager is a pytest fixture
    ACCOUNT_SIZE,
    CP_MS,
    DAILY_MS,
    HOTKEY,
    LEDGER_START_MS,
    MIDNIGHT_MS,
    NOW_MS,
    _even_ledger,
    manager,
)
from vali_objects.challenge_period.challengeperiod_manager import (
    AutoPromotion,
    ChallengePeriodManager,
    DrawdownStats,
    ProStats,
)
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.misc import TradePairReturnStatus
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.penalty.penalty_ledger import PenaltyInputType, PenaltyLedgerManager
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

HL_ADDRESS = "0x" + "d" * 40
DAYS = 60
REALIZED_USD = 7_000.0  # 7% on the account
PRO_BUCKETS = (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
               MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED)


def _account(hotkey: str = HOTKEY, hl_address: str | None = None,
             asset_class: MinerAssetClass | None = None) -> MinerAccount:
    record = CollateralRecord(account_size=ACCOUNT_SIZE, account_size_theta=ACCOUNT_SIZE / ValiConfig.COST_PER_THETA,
                              update_time_ms=NOW_MS - 100 * DAILY_MS, is_first_record=True)
    return MinerAccount(miner_hotkey=hotkey, collateral_records=[record], hl_address=hl_address,
                        asset_class=asset_class)


def _ledger(low: float = 1.0, close: float = 1.0, days: int = DAYS) -> PerfLedger:
    """DAYS of small gains and REALIZED_USD realized, with every checkpoint at `close` and dipping to `low`."""
    ledger = _even_ledger(days, realized_pnl_usd=REALIZED_USD)
    for cp in ledger.cps:
        cp.equity_ret = close
        cp.min_equity_ret = low
    return ledger


def _seed(manager, hotkey: str = HOTKEY, bucket: MinerBucket = MinerBucket.SUBACCOUNT_FUNDED,
          current_equity: float = 1.0, eod_hwm: float = 1.0) -> None:
    manager.set_miner_bucket(hotkey, bucket, NOW_MS - DAYS * DAILY_MS)
    manager.miner_states[hotkey].drawdown = DrawdownStats(current_equity=current_equity, current_balance=current_equity,
                                                          eod_hwm=eod_hwm, last_eod_equity=1.0,
                                                          last_eod_checked_ms=MIDNIGHT_MS)


def _refresh(manager, ledgers: dict, accounts: dict | None = None, now_ms: int = NOW_MS) -> None:
    accounts = accounts if accounts is not None else {hk: _account(hk) for hk in ledgers}
    manager._refresh_auto_promotion_stats(list(ledgers), ledgers, accounts, now_ms)


def _stats(manager, hotkey: str = HOTKEY):
    return manager._auto_promotion_stats.get(hotkey)


@pytest.fixture(autouse=True)
def _standard_subaccount(manager):
    """The entity record of a subaccount created through SUBACCOUNT_CHALLENGE."""
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {"initial_bucket": None}


# ── LedgerUtils.eod_hwm_drawdown ──────────────────────────────────────────────

def _cp(end_ms: int, low: float | None, close: float) -> PerfCheckpoint:
    return PerfCheckpoint(last_update_ms=end_ms, prev_portfolio_ret=1.0, accum_ms=CP_MS,
                          equity_ret=close, min_equity_ret=low)


DAY0 = MIDNIGHT_MS - 10 * DAILY_MS


def test_an_intraday_peak_never_raises_the_mark():
    """The account ran to 1.05 before noon and fell back to 0.98: 2% under the 1.0 mark, not 6.7%."""
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=0.98, close=1.0)])
    ledger.cps[0].mpv = 1.05
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(0.98)


def test_a_midday_close_never_becomes_the_mark():
    """Closing noon at 1.10 and midnight at 1.00 leaves no drawdown against the 1.0 mark."""
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=1.0, close=1.10),
                             _cp(DAY0 + DAILY_MS, low=1.0, close=1.0),
                             _cp(DAY0 + DAILY_MS + CP_MS, low=1.0, close=1.0)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(1.0)


def test_the_mark_moves_at_midnight_and_governs_the_next_day():
    ledger = PerfLedger(cps=[
        _cp(DAY0 + CP_MS, low=1.00, close=1.05),
        _cp(DAY0 + DAILY_MS, low=1.04, close=1.10),           # closes day 0 at 1.10, still on the 1.0 mark
        _cp(DAY0 + DAILY_MS + CP_MS, low=1.045, close=1.08),  # day 1 runs on the 1.10 mark: 5% under
    ])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(1.045 / 1.10)


def test_a_lower_close_never_lowers_the_mark():
    ledger = PerfLedger(cps=[
        _cp(DAY0 + DAILY_MS, low=1.0, close=1.10),
        _cp(DAY0 + 2 * DAILY_MS, low=1.0, close=1.02),
        _cp(DAY0 + 2 * DAILY_MS + CP_MS, low=1.02, close=1.02),
    ])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(1.0 / 1.10)


def test_the_mark_is_floored_at_the_starting_equity():
    ledger = PerfLedger(cps=[_cp(DAY0 + DAILY_MS, low=0.97, close=0.98),
                             _cp(DAY0 + DAILY_MS + CP_MS, low=0.96, close=0.97)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(0.96)


def test_a_checkpoint_that_tracked_equity_without_its_low_makes_the_replay_unknown():
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=None, close=1.0), _cp(DAY0 + DAILY_MS, low=0.99, close=1.0)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) is None


def test_a_checkpoint_with_a_position_open_but_no_equity_tracked_makes_the_replay_unknown():
    """Equity moved (a position was open) but was never recorded, e.g. built while the account had no
    account size: not a checkpoint without a drawdown."""
    untracked = _cp(DAY0 + CP_MS, low=None, close=0.0)
    untracked.open_ms = CP_MS
    ledger = PerfLedger(cps=[untracked, _cp(DAY0 + DAILY_MS, low=0.99, close=1.0)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) is None


def test_a_checkpoint_that_never_tracked_equity_is_skipped():
    """Before the first position opens the ledger has no equity sample, and nothing to replay."""
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=None, close=0.0), _cp(DAY0 + DAILY_MS, low=None, close=0.0),
                             _cp(DAY0 + DAILY_MS + CP_MS, low=0.97, close=0.99)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(0.97)


def test_the_replay_stops_at_the_boundary():
    """Only checkpoints whose window starts before before_ms count: the one straddling it does, the
    next does not, and nor does an unknown low after it."""
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=0.99, close=1.0),
                             _cp(DAY0 + DAILY_MS, low=0.97, close=1.0),
                             _cp(DAY0 + DAILY_MS + CP_MS, low=0.90, close=1.0),
                             _cp(DAY0 + 2 * DAILY_MS, low=None, close=1.0)])
    assert LedgerUtils.eod_hwm_drawdown(ledger, before_ms=DAY0 + DAILY_MS - 1) == pytest.approx(0.97)
    assert LedgerUtils.eod_hwm_drawdown(ledger, before_ms=DAY0 + CP_MS) == pytest.approx(0.99)
    assert LedgerUtils.eod_hwm_drawdown(ledger, before_ms=DAY0 + DAILY_MS + CP_MS) == pytest.approx(0.90)
    assert LedgerUtils.eod_hwm_drawdown(ledger) is None


def test_it_replays_the_live_ratchet_over_a_ledger_built_minute_by_minute():
    """Built through update_pl, the replay equals the worst sampled equity over the mark in force when
    it was sampled: the closing equity of every UTC day already over, floored at 1.0."""
    ledger = PerfLedger()
    samples = []
    for minute in range(1, 4 * 24 * 60, 11):
        t_ms = DAY0 + minute * 60_000
        # Drifts up across the days so the mark moves, with dips of up to 4%
        unrealized = (minute / (4 * 24 * 60)) * 12_000 - ((minute * 53) % 400) * 10.0
        ledger.update_pl(1.0 + unrealized / ACCOUNT_SIZE, t_ms, "hk", TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                         0.0, unrealized, account_size=ACCOUNT_SIZE, cumulative_fees_usd=0.0)
        samples.append((t_ms, (ACCOUNT_SIZE + unrealized) / ACCOUNT_SIZE))

    worst = 1.0
    for t_ms, equity in samples:
        day_start = (t_ms // DAILY_MS) * DAILY_MS
        day_closes = {}
        for s_ms, e in samples:
            if s_ms < day_start:
                day_closes[s_ms // DAILY_MS] = e  # ends as the last sample of each finished day
        worst = min(worst, equity / max([1.0, *day_closes.values()]))

    assert worst < 0.99, "fixture should produce a drawdown"
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(worst)


# ── The live ratchet, in DrawdownStats ────────────────────────────────────────

def test_tracking_starts_at_the_first_refresh_and_never_recovers(manager):
    _seed(manager, current_equity=0.97)
    _refresh(manager, {HOTKEY: _ledger()})
    drawdown = manager.miner_states[HOTKEY].drawdown
    assert drawdown.max_drawdown == pytest.approx(0.97)
    assert drawdown.max_drawdown_since_ms == NOW_MS

    drawdown.current_equity = 1.02
    _refresh(manager, {HOTKEY: _ledger()}, now_ms=NOW_MS + 60_000)
    assert drawdown.max_drawdown == pytest.approx(0.97)
    assert drawdown.max_drawdown_since_ms == NOW_MS


def test_the_ratchet_reads_live_equity_against_the_eod_high_water_mark(manager):
    _seed(manager, current_equity=1.04, eod_hwm=1.08)
    _refresh(manager, {HOTKEY: _ledger(close=1.08)})
    assert manager.miner_states[HOTKEY].drawdown.max_drawdown == pytest.approx(1.04 / 1.08)


def test_the_ratchet_is_saved_with_the_drawdown_stats():
    stats = DrawdownStats(max_drawdown=0.95, max_drawdown_since_ms=NOW_MS)
    restored = DrawdownStats.from_dict(stats.to_dict())
    assert (restored.max_drawdown, restored.max_drawdown_since_ms) == (0.95, NOW_MS)
    assert DrawdownStats.from_dict({"current_equity": 1.0}).max_drawdown_since_ms is None


# ── Stats, measured as in the pro challenge ──────────────────────────────────

def test_a_candidate_gets_stats_measured_as_in_pro(manager):
    """7% realized over a 4% replayed drawdown: calmar 1.75."""
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})

    stats = _stats(manager)
    assert stats.max_drawdown == pytest.approx(0.96)
    assert stats.calmar == pytest.approx(1.75)
    assert stats.trading_days == DAYS
    assert stats.daily_consistency == pytest.approx(1 / DAYS)


def test_the_ledger_only_replays_the_time_before_live_tracking(manager):
    """A ledger low after tracking began is the live ratchet's to see, not the replay's."""
    _seed(manager)
    manager.miner_states[HOTKEY].drawdown.max_drawdown_since_ms = LEDGER_START_MS + 30 * DAILY_MS
    ledger = _ledger(low=0.99)
    ledger.cps[20].min_equity_ret = 0.97   # day 10: before tracking
    ledger.cps[80].min_equity_ret = 0.90   # day 40: after it

    _refresh(manager, {HOTKEY: ledger})

    assert _stats(manager).max_drawdown == pytest.approx(0.97)


def test_the_live_ratchet_and_the_replay_both_count_and_only_the_ratchet_is_kept(manager):
    _seed(manager, current_equity=0.95)
    _refresh(manager, {HOTKEY: _ledger(low=0.98)})
    assert _stats(manager).max_drawdown == pytest.approx(0.95)

    # Recovered live, but the replay now shows a deeper dip
    manager.miner_states[HOTKEY].drawdown.current_equity = 1.02
    _refresh(manager, {HOTKEY: _ledger(low=0.93)})
    assert _stats(manager).max_drawdown == pytest.approx(0.93)

    # A corrected ledger is read afresh; the live ratchet still holds its own worst
    _refresh(manager, {HOTKEY: _ledger(low=0.99)})
    assert _stats(manager).max_drawdown == pytest.approx(0.95)
    assert manager.miner_states[HOTKEY].drawdown.max_drawdown == pytest.approx(0.95)


@pytest.mark.parametrize("record", [
    {"initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value},             # Instant Funded
    {"initial_bucket": MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value},   # Instant Funded
    None,                                                                # no entity record
])
def test_only_subaccounts_created_through_the_challenge_are_candidates(manager, record):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = record
    _seed(manager, current_equity=0.97)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert _stats(manager) is None
    assert manager.miner_states[HOTKEY].drawdown.max_drawdown_since_ms is None
    assert HOTKEY not in manager._auto_promotion_candidates


@pytest.mark.parametrize("account_kwargs", [{"hl_address": HL_ADDRESS}, {"asset_class": MinerAssetClass.HL_ALL}])
def test_hl_linked_subaccounts_are_not_candidates(manager, account_kwargs):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)}, {HOTKEY: _account(**account_kwargs)})
    assert _stats(manager) is None


@pytest.mark.parametrize("bucket", [MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.MAINCOMP, *PRO_BUCKETS])
def test_no_other_bucket_is_tracked(manager, bucket):
    _seed(manager, bucket=bucket, current_equity=0.97)
    _refresh(manager, {HOTKEY: _ledger(low=0.50)})
    assert _stats(manager) is None
    assert manager.miner_states[HOTKEY].drawdown.max_drawdown == 1.0
    assert manager.miner_states[HOTKEY].pro_stats == ProStats()


@pytest.mark.parametrize("bucket", PRO_BUCKETS)
def test_the_pro_track_keeps_its_own_ratchet(manager, bucket):
    _seed(manager, bucket=bucket, current_equity=0.97)
    manager._refresh_pro_stats([HOTKEY], {HOTKEY: _ledger(low=0.50)}, {HOTKEY: _account()})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.97)


def test_the_creation_record_is_looked_up_once(manager):
    _seed(manager)
    for _ in range(3):
        _refresh(manager, {HOTKEY: _ledger()})
    manager._entity_client.get_subaccount_info_for_synthetic.assert_called_once_with(HOTKEY)


def test_a_failed_lookup_skips_the_account_until_the_next_refresh(manager):
    _seed(manager)
    manager._entity_client.get_subaccount_info_for_synthetic.side_effect = RuntimeError("entity server down")
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert _stats(manager) is None

    manager._entity_client.get_subaccount_info_for_synthetic.side_effect = None
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert _stats(manager).max_drawdown == pytest.approx(0.96)


def test_a_candidate_with_no_ledger_yet_is_tracked_but_has_no_stats(manager):
    _seed(manager, current_equity=0.98)
    _refresh(manager, {HOTKEY: None})
    assert _stats(manager) is None
    assert manager.miner_states[HOTKEY].drawdown.max_drawdown == pytest.approx(0.98)


def test_a_ledger_missing_lows_gets_no_stats_and_is_logged_once(manager):
    """Until the one-time regeneration rebuilds it, a ledger from before lows were recorded cannot be
    replayed, so the account is not evaluated."""
    _seed(manager)
    ledger = _ledger(low=0.99)
    ledger.cps[0].min_equity_ret = None

    with patch("vali_objects.challenge_period.challengeperiod_manager.logger") as log:
        _refresh(manager, {HOTKEY: ledger})
        _refresh(manager, {HOTKEY: ledger})
    assert _stats(manager) is None
    assert log.warning.call_count == 1

    _refresh(manager, {HOTKEY: _ledger(low=0.94)})
    assert _stats(manager).max_drawdown == pytest.approx(0.94)


# ── What the dashboard is told ────────────────────────────────────────────────

def test_a_candidate_shows_what_the_refresh_decided(manager):
    """A subaccount from before the promotion rules, 7% realized over a 4% drawdown, flat, never traded as funded."""
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})

    section = manager.get_auto_promotion(HOTKEY)
    assert section == {
        "on_promotion_rules": False,
        "fast_track_target": ValiConfig.AUTO_PROMOTION_FAST_TRACK_EQUITY_RETURN,
        "first_funded_order_ms": None,
        "peak_leverage_use": 0.0,
        "fast_track_held_since_ms": None,
        "fast_track_hold_ms": ChallengePeriodManager.AUTO_PROMOTION_FAST_TRACK_HOLD_MS,
        "trading_days": DAYS,
        "calmar": pytest.approx(1.75),
        "daily_consistency": pytest.approx(1 / DAYS),
        "as_of_ms": NOW_MS,
    }
    assert {key: type(value) for key, value in section.items()} == {
        "on_promotion_rules": bool, "fast_track_target": float, "first_funded_order_ms": type(None),
        "peak_leverage_use": float, "fast_track_held_since_ms": type(None), "fast_track_hold_ms": int,
        "trading_days": int, "calmar": float, "daily_consistency": float, "as_of_ms": int,
    }
    stats = _stats(manager)
    assert (section["trading_days"], section["calmar"], section["daily_consistency"]) == (
        stats.trading_days, stats.calmar, stats.daily_consistency)


def test_every_refresh_replaces_what_is_shown(manager):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    _refresh(manager, {HOTKEY: _ledger(low=0.93)}, now_ms=NOW_MS + 30_000)

    section = manager.get_auto_promotion(HOTKEY)
    assert section["as_of_ms"] == NOW_MS + 30_000
    assert section["calmar"] == pytest.approx(1.0)


def test_a_candidate_with_no_ledger_yet_shows_its_target_and_no_stats(manager):
    _seed(manager)
    _refresh(manager, {HOTKEY: None})

    section = manager.get_auto_promotion(HOTKEY)
    assert (section["trading_days"], section["calmar"], section["daily_consistency"]) == (None, None, None)
    assert section["fast_track_target"] == ValiConfig.AUTO_PROMOTION_FAST_TRACK_EQUITY_RETURN
    assert section["on_promotion_rules"] is False and section["as_of_ms"] == NOW_MS


def test_a_candidate_whose_ledger_cannot_be_replayed_shows_no_stats(manager):
    _seed(manager)
    ledger = _ledger(low=0.99)
    ledger.cps[0].min_equity_ret = None
    _refresh(manager, {HOTKEY: ledger})

    section = manager.get_auto_promotion(HOTKEY)
    assert (section["trading_days"], section["calmar"], section["daily_consistency"]) == (None, None, None)


@pytest.mark.parametrize("record", [
    {"initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value},             # Instant Funded
    {"initial_bucket": MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value},   # Instant Funded
    None,                                                                # no entity record
])
def test_an_account_not_created_through_the_challenge_shows_nothing(manager, record):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = record
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.get_auto_promotion(HOTKEY) is None


@pytest.mark.parametrize("account_kwargs", [{"hl_address": HL_ADDRESS}, {"asset_class": MinerAssetClass.HL_ALL}])
def test_an_hl_linked_subaccount_shows_nothing(manager, account_kwargs):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)}, {HOTKEY: _account(**account_kwargs)})
    assert manager.get_auto_promotion(HOTKEY) is None


@pytest.mark.parametrize("bucket", [MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.MAINCOMP, *PRO_BUCKETS])
def test_no_other_bucket_shows_anything(manager, bucket):
    _seed(manager, bucket=bucket)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.get_auto_promotion(HOTKEY) is None


def test_an_unknown_account_shows_nothing(manager):
    assert manager.get_auto_promotion("nobody") is None


def test_nothing_is_shown_before_the_first_refresh(manager):
    """A restarted validator has decided nothing yet, and cannot tell a candidate from any other funded account."""
    _seed(manager)
    assert manager.get_auto_promotion(HOTKEY) is None


def test_an_account_that_left_the_bucket_shows_nothing_at_once(manager):
    """Eliminated or promoted later in the same refresh: the bucket is read live, not from the last pass."""
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.get_auto_promotion(HOTKEY) is not None

    manager.miner_states[HOTKEY].add_bucket_entry(MinerBucket.ELIMINATED, NOW_MS + 1)
    assert manager.get_auto_promotion(HOTKEY) is None

    _refresh(manager, {HOTKEY: _ledger(low=0.96)}, now_ms=NOW_MS + 30_000)
    assert manager._auto_promotion_snapshot == {}


def test_an_account_promoted_automatically_before_shows_nothing(manager):
    """It is never evaluated again, so there is no progress to show."""
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    manager.miner_states[HOTKEY].auto_promotion = AutoPromotion(path=AutoPromotion.FAST_TRACK, multiplier=3)
    assert manager.get_auto_promotion(HOTKEY) is None


# ── A state reset or replaced between two refreshes ──────────────────────────

ON_THE_RULES_ONLY = {"on_promotion_rules": False, "as_of_ms": NOW_MS}


def _eliminate_then_revert(manager) -> None:
    """A breach eliminates the account after the pass and an operator reverts it (as POST /admin/reset-hotkey
    does too), which restarts its drawdown."""
    manager.eliminate_hotkeys({HOTKEY: EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN}, NOW_MS + 1)
    assert manager.get_auto_promotion(HOTKEY) is None
    assert manager.revert_elimination(HOTKEY)


def _admin_moves_it_back_into_funded(manager) -> None:
    """An operator moves the account back to the challenge and on to funded again (move_to_funded): each
    move switches to a new account."""
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, NOW_MS + 1)[0]
    assert manager.get_auto_promotion(HOTKEY) is None
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS + 2)[0]


def _sync_replaces_the_states(manager) -> None:
    """The nightly sync swaps in another validator's states, here the very same ones."""
    manager.sync_challenge_period_data(manager.to_checkpoint_dict())


def _its_drawdown_is_reset(manager) -> None:
    """As for a standard miner whose traded size changed."""
    manager._reset_drawdown_stats_cache(HOTKEY)


def _it_leaves_the_bucket_and_comes_back(manager) -> None:
    state = manager.miner_states[HOTKEY]
    state.add_bucket_entry(MinerBucket.ELIMINATED, NOW_MS + 1)
    state.add_bucket_entry(MinerBucket.SUBACCOUNT_FUNDED, NOW_MS + 2)


@pytest.mark.parametrize("change", [_eliminate_then_revert, _admin_moves_it_back_into_funded,
                                    _sync_replaces_the_states, _its_drawdown_is_reset,
                                    _it_leaves_the_bucket_and_comes_back])
def test_a_state_reset_or_replaced_since_the_refresh_shows_only_whether_it_is_on_the_rules(manager, change):
    """Funded again with no record, so the live checks pass, but the hold, the target and the stats were read
    from the state before: none of them is shown until a refresh has read the one in its place. What the
    refresh keeps for the promotion itself is left exactly as it was."""
    _seed(manager, current_equity=1.12)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    shown = manager.get_auto_promotion(HOTKEY)
    assert shown["fast_track_held_since_ms"] == NOW_MS and shown["trading_days"] == DAYS
    kept = (dict(manager._auto_promotion_stats), dict(manager._fast_track_since_ms),
            set(manager._auto_promotion_candidates), dict(manager._promotion_rules))

    change(manager)

    state = manager.miner_states[HOTKEY]
    assert state.current_bucket == MinerBucket.SUBACCOUNT_FUNDED and state.auto_promotion is None
    assert manager.get_auto_promotion(HOTKEY) == ON_THE_RULES_ONLY
    assert (manager._auto_promotion_stats, manager._fast_track_since_ms, manager._auto_promotion_candidates,
            manager._promotion_rules) == kept

    state.drawdown.current_equity = 1.0  # the new account's, as the drawdown cache fills it in
    _refresh(manager, {HOTKEY: None}, {HOTKEY: _account()}, now_ms=NOW_MS + 30_000)
    section = manager.get_auto_promotion(HOTKEY)
    assert (section["fast_track_held_since_ms"], section["trading_days"], section["as_of_ms"]) == (
        None, None, NOW_MS + 30_000)
    assert section["fast_track_target"] == shown["fast_track_target"]


def test_a_reset_landing_while_the_refresh_runs_is_not_shown_as_what_it_decided(manager):
    """The pass read the drawdown that the reset then replaced: what it publishes is of the state before."""
    _seed(manager, current_equity=1.12)
    daily_return_log = LedgerUtils.daily_return_log

    def reset_then_measure(ledger):
        manager._reset_drawdown_stats_cache(HOTKEY)
        return daily_return_log(ledger)

    with patch("vali_objects.challenge_period.challengeperiod_manager.LedgerUtils.daily_return_log",
               side_effect=reset_then_measure):
        _refresh(manager, {HOTKEY: _ledger(low=0.96)})

    assert manager._auto_promotion_stats[HOTKEY].trading_days == DAYS
    assert manager.get_auto_promotion(HOTKEY) == ON_THE_RULES_ONLY


def test_an_elimination_landing_while_the_dashboard_reads_is_not_shown_as_a_candidate(manager):
    """The bucket was SUBACCOUNT_FUNDED when the read began and ELIMINATED before it ended: the candidates
    of the last pass still list the account, and it must not be shown as one."""
    _seed(manager, current_equity=1.12)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.get_auto_promotion(HOTKEY) is not None

    class EliminatedOnRead(dict):
        def get(self, key, default=None):
            manager.miner_states[key].add_bucket_entry(MinerBucket.ELIMINATED, NOW_MS + 1)
            return super().get(key, default)

    manager._auto_promotion_snapshot = EliminatedOnRead(manager._auto_promotion_snapshot)

    assert manager.get_auto_promotion(HOTKEY) is None


def test_an_edit_of_the_drawdown_in_place_keeps_what_the_refresh_decided(manager):
    """Neither the refresh's own drawdown cache nor an operator's edit replaces anything: both write onto the
    drawdown the pass read."""
    _seed(manager, current_equity=1.12)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    shown = manager.get_auto_promotion(HOTKEY)

    manager.miner_states[HOTKEY].drawdown.current_equity = 1.13
    success, _, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.01}, now_ms=NOW_MS)

    assert success and manager.get_auto_promotion(HOTKEY) == shown


def test_what_is_shown_is_a_copy(manager):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    manager.get_auto_promotion(HOTKEY)["fast_track_target"] = 0.0
    assert manager.get_auto_promotion(HOTKEY)["fast_track_target"] == ValiConfig.AUTO_PROMOTION_FAST_TRACK_EQUITY_RETURN


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_number_that_is_not_finite_is_shown_as_none(manager, bad):
    """json.dumps would write it as a bare NaN or Infinity, which no JSON parser accepts."""
    _seed(manager)
    account = _account()
    account.peak_leverage_use = bad
    with patch("vali_objects.challenge_period.challengeperiod_manager.Metrics.all_time_calmar", return_value=bad):
        _refresh(manager, {HOTKEY: _ledger(low=0.96)}, {HOTKEY: account})

    section = manager.get_auto_promotion(HOTKEY)
    assert section["calmar"] is None and section["peak_leverage_use"] is None
    assert section["trading_days"] == DAYS and section["fast_track_target"] == 0.10
    json.dumps(section, allow_nan=False)


# ── Nothing else reads the candidates' stats ─────────────────────────────────

def test_no_penalty_reads_the_ratcheted_drawdown_of_a_funded_account():
    for name, config in PenaltyLedgerManager.PENALTIES_CONFIG.items():
        if config.input_type == PenaltyInputType.LEDGER_MAX_DRAWDOWN:
            assert config.buckets is not None and MinerBucket.SUBACCOUNT_FUNDED not in config.buckets, name


def test_a_candidate_never_latches_a_soft_breach_or_shows_pro_stats(manager):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.50)})  # calmar far below every pro line
    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].pro_stats == ProStats()
    assert manager.get_pro_stats(HOTKEY) is None


def test_an_admin_edit_cannot_silently_deepen_a_candidates_drawdown(manager):
    _seed(manager, current_equity=0.99)
    _refresh(manager, {HOTKEY: _ledger(low=0.99)})

    success, message, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.05}, now_ms=NOW_MS)
    assert not success and "auto-promotion max_drawdown" in message
    success, _, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.05}, force=True, now_ms=NOW_MS)
    assert success


def test_an_admin_edit_on_a_funded_non_candidate_needs_no_force(manager):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value}
    _seed(manager, current_equity=0.99)
    _refresh(manager, {HOTKEY: _ledger(low=0.99)})

    success, _, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.05}, now_ms=NOW_MS)
    assert success
