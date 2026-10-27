"""
Auto-promotion candidates are standard subaccounts in SUBACCOUNT_FUNDED (created through SUBACCOUNT_CHALLENGE,
not HL-linked). From the first refresh that sees one, its drawdown is tracked live in DrawdownStats, ratcheted
as the pro challenge ratchets its own; the time before is replayed from the perf ledger
(LedgerUtils.eod_hwm_drawdown). Calmar, daily consistency and trading days are measured as in the pro challenge
into AutoPromotionStats. The pro track and every other account are unaffected.
"""
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
    TRANSITION,
    _bucket,
    _even_ledger,
    manager,
)
from vali_objects.challenge_period.challengeperiod_manager import DrawdownStats, ProStats
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
PRO_BUCKETS = (TRANSITION, MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED)


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


def _seed(manager, hotkey: str = HOTKEY, bucket=MinerBucket.SUBACCOUNT_FUNDED,
          current_equity: float = 1.0, eod_hwm: float = 1.0) -> None:
    manager.set_miner_bucket(hotkey, _bucket(bucket), NOW_MS - DAYS * DAILY_MS)
    if bucket == TRANSITION:
        manager._miner_account_client.set_transition(hotkey, NOW_MS - DAYS * DAILY_MS)
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
    {"initial_bucket": MinerBucket.PRO_CHALLENGE_FUNDED.value},          # Instant Funded
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
    manager._refresh_pro_stats([HOTKEY], {HOTKEY: _ledger(low=0.50)}, {HOTKEY: _account()},
                               manager._miner_account_client.get_transition_starts([HOTKEY]))
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
