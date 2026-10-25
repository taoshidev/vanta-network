"""
Auto-promotion moves a standard subaccount in SUBACCOUNT_FUNDED into its transition, with a
pro size at a multiple of its standard account, on whichever of two paths it qualifies for first:
  * fast track: equity, unrealized PnL included, at least 10% above the account size and held there on
    every refresh for 3 minutes, at any time (3x)
  * steady track: 60 trading days, the lower of balance and equity more than 3% above the account size,
    daily consistency at most 0.5, and calmar of at least 1.5 (5x) or 2.0 (10x)
On the same check it takes the larger multiple.
"""
import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from tests.vali_tests.test_auto_promotion_stats import _account, _ledger, _refresh, _seed
from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager and hop_manager are fixtures
    ACCOUNT_SIZE,
    DAILY_MS,
    HOTKEY,
    MONDAY_MS,
    NEXT_MONDAY_MS,
    NOW_MS,
    hop_manager,
    manager,
)
from entity_management.entity_client import EntityClient
from entity_management.entity_server import EntityServer
from tests.vali_tests.test_pro_account_size import STANDARD_SIZE, _add_standard, _bare_manager
from vali_objects.challenge_period.challengeperiod_manager import (
    AutoPromotion,
    AutoPromotionStats,
    ChallengePeriodManager,
    MinerBucketState,
)
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import BucketPhase, MinerBucket
from vali_objects.vali_config import RPCConnectionMode, ValiConfig

FAST = AutoPromotion.FAST_TRACK
STEADY = AutoPromotion.STEADY_TRACK
FUNDED = BucketPhase(MinerBucket.SUBACCOUNT_FUNDED)
TRANSITION = BucketPhase(MinerBucket.SUBACCOUNT_FUNDED, in_transition=True)
GROW = BucketPhase(MinerBucket.PRO_CHALLENGE_FUNDED)
WEEK_MS = 7 * DAILY_MS
SEVEN_DAYS_MS = ValiConfig.AUTO_PROMOTION_TRANSITION_DAYS * DAILY_MS
HOLD_MS = ChallengePeriodManager.AUTO_PROMOTION_FAST_TRACK_HOLD_MS
HELD_SINCE_MS = NOW_MS - HOLD_MS  # a fast track equity held just long enough at NOW_MS


def _stats(calmar: float = 1.5, consistency: float = 0.5, days: int = 60) -> AutoPromotionStats:
    return AutoPromotionStats(calmar=calmar, daily_consistency=consistency, max_drawdown=0.96, trading_days=days)


@pytest.fixture(autouse=True)
def _standard_subaccount(manager):
    """A standard subaccount's record, on the mothership."""
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {"initial_bucket": None}
    manager._entity_client.is_mothership.return_value = True


def _candidate(manager, hotkey: str = HOTKEY, stats: AutoPromotionStats | None = None, equity: float = 1.0,
               balance: float | None = None, held_since_ms: int | None = None) -> None:
    """A candidate in SUBACCOUNT_FUNDED with the given stats, equity and balance (over its account size,
    the balance defaulting to the equity), and a fast track hold that started at held_since_ms."""
    _seed(manager, hotkey, current_equity=equity)
    manager.miner_states[hotkey].drawdown.current_balance = equity if balance is None else balance
    manager._auto_promotion_stats[hotkey] = stats or _stats()
    if held_since_ms is not None:
        manager._fast_track_since_ms[hotkey] = held_since_ms


def _on_the_fast_track(manager, hotkey: str = HOTKEY) -> None:
    """A candidate that qualifies on the fast track only: 12% up on an open position, held long enough."""
    _candidate(manager, hotkey, equity=1.12, balance=1.0, held_since_ms=HELD_SINCE_MS)


def _auto_promote(manager, hotkeys=(HOTKEY,), now_ms: int = NOW_MS, eliminating=()) -> list[str]:
    return manager._auto_promote({hk: _account(hk) for hk in hotkeys}, set(eliminating), now_ms)


# ── Which path, and which multiple ────────────────────────────────────────────

_path = ChallengePeriodManager._auto_promotion_path


@pytest.mark.parametrize("held,days,expected", [
    (False, 60, None),
    (True, 60, (FAST, 3)),
    (True, 1, (FAST, 3)),      # no minimum time
])
def test_the_fast_track(held, days, expected):
    assert _path(_stats(calmar=0.0, days=days), held, current_return=0.0) == expected


@pytest.mark.parametrize("change,expected", [
    ({}, (STEADY, 5)),
    ({"calmar": 1.4999}, None),
    ({"calmar": 2.0}, (STEADY, 10)),
    ({"calmar": 1.9999}, (STEADY, 5)),
    ({"days": 59}, None),
    ({"return": 0.03}, None),    # must exceed 3%
    ({"consistency": 0.5001}, None),
])
def test_the_steady_track_at_each_boundary(change, expected):
    stats = _stats(calmar=change.get("calmar", 1.5), consistency=change.get("consistency", 0.5),
                   days=change.get("days", 60))
    assert _path(stats, False, change.get("return", 0.0301)) == expected


def test_qualifying_on_both_paths_takes_the_larger_multiple():
    assert _path(_stats(calmar=1.5), True, 0.12) == (STEADY, 5)
    assert _path(_stats(calmar=2.0), True, 0.12) == (STEADY, 10)
    assert _path(_stats(calmar=1.0), True, 0.12) == (FAST, 3)


def test_the_calmar_tiers_do_not_depend_on_their_order():
    with patch.object(ValiConfig, "AUTO_PROMOTION_STEADY_TRACK_CALMAR_MULTIPLIERS", ((1.5, 5), (2.0, 10))):
        assert _path(_stats(calmar=2.5), False, 0.05) == (STEADY, 10)


# ── The fast track's hold ─────────────────────────────────────────────────────

def _tick(manager, now_ms: int, equity: float, balance: float = 1.0) -> list[str]:
    """One refresh's stats and auto-promotion, with the account's equity and balance as given."""
    drawdown = manager.miner_states[HOTKEY].drawdown
    drawdown.current_equity, drawdown.current_balance = equity, balance
    _refresh(manager, {HOTKEY: _ledger(low=0.98)}, now_ms=now_ms)
    return _auto_promote(manager, now_ms=now_ms)


def test_the_fast_track_needs_its_equity_held_for_three_minutes(hop_manager):
    """10% up on an open position: unrealized gains count, but only once they have held."""
    _seed(hop_manager)
    assert _tick(hop_manager, NOW_MS, equity=1.10) == []
    assert _tick(hop_manager, NOW_MS + HOLD_MS - 30_000, equity=1.11) == []
    assert _tick(hop_manager, NOW_MS + HOLD_MS, equity=1.10) == [HOTKEY]

    record = hop_manager.miner_states[HOTKEY].auto_promotion
    assert (record.path, record.multiplier, record.promoted_ms) == (FAST, 3, NOW_MS + HOLD_MS)
    assert record.equity_return == pytest.approx(0.10) and record.balance_return == 0.0


def test_a_refresh_below_the_line_starts_the_hold_over(hop_manager):
    """A tick to 10% that falls back on the next refresh promotes nothing."""
    _seed(hop_manager)
    assert _tick(hop_manager, NOW_MS, equity=1.10) == []
    assert _tick(hop_manager, NOW_MS + 30_000, equity=1.0999) == []
    assert _tick(hop_manager, NOW_MS + HOLD_MS, equity=1.10) == []
    assert _tick(hop_manager, NOW_MS + 2 * HOLD_MS - 1, equity=1.10) == []
    assert _tick(hop_manager, NOW_MS + 2 * HOLD_MS, equity=1.10) == [HOTKEY]


def test_the_hold_is_dropped_once_the_account_is_no_longer_a_candidate(hop_manager):
    _seed(hop_manager)
    _tick(hop_manager, NOW_MS, equity=1.10)
    assert hop_manager._fast_track_since_ms == {HOTKEY: NOW_MS}

    hop_manager.miner_states[HOTKEY].add_bucket_entry(MinerBucket.ELIMINATED, NOW_MS + 30_000)
    _refresh(hop_manager, {HOTKEY: _ledger(low=0.98)}, now_ms=NOW_MS + 30_000)
    assert hop_manager._fast_track_since_ms == {}


@pytest.mark.parametrize("equity,balance,expected", [
    (1.0301, 1.0301, [HOTKEY]),
    (1.05, 1.02, []),   # a gain still open does not count
    (1.02, 1.05, []),   # a closed gain given back on an open position
])
def test_the_steady_track_reads_the_lower_of_balance_and_equity(hop_manager, equity, balance, expected):
    _candidate(hop_manager, equity=equity, balance=balance)
    assert _auto_promote(hop_manager) == expected


# ── The move ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("calmar,equity,balance,path,multiplier", [
    (0.0, 1.12, 1.0, FAST, 3),
    (1.5, 1.035, 1.035, STEADY, 5),
    (2.0, 1.035, 1.035, STEADY, 10),
])
def test_a_qualifying_candidate_moves_into_the_transition_at_its_multiple(hop_manager, calmar, equity, balance,
                                                                          path, multiplier):
    _candidate(hop_manager, stats=_stats(calmar=calmar), equity=equity, balance=balance,
               held_since_ms=HELD_SINCE_MS if path == FAST else None)

    assert _auto_promote(hop_manager) == [HOTKEY]

    state = hop_manager.miner_states[HOTKEY]
    assert state.current_phase == TRANSITION
    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, multiplier * ACCOUNT_SIZE, in_transition=True,
        waive_promotion_fee=False)
    assert state.auto_promotion == AutoPromotion(
        path=path, multiplier=multiplier, pro_account_size=multiplier * ACCOUNT_SIZE, promoted_ms=NOW_MS,
        equity_return=pytest.approx(equity - 1.0), balance_return=pytest.approx(balance - 1.0), calmar=calmar,
        daily_consistency=0.5, trading_days=60)


@pytest.mark.parametrize("mothership", [True, False])
def test_a_pro_size_already_on_the_record_is_kept(hop_manager, mothership):
    """The mothership's sizing (or the entity's) is already on the record: it is followed, never
    overwritten, and the record states the multiple actually granted."""
    hop_manager._entity_client.is_mothership.return_value = mothership
    hop_manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "account_type": "pro", "pro_account_size": 500_000}
    _on_the_fast_track(hop_manager)

    assert _auto_promote(hop_manager) == [HOTKEY]

    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, None, in_transition=True, waive_promotion_fee=False)
    record = hop_manager.miner_states[HOTKEY].auto_promotion
    assert (record.path, record.multiplier, record.pro_account_size) == (FAST, 5.0, 500_000)


def test_a_validator_other_than_the_mothership_waits_for_its_size(hop_manager):
    """It qualifies on its own, but only moves once the mothership's sizing reaches its record."""
    hop_manager._entity_client.is_mothership.return_value = False
    _on_the_fast_track(hop_manager)

    with patch("vali_objects.challenge_period.challengeperiod_manager.logger") as log:
        assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()
    log.error.assert_not_called()
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None

    hop_manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "account_type": "pro", "pro_account_size": 300_000}
    assert _auto_promote(hop_manager, now_ms=NOW_MS + 30_000) == [HOTKEY]
    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, None, in_transition=True, waive_promotion_fee=False)
    assert hop_manager.miner_states[HOTKEY].auto_promotion.pro_account_size == 300_000


def test_a_candidate_that_does_not_qualify_stays_funded(hop_manager):
    _candidate(hop_manager, stats=_stats(calmar=1.0), equity=1.035)
    assert _auto_promote(hop_manager) == []
    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None


def test_a_candidate_without_stats_is_not_evaluated(hop_manager):
    """Its equity alone would qualify, but its ledger cannot replay its drawdown yet."""
    _on_the_fast_track(hop_manager)
    hop_manager._auto_promotion_stats.clear()

    assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()


def test_an_account_promoted_automatically_before_is_never_evaluated_again(hop_manager):
    _on_the_fast_track(hop_manager)
    hop_manager.miner_states[HOTKEY].auto_promotion = AutoPromotion(
        path=FAST, multiplier=3, pro_account_size=300_000, promoted_ms=NOW_MS - WEEK_MS, equity_return=0.11,
        balance_return=0.02, calmar=0.0, daily_consistency=1.0, trading_days=20)

    with patch.object(hop_manager, "_move_to_auto_promotion") as move:
        assert _auto_promote(hop_manager) == []
    move.assert_not_called()


def test_an_eliminated_account_is_not_evaluated(hop_manager):
    _on_the_fast_track(hop_manager)
    hop_manager.miner_states[HOTKEY].add_bucket_entry(MinerBucket.ELIMINATED, MONDAY_MS - DAILY_MS)
    with patch.object(hop_manager, "_move_to_auto_promotion") as move:
        assert _auto_promote(hop_manager) == []
    move.assert_not_called()


def test_an_account_this_refresh_is_eliminating_is_not_evaluated(hop_manager):
    """Even when its elimination failed and left it in SUBACCOUNT_FUNDED."""
    _on_the_fast_track(hop_manager)
    assert _auto_promote(hop_manager, eliminating=[HOTKEY]) == []
    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED


def test_a_failed_move_leaves_no_record_and_is_retried_but_logged_once_a_week(hop_manager):
    _on_the_fast_track(hop_manager)
    hop_manager._entity_client.apply_bucket_account_size.return_value = (False, "rejected")

    with patch("vali_objects.challenge_period.challengeperiod_manager.logger") as log:
        assert _auto_promote(hop_manager) == []
        assert _auto_promote(hop_manager, now_ms=NOW_MS + 30_000) == []
        assert log.error.call_count == 1
        _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS)
        assert log.error.call_count == 2
    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None

    hop_manager._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
    assert _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + 30_000) == [HOTKEY]


def test_a_move_that_lands_before_raising_keeps_its_record(hop_manager):
    _on_the_fast_track(hop_manager)
    real_promote = hop_manager.promote_subaccount

    def promote_then_raise(*args, **kwargs):
        real_promote(*args, **kwargs)
        raise OSError("disk full")

    with patch.object(hop_manager, "promote_subaccount", side_effect=promote_then_raise):
        assert _auto_promote(hop_manager) == [HOTKEY]
    assert hop_manager.miner_states[HOTKEY].auto_promotion.multiplier == 3


@pytest.mark.parametrize("mothership_record", [
    None,  # the entity promoted itself on the mothership
    AutoPromotion(path=STEADY, multiplier=5.0, pro_account_size=500_000, promoted_ms=NOW_MS - 60_000),
])
def test_a_sync_during_the_move_leaves_the_mothership_state_as_it_came(hop_manager, mothership_record):
    """An auto-sync swaps in the mothership's state, already in the transition, while the move runs:
    that state stands, with its own record or none."""
    _on_the_fast_track(hop_manager)
    real_promote = hop_manager.promote_subaccount

    def promote_then_sync(*args, **kwargs):
        result = real_promote(*args, **kwargs)
        synced = MinerBucketState.from_checkpoint_dict(HOTKEY, hop_manager.miner_states[HOTKEY].to_checkpoint_dict())
        synced.auto_promotion = mothership_record
        hop_manager.miner_states[HOTKEY] = synced
        return result

    with patch.object(hop_manager, "promote_subaccount", side_effect=promote_then_sync):
        _auto_promote(hop_manager)
    assert hop_manager.miner_states[HOTKEY].auto_promotion == mothership_record


def test_a_move_that_raises_before_landing_does_not_claim_the_entitys_move(hop_manager):
    """The call fails before its own bucket entry while the entity's request moves the subaccount."""
    _on_the_fast_track(hop_manager)
    real_promote = hop_manager.promote_subaccount

    def entity_lands_then_ours_raises(*args, **kwargs):
        assert real_promote(HOTKEY, NOW_MS - 1_000, 250_000)[0]  # the entity's request
        raise TimeoutError("apply_bucket_account_size")

    with patch.object(hop_manager, "promote_subaccount", side_effect=entity_lands_then_ours_raises):
        assert _auto_promote(hop_manager) == []
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None


def test_an_entity_promotion_landing_first_is_left_alone(hop_manager):
    """The entity's own request moves the subaccount into the transition between the decision and
    the move: the auto-promotion does not hop it onward, nor claim the entity's move as its own."""
    _on_the_fast_track(hop_manager)
    real_promote = hop_manager.promote_subaccount

    def entity_request_lands_first(*args, **kwargs):
        assert real_promote(HOTKEY, NOW_MS, 250_000)[0]  # the entity's request, no from_phase
        return real_promote(*args, **kwargs)

    with patch.object(hop_manager, "promote_subaccount", side_effect=entity_request_lands_first):
        assert _auto_promote(hop_manager) == []

    state = hop_manager.miner_states[HOTKEY]
    assert state.current_phase == TRANSITION
    assert state.auto_promotion is None
    hop_manager._entity_client.apply_bucket_account_size.assert_called_once_with(
        HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, 250_000, in_transition=True, waive_promotion_fee=False)


def test_the_move_is_only_ever_the_hop_out_of_funded(hop_manager):
    hop_manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS, in_transition=True)

    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS, 300_000,
                                                      from_phase=FUNDED)

    assert not success and "no longer" in message
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()


# ── The record ────────────────────────────────────────────────────────────────

def test_the_record_is_saved_with_the_state():
    record = AutoPromotion(path=STEADY, multiplier=10, pro_account_size=1_000_000, promoted_ms=NOW_MS,
                           equity_return=0.06, balance_return=0.05, calmar=2.3, daily_consistency=0.3,
                           trading_days=75)
    state = MinerBucketState.from_checkpoint_dict(HOTKEY, {"entries": [{"bucket": TRANSITION.value,
                                                                        "in_transition": True,
                                                                        "start_time_ms": NOW_MS}]})
    assert state.auto_promotion is None
    state.auto_promotion = record
    assert MinerBucketState.from_checkpoint_dict(HOTKEY, state.to_checkpoint_dict()).auto_promotion == record


def test_a_record_with_missing_or_unknown_fields_still_loads():
    """Written by another version: missing fields take their defaults, unknown ones are ignored."""
    data = {"entries": [{"bucket": TRANSITION.value, "in_transition": True, "start_time_ms": NOW_MS}],
            "auto_promotion": {"path": FAST, "multiplier": 3, "a_later_field": 1}}
    record = MinerBucketState.from_checkpoint_dict(HOTKEY, data).auto_promotion
    assert (record.path, record.multiplier, record.pro_account_size) == (FAST, 3, 0.0)


def test_a_sync_that_fails_to_parse_keeps_the_current_states(manager):
    _seed(manager)
    with pytest.raises(Exception):
        manager.sync_challenge_period_data({"other": {"entries": [{"bucket": "NOT_A_BUCKET"}]}})
    assert manager.miner_states[HOTKEY].current_phase == FUNDED


# ── In the transition: seven days at most, no fee, the granted size, one promotion at a time ──

_expired = ChallengePeriodManager._check_transition_expiry


def _in_transition(manager, entered_ms: int = NOW_MS, auto_promoted: bool = True,
                   hotkey: str = HOTKEY) -> MinerBucketState:
    manager.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - 60 * DAILY_MS)
    state = manager.miner_states[hotkey]
    state.add_bucket_entry(MinerBucket.SUBACCOUNT_FUNDED, entered_ms, in_transition=True)
    if auto_promoted:
        state.auto_promotion = AutoPromotion(path=FAST, multiplier=3.0, pro_account_size=300_000,
                                             promoted_ms=entered_ms)
    return state


def test_an_auto_promoted_account_moves_on_seven_days_after_its_promotion(manager):
    state = _in_transition(manager)
    assert not _expired(state, NEXT_MONDAY_MS + 60_000)   # not at the next Monday
    assert not _expired(state, NOW_MS + SEVEN_DAYS_MS - 1)
    assert _expired(state, NOW_MS + SEVEN_DAYS_MS)


def test_a_self_service_transition_still_moves_on_at_the_next_monday(manager):
    state = _in_transition(manager, auto_promoted=False)
    assert not _expired(state, NEXT_MONDAY_MS - 1)
    assert _expired(state, NEXT_MONDAY_MS)


def test_an_auto_promoted_account_moves_on_as_soon_as_it_is_flat(manager):
    state = _in_transition(manager)
    assert _expired(state, NOW_MS, flat=True)
    assert not _expired(state, NOW_MS)


def test_a_flat_self_service_transition_still_waits_for_the_next_monday(manager):
    state = _in_transition(manager, auto_promoted=False)
    assert not _expired(state, NEXT_MONDAY_MS - 1, flat=True)


def test_only_auto_promoted_transitions_are_looked_up_and_only_the_flat_ones_returned(manager):
    flat, holding, self_service = HOTKEY, "entity_hk_1", "entity_hk_2"
    for hotkey in (flat, holding):
        _in_transition(manager, hotkey=hotkey)
    _in_transition(manager, auto_promoted=False, hotkey=self_service)
    manager.set_miner_bucket("entity_hk_3", MinerBucket.SUBACCOUNT_FUNDED, NOW_MS)
    manager._position_client.get_positions_for_hotkeys.return_value = {holding: [MagicMock()]}

    assert manager._flat_auto_promoted_transitions() == {flat}

    manager._position_client.get_positions_for_hotkeys.assert_called_once_with(
        [flat, holding], only_open_positions=True)


def test_no_open_position_lookup_without_an_auto_promoted_transition(manager):
    _in_transition(manager, auto_promoted=False)
    assert manager._flat_auto_promoted_transitions() == set()
    manager._position_client.get_positions_for_hotkeys.assert_not_called()


def test_the_seven_days_count_from_the_promotion_even_when_the_transition_was_entered_later(manager):
    state = _in_transition(manager, entered_ms=NOW_MS + 3 * DAILY_MS)
    state.auto_promotion.promoted_ms = NOW_MS
    assert not _expired(state, NOW_MS + SEVEN_DAYS_MS - 1)
    assert _expired(state, NOW_MS + SEVEN_DAYS_MS)


def test_a_record_without_its_promotion_time_counts_from_entering_the_transition(manager):
    state = _in_transition(manager)
    state.auto_promotion.promoted_ms = 0
    assert not _expired(state, NOW_MS + SEVEN_DAYS_MS - 1)
    assert _expired(state, NOW_MS + SEVEN_DAYS_MS)


@pytest.mark.parametrize("auto_promoted", [True, False])
def test_only_an_auto_promoted_account_moves_on_free_of_the_fee(hop_manager, auto_promoted):
    subaccount = "entity_hk_0"  # the switch only sizes subaccounts
    _in_transition(hop_manager, auto_promoted=auto_promoted, hotkey=subaccount)
    assert hop_manager.promote_hotkeys({subaccount: TRANSITION}, NOW_MS + SEVEN_DAYS_MS)
    assert hop_manager.miner_states[subaccount].current_phase == GROW
    hop_manager._entity_client.apply_bucket_account_size.assert_called_with(
        subaccount, MinerBucket.PRO_CHALLENGE_FUNDED, in_transition=False, waive_promotion_fee=auto_promoted)


@pytest.mark.parametrize("waive", [True, False])
def test_the_entity_manager_waives_the_fee_and_its_collateral_check_on_request(waive):
    """A fee-paying subaccount whose entity holds no collateral at all."""
    entity_manager = _bare_manager()
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.0
    hotkey = _add_standard(entity_manager)
    entity_manager.get_subaccount_info_for_synthetic(hotkey).reg_fee_theta = STANDARD_SIZE / 1_500
    assert entity_manager.apply_bucket_account_size(hotkey, MinerBucket.SUBACCOUNT_FUNDED, 300_000,
                                                   in_transition=True)[0]

    success, message = entity_manager.apply_bucket_account_size(hotkey, MinerBucket.PRO_CHALLENGE_FUNDED,
                                                               waive_promotion_fee=waive)

    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    if waive:
        assert success and info.account_size == 300_000 and info.pro_fee_theta == 0
    else:
        assert not success and "Insufficient collateral" in message and info.account_size == STANDARD_SIZE


@pytest.mark.parametrize("waive", [True, False])
def test_the_waiver_reaches_the_entity_manager_through_the_client_and_server(waive):
    server = EntityServer.__new__(EntityServer)  # only its forwarding is under test
    server._manager = MagicMock()
    client = EntityClient(connection_mode=RPCConnectionMode.LOCAL, running_unit_tests=True)
    client.set_direct_server(server)
    client.apply_bucket_account_size(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, waive_promotion_fee=waive)
    server._manager.apply_bucket_account_size.assert_called_once_with(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, None, in_transition=False, waive_promotion_fee=waive)


def test_the_entity_can_move_an_auto_promoted_account_on_early_free_of_the_fee(hop_manager):
    _in_transition(hop_manager)
    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS)[0]
    assert hop_manager.miner_states[HOTKEY].current_phase == GROW
    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, None, in_transition=False, waive_promotion_fee=True)


def test_the_entity_cannot_change_an_auto_promoted_accounts_size(hop_manager):
    _in_transition(hop_manager)
    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS, 500_000)
    assert not success and "promoted automatically" in message
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()


def test_an_early_entity_hop_goes_onto_the_granted_size_without_charging_the_fee(hop_manager):
    """Through a real entity manager whose entity holds no collateral at all."""
    entity_manager = _bare_manager()
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.0
    hotkey = _add_standard(entity_manager)
    entity_manager.get_subaccount_info_for_synthetic(hotkey).reg_fee_theta = STANDARD_SIZE / 1_500
    assert entity_manager.apply_bucket_account_size(hotkey, MinerBucket.SUBACCOUNT_FUNDED, 300_000,
                                                   in_transition=True)[0]
    hop_manager._entity_client = _EntityClient(entity_manager)
    _in_transition(hop_manager, hotkey=hotkey)

    success, message = hop_manager.promote_subaccount(hotkey, NOW_MS + DAILY_MS)

    assert success, message
    assert hop_manager.miner_states[hotkey].current_phase == GROW
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_size == 300_000 and info.pro_fee_theta == 0
    entity_manager._entity_collateral_client.offset_collateral_cache.assert_not_called()


def test_the_entity_can_still_move_its_own_promotion_on(hop_manager):
    _in_transition(hop_manager, auto_promoted=False)
    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS)[0]
    assert hop_manager.miner_states[HOTKEY].current_phase == GROW


def _runs_only_after_the_lock_is_released(manager, call) -> None:
    """Hold the subaccount's promotion lock, start `call` on another thread, and check it waits for it."""
    done = threading.Event()
    worker = threading.Thread(target=lambda: (call(), done.set()))
    with manager._promotion_lock(HOTKEY):
        worker.start()
        assert not done.wait(0.2)
        state = manager.miner_states[HOTKEY]
        assert state.current_phase == FUNDED and state.auto_promotion is None
    worker.join(5)
    assert done.is_set()


def test_an_entity_promotion_waits_for_the_subaccounts_promotion_lock(hop_manager):
    hop_manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - DAILY_MS)
    _runs_only_after_the_lock_is_released(hop_manager, lambda: hop_manager.promote_subaccount(HOTKEY, NOW_MS, 250_000))
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION


def test_an_auto_promotion_waits_for_the_subaccounts_promotion_lock(hop_manager):
    _on_the_fast_track(hop_manager)
    _runs_only_after_the_lock_is_released(hop_manager, lambda: _auto_promote(hop_manager))
    assert hop_manager.miner_states[HOTKEY].auto_promotion.multiplier == 3


@pytest.mark.parametrize("auto_promoted", [True, False])
def test_an_entity_promotion_after_the_evaluation_is_not_promoted_again(hop_manager, auto_promoted):
    """The refresh found the account in the transition, then the entity's request moved it on to Grow:
    the refresh does not take it a step further, onto pro funded."""
    _in_transition(hop_manager, auto_promoted=auto_promoted)
    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS)[0]

    assert not hop_manager.promote_hotkeys({HOTKEY: TRANSITION}, NOW_MS + DAILY_MS)
    assert hop_manager.miner_states[HOTKEY].current_phase == GROW


def test_a_scheduled_promotion_waits_for_an_entity_promotion_in_progress(hop_manager):
    _in_transition(hop_manager)
    done = threading.Event()
    worker = threading.Thread(
        target=lambda: (hop_manager.promote_hotkeys({HOTKEY: TRANSITION}, NOW_MS + DAILY_MS), done.set()))
    with hop_manager._promotion_lock(HOTKEY):
        worker.start()
        assert not done.wait(0.2)
        assert hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS)[0]
    worker.join(5)
    assert done.is_set()
    assert hop_manager.miner_states[HOTKEY].current_phase == GROW


def test_through_refresh_an_entity_promotion_during_the_refresh_is_not_promoted_again(hop_manager):
    """The entity's request lands after the refresh found the account flat, before it moves it."""
    _in_transition(hop_manager)

    def entity_request_lands(*args, **kwargs):
        assert hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS)[0]
        return False

    with (patch.object(hop_manager, "_refresh_auto_promotion_stats"),
          patch.object(hop_manager, "eliminate_hotkeys", side_effect=entity_request_lands)):
        _full_refresh(hop_manager, HOTKEY, NOW_MS + DAILY_MS)

    assert hop_manager.miner_states[HOTKEY].current_phase == GROW


def test_a_duplicate_request_that_waited_on_the_lock_makes_no_second_hop(hop_manager):
    hop_manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - DAILY_MS)
    results = []
    duplicate = threading.Thread(
        target=lambda: results.append(hop_manager.promote_subaccount(HOTKEY, NOW_MS, 250_000)))
    with hop_manager._promotion_lock(HOTKEY):
        duplicate.start()
        duplicate.join(0.2)
        assert duplicate.is_alive()  # waiting on the lock
        assert hop_manager.promote_subaccount(HOTKEY, NOW_MS, 250_000)[0]
    duplicate.join(5)
    success, message = results[0]
    assert not success and "no longer" in message
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION


@pytest.mark.parametrize("since_the_decision", ["promoted by the entity", "already auto-promoted"])
def test_the_move_checks_the_account_again_before_promoting_it(hop_manager, since_the_decision):
    """Nothing is moved or recorded when the account changed after the decision."""
    _on_the_fast_track(hop_manager)
    state = hop_manager.miner_states[HOTKEY]
    if since_the_decision == "promoted by the entity":
        state.add_bucket_entry(MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - 1_000, in_transition=True)
    else:
        state.auto_promotion = AutoPromotion(path=FAST, multiplier=3.0, pro_account_size=300_000,
                                             promoted_ms=NOW_MS - SEVEN_DAYS_MS)
    record = state.auto_promotion

    with patch.object(hop_manager, "promote_subaccount") as promote:
        assert not hop_manager._move_to_auto_promotion(HOTKEY, _account(), _stats(), (FAST, 3), NOW_MS)
    promote.assert_not_called()
    assert state.auto_promotion is record


# ── Through refresh(), and on to Grow ─────────────────────────────────────────

class _EntityClient:
    """A real EntityManager behind the client's interface: records cross RPC as dicts."""

    def __init__(self, entity_manager):
        self._manager = entity_manager

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def get_subaccount_info_for_synthetic(self, hotkey):
        info = self._manager.get_subaccount_info_for_synthetic(hotkey)
        return info.model_dump() if info else None

    def is_mothership(self):
        return True


def _seed_on_the_fast_track(manager, hotkey: str = HOTKEY) -> None:
    """As _on_the_fast_track, for a full refresh: 12% up on an open position, held long enough."""
    _seed(manager, hotkey, current_equity=1.12)
    manager.miner_states[hotkey].drawdown.current_balance = 1.0
    manager._fast_track_since_ms[hotkey] = HELD_SINCE_MS


def _full_refresh(manager, hotkey: str, now_ms: int, ledger=None, eliminated: bool = False, holding: bool = False,
                  account=None, real_drawdown_cache: bool = False):
    """One refresh() pass with the drawdown cache pinned, so the stats, auto-promotion and routing run for real.
    holding gives the account an open position. real_drawdown_cache reads equity and balance from the
    account instead."""
    manager._position_client.get_all_hotkeys.return_value = [hotkey]
    manager._position_client.filtered_positions_for_scoring.return_value = ({hotkey: []}, {})
    manager._position_client.get_positions_for_hotkeys.return_value = {hotkey: [MagicMock()] if holding else []}
    manager._elimination_client.get_eliminated_hotkeys.return_value = []
    manager._plagiarism_client.get_plagiarism_miners.return_value = []
    manager._miner_account_client.get_accounts.return_value = {hotkey: account or _account(hotkey)}
    manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = {hotkey: ledger or _ledger(low=0.98)}
    manager._asset_selection_client.get_asset_selections.return_value = {hotkey: MinerAssetClass.FOREX}
    pinned = ["_refresh_rank_cache", "_save_to_disk", "_sync_buckets_to_accounts"]
    if real_drawdown_cache:
        manager._position_client.get_first_order_times.return_value = {hotkey: NOW_MS - 60 * DAILY_MS}
    else:
        pinned.append("_refresh_drawdown_cache")
    with contextlib.ExitStack() as stack:
        for name in pinned:
            stack.enter_context(patch.object(manager, name))
        return manager.refresh(current_time_ms=now_ms)


def test_through_refresh_a_qualifying_account_reaches_grow_seven_days_later_free_of_the_fee(hop_manager):
    """7% realized over a 2% drawdown is calmar 3.5: the 10x tier. The transition records the pro size
    and keeps trading the standard account, past the next Monday while it holds a position; seven days
    after the promotion it goes onto the pro size, with no promotion fee charged even though the entity
    holds no collateral."""
    entity_manager = _bare_manager()
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.0
    hotkey = _add_standard(entity_manager)
    entity_manager.get_subaccount_info_for_synthetic(hotkey).reg_fee_theta = STANDARD_SIZE / 1_500
    hop_manager._entity_client = _EntityClient(entity_manager)
    _seed(hop_manager, hotkey, current_equity=1.07)

    message = _full_refresh(hop_manager, hotkey, NOW_MS)

    state = hop_manager.miner_states[hotkey]
    assert state.current_phase == TRANSITION
    assert state.auto_promotion.path == STEADY and state.auto_promotion.multiplier == 10
    assert hotkey in message
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 10 * STANDARD_SIZE
    assert info.account_size == STANDARD_SIZE

    _full_refresh(hop_manager, hotkey, NEXT_MONDAY_MS + 60_000, holding=True)
    assert hop_manager.miner_states[hotkey].current_phase == TRANSITION

    _full_refresh(hop_manager, hotkey, NOW_MS + SEVEN_DAYS_MS, holding=True)

    assert hop_manager.miner_states[hotkey].current_phase == GROW
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_size == 10 * STANDARD_SIZE
    assert info.pro_fee_theta == 0
    entity_manager._entity_collateral_client.offset_collateral_cache.assert_not_called()


@pytest.mark.parametrize("realized_usd,unrealized_usd,promoted", [
    (0.0, 12_000.0, True),         # 12% up on an open position
    (12_000.0, -3_000.0, False),   # 12% realized, 3% of it given back on an open position: equity 9% up
])
def test_through_refresh_the_hold_reads_the_accounts_own_equity(hop_manager, realized_usd, unrealized_usd, promoted):
    """The drawdown cache turns the account's realized and unrealized PnL into the equity the hold reads,
    so an open loss counts against it. The ledger's calmar of 0.7 keeps the steady track out of it."""
    hop_manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - 60 * DAILY_MS)
    account = _account()
    account.total_realized_pnl, account.unrealized_pnl = realized_usd, unrealized_usd

    for now_ms in (NOW_MS, NOW_MS + HOLD_MS):
        _full_refresh(hop_manager, HOTKEY, now_ms, ledger=_ledger(low=0.90), holding=True, account=account,
                      real_drawdown_cache=True)

    drawdown = hop_manager.miner_states[HOTKEY].drawdown
    assert (drawdown.current_equity, drawdown.current_balance) == pytest.approx(
        (1.0 + (realized_usd + unrealized_usd) / ACCOUNT_SIZE, 1.0 + realized_usd / ACCOUNT_SIZE))
    state = hop_manager.miner_states[HOTKEY]
    if promoted:
        assert state.current_phase == TRANSITION and state.auto_promotion.path == FAST
    else:
        assert state.current_phase == FUNDED and state.auto_promotion is None


def test_through_refresh_the_fast_track_promotes_once_its_equity_has_held(hop_manager):
    _seed(hop_manager, current_equity=1.12)
    hop_manager.miner_states[HOTKEY].drawdown.current_balance = 1.0

    _full_refresh(hop_manager, HOTKEY, NOW_MS, holding=True)
    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED

    _full_refresh(hop_manager, HOTKEY, NOW_MS + HOLD_MS, holding=True)
    state = hop_manager.miner_states[HOTKEY]
    assert state.current_phase == TRANSITION
    assert (state.auto_promotion.path, state.auto_promotion.multiplier) == (FAST, 3)


def test_through_refresh_an_account_flat_when_promoted_reaches_grow_on_the_next_refresh(hop_manager):
    _seed_on_the_fast_track(hop_manager)

    _full_refresh(hop_manager, HOTKEY, NOW_MS)
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION

    _full_refresh(hop_manager, HOTKEY, NOW_MS + 30_000)
    assert hop_manager.miner_states[HOTKEY].current_phase == GROW


def test_through_refresh_the_transition_ends_once_its_last_position_is_closed(hop_manager):
    _seed_on_the_fast_track(hop_manager)
    _full_refresh(hop_manager, HOTKEY, NOW_MS, holding=True)

    _full_refresh(hop_manager, HOTKEY, NOW_MS + DAILY_MS, holding=True)
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION

    _full_refresh(hop_manager, HOTKEY, NOW_MS + 2 * DAILY_MS)
    assert hop_manager.miner_states[HOTKEY].current_phase == GROW


def test_through_refresh_a_failed_open_position_lookup_keeps_the_transition(hop_manager):
    _in_transition(hop_manager)
    hop_manager._position_client.get_positions_for_hotkeys.side_effect = TimeoutError("position RPC")
    with patch.object(hop_manager, "_refresh_auto_promotion_stats"):
        _full_refresh(hop_manager, HOTKEY, NOW_MS + DAILY_MS)
    hop_manager._position_client.get_positions_for_hotkeys.assert_called_once()
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION


def test_through_refresh_a_ledger_missing_lows_holds_the_promotion(hop_manager):
    """Its equity alone would qualify, but a checkpoint's low is unknown until the ledger is regenerated."""
    _seed_on_the_fast_track(hop_manager)
    ledger = _ledger(low=0.98)
    ledger.cps[0].min_equity_ret = None

    _full_refresh(hop_manager, HOTKEY, NOW_MS, ledger)
    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED

    _full_refresh(hop_manager, HOTKEY, NOW_MS + 60_000)
    assert hop_manager.miner_states[HOTKEY].current_phase == TRANSITION


def test_through_refresh_an_account_whose_elimination_failed_is_not_promoted(hop_manager):
    """An intraday breach whose elimination row could not be written leaves the account in its bucket,
    though its equity, 12% up overall but down from a day that opened 30% up, holds the fast track."""
    _seed_on_the_fast_track(hop_manager)
    hop_manager.miner_states[HOTKEY].drawdown.daily_open_equity = 1.30
    hop_manager._elimination_client.append_elimination_row.side_effect = TimeoutError("elimination RPC")

    _full_refresh(hop_manager, HOTKEY, NOW_MS)

    hop_manager._elimination_client.append_elimination_row.assert_called_once()
    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED


def test_a_failing_auto_promotion_pass_never_holds_up_the_refresh(hop_manager):
    _seed(hop_manager)
    with (patch.object(hop_manager, "_auto_promote", side_effect=RuntimeError("unexpected")),
          patch.object(hop_manager, "_save_to_disk") as save):
        hop_manager._position_client.get_all_hotkeys.return_value = [HOTKEY]
        hop_manager._position_client.filtered_positions_for_scoring.return_value = ({HOTKEY: []}, {})
        hop_manager._position_client.get_positions_for_hotkeys.return_value = {HOTKEY: []}
        hop_manager._elimination_client.get_eliminated_hotkeys.return_value = []
        hop_manager._plagiarism_client.get_plagiarism_miners.return_value = []
        hop_manager._miner_account_client.get_accounts.return_value = {HOTKEY: _account()}
        hop_manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = {HOTKEY: _ledger()}
        hop_manager._asset_selection_client.get_asset_selections.return_value = {HOTKEY: MinerAssetClass.FOREX}
        with patch.object(hop_manager, "_refresh_drawdown_cache"), patch.object(hop_manager, "_refresh_rank_cache"):
            hop_manager.refresh(current_time_ms=NOW_MS)
    save.assert_called()


def test_failing_auto_promotion_stats_never_stop_the_eliminations(hop_manager):
    """The stats run before the eliminations: if they fail, the breach is still eliminated, nothing
    is evaluated for auto-promotion on the stale stats, and the fast track's hold starts over."""
    _seed(hop_manager, current_equity=0.90)
    hop_manager.miner_states[HOTKEY].drawdown.daily_open_equity = 1.0
    hop_manager._auto_promotion_stats[HOTKEY] = _stats()
    hop_manager._fast_track_since_ms[HOTKEY] = HELD_SINCE_MS

    with patch.object(hop_manager, "_refresh_auto_promotion_stats", side_effect=RuntimeError("unexpected")):
        _full_refresh(hop_manager, HOTKEY, NOW_MS)

    hop_manager._elimination_client.append_elimination_row.assert_called_once()
    assert hop_manager._auto_promotion_stats == {}
    assert hop_manager._fast_track_since_ms == {}


@pytest.mark.parametrize("record", [{"initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value}, None])
def test_through_refresh_a_non_candidate_is_never_promoted(hop_manager, record):
    hop_manager._entity_client.get_subaccount_info_for_synthetic.return_value = record
    _seed_on_the_fast_track(hop_manager)

    _full_refresh(hop_manager, HOTKEY, NOW_MS)

    assert hop_manager.miner_states[HOTKEY].current_phase == FUNDED
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()
    assert hop_manager._fast_track_since_ms == {}
