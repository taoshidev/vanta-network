"""
Auto-promotion moves a standard subaccount out of SUBACCOUNT_FUNDED into PRO_CHALLENGE_TRANSITION, with a
pro size at a multiple of its standard account, on whichever of two paths it qualifies for first:
  * fast track: total payouts above 10% of the account size, at any time (3x)
  * steady track: 60 trading days, 3 paid weeks, payouts above 3%, daily consistency at most 0.5,
    and calmar of at least 1.5 (5x) or 2.0 (10x)
On the same check it takes the larger multiple. Payouts are the payout endpoint's weekly figures over closed
payout weeks.
"""
import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from tests.vali_tests.test_auto_promotion_stats import _account, _ledger, _seed
from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager and hop_manager are fixtures
    ACCOUNT_SIZE,
    DAILY_MS,
    HOTKEY,
    LEDGER_START_MS,
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
    FundedPayouts,
    MinerBucketState,
)
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import RPCConnectionMode, ValiConfig

FAST = AutoPromotion.FAST_TRACK
STEADY = AutoPromotion.STEADY_TRACK
TRANSITION = MinerBucket.PRO_CHALLENGE_TRANSITION
WEEK_MS = 7 * DAILY_MS
READABLE_MS = ChallengePeriodManager.AUTO_PROMOTION_PAYOUT_LOOKUP_DELAY_MS  # into a week, its payouts can be read
SEVEN_DAYS_MS = ValiConfig.AUTO_PROMOTION_TRANSITION_DAYS * DAILY_MS


def _stats(calmar: float = 1.5, consistency: float = 0.5, days: int = 60,
           ledger_start_ms: int = LEDGER_START_MS) -> AutoPromotionStats:
    return AutoPromotionStats(calmar=calmar, daily_consistency=consistency, max_drawdown=0.96, trading_days=days,
                              ledger_start_ms=ledger_start_ms)


def _weeks(*payouts: float) -> dict[int, float]:
    """Closed payout weeks ending at this week's Monday, oldest first."""
    return {MONDAY_MS - (len(payouts) - i) * WEEK_MS: p for i, p in enumerate(payouts)}


@pytest.fixture(autouse=True)
def _standard_subaccount(manager):
    """A standard subaccount's record, on the mothership."""
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {"initial_bucket": None}
    manager._entity_client.get_funded_week_payouts.return_value = {}
    manager._entity_client.is_mothership.return_value = True


def _candidate(manager, hotkey: str = HOTKEY, stats: AutoPromotionStats | None = None,
               weeks: dict | None = None) -> None:
    """A candidate in SUBACCOUNT_FUNDED with the given stats and payout weeks."""
    _seed(manager, hotkey)
    manager._auto_promotion_stats[hotkey] = stats or _stats()
    if weeks is not None:
        manager._entity_client.get_funded_week_payouts.return_value = weeks


def _auto_promote(manager, hotkeys=(HOTKEY,), now_ms: int = NOW_MS, eliminating=()) -> list[str]:
    return manager._auto_promote({hk: _account(hk) for hk in hotkeys}, set(eliminating), now_ms)


# ── Which path, and which multiple ────────────────────────────────────────────

_path = ChallengePeriodManager._auto_promotion_path


@pytest.mark.parametrize("share,days,expected", [
    (0.10, 60, None),            # payouts must exceed 10%
    (0.1001, 60, (FAST, 3)),
    (0.1001, 1, (FAST, 3)),    # no minimum time
])
def test_the_fast_track(share, days, expected):
    stats = _stats(calmar=0.0, days=days)
    assert _path(stats, share, paid_weeks=1) == expected


@pytest.mark.parametrize("change,expected", [
    ({}, (STEADY, 5)),
    ({"calmar": 1.4999}, None),
    ({"calmar": 2.0}, (STEADY, 10)),
    ({"calmar": 1.9999}, (STEADY, 5)),
    ({"days": 59}, None),
    ({"paid_weeks": 2}, None),
    ({"share": 0.03}, None),     # payouts must exceed 3%
    ({"consistency": 0.5001}, None),
])
def test_the_steady_track_at_each_boundary(change, expected):
    stats = _stats(calmar=change.get("calmar", 1.5), consistency=change.get("consistency", 0.5),
                   days=change.get("days", 60))
    assert _path(stats, change.get("share", 0.0301), change.get("paid_weeks", 3)) == expected


def test_qualifying_on_both_paths_takes_the_larger_multiple():
    assert _path(_stats(calmar=1.5), 0.12, 3) == (STEADY, 5)
    assert _path(_stats(calmar=2.0), 0.12, 3) == (STEADY, 10)
    assert _path(_stats(calmar=1.0), 0.12, 3) == (FAST, 3)


def test_the_calmar_tiers_do_not_depend_on_their_order():
    with patch.object(ValiConfig, "AUTO_PROMOTION_STEADY_TRACK_CALMAR_MULTIPLIERS", ((1.5, 5), (2.0, 10))):
        assert _path(_stats(calmar=2.5), 0.05, 3) == (STEADY, 10)


# ── The move ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("calmar,weeks,path,multiplier", [
    (0.0, _weeks(4_000, 7_000), FAST, 3),
    (1.5, _weeks(1_000, 1_000, 1_500), STEADY, 5),
    (2.0, _weeks(1_000, 1_000, 1_500), STEADY, 10),
])
def test_a_qualifying_candidate_moves_into_the_transition_at_its_multiple(hop_manager, calmar, weeks, path, multiplier):
    _candidate(hop_manager, stats=_stats(calmar=calmar), weeks=weeks)

    assert _auto_promote(hop_manager) == [HOTKEY]

    state = hop_manager.miner_states[HOTKEY]
    assert state.current_bucket == TRANSITION
    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(HOTKEY, TRANSITION, multiplier * ACCOUNT_SIZE)
    assert state.auto_promotion == AutoPromotion(
        path=path, multiplier=multiplier, pro_account_size=multiplier * ACCOUNT_SIZE, promoted_ms=NOW_MS,
        payout_usd=sum(weeks.values()), paid_weeks=len(weeks), calmar=calmar, daily_consistency=0.5,
        trading_days=60)


@pytest.mark.parametrize("mothership", [True, False])
def test_a_pro_size_already_on_the_record_is_kept(hop_manager, mothership):
    """The mothership's sizing (or the entity's) is already on the record: it is followed, never
    overwritten, and the record states the multiple actually granted."""
    hop_manager._entity_client.is_mothership.return_value = mothership
    hop_manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "account_type": "pro", "pro_account_size": 500_000}
    _candidate(hop_manager, weeks=_weeks(20_000))

    assert _auto_promote(hop_manager) == [HOTKEY]

    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(HOTKEY, TRANSITION, None)
    record = hop_manager.miner_states[HOTKEY].auto_promotion
    assert (record.path, record.multiplier, record.pro_account_size) == (FAST, 5.0, 500_000)


def test_a_validator_other_than_the_mothership_waits_for_its_size(hop_manager):
    """It qualifies on its own, but only moves once the mothership's sizing reaches its record."""
    hop_manager._entity_client.is_mothership.return_value = False
    _candidate(hop_manager, weeks=_weeks(20_000))

    with patch("vali_objects.challenge_period.challengeperiod_manager.logger") as log:
        assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()
    log.error.assert_not_called()
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None

    hop_manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": None, "account_type": "pro", "pro_account_size": 300_000}
    assert _auto_promote(hop_manager, now_ms=NOW_MS + 30_000) == [HOTKEY]
    hop_manager._entity_client.apply_bucket_account_size.assert_any_call(HOTKEY, TRANSITION, None)
    assert hop_manager.miner_states[HOTKEY].auto_promotion.pro_account_size == 300_000


def test_a_candidate_that_does_not_qualify_stays_funded(hop_manager):
    _candidate(hop_manager, stats=_stats(calmar=1.0), weeks=_weeks(1_000, 1_000, 1_500))
    assert _auto_promote(hop_manager) == []
    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None


def test_a_candidate_without_stats_is_not_evaluated(hop_manager):
    """Its payouts alone would qualify, but its ledger cannot replay its drawdown yet."""
    _candidate(hop_manager, weeks=_weeks(20_000))
    hop_manager._auto_promotion_stats.clear()

    assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.get_funded_week_payouts.assert_not_called()


def test_an_account_promoted_automatically_before_is_never_evaluated_again(hop_manager):
    _candidate(hop_manager, weeks=_weeks(20_000))
    hop_manager.miner_states[HOTKEY].auto_promotion = AutoPromotion(
        path=FAST, multiplier=3, pro_account_size=300_000, promoted_ms=NOW_MS - WEEK_MS, payout_usd=11_000,
        paid_weeks=2, calmar=0.0, daily_consistency=1.0, trading_days=20)

    assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.get_funded_week_payouts.assert_not_called()


def test_an_eliminated_account_is_not_evaluated(hop_manager):
    _candidate(hop_manager, weeks=_weeks(20_000))
    hop_manager.miner_states[HOTKEY].add_bucket_entry(MinerBucket.ELIMINATED, MONDAY_MS - DAILY_MS)
    assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.get_funded_week_payouts.assert_not_called()


def test_an_account_this_refresh_is_eliminating_is_not_evaluated(hop_manager):
    """Even when its elimination failed and left it in SUBACCOUNT_FUNDED."""
    _candidate(hop_manager, weeks=_weeks(20_000))
    assert _auto_promote(hop_manager, eliminating=[HOTKEY]) == []
    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED


def test_a_failed_move_leaves_no_record_and_is_retried_but_logged_once_a_week(hop_manager):
    _candidate(hop_manager, weeks=_weeks(20_000))
    hop_manager._entity_client.apply_bucket_account_size.return_value = (False, "rejected")

    with patch("vali_objects.challenge_period.challengeperiod_manager.logger") as log:
        assert _auto_promote(hop_manager) == []
        assert _auto_promote(hop_manager, now_ms=NOW_MS + 30_000) == []
        assert log.error.call_count == 1
        _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + READABLE_MS)
        assert log.error.call_count == 2
    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None

    hop_manager._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
    assert _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + READABLE_MS + 30_000) == [HOTKEY]


def test_a_move_that_lands_before_raising_keeps_its_record(hop_manager):
    _candidate(hop_manager, weeks=_weeks(20_000))
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
    _candidate(hop_manager, weeks=_weeks(20_000))
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
    _candidate(hop_manager, weeks=_weeks(20_000))
    real_promote = hop_manager.promote_subaccount

    def entity_lands_then_ours_raises(*args, **kwargs):
        assert real_promote(HOTKEY, NOW_MS - 1_000, 250_000)[0]  # the entity's request
        raise TimeoutError("apply_bucket_account_size")

    with patch.object(hop_manager, "promote_subaccount", side_effect=entity_lands_then_ours_raises):
        assert _auto_promote(hop_manager) == []
    assert hop_manager.miner_states[HOTKEY].current_bucket == TRANSITION
    assert hop_manager.miner_states[HOTKEY].auto_promotion is None


def test_an_entity_promotion_landing_first_is_left_alone(hop_manager):
    """The entity's own request moves the subaccount into the transition between the decision and
    the move: the auto-promotion does not hop it onward, nor claim the entity's move as its own."""
    _candidate(hop_manager, weeks=_weeks(20_000))
    real_promote = hop_manager.promote_subaccount

    def entity_request_lands_first(*args, **kwargs):
        assert real_promote(HOTKEY, NOW_MS, 250_000)[0]  # the entity's request, no from_bucket
        return real_promote(*args, **kwargs)

    with patch.object(hop_manager, "promote_subaccount", side_effect=entity_request_lands_first):
        assert _auto_promote(hop_manager) == []

    state = hop_manager.miner_states[HOTKEY]
    assert state.current_bucket == TRANSITION
    assert state.auto_promotion is None
    hop_manager._entity_client.apply_bucket_account_size.assert_called_once_with(HOTKEY, TRANSITION, 250_000)


def test_the_move_is_only_ever_the_hop_out_of_funded(hop_manager):
    hop_manager.set_miner_bucket(HOTKEY, TRANSITION, NOW_MS)

    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS, 300_000,
                                                      from_bucket=MinerBucket.SUBACCOUNT_FUNDED)

    assert not success and "no longer" in message
    assert hop_manager.miner_states[HOTKEY].current_bucket == TRANSITION
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()


# ── The record ────────────────────────────────────────────────────────────────

def test_the_record_is_saved_with_the_state():
    record = AutoPromotion(path=STEADY, multiplier=10, pro_account_size=1_000_000, promoted_ms=NOW_MS,
                           payout_usd=8_000, paid_weeks=6, calmar=2.3, daily_consistency=0.3, trading_days=75)
    state = MinerBucketState.from_checkpoint_dict(HOTKEY, {"entries": [{"bucket": TRANSITION.value,
                                                                        "start_time_ms": NOW_MS}]})
    assert state.auto_promotion is None
    state.auto_promotion = record
    assert MinerBucketState.from_checkpoint_dict(HOTKEY, state.to_checkpoint_dict()).auto_promotion == record


def test_a_record_with_missing_or_unknown_fields_still_loads():
    """Written by another version: missing fields take their defaults, unknown ones are ignored."""
    data = {"entries": [{"bucket": TRANSITION.value, "start_time_ms": NOW_MS}],
            "auto_promotion": {"path": FAST, "multiplier": 3, "a_later_field": 1}}
    record = MinerBucketState.from_checkpoint_dict(HOTKEY, data).auto_promotion
    assert (record.path, record.multiplier, record.pro_account_size) == (FAST, 3, 0.0)


def test_a_sync_that_fails_to_parse_keeps_the_current_states(manager):
    _seed(manager)
    with pytest.raises(Exception):
        manager.sync_challenge_period_data({"other": {"entries": [{"bucket": "NOT_A_BUCKET"}]}})
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED


# ── Payouts: closed weeks, once a week, within a time budget ──────────────────

def test_payouts_are_read_up_to_this_weeks_monday_and_cached_for_the_week(hop_manager):
    _candidate(hop_manager, stats=_stats(calmar=0.0), weeks=_weeks(1_000))
    lookups = hop_manager._entity_client.get_funded_week_payouts

    _auto_promote(hop_manager)
    _auto_promote(hop_manager, now_ms=NOW_MS + DAILY_MS)
    lookups.assert_called_once_with(HOTKEY, MONDAY_MS)

    _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + READABLE_MS)
    assert lookups.call_count == 2
    lookups.assert_called_with(HOTKEY, NEXT_MONDAY_MS)


def test_a_new_weeks_payouts_wait_for_its_monday_snapshots(hop_manager):
    _candidate(hop_manager, weeks=_weeks(20_000))
    lookups = hop_manager._entity_client.get_funded_week_payouts

    assert _auto_promote(hop_manager, now_ms=MONDAY_MS + READABLE_MS - 1) == []
    lookups.assert_not_called()
    assert _auto_promote(hop_manager, now_ms=MONDAY_MS + READABLE_MS) == [HOTKEY]


def test_payouts_are_read_again_when_the_account_restarts(hop_manager):
    """A reset account starts a new ledger, whose payouts start over."""
    _candidate(hop_manager, stats=_stats(calmar=0.0), weeks=_weeks(1_000))
    _auto_promote(hop_manager)

    hop_manager._auto_promotion_stats[HOTKEY] = _stats(calmar=0.0, ledger_start_ms=MONDAY_MS - DAILY_MS)
    _auto_promote(hop_manager, now_ms=NOW_MS + READABLE_MS)

    assert hop_manager._entity_client.get_funded_week_payouts.call_count == 2


def test_an_account_with_no_closed_payout_week_is_not_evaluated(hop_manager):
    _candidate(hop_manager, stats=_stats(ledger_start_ms=MONDAY_MS + 1), weeks=_weeks(20_000))

    assert _auto_promote(hop_manager) == []
    hop_manager._entity_client.get_funded_week_payouts.assert_not_called()

    # Once its first week closes, it is
    assert _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + READABLE_MS) == [HOTKEY]


def _fake_clock(manager, seconds_per_lookup: float, weeks_for=None):
    """time.monotonic for the manager, advanced only by payout lookups. weeks_for(hotkey) gives each
    lookup's result; by default the lookup's current return value."""
    clock = MagicMock()
    clock.monotonic.return_value = 0.0
    default_weeks = manager._entity_client.get_funded_week_payouts.return_value

    def lookup(hotkey, _end_time_ms):
        clock.monotonic.return_value += seconds_per_lookup
        return weeks_for(hotkey) if weeks_for else default_weeks

    manager._entity_client.get_funded_week_payouts.side_effect = lookup
    return patch("vali_objects.challenge_period.challengeperiod_manager.time", clock)


def test_payout_lookups_stop_once_the_refresh_has_spent_its_budget(hop_manager):
    hotkeys = [f"hk{i:03d}" for i in range(10)]
    for hotkey in hotkeys:
        _candidate(hop_manager, hotkey, stats=_stats(calmar=0.0), weeks=_weeks(1_000))
    lookups = hop_manager._entity_client.get_funded_week_payouts

    # A 5 s budget at 2 s a lookup: the third starts at 4 s, the fourth would start past it
    with _fake_clock(hop_manager, seconds_per_lookup=2.0):
        _auto_promote(hop_manager, hotkeys)
        assert lookups.call_count == 3
        _auto_promote(hop_manager, hotkeys)
        assert lookups.call_count == 6


def test_a_lookup_that_keeps_failing_never_holds_up_the_rest(hop_manager):
    _candidate(hop_manager, "hk_a", stats=_stats(calmar=0.0))
    _candidate(hop_manager, "hk_b", stats=_stats(calmar=0.0))
    looked_up = []

    def weeks_for(hotkey):
        looked_up.append(hotkey)
        return None if hotkey == "hk_a" else _weeks(1_000)

    # Each lookup outlasts the whole budget, so a refresh makes exactly one; by the second, the failed
    # one may be retried again, but the one never tried goes first
    with _fake_clock(hop_manager, seconds_per_lookup=10.0, weeks_for=weeks_for):
        _auto_promote(hop_manager, ["hk_a", "hk_b"])
        _auto_promote(hop_manager, ["hk_a", "hk_b"], now_ms=NOW_MS + READABLE_MS)
    assert looked_up == ["hk_a", "hk_b"]


@pytest.mark.parametrize("failure", [RuntimeError("entity server down"), None])
def test_a_failed_payout_lookup_is_retried_hourly_and_logged_once_a_week(hop_manager, failure):
    _candidate(hop_manager)
    lookups = hop_manager._entity_client.get_funded_week_payouts
    if failure is None:
        lookups.return_value = None
    else:
        lookups.side_effect = failure

    with patch("vali_objects.challenge_period.challengeperiod_manager.logger") as log:
        assert _auto_promote(hop_manager) == []
        assert _auto_promote(hop_manager, now_ms=NOW_MS + 30_000) == []    # too soon to retry
        assert lookups.call_count == 1
        assert _auto_promote(hop_manager, now_ms=NOW_MS + READABLE_MS) == []
        assert lookups.call_count == 2
        assert log.warning.call_count == 1                                 # same payout week
        _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + READABLE_MS)
        assert log.warning.call_count == 2
    assert lookups.call_count == 3

    lookups.side_effect = None
    lookups.return_value = _weeks(20_000)
    assert _auto_promote(hop_manager, now_ms=NEXT_MONDAY_MS + 2 * READABLE_MS) == [HOTKEY]


def test_a_week_counts_as_paid_only_when_it_paid_something(hop_manager):
    """Three weeks traded, one of them paying nothing: two paid weeks, short of the steady track."""
    _candidate(hop_manager, stats=_stats(calmar=2.5), weeks=_weeks(2_000, 0.0, 2_000))
    assert _auto_promote(hop_manager) == []


# ── In the transition: seven days, no fee, no entity hop, one promotion at a time ──

GROW = MinerBucket.PRO_CHALLENGE_FROM_STANDARD
_expired = ChallengePeriodManager._check_transition_expiry


def _in_transition(manager, entered_ms: int = NOW_MS, auto_promoted: bool = True,
                   hotkey: str = HOTKEY) -> MinerBucketState:
    manager.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - 60 * DAILY_MS)
    state = manager.miner_states[hotkey]
    state.add_bucket_entry(TRANSITION, entered_ms)
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
    assert hop_manager.promote_hotkeys([subaccount], NOW_MS + SEVEN_DAYS_MS)
    assert hop_manager.miner_states[subaccount].current_bucket == GROW
    hop_manager._entity_client.apply_bucket_account_size.assert_called_with(
        subaccount, GROW, waive_promotion_fee=auto_promoted)


@pytest.mark.parametrize("waive", [True, False])
def test_the_entity_manager_waives_the_fee_and_its_collateral_check_on_request(waive):
    """A fee-paying subaccount whose entity holds no collateral at all."""
    entity_manager = _bare_manager()
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.0
    hotkey = _add_standard(entity_manager)
    entity_manager.get_subaccount_info_for_synthetic(hotkey).reg_fee_theta = STANDARD_SIZE / 1_500
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 300_000)[0]

    success, message = entity_manager.apply_bucket_account_size(hotkey, GROW, waive_promotion_fee=waive)

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
    client.apply_bucket_account_size(HOTKEY, GROW, waive_promotion_fee=waive)
    server._manager.apply_bucket_account_size.assert_called_once_with(HOTKEY, GROW, None, waive_promotion_fee=waive)


def test_the_entity_cannot_promote_an_auto_promoted_account_out_of_the_transition(hop_manager):
    _in_transition(hop_manager)
    success, message = hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS, 500_000)
    assert not success and "promoted automatically" in message
    assert hop_manager.miner_states[HOTKEY].current_bucket == TRANSITION
    hop_manager._entity_client.apply_bucket_account_size.assert_not_called()


def test_the_entity_can_still_move_its_own_promotion_on(hop_manager):
    _in_transition(hop_manager, auto_promoted=False)
    assert hop_manager.promote_subaccount(HOTKEY, NOW_MS + DAILY_MS)[0]
    assert hop_manager.miner_states[HOTKEY].current_bucket == GROW


def _runs_only_after_the_lock_is_released(manager, call) -> None:
    """Hold the subaccount's promotion lock, start `call` on another thread, and check it waits for it."""
    done = threading.Event()
    worker = threading.Thread(target=lambda: (call(), done.set()))
    with manager._promotion_lock(HOTKEY):
        worker.start()
        assert not done.wait(0.2)
        state = manager.miner_states[HOTKEY]
        assert state.current_bucket == MinerBucket.SUBACCOUNT_FUNDED and state.auto_promotion is None
    worker.join(5)
    assert done.is_set()


def test_an_entity_promotion_waits_for_the_subaccounts_promotion_lock(hop_manager):
    hop_manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - DAILY_MS)
    _runs_only_after_the_lock_is_released(hop_manager, lambda: hop_manager.promote_subaccount(HOTKEY, NOW_MS, 250_000))
    assert hop_manager.miner_states[HOTKEY].current_bucket == TRANSITION


def test_an_auto_promotion_waits_for_the_subaccounts_promotion_lock(hop_manager):
    _candidate(hop_manager, weeks=_weeks(20_000))
    _runs_only_after_the_lock_is_released(hop_manager, lambda: _auto_promote(hop_manager))
    assert hop_manager.miner_states[HOTKEY].auto_promotion.multiplier == 3


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
    assert hop_manager.miner_states[HOTKEY].current_bucket == TRANSITION


@pytest.mark.parametrize("since_the_decision", ["promoted by the entity", "already auto-promoted"])
def test_the_move_checks_the_account_again_before_promoting_it(hop_manager, since_the_decision):
    """Nothing is moved or recorded when the account changed after the decision."""
    _candidate(hop_manager, weeks=_weeks(20_000))
    state = hop_manager.miner_states[HOTKEY]
    if since_the_decision == "promoted by the entity":
        state.add_bucket_entry(TRANSITION, NOW_MS - 1_000)
    else:
        state.auto_promotion = AutoPromotion(path=FAST, multiplier=3.0, pro_account_size=300_000,
                                             promoted_ms=NOW_MS - SEVEN_DAYS_MS)
    record = state.auto_promotion

    payouts = FundedPayouts(week_start_ms=MONDAY_MS, ledger_start_ms=LEDGER_START_MS, total_usd=20_000, paid_weeks=1)
    with patch.object(hop_manager, "promote_subaccount") as promote:
        assert not hop_manager._move_to_auto_promotion(HOTKEY, _account(), _stats(), (FAST, 3), payouts, NOW_MS)
    promote.assert_not_called()
    assert state.auto_promotion is record


# ── EntityManager.get_funded_week_payouts ─────────────────────────────────────

def _row(start_ms: int, end_ms: int, payout: float, bucket: MinerBucket = MinerBucket.SUBACCOUNT_FUNDED) -> dict:
    return {"start_ms": start_ms, "end_ms": end_ms, "payout": payout, "bucket": bucket.value}


def test_funded_week_payouts_sum_each_weeks_funded_rows():
    entity_manager = _bare_manager()
    hotkey = _add_standard(entity_manager)
    week_1, week_2 = MONDAY_MS - 2 * WEEK_MS, MONDAY_MS - WEEK_MS
    rows = [
        _row(week_1, week_1 + 2 * DAILY_MS, 50.0, MinerBucket.SUBACCOUNT_CHALLENGE),  # before funding
        _row(week_1 + 2 * DAILY_MS, week_2, 300.0),
        _row(week_2, week_2 + 3 * DAILY_MS, 100.0),
        _row(week_2 + 3 * DAILY_MS, MONDAY_MS, 0.0),
        _row(MONDAY_MS, MONDAY_MS + DAILY_MS, 900.0),                                  # the running week
    ]
    with patch.object(entity_manager, "_calculate_payout_for_hotkey",
                      return_value={"weekly_settlements": rows}) as calculate:
        weeks = entity_manager.get_funded_week_payouts(hotkey, MONDAY_MS)

    calculate.assert_called_once_with(hotkey, 0, MONDAY_MS, bucket_override=MinerBucket.SUBACCOUNT_FUNDED)
    assert weeks == {week_1: 300.0, week_2: 100.0}


def test_funded_week_payouts_pass_through_a_failed_calculation():
    entity_manager = _bare_manager()
    with patch.object(entity_manager, "_calculate_payout_for_hotkey", return_value=None):
        assert entity_manager.get_funded_week_payouts(_add_standard(entity_manager), MONDAY_MS) is None


@pytest.mark.parametrize("bucket", [None, MinerBucket.UNKNOWN.value])
def test_funded_week_payouts_are_unknown_until_the_debt_ledger_labels_the_weeks(bucket):
    """With no debt ledger yet, rows carry no bucket: that is not a week paying nothing."""
    entity_manager = _bare_manager()
    rows = [{"start_ms": MONDAY_MS - WEEK_MS, "end_ms": MONDAY_MS, "payout": 900.0, "bucket": bucket}]
    with patch.object(entity_manager, "_calculate_payout_for_hotkey", return_value={"weekly_settlements": rows}):
        assert entity_manager.get_funded_week_payouts(_add_standard(entity_manager), MONDAY_MS) is None


# ── Through refresh(), and on to Grow ─────────────────────────────────────────

class _EntityClient:
    """A real EntityManager behind the client's interface: records cross RPC as dicts."""

    def __init__(self, entity_manager, weeks: dict):
        self._manager = entity_manager
        self._weeks = weeks

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def get_subaccount_info_for_synthetic(self, hotkey):
        info = self._manager.get_subaccount_info_for_synthetic(hotkey)
        return info.model_dump() if info else None

    def get_funded_week_payouts(self, hotkey, end_time_ms):
        return self._weeks

    def is_mothership(self):
        return True


def _full_refresh(manager, hotkey: str, now_ms: int, ledger=None, eliminated: bool = False):
    """One refresh() pass with the drawdown cache pinned, so the stats, auto-promotion and routing run for real."""
    manager._position_client.get_all_hotkeys.return_value = [hotkey]
    manager._position_client.filtered_positions_for_scoring.return_value = ({hotkey: []}, {})
    manager._position_client.get_positions_for_hotkeys.return_value = {hotkey: []}
    manager._elimination_client.get_eliminated_hotkeys.return_value = []
    manager._plagiarism_client.get_plagiarism_miners.return_value = []
    manager._miner_account_client.get_accounts.return_value = {hotkey: _account(hotkey)}
    manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = {hotkey: ledger or _ledger(low=0.98)}
    manager._asset_selection_client.get_asset_selections.return_value = {hotkey: MinerAssetClass.FOREX}
    with contextlib.ExitStack() as stack:
        for name in ("_refresh_drawdown_cache", "_refresh_rank_cache", "_save_to_disk", "_sync_buckets_to_accounts"):
            stack.enter_context(patch.object(manager, name))
        return manager.refresh(current_time_ms=now_ms)


def test_through_refresh_a_qualifying_account_reaches_grow_seven_days_later_free_of_the_fee(hop_manager):
    """7% realized over a 2% drawdown is calmar 3.5: the 10x tier. The transition records the pro size
    and keeps trading the standard account, past the next Monday; seven days after the promotion it
    goes onto the pro size, with no promotion fee charged even though the entity holds no collateral."""
    entity_manager = _bare_manager()
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.0
    hotkey = _add_standard(entity_manager)
    entity_manager.get_subaccount_info_for_synthetic(hotkey).reg_fee_theta = STANDARD_SIZE / 1_500
    hop_manager._entity_client = _EntityClient(entity_manager, _weeks(1_000, 1_000, 1_500))
    _seed(hop_manager, hotkey)

    message = _full_refresh(hop_manager, hotkey, NOW_MS)

    state = hop_manager.miner_states[hotkey]
    assert state.current_bucket == TRANSITION
    assert state.auto_promotion.path == STEADY and state.auto_promotion.multiplier == 10
    assert hotkey in message
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 10 * STANDARD_SIZE
    assert info.account_size == STANDARD_SIZE

    _full_refresh(hop_manager, hotkey, NEXT_MONDAY_MS + 60_000)
    assert hop_manager.miner_states[hotkey].current_bucket == TRANSITION

    _full_refresh(hop_manager, hotkey, NOW_MS + SEVEN_DAYS_MS)

    assert hop_manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_size == 10 * STANDARD_SIZE
    assert info.pro_fee_theta == 0
    entity_manager._entity_collateral_client.offset_collateral_cache.assert_not_called()


def test_through_refresh_a_ledger_missing_lows_holds_the_promotion(hop_manager):
    """Payouts alone would qualify, but a checkpoint's low is unknown until the ledger is regenerated."""
    hop_manager._entity_client.get_funded_week_payouts.return_value = _weeks(20_000)
    _seed(hop_manager)
    ledger = _ledger(low=0.98)
    ledger.cps[0].min_equity_ret = None

    _full_refresh(hop_manager, HOTKEY, NOW_MS, ledger)
    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED

    _full_refresh(hop_manager, HOTKEY, NOW_MS + 60_000)
    assert hop_manager.miner_states[HOTKEY].current_bucket == TRANSITION


def test_through_refresh_an_account_whose_elimination_failed_is_not_promoted(hop_manager):
    """An intraday breach whose elimination row could not be written leaves the account in its bucket."""
    hop_manager._entity_client.get_funded_week_payouts.return_value = _weeks(20_000)
    _seed(hop_manager, current_equity=0.90)
    hop_manager.miner_states[HOTKEY].drawdown.daily_open_equity = 1.0
    hop_manager._elimination_client.append_elimination_row.side_effect = TimeoutError("elimination RPC")

    _full_refresh(hop_manager, HOTKEY, NOW_MS)

    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED


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
    """The stats run before the eliminations: if they fail, the breach is still eliminated and nothing
    is evaluated for auto-promotion on the stale stats."""
    hop_manager._entity_client.get_funded_week_payouts.return_value = _weeks(20_000)
    _seed(hop_manager, current_equity=0.90)
    hop_manager.miner_states[HOTKEY].drawdown.daily_open_equity = 1.0
    hop_manager._auto_promotion_stats[HOTKEY] = _stats()

    with patch.object(hop_manager, "_refresh_auto_promotion_stats", side_effect=RuntimeError("unexpected")):
        _full_refresh(hop_manager, HOTKEY, NOW_MS)

    hop_manager._elimination_client.append_elimination_row.assert_called_once()
    assert hop_manager._auto_promotion_stats == {}


@pytest.mark.parametrize("record", [{"initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value}, None])
def test_through_refresh_a_non_candidate_is_never_promoted(hop_manager, record):
    hop_manager._entity_client.get_subaccount_info_for_synthetic.return_value = record
    hop_manager._entity_client.get_funded_week_payouts.return_value = _weeks(20_000)
    _seed(hop_manager)

    _full_refresh(hop_manager, HOTKEY, NOW_MS)

    assert hop_manager.miner_states[HOTKEY].current_bucket == MinerBucket.SUBACCOUNT_FUNDED
    hop_manager._entity_client.get_funded_week_payouts.assert_not_called()
