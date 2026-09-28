"""
Focused unit tests for ChallengePeriodManager and related dataclasses, on the standard track.
The pro account buckets and their promotion criteria live in test_challengeperiod_pro.py.
No RPC connections, no disk I/O, no daemon simulation.
Each test uses a manager fixture with all RPC clients mocked and
is_backtesting=True to skip all file system access.
"""
import contextlib
import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from vali_objects.challenge_period.challengeperiod_manager import (
    ChallengePeriodManager,
    DrawdownStats,
    MinerBucketState,
)
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vanta_api.validator_rest_server import ValidatorRestServer

# ── Constants ─────────────────────────────────────────────────────────────────

DAILY_MS = ValiConfig.DAILY_MS
MIN_CHALLENGE_MS = ValiConfig.CHALLENGE_PERIOD_MINIMUM_MS    # 61 days
MAX_CHALLENGE_MS = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS    # 90 days
THRESHOLD = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT  # 0.1
RANK_LIMIT = ValiConfig.PROMOTION_THRESHOLD_RANK             # 25
INTRADAY_THRESHOLD_PCT = ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD * 100  # 5.0

NOW_MS = 1_748_000_000_000  # fixed reference timestamp (ms)

_CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager.PerfLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.PositionManagerClient",
    "vali_objects.challenge_period.challengeperiod_manager.EliminationClient",
    "vali_objects.challenge_period.challengeperiod_manager.PlagiarismClient",
    "vali_objects.challenge_period.challengeperiod_manager.MinerAccountClient",
    "vali_objects.challenge_period.challengeperiod_manager.CommonDataClient",
    "vali_objects.challenge_period.challengeperiod_manager.AssetSelectionClient",
    "vali_objects.challenge_period.challengeperiod_manager.DebtLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.LimitOrderClient",
    "vali_objects.challenge_period.challengeperiod_manager.EntityClient",
]


# ── Fixtures & helpers ────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    with contextlib.ExitStack() as stack:
        for path in _CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        yield mgr


def _state(bucket: MinerBucket, start_ms: int = NOW_MS) -> MinerBucketState:
    return MinerBucketState("test_hk", [BucketEntry(bucket, start_ms)])


def _setup_refresh_clients(manager, hk: str):
    """Minimal client stubs so refresh() doesn't crash."""
    manager._position_client.get_all_hotkeys.return_value = [hk]
    manager._position_client.filtered_positions_for_scoring.return_value = ({hk: []}, {})
    manager._elimination_client.get_eliminated_hotkeys.return_value = []
    manager._plagiarism_client.get_plagiarism_miners.return_value = []
    manager._miner_account_client.get_accounts.return_value = {}
    manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = {}
    manager._asset_selection_client.get_asset_selections.return_value = {
        hk: TradePairCategory.CRYPTO
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — MinerBucketState
# ═══════════════════════════════════════════════════════════════════════════════

def test_bucket_state_sorted_on_init():
    later = BucketEntry(MinerBucket.MAINCOMP, NOW_MS + 1000)
    earlier = BucketEntry(MinerBucket.CHALLENGE, NOW_MS)
    state = MinerBucketState("test_hk", [later, earlier])
    assert state.entries[0].start_time_ms == NOW_MS
    assert state.entries[1].start_time_ms == NOW_MS + 1000


def test_bucket_state_empty_raises():
    with pytest.raises(ValueError):
        MinerBucketState("test_hk", [])


def test_add_bucket_entry_same_bucket_noop():
    state = _state(MinerBucket.CHALLENGE)
    result = state.add_bucket_entry(MinerBucket.CHALLENGE, NOW_MS + 1)
    assert result is False
    assert len(state.entries) == 1


def test_add_bucket_entry_different_bucket():
    state = _state(MinerBucket.CHALLENGE)
    result = state.add_bucket_entry(MinerBucket.MAINCOMP, NOW_MS + 1)
    assert result is True
    assert len(state.entries) == 2
    assert state.current_bucket == MinerBucket.MAINCOMP


def test_add_bucket_entry_replace_top():
    state = _state(MinerBucket.CHALLENGE, NOW_MS)
    result = state.add_bucket_entry(MinerBucket.CHALLENGE, NOW_MS + 5000, replace_top=True)
    assert result is True
    assert len(state.entries) == 1
    assert state.entries[-1].start_time_ms == NOW_MS + 5000


def test_pop_bucket_entry_matching():
    state = _state(MinerBucket.PLAGIARISM)
    popped = state.pop_bucket_entry(MinerBucket.PLAGIARISM)
    assert isinstance(popped, BucketEntry)
    assert popped.bucket == MinerBucket.PLAGIARISM


def test_pop_bucket_entry_no_match():
    state = _state(MinerBucket.CHALLENGE)
    result = state.pop_bucket_entry(MinerBucket.PLAGIARISM)
    assert result is None
    assert len(state.entries) == 1


def test_to_json_excludes_drawdown():
    state = _state(MinerBucket.CHALLENGE)
    state.drawdown = DrawdownStats(current_equity=1.5)
    state.rank = 3
    json_list = state.to_checkpoint_dict()["entries"]
    assert isinstance(json_list, list)
    assert len(json_list) == 1
    assert "bucket" in json_list[0]
    assert "current_equity" not in json_list[0]
    assert "rank" not in json_list[0]


def test_current_bucket_start_ms():
    state = _state(MinerBucket.CHALLENGE, NOW_MS)
    state.add_bucket_entry(MinerBucket.MAINCOMP, NOW_MS + 100)
    assert state.current_bucket_start_ms == NOW_MS + 100


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — _should_promote / _should_demote
# ═══════════════════════════════════════════════════════════════════════════════

def test_should_promote_challenge_too_early():
    start_ms = NOW_MS - MIN_CHALLENGE_MS + DAILY_MS  # one day short
    state = _state(MinerBucket.CHALLENGE, start_ms)
    state.drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    state.rank = 1
    assert ChallengePeriodManager._check_promotion(state, THRESHOLD, NOW_MS) is False


def test_should_promote_challenge_met_threshold():
    start_ms = NOW_MS - MIN_CHALLENGE_MS - DAILY_MS  # past minimum
    state = _state(MinerBucket.CHALLENGE, start_ms)
    # current_returns = equity - 1.0 = THRESHOLD + 0.01 > THRESHOLD → promote
    state.drawdown = DrawdownStats(current_equity=1.0 + THRESHOLD + 0.01, current_balance=1.0 + THRESHOLD + 0.01)
    state.rank = 1
    assert ChallengePeriodManager._check_promotion(state, THRESHOLD, NOW_MS) is True


def test_should_promote_challenge_below_equity_threshold():
    start_ms = NOW_MS - MIN_CHALLENGE_MS - DAILY_MS
    state = _state(MinerBucket.CHALLENGE, start_ms)
    # current_returns = THRESHOLD - 0.01 < THRESHOLD → no promote
    state.drawdown = DrawdownStats(current_equity=1.0 + THRESHOLD - 0.01, current_balance=1.0 + THRESHOLD - 0.01)
    state.rank = 1
    assert ChallengePeriodManager._check_promotion(state, THRESHOLD, NOW_MS) is False


def test_should_promote_rank_based_good_rank():
    state = _state(MinerBucket.PROBATION)  # rank-based, promotes to MAINCOMP
    # current_returns = 0.5 > THRESHOLD → equity condition met
    state.drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    state.rank = RANK_LIMIT  # at the boundary (≤ 25 passes)
    assert ChallengePeriodManager._check_promotion(state, THRESHOLD, NOW_MS) is True


def test_should_promote_rank_based_bad_rank():
    state = _state(MinerBucket.PROBATION)
    state.drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    state.rank = RANK_LIMIT + 1
    assert ChallengePeriodManager._check_promotion(state, THRESHOLD, NOW_MS) is False


def test_should_promote_rank_based_no_rank():
    state = _state(MinerBucket.PROBATION)
    state.drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    state.rank = None
    assert ChallengePeriodManager._check_promotion(state, THRESHOLD, NOW_MS) is False


def test_should_demote_maincomp_bad_rank():
    state = _state(MinerBucket.MAINCOMP)
    state.rank = RANK_LIMIT + 1
    state.drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    assert ChallengePeriodManager._check_demotion(state) is True


def test_should_demote_maincomp_good_rank():
    state = _state(MinerBucket.MAINCOMP)
    state.rank = RANK_LIMIT  # at boundary (not > 25) → no demotion
    state.drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    assert ChallengePeriodManager._check_demotion(state) is False


def test_should_demote_maincomp_good_rank_low_equity():
    state = _state(MinerBucket.MAINCOMP)
    state.rank = 1  # good rank → demotion is rank-only, equity does not trigger it
    state.drawdown = DrawdownStats(current_equity=1.0 + THRESHOLD - 0.01)
    assert ChallengePeriodManager._check_demotion(state) is False


def test_should_demote_maincomp_no_rank():
    state = _state(MinerBucket.MAINCOMP)
    state.rank = None
    assert ChallengePeriodManager._check_demotion(state) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3b — _check_static_drawdown (subaccounts)
# ═══════════════════════════════════════════════════════════════════════════════

STATIC_DD_PCT = ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD * 100      # 5.0


def test_check_static_drawdown_below_threshold_survives():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(current_equity=1 - (STATIC_DD_PCT - 0.01) / 100)
    assert ChallengePeriodManager._check_static_drawdown(state) is None


def test_check_static_drawdown_funded_reason():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(current_equity=1 - (STATIC_DD_PCT + 0.01) / 100)
    assert ChallengePeriodManager._check_static_drawdown(state) == EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN


def test_check_static_drawdown_challenge_reason():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE)
    state.drawdown = DrawdownStats(current_equity=1 - (STATIC_DD_PCT + 0.01) / 100)
    assert ChallengePeriodManager._check_static_drawdown(state) == EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_DRAWDOWN


def test_check_static_ignores_trailing_drawdown_pcts():
    # Trailing drawdowns (12% intraday, 9% EOD) while equity is still above the
    # starting balance — the static check is absolute, so it does not fire.
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(
        current_equity=1.1,
        daily_open_equity=1.25,
        last_eod_equity=1.092,
        eod_hwm=1.2,
    )
    assert state.drawdown.intraday_drawdown_pct == pytest.approx(12.0)
    assert state.drawdown.eod_drawdown_pct == pytest.approx(9.0)
    assert ChallengePeriodManager._check_static_drawdown(state) is None


def test_should_demote_non_maincomp():
    for bucket in (MinerBucket.CHALLENGE, MinerBucket.PROBATION):
        state = _state(bucket)
        state.rank = RANK_LIMIT + 10
        assert ChallengePeriodManager._check_demotion(state) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — set_miner_bucket / remove_miners
# ═══════════════════════════════════════════════════════════════════════════════

def test_set_miner_bucket_new(manager):
    result = manager.set_miner_bucket("hk1", MinerBucket.CHALLENGE, NOW_MS)
    assert result is True
    assert "hk1" in manager.miner_states
    assert manager.miner_states["hk1"].current_bucket == MinerBucket.CHALLENGE


def test_set_miner_bucket_existing(manager):
    manager.set_miner_bucket("hk1", MinerBucket.CHALLENGE, NOW_MS)
    result = manager.set_miner_bucket("hk1", MinerBucket.MAINCOMP, NOW_MS + 1)
    assert result is False
    assert manager.miner_states["hk1"].current_bucket == MinerBucket.MAINCOMP
    assert len(manager.miner_states["hk1"].entries) == 2


def test_set_miner_bucket_replace_top(manager):
    manager.set_miner_bucket("hk1", MinerBucket.CHALLENGE, NOW_MS)
    manager.set_miner_bucket("hk1", MinerBucket.CHALLENGE, NOW_MS + 5000, replace_top=True)
    assert len(manager.miner_states["hk1"].entries) == 1
    assert manager.miner_states["hk1"].current_bucket_start_ms == NOW_MS + 5000


def test_remove_miners_single_string(manager):
    manager.miner_states["hk1"] = _state(MinerBucket.CHALLENGE)
    result = manager.remove_miners("hk1")
    assert result is True
    assert "hk1" not in manager.miner_states


def test_remove_miners_list(manager):
    manager.miner_states["hk1"] = _state(MinerBucket.CHALLENGE)
    manager.miner_states["hk2"] = _state(MinerBucket.MAINCOMP)
    result = manager.remove_miners(["hk1", "hk2"])
    assert result is True
    assert "hk1" not in manager.miner_states
    assert "hk2" not in manager.miner_states


def test_remove_miners_idempotent(manager):
    result = manager.remove_miners(["nonexistent"])
    assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — refresh (mocked clients)
# ═══════════════════════════════════════════════════════════════════════════════

def test_refresh_promotes_eligible_challenge_miner(manager):
    hk = "hk_promote"
    manager.miner_states[hk] = _state(MinerBucket.CHALLENGE, NOW_MS - MIN_CHALLENGE_MS - DAILY_MS)
    manager.miner_states[hk].drawdown = DrawdownStats(
        current_equity=1.5, current_balance=1.5  # current_returns = 0.5 > THRESHOLD
    )
    manager.miner_states[hk].rank = 1
    _setup_refresh_clients(manager, hk)

    with (
        patch.object(manager, '_refresh_drawdown_cache'),
        patch.object(manager, '_refresh_rank_cache'),
        patch.object(manager, '_save_to_disk'),
        patch.object(manager, '_sync_buckets_to_accounts'),
    ):
        manager.refresh(current_time_ms=NOW_MS)

    assert manager.miner_states[hk].current_bucket == MinerBucket.MAINCOMP


def test_refresh_demotes_maincomp_miner(manager):
    hk = "hk_demote"
    manager.miner_states[hk] = _state(MinerBucket.MAINCOMP, NOW_MS - DAILY_MS)
    manager.miner_states[hk].drawdown = DrawdownStats(current_equity=1.5, current_balance=1.5)
    manager.miner_states[hk].rank = RANK_LIMIT + 1
    _setup_refresh_clients(manager, hk)

    with (
        patch.object(manager, '_refresh_drawdown_cache'),
        patch.object(manager, '_refresh_rank_cache'),
        patch.object(manager, '_save_to_disk'),
        patch.object(manager, '_sync_buckets_to_accounts'),
    ):
        manager.refresh(current_time_ms=NOW_MS)

    assert manager.miner_states[hk].current_bucket == MinerBucket.PROBATION


def test_refresh_eliminates_time_expired(manager):
    hk = "hk_expired"
    manager.miner_states[hk] = _state(MinerBucket.CHALLENGE, NOW_MS - MAX_CHALLENGE_MS - DAILY_MS)
    _setup_refresh_clients(manager, hk)

    with (
        patch.object(manager, '_refresh_drawdown_cache'),
        patch.object(manager, '_refresh_rank_cache'),
        patch.object(manager, '_save_to_disk'),
        patch.object(manager, '_sync_buckets_to_accounts'),
    ):
        manager.refresh(current_time_ms=NOW_MS)

    assert manager.miner_states[hk].current_bucket == MinerBucket.ELIMINATED


def test_refresh_eliminates_intraday_drawdown(manager):
    hk = "hk_drawdown"
    after_activation_ms = ChallengePeriodManager.DRAWDOWN_ACTIVATION_MS + DAILY_MS
    manager.miner_states[hk] = _state(MinerBucket.CHALLENGE, after_activation_ms - DAILY_MS)
    manager.miner_states[hk].drawdown = DrawdownStats(
        daily_open_equity=1 / (1 - (INTRADAY_THRESHOLD_PCT + 1.0) / 100)
    )
    _setup_refresh_clients(manager, hk)

    with (
        patch.object(manager, '_refresh_drawdown_cache'),
        patch.object(manager, '_refresh_rank_cache'),
        patch.object(manager, '_save_to_disk'),
        patch.object(manager, '_sync_buckets_to_accounts'),
    ):
        manager.refresh(current_time_ms=after_activation_ms)

    assert manager.miner_states[hk].current_bucket == MinerBucket.ELIMINATED


def test_refresh_no_changes_still_saves(manager):
    # CHALLENGE miner only 1 day old — too early for promotion, no drawdown issues.
    # refresh() always persists (equity snapshots move even when buckets don't),
    # but skips the account sync when no bucket changed.
    hk = "hk_stable"
    manager.miner_states[hk] = _state(MinerBucket.CHALLENGE, NOW_MS - DAILY_MS)
    manager.miner_states[hk].drawdown = DrawdownStats()
    manager.miner_states[hk].rank = 1
    _setup_refresh_clients(manager, hk)

    with (
        patch.object(manager, '_refresh_drawdown_cache'),
        patch.object(manager, '_refresh_rank_cache'),
        patch.object(manager, '_save_to_disk') as mock_save,
        patch.object(manager, '_sync_buckets_to_accounts') as mock_sync,
    ):
        manager.refresh(current_time_ms=NOW_MS)

    mock_save.assert_called_once()
    mock_sync.assert_not_called()
    assert manager.miner_states[hk].current_bucket == MinerBucket.CHALLENGE


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — sync_elimination_miners / _prune_hotkeys_no_positions
# ═══════════════════════════════════════════════════════════════════════════════

def test_sync_elimination_miners_marks_eliminated(manager):
    manager.miner_states["hk1"] = _state(MinerBucket.CHALLENGE)
    manager.miner_states["hk2"] = _state(MinerBucket.MAINCOMP)
    result = manager.sync_elimination_miners(["hk1"], NOW_MS)
    assert result is True
    assert manager.miner_states["hk1"].current_bucket == MinerBucket.ELIMINATED
    assert manager.miner_states["hk1"].current_bucket_start_ms == NOW_MS
    assert manager.miner_states["hk2"].current_bucket == MinerBucket.MAINCOMP


def test_sync_elimination_miners_empty(manager):
    manager.miner_states["hk1"] = _state(MinerBucket.CHALLENGE)
    result = manager.sync_elimination_miners([], NOW_MS)
    assert result is False
    assert "hk1" in manager.miner_states


def test_prune_skips_entity_bucket(manager):
    hk = "entity_hk"
    manager.miner_states[hk] = _state(MinerBucket.ENTITY)
    manager._position_client.get_all_hotkeys.return_value = []
    state_changed = manager._prune_hotkeys_no_positions()
    assert hk in manager.miner_states
    assert state_changed is False


def test_prune_skips_subaccount_funded(manager):
    hk = "funded_hk"
    manager.miner_states[hk] = _state(MinerBucket.SUBACCOUNT_FUNDED)
    manager._position_client.get_all_hotkeys.return_value = []
    state_changed = manager._prune_hotkeys_no_positions()
    assert hk in manager.miner_states
    assert state_changed is False


def test_prune_removes_missing_regular(manager):
    """CHALLENGE miners absent from position hotkeys should be removed."""
    hk = "regular_hk"
    manager.miner_states[hk] = _state(MinerBucket.CHALLENGE)
    manager._position_client.get_all_hotkeys.return_value = []
    manager._prune_hotkeys_no_positions()
    assert hk not in manager.miner_states


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — set_miner_drawdown_stats
# ═══════════════════════════════════════════════════════════════════════════════

TODAY_MS = NOW_MS - NOW_MS % DAILY_MS


def _dd_manager(manager, bucket=MinerBucket.MAINCOMP, **drawdown):
    hk = "dd_hk"
    manager.miner_states[hk] = _state(bucket)
    manager.miner_states[hk].drawdown = DrawdownStats(**drawdown)
    return hk


def _snapshot_refresh(manager, hk, day_ms, equity_return):
    """One refresh on day_ms with the account's daily open snapshot taken that day."""
    account = MagicMock(account_size=100.0, equity=100.0 * equity_return, balance=100.0 * equity_return)
    account.daily_open_snapshot = MagicMock(snapshot_ms=day_ms, equity_return=equity_return)
    manager._refresh_drawdown_cache([hk], {hk: account}, {}, {hk: [MagicMock()]}, day_ms + 60_000)


def test_set_drawdown_stats_lowered_eod_hwm_sticks_across_snapshot_days(manager):
    hk = _dd_manager(manager, current_equity=0.97, eod_hwm=1.20, last_eod_equity=1.0,
                     daily_open_equity=1.0, last_eod_checked_ms=TODAY_MS)

    success, _, result = manager.set_miner_drawdown_stats(hk, {"eod_hwm": 1.05}, now_ms=NOW_MS)
    dd = manager.miner_states[hk].drawdown
    assert success is True
    assert result["eod_hwm"] == dd.eod_hwm == 1.05
    assert dd.current_equity == 0.97  # live fields untouched

    _snapshot_refresh(manager, hk, TODAY_MS + DAILY_MS, 1.01)
    _snapshot_refresh(manager, hk, TODAY_MS + 2 * DAILY_MS, 1.03)
    assert dd.eod_hwm == 1.05
    assert dd.last_eod_equity == 1.03


def test_set_drawdown_stats_raise_past_threshold_needs_force(manager):
    hk = _dd_manager(manager, current_equity=1.0, eod_hwm=1.0, last_eod_equity=1.0, last_eod_checked_ms=TODAY_MS)

    success, message, _ = manager.set_miner_drawdown_stats(hk, {"eod_hwm": 10.3}, now_ms=NOW_MS)
    assert success is False and "force" in message
    assert manager.miner_states[hk].drawdown.eod_hwm == 1.0

    success, _, _ = manager.set_miner_drawdown_stats(hk, {"eod_hwm": 1.02}, now_ms=NOW_MS)
    assert success is True
    success, _, _ = manager.set_miner_drawdown_stats(hk, {"eod_hwm": 10.3}, force=True, now_ms=NOW_MS)
    assert success is True and manager.miner_states[hk].drawdown.eod_hwm == 10.3


@pytest.mark.parametrize("updates, stored_checked_ms", [
    ({"eod_hwm": 0.99}, TODAY_MS),                                  # below the 1.0 floor
    ({"eod_hwm": 1.01}, TODAY_MS),                                  # below last_eod_equity
    ({"last_eod_checked_ms": TODAY_MS + 1}, TODAY_MS),              # not a day boundary
    ({"last_eod_checked_ms": TODAY_MS + DAILY_MS}, TODAY_MS),       # future day
    ({"daily_open_equity": 1.02}, TODAY_MS - DAILY_MS),             # today's EOD not captured yet
])
def test_set_drawdown_stats_rejected(manager, updates, stored_checked_ms):
    hk = _dd_manager(manager, eod_hwm=1.10, last_eod_equity=1.02, daily_open_equity=1.02,
                     last_eod_checked_ms=stored_checked_ms)
    before = DrawdownStats(**vars(manager.miner_states[hk].drawdown))

    success, _, result = manager.set_miner_drawdown_stats(hk, updates, now_ms=NOW_MS)
    assert success is False and result is None
    assert manager.miner_states[hk].drawdown == before


def test_set_drawdown_stats_unknown_hotkey(manager):
    success, message, _ = manager.set_miner_drawdown_stats("missing_hk", {"eod_hwm": 1.1})
    assert success is False
    assert "not found" in message


# POST /admin/drawdown-stats/<hotkey>: the real Flask handler with a mocked challenge period client

@pytest.fixture
def dd_endpoint():
    server = object.__new__(ValidatorRestServer)
    server._get_api_key_safe = MagicMock(return_value="key")
    server.is_valid_api_key = MagicMock(return_value=True)
    server.can_access_tier = MagicMock(return_value=True)
    server._challenge_period_client = MagicMock()
    set_stats = server._challenge_period_client.set_miner_drawdown_stats
    set_stats.return_value = (True, "ok", {"eod_hwm": 1.05})

    app = Flask(__name__)
    app.route("/admin/drawdown-stats/<hotkey>", methods=["POST"])(server.set_miner_drawdown_stats)
    client = app.test_client()

    def post(raw: str):
        response = client.post("/admin/drawdown-stats/dd_hk", data=raw, content_type="application/json")
        return response.status_code, response.get_json()

    return post, set_stats


def test_drawdown_stats_endpoint_passes_updates_and_force(dd_endpoint):
    post, set_stats = dd_endpoint
    status, payload = post(json.dumps({"eod_hwm": 1.05, "last_eod_checked_ms": 1.728e12, "force": True}))
    assert status == 200
    set_stats.assert_called_once_with("dd_hk", {"eod_hwm": 1.05, "last_eod_checked_ms": 1_728_000_000_000}, True)
    assert payload["drawdown"] == {"eod_hwm": 1.05}


def test_drawdown_stats_endpoint_manager_rejection_is_400(dd_endpoint):
    post, set_stats = dd_endpoint
    set_stats.return_value = (False, "pass force to apply", None)
    status, payload = post(json.dumps({"eod_hwm": 10.3}))
    assert status == 400
    assert "force" in payload["error"]


@pytest.mark.parametrize("raw", [
    '{}',
    '{"force": true}',
    '{"current_equity": 0.5}',
    '{"eod_hwm": "1.1"}',
    '{"eod_hwm": true}',
    '{"eod_hwm": NaN}',
    '{"eod_hwm": -1}',
    '{"eod_hwm": 1' + '0' * 400 + '}',
    '{"last_eod_checked_ms": 1728000000000.5}',
    '{"eod_hwm": 1.05, "force": "yes"}',
])
def test_drawdown_stats_endpoint_rejects_invalid_body(dd_endpoint, raw):
    post, set_stats = dd_endpoint
    status, payload = post(raw)
    assert status == 400
    assert "error" in payload
    set_stats.assert_not_called()
