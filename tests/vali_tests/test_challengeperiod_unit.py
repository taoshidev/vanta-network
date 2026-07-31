"""
Focused unit tests for ChallengePeriodManager and related dataclasses.
No RPC connections, no disk I/O, no daemon simulation.
Each test uses a manager fixture with all RPC clients mocked and
is_backtesting=True to skip all file system access.
"""
import contextlib
from unittest.mock import patch

import pytest

from vali_objects.challenge_period.challengeperiod_manager import (
    ChallengePeriodManager,
    DrawdownStats,
    MinerBucketState,
)
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.vali_config import TradePairCategory, ValiConfig

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
    json_list = state.to_json()
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
# Section 2 — parse_checkpoint_dict
# ═══════════════════════════════════════════════════════════════════════════════

def test_parse_legacy_format():
    data = {"testing": {"hk_a": NOW_MS}, "success": {"hk_b": NOW_MS + 1}}
    result = ChallengePeriodManager.parse_checkpoint_dict(data)
    assert result["hk_a"].current_bucket == MinerBucket.CHALLENGE
    assert result["hk_b"].current_bucket == MinerBucket.MAINCOMP


def test_parse_dict_format():
    data = {"hk_x": {"bucket": "CHALLENGE", "bucket_start_time": NOW_MS}}
    result = ChallengePeriodManager.parse_checkpoint_dict(data)
    assert result["hk_x"].current_bucket == MinerBucket.CHALLENGE
    assert result["hk_x"].current_bucket_start_ms == NOW_MS


def test_parse_dict_format_with_previous_bucket():
    data = {
        "hk_x": {
            "bucket": "MAINCOMP",
            "bucket_start_time": NOW_MS + 1000,
            "previous_bucket": "CHALLENGE",
            "previous_bucket_start_time": NOW_MS,
        }
    }
    result = ChallengePeriodManager.parse_checkpoint_dict(data)
    state = result["hk_x"]
    assert len(state.entries) == 2
    assert state.entries[0].bucket == MinerBucket.CHALLENGE
    assert state.entries[1].bucket == MinerBucket.MAINCOMP


def test_parse_list_format():
    entries = [
        {"bucket": "CHALLENGE", "start_time_ms": NOW_MS},
        {"bucket": "MAINCOMP", "start_time_ms": NOW_MS + 1000},
    ]
    result = ChallengePeriodManager.parse_checkpoint_dict({"hk_z": entries})
    assert result["hk_z"].current_bucket == MinerBucket.MAINCOMP
    assert len(result["hk_z"].entries) == 2


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
# Section 3b — _check_static_drawdown / _check_static_eod_drawdown (subaccounts)
# ═══════════════════════════════════════════════════════════════════════════════

STATIC_DD_PCT = ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD * 100      # 5.0
STATIC_EOD_DD_PCT = ValiConfig.SUBACCOUNT_STATIC_EOD_DRAWDOWN_THRESHOLD * 100  # 5.0


def test_check_static_drawdown_at_threshold_survives():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(static_drawdown_pct=STATIC_DD_PCT)
    assert ChallengePeriodManager._check_static_drawdown(state) is None


def test_check_static_drawdown_funded_reason():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(static_drawdown_pct=STATIC_DD_PCT + 0.01)
    assert ChallengePeriodManager._check_static_drawdown(state) == EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN


def test_check_static_drawdown_challenge_reason():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE)
    state.drawdown = DrawdownStats(static_drawdown_pct=STATIC_DD_PCT + 0.01)
    assert ChallengePeriodManager._check_static_drawdown(state) == EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_DRAWDOWN


def test_check_static_eod_drawdown_at_threshold_survives():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(static_eod_drawdown_pct=STATIC_EOD_DD_PCT)
    assert ChallengePeriodManager._check_static_eod_drawdown(state) is None


def test_check_static_eod_drawdown_funded_reason():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(static_eod_drawdown_pct=STATIC_EOD_DD_PCT + 0.01)
    assert ChallengePeriodManager._check_static_eod_drawdown(state) == EliminationReason.FAILED_FUNDED_PERIOD_STATIC_EOD_DRAWDOWN


def test_check_static_eod_drawdown_challenge_reason():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE)
    state.drawdown = DrawdownStats(static_eod_drawdown_pct=STATIC_EOD_DD_PCT + 0.01)
    assert ChallengePeriodManager._check_static_eod_drawdown(state) == EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_EOD_DRAWDOWN


def test_check_static_ignores_legacy_drawdown_pcts():
    state = _state(MinerBucket.SUBACCOUNT_FUNDED)
    state.drawdown = DrawdownStats(intraday_drawdown_pct=12.0, eod_drawdown_pct=9.0)
    assert ChallengePeriodManager._check_static_drawdown(state) is None
    assert ChallengePeriodManager._check_static_eod_drawdown(state) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3c — uses_static_drawdown_rules (effective-time gate)
# ═══════════════════════════════════════════════════════════════════════════════

EFFECTIVE_MS = ValiConfig.SUBACCOUNT_STATIC_RULES_EFFECTIVE_MS


def test_uses_static_rules_false_for_regular_miner():
    state = _state(MinerBucket.CHALLENGE, EFFECTIVE_MS + DAILY_MS)
    assert state.uses_static_drawdown_rules is False


def test_uses_static_rules_challenge_registered_after_effective():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE, EFFECTIVE_MS + DAILY_MS)
    assert state.uses_static_drawdown_rules is True


def test_uses_static_rules_challenge_registered_before_effective():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE, EFFECTIVE_MS - DAILY_MS)
    assert state.uses_static_drawdown_rules is False


def test_uses_static_rules_registered_exactly_at_effective():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE, EFFECTIVE_MS)
    assert state.uses_static_drawdown_rules is True


def test_uses_static_rules_funded_keyed_on_challenge_registration():
    state = MinerBucketState("test_hk", [
        BucketEntry(MinerBucket.SUBACCOUNT_CHALLENGE, EFFECTIVE_MS - DAILY_MS),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, EFFECTIVE_MS + DAILY_MS * 5),
    ])
    assert state.uses_static_drawdown_rules is False


def test_uses_static_rules_eliminated_falls_back_to_previous_bucket():
    state = MinerBucketState("test_hk", [
        BucketEntry(MinerBucket.SUBACCOUNT_CHALLENGE, EFFECTIVE_MS + DAILY_MS),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, EFFECTIVE_MS + DAILY_MS * 5),
        BucketEntry(MinerBucket.ELIMINATED, EFFECTIVE_MS + DAILY_MS * 10),
    ])
    assert state.uses_static_drawdown_rules is True


def test_uses_static_rules_helper_excludes_hyperscaled():
    state = _state(MinerBucket.SUBACCOUNT_CHALLENGE, EFFECTIVE_MS + DAILY_MS)
    assert ChallengePeriodManager._uses_static_drawdown_rules(state, MinerAssetClass.HL_ALL) is False
    assert ChallengePeriodManager._uses_static_drawdown_rules(state, MinerAssetClass.CRYPTO) is True
    assert ChallengePeriodManager._uses_static_drawdown_rules(state, None) is True


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
        intraday_drawdown_pct=INTRADAY_THRESHOLD_PCT + 1.0
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


def test_refresh_no_changes_skips_save(manager):
    # CHALLENGE miner only 1 day old — too early for promotion, no drawdown issues
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

    mock_save.assert_not_called()
    mock_sync.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — sync_elimination_miners / _prune_hotkeys_no_positions
# ═══════════════════════════════════════════════════════════════════════════════

def test_sync_elimination_miners_removes(manager):
    manager.miner_states["hk1"] = _state(MinerBucket.CHALLENGE)
    manager.miner_states["hk2"] = _state(MinerBucket.MAINCOMP)
    result = manager.sync_elimination_miners(["hk1"])
    assert result is True
    assert "hk1" not in manager.miner_states
    assert "hk2" in manager.miner_states


def test_sync_elimination_miners_empty(manager):
    manager.miner_states["hk1"] = _state(MinerBucket.CHALLENGE)
    result = manager.sync_elimination_miners([])
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
