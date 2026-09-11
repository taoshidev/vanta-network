"""
Focused unit tests for the pro account drawdown rules.
Rule 1 is the daily loss limit against the day's opening equity; Rule 2 is the trailing
loss limit against the end-of-day high-water mark, measured on live equity.
No RPC connections, no disk I/O, no daemon simulation.
"""
import contextlib
from unittest.mock import patch

import pytest

from vali_objects.challenge_period.challengeperiod_manager import (
    ChallengePeriodManager,
    DrawdownStats,
    MinerBucketState,
)
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.vali_config import TradePairCategory, ValiConfig

# ── Constants ─────────────────────────────────────────────────────────────────

DAILY_MS = ValiConfig.DAILY_MS
PRO_BUCKETS = (
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
    MinerBucket.PRO_CHALLENGE_DIRECT,
    MinerBucket.PRO_FUNDED,
)

NOW_MS = 1_748_000_000_000  # fixed reference timestamp (ms)

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
        yield mgr


def _state(bucket: MinerBucket, start_ms: int = NOW_MS) -> MinerBucketState:
    return MinerBucketState("test_hk", [BucketEntry(bucket, start_ms)])


def _intraday_breach() -> DrawdownStats:
    """6% below the day's open with the EOD mark intact, so only Rule 1 binds."""
    return DrawdownStats(current_equity=0.94, daily_open_equity=1.0, eod_hwm=1.0, last_eod_equity=1.0)


def _trailing_breach() -> DrawdownStats:
    """8.6% below the EOD high-water mark but only 3% below the day's open, so only Rule 2 binds.
    last_eod_equity sits at the mark, so the once-a-day EOD comparison stays at zero."""
    return DrawdownStats(current_equity=0.96, daily_open_equity=0.99, eod_hwm=1.05, last_eod_equity=1.05)


def _refresh_pro(manager, hk: str, bucket: MinerBucket, drawdown: DrawdownStats,
                 criteria: DrawdownCriteria = DrawdownCriteria.TRAILING):
    """Run one refresh() pass with the drawdown cache pinned to the given stats."""
    manager.set_miner_bucket(hk, bucket, NOW_MS - DAILY_MS, drawdown_criteria=criteria)
    manager.miner_states[hk].drawdown = drawdown
    manager._position_client.get_all_hotkeys.return_value = [hk]
    manager._position_client.filtered_positions_for_scoring.return_value = ({hk: []}, {})
    manager._position_client.get_positions_for_hotkeys.return_value = {hk: []}
    manager._elimination_client.get_eliminated_hotkeys.return_value = []
    manager._plagiarism_client.get_plagiarism_miners.return_value = []
    manager._miner_account_client.get_accounts.return_value = {}
    manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = {}
    manager._asset_selection_client.get_asset_selections.return_value = {hk: TradePairCategory.CRYPTO}
    with (
        patch.object(manager, "_refresh_drawdown_cache"),
        patch.object(manager, "_refresh_rank_cache"),
        patch.object(manager, "_save_to_disk"),
        patch.object(manager, "_sync_buckets_to_accounts"),
    ):
        manager.refresh(current_time_ms=NOW_MS)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

def test_pro_challenge_thresholds():
    for bucket in (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_CHALLENGE_DIRECT):
        assert bucket.intraday_drawdown_threshold() == ValiConfig.PRO_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD
        assert bucket.eod_drawdown_threshold() == ValiConfig.PRO_CHALLENGE_EOD_DRAWDOWN_THRESHOLD


def test_pro_funded_thresholds():
    assert MinerBucket.PRO_FUNDED.intraday_drawdown_threshold() == ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD
    assert MinerBucket.PRO_FUNDED.eod_drawdown_threshold() == ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD


@pytest.mark.parametrize("bucket", PRO_BUCKETS)
def test_pro_thresholds_configured_independently(bucket):
    """Pro values come from their own config keys, so the standard ones can move on their own."""
    assert bucket.eod_drawdown_threshold() == 0.08
    assert bucket.intraday_drawdown_threshold() == 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Rule checks
# ═══════════════════════════════════════════════════════════════════════════════

def test_pro_intraday_drawdown_reasons():
    challenge = _state(MinerBucket.PRO_CHALLENGE_DIRECT)
    challenge.drawdown = _intraday_breach()
    assert (ChallengePeriodManager._check_intraday_drawdown(challenge)
            == EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN)

    funded = _state(MinerBucket.PRO_FUNDED)
    funded.drawdown = _intraday_breach()
    assert (ChallengePeriodManager._check_intraday_drawdown(funded)
            == EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN)


def test_pro_trailing_drawdown_measures_current_equity():
    """Rule 2 breaches on live equity while the last EOD snapshot is still at the high-water mark."""
    funded = _state(MinerBucket.PRO_FUNDED)
    funded.drawdown = _trailing_breach()
    assert ChallengePeriodManager._check_eod_drawdown(funded) is None
    assert (ChallengePeriodManager._check_trailing_drawdown(funded)
            == EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN)


def test_pro_trailing_drawdown_challenge_reason():
    challenge = _state(MinerBucket.PRO_CHALLENGE_DIRECT)
    challenge.drawdown = _trailing_breach()
    assert (ChallengePeriodManager._check_trailing_drawdown(challenge)
            == EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_EOD_DRAWDOWN)


def test_pro_trailing_drawdown_below_threshold_survives():
    funded = _state(MinerBucket.PRO_FUNDED)
    funded.drawdown = DrawdownStats(current_equity=1.0 - ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD + 0.001,
                                    daily_open_equity=1.0, eod_hwm=1.0, last_eod_equity=1.0)
    assert ChallengePeriodManager._check_trailing_drawdown(funded) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — refresh() routing
# ═══════════════════════════════════════════════════════════════════════════════

def test_refresh_eliminates_pro_funded_on_trailing_drawdown(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _trailing_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN
    # Rule 2 fires on live equity, so the breach is stamped now rather than backdated to midnight
    assert kwargs["elimination_time_ms"] == NOW_MS


def test_refresh_eliminates_pro_funded_on_daily_loss_limit(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _intraday_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN


def test_refresh_demotes_pro_challenge_from_standard_on_trailing_drawdown(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_CHALLENGE_FROM_STANDARD, _trailing_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.SUBACCOUNT_FUNDED
    manager._elimination_client.append_elimination_row.assert_not_called()


def test_refresh_demotes_pro_challenge_direct_on_trailing_drawdown(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_CHALLENGE_DIRECT, _trailing_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.SUBACCOUNT_CHALLENGE
    manager._elimination_client.append_elimination_row.assert_not_called()


def test_refresh_applies_pro_rules_to_static_pro_subaccount(manager):
    """Pro buckets run the pro rules whatever drawdown_criteria the subaccount was created with."""
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _intraday_breach(),
                 criteria=DrawdownCriteria.STATIC)

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN


def test_refresh_eliminates_static_subaccount_on_daily_loss_limit(manager):
    """Static accounts keep the daily loss limit: equity is up on the starting balance, so Rule 1
    is clear, but it is more than the flat threshold below the day's open, so Rule 2 binds."""
    day_open = 1.10
    breach = day_open * (1 - ValiConfig.SUBACCOUNT_STATIC_INTRADAY_DRAWDOWN_THRESHOLD) - 0.001
    drawdown = DrawdownStats(current_equity=breach, daily_open_equity=day_open,
                             eod_hwm=day_open, last_eod_equity=day_open)
    clear_of_rule_1 = _state(MinerBucket.SUBACCOUNT_FUNDED)
    clear_of_rule_1.drawdown = drawdown
    assert ChallengePeriodManager._check_static_drawdown(clear_of_rule_1) is None

    _refresh_pro(manager, "static_hk", MinerBucket.SUBACCOUNT_FUNDED, drawdown,
                 criteria=DrawdownCriteria.STATIC)

    assert manager.get_miner_bucket("static_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN


def test_refresh_leaves_transition_on_standard_rules(manager):
    """PRO_CHALLENGE_TRANSITION still trades the standard account, so a static breach binds it."""
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_CHALLENGE_TRANSITION,
                 DrawdownStats(current_equity=1.0 - ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD - 0.001),
                 criteria=DrawdownCriteria.STATIC)

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN
