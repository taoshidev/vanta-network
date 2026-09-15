"""
Focused unit tests for the pro account drawdown rules.
Rule 1 is the daily loss limit against the day's opening equity; Rule 2 is the loss limit on the
latched end-of-day equity against the end-of-day high-water mark, checked once per UTC day.
No RPC connections, no disk I/O, no daemon simulation.
"""
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.shared_objects.test_utilities import create_daily_checkpoints_with_pnl

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
MIDNIGHT_MS = (NOW_MS // DAILY_MS) * DAILY_MS  # last UTC midnight latch

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


def _live_dip_only() -> DrawdownStats:
    """Live equity 8.57% below the EOD high-water mark, but the latched EOD equity sits at the
    mark and the day's open is only 3% away. Breaches nothing now that Rule 2 is EOD-only."""
    return DrawdownStats(current_equity=0.96, daily_open_equity=0.99, eod_hwm=1.05, last_eod_equity=1.05)


def _eod_breach() -> DrawdownStats:
    """Latched EOD equity 8.6% below the high-water mark, only 1% below the day's open, so only
    Rule 2 binds."""
    return DrawdownStats(current_equity=0.95, daily_open_equity=0.9597,
                         eod_hwm=1.05, last_eod_equity=0.9597,
                         last_eod_checked_ms=MIDNIGHT_MS)


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


def test_pro_eod_drawdown_ignores_intraday_dip_below_mark():
    """Live equity 8.57% under the mark is not a breach: only the latched EOD equity counts."""
    funded = _state(MinerBucket.PRO_FUNDED)
    funded.drawdown = _live_dip_only()
    assert ChallengePeriodManager._check_eod_drawdown(funded) is None


def test_pro_eod_drawdown_funded_reason():
    funded = _state(MinerBucket.PRO_FUNDED)
    funded.drawdown = _eod_breach()
    assert (ChallengePeriodManager._check_eod_drawdown(funded)
            == EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN)


def test_pro_eod_drawdown_challenge_reason():
    challenge = _state(MinerBucket.PRO_CHALLENGE_DIRECT)
    challenge.drawdown = _eod_breach()
    assert (ChallengePeriodManager._check_eod_drawdown(challenge)
            == EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_EOD_DRAWDOWN)


def test_pro_eod_drawdown_below_threshold_survives():
    funded = _state(MinerBucket.PRO_FUNDED)
    just_inside = 1.0 - ValiConfig.PRO_FUNDED_EOD_DRAWDOWN_THRESHOLD + 0.001
    funded.drawdown = DrawdownStats(current_equity=just_inside, daily_open_equity=1.0,
                                    eod_hwm=1.0, last_eod_equity=just_inside)
    assert ChallengePeriodManager._check_eod_drawdown(funded) is None


def test_check_trailing_drawdown_removed():
    """Pro Rule 2 is EOD-only; the live-equity elimination check must not come back."""
    assert not hasattr(ChallengePeriodManager, "_check_trailing_drawdown")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — refresh() routing
# ═══════════════════════════════════════════════════════════════════════════════

def test_refresh_pro_funded_survives_intraday_dip_below_hwm(manager):
    """End-to-end mirror of the EOD-only rule: a live dip past the mark neither eliminates nor demotes."""
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _live_dip_only())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.PRO_FUNDED
    manager._elimination_client.append_elimination_row.assert_not_called()


def test_refresh_eliminates_pro_funded_on_eod_drawdown(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _eod_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN
    assert kwargs["elimination_drawdown_pct"] == pytest.approx(8.6, abs=0.05)
    assert kwargs["eod_drawdown_pct"] == pytest.approx(8.6, abs=0.05)
    assert kwargs["intraday_drawdown_pct"] == pytest.approx(1.01, abs=0.05)


def test_pro_elimination_row_backdates_to_last_eod_latch(manager):
    """The breach happened at the midnight latch, not on the refresh pass that noticed it."""
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _eod_breach())

    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["elimination_time_ms"] == MIDNIGHT_MS


def test_pro_elimination_row_falls_back_to_now_without_a_latch(manager):
    drawdown = _eod_breach()
    drawdown.last_eod_checked_ms = None
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, drawdown)

    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["elimination_time_ms"] == NOW_MS


def test_static_pro_elimination_row_reports_eod_not_static(manager):
    """A pro subaccount created static still runs the pro rules, so the row carries the EOD
    number (8.6%), not static equity-vs-starting-balance (5.0%)."""
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _eod_breach(),
                 criteria=DrawdownCriteria.STATIC)

    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN
    assert kwargs["eod_drawdown_pct"] == pytest.approx(8.6, abs=0.05)


def test_refresh_eliminates_pro_funded_on_daily_loss_limit(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_FUNDED, _intraday_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.ELIMINATED
    kwargs = manager._elimination_client.append_elimination_row.call_args.kwargs
    assert kwargs["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN


def test_refresh_demotes_pro_challenge_from_standard_on_eod_drawdown(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_CHALLENGE_FROM_STANDARD, _eod_breach())

    assert manager.get_miner_bucket("pro_hk") == MinerBucket.SUBACCOUNT_FUNDED
    manager._elimination_client.append_elimination_row.assert_not_called()


def test_refresh_demotes_pro_challenge_direct_on_eod_drawdown(manager):
    _refresh_pro(manager, "pro_hk", MinerBucket.PRO_CHALLENGE_DIRECT, _eod_breach())

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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — the two rules together
# ═══════════════════════════════════════════════════════════════════════════════

def test_live_dip_survives_rule_2_but_still_deepens_the_calmar_drawdown(manager):
    """The spec's two halves in one scenario: live equity 8.57% under the midnight mark does not
    breach the EOD rule, but it does ratchet the calmar denominator, so the miner keeps trading
    and still cannot promote."""
    hk = "pro_hk"
    drawdown = _live_dip_only()
    _refresh_pro(manager, hk, MinerBucket.PRO_CHALLENGE_DIRECT, drawdown)

    # Rule 2 is EOD-only, so the dip neither eliminates nor demotes
    assert manager.get_miner_bucket(hk) == MinerBucket.PRO_CHALLENGE_DIRECT
    manager._elimination_client.append_elimination_row.assert_not_called()

    # ...but the same dip is what the calmar denominator measures
    ledger = create_daily_checkpoints_with_pnl([0.0] * 5, [0.0] * 5)
    ledger.cps[-1].prev_portfolio_realized_pnl = 6_000.0  # 6% on a 100k account
    ledger.cps[-1].cumulative_fees_usd = 0.0
    manager.miner_states[hk].drawdown = drawdown
    manager._refresh_pro_stats([hk], {hk: ledger}, {hk: SimpleNamespace(account_size=100_000.0)})

    pro_stats = manager.miner_states[hk].pro_stats
    assert pro_stats.max_drawdown == pytest.approx(0.96 / 1.05)          # 8.57% under the mark
    assert pro_stats.calmar == pytest.approx(0.70)                        # 0.06 / 0.0857
    assert pro_stats.calmar < ValiConfig.PRO_CHALLENGE_CALMAR_THRESHOLD   # so promotion is blocked
