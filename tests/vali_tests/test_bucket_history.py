"""
Per-bucket history: every bucket stint keeps its positions, snapshots and daily drawdown/ProStats
rows in validation/miners/<hk>/buckets/<BUCKET>_<start_ms>/, readable through
GET /bucket-statistics/<minerid>.

  * Segment derivation from bucket entries: the SUBACCOUNT_FUNDED transition week lives on the account,
    not in a bucket, so it stays in one segment, which records the week as one range.
  * A hop that runs _switch_account (SUBACCOUNT_CHALLENGE -> SUBACCOUNT_FUNDED, the transition ->
    PRO_CHALLENGE_FUNDED) wipes the account: the closed positions are copied into the segment
    before the archive, and the next account starts with no positions.
  * Entering the transition keeps the account and stays in SUBACCOUNT_FUNDED; a hop that keeps the
    account (-> PRO_FUNDED) closes the positions but keeps them on the account, and the next
    segment inherits the account start.
  * Leaving an earning bucket for any bucket but ELIMINATED pins a payout record on the segment,
    computed from the positions copied into its directory.
  * The live state the statistics are computed from is unchanged: positions/ still only holds the
    current account, and archived_positions/ is still written.

Segment files go to tests/validation/ through running_unit_tests; the RPC clients are mocked.
"""
import contextlib
import json
import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from time_util.time_util import TimeUtil
from tests.vali_tests.mock_utils import fake_account_transitions
from vali_objects.challenge_period.bucket_history import BucketHistoryStore, BucketSegment, segment_runs
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager, ProStats
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.account_snapshot import DEFAULT_SNAPSHOT_LIMIT, AccountSnapshot
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.position import Position
from vanta_api.validator_rest_server import ValidatorRestServer

DAILY_MS = ValiConfig.DAILY_MS
HOTKEY = "bucket_history_hk"
T0 = 1_767_225_600_000  # 2026-01-01 00:00:00 UTC
T1 = T0 + 10 * DAILY_MS
T2 = T0 + 20 * DAILY_MS
T3 = T0 + 25 * DAILY_MS
T4 = T0 + 30 * DAILY_MS
T5 = T0 + 40 * DAILY_MS

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


class FakePositions:
    """In-memory stand-in for the position client: live positions plus an archive, with the
    close-all and archive-all behaviour _switch_account relies on."""

    def __init__(self):
        self.live: dict[str, list[Position]] = {}
        self.archived: dict[str, list[Position]] = {}

    def add(self, position: Position) -> Position:
        self.live.setdefault(position.miner_hotkey, []).append(position)
        return position

    def get_positions_for_one_hotkey(self, hotkey, sort_positions=False, **kwargs):
        return list(self.live.get(hotkey, []))

    def close_all_positions(self, hotkey, close_time_ms, order_source=None):
        for position in self.live.get(hotkey, []):
            if position.is_open_position:
                position.is_closed_position = True
                position.close_ms = close_time_ms

    def archive_positions_for_hotkey(self, hotkey, positions=None, archive_all=False):
        moved = self.live.pop(hotkey, [])
        self.archived.setdefault(hotkey, []).extend(moved)
        return len(moved)


def _position(uuid: str, open_ms: int, close_ms: int | None = None) -> Position:
    return Position(
        miner_hotkey=HOTKEY, position_uuid=uuid, open_ms=open_ms, trade_pair=TradePair.EURUSD,
        position_type=OrderType.LONG, close_ms=close_ms, is_closed_position=close_ms is not None,
    )


def _account(balance: float = 100_000.0, equity: float = 100_500.0, size: float = 100_000.0):
    return SimpleNamespace(balance=balance, equity=equity, account_size=size, get_account_size=lambda: size)


@pytest.fixture
def manager():
    """A ChallengePeriodManager with a live bucket history (running_unit_tests) and mocked clients."""
    with contextlib.ExitStack() as stack:
        for path in _CP_CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True, running_unit_tests=True)
        mgr._bucket_history.clear_for_test()
        fake_account_transitions(mgr)
        # The miner directory its positions/ would live in; bucket history never creates it
        os.makedirs(_miner_dir(), exist_ok=True)
        mgr._position_client = FakePositions()
        mgr._miner_account_client.get_accounts.side_effect = lambda hks: {hk: _account() for hk in hks}
        mgr._entity_client.is_instant_funded.return_value = False
        stack.enter_context(patch.object(mgr, "_sync_buckets_to_accounts"))
        yield mgr
        mgr._bucket_history.clear_for_test()
        shutil.rmtree(_miner_dir(), ignore_errors=True)


def _miner_dir() -> str:
    return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=True)}{HOTKEY}"


def _segment(manager, bucket: MinerBucket, start_ms: int) -> BucketSegment:
    segment = manager._bucket_history.get_segment(HOTKEY, start_ms, bucket.value)
    assert segment is not None, f"no {bucket.value} segment at {start_ms}"
    return segment


def _copied(manager, segment: BucketSegment) -> dict[str, dict]:
    return {p["position_uuid"]: p for p in manager._bucket_history.read_positions(segment)}


def _live_uuids(manager) -> set[str]:
    return {p.position_uuid for p in manager._position_client.get_positions_for_one_hotkey(HOTKEY)}


# ── Segment derivation ────────────────────────────────────────────────────────

def test_the_transition_is_one_range_inside_the_funded_segment():
    runs = segment_runs([
        BucketEntry(MinerBucket.SUBACCOUNT_CHALLENGE, T0),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, T1),
        BucketEntry(MinerBucket.PRO_CHALLENGE_FUNDED, T3),
        BucketEntry(MinerBucket.PRO_FUNDED, T5),
    ], transition_start_ms=T2)
    assert [(r.bucket, r.start_ms, r.end_ms) for r in runs] == [
        ("SUBACCOUNT_CHALLENGE", T0, T1),
        ("SUBACCOUNT_FUNDED", T1, T3),
        ("PRO_CHALLENGE_FROM_STANDARD", T3, T5),
        ("PRO_FUNDED", T5, None),
    ]
    funded, pro_challenge, pro_funded = runs[1], runs[2], runs[3]
    assert funded.transition_start_ms == T2
    assert funded.account_start_ms == T1                   # entered through an account switch
    assert pro_challenge.account_start_ms == T3            # leaving the transition switched account
    assert pro_funded.account_start_ms == T3               # PRO_FUNDED keeps that account
    assert all(r.transition_start_ms is None for r in (runs[0], pro_challenge, pro_funded))


def test_a_running_transition_has_no_end():
    funded = segment_runs([BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, T1)], transition_start_ms=T2)[0]
    assert (funded.end_ms, funded.transition_start_ms) == (None, T2)


def test_a_transition_reopened_by_a_revert_starts_with_the_reopened_segment():
    runs = segment_runs([
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, T1),
        BucketEntry(MinerBucket.ELIMINATED, T3),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, T4),
    ], transition_start_ms=T2)
    assert [r.transition_start_ms for r in runs] == [T2, None, T4]


def test_a_segment_keeps_its_transition_once_the_account_moves_on(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T2, in_transition=True)[0]
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T3)[0]
    manager._reconcile_bucket_history([HOTKEY], T4)

    funded = _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert (funded.transition_start_ms, funded.transition_end_ms) == (T2, T3)


def test_reverted_entries_are_skipped():
    runs = segment_runs([
        BucketEntry(MinerBucket.PRO_FUNDED, T0),
        BucketEntry(MinerBucket.ELIMINATED, T1, reverted_ms=T2),
        BucketEntry(MinerBucket.PRO_FUNDED, T2),
    ])
    assert [(r.bucket, r.start_ms, r.end_ms) for r in runs] == [("PRO_FUNDED", T0, None)]


# ── Account switches wipe, other hops carry ──────────────────────────────────

def test_challenge_to_funded_wipes_the_positions_into_the_challenge_segment(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, T0)
    manager._position_client.add(_position("closed_1", T0 + 1, T0 + 2))
    manager._position_client.add(_position("open_1", T0 + 3))

    ok, _ = manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert ok

    challenge = _segment(manager, MinerBucket.SUBACCOUNT_CHALLENGE, T0)
    assert challenge.end_ms == T1
    copied = _copied(manager, challenge)
    assert set(copied) == {"closed_1", "open_1"}
    assert copied["open_1"]["close_ms"] == T1 - 1          # copied after the switch closed it
    assert challenge.final_drawdown is not None and challenge.end_balance == 100_000.0

    # The live state the statistics read is untouched: archived, and nothing carried over
    assert _live_uuids(manager) == set()
    assert {p.position_uuid for p in manager._position_client.archived[HOTKEY]} == {"closed_1", "open_1"}

    funded = _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert funded.is_live and funded.account_start_ms == T1
    stats = manager.get_bucket_statistics(HOTKEY, bucket="SUBACCOUNT_FUNDED", include=["positions"])
    assert stats["segments"][0]["positions"] == []


def test_the_pro_track_carries_positions_except_across_the_transition_exit(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    manager._position_client.add(_position("standard_open", T1 + 1))

    # Entering the transition keeps the standard account and stays in the SUBACCOUNT_FUNDED segment
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T2, in_transition=True)[0]
    funded = _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert [s.key for s in manager._bucket_history.get_segments(HOTKEY)] == [funded.key]
    assert funded.is_live
    assert (funded.transition_start_ms, funded.transition_end_ms) == (T2, None)
    assert _live_uuids(manager) == {"standard_open"}

    # Leaving it for PRO_CHALLENGE_FUNDED switches onto the pro account
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T3)[0]
    assert (funded.end_ms, funded.transition_end_ms) == (T3, T3)
    assert _copied(manager, funded)["standard_open"]["close_ms"] == T3 - 1
    assert funded.final_drawdown is not None
    assert _live_uuids(manager) == set()                 # only positions opened after the switch count
    pcf = _segment(manager, MinerBucket.PRO_CHALLENGE_FUNDED, T3)
    assert pcf.is_live and pcf.account_start_ms == T3

    # The pro account trades, then passes: PRO_FUNDED closes the positions but keeps the account
    manager._position_client.add(_position("pro_open", T4))
    archived_before = len(manager._position_client.archived[HOTKEY])
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T5)[0]

    assert pcf.end_ms == T5
    assert set(_copied(manager, pcf)) == {"pro_open"}
    assert _live_uuids(manager) == {"pro_open"}
    assert len(manager._position_client.archived[HOTKEY]) == archived_before

    pro_funded = _segment(manager, MinerBucket.PRO_FUNDED, T5)
    assert pro_funded.account_start_ms == T3
    stats = manager.get_bucket_statistics(HOTKEY, bucket="PRO_FUNDED", include=["positions"])
    assert [p["position_uuid"] for p in stats["segments"][0]["positions"]] == ["pro_open"]


def test_a_closed_segment_keeps_its_snapshot_slice(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    for ms in (T0, T1 + 1000, T2 - 1000, T2 + 1000):
        ValiBkpUtils.log_snapshot(HOTKEY, AccountSnapshot(ms, 100_000.0, 100_000.0, 100_100.0).to_dict(),
                                  running_unit_tests=True)

    manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T2)

    funded = _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1)
    snapshots = manager._bucket_history.read_snapshots(funded)
    assert [s["snapshot_ms"] for s in snapshots] == [T1 + 1000, T2 - 1000]
    live = manager.get_bucket_statistics(HOTKEY, start_ms=T2, include=["snapshots"])
    assert [s["snapshot_ms"] for s in live["segments"][0]["snapshots"]] == [T2 + 1000]


# ── Daily stats rows ──────────────────────────────────────────────────────────

def test_stats_rows_are_written_at_most_once_per_utc_day(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_DIRECT, T0)
    manager.miner_states[HOTKEY].pro_stats = ProStats(calmar=2.5, trading_days=12)

    manager._record_bucket_stats([HOTKEY], T1 + 1000)
    manager._record_bucket_stats([HOTKEY], T1 + 5000)
    manager._record_bucket_stats([HOTKEY], T1 + DAILY_MS, {HOTKEY: T1})

    rows = manager._bucket_history.read_stats(_segment(manager, MinerBucket.PRO_CHALLENGE_DIRECT, T0))
    assert [r["day_ms"] for r in rows] == [T1, T1 + DAILY_MS]
    assert [r["in_transition"] for r in rows] == [False, True]
    assert rows[0]["calmar"] == 2.5 and rows[0]["trading_days"] == 12
    assert {"intraday_drawdown_pct", "eod_drawdown_pct", "eod_hwm", "current_equity"} <= set(rows[0])


# ── Undone entries ────────────────────────────────────────────────────────────

def test_a_popped_entry_reverts_its_segment_and_reopens_the_one_before(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.CHALLENGE, T0)
    manager.sync_plagiarism_miners([HOTKEY], T1)
    manager._reconcile_bucket_history([HOTKEY], T1)
    assert _segment(manager, MinerBucket.CHALLENGE, T0).end_ms == T1

    manager.sync_plagiarism_miners([], T2)
    manager._reconcile_bucket_history([HOTKEY], T2)

    assert _segment(manager, MinerBucket.PLAGIARISM, T1).reverted_ms == T2
    assert _segment(manager, MinerBucket.CHALLENGE, T0).is_live
    stats = manager.get_bucket_statistics(HOTKEY)
    assert [s["bucket"] for s in stats["segments"]] == ["CHALLENGE"]


def test_bucket_history_never_creates_a_miner_directory(manager):
    """The elimination manager reads a miner directory as a sign of a hotkey (handle_zombies), so a
    hotkey with no files of its own must not get one from its bucket history."""
    shutil.rmtree(_miner_dir())
    manager.set_miner_bucket(HOTKEY, MinerBucket.CHALLENGE, T0)
    manager._record_bucket_stats([HOTKEY], T1)
    manager.admin_set_bucket(HOTKEY, MinerBucket.MAINCOMP, T2)

    assert not os.path.exists(_miner_dir())
    assert [s.bucket for s in manager._bucket_history.get_segments(HOTKEY)] == ["CHALLENGE", "MAINCOMP"]
    assert [s.key for s in BucketHistoryStore(running_unit_tests=True).get_segments(HOTKEY)] == [
        ("CHALLENGE", T0), ("MAINCOMP", T2)]


def test_backtesting_keeps_no_history():
    with contextlib.ExitStack() as stack:
        for path in _CP_CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        mgr.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, T0)
        assert not mgr._bucket_history.enabled
        assert mgr.get_bucket_statistics(HOTKEY) is None


# ── Snapshots and checkpoint ──────────────────────────────────────────────────

def test_account_snapshots_carry_drawdown_and_legacy_rows_still_parse():
    snapshot = AccountSnapshot(T0, 100_000.0, 99_000.0, 98_000.0,
                               intraday_drawdown_pct=1.5, eod_drawdown_pct=2.0, eod_hwm=1.01)
    assert AccountSnapshot.from_dict(snapshot.to_dict()) == snapshot
    legacy = AccountSnapshot.from_dict({"snapshot_ms": T0, "account_size": 1.0, "balance": 1.0, "equity": 1.0})
    assert legacy.intraday_drawdown_pct is None and legacy.eod_hwm is None


def test_take_account_snapshot_stamps_the_drawdown_fields():
    from vali_objects.miner_account.miner_account_manager import MinerAccountManager
    mgr = MinerAccountManager.__new__(MinerAccountManager)
    mgr.accounts = {HOTKEY: SimpleNamespace(miner_hotkey=HOTKEY, miner_bucket=MinerBucket.SUBACCOUNT_FUNDED,
                                            get_account_size=lambda: 100_000.0, balance=100_000.0,
                                            equity=99_000.0, daily_open_snapshot=None)}
    mgr._accounts_lock = MagicMock()
    mgr._save_accounts_to_disk = MagicMock()
    mgr.running_unit_tests = True
    with patch.object(ValiBkpUtils, "log_snapshot") as log_snapshot:
        mgr.take_account_snapshot(timestamp_ms=T0, drawdown_by_hotkey={
            HOTKEY: {"intraday_drawdown_pct": 1.0, "eod_drawdown_pct": 0.5, "eod_hwm": 1.02}})
    row = log_snapshot.call_args[0][1]
    assert (row["intraday_drawdown_pct"], row["eod_drawdown_pct"], row["eod_hwm"]) == (1.0, 0.5, 1.02)


def test_checkpoint_sync_prefers_the_checkpoint_and_keeps_local_only_segments(manager):
    store = manager._bucket_history
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, T0)
    manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    checkpoint = json.loads(json.dumps(store.to_checkpoint_dict()))

    checkpoint[HOTKEY][0]["end_balance"] = 123.0
    del checkpoint[HOTKEY][1]  # the live segment only this validator has seen yet
    stats = store.sync_from_checkpoint(checkpoint)

    assert stats == {"segments_added": 0, "segments_replaced": 1, "errors": 0}
    assert _segment(manager, MinerBucket.SUBACCOUNT_CHALLENGE, T0).end_balance == 123.0
    assert _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1).is_live

    reloaded = BucketHistoryStore(running_unit_tests=True)
    assert [s.key for s in reloaded.get_segments(HOTKEY)] == [("SUBACCOUNT_CHALLENGE", T0), ("SUBACCOUNT_FUNDED", T1)]


def test_a_synced_challenge_period_builds_its_segments(manager):
    manager.sync_challenge_period_data({HOTKEY: {"entries": [
        BucketEntry(MinerBucket.SUBACCOUNT_CHALLENGE, T0).to_dict(),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, T1).to_dict(),
    ]}})
    segments = manager._bucket_history.get_segments(HOTKEY)
    assert [(s.bucket, s.end_ms, s.backfilled) for s in segments] == [
        ("SUBACCOUNT_CHALLENGE", T1, True),
        ("SUBACCOUNT_FUNDED", None, False),
    ]


# ── Payout records ────────────────────────────────────────────────────────────

@pytest.fixture
def subaccount(manager):
    """The manager with HOTKEY treated as a subaccount and the entity's record calculation stubbed:
    the record pays 10 per position it was handed."""
    calls = []

    def _record(hotkey, segment, positions, end_ms):
        calls.append((segment["bucket"], sorted(p["position_uuid"] for p in positions), end_ms))
        payout = 10.0 * len(positions)
        return {"computed_ms": end_ms, "end_ms": end_ms, "payout_usd": payout, "gross_payout_usd": payout,
                "settlements": [{"start_ms": segment["start_ms"], "end_ms": end_ms, "payout": payout}],
                "amended_ms": None}

    manager._entity_client.calculate_segment_payout_record.side_effect = _record
    manager._entity_client.apply_bucket_account_size.return_value = (True, "")
    with patch("vali_objects.challenge_period.challengeperiod_manager.is_synthetic_hotkey", return_value=True):
        yield manager, calls


def test_leaving_an_earning_bucket_pins_a_payout_record(subaccount):
    manager, calls = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    manager._position_client.add(_position("standard_open", T1 + 1))
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T2, in_transition=True)[0]
    assert calls == []  # the transition stays in the segment

    assert manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T3)[0]
    manager._position_client.add(_position("pro_open", T4))
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FUNDED}, T5)

    # Each record is computed from the positions copied into its own segment directory
    assert calls == [("SUBACCOUNT_FUNDED", ["standard_open"], T3), ("PRO_CHALLENGE_FROM_STANDARD", ["pro_open"], T5)]
    assert _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1).payout_record["payout_usd"] == 10.0
    assert _segment(manager, MinerBucket.PRO_CHALLENGE_FUNDED, T3).payout_record["end_ms"] == T5
    assert _segment(manager, MinerBucket.PRO_FUNDED, T5).payout_record is None


def test_elimination_writes_no_payout_record(subaccount):
    manager, calls = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T1)
    manager._position_client.add(_position("open", T1 + 1))

    assert manager.admin_set_bucket(HOTKEY, MinerBucket.ELIMINATED, T2)[0]

    assert calls == []
    assert _segment(manager, MinerBucket.PRO_FUNDED, T1).payout_record is None


def test_a_failed_record_does_not_block_the_move(subaccount):
    manager, _ = subaccount
    manager._entity_client.calculate_segment_payout_record.side_effect = None
    manager._entity_client.calculate_segment_payout_record.return_value = None
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T1)

    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FUNDED}, T2)

    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_FUNDED
    assert _segment(manager, MinerBucket.PRO_CHALLENGE_FUNDED, T1).payout_record is None


def test_a_reopened_segment_drops_its_record(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_ALPHA, T2)[0]
    assert _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1).payout_record is not None

    manager.miner_states[HOTKEY].pop_bucket_entry(MinerBucket.SUBACCOUNT_ALPHA)
    manager._reconcile_bucket_history([HOTKEY], T3)

    funded = _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert funded.is_live and funded.payout_record is None


def test_closed_weeks_freeze_once_final_and_once_per_week(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T1)
    calculate = manager._entity_client.calculate_segment_closed_weeks
    calculate.return_value = None  # the week that just ended is not final yet

    manager._freeze_closed_weeks([HOTKEY], T3)
    manager._freeze_closed_weeks([HOTKEY], T3 + 60_000)  # inside the retry interval
    assert calculate.call_count == 1

    row = {"start_ms": T1, "end_ms": T3, "payout": 5.0}
    calculate.return_value = [row]
    later = T3 + manager.CLOSED_WEEKS_RETRY_MS
    manager._freeze_closed_weeks([HOTKEY], later)
    manager._freeze_closed_weeks([HOTKEY], later + manager.CLOSED_WEEKS_RETRY_MS)  # this week is done
    assert calculate.call_count == 2
    assert _segment(manager, MinerBucket.PRO_FUNDED, T1).closed_weeks == [row]
    assert BucketHistoryStore(running_unit_tests=True).get_segment(HOTKEY, T1).closed_weeks == [row]

    calculate.return_value = []
    manager._freeze_closed_weeks([HOTKEY], later + 7 * DAILY_MS)  # the next week is checked again
    assert calculate.call_count == 3
    assert calculate.call_args.args[1]["closed_weeks"] == [row]


def test_admin_unfreezes_a_week_and_every_later_one_of_its_segment(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T1)
    week = TimeUtil.ms_at_start_of_week(T1) + 7 * DAILY_MS
    weeks = [week + i * 7 * DAILY_MS for i in range(3)]
    segment = _segment(manager, MinerBucket.PRO_FUNDED, T1)
    segment.closed_weeks = [{"start_ms": w, "end_ms": w + 7 * DAILY_MS, "payout": 1.0} for w in weeks]
    manager._closed_weeks_checked[HOTKEY] = weeks[-1]

    ok, _, dropped = manager.admin_drop_closed_weeks(HOTKEY, [weeks[1]])
    assert ok and dropped == weeks[1:]
    assert [row["start_ms"] for row in segment.closed_weeks] == weeks[:1]
    assert HOTKEY not in manager._closed_weeks_checked  # frozen again on the next refresh
    assert BucketHistoryStore(running_unit_tests=True).get_segment(HOTKEY, T1).closed_weeks == segment.closed_weeks

    assert manager.admin_drop_closed_weeks(HOTKEY, [weeks[2]])[0] is False
    assert manager.admin_drop_closed_weeks(HOTKEY)[2] == weeks[:1]
    assert segment.closed_weeks is None
    assert manager.admin_drop_closed_weeks("unknown_hk")[0] is False


def test_closed_weeks_are_only_frozen_for_earning_buckets(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, T1)
    manager._freeze_closed_weeks([HOTKEY], T3)
    manager._entity_client.calculate_segment_closed_weeks.assert_not_called()


def test_the_record_rides_the_checkpoint(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T1)
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FUNDED}, T2)
    checkpoint = json.loads(json.dumps(manager.get_bucket_history_checkpoint_dict()))

    manager._bucket_history.clear_for_test()
    manager._bucket_history.sync_from_checkpoint(checkpoint)

    assert _segment(manager, MinerBucket.PRO_CHALLENGE_FUNDED, T1).payout_record["end_ms"] == T2
    assert [r["start_ms"] for r in manager.get_payout_records(HOTKEY)] == [T1]


def test_admin_amends_removes_and_recomputes_a_record(subaccount):
    manager, calls = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FUNDED, T1)
    manager._position_client.add(_position("pro_open", T1 + 1))
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FUNDED}, T2)
    segment = _segment(manager, MinerBucket.PRO_CHALLENGE_FUNDED, T1)

    ok, _, record = manager.admin_update_payout_record(HOTKEY, T1, payout_usd=25.0)
    assert ok and record["payout_usd"] == 25.0 and record["amended_ms"] is not None
    assert sum(row["payout"] for row in record["settlements"]) == 25.0

    assert manager.admin_update_payout_record(HOTKEY, T1, remove=True)[0]
    assert segment.payout_record is None
    assert manager.admin_update_payout_record(HOTKEY, T1, payout_usd=1.0)[0] is False

    ok, _, record = manager.admin_update_payout_record(HOTKEY, T1, recompute=True)
    assert ok and record["payout_usd"] == 10.0
    assert calls[-1] == ("PRO_CHALLENGE_FROM_STANDARD", ["pro_open"], T2)

    # The live segment cannot be recomputed, and an unknown one is not found
    assert manager.admin_update_payout_record(HOTKEY, T2, recompute=True)[0] is False
    assert manager.admin_update_payout_record(HOTKEY, T0, remove=True)[0] is False


@pytest.fixture
def payout_rest():
    server = object.__new__(ValidatorRestServer)
    server._get_api_key_safe = MagicMock(return_value="key")
    server.is_valid_api_key = MagicMock(return_value=True)
    server.can_access_tier = MagicMock(return_value=True)
    server._challenge_period_client = MagicMock()
    server._challenge_period_client.get_payout_records.return_value = [{"start_ms": T1}]
    server._challenge_period_client.admin_update_payout_record.return_value = (True, "updated", {})
    app = Flask(__name__)
    app.route("/admin/payout-record/<hotkey>", methods=["GET"])(server.get_payout_records)
    app.route("/admin/payout-record/<hotkey>", methods=["POST"])(server.update_payout_record)
    return server, app.test_client()


def test_rest_payout_record_actions(payout_rest):
    server, client = payout_rest
    assert client.get(f"/admin/payout-record/{HOTKEY}").get_json()["payout_records"] == [{"start_ms": T1}]
    update = server._challenge_period_client.admin_update_payout_record
    for query, args in ((f"payout_usd=12.5", (12.5, False, False)), ("remove=true", (None, True, False)),
                        ("recompute=true", (None, False, True))):
        assert client.post(f"/admin/payout-record/{HOTKEY}?segment_start_ms={T1}&{query}").status_code == 200
        update.assert_called_with(HOTKEY, T1, *args)


@pytest.mark.parametrize("query", ["payout_usd=1", f"segment_start_ms={T1}", f"segment_start_ms={T1}&payout_usd=-1",
                                   f"segment_start_ms={T1}&payout_usd=1&remove=true"])
def test_rest_payout_record_rejects_bad_params(payout_rest, query):
    server, client = payout_rest
    assert client.post(f"/admin/payout-record/{HOTKEY}?{query}").status_code == 400
    server._challenge_period_client.admin_update_payout_record.assert_not_called()


@pytest.fixture
def reset_rest():
    server = object.__new__(ValidatorRestServer)
    server._get_api_key_safe = MagicMock(return_value="key")
    server.is_valid_api_key = MagicMock(return_value=True)
    server.can_access_tier = MagicMock(return_value=True)
    for client in ("_position_client", "_perf_ledger_client", "_debt_ledger_client", "_elimination_client",
                   "_challenge_period_client", "_miner_account_client", "_entity_client"):
        setattr(server, client, MagicMock())
    server._position_client.wipe_hotkey.return_value = {}
    server._challenge_period_client.admin_drop_closed_weeks.return_value = (True, "unfroze", [T1])
    app = Flask(__name__)
    app.route("/admin/reset/<hotkey>", methods=["POST"])(server.reset_hotkey)
    return server, app.test_client()


@pytest.mark.parametrize("body, weeks", [({"recompute_weeks": [T1, T2]}, [T1, T2]), ({"recompute_weeks": "all"}, None)])
def test_reset_unfreezes_the_closed_weeks_asked_for(reset_rest, body, weeks):
    server, client = reset_rest
    response = client.post(f"/admin/reset/{HOTKEY}", json=body)
    assert response.status_code == 200 and response.get_json()["recomputed_weeks"] == [T1]
    server._challenge_period_client.admin_drop_closed_weeks.assert_called_once_with(HOTKEY, weeks)


def test_reset_keeps_closed_weeks_by_default(reset_rest):
    server, client = reset_rest
    assert client.post(f"/admin/reset/{HOTKEY}", json={}).status_code == 200
    server._challenge_period_client.admin_drop_closed_weeks.assert_not_called()


@pytest.mark.parametrize("weeks", [[], "some", ["1"], [True], 5])
def test_reset_rejects_bad_recompute_weeks_before_wiping(reset_rest, weeks):
    server, client = reset_rest
    assert client.post(f"/admin/reset/{HOTKEY}", json={"recompute_weeks": weeks}).status_code == 400
    server._position_client.wipe_hotkey.assert_not_called()


# ── REST ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def rest():
    server = object.__new__(ValidatorRestServer)
    server._get_api_key_safe = MagicMock(return_value="key")
    server.is_valid_api_key = MagicMock(return_value=True)
    server._challenge_period_client = MagicMock()
    server._challenge_period_client.get_bucket_statistics.return_value = {
        "hotkey": HOTKEY, "segments": [{"bucket": "PRO_CHALLENGE_FROM_STANDARD", "start_ms": T2}]}
    app = Flask(__name__)
    app.route("/bucket-statistics/<minerid>", methods=["GET"])(server.get_bucket_statistics)
    return server, app.test_client()


@pytest.mark.parametrize("include", ["summary", "positions", "snapshots", "stats", "summary,positions,snapshots,stats"])
def test_rest_passes_each_include_through(rest, include):
    server, client = rest
    response = client.get(f"/bucket-statistics/{HOTKEY}?include={include}&bucket=PRO_CHALLENGE_FROM_STANDARD&start_ms={T2}")
    assert response.status_code == 200
    assert response.get_json()["segments"][0]["start_ms"] == T2
    server._challenge_period_client.get_bucket_statistics.assert_called_once_with(
        HOTKEY, bucket="PRO_CHALLENGE_FROM_STANDARD", start_ms=T2, include=include.split(","),
        limit=DEFAULT_SNAPSHOT_LIMIT)


@pytest.mark.parametrize("query", ["include=everything", "bucket=NOT_A_BUCKET", "start_ms=abc", "limit=0"])
def test_rest_rejects_bad_params(rest, query):
    _, client = rest
    assert client.get(f"/bucket-statistics/{HOTKEY}?{query}").status_code == 400


def test_rest_unauthorized_and_not_found(rest):
    server, client = rest
    server._challenge_period_client.get_bucket_statistics.return_value = None
    assert client.get(f"/bucket-statistics/{HOTKEY}").status_code == 404
    server.is_valid_api_key.return_value = False
    assert client.get(f"/bucket-statistics/{HOTKEY}").status_code == 401


def test_rest_reads_a_real_manager(manager, rest):
    server, client = rest
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, T0)
    manager._position_client.add(_position("closed_1", T0 + 1, T0 + 2))
    manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    server._challenge_period_client = manager

    body = client.get(f"/bucket-statistics/{HOTKEY}?bucket=SUBACCOUNT_CHALLENGE&include=summary,positions").get_json()
    segment = body["segments"][0]
    assert (segment["bucket"], segment["live"], segment["end_ms"]) == ("SUBACCOUNT_CHALLENGE", False, T1)
    assert [p["position_uuid"] for p in segment["positions"]] == ["closed_1"]
