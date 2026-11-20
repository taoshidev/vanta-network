"""
Per-bucket history: every bucket stint keeps its positions, snapshots and daily drawdown/ProStats
rows in validation/miners/<hk>/buckets/<BUCKET>_<start_ms>/, readable through
GET /bucket-statistics/<minerid>.

  * Segment derivation from bucket entries.
  * A hop that runs _switch_account wipes the account: the closed positions are copied into the
    segment before the archive, and the next account starts with no positions.
  * A hop that keeps the account (-> PRO_FUNDED) keeps its positions on the account, and the next
    segment inherits the account start.
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
        # The miner directory its positions/ would live in; bucket history never creates it
        os.makedirs(_miner_dir(), exist_ok=True)
        mgr._position_client = FakePositions()
        mgr._miner_account_client.get_accounts.side_effect = lambda hks: {hk: _account() for hk in hks}
        mgr._miner_account_client.get_standard_account.return_value = None
        mgr._entity_client.get_subaccount_info_for_synthetic.return_value = None
        mgr._entity_client.apply_bucket_account_size.return_value = (True, "")
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

def test_segments_follow_the_bucket_entries_and_account_switches():
    runs = segment_runs([
        BucketEntry(MinerBucket.SUBACCOUNT_CHALLENGE, T0),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, T1),
        BucketEntry(MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T3),
        BucketEntry(MinerBucket.PRO_FUNDED, T5),
    ])
    assert [(r.bucket, r.start_ms, r.end_ms) for r in runs] == [
        ("SUBACCOUNT_CHALLENGE", T0, T1),
        ("SUBACCOUNT_FUNDED", T1, T3),
        ("PRO_CHALLENGE_FROM_STANDARD", T3, T5),
        ("PRO_FUNDED", T5, None),
    ]
    assert [r.account_start_ms for r in runs] == [T0, T1, T3, T3]  # PRO_FUNDED keeps the pro account


def test_a_stored_transition_reads_as_part_of_the_funded_segment():
    """The retired PRO_CHALLENGE_TRANSITION parses as SUBACCOUNT_FUNDED, in entries and in segment summaries."""
    runs = segment_runs([
        BucketEntry.from_dict({"bucket": "SUBACCOUNT_FUNDED", "start_time_ms": T1}),
        BucketEntry.from_dict({"bucket": "PRO_CHALLENGE_TRANSITION", "start_time_ms": T2}),
        BucketEntry(MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T2 + 1),
    ])
    assert [(r.bucket, r.start_ms, r.end_ms) for r in runs] == [
        ("SUBACCOUNT_FUNDED", T1, T2 + 1), ("PRO_CHALLENGE_FROM_STANDARD", T2 + 1, None)]
    legacy = BucketSegment.from_dict({"hotkey": HOTKEY, "bucket": "PRO_CHALLENGE_TRANSITION", "start_ms": T2})
    assert legacy.bucket == "SUBACCOUNT_FUNDED"


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
    assert (funded.account_size, funded.start_balance) == (100_000.0, 100_000.0)
    stats = manager.get_bucket_statistics(HOTKEY, bucket="SUBACCOUNT_FUNDED", include=["positions"])
    assert stats["segments"][0]["positions"] == []


def test_pro_funded_keeps_the_account_and_its_positions(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T3)
    manager._position_client.add(_position("pro_open", T4))

    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T5)

    pcf = _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T3)
    pro_funded = _segment(manager, MinerBucket.PRO_FUNDED, T5)
    assert pcf.end_ms == T5
    assert _copied(manager, pcf)["pro_open"]["close_ms"] == T5 - 1
    assert set(_copied(manager, pcf)) == {"pro_open"}
    assert _live_uuids(manager) == {"pro_open"}                 # closed, not archived
    assert HOTKEY not in manager._position_client.archived
    assert pro_funded.account_start_ms == T3
    stats = manager.get_bucket_statistics(HOTKEY, bucket="PRO_FUNDED", include=["positions"])
    assert [p["position_uuid"] for p in stats["segments"][0]["positions"]] == ["pro_open"]


# ── Close-outs land in the segment being left ─────────────────────────────────

@pytest.mark.parametrize("source, target, closes", [
    (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, True),        # account switch
    (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, True), # account switch
    (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED, True),       # keeps the account
    (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA, True),            # earning to earning
    (MinerBucket.PRO_FUNDED, MinerBucket.ELIMINATED, False),                        # forfeits instead
    (MinerBucket.CHALLENGE, MinerBucket.MAINCOMP, False),                           # never earning
])
def test_an_admin_move_closes_1ms_before_it_lands(manager, source, target, closes):
    manager.set_miner_bucket(HOTKEY, source, T0)
    manager._position_client.add(_position("open", T0 + 1))

    assert manager.admin_set_bucket(HOTKEY, target, T1)[0]

    left = _segment(manager, source, T0)
    assert left.end_ms == T1 and manager.miner_states[HOTKEY].current_bucket_start_ms == T1
    position = next(p for p in [*manager._position_client.live.get(HOTKEY, []),
                                *manager._position_client.archived.get(HOTKEY, [])])
    if closes:
        assert position.close_ms == T1 - 1 and left.contains(position.close_ms)
        manager._limit_order_client.cancel_limit_order.assert_called_with(HOTKEY, None, "ALL", T1)
    else:
        assert position.is_open_position
        manager._limit_order_client.cancel_limit_order.assert_not_called()


def test_closes_positions_from_covers_every_earning_boundary():
    assert MinerBucket.PLAGIARISM.closes_positions_from(MinerBucket.PRO_FUNDED)          # leaving earning
    assert MinerBucket.PRO_FUNDED.closes_positions_from(MinerBucket.PLAGIARISM)          # entering earning
    assert not MinerBucket.PRO_FUNDED.closes_positions_from(MinerBucket.PRO_FUNDED)      # no move
    assert not MinerBucket.ELIMINATED.closes_positions_from(MinerBucket.SUBACCOUNT_FUNDED)
    assert not MinerBucket.PRO_CHALLENGE_DIRECT.closes_positions_from(MinerBucket.SUBACCOUNT_CHALLENGE)


def test_a_closed_segment_keeps_its_snapshot_slice(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    for ms in (T0, T1 + 1000, T2 - 1000, T2 + 1000):
        ValiBkpUtils.log_snapshot(HOTKEY, AccountSnapshot(ms, 100_000.0, 100_000.0, 100_100.0).to_dict(),
                                  running_unit_tests=True)

    manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T2)

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
    manager._record_bucket_stats([HOTKEY], T1 + DAILY_MS)

    rows = manager._bucket_history.read_stats(_segment(manager, MinerBucket.PRO_CHALLENGE_DIRECT, T0))
    assert [r["day_ms"] for r in rows] == [T1, T1 + DAILY_MS]
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

    # The flag comes back: the segment is restored rather than duplicated
    manager.sync_plagiarism_miners([HOTKEY], T1)
    manager._reconcile_bucket_history([HOTKEY], T3)
    assert _segment(manager, MinerBucket.PLAGIARISM, T1).reverted_ms is None


def test_an_elimination_closes_the_segment_and_its_revert_reopens_the_bucket(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T0)
    manager._elimination_client.append_elimination_row.return_value = None
    from vali_objects.enums.elimination_reason_enum import EliminationReason
    manager.eliminate_hotkeys({HOTKEY: EliminationReason.MAX_TOTAL_DRAWDOWN}, T1)
    assert _segment(manager, MinerBucket.PRO_FUNDED, T0).end_ms == T1
    assert manager._bucket_history.get_live_segment(HOTKEY).bucket == MinerBucket.ELIMINATED.value

    # A reverted PRO_FUNDED elimination is marked reverted, so its stint resumes as one segment
    assert manager.revert_elimination(HOTKEY)
    pro_funded = _segment(manager, MinerBucket.PRO_FUNDED, T0)
    assert pro_funded.is_live
    assert _segment(manager, MinerBucket.ELIMINATED, T1).reverted_ms is not None


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


def test_a_corrupt_index_fails_loudly(manager):
    path = ValiBkpUtils.get_bucket_history_index_location(running_unit_tests=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not gzip")
    with pytest.raises(Exception):
        BucketHistoryStore(running_unit_tests=True)


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


def test_drawdown_snapshot_fields_come_from_the_live_state(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T0)
    manager.miner_states[HOTKEY].drawdown.eod_hwm = 1.05
    fields = manager.get_drawdown_snapshot_fields([HOTKEY, "unknown"])
    assert set(fields) == {HOTKEY}
    assert fields[HOTKEY]["eod_hwm"] == 1.05


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
    manager._entity_client.get_payout_scale.return_value = 0.6
    with patch("vali_objects.challenge_period.challengeperiod_manager.is_synthetic_hotkey", return_value=True):
        yield manager, calls


def test_leaving_an_earning_bucket_pins_a_payout_record(subaccount):
    manager, calls = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    manager._position_client.add(_position("standard_open", T1 + 1))
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T3)[0]
    manager._position_client.add(_position("pro_open", T4))
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T5)

    # Each record is computed from the positions copied into its own segment directory
    assert calls == [("SUBACCOUNT_FUNDED", ["standard_open"], T3), ("PRO_CHALLENGE_FROM_STANDARD", ["pro_open"], T5)]
    assert _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1).payout_record["payout_usd"] == 10.0
    pcf = _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T3)
    assert pcf.payout_record["end_ms"] == T5 and pcf.payout_scale == 0.6
    assert _segment(manager, MinerBucket.PRO_FUNDED, T5).payout_record is None


def test_elimination_writes_no_payout_record(subaccount):
    manager, calls = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T1)
    manager._position_client.add(_position("open", T1 + 1))

    assert manager.admin_set_bucket(HOTKEY, MinerBucket.ELIMINATED, T2)[0]

    assert calls == []
    assert _segment(manager, MinerBucket.PRO_FUNDED, T1).payout_record is None


def test_a_failed_record_holds_the_move_for_a_retry(subaccount):
    """D5: a stuck promotion is visible and retried; a missing record would drop the last partial week."""
    manager, _ = subaccount
    manager._entity_client.calculate_segment_payout_record.side_effect = None
    manager._entity_client.calculate_segment_payout_record.return_value = None
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)

    assert not manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T2)
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    assert not manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T2)[0]
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    assert _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1).payout_record is None

    manager._entity_client.calculate_segment_payout_record.return_value = {"payout_usd": 1.0, "settlements": []}
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T3)
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_FUNDED


def test_a_regular_miner_writes_no_record(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_ALPHA, T2)[0]
    manager._entity_client.calculate_segment_payout_record.assert_not_called()


def test_a_standard_miner_pins_a_record_at_its_records_scale(manager):
    """D4: a standard miner's segments get records too, priced at the scale fixed on its record."""
    from vali_objects.miner_account.standard_miner import StandardMinerAccount
    record = StandardMinerAccount(opened_at_ms=T0, opening_leverage_tier=2, asset_class="crypto", account_size=50_000.0,
                                  locked_theta=70.0, opening_price_usd=7.857, pro_account_size=125_000.0,
                                  trading_account_size=125_000.0, payout_scale=1.5)
    manager._miner_account_client.get_standard_account.return_value = record
    manager._entity_client.calculate_segment_payout_record.return_value = {"payout_usd": 5.0, "settlements": []}
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)

    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T2)

    segment = _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)
    assert segment.payout_record["payout_usd"] == 5.0
    assert segment.payout_scale == pytest.approx(1.5 * 50_000 / 125_000)


def test_a_reopened_segment_drops_its_record_but_keeps_its_overrides(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, T1)
    week = TimeUtil.ms_at_start_of_week(T1) + 7 * DAILY_MS
    assert manager.admin_set_payout_override(HOTKEY, T1, week, payout_usd=3.0, now_ms=T2)[0]
    assert manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_ALPHA, T2)[0]
    assert _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1).payout_record is not None

    manager.miner_states[HOTKEY].pop_bucket_entry(MinerBucket.SUBACCOUNT_ALPHA)
    manager._reconcile_bucket_history([HOTKEY], T3)

    funded = _segment(manager, MinerBucket.SUBACCOUNT_FUNDED, T1)
    assert funded.is_live and funded.payout_record is None
    assert funded.payout_overrides[str(week)]["payout_usd"] == 3.0


def test_the_record_rides_the_checkpoint(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T2)
    checkpoint = json.loads(json.dumps(manager.get_bucket_history_checkpoint_dict()))

    manager._bucket_history.clear_for_test()
    manager._bucket_history.sync_from_checkpoint(checkpoint)

    assert _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1).payout_record["end_ms"] == T2
    assert [r["start_ms"] for r in manager.get_payout_records(HOTKEY)] == [T1]


def test_admin_removes_and_recomputes_a_record(subaccount):
    manager, calls = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)
    manager._position_client.add(_position("pro_open", T1 + 1))
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T2)
    segment = _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)

    assert manager.admin_update_payout_record(HOTKEY, T1, remove=True)[0]
    assert segment.payout_record is None

    ok, _, record = manager.admin_update_payout_record(HOTKEY, T1, recompute=True)
    assert ok and record["payout_usd"] == 10.0
    assert calls[-1] == ("PRO_CHALLENGE_FROM_STANDARD", ["pro_open"], T2)

    # The live segment cannot be recomputed, an unknown one is not found, and one action is required
    assert manager.admin_update_payout_record(HOTKEY, T2, recompute=True)[0] is False
    assert manager.admin_update_payout_record(HOTKEY, T0, remove=True)[0] is False
    assert manager.admin_update_payout_record(HOTKEY, T1)[0] is False


# ── Closed weeks ──────────────────────────────────────────────────────────────

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


# ── Payout overrides ──────────────────────────────────────────────────────────

WEEK = TimeUtil.ms_at_start_of_week(T1) + 7 * DAILY_MS  # a Monday inside a segment starting at T1


@pytest.mark.parametrize("kwargs, fragment", [
    ({"payout_usd": -1.0}, "finite"),
    ({"payout_usd": float("nan")}, "finite"),
    ({"payout_usd": True}, "finite"),
    ({}, "exactly one"),
    ({"payout_usd": 1.0, "clear": True}, "exactly one"),
    ({"payout_usd": 1.0, "week_start_ms": WEEK + 1}, "Monday"),
    ({"payout_usd": 1.0, "week_start_ms": WEEK + 70 * DAILY_MS}, "has not started"),
    ({"payout_usd": 1.0, "week_start_ms": TimeUtil.ms_at_start_of_week(T1) - 7 * DAILY_MS}, "outside"),
    ({"payout_usd": 1.0, "segment_start_ms": T0}, "no segment"),
])
def test_an_override_is_refused_when_invalid(subaccount, kwargs, fragment):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_FUNDED, T1)
    args = {"segment_start_ms": T1, "week_start_ms": WEEK, **kwargs}
    ok, message, _ = manager.admin_set_payout_override(HOTKEY, args.pop("segment_start_ms"), args.pop("week_start_ms"),
                                                       now_ms=T3, **args)
    assert not ok and fragment in message
    assert _segment(manager, MinerBucket.PRO_FUNDED, T1).payout_overrides is None


def test_an_override_is_refused_on_a_non_earning_or_reverted_segment(subaccount):
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, T1)
    assert "not an earning" in manager.admin_set_payout_override(HOTKEY, T1, WEEK, 1.0, now_ms=T3)[1]
    manager.set_miner_bucket(HOTKEY, MinerBucket.PLAGIARISM, T2)
    manager._reconcile_bucket_history([HOTKEY], T2)
    manager.miner_states[HOTKEY].pop_bucket_entry(MinerBucket.PLAGIARISM)
    manager._reconcile_bucket_history([HOTKEY], T3)
    assert "reverted" in manager.admin_set_payout_override(HOTKEY, T2, TimeUtil.ms_at_start_of_week(T2),
                                                           1.0, now_ms=T3)[1]


def test_an_override_is_stored_cleared_and_survives_every_recompute(subaccount):
    """E29: recompute, remove, unfreezing, the closed-week freeze and checkpoint sync all leave it in place."""
    manager, _ = subaccount
    manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)
    segment = _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)
    ok, _, overrides = manager.admin_set_payout_override(HOTKEY, T1, WEEK, 0.0, note="paid offline", now_ms=T2)
    assert ok and overrides[str(WEEK)]["payout_usd"] == 0.0 and overrides[str(WEEK)]["note"] == "paid offline"
    assert manager.admin_set_payout_override(HOTKEY, T1, WEEK + 7 * DAILY_MS, 12.5, now_ms=T3)[0]  # the open week

    manager._entity_client.calculate_segment_closed_weeks.return_value = [{"start_ms": WEEK, "end_ms": WEEK + 1,
                                                                           "payout": 4.0}]
    manager._freeze_closed_weeks([HOTKEY], T3)
    assert manager.admin_drop_closed_weeks(HOTKEY)[0]
    assert manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, T4)
    assert manager.admin_update_payout_record(HOTKEY, T1, recompute=True)[0]
    assert manager.admin_update_payout_record(HOTKEY, T1, remove=True)[0]
    checkpoint = json.loads(json.dumps(manager.get_bucket_history_checkpoint_dict()))
    manager._bucket_history.clear_for_test()
    manager._bucket_history.sync_from_checkpoint(checkpoint)

    segment = _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1)
    assert set(segment.payout_overrides) == {str(WEEK), str(WEEK + 7 * DAILY_MS)}
    assert manager.get_payout_overrides(HOTKEY)[0]["start_ms"] == T1

    assert manager.admin_set_payout_override(HOTKEY, T1, WEEK, clear=True, now_ms=T5)[0]
    assert manager.admin_set_payout_override(HOTKEY, T1, WEEK + 7 * DAILY_MS, clear=True, now_ms=T5)[0]
    assert _segment(manager, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, T1).payout_overrides is None
    assert manager.get_payout_overrides(HOTKEY) == []


# ── Admin REST ────────────────────────────────────────────────────────────────

def _admin_server():
    server = object.__new__(ValidatorRestServer)
    server._get_api_key_safe = MagicMock(return_value="key")
    server.is_valid_api_key = MagicMock(return_value=True)
    server.can_access_tier = MagicMock(return_value=True)
    server._challenge_period_client = MagicMock()
    return server


@pytest.fixture
def payout_rest():
    server = _admin_server()
    server._challenge_period_client.get_payout_records.return_value = [{"start_ms": T1}]
    server._challenge_period_client.admin_update_payout_record.return_value = (True, "updated", {})
    server._challenge_period_client.get_payout_overrides.return_value = [{"start_ms": T1}]
    server._challenge_period_client.admin_set_payout_override.return_value = (True, "updated", {})
    app = Flask(__name__)
    app.route("/admin/payout-record/<hotkey>", methods=["GET"])(server.get_payout_records)
    app.route("/admin/payout-record/<hotkey>", methods=["POST"])(server.update_payout_record)
    app.route("/admin/payout-override/<hotkey>", methods=["GET"])(server.get_payout_overrides)
    app.route("/admin/payout-override/<hotkey>", methods=["POST"])(server.set_payout_override)
    return server, app.test_client()


def test_rest_payout_record_actions(payout_rest):
    server, client = payout_rest
    assert client.get(f"/admin/payout-record/{HOTKEY}").get_json()["payout_records"] == [{"start_ms": T1}]
    update = server._challenge_period_client.admin_update_payout_record
    for query, args in (("remove=true", (True, False)), ("recompute=true", (False, True))):
        assert client.post(f"/admin/payout-record/{HOTKEY}?segment_start_ms={T1}&{query}").status_code == 200
        update.assert_called_with(HOTKEY, T1, *args)


@pytest.mark.parametrize("query", ["remove=true", f"segment_start_ms={T1}", f"segment_start_ms={T1}&payout_usd=1",
                                   f"segment_start_ms={T1}&remove=true&recompute=true"])
def test_rest_payout_record_rejects_bad_params(payout_rest, query):
    server, client = payout_rest
    assert client.post(f"/admin/payout-record/{HOTKEY}?{query}").status_code == 400
    server._challenge_period_client.admin_update_payout_record.assert_not_called()


def test_rest_payout_override_actions(payout_rest):
    server, client = payout_rest
    assert client.get(f"/admin/payout-override/{HOTKEY}").get_json()["payout_overrides"] == [{"start_ms": T1}]
    override = server._challenge_period_client.admin_set_payout_override
    body = {"segment_start_ms": T1, "week_start_ms": WEEK}
    assert client.post(f"/admin/payout-override/{HOTKEY}", json={**body, "payout_usd": 12.5, "note": "x"}).status_code == 200
    override.assert_called_with(HOTKEY, T1, WEEK, 12.5, False, "x")
    assert client.post(f"/admin/payout-override/{HOTKEY}", json={**body, "clear": True}).status_code == 200
    override.assert_called_with(HOTKEY, T1, WEEK, None, True, None)


@pytest.mark.parametrize("body", [
    {"week_start_ms": WEEK, "payout_usd": 1}, {"segment_start_ms": T1, "payout_usd": 1},
    {"segment_start_ms": T1, "week_start_ms": WEEK}, {"segment_start_ms": T1, "week_start_ms": WEEK, "payout_usd": -1},
    {"segment_start_ms": T1, "week_start_ms": WEEK, "payout_usd": 1, "clear": True},
    {"segment_start_ms": "1", "week_start_ms": WEEK, "payout_usd": 1},
    {"segment_start_ms": T1, "week_start_ms": WEEK, "payout_usd": 1, "note": 5},
])
def test_rest_payout_override_rejects_bad_params(payout_rest, body):
    server, client = payout_rest
    assert client.post(f"/admin/payout-override/{HOTKEY}", json=body).status_code == 400
    server._challenge_period_client.admin_set_payout_override.assert_not_called()


@pytest.fixture
def reset_rest():
    server = _admin_server()
    for client in ("_position_client", "_perf_ledger_client", "_debt_ledger_client", "_elimination_client",
                   "_miner_account_client", "_entity_client"):
        setattr(server, client, MagicMock())
    server._closed_by_its_miner = MagicMock(return_value=None)
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


# ── Segment settlement (EntityManager) ────────────────────────────────────────

from entity_management.entity_manager import EntityManager  # noqa: E402
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger  # noqa: E402
from vali_objects.vali_dataclasses.order import Order  # noqa: E402

MON = 1_767_571_200_000  # 2026-01-05 00:00:00 UTC, a Monday
WK = 7 * DAILY_MS
CP_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS


def _closed(uuid, open_ms, close_ms, pnl, hotkey=HOTKEY):
    orders = [Order(order_type=OrderType.LONG, leverage=1.0, price=1.0, processed_ms=open_ms, order_uuid=f"{uuid}-o",
                    trade_pair=TradePair.EURUSD),
              Order(order_type=OrderType.FLAT, leverage=0.0, price=1.0, processed_ms=close_ms, order_uuid=f"{uuid}-c",
                    trade_pair=TradePair.EURUSD, realized_pnl=pnl)]
    return Position(miner_hotkey=hotkey, position_uuid=uuid, open_ms=open_ms, trade_pair=TradePair.EURUSD,
                    position_type=OrderType.FLAT, is_closed_position=True, close_ms=close_ms, orders=orders)


def _ledger(bucket, start_ms, end_ms, breached_weeks=(), hotkey=HOTKEY):
    cps, ms = [], start_ms + CP_MS
    while ms <= end_ms:
        breached = TimeUtil.ms_at_start_of_week(ms - 1) in breached_weeks
        cps.append(DebtCheckpoint(timestamp_ms=ms, challenge_period_status=bucket.value,
                                  weekly_penalty=0.0 if breached else 1.0))
        ms += CP_MS
    return DebtLedger(hotkey, checkpoints=cps)


def _entity(positions, ledger, scale=1.0, settled=None):
    em = object.__new__(EntityManager)
    em.running_unit_tests = True
    em.get_payout_scale = lambda _hk: scale
    em._position_client = SimpleNamespace(get_positions_for_one_hotkey=lambda hk, **_: list(positions))
    em._perf_ledger_client = SimpleNamespace(
        get_perf_ledger_for_hotkey=lambda hk: {hk: SimpleNamespace(get_checkpoint_at_time=lambda *_a: None)},
        get_frozen_ledgers=lambda: {})
    em._debt_ledger_client = SimpleNamespace(
        get_ledger=lambda _hk: ledger, get_settled_segments=lambda _hk: [], get_sealed_weeks=lambda _hk: {},
        record_settled_segment=lambda *args: settled.append(args) or True)
    return em


def _payouts(rows):
    return [(row["start_ms"], round(row["payout"], 6)) for row in rows]


def test_a_record_pays_every_week_of_its_segment_and_agrees_with_the_settled_segment():
    """Rule 5, a subaccount leaving SUBACCOUNT_FUNDED. Development settles only the payout week of the switch
    (earlier weeks were paid by the regular path); the record pins every week of the segment, and its last
    week matches what development settles."""
    hotkey, switch_ms = "entity_hk_1", MON + WK + 3 * DAILY_MS
    positions = [_closed("a", MON + DAILY_MS, MON + 2 * DAILY_MS, 100.0, hotkey),
                 _closed("b", MON + WK + DAILY_MS, switch_ms - 1, 50.0, hotkey)]
    settled = []
    em = _entity(positions, _ledger(MinerBucket.SUBACCOUNT_FUNDED, MON, switch_ms, hotkey=hotkey), settled=settled)
    em._challenge_period_client = SimpleNamespace(get_miner_bucket=lambda *_a: MinerBucket.SUBACCOUNT_FUNDED)
    em.get_entity_data = lambda _hk: SimpleNamespace(subaccounts={1: object()})
    segment = {"hotkey": hotkey, "bucket": "SUBACCOUNT_FUNDED", "start_ms": MON, "end_ms": None}

    record = em.calculate_segment_payout_record(hotkey, segment, [p.to_dict() for p in positions], switch_ms)
    assert em.settle_wound_down_segment(hotkey, "SUBACCOUNT_FUNDED", switch_ms)

    assert _payouts(record["settlements"]) == [(MON, 100.0), (MON + WK, 50.0)]
    assert record["payout_usd"] == 150.0 and record["end_ms"] == switch_ms
    assert all(row["payout_record"] for row in record["settlements"])
    development = settled[0]
    assert (development[1], development[5]) == (MON + WK, record["settlements"][-1]["payout"])


def test_a_standard_miners_record_matches_the_weeks_development_settles():
    """Rule 5, a standard miner: TestStandardMinerSettlement pins +100 last week and +50 this week at the standard
    size when the account leaves SUBACCOUNT_FUNDED. The record of the same segment pays the same weeks."""
    switch_ms = MON + WK + 3 * DAILY_MS
    positions = [_closed("a", MON + 6 * 3_600_000, MON + 7 * 3_600_000, 100.0),
                 _closed("b", MON + WK + 6 * 3_600_000, MON + WK + 7 * 3_600_000, 50.0)]
    em = _entity(positions, _ledger(MinerBucket.SUBACCOUNT_FUNDED, MON, switch_ms))
    segment = {"hotkey": HOTKEY, "bucket": "SUBACCOUNT_FUNDED", "start_ms": MON, "end_ms": None, "payout_scale": 0.6}
    record = em.calculate_segment_payout_record(HOTKEY, segment, positions, switch_ms)
    assert _payouts(record["settlements"]) == [(MON, 100.0), (MON + WK, 50.0)]
    # In the pro challenge the same segment is paid at its fixed scale, as development's stretch pricing does
    grow = {**segment, "bucket": "PRO_CHALLENGE_FROM_STANDARD"}
    record = em.calculate_segment_payout_record(HOTKEY, grow, positions, switch_ms)
    assert _payouts(record["settlements"]) == [(MON, 60.0), (MON + WK, 30.0)]


def test_frozen_weeks_keep_their_dollars_and_the_open_week_resumes_after_them():
    """D21: a week frozen once final is paid as frozen; a position edited afterwards does not restate it,
    and what follows is settled from the frozen row's balance and high-water mark."""
    first = _closed("a", MON + DAILY_MS, MON + 2 * DAILY_MS, 100.0)
    second = _closed("b", MON + WK + DAILY_MS, MON + WK + 2 * DAILY_MS, 150.0)
    positions = [first, second]
    em = _entity(positions, _ledger(MinerBucket.PRO_FUNDED, MON, MON + WK + 3 * DAILY_MS))
    segment = {"hotkey": HOTKEY, "bucket": "PRO_FUNDED", "start_ms": MON, "end_ms": None, "closed_weeks": None}

    assert em.calculate_segment_closed_weeks(HOTKEY, segment, MON + WK - 1) == []  # no week has closed
    full = em._debt_ledger_client.get_ledger
    em._debt_ledger_client.get_ledger = lambda _hk: _ledger(MinerBucket.PRO_FUNDED, MON, MON + WK - 1)
    assert em.calculate_segment_closed_weeks(HOTKEY, segment, MON + WK + DAILY_MS) is None  # not final yet
    em._debt_ledger_client.get_ledger = full
    frozen = em.calculate_segment_closed_weeks(HOTKEY, segment, MON + WK + DAILY_MS)
    assert _payouts(frozen) == [(MON, 100.0)] and frozen[0]["frozen_ms"] is not None
    segment["closed_weeks"] = frozen
    assert em.calculate_segment_closed_weeks(HOTKEY, segment, MON + WK + 2 * DAILY_MS) == []  # once per week

    first.orders[-1].realized_pnl = 40.0  # edited after the freeze
    record = em.calculate_segment_payout_record(HOTKEY, segment, positions, MON + WK + 3 * DAILY_MS)
    assert _payouts(record["settlements"]) == [(MON, 100.0), (MON + WK, 150.0)]
    assert record["settlements"][-1]["eow_balance"] == 250.0


def test_a_record_forfeits_the_escrow_it_still_holds():
    """E11: a breached PRO_FUNDED week defers its payout; leaving the bucket forfeits it."""
    positions = [_closed("a", MON + DAILY_MS, MON + 2 * DAILY_MS, 100.0)]
    em = _entity(positions, _ledger(MinerBucket.PRO_FUNDED, MON, MON + 3 * DAILY_MS, breached_weeks={MON}))
    segment = {"hotkey": HOTKEY, "bucket": "PRO_FUNDED", "start_ms": MON, "end_ms": None}
    record = em.calculate_segment_payout_record(HOTKEY, segment, positions, MON + 3 * DAILY_MS)
    row = record["settlements"][-1]
    assert (row["payout"], row["deferred"], row["deferred_forfeited"], row["deferred_balance"]) == (0.0, 100.0, 100.0, 0.0)


# ── Soft breach admin REST ────────────────────────────────────────────────────

@pytest.fixture
def soft_breach_rest():
    server = _admin_server()
    server._challenge_period_client.get_soft_breach_weeks.return_value = {WEEK: [WEEK + DAILY_MS]}
    server._challenge_period_client.admin_set_soft_breach_week.return_value = (True, "updated")
    app = Flask(__name__)
    app.route("/admin/soft-breach/<hotkey>", methods=["GET"])(server.get_soft_breach_weeks)
    app.route("/admin/soft-breach/<hotkey>", methods=["POST"])(server.set_soft_breach_week)
    return server, app.test_client()


def test_rest_soft_breach_actions(soft_breach_rest):
    server, client = soft_breach_rest
    assert client.get(f"/admin/soft-breach/{HOTKEY}").get_json()["soft_breach_weeks"] == {str(WEEK): [WEEK + DAILY_MS]}
    update = server._challenge_period_client.admin_set_soft_breach_week
    for query, days in ((f"days={WEEK},{WEEK + DAILY_MS}", [WEEK, WEEK + DAILY_MS]), ("days=", []),
                        ("remove=true", None)):
        assert client.post(f"/admin/soft-breach/{HOTKEY}?week_start_ms={WEEK}&{query}").status_code == 200
        update.assert_called_with(HOTKEY, WEEK, days)
    server._challenge_period_client.get_soft_breach_weeks.return_value = None
    assert client.get(f"/admin/soft-breach/{HOTKEY}").status_code == 404


@pytest.mark.parametrize("query", ["days=1", f"week_start_ms={WEEK}", f"week_start_ms={WEEK}&days=x"])
def test_rest_soft_breach_rejects_bad_params(soft_breach_rest, query):
    server, client = soft_breach_rest
    assert client.post(f"/admin/soft-breach/{HOTKEY}?{query}").status_code == 400
    server._challenge_period_client.admin_set_soft_breach_week.assert_not_called()
