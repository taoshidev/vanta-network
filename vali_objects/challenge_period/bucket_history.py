"""
Bucket history - one record per bucket stint, kept after the account it traded is wound down.

A miner's statistics are tracked on its live state: positions/, the perf and debt ledgers,
DrawdownStats and ProStats. A bucket change that switches account (MinerBucket.switches_account)
archives the positions and restarts all of that, so the previous stint can no longer be looked
at. This module keeps a copy of each stint, so its statistics stay available after the live
state has moved on.

  * `BucketSegment` - one stint in one MinerBucket, identified by (hotkey, bucket, start_ms). The
    SUBACCOUNT_FUNDED transition week is not a bucket: it stays in the same segment, which records
    when it ran (MinerAccount.transition_start_ms).
  * `segment_runs` - derives the segments a miner's bucket entries describe. The bucket entries
    are the source of truth, so a validator that only follows another validator's record
    (sync_challenge_period_data) derives the same segments.
  * Closed payout weeks - an earning segment's settlement rows are frozen onto it as each payout
    week closes, so the history paid out is never recomputed.
  * `BucketHistoryStore` - holds the summaries, persists them, and reads and writes the files of
    each segment directory (ValiBkpUtils.get_bucket_segment_dir):

        validation/miners/<hk>/buckets/<BUCKET>_<start_ms>/
            summary.json                          the BucketSegment record
            positions/<tp>/{open,closed}/<uuid>   copies of the positions traded in the stint
            snapshots.jsonl                       the account snapshots taken during the stint
            stats.jsonl                           one drawdown/ProStats row per UTC day while live

Only the summaries are shared through the validator checkpoint; the files are local to each
validator. They are only written into a miner directory that already exists: the elimination
manager reads a directory under validation/miners/ as a sign of a hotkey (handle_zombies), so this
record must never be what creates one. The summaries are also kept in one index file
(validation/bucket_history.json.gz), so they survive without the directory. Nothing here changes
the live state: positions/, archived_positions/ and snapshots.jsonl are read, never moved or
rewritten.

Every account switch is a bucket change, so segments never straddle two accounts. A segment
entered through an account switch starts a new account (account_start_ms = start_ms); one entered
without one (e.g. -> PRO_FUNDED) keeps the account of the segment before it, so account_start_ms
is inherited.
"""
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, Iterable, List, Optional

from shared_objects.log import logger
from time_util.time_util import TimeUtil
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

POSITIONS_DIR = "positions"
SUMMARY_FILE = "summary.json"
SNAPSHOTS_FILE = "snapshots.jsonl"
STATS_FILE = "stats.jsonl"


@dataclass
class BucketSegment:
    """One stint of a miner in one bucket."""
    hotkey: str
    bucket: str                         # MinerBucket value
    start_ms: int
    end_ms: Optional[int] = None        # None while live
    # Start of the account this segment trades (see the module docstring)
    account_start_ms: Optional[int] = None
    # SUBACCOUNT_FUNDED only: when the transition week started and ended (None while it runs)
    transition_start_ms: Optional[int] = None
    transition_end_ms: Optional[int] = None
    # Set when the bucket entries this segment came from were undone (popped or reverted)
    reverted_ms: Optional[int] = None
    # Created after the stint had already ended, e.g. bucket entries older than this record.
    # Holds the summary only: the positions and snapshots of the stint were never copied.
    backfilled: bool = False
    account_size: Optional[float] = None
    start_balance: Optional[float] = None
    end_balance: Optional[float] = None
    end_equity: Optional[float] = None
    final_drawdown: Optional[dict] = None
    final_pro_stats: Optional[dict] = None
    # Subaccount payout scale when the segment opened, so a later resize cannot reprice it
    payout_scale: Optional[float] = None
    # What the segment was owed when it left an earning bucket (EntityManager.calculate_segment_payout_record).
    # Pinned: the payout paths read it instead of recomputing the segment.
    payout_record: Optional[dict] = None
    # Settlement rows of the payout weeks that closed while the segment was live, oldest first
    # (EntityManager.calculate_segment_closed_weeks). Frozen: the payout paths resume after the
    # last one instead of recomputing them, so a later ledger rebuild cannot restate a closed week.
    closed_weeks: Optional[List[dict]] = None

    @property
    def key(self) -> tuple:
        return self.bucket, self.start_ms

    @property
    def is_live(self) -> bool:
        return self.end_ms is None and self.reverted_ms is None

    @property
    def dir_name(self) -> str:
        return f"{self.bucket}_{self.start_ms}"

    def contains(self, time_ms: int) -> bool:
        return self.start_ms <= time_ms and (self.end_ms is None or time_ms < self.end_ms)

    def update_from_final_state(self, end_balance=None, end_equity=None, drawdown=None, pro_stats=None) -> None:
        """Record the figures the segment's account closed on."""
        self.end_balance = end_balance
        self.end_equity = end_equity
        self.final_drawdown = drawdown
        self.final_pro_stats = pro_stats

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'BucketSegment':
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class SegmentRun:
    """A segment as derived from bucket entries, before it is matched against the store."""
    bucket: str
    start_ms: int
    end_ms: Optional[int]
    account_start_ms: int
    transition_start_ms: Optional[int] = None

    @property
    def key(self) -> tuple:
        return self.bucket, self.start_ms


def segment_runs(entries: Iterable[BucketEntry], transition_start_ms: Optional[int] = None) -> List[SegmentRun]:
    """Group bucket entries into segments: consecutive entries of the same bucket are one segment.
    Reverted entries are skipped, as MinerBucketState.bucket skips them.

    transition_start_ms is the miner's live transition week, which lands on the SUBACCOUNT_FUNDED
    run holding it, or on the live one when an elimination revert reopened the bucket after it.
    """
    runs: List[SegmentRun] = []
    account_start_ms = None
    for entry in sorted(entries, key=lambda e: e.start_time_ms):
        if entry.is_reverted:
            continue
        if runs and runs[-1].bucket == entry.bucket.value:
            continue
        if entry.bucket.switches_account or account_start_ms is None:
            account_start_ms = entry.start_time_ms
        if runs:
            runs[-1].end_ms = entry.start_time_ms
        runs.append(SegmentRun(bucket=entry.bucket.value, start_ms=entry.start_time_ms,
                               end_ms=None, account_start_ms=account_start_ms))
    if transition_start_ms is not None:
        for run in runs:
            if run.bucket != MinerBucket.SUBACCOUNT_FUNDED.value:
                continue
            if run.start_ms <= transition_start_ms and (run.end_ms is None or transition_start_ms < run.end_ms):
                run.transition_start_ms = transition_start_ms
            elif run.end_ms is None and transition_start_ms < run.start_ms:
                run.transition_start_ms = run.start_ms
    return runs


class BucketHistoryStore:
    """Bucket segment summaries plus the files of each segment directory.

    Pure storage: it never calls another service. ChallengePeriodManager decides when segments
    open and close (`reconcile`) and supplies what gets written into them.
    """

    def __init__(self, running_unit_tests: bool = False, enabled: bool = True):
        self.running_unit_tests = running_unit_tests
        # Disabled stores accept every call and persist nothing (backtesting)
        self.enabled = enabled
        self.segments: Dict[str, List[BucketSegment]] = {}
        # (hotkey, bucket, start_ms) -> UTC day of the segment's last stats row, read lazily
        self._last_stats_day: Dict[tuple, Optional[int]] = {}
        self._lock = threading.RLock()
        if self.enabled:
            self.load_from_disk()

    # ============================ Paths ============================

    def segment_dir(self, segment: BucketSegment) -> str:
        return ValiBkpUtils.get_bucket_segment_dir(
            segment.hotkey, segment.bucket, segment.start_ms, self.running_unit_tests)

    def _can_write(self, hotkey: str) -> bool:
        """True when the miner directory exists to write the segment files into."""
        return self.enabled and os.path.isdir(
            f"{ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)}{hotkey}")

    def _index_path(self) -> str:
        return ValiBkpUtils.get_bucket_history_index_location(running_unit_tests=self.running_unit_tests)

    # ============================ Persistence ============================

    def to_checkpoint_dict(self) -> dict:
        """Segment summaries, JSON-ready. Shared by the index file and the validator checkpoint."""
        with self._lock:
            return {
                hotkey: [segment.to_dict() for segment in segments]
                for hotkey, segments in self.segments.items()
            }

    def save_to_disk(self, hotkeys: Iterable[str] = ()) -> None:
        """Write the index, plus summary.json of every segment of the given hotkeys."""
        if not self.enabled:
            return
        with self._lock:
            for hotkey in hotkeys:
                if not self._can_write(hotkey):
                    continue
                for segment in self.segments.get(hotkey, []):
                    ValiBkpUtils.write_file(os.path.join(self.segment_dir(segment), SUMMARY_FILE), segment.to_dict())
            ValiBkpUtils.write_compressed_json(self._index_path(), {
                "format_version": "1.0",
                "last_update_ms": TimeUtil.now_in_millis(),
                "segments": self.to_checkpoint_dict(),
            })

    def load_from_disk(self) -> int:
        path = self._index_path()
        if not os.path.exists(path):
            logger.info("[BUCKET_HISTORY] No existing bucket history found")
            return 0
        try:
            data = ValiBkpUtils.read_compressed_json(path)
        except Exception as e:
            # Losing this file loses the only record of wound-down accounts, so fail loudly
            logger.error(f"[BUCKET_HISTORY] Failed to read {path}: {e}")
            raise
        with self._lock:
            self.segments = {
                hotkey: sorted((BucketSegment.from_dict(s) for s in segments), key=lambda s: s.start_ms)
                for hotkey, segments in (data.get("segments") or {}).items()
            }
        total = sum(len(s) for s in self.segments.values())
        logger.info(f"[BUCKET_HISTORY] Loaded {total} bucket segments across {len(self.segments)} hotkeys")
        return total

    def sync_from_checkpoint(self, checkpoint_dict: dict) -> dict:
        """Merge the validator checkpoint's segment summaries into this store.

        The checkpoint wins on every segment both sides hold. Segments only this validator holds
        are kept, since the checkpoint lags behind.
        """
        stats = {'segments_added': 0, 'segments_replaced': 0, 'errors': 0}
        if not isinstance(checkpoint_dict, dict):
            logger.warning(f"[BUCKET_HISTORY] Ignoring bucket history checkpoint of type {type(checkpoint_dict)}")
            return stats

        changed_hotkeys = set()
        with self._lock:
            for hotkey, segments in checkpoint_dict.items():
                local = self.segments.setdefault(hotkey, [])
                by_key = {s.key: i for i, s in enumerate(local)}
                for raw in (segments or []):
                    try:
                        candidate = BucketSegment.from_dict(raw)
                    except Exception as e:
                        logger.warning(f"[BUCKET_HISTORY] Skipping malformed segment for {hotkey}: {e}")
                        stats['errors'] += 1
                        continue
                    index = by_key.get(candidate.key)
                    if index is None:
                        local.append(candidate)
                        by_key[candidate.key] = len(local) - 1
                        stats['segments_added'] += 1
                        changed_hotkeys.add(hotkey)
                    elif local[index].to_dict() != candidate.to_dict():
                        local[index] = candidate
                        stats['segments_replaced'] += 1
                        changed_hotkeys.add(hotkey)
                local.sort(key=lambda s: s.start_ms)
                if not local:
                    del self.segments[hotkey]

            if changed_hotkeys:
                self.save_to_disk(changed_hotkeys)
                logger.info(f"[BUCKET_HISTORY] Autosync merged bucket history: {stats}")
        return stats

    def clear_for_test(self) -> None:
        assert self.running_unit_tests, "clear_for_test is only for unit tests"
        with self._lock:
            hotkeys = list(self.segments)
            self.segments = {}
            self._last_stats_day = {}
            for hotkey in hotkeys:
                shutil.rmtree(ValiBkpUtils.get_bucket_history_dir(hotkey, self.running_unit_tests), ignore_errors=True)
            if os.path.exists(self._index_path()):
                os.remove(self._index_path())

    # ============================ Segment lifecycle ============================

    def reconcile(
        self,
        hotkey: str,
        entries: Iterable[BucketEntry],
        now_ms: int,
        on_open: Callable[[BucketSegment], None],
        on_close: Callable[[BucketSegment], None],
        save: bool = True,
        transition_start_ms: Optional[int] = None,
    ) -> bool:
        """Bring the hotkey's segments in line with its bucket entries. transition_start_ms is the
        live transition week, if any; a segment keeps the one it recorded once the account moves on.

        on_open runs for a segment that has just become the live one, on_close for one that has
        just ended (not for a backfilled segment, which never was live here). Both run under the
        store lock with the segment already updated, and may fill in its fields. Returns True
        when anything changed. save=False leaves saving to the caller, for a batch of hotkeys.
        """
        runs = segment_runs(entries, transition_start_ms)
        wanted = {run.key for run in runs}
        changed = False
        with self._lock:
            segments = self.segments.setdefault(hotkey, [])
            by_key = {s.key: s for s in segments}

            # Entries that were popped or reverted take their segment with them
            for segment in segments:
                if segment.key not in wanted and segment.reverted_ms is None:
                    segment.reverted_ms = now_ms
                    changed = True

            for run in runs:
                segment = by_key.get(run.key)
                if segment is None:
                    segment = BucketSegment(
                        hotkey=hotkey, bucket=run.bucket, start_ms=run.start_ms, end_ms=run.end_ms,
                        account_start_ms=run.account_start_ms,
                        transition_start_ms=run.transition_start_ms,
                        transition_end_ms=run.end_ms if run.transition_start_ms is not None else None,
                        backfilled=run.end_ms is not None,
                    )
                    segments.append(segment)
                    by_key[run.key] = segment
                    changed = True
                    if segment.is_live:
                        on_open(segment)
                    continue

                if segment.reverted_ms is not None:
                    # Its entries came back (e.g. a plagiarism flag set, cleared, then set again)
                    segment.reverted_ms = None
                    changed = True
                transition_start = (run.transition_start_ms if run.transition_start_ms is not None
                                    else segment.transition_start_ms)
                derived = (run.account_start_ms, transition_start, run.end_ms if transition_start is not None else None)
                if (segment.account_start_ms, segment.transition_start_ms, segment.transition_end_ms) != derived:
                    segment.account_start_ms, segment.transition_start_ms, segment.transition_end_ms = derived
                    changed = True
                if segment.end_ms != run.end_ms:
                    was_live = segment.end_ms is None
                    segment.end_ms = run.end_ms
                    changed = True
                    if was_live and run.end_ms is not None:
                        on_close(segment)
                    elif run.end_ms is None:
                        # Reopened: the entry that ended it was undone, and so was its payout
                        segment.payout_record = None
                        on_open(segment)

            segments.sort(key=lambda s: s.start_ms)
            if changed and save:
                self.save_to_disk([hotkey])
        return changed

    def save_segment(self, segment: BucketSegment) -> None:
        self.save_to_disk([segment.hotkey])

    # ============================ Segment files ============================

    def write_positions(self, segment: BucketSegment, positions: Iterable) -> int:
        """Copy positions into the segment directory. Returns how many were written."""
        if not self._can_write(segment.hotkey):
            return 0
        root = os.path.join(self.segment_dir(segment), POSITIONS_DIR)
        count = 0
        for position in positions:
            ValiBkpUtils.copy_position_to_dir(position, root)
            count += 1
        return count

    def write_snapshots(self, segment: BucketSegment, snapshot_rows: Iterable[dict]) -> None:
        """Replace the segment's snapshots.jsonl with the given rows."""
        if not self._can_write(segment.hotkey):
            return
        path = os.path.join(self.segment_dir(segment), SNAPSHOTS_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for row in snapshot_rows:
                f.write(json.dumps(row, default=str) + '\n')

    def append_stats_row(self, segment: BucketSegment, row: dict, day_ms: int) -> bool:
        """Append the day's stats row unless the segment already has one for that UTC day."""
        if not self._can_write(segment.hotkey):
            return False
        path = os.path.join(self.segment_dir(segment), STATS_FILE)
        cache_key = (segment.hotkey, *segment.key)
        if cache_key not in self._last_stats_day:
            last = self._read_jsonl(path, limit=1)
            self._last_stats_day[cache_key] = last[-1].get("day_ms") if last else None
        if self._last_stats_day[cache_key] == day_ms:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps({"day_ms": day_ms, **row}, default=str) + '\n')
        self._last_stats_day[cache_key] = day_ms
        return True

    def read_positions(self, segment: BucketSegment) -> List[dict]:
        root = os.path.join(self.segment_dir(segment), POSITIONS_DIR)
        if not os.path.isdir(root):
            return []
        positions = []
        for path in ValiBkpUtils.get_all_files_in_dir(root):
            try:
                with open(path, 'r') as f:
                    positions.append(json.load(f))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"[BUCKET_HISTORY] Unreadable position copy {path}: {e}")
        return positions

    def read_snapshots(self, segment: BucketSegment, limit: int | None = None) -> List[dict]:
        return self._read_jsonl(os.path.join(self.segment_dir(segment), SNAPSHOTS_FILE), limit)

    def read_stats(self, segment: BucketSegment, limit: int | None = None) -> List[dict]:
        return self._read_jsonl(os.path.join(self.segment_dir(segment), STATS_FILE), limit)

    @staticmethod
    def _read_jsonl(path: str, limit: int | None = None) -> List[dict]:
        """Rows of a .jsonl file, oldest first; limit keeps the last rows."""
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:] if limit else rows

    # ============================ Reads ============================

    def get_segments(self, hotkey: str) -> List[BucketSegment]:
        with self._lock:
            return list(self.segments.get(hotkey, []))

    def get_segment(self, hotkey: str, start_ms: int, bucket: str | None = None) -> BucketSegment | None:
        with self._lock:
            for segment in self.segments.get(hotkey, []):
                if segment.start_ms == start_ms and (bucket is None or segment.bucket == bucket):
                    return segment
        return None

    def get_live_segment(self, hotkey: str) -> BucketSegment | None:
        with self._lock:
            for segment in reversed(self.segments.get(hotkey, [])):
                if segment.is_live:
                    return segment
        return None
