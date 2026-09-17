"""
Weekly Seal Ledger - the write-once record of how a payout week was settled.

A payout week is settled money. Everything the payout paths need in order to decide whether a week
was paid, withheld or forfeited is derived from the penalty and debt ledgers, and those are rebuilt
from the perf ledger on a schedule and deleted outright by several admin operations:

  * the 48h full penalty rebuild recomputes historical checkpoints against today's ratcheted
    drawdown and today's account size
  * `delete_debt_ledger` drops the debt *and* penalty ledgers, and runs on revert-elimination,
    /admin/reset and every account switch
  * `get_payout_scale` reads the subaccount's sizes live, so resizing an account would rewrite the
    scale that applied to every past week - the sealed ratio is what both payout paths gate on
    instead, so only the bucket a checkpoint actually held still comes from the ledgers

Any one of those can silently move a closed week between "paid" and "withheld". This ledger pins the
classification - not the dollar amount, which legitimately moves when positions are corrected - so a
rebuild reproduces the same decision. A record is written once and never overwritten; a later build
that disagrees logs loudly instead, which is the canary for a rebuild that would have changed
history.

Deliberate corrections go through `unseal()` (exposed as POST /admin/unseal-week), never through a
silent recompute.

The one thing that does overwrite a record is `sync_from_checkpoint()`: the records ride in the
validator checkpoint so every validator settles a week the same way, and there the checkpoint's
verdict wins over a local one that disagrees. That is the same guarantee one rung up - a week is
pinned across rebuilds *and* across validators - rather than an exception to it.

This file holds a second, narrower record: `SettledSegment`. A sealed week pins a classification
because the inputs survive and get replayed against it. Stored when account needs history wiped due
to promotion, but still has pending payouts pre-promotion.

Both records live in one file for one reason: this ledger is already the thing that survives
`delete_debt_ledger`, `_switch_account`, revert-elimination and /admin/reset.

Deliberate corrections to a segment go through `amend_settled()` / `remove_settled()` (exposed as
POST /admin/settled-segment), never through a silent recompute.
"""
import os
import threading
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional

from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from shared_objects.log import logger


@dataclass
class SealedWeek:
    """How one Monday-anchored payout week was classified, as settled."""
    week_start_ms: int
    # 0.0 withheld, 1.0 paid - the min over the week, before the per-bucket gate the payout
    # paths apply, so a week a subaccount was promoted through seals one value for both halves
    weekly_penalty: float
    payout_scale: float             # the account's standard/pro ratio, ungated by bucket
    track: str                      # WeekTrack name: NO_DATA / ON_TRACK / OFF_TRACK
    first_earning_ms: Optional[int]
    sealed_ms: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SealedWeek':
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def matches(self, weekly_penalty: float, payout_scale: float, track: str) -> bool:
        return (self.weekly_penalty == weekly_penalty
                and self.payout_scale == payout_scale
                and self.track == track)


@dataclass
class SettledSegment:
    """A stretch of a payout week that was settled early because the account was wound down.

    Unlike SealedWeek, this pins the *amount*. The account switch destroys the positions and the
    ledgers the amount would be recomputed from, so there is nothing left to replay - the dollars
    are the only thing that can survive. `segment_end_ms` is the moment of the switch, and is what
    a correction names the record by; the money itself is keyed on (hotkey, week, bucket).
    """
    week_start_ms: int          # Monday 00:00 UTC, TimeUtil.ms_at_start_of_week
    segment_start_ms: int
    segment_end_ms: int
    bucket: str                 # the bucket wound down, e.g. PRO_CHALLENGE_TRANSITION
    payout_usd: float           # settled - what both payout paths add
    gross_payout_usd: float     # pre-penalty, for audit
    weekly_penalty: float
    payout_scale: float
    recorded_ms: int
    amended_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SettledSegment':
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def matches(self, payout_usd: float, gross_payout_usd: float) -> bool:
        return (self.payout_usd == payout_usd
                and self.gross_payout_usd == gross_payout_usd)


class WeeklySealLedger:
    """Per-hotkey settled-payout records, persisted independently of every other ledger.

    Two maps, both keyed by hotkey:
      * `sealed`   - week_start_ms -> SealedWeek, how a closed week was classified
      * `settled`  - a list of SettledSegment, payouts settled early by an account switch

    Deliberately NOT deleted by delete_debt_ledger, _switch_account, revert-elimination or
    /admin/reset: it is the audit trail those operations would otherwise destroy, and for
    `settled` it is the only surviving record of the money itself.
    """

    def __init__(self, running_unit_tests: bool = False):
        self.running_unit_tests = running_unit_tests
        self.sealed: Dict[str, Dict[int, SealedWeek]] = {}
        self.settled: Dict[str, List[SettledSegment]] = {}
        # The debt ledger daemon seals while an autosync RPC call may be merging a peer's records.
        self._lock = threading.RLock()
        self.load_from_disk()

    # ============================ Persistence ============================

    def _get_path(self) -> str:
        return ValiBkpUtils.get_weekly_seal_ledger_file_location(
            running_unit_tests=self.running_unit_tests
        )

    def to_checkpoint_dict(self) -> dict:
        """The sealed and settled records, JSON-ready. Shared by the on-disk file and the
        validator checkpoint, so what autosync ships is exactly what a validator persists."""
        with self._lock:
            return {
                "sealed": {
                    hotkey: {str(week_start_ms): week.to_dict() for week_start_ms, week in weeks.items()}
                    for hotkey, weeks in self.sealed.items()
                },
                "settled": {
                    hotkey: [segment.to_dict() for segment in segments]
                    for hotkey, segments in self.settled.items()
                },
            }

    def save_to_disk(self) -> None:
        with self._lock:
            data = {
                "format_version": "1.1",
                "last_update_ms": TimeUtil.now_in_millis(),
                **self.to_checkpoint_dict(),
            }
            ValiBkpUtils.write_compressed_json(self._get_path(), data)

    def load_from_disk(self) -> int:
        path = self._get_path()
        if not os.path.exists(path):
            logger.info("[WEEKLY_SEAL] No existing weekly seal ledger found")
            return 0

        try:
            data = ValiBkpUtils.read_compressed_json(path)
        except Exception as e:
            # Losing this file means losing the guarantee, so it must be loud rather than silent.
            logger.error(f"[WEEKLY_SEAL] Failed to read {path}: {e}")
            raise

        for hotkey, weeks in (data.get("sealed") or {}).items():
            self.sealed[hotkey] = {
                int(week_start_ms): SealedWeek.from_dict(week)
                for week_start_ms, week in weeks.items()
            }

        for hotkey, segments in (data.get("settled") or {}).items():
            self.settled[hotkey] = sorted(
                (SettledSegment.from_dict(segment) for segment in segments),
                key=lambda s: s.segment_end_ms,
            )

        total = sum(len(w) for w in self.sealed.values())
        total_settled = sum(len(s) for s in self.settled.values())
        logger.info(
            f"[WEEKLY_SEAL] Loaded {total} sealed weeks across {len(self.sealed)} hotkeys, "
            f"{total_settled} settled segments across {len(self.settled)} hotkeys"
        )
        return total

    # ============================ Reads ============================

    def get_sealed(self, hotkey: str) -> Dict[int, SealedWeek]:
        """Sealed weeks for one hotkey, keyed by Monday 00:00 UTC."""
        with self._lock:
            return dict(self.sealed.get(hotkey, {}))

    def is_sealed(self, hotkey: str, week_start_ms: int) -> bool:
        with self._lock:
            return week_start_ms in self.sealed.get(hotkey, {})

    def get_settled(self, hotkey: str) -> List[SettledSegment]:
        """Segments settled early for one hotkey, oldest first."""
        with self._lock:
            return list(self.settled.get(hotkey, []))

    # ============================ Writes ============================

    def seal(
        self,
        hotkey: str,
        week_start_ms: int,
        *,
        weekly_penalty: float,
        payout_scale: float,
        track: str,
        first_earning_ms: Optional[int] = None,
    ) -> bool:
        """Record a closed week's classification. Write-once.

        Returns True when a new record was written. A second seal for the same week never
        overwrites; it logs at warning level when the new values disagree with what is stored,
        which is the signal that a rebuild would have rewritten settled history.
        """
        with self._lock:
            existing = self.sealed.get(hotkey, {}).get(week_start_ms)
            if existing is not None:
                if not existing.matches(weekly_penalty, payout_scale, track):
                    logger.warning(
                        f"[WEEKLY_SEAL] Rebuild disagrees with sealed week for {hotkey} at "
                        f"{TimeUtil.millis_to_formatted_date_str(week_start_ms)}: sealed "
                        f"(penalty={existing.weekly_penalty}, scale={existing.payout_scale}, "
                        f"track={existing.track}) vs recomputed "
                        f"(penalty={weekly_penalty}, scale={payout_scale}, track={track}). "
                        f"Keeping the sealed values."
                    )
                return False

            self.sealed.setdefault(hotkey, {})[week_start_ms] = SealedWeek(
                week_start_ms=week_start_ms,
                weekly_penalty=weekly_penalty,
                payout_scale=payout_scale,
                track=track,
                first_earning_ms=first_earning_ms,
                sealed_ms=TimeUtil.now_in_millis(),
            )
            return True

    def unseal(self, hotkey: str, week_start_ms: int) -> bool:
        """Drop one sealed week so the next build reseals it. For deliberate corrections only."""
        with self._lock:
            weeks = self.sealed.get(hotkey)
            if not weeks or week_start_ms not in weeks:
                return False
            del weeks[week_start_ms]
            if not weeks:
                del self.sealed[hotkey]
            logger.warning(
                f"[WEEKLY_SEAL] Unsealed {hotkey} week "
                f"{TimeUtil.millis_to_formatted_date_str(week_start_ms)}; it will be recomputed"
            )
            self.save_to_disk()
            return True

    def record_settled(
        self,
        hotkey: str,
        *,
        week_start_ms: int,
        segment_start_ms: int,
        segment_end_ms: int,
        bucket: str,
        payout_usd: float,
        gross_payout_usd: float,
        weekly_penalty: float,
        payout_scale: float,
    ) -> bool:
        """Pin a payout settled early by an account switch.

        Write-once on (hotkey, week_start_ms, bucket), deliberately not on segment_end_ms: a switch
        that fails after this record is written leaves the miner in the bucket it was leaving, and
        the move is retried - by the grace-period sweep, or by the entity asking again - with a
        fresh `current_time_ms`. The positions and ledgers the amount came from are still intact at
        that point, so the retry recomputes the same dollars; keying on the timestamp would file
        them a second time and pay the week twice.

        Returns True when a new record was written. A repeat never overwrites, and logs at warning
        level when the new values disagree with what is stored.

        Saves to disk immediately: the caller is mid-wipe, and everything this record describes is
        about to be deleted.
        """
        with self._lock:
            segments = self.settled.setdefault(hotkey, [])
            existing = next(
                (s for s in segments if s.week_start_ms == week_start_ms and s.bucket == bucket), None
            )
            if existing is not None:
                if not existing.matches(payout_usd, gross_payout_usd):
                    logger.warning(
                        f"[WEEKLY_SEAL] Re-settle disagrees with the settled segment for {hotkey} "
                        f"in {bucket} for the week of "
                        f"{TimeUtil.millis_to_formatted_date_str(week_start_ms)}: settled at "
                        f"{TimeUtil.millis_to_formatted_date_str(existing.segment_end_ms)} "
                        f"(payout={existing.payout_usd}, gross={existing.gross_payout_usd}) vs "
                        f"recomputed at {TimeUtil.millis_to_formatted_date_str(segment_end_ms)} "
                        f"(payout={payout_usd}, gross={gross_payout_usd}). "
                        f"Keeping the settled values."
                    )
                return False

            segments.append(SettledSegment(
                week_start_ms=week_start_ms,
                segment_start_ms=segment_start_ms,
                segment_end_ms=segment_end_ms,
                bucket=bucket,
                payout_usd=payout_usd,
                gross_payout_usd=gross_payout_usd,
                weekly_penalty=weekly_penalty,
                payout_scale=payout_scale,
                recorded_ms=TimeUtil.now_in_millis(),
            ))
            segments.sort(key=lambda s: s.segment_end_ms)
            self.save_to_disk()
            logger.info(
                f"[WEEKLY_SEAL] Settled {hotkey} segment in {bucket} ending "
                f"{TimeUtil.millis_to_formatted_date_str(segment_end_ms)}: ${payout_usd:.2f}"
            )
            return True

    def amend_settled(self, hotkey: str, segment_end_ms: int, payout_usd: float) -> bool:
        """Correct a settled segment's amount. For deliberate corrections only.

        Nothing recomputes this record, so a wrong figure stays wrong until it is amended here.
        """
        with self._lock:
            segment = next(
                (s for s in self.settled.get(hotkey, []) if s.segment_end_ms == segment_end_ms), None
            )
            if segment is None:
                return False
            logger.warning(
                f"[WEEKLY_SEAL] Amended {hotkey} settled segment ending "
                f"{TimeUtil.millis_to_formatted_date_str(segment_end_ms)}: "
                f"${segment.payout_usd:.2f} -> ${payout_usd:.2f}"
            )
            segment.payout_usd = payout_usd
            segment.amended_ms = TimeUtil.now_in_millis()
            self.save_to_disk()
            return True

    def remove_settled(self, hotkey: str, segment_end_ms: int) -> bool:
        """Drop a settled segment entirely. For deliberate corrections only.

        Unlike `unseal`, nothing rebuilds this record afterwards - the data it was derived from is
        gone - so removing it drops the payout for good.
        """
        with self._lock:
            segments = self.settled.get(hotkey)
            if not segments:
                return False
            remaining = [s for s in segments if s.segment_end_ms != segment_end_ms]
            if len(remaining) == len(segments):
                return False
            if remaining:
                self.settled[hotkey] = remaining
            else:
                del self.settled[hotkey]
            logger.warning(
                f"[WEEKLY_SEAL] Removed {hotkey} settled segment ending "
                f"{TimeUtil.millis_to_formatted_date_str(segment_end_ms)}; it will not be rebuilt"
            )
            self.save_to_disk()
            return True

    def clear_for_test(self) -> None:
        """Drop every record, in memory and on disk. Unit tests only.

        This ledger outlives every other one by design, including across process restarts, so a
        test that seals a week would otherwise pin that decision for every later test and run.
        """
        assert self.running_unit_tests, "Weekly seal records can only be cleared in unit tests"
        with self._lock:
            self.sealed.clear()
            self.settled.clear()
            path = self._get_path()
            if os.path.exists(path):
                os.remove(path)

    # ============================ Autosync ============================

    def sync_from_checkpoint(self, checkpoint_dict: dict) -> dict:
        """Merge the validator checkpoint's seal records into this ledger.

        The checkpoint wins on every record both sides hold. Agreeing on what was sealed is the
        point, and a validator that classified a closed week differently is exactly the divergence
        this repairs - so it adopts the checkpoint's verdict and logs the disagreement rather than
        keeping its own, which is the opposite of how a local rebuild is treated by `seal()`.

        Records only this validator holds are kept. The checkpoint is time-lagged, so a week sealed
        since it was written would otherwise be dropped and resealed against today's rebuilt
        ledgers - the recompute the seal exists to prevent.
        """
        stats = {'sealed_added': 0, 'sealed_replaced': 0, 'settled_added': 0,
                 'settled_replaced': 0, 'errors': 0}

        if not isinstance(checkpoint_dict, dict):
            logger.warning(f"[WEEKLY_SEAL] Ignoring weekly seal checkpoint of type {type(checkpoint_dict)}")
            return stats

        with self._lock:
            for hotkey, weeks in (checkpoint_dict.get('sealed') or {}).items():
                local_weeks = self.sealed.setdefault(hotkey, {})
                for raw_week_start_ms, week in (weeks or {}).items():
                    try:
                        week_start_ms = int(raw_week_start_ms)
                        candidate = SealedWeek.from_dict(week)
                        candidate.week_start_ms = week_start_ms
                    except Exception as e:
                        logger.warning(
                            f"[WEEKLY_SEAL] Skipping malformed sealed week for {hotkey} "
                            f"at {raw_week_start_ms}: {e}"
                        )
                        stats['errors'] += 1
                        continue

                    existing = local_weeks.get(week_start_ms)
                    if existing is None:
                        local_weeks[week_start_ms] = candidate
                        stats['sealed_added'] += 1
                    elif not existing.matches(candidate.weekly_penalty, candidate.payout_scale,
                                              candidate.track):
                        logger.warning(
                            f"[WEEKLY_SEAL] Autosync disagrees with sealed week for {hotkey} at "
                            f"{TimeUtil.millis_to_formatted_date_str(week_start_ms)}: local "
                            f"(penalty={existing.weekly_penalty}, scale={existing.payout_scale}, "
                            f"track={existing.track}) vs checkpoint "
                            f"(penalty={candidate.weekly_penalty}, scale={candidate.payout_scale}, "
                            f"track={candidate.track}). Adopting the checkpoint's values."
                        )
                        local_weeks[week_start_ms] = candidate
                        stats['sealed_replaced'] += 1
                if not local_weeks:
                    del self.sealed[hotkey]

            for hotkey, segments in (checkpoint_dict.get('settled') or {}).items():
                local_segments = self.settled.setdefault(hotkey, [])
                # Keyed exactly as record_settled is, so a retried account switch does not file twice
                by_key = {(s.week_start_ms, s.bucket): s for s in local_segments}
                changed = False
                for segment in (segments or []):
                    try:
                        candidate = SettledSegment.from_dict(segment)
                    except Exception as e:
                        logger.warning(f"[WEEKLY_SEAL] Skipping malformed settled segment for {hotkey}: {e}")
                        stats['errors'] += 1
                        continue

                    key = (candidate.week_start_ms, candidate.bucket)
                    existing = by_key.get(key)
                    if existing is None:
                        local_segments.append(candidate)
                        by_key[key] = candidate
                        stats['settled_added'] += 1
                        changed = True
                    elif not existing.matches(candidate.payout_usd, candidate.gross_payout_usd):
                        logger.warning(
                            f"[WEEKLY_SEAL] Autosync disagrees with the settled segment for "
                            f"{hotkey} in {candidate.bucket} for the week of "
                            f"{TimeUtil.millis_to_formatted_date_str(candidate.week_start_ms)}: "
                            f"local (payout={existing.payout_usd}, gross={existing.gross_payout_usd}) "
                            f"vs checkpoint (payout={candidate.payout_usd}, "
                            f"gross={candidate.gross_payout_usd}). Adopting the checkpoint's values."
                        )
                        local_segments[local_segments.index(existing)] = candidate
                        by_key[key] = candidate
                        stats['settled_replaced'] += 1
                        changed = True
                if changed:
                    local_segments.sort(key=lambda s: s.segment_end_ms)
                if not local_segments:
                    del self.settled[hotkey]

            if any(stats[k] for k in ('sealed_added', 'sealed_replaced', 'settled_added', 'settled_replaced')):
                self.save_to_disk()
                logger.info(f"[WEEKLY_SEAL] Autosync merged weekly seal records: {stats}")

        return stats
