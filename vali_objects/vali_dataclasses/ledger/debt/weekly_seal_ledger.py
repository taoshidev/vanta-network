"""
Weekly Seal Ledger - the write-once record of how a payout week was classified.

A payout week is settled money. Everything the payout paths need in order to decide whether a week
was paid, withheld or forfeited is derived from the penalty and debt ledgers, and those are rebuilt
from the perf ledger on a schedule and deleted outright by several admin operations:

  * the 48h full penalty rebuild recomputes historical checkpoints against today's ratcheted
    drawdown and today's account size
  * `delete_debt_ledger` drops the debt *and* penalty ledgers, and runs on revert-elimination,
    /admin/reset and every account switch
  * `get_payout_scale` reads the subaccount's sizes live, so resizing an account rewrites the scale
    that applied to every past week

Any one of those can silently move a closed week between "paid" and "withheld". This ledger pins the
classification - not the dollar amount, which legitimately moves when positions are corrected - so a
rebuild reproduces the same decision. A record is written once and never overwritten; a later build
that disagrees logs loudly instead, which is the canary for a rebuild that would have changed
history.

Deliberate corrections go through `unseal()` (exposed as POST /admin/unseal-week), never through a
silent recompute.
"""
import gzip
import json
import os
import shutil
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from shared_objects.log import logger


@dataclass
class SealedWeek:
    """How one Monday-anchored payout week was classified, as settled."""
    week_start_ms: int
    weekly_penalty: float           # 0.0 withheld, 1.0 paid - the min over the week
    payout_scale: float             # standard/pro in a scaled bucket, else 1.0
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


class WeeklySealLedger:
    """Per-hotkey map of week_start_ms -> SealedWeek, persisted independently of every other ledger.

    Deliberately NOT deleted by delete_debt_ledger, _switch_account, revert-elimination or
    /admin/reset: it is the audit trail those operations would otherwise destroy.
    """

    def __init__(self, running_unit_tests: bool = False):
        self.running_unit_tests = running_unit_tests
        self.sealed: Dict[str, Dict[int, SealedWeek]] = {}
        self.load_from_disk()

    # ============================ Persistence ============================

    def _get_path(self) -> str:
        suffix = "/tests" if self.running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/weekly_seal_ledger.json.gz"

    def save_to_disk(self) -> None:
        path = self._get_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "format_version": "1.0",
            "last_update_ms": TimeUtil.now_in_millis(),
            "sealed": {
                hotkey: {str(week_start_ms): week.to_dict() for week_start_ms, week in weeks.items()}
                for hotkey, weeks in self.sealed.items()
            },
        }
        temp_path = path + ".tmp"
        with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        shutil.move(temp_path, path)

    def load_from_disk(self) -> int:
        path = self._get_path()
        if not os.path.exists(path):
            logger.info("[WEEKLY_SEAL] No existing weekly seal ledger found")
            return 0

        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            # Losing this file means losing the guarantee, so it must be loud rather than silent.
            logger.error(f"[WEEKLY_SEAL] Failed to read {path}: {e}")
            raise

        for hotkey, weeks in (data.get("sealed") or {}).items():
            self.sealed[hotkey] = {
                int(week_start_ms): SealedWeek.from_dict(week)
                for week_start_ms, week in weeks.items()
            }

        total = sum(len(w) for w in self.sealed.values())
        logger.info(f"[WEEKLY_SEAL] Loaded {total} sealed weeks across {len(self.sealed)} hotkeys")
        return total

    # ============================ Reads ============================

    def get_sealed(self, hotkey: str) -> Dict[int, SealedWeek]:
        """Sealed weeks for one hotkey, keyed by Monday 00:00 UTC."""
        return dict(self.sealed.get(hotkey, {}))

    def is_sealed(self, hotkey: str, week_start_ms: int) -> bool:
        return week_start_ms in self.sealed.get(hotkey, {})

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
