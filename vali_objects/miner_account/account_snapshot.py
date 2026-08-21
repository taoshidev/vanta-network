"""
AccountSnapshot dataclass and disk-backed read helpers.

Snapshots are appended one-per-line to per-miner `snapshots.jsonl` files
(see `ValiBkpUtils.get_miner_snapshot_path`). This module owns the on-disk
schema and read-side access patterns used by the REST API and by callers
that need historical account state (e.g., entity payout unrealized PnL).
"""
import json
import os
from dataclasses import dataclass
from typing import List

from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


DEFAULT_SNAPSHOT_LIMIT = 720           # 30 days at hourly cadence
MAX_SNAPSHOT_LIMIT = 2160              # 90 days
DEFAULT_TOLERANCE_MS = 1 * 60 * 1000   # 1 minute


@dataclass
class AccountSnapshot:
    """Account state captured at a point in time for a single miner."""
    snapshot_ms: int
    account_size: float
    balance: float
    equity: float

    @property
    def equity_return(self) -> float:
        """equity / account_size — return relative to deposited capital."""
        if not self.account_size:
            return 1.0
        return self.equity / self.account_size

    def to_dict(self) -> dict:
        return {
            'snapshot_ms': self.snapshot_ms,
            'snapshot_date': TimeUtil.millis_to_formatted_date_str(self.snapshot_ms),
            'account_size': self.account_size,
            'balance': self.balance,
            'equity': self.equity,
            'equity_return': self.equity_return,
        }

    @staticmethod
    def from_dict(d: dict) -> 'AccountSnapshot':
        return AccountSnapshot(
            snapshot_ms=d.get('snapshot_ms') or d['day_open_ms'],
            account_size=d['account_size'],
            balance=d['balance'],
            equity=d['equity'],
        )


def read_all_snapshots(hotkey: str, running_unit_tests: bool = False) -> List[AccountSnapshot]:
    """Return all snapshots for `hotkey` in chronological order (oldest first).

    Reads and parses the whole file — callers that need multiple lookups against the
    same hotkey (e.g. a per-week scan) should call this once and search the in-memory
    list rather than re-reading the file per lookup.
    """
    path = ValiBkpUtils.get_miner_snapshot_path(hotkey, running_unit_tests)
    if not os.path.exists(path):
        return []
    out: List[AccountSnapshot] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(AccountSnapshot.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def read_last_n(
    hotkey: str,
    n: int = DEFAULT_SNAPSHOT_LIMIT,
    running_unit_tests: bool = False,
) -> List[AccountSnapshot]:
    """Return the last `n` snapshots for `hotkey` in chronological order (oldest first)."""
    n = max(0, min(n, MAX_SNAPSHOT_LIMIT))
    if n == 0:
        return []
    snapshots = read_all_snapshots(hotkey, running_unit_tests)
    return snapshots[-n:]
