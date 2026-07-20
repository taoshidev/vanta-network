"""
Backfill MinerAccount.max_equity_snapshot from the perf ledger EOD high-water mark.

For each miner account on disk, the migration:

1. Loads the perf ledger (non-frozen) and extracts midnight checkpoints.
2. Computes eod_hwm = max(cp.equity_ret for cp in midnight_cps, default 1.0).
3. Picks the winning equity_return = max(1.0, daily_open_snapshot.equity_return,
   eod_hwm from ledger).
4. Builds max_equity_snapshot:
   - If the winner came from the perf ledger: uses the timestamp of that
     checkpoint; balance = equity = account_size * eod_hwm.
   - If the winner came from daily_open_snapshot: copies that snapshot as-is.
   - If 1.0 wins (nothing exceeded it): equity = balance = account_size, using
     daily_open_snapshot.day_open_ms when available.

Scope:
  * Only writes max_equity_snapshot; leaves all other fields untouched.
  * Skips accounts that already carry a max_equity_snapshot (idempotent).

Usage:
    python runnable/migrations/migrate_max_return_to_eod_hwm.py            # live
    python runnable/migrations/migrate_max_return_to_eod_hwm.py --dry-run  # preview
"""

import argparse
import sys
import traceback

import bittensor as bt

from runnable.migration_utils import MigrationUtils
from time_util.time_util import TimeUtil
from vali_objects.miner_account.miner_account_manager import AccountSnapshot
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_manager import PerfLedgerManager


def _load_perf_ledgers() -> dict[str, PerfLedger]:
    """Load non-frozen perf ledgers directly from disk, bypassing RPC."""
    manager = PerfLedgerManager()
    return manager.get_perf_ledgers(from_disk=True)


def _parse_eod_hwm(ledger: PerfLedger) -> tuple[float, int | None]:
    """Return (eod_hwm, hwm_timestamp_ms) from midnight checkpoints.

    eod_hwm is the max equity_ret across all midnight checkpoints, floored at
    1.0.  hwm_timestamp_ms is the last_update_ms of the checkpoint that achieved
    it (None when no midnight checkpoints exist).
    """
    midnight_cps = [cp for cp in ledger.cps if cp.last_update_ms % 86400000 == 0 and cp.equity_ret > 0]
    if not midnight_cps:
        return 1.0, None
    best_cp = max(midnight_cps, key=lambda cp: cp.equity_ret)
    return max(best_cp.equity_ret, 1.0), best_cp.last_update_ms


def main(dry_run: bool = False) -> bool:
    """Backfill max_equity_snapshot for every hotkey on disk."""
    accounts = MigrationUtils.load_miner_accounts()

    if not accounts:
        print("No accounts found — nothing to migrate.")
        return True

    print("Loading perf ledgers from disk…")
    perf_ledgers = _load_perf_ledgers()
    print(f"Loaded {len(perf_ledgers)} perf ledger(s).")

    updated = 0
    unchanged = 0
    missing_ledger = 0
    capped_at_floor = 0
    failed = 0

    for hotkey, account in accounts.items():
        try:
            if account.max_equity_snapshot is not None:
                unchanged += 1
                continue

            account_size = account.get_account_size()
            dos = account.daily_open_snapshot
            dos_equity_return = dos.equity_return if dos else 1.0
            dos_day_open_ms = dos.day_open_ms if dos else 0

            ledger = perf_ledgers.get(hotkey)
            ledger_hwm = 1.0
            ledger_hwm_ms = None
            if ledger is not None:
                ledger_hwm, ledger_hwm_ms = _parse_eod_hwm(ledger)
            else:
                missing_ledger += 1

            best_return = max(1.0, dos_equity_return, ledger_hwm)
            equity = account_size * best_return
            day_open_ms = ledger_hwm_ms if (ledger_hwm_ms is not None and ledger_hwm == best_return) else dos_day_open_ms
            snapshot = AccountSnapshot(
                day_open_ms=day_open_ms,
                account_size=account_size,
                balance=equity,
                equity=equity,
            )

            if best_return > 1.0:
                date_str = TimeUtil.millis_to_short_date_str(day_open_ms) if day_open_ms else "unknown"
                print(f"  {hotkey}: max_equity_snapshot <- best_return={best_return:.4f} date={date_str}")
            else:
                capped_at_floor += 1
            account.max_equity_snapshot = snapshot
            updated += 1

        except Exception as e:
            failed += 1
            bt.logging.error(f"Failed to backfill {hotkey}: {e}\n{traceback.format_exc()}")

    suffix = " (dry run — nothing written)" if dry_run else ""
    print(
        f"Backfill summary: updated={updated}, unchanged={unchanged}, "
        f"missing_ledger={missing_ledger}, capped_at_floor={capped_at_floor}, failed={failed}{suffix}"
    )

    if failed > 0:
        print(
            f"Aborting migration: {failed} hotkey(s) failed; no changes written. "
            f"See errors above and re-run after fixing."
        )
        return False

    if not dry_run and updated > 0:
        MigrationUtils.save_miner_accounts(accounts)
        print("Wrote updates to miner_account_sizes.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill MinerAccount.max_equity_snapshot from perf ledger EOD high-water mark.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything to disk.",
    )
    args = parser.parse_args()
    sys.exit(0 if main(dry_run=args.dry_run) else 1)
