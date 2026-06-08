"""
Backfill MinerAccount.capital_used_by_class from each miner's open positions.

The capital_used_by_class field was introduced alongside multi-class subaccount
portfolio sub-caps (HL_ALL). Pre-redesign checkpoints do not carry the field;
after upgrade the in-memory MinerAccount defaults it to an empty dict and lazy-
backfills only on specific rebuild events (price corrections, order corrections,
manual restore via REST). That delay means per-class cap enforcement can be slack
for an HL_ALL subaccount until something happens to trigger a rebuild for it.

This migration walks every hotkey on disk, recomputes the per-class breakdown
from open positions, and writes it into the on-disk miner_account_sizes file.
Once it runs (before the validator process comes up after the update), every
account loads with an accurate capital_used_by_class and per-class caps bind
from the first order onward.

Scope — this migration is intentionally additive:
  * Only writes the capital_used_by_class field; leaves capital_used,
    total_realized_pnl, total_fees_paid, total_borrowed_amount untouched. The
    grandfather rule for existing positions ("reduce/close any amount, no new
    increases past the per-class cap") only needs accurate per-class numbers.
  * Only updates hotkeys that already have a record in miner_account_sizes.
    Stub-creating accounts from orphan position files is out of scope.
  * Idempotent — re-running is a no-op (the diff check skips records that
    already match the recomputed value).

Usage:
    python runnable/migrations/migrate_capital_used_by_class.py            # live
    python runnable/migrations/migrate_capital_used_by_class.py --dry-run  # preview
"""

import argparse
import os
import sys
import traceback

import bittensor as bt

from runnable.migration_utils import MigrationUtils
from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


def main(dry_run: bool = False, running_unit_tests: bool = False) -> bool:
    """Backfill capital_used_by_class for every hotkey on disk.

    Args:
        dry_run: when True, no file writes happen (preview only).
        running_unit_tests: routes file lookups to the tests/validation/ tree
            instead of validation/. Production callers (run_migrations.py)
            always invoke main() with no args, so this defaults False.
    """
    accounts_path = ValiBkpUtils.get_miner_account_sizes_file_location(running_unit_tests=running_unit_tests)

    if not os.path.exists(accounts_path):
        print(f"miner_account_sizes file not found at {accounts_path} — nothing to migrate.")
        return True

    accounts_data = ValiUtils.get_vali_json_file_dict(accounts_path)
    # Legacy top-level key — MinerAccountManager pops it on load and never writes
    # it back. Drop it here too so we don't reintroduce it on save.
    accounts_data.pop("_cost_per_theta", None)

    if not accounts_data:
        print("Accounts file empty — nothing to migrate.")
        return True

    all_positions = MigrationUtils.load_all_positions(running_unit_tests=running_unit_tests)

    updated = 0
    unchanged = 0
    skipped = 0
    failed = 0

    for hotkey, records_list in accounts_data.items():
        try:
            if not isinstance(records_list, list) or not records_list:
                skipped += 1
                continue
            last_record = records_list[-1]
            if not isinstance(last_record, dict):
                skipped += 1
                continue

            positions = all_positions.get(hotkey, [])
            computed = MinerAccountManager.compute_account_state_from_positions(positions)
            per_class_enum = computed["capital_used_by_class"]
            # JSON keys must be strings. cat.value is the string form of the enum.
            per_class_str = {cat.value: amt for cat, amt in per_class_enum.items()}

            old_value = last_record.get("capital_used_by_class")
            if old_value == per_class_str:
                unchanged += 1
                continue

            print(f"  {hotkey[:8]}...: {old_value} -> {per_class_str}")
            last_record["capital_used_by_class"] = per_class_str
            updated += 1

        except Exception as e:
            failed += 1
            bt.logging.error(
                f"Failed to backfill {hotkey}: {e}\n{traceback.format_exc()}"
            )

    suffix = " (dry run — nothing written)" if dry_run else ""
    print(
        f"Backfill summary: updated={updated}, unchanged={unchanged}, "
        f"skipped={skipped}, failed={failed}{suffix}"
    )

    if failed > 0:
        # Fail-fast: any per-hotkey failure aborts the whole migration. Successful
        # updates that have been mutated in memory are NOT written back. The
        # operator must investigate the logged errors above, resolve them, then
        # remove this migration from migrations_completed.txt to re-run.
        print(
            f"Aborting migration: {failed} hotkey(s) failed to backfill; "
            f"no changes written to disk. See errors above and re-run after fixing."
        )
        return False

    if not dry_run and updated > 0:
        ValiBkpUtils.write_file(accounts_path, accounts_data)
        print(f"Wrote updates to {accounts_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill MinerAccount.capital_used_by_class from positions on disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything to disk.",
    )
    args = parser.parse_args()
    sys.exit(0 if main(dry_run=args.dry_run) else 1)
