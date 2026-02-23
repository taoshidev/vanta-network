"""
Migration script to backfill max_return (HWM) on MinerAccount.

For each miner account, this script:
- Computes portfolio return as balance / account_size from existing MinerAccount data
  (balance already includes total_realized_pnl, interest, and fees)
- Sets max_return to max(1.0, portfolio_return)

Usage:
    python runnable/migrations/_migrate_max_return.py [--dry-run]

Options:
    --dry-run, -n    Test migration without modifying files
"""

import sys

from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.vali_config import RPCConnectionMode

DRY_RUN = False
for arg in sys.argv[1:]:
    if arg in ['--dry-run', '-n']:
        DRY_RUN = True
        print("*** DRY RUN MODE - No files will be modified ***\n")


def main() -> bool:
    """Run the migration. Returns True on success, False on failure."""
    print("Initializing MinerAccountManager...")
    manager = MinerAccountManager(running_unit_tests=False, connection_mode=RPCConnectionMode.LOCAL)
    print(f"Loaded {len(manager.accounts)} miner accounts")

    updated = 0
    skipped = 0
    results = []

    for hotkey, account in manager.accounts.items():
        account_size = account.get_account_size()
        if account_size <= 0:
            skipped += 1
            continue

        portfolio_return = account.balance / account_size
        max_return = max(1.0, portfolio_return)

        old_max_return = account.max_return
        if max_return > old_max_return:
            results.append((hotkey, old_max_return, max_return))
            if not DRY_RUN:
                account.max_return = max_return
            updated += 1
        else:
            skipped += 1

    # Print results
    results.sort(key=lambda x: x[2], reverse=True)
    for hotkey, old_val, new_val in results:
        print(f"[{hotkey}] max_return: {old_val:.6f} -> {new_val:.6f}")

    # Save
    if not DRY_RUN and updated > 0:
        manager._save_accounts_to_disk()

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Updated:  {updated}")
    print(f"Skipped:  {skipped}")

    if DRY_RUN:
        print("\n[DRY RUN] No files were modified")
    else:
        print("\nMigration completed.")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
