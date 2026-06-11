"""
Re-tag non-HL multi-class subaccounts from "hl_all" to "all_markets".

HL_ALL was originally the multi-class asset class for Hyperliquid-linked
subaccounts (those with an hl_address). The new ALL_MARKETS class mirrors
HL_ALL semantics but is intended for non-HL (i.e. Vanta-source) subaccounts
that want cross-asset-class trading. Subaccounts currently flagged hl_all
that do NOT have an hl_address belong under all_markets.

For each such subaccount we update three files:
  - entities.json            (SubaccountInfo.asset_class)
  - asset_selections.json    (selection keyed by synthetic_hotkey)
  - miner_account_sizes.json (MinerAccount.asset_class)

Usage:
    python runnable/migrations/migrate_hl_all_to_all_markets.py
    python runnable/migrations/migrate_hl_all_to_all_markets.py --dry-run
"""

import argparse
import sys

from runnable.migration_utils import MigrationUtils
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass

OLD_STR = "hl_all"
NEW_STR = "all_markets"
NEW_ENUM = MinerAssetClass.ALL_MARKETS


def _migrate(dry_run: bool = False, running_unit_tests: bool = False) -> bool:
    prefix = "[DRY RUN] " if dry_run else ""

    entities = MigrationUtils.load_entities(running_unit_tests=running_unit_tests)
    if not entities:
        print("entities.json not found or empty — nothing to migrate.")
        return True

    selections = MigrationUtils.load_asset_selections(running_unit_tests=running_unit_tests)
    accounts = MigrationUtils.load_miner_accounts(running_unit_tests=running_unit_tests)

    target_synthetic_hotkeys: set[str] = set()
    subaccounts_changed = 0

    for entity_hotkey, entity in entities.items():
        for sub_id, subaccount in entity.subaccounts.items():
            if subaccount.asset_class != OLD_STR:
                continue
            if subaccount.hl_address:
                # HL-linked subaccounts stay on hl_all.
                continue

            target_synthetic_hotkeys.add(subaccount.synthetic_hotkey)
            print(f"  {prefix}entities.json: {subaccount.synthetic_hotkey}: {OLD_STR} → {NEW_STR}")
            if not dry_run:
                subaccount.asset_class = NEW_STR
            subaccounts_changed += 1

    if not target_synthetic_hotkeys:
        print("No non-HL hl_all subaccounts found — nothing to migrate.")
        return True

    print(f"\nFound {len(target_synthetic_hotkeys)} non-HL hl_all subaccount(s).")

    selections_changed = 0
    for hk in target_synthetic_hotkeys:
        if selections.get(hk) == OLD_STR:
            print(f"  {prefix}asset_selections.json: {hk}: {OLD_STR} → {NEW_STR}")
            if not dry_run:
                selections[hk] = NEW_STR
            selections_changed += 1

    accounts_changed = 0
    for hk in target_synthetic_hotkeys:
        account = accounts.get(hk)
        if account is None:
            continue
        if account.asset_class == MinerAssetClass.HL_ALL:
            print(f"  {prefix}miner_account_sizes.json: {hk}: {OLD_STR} → {NEW_STR}")
            if not dry_run:
                account.asset_class = NEW_ENUM
            accounts_changed += 1

    print(
        f"\nSummary: entities={subaccounts_changed}, "
        f"selections={selections_changed}, accounts={accounts_changed}"
    )

    if not dry_run:
        if subaccounts_changed:
            MigrationUtils.save_entities(entities, running_unit_tests=running_unit_tests)
        if selections_changed:
            MigrationUtils.save_asset_selections(selections, running_unit_tests=running_unit_tests)
        if accounts_changed:
            MigrationUtils.save_miner_accounts(accounts, running_unit_tests=running_unit_tests)
        print("Migration complete.")
    else:
        print("Dry run complete — no files written.")

    return True


def main() -> bool:
    return _migrate(dry_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-tag non-HL hl_all subaccounts to all_markets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()
    sys.exit(0 if _migrate(dry_run=args.dry_run) else 1)
