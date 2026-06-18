"""
Reconcile SubaccountInfo.asset_class in entities.json against asset_selections.json.

Prior to the update_subaccount_asset_selection fix, calling update_asset_selection
on a synthetic hotkey updated asset_selections.json but left entities.json stale.
This script finds every subaccount whose entities.json asset_class differs from
asset_selections.json and updates entities.json to match.

asset_selections.json is the authoritative source of truth for what the validator
actually uses for scoring and challenge period evaluation.

Usage:
    python runnable/migrations/migrate_subaccount_asset_class_from_selections.py
    python runnable/migrations/migrate_subaccount_asset_class_from_selections.py --dry-run
"""

import argparse
import sys

from runnable.migration_utils import MigrationUtils


def _migrate(dry_run: bool = False, running_unit_tests: bool = False) -> bool:
    prefix = "[DRY RUN] " if dry_run else ""

    entities = MigrationUtils.load_entities(running_unit_tests=running_unit_tests)
    if not entities:
        print("entities.json not found or empty — nothing to migrate.")
        return True

    selections = MigrationUtils.load_asset_selections(running_unit_tests=running_unit_tests)
    if not selections:
        print("asset_selections.json not found or empty — nothing to migrate.")
        return True

    changed = 0
    for entity_hotkey, entity in entities.items():
        for sub_id, subaccount in entity.subaccounts.items():
            selection = selections.get(subaccount.synthetic_hotkey)
            if selection is None:
                continue
            if subaccount.asset_class == selection:
                continue

            print(
                f"  {prefix}{subaccount.synthetic_hotkey}: "
                f"entities={subaccount.asset_class} → selections={selection}"
            )
            if not dry_run:
                subaccount.asset_class = selection
            changed += 1

    if changed == 0:
        print("No mismatches found — entities.json is already in sync.")
        return True

    print(f"\nSummary: {changed} subaccount(s) updated.")

    if not dry_run:
        MigrationUtils.save_entities(entities, running_unit_tests=running_unit_tests)
        print("Migration complete.")
    else:
        print("Dry run complete — no files written.")

    return True


def main() -> bool:
    return _migrate(dry_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconcile entities.json asset_class against asset_selections.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()
    sys.exit(0 if _migrate(dry_run=args.dry_run) else 1)
