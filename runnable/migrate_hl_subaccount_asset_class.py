#!/usr/bin/env python3
"""
Migrate existing Hyperliquid-linked subaccounts from asset class "crypto" to "hl_all".

HL subaccounts were previously created with asset_class="crypto" by default. The correct
value is "hl_all", which grants access to all HyperLiquid-sourced trade pairs and applies
the correct challenge period returns threshold (10%, same as crypto).

Updates two files where asset_class must be explicitly corrected:
  - asset_selections.json  (source of truth for miner account loading)
  - entities.json          (SubaccountInfo.asset_class read directly by entity_manager)

Usage:
    python runnable/migrate_hl_subaccount_asset_class.py            # Perform migration
    python runnable/migrate_hl_subaccount_asset_class.py --dry-run  # Preview only
"""

import json
import os
import sys

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

OLD = "crypto"
NEW = "hl_all"


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def migrate_hl_subaccount_asset_class(dry_run: bool = False) -> None:
    entities_path = ValiBkpUtils.get_entity_file_location()
    selections_path = ValiBkpUtils.get_asset_selections_file_location()

    prefix = "[DRY RUN] " if dry_run else ""

    # ------------------------------------------------------------------ #
    # 1. Identify HL synthetic hotkeys from entities.json                 #
    # ------------------------------------------------------------------ #
    entities_data = _load_json(entities_path)
    if not entities_data:
        print(f"entities.json not found or empty at {entities_path} — nothing to migrate.")
        return

    # Collect HL hotkeys and patch entities.json in one pass
    hl_synthetic_hotkeys: set[str] = set()
    entities_changed = 0

    for entity_hotkey, entity in entities_data.items():
        for subaccount_id, subaccount in entity.get("subaccounts", {}).items():
            if not subaccount.get("hl_address"):
                continue
            synthetic_hotkey = (
                subaccount.get("synthetic_hotkey") or f"{entity_hotkey}_{subaccount_id}"
            )
            hl_synthetic_hotkeys.add(synthetic_hotkey)
            if subaccount.get("asset_class") == OLD:
                print(f"  {prefix}entities.json: {synthetic_hotkey}: {OLD} → {NEW}")
                if not dry_run:
                    subaccount["asset_class"] = NEW
                entities_changed += 1

    if not hl_synthetic_hotkeys:
        print("No HL-linked subaccounts found in entities.json — nothing to migrate.")
        return

    print(f"\nFound {len(hl_synthetic_hotkeys)} HL-linked synthetic hotkey(s).")
    print(f"  entities.json: {entities_changed} updated.\n")

    if not dry_run and entities_changed:
        with open(entities_path, "w") as f:
            json.dump(entities_data, f, indent=2)

    # ------------------------------------------------------------------ #
    # 2. Migrate asset_selections.json                                    #
    # ------------------------------------------------------------------ #
    selections = _load_json(selections_path)
    selections_changed = 0

    for hk in hl_synthetic_hotkeys:
        if selections.get(hk) == OLD:
            print(f"  {prefix}asset_selections.json: {hk}: {OLD} → {NEW}")
            if not dry_run:
                selections[hk] = NEW
            selections_changed += 1

    already_correct = sum(1 for hk in hl_synthetic_hotkeys if selections.get(hk) == NEW)
    missing = sum(1 for hk in hl_synthetic_hotkeys if hk not in selections)
    print(f"  asset_selections.json: {selections_changed} updated, "
          f"{already_correct} already correct"
          + (f", {missing} missing selection (skipped)" if missing else "") + ".\n")

    if not dry_run and selections_changed:
        with open(selections_path, "w") as f:
            json.dump(selections, f, indent=2)

    # ------------------------------------------------------------------ #
    # 3. Summary                                                          #
    # ------------------------------------------------------------------ #
    total = entities_changed + selections_changed
    if dry_run:
        print(f"Dry run complete — {total} update(s) would be applied.")
    else:
        print(f"Migration complete — {total} update(s) applied.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate_hl_subaccount_asset_class(dry_run=dry_run)
