#!/usr/bin/env python3
"""
Collapse consecutive duplicate bucket-history entries in challengeperiod.json.

A bug in ChallengePeriodManager.revert_elimination() re-added the previous
bucket using a stale, reused timestamp instead of the current time. Repeated
eliminate/revert cycles for the same hotkey produced multiple consecutive
entries with identical (bucket, start_time_ms), which then sorted into
clusters (e.g. several SUBACCOUNT_FUNDED entries in a row) instead of
alternating with the ELIMINATED entries that separate them.

This script collapses each run of consecutive entries that share the same
"bucket" and "start_time_ms" down to a single entry, keeping the first
occurrence. It does not touch entries that differ by bucket or timestamp.

Usage:
    python runnable/fix_challengeperiod_duplicate_entries.py             # Perform fix
    python runnable/fix_challengeperiod_duplicate_entries.py --dry-run   # Preview only
    python runnable/fix_challengeperiod_duplicate_entries.py --file /path/to/challengeperiod.json
"""

import argparse
import json
import os
import shutil

from vali_objects.vali_config import ValiConfig


def dedupe_entries(entries: list[dict]) -> list[dict]:
    """Collapse consecutive entries sharing the same bucket and start_time_ms."""
    deduped = []
    for entry in entries:
        if deduped:
            prev = deduped[-1]
            if prev.get("bucket") == entry.get("bucket") and prev.get("start_time_ms") == entry.get("start_time_ms"):
                continue
        deduped.append(entry)
    return deduped


def fix_challengeperiod_file(file_path: str, dry_run: bool = False):
    mode_str = "DRY RUN MODE" if dry_run else "LIVE FIX"
    print(f"Starting challenge period duplicate-entry fix - {mode_str}")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"ERROR: file not found at {file_path}")
        return

    print(f"Reading {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    affected = []
    for hotkey, miner_data in data.items():
        entries = miner_data.get("entries")
        if not entries:
            continue
        deduped = dedupe_entries(entries)
        if len(deduped) != len(entries):
            affected.append((hotkey, entries, deduped))

    print(f"Found {len(affected)} hotkeys with duplicate consecutive entries:")
    for hotkey, original, deduped in affected:
        print(f"  - {hotkey}: {len(original)} entries -> {len(deduped)} entries")

    print("=" * 80)

    if not affected:
        print("No duplicate entries found. Nothing to fix.")
        return

    if dry_run:
        print(f"DRY RUN COMPLETE: Would fix {len(affected)} hotkeys")
        print("Run without --dry-run to perform the actual fix")
        return

    for hotkey, _original, deduped in affected:
        data[hotkey]["entries"] = deduped

    backup_path = file_path + ".bak"
    print(f"Backing up original file to {backup_path}")
    shutil.copy2(file_path, backup_path)

    print(f"Writing updated data to {file_path}")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"FIX COMPLETE: Successfully fixed {len(affected)} hotkeys")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument(
        "--file",
        default=os.path.join(ValiConfig.BASE_DIR, "validation", "challengeperiod.json"),
        help="Path to challengeperiod.json (defaults to the standard validation location)",
    )
    args = parser.parse_args()

    fix_challengeperiod_file(args.file, dry_run=args.dry_run)
