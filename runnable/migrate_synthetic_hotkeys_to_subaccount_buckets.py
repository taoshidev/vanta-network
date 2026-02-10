#!/usr/bin/env python3
"""
Migrate existing synthetic hotkeys from old buckets to new subaccount buckets.

Migration mapping:
- CHALLENGE → SUBACCOUNT_CHALLENGE
- PROBATION → SUBACCOUNT_FUNDED
- MAINCOMP → SUBACCOUNT_FUNDED

No synthetic hotkeys are migrated to SUBACCOUNT_ALPHA initially.

Usage:
    python runnable/migrate_synthetic_hotkeys_to_subaccount_buckets.py            # Perform migration
    python runnable/migrate_synthetic_hotkeys_to_subaccount_buckets.py --dry-run  # Preview only
"""

import json
import os
import sys
from entity_management.entity_utils import is_synthetic_hotkey
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig


def migrate_synthetic_hotkeys(dry_run: bool = False):
    """
    Migrate synthetic hotkeys to new subaccount buckets by directly reading and writing challengeperiod.json.

    Args:
        dry_run: If True, only print what would be migrated without making changes
    """
    mode_str = "DRY RUN MODE" if dry_run else "LIVE MIGRATION"
    print(f"Starting synthetic hotkey migration - {mode_str}")
    print("=" * 80)

    # Define bucket migration mapping
    bucket_mapping = {
        MinerBucket.CHALLENGE.value: MinerBucket.SUBACCOUNT_CHALLENGE.value,
        MinerBucket.PROBATION.value: MinerBucket.SUBACCOUNT_FUNDED.value,
        MinerBucket.MAINCOMP.value: MinerBucket.SUBACCOUNT_FUNDED.value,
    }

    # Build file path
    challengeperiod_file = os.path.join(ValiConfig.BASE_DIR, "validation", "challengeperiod.json")

    # Check if file exists
    if not os.path.exists(challengeperiod_file):
        print(f"ERROR: challengeperiod.json not found at {challengeperiod_file}")
        return

    # Read the JSON file
    print(f"Reading {challengeperiod_file}")
    with open(challengeperiod_file, 'r') as f:
        data = json.load(f)

    # Track migrations
    migrations = []
    migration_counts = {
        MinerBucket.CHALLENGE.value: 0,
        MinerBucket.PROBATION.value: 0,
        MinerBucket.MAINCOMP.value: 0,
    }

    # Process each hotkey
    for hotkey, miner_data in data.items():
        current_bucket = miner_data.get("bucket")

        # Check if this is a synthetic hotkey and in a bucket we want to migrate
        if is_synthetic_hotkey(hotkey) and current_bucket in bucket_mapping:
            new_bucket = bucket_mapping[current_bucket]
            migrations.append((hotkey, current_bucket, new_bucket, miner_data))
            migration_counts[current_bucket] += 1

    # Print summary
    print(f"Found {len(migrations)} synthetic hotkeys to migrate:")
    print(f"  - CHALLENGE → SUBACCOUNT_CHALLENGE: {migration_counts[MinerBucket.CHALLENGE.value]}")
    print(f"  - PROBATION → SUBACCOUNT_FUNDED: {migration_counts[MinerBucket.PROBATION.value]}")
    print(f"  - MAINCOMP → SUBACCOUNT_FUNDED: {migration_counts[MinerBucket.MAINCOMP.value]}")
    print("=" * 80)

    if len(migrations) == 0:
        print("No synthetic hotkeys to migrate. Migration complete.")
        return

    # Perform migrations (or preview in dry-run mode)
    for i, (hotkey, old_bucket, new_bucket, miner_data) in enumerate(migrations, 1):
        # Print migration details
        action = "WOULD MIGRATE" if dry_run else "MIGRATING"
        print(
            f"[{i}/{len(migrations)}] {action}: {hotkey}\n"
            f"             Old Bucket: {old_bucket}\n"
            f"             New Bucket: {new_bucket}\n"
            f"             Start Time: {miner_data.get('bucket_start_time')}"
        )

        # Only perform actual migration if not dry-run
        if not dry_run:
            # Update bucket in memory
            data[hotkey]["bucket"] = new_bucket

    print("=" * 80)
    if dry_run:
        print(f"DRY RUN COMPLETE: Would migrate {len(migrations)} synthetic hotkeys")
        print("Run without --dry-run to perform actual migration")
    else:
        # Write updated data back to disk
        print(f"Writing updated data to {challengeperiod_file}")
        with open(challengeperiod_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"MIGRATION COMPLETE: Successfully migrated {len(migrations)} synthetic hotkeys")


if __name__ == "__main__":
    # Check for --dry-run flag
    dry_run = "--dry-run" in sys.argv
    migrate_synthetic_hotkeys(dry_run=dry_run)
