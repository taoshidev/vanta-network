#!/usr/bin/env python3
"""
One-time script to detect and delete ALL positions violating invariants.
This script loads positions directly from disk and fixes them permanently.

Usage:
    python fix_all_position_violations.py [--dry-run]
"""

import argparse
import sys
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.validator_sync_base import ValidatorSyncBase
from vali_utils import ValiUtils


def main():
    parser = argparse.ArgumentParser(
        description='Detect and delete ALL positions violating invariants (loads from disk)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    # Initialize logging
    bt.logging.enable_info()
    bt.logging.enable_warning()

    # Initialize managers
    bt.logging.info("Initializing managers...")
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
    perf_ledger_manager = PerfLedgerManager(None)
    position_manager = PositionManager(
        perf_ledger_manager=perf_ledger_manager,
        live_price_fetcher=live_price_fetcher
    )

    # Create validator sync instance for cleanup
    validator_sync = ValidatorSyncBase(
        position_manager=position_manager,
        live_price_fetcher=live_price_fetcher,
        enable_position_splitting=True
    )

    bt.logging.info("=" * 80)
    bt.logging.info("LOADING ALL POSITIONS FROM DISK...")
    bt.logging.info("=" * 80)

    # Load ALL positions from disk
    disk_positions = position_manager.get_positions_for_all_miners(sort_positions=True)
    total_positions = sum(len(positions) for positions in disk_positions.values())

    bt.logging.info(f"Loaded {total_positions} positions across {len(disk_positions)} hotkeys")

    if args.dry_run:
        bt.logging.warning("DRY RUN MODE: No positions will actually be deleted")
        # Temporarily disable deletion for dry run
        validator_sync.is_mothership = True

    # Run the cleanup
    current_time_ms = TimeUtil.now_in_millis()
    stats = validator_sync.detect_and_delete_overlapping_positions(
        disk_positions,
        current_time_ms=current_time_ms
    )

    # Print final summary
    bt.logging.info("=" * 80)
    bt.logging.info("FINAL SUMMARY")
    bt.logging.info("=" * 80)
    bt.logging.info(f"Total positions scanned: {total_positions}")
    bt.logging.info(f"Total positions deleted: {stats['positions_deleted']}")
    bt.logging.info(f"  - Due to overlaps: {stats['positions_deleted_overlaps']}")
    bt.logging.info(f"  - Due to invariant violations: {stats['positions_deleted_invariant_violations']}")
    bt.logging.info(f"Hotkeys affected: {len(stats['hotkeys_with_overlaps'] | stats['hotkeys_with_invariant_violations'])}")

    if args.dry_run:
        bt.logging.warning("=" * 80)
        bt.logging.warning("DRY RUN: No changes were made to disk")
        bt.logging.warning(f"Run without --dry-run to delete {stats['positions_deleted']} positions")
        bt.logging.warning("=" * 80)
        sys.exit(1 if stats['positions_deleted'] > 0 else 0)
    else:
        if stats['positions_deleted'] > 0:
            bt.logging.success("=" * 80)
            bt.logging.success(f"Successfully deleted {stats['positions_deleted']} problematic positions from disk")
            bt.logging.success("Validator can now be restarted safely")
            bt.logging.success("=" * 80)
            sys.exit(0)
        else:
            bt.logging.success("=" * 80)
            bt.logging.success("No violations found - positions are clean!")
            bt.logging.success("=" * 80)
            sys.exit(0)


if __name__ == '__main__':
    main()
