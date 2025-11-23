#!/usr/bin/env python3
"""
Script to identify and delete positions that violate the perf_ledger assertion:
- For each trade pair, there should be at most 1 open position
- If there is 1 open position, it must be the last position in the historical list

Usage:
    python delete_assertion_violating_positions.py [--dry-run] [--hotkey HOTKEY]

Options:
    --dry-run: Show what would be deleted without actually deleting
    --hotkey: Only check positions for a specific hotkey
"""

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import bittensor as bt

from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.live_price_server import LivePriceFetcherServer
from vali_utils import ValiUtils


def check_position_invariants(
    positions: List[Position],
    hotkey: str
) -> Tuple[bool, List[Position], str]:
    """
    Check if positions for a hotkey violate the perf ledger assertions.

    Returns:
        (has_violations, positions_to_delete, reason)
    """
    # Group positions by trade pair
    tp_to_positions = defaultdict(list)
    for pos in positions:
        tp_to_positions[pos.trade_pair.trade_pair_id].append(pos)

    # Sort each trade pair's positions by close_ms (open positions have close_ms=None, treated as infinity)
    for tp_id in tp_to_positions:
        tp_to_positions[tp_id].sort(
            key=lambda p: p.close_ms if p.close_ms is not None else float('inf')
        )

    violations = []
    positions_to_delete = []

    for tp_id, tp_positions in tp_to_positions.items():
        # Count open and closed positions
        open_positions = [p for p in tp_positions if p.is_open_position]
        closed_positions = [p for p in tp_positions if p.is_closed_position]
        n_open = len(open_positions)
        n_closed = len(closed_positions)

        # Check Assertion 1: At most 1 open position per trade pair
        if n_open > 1:
            reason = f"VIOLATION: {tp_id} has {n_open} open positions (max 1 allowed)"
            violations.append(reason)
            # Delete all but the most recent open position
            open_positions.sort(key=lambda p: p.open_ms)
            positions_to_delete.extend(open_positions[:-1])
            bt.logging.warning(f"  {hotkey}: {reason}")
            bt.logging.warning(f"  Will delete {len(open_positions)-1} older open positions")

        # Check Assertion 2: If 1 open position exists, it must be last in the list
        elif n_open == 1:
            last_position = tp_positions[-1]
            if not last_position.is_open_position:
                reason = (
                    f"VIOLATION: {tp_id} has 1 open position but it's NOT the last in the list. "
                    f"Last position is closed (close_ms={last_position.close_ms})"
                )
                violations.append(reason)
                # The open position is incorrectly placed - delete it
                positions_to_delete.extend(open_positions)
                bt.logging.warning(f"  {hotkey}: {reason}")
                bt.logging.warning(f"  Will delete the misplaced open position: {open_positions[0].position_uuid}")

    has_violations = len(violations) > 0
    reason_summary = "; ".join(violations) if violations else "OK"

    return has_violations, positions_to_delete, reason_summary


def identify_and_delete_violating_positions(
    position_manager: PositionManager,
    dry_run: bool = True,
    target_hotkey: str = None
) -> Dict[str, any]:
    """
    Identify and optionally delete positions that violate perf ledger assertions.

    Args:
        position_manager: PositionManager instance
        dry_run: If True, only report violations without deleting
        target_hotkey: If provided, only check this hotkey

    Returns:
        Dictionary with statistics about violations found and actions taken
    """
    bt.logging.info("="*80)
    bt.logging.info(f"Checking positions for assertion violations (dry_run={dry_run})")
    bt.logging.info("="*80)

    # Get all positions
    if target_hotkey:
        hotkeys_to_check = [target_hotkey]
        hotkey_to_positions = {target_hotkey: position_manager.get_positions_for_one_hotkey(target_hotkey)}
    else:
        hotkey_to_positions = position_manager.get_positions_for_all_miners()
        hotkeys_to_check = list(hotkey_to_positions.keys())

    stats = {
        'total_hotkeys_checked': len(hotkeys_to_check),
        'hotkeys_with_violations': 0,
        'total_positions_to_delete': 0,
        'deleted_positions': 0,
        'failed_deletions': 0,
        'violations_by_hotkey': {}
    }

    for hotkey in hotkeys_to_check:
        positions = hotkey_to_positions.get(hotkey, [])
        if not positions:
            continue

        has_violations, positions_to_delete, reason = check_position_invariants(
            positions, hotkey
        )

        if has_violations:
            stats['hotkeys_with_violations'] += 1
            stats['total_positions_to_delete'] += len(positions_to_delete)
            stats['violations_by_hotkey'][hotkey] = {
                'reason': reason,
                'n_positions_to_delete': len(positions_to_delete),
                'positions_to_delete': [
                    {
                        'uuid': p.position_uuid,
                        'trade_pair': p.trade_pair.trade_pair_id,
                        'is_open': p.is_open_position,
                        'open_ms': p.open_ms,
                        'close_ms': p.close_ms
                    } for p in positions_to_delete
                ]
            }

            bt.logging.warning(f"\nHotkey: {hotkey}")
            bt.logging.warning(f"  Violations: {reason}")
            bt.logging.warning(f"  Positions to delete: {len(positions_to_delete)}")

            if not dry_run:
                # Actually delete the positions
                for pos in positions_to_delete:
                    try:
                        position_manager.delete_position(pos.miner_hotkey, pos.position_uuid)
                        stats['deleted_positions'] += 1
                        bt.logging.success(
                            f"  Deleted position {pos.position_uuid} "
                            f"({pos.trade_pair.trade_pair_id})"
                        )
                    except Exception as e:
                        stats['failed_deletions'] += 1
                        bt.logging.error(
                            f"  Failed to delete position {pos.position_uuid}: {e}"
                        )

    # Print summary
    bt.logging.info("\n" + "="*80)
    bt.logging.info("SUMMARY")
    bt.logging.info("="*80)
    bt.logging.info(f"Total hotkeys checked: {stats['total_hotkeys_checked']}")
    bt.logging.info(f"Hotkeys with violations: {stats['hotkeys_with_violations']}")
    bt.logging.info(f"Total positions to delete: {stats['total_positions_to_delete']}")

    if not dry_run:
        bt.logging.info(f"Successfully deleted: {stats['deleted_positions']}")
        bt.logging.info(f"Failed deletions: {stats['failed_deletions']}")
    else:
        bt.logging.warning("DRY RUN: No positions were actually deleted")

    bt.logging.info("="*80)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Identify and delete positions that violate perf ledger assertions'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--hotkey',
        type=str,
        default=None,
        help='Only check positions for this specific hotkey'
    )

    args = parser.parse_args()

    # Initialize logging
    bt.logging.enable_info()
    bt.logging.enable_warning()

    # Initialize managers
    bt.logging.info("Initializing managers...")
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcherServer(secrets, disable_ws=True)
    perf_ledger_manager = PerfLedgerManager(None)
    position_manager = PositionManager(
        perf_ledger_manager=perf_ledger_manager,
        live_price_fetcher=live_price_fetcher
    )

    # Run the check and delete process
    stats = identify_and_delete_violating_positions(
        position_manager,
        dry_run=args.dry_run,
        target_hotkey=args.hotkey
    )

    # Exit with appropriate code
    if stats['hotkeys_with_violations'] > 0:
        if args.dry_run:
            bt.logging.warning(
                f"\nFound violations. Run without --dry-run to delete the {stats['total_positions_to_delete']} problematic positions."
            )
            sys.exit(1)
        elif stats['failed_deletions'] > 0:
            bt.logging.error(
                f"\nDeleted {stats['deleted_positions']} positions but {stats['failed_deletions']} deletions failed."
            )
            sys.exit(2)
        else:
            bt.logging.success(
                f"\nSuccessfully deleted all {stats['deleted_positions']} problematic positions."
            )
            sys.exit(0)
    else:
        bt.logging.success("\nNo violations found!")
        sys.exit(0)


if __name__ == '__main__':
    main()
