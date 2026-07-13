"""
Migrate or delete positions and limit orders that use dynamic (HL-only) trade pairs.

Dynamic trade pairs are those NOT in TRADE_PAIR_ID_TO_TRADE_PAIR — they were accepted
historically via the HL dynamic registry but are no longer supported.

For each dynamic trade pair found on disk:
  - If appending "C" to its ID produces a known native TradePair (e.g. "SOMEUSD" ->
    "SOMEUSDС" is native), migrate all positions and limit orders to that native pair.
  - Otherwise, delete the position/limit order files.

After processing all miners, purge all entries from validation/hl_dynamic_registry.json.

Usage:
    python runnable/migrations/migrate_dynamic_trade_pairs.py
    python runnable/migrations/migrate_dynamic_trade_pairs.py --dry-run
"""

import argparse
import os
import sys
import traceback

import bittensor as bt

from runnable.migration_utils import MigrationUtils
from vali_objects.enums.misc import OrderStatus
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import (
    TRADE_PAIR_ID_TO_TRADE_PAIR,
    DynamicTradePair,
    TradePair,
)
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position


def _resolve_dynamic_tp(dynamic_tp_id: str) -> TradePair | None:
    """Return the native TradePair for a dynamic ID by appending 'C', or None."""
    return TRADE_PAIR_ID_TO_TRADE_PAIR.get(dynamic_tp_id + "C")


def _find_dynamic_tp_dirs(base_dir: str) -> list[str]:
    """Return subdirectory names under base_dir that are not native trade pair IDs."""
    if not os.path.isdir(base_dir):
        return []
    return [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
        and name not in TRADE_PAIR_ID_TO_TRADE_PAIR
    ]


def _migrate_positions(dry_run: bool, running_unit_tests: bool) -> tuple[int, int, int]:
    """Migrate or delete position files with dynamic trade pairs.

    Returns (migrated, deleted, failed).
    """
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
    migrated = deleted = failed = 0

    if not os.path.isdir(base_dir):
        return migrated, deleted, failed

    for hotkey in os.listdir(base_dir):
        positions_dir = os.path.join(base_dir, hotkey, "positions")
        dynamic_ids = _find_dynamic_tp_dirs(positions_dir)
        if not dynamic_ids:
            continue

        for dynamic_tp_id in dynamic_ids:
            new_tp = _resolve_dynamic_tp(dynamic_tp_id)
            action = f"-> {new_tp.trade_pair_id}" if new_tp else "DELETE"

            tp_dir = os.path.join(positions_dir, dynamic_tp_id)
            for status in ("open", "closed"):
                status_dir = os.path.join(tp_dir, status)
                if not os.path.isdir(status_dir):
                    continue

                for filename in os.listdir(status_dir):
                    old_path = os.path.join(status_dir, filename)
                    try:
                        print(
                            f"[POSITION] hotkey={hotkey} uuid={filename} "
                            f"{dynamic_tp_id} {action} ({status})"
                        )

                        if dry_run:
                            migrated += int(new_tp is not None)
                            deleted += int(new_tp is None)
                            continue

                        if new_tp is None:
                            os.remove(old_path)
                            deleted += 1
                            continue

                        file_string = ValiBkpUtils.get_file(old_path)
                        position = Position.model_validate_json(file_string)
                        position.trade_pair = new_tp
                        for order in position.orders:
                            order.trade_pair = new_tp

                        MigrationUtils.save_position(position, running_unit_tests=running_unit_tests)
                        os.remove(old_path)
                        migrated += 1
                    except Exception as e:
                        failed += 1
                        bt.logging.error(
                            f"Failed to process position {old_path}: {e}\n"
                            f"{traceback.format_exc()}"
                        )

    return migrated, deleted, failed


def _migrate_limit_orders(dry_run: bool, running_unit_tests: bool) -> tuple[int, int, int]:
    """Migrate or delete limit order files with dynamic trade pairs.

    Returns (migrated, deleted, failed).
    """
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
    migrated = deleted = failed = 0

    if not os.path.isdir(base_dir):
        return migrated, deleted, failed

    for hotkey in os.listdir(base_dir):
        limit_orders_root = os.path.join(base_dir, hotkey, "limit_orders")
        dynamic_ids = _find_dynamic_tp_dirs(limit_orders_root)
        if not dynamic_ids:
            continue

        for dynamic_tp_id in dynamic_ids:
            new_tp = _resolve_dynamic_tp(dynamic_tp_id)
            action = f"-> {new_tp.trade_pair_id}" if new_tp else "DELETE"

            tp_dir = os.path.join(limit_orders_root, dynamic_tp_id)
            for status_str in ("unfilled", "closed"):
                status_dir = os.path.join(tp_dir, status_str)
                if not os.path.isdir(status_dir):
                    continue

                for filename in os.listdir(status_dir):
                    old_path = os.path.join(status_dir, filename)
                    try:
                        print(
                            f"[LIMIT_ORDER] hotkey={hotkey} uuid={filename} "
                            f"{dynamic_tp_id} {action} ({status_str})"
                        )

                        if dry_run:
                            migrated += int(new_tp is not None)
                            deleted += int(new_tp is None)
                            continue

                        if new_tp is None:
                            os.remove(old_path)
                            deleted += 1
                            continue

                        file_string = ValiBkpUtils.get_file(old_path)
                        order = Order.model_validate_json(file_string)
                        order.trade_pair = new_tp

                        new_dir = ValiBkpUtils.get_limit_orders_dir(
                            hotkey, new_tp.trade_pair_id, status_str,
                            running_unit_tests=running_unit_tests,
                        )
                        os.makedirs(new_dir, exist_ok=True)
                        ValiBkpUtils.write_file(new_dir + order.order_uuid, order)

                        os.remove(old_path)
                        migrated += 1
                    except Exception as e:
                        failed += 1
                        bt.logging.error(
                            f"Failed to process limit order {old_path}: {e}\n"
                            f"{traceback.format_exc()}"
                        )

    return migrated, deleted, failed


def _cleanup_empty_dirs(dry_run: bool, running_unit_tests: bool) -> None:
    """Remove now-empty dynamic trade pair directories under positions/ and limit_orders/."""
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
    if not os.path.isdir(base_dir):
        return

    for hotkey in os.listdir(base_dir):
        for category in ("positions", "limit_orders"):
            category_dir = os.path.join(base_dir, hotkey, category)
            if not os.path.isdir(category_dir):
                continue
            for name in os.listdir(category_dir):
                if name in TRADE_PAIR_ID_TO_TRADE_PAIR:
                    continue
                tp_dir = os.path.join(category_dir, name)
                if not os.path.isdir(tp_dir):
                    continue
                for sub in os.listdir(tp_dir):
                    sub_path = os.path.join(tp_dir, sub)
                    if os.path.isdir(sub_path) and not os.listdir(sub_path):
                        if not dry_run:
                            os.rmdir(sub_path)
                if not os.listdir(tp_dir):
                    if not dry_run:
                        os.rmdir(tp_dir)


def _migrate(dry_run: bool = False, running_unit_tests: bool = False) -> bool:
    if dry_run:
        print("DRY RUN — no changes will be written to disk.")

    print("\nMigrating positions...")
    pos_migrated, pos_deleted, pos_failed = _migrate_positions(dry_run, running_unit_tests)

    print("\nMigrating limit orders...")
    lo_migrated, lo_deleted, lo_failed = _migrate_limit_orders(dry_run, running_unit_tests)

    print("\nCleaning up empty source directories...")
    _cleanup_empty_dirs(dry_run, running_unit_tests)

    suffix = " (dry run — nothing written)" if dry_run else ""
    print(
        f"\nDone. positions: migrated={pos_migrated}, deleted={pos_deleted}, failed={pos_failed} | "
        f"limit_orders: migrated={lo_migrated}, deleted={lo_deleted}, failed={lo_failed}{suffix}"
    )
    return (pos_failed + lo_failed) == 0


def main() -> bool:
    return _migrate(dry_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate dynamic trade pair positions/limit orders to native equivalents or delete them.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()
    sys.exit(0 if _migrate(dry_run=args.dry_run) else 1)
