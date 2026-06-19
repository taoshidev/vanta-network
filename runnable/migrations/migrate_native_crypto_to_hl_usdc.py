"""
Migrate native-crypto positions and limit orders to their Hyperliquid USDC equivalents.

For every Position and every limit order (filled or unfilled) whose trade_pair is a
native Vanta crypto pair listed in NATIVE_CRYPTO_TO_HL_TRADE_PAIR (e.g. BTCUSD), we:

  1. Rewrite the position-level trade_pair to the HL USDC pair (e.g. BTCUSDC).
  2. Rewrite each child order's trade_pair to match.
  3. Move the on-disk file from
        validation/miners/<hotkey>/positions/BTCUSD/{open,closed}/<position_uuid>
     to
        validation/miners/<hotkey>/positions/BTCUSDC/{open,closed}/<position_uuid>
     (analogous move for limit_orders/<trade_pair_id>/{unfilled,closed}/).
  4. Clean up the now-empty source directories.

Usage:
    python runnable/migrations/migrate_native_crypto_to_hl_usdc.py
    python runnable/migrations/migrate_native_crypto_to_hl_usdc.py --dry-run
"""

import argparse
import os
import sys
import traceback

import bittensor as bt

from runnable.migration_utils import MigrationUtils
from vali_objects.enums.misc import OrderStatus
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import NATIVE_CRYPTO_TO_HL_TRADE_PAIR, TradePair
from vali_objects.vali_dataclasses.order import Order


_ID_MAP: dict[str, TradePair] = {
    old_tp.trade_pair_id: new_tp
    for old_tp, new_tp in NATIVE_CRYPTO_TO_HL_TRADE_PAIR.items()
}


def _migrate_positions(dry_run: bool, running_unit_tests: bool) -> tuple[int, int]:
    """Returns (migrated, failed)."""
    all_positions = MigrationUtils.load_all_positions(running_unit_tests=running_unit_tests)
    migrated = 0
    failed = 0

    for hotkey, positions in all_positions.items():
        for position in positions:
            try:
                old_tp = position.trade_pair
                old_tp_id = old_tp.trade_pair_id if hasattr(old_tp, "trade_pair_id") else None
                new_tp = _ID_MAP.get(old_tp_id) if old_tp_id else None
                if new_tp is None:
                    continue

                status = OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED
                old_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                    hotkey, old_tp_id, order_status=status,
                    running_unit_tests=running_unit_tests,
                )
                old_path = os.path.join(old_dir, position.position_uuid)

                print(
                    f"[POSITION] hotkey={hotkey[:8]}... uuid={position.position_uuid} "
                    f"{old_tp_id} -> {new_tp.trade_pair_id} ({status.name.lower()})"
                )

                if dry_run:
                    migrated += 1
                    continue

                position.trade_pair = new_tp
                for order in position.orders:
                    order.trade_pair = new_tp

                MigrationUtils.save_position(position, running_unit_tests=running_unit_tests)

                if os.path.exists(old_path):
                    os.remove(old_path)

                migrated += 1
            except Exception as e:
                failed += 1
                bt.logging.error(
                    f"Failed to migrate position {getattr(position, 'position_uuid', '?')} "
                    f"({hotkey}): {e}\n{traceback.format_exc()}"
                )

    return migrated, failed


def _migrate_limit_orders(dry_run: bool, running_unit_tests: bool) -> tuple[int, int]:
    """Returns (migrated, failed)."""
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
    if not os.path.exists(base_dir):
        return 0, 0

    migrated = 0
    failed = 0

    for hotkey in os.listdir(base_dir):
        limit_orders_root = os.path.join(base_dir, hotkey, "limit_orders")
        if not os.path.isdir(limit_orders_root):
            continue

        for old_tp_id, new_tp in _ID_MAP.items():
            tp_dir = os.path.join(limit_orders_root, old_tp_id)
            if not os.path.isdir(tp_dir):
                continue

            for status_str in ("unfilled", "closed"):
                status_dir = os.path.join(tp_dir, status_str)
                if not os.path.isdir(status_dir):
                    continue

                for filename in os.listdir(status_dir):
                    old_path = os.path.join(status_dir, filename)
                    try:
                        file_string = ValiBkpUtils.get_file(old_path)
                        order = Order.model_validate_json(file_string)

                        print(
                            f"[LIMIT_ORDER] hotkey={hotkey[:8]}... uuid={order.order_uuid} "
                            f"{old_tp_id} -> {new_tp.trade_pair_id} ({status_str})"
                        )

                        if dry_run:
                            migrated += 1
                            continue

                        order.trade_pair = new_tp

                        new_dir = ValiBkpUtils.get_limit_orders_dir(
                            hotkey, new_tp.trade_pair_id, status_str,
                            running_unit_tests=running_unit_tests,
                        )
                        os.makedirs(new_dir, exist_ok=True)
                        ValiBkpUtils.write_file(new_dir + order.order_uuid, order)

                        if os.path.exists(old_path):
                            os.remove(old_path)

                        migrated += 1
                    except Exception as e:
                        failed += 1
                        bt.logging.error(
                            f"Failed to migrate limit order {old_path}: {e}\n"
                            f"{traceback.format_exc()}"
                        )

    return migrated, failed


def _cleanup_empty_dirs(dry_run: bool, running_unit_tests: bool) -> None:
    """Remove now-empty <native_crypto_id> source directories under positions/ and limit_orders/."""
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
    if not os.path.exists(base_dir):
        return

    for hotkey in os.listdir(base_dir):
        for category in ("positions", "limit_orders"):
            for old_tp_id in _ID_MAP.keys():
                tp_dir = os.path.join(base_dir, hotkey, category, old_tp_id)
                if not os.path.isdir(tp_dir):
                    continue
                # Remove empty status subdirs first, then the tp dir if empty.
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

    print("Migrating positions...")
    pos_migrated, pos_failed = _migrate_positions(dry_run, running_unit_tests)

    print("\nMigrating limit orders...")
    lo_migrated, lo_failed = _migrate_limit_orders(dry_run, running_unit_tests)

    print("\nCleaning up empty source directories...")
    _cleanup_empty_dirs(dry_run, running_unit_tests)

    suffix = " (dry run — nothing written)" if dry_run else ""
    print(
        f"\nDone. positions: migrated={pos_migrated}, failed={pos_failed} | "
        f"limit_orders: migrated={lo_migrated}, failed={lo_failed}{suffix}"
    )
    return (pos_failed + lo_failed) == 0


def main() -> bool:
    return _migrate(dry_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate native crypto positions/limit orders to HL USDC equivalents.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()
    sys.exit(0 if _migrate(dry_run=args.dry_run) else 1)
