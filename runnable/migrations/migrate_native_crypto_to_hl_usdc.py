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


from runnable.migration_utils import MigrationUtils
from vali_objects.enums.misc import OrderStatus
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import NATIVE_CRYPTO_TO_HL_TRADE_PAIR, TradePair
from vali_objects.vali_dataclasses.order import Order
from shared_objects.log import logger


_ID_MAP: dict[str, TradePair] = {
    old_tp.trade_pair_id: new_tp
    for old_tp, new_tp in NATIVE_CRYPTO_TO_HL_TRADE_PAIR.items()
}


def _find_open_position_collisions(running_unit_tests: bool):
    """Scan for hotkeys with open positions in BOTH the native crypto pair and its HL USDC equivalent.

    Returns a list of (hotkey, native_position, hl_position) tuples and prints each collision.
    These would violate the one-open-position-per-pair invariant after migration unless merged.
    """
    all_positions = MigrationUtils.load_all_positions(running_unit_tests=running_unit_tests)
    collisions = []

    for hotkey, positions in all_positions.items():
        # hotkey -> {trade_pair_id: [Position, ...]} for OPEN positions only.
        open_by_tp: dict[str, list] = {}
        for p in positions:
            if not p.is_open_position:
                continue
            tp_id = p.trade_pair.trade_pair_id if hasattr(p.trade_pair, "trade_pair_id") else None
            if tp_id is None:
                continue
            open_by_tp.setdefault(tp_id, []).append(p)

        for old_tp_id, new_tp in _ID_MAP.items():
            new_tp_id = new_tp.trade_pair_id
            if old_tp_id in open_by_tp and new_tp_id in open_by_tp:
                for native_pos in open_by_tp[old_tp_id]:
                    for hl_pos in open_by_tp[new_tp_id]:
                        collisions.append((hotkey, native_pos, hl_pos))
                        print(
                            f"[COLLISION] hotkey={hotkey} "
                            f"{old_tp_id}={native_pos.position_uuid} "
                            f"{new_tp_id}={hl_pos.position_uuid}"
                        )

    if not collisions:
        print("[COLLISION] no open-position collisions detected.")
    else:
        print(f"[COLLISION] {len(collisions)} collision(s) detected.")
    return collisions


def _resolve_collisions(collisions, dry_run: bool, running_unit_tests: bool) -> tuple[int, int]:
    """Merge each native-crypto open position into its colliding HL USDC open position.

    Strategy: move all orders from the native position onto the HL position (re-tagged to the
    HL trade_pair), sort by processed_ms, rebuild, and save the merged HL position. Delete the
    native position's on-disk file. Returns (resolved, failed).
    """
    resolved = 0
    failed = 0

    for hotkey, native_pos, hl_pos in collisions:
        try:
            new_tp = hl_pos.trade_pair
            native_tp_id = native_pos.trade_pair.trade_pair_id

            print(
                f"[MERGE] hotkey={hotkey[:8]}... "
                f"{native_tp_id}/{native_pos.position_uuid} ({len(native_pos.orders)} orders) "
                f"-> {new_tp.trade_pair_id}/{hl_pos.position_uuid} ({len(hl_pos.orders)} orders)"
            )

            if dry_run:
                resolved += 1
                continue

            for order in native_pos.orders:
                order.trade_pair = new_tp
                hl_pos.orders.append(order)

            hl_pos.orders.sort(key=lambda o: o.processed_ms)
            hl_pos.rebuild_position_with_updated_orders(price_fetcher_client=None)

            MigrationUtils.save_position(hl_pos, running_unit_tests=running_unit_tests)

            # Delete the now-merged native position file.
            native_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                hotkey, native_tp_id, order_status=OrderStatus.OPEN,
                running_unit_tests=running_unit_tests,
            )
            native_path = os.path.join(native_dir, native_pos.position_uuid)
            if os.path.exists(native_path):
                os.remove(native_path)

            resolved += 1
        except Exception as e:
            failed += 1
            logger.error(
                f"Failed to merge collision for hotkey={hotkey} "
                f"native={native_pos.position_uuid} hl={hl_pos.position_uuid}: {e}\n"
                f"{traceback.format_exc()}"
            )

    return resolved, failed


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
                logger.error(
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
                        logger.error(
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

    print("Scanning for open-position collisions (native vs HL pair)...")
    collisions = _find_open_position_collisions(running_unit_tests)

    print("\nResolving open-position collisions...")
    col_resolved, col_failed = _resolve_collisions(collisions, dry_run, running_unit_tests)

    print("\nMigrating positions...")
    pos_migrated, pos_failed = _migrate_positions(dry_run, running_unit_tests)

    print("\nMigrating limit orders...")
    lo_migrated, lo_failed = _migrate_limit_orders(dry_run, running_unit_tests)

    print("\nCleaning up empty source directories...")
    _cleanup_empty_dirs(dry_run, running_unit_tests)

    suffix = " (dry run — nothing written)" if dry_run else ""
    print(
        f"\nDone. collisions: resolved={col_resolved}, failed={col_failed} | "
        f"positions: migrated={pos_migrated}, failed={pos_failed} | "
        f"limit_orders: migrated={lo_migrated}, failed={lo_failed}{suffix}"
    )
    return (col_failed + pos_failed + lo_failed) == 0


def main() -> bool:
    return _migrate(dry_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate native crypto positions/limit orders to HL USDC equivalents.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()
    sys.exit(0 if _migrate(dry_run=args.dry_run) else 1)
