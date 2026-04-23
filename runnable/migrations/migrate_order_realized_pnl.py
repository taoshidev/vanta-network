"""
Migration script to populate order.realized_pnl for all positions.

The Order.realized_pnl field was added to track per-order realized PnL as
computed in Position.calculate_pnl(). This script rebuilds all positions via
rebuild_position_with_updated_orders(None) so that each order's realized_pnl
is back-filled from order data (no live price fetcher needed).

Usage:
    python migrate_order_realized_pnl.py            # live run (writes changes)
    python migrate_order_realized_pnl.py --dry-run  # preview only (no writes)
"""

import argparse
import os
import traceback
from collections import defaultdict

import bittensor as bt

from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.enums.misc import OrderStatus


def load_all_positions() -> dict[str, list[Position]]:
    """Load all open and closed positions from disk."""
    all_positions: dict[str, list[Position]] = defaultdict(list)

    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    if not os.path.exists(base_dir):
        bt.logging.error(f"Positions directory not found: {base_dir}")
        return all_positions

    for hotkey in os.listdir(base_dir):
        hotkey_path = os.path.join(base_dir, hotkey)
        if not os.path.isdir(hotkey_path):
            continue

        for trade_pair in TradePair:
            for status in (OrderStatus.OPEN, OrderStatus.CLOSED):
                dir_path = ValiBkpUtils.get_partitioned_miner_positions_dir(
                    hotkey, trade_pair.trade_pair_id,
                    order_status=status,
                    running_unit_tests=False
                )
                if not os.path.exists(dir_path):
                    continue
                for filename in os.listdir(dir_path):
                    filepath = os.path.join(dir_path, filename)
                    try:
                        file_string = ValiBkpUtils.get_file(filepath)
                        position = Position.model_validate_json(file_string)
                        all_positions[hotkey].append(position)
                    except Exception as e:
                        bt.logging.warning(f"Failed to load {filepath}: {e}")

    total = sum(len(v) for v in all_positions.values())
    print(f"Loaded {total} positions from {len(all_positions)} hotkeys")
    return all_positions


def save_position(position: Position) -> None:
    """Persist position back to disk."""
    miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
        position.miner_hotkey,
        position.trade_pair.trade_pair_id,
        order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
        running_unit_tests=False
    )
    ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)


def _log_pnl_diff(hotkey: str, position: Position, before: dict, after: dict) -> None:
    realized_changed = before["realized_pnl"] != after["realized_pnl"]
    unrealized_changed = before["unrealized_pnl"] != after["unrealized_pnl"]

    if not realized_changed and not unrealized_changed:
        return

    print(
        f"[DIFF] hotkey={hotkey}... uuid={position.position_uuid} "
        f"pair={position.trade_pair.trade_pair_id}"
    )
    if realized_changed:
        print(
            f"       realized_pnl:   {before['realized_pnl']} -> {after['realized_pnl']}"
        )
    if unrealized_changed:
        print(
            f"       unrealized_pnl: {before['unrealized_pnl']} -> {after['unrealized_pnl']}"
        )


def main(dry_run: bool = False) -> bool:
    """
    Rebuild all positions so that order.realized_pnl is populated.

    live_price_fetcher=None works because realized_pnl in calculate_pnl()
    only uses order.quantity, order.slippage, order.quote_usd_rate, and
    self.average_entry_price — all already stored on disk.
    """
    if dry_run:
        print("DRY RUN — no changes will be written to disk.")

    all_positions = load_all_positions()
    total_positions = sum(len(v) for v in all_positions.values())

    if total_positions == 0:
        print("No positions found — nothing to migrate.")
        return True

    print(f"Rebuilding {total_positions} positions...")

    migrated = 0
    skipped = 0
    failed = 0
    changed = 0

    for hotkey, positions in all_positions.items():
        for position in positions:
            try:
                if not position.orders:
                    skipped += 1
                    continue

                before = {
                    "realized_pnl": position.realized_pnl,
                    "unrealized_pnl": position.unrealized_pnl,
                }

                position.rebuild_position_with_updated_orders(price_fetcher_client=None)

                # Restore unrealized_pnl — rebuild uses order prices rather than live prices,
                # so the recomputed value is stale for open positions.
                position.unrealized_pnl = before["unrealized_pnl"]

                after = {
                    "realized_pnl": position.realized_pnl,
                    "unrealized_pnl": position.unrealized_pnl,
                }

                _log_pnl_diff(hotkey, position, before, after)

                if before != after:
                    changed += 1

                if not dry_run:
                    save_position(position)

                migrated += 1

            except Exception as e:
                failed += 1
                bt.logging.error(
                    f"Failed to rebuild position {position.position_uuid} ({hotkey}): {e}\n"
                    f"{traceback.format_exc()}"
                )

    suffix = " (dry run — nothing written)" if dry_run else ""
    print(
        f"Done. rebuilt={migrated}, changed={changed}, skipped={skipped}, failed={failed}{suffix}"
    )
    return failed == 0


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Migrate order.realized_pnl for all positions.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing anything to disk.",
    )
    args = parser.parse_args()

    sys.exit(0 if main(dry_run=args.dry_run) else 1)
