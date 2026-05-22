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
import traceback

import bittensor as bt

from runnable.migration_utils import MigrationUtils
from vali_objects.vali_dataclasses.position import Position


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

    all_positions = MigrationUtils.load_all_positions()
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
                    MigrationUtils.save_position(position)

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
