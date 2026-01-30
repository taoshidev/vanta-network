"""
Migration script to track cash balance for open positions.

This script migrates miner accounts to track cash balance by:
- Clearing all existing transaction JSONL files
- Setting initial cash balance based on account size and asset class multiplier
- Processing each order chronologically using MinerAccountManager methods
- Only processes currently OPEN positions

Usage:
    python runnable/migrate_cash_balance.py [--dry-run]

Options:
    --dry-run, -n    Test migration without modifying files
"""

import os
import sys

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.enums.misc import OrderStatus
from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.vali_config import TradePairCategory, ValiConfig, TradePair, RPCConnectionMode

DRY_RUN = False
for arg in sys.argv[1:]:
    if arg in ['--dry-run', '-n']:
        DRY_RUN = True
        print("*** DRY RUN MODE - No files will be modified ***\n")


def load_open_positions() -> dict[str, list[Position]]:
    """Load all OPEN positions from disk."""
    all_positions: dict[str, list[Position]] = {}
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)

    if not os.path.exists(base_dir):
        print(f"Positions directory not found: {base_dir}")
        return all_positions

    for hotkey in os.listdir(base_dir):
        hotkey_path = os.path.join(base_dir, hotkey)
        if not os.path.isdir(hotkey_path):
            continue

        for trade_pair in TradePair:
            open_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                hotkey, trade_pair.trade_pair_id,
                order_status=OrderStatus.OPEN,
                running_unit_tests=False
            )
            if not os.path.exists(open_dir):
                continue

            for filename in os.listdir(open_dir):
                filepath = os.path.join(open_dir, filename)
                try:
                    file_string = ValiBkpUtils.get_file(filepath)
                    position = Position.model_validate_json(file_string)
                    if hotkey not in all_positions:
                        all_positions[hotkey] = []
                    all_positions[hotkey].append(position)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")

    total = sum(len(positions) for positions in all_positions.values())
    print(f"Loaded {total} open positions from {len(all_positions)} hotkeys")
    return all_positions


def process_order_for_migration(
    manager: MinerAccountManager,
    hotkey: str,
    order,
    position: Position
):
    """Process a single order for cash balance migration."""
    account = manager.get_account(hotkey)
    if not account:
        return

    # Determine if this is a buy (adding to position) or sell (reducing/closing)
    is_buy = order.order_type == position.position_type

    if is_buy:
        order_value = abs(order.value) if order.value else 0.0
        margin_loan = order.margin_loan if order.margin_loan else 0.0
        equity_used = order_value - margin_loan
        account.cash_balance -= equity_used
        MinerAccountManager.record_transaction(
            hotkey, order.processed_ms, "BUY",
            cash_delta=-equity_used,
            loan_delta=margin_loan,
            running_unit_tests=False
        )
    else:
        qty = abs(order.quantity) if order.quantity else 0.0
        lot_size = position.trade_pair.lot_size
        sale_proceeds = qty * lot_size / order.usd_base_rate if order.usd_base_rate else 0.0
        margin_loan = order.margin_loan if order.margin_loan else 0.0
        account.cash_balance += abs(sale_proceeds)
        MinerAccountManager.record_transaction(
            hotkey, order.processed_ms, "SELL",
            cash_delta=abs(sale_proceeds),
            loan_delta=margin_loan,
            running_unit_tests=False
        )


def migrate_hotkey(
    manager: MinerAccountManager,
    hotkey: str,
    positions: list[Position],
    asset_selections: dict[str, TradePairCategory],
    dry_run: bool
) -> dict:
    """Migrate cash balance for a single hotkey."""
    stats = {
        'positions_processed': 0,
        'orders_processed': 0,
        'errors': []
    }

    # Get account directly without RPC
    account = manager.get_account(hotkey)
    if not account:
        # Create account manually
        asset_class = asset_selections.get(hotkey)
        multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_class, 1.0) if asset_class else 1.0
        from vali_objects.miner_account.miner_account_manager import MinerAccount
        account = MinerAccount(
            miner_hotkey=hotkey,
            cash_balance=ValiConfig.MIN_CAPITAL * multiplier,
            asset_class=asset_class,
        )
        manager.accounts[hotkey] = account

    account_size = account.get_account_size()

    # Get asset class multiplier
    asset_class = asset_selections.get(hotkey) or account.asset_class
    account.asset_class = asset_class
    multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_class, 1.0) if asset_class else 1.0

    # Reset cash balance and borrowed amount for migration
    initial_cash = account_size * multiplier
    account.cash_balance = initial_cash
    account.total_borrowed_amount = 0.0

    # Sort all positions by their first order timestamp
    positions_sorted = sorted(positions, key=lambda p: p.orders[0].processed_ms if p.orders else 0)

    # Collect all orders across positions and sort by timestamp
    all_orders = []
    for position in positions_sorted:
        for order in position.orders:
            all_orders.append((order, position))

    all_orders.sort(key=lambda x: x[0].processed_ms)

    # Process orders chronologically
    for order, position in all_orders:
        try:
            process_order_for_migration(manager, hotkey, order, position)
            stats['orders_processed'] += 1
        except Exception as e:
            stats['errors'].append(f"Order {order.order_uuid}: {e}")

    stats['positions_processed'] = len(positions)
    return stats


def load_asset_selections() -> dict[str, TradePairCategory]:
    """Load asset selections directly from disk."""
    asset_file = ValiBkpUtils.get_asset_selections_file_location(running_unit_tests=False)
    asset_data = dict(ValiUtils.get_vali_json_file(asset_file))
    result = {}
    for hotkey, asset_str in asset_data.items():
        try:
            result[hotkey] = TradePairCategory(asset_str)
        except ValueError:
            pass
    return result


def clear_all_transactions(dry_run: bool) -> int:
    """Clear all transaction JSONL files for all miners."""
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    cleared_count = 0

    if not os.path.exists(base_dir):
        return 0

    for hotkey in os.listdir(base_dir):
        hotkey_path = os.path.join(base_dir, hotkey)
        if not os.path.isdir(hotkey_path):
            continue

        tx_path = ValiBkpUtils.get_miner_transactions_path(hotkey, running_unit_tests=False)
        if os.path.exists(tx_path):
            if not dry_run:
                ValiBkpUtils.clear_transactions(tx_path)
            cleared_count += 1

    return cleared_count


def main():
    print("Initializing MinerAccountManager...")
    manager = MinerAccountManager(running_unit_tests=False, connection_mode=RPCConnectionMode.LOCAL)

    # Clear all transaction JSONL files
    cleared_count = clear_all_transactions(DRY_RUN)
    print(f"Cleared {cleared_count} transaction files")

    # Load asset selections from disk (bypass RPC)
    asset_selections = load_asset_selections()
    print(f"Loaded {len(asset_selections)} asset selections from disk")

    all_positions = load_open_positions()

    # Get all hotkeys that need processing (from accounts + positions)
    all_hotkeys = set(asset_selections.keys() | manager.accounts.keys())
    print(f"Total hotkeys to process: {len(all_hotkeys)}")

    total_stats = {
        'hotkeys_processed': 0,
        'positions_processed': 0,
        'orders_processed': 0,
        'errors': []
    }

    print(f"\nProcessing {len(all_hotkeys)} hotkeys...")

    for hotkey in all_hotkeys:
        positions = all_positions.get(hotkey, [])
        stats = migrate_hotkey(manager, hotkey, positions, asset_selections, DRY_RUN)

        # Print account status
        account = manager.get_account(hotkey)
        if account:
            print(f"[{hotkey[:8]}] cash: ${account.cash_balance:,.2f}, borrowed: ${account.total_borrowed_amount:,.2f}, positions: {stats['positions_processed']}, orders: {stats['orders_processed']}")

        total_stats['hotkeys_processed'] += 1
        total_stats['positions_processed'] += stats['positions_processed']
        total_stats['orders_processed'] += stats['orders_processed']
        total_stats['errors'].extend(stats['errors'])

    # Save accounts to disk
    if not DRY_RUN:
        manager._save_accounts_to_disk()

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Hotkeys processed:    {total_stats['hotkeys_processed']}")
    print(f"Positions processed:  {total_stats['positions_processed']}")
    print(f"Orders processed:     {total_stats['orders_processed']}")

    if total_stats['errors']:
        print(f"\nErrors ({len(total_stats['errors'])}):")
        for err in total_stats['errors'][:10]:
            print(f"  {err}")
        if len(total_stats['errors']) > 10:
            print(f"  ... and {len(total_stats['errors']) - 10} more")

    if DRY_RUN:
        print("\n[DRY RUN] No files were modified")
    else:
        print("\nMigration completed.")


if __name__ == "__main__":
    main()
