"""
Migration script to reset account state based on positions.

This script migrates miner accounts to the new balance/capital_used model by:
- Clearing all existing transaction JSONL files
- Resetting account state (total_realized_pnl=0, capital_used=0, etc.)
- Setting capital_used from open positions (net_value)
- Setting total_realized_pnl from closed positions + partial closes in open positions
- Only processes positions opened/closed after 2026-01-01 (1767225600000 ms)
- Skips eliminated hotkeys (loaded from eliminations.json)

Usage:
    python runnable/migrate_cash_balance.py [--dry-run]

Options:
    --dry-run, -n    Test migration without modifying files
"""

import os
import sys

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


def load_positions(order_status: OrderStatus) -> dict[str, list[Position]]:
    """Load positions from disk by order status."""
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
            pos_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                hotkey, trade_pair.trade_pair_id,
                order_status=order_status,
                running_unit_tests=False
            )
            if not os.path.exists(pos_dir):
                continue

            for filename in os.listdir(pos_dir):
                filepath = os.path.join(pos_dir, filename)
                try:
                    file_string = ValiBkpUtils.get_file(filepath)
                    position = Position.model_validate_json(file_string)
                    if hotkey not in all_positions:
                        all_positions[hotkey] = []
                    all_positions[hotkey].append(position)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")

    total = sum(len(positions) for positions in all_positions.values())
    print(f"Loaded {total} {order_status.name} positions from {len(all_positions)} hotkeys")
    return all_positions


MIGRATION_CUTOFF_MS = 1767225600000  # 2026-01-01 00:00:00 UTC


def migrate_hotkey(
    manager: MinerAccountManager,
    hotkey: str,
    open_positions: list[Position],
    closed_positions: list[Position],
    asset_selections: dict[str, TradePairCategory],
    dry_run: bool
) -> dict:
    """Migrate account state for a single hotkey to new balance/capital_used model."""
    stats = {
        'open_processed': 0,
        'closed_processed': 0,
        'errors': []
    }

    # Get account directly without RPC
    account = manager.get_account(hotkey)
    if not account:
        # Create account manually
        asset_class = asset_selections.get(hotkey)
        from vali_objects.miner_account.miner_account_manager import MinerAccount
        account = MinerAccount(
            miner_hotkey=hotkey,
            asset_class=asset_class,
        )
        manager.accounts[hotkey] = account

    # Get asset class from asset selections or existing account
    asset_class = asset_selections.get(hotkey) or account.asset_class
    account.asset_class = asset_class

    # Reset account state for migration
    account.total_realized_pnl = 0.0
    account.capital_used = 0.0
    account.total_borrowed_amount = 0.0
    account.total_interest_paid = 0.0

    # Process open positions: accumulate capital_used and realized_pnl (from partial closes)
    for position in open_positions:
        try:
            # Skip positions opened before 2026
            if position.open_ms < MIGRATION_CUTOFF_MS:
                continue
            # Skip positions that don't belong to this asset class
            if asset_class and position.trade_pair.trade_pair_category != asset_class:
                continue
            account.capital_used += abs(position.net_value)
            account.total_realized_pnl += position.realized_pnl
            stats['open_processed'] += 1
        except Exception as e:
            stats['errors'].append(f"Open position {position.position_uuid}: {e}")

    # Process closed positions: accumulate realized_pnl
    for position in closed_positions:
        try:
            # Skip positions opened before 2026
            if position.open_ms < MIGRATION_CUTOFF_MS:
                continue
            # Skip positions that don't belong to this asset class
            if asset_class and position.trade_pair.trade_pair_category != asset_class:
                continue
            account.total_realized_pnl += position.realized_pnl
            stats['closed_processed'] += 1
        except Exception as e:
            stats['errors'].append(f"Closed position {position.position_uuid}: {e}")

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


def load_eliminations() -> set[str]:
    """Load eliminated hotkeys from disk."""
    eliminations_file = ValiBkpUtils.get_eliminations_dir(running_unit_tests=False)
    eliminations_data = dict(ValiUtils.get_vali_json_file(eliminations_file))
    eliminations_list = eliminations_data.get("eliminations", [])
    return {elim["hotkey"] for elim in eliminations_list if "hotkey" in elim}


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

    # Load eliminations from disk
    eliminated_hotkeys = load_eliminations()
    print(f"Loaded {len(eliminated_hotkeys)} eliminated hotkeys from disk")

    open_positions = load_positions(OrderStatus.OPEN)
    closed_positions = load_positions(OrderStatus.CLOSED)

    # Get all hotkeys that need processing (from accounts + positions), excluding eliminated
    all_hotkeys = set(asset_selections.keys() | manager.accounts.keys()) - eliminated_hotkeys
    print(f"Total hotkeys to process: {len(all_hotkeys)} (excluded {len(eliminated_hotkeys)} eliminated)")

    total_stats = {
        'hotkeys_processed': 0,
        'hotkeys_skipped': len(eliminated_hotkeys),
        'open_processed': 0,
        'closed_processed': 0,
        'errors': []
    }

    print(f"\nProcessing {len(all_hotkeys)} hotkeys...")

    for hotkey in all_hotkeys:
        hotkey_open = open_positions.get(hotkey, [])
        hotkey_closed = closed_positions.get(hotkey, [])
        stats = migrate_hotkey(manager, hotkey, hotkey_open, hotkey_closed, asset_selections, DRY_RUN)

        # Print account status
        account = manager.get_account(hotkey)
        if account:
            print(f"[{hotkey[:8]}] balance: ${account.balance:,.2f}, realized_pnl: ${account.total_realized_pnl:,.2f}, capital_used: ${account.capital_used:,.2f}, open: {stats['open_processed']}, closed: {stats['closed_processed']}")

        total_stats['hotkeys_processed'] += 1
        total_stats['open_processed'] += stats['open_processed']
        total_stats['closed_processed'] += stats['closed_processed']
        total_stats['errors'].extend(stats['errors'])

    # Save accounts to disk
    if not DRY_RUN:
        manager._save_accounts_to_disk()

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Hotkeys processed:    {total_stats['hotkeys_processed']}")
    print(f"Hotkeys skipped:      {total_stats['hotkeys_skipped']} (eliminated)")
    print(f"Open positions:       {total_stats['open_processed']}")
    print(f"Closed positions:     {total_stats['closed_processed']}")

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
