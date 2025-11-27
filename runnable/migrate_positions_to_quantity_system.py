"""
Migration script to update positions with quantity and USD conversion fields.

This script migrates historical positions that only have leverage to include:
- order.quantity (number of lots/coins/shares/etc.)
- order.usd_base_rate (USD to base currency conversion)
- order.quote_usd_rate (quote to USD conversion)
- position.account_size (historical account size from collateral)

Usage:
    python runnable/migrate_positions_to_quantity_system.py [--dry-run] [--processes N]

Options:
    --dry-run, -n          Test migration without modifying files
    --processes N, -j N    Number of parallel processes (default: CPU count)

Examples:
    # Dry run with default parallelization
    python runnable/migrate_positions_to_quantity_system.py --dry-run

    # Run migration with 4 parallel processes
    python runnable/migrate_positions_to_quantity_system.py --processes 4

    # Single-process mode for debugging
    python runnable/migrate_positions_to_quantity_system.py --processes 1
"""

import sys
import time
import traceback
from multiprocessing import Pool, cpu_count

import bittensor as bt
from collections import defaultdict

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import OrderStatus
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import ValiConfig, TradePair
from time_util.time_util import TimeUtil

# Configuration
DRY_RUN = False
NUM_PROCESSES = cpu_count()  # Default to number of CPUs
COLLATERAL_START_TIME_MS = 1755302399000

# Check for command line arguments
for i, arg in enumerate(sys.argv[1:], 1):
    if arg in ['--dry-run', '-n']:
        DRY_RUN = True
        print("*** DRY RUN MODE - No files will be modified ***\n")
    elif arg in ['--processes', '-j'] and i + 1 < len(sys.argv):
        try:
            NUM_PROCESSES = int(sys.argv[i + 1])
            print(f"Using {NUM_PROCESSES} processes for parallel execution\n")
        except ValueError:
            print(f"Warning: Invalid process count '{sys.argv[i + 1]}', using default ({NUM_PROCESSES})\n")

# Initialize services
print("Initializing services...")
secrets = ValiUtils.get_secrets()
live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)

# Initialize contract manager for account size lookups
try:
    contract_manager = ValidatorContractManager(
        config=None,
        metagraph=None,
        running_unit_tests=False
    )
    print("Contract manager initialized successfully")
except Exception as e:
    print(f"Could not initialize contract manager: {e}.")
    sys.exit()

def get_account_size_for_order(hotkey, time_ms, miner_account_sizes_cache, contract_mgr):
    """
    Get the miner's account size for an order based on collateral history.
    """
    if time_ms < COLLATERAL_START_TIME_MS:
        return ValiConfig.DEFAULT_CAPITAL

    try:
        account_size = contract_mgr.get_miner_account_size(
            hotkey,
            time_ms,
            records_dict=miner_account_sizes_cache
        )
        return account_size if account_size is not None else ValiConfig.MIN_CAPITAL
    except Exception as e:
        print(f"Error getting account size for {hotkey}: {e}")
        return ValiConfig.MIN_CAPITAL

def migrate_order_quantities(position: Position, price_fetcher) -> tuple[int, int]:
    """
    Migrate orders to include quantity and USD conversion rates.
    Returns (orders_migrated_for_quantity, orders_migrated_for_usd_rates)
    """
    quantity_migrated = 0
    usd_rate_migrated = 0

    if position.position_type == OrderType.FLAT and position.orders[-1].order_type != OrderType.FLAT:
        print(f"Migrating flat position {position} last order type to FLAT")
        position.orders[-1].order_type = OrderType.FLAT

    for order in position.orders:
        # Migrate USD conversion rates
        needs_usd_migration = (order.quote_usd_rate == 0.0 or order.usd_base_rate == 0.0)
        if needs_usd_migration:
            try:
                if order.price == 0 and order.src == 1: # SKIP elimination order where price == 0
                    continue
                order.quote_usd_rate = price_fetcher.get_quote_usd_conversion(
                    order, position
                )
                order.usd_base_rate = price_fetcher.get_usd_base_conversion(
                    order.trade_pair, order.processed_ms, order.price,
                    order.order_type, position
                )
                usd_rate_migrated += 1
            except Exception as e:
                traceback.print_exc()
                bt.logging.warning(
                    f"Failed to migrate USD rates for order {order}: {e}"
                )

        # Migrate quantity
        if (order.quantity is None or order.value is None) and order.leverage is not None:
            order.value = order.leverage * position.account_size
            if order.price == 0:
                order.quantity = 0
            else:
                order.quantity = (order.value * order.usd_base_rate) / position.trade_pair.lot_size
            quantity_migrated += 1

    return quantity_migrated, usd_rate_migrated

def check_position_needs_migration(position: Position) -> tuple[bool, bool]:
    """
    Check if position needs migration.
    Returns (needs_account_size_migration, needs_order_migration)
    """
    needs_account_size = (position.account_size == 0 or position.account_size is None)
    needs_order_migration = any(
        o.quantity is None or
        o.value is None or
        o.quote_usd_rate == 0.0 or
        o.usd_base_rate == 0.0
        for o in position.orders
    )
    return needs_account_size, needs_order_migration

def load_all_positions() -> dict[str, list[Position]]:
    """Load all positions (both open and closed) from disk."""
    print("Loading positions from disk...")

    all_positions = defaultdict(list)

    # Get all hotkey directories
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)

    try:
        import os
        if not os.path.exists(base_dir):
            bt.logging.error(f"Positions directory not found: {base_dir}")
            return all_positions

        # Walk through all hotkey directories
        for hotkey in os.listdir(base_dir):
            hotkey_path = os.path.join(base_dir, hotkey)
            if not os.path.isdir(hotkey_path):
                continue

            # Load open positions
            for trade_pair in TradePair:
                open_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                    hotkey, trade_pair.trade_pair_id,
                    order_status=OrderStatus.OPEN,
                    running_unit_tests=False
                )
                if os.path.exists(open_dir):
                    for filename in os.listdir(open_dir):
                        filepath = os.path.join(open_dir, filename)
                        try:
                            file_string = ValiBkpUtils.get_file(filepath)
                            position = Position.model_validate_json(file_string)
                            all_positions[hotkey].append(position)
                        except Exception as e:
                            bt.logging.warning(f"Failed to load {filepath}: {e}")

                # Load closed positions
                closed_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                    hotkey, trade_pair.trade_pair_id,
                    order_status=OrderStatus.CLOSED,
                    running_unit_tests=False
                )
                if os.path.exists(closed_dir):
                    for filename in os.listdir(closed_dir):
                        filepath = os.path.join(closed_dir, filename)
                        try:
                            file_string = ValiBkpUtils.get_file(filepath)
                            position = Position.model_validate_json(file_string)
                            all_positions[hotkey].append(position)
                        except Exception as e:
                            bt.logging.warning(f"Failed to load {filepath}: {e}")

        total_positions = sum(len(positions) for positions in all_positions.values())
        print(
            f"Loaded {total_positions} positions from {len(all_positions)} hotkeys"
        )

    except Exception as e:
        bt.logging.error(f"Error loading positions: {e}")

    return all_positions

def save_position(position: Position):
    """Save position back to disk."""
    miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
        position.miner_hotkey,
        position.trade_pair.trade_pair_id,
        order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
        running_unit_tests=False
    )
    ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)

def process_hotkey(args):
    """
    Process all positions for a single hotkey.
    This function is called by each worker process in the pool.

    Args:
        args: Tuple of (hotkey, positions, dry_run, miner_account_sizes_cache)

    Returns:
        Dictionary with statistics for this hotkey
    """
    hotkey, positions, dry_run, miner_account_sizes_cache = args

    # Initialize services in worker process
    try:
        secrets = ValiUtils.get_secrets()
        live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)

        contract_manager = ValidatorContractManager(
            config=None,
            metagraph=None,
            running_unit_tests=False
        )
    except Exception as e:
        return {
            'hotkey': hotkey,
            'error': f"Failed to initialize services: {e}",
            'total': len(positions),
            'migrated': 0,
            'failed': len(positions),
            'account_size_migrations': 0,
            'order_quantity_migrations': 0,
            'order_usd_rate_migrations': 0,
            'errors': [f"Initialization failed for all {len(positions)} positions: {e}"]
        }

    # Statistics for this hotkey
    stats = {
        'hotkey': hotkey,
        'total': len(positions),
        'migrated': 0,
        'failed': 0,
        'account_size_migrations': 0,
        'order_quantity_migrations': 0,
        'order_usd_rate_migrations': 0,
        'errors': []
    }

    for position_idx, position in enumerate(positions, 1):
        try:
            # Progress update every 100 positions
            if position_idx % 100 == 0:
                print(
                    f"  [{hotkey[:8]}...] Progress: {position_idx}/{len(positions)} positions "
                    f"({position_idx/len(positions)*100:.1f}%)"
                )

            if not position.orders:
                bt.logging.warning(
                    f"Skipping position {position.position_uuid} - no orders"
                )
                continue

            # Check if migration needed
            needs_account_size, needs_order_migration = check_position_needs_migration(position)

            if not needs_account_size and not needs_order_migration:
                continue

            # Migrate account size first (needed for order calculations)
            if needs_account_size:
                old_account_size = position.account_size
                position.account_size = get_account_size_for_order(
                    hotkey, position.orders[0].processed_ms, miner_account_sizes_cache, contract_manager
                )
                stats['account_size_migrations'] += 1
                bt.logging.debug(
                    f"Migrated account_size for {position.position_uuid}: "
                    f"{old_account_size} → ${position.account_size:,.2f}"
                )

            # Migrate orders
            if needs_order_migration:
                quantity_migrated, usd_rate_migrated = migrate_order_quantities(
                    position, live_price_fetcher
                )
                stats['order_quantity_migrations'] += quantity_migrated
                stats['order_usd_rate_migrations'] += usd_rate_migrated

                if quantity_migrated > 0:
                    bt.logging.debug(
                        f"Migrated {quantity_migrated} orders for position "
                        f"{position.position_uuid}: "
                        f"leverage={position.orders[0].leverage} → "
                        f"quantity={position.orders[0].quantity}"
                    )

            # Rebuild position with updated orders
            position.rebuild_position_with_updated_orders(live_price_fetcher)

            if hotkey == "5F6oea6yYMFWETD9KkWj5FMUntRv77yCw7WoiJv5tVsor2Mb":
                print(position)

            # Save position
            if not dry_run:
                save_position(position)

            stats['migrated'] += 1

        except Exception as e:
            stats['failed'] += 1
            error_msg = f"Failed to migrate position {position.position_uuid}: {e}"
            bt.logging.error(error_msg)
            stats['errors'].append(error_msg)
            continue

    return stats

# Statistics tracking
stats = {
    'total_positions': 0,
    'positions_needing_migration': 0,
    'positions_migrated': 0,
    'positions_failed': 0,
    'account_size_migrations': 0,
    'order_quantity_migrations': 0,
    'order_usd_rate_migrations': 0,
    'positions_by_hotkey': defaultdict(lambda: {'total': 0, 'migrated': 0}),
    'errors': []
}

# Load all positions
all_positions = load_all_positions()
stats['total_positions'] = sum(len(positions) for positions in all_positions.values())

if stats['total_positions'] == 0:
    bt.logging.error("No positions found. Exiting.")
    sys.exit(1)

print(f"\nStarting migration of {stats['total_positions']} positions...")
print("=" * 80)

# Start timer
migration_start_time = time.time()

# Cache for miner account sizes
miner_account_sizes_cache = {}
if contract_manager:
    miner_account_sizes_cache = contract_manager.miner_account_sizes.copy()

# Prepare arguments for parallel processing
print(f"Preparing {len(all_positions)} hotkeys for parallel processing with {NUM_PROCESSES} processes...")
process_args = [
    (hotkey, positions, DRY_RUN, miner_account_sizes_cache)
    for hotkey, positions in all_positions.items()
]

# Process hotkeys in parallel
print(f"Starting parallel migration...")
if NUM_PROCESSES > 1:
    with Pool(processes=NUM_PROCESSES) as pool:
        hotkey_results = pool.map(process_hotkey, process_args)
else:
    # Single process mode for debugging
    hotkey_results = [process_hotkey(args) for args in process_args]

# Aggregate results from all workers
print("\nAggregating results from all workers...")
for result in hotkey_results:
    hotkey = result['hotkey']

    # Update global stats
    stats['positions_migrated'] += result['migrated']
    stats['positions_failed'] += result['failed']
    stats['account_size_migrations'] += result['account_size_migrations']
    stats['order_quantity_migrations'] += result['order_quantity_migrations']
    stats['order_usd_rate_migrations'] += result['order_usd_rate_migrations']

    # Track positions needing migration (migrated + failed)
    stats['positions_needing_migration'] += (result['migrated'] + result['failed'])

    # Update per-hotkey stats
    stats['positions_by_hotkey'][hotkey]['total'] = result['total']
    stats['positions_by_hotkey'][hotkey]['migrated'] = result['migrated']

    # Collect errors
    if result['errors']:
        stats['errors'].extend(result['errors'])

# End timer
migration_end_time = time.time()
migration_duration = migration_end_time - migration_start_time

# Print final statistics
print("\n" + "=" * 80)
print("MIGRATION SUMMARY")
print("=" * 80)
print(f"Total positions processed:        {stats['total_positions']}")
print(f"Positions needing migration:      {stats['positions_needing_migration']}")
print(f"Positions successfully migrated:  {stats['positions_migrated']}")
print(f"Positions failed:                 {stats['positions_failed']}")
print("")
print(f"Account size migrations:          {stats['account_size_migrations']}")
print(f"Order quantity migrations:        {stats['order_quantity_migrations']}")
print(f"Order USD rate migrations:        {stats['order_usd_rate_migrations']}")
print("")
print(f"Migration duration:               {migration_duration:.2f} seconds ({migration_duration/60:.2f} minutes)")

if stats['positions_needing_migration'] > 0:
    success_rate = (stats['positions_migrated'] / stats['positions_needing_migration']) * 100
    print(f"\nSuccess rate: {success_rate:.2f}%")

# Per-hotkey statistics
print("\n" + "=" * 80)
print("PER-HOTKEY BREAKDOWN")
print("=" * 80)
for hotkey in sorted(stats['positions_by_hotkey'].keys()):
    hk_stats = stats['positions_by_hotkey'][hotkey]
    if hk_stats['migrated'] > 0:
        print(
            f"{hotkey}: {hk_stats['migrated']}/{hk_stats['total']} positions migrated"
        )

# Show errors if any
if stats['errors']:
    print("\n" + "=" * 80)
    print(f"ERRORS ({len(stats['errors'])} total)")
    print("=" * 80)
    for error in stats['errors'][:10]:  # Show first 10 errors
        print(error)
    if len(stats['errors']) > 10:
        print(f"... and {len(stats['errors']) - 10} more errors")

if DRY_RUN:
    print("\n" + "=" * 80)
    print("[DRY RUN] No files were modified")
    print("Run without --dry-run to apply changes")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("Migration completed successfully!")
    print("=" * 80)
