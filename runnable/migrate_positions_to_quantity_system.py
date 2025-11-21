"""
Migration script to update positions with quantity and USD conversion fields.

This script migrates historical positions that only have leverage to include:
- order.quantity (number of lots/coins/shares/etc.)
- order.usd_base_rate (USD to base currency conversion)
- order.quote_usd_rate (quote to USD conversion)
- position.account_size (historical account size from collateral)

Usage:
    python runnable/migrate_positions_to_quantity_system.py [--dry-run]
"""

import sys
import bittensor as bt
from collections import defaultdict
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
COLLATERAL_START_TIME_MS = 1755302399000

# Check for dry-run argument
if len(sys.argv) > 1 and sys.argv[1] in ['--dry-run', '-n']:
    DRY_RUN = True
    print("*** DRY RUN MODE - No files will be modified ***\n")

# Initialize services
bt.logging.info("Initializing services...")
secrets = ValiUtils.get_secrets()
live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)

# Initialize contract manager for account size lookups
try:
    contract_manager = ValidatorContractManager(
        config=None,
        wallet=None,
        metagraph=None,
        running_unit_tests=False
    )
    bt.logging.info("Contract manager initialized successfully")
except Exception as e:
    bt.logging.warning(f"Could not initialize contract manager: {e}. Will use default account sizes.")
    contract_manager = None

def get_account_size_for_order(position, order, miner_account_sizes_cache):
    """
    Get the miner's account size for an order based on collateral history.
    """
    if order.processed_ms < COLLATERAL_START_TIME_MS:
        return ValiConfig.DEFAULT_CAPITAL

    if not contract_manager:
        return ValiConfig.DEFAULT_CAPITAL

    try:
        account_size = contract_manager.get_miner_account_size(
            position.miner_hotkey,
            order.processed_ms,
            records_dict=miner_account_sizes_cache
        )
        return account_size if account_size is not None else ValiConfig.MIN_CAPITAL
    except Exception as e:
        bt.logging.warning(f"Error getting account size for {position.miner_hotkey}: {e}")
        return ValiConfig.MIN_CAPITAL

def migrate_order_quantities(position: Position, miner_account_sizes_cache: dict) -> tuple[int, int]:
    """
    Migrate orders to include quantity and USD conversion rates.
    Returns (orders_migrated_for_quantity, orders_migrated_for_usd_rates)
    """
    quantity_migrated = 0
    usd_rate_migrated = 0

    for order in position.orders:
        # Migrate USD conversion rates
        needs_usd_migration = (order.quote_usd_rate == 1.0 or order.usd_base_rate == 1.0)
        if needs_usd_migration:
            try:
                order.quote_usd_rate = live_price_fetcher.get_quote_usd_conversion(
                    order, position.orders[0].order_type
                )
                order.usd_base_rate = live_price_fetcher.get_usd_base_conversion(
                    order.trade_pair, order.processed_ms, order.price,
                    order.order_type, position.orders[0].order_type
                )
                usd_rate_migrated += 1
            except Exception as e:
                bt.logging.warning(
                    f"Failed to migrate USD rates for order {order.order_uuid}: {e}"
                )

        # Migrate quantity
        if order.quantity is None and order.leverage is not None:
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
        (o.quantity is None and o.leverage is not None) or
        o.quote_usd_rate == 1.0 or
        o.usd_base_rate == 1.0
        for o in position.orders
    )
    return needs_account_size, needs_order_migration

def load_all_positions() -> dict[str, list[Position]]:
    """Load all positions (both open and closed) from disk."""
    bt.logging.info("Loading positions from disk...")

    all_positions = defaultdict(list)

    # Get all hotkey directories
    base_dir = ValiBkpUtils.get_miner_all_positions_dir(running_unit_tests=False)

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
        bt.logging.info(
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

bt.logging.info(f"\nStarting migration of {stats['total_positions']} positions...")
bt.logging.info("=" * 80)

# Cache for miner account sizes
miner_account_sizes_cache = {}
if contract_manager:
    miner_account_sizes_cache = contract_manager.miner_account_sizes.copy()

# Process each hotkey's positions
for hotkey_idx, (hotkey, positions) in enumerate(all_positions.items(), 1):
    bt.logging.info(
        f"\n[{hotkey_idx}/{len(all_positions)}] Processing hotkey {hotkey} "
        f"with {len(positions)} positions..."
    )

    stats['positions_by_hotkey'][hotkey]['total'] = len(positions)

    for position_idx, position in enumerate(positions, 1):
        try:
            # Progress update every 100 positions
            if position_idx % 100 == 0:
                bt.logging.info(
                    f"  Progress: {position_idx}/{len(positions)} positions "
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

            stats['positions_needing_migration'] += 1

            # Migrate account size first (needed for order calculations)
            if needs_account_size:
                old_account_size = position.account_size
                position.account_size = get_account_size_for_order(
                    position, position.orders[0], miner_account_sizes_cache
                )
                stats['account_size_migrations'] += 1
                bt.logging.debug(
                    f"Migrated account_size for {position.position_uuid}: "
                    f"{old_account_size} → ${position.account_size:,.2f}"
                )

            # Migrate orders
            if needs_order_migration:
                quantity_migrated, usd_rate_migrated = migrate_order_quantities(
                    position, miner_account_sizes_cache
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

            # Save position
            if not DRY_RUN:
                save_position(position)

            stats['positions_migrated'] += 1
            stats['positions_by_hotkey'][hotkey]['migrated'] += 1

        except Exception as e:
            stats['positions_failed'] += 1
            error_msg = f"Failed to migrate position {position.position_uuid}: {e}"
            bt.logging.error(error_msg)
            stats['errors'].append(error_msg)

            # Continue with next position instead of crashing
            continue

# Print final statistics
bt.logging.info("\n" + "=" * 80)
bt.logging.info("MIGRATION SUMMARY")
bt.logging.info("=" * 80)
bt.logging.info(f"Total positions processed:        {stats['total_positions']}")
bt.logging.info(f"Positions needing migration:      {stats['positions_needing_migration']}")
bt.logging.info(f"Positions successfully migrated:  {stats['positions_migrated']}")
bt.logging.info(f"Positions failed:                 {stats['positions_failed']}")
bt.logging.info("")
bt.logging.info(f"Account size migrations:          {stats['account_size_migrations']}")
bt.logging.info(f"Order quantity migrations:        {stats['order_quantity_migrations']}")
bt.logging.info(f"Order USD rate migrations:        {stats['order_usd_rate_migrations']}")

if stats['positions_needing_migration'] > 0:
    success_rate = (stats['positions_migrated'] / stats['positions_needing_migration']) * 100
    bt.logging.info(f"\nSuccess rate: {success_rate:.2f}%")

# Per-hotkey statistics
bt.logging.info("\n" + "=" * 80)
bt.logging.info("PER-HOTKEY BREAKDOWN")
bt.logging.info("=" * 80)
for hotkey in sorted(stats['positions_by_hotkey'].keys()):
    hk_stats = stats['positions_by_hotkey'][hotkey]
    if hk_stats['migrated'] > 0:
        bt.logging.info(
            f"{hotkey}: {hk_stats['migrated']}/{hk_stats['total']} positions migrated"
        )

# Show errors if any
if stats['errors']:
    bt.logging.info("\n" + "=" * 80)
    bt.logging.info(f"ERRORS ({len(stats['errors'])} total)")
    bt.logging.info("=" * 80)
    for error in stats['errors'][:10]:  # Show first 10 errors
        bt.logging.info(error)
    if len(stats['errors']) > 10:
        bt.logging.info(f"... and {len(stats['errors']) - 10} more errors")

if DRY_RUN:
    bt.logging.info("\n" + "=" * 80)
    bt.logging.info("[DRY RUN] No files were modified")
    bt.logging.info("Run without --dry-run to apply changes")
    bt.logging.info("=" * 80)
else:
    bt.logging.info("\n" + "=" * 80)
    bt.logging.info("Migration completed successfully!")
    bt.logging.info("=" * 80)
