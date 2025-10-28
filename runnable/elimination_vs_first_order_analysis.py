#!/usr/bin/env python3
"""
Daily Portfolio Returns Calculator

This script calculates daily portfolio returns for each miner by:
1. Loading positions from database using PositionSourceManager API
2. Loading eliminations for date boundary calculation
3. Stepping through time day by day (UTC)
4. Calculating total portfolio return at the beginning of each UTC day
5. Using live_price_fetcher.get_close_at_date for pricing

Usage:
    python daily_portfolio_returns.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--hotkeys HOTKEY1,HOTKEY2,...] [--elimination-source DATABASE]
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict

from daily_portfolio_returns import SharedDataManager, get_database_url_from_config, EliminationTracker
import bittensor as bt
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, text, inspect, tuple_
from sqlalchemy.orm import declarative_base, sessionmaker

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_source import PositionSourceManager, PositionSource
from vali_objects.vali_config import TradePair, TradePairCategory, CryptoSubcategory, ForexSubcategory
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.utils.vali_utils import ValiUtils

# Database setup
Base = declarative_base()



class MinerPortValuesModel(Base):
    """Database model for storing miner port values matching taoshi-ts-ptn.miner_port_values schema."""
    __tablename__ = 'miner_port_values'

    miner_hotkey = Column(String(255), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    date = Column(String(10))  # YYYY-MM-DD format
    # Aggregate port values
    all_port_value = Column(Float)
    crypto_port_value = Column(Float)
    forex_port_value = Column(Float)
    # Crypto subcategory port values
    crypto_majors_port_value = Column(Float)
    crypto_alts_port_value = Column(Float)
    # Forex subcategory port values
    forex_group1_port_value = Column(Float)
    forex_group2_port_value = Column(Float)
    forex_group3_port_value = Column(Float)
    forex_group4_port_value = Column(Float)
    forex_group5_port_value = Column(Float)
    # Count columns
    all_count = Column(Integer)
    crypto_count = Column(Integer)
    forex_count = Column(Integer)
    crypto_majors_count = Column(Integer)
    crypto_alts_count = Column(Integer)
    forex_group1_count = Column(Integer)
    forex_group2_count = Column(Integer)
    forex_group3_count = Column(Integer)
    forex_group4_count = Column(Integer)
    forex_group5_count = Column(Integer)



class ReturnCalculator:
    """Handles portfolio and position return calculations."""

    @staticmethod
    def calculate_position_return(
            position: Position,
            target_date_ms: int,
            cached_price_sources: Dict[TradePair, PriceSource],
            live_price_fetcher: LivePriceFetcher
    ) -> float:
        """Calculate return for a single position."""
        # If position is closed and closed before/at target date, use actual return
        if position.is_closed_position and position.close_ms and position.close_ms <= target_date_ms:
            return position.return_at_close

        # For open positions, calculate using cached price
        if position.trade_pair not in cached_price_sources:
            raise ValueError(f"Price not available for {position.trade_pair.trade_pair} at target date")

        price_source = cached_price_sources[position.trade_pair]

        # Create a copy to avoid modifying the original position
        position_copy = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms,
            trade_pair=position.trade_pair,
            orders=position.orders[:],
            position_type=position.position_type,
            is_closed_position=position.is_closed_position,
        )
        position_copy.rebuild_position_with_updated_orders(live_prive_fetcher)

        price = price_source.parse_appropriate_price(
            now_ms=target_date_ms,
            is_forex=position.trade_pair.is_forex,
            order_type=position.orders[0].order_type,
            position_type=position_copy.orders[0].order_type
        )

        position_copy.set_returns(realtime_price=price, time_ms=target_date_ms)
        return position_copy.return_at_close


class PositionCategorizer:
    """Handles categorization of positions into crypto/forex subcategories."""

    @staticmethod
    def categorize_position(trade_pair_obj: TradePair) -> List[str]:
        """
        Determine which category(ies) a trade pair belongs to using the TradePair enum.

        Args:
            trade_pair_obj: TP

        Returns:
            List[str]: A list of category names
        """
        categories = []

        # Add to 'all' category by default
        categories.append('all')

        # Get the TradePair object from the ID

        if trade_pair_obj is None:
            bt.logging.warning(f"Unknown trade pair ID: {trade_pair_obj.trade_pair_id}")
            return categories

        # Get the trade pair info (list format from enum)
        trade_pair_info = trade_pair_obj.value

        # Extract category and subcategory from the trade pair info
        # Format: [id, display_name, tick_size, min_leverage, max_leverage, category, subcategory]
        if len(trade_pair_info) >= 7:
            category = trade_pair_info[5]  # TradePairCategory
            subcategory = trade_pair_info[6]  # Subcategory (CryptoSubcategory or ForexSubcategory)

            # Add the primary category
            if category == TradePairCategory.CRYPTO:
                categories.append('crypto')

                # Add crypto subcategories
                if subcategory == CryptoSubcategory.MAJORS:
                    categories.append('crypto_majors')
                elif subcategory == CryptoSubcategory.ALTS:
                    categories.append('crypto_alts')

            elif category == TradePairCategory.FOREX:
                categories.append('forex')

                # Add forex subcategories
                if subcategory == ForexSubcategory.G1:
                    categories.append('forex_group1')
                elif subcategory == ForexSubcategory.G2:
                    categories.append('forex_group2')
                elif subcategory == ForexSubcategory.G3:
                    categories.append('forex_group3')
                elif subcategory == ForexSubcategory.G4:
                    categories.append('forex_group4')
                elif subcategory == ForexSubcategory.G5:
                    categories.append('forex_group5')

        return categories


class CategoryReturnCalculator:
    """Calculates returns by asset categories and subcategories."""

    @staticmethod
    def calculate_miner_returns_by_category(
            positions: List[Position],
            target_date_ms: int,
            cached_price_sources: Dict[TradePair, PriceSource]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate portfolio values by category for each miner based on positions.
        Follows the same logic as miner_returns.py reference implementation.

        Args:
            positions: List of positions for all miners
            target_date_ms: Target timestamp
            cached_price_sources: Cached price sources for calculations

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Nested dict of miner_hotkey -> category -> {"return": value, "count": count}
        """

        # Initialize data structure for results (following reference exactly)
        # Format: {miner_hotkey: {category: {"return": cumulative_portfolio_value, "count": count}}}
        miner_category_returns = defaultdict(
            lambda: defaultdict(lambda: {"return": 1.0, "count": 0}))

        secrets = ValiUtils.get_secrets()
        live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
        # Process each position (following reference logic exactly)
        for position in positions:
            # Calculate return_at_close for this position at the target date
            position_return = ReturnCalculator.calculate_position_return(
                position, target_date_ms, cached_price_sources, live_price_fetcher
            )

            # Fail fast - if we can't calculate return, something is wrong
            # if not position_return:
            #    raise ValueError(f"Position {position.position_uuid} returned invalid return: {position_return}")

            # Fail fast - if return is not positive, something is wrong
            if position_return < 0:
                raise ValueError(f"Position {position.position_uuid} returned non-positive return: {position_return}")

            miner_hotkey = position.miner_hotkey
            trade_pair_id = position.trade_pair.trade_pair_id  # Use ID not display name

            # Determine categories for this position
            categories = PositionCategorizer.categorize_position(position.trade_pair)

            # Add return to each relevant category
            for category in categories:
                # Directly multiply the return values together (reference logic)
                miner_category_returns[miner_hotkey][category]["return"] *= position_return
                miner_category_returns[miner_hotkey][category]["count"] += 1

        # Convert the results to the final format (following reference)
        results = {}
        for miner_hotkey, categories in miner_category_returns.items():
            results[miner_hotkey] = {}
            for category, data in categories.items():
                results[miner_hotkey][category] = {
                    "return": data["return"],
                    "count": data["count"]
                }

        return results

    @staticmethod
    def prepare_insert_values(
            miner_returns: Dict[str, Dict[str, Dict[str, float]]],
            target_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """
        Prepare insert values for database from calculated portfolio values.

        Args:
            miner_returns: Dictionary of miner portfolio values by category
            target_timestamp: The timestamp for the database records

        Returns:
            List[Dict[str, Any]]: A list of dictionaries for database insertion
        """
        insert_values = []
        for miner_hotkey, categories in miner_returns.items():
            insert_values.append({
                "miner_hotkey": miner_hotkey,
                "timestamp": target_timestamp,
                "date": target_timestamp.strftime("%Y-%m-%d"),
                # Aggregate port values
                "all_port_value": categories.get("all", {}).get("return", 1.0),
                "crypto_port_value": categories.get("crypto", {}).get("return", 1.0),
                "forex_port_value": categories.get("forex", {}).get("return", 1.0),
                # Crypto subcategory port values
                "crypto_majors_port_value": categories.get("crypto_majors", {}).get("return", 1.0),
                "crypto_alts_port_value": categories.get("crypto_alts", {}).get("return", 1.0),
                # Forex subcategory port values
                "forex_group1_port_value": categories.get("forex_group1", {}).get("return", 1.0),
                "forex_group2_port_value": categories.get("forex_group2", {}).get("return", 1.0),
                "forex_group3_port_value": categories.get("forex_group3", {}).get("return", 1.0),
                "forex_group4_port_value": categories.get("forex_group4", {}).get("return", 1.0),
                "forex_group5_port_value": categories.get("forex_group5", {}).get("return", 1.0),
                # Count columns
                "all_count": categories.get("all", {}).get("count", 0),
                "crypto_count": categories.get("crypto", {}).get("count", 0),
                "forex_count": categories.get("forex", {}).get("count", 0),
                "crypto_majors_count": categories.get("crypto_majors", {}).get("count", 0),
                "crypto_alts_count": categories.get("crypto_alts", {}).get("count", 0),
                "forex_group1_count": categories.get("forex_group1", {}).get("count", 0),
                "forex_group2_count": categories.get("forex_group2", {}).get("count", 0),
                "forex_group3_count": categories.get("forex_group3", {}).get("count", 0),
                "forex_group4_count": categories.get("forex_group4", {}).get("count", 0),
                "forex_group5_count": categories.get("forex_group5", {}).get("count", 0)
            })
        return insert_values





def analyze_miners_with_orders(all_positions: Dict[str, List[Position]]) -> Dict[str, Dict[str, Any]]:
    """Analyze miners to find their order date ranges.

    Args:
        all_positions: Dict mapping hotkey -> list of positions

    Returns:
        Dict mapping hotkey -> {first_order_date, last_order_date, missing_days}
    """
    miners_with_orders = {}

    bt.logging.info("Analyzing order data for all miners...")

    for hotkey, positions in all_positions.items():
        order_timestamps = []

        for position in positions:
            for order in position.orders:
                order_timestamps.append(order.processed_ms)

        if order_timestamps:
            first_order_ms = min(order_timestamps)
            last_order_ms = max(order_timestamps)

            first_order_date = datetime.fromtimestamp(first_order_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            last_order_date = datetime.fromtimestamp(last_order_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            # Note: Don't calculate missing_days here - it requires database lookup
            # The actual missing days will be calculated in check_portfolio_coverage()

            miners_with_orders[hotkey] = {
                "first_order_date": first_order_date,
                "last_order_date": last_order_date
            }

    return miners_with_orders




def main():
    """Main function to calculate daily portfolio returns."""
    # Initialize position source manager
    position_source_manager = PositionSourceManager(source_type=PositionSource.DATABASE)

    # Load all positions first to determine date bounds
    bt.logging.info("Loading positions from database...")
    all_positions = position_source_manager.load_positions(
        start_time_ms=None,  # Get all positions to determine bounds
        end_time_ms=None,
        hotkeys=None
    )

    if not all_positions:
        bt.logging.error("No positions found in database")
        return

    # Initialize elimination tracker
    elimination_tracker = EliminationTracker()

    # Load all eliminations for filtering
    elimination_timestamps = elimination_tracker.load_all_eliminations(hotkeys=None)

    # Log elimination loading summary
    if elimination_timestamps:
        bt.logging.info(
            f"✓ Elimination filtering enabled: {len(elimination_timestamps)} miners have eliminations on record")
    else:
        bt.logging.info("ℹ Elimination filtering disabled: No elimination data available or failed to load")

    bt.logging.info("🚀 Running automated backfill mode...")

    # Analyze miners with orders to determine what needs backfilling
    miners_with_orders = analyze_miners_with_orders(all_positions)

    bt.logging.info(f"Found {len(miners_with_orders)} miners with orders. Checking portfolio coverage...")

    # Initialize shared data manager for efficient backfill processing
    database_url = None
    if not database_url:
        database_url = get_database_url_from_config()
    if not database_url:
        try:
            secrets = ValiUtils.get_secrets()
            database_url = secrets.get('database_url') or secrets.get('db_ptn_editor_url')
        except Exception as e:
            bt.logging.debug(f"Could not get database URL from ValiUtils secrets: {e}")

    if not database_url:
        bt.logging.error("No database URL available for automated backfill. Use --database-url argument.")
        return

    # Initialize SharedDataManager with all data loaded at once
    bt.logging.info("🚀 Initializing shared data manager for efficient backfill processing...")
    shared_data_manager = SharedDataManager(database_url, hotkeys=None)
    shared_data_manager.initialize_all_data()

    # Analyze first order times for miners where order.src = 1
    bt.logging.info("📊 Analyzing first order times for orders with src = 1...")
    
    first_order_timestamps = {}
    
    for hotkey, positions in all_positions.items():
        qualifying_order_times = []
        
        for position in positions:
            for order in position.orders:
                # Filter for orders with src = 1 only
                if hasattr(order, 'src') and order.src == 1:
                    qualifying_order_times.append(order.processed_ms)
        
        if qualifying_order_times:
            # Get the earliest qualifying order time
            first_order_ms = min(qualifying_order_times)
            first_order_timestamps[hotkey] = first_order_ms
    
    bt.logging.info(f"Found {len(first_order_timestamps)} miners with qualifying orders (src = 1)")
    
    # Compare elimination times to first order times
    bt.logging.info("📊 Comparing elimination times to first order times...")
    
    # Find miners that have both elimination data and first order data
    elim_hotkeys = set(elimination_timestamps.keys())
    order_hotkeys = set(first_order_timestamps.keys())
    common_miners = elim_hotkeys & order_hotkeys
    only_eliminated = elim_hotkeys - order_hotkeys
    only_orders = order_hotkeys - elim_hotkeys
    
    bt.logging.info(f"Data summary:")
    bt.logging.info(f"  - {len(common_miners)} miners with both elimination and first order data")
    bt.logging.info(f"  - {len(only_eliminated)} miners with only elimination data (no src = 1 orders)")
    bt.logging.info(f"  - {len(only_orders)} miners with only first order data (not eliminated)")
    bt.logging.info("📊 Processing complete - generating analysis report...")
    
    # Collect results for miners with both data points
    both_data_results = []
    for hotkey in common_miners:
        first_order_ms = first_order_timestamps[hotkey]
        elimination_ms = elimination_timestamps[hotkey]
        
        # Get elimination reason from the full elimination data
        elimination_reason = "Unknown"
        if hotkey in shared_data_manager.elimination_tracker.eliminations_by_hotkey:
            eliminations = shared_data_manager.elimination_tracker.eliminations_by_hotkey[hotkey]
            if eliminations:
                # Find the elimination record that matches our timestamp
                for elim in eliminations:
                    if elim.get('elimination_time_ms') == elimination_ms:
                        elimination_reason = elim.get('elimination_reason', 'Unknown')
                        break
                # If no exact match, use the first elimination's reason
                if elimination_reason == "Unknown" and eliminations:
                    elimination_reason = eliminations[0].get('elimination_reason', 'Unknown')
        
        # Convert to formatted UTC date strings
        first_order_date = TimeUtil.millis_to_formatted_date_str(first_order_ms)
        elimination_date = TimeUtil.millis_to_formatted_date_str(elimination_ms)
        
        # Calculate time difference in hours and days (NOT absolute - shows direction)
        time_diff_ms = elimination_ms - first_order_ms  # Positive = elimination after order, Negative = elimination before order
        time_diff_hours = time_diff_ms / (60 * 60 * 1000)
        time_diff_days = time_diff_ms / (24 * 60 * 60 * 1000)  # Use float division for more precision
        
        both_data_results.append((hotkey, first_order_date, elimination_date, time_diff_hours, time_diff_days, elimination_reason))
    
    # Sort by time difference (shortest to longest survival time)
    both_data_results.sort(key=lambda x: x[3])
    
    # Print comprehensive results
    print("\n" + "="*140)
    print("ELIMINATION vs FIRST ORDER ANALYSIS RESULTS")
    print("="*140)
    
    # Section 1: Miners with both elimination and first order data
    if both_data_results:
        print(f"\n1. MINERS WITH BOTH ELIMINATION AND FIRST ORDER DATA ({len(both_data_results)} miners)")
        print("-"*140)
        print(f"{'Miner Hotkey':<50} {'First Order (UTC)':<20} {'Elimination (UTC)':<20} {'Hours Diff':<12} {'Days Diff':<10} {'Elimination Reason':<18}")
        print("-"*140)
        
        # Track miners that need elimination time updates
        mismatched_miners = []
        
        for hotkey, first_order_date, elimination_date, hours_diff, days_diff, elimination_reason in both_data_results:
            # Truncate hotkey for display
            display_hotkey = hotkey[:47] + "..." if len(hotkey) > 50 else hotkey
            # Handle None elimination reason and truncate for display
            if elimination_reason is None:
                display_reason = "Unknown"
            else:
                display_reason = elimination_reason[:17] + "..." if len(elimination_reason) > 20 else elimination_reason
            print(f"{display_hotkey:<50} {first_order_date:<20} {elimination_date:<20} {hours_diff:<12.1f} {days_diff:<10.1f} {display_reason:<18}")
            
            # Check if elimination time doesn't match first_order_ms exactly
            first_order_ms = first_order_timestamps[hotkey]
            elimination_ms = elimination_timestamps[hotkey]
            expected_elimination_ms = first_order_ms
            
            if elimination_ms != expected_elimination_ms:
                mismatched_miners.append((hotkey, first_order_ms, elimination_ms, expected_elimination_ms))
        
        print("-"*140)
        print(f"Total: {len(both_data_results)} miners analyzed")
        
        # Generate SQL for mismatched elimination times
        if mismatched_miners:
            print(f"\n1a. BULK SQL TO UPDATE MISMATCHED ELIMINATION TIMES ({len(mismatched_miners)} miners)")
            print("-"*140)
            print("-- Execute this bulk UPDATE to set elimination times to match first_order_ms exactly:")
            print("-- Updates elimination_ms and updated_ms, keeps original elimination_reason and max_drawdown")
            print("-"*140)
            
            current_time_ms = TimeUtil.now_in_millis()
            
            # Create bulk UPDATE with CASE statements
            print("UPDATE eliminations SET")
            print(f"  elimination_ms = CASE miner_hotkey")
            for hotkey, first_order_ms, current_elimination_ms, expected_elimination_ms in mismatched_miners:
                print(f"    WHEN '{hotkey}' THEN {expected_elimination_ms}")
            print(f"    ELSE elimination_ms")
            print(f"  END,")
            print(f"  updated_ms = {current_time_ms}")
            print(f"WHERE miner_hotkey IN (")
            hotkey_list = [f"'{hotkey}'" for hotkey, _, _, _ in mismatched_miners]
            print(f"  {', '.join(hotkey_list)}")
            print(f");")
            
            print("-"*140)
            print(f"-- Total: {len(mismatched_miners)} miners will be updated in single bulk command")
            print(f"-- These will set elimination_ms = first_order_ms for mismatched miners")
        else:
            print(f"\n1a. NO ELIMINATION TIME UPDATES NEEDED")
            print("All miners already have elimination_ms = first_order_ms")
            
    else:
        print(f"\n1. MINERS WITH BOTH ELIMINATION AND FIRST ORDER DATA (0 miners)")
        print("No miners found with both elimination data and first order data with src = 1")
    
    # Section 2: Miners with only elimination data (no src = 1 orders)
    if only_eliminated:
        print(f"\n2. ELIMINATED MINERS WITHOUT SRC = 1 ORDERS ({len(only_eliminated)} miners)")
        print("-"*140)
        print(f"{'Miner Hotkey':<50} {'Elimination Date (UTC)':<30} {'Note':<60}")
        print("-"*140)
        
        for hotkey in sorted(only_eliminated):
            elimination_ms = elimination_timestamps[hotkey]
            elimination_date = TimeUtil.millis_to_formatted_date_str(elimination_ms)
            display_hotkey = hotkey[:47] + "..." if len(hotkey) > 50 else hotkey
            print(f"{display_hotkey:<50} {elimination_date:<30} {'No orders with src = 1 found':<60}")
        
        print("-"*140)
        print(f"Total: {len(only_eliminated)} eliminated miners without qualifying orders")
    else:
        print(f"\n2. ELIMINATED MINERS WITHOUT SRC = 1 ORDERS (0 miners)")
        print("All eliminated miners have at least one order with src = 1")
    
    # Section 3: Miners with only first order data (not eliminated) - Generate SQL for these
    if only_orders:
        print(f"\n3. NON-ELIMINATED MINERS WITH SRC = 1 ORDERS ({len(only_orders)} miners)")
        print("-"*140)
        print(f"{'Miner Hotkey':<50} {'First Order Date (UTC)':<30} {'Note':<60}")
        print("-"*140)
        
        # Show all miners and collect SQL statements
        sql_statements = []
        for hotkey in sorted(only_orders):
            first_order_ms = first_order_timestamps[hotkey]
            first_order_date = TimeUtil.millis_to_formatted_date_str(first_order_ms)
            display_hotkey = hotkey[:47] + "..." if len(hotkey) > 50 else hotkey
            print(f"{display_hotkey:<50} {first_order_date:<30} {'Not eliminated (needs reconciliation)':<60}")
            
            # Generate SQL statement: elimination_ms = first_order_ms + 1, dd is NULL
            elimination_ms = first_order_ms + 1
            creation_ms = TimeUtil.now_in_millis()
            
            sql_statement = (
                f"INSERT INTO eliminations (miner_hotkey, elimination_ms, elimination_reason, "
                f"max_drawdown, creation_ms, updated_ms) VALUES "
                f"('{hotkey}', {elimination_ms}, 'RECONCILE_ORDER_SRC', NULL, {creation_ms}, {creation_ms});"
            )
            sql_statements.append(sql_statement)
        
        print("-"*140)
        print(f"Total: {len(only_orders)} non-eliminated miners with qualifying orders")
        
        # Print SQL statements section
        print(f"\n4. BULK SQL TO RECONCILE NON-ELIMINATED MINERS")
        print("-"*140)
        print("-- Execute this bulk INSERT to add elimination records for miners with src = 1 orders but no elimination:")
        print("-- Elimination time is set to first_order_ms, reason = 'RECONCILE_ORDER_SRC', max_drawdown = NULL")
        print("-"*140)
        
        # Create bulk INSERT with VALUES
        creation_ms = TimeUtil.now_in_millis()
        print("INSERT INTO eliminations (miner_hotkey, elimination_ms, elimination_reason, max_drawdown, creation_ms, updated_ms)")
        print("VALUES")
        
        value_rows = []
        for hotkey in sorted(only_orders):
            first_order_ms = first_order_timestamps[hotkey]
            elimination_ms = first_order_ms  # Set elimination time to exactly match first order time
            value_row = f"  ('{hotkey}', {elimination_ms}, 'RECONCILE_ORDER_SRC', NULL, {creation_ms}, {creation_ms})"
            value_rows.append(value_row)
        
        print(",\n".join(value_rows))
        print(";")
        
        print("-"*140)
        print(f"-- Total: {len(only_orders)} miners will be inserted in single bulk command")
        
    else:
        print(f"\n3. NON-ELIMINATED MINERS WITH SRC = 1 ORDERS (0 miners)")
        print("All miners with src = 1 orders have been eliminated")
    
    # Section 5: Generate SQL to delete invalid portfolio records
    if elimination_timestamps:
        print(f"\n5. BULK SQL TO DELETE INVALID PORTFOLIO RECORDS")
        print("-" * 140)
        print("-- Delete portfolio records beyond elimination_date + 1 day")
        print("-- Only affects eliminated miners with portfolio data after their elimination")
        print("-" * 140)
        
        delete_conditions = []
        for hotkey, elimination_ms in elimination_timestamps.items():
            # Calculate cutoff date (elimination + 1 day)
            cutoff_date_ms = elimination_ms + (24 * 60 * 60 * 1000)
            cutoff_date_str = TimeUtil.millis_to_formatted_date_str(cutoff_date_ms)[:10]  # YYYY-MM-DD
            delete_conditions.append(f"(miner_hotkey = '{hotkey}' AND date > '{cutoff_date_str}')")
        
        print("DELETE FROM `taoshi-ts-ptn`.`miner_port_values`")
        print("WHERE")
        print("  " + " OR\n  ".join(delete_conditions))
        print(";")
        
        print("-" * 140)
        print(f"-- Total: {len(elimination_timestamps)} eliminated miners checked")
        print("-- Records with dates beyond elimination_date + 1 day will be deleted")
    else:
        print(f"\n5. NO PORTFOLIO CLEANUP NEEDED")
        print("No elimination data available - no portfolio records to clean up")
    
    print("="*140)
    
    # Summary statistics
    total_unique_miners = len(elim_hotkeys | order_hotkeys)
    bt.logging.info(f"Analysis summary:")
    bt.logging.info(f"  - Total unique miners: {total_unique_miners}")
    bt.logging.info(f"  - Miners with both data: {len(both_data_results)}")
    bt.logging.info(f"  - Eliminated only: {len(only_eliminated)}")
    bt.logging.info(f"  - Orders only: {len(only_orders)}")

    return



if __name__ == "__main__":
    # Prevent duplicate logging setup
    if not hasattr(bt.logging._logger, '_handlers_configured'):
        bt.logging.enable_info()
        bt.logging._logger._handlers_configured = True

    # Suppress noisy urllib3 warnings
    import logging

    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

    main()
