#!/usr/bin/env python3
"""
Fix src=1 Positions Script

This script finds all positions that have orders with src=1, fetches proper prices for those orders,
recalculates position returns, and generates SQL statements to update the database.

Based on daily_portfolio_returns.py logic, particularly the "do_special_fetch" pattern.

The script generates SQL statements only - no direct database modifications are made.
SQL statements can be reviewed and executed manually.

Usage:
    python fix_src1_positions.py [--hotkeys HOTKEY1,HOTKEY2,...] [--limit N] [--output-file file.sql]
"""

import argparse
import sys
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import bittensor as bt
from vali_objects.position import Position
from vali_objects.utils.position_source import PositionSourceManager, PositionSource
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from time_util.time_util import TimeUtil


class PositionFixer:
    """Handles finding and fixing positions with src=1 orders."""
    
    def __init__(self):
        """
        Initialize the position fixer.
        """
        
        # Initialize price fetcher
        bt.logging.info("🔧 Initializing price fetcher...")
        secrets = ValiUtils.get_secrets()
        self.live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
        
        # Initialize position source manager
        self.position_source_manager = PositionSourceManager(source_type=PositionSource.DATABASE)
        
        # Statistics
        self.positions_found = 0
        self.positions_fixed = 0
        self.positions_failed = 0
        self.update_data_list = []  # Store update data for bulk SQL generation
        
    def find_src1_positions(self, hotkeys: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict[str, List[Position]]:
        """
        Find all positions that have orders with src=1.
        
        Args:
            hotkeys: Optional list of hotkeys to filter by
            limit: Optional limit on number of positions to process
            
        Returns:
            Dictionary mapping hotkey -> list of positions with src=1 orders
        """
        bt.logging.info("🔍 Loading all positions...")
        
        # Load all positions
        all_positions = self.position_source_manager.load_positions(
            start_time_ms=None,
            end_time_ms=None,
            hotkeys=hotkeys
        )
        
        bt.logging.info(f"✅ Loaded positions for {len(all_positions)} miners")
        
        # Find positions with src=1 orders
        src1_positions = defaultdict(list)
        total_positions_checked = 0
        
        for hotkey, positions in all_positions.items():
            for position in positions:
                if position.trade_pair.is_equities or position.trade_pair.is_indices:
                    continue  # Skip equities and indices
                total_positions_checked += 1
                
                # Check if any order has src=1
                has_src1_order = any(order.src == 1 for order in position.orders)
                
                if has_src1_order:
                    src1_positions[hotkey].append(position)
                    self.positions_found += 1
                    
                    bt.logging.debug(f"Found position {position.position_uuid} for {hotkey[:12]}... with src=1 order")
                    
                # Apply limit if specified
                if limit and self.positions_found >= limit:
                    bt.logging.info(f"🚫 Reached limit of {limit} positions, stopping search")
                    break
                    
            if limit and self.positions_found >= limit:
                break
        
        bt.logging.info(f"📊 Found {self.positions_found} positions with src=1 orders out of {total_positions_checked} total positions")
        return dict(src1_positions)
    
    def fix_position(self, position: Position) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Fix a single position by fetching proper prices and recalculating returns.
        
        Args:
            position: Position object to fix
            
        Returns:
            Tuple of (success, update_data, error_message)
        """
        # Find orders with src=1
        src1_orders = [order for order in position.orders if order.src == 1]

        if not src1_orders:
            return False, None, "No src=1 orders found"

        assert len(src1_orders) == 1, "Expected exactly one src=1 order per position"
        assert position.orders[-1].src == 1, "Last order must be src=1"

        #bt.logging.info(f"🔧 Fixing position {position.position_uuid} with {len(src1_orders)} src=1 orders")

        # Create a copy of the position to work with
        position_copy = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms,
            trade_pair=position.trade_pair,
            orders=position.orders[:],  # Copy orders
            position_type=position.position_type,
            is_closed_position=position.is_closed_position,
        )

        # Fix each src=1 order
        fixed_orders = []

        for i, order in enumerate(position_copy.orders):
            if order.src == 1:
                # Fetch proper price for this order
                bt.logging.debug(f"  📈 Fetching price for order at {TimeUtil.millis_to_formatted_date_str(order.processed_ms)}")

                price_source = self.live_price_fetcher.get_close_at_date(
                    trade_pair=position.trade_pair,
                    timestamp_ms=order.processed_ms,
                    verbose=False
                )

                if not price_source:
                   raise Exception(f"Failed to fetch price for order {order}")

                # Parse the appropriate price
                new_price = price_source.parse_appropriate_price(
                    now_ms=order.processed_ms,
                    is_forex=position.trade_pair.is_forex,
                    order_type=order.order_type,
                    position=position_copy
                )


                # Update order with new price and src=3
                order.price = new_price
                order.src = 3  # Change from 1 to 3

                bt.logging.debug(f"  ✅ Updated order price to {new_price}, src changed to 3")

                fixed_orders.append({
                    'order_uuid': order.order_uuid,
                    'old_price': position.orders[-1].price,
                    'new_price': new_price,
                    'old_src': 1,
                    'new_src': 3,
                    'processed_ms': order.processed_ms  # Add timestamp for display
                })

        # Rebuild position with updated orders
        position_copy.rebuild_position_with_updated_orders(live_price_fetcher=self.live_price_fetcher)

        # Calculate new returns
        if position_copy.is_closed_position:
            # For closed positions, the return_at_close should now be properly calculated
            old_return_at_close = position.return_at_close
            old_curr_return = position.current_return  # Position object uses 'current_return'
            new_return_at_close = position_copy.return_at_close
            new_curr_return = position_copy.current_return  # Position object uses 'current_return'
        else:
            # For open positions, we need to calculate current return at some reference time
            # Use the last order time as reference
            last_order_time = max(order.processed_ms for order in position_copy.orders)

            # Set returns based on current state
            position_copy.set_returns(realtime_price=new_price, time_ms=last_order_time, live_price_fetcher=self.live_price_fetcher)

            old_return_at_close = position.return_at_close
            old_curr_return = position.current_return  # Position object uses 'current_return'
            new_return_at_close = position_copy.return_at_close
            new_curr_return = position_copy.current_return  # Position object uses 'current_return'

        update_data = {
            'position_uuid': position.position_uuid,
            'old_return_at_close': old_return_at_close,
            'new_return_at_close': new_return_at_close,
            'old_curr_return': old_curr_return,
            'new_curr_return': new_curr_return,
            'fixed_orders': fixed_orders
        }

        bt.logging.debug(f"  📊 Return at close: {old_return_at_close} -> {new_return_at_close}")
        bt.logging.debug(f"  📊 Current return: {old_curr_return} -> {new_curr_return}")

        return True, update_data, None

    
    def generate_sql_statements(self, update_data: Dict) -> List[str]:
        """
        Generate SQL statements to update position and order tables.
        
        Args:
            update_data: Data from fix_position containing update information
            
        Returns:
            List of SQL update statements
        """
        statements = []
        
        # Update position table
        position_sql = f"""
UPDATE positions 
SET return_at_close = {update_data['new_return_at_close']}, 
    curr_return = {update_data['new_curr_return']} 
WHERE position_uuid = '{update_data['position_uuid']}';
        """.strip()
        
        statements.append(position_sql)
        
        # Update order table for each fixed order
        for order_data in update_data['fixed_orders']:
            order_sql = f"""
UPDATE orders 
SET price = {order_data['new_price']}, 
    src = {order_data['new_src']} 
WHERE order_uuid = '{order_data['order_uuid']}';
            """.strip()
            
            statements.append(order_sql)
        
        return statements
    
    def _print_detailed_comparisons(self, comparisons: List[Dict]) -> None:
        """
        Print detailed comparison of old vs new values.
        
        Args:
            comparisons: List of comparison data for each position
        """
        print("\n" + "="*100)
        print("DETAILED COMPARISON: OLD vs NEW VALUES (Sorted by % change in return_at_close)")
        print("="*100)
        
        # Calculate percent changes and sort comparisons
        for comp in comparisons:
            update_data = comp['update_data']
            old_val = update_data['old_return_at_close']
            new_val = update_data['new_return_at_close']
            comp['percent_change'] = ((new_val / old_val) - 1) * 100 if old_val != 0 else 0
        
        # Sort by percent change (largest change first)
        sorted_comparisons = sorted(comparisons, key=lambda x: abs(x['percent_change']), reverse=True)
        
        for i, comp in enumerate(sorted_comparisons, 1):
            print(f"\n📊 Position {i}/{len(sorted_comparisons)}")
            print(f"   Hotkey: {comp['hotkey'][:16]}...")
            print(f"   Position UUID: {comp['position_uuid']}")
            print(f"   Trade Pair: {comp['trade_pair']}")
            print(f"   " + "-"*80)
            
            update_data = comp['update_data']
            
            # Show return comparisons
            print(f"   RETURNS:")
            print(f"     return_at_close: {update_data['old_return_at_close']:.8f} → {update_data['new_return_at_close']:.8f}")
            
            diff_at_close = update_data['new_return_at_close'] - update_data['old_return_at_close']
            percent_change_at_close = comp['percent_change']
            print(f"       Difference: {diff_at_close:+.8f} ({percent_change_at_close:+.4f}%)")
            
            # Just show the current_return values without delta
            print(f"     current_return: {update_data['old_curr_return']:.8f} → {update_data['new_curr_return']:.8f}")
            
            # Show order updates
            if update_data['fixed_orders']:
                print(f"   ORDERS FIXED: {len(update_data['fixed_orders'])}")
                for j, order in enumerate(update_data['fixed_orders'], 1):
                    # Convert timestamp to UTC string
                    order_time_utc = TimeUtil.millis_to_formatted_date_str(order['processed_ms'])
                    print(f"     Order {j}:")
                    print(f"       UUID: {order['order_uuid']}")
                    print(f"       Time (UTC): {order_time_utc}")
                    print(f"       Price: {order['old_price']} → {order['new_price']}")
                    print(f"       Src: {order['old_src']} → {order['new_src']}")
        
        print("\n" + "="*100)
        print(f"SUMMARY: {len(comparisons)} positions compared")
        print("="*100 + "\n")
    
    def process_all_positions(self, src1_positions: Dict[str, List[Position]], show_details: bool = False) -> None:
        """
        Process all positions with src=1 orders.
        
        Args:
            src1_positions: Dictionary mapping hotkey -> list of positions to fix
            show_details: If True, show detailed comparison of old vs new values
        """
        bt.logging.info(f"🔧 Processing {self.positions_found} positions with src=1 orders...")
        
        all_sql_statements = []
        all_comparisons = []  # Store comparisons for detailed output
        
        for hotkey, positions in src1_positions.items():
            bt.logging.info(f"📋 Processing {len(positions)} positions for {hotkey[:12]}...")
            
            for position in positions:
                success, update_data, error = self.fix_position(position)
                
                if success:
                    self.positions_fixed += 1
                    
                    # Store update data for bulk SQL generation
                    self.update_data_list.append(update_data)
                    
                    bt.logging.info(f"  ✅ Fixed position {position.position_uuid}")
                    
                    # Store comparison data
                    if show_details:
                        all_comparisons.append({
                            'hotkey': hotkey,
                            'position_uuid': position.position_uuid,
                            'trade_pair': position.trade_pair.trade_pair,
                            'update_data': update_data
                        })
                    
                else:
                    self.positions_failed += 1
                    bt.logging.warning(f"  ❌ Failed to fix position {position.position_uuid}: {error}")
        
        # Show detailed comparisons if requested
        if show_details and all_comparisons:
            self._print_detailed_comparisons(all_comparisons)
        
        bt.logging.info(f"📊 Processing complete: {self.positions_fixed} fixed, {self.positions_failed} failed")
    
    def _generate_bulk_sql_commands(self) -> str:
        """
        Generate bulk SQL UPDATE commands for positions and orders tables.
        
        Returns:
            String containing bulk SQL commands
        """
        if not self.update_data_list:
            return "-- No updates to generate\n"
        
        # Separate position and order updates
        position_updates = []
        order_updates = []
        
        for update_data in self.update_data_list:
            # Add position update
            position_updates.append({
                'uuid': update_data['position_uuid'],
                'return_at_close': update_data['new_return_at_close'],
                'curr_return': update_data['new_curr_return']
            })
            
            # Add order updates
            for order in update_data['fixed_orders']:
                order_updates.append({
                    'uuid': order['order_uuid'],
                    'price': order['new_price'],
                    'src': order['new_src']
                })
        
        # Generate bulk SQL
        sql_parts = []
        
        # Positions table bulk update using CASE statements
        if position_updates:
            sql_parts.append("-- BULK UPDATE: Positions Table")
            sql_parts.append("-- Updates return_at_close and curr_return for fixed positions")
            sql_parts.append("UPDATE positions")
            sql_parts.append("SET ")
            sql_parts.append("    return_at_close = CASE position_uuid")
            
            for pos in position_updates:
                sql_parts.append(f"        WHEN '{pos['uuid']}' THEN {pos['return_at_close']}")
            sql_parts.append("        ELSE return_at_close")
            sql_parts.append("    END,")
            sql_parts.append("    curr_return = CASE position_uuid")
            
            for pos in position_updates:
                sql_parts.append(f"        WHEN '{pos['uuid']}' THEN {pos['curr_return']}")
            sql_parts.append("        ELSE curr_return")
            sql_parts.append("    END")
            
            # Add WHERE clause with all position UUIDs
            uuids = "', '".join(pos['uuid'] for pos in position_updates)
            sql_parts.append(f"WHERE position_uuid IN ('{uuids}');")
            sql_parts.append("")
        
        # Orders table bulk update using CASE statements
        if order_updates:
            sql_parts.append("-- BULK UPDATE: Orders Table")
            sql_parts.append("-- Updates price and src for fixed orders")
            sql_parts.append("UPDATE orders")
            sql_parts.append("SET ")
            sql_parts.append("    price = CASE order_uuid")
            
            for order in order_updates:
                sql_parts.append(f"        WHEN '{order['uuid']}' THEN {order['price']}")
            sql_parts.append("        ELSE price")
            sql_parts.append("    END,")
            sql_parts.append("    src = CASE order_uuid")
            
            for order in order_updates:
                sql_parts.append(f"        WHEN '{order['uuid']}' THEN {order['src']}")
            sql_parts.append("        ELSE src")
            sql_parts.append("    END")
            
            # Add WHERE clause with all order UUIDs
            uuids = "', '".join(order['uuid'] for order in order_updates)
            sql_parts.append(f"WHERE order_uuid IN ('{uuids}');")
            sql_parts.append("")
        
        # Add summary
        sql_parts.append(f"-- SUMMARY:")
        sql_parts.append(f"-- Positions updated: {len(position_updates)}")
        sql_parts.append(f"-- Orders updated: {len(order_updates)}")
        
        return "\n".join(sql_parts)
    
    def output_sql_statements(self, output_file: Optional[str] = None) -> None:
        """
        Output bulk SQL statements to file or console.
        
        Args:
            output_file: Optional file path to write SQL statements
        """
        if not self.positions_fixed:
            bt.logging.warning("No positions were fixed - no SQL to output")
            return
        
        # Generate bulk SQL commands
        bulk_sql = self._generate_bulk_sql_commands()
        
        bt.logging.info(f"📝 Generated bulk SQL commands for {self.positions_fixed} positions")
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write("-- Bulk SQL commands to fix src=1 positions\n")
                    f.write(f"-- Generated on {TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())}\n")
                    f.write(f"-- Total positions fixed: {self.positions_fixed}\n")
                    f.write(f"-- Total orders updated: {len([u for u in self.update_data_list for _ in u['fixed_orders']])}\n\n")
                    
                    f.write(bulk_sql)
                
                bt.logging.info(f"✅ Bulk SQL commands written to {output_file}")
            except Exception as e:
                bt.logging.error(f"Failed to write SQL commands to file: {e}")
        else:
            # Output to console
            print("\n" + "="*100)
            print("BULK SQL COMMANDS TO FIX SRC=1 POSITIONS")
            print("="*100)
            print(bulk_sql)
            print("="*100)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate SQL statements to fix positions with src=1 orders")
    
    parser.add_argument(
        "--hotkeys",
        type=str,
        help="Comma-separated list of hotkeys to process (optional)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of positions to process (optional)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="File to write SQL statements to (optional, defaults to console)",
        default='fix_src1_positions.txt'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    bt.logging.enable_info()
    
    # Parse hotkeys if provided
    hotkeys = None
    if args.hotkeys:
        hotkeys = [hk.strip() for hk in args.hotkeys.split(',') if hk.strip()]
        bt.logging.info(f"🎯 Filtering for {len(hotkeys)} specific hotkeys")
    
    # Initialize position fixer
    fixer = PositionFixer()

    # Find positions with src=1 orders
    src1_positions = fixer.find_src1_positions(hotkeys=hotkeys, limit=args.limit)

    if not src1_positions:
        bt.logging.info("🎉 No positions with src=1 orders found!")
        return

    # Process all positions (show details when limit is used)
    show_details = args.limit is not None
    fixer.process_all_positions(src1_positions, show_details=show_details)

    # Output SQL statements
    fixer.output_sql_statements(args.output_file)

    # Summary
    bt.logging.info(f"🎯 SUMMARY:")
    bt.logging.info(f"  📊 Positions found: {fixer.positions_found}")
    bt.logging.info(f"  ✅ Positions fixed: {fixer.positions_fixed}")
    bt.logging.info(f"  ❌ Positions failed: {fixer.positions_failed}")
    bt.logging.info(f"  📝 SQL statements generated: {len(fixer.sql_statements)}")




if __name__ == "__main__":
    main()