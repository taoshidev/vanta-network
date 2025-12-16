#!/usr/bin/env python3
"""
Elimination Timing Validation Script

Cross-checks elimination data against position/order data to identify miners that were 
eliminated before their final order time, indicating potential data consistency issues.

Usage:
    python validate_elimination_timing_final.py [--hotkeys HOTKEY1,HOTKEY2] [--output-csv results.csv]
    
Example:
    python validate_elimination_timing_final.py --hotkeys 5Da5hqCMSVgeGWmzeEnNrime3JKfgTpQmh7dXsdMP58dgeBd --output-csv validation_results.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import bittensor as bt
from sqlalchemy import create_engine, text

from vali_objects.position_management.position_utils.position_source import PositionSourceManager, PositionSource

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time_util.time_util import TimeUtil


@dataclass
class MinerValidationResult:
    """Result of validation for a single miner."""
    hotkey: str
    elimination_date: Optional[str]
    elimination_timestamp: Optional[int]
    final_order_date: Optional[str]
    final_order_timestamp: Optional[int]
    orders_after_elimination: int
    positions_after_elimination: int
    days_between: Optional[int]
    status: str  # 'VALID', 'INVALID', 'NO_ELIMINATION', 'NO_ORDERS'
    issue_description: Optional[str]
    elimination_reason: Optional[str]
    # Portfolio return validation fields
    portfolio_rows_after_elimination: int = 0
    final_portfolio_date: Optional[str] = None
    portfolio_days_after_elimination: Optional[int] = None
    has_portfolio_violation: bool = False


class EliminationTimingValidator:
    """Validates elimination timing against position/order data."""
    
    def __init__(self, database_url: str):
        """Initialize the validator with database connection."""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        # Cache for loaded data
        self.all_positions = {}
        self.elimination_data = {}
        self.portfolio_data = {}  # hotkey -> list of portfolio dates
        
    def load_positions(self, hotkeys: Optional[List[str]] = None) -> bool:
        """Load positions using the position source manager."""
        bt.logging.info("Loading position data...")
        try:
            position_source_manager = PositionSourceManager(source_type=PositionSource.DATABASE)
            self.all_positions = position_source_manager.load_positions(
                start_time_ms=None,
                end_time_ms=None,
                hotkeys=hotkeys
            )
            bt.logging.info(f"Loaded positions for {len(self.all_positions)} miners")
            return True
            
        except Exception as e:
            bt.logging.error(f"Failed to load position data: {e}")
            return False
    
    def load_eliminations(self, hotkeys: Optional[List[str]] = None) -> bool:
        """Load elimination data directly from database."""
        bt.logging.info("Loading elimination data from database...")
        try:
            with self.engine.connect() as conn:
                # Query elimination table with correct column names
                query = "SELECT miner_hotkey, elimination_ms, elimination_reason FROM eliminations"
                params = {}
                
                if hotkeys:
                    placeholders = ", ".join([f":hotkey_{i}" for i in range(len(hotkeys))])
                    query += f" WHERE miner_hotkey IN ({placeholders})"
                    for i, hotkey in enumerate(hotkeys):
                        params[f"hotkey_{i}"] = hotkey
                
                result = conn.execute(text(query), params)
                eliminations = result.fetchall()
                
                bt.logging.info(f"Loaded {len(eliminations)} elimination records")
                
                # Process elimination data
                for elim in eliminations:
                    miner_hotkey, elimination_ms, elimination_reason = elim
                    if miner_hotkey not in self.elimination_data:
                        self.elimination_data[miner_hotkey] = []
                    
                    self.elimination_data[miner_hotkey].append({
                        'elimination_ms': elimination_ms,
                        'elimination_reason': elimination_reason
                    })
                
                return True
                
        except Exception as e:
            bt.logging.error(f"Failed to load elimination data: {e}")
            return False
    
    def load_portfolio_data(self, hotkeys: Optional[List[str]] = None) -> bool:
        """Load portfolio return dates from miner_port_values table."""
        bt.logging.info("Loading portfolio return data from database...")
        try:
            with self.engine.connect() as conn:
                # Query miner_port_values table for dates
                query = "SELECT miner_hotkey, date FROM miner_port_values"
                params = {}
                
                if hotkeys:
                    placeholders = ", ".join([f":hotkey_{i}" for i in range(len(hotkeys))])
                    query += f" WHERE miner_hotkey IN ({placeholders})"
                    for i, hotkey in enumerate(hotkeys):
                        params[f"hotkey_{i}"] = hotkey
                
                result = conn.execute(text(query), params)
                portfolio_records = result.fetchall()
                
                bt.logging.info(f"Loaded {len(portfolio_records)} portfolio return records")
                
                # Process portfolio data
                for record in portfolio_records:
                    miner_hotkey, date_str = record
                    if miner_hotkey not in self.portfolio_data:
                        self.portfolio_data[miner_hotkey] = []
                    
                    self.portfolio_data[miner_hotkey].append(date_str)
                
                # Sort dates for each miner
                for hotkey in self.portfolio_data:
                    self.portfolio_data[hotkey].sort()
                
                return True
                
        except Exception as e:
            bt.logging.error(f"Failed to load portfolio data: {e}")
            return False
    
    def get_first_order_timestamp(self, positions: List[Any]) -> Optional[int]:
        """Get the timestamp of the first order across all positions for a miner."""
        first_timestamp = None
        
        for position in positions:
            if hasattr(position, 'orders') and position.orders:
                for order in position.orders:
                    if hasattr(order, 'processed_ms') and order.processed_ms:
                        if first_timestamp is None or order.processed_ms < first_timestamp:
                            first_timestamp = order.processed_ms
        
        return first_timestamp
    
    def get_final_order_timestamp(self, positions: List[Any]) -> Optional[int]:
        """Get the timestamp of the final order across all positions for a miner."""
        final_timestamp = None
        
        for position in positions:
            if hasattr(position, 'orders') and position.orders:
                for order in position.orders:
                    if hasattr(order, 'processed_ms') and order.processed_ms:
                        if final_timestamp is None or order.processed_ms > final_timestamp:
                            final_timestamp = order.processed_ms
        
        return final_timestamp
    
    def count_activity_after_elimination(self, positions: List[Any], elimination_timestamp: int) -> Tuple[int, int]:
        """Count orders and positions that occurred after elimination."""
        orders_after = 0
        positions_after = 0
        
        for position in positions:
            # Check if position was opened after elimination
            if hasattr(position, 'open_ms') and position.open_ms > elimination_timestamp:
                positions_after += 1
            
            # Count orders after elimination
            if hasattr(position, 'orders') and position.orders:
                for order in position.orders:
                    if hasattr(order, 'processed_ms') and order.processed_ms:
                        if order.processed_ms > elimination_timestamp:
                            orders_after += 1
        
        return orders_after, positions_after
    
    def check_portfolio_returns_after_elimination(self, hotkey: str, elimination_timestamp: int) -> Tuple[int, Optional[str], Optional[int]]:
        """Check for portfolio return rows after elimination day + 1.
        
        Args:
            hotkey: Miner hotkey to check
            elimination_timestamp: Elimination timestamp in milliseconds
            
        Returns:
            Tuple of (portfolio_rows_after_elimination, final_portfolio_date, portfolio_days_after_elimination)
        """
        portfolio_dates = self.portfolio_data.get(hotkey, [])
        if not portfolio_dates:
            return 0, None, None
        
        from datetime import datetime, timezone, timedelta
        
        # Convert elimination timestamp to date (elimination day + 1)
        elimination_date = datetime.fromtimestamp(elimination_timestamp / 1000, tz=timezone.utc)
        # Expected final portfolio date should be elimination day + 1 (start of next day)
        expected_final_date = elimination_date + timedelta(days=1)
        expected_final_date_str = expected_final_date.strftime('%Y-%m-%d')
        
        # Count portfolio rows after elimination day + 1
        rows_after_elimination = 0
        final_portfolio_date = None
        
        for date_str in portfolio_dates:
            if date_str > expected_final_date_str:
                rows_after_elimination += 1
            # Track the actual final portfolio date
            if final_portfolio_date is None or date_str > final_portfolio_date:
                final_portfolio_date = date_str
        
        # Calculate days between expected final date and actual final date
        portfolio_days_after_elimination = None
        if final_portfolio_date and final_portfolio_date > expected_final_date_str:
            final_date_obj = datetime.strptime(final_portfolio_date, '%Y-%m-%d')
            days_diff = (final_date_obj - expected_final_date).days
            portfolio_days_after_elimination = days_diff
        
        return rows_after_elimination, final_portfolio_date, portfolio_days_after_elimination
    
    def check_all_portfolio_rows_optimized(self, miners_with_orders: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Optimized function to check portfolio coverage for all miners with a single query.
        
        Args:
            miners_with_orders: Dict with hotkey -> {'first_order_ms': timestamp, 'last_order_ms': timestamp}
            
        Returns:
            Dict with hotkey -> coverage analysis
        """
        from datetime import datetime, timezone, timedelta
        
        if not miners_with_orders:
            return {}
        
        # Get all existing portfolio data with optimized queries
        existing_portfolio_data = {}
        try:
            with self.engine.connect() as conn:
                hotkey_list = list(miners_with_orders.keys())
                
                # Process in batches to avoid SQL parameter limits (typically 65535)
                batch_size = 1000
                for i in range(0, len(hotkey_list), batch_size):
                    batch = hotkey_list[i:i + batch_size]
                    
                    placeholders = ", ".join([f":hotkey_{j}" for j in range(len(batch))])
                    params = {f"hotkey_{j}": hotkey for j, hotkey in enumerate(batch)}
                    
                    # Query for this batch
                    result = conn.execute(text(
                        f"SELECT miner_hotkey, date FROM miner_port_values WHERE miner_hotkey IN ({placeholders})"
                    ), params)
                    
                    # Group by hotkey
                    for row in result.fetchall():
                        hotkey, date = row
                        if hotkey not in existing_portfolio_data:
                            existing_portfolio_data[hotkey] = set()
                        existing_portfolio_data[hotkey].add(date)
                    
                    if len(hotkey_list) > batch_size:
                        bt.logging.info(f"Processed portfolio data for {min(i + batch_size, len(hotkey_list))}/{len(hotkey_list)} miners...")
                    
        except Exception as e:
            bt.logging.error(f"Error checking portfolio rows: {e}")
            return {}
        
        # Now analyze each miner
        coverage_results = {}
        for hotkey, order_data in miners_with_orders.items():
            first_order_ms = order_data['first_order_ms']
            last_order_ms = order_data['last_order_ms']
            
            # Convert timestamps to dates
            first_date = datetime.fromtimestamp(first_order_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            last_date = datetime.fromtimestamp(last_order_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get all expected dates between first and last order
            expected_dates = set()
            current_date = first_date
            while current_date <= last_date:
                expected_dates.add(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
            
            # Get existing dates for this miner (filter by date range)
            existing_dates = existing_portfolio_data.get(hotkey, set())
            relevant_existing = existing_dates.intersection(expected_dates)
            
            # Find missing dates
            missing_dates = sorted(expected_dates - relevant_existing)
            
            coverage_results[hotkey] = {
                'expected_dates': len(expected_dates),
                'existing_dates': len(relevant_existing),
                'missing_dates': missing_dates,
                'missing_count': len(missing_dates),
                'coverage_percentage': (len(relevant_existing) / len(expected_dates) * 100) if expected_dates else 0
            }
        
        return coverage_results
    
    def validate_miner(self, hotkey: str) -> MinerValidationResult:
        """Validate elimination timing for a single miner."""
        positions = self.all_positions.get(hotkey, [])
        eliminations = self.elimination_data.get(hotkey, [])
        
        # Initialize result
        result = MinerValidationResult(
            hotkey=hotkey,
            elimination_date=None,
            elimination_timestamp=None,
            final_order_date=None,
            final_order_timestamp=None,
            orders_after_elimination=0,
            positions_after_elimination=0,
            days_between=None,
            status='UNKNOWN',
            issue_description=None,
            elimination_reason=None
        )
        
        # Check if miner has any positions/orders
        if not positions:
            result.status = 'NO_ORDERS'
            result.issue_description = 'No positions or orders found for this miner'
            return result
        
        # Get final order timestamp
        final_order_timestamp = self.get_final_order_timestamp(positions)
        if final_order_timestamp:
            result.final_order_timestamp = final_order_timestamp
            result.final_order_date = TimeUtil.millis_to_formatted_date_str(final_order_timestamp)
        
        # Check if miner has eliminations
        if not eliminations:
            result.status = 'NO_ELIMINATION'
            result.issue_description = 'No elimination record found for this miner'
            return result
        
        # Get earliest elimination timestamp (miners could have multiple eliminations)
        earliest_elimination = min(eliminations, key=lambda x: x['elimination_ms'])
        elimination_timestamp = earliest_elimination['elimination_ms']
        result.elimination_timestamp = elimination_timestamp
        result.elimination_date = TimeUtil.millis_to_formatted_date_str(elimination_timestamp)
        result.elimination_reason = earliest_elimination['elimination_reason']
        
        # Count activity after elimination
        orders_after, positions_after = self.count_activity_after_elimination(positions, elimination_timestamp)
        result.orders_after_elimination = orders_after
        result.positions_after_elimination = positions_after
        
        # Check portfolio returns after elimination day + 1
        portfolio_rows_after, final_portfolio_date, portfolio_days_after = self.check_portfolio_returns_after_elimination(
            hotkey, elimination_timestamp
        )
        result.portfolio_rows_after_elimination = portfolio_rows_after
        result.final_portfolio_date = final_portfolio_date
        result.portfolio_days_after_elimination = portfolio_days_after
        result.has_portfolio_violation = portfolio_rows_after > 0
        
        # Calculate days between elimination and final order
        if final_order_timestamp and elimination_timestamp:
            days_diff = (final_order_timestamp - elimination_timestamp) / (1000 * 60 * 60 * 24)
            result.days_between = int(days_diff)
        
        # Determine validation status (updated to include portfolio violations)
        has_order_violation = final_order_timestamp and final_order_timestamp > elimination_timestamp
        has_portfolio_violation = result.has_portfolio_violation
        
        if not final_order_timestamp:
            result.status = 'NO_ORDERS'
            result.issue_description = 'No orders found in position data'
        elif not has_order_violation and not has_portfolio_violation:
            result.status = 'VALID'
            result.issue_description = 'All elimination timing rules followed correctly'
        else:
            result.status = 'INVALID'
            violations = []
            
            if has_order_violation:
                violations.append(f'Final order occurred {result.days_between} days AFTER elimination')
                if orders_after > 0:
                    violations[-1] += f' ({orders_after} orders after elimination)'
            
            if has_portfolio_violation:
                violations.append(f'Portfolio returns continue {portfolio_days_after} days beyond elimination day + 1')
                violations[-1] += f' ({portfolio_rows_after} rows after expected cutoff)'
            
            result.issue_description = '; '.join(violations)
        
        return result
    
    def check_all_miners_portfolio_coverage(self, hotkeys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimized portfolio value coverage check for all miners using single query.
        
        Returns:
            Dictionary with comprehensive portfolio coverage analysis
        """
        all_miners = set(self.all_positions.keys())
        
        if hotkeys:
            all_miners = all_miners.intersection(set(hotkeys))
        
        bt.logging.info(f"Checking portfolio value coverage for {len(all_miners)} miners...")
        
        # First, collect all miners with their order timestamps
        miners_with_orders = {}
        bt.logging.info("Analyzing order data for all miners...")
        
        for hotkey in all_miners:
            positions = self.all_positions.get(hotkey, [])
            if not positions:
                continue
            
            # Get first and last order timestamps
            first_order_ms = self.get_first_order_timestamp(positions)
            last_order_ms = self.get_final_order_timestamp(positions)
            
            if first_order_ms and last_order_ms:
                miners_with_orders[hotkey] = {
                    'first_order_ms': first_order_ms,
                    'last_order_ms': last_order_ms
                }
        
        bt.logging.info(f"Found {len(miners_with_orders)} miners with orders. Fetching portfolio coverage...")
        
        # Use optimized single-query method
        coverage_results = self.check_all_portfolio_rows_optimized(miners_with_orders)
        
        # Process results
        miners_with_missing_data = []
        perfect_coverage_miners = []
        no_portfolio_data_miners = []
        
        for hotkey, coverage in coverage_results.items():
            order_data = miners_with_orders[hotkey]
            
            if coverage['missing_count'] > 0:
                miners_with_missing_data.append({
                    'hotkey': hotkey,
                    'first_order_date': TimeUtil.millis_to_formatted_date_str(order_data['first_order_ms'])[:10],
                    'last_order_date': TimeUtil.millis_to_formatted_date_str(order_data['last_order_ms'])[:10],
                    'expected_days': coverage['expected_dates'],
                    'existing_days': coverage['existing_dates'],
                    'missing_days': coverage['missing_count'],
                    'coverage_pct': coverage['coverage_percentage'],
                    'missing_dates': coverage['missing_dates'][:10] if len(coverage['missing_dates']) > 10 else coverage['missing_dates'],
                    'total_missing_dates': len(coverage['missing_dates'])
                })
            elif coverage['existing_dates'] == 0:
                no_portfolio_data_miners.append(hotkey)
            else:
                perfect_coverage_miners.append(hotkey)
        
        # Add miners that had no orders at all
        miners_no_orders = all_miners - set(miners_with_orders.keys())
        
        # Sort by number of missing days (worst first)
        miners_with_missing_data.sort(key=lambda x: x['missing_days'], reverse=True)
        
        bt.logging.info(f"Coverage analysis complete: {len(perfect_coverage_miners)} perfect, {len(miners_with_missing_data)} missing data, {len(no_portfolio_data_miners)} no data, {len(miners_no_orders)} no orders")
        
        return {
            'total_miners': len(all_miners),
            'perfect_coverage': len(perfect_coverage_miners),
            'missing_data': len(miners_with_missing_data),
            'no_data': len(no_portfolio_data_miners),
            'no_orders': len(miners_no_orders),
            'miners_with_missing_data': miners_with_missing_data
        }
    
    def validate_all_miners(self, hotkeys: Optional[List[str]] = None) -> List[MinerValidationResult]:
        """Validate elimination timing for all miners."""
        # Get all unique miners from both positions and eliminations
        all_miners = set(self.all_positions.keys()) | set(self.elimination_data.keys())
        
        # Filter to specific hotkeys if provided
        if hotkeys:
            all_miners = all_miners.intersection(set(hotkeys))
        
        bt.logging.info(f"Validating elimination timing for {len(all_miners)} miners...")
        
        results = []
        for i, hotkey in enumerate(sorted(all_miners), 1):
            if i % 100 == 0:
                bt.logging.info(f"Processed {i}/{len(all_miners)} miners...")
            
            result = self.validate_miner(hotkey)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[MinerValidationResult]) -> Dict[str, Any]:
        """Generate a summary report from validation results."""
        report = {
            'total_miners': len(results),
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': defaultdict(int),
            'invalid_miners': [],
            'statistics': {
                'days_between_stats': [],
                'orders_after_elimination_stats': [],
                'portfolio_rows_after_elimination_stats': []
            }
        }
        
        for result in results:
            report['summary'][result.status] += 1
            
            if result.status == 'INVALID':
                report['invalid_miners'].append({
                    'hotkey': result.hotkey,
                    'elimination_date': result.elimination_date,
                    'final_order_date': result.final_order_date,
                    'final_order_timestamp': result.final_order_timestamp,
                    'days_between': result.days_between,
                    'orders_after_elimination': result.orders_after_elimination,
                    'positions_after_elimination': result.positions_after_elimination,
                    'elimination_reason': result.elimination_reason,
                    'issue_description': result.issue_description,
                    'portfolio_rows_after_elimination': result.portfolio_rows_after_elimination,
                    'final_portfolio_date': result.final_portfolio_date,
                    'portfolio_days_after_elimination': result.portfolio_days_after_elimination,
                    'has_portfolio_violation': result.has_portfolio_violation
                })
            
            # Collect statistics
            if result.days_between is not None:
                report['statistics']['days_between_stats'].append(result.days_between)
            if result.orders_after_elimination is not None:
                report['statistics']['orders_after_elimination_stats'].append(result.orders_after_elimination)
            if result.portfolio_rows_after_elimination is not None:
                report['statistics']['portfolio_rows_after_elimination_stats'].append(result.portfolio_rows_after_elimination)
        
        # Calculate summary statistics
        days_stats = report['statistics']['days_between_stats']
        if days_stats:
            report['statistics']['days_between_summary'] = {
                'min': min(days_stats),
                'max': max(days_stats),
                'avg': sum(days_stats) / len(days_stats),
                'count': len(days_stats)
            }
        
        orders_stats = report['statistics']['orders_after_elimination_stats']
        if orders_stats:
            report['statistics']['orders_after_elimination_summary'] = {
                'min': min(orders_stats),
                'max': max(orders_stats),
                'avg': sum(orders_stats) / len(orders_stats),
                'total': sum(orders_stats),
                'count': len(orders_stats)
            }
        
        portfolio_stats = report['statistics']['portfolio_rows_after_elimination_stats']
        if portfolio_stats:
            report['statistics']['portfolio_rows_after_elimination_summary'] = {
                'min': min(portfolio_stats),
                'max': max(portfolio_stats),
                'avg': sum(portfolio_stats) / len(portfolio_stats),
                'total': sum(portfolio_stats),
                'count': len(portfolio_stats)
            }
        
        return report
    
    def print_portfolio_coverage_report(self, coverage_report: Dict[str, Any]):
        """Print portfolio value coverage report."""
        bt.logging.info("=" * 80)
        bt.logging.info("PORTFOLIO VALUE COVERAGE REPORT")
        bt.logging.info("=" * 80)
        bt.logging.info(f"Total Miners Analyzed: {coverage_report['total_miners']}")
        bt.logging.info("")
        
        bt.logging.info("COVERAGE SUMMARY:")
        bt.logging.info(f"  ✅ Perfect Coverage: {coverage_report['perfect_coverage']} miners")
        bt.logging.info(f"  ⚠️  Missing Data: {coverage_report['missing_data']} miners")
        bt.logging.info(f"  ❌ No Portfolio Data: {coverage_report['no_data']} miners")
        bt.logging.info("")
        
        if coverage_report['miners_with_missing_data']:
            bt.logging.info(f"🚨 MINERS WITH MISSING PORTFOLIO DATA ({coverage_report['missing_data']}):")
            bt.logging.info("")
            
            for i, miner in enumerate(coverage_report['miners_with_missing_data'][:20]):  # Show top 20
                bt.logging.info(f"  {i+1}. {miner['hotkey']}")
                bt.logging.info(f"     Period: {miner['first_order_date']} to {miner['last_order_date']}")
                bt.logging.info(f"     Expected Days: {miner['expected_days']}")
                bt.logging.info(f"     Existing Days: {miner['existing_days']}")
                bt.logging.info(f"     Missing Days: {miner['missing_days']} ({100 - miner['coverage_pct']:.1f}% missing)")
                
                if miner['missing_dates']:
                    dates_to_show = miner['missing_dates']
                    if miner['total_missing_dates'] > 10:
                        bt.logging.info(f"     Sample Missing Dates: {', '.join(dates_to_show[:5])}... ({miner['total_missing_dates']} total)")
                    else:
                        bt.logging.info(f"     Missing Dates: {', '.join(dates_to_show)}")
                bt.logging.info("")
            
            if len(coverage_report['miners_with_missing_data']) > 20:
                bt.logging.info(f"     ... and {len(coverage_report['miners_with_missing_data']) - 20} more miners with missing data")
        else:
            bt.logging.info("✅ All miners have complete portfolio value coverage!")
        
        bt.logging.info("=" * 80)
    
    def print_report(self, report: Dict[str, Any]):
        """Print a formatted report to console."""
        bt.logging.info("=" * 80)
        bt.logging.info("ELIMINATION TIMING VALIDATION REPORT")
        bt.logging.info("=" * 80)
        bt.logging.info(f"Validation Time: {report['validation_timestamp']}")
        bt.logging.info(f"Total Miners Analyzed: {report['total_miners']}")
        bt.logging.info("")
        
        bt.logging.info("STATUS SUMMARY:")
        for status, count in report['summary'].items():
            percentage = (count / report['total_miners']) * 100 if report['total_miners'] > 0 else 0
            bt.logging.info(f"  {status}: {count} ({percentage:.1f}%)")
        bt.logging.info("")
        
        # Show invalid miners (the main concern)
        invalid_count = len(report['invalid_miners'])
        if invalid_count > 0:
            bt.logging.info(f"🚨 INVALID MINERS ({invalid_count}):")
            bt.logging.info("These miners have violations after their elimination date:")
            bt.logging.info("")
            
            order_violations = []
            portfolio_violations = []
            both_violations = []
            
            # Categorize violations
            for miner in report['invalid_miners']:
                has_order_violation = miner.get('orders_after_elimination', 0) > 0
                has_portfolio_violation = miner.get('portfolio_rows_after_elimination', 0) > 0
                
                if has_order_violation and has_portfolio_violation:
                    both_violations.append(miner)
                elif has_order_violation:
                    order_violations.append(miner)
                elif has_portfolio_violation:
                    portfolio_violations.append(miner)
            
            # Display categorized violations
            if both_violations:
                bt.logging.info(f"📋 MINERS WITH BOTH ORDER & PORTFOLIO VIOLATIONS ({len(both_violations)}):")
                for i, miner in enumerate(both_violations):
                    bt.logging.info(f"  {i+1}. {miner['hotkey']}")
                    bt.logging.info(f"     Elimination: {miner['elimination_date']} ({miner['elimination_reason']})")
                    bt.logging.info(f"     Final Order: {miner['final_order_date']} ({miner['days_between']} days after)")
                    bt.logging.info(f"     Orders After: {miner['orders_after_elimination']}")
                    bt.logging.info(f"     Portfolio Final: {miner.get('final_portfolio_date', 'N/A')} ({miner.get('portfolio_days_after_elimination', 0)} days after)")
                    bt.logging.info(f"     Portfolio Rows After: {miner.get('portfolio_rows_after_elimination', 0)}")
                    bt.logging.info("")
            
            if order_violations:
                bt.logging.info(f"📊 MINERS WITH ORDER VIOLATIONS ONLY ({len(order_violations)}):")
                for i, miner in enumerate(order_violations):
                    bt.logging.info(f"  {i+1}. {miner['hotkey']}")
                    bt.logging.info(f"     Elimination: {miner['elimination_date']} ({miner['elimination_reason']})")
                    bt.logging.info(f"     Final Order: {miner['final_order_date']} ({miner['days_between']} days after)")
                    bt.logging.info(f"     Orders After: {miner['orders_after_elimination']}")
                    bt.logging.info("")
            
            if portfolio_violations:
                bt.logging.info(f"💼 MINERS WITH PORTFOLIO VIOLATIONS ONLY ({len(portfolio_violations)}):")
                for i, miner in enumerate(portfolio_violations):
                    bt.logging.info(f"  {i+1}. {miner['hotkey']}")
                    bt.logging.info(f"     Elimination: {miner['elimination_date']} ({miner['elimination_reason']})")
                    bt.logging.info(f"     Expected Final Portfolio: elimination day + 1")
                    bt.logging.info(f"     Actual Final Portfolio: {miner.get('final_portfolio_date', 'N/A')} ({miner.get('portfolio_days_after_elimination', 0)} days after)")
                    bt.logging.info(f"     Extra Portfolio Rows: {miner.get('portfolio_rows_after_elimination', 0)}")
                    bt.logging.info("")
                
            bt.logging.info("🔍 RECOMMENDATION:")
            bt.logging.info("   These violations suggest data pipeline issues:")
            if order_violations or both_violations:
                bt.logging.info("   • Review position/order elimination timing logic")
            if portfolio_violations or both_violations:
                bt.logging.info("   • Review portfolio return calculation cutoff logic")
            bt.logging.info("   • Investigate data integrity and elimination timing implementation")
            bt.logging.info("")
            
            # Generate SQL fix queries
            bt.logging.info("💾 SQL FIX QUERIES:")
            bt.logging.info("   Run these queries to fix elimination timing issues:")
            bt.logging.info("")
            
            for i, miner in enumerate(report['invalid_miners']):
                # Calculate new elimination time = final order time + 1ms
                new_elimination_ms = miner['final_order_timestamp'] + 1 if miner['final_order_timestamp'] else None
                
                if new_elimination_ms:
                    bt.logging.info(f"-- Fix #{i+1}: {miner['hotkey']}")
                    bt.logging.info(f"UPDATE eliminations SET elimination_ms = {new_elimination_ms}")
                    bt.logging.info(f"WHERE miner_hotkey = '{miner['hotkey']}';")
                    bt.logging.info("")
            
            bt.logging.info("⚠️  WARNING: Review these queries carefully before execution!")
            bt.logging.info("   Test in a development environment first.")
        else:
            bt.logging.info("✅ No invalid miners found!")
            bt.logging.info("   All miners show consistent elimination timing.")
        
        bt.logging.info("=" * 80)
    
    def save_csv_report(self, results: List[MinerValidationResult], filename: str):
        """Save detailed results to CSV file."""
        bt.logging.info(f"Saving detailed results to {filename}...")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'hotkey', 'status', 'elimination_date', 'final_order_date',
                'days_between', 'orders_after_elimination', 'positions_after_elimination',
                'portfolio_rows_after_elimination', 'final_portfolio_date', 'portfolio_days_after_elimination',
                'has_portfolio_violation', 'elimination_reason', 'issue_description'
            ])
            
            # Write data
            for result in results:
                writer.writerow([
                    result.hotkey, result.status, result.elimination_date,
                    result.final_order_date, result.days_between,
                    result.orders_after_elimination, result.positions_after_elimination,
                    result.portfolio_rows_after_elimination, result.final_portfolio_date,
                    result.portfolio_days_after_elimination, result.has_portfolio_violation,
                    result.elimination_reason, result.issue_description
                ])
        
        bt.logging.info(f"✅ CSV report saved to {filename}")
    
    def save_sql_fix_queries(self, report: Dict[str, Any], filename: str):
        """Save SQL fix queries to a file."""
        invalid_miners = report.get('invalid_miners', [])
        
        if not invalid_miners:
            bt.logging.info("No invalid miners found, no SQL fixes needed.")
            return
            
        bt.logging.info(f"Saving SQL fix queries to {filename}...")
        
        with open(filename, 'w') as sqlfile:
            sqlfile.write("-- SQL Queries to Fix Elimination Timing Issues\n")
            sqlfile.write(f"-- Generated on: {report['validation_timestamp']}\n")
            sqlfile.write(f"-- Total invalid miners: {len(invalid_miners)}\n")
            sqlfile.write("-- \n")
            sqlfile.write("-- WARNING: Review these queries carefully before execution!\n")
            sqlfile.write("-- Test in a development environment first.\n")
            sqlfile.write("-- \n\n")
            
            for i, miner in enumerate(invalid_miners):
                new_elimination_ms = miner['final_order_timestamp'] + 1 if miner['final_order_timestamp'] else None
                
                if new_elimination_ms:
                    sqlfile.write(f"-- Fix #{i+1}: {miner['hotkey']}\n")
                    sqlfile.write(f"-- Old elimination: {miner['elimination_date']}\n")
                    sqlfile.write(f"-- Final order: {miner['final_order_date']}\n")
                    sqlfile.write(f"-- Gap: {miner['days_between']} days\n")
                    sqlfile.write(f"-- Orders after elimination: {miner['orders_after_elimination']}\n")
                    sqlfile.write(f"-- Positions after elimination: {miner['positions_after_elimination']}\n")
                    sqlfile.write(f"UPDATE eliminations SET elimination_ms = {new_elimination_ms}\n")
                    sqlfile.write(f"WHERE miner_hotkey = '{miner['hotkey']}';\n")
                    sqlfile.write("\n")
        
        bt.logging.info(f"✅ SQL fix queries saved to {filename}")
    


def get_database_url_from_config() -> Optional[str]:
    """Read database URL from config file."""
    config_file = "../config-development.json"
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get('secrets', {}).get('db_ptn_editor_url')
    except Exception as e:
        bt.logging.error(f"Error reading {config_file}: {e}")
    
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate elimination timing against position data")
    parser.add_argument(
        "--hotkeys",
        type=str,
        help="Comma-separated list of hotkeys to analyze (defaults to all)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database connection string",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Save detailed results to CSV file",
    )
    parser.add_argument(
        "--output-sql",
        type=str,
        help="Save SQL fix queries to file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--check-portfolio-coverage",
        action="store_true",
        help="Check for missing portfolio value rows instead of elimination timing",
    )
    return parser.parse_args()


def main():
    """Main function to run elimination timing validation."""
    args = parse_args()
    
    # Configure logging
    bt.logging.set_trace(args.log_level == "DEBUG")
    bt.logging.set_debug(args.log_level == "DEBUG")
    
    # Get database URL
    database_url = args.database_url
    if not database_url:
        database_url = get_database_url_from_config()
    
    if not database_url:
        bt.logging.error("No database URL provided. Use --database-url or config file.")
        return
    
    # Parse hotkeys
    hotkeys = None
    if args.hotkeys:
        hotkeys = [hk.strip() for hk in args.hotkeys.split(',')]
        bt.logging.info(f"Filtering for {len(hotkeys)} specific hotkeys")
    
    # Initialize validator
    validator = EliminationTimingValidator(database_url)
    
    # Load position data (always needed)
    if not validator.load_positions(hotkeys):
        bt.logging.error("Failed to load position data. Exiting.")
        return
    
    if args.check_portfolio_coverage or args.auto_backfill:
        # Check portfolio value coverage
        bt.logging.info("Running portfolio value coverage check...")
        coverage_report = validator.check_all_miners_portfolio_coverage(hotkeys)
        
        # Show the coverage report
        validator.print_portfolio_coverage_report(coverage_report)
        
        # Show summary with manual command suggestions
        if coverage_report['missing_data'] > 0:
            bt.logging.info("")
            bt.logging.info(f"⚠️  Found {coverage_report['missing_data']} miners with missing portfolio value rows!")
            bt.logging.info("   Use daily_portfolio_returns.py --auto-backfill to automatically fix all missing data.")
            
            # Generate backfill commands for miners with missing data
            if coverage_report['miners_with_missing_data']:
                bt.logging.info("")
                bt.logging.info("📝 MANUAL COMMANDS (if preferred):")
                for miner in coverage_report['miners_with_missing_data'][:5]:
                    bt.logging.info(f"   python daily_portfolio_returns.py --hotkeys {miner['hotkey']}")
                if len(coverage_report['miners_with_missing_data']) > 5:
                    bt.logging.info(f"   ... and {len(coverage_report['miners_with_missing_data']) - 5} more miners")
                bt.logging.info("")
                bt.logging.info("💡 TIP: Use daily_portfolio_returns.py --auto-backfill for automated processing!")
        else:
            bt.logging.info("")
            bt.logging.info("✅ All miners have complete portfolio value coverage!")
    else:
        # Original elimination timing validation
        if not validator.load_eliminations(hotkeys):
            bt.logging.error("Failed to load elimination data. Exiting.")
            return
        
        # Load portfolio data for enhanced validation
        if not validator.load_portfolio_data(hotkeys):
            bt.logging.warning("Failed to load portfolio data. Portfolio validation will be skipped.")
        
        # Validate all miners
        results = validator.validate_all_miners(hotkeys)
        
        # Generate and print report
        report = validator.generate_report(results)
        validator.print_report(report)
        
        # Save CSV if requested
        if args.output_csv:
            validator.save_csv_report(results, args.output_csv)
        
        # Save SQL fix queries if requested
        if args.output_sql:
            validator.save_sql_fix_queries(report, args.output_sql)
        
        # Show summary of critical issues
        invalid_count = report['summary']['INVALID']
        if invalid_count > 0:
            bt.logging.info("")
            bt.logging.info(f"⚠️  CRITICAL: Found {invalid_count} miners with orders after elimination!")
            bt.logging.info("   This indicates serious data consistency issues.")
            bt.logging.info("   Investigate elimination timing and position data integrity.")
        else:
            bt.logging.info("")
            bt.logging.info("✅ VALIDATION PASSED: All miners show consistent elimination timing.")


if __name__ == "__main__":
    bt.logging.enable_info()
    main()