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
import time
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

import bittensor as bt
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, text, inspect, tuple_
from sqlalchemy.orm import declarative_base, sessionmaker

from time_util.time_util import TimeUtil
from time_util.time_util import MS_IN_24_HOURS
from vali_objects.position import Position
from vali_objects.utils.position_source import PositionSourceManager, PositionSource
from vali_objects.utils.elimination_source import EliminationSourceManager, EliminationSource
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.vali_config import TradePair, TradePairCategory, CryptoSubcategory, ForexSubcategory
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.position_filter import PositionFilter, FilterStats
from collections import defaultdict
from datetime import datetime, timezone

# Database setup
Base = declarative_base()


class SharedDataManager:
    """Manages shared data loaded once at initialization to minimize database calls."""
    
    def __init__(self, database_url: str, hotkeys: Optional[List[str]] = None):
        """Initialize shared data manager.
        
        Args:
            database_url: Database connection string
            hotkeys: Optional list of specific hotkeys to filter data loading
        """
        self.database_url = database_url
        self.filter_hotkeys = hotkeys
        
        # Shared data loaded at initialization
        self.all_positions = {}  # hotkey -> list of positions
        self.db_manager = None   # DatabaseManager instance with portfolio data cache
        self.elimination_tracker = None  # EliminationTracker with preloaded data
        self.live_price_fetcher = None  # LivePriceFetcher for cached price requests across miners
        
        # Price source cache keyed by date string for maximum reuse
        # Format: {"YYYY-MM-DD": {trade_pair_id: PriceSource}}
        self.price_source_cache: Dict[str, Dict[str, Any]] = {}
        self.price_cache_hits = 0
        self.price_cache_misses = 0
        
        # Per-miner price source tracking for detailed logging
        self.current_miner_price_stats = {
            'cache_hits': 0,
            'live_fetches': 0,
            'total_days': 0
        }
        
        # Load flags
        self.positions_loaded = False
        self.portfolio_data_loaded = False
        self.eliminations_loaded = False
    
    def initialize_all_data(self, elimination_source: EliminationSource = EliminationSource.DATABASE):
        """Load all required data at once to minimize database calls."""
        bt.logging.info("🚀 Initializing shared data manager - loading all data at once...")
        
        # 1. Load positions
        self._load_all_positions()
        
        # 2. Initialize database manager and load portfolio data
        self._load_database_manager_with_portfolio_data()
        
        # 3. Load elimination data
        self._load_elimination_data(elimination_source)
        
        # 4. Initialize shared price fetcher for efficiency across miners
        self._initialize_price_fetcher()
        
        # Summary
        total_miners = len(self.all_positions)
        total_positions = sum(len(positions) for positions in self.all_positions.values())
        
        bt.logging.info("✅ Shared data manager initialization complete:")
        bt.logging.info(f"   📊 Positions: {total_miners} miners, {total_positions} total positions")
        if self.db_manager and self.db_manager.portfolio_data_loaded:
            total_portfolio_records = sum(len(dates) for dates in self.db_manager.all_portfolio_data.values())
            bt.logging.info(f"   📊 Portfolio data: {len(self.db_manager.all_portfolio_data)} miners, {total_portfolio_records} records")
        if self.elimination_tracker and self.elimination_tracker.loaded:
            bt.logging.info(f"   📊 Eliminations: {len(self.elimination_tracker.elimination_timestamps)} eliminated miners")
        bt.logging.info(f"   📊 Price cache: Ready (0 cached prices)")
        
    def _load_all_positions(self):
        """Load all positions using PositionSourceManager."""
        bt.logging.info("📊 Loading all positions...")
        try:
            position_source_manager = PositionSourceManager(source_type=PositionSource.DATABASE)
            self.all_positions = position_source_manager.load_positions(
                start_time_ms=None,
                end_time_ms=None, 
                hotkeys=self.filter_hotkeys
            )
            self.positions_loaded = True
            bt.logging.info(f"✅ Loaded positions for {len(self.all_positions)} miners")
        except Exception as e:
            bt.logging.error(f"Failed to load positions: {e}")
            raise
    
    def _load_database_manager_with_portfolio_data(self):
        """Initialize database manager and load all portfolio data."""
        bt.logging.info("📊 Initializing database manager with portfolio data cache...")
        try:
            self.db_manager = DatabaseManager(self.database_url)
            # Load all portfolio data into memory cache
            self.db_manager.load_all_portfolio_data(hotkeys=self.filter_hotkeys)
            self.portfolio_data_loaded = self.db_manager.portfolio_data_loaded
            bt.logging.info("✅ Database manager initialized with portfolio data cache")
        except Exception as e:
            bt.logging.error(f"Failed to initialize database manager: {e}")
            raise
    
    def _load_elimination_data(self, elimination_source: EliminationSource):
        """Load elimination data."""
        bt.logging.info("📊 Loading elimination data...")
        self.elimination_tracker = EliminationTracker(elimination_source_type=elimination_source)
        elimination_timestamps = self.elimination_tracker.load_all_eliminations(hotkeys=self.filter_hotkeys)
        self.eliminations_loaded = self.elimination_tracker.loaded
        
        # Fast-fail validation: eliminations must be loaded
        if not self.eliminations_loaded:
            raise RuntimeError("❌ Failed to load elimination data - this is required for accurate date bounds calculation")
        
        bt.logging.info(f"✅ Loaded elimination data for {len(elimination_timestamps)} miners")

    
    def _initialize_price_fetcher(self):
        """Initialize shared price fetcher for efficient caching across miners."""
        bt.logging.info("📊 Initializing shared price fetcher...")
        secrets = ValiUtils.get_secrets()
        self.live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
        bt.logging.info("✅ Shared price fetcher initialized (will cache prices across miners)")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_dates = len(self.price_source_cache)
        total_cached_prices = sum(len(prices) for prices in self.price_source_cache.values())
        
        return {
            'dates_cached': total_dates,
            'total_price_sources_cached': total_cached_prices,
            'cache_hits': self.price_cache_hits,
            'cache_misses': self.price_cache_misses,
            'hit_rate': self.price_cache_hits / (self.price_cache_hits + self.price_cache_misses) * 100 
                       if (self.price_cache_hits + self.price_cache_misses) > 0 else 0,
            'dates_list': sorted(self.price_source_cache.keys()) if total_dates < 100 else None
        }
    
    def clear_cache_before_date(self, cutoff_date_str: str):
        """Clear cache entries older than cutoff date to manage memory."""
        dates_to_remove = [date_str for date_str in self.price_source_cache.keys() 
                          if date_str < cutoff_date_str]
        for date_str in dates_to_remove:
            del self.price_source_cache[date_str]
        
        if dates_to_remove:
            bt.logging.info(f"🗑️ Cleared {len(dates_to_remove)} old cache entries before {cutoff_date_str}")
    
    def reset_miner_price_stats(self):
        """Reset per-miner price statistics for tracking individual miner processing."""
        self.current_miner_price_stats = {
            'cache_hits': 0,
            'live_fetches': 0,
            'total_days': 0
        }
    
    def get_miner_price_stats(self) -> Dict[str, int]:
        """Get current miner price statistics."""
        return self.current_miner_price_stats.copy()


def get_cached_or_fetch_price_sources(
    shared_data_manager: SharedDataManager,
    date_str: str,  # Date string passed in, not calculated
    target_date_ms: int,  # Still needed for fetching new prices
    required_trade_pairs: Set[str]
) -> Dict[str, Any]:
    """
    Get price sources using cache-first strategy.
    
    Args:
        shared_data_manager: Shared data manager with price cache
        date_str: YYYY-MM-DD date string (cache key)
        target_date_ms: Timestamp in ms (for fetching new prices)
        required_trade_pairs: Set of trade pair IDs needed
    
    Returns:
        Dict mapping trade pair ID to PriceSource
    """
    if not required_trade_pairs:
        return {}
    
    # Use provided date string directly - no conversion needed
    date_cache = shared_data_manager.price_source_cache.get(date_str, {})
    
    # Identify which prices we need to fetch
    missing_trade_pairs = required_trade_pairs - set(date_cache.keys())
    cached_trade_pairs = required_trade_pairs & set(date_cache.keys())
    
    # Update cache statistics (global and per-miner)
    shared_data_manager.price_cache_hits += len(cached_trade_pairs)
    shared_data_manager.price_cache_misses += len(missing_trade_pairs)
    
    # Track per-miner stats
    if cached_trade_pairs:
        shared_data_manager.current_miner_price_stats['cache_hits'] += 1
    if missing_trade_pairs:
        shared_data_manager.current_miner_price_stats['live_fetches'] += 1
    
    # Always count this as a day processed
    shared_data_manager.current_miner_price_stats['total_days'] += 1
    
    if cached_trade_pairs:
        bt.logging.debug(f"Cache hit for {date_str}: {len(cached_trade_pairs)} prices reused")
    
    if missing_trade_pairs:
        bt.logging.debug(f"Cache miss for {date_str}: fetching {len(missing_trade_pairs)} new prices")
        
        # Convert to TradePair objects for fetching
        trade_pairs_to_fetch = set()
        for tp_id in missing_trade_pairs:
            # Create TradePair from ID (simplified - may need adjustment based on actual implementation)
            trade_pair = TradePair.get_latest_trade_pair_from_trade_pair_id(tp_id)
            trade_pairs_to_fetch.add(trade_pair)
        
        # Fetch only missing prices
        price_fetcher = PriceFetcher(shared_data_manager.live_price_fetcher, max_workers=30)
        new_prices = price_fetcher.fetch_multiple_price_sources(
            trade_pairs_to_fetch, target_date_ms
        )
        
        # Convert back to use trade pair IDs as keys
        new_prices_by_id = {tp.trade_pair_id: ps for tp, ps in new_prices.items()}
        
        # Update cache with provided date string key
        if date_str not in shared_data_manager.price_source_cache:
            shared_data_manager.price_source_cache[date_str] = {}
        
        shared_data_manager.price_source_cache[date_str].update(new_prices_by_id)
        date_cache.update(new_prices_by_id)
        
        bt.logging.debug(f"📊 Price cache for {date_str}: now contains {len(shared_data_manager.price_source_cache[date_str])} trade pairs")
    
    # Return combined cached + newly fetched prices as TradePair -> PriceSource mapping
    # Need to convert back from IDs to TradePair objects for compatibility
    result = {}
    for tp_id in required_trade_pairs:
        if tp_id in date_cache:
            trade_pair = TradePair.get_latest_trade_pair_from_trade_pair_id(tp_id)
            result[trade_pair] = date_cache[tp_id]
    
    return result


def process_miner_with_main_logic(
    all_positions: Dict[str, List],
    hotkeys: List[str], 
    shared_data_manager: SharedDataManager,
    dry_run: bool = False,
    collect_only: bool = False
) -> Union[int, List[Dict]]:
    """
    Process a single miner using the exact same logic as the main processing loop.
    This ensures auto-backfill uses the same code path as single miner mode.
    Fast fails on any error - no try-catch masking.
    
    Args:
        collect_only: If True, return the collected returns instead of inserting them
    
    Returns:
        If collect_only=False: Number of days processed successfully
        If collect_only=True: List of daily return records for bulk processing
    """
    # Use the exact same logic as main processing but for this specific miner
    hotkey = hotkeys[0]  # Should only be one hotkey
    
    # Reset per-miner price statistics
    shared_data_manager.reset_miner_price_stats()
    
    # Get positions for this miner
    positions = all_positions.get(hotkey, [])
    if not positions:
        bt.logging.warning(f"No positions found for {hotkey}")
        return [] if collect_only else 0
    
    # Determine date bounds from positions (same as main logic)
    start_ms, end_ms = DateUtils.determine_date_bounds(all_positions, shared_data_manager.elimination_tracker.elimination_timestamps)
    
    # Use shared database manager and elimination tracker
    db_manager = shared_data_manager.db_manager
    elimination_tracker = shared_data_manager.elimination_tracker
    live_price_fetcher = shared_data_manager.live_price_fetcher
    
    # Get existing hotkey-date pairs for this miner (same as main logic)
    start_date_str = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
    end_date_str = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
    
    existing_hotkey_date_pairs = db_manager.get_existing_hotkey_date_pairs(
        start_date_str, end_date_str, [hotkey]
    )
    
    # Calculate total number of days to process (same as main logic)
    total_days = int((end_ms - start_ms) // MS_IN_24_HOURS) + 1
    bt.logging.info(f"📅 Total days in range: {total_days}")
    
    # Step through each day (same as main logic)
    current_ms = end_ms
    day_counter = 0

    # Batch process: Calculate all returns first, then insert all at once
    all_daily_returns_flattened = []  # Collect all return records for single bulk insert
    dates_processed = []
    
    bt.logging.info(f"📊 Calculating returns for all days...")
    while current_ms >= start_ms:
        day_counter += 1
        current_date = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc)
        
        # Calculate date string ONCE at the start of each day's processing
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip if this hotkey-date pair already exists (fine-grained checking)
        if (hotkey, current_date_str) in existing_hotkey_date_pairs:
            current_ms -= MS_IN_24_HOURS
            continue
        
        # Filter positions and identify required trade pairs using date string
        filtered_positions_by_hotkey, required_trade_pair_ids, skip_stats = PositionFilterAnalyzer.filter_and_analyze_positions_for_date(
            all_positions, current_ms, current_date_str, live_price_fetcher, elimination_tracker
        )
        
        if not filtered_positions_by_hotkey or hotkey not in filtered_positions_by_hotkey:
            bt.logging.info(f"⏭️  No positions or eliminated for {hotkey[:12]}... on {current_date_str}")
            current_ms -= MS_IN_24_HOURS
            continue
        
        # Process this day
        bt.logging.debug(f"⚡ Calculating returns for {current_date_str} for {hotkey}...")
        
        # Use cached price sources with date string key
        if not required_trade_pair_ids:
            bt.logging.debug(f"No open positions needing prices on {current_date_str}")
            cached_price_sources = {}  # Empty dict but not None
        else:
            # Use smart caching strategy with date string
            cached_price_sources = get_cached_or_fetch_price_sources(
                shared_data_manager,
                current_date_str,  # Pass the pre-calculated date string
                current_ms,
                required_trade_pair_ids
            )

        
        # Calculate returns for this day
        calculator = PortfolioCalculator(filtered_positions_by_hotkey, live_price_fetcher)
        daily_returns = calculator.calculate_daily_returns(current_date, cached_price_sources, live_price_fetcher)
        
        if daily_returns:
            # Flatten the returns and collect for single bulk insert
            all_daily_returns_flattened.extend(daily_returns)
            dates_processed.append(current_date_str)
            bt.logging.debug(f"✅ Calculated {len(daily_returns)} returns for {current_date_str}")
        else:
            bt.logging.warning(f"⚠️  No returns calculated for {current_date_str}")

        current_ms -= MS_IN_24_HOURS
    
    # Handle collect_only mode (for auto-backfill optimization)
    if collect_only:
        bt.logging.debug(f"✅ Collected {len(all_daily_returns_flattened)} return records for {hotkey}")
        return all_daily_returns_flattened
    
    # Single bulk insert for all returns at once to minimize database round trips
    days_inserted = 0
    if all_daily_returns_flattened and not dry_run:
        bt.logging.info(f"💾 Single bulk inserting {len(all_daily_returns_flattened)} return records across {len(dates_processed)} days...")
        
        # Single database call for all records  
        # Use a generic date string since we're inserting multiple dates
        success = db_manager.insert_daily_returns(all_daily_returns_flattened, "bulk_insert", skip_duplicates=True)
        if not success:
            raise RuntimeError(f"Failed to bulk insert {len(all_daily_returns_flattened)} return records")
        
        days_inserted = len(dates_processed)
        
        # Get price source statistics for detailed logging
        price_stats = shared_data_manager.get_miner_price_stats()
        bt.logging.info(f"✅ Miner {hotkey}: Successfully inserted {len(all_daily_returns_flattened)} portfolio return records across {len(dates_processed)} days "
                       f"(price fetching: {price_stats['cache_hits']} days cached, {price_stats['live_fetches']} days live-fetched, {price_stats['total_days']} days needed prices)")
    elif all_daily_returns_flattened and dry_run:
        days_inserted = len(dates_processed)
        
        # Get price source statistics for detailed logging
        price_stats = shared_data_manager.get_miner_price_stats()
        bt.logging.info(f"✅ [DRY RUN] Miner {hotkey}: Would bulk insert {len(all_daily_returns_flattened)} records across {len(dates_processed)} days "
                       f"(price fetching: {price_stats['cache_hits']} days cached, {price_stats['live_fetches']} days live-fetched, {price_stats['total_days']} days needed prices)")
    else:
        bt.logging.info(f"⚠️  No returns to insert")
    
    bt.logging.info(f"✨ Completed! Processed {day_counter} days, batch inserted {days_inserted} records")
    return days_inserted


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


class DatabaseManager:
    """Handles database operations for daily portfolio returns."""
    
    def __init__(self, connection_string: str):
        """Initialize database connection."""
        self.engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_timeout=60,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.ensure_table_exists()
        
        # In-memory cache for all portfolio data (loaded at initialization)
        self.all_portfolio_data = {}  # hotkey -> set of dates
        self.portfolio_data_loaded = False
    
    def ensure_table_exists(self):
        """Check that miner_port_values table exists - don't create new tables."""
        try:
            inspector = inspect(self.engine)
            if not inspector.has_table('miner_port_values'):
                raise RuntimeError("miner_port_values table does not exist in database. This script requires an existing miner_port_values table.")
            
            bt.logging.info("Database table 'miner_port_values' found and ready")
        except Exception as e:
            bt.logging.error(f"Failed to verify database table exists: {e}")
            raise
    
    def load_all_portfolio_data(self, hotkeys: Optional[List[str]] = None):
        """Load all portfolio value rows at initialization to minimize future database calls."""
        if self.portfolio_data_loaded:
            return  # Already loaded
            
        bt.logging.info("📊 Loading all portfolio value data at initialization...")
        try:
            with self.SessionFactory() as session:
                # Build query to load all portfolio data
                query = "SELECT miner_hotkey, date FROM miner_port_values"
                params = {}
                
                if hotkeys:
                    # Process in batches to avoid MySQL parameter limits
                    batch_size = 1000
                    for i in range(0, len(hotkeys), batch_size):
                        batch_hotkeys = hotkeys[i:i + batch_size]
                        
                        placeholders = ", ".join([f":hotkey_{j}" for j in range(len(batch_hotkeys))])
                        batch_query = f"{query} WHERE miner_hotkey IN ({placeholders})"
                        batch_params = {f"hotkey_{j}": hotkey for j, hotkey in enumerate(batch_hotkeys)}
                        
                        result = session.execute(text(batch_query), batch_params)
                        
                        for row in result.fetchall():
                            hotkey, date = row[0], row[1]
                            if hotkey not in self.all_portfolio_data:
                                self.all_portfolio_data[hotkey] = set()
                            self.all_portfolio_data[hotkey].add(date)
                        
                        if len(hotkeys) > batch_size:
                            bt.logging.info(f"   Loaded portfolio data for {min(i + batch_size, len(hotkeys))}/{len(hotkeys)} specified miners...")
                else:
                    # Load all portfolio data if no specific hotkeys
                    result = session.execute(text(query), params)
                    
                    for row in result.fetchall():
                        hotkey, date = row[0], row[1]
                        if hotkey not in self.all_portfolio_data:
                            self.all_portfolio_data[hotkey] = set()
                        self.all_portfolio_data[hotkey].add(date)
                
                self.portfolio_data_loaded = True
                total_miners = len(self.all_portfolio_data)
                total_records = sum(len(dates) for dates in self.all_portfolio_data.values())
                
                bt.logging.info(f"✅ Loaded portfolio data for {total_miners} miners with {total_records} total records in memory")
                
        except Exception as e:
            bt.logging.error(f"Failed to load all portfolio data: {e}")
            # Continue without in-memory cache - will fall back to per-query methods
            self.all_portfolio_data = {}
            self.portfolio_data_loaded = False
    
    def get_existing_dates(self, start_date: str, end_date: str) -> Set[str]:
        """Get all dates that already exist in database within date range."""
        try:
            with self.SessionFactory() as session:
                result = session.execute(text(
                    "SELECT DISTINCT date FROM miner_port_values "
                    "WHERE date >= :start_date AND date <= :end_date"
                ), {"start_date": start_date, "end_date": end_date})
                
                existing_dates = {row[0] for row in result.fetchall()}
                bt.logging.info(f"Found {len(existing_dates)} existing dates in database")
                return existing_dates
                
        except Exception as e:
            bt.logging.error(f"Failed to fetch existing dates: {e}")
            return set()

    def get_existing_hotkey_date_pairs(self, start_date: str, end_date: str, hotkeys: Optional[List[str]] = None) -> \
            Set[Tuple[str, str]]:
        """Get all (hotkey, date) pairs that already exist in database within date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            hotkeys: Optional list of specific hotkeys to check

        Returns:
            Set of (hotkey, date) tuples that already exist in database
        """
        # Use in-memory cache if available
        if self.portfolio_data_loaded:
            existing_pairs = set()

            # Filter by hotkeys if specified
            miners_to_check = set(hotkeys) if hotkeys else set(self.all_portfolio_data.keys())

            min_found_date_str = None
            max_found_date_str = None
            for hotkey in miners_to_check:
                if hotkey in self.all_portfolio_data:
                    # Filter dates within range
                    for date_str in self.all_portfolio_data[hotkey]:
                        if start_date <= date_str <= end_date:
                            existing_pairs.add((hotkey, date_str))
                            if min_found_date_str is None or date_str < min_found_date_str:
                                min_found_date_str = date_str
                            if max_found_date_str is None or date_str > max_found_date_str:
                                max_found_date_str = date_str

            bt.logging.info(
                f"Found {len(existing_pairs)} existing hotkey-date pairs from in-memory cache. min/max found_date_str {min_found_date_str}/{max_found_date_str}")
            return existing_pairs

        # Fall back to database query if cache not loaded
        try:
            with self.SessionFactory() as session:
                query = (
                    "SELECT DISTINCT miner_hotkey, date FROM miner_port_values "
                    "WHERE date >= :start_date AND date <= :end_date"
                )
                params = {"start_date": start_date, "end_date": end_date}

                if hotkeys:
                    placeholders = ", ".join([f":hotkey_{i}" for i in range(len(hotkeys))])
                    query += f" AND miner_hotkey IN ({placeholders})"
                    for i, hotkey in enumerate(hotkeys):
                        params[f"hotkey_{i}"] = hotkey

                result = session.execute(text(query), params)

                existing_pairs = {(row[0], row[1]) for row in result.fetchall()}
                bt.logging.info(f"Found {len(existing_pairs)} existing hotkey-date pairs from database query")
                return existing_pairs

        except Exception as e:
            bt.logging.error(f"Failed to fetch existing hotkey-date pairs: {e}")
            return set()
    
    def insert_daily_returns(self, daily_returns: List[Dict], current_date_str: str, skip_duplicates: bool = False) -> bool:
        """Insert daily returns for a specific date using miner_port_values schema with bulk operations.
        
        Args:
            daily_returns: List of return dictionaries to insert
            current_date_str: current_date_str
            skip_duplicates: If True, skip individual records that already exist
        """
        if not daily_returns:
            bt.logging.warning(f"No returns to insert for {current_date_str}")
            return False

        try:
            with self.SessionFactory() as session:
                records_to_insert = daily_returns
                skipped_count = 0
                
                if skip_duplicates:
                    # Single bulk query to check all existing records
                    hotkey_timestamp_pairs = [(r['miner_hotkey'], r['timestamp']) for r in daily_returns]
                    
                    # Use bulk query with IN clause for all hotkeys and timestamps
                    existing_query = session.query(
                        MinerPortValuesModel.miner_hotkey, 
                        MinerPortValuesModel.timestamp
                    ).filter(
                        tuple_(MinerPortValuesModel.miner_hotkey, MinerPortValuesModel.timestamp).in_(hotkey_timestamp_pairs)
                    )
                    
                    existing_records = set(existing_query.all())
                    
                    # Filter out existing records
                    records_to_insert = []
                    for record in daily_returns:
                        key = (record['miner_hotkey'], record['timestamp'])
                        if key in existing_records:
                            skipped_count += 1
                        else:
                            records_to_insert.append(record)
                
                if records_to_insert:
                    # Use bulk_insert_mappings for maximum efficiency
                    session.bulk_insert_mappings(MinerPortValuesModel, records_to_insert)
                
                session.commit()
                
                inserted_count = len(records_to_insert)
                if skipped_count > 0:
                    bt.logging.info(f"✓ Bulk inserted {inserted_count} returns, skipped {skipped_count} duplicates for {current_date_str}")
                else:
                    bt.logging.info(f"✓ Bulk inserted {inserted_count} returns for {current_date_str}")
                return True
                
        except Exception as e:
            bt.logging.error(f"Failed to bulk insert returns for {current_date_str}: {e}")
            return False



@dataclass
class DailyStats:
    """Statistics for a single day's return calculations."""
    successful_miners: int = 0
    failed_miners: int = 0
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    position_returns: List[float] = None
    portfolio_returns: List[float] = None
    extreme_returns: Dict = None
    trade_pair_usage: Dict[str, int] = None
    skip_stats: FilterStats = None
    elimination_stats: Dict = None
    
    def __post_init__(self):
        if self.position_returns is None:
            self.position_returns = []
        if self.portfolio_returns is None:
            self.portfolio_returns = []
        if self.extreme_returns is None:
            self.extreme_returns = {'best': None, 'worst': None}
        if self.trade_pair_usage is None:
            self.trade_pair_usage = defaultdict(int)
        if self.skip_stats is None:
            self.skip_stats = FilterStats()
        if self.elimination_stats is None:
            self.elimination_stats = {"total_hotkeys": 0, "eliminated_hotkeys": 0}


class PositionFilterAnalyzer:
    """Extended position filtering with analysis capabilities for daily_portfolio_returns.py

    Uses PositionFilter from position_filter.py for the core filtering logic.
    """

    @staticmethod
    def filter_and_analyze_positions(
        all_positions: Dict[str, List[Position]],
        target_date_ms: int
    ) -> Tuple[Dict[str, List[Position]], Set[TradePair], FilterStats]:
        """
        Filter positions by date and identify trade pairs needing prices.
        
        Returns:
            Tuple of (filtered_positions_by_hotkey, trade_pairs_needing_prices, filter_stats)
        """
        filtered_positions_by_hotkey = {}
        trade_pairs_needing_prices = set()
        stats = FilterStats()
        
        for hotkey, positions in all_positions.items():
            stats.total_positions_before_filter += len(positions)
            filtered_positions = []
            
            for position in positions:
                filtered_position, skip_reason = PositionFilter.filter_single_position(position, target_date_ms, self.live_price_fetcher)
                
                if skip_reason == "equities":
                    stats.equities_positions_skipped += 1
                    bt.logging.debug(f"Skipping {position.trade_pair.trade_pair} position - equities not supported")
                elif skip_reason == "indices":
                    stats.indices_positions_skipped += 1
                    bt.logging.debug(f"Skipping {position.trade_pair.trade_pair} position - indices not supported")
                elif skip_reason == "date_filtered":
                    stats.date_filtered_out += 1
                elif skip_reason == "kept" and filtered_position:
                    filtered_positions.append(filtered_position)
                    stats.final_positions += 1
                    
                    # Check if position needs a price (open at target date)
                    if not filtered_position.is_closed_position or (filtered_position.close_ms and filtered_position.close_ms > target_date_ms):
                        trade_pairs_needing_prices.add(filtered_position.trade_pair)
            
            if filtered_positions:
                filtered_positions_by_hotkey[hotkey] = filtered_positions
        
        return filtered_positions_by_hotkey, trade_pairs_needing_prices, stats
    
    @staticmethod
    def filter_and_analyze_positions_for_date(
        all_positions: Dict[str, List[Position]],
        target_date_ms: int,
        date_str: str,  # Date string passed in
        live_price_fetcher,
        elimination_tracker: Optional['EliminationTracker'] = None
    ) -> Tuple[Dict[str, List[Position]], Set[str], FilterStats]:
        """
        Filter positions and identify required trade pairs for a specific date.
        Returns trade pair IDs (not TradePair objects) for easier caching.
        
        Args:
            all_positions: All positions by hotkey
            target_date_ms: Target timestamp in milliseconds
            date_str: YYYY-MM-DD date string for logging
        
        Returns:
            Tuple of (filtered_positions, required_trade_pair_ids, filter_stats)
        """
        filtered_positions_by_hotkey = {}
        required_trade_pair_ids = set()  # Use IDs for cache keys
        stats = FilterStats()
        
        bt.logging.debug(f"Filtering positions for {date_str} ({target_date_ms} ms)")
        
        for hotkey, positions in all_positions.items():
            # Skip eliminated miners entirely to avoid unnecessary position processing
            if elimination_tracker and elimination_tracker.is_hotkey_eliminated_at_date(hotkey, target_date_ms):
                bt.logging.debug(f"Skipping eliminated miner {hotkey[:12]} for {date_str}")
                continue
                
            stats.total_positions_before_filter += len(positions)
            filtered_positions = []
            
            for position in positions:
                filtered_position, skip_reason = PositionFilter.filter_single_position(position, target_date_ms, live_price_fetcher)
                
                if skip_reason == "equities":
                    stats.equities_positions_skipped += 1
                elif skip_reason == "indices":
                    stats.indices_positions_skipped += 1
                elif skip_reason == "date_filtered":
                    stats.date_filtered_out += 1
                elif skip_reason == "kept" and filtered_position:
                    filtered_positions.append(filtered_position)
                    stats.final_positions += 1
                    
                    # Precisely check if position needs a price on this specific date
                    position_is_active = False
                    if filtered_position.open_ms <= target_date_ms:
                        if filtered_position.is_open_position:
                            # Open position: always needs pricing
                            position_is_active = True
                    
                    if position_is_active:
                        # Use trade pair ID for cache key
                        required_trade_pair_ids.add(filtered_position.trade_pair.trade_pair_id)
            
            if filtered_positions:
                filtered_positions_by_hotkey[hotkey] = filtered_positions
        
        bt.logging.debug(f"Date {date_str}: {len(filtered_positions_by_hotkey)} miners, "
                        f"{len(required_trade_pair_ids)} unique trade pairs needed")
        
        return filtered_positions_by_hotkey, required_trade_pair_ids, stats


class PriceFetcher:
    """Handles multi-threaded price fetching for trade pairs."""
    
    def __init__(self, live_price_fetcher: LivePriceFetcher, max_workers: int = 30):
        self.live_price_fetcher = live_price_fetcher
        self.max_workers = max_workers
    
    def fetch_single_price_source(self, trade_pair: TradePair, target_date_ms: int) -> Tuple[TradePair, PriceSource, str]:
        """Fetch price source for a single trade pair."""
        target_datetime = datetime.fromtimestamp(target_date_ms / 1000, tz=timezone.utc)
        target_date_utc_str = target_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        bt.logging.debug(f"Fetching price for {trade_pair.trade_pair} at {target_date_utc_str}")
        
        price_source = self.live_price_fetcher.get_close_at_date(
            trade_pair=trade_pair,
            timestamp_ms=target_date_ms,
            verbose=False
        )
        
        if price_source is None:
            error_msg = f"Failed to fetch price for {trade_pair.trade_pair} at {target_date_utc_str}"
            raise Exception(error_msg)
        
        return trade_pair, price_source, None
    
    def fetch_multiple_price_sources(
        self, 
        trade_pairs: Set[TradePair], 
        target_date_ms: int
    ) -> Dict[TradePair, PriceSource]:
        """Fetch price sources for multiple trade pairs concurrently."""
        if not trade_pairs:
            return {}
        
        start_time = time.time()
        
        target_datetime = datetime.fromtimestamp(target_date_ms / 1000, tz=timezone.utc)
        target_date_utc_str = target_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        price_sources = {}
        errors = []
        
        bt.logging.debug(f"Starting parallel price fetch for {len(trade_pairs)} trade pairs at {target_date_utc_str} using {self.max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_trade_pair = {
                executor.submit(self.fetch_single_price_source, tp, target_date_ms): tp
                for tp in trade_pairs
            }
            
            for future in as_completed(future_to_trade_pair):
                trade_pair, price_source, error_msg = future.result()
                
                if error_msg is None:
                    price_sources[trade_pair] = price_source
                else:
                    errors.append(error_msg)
                    bt.logging.error(error_msg)
        
        elapsed_time = time.time() - start_time
        bt.logging.debug(f"Price fetch results for {target_date_utc_str}: {len(price_sources)} successful, {len(errors)} failed (took {elapsed_time:.2f}s)")
        
        if errors:
            error_summary = f"Failed to fetch prices for {len(errors)} trade pairs: {errors[:3]}"
            if len(errors) > 3:
                error_summary += f" (and {len(errors) - 3} more)"
            raise Exception(error_summary)
        
        return price_sources


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
        do_special_fetch = position.is_closed_position and \
                position.return_at_close == 1.0 and \
                position.orders and \
                position.orders[-1].src == 1

        if position.is_closed_position and not do_special_fetch:
            return position.return_at_close
        
        # For open positions, calculate using cached price
        if position.trade_pair not in cached_price_sources:
            if not do_special_fetch:
                raise ValueError(f"Price not available for {position.trade_pair.trade_pair} at target date")

        if do_special_fetch:
            orders_to_use = position.orders[:-1]  # Exclude the last order which is the closing order with src = 1
            price_source = live_price_fetcher.get_close_at_date(
                trade_pair=position.trade_pair,
                timestamp_ms=position.close_ms,
                verbose=False
            )
        else:
            price_source = cached_price_sources[position.trade_pair]
            orders_to_use = position.orders[:]
        
        # Create a copy to avoid modifying the original position
        position_copy = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms,
            trade_pair=position.trade_pair,
            orders=orders_to_use,
            position_type=position.position_type,
            is_closed_position=position.is_closed_position,
        )
        position_copy.rebuild_position_with_updated_orders(live_price_fetcher)
        
        price = price_source.parse_appropriate_price(
            now_ms=target_date_ms,
            is_forex=position.trade_pair.is_forex,
            order_type=position.orders[0].order_type,
            position=position_copy
        )
        
        position_copy.set_returns(price, live_price_fetcher, time_ms=target_date_ms)
        if do_special_fetch:
            bt.logging.warning(f'Special fetch for closed position {position.position_uuid} at {position.close_ms} ms. '
                               f'Trade pair {position.trade_pair.trade_pair} Return: {position_copy.return_at_close}. price_source: {price_source}')
        return position_copy.return_at_close


class PositionCategorizer:
    """Handles categorization of positions into crypto/forex subcategories."""
    
    @staticmethod
    def categorize_position(trade_pair_obj:TradePair) -> List[str]:
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


class PositionAnalyzer:
    """Analyzes positions for statistical insights."""
    
    @staticmethod
    def analyze_positions(
        positions: List[Position],
        target_date_ms: int,
        cached_price_sources: Dict[TradePair, PriceSource]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze positions and calculate categorized returns.
        This is the main method used by PortfolioCalculator.
        
        Args:
            positions: List of positions to analyze
            target_date_ms: Target timestamp in milliseconds
            cached_price_sources: Cached price sources for calculations
            
        Returns:
            Dictionary of categories with returns and counts
        """
        # Use CategoryReturnCalculator to calculate returns by category
        return CategoryReturnCalculator.calculate_miner_returns_by_category(
            positions, target_date_ms, cached_price_sources
        )
    
    @staticmethod
    def prepare_database_records(
        miner_returns: Dict[str, Dict[str, Dict[str, float]]],
        target_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """
        Prepare database records from miner returns.
        This is used by PortfolioCalculator to format data for insertion.
        
        Args:
            miner_returns: Dictionary of miner portfolio values by category
            target_timestamp: The timestamp for the database records
            
        Returns:
            List of dictionaries ready for database insertion
        """
        # Use CategoryReturnCalculator's prepare_insert_values method
        return CategoryReturnCalculator.prepare_insert_values(
            miner_returns, target_timestamp
        )


class CategoryReturnCalculator:
    """Calculates returns by asset categories and subcategories."""
    
    @staticmethod
    def calculate_miner_returns_by_category(
        positions: List[Position],
        target_date_ms: int,
        cached_price_sources: Dict[TradePair, PriceSource], live_price_fetcher, pos_debug=False
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
        
        # Collect position returns for debugging
        position_returns_debug = []
        
        # Process each position (following reference logic exactly)
        for position in positions:
            # Calculate return_at_close for this position at the target date
            position_return = ReturnCalculator.calculate_position_return(
                position, target_date_ms, cached_price_sources, live_price_fetcher
            )
            
            # Fail fast - if we can't calculate return, something is wrong
            #if not position_return:
            #    raise ValueError(f"Position {position.position_uuid} returned invalid return: {position_return}")
            
            # Fail fast - if return is not positive, something is wrong
            if position_return < 0:
                raise ValueError(f"Position {position.position_uuid} returned non-positive return: {position_return}")
                
            miner_hotkey = position.miner_hotkey

            # Collect for debugging
            if pos_debug:
                trade_pair_str = position.trade_pair.trade_pair  # Get the actual trade pair string
                position_returns_debug.append((trade_pair_str, position_return))
            
            # Determine categories for this position
            categories = PositionCategorizer.categorize_position(position.trade_pair)
            
            # Add return to each relevant category
            for category in categories:
                # Directly multiply the return values together (reference logic)
                miner_category_returns[miner_hotkey][category]["return"] *= position_return
                miner_category_returns[miner_hotkey][category]["count"] += 1
        
        # Log position returns for debugging
        if position_returns_debug:
            target_date = datetime.fromtimestamp(target_date_ms / 1000, tz=timezone.utc)
            bt.logging.info(f"📊 Position returns for {target_date.strftime('%Y-%m-%d')} ({len(position_returns_debug)} positions): {position_returns_debug}")
        
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


class PortfolioCalculator:
    """Handles portfolio return calculations for miners."""
    
    def __init__(self, positions_by_hotkey: Dict[str, List], live_price_fetcher):
        """
        Initialize portfolio calculator.
        
        Args:
            positions_by_hotkey: Dictionary mapping hotkey -> list of positions
            live_price_fetcher: LivePriceFetcher instance for price data
        """
        self.positions_by_hotkey = positions_by_hotkey
        self.live_price_fetcher = live_price_fetcher
    
    def calculate_daily_returns(self, target_date: datetime, cached_price_sources: Dict, live_price_fetcher) -> List[Dict]:
        """
        Calculate daily portfolio returns for all miners.
        
        Args:
            target_date: The date to calculate returns for
            cached_price_sources: REQUIRED pre-fetched price sources (must not be None)
            
        Returns:
            List of dictionaries ready for database insertion
        """
        # Enforce cached_price_sources is provided
        if cached_price_sources is None:
            raise ValueError("cached_price_sources is required and cannot be None")
        
        target_date_ms = int(target_date.timestamp() * 1000)
        
        # Process each miner separately
        all_results = []
        for hotkey, positions in self.positions_by_hotkey.items():
            if not positions:
                continue
            
            # Calculate returns for this specific miner's positions
            miner_returns = CategoryReturnCalculator.calculate_miner_returns_by_category(
                positions, target_date_ms, cached_price_sources, live_price_fetcher
            )
            
            # Convert to database format for this miner
            if miner_returns:
                miner_records = CategoryReturnCalculator.prepare_insert_values(miner_returns, target_date)
                all_results.extend(miner_records)
        
        return all_results

class DateUtils:
    """Utility functions for date handling."""
    
    @staticmethod
    def determine_date_bounds(positions_dict: Dict[str, List[Position]], 
                            elimination_timestamps: Dict[str, int]) -> Tuple[int, int]:
        """
        Determine start and end dates based on first and last order times.
        Adjusts end date per miner to elimination_time + 1 day.
        
        Args:
            positions_dict: Dictionary mapping hotkey -> list of positions
            elimination_timestamps: Dict mapping hotkey -> elimination_ms (required)
            
        Returns:
            Tuple of (start_date_ms, end_date_ms)
        """
        if not positions_dict:
            raise ValueError("No positions found to determine date bounds")
        
        if elimination_timestamps is None:
            raise ValueError("Elimination timestamps are required for date bounds calculation")

        first_order_ms = float('inf')
        has_active_miner = False  # Track if any miner is not eliminated
        latest_elimination_ms = 0  # Track the latest elimination date

        # First pass: find earliest order and check elimination status
        for hotkey, positions in positions_dict.items():
            for position in positions:
                first_order_ms = min(position.orders[0].processed_ms, first_order_ms)

            # Check elimination status once per hotkey (not per position)
            if hotkey in elimination_timestamps:
                elimination_ms = elimination_timestamps[hotkey]
                latest_elimination_ms = max(latest_elimination_ms, elimination_ms)
            else:
                has_active_miner = True

        if first_order_ms == float('inf'):
            raise ValueError("No valid orders found in positions")

        # Determine final end date based on whether any miner is still active
        if has_active_miner:
            # If any miner is active, process through today for all miners
            today_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            final_end_ms = (today_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS
            elim_time = None
        else:
            # All miners eliminated, use latest elimination + 1 day
            elimination_day_start = (latest_elimination_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS
            final_end_ms = elimination_day_start + MS_IN_24_HOURS
            elim_time = latest_elimination_ms

        # Round to start of UTC day
        first_date_ms = (first_order_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS

        bt.logging.info(f"Date bounds determined: "
                        f"{TimeUtil.millis_to_formatted_date_str(first_date_ms)} to "
                        f"{TimeUtil.millis_to_formatted_date_str(final_end_ms)}. "
                        f"elim time ({TimeUtil.millis_to_formatted_date_str(elim_time) if elim_time else 'N/A'})")

        return first_date_ms, final_end_ms


def get_database_url_from_config() -> Optional[str]:
    """
    Read database URL from config-development.json file in working directory.
    Ensures the URL includes the database name for a complete connection string.
    
    Returns:
        Complete database URL string if found, None otherwise
    """
    config_file = "config-development.json"
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Navigate to secrets.db_ptn_editor_url
            base_url = config.get('secrets', {}).get('db_ptn_editor_url')
            
            if base_url:
                # Parse URL to ensure it has a database name
                parts = base_url.split('/')
                
                if len(parts) >= 4 and parts[-1]:
                    # URL already complete
                    bt.logging.info(f"Found complete database URL in {config_file}")
                    return base_url
                elif len(parts) >= 3:
                    # Has connection info but missing database name
                    if not base_url.endswith('/'):
                        base_url += '/'
                    complete_url = f"{base_url}taoshi-ts-ptn"
                    bt.logging.info(f"Completed database URL from {config_file} by appending database name")
                    return complete_url
                else:
                    bt.logging.error(f"Invalid database URL format in {config_file}")
                    return None
            else:
                bt.logging.warning(f"No db_ptn_editor_url found in {config_file}")
                return None
        else:
            bt.logging.debug(f"Config file {config_file} not found in working directory")
            return None
            
    except json.JSONDecodeError as e:
        bt.logging.error(f"Failed to parse {config_file}: {e}")
        return None
    except Exception as e:
        bt.logging.error(f"Error reading {config_file}: {e}")
        return None


class EliminationTracker:
    """Handles elimination checking and filtering for miners."""
    
    def __init__(self, elimination_source_type: EliminationSource = EliminationSource.DATABASE):
        """Initialize elimination tracker with specified source."""
        self.elimination_source_manager = EliminationSourceManager(source_type=elimination_source_type)
        self.eliminations_by_hotkey = {}
        self.elimination_timestamps = {}  # hotkey -> elimination_time_ms
        self.loaded = False
        
    def load_all_eliminations(self, hotkeys: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Load all eliminations for the specified hotkeys (or all if None).
        Returns a dict mapping hotkeys to their elimination timestamps.
        """
        bt.logging.info("Loading elimination data for filtering...")
        # Load eliminations over all time (no time restrictions)
        self.eliminations_by_hotkey = self.elimination_source_manager.load_eliminations(
            start_time_ms=None,  # All time
            end_time_ms=None,    # All time
            hotkeys=hotkeys
        )

        # Create fast lookup: hotkey -> earliest elimination timestamp
        self.elimination_timestamps = {}

        for hotkey, eliminations in self.eliminations_by_hotkey.items():
            if eliminations:
                # Get the earliest elimination time for this miner
                earliest_elimination_ms = min(
                    elim.get('elimination_time_ms', float('inf'))
                    for elim in eliminations
                )
                if earliest_elimination_ms != float('inf'):
                    self.elimination_timestamps[hotkey] = earliest_elimination_ms

        self.loaded = True

        # Log summary
        total_eliminations = sum(len(elims) for elims in self.eliminations_by_hotkey.values())
        summary = self.elimination_source_manager.get_eliminations_summary(self.eliminations_by_hotkey)

        bt.logging.info(f"Loaded {total_eliminations} elimination records for {len(self.eliminations_by_hotkey)} miners")
        bt.logging.info(f"Elimination summary: {summary}")

        if self.elimination_timestamps:
            earliest_elim = min(self.elimination_timestamps.values())
            latest_elim = max(self.elimination_timestamps.values())
            bt.logging.info(f"Elimination time range: "
                           f"{TimeUtil.millis_to_formatted_date_str(earliest_elim)} to "
                           f"{TimeUtil.millis_to_formatted_date_str(latest_elim)}")

        return self.elimination_timestamps

    def is_hotkey_eliminated_at_date(self, hotkey: str, target_date_ms: int) -> bool:
        """
        Check if a hotkey is eliminated at the given date.
        Returns True if eliminated (should be skipped).
        
        Fixed behavior: Miners should be processed THROUGH their elimination date,
        then have a final row inserted on elimination day + 1, then be excluded.
        """
        if not self.loaded or hotkey not in self.elimination_timestamps:
            return False  # Not eliminated or no elimination data
        
        elimination_time = self.elimination_timestamps[hotkey]
        # Allow processing through elimination date, then stop after elimination day + 1
        elimination_day_plus_1_ms = elimination_time + MS_IN_24_HOURS
        return target_date_ms > elimination_day_plus_1_ms  # Skip only after elimination day + 1
    


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


def run_automated_backfill(miners_with_orders: Dict[str, Dict[str, Any]], shared_data_manager: SharedDataManager = None,
                           dry_run: bool = False) -> Dict[str, Any]:
    """Run automated backfill process for miners with missing data.

    Args:
        miners_with_orders: Dict mapping hotkey -> {first_order_date, last_order_date, missing_days}
        shared_data_manager: SharedDataManager with pre-loaded data for efficient processing
        dry_run: If True, only show what would be done without executing

    Returns:
        Dict with backfill results
    """


    if not miners_with_orders:
        bt.logging.info("✅ No miners need backfill - all portfolio data is up to date!")
        return {"successful": [], "failed": [], "total_days_backfilled": 0}

    bt.logging.info("=" * 80)
    bt.logging.info("AUTOMATED PORTFOLIO BACKFILL")
    bt.logging.info("=" * 80)
    bt.logging.info(f"Found {len(miners_with_orders)} miners needing backfill")

    if dry_run:
        raise Exception

    bt.logging.info("🚀 Starting automated backfill process...")
    bt.logging.info("Processing miners serially to avoid database conflicts...")

    # Use shared data manager if provided (more efficient), otherwise fall back to individual loading
    assert shared_data_manager

    successful_backfills = []
    failed_backfills = []
    total_days_backfilled = 0

    bt.logging.info("📋 Processing ALL miners with missing data (no filtering applied)")
    bt.logging.info(f"🚀 Immediate write processing: Calculate and write each of {len(miners_with_orders)} miners immediately")

    # Process each miner and write immediately
    bt.logging.info("📊 Processing miners with immediate writes...")
    total_start = time.time()

    for i, (hotkey, info) in enumerate(miners_with_orders.items(), 1):
        bt.logging.info(f"📊 [{i}/{len(miners_with_orders)}] Processing {hotkey}...")
        miner_start_time = time.time()

        # Use the same main processing logic but don't collect - directly insert
        single_hotkey_positions = {hotkey: shared_data_manager.all_positions.get(hotkey, [])}

        # Fast fail on any error - no exception masking
        # Use collect_only=False to directly insert into database
        days_processed = process_miner_with_main_logic(
            single_hotkey_positions,
            [hotkey],
            shared_data_manager,
            dry_run=dry_run,
            collect_only=False  # Write immediately instead of collecting
        )

        elapsed_time = time.time() - miner_start_time
        
        # Get price source statistics for detailed logging
        price_stats = shared_data_manager.get_miner_price_stats()
        
        bt.logging.info(f"✅ [{i}/{len(miners_with_orders)}] {hotkey}: {days_processed} days processed and written in {elapsed_time:.1f}s "
                       f"(price fetching: {price_stats['cache_hits']} cached, {price_stats['live_fetches']} live-fetched)")
        
        # Track statistics
        if days_processed > 0:
            successful_backfills.append({
                "hotkey": hotkey,
                "days_processed": days_processed,
                "output_sample": f"Processed and written in {elapsed_time:.1f}s"
            })
            total_days_backfilled += days_processed
        else:
            bt.logging.warning(f"⚠️  No days processed for {hotkey}")

    total_time = time.time() - total_start
    bt.logging.info(f"✅ All miners completed in {total_time:.1f}s with immediate writes")

    # Print summary
    bt.logging.info("")
    bt.logging.info("=" * 80)
    bt.logging.info("OPTIMIZED BATCH BACKFILL SUMMARY")
    bt.logging.info("=" * 80)
    bt.logging.info(f"Total Miners Processed: {len(miners_with_orders)}")
    bt.logging.info(f"✅ Successful: {len(successful_backfills)}")
    bt.logging.info(f"❌ Failed: {len(failed_backfills)}")
    bt.logging.info(f"📊 Total Records Processed: {total_days_backfilled}")
    if not dry_run and total_days_backfilled > 0:
        bt.logging.info(f"🚀 Database Optimization: Used single bulk insert instead of {len(miners_with_orders)} individual inserts")
        bt.logging.info(f"⚡ Performance: Reduced database calls by {len(miners_with_orders)-1}x ({len(miners_with_orders)} → 1 calls)")
    bt.logging.info("")

    if successful_backfills:
        bt.logging.info("✅ SUCCESSFUL BACKFILLS:")
        for success in successful_backfills[:10]:  # Show first 10 to avoid spam
            bt.logging.info(f"  {success['hotkey']} - {success['days_processed']} records - {success['output_sample']}")
        if len(successful_backfills) > 10:
            bt.logging.info(f"  ... and {len(successful_backfills) - 10} more successful backfills")

    if failed_backfills:
        bt.logging.info("")
        bt.logging.info("❌ FAILED BACKFILLS:")
        for failure in failed_backfills:
            bt.logging.info(f"{failure['hotkey']} - {failure['error']}")

    bt.logging.info("")
    bt.logging.info("=" * 80)

    return {
        "successful": successful_backfills,
        "failed": failed_backfills,
        "total_days_backfilled": total_days_backfilled
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate daily portfolio returns for miners")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (defaults to first order date)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (defaults to last order date)",
    )
    parser.add_argument(
        "--hotkeys",
        type=str,
        help="Comma-separated list of hotkeys to analyze (defaults to all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (only used with --save-csv)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Maximum number of threads for price fetching (default: 5)",
    )
    parser.add_argument(
        "--elimination-source",
        type=str,
        default="DATABASE",
        choices=["DATABASE", "TEST", "DISK"],
        help="Source for elimination data (default: DATABASE)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database connection string (if not provided, will read from config-development.json or secrets)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV file (in addition to or instead of database)",
    )
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="Run in diagnostic mode to detect missing days and miners without processing",
    )
    parser.add_argument(
        "--auto-backfill",
        action="store_true",
        help="Automatically detect and backfill missing portfolio data for all miners",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backfilled without actually processing (use with --auto-backfill)",
    )
    return parser.parse_args()




def main():
    """Main function to calculate daily portfolio returns."""
    args = parse_args()
    
    # Configure logging
    bt.logging.set_trace(args.log_level == "DEBUG")
    bt.logging.set_debug(args.log_level == "DEBUG")
    
    # Parse hotkeys if provided
    hotkeys = None
    if args.hotkeys:
        hotkeys = [h.strip() for h in args.hotkeys.split(",")]
        bt.logging.info(f"Filtering for hotkeys: {hotkeys}")
    
    # Initialize position source manager
    position_source_manager = PositionSourceManager(source_type=PositionSource.DATABASE)
    
    # Load all positions first to determine date bounds
    bt.logging.info("Loading positions from database...")
    all_positions = position_source_manager.load_positions(
        start_time_ms=None,  # Get all positions to determine bounds
        end_time_ms=None,
        hotkeys=hotkeys
    )
    
    if not all_positions:
        bt.logging.error("No positions found in database")
        return
    
    # Initialize elimination tracker
    elimination_source = getattr(EliminationSource, args.elimination_source)
    elimination_tracker = EliminationTracker(elimination_source_type=elimination_source)
    
    # Load all eliminations for filtering
    elimination_timestamps = elimination_tracker.load_all_eliminations(hotkeys=hotkeys)
    
    # Fast-fail validation: eliminations must be loaded
    if not elimination_tracker.loaded:
        bt.logging.error("❌ Failed to load elimination data - this is required for accurate date bounds calculation")
        raise RuntimeError("Elimination data loading failed - cannot continue")
    
    # Log elimination loading summary
    bt.logging.info(f"✅ Elimination data loaded successfully: {len(elimination_timestamps)} miners have eliminations on record")
    
    # Handle auto-backfill mode
    if args.auto_backfill:
        bt.logging.info("🚀 Running automated backfill mode...")
        
        # Analyze miners with orders to determine what needs backfilling
        miners_with_orders = analyze_miners_with_orders(all_positions)
        
        bt.logging.info(f"Found {len(miners_with_orders)} miners with orders. Checking portfolio coverage...")
        
        # Initialize shared data manager for efficient backfill processing
        database_url = args.database_url
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
        shared_data_manager = SharedDataManager(database_url, hotkeys=hotkeys)
        shared_data_manager.initialize_all_data(elimination_source=getattr(EliminationSource, args.elimination_source))


        # Run automated backfill with shared data manager
        _ = run_automated_backfill(miners_with_orders, shared_data_manager, dry_run=args.dry_run)

        bt.logging.info("🎯 Automated backfill process completed!")
        return


    # Determine date bounds dynamically if not provided
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
    else:
        # Determine bounds from positions
        start_ms, end_ms = DateUtils.determine_date_bounds(all_positions, elimination_timestamps)
        
        # Override with user-provided dates if any
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ms = int(start_date.timestamp() * 1000)
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_ms = int(end_date.timestamp() * 1000)
    
    bt.logging.info(f"Calculating returns from {TimeUtil.millis_to_formatted_date_str(start_ms)} "
                    f"to {TimeUtil.millis_to_formatted_date_str(end_ms)}")
    
    # Initialize database manager (default behavior)
    db_manager = None
    existing_dates = set()
    existing_hotkey_date_pairs = set()
    
    try:
        # Get database connection string
        database_url = args.database_url
        if not database_url:
            # Try to get from config-development.json
            database_url = get_database_url_from_config()
            
        if not database_url:
            # Fallback: try to get from ValiUtils secrets
            try:
                secrets = ValiUtils.get_secrets()
                database_url = secrets.get('database_url') or secrets.get('db_ptn_editor_url')
            except Exception as e:
                bt.logging.debug(f"Could not get database URL from ValiUtils secrets: {e}")
        
        if database_url:
            bt.logging.info("Initializing database connection...")
            db_manager = DatabaseManager(database_url)
            
            start_date_str = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
            end_date_str = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
            
            # Get existing hotkey-date pairs for fine-grained duplicate checking
            existing_hotkey_date_pairs = db_manager.get_existing_hotkey_date_pairs(
                start_date_str, end_date_str, hotkeys
            )
            
            if existing_hotkey_date_pairs:
                bt.logging.info(f"Found {len(existing_hotkey_date_pairs)} existing hotkey-date pairs")
                # Show sample of what will be skipped
                sample_pairs = list(existing_hotkey_date_pairs)[:3]
                for hotkey, date in sample_pairs:
                    bt.logging.info(f"  Will skip: {hotkey[:12]}... on {date}")
                if len(existing_hotkey_date_pairs) > 3:
                    bt.logging.info(f"  ... and {len(existing_hotkey_date_pairs) - 3} more pairs")
        else:
            if not args.save_csv:
                bt.logging.error("No database URL available and --save-csv not specified")
                bt.logging.error("Either provide --database-url or use --save-csv flag")
                return
            bt.logging.warning("No database URL provided, will save to CSV only")
    except Exception as e:
        bt.logging.error(f"Failed to initialize database: {e}")
        if not args.save_csv:
            bt.logging.error("Database initialization failed and --save-csv not specified")
            bt.logging.error("Use --save-csv flag to save results to file instead")
            return
        bt.logging.warning("Database failed, continuing with CSV output")

    # Initialize live price fetcher
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
    
    # Initialize SharedDataManager for efficient price caching in regular mode
    # This is needed for the regular flow (non-auto-backfill) to cache prices across days
    bt.logging.info("Initializing shared data manager for price caching...")
    shared_data_manager = SharedDataManager(database_url if db_manager else None, hotkeys=hotkeys)
    shared_data_manager.live_price_fetcher = live_price_fetcher
    shared_data_manager.all_positions = all_positions
    shared_data_manager.elimination_tracker = elimination_tracker
    
    # Store in args for use in get_cached_or_fetch_price_sources
    args.shared_data_manager = shared_data_manager

    # Results storage (only used if CSV output is requested)
    results = [] if args.save_csv else None
    
    # Calculate total number of days to process
    total_days = int((end_ms - start_ms) // MS_IN_24_HOURS) + 1
    
    # Step through each day
    current_ms = end_ms
    day_counter = 0
    while current_ms >= start_ms:
        day_start_time = time.time()
        
        day_counter += 1
        current_date = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc)
        current_date_utc_str = current_date.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Calculate date string ONCE at the start of each day's processing
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        bt.logging.info(f"=== Processing date: {current_date_str} ({current_date_utc_str}) [{day_counter}/{total_days}] ===")
        
        
        try:
            # Step 1: Filter positions and identify required trade pairs using date string
            filtered_positions_by_hotkey, required_trade_pair_ids, skip_stats = PositionFilterAnalyzer.filter_and_analyze_positions_for_date(
                all_positions, current_ms, current_date_str, live_price_fetcher, elimination_tracker
            )

            
            if not filtered_positions_by_hotkey:
                day_elapsed_time = time.time() - day_start_time
                bt.logging.info(f"No valid positions remaining after filtering on {current_date_str} UTC, skipping... (took {day_elapsed_time:.2f}s)")
                current_ms -= MS_IN_24_HOURS
                continue

            bt.logging.info(f"Found {len(filtered_positions_by_hotkey)} active miners with {sum(len(positions) for positions in filtered_positions_by_hotkey.values())} valid positions on {current_date_str} UTC")

            # Early duplicate check - skip price fetching if all miners already have data for this date
            if existing_hotkey_date_pairs:
                # Check which miners would be skipped
                miners_to_skip = set()
                for miner_hotkey in filtered_positions_by_hotkey.keys():
                    if (miner_hotkey, current_date_str) in existing_hotkey_date_pairs:
                        miners_to_skip.add(miner_hotkey)

                if miners_to_skip:
                    bt.logging.info(f"⏭️  Will skip {len(miners_to_skip)} miners that already have data for {current_date_str}")

                    # If ALL miners would be skipped, skip the entire day
                    if len(miners_to_skip) == len(filtered_positions_by_hotkey):
                        bt.logging.info(f"All {len(filtered_positions_by_hotkey)} miners already have data for {current_date_str}, skipping entire day...")
                        current_ms -= MS_IN_24_HOURS
                        continue

                    # Remove miners that already have data from our working set
                    for miner_to_skip in miners_to_skip:
                        del filtered_positions_by_hotkey[miner_to_skip]

                    # Recalculate required trade pairs after removing miners
                    remaining_positions = []
                    for positions in filtered_positions_by_hotkey.values():
                        remaining_positions.extend(positions)

                    required_trade_pair_ids = set()
                    for position in remaining_positions:
                        if position.is_closed_position and position.close_ms <= current_ms:
                            continue
                        required_trade_pair_ids.add(position.trade_pair.trade_pair_id)

                    bt.logging.info(f"After skipping duplicates: {len(filtered_positions_by_hotkey)} miners with {len(remaining_positions)} positions remain")

            # Step 2: Get prices using smart caching strategy
            if not required_trade_pair_ids:
                bt.logging.info(f"No open positions needing prices on {current_date_str} UTC, using closed position returns...")
                cached_price_sources = {}  # Empty dict but not None
            else:
                bt.logging.info(f"Need prices for {len(required_trade_pair_ids)} trade pairs on {current_date_str}")

                # Use smart caching strategy with date string
                cached_price_sources = get_cached_or_fetch_price_sources(
                    args.shared_data_manager,
                    current_date_str,  # Pass the pre-calculated date string
                    current_ms,
                    required_trade_pair_ids
                )

                # Log cache performance periodically
                if day_counter % 10 == 0:
                    stats = args.shared_data_manager.get_cache_statistics()
                    bt.logging.info(f"📊 Price cache performance: {stats['hit_rate']:.1f}% hit rate, "
                                   f"{stats['total_price_sources_cached']} cached prices")

                bt.logging.info(f"✓ Successfully got prices for {current_date_str} UTC")

            # Step 3: Calculate returns for each miner using categorized approach
            # Flatten all positions for category-based calculation
            all_filtered_positions = []
            for positions in filtered_positions_by_hotkey.values():
                all_filtered_positions.extend(positions)
            
            # Calculate returns by category
            miner_category_returns = CategoryReturnCalculator.calculate_miner_returns_by_category(
                all_filtered_positions, current_ms, cached_price_sources, live_price_fetcher
            )
            
            # Prepare insert values for miner_port_values table
            daily_returns = CategoryReturnCalculator.prepare_insert_values(
                miner_category_returns, current_date
            )
            
            # Note: Duplicate checking already done before price fetching - this is just a safety check
            if existing_hotkey_date_pairs:
                original_count = len(daily_returns)
                daily_returns = [
                    dr for dr in daily_returns
                    if (dr['miner_hotkey'], dr['date']) not in existing_hotkey_date_pairs
                ]
                skipped_count = original_count - len(daily_returns)
                if skipped_count > 0:
                    bt.logging.warning(f"⚠️  Found {skipped_count} unexpected duplicates that should have been filtered earlier")
                if not daily_returns:
                    bt.logging.warning(f"All {original_count} miners unexpectedly had duplicates, this shouldn't happen")
                    current_ms -= MS_IN_24_HOURS
                    continue

            
            # Save results based on configuration
            if db_manager and daily_returns:
                # Primary: Insert into database
                # Use skip_duplicates=True when fine-grained skipping is enabled for extra safety
                success = db_manager.insert_daily_returns(
                    daily_returns, 
                    current_date,
                    skip_duplicates=True
                )
                if success:
                    bt.logging.info(f"💾 Successfully saved {len(daily_returns)} returns to database for {current_date_str}")
                else:
                    bt.logging.error(f"Failed to save returns to database for {current_date_str}")
                    raise RuntimeError(f"Database insertion failed for {current_date_str}")


            else:
                # No valid output method
                bt.logging.error(f"No valid output method configured for {current_date_str}")
                raise RuntimeError("No database connection and CSV output not enabled")

            # Log day processing time
            day_elapsed_time = time.time() - day_start_time
            bt.logging.info(f"✓ Completed processing {current_date_str} UTC (took {day_elapsed_time:.2f}s)\n")
            
        except Exception as e:
            day_elapsed_time = time.time() - day_start_time
            bt.logging.error(f"✗ CRITICAL ERROR: Failed to process date {current_date_str} UTC: {e} (took {day_elapsed_time:.2f}s)")
            bt.logging.error("Stopping script execution due to date processing failure")
            raise RuntimeError(f"Date processing failed for {current_date_str}: {e}") from e
        
        # Move to next day
        current_ms -= MS_IN_24_HOURS
    
    # Handle CSV output if requested
    if args.save_csv and results:
        output_file = args.output or "daily_portfolio_returns.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        if db_manager:
            bt.logging.info(f"📄 CSV output also saved to {output_file} ({len(results)} records)")
        else:
            bt.logging.info(f"📄 Results saved to {output_file} ({len(results)} records)")

    elif db_manager:
        bt.logging.info("✅ All daily returns successfully saved to database")
    else:
        bt.logging.warning("No output method was used")
    
    # Final summary
    if db_manager:
        processed_days = day_counter - len(existing_dates)
        bt.logging.info("=" * 80)
        bt.logging.info("📊 DATABASE INSERTION SUMMARY")
        bt.logging.info("=" * 80)
        bt.logging.info(f"📅 Total days in range: {total_days}")
        bt.logging.info(f"⏭️  Skipped existing: {len(existing_dates)}")
        bt.logging.info(f"✅ Processed new days: {processed_days}")
        bt.logging.info(f"💾 Database: daily_portfolio_returns table updated")
        bt.logging.info("=" * 80)




if __name__ == "__main__":
    # Prevent duplicate logging setup
    if not hasattr(bt.logging._logger, '_handlers_configured'):
        bt.logging.enable_info()
        bt.logging._logger._handlers_configured = True
    
    # Suppress noisy urllib3 warnings
    import logging
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
    
    main()