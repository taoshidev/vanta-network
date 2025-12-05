# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
import os
from enum import Enum
from typing import Dict, List, Optional
from collections import defaultdict
import bittensor as bt
import traceback
from vali_objects.vali_dataclasses.position import Position
from time_util.time_util import TimeUtil


class PositionSource(Enum):
    """Enumeration of available position sources."""
    DISK = "disk"
    DATABASE = "database"
    TEST = "test"


class PositionSourceManager:
    """
    Centralized manager for loading positions from various sources.
    
    This class provides a unified interface for loading positions from:
    - Disk (cached positions from local files)
    - Database (via taoshi.ts.ptn)
    - Test data (hardcoded test positions)
    """
    
    def __init__(self, source_type: PositionSource = PositionSource.DISK):
        """
        Initialize the position source manager.
        
        Args:
            source_type: The type of position source to use
        """
        self.source_type = source_type

    def load_positions(self, 
                      start_time_ms: Optional[int] = None,
                      end_time_ms: Optional[int] = None,
                      hotkeys: Optional[List[str]] = None) -> Dict[str, List[Position]]:
        """
        Load positions based on the configured source.
        
        Args:
            start_time_ms: Start time in milliseconds (for database queries)
            end_time_ms: End time in milliseconds (for database queries)
            hotkeys: List of hotkeys to filter by (None for all)
            
        Returns:
            Dictionary mapping hotkeys to their Position objects
        """
        bt.logging.info(f"Loading positions from source: {self.source_type.value}")
        
        if self.source_type == PositionSource.DATABASE:
            ans = self._load_from_database(start_time_ms, end_time_ms, hotkeys)
        elif self.source_type == PositionSource.TEST:
            ans = self._load_test_positions()
        else:  # DISK
            # For disk-based loading, return empty dict as positions are loaded
            # through existing PositionManager/PerfLedgerManager mechanisms
            bt.logging.info("Disk source selected - positions will be loaded via PositionManager")
            ans = {}
        return ans
            
    def _load_from_database(self, 
                           start_time_ms: Optional[int], 
                           end_time_ms: Optional[int], 
                           hotkeys: Optional[List[str]]) -> Dict[str, List[Position]]:
        """
        Load positions from database using taoshi.ts.ptn.
        
        Args:
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds  
            hotkeys: List of hotkeys to filter by
            
        Returns:
            Dictionary mapping hotkeys to their Position objects
        """
        bt.logging.info(f"Loading positions from database for period "
                       f"{TimeUtil.millis_to_formatted_date_str(start_time_ms) if start_time_ms else 'beginning'} to "
                       f"{TimeUtil.millis_to_formatted_date_str(end_time_ms) if end_time_ms else 'now'}")
        
        if hotkeys:
            bt.logging.info(f"Filtering for {len(hotkeys)} specific hotkeys")

        # Import taoshi.ts.ptn locally to avoid circular import
        try:
            os.environ["TAOSHI_TS_DEPLOYMENT"] = "DEVELOPMENT"
            os.environ["TAOSHI_TS_PLATFORM"] = "LOCAL"
            from taoshi.ts.ptn import wiring
            from taoshi.ts import ptn as ptn_utils
        except ImportError as e:
            bt.logging.error(f"Failed to import taoshi.ts.ptn: {e}")
            traceback.print_exc()
            return {}

        # Initialize database position source
        miner_db = ptn_utils.DatabasePositionOrderSource()

        # Get positions from database
        filtered_positions = miner_db.get_positions_with_orders(
            start_ms=start_time_ms if start_time_ms else 0,
            end_ms=end_time_ms if end_time_ms else TimeUtil.now_in_millis(),
            miner_hotkeys=hotkeys if hotkeys else []
        )

        bt.logging.info(f"Retrieved {len(filtered_positions)}/{sum(len(v) for _, v in filtered_positions.items())} hotkeys/positions from database")


        return filtered_positions

            
    def _load_test_positions(self) -> Dict[str, List[Position]]:
        """
        Load test positions from hardcoded test data.
        
        Returns:
            Dictionary mapping hotkeys to their Position objects
        """
        try:
            from tests.test_data.backtest_test_positions import get_test_positions
        except ImportError as e:
            bt.logging.error(f"Failed to import test positions: {e}")
            raise
            
        bt.logging.info("Loading test positions")
        
        test_positions = get_test_positions()
        hk_to_positions = defaultdict(list)
        
        # Calculate time range from test data for logging
        if test_positions:
            start_time_ms = min(min(o['processed_ms'] for o in pos['orders']) for pos in test_positions)
            end_time_ms = max(max(o['processed_ms'] for o in pos['orders']) for pos in test_positions)
            bt.logging.info(f"Test data time range: "
                           f"{TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                           f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}")
        
        for pos_data in test_positions:
            try:
                position_obj = Position(**pos_data)
                hk_to_positions[position_obj.miner_hotkey].append(position_obj)
            except Exception as e:
                bt.logging.error(f"Failed to create Position object from test data: {e}")
                continue
                
        bt.logging.info(f"Loaded {sum(len(positions) for positions in hk_to_positions.values())} "
                       f"test positions for {len(hk_to_positions)} miners")
        
        return dict(hk_to_positions)
    
    def save_to_position_manager(self, position_manager, hk_to_positions: Dict[str, List[Position]]):
        """
        Helper method to save loaded positions to a position manager.
        
        Args:
            position_manager: The position manager instance
            hk_to_positions: Dictionary mapping hotkeys to Position objects
        """
        if not hk_to_positions:
            bt.logging.warning("No positions to save to position manager")
            return
            
        position_count = 0
        for hk, positions in hk_to_positions.items():
            for position in positions:
                position_manager.save_miner_position(position)
                position_count += 1
        
        bt.logging.info(f"Saved {position_count} positions for {len(hk_to_positions)} miners to position manager")