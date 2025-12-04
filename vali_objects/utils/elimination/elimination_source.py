# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import os
import asyncio
import json
from enum import Enum
from typing import Dict, List, Optional, Any
from collections import defaultdict
import bittensor as bt
import traceback
from time_util.time_util import TimeUtil


class EliminationSource(Enum):
    """Enumeration of available elimination sources."""
    DISK = "disk"
    DATABASE = "database"
    TEST = "test"


class EliminationSourceManager:
    """
    Centralized manager for loading eliminations from various sources.
    
    This class provides a unified interface for loading eliminations from:
    - Disk (cached eliminations from local files)
    - Database (via taoshi.ts.ptnhs.eliminationsdatabase)
    - Test data (hardcoded test eliminations)
    """
    
    def __init__(self, source_type: EliminationSource = EliminationSource.DISK):
        """
        Initialize the elimination source manager.
        
        Args:
            source_type: The type of elimination source to use
        """
        self.source_type = source_type

    def load_eliminations(self, 
                         start_time_ms: Optional[int] = None,
                         end_time_ms: Optional[int] = None,
                         hotkeys: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load eliminations based on the configured source.
        
        Args:
            start_time_ms: Start time in milliseconds (for database queries)
            end_time_ms: End time in milliseconds (for database queries)
            hotkeys: List of hotkeys to filter by (None for all)
            
        Returns:
            Dictionary mapping hotkeys to their elimination records
        """
        bt.logging.info(f"Loading eliminations from source: {self.source_type.value}")
        
        if self.source_type == EliminationSource.DATABASE:
            return asyncio.run(self._load_from_database_async(start_time_ms, end_time_ms, hotkeys))
        elif self.source_type == EliminationSource.TEST:
            return self._load_test_eliminations()
        else:  # DISK
            # For disk-based loading, return empty dict as eliminations are loaded
            # through existing elimination manager mechanisms
            bt.logging.info("Disk source selected - eliminations will be loaded via existing elimination manager")
            return {}
            
    async def _load_from_database_async(self, 
                                       start_time_ms: Optional[int], 
                                       end_time_ms: Optional[int], 
                                       hotkeys: Optional[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load eliminations from database using taoshi.ts.ptnhs.eliminationsdatabase.
        
        Args:
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds  
            hotkeys: List of hotkeys to filter by
            
        Returns:
            Dictionary mapping hotkeys to their elimination records
        """
        bt.logging.info(f"Loading eliminations from database for period "
                       f"{TimeUtil.millis_to_formatted_date_str(start_time_ms) if start_time_ms else 'beginning'} to "
                       f"{TimeUtil.millis_to_formatted_date_str(end_time_ms) if end_time_ms else 'now'}")
        
        if hotkeys:
            bt.logging.info(f"Filtering for {len(hotkeys)} specific hotkeys")

        # Import taoshi.ts.ptnhs.eliminationsdatabase locally to avoid circular import
        try:
            os.environ["TAOSHI_TS_DEPLOYMENT"] = "DEVELOPMENT"
            os.environ["TAOSHI_TS_PLATFORM"] = "LOCAL"
            from taoshi.ts import ptn as ptn_utils
            from taoshi.ts.ptnhs.eliminationsdatabase import EliminationsDatabase
        except ImportError as e:
            raise Exception(f"Failed to import taoshi.ts.ptnhs.eliminationsdatabase: {e}")

        # Initialize database elimination source
        eliminations_db = EliminationsDatabase()

        # Query all elimination records directly from the database instead of using the external API
        # which only returns 1 record instead of the expected ~300
        from sqlalchemy.orm import Session
        from taoshi.ts.model import EliminationModel

        session = Session(eliminations_db.ptn_db_editor)

        try:
            # Query all elimination records from the database
            elimination_records = session.query(EliminationModel).all()
            bt.logging.info(f"Retrieved {len(elimination_records)} elimination records from database")

            # Convert SQLAlchemy objects to dictionaries
            api_data = []
            for elim in elimination_records:
                api_data.append({
                    'miner_hotkey': elim.miner_hotkey,
                    'max_drawdown': elim.max_drawdown,
                    'elimination_time_ms': elim.elimination_ms,  # Map elimination_ms to elimination_time_ms for consistency
                    'elimination_ms': elim.elimination_ms,
                    'elimination_reason': elim.elimination_reason,
                    'creation_ms': elim.creation_ms,
                    'updated_ms': elim.updated_ms,
                    'hotkey': elim.miner_hotkey  # Add hotkey alias for compatibility
                })

        finally:
            session.close()

        # Apply local filtering since database API doesn't support parameters
        filtered_api_data = api_data

        # Filter by time range if specified
        if start_time_ms is not None or end_time_ms is not None:
            original_count = len(filtered_api_data)
            filtered_api_data = []

            for elimination_record in api_data:
                elim_time = elimination_record.get('elimination_time_ms', 0)

                # Apply time range filters
                if start_time_ms is not None and elim_time < start_time_ms:
                    continue
                if end_time_ms is not None and elim_time > end_time_ms:
                    continue

                filtered_api_data.append(elimination_record)

            bt.logging.info(f"Time filtering: {len(filtered_api_data)}/{original_count} records within time range")

        # Filter by hotkeys if specified
        if hotkeys:
            original_count = len(filtered_api_data)
            hotkey_set = set(hotkeys)
            filtered_api_data = [
                record for record in filtered_api_data
                if (record.get('miner_hotkey') in hotkey_set or record.get('hotkey') in hotkey_set)
            ]
            bt.logging.info(f"Hotkey filtering: {len(filtered_api_data)}/{original_count} records match target hotkeys")

        # Group eliminations by hotkey
        hk_to_eliminations = defaultdict(list)

        for elimination_record in filtered_api_data:
            # Extract hotkey from elimination record
            hotkey = elimination_record.get('miner_hotkey') or elimination_record.get('hotkey')
            if hotkey:
                hk_to_eliminations[hotkey].append(elimination_record)
            else:
                bt.logging.warning(f"Elimination record missing hotkey: {elimination_record}")

        bt.logging.info(f"Final result: {len(filtered_api_data)} elimination records for {len(hk_to_eliminations)} miners")
        return dict(hk_to_eliminations)

            
    def _load_test_eliminations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load test eliminations from hardcoded test data.
        
        Returns:
            Dictionary mapping hotkeys to their elimination records
        """
        try:
            from tests.test_data.backtest_test_eliminations import get_test_eliminations
        except ImportError as e:
            bt.logging.error(f"Failed to import test eliminations: {e}")
            # Create some basic test data if import fails
            return self._create_basic_test_eliminations()
            
        bt.logging.info("Loading test eliminations")
        
        test_eliminations = get_test_eliminations()
        hk_to_eliminations = defaultdict(list)
        
        # Calculate time range from test data for logging
        if test_eliminations:
            timestamps = [elim.get('elimination_time_ms', 0) for elim in test_eliminations]
            if timestamps:
                start_time_ms = min(timestamps)
                end_time_ms = max(timestamps)
                bt.logging.info(f"Test elimination data time range: "
                               f"{TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                               f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}")
        
        for elimination_record in test_eliminations:
            hotkey = elimination_record.get('miner_hotkey') or elimination_record.get('hotkey')
            if hotkey:
                hk_to_eliminations[hotkey].append(elimination_record)
            else:
                bt.logging.warning(f"Test elimination record missing hotkey: {elimination_record}")
                
        bt.logging.info(f"Loaded {sum(len(eliminations) for eliminations in hk_to_eliminations.values())} "
                       f"test elimination records for {len(hk_to_eliminations)} miners")
        
        return dict(hk_to_eliminations)
    
    def _create_basic_test_eliminations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create basic test elimination data when test files are not available.
        
        Returns:
            Dictionary with sample elimination data
        """
        bt.logging.info("Creating basic test elimination data")
        
        current_time_ms = TimeUtil.now_in_millis()
        one_day_ms = 24 * 60 * 60 * 1000
        
        test_eliminations = {
            "test_miner_1": [
                {
                    "miner_hotkey": "test_miner_1",
                    "elimination_time_ms": current_time_ms - (7 * one_day_ms),
                    "elimination_reason": "max_drawdown",
                    "elimination_type": "drawdown",
                    "drawdown_percentage": 12.5,
                    "challenge_period_ms": 30 * one_day_ms
                }
            ],
            "test_miner_2": [
                {
                    "miner_hotkey": "test_miner_2", 
                    "elimination_time_ms": current_time_ms - (14 * one_day_ms),
                    "elimination_reason": "plagiarism",
                    "elimination_type": "plagiarism",
                    "similarity_score": 0.95,
                    "challenge_period_ms": 30 * one_day_ms
                }
            ]
        }
        
        bt.logging.info(f"Created {sum(len(eliminations) for eliminations in test_eliminations.values())} "
                       f"basic test elimination records for {len(test_eliminations)} miners")
        
        return test_eliminations
    
    def save_to_elimination_manager(self, elimination_manager, hk_to_eliminations: Dict[str, List[Dict[str, Any]]]):
        """
        Helper method to save loaded eliminations to an elimination manager.
        
        Args:
            elimination_manager: The elimination manager instance
            hk_to_eliminations: Dictionary mapping hotkeys to elimination records
        """
        if not hk_to_eliminations:
            bt.logging.warning("No eliminations to save to elimination manager")
            return
            
        elimination_count = 0
        for hk, eliminations in hk_to_eliminations.items():
            for elimination_record in eliminations:
                # The exact method depends on the elimination manager's interface
                # This is a placeholder - adjust based on actual elimination manager API
                if hasattr(elimination_manager, 'add_elimination_record'):
                    elimination_manager.add_elimination_record(hk, elimination_record)
                    elimination_count += 1
                elif hasattr(elimination_manager, 'save_elimination'):
                    elimination_manager.save_elimination(elimination_record)
                    elimination_count += 1
                else:
                    bt.logging.warning(f"Elimination manager does not have expected methods for saving eliminations")
                    break
        
        bt.logging.info(f"Saved {elimination_count} elimination records for {len(hk_to_eliminations)} miners to elimination manager")

    def get_eliminations_summary(self, hk_to_eliminations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate a summary of loaded eliminations.
        
        Args:
            hk_to_eliminations: Dictionary mapping hotkeys to elimination records
            
        Returns:
            Dictionary with elimination summary statistics
        """
        if not hk_to_eliminations:
            return {"total_miners": 0, "total_eliminations": 0}
        
        total_eliminations = sum(len(eliminations) for eliminations in hk_to_eliminations.values())
        elimination_reasons = defaultdict(int)
        elimination_types = defaultdict(int)
        timestamps = []
        
        for hotkey, eliminations in hk_to_eliminations.items():
            for elimination in eliminations:
                reason = elimination.get('elimination_reason', 'unknown')
                elim_type = elimination.get('elimination_type', 'unknown')
                timestamp = elimination.get('elimination_time_ms', 0)
                
                elimination_reasons[reason] += 1
                elimination_types[elim_type] += 1
                if timestamp > 0:
                    timestamps.append(timestamp)
        
        summary = {
            "total_miners": len(hk_to_eliminations),
            "total_eliminations": total_eliminations,
            "elimination_reasons": dict(elimination_reasons),
            "elimination_types": dict(elimination_types),
        }
        
        if timestamps:
            summary["time_range"] = {
                "start_ms": min(timestamps),
                "end_ms": max(timestamps),
                "start_date": TimeUtil.millis_to_formatted_date_str(min(timestamps)),
                "end_date": TimeUtil.millis_to_formatted_date_str(max(timestamps))
            }
        
        return summary