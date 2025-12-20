"""
MinerAccountManager - Manages per-miner account state and account size tracking.

This manager is the source of truth for miner account state including:
- Account size (via CollateralRecord tracking)
- Cash balance (for future equities margin)
- Disk persistence of account sizes

This module contains ALL account size functionality, previously split across
ValidatorContractManager. The contract manager now delegates to this module.
"""
import threading
from dataclasses import dataclass
from datetime import timezone, datetime, timedelta
from typing import Dict, Optional, List, Any
import bittensor as bt
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


# ==================== Data Classes ====================


class CollateralRecord:
    """Record of a collateral/account size update at a specific timestamp."""

    def __init__(self, account_size: float, account_size_theta: float, update_time_ms: int):
        self.account_size = account_size
        self.account_size_theta = account_size_theta
        self.update_time_ms = update_time_ms
        self.valid_date_timestamp = CollateralRecord.valid_from_ms(update_time_ms)

    @staticmethod
    def valid_from_ms(update_time_ms: int) -> int:
        """Returns timestamp of start of next day (00:00:00 UTC) when this record is valid"""
        dt = datetime.fromtimestamp(update_time_ms / 1000, tz=timezone.utc)
        start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Record is valid from the start of the next day
        start_of_next_day = start_of_day + timedelta(days=1)
        return int(start_of_next_day.timestamp() * 1000)

    @property
    def valid_date_str(self) -> str:
        """Returns YYYY-MM-DD format for easy reading"""
        return TimeUtil.millis_to_short_date_str(self.valid_date_timestamp)

    def __repr__(self):
        """String representation"""
        return str(vars(self))


@dataclass
class MinerAccount:
    """Per-miner account state. Source of truth for account_size."""
    miner_hotkey: str
    account_size: float              # From CollateralRecord (updated when chain updates)
    cash_balance: float              # Available cash (for future equities margin)

    def update_account_size(self, new_size: float):
        """Called when CollateralRecord updates from chain."""
        delta = new_size - self.account_size
        self.account_size = new_size
        self.cash_balance += delta  # Adjust cash by the delta


# ==================== Manager Implementation ====================


class MinerAccountManager:
    """
    Manages all miner accounts and account size tracking.

    This is the source of truth for:
    - Account sizes (via CollateralRecord history)
    - Cash balances (for equities margin)
    - Disk persistence of account data

    The ValidatorContractManager delegates all account size operations here.
    """

    def __init__(self, running_unit_tests: bool = False, collateral_balance_getter=None):
        """
        Initialize the manager.

        Args:
            running_unit_tests: Whether running in test mode
            collateral_balance_getter: Callable to get collateral balance for a hotkey.
                                       Signature: (hotkey: str) -> Optional[float]
                                       Returns balance in theta tokens, or None.
        """
        self.running_unit_tests = running_unit_tests
        self._collateral_balance_getter = collateral_balance_getter

        # MinerAccount cache for dynamic account state (cash balance, etc.)
        self.accounts: Dict[str, MinerAccount] = {}

        # Account size history (CollateralRecords)
        self.miner_account_sizes: Dict[str, List[CollateralRecord]] = {}

        # Locking strategy - EAGER initialization (not lazy!)
        # RLock allows same thread to acquire lock multiple times (needed for nested calls)
        self._account_sizes_lock = threading.RLock()
        # Lock for disk I/O serialization to prevent concurrent file writes
        self._disk_lock = threading.Lock()

        # Initialize miner account sizes file location
        self.MINER_ACCOUNT_SIZES_FILE = ValiBkpUtils.get_miner_account_sizes_file_location(
            running_unit_tests=running_unit_tests
        )

        # Load from disk
        self._load_miner_account_sizes_from_disk()

    def set_collateral_balance_getter(self, getter):
        """Set the collateral balance getter (for lazy initialization)."""
        self._collateral_balance_getter = getter

    # ==================== Disk Persistence ====================

    def _load_miner_account_sizes_from_disk(self):
        """Load miner account sizes from disk during initialization - protected by locks"""
        with self._disk_lock:
            try:
                disk_data = ValiUtils.get_vali_json_file_dict(self.MINER_ACCOUNT_SIZES_FILE)
                parsed_data = self._parse_miner_account_sizes_dict(disk_data)

                # Acquire account_sizes_lock to update the dict
                with self._account_sizes_lock:
                    self.miner_account_sizes.clear()
                    self.miner_account_sizes.update(parsed_data)

                bt.logging.info(f"Loaded {len(self.miner_account_sizes)} miner account size records from disk")
            except Exception as e:
                bt.logging.warning(f"Failed to load miner account sizes from disk: {e}")

    def re_init_account_sizes(self):
        """Public method to reload account sizes from disk (useful for tests)"""
        self._load_miner_account_sizes_from_disk()

    def _save_miner_account_sizes_to_disk(self):
        """Save miner account sizes to disk - protected by _disk_lock to prevent concurrent writes"""
        with self._disk_lock:
            try:
                data_dict = self.miner_account_sizes_dict()
                ValiBkpUtils.write_file(self.MINER_ACCOUNT_SIZES_FILE, data_dict)
            except Exception as e:
                bt.logging.error(f"Failed to save miner account sizes to disk: {e}")

    def miner_account_sizes_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync

        Args:
            most_recent_only: If True, only return the most recent record for each miner

        Returns:
            Dictionary with hotkeys as keys and list of record dicts as values
        """
        with self._account_sizes_lock:
            json_dict = {}
            for hotkey, records in self.miner_account_sizes.items():
                if most_recent_only and records:
                    # Only include the most recent (last) record
                    json_dict[hotkey] = [vars(records[-1])]
                else:
                    json_dict[hotkey] = [vars(record) for record in records]
            return json_dict

    @staticmethod
    def _parse_miner_account_sizes_dict(data_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[
        str, List[CollateralRecord]]:
        """Parse miner account sizes from disk format back to CollateralRecord objects"""
        parsed_dict = {}
        for hotkey, records_data in data_dict.items():
            try:
                parsed_records = []
                for record_data in records_data:
                    if isinstance(record_data, dict) and all(
                            key in record_data for key in ["account_size", "update_time_ms"]):
                        record = CollateralRecord(
                            record_data["account_size"],
                            record_data.get("account_size_theta", 0),
                            record_data["update_time_ms"]
                        )
                        parsed_records.append(record)

                if parsed_records:  # Only add if we have valid records
                    parsed_dict[hotkey] = parsed_records
            except Exception as e:
                bt.logging.warning(f"Failed to parse account size records for {hotkey}: {e}")

        return parsed_dict

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]):
        """
        Sync miner account sizes data from external source (backup/sync).
        If empty dict is passed, clears all account sizes (useful for tests).
        """
        try:
            with self._account_sizes_lock:
                if not account_sizes_data:
                    assert self.running_unit_tests, "Empty account sizes data can only be used in test mode"
                    # Empty dict = clear all data (useful for test cleanup)
                    bt.logging.info("Clearing all miner account sizes")
                    self.miner_account_sizes.clear()
                    self._save_miner_account_sizes_to_disk()
                    return

                synced_data = self._parse_miner_account_sizes_dict(account_sizes_data)
                self.miner_account_sizes.clear()
                self.miner_account_sizes.update(synced_data)
                self._save_miner_account_sizes_to_disk()
                bt.logging.info(f"Synced {len(self.miner_account_sizes)} miner account size records")
        except Exception as e:
            bt.logging.error(f"Failed to sync miner account sizes data: {e}")

    # ==================== Account Size Methods ====================

    def set_miner_account_size(self, hotkey: str, collateral_balance_theta: float, timestamp_ms: Optional[int] = None) -> Optional[CollateralRecord]:
        """
        Set the account size for a miner. Saves to memory and disk.
        Records are kept in chronological order.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            collateral_balance_theta: Collateral balance in theta tokens
            timestamp_ms: Timestamp for the record (defaults to now)

        Returns:
            CollateralRecord if successful, None otherwise
        """
        if collateral_balance_theta is None:
            bt.logging.warning(f"Could not set account size for {hotkey}: collateral_balance is None")
            return None

        # CRITICAL SECTION: Acquire lock for timestamp + record creation + append + save
        # Timestamp MUST be generated inside lock to ensure chronological ordering
        with self._account_sizes_lock:
            # Generate timestamp inside lock if not provided
            # This ensures records are added in strictly chronological order
            if timestamp_ms is None:
                timestamp_ms = TimeUtil.now_in_millis()

            account_size = min(ValiConfig.MAX_COLLATERAL_BALANCE_THETA, collateral_balance_theta) * ValiConfig.COST_PER_THETA
            collateral_record = CollateralRecord(account_size, collateral_balance_theta, timestamp_ms)

            # Skip if the new record matches the last existing record
            if hotkey in self.miner_account_sizes and self.miner_account_sizes[hotkey]:
                last_record = self.miner_account_sizes[hotkey][-1]
                if (last_record.account_size == collateral_record.account_size and
                        last_record.account_size_theta == collateral_record.account_size_theta):
                    bt.logging.info(f"Skipping save for {hotkey} - new record matches last record")
                    return collateral_record

            if hotkey not in self.miner_account_sizes:
                self.miner_account_sizes[hotkey] = []

            # Add the new record
            self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [collateral_record]

            # Save to disk (still inside account_sizes_lock, but _save will acquire _disk_lock)
            self._save_miner_account_sizes_to_disk()

            # Update MinerAccount cache if exists
            if hotkey in self.accounts:
                self.accounts[hotkey].update_account_size(account_size)

        bt.logging.info(
            f"Updated account size for {hotkey}: ${account_size:,.2f} (valid from {collateral_record.valid_date_str})")

        return collateral_record

    def get_miner_account_size(self, hotkey: str, timestamp_ms: Optional[int] = None, most_recent: bool = False,
                               records_dict: Optional[dict] = None, use_account_floor: bool = False) -> float | None:
        """
        Get the account size for a miner at a given timestamp. Iterate list in reverse chronological order, and return
        the first record whose valid_date_timestamp <= start_of_day_ms

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp to query for (defaults to now)
            most_recent: If True, return most recent record regardless of timestamp
            records_dict: Optional dict to use instead of self.miner_account_sizes (for cached lookups)
            use_account_floor: If True, return MIN_CAPITAL instead of None when no records exist

        Returns:
            Account size in USD, or None if no applicable records (or MIN_CAPITAL if use_account_floor=True)
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        # Use provided records_dict or default to self.miner_account_sizes
        # If using external dict, assume caller handles locking
        # If using self.miner_account_sizes, acquire lock
        if records_dict is not None:
            source_records = records_dict
            lock_needed = False
        else:
            source_records = self.miner_account_sizes
            lock_needed = True

        def _get_account_size_locked():
            """Inner function with the actual logic"""
            if hotkey not in source_records or not source_records[hotkey]:
                # Use account floor if requested (for miners without collateral records)
                return ValiConfig.MIN_CAPITAL if use_account_floor else None

            # Get start of the requested day
            start_of_day_ms = int(
                datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .timestamp() * 1000
            )

            # Return most recent record
            if most_recent:
                most_recent_record = source_records[hotkey][-1]
                return most_recent_record.account_size

            # Iterate in reversed order, and return the first record that is valid for or before the requested day
            for record in reversed(source_records[hotkey]):
                if record.valid_date_timestamp <= start_of_day_ms:
                    return record.account_size

            # No applicable records found - use account floor if requested
            return ValiConfig.MIN_CAPITAL if use_account_floor else None

        # Execute with or without lock depending on source
        if lock_needed:
            with self._account_sizes_lock:
                return _get_account_size_locked()
        else:
            return _get_account_size_locked()

    def get_all_miner_account_sizes(self, miner_account_sizes: dict[str, List[CollateralRecord]] | None = None, timestamp_ms: Optional[int] = None) -> dict[str, float]:
        """
        Return a dict of all miner account sizes at a timestamp_ms
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        # If external dict provided, use it directly (caller handles locking)
        if miner_account_sizes is not None:
            all_miner_account_sizes = {}
            for hotkey in miner_account_sizes.keys():
                all_miner_account_sizes[hotkey] = self.get_miner_account_size(
                    hotkey, timestamp_ms=timestamp_ms, records_dict=miner_account_sizes
                )
            return all_miner_account_sizes

        # Using self.miner_account_sizes - must prevent race conditions
        # Copy the ENTIRE dict (not just keys) while holding lock to prevent iterator invalidation
        # This prevents sync_miner_account_sizes_data() from clearing the dict while we're reading it
        with self._account_sizes_lock:
            # Deep copy: create new dict with shallow copies of record lists
            # We don't need deep copy of CollateralRecord objects (they're immutable)
            miner_account_sizes_snapshot = {
                hotkey: list(records)  # Shallow copy of list
                for hotkey, records in self.miner_account_sizes.items()
            }

        # Now work with the snapshot (no lock needed - we own this copy)
        all_miner_account_sizes = {}
        for hotkey in miner_account_sizes_snapshot.keys():
            all_miner_account_sizes[hotkey] = self.get_miner_account_size(
                hotkey, timestamp_ms=timestamp_ms, records_dict=miner_account_sizes_snapshot
            )
        return all_miner_account_sizes

    def receive_collateral_record_update(self, collateral_record_data: dict, is_mothership: bool = False) -> bool:
        """
        Process an incoming CollateralRecord synapse and update miner_account_sizes.

        Args:
            collateral_record_data: Dictionary containing hotkey, account_size, update_time_ms, valid_date_timestamp
            is_mothership: Whether this validator is the mothership (should not receive updates)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if is_mothership:
                return False
            with self._account_sizes_lock:
                # Extract data from the synapse
                hotkey = collateral_record_data.get("hotkey")
                account_size = collateral_record_data.get("account_size")
                account_size_theta = collateral_record_data.get("account_size_theta")
                update_time_ms = collateral_record_data.get("update_time_ms")
                bt.logging.info(f"Processing collateral record update for miner {hotkey}")

                if not all([hotkey, account_size is not None, update_time_ms]):
                    bt.logging.warning(f"Invalid collateral record data received: {collateral_record_data}")
                    return False

                # Create a CollateralRecord object
                collateral_record = CollateralRecord(account_size, account_size_theta, update_time_ms)

                # Update miner account sizes
                if hotkey not in self.miner_account_sizes:
                    self.miner_account_sizes[hotkey] = []

                # Check if we already have this record (avoid duplicates)
                if self.get_miner_account_size(hotkey, most_recent=True) == account_size:
                    bt.logging.debug(f"Most recent collateral record for {hotkey} already exists")
                    return True

                # Add the new record
                self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [collateral_record]

                # Save to disk
                self._save_miner_account_sizes_to_disk()

                # Update MinerAccount cache if exists
                if hotkey in self.accounts:
                    self.accounts[hotkey].update_account_size(account_size)

                bt.logging.info(
                    f"Updated miner account size for {hotkey}: ${account_size} (valid from {collateral_record.valid_date_str})")
                return True

        except Exception as e:
            bt.logging.error(f"Error processing collateral record update: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False

    # ==================== MinerAccount Cache Methods ====================

    def get_or_create(self, hotkey: str) -> MinerAccount:
        """Get existing account or create from CollateralRecord."""
        if hotkey not in self.accounts:
            account_size = self.get_miner_account_size(hotkey)
            if account_size is None:
                account_size = ValiConfig.MIN_CAPITAL

            self.accounts[hotkey] = MinerAccount(
                miner_hotkey=hotkey,
                account_size=account_size,
                cash_balance=account_size,
            )
        return self.accounts[hotkey]

    def get_account(self, hotkey: str) -> Optional[MinerAccount]:
        """Get account if it exists, without creating."""
        return self.accounts.get(hotkey)

    def get_all_hotkeys(self) -> list:
        """Get all hotkeys with account size records."""
        with self._account_sizes_lock:
            return list(self.miner_account_sizes.keys())

    def health_check(self) -> dict:
        """Health check for monitoring."""
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "num_account_records": len(self.miner_account_sizes),
            "num_cached_accounts": len(self.accounts)
        }

    # ==================== Static Methods ====================

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """
        Penalize miners who do not reach the min collateral
        """
        if collateral >= ValiConfig.MIN_COLLATERAL_VALUE:
            return 1
        return 0.01
