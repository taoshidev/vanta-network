"""
MinerAccountManager - Manages per-miner account state and account size tracking.

This manager is the source of truth for miner account state including:
- Account size (via CollateralRecord tracking)
- Cash balance (for equities margin)
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
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.exceptions.signal_exception import SignalException


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
    """Per-miner account state. Unified source of truth for account data."""
    miner_hotkey: str
    cash_balance: float              # Available cash (for equities margin)
    total_borrowed_amount: float = 0.0  # Total margin loans outstanding
    collateral_records: List[CollateralRecord] = None  # Historical CollateralRecords (List[CollateralRecord])

    def __post_init__(self):
        if self.collateral_records is None:
            self.collateral_records = []

    def add_collateral_record(self, record: 'CollateralRecord'):
        """Add a new collateral record and update account_size."""
        previous_size = self.get_account_size()
        new_size = record.account_size
        self.collateral_records.append(record)

        if previous_size:
            size_increase = new_size - previous_size
            self.cash_balance += size_increase

    def get_account_size(self, timestamp_ms: Optional[int] = None) -> Optional[float]:
        if not self.collateral_records:
            return None

        if timestamp_ms is None:
            return self.collateral_records[-1].account_size

        # Get start of the requested day
        start_of_day_ms = int(
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )

        # Iterate in reversed order, return first record valid for or before the requested day
        for record in reversed(self.collateral_records):
            if record.valid_date_timestamp <= start_of_day_ms:
                return record.account_size

        return None


# ==================== Manager Implementation ====================


class MinerAccountManager:
    """
    Manages all miner accounts and account size tracking.

    This is the unified source of truth for:
    - Account sizes (via CollateralRecord history in MinerAccount)
    - Cash balances (for equities margin)
    - Margin loans (total_borrowed_amount)
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

        # Unified MinerAccount storage - single source of truth
        self.accounts: Dict[str, MinerAccount] = {}

        # Locking strategy - EAGER initialization (not lazy!)
        # RLock allows same thread to acquire lock multiple times (needed for nested calls)
        self._accounts_lock = threading.RLock()
        # Lock for disk I/O serialization to prevent concurrent file writes
        self._disk_lock = threading.Lock()

        # Initialize miner accounts file location
        self.MINER_ACCOUNTS_FILE = ValiBkpUtils.get_miner_account_sizes_file_location(
            running_unit_tests=running_unit_tests
        )

        # Load from disk
        self._load_accounts_from_disk()

    def set_collateral_balance_getter(self, getter):
        """Set the collateral balance getter (for lazy initialization)."""
        self._collateral_balance_getter = getter

    # ==================== Disk Persistence ====================

    def _load_accounts_from_disk(self):
        """Load miner accounts from disk during initialization - protected by locks"""
        with self._disk_lock:
            try:
                disk_data = ValiUtils.get_vali_json_file_dict(self.MINER_ACCOUNTS_FILE)
                parsed_accounts = self._parse_accounts_dict(disk_data)

                with self._accounts_lock:
                    self.accounts.clear()
                    self.accounts.update(parsed_accounts)

                bt.logging.info(f"Loaded {len(self.accounts)} miner accounts from disk")
            except Exception as e:
                bt.logging.warning(f"Failed to load miner accounts from disk: {e}")

    def re_init_account_sizes(self):
        """Public method to reload accounts from disk (useful for tests)"""
        self._load_accounts_from_disk()

    def _save_accounts_to_disk(self):
        """Save miner accounts to disk - protected by _disk_lock to prevent concurrent writes"""
        with self._disk_lock:
            try:
                data_dict = self.accounts_dict()
                ValiBkpUtils.write_file(self.MINER_ACCOUNTS_FILE, data_dict)
            except Exception as e:
                bt.logging.error(f"Failed to save miner accounts to disk: {e}")

    def accounts_dict(self, most_recent_only: bool = False) -> Dict[str, Any]:
        """Convert miner accounts to checkpoint format for backup/sync

        Args:
            most_recent_only: If True, only return the most recent collateral record for each miner

        Returns:
            Dictionary with hotkeys as keys and account data as values
        """
        with self._accounts_lock:
            json_dict = {}
            for hotkey, account in self.accounts.items():
                if most_recent_only and account.collateral_records:
                    records = [vars(account.collateral_records[-1])]
                else:
                    records = [vars(record) for record in account.collateral_records]

                json_dict[hotkey] = {
                    "collateral_records": records,
                    "cash_balance": account.cash_balance,
                    "total_borrowed_amount": account.total_borrowed_amount
                }
            return json_dict

    # Backwards compatibility alias
    def miner_account_sizes_dict(self, most_recent_only: bool = False) -> Dict[str, Any]:
        """Backwards compatible method - converts to old format (hotkey -> list of CollateralRecords)"""
        with self._accounts_lock:
            json_dict = {}
            for hotkey, account in self.accounts.items():
                if most_recent_only and account.collateral_records:
                    json_dict[hotkey] = [vars(account.collateral_records[-1])]
                else:
                    json_dict[hotkey] = [vars(record) for record in account.collateral_records]
            return json_dict

    @staticmethod
    def _parse_accounts_dict(data_dict: Dict[str, Any]) -> Dict[str, MinerAccount]:
        """Parse miner accounts from disk format back to MinerAccount objects.

        Supports:
        - Legacy format: {"hotkey": [list of CollateralRecord dicts]}
        - New format: {"hotkey": {"collateral_records": [...], "cash_balance": ..., "total_borrowed_amount": ...}}
        """
        parsed_accounts = {}

        for hotkey, account_data in data_dict.items():
            try:
                collateral_records = []

                # Determine format and extract records
                if isinstance(account_data, dict) and "collateral_records" in account_data:
                    records_list = account_data.get("collateral_records", [])
                    cash_balance = account_data.get("cash_balance")
                    total_borrowed = account_data.get("total_borrowed_amount", 0.0)
                elif isinstance(account_data, list):
                    records_list = account_data
                    cash_balance = None  # Will default to account_size
                    total_borrowed = 0.0
                else:
                    continue

                # Parse collateral records
                for record_data in records_list:
                    if isinstance(record_data, dict) and "account_size" in record_data and "update_time_ms" in record_data:
                        record = CollateralRecord(
                            record_data["account_size"],
                            record_data.get("account_size_theta", 0),
                            record_data["update_time_ms"]
                        )
                        collateral_records.append(record)

                if collateral_records:
                    account_size = collateral_records[-1].account_size
                    parsed_accounts[hotkey] = MinerAccount(
                        miner_hotkey=hotkey,
                        account_size=account_size,
                        cash_balance=cash_balance if cash_balance is not None else account_size,
                        total_borrowed_amount=total_borrowed,
                        collateral_records=collateral_records
                    )
            except Exception as e:
                bt.logging.warning(f"Failed to parse account for {hotkey}: {e}")

        return parsed_accounts

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, Any]):
        """
        Sync miner account sizes data from external source (backup/sync).
        If empty dict is passed, clears all accounts (useful for tests).
        """
        try:
            with self._accounts_lock:
                if not account_sizes_data:
                    assert self.running_unit_tests, "Empty account sizes data can only be used in test mode"
                    # Empty dict = clear all data (useful for test cleanup)
                    bt.logging.info("Clearing all miner accounts")
                    self.accounts.clear()
                    self._save_accounts_to_disk()
                    return

                parsed_accounts = self._parse_accounts_dict(account_sizes_data)
                self.accounts.clear()
                self.accounts.update(parsed_accounts)

                self._save_accounts_to_disk()
                bt.logging.info(f"Synced {len(self.accounts)} miner accounts")
        except Exception as e:
            bt.logging.error(f"Failed to sync miner accounts data: {e}")

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
        with self._accounts_lock:
            # Generate timestamp inside lock if not provided
            # This ensures records are added in strictly chronological order
            if timestamp_ms is None:
                timestamp_ms = TimeUtil.now_in_millis()

            account_size = min(ValiConfig.MAX_COLLATERAL_BALANCE_THETA, collateral_balance_theta) * ValiConfig.COST_PER_THETA
            collateral_record = CollateralRecord(account_size, collateral_balance_theta, timestamp_ms)

            # Get or create account
            account = self.get_or_create(hotkey)

            # Skip if the new record matches the last existing record
            if account.collateral_records:
                last_record = account.collateral_records[-1]
                if (last_record.account_size == collateral_record.account_size and
                        last_record.account_size_theta == collateral_record.account_size_theta):
                    bt.logging.info(f"Skipping save for {hotkey} - new record matches last record")
                    return collateral_record

            # Add the new record and update account size
            account.add_collateral_record(collateral_record)

            # Save to disk
            self._save_accounts_to_disk()

        bt.logging.info(
            f"Updated account size for {hotkey}: ${account_size:,.2f} (valid from {collateral_record.valid_date_str})")

        return collateral_record

    def get_miner_account_size(self, hotkey: str, timestamp_ms: Optional[int] = None, most_recent: bool = False,
                               use_account_floor: bool = False) -> float | None:
        """
        Get the account size for a miner at a given timestamp. Iterate list in reverse chronological order, and return
        the first record whose valid_date_timestamp <= start_of_day_ms

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp to query for (defaults to now)
            most_recent: If True, return most recent record regardless of timestamp
            use_account_floor: If True, return MIN_CAPITAL instead of None when no records exist

        Returns:
            Account size in USD, or None if no applicable records (or MIN_CAPITAL if use_account_floor=True)
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        with self._accounts_lock:
            account = self.accounts.get(hotkey)
            if not account or not account.collateral_records:
                # Use account floor if requested (for miners without collateral records)
                return ValiConfig.MIN_CAPITAL if use_account_floor else None

            # Return most recent record
            if most_recent:
                return account.get_account_size()

            # Use MinerAccount's method to get account size at timestamp
            result = account.get_account_size(timestamp_ms)
            if result is None:
                return ValiConfig.MIN_CAPITAL if use_account_floor else None
            return result

    def get_all_miner_account_sizes(self, timestamp_ms: Optional[int] = None) -> dict[str, float]:
        """
        Return a dict of all miner account sizes at a timestamp_ms
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        with self._accounts_lock:
            all_miner_account_sizes = {}
            for hotkey in self.accounts.keys():
                account_size = self.get_miner_account_size(hotkey, timestamp_ms=timestamp_ms)
                if account_size is not None:
                    all_miner_account_sizes[hotkey] = account_size
            return all_miner_account_sizes

    def receive_collateral_record_update(self, collateral_record_data: dict, is_mothership: bool = False) -> bool:
        """
        Process an incoming CollateralRecord synapse and update accounts.

        Args:
            collateral_record_data: Dictionary containing hotkey, account_size, update_time_ms, valid_date_timestamp
            is_mothership: Whether this validator is the mothership (should not receive updates)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if is_mothership:
                return False
            with self._accounts_lock:
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

                # Get or create account
                account = self.get_or_create(hotkey)

                # Check if we already have this record (avoid duplicates)
                if account.collateral_records:
                    if account.collateral_records[-1].account_size == account_size:
                        bt.logging.debug(f"Most recent collateral record for {hotkey} already exists")
                        return True

                # Add the new record and update account size
                account.add_collateral_record(collateral_record)

                # Save to disk
                self._save_accounts_to_disk()

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
        """Get all hotkeys with accounts."""
        with self._accounts_lock:
            return list(self.accounts.keys())

    def health_check(self) -> dict:
        """Health check for monitoring."""
        with self._accounts_lock:
            total_collateral_records = sum(
                len(account.collateral_records) for account in self.accounts.values()
            )
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "num_accounts": len(self.accounts),
            "num_collateral_records": total_collateral_records
        }

    # ==================== Margin/Cash Processing Methods ====================

    def process_order_buy(self, hotkey: str, order_value_usd: float,
                          trade_pair_category: TradePairCategory) -> float:
        """
        Process buy order cash/margin.

        Args:
            hotkey: Miner's hotkey
            order_value_usd: Order value in USD
            trade_pair_category: TradePairCategory enum value

        Returns: {cash_used, borrowed_amount}
        Raises: SignalException if insufficient funds for margin
        """
        account = self.get_or_create(hotkey)

        if trade_pair_category != TradePairCategory.EQUITIES:
            return 0.0

        with self._accounts_lock:
            if order_value_usd <= account.cash_balance:
                # Pure cash purchase - no margin needed
                account.cash_balance -= order_value_usd
                self._save_accounts_to_disk()
                bt.logging.info(f"[{hotkey[:8]}] Cash purchase: ${order_value_usd:.2f}, remaining cash: ${account.cash_balance:.2f}")
                return 0.0

            # Margin purchase (50% initial margin requirement)
            initial_margin = order_value_usd * 0.5
            if account.cash_balance < initial_margin:
                raise SignalException(
                    f"Insufficient funds. Need ${initial_margin:.2f} (50% margin), have ${account.cash_balance:.2f}"
                )

            borrowed_amount = order_value_usd - initial_margin
            account.cash_balance -= initial_margin
            account.total_borrowed_amount += borrowed_amount

            self._save_accounts_to_disk()
            bt.logging.info(
                f"[{hotkey[:8]}] Margin purchase: ${order_value_usd:.2f}, margin used: ${initial_margin:.2f}, "
                f"borrowed: ${borrowed_amount:.2f}, total borrowed: ${account.total_borrowed_amount:.2f}"
            )
            return borrowed_amount

    def process_order_sell(self, hotkey: str, sale_proceeds_usd: float,
                           borrowed_for_position: float, trade_pair_category: TradePairCategory) -> dict:
        """
        Process sell/close order. Pay off loan first, return rest to cash.

        Args:
            hotkey: Miner's hotkey
            sale_proceeds_usd: Proceeds from sale in USD
            borrowed_for_position: Amount borrowed for this position
            trade_pair_category: TradePairCategory enum value

        Returns: {loan_repaid, cash_returned}
        """
        account = self.get_or_create(hotkey)

        if trade_pair_category != TradePairCategory.EQUITIES:
            return {"loan_repaid": 0.0, "cash_returned": 0.0}

        with self._accounts_lock:
            loan_repaid = min(borrowed_for_position, sale_proceeds_usd)
            cash_returned = sale_proceeds_usd - loan_repaid

            account.total_borrowed_amount -= loan_repaid
            account.cash_balance += cash_returned

            self._save_accounts_to_disk()
            bt.logging.info(
                f"[{hotkey[:8]}] Position closed: proceeds ${sale_proceeds_usd:.2f}, loan repaid: ${loan_repaid:.2f}, "
                f"cash returned: ${cash_returned:.2f}, remaining borrowed: ${account.total_borrowed_amount:.2f}"
            )
            return {"loan_repaid": loan_repaid, "cash_returned": cash_returned}

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        account = self.get_account(hotkey)
        if not account:
            return 0.0
        return account.total_borrowed_amount
