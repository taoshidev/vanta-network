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
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient


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
    asset_class: Optional[TradePairCategory] = None  # EQUITIES, CRYPTO, FOREX
    collateral_records: List[CollateralRecord] = None  # Historical CollateralRecords (List[CollateralRecord])
    last_interest_date_ms: Optional[int] = None  # Last date interest was applied

    def __post_init__(self):
        """Initialize collateral_records to empty list if None."""
        if self.collateral_records is None:
            self.collateral_records = []

    def add_collateral_record(self, record: 'CollateralRecord', multiplier: float = 1.0):
        """Add a new collateral record and update account_size.

        Args:
            record: The CollateralRecord to add
            multiplier: Cash balance multiplier based on asset selection (default 1.0)
        """
        previous_size = self.get_account_size()
        new_size = record.account_size
        self.collateral_records.append(record)

        if previous_size:
            size_increase = new_size - previous_size
            self.cash_balance += size_increase * multiplier

    def get_account_size(self, timestamp_ms: Optional[int] = None) -> float:
        """Get account size at a given timestamp. Returns MIN_CAPITAL if no collateral records."""
        if not self.collateral_records:
            return ValiConfig.MIN_CAPITAL

        if timestamp_ms is None:
            theta = min(self.collateral_records[-1].account_size_theta, ValiConfig.MAX_COLLATERAL_BALANCE_THETA)
            return max(theta * ValiConfig.COST_PER_THETA, ValiConfig.MIN_CAPITAL)

        # Get start of the requested day
        start_of_day_ms = int(
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )

        # Iterate in reversed order, return first record valid for or before the requested day
        for record in reversed(self.collateral_records):
            if record.valid_date_timestamp <= start_of_day_ms:
                theta = min(record.account_size_theta, ValiConfig.MAX_COLLATERAL_BALANCE_THETA)
                return max(theta * ValiConfig.COST_PER_THETA, ValiConfig.MIN_CAPITAL)

        # No valid record for the timestamp, return MIN_CAPITAL
        return ValiConfig.MIN_CAPITAL

    def apply_interest(self, current_time_ms: int, running_unit_tests: bool = False) -> bool:
        """
        Apply daily interest to this account if needed.

        Args:
            current_time_ms: Current timestamp in milliseconds
            running_unit_tests: Whether running in test mode (for transaction recording)

        Returns:
            True if interest was processed for this hotkey, False otherwise
        """
        daily_interest_rate = ValiConfig.DAILY_INTEREST_RATE

        # Skip if no borrowed amount
        if self.total_borrowed_amount <= 0:
            return False

        current_date = datetime.fromtimestamp(current_time_ms / 1000, tz=timezone.utc).date()

        # First time seeing this loan - mark date, don't charge (first day free)
        if self.last_interest_date_ms is None:
            self.last_interest_date_ms = current_time_ms
            return True

        # Check last applied date
        last_applied_date = datetime.fromtimestamp(
            self.last_interest_date_ms / 1000, tz=timezone.utc
        ).date()
        if last_applied_date == current_date:
            return False

        # Calculate daily interest
        daily_interest = self.total_borrowed_amount * daily_interest_rate
        unpaid_interest = 0.0

        if self.cash_balance >= daily_interest:
            # Full interest paid from cash
            self.cash_balance -= daily_interest
            bt.logging.info(
                f"[{self.miner_hotkey[:8]}] Interest charged: ${daily_interest:.4f} (paid from cash), "
                f"remaining cash: ${self.cash_balance:.2f}, total borrowed: ${self.total_borrowed_amount:.2f}"
            )
        else:
            # Partial/no cash available: use all cash, add remainder to loan
            unpaid_interest = daily_interest - self.cash_balance
            self.total_borrowed_amount += unpaid_interest
            self.cash_balance = 0.0
            bt.logging.warning(
                f"[{self.miner_hotkey[:8]}] Interest charged: ${daily_interest:.4f} "
                f"(${unpaid_interest:.4f} added to loan - compounding), "
                f"total borrowed: ${self.total_borrowed_amount:.2f}"
            )

        # Update last interest date
        self.last_interest_date_ms = current_time_ms

        # Record interest transaction
        MinerAccountManager.record_transaction(self.miner_hotkey, {
            "timestamp_ms": current_time_ms,
            "type": "INTEREST",
            "cash_delta": -(daily_interest - unpaid_interest),  # cash paid (negative)
            "loan_delta": unpaid_interest  # unpaid added to loan (positive)
        }, running_unit_tests=running_unit_tests)

        return True

    def to_dict(self, include_collateral_records: bool = False) -> dict:
        """
        Convert MinerAccount to dictionary representation.

        Args:
            include_collateral_records: If True, include full collateral records history

        Returns:
            dict with account data
        """
        result = {
            'miner_hotkey': self.miner_hotkey,
            'account_size': self.get_account_size(),
            'cash_balance': self.cash_balance,
            'asset_class': self.asset_class.value if self.asset_class else None,
            'total_borrowed_amount': self.total_borrowed_amount,
            'last_interest_date_ms': self.last_interest_date_ms
        }

        if include_collateral_records:
            result['collateral_records'] = [vars(record) for record in self.collateral_records]

        return result


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

    def __init__(
        self,
        running_unit_tests: bool = False,
        collateral_balance_getter=None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize the manager.

        Args:
            running_unit_tests: Whether running in test mode
            collateral_balance_getter: Callable to get collateral balance for a hotkey.
                                       Signature: (hotkey: str) -> Optional[float]
                                       Returns balance in theta tokens, or None.
            connection_mode: RPC or LOCAL mode for asset selection client
        """
        self.running_unit_tests = running_unit_tests
        self._collateral_balance_getter = collateral_balance_getter
        self.connection_mode = connection_mode

        # Unified MinerAccount storage - single source of truth
        self.accounts: Dict[str, MinerAccount] = {}

        # Locking strategy - EAGER initialization (not lazy!)
        # RLock allows same thread to acquire lock multiple times (needed for nested calls)
        self._accounts_lock = threading.RLock()
        # Lock for disk I/O serialization to prevent concurrent file writes
        self._disk_lock = threading.Lock()

        # Asset selection client for determining miner's trading category
        self._asset_selection_client = AssetSelectionClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests
        )

        # Initialize miner accounts file location
        self.MINER_ACCOUNTS_FILE = ValiBkpUtils.get_miner_account_sizes_file_location(
            running_unit_tests=running_unit_tests
        )
        self.ASSET_SELECTIONS_FILE = ValiBkpUtils.get_asset_selections_file_location(
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
                accounts_data = ValiUtils.get_vali_json_file_dict(self.MINER_ACCOUNTS_FILE)
                asset_selection_data = dict(ValiUtils.get_vali_json_file(self.ASSET_SELECTIONS_FILE))
                parsed_accounts = self._parse_accounts_dict(accounts_data, asset_selection_data)

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
            Dictionary with hotkeys as keys and list of collateral records as values.
            Account-level fields are added to the last record in the list.
            If no collateral records exist, a single record with only account-level fields is saved.
        """
        with self._accounts_lock:
            json_dict = {}
            for hotkey, account in self.accounts.items():
                # Build list of collateral records
                if most_recent_only and account.collateral_records:
                    records = [account.collateral_records[-1]]
                else:
                    records = account.collateral_records

                records_list = [vars(record).copy() for record in records]

                # If no collateral records, create empty record for account-level fields
                if not records_list:
                    records_list.append({})

                # Add account-level fields to the last record
                records_list[-1]["cash_balance"] = account.cash_balance
                records_list[-1]["asset_class"] = account.asset_class.value if account.asset_class else None
                records_list[-1]["total_borrowed_amount"] = account.total_borrowed_amount
                records_list[-1]["last_interest_date_ms"] = account.last_interest_date_ms

                json_dict[hotkey] = records_list
            return json_dict

    @staticmethod
    def _parse_accounts_dict(data_dict: Dict[str, Any], asset_selection_dict: Optional[Dict[str, str]] = None) -> Dict[str, MinerAccount]:
        """Parse miner accounts from disk format back to MinerAccount objects.

        Format: {"hotkey": [list of CollateralRecord dicts]}
        Account-level fields (cash_balance, asset_class, total_borrowed_amount, last_interest_date_ms)
        are stored on the last record in the list.

        Args:
            data_dict: Dict of hotkey -> list of collateral record dicts
            asset_selection_dict: Optional dict of hotkey -> asset class string (for initial sync)
        """
        parsed_accounts = {}

        for hotkey, account_data in data_dict.items():
            try:
                if not isinstance(account_data, list):
                    continue

                records_list = account_data
                collateral_records = []

                # Extract account-level fields from the last record in the list
                if records_list and isinstance(records_list[-1], dict):
                    last_record = records_list[-1]
                    cash_balance = last_record.get("cash_balance")
                    total_borrowed = last_record.get("total_borrowed_amount", 0.0)
                    last_interest_date_ms = last_record.get("last_interest_date_ms")
                else:
                    cash_balance = None  # Will default to account_size
                    total_borrowed = 0.0
                    last_interest_date_ms = None

                # Parse collateral records
                for record_data in records_list:
                    if isinstance(record_data, dict) and "account_size" in record_data and "update_time_ms" in record_data:
                        record = CollateralRecord(
                            record_data["account_size"],
                            record_data.get("account_size_theta", 0),
                            record_data["update_time_ms"]
                        )
                        collateral_records.append(record)

                # Get account_size from collateral records, or fall back to cash_balance, or MIN_CAPITAL
                if collateral_records:
                    account_size = collateral_records[-1].account_size
                else:
                    account_size = ValiConfig.MIN_CAPITAL

                # Get asset_class from asset_selections file (source of truth during migration)
                asset_class = None
                if asset_selection_dict:
                    asset_class_str = asset_selection_dict.get(hotkey)
                    if asset_class_str:
                        try:
                            asset_class = TradePairCategory(asset_class_str)
                        except ValueError:
                            bt.logging.warning(f"Unknown asset_class '{asset_class_str}' for {hotkey}")

                parsed_accounts[hotkey] = MinerAccount(
                    miner_hotkey=hotkey,
                    cash_balance=cash_balance if cash_balance is not None else account_size,
                    total_borrowed_amount=total_borrowed,
                    asset_class=asset_class,
                    collateral_records=collateral_records,
                    last_interest_date_ms=last_interest_date_ms
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
                    bt.logging.info("Clearing all miner accounts")
                    self.accounts.clear()
                    self._save_accounts_to_disk()
                    return

                asset_data = dict(ValiUtils.get_vali_json_file(self.ASSET_SELECTIONS_FILE))
                parsed_accounts = self._parse_accounts_dict(account_sizes_data, asset_data)
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

            # Get asset selection multiplier for cash balance scaling
            asset_selection = self._asset_selection_client.get_asset_selection(hotkey)
            multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_selection, 1.0)

            # Update asset_class if not already set
            if account.asset_class is None:
                account.asset_class = asset_selection

            # Add the new record and update account size
            account.add_collateral_record(collateral_record, multiplier=multiplier)

            # Save to disk
            self._save_accounts_to_disk()

        bt.logging.info(
            f"Updated account size for {hotkey}: ${account_size:,.2f} (valid from {collateral_record.valid_date_str})")

        return collateral_record

    def get_miner_account_size(self, hotkey: str, timestamp_ms: Optional[int] = None, most_recent: bool = False,
                               use_account_floor: bool = False) -> float | None:
        """
        Get the account size for a miner at a given timestamp.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp to query for. If None, returns most recent record.
            most_recent: If True, return most recent record regardless of timestamp
            use_account_floor: If True, return MIN_CAPITAL instead of None when no account exists

        Returns:
            Account size in USD. Returns MIN_CAPITAL for accounts without collateral records.
            Returns None if account doesn't exist (or MIN_CAPITAL if use_account_floor=True).
        """
        with self._accounts_lock:
            account = self.accounts.get(hotkey)
            if not account:
                return ValiConfig.MIN_CAPITAL if use_account_floor else None

            # Return most recent record when no timestamp provided or most_recent=True
            if most_recent or timestamp_ms is None:
                return account.get_account_size()

            # Get account size at timestamp (returns MIN_CAPITAL if no applicable records)
            return account.get_account_size(timestamp_ms)

    def get_all_miner_account_sizes(self, timestamp_ms: Optional[int] = None) -> dict[str, float]:
        """
        Return a dict of all miner account sizes. If timestamp_ms is None, returns most recent sizes.
        """
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

                # Get asset selection multiplier for cash balance scaling
                asset_selection = self._asset_selection_client.get_asset_selection(hotkey)
                multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_selection, 1.0)

                # Update asset_class if not already set
                if account.asset_class is None:
                    account.asset_class = asset_selection

                # Add the new record and update account size
                account.add_collateral_record(collateral_record, multiplier=multiplier)

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
        """Get existing account or create new one with MIN_CAPITAL scaled by asset selection multiplier."""
        if hotkey not in self.accounts:
            # Get asset selection for initial cash balance and asset_class
            asset_selection = self._asset_selection_client.get_asset_selection(hotkey)
            multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_selection, 1.0)
            self.accounts[hotkey] = MinerAccount(
                miner_hotkey=hotkey,
                cash_balance=ValiConfig.MIN_CAPITAL * multiplier,
                asset_class=asset_selection,
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

    @staticmethod
    def record_transaction(hotkey: str, transaction: dict, running_unit_tests: bool = False) -> None:
        """Record a transaction to the miner's transaction history."""
        try:
            tx_path = ValiBkpUtils.get_miner_transactions_path(
                hotkey, running_unit_tests=running_unit_tests
            )
            ValiBkpUtils.append_transaction(tx_path, transaction)
        except Exception as e:
            bt.logging.error(f"Failed to record transaction for {hotkey}: {e}")

    def process_order_buy(self, hotkey: str, order_value_usd: float,
                          trade_pair_category: TradePairCategory) -> float:
        """
        Process buy order cash/margin.

        Args:
            hotkey: Miner's hotkey
            order_value_usd: Order value in USD
            trade_pair_category: TradePairCategory enum value

        Returns: borrowed_amount
        Raises: SignalException if insufficient funds for margin
        """
        account = self.get_or_create(hotkey)

        if trade_pair_category != TradePairCategory.EQUITIES:
            bt.logging.info(f"[PROCESS ORDER BUY] ${order_value_usd} for {trade_pair_category}")
            return 0.0

        with self._accounts_lock:
            if order_value_usd <= account.cash_balance:
                # Pure cash purchase - no margin needed
                account.cash_balance -= order_value_usd
                self._save_accounts_to_disk()
                MinerAccountManager.record_transaction(hotkey, {
                    "timestamp_ms": TimeUtil.now_in_millis(),
                    "type": "BUY",
                    "cash_delta": -order_value_usd,
                    "loan_delta": 0.0
                }, running_unit_tests=self.running_unit_tests)
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
            MinerAccountManager.record_transaction(hotkey, {
                "timestamp_ms": TimeUtil.now_in_millis(),
                "type": "BUY",
                "cash_delta": -initial_margin,
                "loan_delta": borrowed_amount
            }, running_unit_tests=self.running_unit_tests)
            bt.logging.info(
                f"[PROCESS ORDER BUY] {hotkey} Margin purchase: ${order_value_usd:.2f}, margin used: ${initial_margin:.2f}, "
                f"borrowed: ${borrowed_amount:.2f}, total borrowed: ${account.total_borrowed_amount:.2f}"
            )
            return borrowed_amount

    def process_order_sell(self, hotkey: str, sale_proceeds_usd: float,
                           position_margin_loan: float, trade_pair_category: TradePairCategory) -> float:
        """
        Process sell/close order. Pay off loan first, return rest to cash.

        Args:
            hotkey: Miner's hotkey
            sale_proceeds_usd: Proceeds from sale in USD
            position_margin_loan: Margin loan amount for this position
            trade_pair_category: TradePairCategory enum value

        Returns: loan_repaid
        """
        account = self.get_or_create(hotkey)

        if trade_pair_category != TradePairCategory.EQUITIES:
            bt.logging.info(f"[PROCESS ORDER SELL] ${sale_proceeds_usd} for {trade_pair_category}")
            return 0.0

        with self._accounts_lock:
            loan_repaid = min(position_margin_loan, sale_proceeds_usd)
            cash_returned = sale_proceeds_usd - loan_repaid

            account.total_borrowed_amount -= loan_repaid
            account.cash_balance += cash_returned

            self._save_accounts_to_disk()
            MinerAccountManager.record_transaction(hotkey, {
                "timestamp_ms": TimeUtil.now_in_millis(),
                "type": "SELL",
                "cash_delta": cash_returned,
                "loan_delta": -loan_repaid
            }, running_unit_tests=self.running_unit_tests)
            bt.logging.info(
                f"[PROCESS ORDER SELL] {hotkey} Sell processed: proceeds ${sale_proceeds_usd:.2f}, loan repaid: ${loan_repaid:.2f}, "
                f"cash returned: ${cash_returned:.2f}, remaining borrowed: ${account.total_borrowed_amount:.2f}"
            )
            return loan_repaid

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        account = self.get_account(hotkey)
        if not account:
            return 0.0
        return account.total_borrowed_amount

    def apply_daily_interest(self) -> int:
        """
        Apply daily interest to accounts with outstanding margin loans that need it.
        Interest is applied on a 24-hour interval basis per account via MinerAccount.apply_interest().
        """
        accounts_processed = 0
        current_time_ms = TimeUtil.now_in_millis()

        with self._accounts_lock:
            for hotkey, account in self.accounts.items():
                # Let the account handle its own interest calculation
                processed = account.apply_interest(current_time_ms, running_unit_tests=self.running_unit_tests)
                if processed:
                    accounts_processed += 1

            # Save to disk
            if accounts_processed > 0:
                self._save_accounts_to_disk()
                bt.logging.success(f"Daily interest applied to {accounts_processed} accounts")

        return accounts_processed

    def reconstruct_account_from_transactions(self, hotkey: str) -> Optional[MinerAccount]:
        """
        Reconstruct a MinerAccount from collateral records and transaction history.
        Returns None if account doesn't exist.
        """
        account = self.get_account(hotkey)
        if not account or not account.collateral_records:
            return None

        multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(account.asset_class, 1.0) if account.asset_class else 1.0

        # Start with first collateral record
        initial_size = account.collateral_records[0].account_size
        cash_balance = initial_size * multiplier
        total_borrowed = 0.0

        # Read transactions
        tx_path = ValiBkpUtils.get_miner_transactions_path(hotkey, running_unit_tests=self.running_unit_tests)
        transactions = ValiBkpUtils.read_transactions(tx_path)

        # Track collateral changes
        cr_idx = 1
        for tx in sorted(transactions, key=lambda x: x['timestamp_ms']):
            tx_time = tx['timestamp_ms']

            # Apply collateral changes before this transaction
            while cr_idx < len(account.collateral_records):
                cr = account.collateral_records[cr_idx]
                if cr.valid_date_timestamp <= tx_time:
                    prev_size = account.collateral_records[cr_idx - 1].account_size
                    cash_balance += (cr.account_size - prev_size) * multiplier
                    cr_idx += 1
                else:
                    break

            cash_balance += tx['cash_delta']
            total_borrowed += tx['loan_delta']

        # Apply remaining collateral records
        while cr_idx < len(account.collateral_records):
            cr = account.collateral_records[cr_idx]
            prev_size = account.collateral_records[cr_idx - 1].account_size
            cash_balance += (cr.account_size - prev_size) * multiplier
            cr_idx += 1

        return MinerAccount(
            miner_hotkey=hotkey,
            cash_balance=cash_balance,
            total_borrowed_amount=total_borrowed,
            asset_class=account.asset_class,
            collateral_records=account.collateral_records
        )

    # ==================== Asset Selection / Withdrawal Methods ====================

    def set_asset_selection_client(self, client: AssetSelectionClient) -> None:
        """Set the asset selection client (for testing or lazy initialization)."""
        self._asset_selection_client = client

    def can_withdraw_collateral(self, hotkey: str, amount_theta: float) -> bool:
        """
        Check if miner can withdraw the specified amount of collateral.

        The cash balance represents available trading capacity. If positions are open,
        some collateral must be withheld to support those positions.

        Formula:
            total_cash_capacity = account_size * multiplier
            cash_used = total_cash_capacity - cash_balance
            collateral_needed_usd = cash_used / multiplier
            max_withdrawable_usd = account_size - collateral_needed_usd
            max_withdrawable_theta = max_withdrawable_usd / COST_PER_THETA

        Args:
            hotkey: Miner's hotkey
            amount_theta: Requested withdrawal amount in theta

        Returns:
            True if withdrawal is allowed, False otherwise
        """
        # No asset selection = no positions possible = no restrictions
        # TODO update for crypto and forex, ignore initially for equities
        asset_selection = self._asset_selection_client.get_asset_selection(hotkey)
        if asset_selection is None or asset_selection != TradePairCategory.EQUITIES:
            return True

        with self._accounts_lock:
            account = self.accounts.get(hotkey)
            if account is None:
                return True

            account_size = account.get_account_size()
            cash_balance = account.cash_balance
            multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_selection, 1.0)

            # Total virtual cash capacity based on account size and multiplier
            total_cash_capacity = account_size * multiplier

            # Cash used in positions = total capacity - available cash
            cash_used = total_cash_capacity - cash_balance

            # Collateral needed to back the used cash (inverse of multiplier)
            collateral_needed_usd = cash_used / multiplier

            # Max withdrawable is current account size minus what's needed
            max_withdrawable_usd = account_size - collateral_needed_usd

            # Convert to theta
            max_withdrawable_theta = max_withdrawable_usd / ValiConfig.COST_PER_THETA

            return amount_theta <= max(0.0, max_withdrawable_theta)

    def recalculate_cash_balance_for_asset_selection(self, hotkey: str, asset_selection: TradePairCategory) -> bool:
        """
        Recalculate cash balance when a miner selects an asset class.

        This handles the case where a miner deposits collateral before selecting an asset class.
        When they later select an asset class, the cash balance needs to be recalculated
        based on the new multiplier.

        Formula:
            new_cash_balance = account_size * multiplier

        Args:
            hotkey: Miner's hotkey
            asset_selection: The TradePairCategory the miner selected

        Returns:
            True if cash balance was updated, False otherwise
        """
        with self._accounts_lock:
            account = self.accounts.get(hotkey)
            if account is None:
                bt.logging.debug(f"[{hotkey[:8]}] No account found for asset selection recalculation")
                return False

            account_size = account.get_account_size()
            multiplier = ValiConfig.CASH_BALANCE_MULTIPLIER.get(asset_selection, 1.0)

            # Calculate new cash balance based on account size and multiplier
            new_cash_balance = account_size * multiplier
            old_cash_balance = account.cash_balance

            # Update cash balance and asset_class
            account.cash_balance = new_cash_balance
            account.asset_class = asset_selection

            # Save to disk
            self._save_accounts_to_disk()

            bt.logging.info(
                f"[{hotkey[:8]}] Recalculated cash balance for {asset_selection.value}: "
                f"${old_cash_balance:,.2f} -> ${new_cash_balance:,.2f} (multiplier: {multiplier}x)"
            )
            return True
