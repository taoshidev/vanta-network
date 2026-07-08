"""
MinerAccountManager - Manages per-miner account state and account size tracking.

This manager is the source of truth for miner account state including:
- Account size (via CollateralRecord tracking)
- balance-based buying power model (balance = account_size + total_realized_pnl,
  buying_power = balance * multiplier - capital_used)
- Margin loans (equities only)
- Disk persistence of account sizes

This module contains ALL account size functionality, previously split across
ValidatorContractManager. The contract manager now delegates to this module.
"""
import threading
from dataclasses import dataclass, field
from datetime import timezone, datetime, timedelta
from typing import Dict, Optional, List, Any
import bittensor as bt

from entity_management.entity_utils import is_synthetic_hotkey
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_dataclasses.position import Position
from vali_objects.validator_broadcast_base import ValidatorBroadcastBase


# ==================== Data Classes ====================


class CollateralRecord:
    """Record of a collateral/account size update at a specific timestamp."""

    def __init__(self, account_size: float, account_size_theta: float, update_time_ms: int, is_first_record: bool = False):
        self.account_size = account_size
        self.account_size_theta = account_size_theta
        self.update_time_ms = update_time_ms
        self.valid_date_timestamp = CollateralRecord.valid_from_ms(update_time_ms, is_first_record)

    @staticmethod
    def valid_from_ms(update_time_ms: int, is_first_record: bool = False) -> int:
        """Returns timestamp of start of next day (00:00:00 UTC) when this record is valid"""
        dt = datetime.fromtimestamp(update_time_ms / 1000, tz=timezone.utc)
        start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if is_first_record:
            # First record: valid immediately from start of current day
            return int(start_of_day.timestamp() * 1000)
        else:
            # Subsequent records: valid from start of next day
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
    total_realized_pnl: float = 0.0     # Cumulative realized PNL from closed trades
    capital_used: float = 0.0            # Total leveraged USD value of open positions
    total_borrowed_amount: float = 0.0   # Total margin loans outstanding (equities only)
    total_fees_paid: float = 0.0         # Cumulative fees paid (transaction, funding, interest, ...)
    total_dividend_income: float = 0.0   # Net dividend income
    asset_class: Optional[MinerAssetClass] = None  # CRYPTO, FOREX, EQUITIES, COMMODITIES, HL_ALL
    collateral_records: List[CollateralRecord] = None  # Historical CollateralRecords (List[CollateralRecord])
    miner_bucket: Optional[MinerBucket] = None  # Pushed by ChallengePeriodManager
    hl_address: Optional[str] = None            # Set for HS subaccounts; None for VT
    max_return: float = 1.0  # High water mark for portfolio return
    unrealized_pnl: float = 0.0  # Current unrealized PNL from open positions
    # Per-asset-class breakdown of capital_used. Required by multi-class subaccounts
    # (HL_ALL) for per-class portfolio cap enforcement. Empty for older checkpoints;
    # lazy-backfilled on next rebuild_account_state_from_positions call.
    capital_used_by_class: Dict[TradePairCategory, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize collateral_records to empty list if None."""
        if self.collateral_records is None:
            self.collateral_records = []

    @property
    def balance(self) -> float:
        """Current balance = account_size + total_realized_pnl + total_dividend_income - total_fees_paid."""
        return self.get_account_size() + self.total_realized_pnl + self.total_dividend_income - self.total_fees_paid

    @property
    def equity(self) -> float:
        """Current equity = balance + unrealized_pnl"""
        return self.balance + self.unrealized_pnl

    @property
    def buying_power(self) -> float:
        """Available buying power"""
        if self.asset_class == MinerAssetClass.EQUITIES:
            # balance - cash used
            return (self.balance - (self.capital_used - self.total_borrowed_amount)) * self.multiplier
        else:
            return self.balance * self.multiplier - self.capital_used

    @property
    def multiplier(self) -> float:
        """Subaccount-wide portfolio cap multiplier used by `buying_power`.

        Returns TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier][asset_class]. For multi-class
        subaccounts (HL_ALL, ALL_MARKETS) this is the cross-class overall ceiling; per-class
        sub-caps are enforced separately at order entry via get_portfolio_caps.
        """
        if not self.asset_class:
            return 1

        from vali_objects.utils.leverage_utils import get_leverage_tier
        tier = get_leverage_tier(self.miner_bucket, self.get_account_size())
        return ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier].get(self.asset_class, 1.0)

    def add_collateral_record(self, record: 'CollateralRecord'):
        """Add a new collateral record. Account size flows through balance property."""
        self.collateral_records.append(record)

    def get_account_size(self, timestamp_ms: Optional[int] = None) -> float:
        """Get account size at a given timestamp. Returns MIN_CAPITAL if no collateral records."""
        if not self.collateral_records:
            return ValiConfig.MIN_CAPITAL

        if is_synthetic_hotkey(self.miner_hotkey):
            return self.collateral_records[-1].account_size

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

    def reset_account_fields(self):
        self.total_realized_pnl = 0
        self.capital_used = 0
        self.total_borrowed_amount = 0
        self.total_fees_paid = 0
        self.total_dividend_income = 0
        self.miner_bucket = None
        self.max_return = 1.0
        self.unrealized_pnl = 0.0
        self.capital_used_by_class = {}


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
            'total_realized_pnl': self.total_realized_pnl,
            'capital_used': self.capital_used,
            'balance': self.balance,
            'buying_power': self.buying_power,
            'asset_class': self.asset_class.value if self.asset_class else None,
            'total_borrowed_amount': self.total_borrowed_amount,
            'total_fees_paid': self.total_fees_paid,
            'total_dividend_income': self.total_dividend_income,
            'miner_bucket': self.miner_bucket.value if self.miner_bucket else None,
            'hl_address': self.hl_address,
            'max_return': self.max_return,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': self.equity,
            # JSON keys must be str; convert TradePairCategory enum to its .value
            'capital_used_by_class': {cat.value: amt for cat, amt in self.capital_used_by_class.items()},
        }

        if include_collateral_records:
            result['collateral_records'] = [vars(record) for record in self.collateral_records]

        return result

    def to_dashboard(self) -> dict:
        return {
            'account_size': self.get_account_size(),
            'total_realized_pnl': self.total_realized_pnl,
            'capital_used': self.capital_used,
            'balance': self.balance,
            'total_borrowed_amount': self.total_borrowed_amount,
            'total_fees_paid': self.total_fees_paid,
            'buying_power': self.buying_power,
            'max_return': self.max_return,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': self.equity,
        }


# ==================== Manager Implementation ====================


class MinerAccountManager(ValidatorBroadcastBase):
    """
    Manages all miner accounts and account size tracking.

    This is the unified source of truth for:
    - Account sizes (via CollateralRecord history in MinerAccount)
    - balance and buying power (derived from account_size + total_realized_pnl)
    - Capital used (leveraged value of open positions)
    - Margin loans (total_borrowed_amount, equities only)
    - Disk persistence of account data

    The ValidatorContractManager delegates all account size operations here.
    """

    def __init__(
        self,
        running_unit_tests: bool = False,
        collateral_balance_getter=None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        config=None,
        is_testnet: bool = False
    ):
        """
        Initialize the manager.

        Args:
            running_unit_tests: Whether running in test mode
            collateral_balance_getter: Callable to get collateral balance for a hotkey.
                                       Signature: (hotkey: str) -> Optional[float]
                                       Returns balance in theta tokens, or None.
            connection_mode: RPC or LOCAL mode for asset selection client
            config: Bittensor config (for ValidatorBroadcastBase)
            is_testnet: Whether running on testnet (for ValidatorBroadcastBase)
        """
        # Initialize ValidatorBroadcastBase first
        super().__init__(
            running_unit_tests=running_unit_tests,
            is_testnet=is_testnet,
            config=config,
            connection_mode=connection_mode
        )

        self.running_unit_tests = running_unit_tests
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

    # ==================== Disk Persistence ====================

    def _load_accounts_from_disk(self):
        """Load miner accounts from disk during initialization - protected by locks"""
        try:
            with self._disk_lock:
                accounts_data = ValiUtils.get_vali_json_file_dict(self.MINER_ACCOUNTS_FILE)
                accounts_data.pop("_cost_per_theta", None)  # ignore legacy field
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
                records_list.append(account.to_dict(include_collateral_records=False))

                json_dict[hotkey] = records_list
            return json_dict

    @staticmethod
    def _parse_accounts_dict(data_dict: Dict[str, Any], asset_selection_dict: Optional[Dict[str, str]] = None) -> Dict[str, MinerAccount]:
        """Parse miner accounts from disk format back to MinerAccount objects.

        Format: {"hotkey": [list of CollateralRecord dicts]}
        Account-level fields (cash_balance, asset_class, total_borrowed_amount, total_fees_paid)
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
                    total_realized_pnl = last_record.get("total_realized_pnl")
                    capital_used = last_record.get("capital_used")
                    total_borrowed = last_record.get("total_borrowed_amount", 0.0)
                    total_fees_paid = last_record.get("total_fees_paid", 0.0)
                    total_dividend_income = last_record.get("total_dividend_income", 0.0)
                    miner_bucket_str = last_record.get("miner_bucket")
                    hl_address = last_record.get("hl_address")
                    max_return = last_record.get("max_return", 1.0)
                    unrealized_pnl = last_record.get("unrealized_pnl", 0.0)
                    capital_used_by_class_raw = last_record.get("capital_used_by_class", {})
                else:
                    total_realized_pnl = None
                    capital_used = None
                    total_borrowed = 0.0
                    total_fees_paid = 0.0
                    total_dividend_income = 0.0
                    miner_bucket_str = None
                    hl_address = None
                    max_return = 1.0
                    unrealized_pnl = 0.0
                    capital_used_by_class_raw = {}

                # Deserialize capital_used_by_class: string keys → TradePairCategory enum.
                # Missing field (older checkpoint) → empty dict; will be lazy-backfilled by
                # the next rebuild_account_state_from_positions call.
                capital_used_by_class: Dict[TradePairCategory, float] = {}
                for cat_str, amount in (capital_used_by_class_raw or {}).items():
                    try:
                        capital_used_by_class[TradePairCategory(cat_str)] = float(amount)
                    except ValueError:
                        bt.logging.warning(f"Unknown asset_class '{cat_str}' in capital_used_by_class for {hotkey}; skipping")

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
                            asset_class = MinerAssetClass(asset_class_str)
                        except ValueError:
                            bt.logging.warning(f"Unknown asset_class '{asset_class_str}' for {hotkey}")

                # Parse miner_bucket from disk (None for legacy data, filled on first CP refresh)
                miner_bucket = None
                if miner_bucket_str:
                    try:
                        miner_bucket = MinerBucket(miner_bucket_str)
                    except ValueError:
                        bt.logging.warning(f"Unknown miner_bucket '{miner_bucket_str}' for {hotkey}")

                parsed_accounts[hotkey] = MinerAccount(
                    miner_hotkey=hotkey,
                    total_realized_pnl=total_realized_pnl if total_realized_pnl is not None else 0.0,
                    capital_used=capital_used if capital_used is not None else 0.0,
                    total_borrowed_amount=total_borrowed,
                    total_fees_paid=total_fees_paid,
                    total_dividend_income=total_dividend_income,
                    asset_class=asset_class,
                    collateral_records=collateral_records,
                    miner_bucket=miner_bucket,
                    hl_address=hl_address,
                    max_return=max_return,
                    unrealized_pnl=unrealized_pnl,
                    capital_used_by_class=capital_used_by_class,
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

    def set_miner_account_size(self, hotkey: str, collateral_balance_theta: float, timestamp_ms: Optional[int] = None, account_size: float = None) -> Optional[CollateralRecord]:
        """
        Set the account size for a miner. Saves to memory and disk.
        Records are kept in chronological order.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            collateral_balance_theta: Collateral balance in theta tokens
            timestamp_ms: Timestamp for the record (defaults to now)
            account_size: Optional USD account size. If not provided, calculated from collateral balance

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

            if account_size is None:
                account_size = min(ValiConfig.MAX_COLLATERAL_BALANCE_THETA, collateral_balance_theta) * ValiConfig.COST_PER_THETA

            # Check if this is the first record for this miner
            is_first_record = hotkey not in self.accounts or not self.accounts[hotkey].collateral_records
            collateral_record = CollateralRecord(account_size, collateral_balance_theta, timestamp_ms, is_first_record)

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

    def reset_account_fields(self, hotkey: str, miner_bucket: MinerBucket | None = None) -> bool:
        with self._accounts_lock:
            account = self.accounts.get(hotkey)
            if not account:
                return False

            account.reset_account_fields()
            if miner_bucket:
                account.miner_bucket = miner_bucket

            self._save_accounts_to_disk()

        return True


    def delete_miner_account_size(self, hotkey: str) -> bool:
        """
        Delete the account size for a miner. Used for rollback when operations fail.

        Args:
            hotkey: Miner's hotkey (SS58 address)

        Returns:
            bool: True if deleted (or didn't exist), False on error
        """
        with self._accounts_lock:
            if hotkey in self.accounts:
                del self.accounts[hotkey]
                bt.logging.info(f"Deleted account size for {hotkey}")

                # Save to disk
                self._save_accounts_to_disk()
                return True
            else:
                bt.logging.debug(f"No account size to delete for {hotkey}")
                return True  # Return True - idempotent behavior

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

    def receive_collateral_record_update(self, collateral_record_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process an incoming CollateralRecord synapse and update accounts.

        Args:
            collateral_record_data: Dictionary containing hotkey, account_size, update_time_ms, valid_date_timestamp
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # SECURITY: Verify sender using shared base class method
            if not self.verify_broadcast_sender(sender_hotkey, "CollateralRecord"):
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
                is_first_record = hotkey not in self.accounts or not self.accounts[hotkey].collateral_records
                collateral_record = CollateralRecord(account_size, account_size_theta, update_time_ms, is_first_record)

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
        """Get existing account or create new one with zero realized PNL and zero capital used."""
        if hotkey not in self.accounts:
            self.accounts[hotkey] = MinerAccount(
                miner_hotkey=hotkey,
                total_realized_pnl=0.0,
                capital_used=0.0,
            )
        return self.accounts[hotkey]

    def get_account(self, hotkey: str) -> Optional[MinerAccount]:
        """Get account if it exists, without creating."""
        return self.accounts.get(hotkey)

    def get_accounts(self, hotkeys: List[str]) -> Dict[str, MinerAccount]:
        """Get accounts for multiple hotkeys. Returns dict of hotkey -> MinerAccount for existing accounts."""
        with self._accounts_lock:
            return {hk: self.accounts[hk] for hk in hotkeys if hk in self.accounts}

    def get_dashboard(self, hotkey: str) -> dict | None:
        account = self.accounts.get(hotkey)
        if account is None:
            return None
        return account.to_dashboard()

    def update_unrealized_pnl(self, hotkey_to_unrealized_pnl: Dict[str, float]) -> None:
        """Batch update unrealized PNL for multiple hotkeys and advance max_return HWM."""
        with self._accounts_lock:
            for hotkey, unrealized_pnl in hotkey_to_unrealized_pnl.items():
                account = self.accounts.get(hotkey)
                if account is None:
                    continue

                account.unrealized_pnl = unrealized_pnl

                current_return = account.equity / account.get_account_size()
                account.max_return = max(account.max_return, current_return)

            self._save_accounts_to_disk()

    def set_miner_bucket(self, hotkey: str, bucket: Optional[MinerBucket]) -> None:
        """Set the miner bucket on an account. Called by ChallengePeriodManager via RPC."""
        with self._accounts_lock:
            account = self.get_or_create(hotkey)
            account.miner_bucket = bucket
            self._save_accounts_to_disk()

    def set_miner_buckets(self, hotkey_to_bucket: Dict[str, MinerBucket]) -> None:
        """Bulk update miner buckets across multiple accounts. Saves to disk once."""
        if not hotkey_to_bucket:
            return
        with self._accounts_lock:
            for hotkey, bucket in hotkey_to_bucket.items():
                account = self.get_or_create(hotkey)
                account.miner_bucket = bucket
            self._save_accounts_to_disk()

    def get_hl_address(self, hotkey: str) -> Optional[str]:
        """Return the HL address for an account, or None if not an HS subaccount."""
        with self._accounts_lock:
            account = self.accounts.get(hotkey)
            return account.hl_address if account else None

    def set_hl_address(self, hotkey: str, hl_address: Optional[str]) -> None:
        """Set the HL address on an account. Called by EntityManager when an HL subaccount is created/synced."""
        with self._accounts_lock:
            account = self.get_or_create(hotkey)
            account.hl_address = hl_address
            self._save_accounts_to_disk()

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

    def process_order_buy(self, hotkey: str, order_value_usd: float, borrowed_amount: float, fee_usd: float = 0, trade_pair_category: Optional[TradePairCategory] = None) -> None:
        """
        Process buy order. Check buying_power and track capital_used.

        Args:
            hotkey: Miner's hotkey
            order_value_usd: Order value in USD (full leveraged value)
            borrowed_amount: Amount borrowed (calculated by caller, equities only)
            fee_usd: Transaction fee in USD
            trade_pair_category: Asset class of the order's trade pair. Required to maintain
                capital_used_by_class. Optional for backward compat with callers that haven't
                been updated yet; if None, capital_used_by_class is not updated for this order.

        Raises: SignalException if insufficient buying power
        """
        account = self.get_or_create(hotkey)
        order_value_usd = abs(order_value_usd)
        borrowed_amount = abs(borrowed_amount)

        with self._accounts_lock:
            tolerance = 0.001  # floating point errors
            if order_value_usd + fee_usd * account.multiplier > account.buying_power + tolerance:
                raise SignalException(
                    f"Insufficient buying power. Need ${order_value_usd + fee_usd:.2f}, have ${account.buying_power:.2f}"
                )

            if account.asset_class == MinerAssetClass.EQUITIES and borrowed_amount > 0:
                account.total_borrowed_amount += borrowed_amount

            account.capital_used += order_value_usd
            account.total_fees_paid += fee_usd
            if trade_pair_category is not None:
                account.capital_used_by_class[trade_pair_category] = (
                    account.capital_used_by_class.get(trade_pair_category, 0.0) + order_value_usd
                )

            self._save_accounts_to_disk()

            bt.logging.info(
                f"[PROCESS ORDER BUY {hotkey}] ${order_value_usd:.2f}, capital_used: ${account.capital_used:.2f}, "
                f"buying_power: ${account.buying_power:.2f}, borrowed: ${borrowed_amount:.2f}"
            )

    def process_order_sell(self, hotkey: str, entry_value_usd: float, realized_pnl: float, loan_repaid: float, fee_usd: float = 0, trade_pair_category: Optional[TradePairCategory] = None) -> None:
        """
        Process sell/close order. Free capital_used, compound realized PNL to balance.

        Args:
            hotkey: Miner's hotkey
            entry_value_usd: Original entry value of the position being closed (full leveraged value)
            realized_pnl: Realized PNL from this sale (raw, unmultiplied)
            loan_repaid: Amount of loan repaid (calculated by caller, equities only)
            fee_usd: Transaction fee in USD
            trade_pair_category: Asset class of the position being closed. Required to maintain
                capital_used_by_class. Optional for backward compat; if None, the per-class
                bookkeeping is not adjusted for this close.
        """
        account = self.get_or_create(hotkey)
        entry_value_usd = abs(entry_value_usd)
        loan_repaid = abs(loan_repaid)

        with self._accounts_lock:
            # All asset classes: free capital and compound realized PNL
            account.capital_used = max(0.0, account.capital_used - entry_value_usd)
            account.total_realized_pnl += realized_pnl
            account.total_fees_paid += fee_usd
            if trade_pair_category is not None:
                account.capital_used_by_class[trade_pair_category] = max(
                    0.0, account.capital_used_by_class.get(trade_pair_category, 0.0) - entry_value_usd
                )

            if account.asset_class == MinerAssetClass.EQUITIES and loan_repaid > 0:
                # Clamp to actual borrowed amount and repay
                loan_repaid = min(loan_repaid, account.total_borrowed_amount)
                account.total_borrowed_amount -= loan_repaid

            self._save_accounts_to_disk()

            bt.logging.info(
                f"[PROCESS ORDER SELL {hotkey}] entry_value=${entry_value_usd:.2f}, pnl=${realized_pnl:.2f}, "
                f"loan_repaid=${loan_repaid:.2f}, balance=${account.balance:.2f}, buying_power=${account.buying_power:.2f}"
            )

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        account = self.get_account(hotkey)
        if not account:
            return 0.0
        return account.total_borrowed_amount

    def process_fees(self, hotkey_to_fee: Dict[str, float]) -> None:
        """Batch update total_fees_paid for multiple hotkeys. Saves to disk once at the end."""
        with self._accounts_lock:
            for hotkey, fee_usd in hotkey_to_fee.items():
                account = self.get_or_create(hotkey)
                account.total_fees_paid += fee_usd
            self._save_accounts_to_disk()

    def process_dividend_income(self, hotkey_to_credit: Dict[str, float]) -> None:
        """Batch update total_dividend_income for multiple hotkeys. Saves to disk once."""
        with self._accounts_lock:
            for hotkey, credit_usd in hotkey_to_credit.items():
                account = self.get_or_create(hotkey)
                account.total_dividend_income += credit_usd
            self._save_accounts_to_disk()

    # ==================== Asset Selection / Withdrawal Methods ====================

    def set_asset_selection_client(self, client: AssetSelectionClient) -> None:
        """Set the asset selection client (for testing or lazy initialization)."""
        self._asset_selection_client = client

    @staticmethod
    def compute_account_state_from_positions(positions: list) -> dict:
        """
        Compute account state fields from a list of positions.

        Returns:
            dict with total_realized_pnl, total_fees_paid, capital_used, total_borrowed_amount,
            and capital_used_by_class (per-asset-class breakdown of capital_used).
        """
        total_realized_pnl = 0.0
        total_fees_paid = 0.0
        capital_used = 0.0
        total_borrowed_amount = 0.0
        capital_used_by_class: Dict[TradePairCategory, float] = {}

        for position in positions:
            total_realized_pnl += position.realized_pnl
            total_fees_paid += position.total_fees

            if not position.is_closed_position:
                position_value = abs(position.net_value)
                capital_used += position_value
                total_borrowed_amount += position.margin_loan
                category = position.trade_pair.trade_pair_category
                capital_used_by_class[category] = capital_used_by_class.get(category, 0.0) + position_value

        return {
            'total_realized_pnl': total_realized_pnl,
            'total_fees_paid': total_fees_paid,
            'capital_used': capital_used,
            'total_borrowed_amount': total_borrowed_amount,
            'capital_used_by_class': capital_used_by_class,
        }

    def rebuild_account_state_from_positions(
        self,
        hotkey: str,
        positions: List['Position'],
        miner_bucket: Optional[MinerBucket] = None,
        max_return: Optional[float] = None,
    ) -> None:
        """
        Rebuild a miner's account state (capital_used, total_realized_pnl, total_fees_paid)
        from a list of positions. Preserves collateral_records and asset_class.

        Args:
            hotkey: Miner's hotkey
            positions: All positions (open and closed) for this miner
            miner_bucket: Miner bucket to restore after reset
            max_return: Max return (high water mark) to restore after reset. If None, preserves existing value.
        """
        computed = self.compute_account_state_from_positions(positions)

        with self._accounts_lock:
            account = self.get_or_create(hotkey)

            if miner_bucket is None:
                miner_bucket = account.miner_bucket

            if max_return is None:
                max_return = account.max_return

            account.reset_account_fields()
            account.miner_bucket = miner_bucket
            account.max_return = max_return
            account.total_realized_pnl = computed['total_realized_pnl']
            account.total_fees_paid = computed['total_fees_paid']
            account.capital_used = computed['capital_used']
            account.total_borrowed_amount = computed['total_borrowed_amount']
            account.capital_used_by_class = computed['capital_used_by_class']

            self._save_accounts_to_disk()

            bt.logging.info(
                f"[REBUILD {hotkey}] capital_used=${account.capital_used:.2f}, "
                f"realized_pnl=${account.total_realized_pnl:.2f}, "
                f"fees_paid=${account.total_fees_paid:.2f}, "
                f"balance=${account.balance:.2f}"
            )

    def update_asset_selection(self, hotkey: str, asset_selection: MinerAssetClass) -> bool:

        with self._accounts_lock:
            account = self.get_or_create(hotkey)
            account.asset_class = asset_selection

            # Save to disk
            self._save_accounts_to_disk()

            bt.logging.info(
                f"[{hotkey}] Set asset class to {asset_selection.value}: "
                f"balance: ${account.balance:.2f}, buying_power: ${account.buying_power:.2f}"
            )
            return True
