"""
MinerAccount and CollateralRecord dataclasses.

Per-miner account state (balance, buying power, capital usage) and the
historical collateral records that determine account size over time.
"""
from dataclasses import dataclass, field
from datetime import timezone, datetime, timedelta
from typing import Dict, Optional, List

from entity_management.entity_utils import is_synthetic_hotkey
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket


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
            return int(start_of_day.timestamp() * 1000)
        else:
            start_of_next_day = start_of_day + timedelta(days=1)
            return int(start_of_next_day.timestamp() * 1000)

    @property
    def valid_date_str(self) -> str:
        """Returns YYYY-MM-DD format for easy reading"""
        return TimeUtil.millis_to_short_date_str(self.valid_date_timestamp)

    def __repr__(self):
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
    # Per-asset-class breakdown of capital_used. Required by multi-class subaccounts
    # (HL_ALL) for per-class portfolio cap enforcement. Empty for older checkpoints;
    # lazy-backfilled on next rebuild_account_state_from_positions call.
    capital_used_by_class: Dict[TradePairCategory, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize collateral_records to empty list if None."""
        if self.collateral_records is None:
            self.collateral_records = []

    @property
    def account_size(self) -> float:
        return self.get_account_size()

    @property
    def balance(self) -> float:
        """Current balance = account_size + total_realized_pnl + total_dividend_income - total_fees_paid."""
        return self.get_account_size() + self.total_realized_pnl + self.total_dividend_income - self.total_fees_paid

    @property
    def buying_power(self) -> float:
        """Available buying power"""
        if self.asset_class == MinerAssetClass.EQUITIES:
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

        start_of_day_ms = int(
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp() * 1000
        )

        for record in reversed(self.collateral_records):
            if record.valid_date_timestamp <= start_of_day_ms:
                theta = min(record.account_size_theta, ValiConfig.MAX_COLLATERAL_BALANCE_THETA)
                return max(theta * ValiConfig.COST_PER_THETA, ValiConfig.MIN_CAPITAL)

        return ValiConfig.MIN_CAPITAL

    def reset_account_fields(self):
        self.total_realized_pnl = 0
        self.capital_used = 0
        self.total_borrowed_amount = 0
        self.total_fees_paid = 0
        self.total_dividend_income = 0
        self.miner_bucket = None
        self.max_return = 1.0
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
            'max_return': self.max_return
        }
