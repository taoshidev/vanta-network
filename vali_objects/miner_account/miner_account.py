from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone, datetime, timedelta

from entity_management.entity_utils import is_synthetic_hotkey
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket


@dataclass(frozen=True)
class DailyOpenSnapshot:
    """Account state captured at 00:00:00 UTC for a single miner."""
    day_open_ms: int           # Unix ms for 00:00:00 UTC of this day
    account_size: float
    balance: float
    equity: float

    @property
    def equity_return(self) -> float:
        """equity / account_size — return relative to deposited capital."""
        if not self.account_size:
            return 1.0
        return self.equity / self.account_size

    def to_dict(self) -> dict:
        return {
            'day_open_ms': self.day_open_ms,
            'account_size': self.account_size,
            'balance': self.balance,
            'equity': self.equity,
            'equity_return': self.equity_return,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DailyOpenSnapshot:
        return cls(
            day_open_ms=d['day_open_ms'],
            account_size=d['account_size'],
            balance=d['balance'],
            equity=d['equity'],
        )

    @classmethod
    def from_account_size(cls, account_size: float, timestamp_ms: int) -> DailyOpenSnapshot:
        """Create a snapshot for a miner's first deposit, where balance and equity equal account_size."""
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        day_open_ms = int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        return cls(
            day_open_ms=day_open_ms,
            account_size=account_size,
            balance=account_size,
            equity=account_size,
        )


@dataclass(frozen=True)
class CollateralRecord:
    """Record of a collateral/account size update at a specific timestamp."""
    account_size: float
    account_size_theta: float
    update_time_ms: int
    valid_date_timestamp: int = field(init=False)

    def __post_init__(self):
        dt = datetime.fromtimestamp(self.update_time_ms / 1000, tz=timezone.utc)
        start_of_next_day = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        object.__setattr__(self, 'valid_date_timestamp', int(start_of_next_day.timestamp() * 1000))

    @property
    def valid_date_str(self) -> str:
        return TimeUtil.millis_to_short_date_str(self.valid_date_timestamp)


@dataclass
class MinerAccount:
    """Per-miner account state. Unified source of truth for account data."""
    miner_hotkey: str
    total_realized_pnl: float = 0.0     # Cumulative realized PNL from closed trades
    capital_used: float = 0.0            # Total leveraged USD value of open positions
    total_borrowed_amount: float = 0.0   # Total margin loans outstanding (equities only)
    total_fees_paid: float = 0.0         # Cumulative fees paid (transaction, funding, interest, ...)
    total_dividend_income: float = 0.0   # Net dividend income
    asset_class: MinerAssetClass | None = None  # CRYPTO, FOREX, EQUITIES, COMMODITIES, HL_ALL
    collateral_records: list[CollateralRecord] = field(default_factory=list)
    miner_bucket: MinerBucket | None = None  # Pushed by ChallengePeriodManager
    hl_address: str | None = None            # Set for HS subaccounts; None for VT
    max_return: float = 1.0  # High water mark for portfolio return
    unrealized_pnl: float = 0.0  # Current unrealized PNL from open positions
    # Per-asset-class breakdown of capital_used. Required by multi-class subaccounts
    # (HL_ALL) for per-class portfolio cap enforcement. Empty for older checkpoints;
    # lazy-backfilled on next rebuild_account_state_from_positions call.
    capital_used_by_class: dict[TradePairCategory, float] = field(default_factory=dict)
    daily_open_snapshot: DailyOpenSnapshot | None = None

    @property
    def account_size(self) -> float:
        """Current account size"""
        if not self.collateral_records:
            return ValiConfig.MIN_CAPITAL
        return self.collateral_records[-1].account_size

    @property
    def balance(self) -> float:
        """Current balance = account_size + total_realized_pnl + total_dividend_income - total_fees_paid."""
        return self.account_size + self.total_realized_pnl + self.total_dividend_income - self.total_fees_paid

    @property
    def equity(self) -> float:
        """Current equity = balance + unrealized_pnl"""
        return self.balance + self.unrealized_pnl

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
        tier = get_leverage_tier(self.miner_bucket, self.account_size)
        return ValiConfig.TIER_PORTFOLIO_LEVERAGE_BY_ASSET_CLASS[tier].get(self.asset_class, 1.0)

    def add_collateral_record(self, record: 'CollateralRecord'):
        """Add a new collateral record. Account size flows through balance property."""
        self.collateral_records.append(record)

    def get_account_size(self, timestamp_ms: int | None = None) -> float:
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

    @classmethod
    def from_dict(cls, d: dict) -> MinerAccount:
        """Deserialize from the dict produced by to_dict."""
        collateral_records = [
            CollateralRecord(r['account_size'], r.get('account_size_theta', 0), r['update_time_ms'])
            for r in d.get('collateral_records', [])
            if 'account_size' in r and 'update_time_ms' in r
        ]
        capital_used_by_class: dict[TradePairCategory, float] = {
            TradePairCategory(cat_str): float(amount)
            for cat_str, amount in (d.get('capital_used_by_class') or {}).items()
        }

        return cls(
            miner_hotkey=d['miner_hotkey'],
            asset_class=MinerAssetClass(d['asset_class']) if d.get('asset_class') is not None else None,
            miner_bucket=MinerBucket(d['miner_bucket']) if d.get('miner_bucket') is not None else None,
            hl_address=d.get('hl_address'),
            total_realized_pnl=d.get('total_realized_pnl', 0.0),
            capital_used=d.get('capital_used', 0.0),
            total_borrowed_amount=d.get('total_borrowed_amount', 0.0),
            total_fees_paid=d.get('total_fees_paid', 0.0),
            total_dividend_income=d.get('total_dividend_income', 0.0),
            max_return=d.get('max_return', 1.0),
            unrealized_pnl=d.get('unrealized_pnl', 0.0),
            capital_used_by_class=capital_used_by_class,
            collateral_records=collateral_records,
            daily_open_snapshot=DailyOpenSnapshot.from_dict(d['daily_open_snapshot']) if d.get('daily_open_snapshot') else None,
        )

    def to_dict(self, include_computed: bool = True) -> dict:
        d = {
            'miner_hotkey': self.miner_hotkey,
            'asset_class': self.asset_class.value if self.asset_class else None,
            'miner_bucket': self.miner_bucket.value if self.miner_bucket else None,
            'hl_address': self.hl_address,
            'total_realized_pnl': self.total_realized_pnl,
            'capital_used': self.capital_used,
            'total_borrowed_amount': self.total_borrowed_amount,
            'total_fees_paid': self.total_fees_paid,
            'total_dividend_income': self.total_dividend_income,
            'max_return': self.max_return,
            'unrealized_pnl': self.unrealized_pnl,
            'capital_used_by_class': {cat.value: amt for cat, amt in self.capital_used_by_class.items()},
            'daily_open_snapshot': self.daily_open_snapshot.to_dict() if self.daily_open_snapshot else None,
            'collateral_records': [vars(r).copy() for r in self.collateral_records],
        }
        if include_computed:
            d['account_size'] = self.account_size
            d['balance'] = self.balance
            d['buying_power'] = self.buying_power
            d['equity'] = self.equity
        return d

    def to_dashboard(self) -> dict:
        return {
            'account_size': self.account_size,
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

