"""
A standard miner is a regular miner that trades on the standard account pipeline under its own hotkey: its
account, and the pipeline account view shared with entity subaccounts.
"""
import math
from dataclasses import asdict, dataclass, fields
from typing import Optional

from entity_management.entity_utils import is_created_through_challenge, is_instant_funded
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig


@dataclass(frozen=True)
class StandardMinerAccount:
    """Created at opening; the pipeline later adds the pro size and the closing time (dataclasses.replace). A field
    added later needs a default so older records load. The locked theta is the collateral the size required at opening."""
    opened_at_ms: int
    opening_leverage_tier: int  # priced the locked theta; the live tier is MinerAccount.leverage_tier
    asset_class: str  # for audit; the asset selection stays the source of truth
    account_size: float  # fixed at opening, and the size the pro challenge is paid on
    locked_theta: float
    opening_price_usd: float
    trading_account_size: Optional[float] = None  # the size traded now when it differs: the pro size on the pro track
    pro_account_size: Optional[float] = None
    payout_scale: Optional[float] = None  # None pays STANDARD_MINER_PRO_TRANSITION_PAYOUT_MULTIPLIER
    closed_at_ms: Optional[int] = None

    def __post_init__(self):
        # Checked here, so a bad record fails where it is made rather than where it is used
        for name in ("opened_at_ms", "opening_leverage_tier", "closed_at_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError(f"{name} must be an integer, not {value!r}")
            object.__setattr__(self, name, int(value))
        for name in ("account_size", "locked_theta", "opening_price_usd", "trading_account_size", "pro_account_size",
                     "payout_scale"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number, not {value!r}")
            value = float(value)
            if not math.isfinite(value) or value < 0 or (value == 0 and name != "payout_scale"):
                raise ValueError(f"{name} must be a positive number, not {value!r}")
            object.__setattr__(self, name, value)
        if None in (self.opened_at_ms, self.opening_leverage_tier, self.asset_class, self.account_size,
                    self.locked_theta, self.opening_price_usd):
            raise ValueError("a required field of the record is None")

    @property
    def is_closed(self) -> bool:
        return self.closed_at_ms is not None

    @property
    def stage(self) -> tuple:
        """How far the account has come, each step final: closed, trading a size other than the opening one, granted
        a pro size. Of two records of one account, the later stage wins."""
        return (self.is_closed, self.size_traded != self.account_size, self.pro_account_size is not None)

    @property
    def size_traded(self) -> float:
        return self.trading_account_size if self.trading_account_size is not None else self.account_size

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StandardMinerAccount":
        """Unknown keys are ignored, so a record written by a later version still loads."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class PipelineAccountInfo:
    """What the pipeline code reads about an account, the same way from an entity subaccount (SubaccountInfo) and a
    standard miner's record. The size traded now depends on the bucket and lives on the MinerAccount instead."""
    initial_bucket: Optional[str]
    account_type: str  # pro once a pro size is granted, as EntityManager sets it on every pro-track move
    standard_account_size: Optional[float]  # the size the pro challenge is paid on, kept with the pro size
    pro_account_size: Optional[float]
    payout_scale: float  # the multiplier PRO_CHALLENGE_FROM_STANDARD is paid with

    @property
    def created_through_challenge(self) -> bool:
        return is_created_through_challenge(self.initial_bucket)

    @property
    def instant_funded(self) -> bool:
        return is_instant_funded(self.initial_bucket)

    @classmethod
    def from_subaccount(cls, info: dict) -> "PipelineAccountInfo":
        """From the SubaccountInfo dict the entity client serves (never None: callers check first)."""
        payout_scale = info.get("payout_scale")
        return cls(
            initial_bucket=info.get("initial_bucket"),
            account_type=info.get("account_type") or AccountType.STANDARD.value,
            standard_account_size=info.get("standard_account_size"),
            pro_account_size=info.get("pro_account_size"),
            payout_scale=payout_scale if payout_scale is not None else ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER,
        )

    @classmethod
    def from_standard_miner(cls, record: StandardMinerAccount) -> "PipelineAccountInfo":
        """From a standard miner's record: always created through the challenge, on the pro track once
        auto-promotion granted a pro size, paid STANDARD_MINER_PRO_TRANSITION_PAYOUT_MULTIPLIER unless pinned."""
        on_pro_track = record.pro_account_size is not None
        return cls(
            initial_bucket=MinerBucket.SUBACCOUNT_CHALLENGE.value,
            account_type=AccountType.PRO.value if on_pro_track else AccountType.STANDARD.value,
            standard_account_size=record.account_size if on_pro_track else None,
            pro_account_size=record.pro_account_size,
            payout_scale=(record.payout_scale if record.payout_scale is not None
                          else ValiConfig.STANDARD_MINER_PRO_TRANSITION_PAYOUT_MULTIPLIER),
        )


class OpeningRejected(ValueError):
    """An opening request the network cannot honor; the message is for the miner."""


def open_account(collateral_theta, price_usd, leverage_tier, asset_class: str, opened_at_ms: int) -> StandardMinerAccount:
    """Size the account from the collateral's USD value and the tier's collateral percentage, within the size bounds,
    and lock the theta that size requires at this price. Collateral beyond the locked theta stays free."""
    if not ValiConfig.is_valid_standard_leverage_tier(leverage_tier):
        raise OpeningRejected(f"The leverage tier must be one of {list(ValiConfig.STANDARD_LEVERAGE_TIERS)}")
    percentage = ValiConfig.STANDARD_MINER_COLLATERAL_PCT_BY_TIER[leverage_tier]
    if price_usd is None or not math.isfinite(price_usd) or price_usd <= 0:
        raise OpeningRejected("The collateral price is not available; try again shortly")
    if collateral_theta is None or collateral_theta <= 0:
        raise OpeningRejected("No collateral is posted for this hotkey")
    account_size = min(ValiConfig.STANDARD_MINER_MAX_ACCOUNT_SIZE, collateral_theta * price_usd / percentage)
    minimum = ValiConfig.STANDARD_MINER_MIN_ACCOUNT_SIZE
    if account_size < minimum:
        raise OpeningRejected(f"The collateral is worth ${collateral_theta * price_usd:,.2f}; a ${minimum:,.0f} account "
                              f"needs ${minimum * percentage:,.2f} at this tier")
    return StandardMinerAccount(
        opened_at_ms=opened_at_ms, opening_leverage_tier=leverage_tier, asset_class=asset_class,
        account_size=account_size, locked_theta=min(collateral_theta, account_size * percentage / price_usd),
        opening_price_usd=price_usd,
    )


def opening_bucket_thresholds() -> dict:
    """The drawdown limits a standard miner trades under before PRO_FUNDED, as set_miner_bucket overrides."""
    return dict(intraday_drawdown_threshold=ValiConfig.STANDARD_MINER_INTRADAY_DRAWDOWN_THRESHOLD,
                eod_hwm_threshold=ValiConfig.STANDARD_MINER_EOD_HWM_THRESHOLD)
