"""
The account a new regular miner opens on the standard pipeline under its own hotkey, and the view of a
pipeline account that the challenge period and payout code share with entity subaccounts.
"""
import math
from dataclasses import asdict, dataclass, fields
from typing import Optional

from entity_management.entity_utils import is_created_through_challenge, is_instant_funded
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig


@dataclass(frozen=True)
class NewRegularMinerRecord:
    """Created at opening; auto-promotion adds the pro size and closing the time (dataclasses.replace). A field
    added later needs a default so older records still load. The locked theta is the collateral the account size
    required at the opening price, the only collateral ever at risk."""
    opened_at_ms: int
    opening_leverage_tier: int  # priced the locked theta; the live tier is MinerAccount.leverage_tier
    asset_class: str  # for audit; the asset selection stays the source of truth
    account_size: float  # fixed at opening, and the size the pro challenge is paid on
    locked_theta: float
    opening_price_usd: float
    pro_account_size: Optional[float] = None
    payout_scale: Optional[float] = None  # None pays NEW_REGULAR_MINER_PRO_TRANSITION_PAYOUT_MULTIPLIER
    closed_at_ms: Optional[int] = None

    def __post_init__(self):
        # Values are checked here, so a bad record fails where it is made (peers apply the mothership's as is)
        for name in ("opened_at_ms", "opening_leverage_tier", "closed_at_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError(f"{name} must be an integer, not {value!r}")
            object.__setattr__(self, name, int(value))
        for name in ("account_size", "locked_theta", "opening_price_usd", "pro_account_size", "payout_scale"):
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NewRegularMinerRecord":
        """Unknown keys are ignored, so a record written by a later version still loads."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class PipelineAccountInfo:
    """What the pipeline code needs to know about an account on the standard pipeline, read the same way from
    an entity subaccount (SubaccountInfo) and from a new regular miner's record. The size traded now is not here:
    it depends on the bucket and lives on the MinerAccount."""
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
    def from_new_regular_miner(cls, record: NewRegularMinerRecord) -> "PipelineAccountInfo":
        """From a new regular miner's record: always created through the challenge, on the pro track once
        auto-promotion granted a pro size, paid NEW_REGULAR_MINER_PRO_TRANSITION_PAYOUT_MULTIPLIER unless pinned."""
        on_pro_track = record.pro_account_size is not None
        return cls(
            initial_bucket=MinerBucket.SUBACCOUNT_CHALLENGE.value,
            account_type=AccountType.PRO.value if on_pro_track else AccountType.STANDARD.value,
            standard_account_size=record.account_size if on_pro_track else None,
            pro_account_size=record.pro_account_size,
            payout_scale=(record.payout_scale if record.payout_scale is not None
                          else ValiConfig.NEW_REGULAR_MINER_PRO_TRANSITION_PAYOUT_MULTIPLIER),
        )
