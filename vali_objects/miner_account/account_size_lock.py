"""
The fixed account size of a miner on the existing rules: from the lock start on it keeps the size in force when it is
first seen trading, deposits never raise it, and withdrawing into the collateral behind it closes the account.
"""
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

from vali_objects.vali_config import ValiConfig


@dataclass(frozen=True)
class AccountSizeLock:
    locked_ms: int  # when the lock was written; a shrink to a lower balance writes it again
    account_size: float
    locked_theta: float  # the collateral behind the size; a withdrawal into it closes the account
    closed_at_ms: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "locked_ms", int(self.locked_ms))
        object.__setattr__(self, "account_size", float(self.account_size))
        object.__setattr__(self, "locked_theta", float(self.locked_theta))
        if self.closed_at_ms is not None:
            object.__setattr__(self, "closed_at_ms", int(self.closed_at_ms))
        if not (math.isfinite(self.account_size) and self.account_size > 0
                and math.isfinite(self.locked_theta) and self.locked_theta >= 0):
            raise ValueError(f"Invalid account size lock: {self}")

    @property
    def is_closed(self) -> bool:
        return self.closed_at_ms is not None

    def closed(self, closed_at_ms: int) -> "AccountSizeLock":
        return self if self.is_closed else replace(self, closed_at_ms=closed_at_ms)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AccountSizeLock":
        """Unknown keys are ignored, so a lock written by a later version still loads."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def from_collateral(cls, theta: float, locked_ms: int) -> "AccountSizeLock":
        """The lock a collateral balance gives: the size it buys (COST_PER_THETA a theta up to
        MAX_COLLATERAL_BALANCE_THETA, at least MIN_CAPITAL) and the theta behind that size."""
        locked_theta = min(max(float(theta or 0.0), 0.0), ValiConfig.MAX_COLLATERAL_BALANCE_THETA)
        return cls(locked_ms, max(locked_theta * ValiConfig.COST_PER_THETA, ValiConfig.MIN_CAPITAL), locked_theta)
