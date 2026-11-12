"""
A standard miner is a regular miner that trades on the standard account pipeline under its own hotkey: its
account, and the pipeline account view shared with entity subaccounts.
"""
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

from entity_management.entity_utils import (
    is_created_through_challenge,
    is_instant_funded,
    is_on_promotion_rules,
    pro_account_size_error,
)
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
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
    payout_scale: Optional[float] = None  # the Grow payout multiplier, fixed with the pro size
    closed_at_ms: Optional[int] = None
    reopened_at_ms: Optional[int] = None  # when an admin last undid a close; a later reopening outranks a close

    def __post_init__(self):
        # Checked here, so a bad record fails where it is made rather than where it is used
        for name in ("opened_at_ms", "opening_leverage_tier", "closed_at_ms", "reopened_at_ms"):
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
        """Mothership-set progress (closed, granted a pro size): of two records the later stage wins, unless a close was
        undone (supersedes). The size traded is not a stage: each validator sets it at its own bucket move."""
        return (self.is_closed, self.pro_account_size is not None)

    def supersedes(self, other: "StandardMinerAccount") -> bool:
        """Whether this copy of the record outranks another: the one reopened later, else the later stage."""
        reopened, other_reopened = self.reopened_at_ms or -1, other.reopened_at_ms or -1
        if reopened != other_reopened:
            return reopened > other_reopened
        return self.stage > other.stage

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
    hl_linked: bool = False  # a Hyperliquid-linked subaccount (hl_address, or hl_all from before it was recorded)
    created_at_ms: Optional[int] = None  # a subaccount's creation time
    standard_miner: bool = False

    def on_promotion_rules(self, is_testnet: bool) -> bool:
        """A standard miner, or a subaccount created from ValiConfig.SUBACCOUNT_PROMOTION_RULES_START_MS on."""
        return self.standard_miner or is_on_promotion_rules(self.created_at_ms, is_testnet)

    @property
    def created_through_challenge(self) -> bool:
        return is_created_through_challenge(self.initial_bucket)

    @property
    def instant_funded(self) -> bool:
        return is_instant_funded(self.initial_bucket)

    @classmethod
    def from_subaccount(cls, info) -> "PipelineAccountInfo":
        """From the SubaccountInfo the entity client serves, as the dict it sends over RPC or the model itself
        (never None: callers check first)."""
        if hasattr(info, "model_dump"):
            info = info.model_dump()
        payout_scale = info.get("payout_scale")
        return cls(
            initial_bucket=info.get("initial_bucket"),
            account_type=info.get("account_type") or AccountType.STANDARD.value,
            standard_account_size=info.get("standard_account_size"),
            pro_account_size=info.get("pro_account_size"),
            payout_scale=payout_scale if payout_scale is not None else ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER,
            hl_linked=bool(info.get("hl_address")) or info.get("asset_class") == MinerAssetClass.HL_ALL.value,
            created_at_ms=info.get("created_at_ms"),
        )

    @classmethod
    def from_standard_miner(cls, record: StandardMinerAccount) -> "PipelineAccountInfo":
        """From a standard miner's record: always created through the challenge, on the pro track once
        auto-promotion granted a pro size, paid the multiplier fixed with it (the lowest if none was)."""
        on_pro_track = record.pro_account_size is not None
        return cls(
            initial_bucket=MinerBucket.SUBACCOUNT_CHALLENGE.value,
            account_type=AccountType.PRO.value if on_pro_track else AccountType.STANDARD.value,
            standard_account_size=record.account_size if on_pro_track else None,
            pro_account_size=record.pro_account_size,
            payout_scale=(record.payout_scale if record.payout_scale is not None
                          else ValiConfig.grow_payout_multiplier()),
            standard_miner=True,
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


def opening_drawdown_rules() -> dict:
    """The drawdown rules of a standard miner, as set_miner_bucket takes them: the static rules until the pro buckets,
    then the EOD high-water mark limit, with the daily loss limit, both overriding every bucket but PRO_FUNDED."""
    return dict(drawdown_criteria=DrawdownCriteria.STATIC,
                intraday_drawdown_threshold=ValiConfig.STANDARD_MINER_INTRADAY_DRAWDOWN_THRESHOLD,
                eod_hwm_threshold=ValiConfig.STANDARD_MINER_EOD_HWM_THRESHOLD)


# Eliminations that take the closing share on the drawdown at that moment: the miner left, or its UID did
LIVE_DRAWDOWN_REASONS = (EliminationReason.ACCOUNT_CLOSED, EliminationReason.INACTIVE, EliminationReason.ZOMBIE,
                         EliminationReason.DEREGISTERED)


def closing_slash_proportion(drawdown: Optional[dict]) -> float:
    """The share of the locked theta an account forfeits on closing: the worse of today's loss and the account's other
    drawdown, each against its limit, at most all of it (the drawdown stats dict). The other drawdown is the one from
    the starting balance while the static rules apply, else the live one from the end-of-day high-water mark."""
    if not drawdown:
        return 0.0
    other = (("static_drawdown_pct", "static_drawdown_threshold") if drawdown.get("static_rules_apply")
             else ("trailing_drawdown_pct", "eod_drawdown_threshold"))
    shares = []
    for pct_key, limit_key in (("intraday_drawdown_pct", "intraday_drawdown_threshold"), other):
        limit = drawdown.get(limit_key)
        if limit:
            shares.append(max(0.0, drawdown.get(pct_key) or 0.0) / (limit * 100.0))
    return min(1.0, max(shares, default=0.0))


def elimination_slash_proportion(reason: EliminationReason, elimination_drawdown_pct: Optional[float],
                                 drawdown: Optional[dict]) -> float:
    """The share of the locked theta an elimination takes: all of it past a drawdown limit (daily loss, static or EOD),
    the closing share when the account closed, went inactive or left the metagraph, else the drawdown against the
    maximum total drawdown."""
    if reason.is_intraday_drawdown or reason.is_eod_drawdown or reason.is_static_drawdown:
        return 1.0
    if reason in LIVE_DRAWDOWN_REASONS:
        return closing_slash_proportion(drawdown)
    proportion = (elimination_drawdown_pct or 0.0) / ((1 - ValiConfig.MAX_TOTAL_DRAWDOWN) * 100.0)
    return max(0.0, min(1.0, proportion))


class SizingRejected(ValueError):
    """A bucket move the record cannot be sized for; the message says why."""


def bucket_sizing(record: StandardMinerAccount, target_bucket: MinerBucket,
                  pro_account_size=None, payout_scale=None) -> StandardMinerAccount:
    """The record after a move into target_bucket, sized as EntityManager.apply_bucket_account_size sizes a
    subaccount, no fee. A standard bucket keeps any granted pro size; the first pro size given fixes payout_scale."""
    if not target_bucket.is_pro_track:
        return replace(record, trading_account_size=None)
    if pro_account_size is not None:
        error = pro_account_size_error(pro_account_size, record.account_size)
        if error:
            raise SizingRejected(error)
        if record.pro_account_size is None and record.payout_scale is None:
            record = replace(record, payout_scale=(payout_scale if payout_scale is not None
                                                   else ValiConfig.grow_payout_multiplier()))
    elif record.pro_account_size is not None:
        pro_account_size = record.pro_account_size
    else:
        raise SizingRejected("A pro account size is required to enter the pro track")
    return replace(record, pro_account_size=pro_account_size,
                   trading_account_size=pro_account_size if target_bucket.is_pro else None)
