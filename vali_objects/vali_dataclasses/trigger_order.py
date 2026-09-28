# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
"""
Trigger order primitive: a trigger (price + direction) and the fill and/or attached child
orders to place when it fires.

Every order is either MARKET (fills now) or a trigger order. A trigger order fires a MARKET
fill and/or arms attached child trigger orders when its trigger crosses; limit prices for
those children live on their own trigger, not on the fill.

Replaces LIMIT / STOP_LIMIT / BRACKET as the internal representation (long side shown):
  LIMIT @ P            -> trigger=ASK LTE P, fill=MARKET LONG fill_at_trigger
  STOP_LIMIT (S, L)    -> trigger=MID GTE S, attached=[{trigger=ASK LTE L, fill=MARKET LONG fill_at_trigger}]
  BRACKET SL           -> trigger=BID LTE SL (fixed or trailing), fill=MARKET FLAT reduce_only
  BRACKET TP           -> trigger=BID GTE TP, fill=MARKET FLAT reduce_only fill_at_trigger
  LIMIT @ P + SL/TP    -> trigger=ASK LTE P, fill=MARKET LONG fill_at_trigger,
                          attached=[{SL}, {TP}]  (attached children share an oco_group)

A trigger order carries at most one fill (what happens when *its own* trigger fires) plus zero
or more attached child trigger orders (armed once this trigger fires, each with their own
trigger and, recursively, their own fill/attached). This mirrors how brokers actually compose
conditional orders -- IBKR bracket orders, Binance/Bybit "order with TP/SL", Hyperliquid grouped
orders -- rather than allowing an arbitrary list of unrelated actions per trigger.

execution_type records the API-facing type (LIMIT / STOP_LIMIT / BRACKET) so responses such
as to_dashboard keep their existing shape.

Lifecycle: an order is open until it fires or is cancelled. Firing runs the fill (if any) and
arms/places the attached children, then closes the order; the fill becomes an Order on the
Position, placed children become new TriggerOrders, and cancelled orders are persisted to the
cancelled/ directory.
"""
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.utils.limit_order.order_utils import OrderSize
from vali_objects.vali_config import TradePair


class PriceField(str, Enum):
    """Which side of the book the trigger observes."""
    BID = "BID"
    ASK = "ASK"
    MID = "MID"


class FixedTrigger(BaseModel):
    kind: Literal["fixed"] = "fixed"
    price: float = Field(gt=0)
    direction: StopCondition  # GTE fires when observed >= price, LTE when observed <= price
    price_field: PriceField = PriceField.MID


class TrailingTrigger(BaseModel):
    """
    price is the level the trigger fires at; the evaluator only ever reads it.
    update_trailing_price() moves price along with best_price as new observations come in. For
    LTE (protects a LONG) best_price is the high-water mark and price sits below it; for GTE
    (protects a SHORT) best_price is the low-water mark and price sits above it. best_price and
    price start unset and are seeded from the first observation.
    """
    kind: Literal["trailing"] = "trailing"
    price: float = Field(gt=0)
    direction: StopCondition
    price_field: PriceField = PriceField.MID

    trailing_pct: float | None = Field(default=None, gt=0, lt=1)
    trailing_val: float | None = Field(default=None, gt=0)
    best_price: float | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def validate_trailing_amount(self):
        if (self.trailing_pct is None) == (self.trailing_val is None):
            raise ValueError("TrailingTrigger requires exactly one of trailing_pct or trailing_val")
        return self

    def update_trailing_price(self, observed: float) -> None:
        if self.best_price is None:
            self.best_price = observed

        if self.direction == StopCondition.LTE:
            self.best_price = max(self.best_price, observed)
            if self.trailing_pct is not None:
                self.price = self.best_price * (1 - self.trailing_pct)
            else:
                self.price = self.best_price - self.trailing_val
        else:
            self.best_price = min(self.best_price, observed)
            if self.trailing_pct is not None:
                self.price = self.best_price * (1 + self.trailing_pct)
            else:
                self.price = self.best_price + self.trailing_val


Trigger = Annotated[
    FixedTrigger | TrailingTrigger,
    Field(discriminator="kind"),
]


class MarketAction(BaseModel):
    order_type: OrderType
    size: OrderSize
    fill_at_trigger: bool = False  # fill at the trigger price (maker) rather than the observed price (taker)
    reduce_only: bool = False      # may only shrink the bound position

    @model_validator(mode='after')
    def validate_size(self):
        if self.size.bracket_pct is not None and not self.reduce_only:
            raise ValueError("bracket_pct sizing is only valid for reduce_only actions")
        return self

    @property
    def is_taker(self) -> bool:
        return not self.fill_at_trigger


class TriggerOrder(BaseModel):
    """
    A top-level trigger order, or a child template attached to another trigger order. A child's
    identity fields are None until the parent fires and places it; the parent assigns order_uuid
    f"{parent_uuid}-{index}" and processed_ms = fire time, so the child only sees prices from
    after the parent fired.
    """
    execution_type: ExecutionType  # API-facing type, kept for backwards-compatible responses

    order_uuid: str | None = None        # None on a child template
    miner_hotkey: str | None = None      # None on a child template
    trade_pair: TradePair | None = None  # None on a child template; inherits the parent's

    processed_ms: int | None = None  # None on a child template
    closed_ms: int | None = None     # set on fire or cancel; keeps processed_ms as placement time

    trigger: Trigger
    fill: MarketAction | None = None              # executed when this trigger fires
    attached: list["TriggerOrder"] = Field(default_factory=list)  # armed when this trigger fires

    oco_group: str | None = None      # a sibling firing cancels the others
    position_uuid: str | None = None  # bound position; cancel if it closes or flips
    bind_to_position: bool = False    # child templates: bind to the position open when placed

    @field_validator('trade_pair', mode='before')
    @classmethod
    def convert_trade_pair(cls, v):
        if isinstance(v, str):
            return TradePair.from_trade_pair_id(v)
        if isinstance(v, dict) and 'trade_pair_id' in v:
            return TradePair.from_trade_pair_id(v['trade_pair_id'])
        if isinstance(v, list) and len(v) >= 1:
            return TradePair.from_trade_pair_id(v[0])
        return v

    @model_validator(mode='after')
    def validate_binding(self):
        is_bound = self.position_uuid is not None or self.bind_to_position
        if self.fill is None and not self.attached:
            raise ValueError("TriggerOrder requires a fill, attached orders, or both")
        if isinstance(self.trigger, TrailingTrigger):
            if not is_bound or self.fill is None or not self.fill.reduce_only or self.attached:
                raise ValueError("Trailing triggers require a bound position and a single reduce_only MARKET fill")
        if self.fill is not None and self.fill.reduce_only and not is_bound:
            raise ValueError("reduce_only fills require a bound position")
        return self

    @property
    def is_open(self) -> bool:
        return self.closed_ms is None


TriggerOrder.model_rebuild()
