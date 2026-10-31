"""
Trigger order primitive: a trigger (price + direction) plus a fill and/or child orders to place
when it fires. Every order is either MARKET or a trigger order.

Replaces LIMIT / STOP_LIMIT / BRACKET internally (long side shown):
  LIMIT @ P          -> trigger=ASK LTE P, fill=MARKET LONG is_maker
  STOP_LIMIT (S, L)  -> trigger=MID <stop_condition> S, children=[LIMIT @ L]
  BRACKET SL         -> trigger=BID LTE SL (fixed or trailing), fill=MARKET SHORT reduce_only bracket_pct
  BRACKET TP         -> trigger=BID GTE TP, fill=MARKET SHORT reduce_only bracket_pct is_maker
  LIMIT @ P + SL/TP  -> LIMIT @ P, children=[SL, TP] bound to the filled position

The legs of one bracket (SL, TP, trailing SL) are independent orders bound to the position; they
are cancelled when it closes or flips, so a full-size SL or TP firing cancels the rest. oco_group
is set by the user: firing one order in the group drops the rest. execution_type keeps the
API-facing type for backwards-compatible responses.
"""
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_source_enum import OrderSource
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
    price_field: PriceField


class TrailingTrigger(BaseModel):
    """
    Fires at price, which trails best_price by trailing_pct or trailing_val. LTE (protects a LONG)
    tracks the high-water mark; GTE (protects a SHORT) tracks the low-water mark. best_price is
    unset on a template and seeded on placement: the parent's fill price, or the current price.
    """
    kind: Literal["trailing"] = "trailing"
    direction: StopCondition
    price_field: PriceField

    best_price: float | None = Field(default=None, gt=0)
    trailing_pct: float | None = Field(default=None, gt=0, lt=1)
    trailing_val: float | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def validate_trailing_amount(self):
        if (self.trailing_pct is None) == (self.trailing_val is None):
            raise ValueError("TrailingTrigger requires exactly one of trailing_pct or trailing_val")
        return self

    @property
    def price(self) -> float:
        if self.best_price is None:
            raise ValueError("TrailingTrigger best_price is not seeded")
        sign = 1 if self.direction == StopCondition.GTE else -1
        if self.trailing_pct is not None:
            return self.best_price * (1 + sign * self.trailing_pct)
        return self.best_price + sign * self.trailing_val

    def update_best_price(self, observed: float) -> None:
        if self.best_price is None:
            self.best_price = observed
        elif self.direction == StopCondition.LTE:
            self.best_price = max(self.best_price, observed)
        else:
            self.best_price = min(self.best_price, observed)


Trigger = Annotated[
    FixedTrigger | TrailingTrigger,
    Field(discriminator="kind"),
]


class MarketAction(BaseModel):
    order_type: OrderType
    size: OrderSize
    is_maker: bool = False     # charged maker (rested on the book) rather than taker
    reduce_only: bool = False  # may only shrink the bound position

    @model_validator(mode='after')
    def validate_size(self):
        if self.size.bracket_pct is not None and not self.reduce_only:
            raise ValueError("bracket_pct sizing is only valid for reduce_only actions")
        return self

    @property
    def is_taker(self) -> bool:
        return not self.is_maker


class TriggerAction(BaseModel):
    """A trigger order template: what fires, what it fills, and which children it places."""
    execution_type: ExecutionType  # API-facing type, kept for backwards-compatible responses
    order_uuid: str                # client-known uuid; the placed TriggerOrder keeps it

    trigger: Trigger
    market_action: MarketAction | None
    child_actions: list["TriggerAction"] = Field(default_factory=list)  # placed when trigger fires
    bind_to_position: bool = False  # bind to the trade pair's position when placed
    oco_group: str | None = None    # firing one order in the group drops the rest

    @model_validator(mode='after')
    def validate_actions(self):
        if not self.market_action and not self.child_actions:
            raise ValueError("TriggerAction requires a fill, child actions, or both")
        if self.market_action and self.market_action.reduce_only and not self.bind_to_position:
            raise ValueError("reduce_only actions must bind to a position")
        return self


class TriggerOrder(TriggerAction):
    """
    A placed trigger order. Open until it fires or is cancelled; closed_ms is set on either.
    Children are placed with processed_ms = fire time, so they only see later prices.
    """
    miner_hotkey: str
    trade_pair: TradePair

    processed_ms: int
    closed_ms: int | None = None
    close_reason: OrderSource | None = None  # set on cancel
    position_uuid: str | None = None         # bound position; cancel if it closes or flips

    @field_validator('trade_pair', mode='before')
    @classmethod
    def parse_trade_pair(cls, v):
        return TradePair.from_trade_pair_id(v) if isinstance(v, str) else v

    @field_serializer('trade_pair')
    def serialize_trade_pair(self, trade_pair: TradePair) -> str:
        return trade_pair.trade_pair_id

    def create_child_orders(self, fire_ms: int, fill_price: float, position_uuid: str | None) -> list["TriggerOrder"]:
        """Place child_actions at fire_ms. Trailing triggers are seeded with fill_price."""
        children = []
        for action in self.child_actions:
            trigger = action.trigger.model_copy(deep=True)
            if isinstance(trigger, TrailingTrigger) and trigger.best_price is None:
                trigger.best_price = fill_price
            children.append(TriggerOrder(
                **action.model_dump(exclude={"trigger", "market_action", "child_actions"}),
                trigger=trigger,
                market_action=action.market_action,
                child_actions=action.child_actions,
                miner_hotkey=self.miner_hotkey,
                trade_pair=self.trade_pair,
                processed_ms=fire_ms,
                position_uuid=position_uuid if action.bind_to_position else None,
            ))
        return children


TriggerAction.model_rebuild()
TriggerOrder.model_rebuild()
