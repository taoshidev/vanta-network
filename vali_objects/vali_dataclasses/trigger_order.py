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

from pydantic import BaseModel, Field, model_validator

from time_util.time_util import TimeUtil
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
    direction: StopCondition
    price_field: PriceField = PriceField.MID

    best_price: float
    trailing_pct: float | None = Field(default=None, gt=0, lt=1)
    trailing_val: float | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def validate_trailing_amount(self):
        if (self.trailing_pct is None) == (self.trailing_val is None):
            raise ValueError("TrailingTrigger requires exactly one of trailing_pct or trailing_val")
        return self

    @property
    def price(self) -> float:
        sign = 1 if self.direction == StopCondition.GTE else -1
        if self.trailing_pct is not None:
            return self.best_price * (1 + sign * self.trailing_pct)
        elif self.trailing_val is not None:
            return self.best_price + sign * self.trailing_val
        else:
            raise ValueError("TrailingTrigger missing both trailing_pct and trailing_val")

    def update_best_price(self, observed: float) -> None:
        if self.direction == StopCondition.LTE:
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


class TriggerAction(BaseModel):
    execution_type: ExecutionType = ExecutionType.LIMIT
    order_uuid: str  # pending actions need uuid for v2 api

    trigger: Trigger
    market_action: MarketAction | None
    child_actions: list["TriggerAction"] = Field(default_factory=list)
    bind_to_position: bool = False  # Attach child actions to live trade pair position

    @model_validator(mode='after')
    def validate_actions(self):
        if not self.market_action and not self.child_actions:
            raise ValueError("TriggerAction requires a fill, attached child actions, or both")
        return self


class TriggerOrder(BaseModel):
    """
    A top-level trigger order, or a child template attached to another trigger order. A child's
    identity fields are None until the parent fires and places it; the parent assigns order_uuid
    f"{parent_uuid}-{index}" and processed_ms = fire time, so the child only sees prices from
    after the parent fired.
    """
    execution_type: ExecutionType  # API-facing type, kept for backwards-compatible responses

    order_uuid: str
    miner_hotkey: str
    trade_pair: TradePair

    processed_ms: int
    closed_ms: int | None = None     # set on fire or cancel; keeps processed_ms as placement time

    trigger: Trigger
    market_action: MarketAction | None
    child_actions: list[TriggerAction] = Field(default_factory=list)  # armed when trigger fires
    bind_to_position: bool = False

    # oco_group: str | None = None      # a sibling firing cancels the others
    position_uuid: str | None = None  # bound position; cancel if it closes or flips

    @model_validator(mode='after')
    def validate_actions(self):
        if not self.market_action and not self.child_actions:
            raise ValueError("TriggerOrder requires a fill, attached child actions, or both")
        return self

    def create_child_orders(self, linked_position_uuid) -> list["TriggerOrder"]:
        orders = []
        time_ms = self.closed_ms or TimeUtil.now_in_millis()
        for i, action in enumerate(self.child_actions):
            new_order = TriggerOrder(
                execution_type=action.execution_type,
                order_uuid=f"{action.order_uuid}-trigger-{i}",
                miner_hotkey=self.miner_hotkey,
                trade_pair=self.trade_pair,
                processed_ms=time_ms,
                trigger=action.trigger,
                market_action=action.market_action,
                child_actions=action.child_actions,
                bind_to_position=action.bind_to_position,
                position_uuid=linked_position_uuid if self.bind_to_position else None,
            )
            orders.append(new_order)

        return orders


TriggerOrder.model_rebuild()
