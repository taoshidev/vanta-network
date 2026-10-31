"""
Converts between the legacy API shapes (Signal / Order with LIMIT, STOP_LIMIT, BRACKET) and
TriggerOrders, so the API keeps its request and response format.

  to trigger: signal_to_trigger_orders, bracket_to_trigger_orders, legacy_order_to_trigger_orders
  to legacy:  to_legacy_orders (Orders for to_python_dict / to_dashboard), to_bracket_dict
  stored:     sync_legacy_orders (checkpoints), migrate_legacy_disk (limit_orders/ on disk)

Raises ValueError for requests the legacy manager rejected.

uuids match today's: a LIMIT's brackets are groups "{uuid}-bracket-{i}", a STOP_LIMIT's limit is
"{uuid}-limit", and each bracket leg is "{group}-sl" / "-tp" / "-trail" in oco_group=group.
"""
import os

from shared_objects.log import logger
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.utils.limit_order.order_utils import OrderSize
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.vali_dataclasses.trigger_order import (
    FixedTrigger,
    MarketAction,
    PriceField,
    TrailingTrigger,
    TriggerAction,
    TriggerOrder,
)

UNFILLED_SOURCE = {
    ExecutionType.LIMIT: OrderSource.LIMIT_UNFILLED,
    ExecutionType.BRACKET: OrderSource.BRACKET_UNFILLED,
    ExecutionType.STOP_LIMIT: OrderSource.STOP_LIMIT_UNFILLED,
}


# ==================== Legacy -> trigger ====================

def signal_to_trigger_orders(hotkey: str, signal: Signal, order_uuid: str, now_ms: int,
                             position_type: OrderType | None = None) -> list[TriggerOrder]:
    """
    A LIMIT, STOP_LIMIT or BRACKET signal as the trigger orders to place. LIMIT and STOP_LIMIT
    give one order (brackets become children); a BRACKET gives its legs, which need the open
    position's side as position_type.
    """
    execution_type = signal.execution_type
    if execution_type == ExecutionType.LIMIT:
        actions = [_limit_action(signal, order_uuid)]
    elif execution_type == ExecutionType.STOP_LIMIT:
        if signal.order_type == OrderType.LONG and signal.limit_price < signal.stop_price:
            raise ValueError(
                f"STOP_LIMIT LONG orders require limit_price ({signal.limit_price}) >= stop_price ({signal.stop_price})"
            )
        if signal.order_type == OrderType.SHORT and signal.limit_price > signal.stop_price:
            raise ValueError(
                f"STOP_LIMIT SHORT orders require limit_price ({signal.limit_price}) <= stop_price ({signal.stop_price})"
            )
        actions = [TriggerAction(
            execution_type=ExecutionType.STOP_LIMIT,
            order_uuid=order_uuid,
            trigger=FixedTrigger(price=signal.stop_price, direction=signal.stop_condition, price_field=PriceField.MID),
            market_action=None,
            child_actions=[_limit_action(signal, f"{order_uuid}-limit")],
        )]
    elif execution_type == ExecutionType.BRACKET:
        if position_type not in (OrderType.LONG, OrderType.SHORT):
            raise ValueError("BRACKET orders need the open position's side")
        bracket = (signal.bracket_orders or [{}])[0] | _bracket_fields(signal)
        actions = _bracket_legs(bracket, position_type, order_uuid)
    else:
        raise ValueError(f"{execution_type} is not a trigger order")
    return [_to_order(action, hotkey, signal.trade_pair, now_ms) for action in actions]


def legacy_order_to_trigger_orders(hotkey: str, order: Order) -> list[TriggerOrder]:
    """
    A stored legacy Order (disk or checkpoint) as trigger orders. A BRACKET's order_type is its
    position's side; its legs bind to that position on the next tick. Cancelled orders keep src
    as their close reason.
    """
    if order.execution_type == ExecutionType.BRACKET:
        orders = bracket_to_trigger_orders(
            hotkey, order.trade_pair, _bracket_fields(order), order.order_uuid, order.order_type,
            order.processed_ms, best_price=order.price or None,
        )
    else:
        orders = signal_to_trigger_orders(hotkey, order, order.order_uuid, order.processed_ms)
    if OrderSource.is_cancelled(order.src):
        for o in orders:
            o.close_reason = OrderSource(order.src)
            o.closed_ms = order.processed_ms
    return orders


def bracket_to_trigger_orders(hotkey: str, trade_pair: TradePair, bracket: dict, group: str,
                              position_type: OrderType, now_ms: int,
                              best_price: float | None = None) -> list[TriggerOrder]:
    """
    One bracket dict (stop_loss / take_profit / trailing_* / size) as its legs on an open position.
    Used for brackets attached to a MARKET order (best_price = fill price) and bulk bracket edits.
    """
    return [
        _to_order(leg, hotkey, trade_pair, now_ms, best_price)
        for leg in _bracket_legs(bracket, position_type, group)
    ]


def _limit_action(signal: Signal, order_uuid: str) -> TriggerAction:
    if signal.order_type not in (OrderType.LONG, OrderType.SHORT):
        raise ValueError(f"{signal.order_type} order is not supported for LIMIT orders")
    long = signal.order_type == OrderType.LONG
    return TriggerAction(
        execution_type=ExecutionType.LIMIT,
        order_uuid=order_uuid,
        trigger=FixedTrigger(
            price=signal.limit_price,
            direction=StopCondition.LTE if long else StopCondition.GTE,
            price_field=PriceField.ASK if long else PriceField.BID,
        ),
        market_action=MarketAction(
            order_type=signal.order_type,
            size=_unsigned(OrderSize(quantity=signal.quantity, leverage=signal.leverage, value=signal.value)),
            is_maker=True,
        ),
        child_actions=[
            leg
            for i, bracket in enumerate(signal.bracket_orders or [])
            for leg in _bracket_legs(bracket, signal.order_type, f"{order_uuid}-bracket-{i}")
        ],
    )


def _bracket_fields(signal: Signal) -> dict:
    """A BRACKET signal's own SL/TP/trailing/size, as a bracket dict."""
    trailing = signal.trailing_stop or {}
    fields = {
        'stop_loss': signal.stop_loss,
        'take_profit': signal.take_profit,
        'trailing_percent': trailing.get('trailing_percent'),
        'trailing_value': trailing.get('trailing_value'),
        'leverage': signal.leverage,
        'value': signal.value,
        'quantity': signal.quantity,
        'bracket_pct': signal.bracket_pct,
    }
    return {k: v for k, v in fields.items() if v is not None}


def _bracket_legs(bracket: dict, position_type: OrderType, group: str) -> list[TriggerAction]:
    """SL, trailing SL and TP legs of one bracket, protecting a position_type position."""
    long = position_type == OrderType.LONG
    side = PriceField.BID if long else PriceField.ASK
    stop = StopCondition.LTE if long else StopCondition.GTE
    take = StopCondition.GTE if long else StopCondition.LTE

    size = OrderSize(
        quantity=_float(bracket.get('quantity')),
        leverage=_float(bracket.get('leverage')),
        value=_float(bracket.get('value')),
        bracket_pct=_float(bracket.get('bracket_pct')),
    )
    if size == OrderSize():
        size = OrderSize(bracket_pct=1.0)  # no size closes the whole position, as today

    def leg(suffix, trigger, is_maker=False):
        return TriggerAction(
            execution_type=ExecutionType.BRACKET,
            order_uuid=f"{group}-{suffix}",
            trigger=trigger,
            market_action=MarketAction(
                order_type=OrderType.SHORT if long else OrderType.LONG,
                size=_unsigned(size),
                is_maker=is_maker,
                reduce_only=True,
            ),
            bind_to_position=True,
            oco_group=group,
        )

    legs = []
    if bracket.get('stop_loss') is not None:
        legs.append(leg("sl", FixedTrigger(price=float(bracket['stop_loss']), direction=stop, price_field=side)))
    if bracket.get('trailing_percent') is not None or bracket.get('trailing_value') is not None:
        legs.append(leg("trail", TrailingTrigger(
            direction=stop, price_field=side,
            trailing_pct=_float(bracket.get('trailing_percent')),
            trailing_val=_float(bracket.get('trailing_value')),
        )))
    if bracket.get('take_profit') is not None:
        legs.append(leg("tp", FixedTrigger(price=float(bracket['take_profit']), direction=take, price_field=side), is_maker=True))
    if not legs:
        raise ValueError("BRACKET orders must have at least one of stop_loss, take_profit, or trailing_stop set")
    return legs


def _to_order(action: TriggerAction, hotkey: str, trade_pair: TradePair, now_ms: int,
              best_price: float | None = None) -> TriggerOrder:
    trigger = action.trigger.model_copy(deep=True)
    if isinstance(trigger, TrailingTrigger) and trigger.best_price is None and best_price:
        trigger.best_price = best_price
    return TriggerOrder(
        **action.model_dump(exclude={"trigger", "market_action", "child_actions"}),
        trigger=trigger,
        market_action=action.market_action,
        child_actions=action.child_actions,
        miner_hotkey=hotkey,
        trade_pair=trade_pair,
        processed_ms=now_ms,
    )


# ==================== Stored legacy orders ====================

def sync_legacy_orders(store, sync_data: dict) -> int:
    """Load a checkpoint's {trade_pair_id: {hotkey: [legacy order dicts]}} into the store."""
    orders = []
    for hotkey_dict in (sync_data or {}).values():
        for hotkey, order_dicts in (hotkey_dict or {}).items():
            orders.extend(_convert_dicts(hotkey, order_dicts))
    store.load(orders)
    return len(orders)


def migrate_legacy_disk(store, running_unit_tests=False) -> int:
    """
    Move orders saved by the legacy limit order manager ({hotkey}/limit_orders/) into the store.
    Each migrated folder is renamed to limit_orders_migrated/ so it isn't read twice.
    """
    miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
    if not os.path.exists(miner_dir):
        return 0
    migrated = 0
    for hotkey in ValiBkpUtils.get_directories_in_dir(miner_dir):
        legacy_dir = f"{miner_dir}{hotkey}/limit_orders"
        if not os.path.exists(legacy_dir):
            continue
        orders = []
        for status in ("unfilled", "cancelled"):
            orders.extend(_convert_dicts(hotkey, ValiBkpUtils.get_limit_orders(hotkey, status, running_unit_tests=running_unit_tests)))
        store.load(orders)
        os.rename(legacy_dir, f"{legacy_dir}_migrated")
        migrated += len(orders)
    if migrated:
        logger.info(f"[TRIGGER] Migrated {migrated} legacy limit orders to trigger orders")
    return migrated


def _convert_dicts(hotkey: str, order_dicts: list[dict]) -> list[TriggerOrder]:
    orders = []
    for order_dict in order_dicts:
        try:
            order = Order.from_dict(order_dict)
            if OrderSource.is_open(order.src) or OrderSource.is_cancelled(order.src):
                orders.extend(legacy_order_to_trigger_orders(hotkey, order))
        except Exception as e:
            logger.error(f"[TRIGGER] Could not convert legacy order for {hotkey}: {e} | {order_dict}")
    return orders


# ==================== Trigger -> legacy ====================

def to_legacy_orders(orders: list[TriggerOrder]) -> list[Order]:
    """
    Legacy Orders for the API, oldest first: one per LIMIT / STOP_LIMIT, and one BRACKET per
    oco_group. Cancelled orders keep their close reason as src and close time as processed_ms.
    """
    legacy = []
    seen_groups = set()
    for order in sorted(orders, key=lambda o: o.processed_ms):
        if order.execution_type == ExecutionType.BRACKET:
            key = (order.close_reason, legacy_uuid(order))
            if key in seen_groups:
                continue
            seen_groups.add(key)
            legs = [o for o in orders if o.close_reason == order.close_reason and legacy_uuid(o) == legacy_uuid(order)]
            legacy_order = Order.from_dict(to_bracket_dict(legs))
        else:
            legacy_order = _entry_to_order(order)

        if order.close_reason is not None:
            legacy_order.src = order.close_reason
            legacy_order.processed_ms = order.closed_ms or order.processed_ms
        legacy.append(legacy_order)
    return legacy


def _entry_to_order(order: TriggerOrder) -> Order:
    """A LIMIT or STOP_LIMIT trigger order as its legacy Order."""
    limit = order if order.execution_type == ExecutionType.LIMIT else order.child_actions[0]
    fill = limit.market_action
    sign = -1 if fill.order_type == OrderType.SHORT else 1
    stop_limit = order.execution_type == ExecutionType.STOP_LIMIT
    return Order(
        trade_pair=order.trade_pair,
        order_uuid=order.order_uuid,
        processed_ms=order.processed_ms,
        price=0.0,
        order_type=fill.order_type,
        leverage=sign * fill.size.leverage if fill.size.leverage is not None else None,
        value=sign * fill.size.value if fill.size.value is not None else None,
        quantity=sign * fill.size.quantity if fill.size.quantity is not None else None,
        execution_type=order.execution_type,
        limit_price=limit.trigger.price,
        stop_price=order.trigger.price if stop_limit else None,
        stop_condition=order.trigger.direction if stop_limit else None,
        bracket_orders=_bracket_orders(limit.child_actions) or None,
        src=UNFILLED_SOURCE[order.execution_type],
    )


def _bracket_orders(legs: list[TriggerAction]) -> list[dict]:
    """bracket_orders entries rebuilt from a LIMIT's leg templates, one per oco_group."""
    groups: dict[str, list[TriggerAction]] = {}
    for leg in legs:
        groups.setdefault(leg.oco_group or leg.order_uuid, []).append(leg)
    entries = []
    for group, group_legs in groups.items():
        bracket = to_bracket_dict(group_legs)
        entries.append({
            'order_uuid': group,
            'stop_loss': bracket['stop_loss'],
            'take_profit': bracket['take_profit'],
            **(bracket['trailing_stop'] or {}),
            'leverage': bracket['leverage'],
            'value': bracket['value'],
            'quantity': bracket['quantity'],
            'bracket_pct': bracket['bracket_pct'],
        })
    return entries


def legacy_uuid(order: TriggerAction) -> str:
    """uuid of the legacy bracket an order belongs to: its oco_group, or itself."""
    return order.oco_group or order.order_uuid


def _leg(order: TriggerAction) -> tuple[OrderType, str]:
    """(protected position type, 'stop' | 'take' | 'trail') for a reduce_only bracket leg."""
    action = order.market_action
    trigger = order.trigger
    if isinstance(trigger, TrailingTrigger):
        kind = 'trail'
    elif action.order_type == OrderType.FLAT:  # TPs rest on the book, SLs sweep it
        kind = 'take' if action.is_maker else 'stop'
    else:
        stop_direction = StopCondition.LTE if action.order_type == OrderType.SHORT else StopCondition.GTE
        kind = 'stop' if trigger.direction == stop_direction else 'take'

    if action.order_type == OrderType.SHORT:
        return OrderType.LONG, kind
    if action.order_type == OrderType.LONG:
        return OrderType.SHORT, kind
    losing = StopCondition.GTE if kind == 'take' else StopCondition.LTE
    return (OrderType.LONG if trigger.direction == losing else OrderType.SHORT), kind


def to_bracket_dict(legs: list[TriggerAction]) -> dict:
    """
    One legacy BRACKET Order dict merging the legs of an oco_group (SL, TP, trailing SL). Also
    the copy kept on Position.unfilled_orders.
    """
    first = legs[0]
    fields = {'stop': None, 'take': None, 'trail': None}
    position_type = None
    for leg in legs:
        position_type, kind = _leg(leg)
        fields[kind] = leg.trigger

    trailing = fields['trail']
    trailing_stop = None
    if trailing is not None:
        trailing_stop = (
            {'trailing_percent': trailing.trailing_pct} if trailing.trailing_pct is not None
            else {'trailing_value': trailing.trailing_val}
        )

    size = first.market_action.size
    return {
        'trade_pair_id': first.trade_pair.trade_pair_id if isinstance(first, TriggerOrder) else None,
        'order_uuid': legacy_uuid(first),
        'processed_ms': min(getattr(leg, 'processed_ms', 0) for leg in legs),
        'price': (trailing.best_price or 0.0) if trailing is not None else 0.0,
        'order_type': position_type.value,
        'leverage': abs(size.leverage) if size.leverage is not None else None,
        'value': abs(size.value) if size.value is not None else None,
        'quantity': abs(size.quantity) if size.quantity is not None else None,
        'bracket_pct': size.bracket_pct,
        'execution_type': ExecutionType.BRACKET.value,
        'stop_loss': fields['stop'].price if fields['stop'] is not None else None,
        'take_profit': fields['take'].price if fields['take'] is not None else None,
        'trailing_stop': trailing_stop,
        'src': OrderSource.BRACKET_UNFILLED,
    }


def _unsigned(size: OrderSize) -> OrderSize:
    return OrderSize(*(abs(v) if v is not None else None for v in size))


def _float(value) -> float | None:
    return float(value) if value is not None else None
