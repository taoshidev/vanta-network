"""
Stateless trigger evaluation for limit, stop-limit, and bracket (SL/TP) orders.

Bracket SL/TP rules:
  LONG  SL: min_bid <= SL    LONG  TP: max_bid >= TP
  SHORT SL: max_ask >= SL    SHORT TP: min_ask <= TP

Trailing-stop best-price tracking mutates `order.price` in place; otherwise
this module reads but does not retain state.
"""
from collections import namedtuple

import bittensor as bt

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType, StopCondition


# Extrema across the price-source window. LIMIT/STOP_LIMIT use aggressive
# matching: max_bid_ps for SHORT/LTE legs, min_ask_ps for LONG/GTE legs.
LimitPriceSources = namedtuple(
    "LimitPriceSources",
    ["min_bid_ps", "max_bid_ps", "min_ask_ps", "max_ask_ps"],
)


def single_source(ps):
    """Build a LimitPriceSources where every extremum is the same single source."""
    return LimitPriceSources(
        min_bid_ps=ps, max_bid_ps=ps,
        min_ask_ps=ps, max_ask_ps=ps,
    )


def evaluate_order_trigger(order, position, sources):
    """Dispatch to the right evaluator. Returns (trigger_ps, trigger_price)."""
    trigger_ps, trigger_price = None, None

    if order.execution_type == ExecutionType.LIMIT:
        # LONG buys at min_ask, SHORT sells at max_bid (aggressive matching).
        trigger_ps = sources.min_ask_ps if order.order_type == OrderType.LONG else sources.max_bid_ps
        trigger_price = evaluate_limit_trigger_price(order, trigger_ps)

    elif order.execution_type == ExecutionType.BRACKET:
        if not position:
            return None, None
        # Bracket evaluation picks the right extremum per (side, leg) inside.
        # trigger_ps is reported as the same single-extremum used for trailing updates.
        trigger_ps = sources.max_bid_ps if position.position_type == OrderType.LONG else sources.min_ask_ps
        trigger_price = evaluate_bracket_trigger_price(order, position, sources)

    elif order.execution_type == ExecutionType.STOP_LIMIT:
        # GTE triggers on upside breakout: min_ask is higher of the two and more favorable.
        # LTE triggers on downside breakout: max_bid is lower of the two and more favorable.
        trigger_ps = sources.min_ask_ps if order.stop_condition == StopCondition.GTE else sources.max_bid_ps
        trigger_price = evaluate_stop_limit_trigger_price(order, trigger_ps)

    if trigger_price:
        bt.logging.info(
            f"{order.execution_type} triggered: {order.trade_pair.trade_pair_id} "
            f"{order.order_uuid} trigger_price={trigger_price} price_source={trigger_ps}"
        )

    return trigger_ps, trigger_price


def evaluate_limit_trigger_price(order, ps):
    """Return limit_price if triggered, else None."""
    bid_price = ps.bid if ps.bid > 0 else ps.open
    ask_price = ps.ask if ps.ask > 0 else ps.open
    limit_price = order.limit_price

    if order.order_type == OrderType.LONG:
        return limit_price if ask_price <= limit_price else None
    if order.order_type == OrderType.SHORT:
        return limit_price if bid_price >= limit_price else None
    return None


def evaluate_stop_limit_trigger_price(order, ps):
    """Return stop_price if triggered (mid-price crosses stop_price per stop_condition)."""
    bid_price = ps.bid if ps.bid > 0 else ps.open
    ask_price = ps.ask if ps.ask > 0 else ps.open
    mid_price = (bid_price + ask_price) / 2

    if order.stop_condition == StopCondition.GTE and mid_price >= order.stop_price:
        bt.logging.info(f"Stop-limit triggered (GTE): mid={mid_price} >= stop_price={order.stop_price}")
        return order.stop_price
    if order.stop_condition == StopCondition.LTE and mid_price <= order.stop_price:
        bt.logging.info(f"Stop-limit triggered (LTE): mid={mid_price} <= stop_price={order.stop_price}")
        return order.stop_price
    return None


def _compute_trailing_sl(order, position_type):
    """Compute trailing stop loss price from order.price (assumed already updated)."""
    if order.trailing_stop is None or order.price <= 0:
        return None

    trailing_percent = order.trailing_stop.get('trailing_percent')
    trailing_value = order.trailing_stop.get('trailing_value')

    if position_type == OrderType.LONG:
        if trailing_percent is not None:
            return order.price * (1 - float(trailing_percent))
        return order.price - float(trailing_value)
    if position_type == OrderType.SHORT:
        if trailing_percent is not None:
            return order.price * (1 + float(trailing_percent))
        return order.price + float(trailing_value)
    return None


def evaluate_bracket_trigger_price(order, position, sources):
    """
    Return trigger price if SL/TP boundary is hit, else None.

    Callers must invoke update_trailing_best_price() beforehand for trailing-stop
    orders; this function reads order.price but does not mutate it.
    """
    if not position:
        return None

    if order.processed_ms < position.open_ms:
        bt.logging.info(
            f"[BRACKET CANCELLED] Bracket {order.order_uuid} (processed_ms={order.processed_ms}) "
            f"predates current position (open_ms={position.open_ms}), skipping trigger as orphan"
        )
        return None

    position_type = position.position_type
    order.order_type = position_type

    trailing_sl = _compute_trailing_sl(order, position_type)

    # Effective stop loss: more protective of static SL vs trailing SL.
    effective_sl = order.stop_loss
    if trailing_sl is not None:
        if effective_sl is None:
            effective_sl = trailing_sl
        elif position_type == OrderType.LONG:
            effective_sl = max(effective_sl, trailing_sl)
        elif position_type == OrderType.SHORT:
            effective_sl = min(effective_sl, trailing_sl)

    if position_type == OrderType.LONG:
        min_bid = sources.min_bid_ps.bid if sources.min_bid_ps.bid > 0 else sources.min_bid_ps.open
        max_bid = sources.max_bid_ps.bid if sources.max_bid_ps.bid > 0 else sources.max_bid_ps.open
        if effective_sl is not None and min_bid <= effective_sl:
            bt.logging.info(f"Bracket order stop loss triggered: min_bid={min_bid} <= SL={effective_sl}")
            return effective_sl
        if order.take_profit is not None and max_bid >= order.take_profit:
            bt.logging.info(f"Bracket order take profit triggered: max_bid={max_bid} >= TP={order.take_profit}")
            return order.take_profit

    elif position_type == OrderType.SHORT:
        max_ask = sources.max_ask_ps.ask if sources.max_ask_ps.ask > 0 else sources.max_ask_ps.open
        min_ask = sources.min_ask_ps.ask if sources.min_ask_ps.ask > 0 else sources.min_ask_ps.open
        if effective_sl is not None and max_ask >= effective_sl:
            bt.logging.info(f"Bracket order stop loss triggered: max_ask={max_ask} >= SL={effective_sl}")
            return effective_sl
        if order.take_profit is not None and min_ask <= order.take_profit:
            bt.logging.info(f"Bracket order take profit triggered: min_ask={min_ask} <= TP={order.take_profit}")
            return order.take_profit

    return None
