"""
Stateless trigger evaluation for limit, stop-limit, and bracket (SL/TP) orders.

Bracket SL/TP rules:
  LONG  SL: min_bid <= SL    LONG  TP: max_bid >= TP
  SHORT SL: max_ask >= SL    SHORT TP: min_ask <= TP

Trailing-stop best-price tracking mutates `order.price` in place; otherwise
this module reads but does not retain state.
"""
from typing import NamedTuple

import bittensor as bt

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_dataclasses.price_source import PriceSource
from shared_objects.log import logger


# Extrema across the price-source window. LIMIT/STOP_LIMIT use aggressive
# matching: max_bid_ps for SHORT/LTE legs, min_ask_ps for LONG/GTE legs.
class LimitPriceSources(NamedTuple):
    min_bid_ps: PriceSource
    max_bid_ps: PriceSource
    min_ask_ps: PriceSource
    max_ask_ps: PriceSource


def build_limit_price_sources(price_sources, cutoff_ms: int = 0):
    """
    Compute bid/ask extrema across price sources that satisfy the time constraint
    (start_ms >= cutoff_ms). cutoff_ms=0 (default) applies no filtering.
    Returns LimitPriceSources, or None if no valid sources remain.
    """
    valid = [ps for ps in price_sources if ps.start_ms >= cutoff_ms] if cutoff_ms else price_sources
    if not valid:
        return None
    bid_sorted = sorted(valid, key=lambda ps: ps.bid if ps.bid > 0 else ps.open, reverse=True)
    ask_sorted = sorted(valid, key=lambda ps: ps.ask if ps.ask > 0 else ps.open)
    max_bid_ps, min_bid_ps = bid_sorted[0], bid_sorted[-1]
    min_ask_ps, max_ask_ps = ask_sorted[0], ask_sorted[-1]
    if max_bid_ps.bid == 0 or min_ask_ps.ask == 0:
        return None
    return LimitPriceSources(
        min_bid_ps=min_bid_ps, max_bid_ps=max_bid_ps,
        min_ask_ps=min_ask_ps, max_ask_ps=max_ask_ps,
    )


def evaluate_order_trigger(hotkey: str, order: Order, position: Position | None, price_sources, cutoff_ms: int = 0):
    """
    Dispatch to the right evaluator. Returns (trigger_ps, trigger_price, is_taker).

    is_taker reports whether the fill takes liquidity: False for orders that rested
    on the book (LIMIT, bracket TP), True for those that sweep it (bracket SL).

    price_sources: full list of price-source objects for the look-back window.
    cutoff_ms: only sources with start_ms >= cutoff_ms are used to build extrema.
    """
    sources = build_limit_price_sources(price_sources, cutoff_ms)
    if sources is None:
        return None, None, None

    trigger_ps, trigger_price, is_taker = None, None, None

    if order.execution_type == ExecutionType.LIMIT:
        # LONG buys at min_ask, SHORT sells at max_bid (aggressive matching).
        trigger_ps = sources.min_ask_ps if order.order_type == OrderType.LONG else sources.max_bid_ps
        trigger_price = evaluate_limit_trigger_price(order, trigger_ps)
        is_taker = False

    elif order.execution_type == ExecutionType.BRACKET:
        if not position:
            return None, None, None
        # Bracket evaluator returns the actual ps for the leg that fired.
        trigger_ps, trigger_price, is_taker = evaluate_bracket_trigger_price(order, position, sources)

    elif order.execution_type == ExecutionType.STOP_LIMIT:
        # GTE triggers on upside breakout: min_ask is higher of the two and more favorable.
        # LTE triggers on downside breakout: max_bid is lower of the two and more favorable.
        trigger_ps = sources.min_ask_ps if order.stop_condition == StopCondition.GTE else sources.max_bid_ps
        trigger_price = evaluate_stop_limit_trigger_price(order, trigger_ps)
        is_taker = True

    if trigger_price:
        logger.info(
            f"{order.execution_type} triggered: {hotkey} {order.trade_pair.trade_pair_id} "
            f"{order.order_uuid} trigger_price={trigger_price} price_source={trigger_ps}"
        )

    if trigger_price is None:
        is_taker = None

    return trigger_ps, trigger_price, is_taker


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
        return order.stop_price
    if order.stop_condition == StopCondition.LTE and mid_price <= order.stop_price:
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
    Return (trigger_ps, trigger_price, is_taker) if SL/TP boundary is hit, else (None, None, None).

    The SL leg sweeps the book on trigger (is_taker=True); the TP leg rests on it
    like a plain limit (is_taker=False).

    The returned trigger_ps is the actual extremum that crossed the boundary
    (not just the favorable side), so callers can correctly gate on its
    start_ms for staleness.

    Callers must invoke update_trailing_best_price() beforehand for trailing-stop
    orders; this function reads order.price but does not mutate it.
    """
    if not position:
        return None, None, None

    if order.processed_ms < position.open_ms:
        logger.info(
            f"[BRACKET CANCELLED] Bracket {order.order_uuid} (processed_ms={order.processed_ms}) "
            f"predates current position (open_ms={position.open_ms}), skipping trigger as orphan"
        )
        return None, None, None

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
        min_bid_ps = sources.min_bid_ps
        max_bid_ps = sources.max_bid_ps
        min_bid = min_bid_ps.bid if min_bid_ps.bid > 0 else min_bid_ps.open
        max_bid = max_bid_ps.bid if max_bid_ps.bid > 0 else max_bid_ps.open
        if effective_sl is not None and min_bid <= effective_sl:
            return min_bid_ps, effective_sl, True
        if order.take_profit is not None and max_bid >= order.take_profit:
            return max_bid_ps, order.take_profit, False

    elif position_type == OrderType.SHORT:
        min_ask_ps = sources.min_ask_ps
        max_ask_ps = sources.max_ask_ps
        max_ask = max_ask_ps.ask if max_ask_ps.ask > 0 else max_ask_ps.open
        min_ask = min_ask_ps.ask if min_ask_ps.ask > 0 else min_ask_ps.open
        if effective_sl is not None and max_ask >= effective_sl:
            return max_ask_ps, effective_sl, True
        if order.take_profit is not None and min_ask <= order.take_profit:
            return min_ask_ps, order.take_profit, False

    return None, None, None
