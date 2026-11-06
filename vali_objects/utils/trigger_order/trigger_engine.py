"""
Stateless trigger evaluation. A trigger fires when the most extreme observed price in the window
crosses it: the lowest for LTE, the highest for GTE. Trailing updates are the only mutation.
"""
from typing import NamedTuple

from vali_objects.enums.order_type_enum import StopCondition
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.trigger_order import PriceField, TrailingTrigger, Trigger


class Fire(NamedTuple):
    price_source: PriceSource  # the observation that crossed
    trigger_price: float       # caps the fill so it is never worse than the trigger


def observed_price(ps: PriceSource, field: PriceField) -> float:
    """Bid, ask, or mid of a price source, falling back to open when a side is missing."""
    bid = ps.bid if ps.bid and ps.bid > 0 else ps.open
    ask = ps.ask if ps.ask and ps.ask > 0 else ps.open
    if field == PriceField.BID:
        return bid
    if field == PriceField.ASK:
        return ask
    return (bid + ask) / 2


def would_fire(trigger: Trigger, price: float) -> bool:
    if trigger.direction == StopCondition.GTE:
        return price >= trigger.price
    return price <= trigger.price


def _extreme(trigger: Trigger, price_sources, cutoff_ms: int, lowest: bool):
    """(ps, price) with the lowest or highest observed price at or after cutoff_ms."""
    observed = [
        (ps, observed_price(ps, trigger.price_field))
        for ps in price_sources
        if ps.start_ms >= cutoff_ms
    ]
    observed = [(ps, price) for ps, price in observed if price and price > 0]
    if not observed:
        return None
    return (min if lowest else max)(observed, key=lambda o: o[1])


def evaluate(trigger: Trigger, price_sources, cutoff_ms: int = 0) -> Fire | None:
    """Fire if the trigger crossed in any price source at or after cutoff_ms."""
    extreme = _extreme(trigger, price_sources, cutoff_ms, lowest=trigger.direction == StopCondition.LTE)
    if extreme is None:
        return None
    ps, price = extreme
    return Fire(ps, trigger.price) if would_fire(trigger, price) else None


def update_trailing(trigger: Trigger, price_sources, cutoff_ms: int = 0) -> bool:
    """Move a trailing trigger's best_price with the window. Returns True if it changed."""
    if not isinstance(trigger, TrailingTrigger):
        return False
    # LTE trails the high-water mark, GTE the low-water mark
    extreme = _extreme(trigger, price_sources, cutoff_ms, lowest=trigger.direction == StopCondition.GTE)
    if extreme is None:
        return False
    before = trigger.best_price
    trigger.update_best_price(extreme[1])
    return trigger.best_price != before
