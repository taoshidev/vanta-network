import math
from typing import NamedTuple

from vali_objects.vali_config import ValiConfig
from vali_objects.trade_pair import TradePair


class OrderSize(NamedTuple):
    quantity: float | None = None
    leverage: float | None = None
    value: float | None = None
    bracket_pct: float | None = None

    @classmethod
    def from_dict(cls, _dict: dict) -> "OrderSize":
        def to_float(value):
            return float(value) if value is not None else None

        return cls(
            quantity=to_float(_dict.get("quantity")),
            value=to_float(_dict.get("value")),
            leverage=to_float(_dict.get("leverage")),
            bracket_pct=to_float(_dict.get("bracket_pct")),
        )


def convert_order_sizes(
    order_size: OrderSize,
    usd_base_conversion: float,
    trade_pair: TradePair,
    balance: float,
    round_qty=True,
    use_floor=False,
    use_nano_increment=False,
):
    """
    parses an order signal and calculates leverage, value, and quantity
    """
    leverage = order_size.leverage
    value = order_size.value
    quantity = order_size.quantity

    fields_set = [x is not None for x in (leverage, value, quantity)]
    if sum(fields_set) != 1:
        raise ValueError("Exactly one of 'leverage', 'value', or 'quantity' must be set")

    if leverage is not None:
        quantity = (leverage * balance * usd_base_conversion) / trade_pair.lot_size
    elif value is not None:
        quantity = (value * usd_base_conversion) / trade_pair.lot_size

    if round_qty:
        if trade_pair.is_forex:
            increment = ValiConfig.FOREX_MIN_ORDER_SIZE_SUB_NANO if use_nano_increment else ValiConfig.FOREX_MIN_ORDER_SIZE
        elif trade_pair.is_crypto:
            increment = ValiConfig.CRYPTO_MIN_ORDER_SIZE
        elif trade_pair.is_equities:
            increment = ValiConfig.EQUITIES_MIN_ORDER_SIZE
        elif trade_pair.is_commodities:
            increment = ValiConfig.COMMODITIES_MIN_ORDER_SIZE
        else:
            increment = ValiConfig.CRYPTO_MIN_ORDER_SIZE
        quantity = math.trunc(quantity / increment) * increment if use_floor else round(quantity / increment) * increment

    value = quantity * trade_pair.lot_size / usd_base_conversion
    leverage = value / balance

    return quantity, leverage, value
