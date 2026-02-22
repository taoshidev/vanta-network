# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
from typing import Optional

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from pydantic import BaseModel, model_validator

class Signal(BaseModel):
    trade_pair: Optional[TradePair] = None  # Optional for FLAT_ALL and LIMIT_CANCEL
    order_type: Optional[OrderType] = None
    leverage: Optional[float] = None    # Multiplier of account size
    value: Optional[float] = None       # USD notional value
    quantity: Optional[float] = None    # Base currency, number of lots/coins/shares/etc.
    execution_type: ExecutionType = ExecutionType.MARKET
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    bracket_orders: Optional[list[dict]] = None

    @model_validator(mode='before')
    @classmethod
    def check_bracket_orders(cls, values):
        """
        Validate mutual exclusivity: bracket_orders vs stop_loss/take_profit.
        """
        bracket_orders = values.get('bracket_orders')
        has_sl_tp = values.get('stop_loss') is not None or values.get('take_profit') is not None

        if bracket_orders and has_sl_tp:
            raise ValueError("Cannot set both bracket_orders and stop_loss/take_profit on Signal")

        return values

    @model_validator(mode='before')
    def validate_order_type(cls, values):
        """Validate order type restrictions and normalize size."""
        execution_type = values.get('execution_type')
        if execution_type in [ExecutionType.LIMIT_CANCEL, ExecutionType.FLAT_ALL]:
            return values

        order_type = values.get('order_type')
        is_flat = order_type == OrderType.FLAT or order_type == 'FLAT'

        if execution_type == ExecutionType.LIMIT and is_flat:
            raise ValueError("FLAT order is not supported for LIMIT orders")

        # Normalize size sign based on order_type
        for field in ['leverage', 'value', 'quantity']:
            size = values.get(field)
            if size is not None:
                if order_type == OrderType.LONG and size < 0:
                    raise ValueError(f"{field} must be positive for LONG orders.")
                elif order_type == OrderType.SHORT:
                    values[field] = -1.0 * abs(size)

        return values

    @model_validator(mode='before')
    def validate_size_fields(cls, values):
        """Validate only one size field is filled (leverage/value/quantity)."""
        execution_type = values.get('execution_type')
        order_type = values.get('order_type')
        # Skip size validation for LIMIT_CANCEL, FLAT_ALL, and FLAT orders
        if execution_type in [ExecutionType.LIMIT_CANCEL, ExecutionType.FLAT_ALL] or order_type == OrderType.FLAT:
            return values

        fields = ['leverage', 'value', 'quantity']
        filled = [f for f in fields if values.get(f) is not None]

        # BRACKET allows empty size fields (populated from position)
        if execution_type != ExecutionType.BRACKET and len(filled) != 1:
            raise ValueError(f"Exactly one of {fields} must be provided, got {filled}")

        return values

    @model_validator(mode='before')
    def validate_price_fields(cls, values):
        """Validate price fields based on execution type."""
        execution_type = values.get('execution_type')
        order_type = values.get('order_type')

        if execution_type == ExecutionType.LIMIT:
            limit_price = values.get('limit_price')
            if not limit_price:
                raise ValueError("Limit price must be specified for LIMIT orders")

            sl = values.get('stop_loss')
            tp = values.get('take_profit')
            if order_type == OrderType.LONG and ((sl and sl >= limit_price) or (tp and tp <= limit_price)):
                raise ValueError(
                    f"LONG LIMIT orders must satisfy: stop_loss < limit_price < take_profit. "
                    f"Got stop_loss={sl}, limit_price={limit_price}, take_profit={tp}"
                )
            elif order_type == OrderType.SHORT and ((sl and sl <= limit_price) or (tp and tp >= limit_price)):
                raise ValueError(
                    f"SHORT LIMIT orders must satisfy: take_profit < limit_price < stop_loss. "
                    f"Got take_profit={tp}, limit_price={limit_price}, stop_loss={sl}"
                )

        elif execution_type == ExecutionType.BRACKET:
            sl = values.get('stop_loss')
            tp = values.get('take_profit')
            bracket_orders = values.get('bracket_orders')

            # If top-level SL/TP empty but bracket_orders provided, extract from first entry
            if sl is None and tp is None and bracket_orders:
                if len(bracket_orders) != 1:
                    raise ValueError("bracket_orders must contain exactly one entry when used for BRACKET orders")
                sl = bracket_orders[0].get('stop_loss')
                tp = bracket_orders[0].get('take_profit')

            # Validate at least one of SL or TP is set
            if sl is None and tp is None:
                raise ValueError("Bracket order must specify at least one of stop_loss or take_profit")

            # Validate stop_loss > 0 if present
            if sl is not None and float(sl) <= 0:
                raise ValueError("stop_loss must be greater than 0")

            # Validate take_profit > 0 if present
            if tp is not None and float(tp) <= 0:
                raise ValueError("take_profit must be greater than 0")

            # Validate SL and TP are unique if both set
            if sl is not None and tp is not None and float(sl) == float(tp):
                raise ValueError("stop_loss and take_profit must be unique")

        return values

    @staticmethod
    def parse_trade_pair_from_signal(signal) -> TradePair | None:
        if not signal or not isinstance(signal, dict):
            return None
        if 'trade_pair' not in signal:
            return None
        temp = signal["trade_pair"]
        # Handle list format from model_dump(mode='json'): ['BTCUSD', 'BTC/USD', ...]
        if isinstance(temp, list) and len(temp) >= 1:
            return TradePair.from_trade_pair_id(temp[0])
        # Handle dict format: {'trade_pair_id': 'BTCUSD', ...}
        if isinstance(temp, dict) and 'trade_pair_id' in temp:
            return TradePair.from_trade_pair_id(temp['trade_pair_id'])
        # Handle string format: 'BTCUSD'
        if isinstance(temp, str):
            return TradePair.from_trade_pair_id(temp)
        return None

    def __str__(self):
        base = {
            'trade_pair': str(self.trade_pair) if self.trade_pair else None,
            'order_type': str(self.order_type),
            'leverage': self.leverage,
            'value': self.value,
            'quantity': self.quantity,
            'execution_type': str(self.execution_type),
            'bracket_orders': self.bracket_orders
        }
        if self.execution_type == ExecutionType.MARKET:
            return str(base)

        elif self.execution_type == ExecutionType.LIMIT:
            base.update({
                'limit_price': self.limit_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            })
            return str(base)

        elif self.execution_type == ExecutionType.LIMIT_CANCEL:
            return str(base)

        elif self.execution_type == ExecutionType.FLAT_ALL:
            return str(base)

        return str({**base, 'Error': 'Unknown execution type'})
