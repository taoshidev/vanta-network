# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
from typing import Optional

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from pydantic import BaseModel, model_validator

class Signal(BaseModel):
    trade_pair: TradePair
    order_type: OrderType
    leverage: Optional[float] = None    # Multiplier of account size
    value: Optional[float] = None       # USD notional value
    quantity: Optional[float] = None    # Base currency, number of lots/coins/shares/etc.
    execution_type: ExecutionType = ExecutionType.MARKET
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @model_validator(mode='before')
    def check_exclusive_fields(cls, values):
        """
        Ensure that only ONE of leverage, value, or quantity is filled
        """
        fields = ['leverage', 'value', 'quantity']
        filled = [f for f in fields if values.get(f) is not None]
        if len(filled) != 1:
            raise ValueError(f"Exactly one of {fields} must be provided, got {filled}")
        return values

    @model_validator(mode='before')
    def set_size(cls, values):
        """
        Ensure that long orders have positive size, and short orders have negative size,
        applied to all non-None of leverage, value, and quantity.
        Special handling for BRACKET orders.
        """
        order_type = values['order_type']
        execution_type = values.get('execution_type', ExecutionType.MARKET)

        # Skip validation for FLAT and BRACKET orders
        is_flat_order = order_type == OrderType.FLAT or order_type == 'FLAT'
        is_bracket_order = execution_type == ExecutionType.BRACKET

        # For bracket orders, leverage can be 0 (will be set from position)
        if is_bracket_order:
            lev = values.get('leverage', 0)
            if lev == 0:
                return values

        # Apply sign correction to leverage, value, and quantity
        for field in ['leverage', 'value', 'quantity']:
            size = values.get(field)
            if size is not None:
                if order_type == OrderType.LONG and size < 0:
                    raise ValueError(f"{field} must be positive for LONG orders.")
                elif order_type == OrderType.SHORT:
                    values[field] = -1.0 * abs(size)
        return values

    @staticmethod
    def parse_trade_pair_from_signal(signal) -> TradePair | None:
        if not signal or not isinstance(signal, dict):
            return None
        if 'trade_pair' not in signal:
            return None
        temp = signal["trade_pair"]
        if 'trade_pair_id' not in temp:
            return None
        string_trade_pair = signal["trade_pair"]["trade_pair_id"]
        trade_pair = TradePair.from_trade_pair_id(string_trade_pair)
        return trade_pair

    def __str__(self):
        base = {
            'trade_pair': str(self.trade_pair),
            'order_type': str(self.order_type),
            'leverage': self.leverage,
            'value': self.value,
            'quantity': self.quantity,
            'execution_type': str(self.execution_type)
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
            # No extra fields needed - order_uuid comes from synapse.miner_order_uuid
            return str(base)

        return str({**base, 'Error': 'Unknown execution type'})
