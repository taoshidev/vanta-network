# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
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
        Ensure that only ONE of leverage, value, or quantity is filled.
        Exception: BRACKET orders can have all fields as None (will be populated from position).
        """
        execution_type = values.get('execution_type')
        if execution_type == ExecutionType.LIMIT_CANCEL:
            return values

        fields = ['leverage', 'value', 'quantity']
        filled = [f for f in fields if values.get(f) is not None]
        if len(filled) == 0 and execution_type == ExecutionType.BRACKET:
            return values
        if len(filled) != 1:
            raise ValueError(f"Exactly one of {fields} must be provided, got {filled}")
        return values

    @model_validator(mode='before')
    def check_price_fields(cls, values):
        execution_type = values.get('execution_type')
        order_type = values.get('order_type')

        if execution_type == ExecutionType.LIMIT:
            limit_price = values.get('limit_price')
            if not limit_price:
                raise ValueError(f"Limit price must be specified for LIMIT orders")

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
            if not sl and not tp:
                raise ValueError(f"Either stop_loss or take_profit must be set for BRACKET orders")
            if sl and tp and sl == tp:
                raise ValueError(f"stop_loss and take_profit must be unique")

        return values


    @model_validator(mode='before')
    def set_size(cls, values):
        """
        Ensure that long orders have positive size, and short orders have negative size,
        applied to all non-None of leverage, value, and quantity.
        """
        execution_type = values.get('execution_type')
        if execution_type == ExecutionType.LIMIT_CANCEL:
            return values

        order_type = values['order_type']

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
