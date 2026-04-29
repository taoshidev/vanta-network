# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
from typing import Optional, Union

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair, TradePairLike, ValiConfig
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from pydantic import BaseModel, model_validator, field_validator

class Signal(BaseModel):
    trade_pair: Optional[Union[TradePairLike, str]] = None  # str fallback for dynamic pairs not in local registry
    order_type: Optional[OrderType] = None
    leverage: Optional[float] = None    # Multiplier of account size
    value: Optional[float] = None       # USD notional value
    quantity: Optional[float] = None    # Base currency, number of lots/coins/shares/etc.
    execution_type: ExecutionType = ExecutionType.MARKET
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_price: Optional[float] = None
    stop_condition: Optional[StopCondition] = None
    trailing_stop: Optional[dict] = None
    bracket_orders: Optional[list[dict]] = None

    @field_validator('trade_pair', mode='before')
    @classmethod
    def convert_trade_pair(cls, v):
        if isinstance(v, str):
            return TradePair.from_trade_pair_id(v) or v
        if isinstance(v, dict) and 'trade_pair_id' in v:
            return TradePair.from_trade_pair_id(v['trade_pair_id']) or v['trade_pair_id']
        if isinstance(v, list) and len(v) >= 1:
            return TradePair.from_trade_pair_id(v[0]) or v[0]
        return v

    # Pydantic v2 runs mode='before' validators in REVERSE definition order.
    # normalize_bracket_orders is defined first so it runs LAST.

    @model_validator(mode="before")
    @classmethod
    def normalize_bracket_orders(cls, values):
        """Convert top-level stop_loss/take_profit/trailing_stop into bracket_orders entries."""
        bracket_orders = values.get('bracket_orders')
        stop_loss = values.get('stop_loss')
        take_profit = values.get('take_profit')
        trailing_stop = values.get('trailing_stop')
        has_sl_tp = stop_loss is not None or take_profit is not None
        has_trailing = trailing_stop is not None

        execution_type = values.get('execution_type', ExecutionType.MARKET)
        if execution_type not in [ExecutionType.MARKET, ExecutionType.LIMIT, ExecutionType.STOP_LIMIT]:
            return values

        if (has_sl_tp or has_trailing) and not bracket_orders:
            bracket_entry = {'stop_loss': stop_loss, 'take_profit': take_profit}
            if has_trailing:
                if 'trailing_percent' in trailing_stop:
                    bracket_entry['trailing_percent'] = trailing_stop['trailing_percent']
                if 'trailing_value' in trailing_stop:
                    bracket_entry['trailing_value'] = trailing_stop['trailing_value']
            values['bracket_orders'] = [bracket_entry]
            return values

        if not bracket_orders:
            return values

        price_fields = {'stop_loss', 'take_profit'}
        trailing_fields = {'trailing_percent', 'trailing_value'}

        for i, bracket in enumerate(bracket_orders):
            price_present = [f for f in price_fields if bracket.get(f) is not None]
            trailing_present = [f for f in trailing_fields if bracket.get(f) is not None]
            if len(price_present) < 1 and len(trailing_present) < 1:
                raise ValueError(f"bracket_orders[{i}]: at least one of stop_loss/take_profit/trailing_percent/trailing_value required")

        return values

    @model_validator(mode='before')
    @classmethod
    def validate_trailing_stop(cls, values):
        """Validate trailing_stop dict structure and range."""
        trailing_stop = values.get('trailing_stop')
        if trailing_stop is None:
            return values

        if not isinstance(trailing_stop, dict):
            raise ValueError("trailing_stop must be a dict")

        has_percent = 'trailing_percent' in trailing_stop
        has_value = 'trailing_value' in trailing_stop

        if has_percent == has_value:
            raise ValueError("trailing_stop must contain exactly one of 'trailing_percent' or 'trailing_value'")

        if has_percent:
            pct = float(trailing_stop['trailing_percent'])
            if not (0 < pct < 1):
                raise ValueError(f"trailing_percent must be between 0 and 1 (exclusive), got {pct}")

        if has_value:
            val = float(trailing_stop['trailing_value'])
            if val <= 0:
                raise ValueError(f"trailing_value must be greater than 0, got {val}")

        return values

    @model_validator(mode='before')
    def validate_order_type(cls, values):
        """Validate order type restrictions and normalize size sign for SHORT."""
        execution_type = values.get('execution_type')
        if execution_type in [ExecutionType.LIMIT_CANCEL, ExecutionType.FLAT_ALL]:
            return values

        order_type = values.get('order_type')
        is_flat = order_type == OrderType.FLAT or order_type == 'FLAT'

        if execution_type == ExecutionType.LIMIT and is_flat:
            raise ValueError("FLAT order is not supported for LIMIT orders")

        if execution_type == ExecutionType.STOP_LIMIT and is_flat:
            raise ValueError("FLAT order is not supported for STOP_LIMIT orders")

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
        """Exactly one of leverage/value/quantity must be provided."""
        execution_type = values.get('execution_type')
        order_type = values.get('order_type')
        if execution_type in [ExecutionType.LIMIT_CANCEL, ExecutionType.FLAT_ALL] or order_type == OrderType.FLAT:
            return values

        fields = ['leverage', 'value', 'quantity']
        filled = [f for f in fields if values.get(f) is not None]

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
            has_trailing = values.get('trailing_stop') is not None
            if order_type == OrderType.LONG and (((sl and not has_trailing) and sl >= limit_price) or (tp and tp <= limit_price)):
                raise ValueError(
                    f"LONG LIMIT orders must satisfy: stop_loss < limit_price < take_profit. "
                    f"Got stop_loss={sl}, limit_price={limit_price}, take_profit={tp}"
                )
            elif order_type == OrderType.SHORT and (((sl and not has_trailing) and sl <= limit_price) or (tp and tp >= limit_price)):
                raise ValueError(
                    f"SHORT LIMIT orders must satisfy: take_profit < limit_price < stop_loss. "
                    f"Got take_profit={tp}, limit_price={limit_price}, stop_loss={sl}"
                )

        elif execution_type == ExecutionType.STOP_LIMIT:
            stop_price = values.get('stop_price')
            limit_price = values.get('limit_price')
            stop_condition = values.get('stop_condition')

            if not stop_price or float(stop_price) <= 0:
                raise ValueError("stop_price must be specified and > 0 for STOP_LIMIT orders")
            if not limit_price or float(limit_price) <= 0:
                raise ValueError("limit_price must be specified and > 0 for STOP_LIMIT orders")
            if stop_condition is None:
                raise ValueError("stop_condition must be specified for STOP_LIMIT orders (GTE or LTE)")
            # Validate stop_condition is a valid enum value
            if isinstance(stop_condition, str):
                try:
                    StopCondition.from_string(stop_condition)
                except ValueError:
                    raise ValueError(f"Invalid stop_condition '{stop_condition}'. Must be GTE or LTE")

            # Validate SL/TP against limit_price (same as LIMIT order validation)
            sl = values.get('stop_loss')
            tp = values.get('take_profit')
            has_trailing = values.get('trailing_stop') is not None
            if order_type == OrderType.LONG and (((sl and not has_trailing) and sl >= float(limit_price)) or (tp and tp <= float(limit_price))):
                raise ValueError(
                    f"LONG STOP_LIMIT orders must satisfy: stop_loss < limit_price < take_profit. "
                    f"Got stop_loss={sl}, limit_price={limit_price}, take_profit={tp}"
                )
            elif order_type == OrderType.SHORT and (((sl and not has_trailing) and sl <= float(limit_price)) or (tp and tp >= float(limit_price))):
                raise ValueError(
                    f"SHORT STOP_LIMIT orders must satisfy: take_profit < limit_price < stop_loss. "
                    f"Got take_profit={tp}, limit_price={limit_price}, stop_loss={sl}"
                )

        elif execution_type == ExecutionType.BRACKET:
            sl = values.get('stop_loss')
            tp = values.get('take_profit')
            trailing_stop = values.get('trailing_stop')
            bracket_orders = values.get('bracket_orders')

            if sl is None and tp is None and trailing_stop is None and bracket_orders:
                if len(bracket_orders) != 1:
                    raise ValueError("bracket_orders must contain exactly one entry when used for BRACKET orders")
                sl = bracket_orders[0].get('stop_loss')
                tp = bracket_orders[0].get('take_profit')
                if bracket_orders[0].get('trailing_percent') is not None or bracket_orders[0].get('trailing_value') is not None:
                    trailing_stop = bracket_orders[0]

            if sl is None and tp is None and trailing_stop is None:
                raise ValueError("Bracket order must specify at least one of stop_loss, take_profit, or trailing_stop")

            if sl is not None and float(sl) <= 0:
                raise ValueError("stop_loss must be greater than 0")

            if tp is not None and float(tp) <= 0:
                raise ValueError("take_profit must be greater than 0")

            if sl is not None and tp is not None and float(sl) == float(tp):
                raise ValueError("stop_loss and take_profit must be unique")

        return values

    @staticmethod
    def parse_trade_pair_from_signal(signal) -> TradePairLike | None:
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

        elif self.execution_type == ExecutionType.STOP_LIMIT:
            base.update({
                'stop_price': self.stop_price,
                'stop_condition': str(self.stop_condition) if self.stop_condition else None,
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
