# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from pydantic import field_validator, model_validator

from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.vali_dataclasses.price_source import PriceSource


class Order(Signal):
    price: float                # Quote currency
    bid: float = 0              # Quote currency
    ask: float = 0              # Quote currency
    slippage: float = 0
    quote_usd_rate: float = 0.0 # Conversion rate from quote currency to USD
    usd_base_rate: float = 0.0  # Conversion rate from usd to base currency
    processed_ms: int
    order_uuid: str
    price_sources: list = []
    src: int = OrderSource.ORGANIC
    margin_loan: float = 0.0

    @field_validator('trade_pair', mode='before')
    @classmethod
    def convert_trade_pair(cls, v):
        """Convert trade_pair_id string or dict to TradePair object if needed."""
        if isinstance(v, str):
            return TradePair.from_trade_pair_id(v)
        elif isinstance(v, dict):
            # Handle dict with 'trade_pair_id' key (from disk serialization)
            if 'trade_pair_id' in v:
                return TradePair.from_trade_pair_id(v['trade_pair_id'])
        elif isinstance(v, list) and len(v) >= 1:
            # Handle list format ['BTCUSD', 'BTC/USD'] from to_python_dict()
            return TradePair.from_trade_pair_id(v[0])
        return v

    @field_validator('execution_type', mode='before')
    @classmethod
    def convert_execution_type(cls, v):
        """Convert execution_type string to ExecutionType enum if needed."""
        if isinstance(v, str):
            return ExecutionType.from_string(v)
        return v

    @model_validator(mode='before')
    @classmethod
    def handle_trade_pair_id(cls, values):
        """Handle dict input with 'trade_pair_id' instead of 'trade_pair'."""
        if isinstance(values, dict) and 'trade_pair_id' in values and 'trade_pair' not in values:
            # Create new dict with trade_pair instead of trade_pair_id (immutable approach)
            return {k: v for k, v in values.items() if k != 'trade_pair_id'} | {'trade_pair': values['trade_pair_id']}
        return values

    @model_validator(mode="after")
    def set_conversion_defaults(self):
        """
        Initializes quote_usd_rate and usd_base_rate based on the trade pair.
        Only sets values if they were left at the default 0.
        """
        # Skip for FLAT_ALL or other cases where trade_pair is None
        if self.trade_pair is None:
            return self

        base = self.trade_pair.base  # e.g. BTC in BTCUSD
        quote = self.trade_pair.quote  # e.g. USD in BTCUSD
        price = self.price

        if price == 0:
            return self

        if self.quote_usd_rate == 0:
            if quote == "USD":
                self.quote_usd_rate = 1.0
            elif base == "USD":
                self.quote_usd_rate = 1.0 / price

        if self.usd_base_rate == 0:
            if base == "USD":
                self.usd_base_rate = 1.0
            elif quote == "USD":
                self.usd_base_rate = 1.0 / price

        return self

    @field_validator('price', 'processed_ms', mode='before')
    def validate_values(cls, v, info):
        if info.field_name == 'price' and v < 0:
            raise ValueError("Price must be greater than 0")
        if info.field_name == 'processed_ms' and v < 0:
            raise ValueError("processed_ms must be greater than 0")
        return v

    @field_validator('order_uuid', mode='before')
    def ensure_order_uuid_is_string(cls, v):
        if not isinstance(v, str):
            v = str(v)
        return v

    @field_validator('price_sources', mode='before')
    def validate_price_sources(cls, v):
        if isinstance(v, list):
            return [PriceSource(**ps) if isinstance(ps, dict) else ps for ps in v]
        return v

    # @model_validator(mode='before')
    # def validate_size(cls, values):
    #     """
    #     Ensure that size meets min and maximum requirements
    #     """
    #     order_type = values['order_type']
    #     is_flat_order = order_type == OrderType.FLAT or order_type == 'FLAT'
    #     lev = values['leverage']
    #     val = values.get('value')
    #     if not is_flat_order and not (ValiConfig.ORDER_MIN_LEVERAGE <= abs(lev) <= ValiConfig.ORDER_MAX_LEVERAGE):
    #         raise ValueError(
    #             f"Order leverage must be between {ValiConfig.ORDER_MIN_LEVERAGE} and {ValiConfig.ORDER_MAX_LEVERAGE}, provided - lev [{lev}] and order_type [{order_type}] ({type(order_type)})")
    #     if val is not None and not is_flat_order and not ValiConfig.ORDER_MIN_VALUE <= abs(val):
    #         raise ValueError(f"Order value must be greater than {ValiConfig.ORDER_MIN_VALUE}, provided value is {abs(val)}")
    #     return values

    @model_validator(mode="before")
    def validate_size_fields(cls, values):
        """
        Overrides inherited validate_size_fields from Signal.
        Order can have all three leverage/value/quantity fields filled.
        """
        return values

    @model_validator(mode="before")
    @classmethod
    def check_bracket_orders(cls, values):
        """
        Overrides inherited check_bracket_orders from Signal.
        Order allows both bracket_orders and stop_loss/take_profit for display purposes.
        """
        return values

    @model_validator(mode="before")
    @classmethod
    def normalize_bracket_orders(cls, values):
        """
        Convert stop_loss/take_profit to bracket_orders format and validate entries.
        """
        bracket_orders = values.get('bracket_orders')
        stop_loss = values.get('stop_loss')
        take_profit = values.get('take_profit')
        has_sl_tp = stop_loss is not None or take_profit is not None

        execution_type = values.get('execution_type')
        if execution_type not in [ExecutionType.MARKET, ExecutionType.LIMIT]:
            return values

        # Convert stop_loss/take_profit to bracket_orders
        if has_sl_tp and not bracket_orders:
            values['bracket_orders'] = [{
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': values.get('leverage'),
                'value': values.get('value'),
                'quantity': values.get('quantity'),
            }]
            return values

        # Validate bracket_orders entries
        bracket_orders = values.get('bracket_orders')
        if not bracket_orders:
            return values

        size_fields = {'leverage', 'value', 'quantity'}
        price_fields = {'stop_loss', 'take_profit'}

        for i, bracket in enumerate(bracket_orders):
            # Exactly one size field required
            size_present = [f for f in size_fields if bracket.get(f) is not None]
            if len(size_present) != 1:
                raise ValueError(f"bracket_orders[{i}]: exactly one of leverage/value/quantity required")

            # At least one price field required
            price_present = [f for f in price_fields if bracket.get(f) is not None]
            if len(price_present) < 1:
                raise ValueError(f"bracket_orders[{i}]: at least one of stop_loss/take_profit required")

        return values

    @classmethod
    def from_dict(cls, order_dict):
        """
        Create Order from dict. Pydantic validators handle all conversions:
        - trade_pair_id (str) -> trade_pair (TradePair)
        - order_type (str) -> order_type (OrderType)
        """
        return cls(**order_dict)

    def get_order_age(self, order):
        return TimeUtil.now_in_millis() - order.processed_ms

    def to_python_dict(self):
        trade_pair_id = None
        trade_pair_name = None
        if self.trade_pair is not None:
            trade_pair_id = self.trade_pair.trade_pair_id if hasattr(self.trade_pair, 'trade_pair_id') else 'unknown'
            trade_pair_name = self.trade_pair.trade_pair if hasattr(self.trade_pair, 'trade_pair') else 'unknown'
        return {'trade_pair_id': trade_pair_id,
                'trade_pair': [trade_pair_id, trade_pair_name],
                'order_type': self.order_type.name,
                'leverage': self.leverage,
                'value': self.value,
                'quantity': self.quantity,
                'price': self.price,
                'bid': self.bid,
                'ask': self.ask,
                'slippage': self.slippage,
                'quote_usd_rate': self.quote_usd_rate,
                'usd_base_rate': self.usd_base_rate,
                'processed_ms': self.processed_ms,
                'price_sources': self.price_sources,
                'order_uuid': self.order_uuid,
                'src': self.src,
                'execution_type': self.execution_type.name if self.execution_type else None,
                'limit_price': self.limit_price,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'margin_loan': self.margin_loan,
                'bracket_orders': self.bracket_orders}

    def __str__(self):
        # Ensuring the `trade_pair.trade_pair_id` is accessible for the string representation
        # This assumes that trade_pair_id is a valid attribute of trade_pair
        d = self.to_python_dict()
        return str(d)



