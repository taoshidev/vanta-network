# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

from time_util.time_util import TimeUtil
from pydantic import field_validator, model_validator

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.vali_dataclasses.price_source import PriceSource
from enum import Enum, IntEnum, auto

class OrderSource(IntEnum):
    """Enum representing the source/origin of an order."""
    ORGANIC = 0                        # order generated from a miner's signal
    ELIMINATION_FLAT = 1               # order inserted when a miner is eliminated (0 used for price. DEPRECATED)
    DEPRECATION_FLAT = 2               # order inserted when a trade pair is removed (0 used for price)
    PRICE_FILLED_ELIMINATION_FLAT = 3  # order inserted when a miner is eliminated but we price fill it accurately.
    MAX_ORDERS_PER_POSITION_CLOSE = 4  # order inserted when position hits max orders limit and needs to be closed

# Backward compatibility constants - to be removed after migration
ORDER_SRC_ORGANIC = OrderSource.ORGANIC
ORDER_SRC_ELIMINATION_FLAT = OrderSource.ELIMINATION_FLAT
ORDER_SRC_DEPRECATION_FLAT = OrderSource.DEPRECATION_FLAT
ORDER_SRC_PRICE_FILLED_ELIMINATION_FLAT = OrderSource.PRICE_FILLED_ELIMINATION_FLAT
ORDER_SRC_MAX_ORDERS_PER_POSITION_CLOSE = OrderSource.MAX_ORDERS_PER_POSITION_CLOSE

class Order(Signal):
    price: float
    bid: float = 0
    ask: float = 0
    slippage: float = 0
    processed_ms: int
    order_uuid: str
    price_sources: list = []
    src: int = ORDER_SRC_ORGANIC

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

    @model_validator(mode='before')
    def validate_size(cls, values):
        """
        Ensure that size meets min and maximum requirements
        """
        order_type = values['order_type']
        is_flat_order = order_type == OrderType.FLAT or order_type == 'FLAT'
        lev = values['leverage']
        val = values.get('value')
        if not is_flat_order and not (ValiConfig.ORDER_MIN_LEVERAGE <= abs(lev) <= ValiConfig.ORDER_MAX_LEVERAGE):
            raise ValueError(
                f"Order leverage must be between {ValiConfig.ORDER_MIN_LEVERAGE} and {ValiConfig.ORDER_MAX_LEVERAGE}, provided - lev [{lev}] and order_type [{order_type}] ({type(order_type)})")
        if val is not None and not is_flat_order and not ValiConfig.ORDER_MIN_VALUE <= abs(val):
            raise ValueError(f"Order value must be greater than {ValiConfig.ORDER_MIN_VALUE}, provided value is {abs(val)}")
        return values

    @model_validator(mode="before")
    def check_exclusive_fields(cls, values):
        """
        Overrides inherited check_exclusive_fields from signal. When we populate the order we want to fill in all three leverage/value/quantity fields.
        """
        return values

    # Using Pydantic's constructor instead of a custom from_dict method
    @classmethod
    def from_dict(cls, order_dict):
        # This method is now simplified as Pydantic can automatically
        # handle the conversion from dict to model instance
        return cls(**order_dict)

    def get_order_age(self, order):
        return TimeUtil.now_in_millis() - order.processed_ms

    def to_python_dict(self):
        trade_pair_id = self.trade_pair.trade_pair_id if hasattr(self.trade_pair, 'trade_pair_id') else 'unknown'
        return {'trade_pair_id': trade_pair_id,
                    'order_type': self.order_type.name,
                    'leverage': self.leverage,
                    'value': self.value,
                    'quantity': self.quantity,
                    'price': self.price,
                    'bid': self.bid,
                    'ask': self.ask,
                    'slippage': self.slippage,
                    'processed_ms': self.processed_ms,
                    'price_sources': self.price_sources,
                    'order_uuid': self.order_uuid,
                    'src': self.src}
    def __str__(self):
        # Ensuring the `trade_pair.trade_pair_id` is accessible for the string representation
        # This assumes that trade_pair_id is a valid attribute of trade_pair
        d = self.to_python_dict()
        return str(d)



class OrderStatus(Enum):
    OPEN = auto()
    CLOSED = auto()
    ALL = auto()  # Represents both or neither, depending on your logic

