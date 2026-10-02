# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

from dataclasses import dataclass
from typing import Optional
from time_util.time_util import TimeUtil

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import ValiConfig
from shared_objects.log import logger

# Websocket source names built by the Polygon data services, used by apply_nasdaq_fmv_rule
POLYGON_WS_SOURCE = "Polygon_ws"  # FMV for equities
NASDAQ_BASIC_SOURCE = "Polygon_nasdaq_ws"


# Point-in-time (ws) or second candles only
@dataclass
class PriceSource:
    """
    Dataclass representing a price source for a trading instrument.

    Refactored from Pydantic BaseModel to standard dataclass to avoid
    pickle recursion issues when passing through RPC boundaries.

    Note: Dataclasses are naturally pickleable and don't have the complex
    internal state that Pydantic models have, making them ideal for RPC.
    """
    source: str = 'unknown'
    timespan_ms: int = 0
    open: Optional[float] = None
    close: Optional[float] = None
    vwap: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    start_ms: int = 0
    websocket: bool = False
    lag_ms: int = 0
    bid: Optional[float] = 0.0
    ask: Optional[float] = 0.0

    def to_dict(self):
        """Convert to dictionary (compatibility method for serialization)."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create PriceSource from dictionary.

        Args:
            data: Dictionary containing PriceSource fields

        Returns:
            PriceSource instance
        """
        return cls(**data)

    def __eq__(self, other):
        if not isinstance(other, PriceSource):
            return NotImplemented
        return (self.source == other.source and
                self.start_ms == other.start_ms and
                self.timespan_ms == other.timespan_ms and
                self.open == other.open and
                self.close == other.close and
                self.high == other.high and
                self.low == other.low)



    def __hash__(self):
        return hash((self.source,
                    self.start_ms,
                    self.timespan_ms,
                    self.open,
                    self.close,
                    self.high,
                    self.low))

    @property
    def end_ms(self):
        if self.websocket:
            return self.start_ms
        else:
            return self.start_ms + self.timespan_ms - 1  # Always prioritize a new candle over the previous one

    def get_start_time_ms(self):
        return self.start_ms

    def time_delta_from_now_ms(self, now_ms:int = None) -> int:
        if not now_ms:
            now_ms = TimeUtil.now_in_millis()
        if self.websocket:
            return abs(now_ms - self.start_ms)
        else:
            return min(abs(now_ms - self.start_ms),
                       abs(now_ms - self.end_ms))

    def parse_best_best_price_legacy(self, now_ms: int):
        if not now_ms:
            now_ms = TimeUtil.now_in_millis()
        if self.websocket:
            return self.open
        else:
            if abs(now_ms - self.start_ms) < abs(now_ms - self.end_ms):
                return self.open
            else:
                return self.close

    def parse_appropriate_price(self, now_ms: int, is_forex: bool, order_type: OrderType, position_type: OrderType) -> float:
        ans = None
        # Fill against the side of the book the order takes whenever a real quote is present.
        # Buys (LONG, or FLAT closing a SHORT) lift the ask; sells hit the bid.
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            if order_type == OrderType.LONG:
                ans = self.ask
            elif order_type == OrderType.SHORT:
                ans = self.bid
            elif order_type == OrderType.FLAT:
                if position_type == OrderType.LONG:
                    ans = self.bid
                elif position_type == OrderType.SHORT:
                    ans = self.ask
                else:
                    logger.error(f'Initial position order is FLAT. Unexpected. Position type: {position_type}')
                    ans = self.vwap
            else:
                raise Exception(f'Unexpected order type {order_type}')

        elif self.websocket:
            ans = self.open
        else:
            if abs(now_ms - self.start_ms) < abs(now_ms - self.end_ms):
                ans = self.open
            else:
                ans = self.close
        #logger.info(f'Parsed appropriate price {ans} from price_source {self} for order type {order_type} and trade_pair {position.trade_pair.trade_pair_id}')
        return ans

    @staticmethod
    def get_winning_event(events, now_ms):
        best_event = None
        best_time_delta = float('inf')
        for event in events:
            if event:
                time_delta = event.time_delta_from_now_ms(now_ms)
                if best_event is None or time_delta < best_time_delta:
                    best_event = event
                    best_time_delta = time_delta
        return best_event

    @staticmethod
    def get_winning_price_source(events, now_ms):
        return PriceSource.get_winning_event(events, now_ms)

    @staticmethod
    def apply_nasdaq_fmv_rule(events, time_ms=None):
        """
        Equities: Nasdaq Basic quotes replace FMV. When time_ms is given, only quotes within
        WEBSOCKET_PRICE_MAX_AGE_MS of it count; a stale quote would push a fresh FMV out and send the pair to REST.

        Returns events without FMV if any quote counts, otherwise without the quotes. Events with no Nasdaq quote
        are returned unchanged. Used for a single price (time_ms given) and for a window of events.
        """
        quotes = [e for e in events if e.source == NASDAQ_BASIC_SOURCE]
        if not quotes:
            return events
        kept_quote_ids = {id(q) for q in quotes
                          if time_ms is None or q.time_delta_from_now_ms(time_ms) <= ValiConfig.WEBSOCKET_PRICE_MAX_AGE_MS}
        if kept_quote_ids:
            return [e for e in events if e.source != POLYGON_WS_SOURCE
                    and (e.source != NASDAQ_BASIC_SOURCE or id(e) in kept_quote_ids)]
        return [e for e in events if e.source != NASDAQ_BASIC_SOURCE]

    @staticmethod
    def non_null_events_sorted(events, now_ms):
        ans = sorted(events, key=lambda x: x.time_delta_from_now_ms(now_ms))
        for a in ans:
            a.lag_ms = a.time_delta_from_now_ms(now_ms)
        return ans

    def debug_str(self, time_target_ms):
        return f"(src={self.source} price={self.open} ba=({self.bid}/{self.ask}) delta_ms={self.time_delta_from_now_ms(time_target_ms)})"


