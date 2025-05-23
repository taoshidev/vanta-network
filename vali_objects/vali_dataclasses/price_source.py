# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import bittensor as bt
from typing import Optional
from pydantic import BaseModel
from time_util.time_util import TimeUtil

from vali_objects.enums.order_type_enum import OrderType


# Point-in-time (ws) or second candles only
class PriceSource(BaseModel):
    source: str = 'unknown'
    timespan_ms: int = 0
    open: float = None
    close: float = None
    vwap: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    start_ms: int = 0
    websocket: bool = False
    lag_ms: int = 0
    bid: Optional[float] = 0.0
    ask: Optional[float] = 0.0

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

    def parse_appropriate_price(self, now_ms: int, is_forex: bool, order_type: OrderType, position) -> float:
        ans = None
        # Only secondly candles have bid/ask
        if is_forex and self.timespan_ms == 1000:
            if order_type == OrderType.LONG:
                ans = self.ask
            elif order_type == OrderType.SHORT:
                ans = self.bid
            elif order_type == OrderType.FLAT:
                # Use the position's initial type to determine if the FLAT is increasing or decreasing leverage
                if position.orders[0].order_type == OrderType.LONG:
                    ans = self.bid
                elif position.orders[0].order_type == OrderType.SHORT:
                    ans = self.ask
                else:
                    bt.logging.error(f'Initial position order is FLAT. Unexpected. Position: {position}')
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
        #bt.logging.success(f'Parsed appropriate price {ans} from price_source {self} for order type {order_type} and trade_pair {position.trade_pair.trade_pair_id}')
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
    def non_null_events_sorted(events, now_ms):
        ans = sorted(events, key=lambda x: x.time_delta_from_now_ms(now_ms))
        for a in ans:
            a.lag_ms = a.time_delta_from_now_ms(now_ms)
        return ans

    def debug_str(self, time_target_ms):
        return f"(src={self.source} price={self.open} ba=({self.bid}/{self.ask}) delta_ms={self.time_delta_from_now_ms(time_target_ms)})"


