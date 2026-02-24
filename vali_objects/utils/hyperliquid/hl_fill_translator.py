"""
Hyperliquid Fill Translator — Maps HL WebSocket fill data to Vanta structures.

Parses userFills messages from the Hyperliquid WebSocket API and converts them
into PriceSource and signal dicts compatible with MarketOrderManager.
"""
from dataclasses import dataclass
from typing import Optional

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.price_source import PriceSource


# Map HL coin names to Vanta TradePair enums (crypto pairs only)
HL_COIN_TO_TRADE_PAIR = {
    "BTC": TradePair.BTCUSD,
    "ETH": TradePair.ETHUSD,
    "SOL": TradePair.SOLUSD,
    "XRP": TradePair.XRPUSD,
    "DOGE": TradePair.DOGEUSD,
    "ADA": TradePair.ADAUSD,
}

# Map HL direction strings to Vanta OrderType
_HL_DIR_TO_ORDER_TYPE = {
    "Open Long": OrderType.LONG,
    "Open Short": OrderType.SHORT,
    "Close Long": OrderType.FLAT,
    "Close Short": OrderType.FLAT,
}


@dataclass
class HyperliquidFill:
    """Parsed representation of a single Hyperliquid userFill."""
    coin: str
    price: float
    sz: float
    dir: str
    time_ms: int
    hash: str
    tid: str
    start_position: Optional[str] = None
    closed_pnl: Optional[str] = None

    @classmethod
    def from_ws_data(cls, fill_dict: dict) -> "HyperliquidFill":
        """Parse a fill dict from HL WebSocket userFills message."""
        return cls(
            coin=fill_dict["coin"],
            price=float(fill_dict["px"]),
            sz=float(fill_dict["sz"]),
            dir=fill_dict["dir"],
            time_ms=fill_dict["time"],
            hash=fill_dict["hash"],
            tid=str(fill_dict.get("tid", "")),
            start_position=fill_dict.get("startPosition"),
            closed_pnl=fill_dict.get("closedPnl"),
        )

    @property
    def trade_pair(self) -> Optional[TradePair]:
        return HL_COIN_TO_TRADE_PAIR.get(self.coin)

    @property
    def order_type(self) -> Optional[OrderType]:
        return _HL_DIR_TO_ORDER_TYPE.get(self.dir)

    @property
    def dedup_id(self) -> str:
        """Deterministic UUID for deduplication across reconnects."""
        return f"hl_{self.hash}_{self.tid}"

    def to_price_source(self) -> PriceSource:
        """
        Create a PriceSource using the HL fill price as open/close.

        bid/ask are set to None for now — they will be populated from HL L2
        order book data once we add l2Book subscriptions, so the slippage model
        can calculate real slippage based on Vanta's order size vs HL book depth.
        """
        return PriceSource(
            source="hyperliquid",
            timespan_ms=0,
            open=self.price,
            close=self.price,
            vwap=self.price,
            high=self.price,
            low=self.price,
            start_ms=self.time_ms,
            websocket=True,
            lag_ms=0,
            bid=None,
            ask=None,
        )

    def to_signal_dict(self) -> dict:
        """
        Create a signal dict compatible with MarketOrderManager._process_market_order().

        Sets quantity from the HL fill size and marks the signal as an HL fill
        so downstream code can set the correct OrderSource.
        """
        ot = self.order_type
        if ot is None:
            raise ValueError(f"Unknown HL direction: {self.dir}")

        return {
            "order_type": ot.value,
            "trade_pair": self.trade_pair.trade_pair_id if self.trade_pair else None,
            "quantity": self.sz,
            "execution_type": "MARKET",
            "_hl_fill": True,
        }
