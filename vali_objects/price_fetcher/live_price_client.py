from typing import List, Optional, Tuple, Dict

from shared_objects.rpc.rpc_client_base import RPCClientBase
from time_util.time_util import UnifiedMarketCalendar, TimeUtil
from vali_objects.vali_config import RPCConnectionMode, ValiConfig, TradePair
from vali_objects.vali_dataclasses.price_source import PriceSource


class LivePriceFetcherClient(RPCClientBase):
    """
    Lightweight RPC client for LivePriceFetcherServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_LIVEPRICEFETCHER_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct LivePriceFetcherServer instance.
    """


    def __init__(self, running_unit_tests: bool = False,
                 connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        """
        Initialize live price fetcher client.

        Args:
            port: Port number of the server (default: ValiConfig.RPC_LIVEPRICEFETCHER_PORT)
            running_unit_tests: If True, don't connect via RPC (use set_direct_server() instead)
        """
        self.running_unit_tests = running_unit_tests

        # Market calendar for local (non-RPC) market hours checking
        self._market_calendar = UnifiedMarketCalendar()

        # In test mode, don't connect via RPC - tests will set direct server
        super().__init__(
            service_name=ValiConfig.RPC_LIVEPRICEFETCHER_SERVICE_NAME,
            port=ValiConfig.RPC_LIVEPRICEFETCHER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=False,
            connection_mode=connection_mode
        )

    @property
    def _server(self):
        # Use parent class's _server which handles lazy connection
        return super()._server

    # ========== Local methods (no RPC) ==========

    def is_market_open(self, trade_pair: TradePair, time_ms=None) -> bool:
        """
        Check if market is open for a trade pair. Executes locally (no RPC).

        Args:
            trade_pair: The trade pair to check
            time_ms: Optional timestamp in milliseconds (defaults to now)

        Returns:
            bool: True if market is open, False otherwise
        """
        if self.running_unit_tests:
            return self._server.is_market_open(trade_pair, time_ms)

        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        return self._market_calendar.is_market_open(trade_pair, time_ms)

    def get_unsupported_trade_pairs(self):
        """
        Return static tuple of unsupported trade pairs. Executes locally (no RPC).

        Returns:
            Tuple of TradePair constants that are unsupported
        """
        return ValiConfig.UNSUPPORTED_TRADE_PAIRS

    # ========== RPC proxy methods ==========

    def stop_all_threads(self):
        """Stop all data service threads on the server."""
        return self._server.stop_all_threads()

    def get_usd_base_conversion(self, trade_pair, time_ms, price, order_type, position):
        return self._server.get_usd_base_conversion(trade_pair, time_ms, price, order_type, position)

    def health_check(self) -> dict:
        """Health check - returns server status."""
        return self._server.health_check()

    def get_ws_price_sources_in_window(self, trade_pair: TradePair, start_ms: int, end_ms: int) -> List[PriceSource]:
        """Get WebSocket price sources in time window."""
        return self._server.get_ws_price_sources_in_window(trade_pair, start_ms, end_ms)

    def get_currency_conversion(self, base: str, quote: str):
        """Get currency conversion rate."""
        return self._server.get_currency_conversion(base, quote)

    def unified_candle_fetcher(self, trade_pair, start_date, order_date, timespan="day"):
        """Fetch candles for a trade pair."""
        return self._server.unified_candle_fetcher(trade_pair, start_date, order_date, timespan)

    def get_latest_price(self, trade_pair: TradePair, time_ms=None) -> Tuple[float, List[PriceSource]] | Tuple[None, None]:
        """Get the latest price for a trade pair."""
        return self._server.get_latest_price(trade_pair, time_ms)

    def get_sorted_price_sources_for_trade_pair(self, trade_pair: TradePair, time_ms: int, live=True) -> List[PriceSource] | None:
        """Get sorted price sources for a trade pair."""
        return self._server.get_sorted_price_sources_for_trade_pair(trade_pair, time_ms, live)

    def get_tp_to_sorted_price_sources(self, trade_pairs: List[TradePair], time_ms: int, live=True) -> Dict[TradePair, List[PriceSource]]:
        """Get sorted price sources for multiple trade pairs."""
        return self._server.get_tp_to_sorted_price_sources(trade_pairs, time_ms, live)

    def time_since_last_ws_ping_s(self, trade_pair: TradePair) -> float | None:
        """Get time since last websocket ping for a trade pair."""
        return self._server.time_since_last_ws_ping_s(trade_pair)

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms) -> dict:
        """Fetch candles for multiple trade pairs in a time window."""
        return self._server.get_candles(trade_pairs, start_time_ms, end_time_ms)

    def get_close_at_date(self, trade_pair, timestamp_ms, order=None, verbose=True):
        """Get closing price at a specific date."""
        return self._server.get_close_at_date(trade_pair, timestamp_ms, order, verbose)

    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> Tuple[float, float, int]:
        """Get bid/ask quote for a trade pair."""
        return self._server.get_quote(trade_pair, processed_ms)

    def get_quote_usd_conversion(self, order, position):
        """Get the conversion rate between an order's quote currency and USD."""
        return self._server.get_quote_usd_conversion(order, position)

    def get_stock_splits(self, time_ms: int) -> dict[str, float]:
        return self._server.get_stock_splits(time_ms)

    def set_test_price_source(self, trade_pair: TradePair, price_source: PriceSource) -> None:
        """Set test price source for a specific trade pair (test-only)."""
        return self._server.set_test_price_source(trade_pair, price_source)

    def clear_test_price_sources(self) -> None:
        """Clear all test price sources (test-only)."""
        return self._server.clear_test_price_sources()

    def set_test_market_open(self, is_open: bool) -> None:
        """Set market open override for testing (test-only)."""
        return self._server.set_test_market_open(is_open)

    def clear_test_market_open(self) -> None:
        """Clear market open override (test-only)."""
        return self._server.clear_test_market_open()

    def set_test_candle_data(self, trade_pair: TradePair, start_ms: int, end_ms: int, candles: List[PriceSource]) -> None:
        """Set test candle data for a specific trade pair and time window (test-only)."""
        return self._server.set_test_candle_data(trade_pair, start_ms, end_ms, candles)

    def clear_test_candle_data(self) -> None:
        """Clear all test candle data (test-only)."""
        return self._server.clear_test_candle_data()
