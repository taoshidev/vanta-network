"""
LivePriceFetcher Client/Server - RPC architecture for price fetching.

This module follows the same pattern as PerfLedgerClient/PerfLedgerServer:
- LivePriceFetcherClient: Lightweight RPC client (extends RPCClientBase)
- LivePriceFetcherServer: Server that delegates to LivePriceFetcher instance (inherits from RPCServerBase)
- LivePriceFetcher: Contains all heavy logic for price fetching (in live_price_fetcher.py)

Usage in validator.py:
    # Start the server (once, early in initialization)
    self.live_price_fetcher_server = LivePriceFetcherServer(
        secrets=self.secrets,
        disable_ws=False,
        slack_notifier=self.slack_notifier,
        start_server=True,
        start_daemon=True  # Optional - enables health monitoring
    )

    # In other components, create lightweight clients
    client = LivePriceFetcherClient(running_unit_tests=False)
    price = client.get_latest_price(trade_pair)
"""
import time
from typing import List, Optional, Tuple, Dict

from shared_objects.rpc.rpc_server_base import RPCServerBase
import bittensor as bt
from vali_objects.vali_config import RPCConnectionMode

from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.price_fetcher.live_price_fetcher import LivePriceFetcher
from vali_objects.vali_dataclasses.price_source import PriceSource


class LivePriceFetcherServer(RPCServerBase):
    """
    RPC server for live price fetching.

    Inherits from RPCServerBase for unified RPC server lifecycle and daemon management.
    Manages connections to Polygon and Tiingo data services.
    Exposes methods via RPC to LivePriceFetcherClient.

    Architecture:
    - Runs RPC server in background thread (via RPCServerBase)
    - Optional daemon for health monitoring
    - Automatic shutdown via ShutdownCoordinator (inherited from RPCServerBase)
    - Handles all price fetching logic
    - Port is obtained from ValiConfig.RPC_LIVEPRICEFETCHER_PORT
    """
    service_name = ValiConfig.RPC_LIVEPRICEFETCHER_SERVICE_NAME
    service_port = ValiConfig.RPC_LIVEPRICEFETCHER_PORT

    def __init__(self, secrets, disable_ws=False, is_backtesting=False,
                 running_unit_tests=False, slack_notifier=None,
                 start_server=True, start_daemon=False, connection_mode=RPCConnectionMode.RPC):
        """
        Initialize the LivePriceFetcherServer.

        Args:
            secrets: Dictionary containing API keys for data services
            disable_ws: Whether to disable websocket connections
            is_backtesting: Whether running in backtesting mode
            running_unit_tests: Whether running unit tests
            slack_notifier: SlackNotifier for error reporting
            start_server: If True, start the RPC server immediately
            start_daemon: If True, start daemon for health monitoring
        """
        self.is_backtesting = is_backtesting
        self.running_unit_tests = running_unit_tests
        self.last_health_check_ms = 0
        self._secrets = secrets
        self._disable_ws = disable_ws

        # Create the actual LivePriceFetcher instance (contains all heavy logic)
        # This follows the PerfLedgerServer pattern: server holds manager/fetcher instance
        self._fetcher = LivePriceFetcher(
            secrets=secrets,
            disable_ws=disable_ws,
            is_backtesting=is_backtesting,
            running_unit_tests=running_unit_tests
        )

        # Initialize RPCServerBase (handles RPC server and daemon lifecycle)
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_LIVEPRICEFETCHER_SERVICE_NAME,
            port=ValiConfig.RPC_LIVEPRICEFETCHER_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=10.0,  # Health check every 10 seconds
            hang_timeout_s=120.0  # Alert if no heartbeat for 2 minutes
        )

    # ============================================================================
    # RPCServerBase ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================

    def run_daemon_iteration(self) -> None:
        """
        Called repeatedly by RPCServerBase daemon loop.

        Since price fetching is on-demand (via RPC calls), the daemon
        primarily serves as a health monitoring mechanism. The heartbeat
        updates automatically via RPCServerBase, ensuring watchdog monitoring.
        """
        # Check shutdown signal
        if self._is_shutdown():
            return

        # Optional: Could add periodic health checks for data services here
        # For now, heartbeat is sufficient for monitoring

    # ============================================================================
    # SHUTDOWN OVERRIDE (clean up data services)
    # ============================================================================

    def shutdown(self):
        """Override shutdown to clean up data service threads."""
        bt.logging.info("LivePriceFetcherServer shutting down data services...")
        self.stop_all_threads()
        super().shutdown()

    def stop_all_threads(self):
        """Stop all data service threads - delegates to fetcher."""
        if hasattr(self, '_fetcher'):
            self._fetcher.stop_all_threads()

    # ============================================================================
    # HEALTH CHECK (RPC method)
    # ============================================================================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "is_backtesting": self.is_backtesting
        }

    def health_check(self) -> dict:
        """
        Alias for health_check_rpc() for backward compatibility with client.
        """
        return self.health_check_rpc()

    # ============================================================================
    # DELEGATION METHODS (all business logic delegates to _fetcher)
    # ============================================================================

    def get_usd_base_conversion(self, trade_pair, time_ms, price, order_type, position):
        """Delegate to fetcher."""
        return self._fetcher.get_usd_base_conversion(trade_pair, time_ms, price, order_type, position)

    def get_ws_price_sources_in_window(self, trade_pair: TradePair, start_ms: int, end_ms: int) -> List[PriceSource]:
        """Delegate to fetcher."""
        return self._fetcher.get_ws_price_sources_in_window(trade_pair, start_ms, end_ms)

    def get_currency_conversion(self, base: str, quote: str):
        """Delegate to fetcher."""
        return self._fetcher.get_currency_conversion(base, quote)

    def unified_candle_fetcher(self, trade_pair, start_date, order_date, timespan="day"):
        """Delegate to fetcher."""
        return self._fetcher.unified_candle_fetcher(trade_pair, start_date, order_date, timespan)

    def get_latest_price(self, trade_pair: TradePair, time_ms=None) -> Tuple[float, List[PriceSource]] | Tuple[None, None]:
        """Delegate to fetcher."""
        return self._fetcher.get_latest_price(trade_pair, time_ms)

    def get_sorted_price_sources_for_trade_pair(self, trade_pair: TradePair, time_ms: int, live=True) -> List[PriceSource] | None:
        """Delegate to fetcher."""
        return self._fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, time_ms, live)

    def get_tp_to_sorted_price_sources(self, trade_pairs: List[TradePair], time_ms: int, live=True) -> Dict[TradePair, List[PriceSource]]:
        """Delegate to fetcher."""
        return self._fetcher.get_tp_to_sorted_price_sources(trade_pairs, time_ms, live)

    def time_since_last_ws_ping_s(self, trade_pair: TradePair) -> float | None:
        """Delegate to fetcher."""
        return self._fetcher.time_since_last_ws_ping_s(trade_pair)

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms) -> dict:
        """Delegate to fetcher."""
        return self._fetcher.get_candles(trade_pairs, start_time_ms, end_time_ms)

    def get_close_at_date(self, trade_pair, timestamp_ms, order=None, verbose=True):
        """Delegate to fetcher."""
        return self._fetcher.get_close_at_date(trade_pair, timestamp_ms, order, verbose)

    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> Tuple[float, float, int]:
        """Delegate to fetcher."""
        return self._fetcher.get_quote(trade_pair, processed_ms)

    def get_quote_usd_conversion(self, order, position):
        """Delegate to fetcher."""
        return self._fetcher.get_quote_usd_conversion(order, position)

    def get_stock_splits(self, time_ms: int) -> dict[str, float]:
        return self._fetcher.get_stock_splits(time_ms)

    def set_test_price_source(self, trade_pair: TradePair, price_source: PriceSource) -> None:
        """
        Test-only RPC method to set price source for a trade pair.
        Only available when running_unit_tests=True.
        """
        return self._fetcher.set_test_price_source(trade_pair, price_source)

    def clear_test_price_sources(self) -> None:
        """Test-only RPC method to clear all test price sources."""
        return self._fetcher.clear_test_price_sources()

    def set_test_market_open(self, is_open: bool) -> None:
        """
        Test-only RPC method to override market open status.
        Only available when running_unit_tests=True.
        """
        return self._fetcher.set_test_market_open(is_open)

    def clear_test_market_open(self) -> None:
        """Test-only RPC method to clear market open override."""
        return self._fetcher.clear_test_market_open()

    def set_test_candle_data(self, trade_pair: TradePair, start_ms: int, end_ms: int, candles: List[PriceSource]) -> None:
        """
        Test-only RPC method to inject candle data for specific trade pair and time window.
        Only available when running_unit_tests=True.
        """
        return self._fetcher.set_test_candle_data(trade_pair, start_ms, end_ms, candles)

    def clear_test_candle_data(self) -> None:
        """Test-only RPC method to clear all test candle data."""
        return self._fetcher.clear_test_candle_data()

    def is_market_open(self, trade_pair: TradePair, time_ms: int) -> bool:
        return self._fetcher.is_market_open(trade_pair, time_ms)



if __name__ == "__main__":
    from vali_objects.utils.vali_utils import ValiUtils
    from vali_objects.vali_config import TradePair

    secrets = ValiUtils.get_secrets()
    server = LivePriceFetcherServer(secrets, disable_ws=True, start_server=False)
    ans = server.get_close_at_date(TradePair.TAOUSD, 1733304060475)
    print('@@@@', ans, '@@@@@')
    time.sleep(100000)
