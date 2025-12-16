# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
MDDCheckerServer - RPC server for MDD (Maximum Drawdown) checking and price corrections.

This server runs in its own process and handles:
- Real-time price corrections for recent orders
- Position return updates using live prices
- Periodic MDD checking for all miners

Architecture:
- Wraps MDDChecker and exposes its methods via RPC
- MDDChecker contains all the business logic
- Server handles RPC lifecycle and daemon management

Usage:
    # In validator.py - create server
    from vali_objects.utils.mdd_checker_server import MDDCheckerServer
    mdd_checker_server = MDDCheckerServer(
        slack_notifier=slack_notifier,
        start_server=True,
        start_daemon=True
    )
"""
import bittensor as bt

from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.utils.mdd_checker.mdd_checker import MDDChecker
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class MDDCheckerServer(RPCServerBase):
    """
    RPC server for MDD checking and price corrections.

    Wraps MDDChecker and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC.

    Architecture:
    - Runs in its own process (or thread in test mode)
    - Creates MDDChecker instance which handles business logic
    - Ports are obtained from ValiConfig
    """
    service_name = ValiConfig.RPC_MDDCHECKER_SERVICE_NAME
    service_port = ValiConfig.RPC_MDDCHECKER_PORT

    def __init__(
        self,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize MDDCheckerServer.

        Args:
            running_unit_tests: Whether running in unit test mode
            slack_notifier: Optional SlackNotifier for error alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        # Initialize RPCServerBase (handles RPC server and daemon lifecycle)
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_MDDCHECKER_SERVICE_NAME,
            port=ValiConfig.RPC_MDDCHECKER_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # Defer until initialization complete
            daemon_interval_s=ValiConfig.MDD_CHECK_REFRESH_TIME_MS / 1000.0,  # Convert ms to seconds
            hang_timeout_s=120.0  # MDD check can take a while
        )

        # Create the MDDChecker instance that contains all business logic
        self._checker = MDDChecker(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

        bt.logging.success("MDDCheckerServer initialized")

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Checks for sync in progress, then runs MDD check.
        """
        if self._checker.sync_in_progress:
            bt.logging.debug("MDDCheckerServer: Sync in progress, pausing...")
            return

        iteration_epoch = self._checker.sync_epoch
        self._checker.mdd_check(iteration_epoch=iteration_epoch)

    # ==================== Properties (forward to checker) ====================

    @property
    def price_correction_enabled(self):
        """Get price correction enabled flag (forward to checker)."""
        return self._checker.price_correction_enabled

    @price_correction_enabled.setter
    def price_correction_enabled(self, value: bool):
        """Set price correction enabled flag (forward to checker)."""
        self._checker.price_correction_enabled = value

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "n_orders_corrected": self._checker.n_orders_corrected,
            "n_miners_corrected": len(self._checker.miners_corrected),
            "n_poly_api_requests": self._checker.n_poly_api_requests
        }

    def mdd_check_rpc(self, iteration_epoch: int = None) -> None:
        """
        Trigger MDD check via RPC.

        Args:
            iteration_epoch: Sync epoch captured at start of iteration. Used to detect stale data.
        """
        self._checker.mdd_check(iteration_epoch=iteration_epoch)

    def reset_debug_counters_rpc(self) -> None:
        """Reset debug counters via RPC."""
        self._checker.reset_debug_counters()

    def get_price_correction_enabled_rpc(self) -> bool:
        """Get price correction enabled flag via RPC."""
        return self._checker.price_correction_enabled

    def set_price_correction_enabled_rpc(self, value: bool) -> None:
        """Set price correction enabled flag via RPC."""
        self._checker.price_correction_enabled = value

    def get_last_price_fetch_time_ms_rpc(self) -> int:
        """Get last price fetch time via RPC."""
        return self._checker.last_price_fetch_time_ms

    def set_last_price_fetch_time_ms_rpc(self, value: int) -> None:
        """Set last price fetch time via RPC."""
        self._checker.last_price_fetch_time_ms = value

    # ==================== Direct Access Methods (for backward compatibility in tests) ====================

    def reset_debug_counters(self):
        """Reset debug counters (direct access for tests)."""
        self._checker.reset_debug_counters()

    def mdd_check(self, iteration_epoch: int = None):
        """Run MDD check (direct access for tests/internal use)."""
        self._checker.mdd_check(iteration_epoch=iteration_epoch)

    @property
    def last_price_fetch_time_ms(self):
        """Get last price fetch time (direct access for tests)."""
        return self._checker.last_price_fetch_time_ms

    @last_price_fetch_time_ms.setter
    def last_price_fetch_time_ms(self, value: int):
        """Set last price fetch time (direct access for tests)."""
        self._checker.last_price_fetch_time_ms = value

