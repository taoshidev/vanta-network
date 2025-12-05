# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
MDDCheckerClient - Lightweight RPC client for MDD (Maximum Drawdown) checking.

This client connects to the MDDCheckerServer via RPC.
Can be created in ANY process - just needs the server to be running.

"""

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class MDDCheckerClient(RPCClientBase):
    """
    Lightweight RPC client for MDDCheckerServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_MDDCHECKER_PORT.
    """

    def __init__(
        self,
        port: int = None,
        connect_immediately: bool = False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize MDD checker client.

        Args:
            port: Port number of the MDD checker server (default: ValiConfig.RPC_MDDCHECKER_PORT)
            connect_immediately: Whether to connect immediately
            running_unit_tests: Whether running in unit test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_MDDCHECKER_SERVICE_NAME,
            port=port or ValiConfig.RPC_MDDCHECKER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Main Methods ====================

    def mdd_check(self, iteration_epoch: int = None) -> None:
        """
        Trigger MDD check.

        Args:
            iteration_epoch: Sync epoch captured at start of iteration. Used to detect stale data.
        """
        self._server.mdd_check_rpc(iteration_epoch=iteration_epoch)

    def reset_debug_counters(self) -> None:
        """Reset debug counters."""
        self._server.reset_debug_counters_rpc()

    # ==================== Properties ====================

    @property
    def price_correction_enabled(self) -> bool:
        """Get price correction enabled flag."""
        return self._server.get_price_correction_enabled_rpc()

    @price_correction_enabled.setter
    def price_correction_enabled(self, value: bool):
        """Set price correction enabled flag."""
        self._server.set_price_correction_enabled_rpc(value)

    @property
    def last_price_fetch_time_ms(self) -> int:
        """Get last price fetch time."""
        return self._server.get_last_price_fetch_time_ms_rpc()

    @last_price_fetch_time_ms.setter
    def last_price_fetch_time_ms(self, value: int):
        """Set last price fetch time."""
        self._server.set_last_price_fetch_time_ms_rpc(value)

    # ==================== Daemon Control ====================

    def start_daemon(self) -> None:
        """Request daemon start on server."""
        self._server.start_daemon_rpc()
