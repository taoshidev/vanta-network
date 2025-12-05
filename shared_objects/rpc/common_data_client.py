from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class CommonDataClient(RPCClientBase):
    """
    Lightweight RPC client for accessing shared validator state.

    Usage:
        # Create client (connects automatically unless in test mode)
        client = CommonDataClient(connect_immediately=True)

        # Check shutdown
        if client.is_shutdown():
            return

        # Check sync state
        if client.get_sync_in_progress():
            bt.logging.debug("Sync in progress, pausing...")

        # Get sync epoch for stale data detection
        epoch = client.get_sync_epoch()
        # ... do work ...
        if client.get_sync_epoch() != epoch:
            bt.logging.warning("Sync occurred, data may be stale")

    Test Mode:
        # For unit tests, use direct server reference
        client = CommonDataClient(connect_immediately=False)
        client.set_direct_server(server_instance)
    """

    def __init__(
        self,
        port: int = None,
        connect_immediately: bool = True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False
    ):
        """
        Initialize CommonDataClient.

        Args:
            port: RPC port (default: ValiConfig.RPC_COMMONDATA_PORT)
            connect_immediately: Whether to connect on init
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_COMMONDATA_SERVICE_NAME,
            port=port or ValiConfig.RPC_COMMONDATA_PORT,
            connect_immediately=connect_immediately and connection_mode == RPCConnectionMode.RPC,
            connection_mode=connection_mode
        )

    # ==================== Shutdown Dict Methods ====================

    def get_shutdown_dict(self) -> dict:
        """Get the shutdown dict."""
        return self.call("get_shutdown_dict_rpc")

    def is_shutdown(self) -> bool:
        """Check if shutdown is in progress."""
        return self.call("is_shutdown_rpc")

    def set_shutdown(self, value: bool = True) -> None:
        """Set shutdown state."""
        self.call("set_shutdown_rpc", value)

    # ==================== Sync In Progress Methods ====================

    def get_sync_in_progress(self) -> bool:
        """Get sync_in_progress flag."""
        return self.call("get_sync_in_progress_rpc")

    def set_sync_in_progress(self, value: bool) -> None:
        """Set sync_in_progress flag."""
        self.call("set_sync_in_progress_rpc", value)

    # ==================== Sync Epoch Methods ====================

    def get_sync_epoch(self) -> int:
        """Get current sync epoch."""
        return self.call("get_sync_epoch_rpc")

    def increment_sync_epoch(self) -> int:
        """Increment sync epoch and return new value."""
        return self.call("increment_sync_epoch_rpc")

    def set_sync_epoch(self, value: int) -> None:
        """Set sync epoch to specific value."""
        self.call("set_sync_epoch_rpc", value)

    # ==================== Combined State Methods ====================

    def get_all_state(self) -> dict:
        """Get all shared state in a single call."""
        return self.call("get_all_state_rpc")

    # ==================== Test State Cleanup ====================

    def clear_test_state(self) -> None:
        """
        Clear ALL test-sensitive state (comprehensive reset for test isolation).

        This resets:
        - shutdown_dict (prevents false shutdown in tests)
        - sync_in_progress (prevents daemons from incorrectly pausing)
        - sync_epoch (resets stale data detection counter)

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.
        """
        self.call("clear_test_state_rpc")

    # ==================== Convenience Properties ====================

    @property
    def shutdown_dict(self) -> dict:
        """Property access to shutdown dict (for backward compatibility)."""
        return self.get_shutdown_dict()

    @property
    def sync_in_progress_value(self) -> bool:
        """Property access to sync_in_progress (mimics IPC Value.value pattern)."""
        return self.get_sync_in_progress()

    @property
    def sync_epoch_value(self) -> int:
        """Property access to sync_epoch (mimics IPC Value.value pattern)."""
        return self.get_sync_epoch()
