# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
CommonDataServer - Centralized RPC server for shared validator state.

This server manages cross-process shared data that was previously managed via IPC Manager:
- shutdown_dict: Global shutdown flag for graceful termination
- sync_in_progress: Flag to pause daemon processes during position sync
- sync_epoch: Counter incremented each sync cycle to detect stale data

Architecture:
- CommonDataServer: RPC server that manages shared state (runs in validator process)
- CommonDataClient: Lightweight RPC client for consumers to access/modify state

Forward Compatibility Pattern:
All consumers create their own CommonDataClient internally:
    self._common_data_client = CommonDataClient(connection_mode=connection_mode)

Usage in validator.py:
    # Start CommonDataServer early in initialization
    self.common_data_server = CommonDataServer(
        slack_notifier=self.slack_notifier,
        start_server=True,
        connection_mode=RPCConnectionMode.RPC
    )

    # Pass to consumers (they create their own clients internally)
    self.elimination_server = EliminationServer(connection_mode=RPCConnectionMode.RPC)
    # EliminationServer creates its own CommonDataClient internally

Usage in consumers:
    class EliminationServer(RPCServerBase):
        def __init__(self, ..., connection_mode=RPCConnectionMode.RPC):
            # Forward compatibility: create own CommonDataClient
            self._common_data_client = CommonDataClient(
                connection_mode=connection_mode
            )

        @property
        def shutdown_dict(self):
            return self._common_data_client.get_shutdown_dict()

        @property
        def sync_in_progress(self):
            return self._common_data_client.get_sync_in_progress()
"""
import threading
import time
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc_server_base import RPCServerBase
from shared_objects.rpc_client_base import RPCClientBase


class CommonDataServer(RPCServerBase):
    """
    RPC server for shared validator state management.

    Manages:
    - shutdown_dict: Global shutdown flag (dict used as truthy check)
    - sync_in_progress: Boolean flag for sync state
    - sync_epoch: Integer counter for sync cycles

    Inherits from RPCServerBase for RPC server lifecycle.
    No daemon needed - this is a simple state server.
    """
    service_name = ValiConfig.RPC_COMMONDATA_SERVICE_NAME
    service_port = ValiConfig.RPC_COMMONDATA_PORT

    def __init__(
        self,
        slack_notifier=None,
        start_server: bool = True,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize CommonDataServer.

        Args:
            slack_notifier: Optional SlackNotifier for alerts
            start_server: Whether to start RPC server immediately
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        # Initialize shared state
        self.running_unit_tests = running_unit_tests
        self._shutdown_dict = {}
        self._sync_in_progress = False
        self._sync_epoch = 0
        self._state_lock = threading.Lock()

        # Initialize RPCServerBase (no daemon needed for this simple state server)
        super().__init__(
            service_name=ValiConfig.RPC_COMMONDATA_SERVICE_NAME,
            port=ValiConfig.RPC_COMMONDATA_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # No daemon needed
            connection_mode=connection_mode
        )

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """No daemon needed for this simple state server."""
        pass

    # ==================== Shutdown Dict RPC Methods ====================

    def get_shutdown_dict_rpc(self) -> dict:
        """Get the shutdown dict (truthy if shutting down)."""
        with self._state_lock:
            return dict(self._shutdown_dict)

    def is_shutdown_rpc(self) -> bool:
        """Check if shutdown is in progress (bool for easier use)."""
        with self._state_lock:
            return bool(self._shutdown_dict)

    def set_shutdown_rpc(self, value: bool = True) -> None:
        """
        Set shutdown state.

        Args:
            value: If True, sets shutdown_dict[True] = True (triggers shutdown)
                   If False, clears shutdown_dict
        """
        with self._state_lock:
            if value:
                self._shutdown_dict[True] = True
                bt.logging.warning("[COMMON_DATA] Shutdown flag set")
            else:
                self._shutdown_dict.clear()
                bt.logging.info("[COMMON_DATA] Shutdown flag cleared")

    # ==================== Sync In Progress RPC Methods ====================

    def get_sync_in_progress_rpc(self) -> bool:
        """Get sync_in_progress flag."""
        with self._state_lock:
            return self._sync_in_progress

    def set_sync_in_progress_rpc(self, value: bool) -> None:
        """Set sync_in_progress flag."""
        with self._state_lock:
            old_value = self._sync_in_progress
            self._sync_in_progress = value
            if old_value != value:
                bt.logging.info(f"[COMMON_DATA] sync_in_progress: {old_value} -> {value}")

    # ==================== Sync Epoch RPC Methods ====================

    def get_sync_epoch_rpc(self) -> int:
        """Get current sync epoch."""
        with self._state_lock:
            return self._sync_epoch

    def increment_sync_epoch_rpc(self) -> int:
        """
        Increment sync epoch and return new value.

        Returns:
            New sync epoch value after increment
        """
        with self._state_lock:
            old_epoch = self._sync_epoch
            self._sync_epoch += 1
            bt.logging.info(f"[COMMON_DATA] Incrementing sync epoch {old_epoch} -> {self._sync_epoch}")
            return self._sync_epoch

    def set_sync_epoch_rpc(self, value: int) -> None:
        """Set sync epoch to specific value."""
        with self._state_lock:
            self._sync_epoch = value

    # ==================== Test State Cleanup ====================

    def clear_test_state_rpc(self) -> None:
        """
        Clear ALL test-sensitive state (for test isolation).

        This includes:
        - shutdown_dict (prevents false shutdown in tests)
        - sync_in_progress (prevents daemons from incorrectly pausing)
        - sync_epoch (resets stale data detection counter)

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.
        """
        with self._state_lock:
            self._shutdown_dict.clear()
            self._sync_in_progress = False
            self._sync_epoch = 0
            bt.logging.debug("[COMMON_DATA] Test state cleared (shutdown, sync_in_progress, sync_epoch reset)")

    # ==================== Combined State RPC Methods ====================

    def get_all_state_rpc(self) -> dict:
        """
        Get all shared state in a single RPC call (reduces round trips).

        Returns:
            dict with keys: shutdown_dict, sync_in_progress, sync_epoch
        """
        with self._state_lock:
            return {
                "shutdown_dict": dict(self._shutdown_dict),
                "sync_in_progress": self._sync_in_progress,
                "sync_epoch": self._sync_epoch,
                "timestamp_ms": TimeUtil.now_in_millis()
            }

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        with self._state_lock:
            return {
                "is_shutdown": bool(self._shutdown_dict),
                "sync_in_progress": self._sync_in_progress,
                "sync_epoch": self._sync_epoch
            }


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


# ==================== Server Entry Point ====================

def start_common_data_server(
    slack_notifier=None,
    server_ready=None
):
    """
    Entry point for starting CommonDataServer in a separate process.

    Args:
        slack_notifier: Optional SlackNotifier for alerts
        server_ready: Event to signal when server is ready
    """
    from setproctitle import setproctitle
    setproctitle("vali_CommonDataServerProcess")

    # Create server
    server = CommonDataServer(
        slack_notifier=slack_notifier,
        start_server=True,
        connection_mode=RPCConnectionMode.RPC
    )

    bt.logging.success(f"CommonDataServer ready on port {ValiConfig.RPC_COMMONDATA_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown
    while not server.is_shutdown_rpc():
        time.sleep(1)

    server.shutdown()
    bt.logging.info("CommonDataServer process exiting")
