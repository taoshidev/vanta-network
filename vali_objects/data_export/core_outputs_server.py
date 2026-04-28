# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
CoreOutputsServer and CoreOutputsClient - RPC-based checkpoint generation service.

This module provides:
- CoreOutputsServer: Wraps CoreOutputsManager and exposes checkpoint generation via RPC
- CoreOutputsClient: Lightweight RPC client for accessing checkpoint data

Architecture:
- CoreOutputsManager (in generate_request_core.py): Contains all heavy business logic
- CoreOutputsServer: Wraps manager and exposes methods via RPC (inherits from RPCServerBase)
- CoreOutputsClient: Lightweight RPC client (inherits from RPCClientBase)
- Forward-compatible: Consumers create their own CoreOutputsClient instances

This follows the same pattern as PerfLedgerServer/PerfLedgerManager and
EliminationServer/EliminationManager.

Usage:
    # In validator.py - create server with daemon for periodic cache refresh
    core_outputs_server = CoreOutputsServer(
        slack_notifier=slack_notifier,
        start_server=True,
        start_daemon=True  # Daemon refreshes checkpoint cache every 60s
    )

    # In consumers - create client
    client = CoreOutputsClient()
    checkpoint = client.generate_request_core()
"""

import traceback

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode, load_hl_dynamic_registry
from vali_objects.data_export.core_outputs_manager import CoreOutputsManager

from shared_objects.rpc.rpc_server_base import RPCServerBase


class CoreOutputsServer(RPCServerBase):
    """
    RPC server for checkpoint generation and core outputs.

    Wraps CoreOutputsManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to clients.

    This follows the same pattern as PerfLedgerServer and EliminationServer.
    """
    service_name = ValiConfig.RPC_COREOUTPUTS_SERVICE_NAME
    service_port = ValiConfig.RPC_COREOUTPUTS_PORT

    def __init__(
        self,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize CoreOutputsServer.

        The server creates its own CoreOutputsManager internally (forward compatibility pattern).

        Args:
            running_unit_tests: Whether running in unit test mode
            slack_notifier: Optional SlackNotifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon (refreshes checkpoint cache every 60s)
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        self._last_upload_hour = -1

        # Initialize RPCServerBase (handles RPC server lifecycle, daemon, watchdog)
        super().__init__(
            service_name=ValiConfig.RPC_COREOUTPUTS_SERVICE_NAME,
            port=ValiConfig.RPC_COREOUTPUTS_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after manager is initialized
            daemon_interval_s=60.0,  # Refresh checkpoint cache every 60 seconds
            hang_timeout_s = 300.0,  # 5 minute hang timeout
            connection_mode=connection_mode,
            daemon_stagger_s=30,
        )

        # Create the actual CoreOutputsManager (contains all business logic)
        self._manager = CoreOutputsManager(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        bt.logging.info(f"[COREOUTPUTS_SERVER] CoreOutputsManager initialized")

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work - delegates to manager's checkpoint generation.

        CoreOutputsServer daemon periodically generates checkpoint data to keep
        the in-memory cache fresh for API requests. This pre-warms the cache so
        API responses are instant rather than requiring on-demand generation.

        Runs every ~60 seconds (controlled by daemon_interval_s in __init__).
        """
        try:
            load_hl_dynamic_registry()

            time_now = TimeUtil.now_in_millis()
            bt.logging.debug(f"CoreOutputsServer daemon: generating checkpoint cache...")

            # Ensure upload occurs at least once per hour, after the 24 minute mark
            datetime_now = TimeUtil.generate_start_timestamp(0)
            if (datetime_now.hour != self._last_upload_hour) and (datetime_now.minute >= 24):
                upload_needed = True
                self._last_upload_hour = datetime_now.hour
            else:
                upload_needed = False

            # Delegate to manager for checkpoint generation
            self._manager.generate_request_core(
                create_production_files=True,
                save_production_files=True,
                upload_production_files=upload_needed
            )

            elapsed_ms = TimeUtil.now_in_millis() - time_now
            bt.logging.info(f"CoreOutputsServer daemon: checkpoint cache refreshed in {elapsed_ms}ms")

        except Exception as e:
            bt.logging.error(f"CoreOutputsServer daemon error: {e}")
            bt.logging.error(traceback.format_exc())
            # Don't re-raise - let daemon continue on next iteration

    # ==================== Properties (Forward Compatibility) ====================

    @property
    def position_manager(self):
        """Get position manager client (via manager)."""
        return self._manager.position_manager

    @property
    def elimination_manager(self):
        """Get elimination manager client (via manager)."""
        return self._manager.elimination_manager

    @property
    def challengeperiod_manager(self):
        """Get challenge period client (via manager)."""
        return self._manager.challengeperiod_manager

    @property
    def contract_manager(self):
        """Get contract client (via manager - forward compatibility)."""
        return self._manager.contract_manager

    # ==================== RPC Methods (exposed to clients) ====================

    def generate_request_core_rpc(
        self,
        get_dash_data_hotkey: str | None = None,
        write_and_upload_production_files: bool = False,
        create_production_files: bool = True,
        save_production_files: bool = False,
        upload_production_files: bool = False
    ) -> dict:
        """
        Generate request core data and optionally create/save/upload production files via RPC.

        Delegates to manager for actual checkpoint generation.
        """
        return self._manager.generate_request_core(
            get_dash_data_hotkey=get_dash_data_hotkey,
            write_and_upload_production_files=write_and_upload_production_files,
            create_production_files=create_production_files,
            save_production_files=save_production_files,
            upload_production_files=upload_production_files
        )

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def generate_request_core(
        self,
        get_dash_data_hotkey: str | None = None,
        write_and_upload_production_files=False,
        create_production_files=True,
        save_production_files=False,
        upload_production_files=False
    ) -> dict:
        """
        Generate request core data - delegates to manager.

        This is a forward-compatible alias for direct server access (tests).
        """
        return self._manager.generate_request_core(
            get_dash_data_hotkey=get_dash_data_hotkey,
            write_and_upload_production_files=write_and_upload_production_files,
            create_production_files=create_production_files,
            save_production_files=save_production_files,
            upload_production_files=upload_production_files
        )

    @staticmethod
    def cleanup_test_files():
        """Clean up test files - delegates to manager."""
        return CoreOutputsManager.cleanup_test_files()


# ==================== Entry Point for Subprocess-Based Server ====================

def start_core_outputs_server(
    slack_notifier,
    address,
    authkey,
    server_ready
):
    """
    Entry point for starting CoreOutputsServer in a separate process.

    Args:
        slack_notifier: Slack notifier instance
        address: RPC server address tuple (host, port)
        authkey: RPC authentication key
        server_ready: Event to signal when server is ready
    """
    from setproctitle import setproctitle
    setproctitle("vali_CoreOutputsServer")

    # Create server instance (creates its own RPC clients internally)
    server = CoreOutputsServer(
        slack_notifier=slack_notifier,
        start_server=False,  # Don't start thread-based server
        start_daemon=False   # No daemon needed
    )

    # Serve via RPC (uses RPCServerBase helper)
    RPCServerBase.serve_rpc(
        server_instance=server,
        service_name=ValiConfig.RPC_COREOUTPUTS_SERVICE_NAME,
        address=address,
        authkey=authkey,
        server_ready=server_ready
    )


if __name__ == "__main__":
    # NOTE: This standalone test script needs the RPC servers running
    # In production, CoreOutputsServer creates its own clients

    # CoreOutputsServer creates its own RPC clients
    server = CoreOutputsServer(
        running_unit_tests=False,
        start_server=True,
        start_daemon=False
    )

    result = server.generate_request_core(
        create_production_files=True,
        save_production_files=True,
        upload_production_files=True
    )
    print(f"Generated checkpoint with keys: {result.keys()}")
