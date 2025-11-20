# developer: jbonilla
# Copyright © 2025 Taoshi Inc
import os
import bittensor as bt
from shared_objects.rpc_service_base import RPCServiceBase
from vali_objects.vali_config import ValiConfig
import traceback
from vali_objects.position import Position
from time_util.time_util import TimeUtil


class WebSocketNotifier(RPCServiceBase):
    """
    RPC Client for WebSocketNotifier - broadcasts position updates to WebSocket clients via RPC.

    This client connects to WebSocketNotifierServer running in the WebSocket server process.
    Replaces the multiprocessing.Queue pattern with proper RPC calls.
    """

    def __init__(self, running_unit_tests=False, slack_notifier=None):
        """
        Initialize WebSocketNotifier.

        Connection to RPC server is deferred until first use.
        If server is not available (--serve not enabled), broadcasts are silently skipped.
        """
        # Initialize RPCServiceBase
        RPCServiceBase.__init__(
            self,
            service_name=ValiConfig.RPC_WEBSOCKET_NOTIFIER_SERVICE_NAME,
            port=ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT,
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )

        # Track if server is unavailable to avoid repeated connection attempts
        self._server_unavailable = False

        # Don't connect here - connection happens on first broadcast
        # Server is started by api_manager (if --serve is enabled)

    def _create_direct_server(self):
        """
        Create direct in-memory instance for tests.

        In test mode, returns a mock instance that doesn't actually broadcast.
        """
        # For tests, we don't need actual websocket broadcasting
        # Return a simple mock that accepts calls but does nothing
        class MockWebSocketNotifierServer:
            def broadcast_position_update_rpc(self, position: Position, miner_repo_version: str = None) -> bool:
                bt.logging.debug("[MOCK_WS_NOTIFIER] broadcast_position_update_rpc called (no-op in test mode)")
                return True

            def health_check_rpc(self):
                return {
                    "status": "ok",
                    "timestamp_ms": TimeUtil.now_in_millis(),
                    "connected_clients": 0,
                    "subscribed_clients": 0,
                    "queue_size": 0,
                    "queue_maxsize": 1000
                }

        return MockWebSocketNotifierServer()

    def _start_server_process(self, address, authkey, server_ready):
        """
        Start RPC server in separate process.

        NOTE: This is not used for WebSocketNotifier because the server is started
        by api_manager in the WebSocket server's process. This method is here for
        compatibility with RPCServiceBase but should not be called.
        """
        raise NotImplementedError(
            "WebSocketNotifier server is started by api_manager, not by the client. "
            "Use api_manager.start_websocket_notifier_server() instead."
        )

    def __getstate__(self):
        """
        Prepare object for pickling (when passed to child processes).

        When websocket_notifier is passed to other components that run in separate
        processes, this method ensures the object can be pickled properly.

        The unpickled object will be a client-only instance that connects to the
        existing RPC server.
        """

        msg = (
            f"[WS_NOTIFIER_PICKLE] __getstate__ called in PID {os.getpid()}, "
            f"running_unit_tests={self.running_unit_tests}, port={self.port}"
        )
        bt.logging.info(msg)
        print(msg, flush=True)

        state = self.__dict__.copy()

        # Mark as client-only so unpickled instance connects to existing server
        state['_is_client_only'] = True

        # Don't pickle process/proxy objects (they're not picklable anyway)
        state['_server_process'] = None
        state['_client_manager'] = None
        state['_server_proxy'] = None

        bt.logging.debug("[WS_NOTIFIER_PICKLE] __getstate__ complete, removed unpicklable objects")

        return state

    def __setstate__(self, state):
        """
        Restore object after unpickling (in child process).
        Connection to RPC server is deferred until first use.
        """
        self.__dict__.update(state)

        # Initialize tracking attribute
        if not hasattr(self, '_server_unavailable'):
            self._server_unavailable = False

        # In test mode, create mock server instance
        if self.running_unit_tests:
            self._server_proxy = self._create_direct_server()
        # In production, connection happens on first use

    def ensure_connected(self) -> bool:
        """
        Ensure client is connected to RPC server. Connects if not already connected.
        If server is unavailable (--serve not enabled), returns False and broadcasts are skipped.
        """
        if self.running_unit_tests:
            if self._server_proxy is None:
                self._server_proxy = self._create_direct_server()
            return True

        # Already connected
        if self._server_proxy is not None:
            return True

        # Server previously determined to be unavailable
        if self._server_unavailable:
            return False

        # Try to connect once
        try:
            self._connect_client()
            bt.logging.success(f"WebSocketNotifier: Connected to RPC server at {self._address}")
            return True
        except Exception:
            # Server not available (--serve not enabled), silently skip broadcasts
            self._server_unavailable = True
            bt.logging.debug("WebSocketNotifier: Server not available, broadcasts will be skipped")
            return False

    # ==================== Client Methods (proxy to RPC) ====================
    def broadcast_position_update(self, position: Position, miner_repo_version: str = None) -> bool:
        """
        Broadcast a position update to all subscribed WebSocket clients.
        If server not available (--serve not enabled), silently returns False.
        """
        # Skip broadcast for development hotkey
        if position.miner_hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
            return True

        # Server not available (--serve not enabled)
        if not self.ensure_connected():
            return False

        try:
            return self._server_proxy.broadcast_position_update_rpc(position, miner_repo_version)
        except Exception as e:
            bt.logging.debug(f"WebSocketNotifier: Broadcast failed: {e}")
            self._server_unavailable = True
            return False
