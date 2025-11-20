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

        Args:
            running_unit_tests: Whether running in test mode
            start_server: Whether to start RPC server (not used - server is started by api_manager)
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

        # Don't start server here - the server is started by api_manager in the websocket process
        # This client is used by other processes (market_order_manager, position_manager, etc.)
        # to connect to the existing server

        # Connect to existing server
        if not running_unit_tests:
            self._connect_client()
            bt.logging.info(f"WebSocketNotifier: Connected to RPC server at {self._address}")

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

        Automatically reconnects to existing RPC server running in the websocket server process.
        """
        import os

        msg = (
            f"[WS_NOTIFIER_UNPICKLE] __setstate__ called in PID {os.getpid()}, "
            f"running_unit_tests={state.get('running_unit_tests', 'UNKNOWN')}"
        )
        bt.logging.info(msg)
        print(msg, flush=True)

        self.__dict__.update(state)

        # In test mode, create mock server instance
        if self.running_unit_tests:
            bt.logging.info("[WS_NOTIFIER_UNPICKLE] Test mode - creating mock server instance")
            self._server_proxy = self._create_direct_server()
            bt.logging.success(
                f"[WS_NOTIFIER_UNPICKLE] Mock server created, type: {type(self._server_proxy)}"
            )
            return

        # Reconnect to existing RPC server
        msg_reconnect = f"[WS_NOTIFIER_UNPICKLE] Production mode - reconnecting to RPC server on port {self.port}"
        bt.logging.info(msg_reconnect)
        print(msg_reconnect, flush=True)

        self._connect_client_only()

        # Verify connection succeeded
        if self._server_proxy is None:
            msg_error = "[WS_NOTIFIER_UNPICKLE] CRITICAL: _server_proxy is still None after _connect_client_only()!"
            bt.logging.error(msg_error)
            print(msg_error, flush=True)
        else:
            msg_success = f"[WS_NOTIFIER_UNPICKLE] _server_proxy successfully set: {type(self._server_proxy)}"
            bt.logging.success(msg_success)
            print(msg_success, flush=True)

    def _connect_client_only(self):
        """
        Connect to existing RPC server (client-only mode for child processes).

        This is called when websocket_notifier is unpickled in a child process.
        """
        bt.logging.info(
            f"[WS_NOTIFIER_UNPICKLE] Starting client-only reconnection on port {self.port}"
        )

        # Use stable authkey (must match what server used)
        if not hasattr(self, '_authkey'):
            self._authkey = ValiConfig.get_rpc_authkey(self.service_name, self.port)
            bt.logging.debug(f"[WS_NOTIFIER_UNPICKLE] Generated authkey from port {self.port}")

        # Connect to existing server (inherited from RPCServiceBase)
        try:
            bt.logging.info(f"[WS_NOTIFIER_UNPICKLE] Attempting to connect to {self._address}")
            self._connect_client()
            bt.logging.success(
                f"[WS_NOTIFIER_UNPICKLE] ✓ Client-only mode connected to existing RPC server at {self._address}"
            )
            bt.logging.info(f"[WS_NOTIFIER_UNPICKLE] _server_proxy type: {type(self._server_proxy)}")
        except Exception as e:
            error_trace = traceback.format_exc()
            bt.logging.error(
                f"[WS_NOTIFIER_UNPICKLE] ✗ Failed to reconnect in client-only mode: {e}\n"
                f"Address: {self._address}\n"
                f"Port: {self.port}\n"
                f"Traceback:\n{error_trace}"
            )
            # Don't raise - allow child process to continue, but log the error
            # The _server_proxy will be None and methods will fail with clearer errors
            bt.logging.error(
                "[WS_NOTIFIER_UNPICKLE] Child process will not be able to broadcast WebSocket updates. "
                "This likely means the RPC server is not running or not accessible from this process."
            )

    # ==================== Client Methods (proxy to RPC) ====================
    def broadcast_position_update(self, position: Position, miner_repo_version: str = None) -> bool:
        """
        Broadcast a position update to all subscribed WebSocket clients.

        This method is called from other processes (MarketOrderManager, PositionManager,
        EliminationManager) to notify WebSocket clients of position changes.

        Args:
            position: Position object to broadcast
            miner_repo_version: Optional miner repository version for the websocket dict

        Returns:
            bool: True if message was queued successfully, False otherwise
        """
        try:
            return self._server_proxy.broadcast_position_update_rpc(position, miner_repo_version)
        except Exception as e:
            bt.logging.error(f"WebSocketNotifier: Error broadcasting position update: {e}")
            return False
