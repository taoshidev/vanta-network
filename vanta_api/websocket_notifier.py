# developer: jbonilla
# Copyright (c) 2025 Taoshi Inc
"""
WebSocketNotifier - RPC server and client for WebSocket broadcasting.

This module provides both the client for WebSocket position broadcasting via RPC.

The server maintains a message queue and broadcasts to WebSocket clients.
The client allows other processes to queue messages for broadcasting.

Client Usage:
    from vanta_api.websocket_notifier import WebSocketNotifierClient

    client = WebSocketNotifierClient()
    client.broadcast_position_update(position)
"""
import threading
from typing import Optional


from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.log import logger


# ==================== Client Implementation ====================

class WebSocketNotifierClient(RPCClientBase):
    """
    Lightweight RPC client for WebSocketNotifierServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT.

    In LOCAL mode (connection_mode=RPCConnectionMode.LOCAL), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct WebSocketNotifierServer instance.
    """

    def __init__(
        self,
        port: int = None,
        connect_immediately: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize WebSocket notifier client.

        Args:
            port: Port number of the WebSocket notifier server (default: ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT)
            connect_immediately: If True, connect in __init__. If False, call connect() later.
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
        """
        super().__init__(
            service_name=ValiConfig.RPC_WEBSOCKET_NOTIFIER_SERVICE_NAME,
            port=port or ValiConfig.RPC_WEBSOCKET_NOTIFIER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )
        # Guards the disconnect+reconnect sequence in _call_with_reconnect(). The notifier is
        # shared across threads (HyperliquidTracker, MarketOrderManager, EntityManager) — without
        # this, two threads hitting a dead proxy simultaneously would race the rebuild.
        self._reconnect_lock = threading.Lock()

    # ==================== Reconnect-on-failure (see runnable/spike_rpc_reconnect.py) ====================

    def _call_with_reconnect(self, method_name: str, *args, **kwargs):
        """
        Invoke an RPC method; on failure, rebuild the connection ONCE and retry.

        Why this exists (empirically verified 2026-07-07 by the spike): after the WebSocket
        server process restarts, a long-lived client's proxy is PERMANENTLY dead — the proxy
        holds a server-assigned object token that died with the old server process, so neither
        the same thread (EOFError) nor a fresh thread (RemoteError/KeyError) recovers. The ONLY
        recovery is disconnect() + connect(), which rebuilds the manager and fetches a fresh
        proxy. Without this, every broadcast after a WS-only restart is silently dropped until
        the CORE restarts — defeating the REST/WS process split.

        The reconnect is BOUNDED (a single quick connect attempt, not the default 5x1s retry):
        callers sit on the HL order path, so a WS outage must fail fast (broadcast dropped,
        same as before) rather than inject seconds of latency. The moment WS is back, the next
        call's reconnect succeeds and broadcasts RESUME.

        Raises on final failure — callers keep their existing swallow-and-log semantics.
        """
        if self.connection_mode == RPCConnectionMode.LOCAL:
            # Direct in-process server (tests): no transport to heal; call as-is.
            return getattr(self._server, method_name)(*args, **kwargs)

        # Own ALL connects on this path so every attempt is bounded to 1 quick try. If we let
        # the lazy `_server` property connect, a not-connected client (e.g. after a failed
        # reconnect below) would use the class defaults (5 x 1s) and inject ~5s of latency into
        # every broadcast while WS is down.
        if self._proxy is None or not self._connected:
            with self._reconnect_lock:
                self.connect(max_retries=1, retry_delay=0.25)  # raises fast if WS is down

        try:
            return getattr(self._server, method_name)(*args, **kwargs)
        except Exception as first_error:
            with self._reconnect_lock:
                # Another thread may have already rebuilt the connection while we waited on
                # the lock — connect() below early-returns in that case, which is fine.
                try:
                    self.disconnect()
                except Exception:
                    pass  # stale-state cleanup is best-effort
                # disconnect() removed us from the client registry (used by disconnect_all()
                # test cleanup); restore registration so this long-lived client stays tracked.
                self._instance_id = RPCClientBase._register_instance(self)
                # Single quick attempt: connection-refused returns immediately when WS is down.
                self.connect(max_retries=1, retry_delay=0.25)
                # Retry INSIDE the lock: broadcasts are millisecond queue-puts server-side, and
                # this prevents a concurrent failer from tearing down the proxy we just rebuilt
                # before we use it (which would punt us onto the slow default-connect path).
                result = getattr(self._server, method_name)(*args, **kwargs)
            logger.info(
                f"WebSocketNotifierClient: reconnected to WS notifier and resumed after: {first_error}"
            )
            return result

    # ==================== Client Methods ====================

    def broadcast_position_update(self, position: Position, miner_repo_version: str = None) -> None:
        """
        Broadcast a position update to all subscribed WebSocket clients.

        Args:
            position: Position object to broadcast
            miner_repo_version: Optional miner repository version for the websocket dict
        """
        # Skip broadcast for development hotkey
        if position.miner_hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
            return

        try:
            self._call_with_reconnect("broadcast_position_update_rpc", position, miner_repo_version)
        except Exception as e:
            logger.debug(f"WebSocketNotifierClient: Broadcast failed (after reconnect attempt): {e}")

    def broadcast_subaccount_dashboard(self, synthetic_hotkey: str) -> None:
        """
        Broadcast subaccount dashboard to subscribed WebSocket clients.

        Args:
            synthetic_hotkey: The synthetic hotkey to broadcast dashboard for
            data: The dashboard data to broadcast

        Returns:
            bool: True if broadcast was successful or skipped, False on error
        """
        try:
            self._call_with_reconnect("broadcast_subaccount_dashboard_rpc", synthetic_hotkey)
        except Exception as e:
            logger.debug(f"WebSocketNotifierClient: Dashboard broadcast failed (after reconnect attempt): {e}")

    def notify_new_subaccount(self, entity_hotkey: str, synthetic_hotkey: str) -> bool:
        """
        Notify the WebSocket server of a newly created subaccount so connected
        entity clients can be auto-subscribed.

        Args:
            entity_hotkey: The entity hotkey that owns the new subaccount
            synthetic_hotkey: The synthetic hotkey of the new subaccount

        Returns:
            bool: True if notification was sent successfully
        """
        try:
            return self._call_with_reconnect("notify_new_subaccount_rpc", entity_hotkey, synthetic_hotkey)
        except Exception as e:
            logger.debug(f"WebSocketNotifierClient: New subaccount notification failed (after reconnect attempt): {e}")
            return False

    def health_check(self) -> Optional[dict]:
        """
        Health check endpoint for monitoring.

        Returns:
            dict: Health status with queue stats, or None if server unavailable
        """
        try:
            return self._call_with_reconnect("health_check_rpc")
        except Exception as e:
            logger.debug(f"WebSocketNotifierClient: Health check failed (after reconnect attempt): {e}")
            return None

    def get_queued_messages(self, max_messages: int = None) -> list:
        """
        Retrieve queued messages from the server.

        Args:
            max_messages: Maximum number of messages to retrieve (None = all)

        Returns:
            list: List of queued message dicts
        """
        try:
            return self._call_with_reconnect("get_queued_messages_rpc", max_messages)
        except Exception as e:
            logger.debug(f"WebSocketNotifierClient: Get queued messages failed (after reconnect attempt): {e}")
            return []

    def clear_queue(self) -> int:
        """
        Clear all queued messages.

        Returns:
            int: Number of messages cleared, or 0 if server unavailable
        """
        try:
            return self._call_with_reconnect("clear_queue_rpc")
        except Exception as e:
            logger.debug(f"WebSocketNotifierClient: Clear queue failed (after reconnect attempt): {e}")
            return 0


# Backward compatibility alias
WebSocketNotifier = WebSocketNotifierClient
