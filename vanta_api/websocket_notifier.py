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
    success = client.broadcast_position_update(position)
"""
from typing import Optional

import bittensor as bt

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


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

    # ==================== Client Methods ====================

    def broadcast_position_update(self, position: Position, miner_repo_version: str = None) -> bool:
        """
        Broadcast a position update to all subscribed WebSocket clients.

        Args:
            position: Position object to broadcast
            miner_repo_version: Optional miner repository version for the websocket dict

        Returns:
            bool: True if message was queued successfully, False otherwise
        """
        # Skip broadcast for development hotkey
        if position.miner_hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
            return True

        try:
            return self._server.broadcast_position_update_rpc(position, miner_repo_version)
        except Exception as e:
            bt.logging.debug(f"WebSocketNotifierClient: Broadcast failed: {e}")
            return False

    def health_check(self) -> Optional[dict]:
        """
        Health check endpoint for monitoring.

        Returns:
            dict: Health status with queue stats, or None if server unavailable
        """
        try:
            return self._server.health_check_rpc()
        except Exception as e:
            bt.logging.debug(f"WebSocketNotifierClient: Health check failed: {e}")
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
            return self._server.get_queued_messages_rpc(max_messages)
        except Exception as e:
            bt.logging.debug(f"WebSocketNotifierClient: Get queued messages failed: {e}")
            return []

    def clear_queue(self) -> int:
        """
        Clear all queued messages.

        Returns:
            int: Number of messages cleared, or 0 if server unavailable
        """
        try:
            return self._server.clear_queue_rpc()
        except Exception as e:
            bt.logging.debug(f"WebSocketNotifierClient: Clear queue failed: {e}")
            return 0


# Backward compatibility alias
WebSocketNotifier = WebSocketNotifierClient
