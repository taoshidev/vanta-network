"""
HyperliquidTrackerClient - Lightweight RPC client for Hyperliquid trade tracking.

Connects to HyperliquidTrackerServer via RPC.
Can be created in ANY process - just needs the server to be running.

Usage:
    from vali_objects.tracking.hyperliquid_tracker_client import HyperliquidTrackerClient

    client = HyperliquidTrackerClient()
    client.register_address(miner_hotkey, "0xabc123...")
    registrations = client.get_registrations()
"""

from typing import Dict, Optional

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class HyperliquidTrackerClient(RPCClientBase):
    """
    Lightweight RPC client for HyperliquidTrackerServer.

    Can be created in ANY process. No server ownership.
    """

    def __init__(
        self,
        port: int = None,
        running_unit_tests: bool = False,
        connect_immediately: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
    ):
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_HYPERLIQUID_TRACKER_SERVICE_NAME,
            port=port or ValiConfig.RPC_HYPERLIQUID_TRACKER_PORT,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode,
        )

    # ==================== Registration Methods ====================

    def register_address(self, miner_hotkey: str, address: str) -> dict:
        """Register a Hyperliquid address for a miner."""
        return self._server.register_address_rpc(miner_hotkey, address)

    def unregister_address(self, miner_hotkey: str) -> dict:
        """Unregister a miner's Hyperliquid address."""
        return self._server.unregister_address_rpc(miner_hotkey)

    def get_registrations(self) -> Dict[str, dict]:
        """Get all registered miner -> address mappings."""
        return self._server.get_registrations_rpc()

    def get_registration(self, miner_hotkey: str) -> Optional[dict]:
        """Get registration for a specific miner."""
        return self._server.get_registration_rpc(miner_hotkey)

    # ==================== Utility Methods ====================

    def health_check(self) -> dict:
        """Check server health."""
        return self._server.health_check_rpc()

    def clear_registrations_for_test(self) -> None:
        """Clear all registrations (TEST ONLY)."""
        self._server.clear_registrations_for_test_rpc()
