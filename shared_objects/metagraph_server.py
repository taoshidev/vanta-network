# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Metagraph RPC Server - Manages metagraph state with local data and cached set for fast lookups.

This server runs in its own process and exposes metagraph data via RPC.
Much faster than direct IPC property access for has_hotkey() checks.
"""
import bittensor as bt
from typing import Set


class MetagraphServer:
    """
    Server-side metagraph with local data and cached hotkeys_set for O(1) lookups.

    All public methods ending in _rpc are exposed via RPC to the client.
    Internal state is kept local to this process for performance.
    """

    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def __init__(self, running_unit_tests=False):
        """
        Initialize metagraph server.

        Args:
            running_unit_tests: Whether running in test mode
        """
        self.running_unit_tests = running_unit_tests

        # Local data (no IPC overhead)
        self._neurons = []
        self._hotkeys = []
        self._uids = []
        self._axons = []
        self._block_at_registration = []
        self._emission = []
        self._tao_reserve_rao = 0.0
        self._alpha_reserve_rao = 0.0
        self._tao_to_usd_rate = 0.0

        # Cached hotkeys_set for O(1) has_hotkey() lookups
        self._hotkeys_set: Set[str] = set()

        bt.logging.info(
            f"MetagraphServer initialized - '{self.DEVELOPMENT_HOTKEY}' hotkey "
            f"will be available for development orders"
        )

    # ==================== RPC Methods (exposed to client) ====================

    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring"""
        return {
            "status": "ok",
            "num_hotkeys": len(self._hotkeys),
            "num_neurons": len(self._neurons)
        }

    def has_hotkey_rpc(self, hotkey: str) -> bool:
        """
        Fast O(1) hotkey existence check using cached set.

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey exists or is DEVELOPMENT, False otherwise
        """
        if hotkey == self.DEVELOPMENT_HOTKEY:
            return True
        return hotkey in self._hotkeys_set

    def get_hotkeys_rpc(self) -> list:
        """Get list of all hotkeys"""
        return list(self._hotkeys)

    def set_hotkeys_rpc(self, hotkeys: list) -> None:
        """
        Set hotkeys and update cached set.

        Args:
            hotkeys: List of hotkeys to set
        """
        self._hotkeys = list(hotkeys)
        # Update cached set for O(1) lookups
        self._hotkeys_set = set(hotkeys)

    def get_neurons_rpc(self) -> list:
        """Get list of neurons"""
        return list(self._neurons)

    def set_neurons_rpc(self, neurons: list) -> None:
        """Set neurons list"""
        self._neurons = list(neurons)

    def get_uids_rpc(self) -> list:
        """Get list of UIDs"""
        return list(self._uids)

    def set_uids_rpc(self, uids: list) -> None:
        """Set UIDs list"""
        self._uids = list(uids)

    def get_axons_rpc(self) -> list:
        """Get list of axons"""
        return list(self._axons)

    def set_axons_rpc(self, axons: list) -> None:
        """Set axons list"""
        self._axons = list(axons)

    def get_block_at_registration_rpc(self) -> list:
        """Get block at registration list"""
        return list(self._block_at_registration)

    def set_block_at_registration_rpc(self, blocks: list) -> None:
        """Set block at registration list"""
        self._block_at_registration = list(blocks)

    def get_emission_rpc(self) -> list:
        """Get emission list"""
        return list(self._emission)

    def set_emission_rpc(self, emission: list) -> None:
        """Set emission list"""
        self._emission = list(emission)

    def get_tao_reserve_rao_rpc(self) -> float:
        """Get TAO reserve in RAO"""
        return self._tao_reserve_rao

    def set_tao_reserve_rao_rpc(self, value: float) -> None:
        """Set TAO reserve in RAO"""
        self._tao_reserve_rao = float(value)

    def get_alpha_reserve_rao_rpc(self) -> float:
        """Get ALPHA reserve in RAO"""
        return self._alpha_reserve_rao

    def set_alpha_reserve_rao_rpc(self, value: float) -> None:
        """Set ALPHA reserve in RAO"""
        self._alpha_reserve_rao = float(value)

    def get_tao_to_usd_rate_rpc(self) -> float:
        """Get TAO to USD conversion rate"""
        return self._tao_to_usd_rate

    def set_tao_to_usd_rate_rpc(self, value: float) -> None:
        """Set TAO to USD conversion rate"""
        self._tao_to_usd_rate = float(value)


def start_metagraph_server(running_unit_tests, address, authkey, server_ready):
    """Entry point for server process"""
    from multiprocessing.managers import BaseManager

    server_instance = MetagraphServer(running_unit_tests=running_unit_tests)

    # Register server with manager
    class MetagraphRPC(BaseManager):
        pass

    MetagraphRPC.register('MetagraphServer', callable=lambda: server_instance)

    manager = MetagraphRPC(address=address, authkey=authkey)
    rpc_server = manager.get_server()

    bt.logging.success(f"MetagraphServer ready on {address}")

    if server_ready:
        server_ready.set()

    # Start serving (blocks forever)
    rpc_server.serve_forever()
