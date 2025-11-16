# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Metagraph RPC Server - Manages metagraph state with local data and cached set for fast lookups.

This server runs in its own process and exposes metagraph data via RPC.
Much faster than direct IPC property access for has_hotkey() checks.

Thread-safe: All RPC methods are protected by a threading lock to ensure atomicity.
"""
import threading
import bittensor as bt
from typing import Set


class MetagraphServer:
    """
    Server-side metagraph with local data and cached hotkeys_set for O(1) lookups.

    All public methods ending in _rpc are exposed via RPC to the client.
    Internal state is kept local to this process for performance.

    Thread-safe: All data access is protected by self._lock to ensure atomicity.
    BaseManager RPC server is multithreaded, so we need to guard against concurrent access.
    """

    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def __init__(self, running_unit_tests=False):
        """
        Initialize metagraph server.

        Args:
            running_unit_tests: Whether running in test mode
        """
        self.running_unit_tests = running_unit_tests

        # Threading lock for atomic operations (server-side only)
        # BaseManager RPC server is multithreaded, so we need to protect shared state
        self._lock = threading.Lock()

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
        with self._lock:
            return {
                "status": "ok",
                "num_hotkeys": len(self._hotkeys),
                "num_neurons": len(self._neurons)
            }

    def has_hotkey_rpc(self, hotkey: str) -> bool:
        """
        Fast O(1) hotkey existence check using cached set.
        Thread-safe via lock.

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey exists or is DEVELOPMENT, False otherwise
        """
        if hotkey == self.DEVELOPMENT_HOTKEY:
            return True
        with self._lock:
            return hotkey in self._hotkeys_set

    def get_hotkeys_rpc(self) -> list:
        """Get list of all hotkeys (thread-safe)"""
        with self._lock:
            return list(self._hotkeys)

    def get_neurons_rpc(self) -> list:
        """Get list of neurons (thread-safe)"""
        with self._lock:
            return list(self._neurons)

    def get_uids_rpc(self) -> list:
        """Get list of UIDs (thread-safe)"""
        with self._lock:
            return list(self._uids)

    def get_axons_rpc(self) -> list:
        """Get list of axons (thread-safe)"""
        with self._lock:
            return list(self._axons)

    def get_block_at_registration_rpc(self) -> list:
        """Get block at registration list (thread-safe)"""
        with self._lock:
            return list(self._block_at_registration)

    def get_emission_rpc(self) -> list:
        """Get emission list (thread-safe)"""
        with self._lock:
            return list(self._emission)

    def get_tao_reserve_rao_rpc(self) -> float:
        """Get TAO reserve in RAO (thread-safe)"""
        with self._lock:
            return self._tao_reserve_rao

    def get_alpha_reserve_rao_rpc(self) -> float:
        """Get ALPHA reserve in RAO (thread-safe)"""
        with self._lock:
            return self._alpha_reserve_rao

    def get_tao_to_usd_rate_rpc(self) -> float:
        """Get TAO to USD conversion rate (thread-safe)"""
        with self._lock:
            return self._tao_to_usd_rate

    def update_metagraph_rpc(self, neurons: list = None, uids: list = None, hotkeys: list = None,
                            block_at_registration: list = None, axons: list = None,
                            emission: list = None, tao_reserve_rao: float = None,
                            alpha_reserve_rao: float = None, tao_to_usd_rate: float = None) -> None:
        """
        Atomically update multiple metagraph fields in a single RPC call (thread-safe).
        Only updates fields that are provided (not None).

        All updates happen within a single lock acquisition for true atomicity.
        This ensures that concurrent reads see a consistent state.

        Args:
            neurons: List of neurons (optional)
            uids: List of UIDs (optional)
            hotkeys: List of hotkeys (optional, will update cached set)
            block_at_registration: List of block numbers (optional)
            axons: List of axons (optional)
            emission: List of emission values (optional)
            tao_reserve_rao: TAO reserve in RAO (optional)
            alpha_reserve_rao: ALPHA reserve in RAO (optional)
            tao_to_usd_rate: TAO to USD conversion rate (optional)
        """
        with self._lock:
            if neurons is not None:
                self._neurons = list(neurons)
            if uids is not None:
                self._uids = list(uids)
            if hotkeys is not None:
                self._hotkeys = list(hotkeys)
                # Update cached set for O(1) lookups
                self._hotkeys_set = set(hotkeys)
            if block_at_registration is not None:
                self._block_at_registration = list(block_at_registration)
            if axons is not None:
                self._axons = list(axons)
            if emission is not None:
                self._emission = list(emission)
            if tao_reserve_rao is not None:
                self._tao_reserve_rao = float(tao_reserve_rao)
            if alpha_reserve_rao is not None:
                self._alpha_reserve_rao = float(alpha_reserve_rao)
            if tao_to_usd_rate is not None:
                self._tao_to_usd_rate = float(tao_to_usd_rate)


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
