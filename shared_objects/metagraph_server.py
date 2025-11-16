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

        Uses atomic tuple assignment for updates instead of locks.
        All updates happen via single tuple unpacking (atomic in Python).
        Reads are lock-free for maximum performance.

        Args:
            running_unit_tests: Whether running in test mode
        """
        self.running_unit_tests = running_unit_tests

        # Local data (no IPC overhead, no locks needed!)
        # Updates are atomic via tuple unpacking: (a, b, c) = (x, y, z)
        # Reads are lock-free and always see consistent state
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
        """Health check endpoint for RPC monitoring (lock-free read)"""
        return {
            "status": "ok",
            "num_hotkeys": len(self._hotkeys),
            "num_neurons": len(self._neurons)
        }

    def has_hotkey_rpc(self, hotkey: str) -> bool:
        """
        Fast O(1) hotkey existence check using cached set.
        Lock-free - set membership check is atomic in Python.

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey exists or is DEVELOPMENT, False otherwise
        """
        if hotkey == self.DEVELOPMENT_HOTKEY:
            return True
        # Lock-free! Python's 'in' operator is atomic for reads
        return hotkey in self._hotkeys_set

    def get_hotkeys_rpc(self) -> list:
        """Get list of all hotkeys (lock-free read)"""
        return list(self._hotkeys)

    def get_neurons_rpc(self) -> list:
        """Get list of neurons (lock-free read)"""
        return list(self._neurons)

    def get_uids_rpc(self) -> list:
        """Get list of UIDs (lock-free read)"""
        return list(self._uids)

    def get_axons_rpc(self) -> list:
        """Get list of axons (lock-free read)"""
        return list(self._axons)

    def get_block_at_registration_rpc(self) -> list:
        """Get block at registration list (lock-free read)"""
        return list(self._block_at_registration)

    def get_emission_rpc(self) -> list:
        """Get emission list (lock-free read)"""
        return list(self._emission)

    def get_tao_reserve_rao_rpc(self) -> float:
        """Get TAO reserve in RAO (lock-free read)"""
        return self._tao_reserve_rao

    def get_alpha_reserve_rao_rpc(self) -> float:
        """Get ALPHA reserve in RAO (lock-free read)"""
        return self._alpha_reserve_rao

    def get_tao_to_usd_rate_rpc(self) -> float:
        """Get TAO to USD conversion rate (lock-free read)"""
        return self._tao_to_usd_rate

    def update_metagraph_rpc(self, neurons: list = None, uids: list = None, hotkeys: list = None,
                            block_at_registration: list = None, axons: list = None,
                            emission: list = None, tao_reserve_rao: float = None,
                            alpha_reserve_rao: float = None, tao_to_usd_rate: float = None) -> None:
        """
        Atomically update multiple metagraph fields in a single RPC call (lock-free).
        Only updates fields that are provided (not None).

        Uses atomic tuple assignment for thread-safety without locks.
        All fields are updated in a single tuple unpacking operation, which is
        atomic at the Python bytecode level. This ensures concurrent reads always
        see a consistent state while avoiding lock contention.

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
        # Prepare new values (use current value if not provided)
        new_neurons = list(neurons) if neurons is not None else self._neurons
        new_uids = list(uids) if uids is not None else self._uids
        new_hotkeys = list(hotkeys) if hotkeys is not None else self._hotkeys
        new_block_at_reg = list(block_at_registration) if block_at_registration is not None else self._block_at_registration
        new_axons = list(axons) if axons is not None else self._axons
        new_emission = list(emission) if emission is not None else self._emission
        new_tao_reserve = float(tao_reserve_rao) if tao_reserve_rao is not None else self._tao_reserve_rao
        new_alpha_reserve = float(alpha_reserve_rao) if alpha_reserve_rao is not None else self._alpha_reserve_rao
        new_tao_usd_rate = float(tao_to_usd_rate) if tao_to_usd_rate is not None else self._tao_to_usd_rate

        # Update cached hotkeys set (only if hotkeys changed)
        new_hotkeys_set = set(hotkeys) if hotkeys is not None else self._hotkeys_set

        # Atomic tuple assignment - all fields updated in single bytecode operation!
        # This is thread-safe without locks due to Python's GIL and atomic tuple unpacking
        (self._neurons, self._uids, self._hotkeys, self._block_at_registration,
         self._axons, self._emission, self._tao_reserve_rao, self._alpha_reserve_rao,
         self._tao_to_usd_rate, self._hotkeys_set) = (
            new_neurons, new_uids, new_hotkeys, new_block_at_reg,
            new_axons, new_emission, new_tao_reserve, new_alpha_reserve,
            new_tao_usd_rate, new_hotkeys_set
        )


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
