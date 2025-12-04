# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Metagraph RPC Server and Client - Manages metagraph state with local data and cached set for fast lookups.

Architecture:
- MetagraphServer: Inherits from RPCServerBase, manages metagraph data with O(1) has_hotkey() lookups
- MetagraphClient: Inherits from RPCClientBase, lightweight client for consumers

Usage:
    # In validator.py - create server
    from shared_objects.metagraph_server import MetagraphServer
    metagraph_server = MetagraphServer(
        slack_notifier=slack_notifier,
        start_server=True
    )

    # In consumers - create own client (forward compatibility pattern)
    from shared_objects.metagraph_server import MetagraphClient
    metagraph_client = MetagraphClient()  # Connects to server via RPC

Thread-safe: All RPC methods are atomic (lock-free via atomic tuple assignment).
"""
import bittensor as bt
from typing import Set, List

from shared_objects.rpc.rpc_server_base import RPCServerBase
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class MetagraphServer(RPCServerBase):
    """
    Server-side metagraph with local data and cached hotkeys_set for O(1) lookups.

    All public methods ending in _rpc are exposed via RPC to the client.
    Internal state is kept local to this process for performance.

    Thread-safe: All data access uses atomic tuple assignment (lock-free).
    BaseManager RPC server is multithreaded, so we need atomic operations.

    Note: This server has NO daemon work - it just stores data that MetagraphUpdater pushes to it.
    """
    service_name = ValiConfig.RPC_METAGRAPH_SERVICE_NAME
    service_port = ValiConfig.RPC_METAGRAPH_PORT

    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def __init__(
        self,
        slack_notifier=None,
        start_server: bool = None,  # None = auto (True for validator, False for miner)
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,  # None = auto (RPC for validator, LOCAL for miner)
        is_miner: bool = False,
        running_unit_tests: bool = False
    ):
        """
        Initialize metagraph server.

        Uses atomic tuple assignment for updates instead of locks.
        All updates happen via single tuple unpacking (atomic in Python).
        Reads are lock-free for maximum performance.

        Args:
            slack_notifier: Optional slack notifier for error reporting
            start_server: Whether to start RPC server immediately (default: True for validator, False for miner)
            connection_mode: RPCConnectionMode.LOCAL for tests/miner, RPCConnectionMode.RPC for validator
            is_miner: Whether this is a miner (simplified mode, no RPC server)
        """
        # Auto-configure based on miner/validator mode
        self.is_miner = is_miner
        self.is_validator = not is_miner
        self.running_unit_tests = running_unit_tests

        # Default start_server: False for miner, True for validator
        if start_server is None:
            start_server = not is_miner
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

        # Initialize RPCServerBase (NO daemon for MetagraphServer - it's just a data store)
        super().__init__(
            service_name=ValiConfig.RPC_METAGRAPH_SERVICE_NAME,
            port=ValiConfig.RPC_METAGRAPH_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # No daemon work - data is pushed by MetagraphUpdater
            connection_mode=connection_mode
        )

        bt.logging.info(
            f"MetagraphServer initialized on port {ValiConfig.RPC_METAGRAPH_PORT} - "
            f"'{self.DEVELOPMENT_HOTKEY}' hotkey will be available for development orders"
        )

    # ==================== RPCServerBase Abstract Methods (no daemon work) ====================

    def run_daemon_iteration(self) -> None:
        """
        No-op: MetagraphServer has no daemon work.

        Data is pushed to this server by MetagraphUpdater via update_metagraph_rpc().
        """
        pass

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details (lock-free read)."""
        return {
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

    # ==================== Convenience Methods (direct access, same API as client) ====================

    def has_hotkey(self, hotkey: str) -> bool:
        """Fast O(1) hotkey existence check (direct access, no RPC)."""
        return self.has_hotkey_rpc(hotkey)

    def get_hotkeys(self) -> list:
        """Get list of all hotkeys (direct access, no RPC)."""
        return self.get_hotkeys_rpc()

    def get_neurons(self) -> list:
        """Get list of neurons (direct access, no RPC)."""
        return self.get_neurons_rpc()

    def get_uids(self) -> list:
        """Get list of UIDs (direct access, no RPC)."""
        return self.get_uids_rpc()

    def get_axons(self) -> list:
        """Get list of axons (direct access, no RPC)."""
        return self.get_axons_rpc()

    def get_block_at_registration(self) -> list:
        """Get block at registration list (direct access, no RPC)."""
        return self.get_block_at_registration_rpc()

    def get_emission(self) -> list:
        """Get emission list (direct access, no RPC)."""
        return self.get_emission_rpc()

    # ==================== Property Accessors (for backward compatibility with attribute access) ====================

    @property
    def hotkeys(self) -> list:
        """Property accessor for hotkeys list (backward compatibility with metagraph.hotkeys)."""
        return self._hotkeys

    @property
    def neurons(self) -> list:
        """Property accessor for neurons list."""
        return self._neurons

    @property
    def uids(self) -> list:
        """Property accessor for UIDs list."""
        return self._uids

    @property
    def axons(self) -> list:
        """Property accessor for axons list."""
        return self._axons

    @property
    def block_at_registration(self) -> list:
        """Property accessor for block_at_registration list."""
        return self._block_at_registration

    @property
    def emission(self) -> list:
        """Property accessor for emission list."""
        return self._emission

    @property
    def tao_reserve_rao(self) -> float:
        """Property accessor for TAO reserve in RAO."""
        return self._tao_reserve_rao

    @property
    def alpha_reserve_rao(self) -> float:
        """Property accessor for ALPHA reserve in RAO."""
        return self._alpha_reserve_rao

    @property
    def tao_to_usd_rate(self) -> float:
        """Property accessor for TAO to USD conversion rate."""
        return self._tao_to_usd_rate

    # ==================== Test Convenience Methods ====================

    def set_hotkeys(self, hotkeys: List[str]) -> None:
        """
        Convenience method for tests: Set hotkeys with auto-generated default values.

        Automatically generates:
        - uids: Sequential integers [0, 1, 2, ...]
        - neurons: Empty list (most tests don't need actual neuron objects)
        - block_at_registration: All set to 1000
        - axons: Empty list
        - emission: All set to 1.0
        - tao_reserve_rao: 1_000_000_000_000 (1000 TAO)
        - alpha_reserve_rao: 1_000_000_000_000 (1000 ALPHA)
        - tao_to_usd_rate: 100.0

        Args:
            hotkeys: List of hotkey strings

        Example:
            metagraph_server.set_hotkeys(["miner1", "miner2", "miner3"])
        """
        n = len(hotkeys)
        self.update_metagraph_rpc(
            hotkeys=hotkeys,
            uids=list(range(n)),
            neurons=[None] * n,
            block_at_registration=[1000] * n,
            axons=[None] * n,
            emission=[1.0] * n,
            tao_reserve_rao=1_000_000_000_000,
            alpha_reserve_rao=1_000_000_000_000,
            tao_to_usd_rate=100.0
        )

    def set_block_at_registration(self, hotkey: str, block: int) -> None:
        """
        Convenience method for tests: Set block_at_registration for a specific hotkey.

        Args:
            hotkey: The hotkey to update
            block: The block number to set

        Raises:
            ValueError: If hotkey is not in metagraph
            AssertionError: If not running in unit test mode

        Example:
            metagraph_server.set_block_at_registration("miner1", 4916373)
        """
        assert self.running_unit_tests, "set_block_at_registration() is only allowed during unit tests"

        # Get current data (direct access, no RPC)
        current_hotkeys = self._hotkeys
        current_blocks = self._block_at_registration

        # Find hotkey index
        if hotkey not in current_hotkeys:
            raise ValueError(f"Hotkey '{hotkey}' not found in metagraph")

        hotkey_index = current_hotkeys.index(hotkey)

        # Update the block_at_registration list
        new_blocks = list(current_blocks)
        new_blocks[hotkey_index] = block

        # Update via RPC method
        self.update_metagraph_rpc(block_at_registration=new_blocks)


class MetagraphClient(RPCClientBase):
    """
    RPC Client for Metagraph - provides fast access to metagraph data via RPC.

    This client connects to MetagraphServer running in the validator.
    The server maintains a cached hotkeys_set for O(1) has_hotkey() lookups.

    Forward compatibility: Consumers create their own MetagraphClient instance
    instead of being passed a metagraph instance.
    """

    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def __init__(
        self,
        connect_immediately: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False
    ):
        """
        Initialize metagraph RPC client.

        Args:
            connect_immediately: Whether to connect immediately or defer
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_METAGRAPH_SERVICE_NAME,
            port=ValiConfig.RPC_METAGRAPH_PORT,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Client Methods (proxy to RPC) ====================

    def has_hotkey(self, hotkey: str) -> bool:
        """
        Fast O(1) hotkey existence check via RPC.
        Server uses cached set for instant lookups.

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey exists or is DEVELOPMENT, False otherwise
        """
        return self._server.has_hotkey_rpc(hotkey)

    def get_hotkeys(self) -> list:
        """Get list of all hotkeys"""
        return self._server.get_hotkeys_rpc()

    def get_neurons(self) -> list:
        """Get list of neurons"""
        return self._server.get_neurons_rpc()

    def get_uids(self) -> list:
        """Get list of UIDs"""
        return self._server.get_uids_rpc()

    def get_axons(self) -> list:
        """Get list of axons"""
        return self._server.get_axons_rpc()

    def get_block_at_registration(self) -> list:
        """Get block at registration list"""
        return self._server.get_block_at_registration_rpc()

    def get_emission(self) -> list:
        """Get emission list"""
        return self._server.get_emission_rpc()

    def get_tao_reserve_rao(self) -> float:
        """Get TAO reserve in RAO"""
        return self._server.get_tao_reserve_rao_rpc()

    def get_alpha_reserve_rao(self) -> float:
        """Get ALPHA reserve in RAO"""
        return self._server.get_alpha_reserve_rao_rpc()

    def get_tao_to_usd_rate(self) -> float:
        """Get TAO to USD conversion rate"""
        return self._server.get_tao_to_usd_rate_rpc()

    def set_tao_reserve_rao(self, tao_reserve_rao: float) -> None:
        """
        Set TAO reserve in RAO.

        Args:
            tao_reserve_rao: TAO reserve amount in RAO (1 RAO = 10^-9 TAO)
        """
        self._server.update_metagraph_rpc(tao_reserve_rao=tao_reserve_rao)

    def set_alpha_reserve_rao(self, alpha_reserve_rao: float) -> None:
        """
        Set ALPHA reserve in RAO.

        Args:
            alpha_reserve_rao: ALPHA reserve amount in RAO (1 RAO = 10^-9 ALPHA)
        """
        self._server.update_metagraph_rpc(alpha_reserve_rao=alpha_reserve_rao)

    def update_metagraph(self, neurons: list = None, uids: list = None, hotkeys: list = None,
                        block_at_registration: list = None, axons: list = None,
                        emission: list = None, tao_reserve_rao: float = None,
                        alpha_reserve_rao: float = None, tao_to_usd_rate: float = None) -> None:
        """
        Atomically update multiple metagraph fields in a single RPC call.
        Much faster than individual setter calls (1 RPC call instead of N).

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

        Example:
            # Update all metagraph fields in one atomic RPC call
            metagraph.update_metagraph(
                neurons=list(metagraph_clone.neurons),
                uids=list(metagraph_clone.uids),
                hotkeys=list(metagraph_clone.hotkeys),
                block_at_registration=list(metagraph_clone.block_at_registration),
                emission=list(metagraph_clone.emission)
            )
        """
        self._server.update_metagraph_rpc(
            neurons=neurons,
            uids=uids,
            hotkeys=hotkeys,
            block_at_registration=block_at_registration,
            axons=axons,
            emission=emission,
            tao_reserve_rao=tao_reserve_rao,
            alpha_reserve_rao=alpha_reserve_rao,
            tao_to_usd_rate=tao_to_usd_rate
        )

    def is_development_hotkey(self, hotkey: str) -> bool:
        """Check if hotkey is the synthetic DEVELOPMENT hotkey"""
        return hotkey == self.DEVELOPMENT_HOTKEY

    # ==================== Property Accessors (for backward compatibility with attribute access) ====================

    @property
    def hotkeys(self) -> list:
        """Property accessor for hotkeys list (backward compatibility with metagraph.hotkeys)."""
        return self.get_hotkeys()

    @property
    def neurons(self) -> list:
        """Property accessor for neurons list."""
        return self.get_neurons()

    @property
    def uids(self) -> list:
        """Property accessor for UIDs list."""
        return self.get_uids()

    @property
    def axons(self) -> list:
        """Property accessor for axons list."""
        return self.get_axons()

    @property
    def block_at_registration(self) -> list:
        """Property accessor for block_at_registration list."""
        return self.get_block_at_registration()

    @property
    def emission(self) -> list:
        """Property accessor for emission list."""
        return self.get_emission()

    @property
    def tao_reserve_rao(self) -> float:
        """Property accessor for TAO reserve in RAO."""
        return self.get_tao_reserve_rao()

    @property
    def alpha_reserve_rao(self) -> float:
        """Property accessor for ALPHA reserve in RAO."""
        return self.get_alpha_reserve_rao()

    @property
    def tao_to_usd_rate(self) -> float:
        """Property accessor for TAO to USD conversion rate."""
        return self.get_tao_to_usd_rate()

    # ==================== Test Convenience Methods ====================

    def set_hotkeys(self, hotkeys: List[str]) -> None:
        """
        Convenience method for tests: Set hotkeys with auto-generated default values.

        Automatically generates:
        - uids: Sequential integers [0, 1, 2, ...]
        - neurons: Empty list (most tests don't need actual neuron objects)
        - block_at_registration: All set to 1000
        - axons: Empty list
        - emission: All set to 1.0
        - tao_reserve_rao: 1_000_000_000_000 (1000 TAO)
        - alpha_reserve_rao: 1_000_000_000_000 (1000 ALPHA)
        - tao_to_usd_rate: 100.0

        Args:
            hotkeys: List of hotkey strings

        Example:
            metagraph_client.set_hotkeys(["miner1", "miner2", "miner3"])
        """
        n = len(hotkeys)
        self._server.update_metagraph_rpc(
            hotkeys=hotkeys,
            uids=list(range(n)),
            neurons=[None] * n,  # Placeholder - most tests don't need actual neurons
            block_at_registration=[1000] * n,
            axons=[None] * n,
            emission=[1.0] * n,
            tao_reserve_rao=1_000_000_000_000,  # 1000 TAO in RAO
            alpha_reserve_rao=1_000_000_000_000,  # 1000 ALPHA in RAO
            tao_to_usd_rate=100.0
        )

    def set_block_at_registration(self, hotkey: str, block: int) -> None:
        """
        Convenience method for tests: Set block_at_registration for a specific hotkey.

        Args:
            hotkey: The hotkey to update
            block: The block number to set

        Raises:
            ValueError: If hotkey is not in metagraph
            AssertionError: If not running in unit test mode

        Example:
            metagraph_client.set_block_at_registration("miner1", 4916373)
        """
        assert self.running_unit_tests, "set_block_at_registration() is only allowed during unit tests"

        # Get current data
        current_hotkeys = self.get_hotkeys()
        current_blocks = self.get_block_at_registration()

        # Find hotkey index
        if hotkey not in current_hotkeys:
            raise ValueError(f"Hotkey '{hotkey}' not found in metagraph")

        hotkey_index = current_hotkeys.index(hotkey)

        # Update the block_at_registration list
        new_blocks = list(current_blocks)
        new_blocks[hotkey_index] = block

        # Update via RPC
        self._server.update_metagraph_rpc(block_at_registration=new_blocks)


# Backward compatibility alias
MetagraphManager = MetagraphClient
