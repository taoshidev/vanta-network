from typing import List

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


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
