# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Metagraph RPC Client - Provides fast metagraph access via RPC.

This client connects to MetagraphServer running in a separate process.
Much faster than IPC for has_hotkey() checks (O(1) with server-side cached set).
"""
from multiprocessing import Process
from shared_objects.rpc_service_base import RPCServiceBase

import bittensor as bt


class MetagraphManager(RPCServiceBase):
    """
    RPC Client for Metagraph - provides fast access to metagraph data via RPC.

    This client connects to MetagraphServer running in a separate process.
    The server maintains a cached hotkeys_set for O(1) has_hotkey() lookups.
    """

    DEVELOPMENT_HOTKEY = "DEVELOPMENT"

    def __init__(self, running_unit_tests=False, slack_notifier=None):
        """
        Initialize metagraph RPC client.

        Args:
            running_unit_tests: Whether running in test mode
            slack_notifier: Optional slack notifier for error reporting
        """
        # Initialize RPCServiceBase
        RPCServiceBase.__init__(
            self,
            service_name="MetagraphServer",
            port=50005,  # Unique port for MetagraphServer
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )

        # Store dependencies
        self.running_unit_tests = running_unit_tests

        # Start the RPC service
        self._initialize_service()

    def _create_direct_server(self):
        """Create direct in-memory instance for tests"""
        from shared_objects.metagraph_server import MetagraphServer

        return MetagraphServer(running_unit_tests=self.running_unit_tests)

    def _start_server_process(self, address, authkey, server_ready):
        """Start RPC server in separate process"""
        from shared_objects.metagraph_server import start_metagraph_server

        process = Process(
            target=start_metagraph_server,
            args=(
                self.running_unit_tests,
                address,
                authkey,
                server_ready
            ),
            daemon=True
        )
        process.start()
        return process

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
        return self._server_proxy.has_hotkey_rpc(hotkey)

    def get_hotkeys(self) -> list:
        """Get list of all hotkeys"""
        return self._server_proxy.get_hotkeys_rpc()

    def set_hotkeys(self, hotkeys: list) -> None:
        """Set hotkeys and update server's cached set"""
        self._server_proxy.set_hotkeys_rpc(hotkeys)

    def get_neurons(self) -> list:
        """Get list of neurons"""
        return self._server_proxy.get_neurons_rpc()

    def set_neurons(self, neurons: list) -> None:
        """Set neurons list"""
        self._server_proxy.set_neurons_rpc(neurons)

    def get_uids(self) -> list:
        """Get list of UIDs"""
        return self._server_proxy.get_uids_rpc()

    def set_uids(self, uids: list) -> None:
        """Set UIDs list"""
        self._server_proxy.set_uids_rpc(uids)

    def get_axons(self) -> list:
        """Get list of axons"""
        return self._server_proxy.get_axons_rpc()

    def set_axons(self, axons: list) -> None:
        """Set axons list"""
        self._server_proxy.set_axons_rpc(axons)

    def get_block_at_registration(self) -> list:
        """Get block at registration list"""
        return self._server_proxy.get_block_at_registration_rpc()

    def set_block_at_registration(self, blocks: list) -> None:
        """Set block at registration list"""
        self._server_proxy.set_block_at_registration_rpc(blocks)

    def get_emission(self) -> list:
        """Get emission list"""
        return self._server_proxy.get_emission_rpc()

    def set_emission(self, emission: list) -> None:
        """Set emission list"""
        self._server_proxy.set_emission_rpc(emission)

    def get_tao_reserve_rao(self) -> float:
        """Get TAO reserve in RAO"""
        return self._server_proxy.get_tao_reserve_rao_rpc()

    def set_tao_reserve_rao(self, value: float) -> None:
        """Set TAO reserve in RAO"""
        self._server_proxy.set_tao_reserve_rao_rpc(value)

    def get_alpha_reserve_rao(self) -> float:
        """Get ALPHA reserve in RAO"""
        return self._server_proxy.get_alpha_reserve_rao_rpc()

    def set_alpha_reserve_rao(self, value: float) -> None:
        """Set ALPHA reserve in RAO"""
        self._server_proxy.set_alpha_reserve_rao_rpc(value)

    def get_tao_to_usd_rate(self) -> float:
        """Get TAO to USD conversion rate"""
        return self._server_proxy.get_tao_to_usd_rate_rpc()

    def set_tao_to_usd_rate(self, value: float) -> None:
        """Set TAO to USD conversion rate"""
        self._server_proxy.set_tao_to_usd_rate_rpc(value)

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
        self._server_proxy.update_metagraph_rpc(
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

    # ==================== Backwards Compatibility Properties ====================

    @property
    def hotkeys(self) -> list:
        """Direct access to hotkeys (for compatibility)"""
        return self.get_hotkeys()

    @hotkeys.setter
    def hotkeys(self, value: list):
        """Allow setting hotkeys (for compatibility)"""
        self.set_hotkeys(value)

    @property
    def neurons(self) -> list:
        """Direct access to neurons (for compatibility)"""
        return self.get_neurons()

    @neurons.setter
    def neurons(self, value: list):
        """Allow setting neurons (for compatibility)"""
        self.set_neurons(value)

    @property
    def uids(self) -> list:
        """Direct access to uids (for compatibility)"""
        return self.get_uids()

    @uids.setter
    def uids(self, value: list):
        """Allow setting uids (for compatibility)"""
        self.set_uids(value)

    @property
    def axons(self) -> list:
        """Direct access to axons (for compatibility)"""
        return self.get_axons()

    @axons.setter
    def axons(self, value: list):
        """Allow setting axons (for compatibility)"""
        self.set_axons(value)

    @property
    def block_at_registration(self) -> list:
        """Direct access to block_at_registration (for compatibility)"""
        return self.get_block_at_registration()

    @block_at_registration.setter
    def block_at_registration(self, value: list):
        """Allow setting block_at_registration (for compatibility)"""
        self.set_block_at_registration(value)

    @property
    def emission(self) -> list:
        """Direct access to emission (for compatibility)"""
        return self.get_emission()

    @emission.setter
    def emission(self, value: list):
        """Allow setting emission (for compatibility)"""
        self.set_emission(value)

    @property
    def tao_reserve_rao(self) -> float:
        """Direct access to TAO reserve (for compatibility)"""
        return self.get_tao_reserve_rao()

    @tao_reserve_rao.setter
    def tao_reserve_rao(self, value: float):
        """Allow setting TAO reserve (for compatibility)"""
        self.set_tao_reserve_rao(value)

    @property
    def alpha_reserve_rao(self) -> float:
        """Direct access to ALPHA reserve (for compatibility)"""
        return self.get_alpha_reserve_rao()

    @alpha_reserve_rao.setter
    def alpha_reserve_rao(self, value: float):
        """Allow setting ALPHA reserve (for compatibility)"""
        self.set_alpha_reserve_rao(value)

    @property
    def tao_to_usd_rate(self) -> float:
        """Direct access to TAO/USD rate (for compatibility)"""
        return self.get_tao_to_usd_rate()

    @tao_to_usd_rate.setter
    def tao_to_usd_rate(self, value: float):
        """Allow setting TAO/USD rate (for compatibility)"""
        self.set_tao_to_usd_rate(value)

    def is_development_hotkey(self, hotkey: str) -> bool:
        """Check if hotkey is the synthetic DEVELOPMENT hotkey"""
        return hotkey == self.DEVELOPMENT_HOTKEY
