"""
Asset Selection Manager - RPC Client for managing asset class selections.

This client connects to AssetSelectionManagerServer running in a separate process.
Much faster than IPC managerized dicts.

Miners can select an asset class (forex, crypto, etc.) only once.
Once selected, the miner cannot trade any trade pair from a different asset class.
"""
import bittensor as bt
from typing import Dict, Optional
from multiprocessing import Process

import template.protocol
from shared_objects.rpc_service_base import RPCServiceBase
from vali_objects.vali_config import TradePairCategory, ValiConfig


ASSET_CLASS_SELECTION_TIME_MS = 1758326340000


class AssetSelectionManager(RPCServiceBase):
    """
    Lightweight RPC client for accessing asset selection data.

    The actual asset selection management happens in AssetSelectionManagerServer.
    This client provides access via RPC.

    Inherits from RPCServiceBase for common RPC infrastructure (connection management,
    process lifecycle, stale server cleanup, health checks).
    """

    def __init__(self, config=None, metagraph=None, running_unit_tests=False, slack_notifier=None):
        """
        Initialize client and start server process.

        Args:
            config: Validator config (for netuid, wallet)
            metagraph: Metagraph instance for broadcasting
            running_unit_tests: Whether running in unit test mode
            slack_notifier: Slack notifier for health check alerts
        """
        # Store dependencies for server creation
        self._config = config
        self._metagraph = metagraph

        # Initialize RPCServiceBase (handles connection, process lifecycle, etc.)
        super().__init__(
            service_name=ValiConfig.RPC_ASSETSELECTION_SERVICE_NAME,
            port=ValiConfig.RPC_ASSETSELECTION_PORT,
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )

        # Cache static file path locally (no need for RPC call)
        from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
        self._asset_selections_file = ValiBkpUtils.get_asset_selections_file_location(running_unit_tests=running_unit_tests)

        # Initialize the service (RPC mode or direct mode for tests)
        self._initialize_service()

    # ============================================================================
    # RPCServiceBase IMPLEMENTATION (required abstract methods)
    # ============================================================================

    def _create_direct_server(self):
        """
        Create a direct in-memory server instance for unit tests.

        Returns:
            AssetSelectionManagerServer instance (not proxied, direct Python object)
        """
        from vali_objects.utils.asset_selection_manager_server import AssetSelectionManagerServer

        return AssetSelectionManagerServer(
            config=self._config,
            metagraph=self._metagraph,
            running_unit_tests=self.running_unit_tests
        )

    def _start_server_process(self, address, authkey, server_ready):
        """
        Start the RPC server in a separate process.

        Args:
            address: (host, port) tuple for RPC server
            authkey: Authentication key for RPC connection
            server_ready: Event to signal when server is ready

        Returns:
            Process object for the server process
        """
        from vali_objects.utils.asset_selection_manager_server import start_asset_selection_manager_server

        process = Process(
            target=start_asset_selection_manager_server,
            args=(
                address,
                authkey,
                self._config,
                self._metagraph,
                self.running_unit_tests,
                server_ready
            ),
            daemon=True
        )
        process.start()
        return process

    # ============================================================================
    # PUBLIC CLIENT METHODS (call server RPC methods)
    # ============================================================================

    def receive_asset_selection(self, synapse: template.protocol.AssetSelection) -> template.protocol.AssetSelection:
        """
        Receive miner's asset selection.

        Args:
            synapse: AssetSelection synapse from miner

        Returns:
            Updated synapse with success/error status
        """
        return self._server_proxy.receive_asset_selection_rpc(synapse)

    def sync_miner_asset_selection_data(self, asset_selection_data: Dict[str, str]) -> None:
        """
        Sync miner asset selection data from external source (backup/sync).

        Args:
            asset_selection_data: Dict mapping hotkey to asset class string
        """
        self._server_proxy.sync_miner_asset_selection_data_rpc(asset_selection_data)

    def is_valid_asset_class(self, asset_class: str) -> bool:
        """
        Validate if the provided asset class is valid.

        Args:
            asset_class: The asset class string to validate

        Returns:
            True if valid, False otherwise
        """
        return self._server_proxy.is_valid_asset_class_rpc(asset_class)

    def validate_order_asset_class(self, miner_hotkey: str, trade_pair_category: TradePairCategory, timestamp_ms: int = None) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check
            timestamp_ms: Optional timestamp in milliseconds

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        return self._server_proxy.validate_order_asset_class_rpc(miner_hotkey, trade_pair_category, timestamp_ms)

    def process_asset_selection_request(self, asset_selection: str, miner: str) -> Dict[str, str]:
        """
        Process an asset selection request from a miner.

        Args:
            asset_selection: The asset class the miner wants to select
            miner: The miner's hotkey

        Returns:
            Dict containing success status and message
        """
        return self._server_proxy.process_asset_selection_request_rpc(asset_selection, miner)

    def get_all_miner_selections(self) -> Dict[str, str]:
        """
        Get all miner asset selections as a dictionary.

        Returns:
            Dict[str, str]: Dictionary mapping miner hotkeys to their asset class selections (as strings).
                           Returns empty dict if no selections exist.
        """
        return self._server_proxy.get_all_miner_selections_rpc()

    def receive_asset_selection_update(self, asset_selection_data: dict) -> bool:
        """
        Process an incoming AssetSelection synapse and update miner asset selection.

        Args:
            asset_selection_data: Dictionary containing hotkey, asset selection

        Returns:
            bool: True if successful, False otherwise
        """
        return self._server_proxy.receive_asset_selection_update_rpc(asset_selection_data)

    def health_check(self) -> dict:
        """Check server health."""
        return self._server_proxy.health_check_rpc()

    # ============================================================================
    # BACKWARD COMPATIBILITY (for tests)
    # ============================================================================

    @property
    def asset_selections(self) -> Dict[str, TradePairCategory]:
        """
        Get asset selections dict from server (backward compatibility for tests).

        In test mode, provides direct access to server's dict for mutation.
        In production, uses RPC call which returns a copy.

        Returns:
            Dict mapping hotkey to TradePairCategory
        """
        if self.running_unit_tests:
            # Direct mode: return actual dict reference for test mutations
            return self._server_proxy.asset_selections
        else:
            # RPC mode: return copy via RPC call
            return self._server_proxy.get_asset_selections_rpc()

    @property
    def ASSET_SELECTIONS_FILE(self) -> str:
        """
        Get asset selections file path (backward compatibility for tests).

        Cached locally during initialization - no RPC call needed for static value.

        Returns:
            File path for asset selections persistence
        """
        return self._asset_selections_file

    def _to_dict(self) -> Dict:
        """Convert asset selections to disk format (backward compatibility)."""
        return self._server_proxy.to_dict_rpc()

    @staticmethod
    def _parse_asset_selections_dict(json_dict: Dict) -> Dict[str, TradePairCategory]:
        """Parse disk format back to in-memory format (backward compatibility)."""
        from vali_objects.utils.asset_selection_manager_server import AssetSelectionManagerServer
        return AssetSelectionManagerServer._parse_asset_selections_dict(json_dict)

    def _save_asset_selections_to_disk(self) -> None:
        """Save asset selections to disk (backward compatibility)."""
        return self._server_proxy.save_asset_selections_to_disk_rpc()
