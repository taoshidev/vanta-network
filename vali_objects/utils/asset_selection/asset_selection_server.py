# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
AssetSelectionServer - RPC server for asset class selection management.

This server runs in its own process and exposes asset selection management via RPC.
Clients connect using AssetSelectionClient.

This follows the same pattern as EliminationServer - the server wraps AssetSelectionManager
and exposes its methods via RPC.

Usage:
    # Validator spawns the server at startup
    from vali_objects.utils.asset_selection_server import AssetSelectionServer

    asset_selection_server = AssetSelectionServer(
        config=config,
        start_server=True,
        start_daemon=False
    )

    # Other processes connect via AssetSelectionClient
    from vali_objects.utils.asset_selection_client import AssetSelectionClient
    client = AssetSelectionClient()  # Uses ValiConfig.RPC_ASSETSELECTION_PORT
"""

import bittensor as bt
from typing import Dict

from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.utils.asset_selection.asset_selection_manager import AssetSelectionManager
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
import template.protocol


class AssetSelectionServer(RPCServerBase):
    """
    RPC server for asset selection management.

    Wraps AssetSelectionManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to AssetSelectionClient.

    This follows the same pattern as EliminationServer.
    """
    service_name = ValiConfig.RPC_ASSETSELECTION_SERVICE_NAME
    service_port = ValiConfig.RPC_ASSETSELECTION_PORT

    def __init__(
        self,
        config=None,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize AssetSelectionServer.

        Args:
            config: Validator config (for netuid, wallet)
            running_unit_tests: Whether running in test mode
            slack_notifier: Slack notifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately (typically False for asset selection)
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        # Create mock config if running tests and config not provided
        if running_unit_tests:
            from shared_objects.rpc.test_mock_factory import TestMockFactory
            config = TestMockFactory.create_mock_config_if_needed(config, netuid=116, network="test")

        self._config = config
        self.running_unit_tests = running_unit_tests

        # Create own MetagraphClient (forward compatibility - no parameter passing)
        from shared_objects.rpc.metagraph_client import MetagraphClient
        self._metagraph_client = MetagraphClient(connection_mode=connection_mode)

        # Determine testnet from config
        if not running_unit_tests and config is not None:
            self.is_testnet = config.netuid == 116
        else:
            self.is_testnet = False

        # Create the actual AssetSelectionManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        # Manager handles wallet initialization in background thread
        self._manager = AssetSelectionManager(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
            config=config
        )

        bt.logging.success("[ASSET_SERVER] AssetSelectionManager initialized")

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        super().__init__(
            service_name=ValiConfig.RPC_ASSETSELECTION_SERVICE_NAME,
            port=ValiConfig.RPC_ASSETSELECTION_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=60.0,  # Low frequency if daemon is used
            hang_timeout_s=120.0,
            connection_mode=connection_mode
        )

        bt.logging.success("[ASSET_SERVER] AssetSelectionServer initialized")

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work - currently no-op for asset selection.

        Asset selection doesn't need periodic processing (unlike eliminations).
        """
        pass  # Asset selection doesn't need periodic updates

    @property
    def metagraph(self):
        """Get metagraph client (forward compatibility - created internally)."""
        return self._metagraph_client

    @property
    def wallet(self):
        """Get wallet from manager (for backward compatibility with receive_asset_selection)."""
        return self._manager.wallet

    # ==================== RPC Methods (exposed to clients) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "total_selections": len(self._manager.asset_selections)
        }

    def get_asset_selections_rpc(self) -> Dict[str, TradePairCategory]:
        """
        Get the asset_selections dict (RPC method).

        Returns:
            Dict[str, TradePairCategory]: Dictionary mapping hotkey to TradePairCategory enum
        """
        return self._manager.get_asset_selections()

    def get_asset_selection_rpc(self, hotkey: str) -> TradePairCategory | None:
        return self._manager.get_asset_selection(hotkey)

    def get_all_miner_selections_rpc(self) -> Dict[str, str]:
        """
        Get all miner asset selections as a string dictionary (RPC method).

        Returns:
            Dict[str, str]: Dictionary mapping miner hotkeys to their asset class selections (as strings).
        """
        return self._manager.get_all_miner_selections()

    def validate_order_asset_class_rpc(
        self,
        miner_hotkey: str,
        trade_pair_category: TradePairCategory,
        timestamp_ms: int = None
    ) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class (RPC method).

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check
            timestamp_ms: Optional timestamp in milliseconds

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        return self._manager.validate_order_asset_class(miner_hotkey, trade_pair_category, timestamp_ms)

    def is_valid_asset_class_rpc(self, asset_class: str) -> bool:
        """
        Validate if the provided asset class is valid (RPC method).

        Args:
            asset_class: The asset class string to validate

        Returns:
            True if valid, False otherwise
        """
        return self._manager.is_valid_asset_class(asset_class)

    def process_asset_selection_request_rpc(
        self,
        asset_selection: str,
        miner: str
    ) -> Dict[str, str]:
        """
        Process an asset selection request from a miner (RPC method).

        Args:
            asset_selection: The asset class the miner wants to select
            miner: The miner's hotkey

        Returns:
            Dict containing success status and message
        """
        result = self._manager.process_asset_selection_request(asset_selection, miner)

        # If successful, broadcast to validators (delegate to manager)
        if result.get('successfully_processed') and 'asset_class' in result:
            asset_class = result['asset_class']
            self._manager.broadcast_asset_selection_to_validators(miner, asset_class)
            # Remove asset_class from result before returning (not needed by client)
            result = {k: v for k, v in result.items() if k != 'asset_class'}

        return result

    def delete_asset_selection_rpc(self, hotkey: str) -> Dict[str, str]:
        """
        Delete an asset selection for a miner (RPC method).

        Args:
            hotkey: The miner's hotkey to delete

        Returns:
            Dict containing success status and message
        """
        return self._manager.delete_asset_selection(hotkey)

    def sync_miner_asset_selection_data_rpc(self, asset_selection_data: Dict[str, str]) -> None:
        """
        Sync miner asset selection data from external source (RPC method).

        Args:
            asset_selection_data: Dict mapping hotkey to asset class string
        """
        self._manager.sync_miner_asset_selection_data(asset_selection_data)

    def receive_asset_selection_update_rpc(self, asset_selection_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process an incoming AssetSelection synapse and update miner asset selection (RPC method).

        Args:
            asset_selection_data: Dictionary containing hotkey, asset selection
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        return self._manager.receive_asset_selection_update(asset_selection_data, sender_hotkey)

    def to_dict_rpc(self) -> Dict:
        """
        Convert asset selections to disk format (RPC method).

        Returns:
            Dict mapping hotkey to asset class string
        """
        return self._manager._to_dict()

    def save_asset_selections_to_disk_rpc(self) -> None:
        """Save asset selections to disk (RPC method)."""
        self._manager._save_asset_selections_to_disk()

    def receive_asset_selection_rpc(
        self,
        synapse: template.protocol.AssetSelection
    ) -> template.protocol.AssetSelection:
        """
        Receive asset selection synapse (RPC method for axon handler).

        This is called by the validator's axon when receiving an AssetSelection synapse.

        Args:
            synapse: AssetSelection synapse from another validator

        Returns:
            Updated synapse with success/error status
        """
        try:
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"[ASSET_SERVER] Received AssetSelection synapse from validator hotkey [{sender_hotkey}]")
            success = self._manager.receive_asset_selection_update(synapse.asset_selection, sender_hotkey)

            if success:
                synapse.successfully_processed = True
                synapse.error_message = ""
                bt.logging.info(f"[ASSET_SERVER] Successfully processed AssetSelection synapse from {sender_hotkey}")
            else:
                synapse.successfully_processed = False
                synapse.error_message = "Failed to process asset selection"
                bt.logging.warning(f"[ASSET_SERVER] Failed to process AssetSelection synapse from {sender_hotkey}")

        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Error processing asset selection: {str(e)}"
            bt.logging.error(f"[ASSET_SERVER] Exception in receive_asset_selection: {e}")

        return synapse

    def clear_asset_selections_for_test_rpc(self) -> None:
        """
        Clear all asset selections (TEST ONLY - requires running_unit_tests=True).

        This method is only available when the server is running in test mode.
        It clears all asset selections from memory and disk.
        """
        if not self.running_unit_tests:
            raise RuntimeError("clear_asset_selections_for_test is only available in test mode")
        self._manager.asset_selections.clear()
        self._manager._save_asset_selections_to_disk()

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def get_asset_selections(self) -> Dict[str, TradePairCategory]:
        """Get asset selections dict (forward-compatible alias)."""
        return self.get_asset_selections_rpc()

    def get_all_miner_selections(self) -> Dict[str, str]:
        """Get all miner selections (forward-compatible alias)."""
        return self.get_all_miner_selections_rpc()

    def validate_order_asset_class(
        self,
        miner_hotkey: str,
        trade_pair_category: TradePairCategory,
        timestamp_ms: int = None
    ) -> bool:
        """Validate order asset class (forward-compatible alias)."""
        return self.validate_order_asset_class_rpc(miner_hotkey, trade_pair_category, timestamp_ms)

    def is_valid_asset_class(self, asset_class: str) -> bool:
        """Validate asset class (forward-compatible alias)."""
        return self.is_valid_asset_class_rpc(asset_class)

    def process_asset_selection_request(self, asset_selection: str, miner: str) -> Dict[str, str]:
        """Process asset selection request (forward-compatible alias)."""
        return self.process_asset_selection_request_rpc(asset_selection, miner)

    def delete_asset_selection(self, hotkey: str) -> Dict[str, str]:
        """Delete asset selection (forward-compatible alias)."""
        return self.delete_asset_selection_rpc(hotkey)

    def sync_miner_asset_selection_data(self, asset_selection_data: Dict[str, str]) -> None:
        """Sync asset selection data (forward-compatible alias)."""
        self.sync_miner_asset_selection_data_rpc(asset_selection_data)

    def receive_asset_selection_update(self, asset_selection_data: dict) -> bool:
        """Receive asset selection update (forward-compatible alias)."""
        return self.receive_asset_selection_update_rpc(asset_selection_data)

    @property
    def asset_selections(self) -> Dict[str, TradePairCategory]:
        """Direct access to asset_selections for backward compatibility."""
        return self._manager.asset_selections
