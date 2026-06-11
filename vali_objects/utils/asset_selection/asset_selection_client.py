# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
AssetSelectionClient - Lightweight RPC client for asset selection management.

This client connects to the AssetSelectionServer via RPC.
Can be created in ANY process - just needs the server to be running.

Usage:
    from vali_objects.utils.asset_selection_client import AssetSelectionClient

    # Connect to server (uses ValiConfig.RPC_ASSETSELECTION_PORT by default)
    client = AssetSelectionClient()

    # Get all selections
    selections = client.get_all_miner_selections()
"""
from typing import Dict, Optional

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
import template.protocol


class AssetSelectionClient(RPCClientBase):
    """
    Lightweight RPC client for AssetSelectionServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_ASSETSELECTION_PORT.

    Supports local caching for fast lookups without RPC calls:
        client = AssetSelectionClient(local_cache_refresh_period_ms=5000)
        # Fast local lookup (no RPC):
        selection = client.get_selection_local_cache(hotkey)
    """

    def __init__(
        self,
        port: int = None,
        running_unit_tests: bool = False,
        connect_immediately: bool = False,
        local_cache_refresh_period_ms: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize AssetSelectionClient.

        Args:
            port: Port number of the AssetSelection server (default: ValiConfig.RPC_ASSETSELECTION_PORT)
            running_unit_tests: If True, don't connect (use set_direct_server() instead)
            connect_immediately: If True, connect in __init__. If False, call connect() later.
            local_cache_refresh_period_ms: If not None, spawn a daemon thread that refreshes
                a local cache at this interval for fast lookups without RPC.
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_ASSETSELECTION_SERVICE_NAME,
            port=port or ValiConfig.RPC_ASSETSELECTION_PORT,
            connect_immediately=connect_immediately,
            local_cache_refresh_period_ms=local_cache_refresh_period_ms,
            connection_mode=connection_mode
        )

    # ==================== Query Methods ====================

    def get_asset_selections(self) -> Dict[str, MinerAssetClass]:
        """
        Get all asset selections.

        Returns:
            Dict mapping hotkey to MinerAssetClass
        """
        return self._server.get_asset_selections_rpc()

    def get_asset_selection(self, hotkey) -> MinerAssetClass | None:
        return self._server.get_asset_selection_rpc(hotkey)

    def get_all_miner_selections(self) -> Dict[str, str]:
        """
        Get all miner asset selections as string dict.

        Returns:
            Dict mapping hotkey to asset class string
        """
        return self._server.get_all_miner_selections_rpc()

    # ==================== Mutation Methods ====================

    def process_asset_selection_request(
        self,
        asset_selection: str,
        miner: str,
        overwrite: bool = False
    ) -> Dict[str, str]:
        """
        Process an asset selection request for a miner.

        Args:
            asset_selection: The asset class to select
            miner: The miner's hotkey
            overwrite: Overwrite existing selection if True, otherwise selection is immutable

        Returns:
            Dict containing success status and message
        """
        return self._server.process_asset_selection_request_rpc(asset_selection, miner, overwrite=overwrite)

    def delete_asset_selection(self, hotkey: str) -> Dict[str, str]:
        """
        Delete an asset selection for a miner.

        This allows the hotkey to select a new asset class, useful for
        rollback scenarios when operations fail.

        Args:
            hotkey: The miner's hotkey to delete

        Returns:
            Dict containing success status and message
        """
        return self._server.delete_asset_selection_rpc(hotkey)

    def sync_miner_asset_selection_data(self, asset_selection_data: Dict[str, str]) -> None:
        """
        Sync miner asset selection data from external source (backup/sync).

        Args:
            asset_selection_data: Dict mapping hotkey to asset class string
        """
        self._server.sync_miner_asset_selection_data_rpc(asset_selection_data)

    def receive_asset_selection_update(self, asset_selection_data: dict) -> bool:
        """
        Process an incoming AssetSelection synapse and update miner asset selection.

        Args:
            asset_selection_data: Dictionary containing hotkey, asset selection

        Returns:
            bool: True if successful, False otherwise
        """
        return self._server.receive_asset_selection_update_rpc(asset_selection_data)

    def receive_asset_selection(
        self,
        synapse: template.protocol.AssetSelection
    ) -> template.protocol.AssetSelection:
        """
        Receive asset selection synapse (for axon attachment).

        This delegates to the server's RPC handler. Used by validator_base.py for axon attachment.

        Args:
            synapse: AssetSelection synapse from another validator

        Returns:
            Updated synapse with success/error status
        """
        return self._server.receive_asset_selection_rpc(synapse)

    # ==================== Utility Methods ====================

    def health_check(self) -> dict:
        """Check server health."""
        return self._server.health_check_rpc()

    def to_dict(self) -> Dict[str, str]:
        """
        Convert asset selections to disk format.

        Returns:
            Dict mapping hotkey to asset class string
        """
        return self._server.to_dict_rpc()

    def save_asset_selections_to_disk(self) -> None:
        """Save asset selections to disk."""
        self._server.save_asset_selections_to_disk_rpc()

    def clear_asset_selections_for_test(self) -> None:
        """
        Clear all asset selections (TEST ONLY).

        This method is only available when the server is running in test mode.
        It clears all asset selections from memory and disk for test isolation.
        """
        self._server.clear_asset_selections_for_test_rpc()

    # ==================== Backward Compatibility Properties ====================

    @property
    def asset_selections(self) -> Dict[str, MinerAssetClass]:
        """
        Get asset selections dict (backward compatibility).

        Returns:
            Dict mapping hotkey to MinerAssetClass
        """
        return self._server.get_asset_selections_rpc()

    # ==================== Local Cache Support ====================

    def populate_cache(self) -> Dict[str, MinerAssetClass]:
        """
        Populate the local cache with asset selection data from the server.

        Called periodically by the cache refresh daemon when
        local_cache_refresh_period_ms is configured.

        Returns:
            Dict mapping hotkey to MinerAssetClass
        """
        return self._server.get_asset_selections_rpc()

    def refresh_local_cache(self) -> None:
        """
        Synchronously refresh the local cache from the server.

        Normally the cache is refreshed by the background daemon (when
        local_cache_refresh_period_ms is configured). This method forces an
        immediate refresh, which is useful for callers/tests that need the cache
        to reflect a just-applied selection change without waiting for the daemon.
        """
        new_cache = self.populate_cache()
        with self._local_cache_lock:
            self._local_cache = new_cache

    def get_selection_local_cache(self, hotkey: str) -> Optional[MinerAssetClass]:
        """
        Get asset selection for a hotkey from the local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            hotkey: The miner's hotkey

        Returns:
            MinerAssetClass if found, None otherwise
        """
        with self._local_cache_lock:
            return self._local_cache.get(hotkey)

