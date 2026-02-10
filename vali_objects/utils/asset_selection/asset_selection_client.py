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

    # Check if asset class is valid
    if client.is_valid_asset_class("forex"):
        print("Valid asset class")

    # Get all selections
    selections = client.get_all_miner_selections()
"""
from typing import Dict, Optional

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
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

    def get_asset_selections(self) -> Dict[str, TradePairCategory]:
        """
        Get all asset selections.

        Returns:
            Dict mapping hotkey to TradePairCategory
        """
        return self._server.get_asset_selections_rpc()

    def get_asset_selection(self, hotkey) -> TradePairCategory | None:
        return self._server.get_asset_selection_rpc(hotkey)

    def get_all_miner_selections(self) -> Dict[str, str]:
        """
        Get all miner asset selections as string dict.

        Returns:
            Dict mapping hotkey to asset class string
        """
        return self._server.get_all_miner_selections_rpc()

    def validate_order_asset_class(
        self,
        miner_hotkey: str,
        trade_pair_category: TradePairCategory,
        timestamp_ms: int = None
    ) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check
            timestamp_ms: Optional timestamp in milliseconds

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        return self._server.validate_order_asset_class_rpc(
            miner_hotkey, trade_pair_category, timestamp_ms
        )

    def is_valid_asset_class(self, asset_class: str) -> bool:
        """
        Validate if the provided asset class is valid.

        Args:
            asset_class: The asset class string to validate

        Returns:
            True if valid, False otherwise
        """
        return self._server.is_valid_asset_class_rpc(asset_class)

    # ==================== Mutation Methods ====================

    def process_asset_selection_request(
        self,
        asset_selection: str,
        miner: str
    ) -> Dict[str, str]:
        """
        Process an asset selection request from a miner.

        Args:
            asset_selection: The asset class the miner wants to select
            miner: The miner's hotkey

        Returns:
            Dict containing success status and message
        """
        return self._server.process_asset_selection_request_rpc(asset_selection, miner)

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
    def asset_selections(self) -> Dict[str, TradePairCategory]:
        """
        Get asset selections dict (backward compatibility).

        Returns:
            Dict mapping hotkey to TradePairCategory
        """
        return self._server.get_asset_selections_rpc()

    # ==================== Local Cache Support ====================

    def populate_cache(self) -> Dict[str, TradePairCategory]:
        """
        Populate the local cache with asset selection data from the server.

        Called periodically by the cache refresh daemon when
        local_cache_refresh_period_ms is configured.

        Returns:
            Dict mapping hotkey to TradePairCategory
        """
        return self._server.get_asset_selections_rpc()

    def get_selection_local_cache(self, hotkey: str) -> Optional[TradePairCategory]:
        """
        Get asset selection for a hotkey from the local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            hotkey: The miner's hotkey

        Returns:
            TradePairCategory if found, None otherwise
        """
        with self._local_cache_lock:
            return self._local_cache.get(hotkey)

    def validate_order_asset_class_local_cache(
        self,
        miner_hotkey: str,
        trade_pair_category: TradePairCategory,
        timestamp_ms: int = None
    ) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class using local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check
            timestamp_ms: Optional timestamp in milliseconds

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        with self._local_cache_lock:
            selected_asset_class = self._local_cache.get(miner_hotkey)
            if selected_asset_class is None:
                return False
            return selected_asset_class == trade_pair_category
