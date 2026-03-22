# developer: jbonilla
# Copyright � 2024 Taoshi Inc
"""
EntityClient - Lightweight RPC client for entity miner management.

This client connects to the EntityServer via RPC.
Can be created in ANY process - just needs the server to be running.

Usage:
    from entitiy_management.entity_client import EntityClient
    from entitiy_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey

    # Connect to server (uses ValiConfig.RPC_ENTITY_PORT by default)
    client = EntityClient()

    # Register an entity
    success, message = client.register_entity("my_entity_hotkey")

    # Create a subaccount
    success, subaccount_info, message = client.create_subaccount("my_entity_hotkey")

    # Check if hotkey is synthetic (use entity_utils directly - no RPC overhead)
    if is_synthetic_hotkey(hotkey):
        entity_hotkey, subaccount_id = parse_synthetic_hotkey(hotkey)
"""
from typing import Optional, Tuple, Dict, List

import template.protocol
from template.protocol import SubaccountRegistration, EntityEndpointUpdate
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class EntityClient(RPCClientBase):
    """
    Lightweight RPC client for EntityServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_ENTITY_PORT.

    In LOCAL mode (connection_mode=RPCConnectionMode.LOCAL), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct EntityServer instance.
    """

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False,
        connect_immediately: bool = False
    ):
        """
        Initialize entity client.

        Args:
            port: Port number of the entity server (default: ValiConfig.RPC_ENTITY_PORT)
            connection_mode: RPCConnectionMode.LOCAL for tests (use set_direct_server()), RPCConnectionMode.RPC for production
            running_unit_tests: Whether running in test mode
            connect_immediately: Whether to connect immediately (default: False for lazy connection)
        """
        self._direct_server = None
        self.running_unit_tests = running_unit_tests

        # In LOCAL mode, don't connect via RPC - tests will set direct server
        super().__init__(
            service_name=ValiConfig.RPC_ENTITY_SERVICE_NAME,
            port=port or ValiConfig.RPC_ENTITY_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Entity Registration Methods ====================

    def register_entity(
        self,
        entity_hotkey: str
    ) -> Tuple[bool, str]:
        """
        Register a new entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            (success: bool, message: str)
        """
        return self._server.register_entity_rpc(entity_hotkey)

    def create_subaccount(
        self,
        entity_hotkey: str,
        account_size: float,
        asset_class: str,
        admin: bool = False
    ) -> Tuple[bool, Optional[dict], str]:
        """
        Create a new subaccount for an entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            account_size: Account size in USD
            asset_class: Asset class selection
            admin: If True, skip collateral slashing and exclude from payouts

        Returns:
            (success: bool, subaccount_info_dict: Optional[dict], message: str)
        """
        return self._server.create_subaccount_rpc(entity_hotkey, account_size, asset_class, admin=admin)

    def create_hl_subaccount(
        self,
        entity_hotkey: str,
        account_size: float,
        hl_address: str,
        asset_class: str = "crypto",
        admin: bool = False,
        payout_address: Optional[str] = None
    ) -> Tuple[bool, Optional[dict], str]:
        """
        Create a new subaccount linked to a Hyperliquid address.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            account_size: Account size in USD
            hl_address: Hyperliquid address (0x-prefixed, 40 hex chars)
            asset_class: Asset class selection (default: "crypto")
            admin: If True, skip collateral slashing
            payout_address: Optional EVM address (0x + 40 hex) for USDC payouts

        Returns:
            (success: bool, subaccount_info_dict: Optional[dict], message: str)
        """
        return self._server.create_hl_subaccount_rpc(entity_hotkey, account_size, hl_address, asset_class=asset_class, admin=admin, payout_address=payout_address)

    def get_all_active_hl_subaccounts(self) -> List[Tuple[str, dict]]:
        """
        Get all active subaccounts with HL addresses.

        Returns:
            List of (hl_address, subaccount_info_dict) tuples
        """
        return self._server.get_all_active_hl_subaccounts_rpc()

    def get_synthetic_hotkey_for_hl_address(self, hl_address: str) -> Optional[str]:
        """
        O(1) lookup of synthetic hotkey for a Hyperliquid address.

        Args:
            hl_address: The Hyperliquid address

        Returns:
            Synthetic hotkey if found, None otherwise
        """
        return self._server.get_synthetic_hotkey_for_hl_address_rpc(hl_address)

    def get_subaccount_info_for_synthetic(self, synthetic_hotkey: str) -> Optional[dict]:
        """
        Get SubaccountInfo for a synthetic hotkey.

        Args:
            synthetic_hotkey: The synthetic hotkey

        Returns:
            SubaccountInfo dict if found, None otherwise
        """
        return self._server.get_subaccount_info_for_synthetic_rpc(synthetic_hotkey)

    def get_hl_subaccount_limits_data(self, hl_address: str) -> Optional[dict]:
        """
        Get lightweight limits data for an HL subaccount.

        Args:
            hl_address: The Hyperliquid address

        Returns:
            Dict with {account_size, asset_class, challenge_bucket} or None
        """
        return self._server.get_hl_subaccount_limits_data_rpc(hl_address)

    def eliminate_subaccount(
        self,
        entity_hotkey: str,
        subaccount_id: int,
        reason: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Eliminate a subaccount.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            subaccount_id: The subaccount ID to eliminate
            reason: Elimination reason

        Returns:
            (success: bool, message: str)
        """
        return self._server.eliminate_subaccount_rpc(entity_hotkey, subaccount_id, reason)

    def restore_subaccount(self, synthetic_hotkey: str) -> Tuple[bool, str]:
        """
        Restore an erroneously eliminated subaccount to active status.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (success: bool, message: str)
        """
        return self._server.restore_subaccount_rpc(synthetic_hotkey)

    # ==================== Query Methods ====================

    def get_subaccount_status(self, synthetic_hotkey: str) -> Tuple[bool, Optional[str], str]:
        """
        Get the status of a subaccount by synthetic hotkey.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (found: bool, status: Optional[str], synthetic_hotkey: str)
        """
        return self._server.get_subaccount_status_rpc(synthetic_hotkey)

    def get_entity_data(self, entity_hotkey: str) -> Optional[dict]:
        """
        Get full entity data.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            Entity data as dict or None
        """
        return self._server.get_entity_data_rpc(entity_hotkey)

    def get_subaccount_dashboard(self, synthetic_hotkey: str) -> dict | None:
        return self._server.get_subaccount_dashboard_rpc(synthetic_hotkey)

    def get_all_entities(self) -> Dict[str, dict]:
        """
        Get all entities.

        Returns:
            Dict mapping entity_hotkey -> entity_data_dict
        """
        return self._server.get_all_entities_rpc()

    def get_hl_leaderboard_data(self) -> dict:
        """
        Get aggregated HL leaderboard data.

        Returns:
            Dict with summary, fundedTraders, challengeTraders, timestamp
        """
        return self._server.get_hl_leaderboard_data_rpc()

    def validate_hotkey_for_orders(self, hotkey: str) -> dict:
        """
        Validate a hotkey for order placement in a single RPC call.

        This consolidates multiple checks into one RPC call:
        - Synthetic hotkey detection (via entity_utils.is_synthetic_hotkey)
        - Subaccount status check
        - Entity data check

        Args:
            hotkey: The hotkey to validate

        Returns:
            dict with:
                - is_valid (bool): Whether hotkey can place orders
                - error_message (str): Error message if not valid, empty if valid
                - hotkey_type (str): 'synthetic', 'entity', or 'regular'
                - status (str|None): Status if synthetic hotkey, None otherwise
        """
        return self._server.validate_hotkey_for_orders_rpc(hotkey)

    def get_subaccount_dashboard_data(self, synthetic_hotkey: str) -> Optional[dict]:
        """
        Get comprehensive dashboard data for a subaccount.

        This method aggregates data from multiple RPC services:
        - Subaccount info (status, timestamps)
        - Challenge period status (bucket, start time)
        - Debt ledger data (DebtLedger instance)
        - Position data (positions, leverage)
        - Statistics (cached miner statistics with metrics, scores, rankings)
        - Elimination status (if eliminated)

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            Dict with aggregated dashboard data, or None if subaccount not found
        """
        return self._server.get_subaccount_dashboard_data_rpc(synthetic_hotkey)

    def broadcast_subaccount_dashboard(self, synthetic_hotkey: str) -> None:
        return self._server.broadcast_subaccount_dashboard_rpc(synthetic_hotkey)

    def calculate_subaccount_payout(
        self,
        subaccount_uuid: str,
        start_time_ms: int,
        end_time_ms: Optional[int]
    ) -> Optional[dict]:
        """
        Calculate payout for a subaccount based on debt ledger checkpoints.

        Args:
            subaccount_uuid: The subaccount UUID
            start_time_ms: Start timestamp (inclusive)
            end_time_ms: End timestamp (inclusive); if None, uses current time

        Returns:
            Dict with {hotkey, total_checkpoints, checkpoints, payout} or None
        """
        return self._server.calculate_subaccount_payout_rpc(
            subaccount_uuid,
            start_time_ms,
            end_time_ms
        )

    # ==================== Validator Broadcast Methods ====================

    def broadcast_subaccount_registration(
        self,
        entity_hotkey: str,
        subaccount_id: int,
        subaccount_uuid: str,
        synthetic_hotkey: str,
        account_size: float,
        asset_class: str,
        status: str = "active",
        hl_address: Optional[str] = None,
        payout_address: Optional[str] = None
    ) -> None:
        """
        Broadcast subaccount registration to other validators.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            subaccount_id: The subaccount ID
            subaccount_uuid: The subaccount UUID
            synthetic_hotkey: The synthetic hotkey
            account_size: Account size in USD
            asset_class: Asset class selection
            status: Subaccount status (active, admin, etc.)
            hl_address: Optional Hyperliquid address for HL-linked subaccounts
            payout_address: Optional EVM address for USDC payouts
        """
        return self._server.broadcast_subaccount_registration_rpc(
            entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey,
            account_size, asset_class, status, hl_address=hl_address, payout_address=payout_address
        )

    def receive_subaccount_registration_update(self, subaccount_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process incoming subaccount registration from another validator.

        Args:
            subaccount_data: Dict containing entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        # Call the data-level RPC method (not the synapse handler)
        return self._server.receive_subaccount_registration_update_rpc(subaccount_data, sender_hotkey)

    def receive_subaccount_registration(
        self,
        synapse: SubaccountRegistration
    ) -> SubaccountRegistration:
        """
        Receive subaccount registration synapse (for axon attachment).

        This delegates to the server's RPC handler. Used by validator_base.py for axon attachment.

        Args:
            synapse: SubaccountRegistration synapse from another validator

        Returns:
            Updated synapse with success/error status
        """
        return self._server.receive_subaccount_registration_rpc(synapse)

    # ==================== Entity Endpoint URL Methods ====================

    def set_endpoint_url(
        self,
        entity_hotkey: str,
        endpoint_url: str
    ) -> Tuple[bool, str]:
        """
        Set the public endpoint URL for an entity miner.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            endpoint_url: The public-facing endpoint URL

        Returns:
            (success: bool, message: str)
        """
        return self._server.set_endpoint_url_rpc(entity_hotkey, endpoint_url)

    def get_endpoint_url_by_address(
        self,
        hl_address: str = None,
        subaccount: str = None
    ) -> Optional[str]:
        """
        Resolve an HL address or synthetic hotkey to the entity's endpoint URL.

        Args:
            hl_address: Hyperliquid address (0x-prefixed)
            subaccount: Synthetic hotkey (entity_hotkey_N)

        Returns:
            The entity's endpoint URL, or None if not found
        """
        return self._server.get_endpoint_url_by_address_rpc(hl_address=hl_address, subaccount=subaccount)

    def receive_entity_endpoint_update(self, endpoint_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process incoming entity endpoint update from another validator.

        Args:
            endpoint_data: Dict containing entity_hotkey and endpoint_url
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        return self._server.receive_entity_endpoint_update_rpc(endpoint_data, sender_hotkey)

    def receive_entity_endpoint_synapse(
        self,
        synapse: EntityEndpointUpdate
    ) -> EntityEndpointUpdate:
        """
        Receive EntityEndpointUpdate synapse (for axon attachment).

        This delegates to the server's RPC handler. Used by validator_base.py for axon attachment.

        Args:
            synapse: EntityEndpointUpdate synapse from another validator

        Returns:
            Updated synapse with success/error status
        """
        return self._server.receive_entity_endpoint_synapse_rpc(synapse)

    # ==================== Health Check Methods ====================

    def health_check(self) -> dict:
        """
        Get health status from server.

        Returns:
            dict: Health status with 'status', 'service', 'timestamp_ms' and service-specific info
        """
        return self._server.health_check_rpc()

    # ==================== Testing/Admin Methods ====================

    def clear_all_entities(self) -> None:
        """Clear all entity data (for testing only)."""
        self._server.clear_all_entities_rpc()

    def to_checkpoint_dict(self) -> dict:
        """Get entity data as a checkpoint dict for serialization."""
        return self._server.to_checkpoint_dict_rpc()

    def sync_entity_data(self, entities_checkpoint_dict: dict) -> dict:
        """
        Sync entity data from checkpoint.

        Args:
            entities_checkpoint_dict: Dict from checkpoint (entity_hotkey -> EntityData dict)

        Returns:
            dict: Sync statistics (entities_added, subaccounts_added, subaccounts_updated)
        """
        return self._server.sync_entity_data_rpc(entities_checkpoint_dict)

    # ==================== Daemon Control Methods ====================

    def start_daemon(self) -> bool:
        """
        Start the daemon thread remotely via RPC.

        Returns:
            bool: True if daemon was started, False if already running
        """
        return self._server.start_daemon_rpc()

    def stop_daemon(self) -> bool:
        """
        Stop the daemon thread remotely via RPC.

        Returns:
            bool: True if daemon was stopped, False if not running
        """
        return self._server.stop_daemon_rpc()

    def is_daemon_running(self) -> bool:
        """
        Check if daemon is running via RPC.

        Returns:
            bool: True if daemon is running, False otherwise
        """
        return self._server.is_daemon_running_rpc()
