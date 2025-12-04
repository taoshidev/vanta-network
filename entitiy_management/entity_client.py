# developer: jbonilla
# Copyright � 2024 Taoshi Inc
"""
EntityClient - Lightweight RPC client for entity miner management.

This client connects to the EntityServer via RPC.
Can be created in ANY process - just needs the server to be running.

Usage:
    from entitiy_management.entity_client import EntityClient

    # Connect to server (uses ValiConfig.RPC_ENTITY_PORT by default)
    client = EntityClient()

    # Register an entity
    success, message = client.register_entity("my_entity_hotkey")

    # Create a subaccount
    success, subaccount_info, message = client.create_subaccount("my_entity_hotkey")

    # Check if hotkey is synthetic
    if client.is_synthetic_hotkey(hotkey):
        entity_hotkey, subaccount_id = client.parse_synthetic_hotkey(hotkey)
"""
from typing import Optional, Tuple, Dict

import template.protocol
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
        entity_hotkey: str,
        collateral_amount: float = 0.0,
        max_subaccounts: int = None
    ) -> Tuple[bool, str]:
        """
        Register a new entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            collateral_amount: Collateral amount (placeholder)
            max_subaccounts: Maximum allowed subaccounts

        Returns:
            (success: bool, message: str)
        """
        return self._server.register_entity_rpc(entity_hotkey, collateral_amount, max_subaccounts)

    def create_subaccount(self, entity_hotkey: str) -> Tuple[bool, Optional[dict], str]:
        """
        Create a new subaccount for an entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            (success: bool, subaccount_info_dict: Optional[dict], message: str)
        """
        return self._server.create_subaccount_rpc(entity_hotkey)

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

    def update_collateral(self, entity_hotkey: str, collateral_amount: float) -> Tuple[bool, str]:
        """
        Update collateral for an entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            collateral_amount: New collateral amount

        Returns:
            (success: bool, message: str)
        """
        return self._server.update_collateral_rpc(entity_hotkey, collateral_amount)

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

    def get_all_entities(self) -> Dict[str, dict]:
        """
        Get all entities.

        Returns:
            Dict mapping entity_hotkey -> entity_data_dict
        """
        return self._server.get_all_entities_rpc()

    def is_synthetic_hotkey(self, hotkey: str) -> bool:
        """
        Check if a hotkey is synthetic (contains underscore with integer suffix).

        Args:
            hotkey: The hotkey to check

        Returns:
            True if synthetic, False otherwise
        """
        return self._server.is_synthetic_hotkey_rpc(hotkey)

    def parse_synthetic_hotkey(self, synthetic_hotkey: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a synthetic hotkey into entity_hotkey and subaccount_id.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (entity_hotkey, subaccount_id) or (None, None) if invalid
        """
        return self._server.parse_synthetic_hotkey_rpc(synthetic_hotkey)

    # ==================== Validator Broadcast Methods ====================

    def broadcast_subaccount_registration(
        self,
        entity_hotkey: str,
        subaccount_id: int,
        subaccount_uuid: str,
        synthetic_hotkey: str
    ) -> None:
        """
        Broadcast subaccount registration to other validators.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            subaccount_id: The subaccount ID
            subaccount_uuid: The subaccount UUID
            synthetic_hotkey: The synthetic hotkey
        """
        return self._server.broadcast_subaccount_registration_rpc(
            entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey
        )

    def receive_subaccount_registration_update(self, subaccount_data: dict) -> bool:
        """
        Process incoming subaccount registration from another validator.

        Args:
            subaccount_data: Dict containing entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey

        Returns:
            bool: True if successful, False otherwise
        """
        # This delegates to EntityManager.receive_subaccount_registration via RPC
        # Returns True if successful, False otherwise
        success = self._server.receive_subaccount_registration_rpc(
            template.protocol.SubaccountRegistration(subaccount_data=subaccount_data)
        ).successfully_processed
        return success

    def receive_subaccount_registration(
        self,
        synapse: 'template.protocol.SubaccountRegistration'
    ) -> 'template.protocol.SubaccountRegistration':
        """
        Receive subaccount registration synapse (for axon attachment).

        This delegates to the server's RPC handler. Used by validator_base.py for axon attachment.

        Args:
            synapse: SubaccountRegistration synapse from another validator

        Returns:
            Updated synapse with success/error status
        """
        return self._server.receive_subaccount_registration_rpc(synapse)

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
