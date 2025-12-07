# developer: jbonilla
# Copyright � 2024 Taoshi Inc
"""
EntityServer - RPC server for entity miner management.

This server runs in its own process and exposes entity management via RPC.
Clients connect using EntityClient.

Follows the same pattern as ChallengePeriodServer.
"""
import bittensor as bt
from typing import Optional, Tuple, Dict, List

import template.protocol
from entitiy_management.entity_manager import EntityManager, SubaccountInfo, EntityData
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.rpc_server_base import RPCServerBase


class EntityServer(RPCServerBase):
    """
    RPC server for entity miner management.

    Wraps EntityManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to EntityClient.

    This follows the same pattern as ChallengePeriodServer and EliminationServer.
    """
    service_name = ValiConfig.RPC_ENTITY_SERVICE_NAME
    service_port = ValiConfig.RPC_ENTITY_PORT

    def __init__(
        self,
        *,
        config=None,
        is_backtesting=False,
        slack_notifier=None,
        start_server=True,
        start_daemon=False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize EntityServer IN-PROCESS (never spawns).

        Args:
            config: Validator config (for netuid, wallet) - required for EntityManager
            is_backtesting: Whether running in backtesting mode
            slack_notifier: Slack notifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests

        # Create mock config if running tests and config not provided
        if running_unit_tests:
            from shared_objects.rpc.test_mock_factory import TestMockFactory
            config = TestMockFactory.create_mock_config_if_needed(config, netuid=116, network="test")

        # Create the actual EntityManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        self._manager = EntityManager(
            is_backtesting=is_backtesting,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
            config=config
        )

        bt.logging.info("[ENTITY_SERVER] EntityManager initialized")

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        # daemon_interval_s: 5 minutes (challenge period + elimination assessment)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms during normal sleep
        daemon_interval_s = ValiConfig.ENTITY_ELIMINATION_CHECK_INTERVAL  # 300s (5 minutes)
        hang_timeout_s = daemon_interval_s * 2.0  # 600s (10 minutes, 2x interval)

        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_ENTITY_SERVICE_NAME,
            port=ValiConfig.RPC_ENTITY_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
            connection_mode=connection_mode
        )

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Runs every 5 minutes to:
        - Check elimination registry and sync subaccount status
        - Mark eliminated subaccounts in EntityManager state
        """
        # Run elimination assessment - sync with central elimination registry
        elim_count = self._manager.assess_eliminations()

        bt.logging.info(
            f"[ENTITY_SERVER] Daemon iteration complete: "
            f"{elim_count} eliminations synced"
        )

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        all_entities = self._manager.get_all_entities()
        total_subaccounts = sum(len(entity.subaccounts) for entity in all_entities.values())
        active_subaccounts = sum(len(entity.get_active_subaccounts()) for entity in all_entities.values())

        return {
            "total_entities": len(all_entities),
            "total_subaccounts": total_subaccounts,
            "active_subaccounts": active_subaccounts
        }

    # ==================== Entity Registration RPC Methods ====================

    def register_entity_rpc(
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
        return self._manager.register_entity(entity_hotkey, collateral_amount, max_subaccounts)

    def create_subaccount_rpc(self, entity_hotkey: str) -> Tuple[bool, Optional[dict], str]:
        """
        Create a new subaccount for an entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            (success: bool, subaccount_info_dict: Optional[dict], message: str)
        """
        success, subaccount_info, message = self._manager.create_subaccount(entity_hotkey)

        # Convert SubaccountInfo to dict for RPC serialization
        subaccount_dict = subaccount_info.model_dump() if subaccount_info else None

        return success, subaccount_dict, message

    def eliminate_subaccount_rpc(
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
        return self._manager.eliminate_subaccount(entity_hotkey, subaccount_id, reason)

    def update_collateral_rpc(self, entity_hotkey: str, collateral_amount: float) -> Tuple[bool, str]:
        """
        Update collateral for an entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            collateral_amount: New collateral amount

        Returns:
            (success: bool, message: str)
        """
        return self._manager.update_collateral(entity_hotkey, collateral_amount)

    # ==================== Query RPC Methods ====================

    def get_subaccount_status_rpc(self, synthetic_hotkey: str) -> Tuple[bool, Optional[str], str]:
        """
        Get the status of a subaccount by synthetic hotkey.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (found: bool, status: Optional[str], synthetic_hotkey: str)
        """
        return self._manager.get_subaccount_status(synthetic_hotkey)

    def get_entity_data_rpc(self, entity_hotkey: str) -> Optional[dict]:
        """
        Get full entity data.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            Entity data as dict or None
        """
        entity_data = self._manager.get_entity_data(entity_hotkey)
        return entity_data.model_dump() if entity_data else None

    def get_all_entities_rpc(self) -> Dict[str, dict]:
        """
        Get all entities.

        Returns:
            Dict mapping entity_hotkey -> entity_data_dict
        """
        all_entities = self._manager.get_all_entities()
        return {hotkey: entity.model_dump() for hotkey, entity in all_entities.items()}

    def is_synthetic_hotkey_rpc(self, hotkey: str) -> bool:
        """
        Check if a hotkey is synthetic (contains underscore with integer suffix).

        Args:
            hotkey: The hotkey to check

        Returns:
            True if synthetic, False otherwise
        """
        return self._manager.is_synthetic_hotkey(hotkey)

    def parse_synthetic_hotkey_rpc(self, synthetic_hotkey: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a synthetic hotkey into entity_hotkey and subaccount_id.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (entity_hotkey, subaccount_id) or (None, None) if invalid
        """
        return self._manager.parse_synthetic_hotkey(synthetic_hotkey)

    def validate_hotkey_for_orders_rpc(self, hotkey: str) -> dict:
        """
        Validate a hotkey for order placement in a single RPC call.

        Consolidates:
        - is_synthetic_hotkey() check
        - get_subaccount_status() check
        - get_entity_data() check

        Args:
            hotkey: The hotkey to validate

        Returns:
            dict with is_valid, error_message, hotkey_type, status
        """
        return self._manager.validate_hotkey_for_orders(hotkey)

    def get_subaccount_dashboard_data_rpc(self, synthetic_hotkey: str) -> Optional[dict]:
        """
        Get comprehensive dashboard data for a subaccount (RPC method).

        Aggregates data from:
        - ChallengePeriodClient: Challenge period status
        - DebtLedgerClient: Debt ledger data
        - PositionManagerClient: Positions and leverage
        - MinerStatisticsClient: Cached statistics (metrics, scores, rankings)
        - EliminationClient: Elimination status

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            Dict with aggregated dashboard data, or None if subaccount not found
        """
        return self._manager.get_subaccount_dashboard_data(synthetic_hotkey)

    # ==================== Validator Broadcast RPC Methods ====================

    def broadcast_subaccount_registration_rpc(
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
        self._manager.broadcast_subaccount_registration(
            entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey
        )

    def receive_subaccount_registration_update_rpc(self, subaccount_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process an incoming SubaccountRegistration synapse and update entity data (RPC method).

        This is the data-level handler that can be called directly via RPC or by the synapse handler.

        Args:
            subaccount_data: Dictionary containing entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        return self._manager.receive_subaccount_registration_update(subaccount_data, sender_hotkey)

    def receive_subaccount_registration_rpc(
        self,
        synapse: template.protocol.SubaccountRegistration
    ) -> template.protocol.SubaccountRegistration:
        """
        Receive subaccount registration synapse (RPC method for axon handler).

        This is called by the validator's axon when receiving a SubaccountRegistration synapse.

        Args:
            synapse: SubaccountRegistration synapse from another validator

        Returns:
            Updated synapse with success/error status
        """
        try:
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(
                f"[ENTITY_SERVER] Received SubaccountRegistration synapse from validator hotkey [{sender_hotkey}]"
            )
            success = self.receive_subaccount_registration_update_rpc(synapse.subaccount_data, sender_hotkey)

            if success:
                synapse.successfully_processed = True
                synapse.error_message = ""
                bt.logging.info(
                    f"[ENTITY_SERVER] Successfully processed SubaccountRegistration synapse from {sender_hotkey}"
                )
            else:
                synapse.successfully_processed = False
                synapse.error_message = "Failed to process subaccount registration"
                bt.logging.warning(
                    f"[ENTITY_SERVER] Failed to process SubaccountRegistration synapse from {sender_hotkey}"
                )

        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Error processing subaccount registration: {e}"
            bt.logging.error(f"[ENTITY_SERVER] Error processing SubaccountRegistration synapse: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

        return synapse

    # ==================== Testing/Admin RPC Methods ====================

    def clear_all_entities_rpc(self) -> None:
        """Clear all entity data (for testing only)."""
        self._manager.clear_all_entities()

    def to_checkpoint_dict_rpc(self) -> dict:
        """Get entity data as a checkpoint dict for serialization."""
        return self._manager.to_checkpoint_dict()
