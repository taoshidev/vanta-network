# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
EntityCollateralServer - RPC server for entity cross-margin collateral.

Wraps EntityCollateralManager and exposes its methods via RPC.
Runs a daemon that periodically refreshes the collateral cache from on-chain contracts.

Clients connect using EntityCollateralClient.
"""

from typing import Dict, Optional, Tuple

from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.utils.entity_collateral.entity_collateral_manager import EntityCollateralManager
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class EntityCollateralServer(RPCServerBase):
    """
    RPC server for entity cross-margin collateral.

    Wraps EntityCollateralManager and exposes its methods via RPC.
    Daemon thread refreshes collateral cache every ~60s.
    """

    service_name = ValiConfig.RPC_ENTITY_COLLATERAL_SERVICE_NAME
    service_port = ValiConfig.RPC_ENTITY_COLLATERAL_PORT

    def __init__(
        self,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
    ):
        """
        Initialize EntityCollateralServer.

        Args:
            running_unit_tests: Whether running in test mode.
            slack_notifier: Slack notifier for alerts.
            start_server: Whether to start the RPC server immediately.
            start_daemon: Whether to start the cache refresh daemon immediately.
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production.
        """
        # TODO: Create EntityCollateralManager FIRST (before RPCServerBase.__init__)
        # TODO: Initialize RPCServerBase with daemon_interval_s=60
        # TODO: Start daemon if requested
        pass

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single daemon iteration: refresh the collateral cache from on-chain contracts.
        """
        # TODO: Delegate to self._manager.refresh_collateral_cache()
        pass

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        pass

    # ==================== RPC Methods (exposed to clients) ====================

    def get_cached_collateral_rpc(self, entity_hotkey: str) -> Optional[float]:
        """Get cached collateral balance for an entity (RPC method)."""
        pass

    def compute_entity_required_collateral_rpc(self, entity_hotkey: str) -> float:
        """Compute required collateral for an entity (RPC method)."""
        pass

    def can_open_position_rpc(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        additional_position_value: float,
    ) -> Tuple[bool, str]:
        """Check if a subaccount can open a new position (RPC method)."""
        pass

    def slash_on_realized_loss_rpc(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        realized_loss: float,
    ) -> float:
        """Slash entity collateral on subaccount realized loss (RPC method)."""
        pass

    def get_cumulative_slashed_rpc(self, synthetic_hotkey: str) -> float:
        """Get cumulative slashed amount for a subaccount (RPC method)."""
        pass

    def get_max_slash_rpc(self, synthetic_hotkey: str) -> float:
        """Get max slashable amount for a subaccount (RPC method)."""
        pass

    def refresh_collateral_cache_rpc(self) -> int:
        """Force-refresh the collateral cache (RPC method)."""
        pass
