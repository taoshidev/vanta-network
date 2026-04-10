# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
EntityCollateralServer - RPC server for entity cross-margin collateral.

Wraps EntityCollateralManager and exposes its methods via RPC.
Runs a daemon that periodically refreshes the collateral cache from on-chain contracts.

Clients connect using EntityCollateralClient.
"""

import bittensor as bt
from typing import Optional, Tuple

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
        # Create manager FIRST before RPCServerBase.__init__ to prevent race conditions
        self._manager = EntityCollateralManager(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
        )

        bt.logging.success("[ENTITY_COLLATERAL_SERVER] EntityCollateralManager initialized")

        super().__init__(
            service_name=ValiConfig.RPC_ENTITY_COLLATERAL_SERVICE_NAME,
            port=ValiConfig.RPC_ENTITY_COLLATERAL_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=float(ValiConfig.ENTITY_COLLATERAL_CACHE_REFRESH_S),
            hang_timeout_s=120.0,
            connection_mode=connection_mode,
        )

        bt.logging.success("[ENTITY_COLLATERAL_SERVER] EntityCollateralServer initialized")

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single daemon iteration: refresh the collateral cache from on-chain contracts.
        """
        self._manager.refresh_collateral_cache()

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "cached_entities": len(self._manager._collateral_cache),
            "slash_records": len(self._manager._slash_tracking),
        }

    # ==================== RPC Methods (exposed to clients) ====================

    def get_cached_collateral_rpc(self, entity_hotkey: str) -> Optional[float]:
        """Get cached collateral balance for an entity (RPC method)."""
        return self._manager.get_cached_collateral(entity_hotkey)

    def compute_entity_required_collateral_rpc(self, entity_hotkey: str) -> float:
        """Compute required collateral for an entity (RPC method)."""
        return self._manager.compute_entity_required_collateral(entity_hotkey)

    def can_open_position_rpc(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        additional_position_value: float,
    ) -> Tuple[bool, str]:
        """Check if a subaccount can open a new position (RPC method)."""
        return self._manager.can_open_position(entity_hotkey, synthetic_hotkey, additional_position_value)

    def slash_on_realized_loss_rpc(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        realized_loss: float,
    ) -> float:
        """Slash entity collateral on subaccount realized loss (RPC method)."""
        return self._manager.slash_on_realized_loss(entity_hotkey, synthetic_hotkey, realized_loss)

    def try_slash_on_elimination_rpc(self, hotkey: str) -> float:
        """Slash all remaining collateral for a funded subaccount being eliminated (RPC method)."""
        return self._manager.try_slash_on_elimination(hotkey)

    def get_cumulative_slashed_rpc(self, synthetic_hotkey: str) -> float:
        """Get cumulative slashed amount for a subaccount (RPC method)."""
        return self._manager.get_cumulative_slashed(synthetic_hotkey)

    def get_max_slash_rpc(self, synthetic_hotkey: str) -> float:
        """Get max slashable amount for a subaccount (RPC method)."""
        return self._manager.get_max_slash(synthetic_hotkey)

    def refresh_collateral_cache_rpc(self) -> int:
        """Force-refresh the collateral cache (RPC method)."""
        return self._manager.refresh_collateral_cache()

    # ==================== Test Helper RPC Methods ====================

    def set_test_collateral_cache_rpc(self, entity_hotkey: str, collateral_usd: float) -> None:
        """Test-only: Inject collateral cache value."""
        return self._manager.set_test_collateral_cache(entity_hotkey, collateral_usd)

    def set_test_slash_tracking_rpc(self, synthetic_hotkey: str, cumulative_realized_loss: float, cumulative_slashed: float) -> None:
        """Test-only: Inject slash tracking data."""
        return self._manager.set_test_slash_tracking(synthetic_hotkey, cumulative_realized_loss, cumulative_slashed)

    def get_test_slash_tracking_rpc(self, synthetic_hotkey: str) -> Optional[dict]:
        """Test-only: Get raw slash tracking data."""
        return self._manager.get_test_slash_tracking(synthetic_hotkey)

    def clear_test_state_rpc(self) -> None:
        """Test-only: Clear all state for test isolation."""
        return self._manager.clear_test_state()

    def get_test_slash_file_path_rpc(self) -> str:
        """Test-only: Get the slash tracking file path."""
        return self._manager.get_test_slash_file_path()
