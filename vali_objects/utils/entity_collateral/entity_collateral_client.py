# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
EntityCollateralClient - Lightweight RPC client for entity cross-margin collateral.

Connects to EntityCollateralServer via RPC. Can be created in ANY process.

Primary consumers:
- MarketOrderManager: calls can_open_position() before accepting orders from subaccounts.
- MarketOrderManager: calls slash_on_realized_loss() when subaccount positions close with loss.
"""

from typing import Optional, Tuple

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class EntityCollateralClient(RPCClientBase):
    """
    Lightweight RPC client for EntityCollateralServer.

    Can be created in ANY process. No server ownership.
    """

    def __init__(
        self,
        port: int = None,
        running_unit_tests: bool = False,
        connect_immediately: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
    ):
        """
        Initialize EntityCollateralClient.

        Args:
            port: Port number (default: ValiConfig.RPC_ENTITY_COLLATERAL_PORT).
            running_unit_tests: If True, don't connect (use set_direct_server() instead).
            connect_immediately: If True, connect in __init__.
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production.
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_ENTITY_COLLATERAL_SERVICE_NAME,
            port=port or ValiConfig.RPC_ENTITY_COLLATERAL_PORT,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode,
        )

    # ==================== Order Gating ====================

    def can_open_position(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        additional_position_value: float,
    ) -> Tuple[bool, str]:
        """
        Check if a subaccount can open a new position given entity cross-margin availability.

        Skips the check for challenge period subaccounts.

        Args:
            entity_hotkey: The entity's hotkey.
            synthetic_hotkey: The subaccount's synthetic hotkey.
            additional_position_value: USD value of the proposed new position.

        Returns:
            (allowed: bool, reason: str) - reason is empty if allowed.
        """
        return self._server.can_open_position_rpc(
            entity_hotkey, synthetic_hotkey, additional_position_value
        )

    # ==================== Slashing ====================

    def slash_on_realized_loss(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        realized_loss: float,
    ) -> float:
        """
        Slash entity collateral when a subaccount closes a position with realized loss.

        Args:
            entity_hotkey: The entity's hotkey.
            synthetic_hotkey: The subaccount's synthetic hotkey.
            realized_loss: The realized loss in USD (positive number).

        Returns:
            Actual amount slashed in USD.
        """
        return self._server.slash_on_realized_loss_rpc(
            entity_hotkey, synthetic_hotkey, realized_loss
        )

    # ==================== Query Methods ====================

    def get_cached_collateral(self, entity_hotkey: str) -> Optional[float]:
        """
        Get the cached collateral balance for an entity.

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Deposited collateral in USD, or None if not found.
        """
        return self._server.get_cached_collateral_rpc(entity_hotkey)

    def compute_entity_required_collateral(self, entity_hotkey: str) -> float:
        """
        Compute the total required collateral for an entity.

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Required collateral in USD.
        """
        return self._server.compute_entity_required_collateral_rpc(entity_hotkey)

    def get_cumulative_slashed(self, synthetic_hotkey: str) -> float:
        """
        Get cumulative slashed amount for a subaccount.

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Cumulative slashed amount in USD.
        """
        return self._server.get_cumulative_slashed_rpc(synthetic_hotkey)

    def get_max_slash(self, synthetic_hotkey: str) -> float:
        """
        Get max slashable amount for a subaccount (account_balance * MDD%).

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Max slash amount in USD.
        """
        return self._server.get_max_slash_rpc(synthetic_hotkey)

    # ==================== Utility ====================

    def health_check(self) -> dict:
        """Check server health."""
        return self._server.health_check_rpc()
