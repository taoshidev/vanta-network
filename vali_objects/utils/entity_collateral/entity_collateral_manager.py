# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
EntityCollateralManager - Core business logic for entity cross-margin collateral.

Manages:
- Background refresh of entity collateral balances from on-chain contracts
- On-disk caching of collateral balances for low-latency order gating
- Cross-margin exposure calculation across entity subaccounts
- Collateral slashing on subaccount realized losses
- Order rejection when entity cross-margin is fully utilized

This manager is wrapped by EntityCollateralServer which exposes methods via RPC.
"""

import threading
from typing import Dict, Optional, Tuple

from shared_objects.cache_controller import CacheController
from vali_objects.vali_config import RPCConnectionMode


class EntityCollateralManager(CacheController):
    """
    Core business logic for entity cross-margin collateral.

    Maintains an on-disk cache of entity collateral balances, refreshed
    periodically from on-chain contracts. Provides fast lookups for
    order gating and cross-margin calculations.

    Pattern follows other managers (EntityManager, MDDChecker):
    - Manager holds all business logic
    - Server wraps this and exposes via RPC
    - Local dicts for performance
    - Disk persistence via JSON
    """

    def __init__(
        self,
        *,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
    ):
        """
        Initialize EntityCollateralManager.

        Args:
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        # TODO: Initialize CacheController, RPC clients, load cache from disk
        pass

    # ==================== Cache Management ====================

    def refresh_collateral_cache(self) -> int:
        """
        Refresh cached collateral balances for all known entities from on-chain contracts.

        Called periodically by the daemon. Reads each entity's collateral balance
        from the ContractClient and writes results to the on-disk cache.

        Returns:
            Number of entities refreshed.
        """
        # TODO: Read all entity hotkeys from EntityClient
        # TODO: For each, call ContractClient.get_miner_collateral_balance()
        # TODO: Write updated cache to disk and memory
        pass

    def get_cached_collateral(self, entity_hotkey: str) -> Optional[float]:
        """
        Get the cached collateral balance for an entity (fast local lookup).

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Deposited collateral in USD, or None if entity not found in cache.
        """
        pass

    def _load_cache_from_disk(self) -> Dict[str, float]:
        """
        Load the entity collateral cache from disk.

        Returns:
            Dict mapping entity_hotkey -> deposited_collateral_usd.
        """
        pass

    def _save_cache_to_disk(self) -> None:
        """
        Persist the current in-memory collateral cache to disk.

        File location: validation/entity_collateral_cache.json
        """
        pass

    # ==================== Cross-Margin Calculation ====================

    def compute_entity_required_collateral(self, entity_hotkey: str) -> float:
        """
        Compute the total required collateral for an entity across all
        non-challenge-period subaccounts.

        For each subaccount:
            risk_exposure = min(sum(abs(position_value)), account_balance * MDD%)

        Entity required collateral = sum of all subaccount risk exposures.

        Challenge period subaccounts are excluded (their risk exposure is 0).

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Total required collateral in USD.
        """
        # TODO: Get all subaccounts from EntityClient
        # TODO: For each non-challenge-period subaccount:
        #   - Get open positions from PositionManagerClient
        #   - Get account balance from MinerAccountClient
        #   - Compute risk_exposure = min(sum(abs(position_value)), balance * MDD%)
        # TODO: Return sum of all risk exposures
        pass

    def compute_subaccount_risk_exposure(
        self,
        synthetic_hotkey: str,
        account_balance: float,
    ) -> float:
        """
        Compute the risk exposure for a single subaccount.

        risk_exposure = min(sum(abs(position_value)), account_balance * MDD%)

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.
            account_balance: The subaccount's account balance in USD.

        Returns:
            Risk exposure in USD.
        """
        # TODO: Get open positions for this subaccount
        # TODO: Sum abs(position_value) across all open positions
        # TODO: Return min(total_position_value, account_balance * MDD%)
        pass

    # ==================== Order Gating ====================

    def can_open_position(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        additional_position_value: float,
    ) -> Tuple[bool, str]:
        """
        Check if a subaccount can open a new position given the entity's
        cross-margin availability.

        Skips the check if the subaccount is in challenge period.

        Args:
            entity_hotkey: The entity's hotkey.
            synthetic_hotkey: The subaccount's synthetic hotkey.
            additional_position_value: The USD value of the proposed new position.

        Returns:
            (allowed: bool, reason: str) - reason is empty if allowed,
            otherwise describes why the order was rejected.
        """
        # TODO: Check if subaccount is in challenge period -> allow
        # TODO: Compute current required collateral (including new position impact)
        # TODO: Compare against cached deposited collateral
        # TODO: Return (False, reason) if required > deposited
        pass

    # ==================== Slashing ====================

    def slash_on_realized_loss(
        self,
        entity_hotkey: str,
        synthetic_hotkey: str,
        realized_loss: float,
    ) -> float:
        """
        Slash entity collateral when a subaccount closes a position with a
        realized loss.

        Slashing formula:
            max_slash = account_balance * MDD%
            remaining_limit = max_slash - cumulative_slashed
            actual_slash = min(abs(realized_loss), remaining_limit)
            cumulative_slashed += actual_slash

        Args:
            entity_hotkey: The entity's hotkey.
            synthetic_hotkey: The subaccount's synthetic hotkey.
            realized_loss: The realized loss in USD (positive number representing loss amount).

        Returns:
            The actual amount slashed in USD.
        """
        # TODO: Look up cumulative slashed for this subaccount
        # TODO: Compute remaining slash limit
        # TODO: Determine actual slash amount
        # TODO: Call ContractClient to execute the slash
        # TODO: Update cumulative slashed tracking
        pass

    def get_cumulative_slashed(self, synthetic_hotkey: str) -> float:
        """
        Get the cumulative amount slashed for a subaccount.

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Cumulative slashed amount in USD.
        """
        pass

    def get_max_slash(self, synthetic_hotkey: str) -> float:
        """
        Get the maximum slashable amount for a subaccount (account_balance * MDD%).

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Maximum slash amount in USD.
        """
        pass
