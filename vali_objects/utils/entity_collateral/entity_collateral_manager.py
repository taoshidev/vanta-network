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

import json
import threading
import bittensor as bt
from typing import Dict, Optional, Tuple

from shared_objects.cache_controller import CacheController
from entity_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


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
        super().__init__(running_unit_tests, connection_mode)
        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode

        # RPC clients (created internally, forward compatibility pattern)
        from entity_management.entity_client import EntityClient
        from vali_objects.contract.contract_client import ContractClient
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient

        self._entity_client = EntityClient(connection_mode=connection_mode, connect_immediately=False)
        self._contract_client = ContractClient(connection_mode=connection_mode, connect_immediately=False,
                                               running_unit_tests=running_unit_tests)
        self._position_client = PositionManagerClient(connection_mode=connection_mode, connect_immediately=False,
                                                      running_unit_tests=running_unit_tests)
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)
        self._challenge_period_client = ChallengePeriodClient(connection_mode=connection_mode,
                                                              running_unit_tests=running_unit_tests)

        # In-memory cache: entity_hotkey -> deposited collateral in USD
        self._collateral_cache: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Slash tracking: synthetic_hotkey -> cumulative slashed USD
        self._slash_tracking: Dict[str, float] = {}
        self._slash_lock = threading.RLock()

        # File locations
        self._cache_file = ValiBkpUtils.get_entity_collateral_cache_file_location(running_unit_tests)
        self._slash_file = ValiBkpUtils.get_entity_slash_tracking_file_location(running_unit_tests)

        # MDD percentage: MAX_TOTAL_DRAWDOWN is 0.9, meaning 10% max drawdown
        self.mdd_percent = 1.0 - ValiConfig.MAX_TOTAL_DRAWDOWN  # 0.10

        # Load persisted state from disk
        self._collateral_cache = self._load_cache_from_disk()
        self._slash_tracking = self._load_slash_tracking_from_disk()

        bt.logging.info(
            f"[ENTITY_COLLATERAL] Initialized with {len(self._collateral_cache)} cached entities, "
            f"{len(self._slash_tracking)} slash records"
        )

    # ==================== Cache Management ====================

    def refresh_collateral_cache(self) -> int:
        """
        Refresh cached collateral balances for all known entities from on-chain contracts.

        Called periodically by the daemon. Reads each entity's collateral balance
        from the ContractClient and writes results to the on-disk cache.

        Returns:
            Number of entities refreshed.
        """
        all_entities = self._entity_client.get_all_entities()
        if not all_entities:
            return 0

        refreshed = 0
        for entity_hotkey in all_entities:
            try:
                balance_theta = self._contract_client.get_miner_collateral_balance(entity_hotkey)
                if balance_theta is not None:
                    collateral_usd = balance_theta * ValiConfig.ENTITY_COST_PER_THETA
                    with self._cache_lock:
                        self._collateral_cache[entity_hotkey] = collateral_usd
                    refreshed += 1
            except Exception as e:
                bt.logging.warning(f"[ENTITY_COLLATERAL] Failed to refresh collateral for {entity_hotkey}: {e}")

        self._save_cache_to_disk()
        bt.logging.info(f"[ENTITY_COLLATERAL] Refreshed collateral cache for {refreshed}/{len(all_entities)} entities")
        return refreshed

    def get_cached_collateral(self, entity_hotkey: str) -> Optional[float]:
        """
        Get the cached collateral balance for an entity (fast local lookup).

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Deposited collateral in USD, or None if entity not found in cache.
        """
        with self._cache_lock:
            return self._collateral_cache.get(entity_hotkey)

    def _load_cache_from_disk(self) -> Dict[str, float]:
        """
        Load the entity collateral cache from disk.

        Returns:
            Dict mapping entity_hotkey -> deposited_collateral_usd.
        """
        try:
            data = ValiUtils.get_vali_json_file_dict(self._cache_file)
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items()}
        except Exception as e:
            bt.logging.warning(f"[ENTITY_COLLATERAL] Failed to load cache from disk: {e}")
        return {}

    def _save_cache_to_disk(self) -> None:
        """
        Persist the current in-memory collateral cache to disk.
        """
        with self._cache_lock:
            data = dict(self._collateral_cache)
        try:
            ValiBkpUtils.write_file(self._cache_file, data)
        except Exception as e:
            bt.logging.error(f"[ENTITY_COLLATERAL] Failed to save cache to disk: {e}")

    def _load_slash_tracking_from_disk(self) -> Dict[str, float]:
        """
        Load the slash tracking data from disk.

        Returns:
            Dict mapping synthetic_hotkey -> cumulative_slashed_usd.
        """
        try:
            data = ValiUtils.get_vali_json_file_dict(self._slash_file)
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items()}
        except Exception as e:
            bt.logging.warning(f"[ENTITY_COLLATERAL] Failed to load slash tracking from disk: {e}")
        return {}

    def _save_slash_tracking_to_disk(self) -> None:
        """
        Persist the slash tracking data to disk.
        """
        with self._slash_lock:
            data = dict(self._slash_tracking)
        try:
            ValiBkpUtils.write_file(self._slash_file, data)
        except Exception as e:
            bt.logging.error(f"[ENTITY_COLLATERAL] Failed to save slash tracking to disk: {e}")

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
        entity_data = self._entity_client.get_entity_data(entity_hotkey)
        if not entity_data:
            return 0.0

        subaccounts = entity_data.get("subaccounts", {})
        total_required = 0.0

        for sa_id, sa_info in subaccounts.items():
            if sa_info.get("status") not in ("active", "admin"):
                continue

            synthetic_hotkey = sa_info.get("synthetic_hotkey")
            if not synthetic_hotkey:
                continue

            # Skip challenge period subaccounts
            bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
            if bucket == MinerBucket.SUBACCOUNT_CHALLENGE:
                continue

            account_balance = self._miner_account_client.get_balance(synthetic_hotkey)
            if not account_balance or account_balance <= 0:
                continue

            exposure = self.compute_subaccount_risk_exposure(synthetic_hotkey, account_balance)
            total_required += exposure

        return total_required

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
        open_positions = self._position_client.get_positions_for_one_hotkey(
            synthetic_hotkey, only_open_positions=True
        )

        total_position_value = 0.0
        for position in open_positions:
            total_position_value += abs(position.net_value)

        max_exposure = account_balance * self.mdd_percent
        return min(total_position_value, max_exposure)

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
        # Challenge period subaccounts are exempt from margin requirements
        bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
        if bucket == MinerBucket.SUBACCOUNT_CHALLENGE:
            return True, ""

        # Compute current required collateral across all entity subaccounts
        current_required = self.compute_entity_required_collateral(entity_hotkey)

        # Add the impact of the proposed new position.
        # The new position increases risk exposure by at most the additional value,
        # but capped by the subaccount's MDD limit.
        account_balance = self._miner_account_client.get_balance(synthetic_hotkey) or 0.0
        max_subaccount_exposure = account_balance * self.mdd_percent

        # Current exposure for this subaccount (already included in current_required)
        current_subaccount_exposure = self.compute_subaccount_risk_exposure(synthetic_hotkey, account_balance)

        # Projected exposure after adding the new position
        open_positions = self._position_client.get_positions_for_one_hotkey(
            synthetic_hotkey, only_open_positions=True
        )
        current_position_value = sum(abs(p.net_value) for p in open_positions)
        projected_position_value = current_position_value + abs(additional_position_value)
        projected_subaccount_exposure = min(projected_position_value, max_subaccount_exposure)

        # Delta is the increase in required collateral from this order
        exposure_delta = projected_subaccount_exposure - current_subaccount_exposure
        projected_required = current_required + exposure_delta

        # Look up deposited collateral from cache
        deposited = self.get_cached_collateral(entity_hotkey)
        if deposited is None:
            return False, (
                f"Entity {entity_hotkey} has no cached collateral. "
                f"Collateral cache may not have refreshed yet."
            )

        if projected_required > deposited:
            return False, (
                f"Insufficient entity cross-margin. "
                f"Required: ${projected_required:.2f}, Deposited: ${deposited:.2f}. "
                f"Order would add ${exposure_delta:.2f} to margin requirement."
            )

        return True, ""

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
            The actual amount slashed in USD (0.0 if no slash executed).
        """
        if realized_loss <= 0:
            return 0.0

        max_slash = self.get_max_slash(synthetic_hotkey)
        if max_slash <= 0:
            bt.logging.warning(
                f"[ENTITY_COLLATERAL] Cannot compute max slash for {synthetic_hotkey}, skipping"
            )
            return 0.0

        with self._slash_lock:
            cumulative = self._slash_tracking.get(synthetic_hotkey, 0.0)
            remaining_limit = max_slash - cumulative
            if remaining_limit <= 0:
                bt.logging.info(
                    f"[ENTITY_COLLATERAL] Slash limit reached for {synthetic_hotkey} "
                    f"(cumulative={cumulative:.2f}, max={max_slash:.2f})"
                )
                return 0.0

            actual_slash = min(realized_loss, remaining_limit)
            self._slash_tracking[synthetic_hotkey] = cumulative + actual_slash

        # Convert USD to theta for the on-chain slash
        slash_theta = actual_slash / ValiConfig.ENTITY_COST_PER_THETA
        try:
            success = self._contract_client.slash_miner_collateral(entity_hotkey, slash_theta)
            if not success:
                bt.logging.error(
                    f"[ENTITY_COLLATERAL] On-chain slash failed for entity {entity_hotkey}, "
                    f"amount={slash_theta:.4f} theta (${actual_slash:.2f})"
                )
                # Revert tracking on failure
                with self._slash_lock:
                    self._slash_tracking[synthetic_hotkey] -= actual_slash
                return 0.0
        except Exception as e:
            bt.logging.error(f"[ENTITY_COLLATERAL] Slash exception for {entity_hotkey}: {e}")
            with self._slash_lock:
                self._slash_tracking[synthetic_hotkey] -= actual_slash
            return 0.0

        # Persist and update collateral cache after successful slash
        self._save_slash_tracking_to_disk()
        with self._cache_lock:
            if entity_hotkey in self._collateral_cache:
                self._collateral_cache[entity_hotkey] -= actual_slash

        bt.logging.info(
            f"[ENTITY_COLLATERAL] Slashed ${actual_slash:.2f} ({slash_theta:.4f} theta) "
            f"from entity {entity_hotkey} for subaccount {synthetic_hotkey}. "
            f"Cumulative: ${cumulative + actual_slash:.2f} / ${max_slash:.2f}"
        )
        return actual_slash

    def get_cumulative_slashed(self, synthetic_hotkey: str) -> float:
        """
        Get the cumulative amount slashed for a subaccount.

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Cumulative slashed amount in USD.
        """
        with self._slash_lock:
            return self._slash_tracking.get(synthetic_hotkey, 0.0)

    def get_max_slash(self, synthetic_hotkey: str) -> float:
        """
        Get the maximum slashable amount for a subaccount (account_balance * MDD%).

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Maximum slash amount in USD.
        """
        account_size = self._miner_account_client.get_miner_account_size(synthetic_hotkey)
        if not account_size or account_size <= 0:
            return 0.0
        return account_size * self.mdd_percent
