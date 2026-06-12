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

from time_util.time_util import TimeUtil

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

        # In-memory cache: entity_hotkey -> deposited collateral in theta
        self._collateral_cache: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Slash tracking: synthetic_hotkey -> {cumulative_realized_loss, cumulative_slashed}
        self._slash_tracking: Dict[str, Dict[str, float]] = {}
        self._slash_lock = threading.RLock()

        # File locations
        self._cache_file = ValiBkpUtils.get_entity_collateral_cache_file_location(running_unit_tests)
        self._slash_file = ValiBkpUtils.get_entity_slash_tracking_file_location(running_unit_tests)

        # Intraday drawdown threshold for funded subaccounts (8%).
        # This is the actual elimination threshold applied to funded subaccounts,
        # so it is also the maximum loss that can ever be slashed from a subaccount.
        self.mdd_percent = ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD  # 0.08

        # Load persisted state from disk
        self._collateral_cache = self._load_cache_from_disk()
        self._slash_tracking = self._load_slash_tracking_from_disk()

        bt.logging.info(
            f"[ENTITY_COLLATERAL] Initialized with {len(self._collateral_cache)} cached entities, "
            f"{len(self._slash_tracking)} slash records"
        )

    # ==================== Cache Management ====================

    def refresh_collateral_cache(self) -> str | None:
        """
        Refresh cached collateral balances for all known entities from on-chain contracts.

        Called periodically by the daemon. Reads each entity's collateral balance
        from the ContractClient and writes results to the on-disk cache.

        Returns:
            Number of entities refreshed.
        """
        all_entities = self._entity_client.get_all_entities()
        if not all_entities:
            return None

        refreshed = 0
        for entity_hotkey in all_entities:
            try:
                balance_theta = self._contract_client.get_miner_collateral_balance(entity_hotkey)
                if balance_theta is not None:
                    with self._cache_lock:
                        self._collateral_cache[entity_hotkey] = balance_theta
                    refreshed += 1
            except Exception as e:
                bt.logging.warning(f"[ENTITY_COLLATERAL] Failed to refresh collateral for {entity_hotkey}: {e}")

        self._save_cache_to_disk()
        bt.logging.info(f"[ENTITY_COLLATERAL] Refreshed collateral cache for {refreshed}/{len(all_entities)} entities")
        return self._to_slack_message(self._collateral_cache)

    def get_cached_collateral(self, entity_hotkey: str) -> Optional[float]:
        """
        Get the cached collateral balance for an entity (fast local lookup).

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Deposited collateral in theta, or None if entity not found in cache.
        """
        with self._cache_lock:
            return self._collateral_cache.get(entity_hotkey)

    def decrement_collateral_cache(self, entity_hotkey: str, theta: float) -> None:
        """
        Decrement the cached collateral balance for an entity.

        Called immediately on subaccount creation to reserve the registration fee
        in the cache before the daemon slashes it on-chain, preventing double-spend.

        Args:
            entity_hotkey: The entity's hotkey.
            theta: Amount to decrement in theta.
        """
        with self._cache_lock:
            if entity_hotkey in self._collateral_cache:
                self._collateral_cache[entity_hotkey] = max(0.0, self._collateral_cache[entity_hotkey] - theta)

    def _load_cache_from_disk(self) -> Dict[str, float]:
        """
        Load the entity collateral cache from disk.

        Returns:
            Dict mapping entity_hotkey -> deposited_collateral_theta.
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

    def _load_slash_tracking_from_disk(self) -> Dict[str, Dict[str, float]]:
        """
        Load the slash tracking data from disk.

        Returns:
            Dict mapping synthetic_hotkey -> {cumulative_realized_loss, cumulative_slashed}.
        """
        try:
            data = ValiUtils.get_vali_json_file_dict(self._slash_file)
            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        # New format: {cumulative_realized_loss, cumulative_slashed}
                        result[k] = {
                            "cumulative_realized_loss": float(v.get("cumulative_realized_loss", 0.0)),
                            "cumulative_slashed": float(v.get("cumulative_slashed", 0.0)),
                        }
                    else:
                        # Legacy format: synthetic_hotkey -> cumulative_slashed_usd
                        result[k] = {
                            "cumulative_realized_loss": float(v),
                            "cumulative_slashed": float(v),
                        }
                return result
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
        Compute the total required collateral for an entity in theta.

        Only funded (non-challenge) subaccounts with open positions contribute.

        Formula per qualifying subaccount:
            margin_usd     = min(open_position_value, max_slash_usd - cumulative_slashed_usd)
            required_theta += margin_usd / CPT_RISK

        where max_slash_usd = account_size * SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD.

        Args:
            entity_hotkey: The entity's hotkey.

        Returns:
            Total required collateral in theta.
        """
        entity_data = self._entity_client.get_entity_data(entity_hotkey)
        if not entity_data:
            return 0.0

        subaccounts = entity_data.get("subaccounts", {})
        total_required_theta = 0.0

        for sa_id, sa_info in subaccounts.items():
            if sa_info.get("status") not in ("active", "admin"):
                continue

            synthetic_hotkey = sa_info.get("synthetic_hotkey")
            if not synthetic_hotkey:
                continue

            # Only funded subaccounts require margin; skip challenge, unknown, or other buckets
            bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
            if bucket not in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
                continue

            margin_usd = self.compute_subaccount_margin_requirement(synthetic_hotkey)
            total_required_theta += margin_usd / ValiConfig.ENTITY_COLLATERAL_CPT_RISK

        return total_required_theta

    def compute_subaccount_margin_requirement(self, synthetic_hotkey: str) -> float:
        """
        Compute the margin requirement in USD for a single funded subaccount.

        margin_usd = min(total_open_position_value, max_slash_usd - cumulative_slashed_usd)

        The position value caps the requirement from below: small positions require
        proportionally less collateral. The remaining slash headroom caps it from above:
        once a subaccount has been substantially slashed, less collateral is needed
        because the worst-case future loss is smaller.

        Returns 0 if the subaccount has no open positions or has been fully slashed.

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Margin requirement in USD.
        """
        open_positions = self._position_client.get_positions_for_one_hotkey(
            synthetic_hotkey, only_open_positions=True
        )
        if not open_positions:
            return 0.0

        total_position_value = sum(abs(p.net_value) for p in open_positions)

        max_slash = self.get_max_slash(synthetic_hotkey)
        cumulative_slashed = self.get_cumulative_slashed(synthetic_hotkey)
        remaining_headroom = max(0.0, max_slash - cumulative_slashed)

        return min(total_position_value, remaining_headroom)

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

        Computes the projected required collateral in theta if this order goes through:
            For all other funded subaccounts: margin = min(position_value, max_slash - slashed)
            For the ordering subaccount:      margin = min(position_value + additional, max_slash - slashed)
            projected_required_theta = sum(margin_usd) / CPT_RISK

        Args:
            entity_hotkey: The entity's hotkey.
            synthetic_hotkey: The subaccount's synthetic hotkey.
            additional_position_value: The USD value of the proposed new position.

        Returns:
            (allowed: bool, reason: str) - reason is empty if allowed,
            otherwise describes why the order was rejected.
        """
        # Only funded subaccounts are subject to margin requirements
        bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
        if bucket not in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            return True, ""

        # Current required collateral across all funded subaccounts with open positions.
        # This already includes the ordering subaccount if it has open positions.
        current_required_theta = self.compute_entity_required_collateral(entity_hotkey)

        # Compute the delta from adding this order to the ordering subaccount.
        # current_margin already accounts for it if it has open positions; we compute
        # projected margin and add only the incremental difference.
        open_positions = self._position_client.get_positions_for_one_hotkey(
            synthetic_hotkey, only_open_positions=True
        )
        current_position_value = sum(abs(p.net_value) for p in open_positions)
        projected_position_value = current_position_value + abs(additional_position_value)

        max_slash = self.get_max_slash(synthetic_hotkey)
        cumulative_slashed = self.get_cumulative_slashed(synthetic_hotkey)
        remaining_headroom = max(0.0, max_slash - cumulative_slashed)

        current_margin_usd = min(current_position_value, remaining_headroom)
        projected_margin_usd = min(projected_position_value, remaining_headroom)
        margin_delta_usd = projected_margin_usd - current_margin_usd
        margin_delta_theta = margin_delta_usd / ValiConfig.ENTITY_COLLATERAL_CPT_RISK

        projected_required_theta = current_required_theta + margin_delta_theta

        # Look up deposited collateral from cache (theta)
        deposited_theta = self.get_cached_collateral(entity_hotkey)
        if deposited_theta is None:
            return False, (
                f"Entity {entity_hotkey} has no cached collateral. "
                f"Collateral cache may not have refreshed yet."
            )

        if projected_required_theta > deposited_theta:
            return False, (
                f"Insufficient entity collateral. "
                f"Required: {projected_required_theta:.2f} theta, Deposited: {deposited_theta:.2f} theta. "
                f"Order would add {margin_delta_theta:.2f} theta to requirement."
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

        Cumulative MDD slashing model:
        - Track cumulative_realized_loss per subaccount (total losses over lifetime)
        - max_slash = current_account_size * MDD% (dynamic, tracks current size)
        - target_slash = min(cumulative_realized_loss, max_slash)
        - slash_delta = target_slash - cumulative_slashed (only slash the new delta)
        - If account size increases, max_slash grows, opening new slash headroom

        Args:
            entity_hotkey: The entity's hotkey.
            synthetic_hotkey: The subaccount's synthetic hotkey.
            realized_loss: The realized loss in USD (positive number representing loss amount).

        Returns:
            The actual amount slashed in USD (0.0 if no slash executed).
        """
        if realized_loss <= 0:
            return 0.0

        # Only funded subaccounts are subject to slashing
        bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
        if bucket not in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            return 0.0

        max_slash = self.get_max_slash(synthetic_hotkey)
        if max_slash <= 0:
            bt.logging.warning(
                f"[ENTITY_COLLATERAL] Cannot compute max slash for {synthetic_hotkey}, skipping"
            )
            return 0.0

        with self._slash_lock:
            tracking = self._slash_tracking.get(synthetic_hotkey, {
                "cumulative_realized_loss": 0.0,
                "cumulative_slashed": 0.0,
            })

            cumulative_realized_loss = tracking["cumulative_realized_loss"] + realized_loss
            cumulative_slashed = tracking["cumulative_slashed"]
            tracking["cumulative_realized_loss"] = cumulative_realized_loss
            self._slash_tracking[synthetic_hotkey] = tracking

            # Compute the pending slash amount for cache reservation, but do NOT mark
            # cumulative_slashed yet — process_pending_slashes will mark-then-slash atomically.
            slash_usd = self._get_loss_slash_usd(cumulative_realized_loss, cumulative_slashed, self.get_max_slash(synthetic_hotkey))

            bt.logging.info(
                f"[ENTITY_COLLATERAL] Queued ({slash_usd / ValiConfig.ENTITY_COLLATERAL_CPT_RISK:.4f} theta) "
                f"loss slash for entity {entity_hotkey} subaccount {synthetic_hotkey}. "
                f"cumulative_loss=${cumulative_realized_loss:.2f}, cumulative_slashed=${cumulative_slashed:.2f}"
            )

        # Decrement cache outside slash lock as an optimistic reservation until
        # refresh_collateral_cache resets it from the on-chain value.
        self.decrement_collateral_cache(entity_hotkey, realized_loss / ValiConfig.ENTITY_COLLATERAL_CPT_RISK)

        self._save_slash_tracking_to_disk()
        return slash_usd

    def try_slash_on_elimination(self, hotkey: str) -> float:
        """
        Slash all remaining collateral for a funded subaccount being eliminated.

        Determines the remaining max_slash (max_slash - cumulative_slashed) for the
        subaccount and passes it to slash_on_realized_loss to collect all outstanding
        collateral in one shot. Challenge subaccounts are exempt.

        Args:
            hotkey: The miner hotkey (may or may not be a synthetic subaccount).

        Returns:
            Actual amount slashed in USD, or 0.0 if not applicable or nothing remaining.
        """
        if not is_synthetic_hotkey(hotkey):
            return 0.0

        entity_hotkey, _ = parse_synthetic_hotkey(hotkey)
        if not entity_hotkey:
            return 0.0

        # Only slash funded subaccounts
        bucket = self._challenge_period_client.get_miner_bucket(hotkey)
        if bucket is not None and bucket not in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            return 0.0

        max_slash = self.get_max_slash(hotkey)
        cumulative_slashed = self.get_cumulative_slashed(hotkey)
        remaining = max(0.0, max_slash - cumulative_slashed)

        if remaining <= 0:
            return 0.0

        return self.slash_on_realized_loss(entity_hotkey, hotkey, remaining)

    def get_cumulative_slashed(self, synthetic_hotkey: str) -> float:
        """
        Get the cumulative amount slashed for a subaccount.

        Args:
            synthetic_hotkey: The subaccount's synthetic hotkey.

        Returns:
            Cumulative slashed amount in USD.
        """
        with self._slash_lock:
            tracking = self._slash_tracking.get(synthetic_hotkey)
            if tracking is None:
                return 0.0
            return tracking.get("cumulative_slashed", 0.0)

    def process_pending_slashes(self) -> str | None:
        all_entities = self._entity_client.get_all_entities()
        if not all_entities:
            return None

        refreshed = 0
        total_theta_slashed = 0.0
        slashed_per_entity: Dict[str, float] = {}

        for entity_hotkey, entity_data in all_entities.items():
            try:
                subaccounts = entity_data.get("subaccounts", {}) if isinstance(entity_data, dict) else {}
                synthetic_hotkeys = {s.get("synthetic_hotkey") for s in subaccounts.values() if isinstance(s, dict)}

                pending_loss_usd: Dict[str, float] = {} # synthetic hotkey -> USD
                with self._slash_lock:
                    for synthetic_hotkey, tracking in self._slash_tracking.items():
                        if synthetic_hotkey not in synthetic_hotkeys:
                            continue
                        cumulative_realized_loss = tracking["cumulative_realized_loss"]
                        cumulative_slashed = tracking["cumulative_slashed"]
                        max_slash = self.get_max_slash(synthetic_hotkey)
                        pending_loss_usd[synthetic_hotkey] = self._get_loss_slash_usd(cumulative_realized_loss, cumulative_slashed, max_slash)
                total_pending_loss_theta = sum(pending_loss_usd.values()) / ValiConfig.ENTITY_COLLATERAL_CPT_RISK

                pending_reg_theta: Dict[int, float] = {} # subaccount id -> theta
                for subaccount_id, subaccount_info in subaccounts.items():
                    if not isinstance(subaccount_info, dict):
                        continue
                    reg_fee_theta = subaccount_info.get("reg_fee_theta") or 0.0
                    reg_fee_slashed_ms = subaccount_info.get("reg_fee_slashed_ms")
                    if reg_fee_theta > 0 and reg_fee_slashed_ms is None:
                        pending_reg_theta[int(subaccount_id)] = reg_fee_theta
                total_pending_reg_theta = sum(pending_reg_theta.values())

                theta_slash = total_pending_reg_theta + total_pending_loss_theta
                if theta_slash <= 0:
                    continue

                # Mark before slash so that if the process crashes after the on-chain
                # call succeeds but before tracking is written, we don't double-slash.
                # On failure we revert both marks so the slash is retried next iteration.
                now_ms = TimeUtil.now_in_millis()
                for subaccount_id in pending_reg_theta:
                    self._entity_client.set_reg_fee_time(entity_hotkey, subaccount_id, now_ms)
                with self._slash_lock:
                    for synthetic_hotkey, usd in pending_loss_usd.items():
                        self._slash_tracking[synthetic_hotkey]["cumulative_slashed"] += usd
                self._save_slash_tracking_to_disk()

                success = self._contract_client.slash_miner_collateral(entity_hotkey, theta_slash)
                if success:
                    refreshed += 1
                    total_theta_slashed += theta_slash
                    slashed_per_entity[entity_hotkey] = theta_slash
                    msg = (
                        f"[ENTITY_COLLATERAL] slash_pending_fees: slashed {theta_slash:.4f} theta "
                        f"({total_pending_reg_theta:.4f} reg fees + {total_pending_loss_theta:.4f} loss theta) "
                        f"from entity {entity_hotkey} for subaccounts {list(pending_reg_theta.keys())}"
                    )
                    bt.logging.info(msg)
                else:
                    # Revert both marks so the slash is retried on the next iteration.
                    for subaccount_id in pending_reg_theta:
                        self._entity_client.set_reg_fee_time(entity_hotkey, subaccount_id, None)
                    with self._slash_lock:
                        for synthetic_hotkey, usd in pending_loss_usd.items():
                            self._slash_tracking[synthetic_hotkey]["cumulative_slashed"] -= usd
                    self._save_slash_tracking_to_disk()
                    msg = (
                        f"[ENTITY_COLLATERAL] slash_pending_fees: on-chain slash failed for entity {entity_hotkey} "
                        f"({theta_slash:.4f} theta, subaccounts: {list(pending_reg_theta.keys())}); marks reverted"
                    )
                    bt.logging.error(msg)

                # Refresh collateral cache is called before slash pending
                # need to offset cache until on-chain value is read in next daemon cycle
                # Doesn't matter if slashing succeeded or failed
                if theta_slash > 0:
                    self.decrement_collateral_cache(entity_hotkey, theta_slash)

            except Exception as e:
                bt.logging.warning(f"[ENTITY_COLLATERAL] slash_pending_fees: failed to process entity {entity_hotkey}: {e}")

        bt.logging.info(
            f"[ENTITY_COLLATERAL] slash_pending_fees: {refreshed}/{len(all_entities)} entities slashed, "
            f"{total_theta_slashed:.4f} theta total"
        )
        return self._to_slack_message(slashed_per_entity=slashed_per_entity)

    def _get_loss_slash_usd(self, cumulative_realized_loss, cumulative_slashed, max_slash):
        target_slash = min(cumulative_realized_loss, max_slash)
        slash_delta = max(0, target_slash - cumulative_slashed)
        return slash_delta


    def _to_slack_message(
        self,
        collateral_cache: dict[str, float] | None = None,
        slashed_per_entity: dict[str, float] | None = None,
    ) -> str | None:
        if collateral_cache:
            msg = ":bank: *Entity Collateral Overview*\n"
            msg += "\n".join(f"`{hotkey}`: {collateral:.4f} theta, required: {self.compute_entity_required_collateral(hotkey)}" for hotkey, collateral in collateral_cache.items())
            return msg

        if slashed_per_entity:
            msg = ":money_with_wings: *Entity Collateral Slashed*\n"
            msg += "\n".join(f"`{hotkey}`: {slashed:.4f} theta slashed" for hotkey, slashed in slashed_per_entity.items())
            return msg

        return None


    # ==================== Test Helpers ====================

    def set_test_collateral_cache(self, entity_hotkey: str, collateral_theta: float) -> None:
        """Test-only: Inject a collateral cache value for an entity (in theta)."""
        if not self.running_unit_tests:
            raise RuntimeError("set_test_collateral_cache can only be used in unit test mode")
        with self._cache_lock:
            self._collateral_cache[entity_hotkey] = collateral_theta

    def set_test_slash_tracking(self, synthetic_hotkey: str, cumulative_realized_loss: float, cumulative_slashed: float) -> None:
        """Test-only: Inject slash tracking data for a subaccount."""
        if not self.running_unit_tests:
            raise RuntimeError("set_test_slash_tracking can only be used in unit test mode")
        with self._slash_lock:
            self._slash_tracking[synthetic_hotkey] = {
                "cumulative_realized_loss": cumulative_realized_loss,
                "cumulative_slashed": cumulative_slashed,
            }

    def get_test_slash_tracking(self, synthetic_hotkey: str) -> Optional[Dict[str, float]]:
        """Test-only: Get raw slash tracking data for a subaccount."""
        if not self.running_unit_tests:
            raise RuntimeError("get_test_slash_tracking can only be used in unit test mode")
        with self._slash_lock:
            tracking = self._slash_tracking.get(synthetic_hotkey)
            return dict(tracking) if tracking else None

    def clear_test_state(self) -> None:
        """Test-only: Clear all in-memory state for test isolation."""
        if not self.running_unit_tests:
            raise RuntimeError("clear_test_state can only be used in unit test mode")
        with self._cache_lock:
            self._collateral_cache.clear()
        with self._slash_lock:
            self._slash_tracking.clear()

    def get_test_slash_file_path(self) -> str:
        """Test-only: Get the slash tracking file path for direct disk testing."""
        if not self.running_unit_tests:
            raise RuntimeError("get_test_slash_file_path can only be used in unit test mode")
        return self._slash_file

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
