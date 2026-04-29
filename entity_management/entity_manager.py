# developer: jbonilla
# Copyright � 2024 Taoshi Inc
"""
EntityManager - Core business logic for entity miner management.

This manager handles all business logic for entity operations including:
- Entity registration and tracking
- Subaccount creation with monotonic IDs
- Subaccount status management (active/eliminated)
- Collateral verification (placeholder)
- Slot allowance checking
- Thread-safe operations with proper locking

Pattern follows ChallengePeriodManager:
- Manager holds all business logic
- Server wraps this and exposes via RPC
- Local dicts (NOT IPC) for performance
- Disk persistence via JSON
"""
import re
import uuid
import time
import threading
import asyncio
import bittensor as bt
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from pydantic import BaseModel, Field

import template.protocol
from entity_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey
from vali_objects.miner_account import MinerAccountClient
from vali_objects.position_management.position_utils.position_utils import PositionUtils
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from datetime import datetime, timezone
from vali_objects.vali_config import ValiConfig, RPCConnectionMode, TradePairCategory
from shared_objects.cache_controller import CacheController
from vali_objects.validator_broadcast_base import ValidatorBroadcastBase
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.statistics.miner_statistics_client import MinerStatisticsClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.contract.contract_client import ContractClient
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from vali_objects.enums.miner_bucket_enum import MinerBucket
from time_util.time_util import MS_IN_24_HOURS, TimeUtil
from vanta_api.websocket_notifier import WebSocketNotifierClient


class SubaccountInfo(BaseModel):
    """Data structure for a single subaccount."""
    subaccount_id: int = Field(description="Monotonically increasing ID")
    subaccount_uuid: str = Field(description="Unique UUID for this subaccount")
    synthetic_hotkey: str = Field(description="Synthetic hotkey: {entity_hotkey}_{subaccount_id}")
    status: str = Field(default="active", description="Status: active, eliminated, or unknown")
    created_at_ms: int = Field(description="Timestamp when subaccount was created")
    eliminated_at_ms: Optional[int] = Field(default=None, description="Timestamp when subaccount was eliminated")
    account_size: float = Field(description="Account size in USD (immutable once set)")
    asset_class: str = Field(description="Asset class selection (immutable once set)")
    hl_address: Optional[str] = Field(default=None, description="Hyperliquid address for HL tracking subaccounts")
    payout_address: Optional[str] = Field(default=None, description="EVM address (0x + 40 hex) for USDC payouts")

    # Note: Challenge period tracking has been migrated to ChallengePeriodManager
    # Synthetic hotkeys are added to challenge period bucket and evaluated via inspect()


class EntityData(BaseModel):
    """Data structure for an entity."""
    entity_hotkey: str = Field(description="The VANTA_ENTITY_HOTKEY")
    subaccounts: Dict[int, SubaccountInfo] = Field(default_factory=dict, description="Map subaccount_id -> SubaccountInfo")
    next_subaccount_id: int = Field(default=0, description="Next subaccount ID to assign (monotonic)")
    registered_at_ms: int = Field(description="Timestamp when entity was registered")
    endpoint_url: Optional[str] = Field(default=None, description="Public-facing endpoint URL for this entity miner")

    class Config:
        arbitrary_types_allowed = True

    def get_active_subaccounts(self) -> List[SubaccountInfo]:
        """Get all active subaccounts."""
        return [sa for sa in self.subaccounts.values() if sa.status == "active"]

    def get_eliminated_subaccounts(self) -> List[SubaccountInfo]:
        """Get all eliminated subaccounts."""
        return [sa for sa in self.subaccounts.values() if sa.status == "eliminated"]

    def get_synthetic_hotkey(self, subaccount_id: int) -> Optional[str]:
        """Get synthetic hotkey for a subaccount ID."""
        sa = self.subaccounts.get(subaccount_id)
        return sa.synthetic_hotkey if sa else None


class EntityManager(ValidatorBroadcastBase):
    """
    Entity Manager - Contains all business logic for entity miner management.

    This manager is wrapped by EntityServer which exposes methods via RPC.
    All heavy logic resides here - server delegates to this manager.

    Pattern:
    - Server holds a `self._manager` instance
    - Server delegates all RPC methods to manager methods
    - Manager creates its own clients internally (forward compatibility)
    - Local dicts (NOT IPC) for fast access
    - Thread-safe operations with locks
    """

    def __init__(
        self,
        *,
        is_backtesting=False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        config=None
    ):
        """
        Initialize EntityManager.

        Args:
            is_backtesting: Whether running in backtesting mode
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
            config: Validator config (for netuid, wallet) - optional, used for broadcasting
        """
        self.is_backtesting = is_backtesting
        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode

        # Determine is_testnet before calling ValidatorBroadcastBase.__init__
        # This prevents wallet creation blocking in ValidatorBroadcastBase
        is_testnet = (config.netuid in (116, 171)) if (config and hasattr(config, 'netuid')) else False

        # ValidatorBroadcastBase derives is_mothership internally
        # CRITICAL: Pass running_unit_tests AND is_testnet to prevent blocking wallet creation
        super().__init__(
            running_unit_tests=running_unit_tests,
            is_testnet=is_testnet,
            connection_mode=connection_mode,
            config=config
        )

        # Local dicts (NOT IPC managerized) - much faster!
        self.entities: Dict[str, EntityData] = {}

        # Reverse index: UUID -> synthetic hotkey for O(1) lookups
        self._uuid_to_hotkey: Dict[str, str] = {}

        # Reverse index: HL address -> synthetic hotkey for O(1) lookups
        self._hl_address_to_synthetic: Dict[str, str] = {}

        # Per-entity locking strategy for better concurrency
        # Master lock protects the entities dict structure and the entity_locks dict
        # Use RLock (reentrant) to allow methods to call each other within locked contexts
        self._entities_lock = threading.RLock()

        # Per-entity locks: only serialize operations on the same entity
        # Operations on different entities can run concurrently
        self._entity_locks: Dict[str, threading.RLock] = {}

        # Store testnet flag (redundant with ValidatorBroadcastBase but kept for clarity)
        self.is_testnet = is_testnet

        # Create DebtLedgerClient with connect_immediately=False to defer connection
        self._debt_ledger_client = DebtLedgerClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create EliminationClient with connect_immediately=False to defer connection
        self._elimination_client = EliminationClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create ChallengePeriodClient with connect_immediately=False to defer connection
        self._challenge_period_client = ChallengePeriodClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests
        )

        # Create MinerStatisticsClient with connect_immediately=False to defer connection
        self._statistics_client = MinerStatisticsClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create PositionManagerClient with connect_immediately=False to defer connection
        self._position_client = PositionManagerClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create ContractClient for collateral verification and slashing
        self._contract_client = ContractClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create MinerAccountClient for setting subaccount miner account size
        self._miner_account_client = MinerAccountClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        # Create AssetSelectionClient for asset class selection
        self._asset_selection_client = AssetSelectionClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create LimitOrderClient for unfilled limit orders
        self._limit_order_client = LimitOrderClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        # Create WebSocketNotifierClient for real-time dashboard updates
        self._websocket_client = WebSocketNotifierClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        self.ENTITY_FILE = ValiBkpUtils.get_entity_file_location(running_unit_tests=running_unit_tests)

        # Load initial entities from disk
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.ENTITY_FILE)
            self.entities = self.parse_checkpoint_dict(disk_data)

            # Build UUID -> hotkey and HL address -> synthetic reverse indices
            for entity_data in self.entities.values():
                for subaccount in entity_data.subaccounts.values():
                    self._uuid_to_hotkey[subaccount.subaccount_uuid] = subaccount.synthetic_hotkey
                    if subaccount.hl_address and subaccount.status in ('active', 'admin', 'pending'):
                        normalized_hl = self._normalize_hl_address(subaccount.hl_address)
                        if normalized_hl:
                            self._hl_address_to_synthetic[normalized_hl] = subaccount.synthetic_hotkey

            # Recreate locks for all loaded entities
            for entity_hotkey in self.entities.keys():
                self._entity_locks[entity_hotkey] = threading.RLock()
            bt.logging.info(f"[ENTITY_MANAGER] Loaded {len(self.entities)} entities from disk with per-entity locks")

        bt.logging.info("[ENTITY_MANAGER] EntityManager initialized")

    # ==================== Lock Management ====================

    @staticmethod
    def _normalize_hl_address(hl_address: Optional[str]) -> Optional[str]:
        """Canonical key form for HL address lookups/indexing."""
        if not isinstance(hl_address, str):
            return None
        return hl_address.lower()

    def _get_entity_lock(self, entity_hotkey: str) -> threading.RLock:
        """
        Get or create a lock for a specific entity.

        This method is thread-safe and ensures each entity has its own lock.
        The master lock protects the entity_locks dict.

        Args:
            entity_hotkey: The entity hotkey

        Returns:
            RLock for this entity
        """
        with self._entities_lock:
            if entity_hotkey not in self._entity_locks:
                self._entity_locks[entity_hotkey] = threading.RLock()
            return self._entity_locks[entity_hotkey]

    # ==================== Core Business Logic ====================

    def register_entity(
        self,
        entity_hotkey: str
    ) -> Tuple[bool, str]:
        """
        Register a new entity.

        Verifies entity has sufficient collateral balance (ENTITY_REGISTRATION_FEE)
        and slashes this amount as a registration fee.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            (success: bool, message: str)
        """
        # Use master lock: adding new entity to dict
        with self._entities_lock:
            if entity_hotkey in self.entities:
                return False, f"Entity {entity_hotkey} already registered"

            # Check if max entities limit is reached
            if len(self.entities) >= ValiConfig.MAX_REGISTERED_ENTITIES:
                return False, f"Maximum number of entities ({ValiConfig.MAX_REGISTERED_ENTITIES}) already registered. No new registrations allowed."

            positions = self._position_client.get_positions_for_one_hotkey(entity_hotkey)
            if positions and len(positions) > 0:
                return False, f"Entity {entity_hotkey} is already used as a miner. Choose a new hotkey."

            if not self.running_unit_tests:
                # Verify collateral balance
                try:
                    current_balance = self._contract_client.get_miner_collateral_balance(entity_hotkey)
                    if current_balance is None:
                        bt.logging.warning(f"[ENTITY_MANAGER] Unable to verify collateral for {entity_hotkey} - balance check returned None")
                        return False, "Unable to verify collateral balance"

                    if current_balance < ValiConfig.ENTITY_REGISTRATION_FEE:
                        bt.logging.warning(
                            f"[ENTITY_MANAGER] Insufficient collateral for entity {entity_hotkey}: "
                            f"has {current_balance} theta, needs {ValiConfig.ENTITY_REGISTRATION_FEE} theta"
                        )
                        return False, f"Insufficient collateral: has {current_balance} theta, needs {ValiConfig.ENTITY_REGISTRATION_FEE} theta"

                    # Slash registration fee
                    slash_success = self._contract_client.slash_miner_collateral(entity_hotkey, ValiConfig.ENTITY_REGISTRATION_FEE)
                    if not slash_success:
                        bt.logging.error(f"[ENTITY_MANAGER] Failed to slash registration fee for {entity_hotkey}")
                        return False, "Failed to slash registration fee"

                    bt.logging.info(
                        f"[ENTITY_MANAGER] Slashed {ValiConfig.ENTITY_REGISTRATION_FEE} theta registration fee for entity {entity_hotkey}"
                    )

                except Exception as e:
                    bt.logging.error(f"[ENTITY_MANAGER] Error verifying/slashing collateral for {entity_hotkey}: {e}")
                    return False, f"Error verifying collateral: {str(e)}"

            # Registration fee slashed - proceed with registration
            entity_data = EntityData(
                entity_hotkey=entity_hotkey,
                subaccounts={},
                next_subaccount_id=0,
                registered_at_ms=TimeUtil.now_in_millis()
            )

            self.entities[entity_hotkey] = entity_data
            # Create lock for this entity
            self._entity_locks[entity_hotkey] = threading.RLock()
            self._write_entities_from_memory_to_disk()

            # Add entity hotkey to ENTITY bucket (4x dust weight)
            self._challenge_period_client.set_miner_bucket(
                entity_hotkey,
                MinerBucket.ENTITY,
                TimeUtil.now_in_millis()
            )

            bt.logging.info(
                f"[ENTITY_MANAGER] Registered entity {entity_hotkey}, "
                f"slashed {ValiConfig.ENTITY_REGISTRATION_FEE} theta"
            )
            return True, f"Entity registered successfully - {ValiConfig.ENTITY_REGISTRATION_FEE} theta registration fee slashed"

    def create_subaccount(
        self,
        entity_hotkey: str,
        account_size: float,
        asset_class: str,
        admin: bool = False
    ) -> Tuple[bool, Optional[SubaccountInfo], str]:
        """
        Create a new subaccount for an entity.

        Verifies entity has sufficient collateral for the requested account size
        and slashes the required amount as a subaccount registration fee.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            account_size: Account size in USD (immutable once set, max 100k)
            asset_class: Asset class selection (immutable once set)
            admin: If True, skip collateral slashing and set status to "admin".
                   Admin subaccounts are excluded from entity aggregation and payouts.

        Returns:
            (success: bool, subaccount_info: Optional[SubaccountInfo], message: str)
        """
        import time
        t_start = time.time()
        timings = {}

        # Validate account size (must be <= MAX_SUBACCOUNT_ACCOUNT_SIZE)
        if account_size > ValiConfig.MAX_SUBACCOUNT_ACCOUNT_SIZE:
            return False, None, (
                f"Account size ${account_size} exceeds maximum allowed "
                f"${ValiConfig.MAX_SUBACCOUNT_ACCOUNT_SIZE}"
            )

        # Use per-entity lock: only operates on single entity
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, None, f"Entity {entity_hotkey} not registered"

            # Check slot allowance
            active_count = len(entity_data.get_active_subaccounts())
            if active_count >= ValiConfig.ENTITY_MAX_SUBACCOUNTS:
                return False, None, f"Entity {entity_hotkey} has reached maximum subaccounts ({ValiConfig.ENTITY_MAX_SUBACCOUNTS})"

            # Calculate required collateral: account_size / ENTITY_COST_PER_THETA (lower rate for <=10k accounts)
            cpt = ValiConfig.ENTITY_COST_PER_THETA_LOW if account_size <= ValiConfig.ENTITY_COST_PER_THETA_LOW_THRESHOLD else ValiConfig.ENTITY_COST_PER_THETA
            required_theta = account_size / cpt if not admin else 0
            current_balance = None  # dummy init for collateral tracking on miner

            # Verify collateral balance
            try:
                if not self.running_unit_tests:
                    t0 = time.time()
                    current_balance = self._contract_client.get_miner_collateral_balance(entity_hotkey)
                    timings['get_collateral_balance'] = int((time.time() - t0) * 1000)

                    if current_balance is None:
                        bt.logging.warning(f"[ENTITY_MANAGER] Unable to verify collateral for {entity_hotkey} - balance check returned None")
                        return False, None, "Unable to verify collateral balance"

                    if current_balance < required_theta:
                        bt.logging.warning(
                            f"[ENTITY_MANAGER] Insufficient collateral for subaccount creation: "
                            f"entity {entity_hotkey} has {current_balance} theta, needs {required_theta} theta "
                            f"to create new subaccount with ${account_size} account size"
                        )
                        return False, None, (
                            f"Insufficient collateral: has {current_balance} theta, needs {required_theta} theta "
                            f"to create new subaccount with ${account_size} account size"
                        )

                # Generate monotonic ID
                subaccount_id = entity_data.next_subaccount_id
                entity_data.next_subaccount_id += 1

                # Generate UUID and synthetic hotkey
                subaccount_uuid = str(uuid.uuid4())
                synthetic_hotkey = f"{entity_hotkey}_{subaccount_id}"

                # Process asset selection for synthetic hotkey
                t0 = time.time()
                asset_selection_result = self._asset_selection_client.process_asset_selection_request(
                    asset_selection=asset_class,
                    miner=synthetic_hotkey
                )
                timings['asset_selection_rpc'] = int((time.time() - t0) * 1000)

                if not asset_selection_result.get('successfully_processed', False):
                    bt.logging.warning(
                        f"[ENTITY_MANAGER] Failed to process asset selection for {synthetic_hotkey}: "
                        f"{asset_selection_result.get('error_message', 'Unknown error')}"
                    )
                    entity_data.next_subaccount_id -= 1
                    self._asset_selection_client.delete_asset_selection(synthetic_hotkey)
                    return False, None, f"Failed to set asset selection {asset_class}"
                bt.logging.info(
                    f"[ENTITY_MANAGER] Asset selection '{asset_class}' set for {synthetic_hotkey}"
                )

                # Set account size for synthetic hotkey with explicit account_size parameter
                # This records the account size in the contract manager's miner_account_sizes
                t0 = time.time()
                set_size_success = self._miner_account_client.set_miner_account_size(
                    synthetic_hotkey,
                    collateral_balance_theta=account_size / cpt,
                    timestamp_ms=TimeUtil.now_in_millis(),
                    account_size=account_size,
                    bucket=MinerBucket.SUBACCOUNT_CHALLENGE
                )
                timings['set_account_size'] = int((time.time() - t0) * 1000)

                if not set_size_success:
                    bt.logging.warning(
                        f"[ENTITY_MANAGER] Failed to set account size for {synthetic_hotkey}"
                    )
                    entity_data.next_subaccount_id -= 1
                    self._asset_selection_client.delete_asset_selection(synthetic_hotkey)
                    self._miner_account_client.delete_miner_account_size(synthetic_hotkey)
                    return False, None, "Failed to set account size for subaccount creation"
                bt.logging.info(
                    f"[ENTITY_MANAGER] Set account size {account_size} for {synthetic_hotkey}"
                )

            except Exception as e:
                bt.logging.error(f"[ENTITY_MANAGER] Error creating subaccount: {e}")
                # Rollback subaccount ID increment and clean up asset selection/account size
                entity_data.next_subaccount_id -= 1
                self._asset_selection_client.delete_asset_selection(synthetic_hotkey)
                self._miner_account_client.delete_miner_account_size(synthetic_hotkey)
                return False, None, f"Error creating subaccount: {str(e)}"

            # Create subaccount info
            now_ms = TimeUtil.now_in_millis()
            initial_status = "admin" if admin else "pending"
            subaccount_info = SubaccountInfo(
                subaccount_id=subaccount_id,
                subaccount_uuid=subaccount_uuid,
                synthetic_hotkey=synthetic_hotkey,
                status=initial_status,
                created_at_ms=now_ms,
                account_size=account_size,
                asset_class=asset_class
            )

            entity_data.subaccounts[subaccount_id] = subaccount_info

            # Update reverse index
            with self._entities_lock:
                self._uuid_to_hotkey[subaccount_uuid] = synthetic_hotkey

            t0 = time.time()
            self._write_entities_from_memory_to_disk()
            timings['write_to_disk'] = int((time.time() - t0) * 1000)

            # Start background slashing thread (not in unit tests, not for admin)
            if not self.running_unit_tests and not admin:
                thread = threading.Thread(
                    target=self._complete_subaccount_slashing,
                    args=(subaccount_id, entity_hotkey, synthetic_hotkey, required_theta),
                    daemon=True
                )
                thread.start()
            else:
                # In tests or admin: mark as active immediately (skip slashing)
                # Admin subaccounts keep "admin" status, test subaccounts become "active"
                if not admin:
                    subaccount_info.status = "active"
                self._write_entities_from_memory_to_disk()

                # Notify WebSocket server so connected entity clients auto-subscribe
                try:
                    self._websocket_client.notify_new_subaccount(entity_hotkey, synthetic_hotkey)
                except Exception as notify_err:
                    bt.logging.debug(f"[ENTITY_MANAGER] New subaccount WS notification failed: {notify_err}")

            total_ms = int((time.time() - t_start) * 1000)

            bt.logging.info(
                f"[ENTITY_MANAGER] Created subaccount {subaccount_id} for entity {entity_hotkey}: "
                f"{synthetic_hotkey}, account_size=${account_size}, asset_class={asset_class}, "
                f"status={initial_status}, slashing {required_theta} theta in background ({total_ms} ms) | timings: {timings}"
            )
            remaining_theta = (current_balance - required_theta) if current_balance else 0.0
            return True, subaccount_info, f"[{initial_status}] Subaccount creation - slashing {required_theta} theta, {remaining_theta:.2f} theta remaining"

    def _complete_subaccount_slashing(
        self,
        subaccount_id: int,
        entity_hotkey: str,
        synthetic_hotkey: str,
        required_theta: float
    ) -> None:
        """Background thread to complete collateral slashing."""
        try:
            slash_success = self._contract_client.slash_miner_collateral(entity_hotkey, required_theta)

            entity_lock = self._get_entity_lock(entity_hotkey)
            with entity_lock:
                entity_data = self.entities.get(entity_hotkey)
                if not entity_data:
                    return

                subaccount = entity_data.subaccounts.get(subaccount_id)
                if not subaccount:
                    return

                if slash_success:
                    subaccount.status = "active"
                    bt.logging.info(f"[ENTITY_MANAGER] Slashing complete for {synthetic_hotkey}")
                else:
                    subaccount.status = "failed"
                    bt.logging.error(f"[ENTITY_MANAGER] Slashing failed for {synthetic_hotkey}")
                self._write_entities_from_memory_to_disk()

            # Broadcast status update to other validators after slashing completes
            if slash_success:
                self.broadcast_subaccount_registration(
                    entity_hotkey=entity_hotkey,
                    subaccount_id=subaccount_id,
                    subaccount_uuid=subaccount.subaccount_uuid,
                    synthetic_hotkey=synthetic_hotkey,
                    account_size=subaccount.account_size,
                    asset_class=subaccount.asset_class,
                    status="active",
                    hl_address=subaccount.hl_address,
                    payout_address=subaccount.payout_address
                )

            # Broadcast dashboard update to WebSocket subscribers after slashing completes
            self.broadcast_subaccount_dashboard(synthetic_hotkey)

            # Notify WebSocket server so connected entity clients auto-subscribe
            if slash_success:
                try:
                    self._websocket_client.notify_new_subaccount(entity_hotkey, synthetic_hotkey)
                except Exception as notify_err:
                    bt.logging.debug(f"[ENTITY_MANAGER] New subaccount WS notification failed: {notify_err}")

        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Slashing error for {synthetic_hotkey}: {e}")
            # Mark as failed
            entity_lock = self._get_entity_lock(entity_hotkey)
            with entity_lock:
                entity_data = self.entities.get(entity_hotkey)
                if entity_data:
                    subaccount = entity_data.subaccounts.get(subaccount_id)
                    if subaccount:
                        subaccount.status = "failed"
                        self._write_entities_from_memory_to_disk()

            # Broadcast dashboard update even on failure so clients see the status change
            self.broadcast_subaccount_dashboard(synthetic_hotkey)

    def _get_hl_max_addresses(self) -> int:
        """
        Return the max number of HL addresses we can track.

        If proxy ports are configured in secrets.json, returns PER_IP * num_ports.
        Otherwise returns PER_IP (10).
        """
        try:
            secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)
        except Exception:
            return ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP

        proxy_url = secrets.get(ValiConfig.HL_PROXY_SECRET_KEY)
        ports_str = secrets.get(ValiConfig.HL_PROXY_PORTS_SECRET_KEY)

        if not proxy_url or not ports_str:
            return ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP

        # Parse ports to count them
        from entity_management.hyperliquid_tracker import HyperliquidTracker
        ports = HyperliquidTracker._parse_ports(ports_str)
        num_ports = min(len(ports), ValiConfig.HL_MAX_PROXY_SHARDS)
        return num_ports * ValiConfig.HL_MAX_TRACKED_ADDRESSES_PER_IP

    def create_hl_subaccount(
        self,
        entity_hotkey: str,
        account_size: float,
        hl_address: str,
        asset_class: str = "crypto",
        admin: bool = False,
        payout_address: Optional[str] = None
    ) -> Tuple[bool, Optional[SubaccountInfo], str]:
        """
        Create a new subaccount linked to a Hyperliquid address.

        Validates the HL address format, checks for duplicates and the max tracked
        addresses limit, then delegates to create_subaccount() for standard validation.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            account_size: Account size in USD
            hl_address: Hyperliquid address (0x-prefixed, 40 hex chars)
            asset_class: Asset class selection (default: "crypto")
            admin: If True, skip collateral slashing
            payout_address: Optional EVM address for payouts (0x-prefixed, 40 hex chars)

        Returns:
            (success: bool, subaccount_info: Optional[SubaccountInfo], message: str)
        """
        # Validate HL address format
        if not re.match(ValiConfig.HL_ADDRESS_REGEX, hl_address):
            return False, None, f"Invalid Hyperliquid address format: {hl_address}. Must be 0x followed by 40 hex characters."
        normalized_hl_address = self._normalize_hl_address(hl_address)

        # Validate payout_address format if provided
        if payout_address is not None:
            if not isinstance(payout_address, str) or not re.match(ValiConfig.HL_ADDRESS_REGEX, payout_address):
                return False, None, f"Invalid payout_address format: {payout_address}. Must be a valid EVM address (0x followed by 40 hex characters)."

        # Check for duplicate HL address across all entities
        with self._entities_lock:
            if normalized_hl_address in self._hl_address_to_synthetic:
                existing = self._hl_address_to_synthetic[normalized_hl_address]
                return False, None, f"Hyperliquid address {hl_address} is already registered to subaccount {existing}"

            # Check total active HL subaccounts < max limit
            active_hl_count = len(self._hl_address_to_synthetic)
            max_addresses = self._get_hl_max_addresses()
            if active_hl_count >= max_addresses:
                return False, None, (
                    f"Maximum number of tracked Hyperliquid addresses ({max_addresses}) reached. "
                    f"Cannot register more HL subaccounts."
                )

        # Delegate to standard create_subaccount for all existing validation
        success, subaccount_info, message = self.create_subaccount(
            entity_hotkey, account_size, asset_class, admin=admin
        )

        if not success:
            return False, None, message

        # Set hl_address and payout_address on the subaccount and re-persist
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if entity_data and subaccount_info:
                subaccount = entity_data.subaccounts.get(subaccount_info.subaccount_id)
                if subaccount:
                    subaccount.hl_address = hl_address
                    if payout_address:
                        subaccount.payout_address = payout_address
                    # Update HL reverse index
                    with self._entities_lock:
                        self._hl_address_to_synthetic[normalized_hl_address] = subaccount.synthetic_hotkey
                    self._write_entities_from_memory_to_disk()
                    # Persist hl_address on the MinerAccount so multiplier/buying_power use HS divisor
                    if self._miner_account_client:
                        self._miner_account_client.set_hl_address(subaccount.synthetic_hotkey, hl_address)

        return True, subaccount_info, message

    def get_all_active_hl_subaccounts(self) -> List[Tuple[str, dict]]:
        """
        Get all active subaccounts with HL addresses.

        Returns:
            List of (hl_address, subaccount_info_dict) tuples
        """
        result = []
        with self._entities_lock:
            for entity_data in self.entities.values():
                for subaccount in entity_data.subaccounts.values():
                    if subaccount.hl_address and subaccount.status in ('active', 'admin'):
                        result.append((subaccount.hl_address, subaccount.model_dump()))
        return result

    def get_synthetic_hotkey_for_hl_address(self, hl_address: str) -> Optional[str]:
        """
        O(1) lookup of synthetic hotkey for a Hyperliquid address.

        Args:
            hl_address: The Hyperliquid address

        Returns:
            Synthetic hotkey if found, None otherwise
        """
        normalized_hl_address = self._normalize_hl_address(hl_address)
        if not normalized_hl_address:
            return None
        with self._entities_lock:
            return self._hl_address_to_synthetic.get(normalized_hl_address)

    def get_subaccount_info_for_synthetic(self, synthetic_hotkey: str) -> Optional[SubaccountInfo]:
        """
        Get SubaccountInfo for a synthetic hotkey.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            SubaccountInfo if found, None otherwise
        """
        if not is_synthetic_hotkey(synthetic_hotkey):
            return None

        entity_hotkey, subaccount_id = parse_synthetic_hotkey(synthetic_hotkey)
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return None
            return entity_data.subaccounts.get(subaccount_id)

    def get_hl_subaccount_limits_data(self, hl_address: str) -> Optional[dict]:
        """
        Get lightweight limits data for an HL subaccount.

        Only fetches subaccount info (O(1) dict lookup) and challenge bucket
        (1 lightweight RPC call), avoiding the 7+ RPC calls of the full dashboard.

        Args:
            hl_address: The Hyperliquid address

        Returns:
            Dict with {account_size, asset_class, challenge_bucket} or None
        """
        synthetic_hotkey = self.get_synthetic_hotkey_for_hl_address(hl_address)
        if not synthetic_hotkey:
            return None

        subaccount_info = self.get_subaccount_info_for_synthetic(synthetic_hotkey)
        if not subaccount_info:
            return None

        # Get challenge bucket (1 lightweight RPC call)
        challenge_bucket = None
        if self._challenge_period_client.has_miner(synthetic_hotkey):
            bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
            if bucket:
                challenge_bucket = bucket.value

        return {
            'account_size': subaccount_info.account_size,
            'asset_class': subaccount_info.asset_class,
            'challenge_bucket': challenge_bucket,
        }

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
        # Use per-entity lock: only operates on single entity
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, f"Entity {entity_hotkey} not found"

            subaccount = entity_data.subaccounts.get(subaccount_id)
            if not subaccount:
                return False, f"Subaccount {subaccount_id} not found for entity {entity_hotkey}"

            if subaccount.status == "eliminated":
                return True, f"Subaccount {subaccount_id} already eliminated"

            subaccount.status = "eliminated"
            subaccount.eliminated_at_ms = TimeUtil.now_in_millis()

            # Remove HL address from reverse index so the wallet can be re-registered
            if subaccount.hl_address:
                normalized_hl = self._normalize_hl_address(subaccount.hl_address)
                if normalized_hl:
                    with self._entities_lock:
                        self._hl_address_to_synthetic.pop(normalized_hl, None)

            self._write_entities_from_memory_to_disk()

            bt.logging.info(
                f"[ENTITY_MANAGER] Eliminated subaccount {subaccount_id} for entity {entity_hotkey}. Reason: {reason}"
            )
            return True, f"Subaccount {subaccount_id} eliminated successfully"

    def restore_subaccount(self, synthetic_hotkey: str) -> Tuple[bool, str]:
        """
        Restore an erroneously eliminated subaccount back to active status.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (success: bool, message: str)
        """
        if not is_synthetic_hotkey(synthetic_hotkey):
            return False, f"{synthetic_hotkey} is not a synthetic hotkey"

        entity_hotkey, subaccount_id = parse_synthetic_hotkey(synthetic_hotkey)

        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, f"Entity {entity_hotkey} not found"

            subaccount = entity_data.subaccounts.get(subaccount_id)
            if not subaccount:
                return False, f"Subaccount {subaccount_id} not found for entity {entity_hotkey}"

            subaccount.status = "active"
            subaccount.eliminated_at_ms = None

            # Re-add HL address to reverse index
            if subaccount.hl_address:
                normalized_hl = self._normalize_hl_address(subaccount.hl_address)
                if normalized_hl:
                    with self._entities_lock:
                        self._hl_address_to_synthetic[normalized_hl] = subaccount.synthetic_hotkey

            self._write_entities_from_memory_to_disk()

        self.broadcast_subaccount_dashboard(synthetic_hotkey)

        bt.logging.info(
            f"[ENTITY_MANAGER] Restored subaccount {synthetic_hotkey} to active status"
        )
        return True, f"Subaccount {synthetic_hotkey} restored to active"

    def get_subaccount_status(self, synthetic_hotkey: str) -> Tuple[bool, Optional[str], str]:
        """
        Get the status of a subaccount by synthetic hotkey.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (found: bool, status: Optional[str], synthetic_hotkey: str)
        """
        if not is_synthetic_hotkey(synthetic_hotkey):
            return False, None, synthetic_hotkey

        entity_hotkey, subaccount_id = parse_synthetic_hotkey(synthetic_hotkey)

        # Use per-entity lock: only reads from single entity
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, None, synthetic_hotkey

            subaccount = entity_data.subaccounts.get(subaccount_id)
            if not subaccount:
                return False, None, synthetic_hotkey

            return True, subaccount.status, synthetic_hotkey

    def get_entity_data(self, entity_hotkey: str) -> Optional[EntityData]:
        """
        Get full entity data.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            EntityData or None
        """
        # Use per-entity lock: only reads from single entity
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            return self.entities.get(entity_hotkey)

    def get_subaccount_dashboard(self, synthetic_hotkey: str) -> dict | None:
        entity_hotkey, subaccount_id = parse_synthetic_hotkey(synthetic_hotkey)
        if not entity_hotkey:
            return None

        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)

            if entity_data is None:
                return None

            subaccount = entity_data.subaccounts.get(subaccount_id)
            if subaccount is None:
                return None

            result = {
                "synthetic_hotkey": synthetic_hotkey,
                "subaccount_uuid": subaccount.subaccount_uuid,
                "subaccount_id": subaccount.subaccount_id,
                "asset_class": subaccount.asset_class,
                "account_size": subaccount.account_size,
                "status": subaccount.status,
                "created_at_ms": subaccount.created_at_ms,
                "eliminated_at_ms": subaccount.eliminated_at_ms,
            }
            if subaccount.hl_address:
                result["hl_address"] = subaccount.hl_address
            if subaccount.payout_address:
                result["payout_address"] = subaccount.payout_address
            return result

    def get_synthetic_hotkey_from_uuid(self, subaccount_uuid: str) -> Optional[str]:
        """
        Translate subaccount UUID to synthetic hotkey using O(1) reverse index.

        Args:
            subaccount_uuid: The subaccount UUID

        Returns:
            Synthetic hotkey if found, None otherwise
        """
        with self._entities_lock:
            return self._uuid_to_hotkey.get(subaccount_uuid)

    def calculate_subaccount_payout(
        self,
        subaccount_uuid: str,
        start_time_ms: int,
        end_time_ms: Optional[int]
    ) -> Optional[dict]:
        """
        Calculate payout for a subaccount based on debt ledger checkpoints in time range.

        Orchestrates:
        1. UUID -> hotkey translation
        2. Debt ledger retrieval
        3. Checkpoint filtering by time range
        4. Payout calculation via DebtBasedScoring.calculate_payout_from_checkpoints()

        Args:
            subaccount_uuid: The subaccount UUID
            start_time_ms: Start timestamp (inclusive)
            end_time_ms: End timestamp (inclusive); if None, uses current time

        Returns:
            Dict with {
                'hotkey': str,
                'total_checkpoints': int,
                'checkpoints': List[dict],
                'payout': float
            } or None if subaccount not found
        """
        realtime = False
        if end_time_ms is None:
            end_time_ms = TimeUtil.now_in_millis()
            realtime = True

        # Translate UUID to hotkey
        synthetic_hotkey = self.get_synthetic_hotkey_from_uuid(subaccount_uuid)
        if not synthetic_hotkey:
            return None

        entity_hotkey, subaccount_id = parse_synthetic_hotkey(synthetic_hotkey)
        if not entity_hotkey or not subaccount_id:
            return None
        entity_data = self.get_entity_data(entity_hotkey)
        if not entity_data:
            return None
        subaccount = entity_data.subaccounts.get(subaccount_id)
        if not subaccount:
            return None

        # Get debt ledger for this hotkey
        try:
            debt_ledger = self._debt_ledger_client.get_ledger(synthetic_hotkey)
            if not debt_ledger:
                return None

            EMPTY_RESPONSE = {
                'hotkey': synthetic_hotkey,
                'total_checkpoints': 0,
                'checkpoints': {},
                'weekly_settlements': [],
                'payout': 0,
            }
            miner_bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey, end_time_ms)
            if miner_bucket not in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
                return EMPTY_RESPONSE

            checkpoints_dict = [cp.to_dict() for cp in debt_ledger.checkpoints]

            positions = self._position_client.get_positions_for_one_hotkey(synthetic_hotkey, sort_positions=True)
            orders = []
            fees = []
            realtime_unrealized = 0
            for position in positions:
                for order in position.orders:
                    if order.processed_ms < end_time_ms:
                        orders.append(order)
                for fee in position.fee_history:
                    if fee["time_ms"] < end_time_ms:
                        fees.append(fee)

                realtime_unrealized += position.unrealized_pnl

            orders.sort(key=lambda x: x.processed_ms)
            fees.sort(key=lambda x: x["time_ms"])
            if not orders:
                return EMPTY_RESPONSE

            weekly_settlements = []
            def _record_week(start_ms, end_ms, balance, prev_hwm, eow_unrealized, week_orders):
                weekly_settlements.append({
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'eow_balance': balance,
                    'eow_unrealized': eow_unrealized,
                    'payout': max(0.0, balance - prev_hwm + min(0.0, eow_unrealized)),
                    'orders': [o.to_python_dict() for o in week_orders],
                })

            running_balance = 0
            eow_hwm = 0
            MS_IN_WEEK = MS_IN_24_HOURS * 7
            CP_DURATION = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

            # Always start week_start at the Monday 00:00 UTC of the first order.
            first_day_index = orders[0].processed_ms // MS_IN_24_HOURS
            days_since_monday = (first_day_index + 3) % 7
            week_start = (first_day_index - days_since_monday) * MS_IN_24_HOURS
            week_end = week_start + MS_IN_WEEK

            idx_order, idx_fee = 0, 0
            while week_start < end_time_ms:
                end_time = min(week_end, end_time_ms)
                week_orders = []
                while idx_order < len(orders) and orders[idx_order].processed_ms < end_time:
                    running_balance += orders[idx_order].realized_pnl
                    if orders[idx_order].realized_pnl != 0:
                        week_orders.append(orders[idx_order])
                    idx_order += 1
                while idx_fee < len(fees) and fees[idx_fee]["time_ms"] < end_time:
                    running_balance -= fees[idx_fee]["amount"]
                    idx_fee += 1

                # debt checkpoint timestamp is start_ms, but unrealized pnl is at the end_ms
                # so must offset by a checkpoint for correct unrealized pnl at end time
                cp = debt_ledger.get_checkpoint_at_time(end_time - CP_DURATION, CP_DURATION)
                unrealized_pnl = cp.unrealized_pnl if cp else 0.0
                if end_time == end_time_ms and realtime:
                    unrealized_pnl = realtime_unrealized
                _record_week(week_start, end_time, running_balance, eow_hwm, unrealized_pnl, week_orders)
                eow_hwm = max(eow_hwm, running_balance)
                week_start, week_end = week_end, week_end + MS_IN_WEEK

            # Only sum weeks that fall within the requested period.
            payout = sum(w['payout'] for w in weekly_settlements if w['start_ms'] >= start_time_ms)

            return {
                'hotkey': synthetic_hotkey,
                'total_checkpoints': len(checkpoints_dict),
                'checkpoints': checkpoints_dict,
                'weekly_settlements': weekly_settlements,
                'payout': payout,
            }

        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Error calculating payout for {subaccount_uuid}: {e}")
            return None

    def validate_hotkey_for_orders(self, hotkey: str) -> dict:
        """
        Validate a hotkey for order placement in a single check.

        This consolidates multiple checks into one RPC call:
        1. Is it a synthetic hotkey (subaccount)?
        2. If synthetic, is it active?
        3. If not synthetic, is it an entity hotkey (not allowed to trade)?

        Args:
            hotkey: The hotkey to validate

        Returns:
            dict with:
                - is_valid (bool): Whether hotkey can place orders
                - error_message (str): Error message if not valid, empty if valid
                - hotkey_type (str): 'synthetic', 'entity', or 'regular'
                - status (str|None): Status if synthetic hotkey, None otherwise
        """
        # Check if synthetic (no lock needed - just string parsing)
        if is_synthetic_hotkey(hotkey):
            # Synthetic hotkey - check if active
            found, status, _ = self.get_subaccount_status(hotkey)

            if not found:
                return {
                    'is_valid': False,
                    'error_message': (f"Synthetic hotkey {hotkey} not found. "
                                    f"Please ensure your subaccount is properly registered."),
                    'hotkey_type': 'synthetic',
                    'status': None
                }

            if status not in ['active', 'admin']:
                return {
                    'is_valid': False,
                    'error_message': (f"Synthetic hotkey {hotkey} is not active (status: {status}). "
                                    f"Please ensure your subaccount is properly registered."),
                    'hotkey_type': 'synthetic',
                    'status': status
                }

            # Valid synthetic hotkey
            return {
                'is_valid': True,
                'error_message': '',
                'hotkey_type': 'synthetic',
                'status': status
            }

        # Not synthetic - check if it's an entity hotkey
        # Use per-entity lock: only reads from single entity
        entity_lock = self._get_entity_lock(hotkey)
        with entity_lock:
            entity_data = self.entities.get(hotkey)

        if entity_data:
            # Entity hotkey cannot place orders directly
            return {
                'is_valid': False,
                'error_message': (f"Entity hotkey {hotkey} cannot place orders directly. "
                                f"Please use a subaccount (synthetic hotkey) to place orders."),
                'hotkey_type': 'entity',
                'status': None
            }

        # Regular hotkey (not synthetic, not entity)
        return {
            'is_valid': True,
            'error_message': '',
            'hotkey_type': 'regular',
            'status': None
        }

    def get_subaccount_dashboard_data(self, synthetic_hotkey: str) -> Optional[dict]:
        """
        Get comprehensive dashboard data for a subaccount by aggregating data from multiple RPC services.

        This method pulls existing data from:
        - ChallengePeriodClient: Challenge period status and bucket
        - DebtLedgerClient: Debt ledger data
        - PositionManagerClient: Open positions and leverage
        - LimitOrderClient: Unfilled limit orders
        - MinerStatisticsClient: Cached statistics (metrics, scores, rankings, etc.)
        - EliminationClient: Elimination status

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            Dict with aggregated dashboard data, or None if subaccount not found
        """
        # 1. Validate subaccount exists
        entity_hotkey, subaccount_id = parse_synthetic_hotkey(synthetic_hotkey)
        if not entity_hotkey:
            return None

        entity_data = self.get_entity_data(entity_hotkey)
        if not entity_data:
            return None

        subaccount = entity_data.subaccounts.get(subaccount_id)
        if not subaccount:
            return None

        # 2. Query each client (with graceful degradation on errors)
        time_now_ms = TimeUtil.now_in_millis()

        # Challenge period data
        challenge_data = None
        try:
            if self._challenge_period_client.has_miner(synthetic_hotkey):
                bucket = self._challenge_period_client.get_miner_bucket(synthetic_hotkey)
                start_time = self._challenge_period_client.get_miner_start_time(synthetic_hotkey)
                challenge_data = {
                    'bucket': bucket.value if bucket else None,
                    'start_time_ms': start_time
                }
        except Exception as e:
            bt.logging.debug(f"[ENTITY_MANAGER] Challenge period data unavailable for {synthetic_hotkey}: {e}")

        # Debt ledger data
        ledger_data = None
        try:
            ledger = self._debt_ledger_client.get_ledger(synthetic_hotkey)
            if ledger:
                ledger_data = ledger.to_dict()  # Convert DebtLedger to dict for JSON serialization
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Ledger data unavailable for {synthetic_hotkey}: {e}")

        # Position data
        positions_data = None
        try:
            positions = self._position_client.get_positions_for_one_hotkey(synthetic_hotkey, sort_positions=True, archived_positions=True)
            if positions:
                positions_data = PositionManagerClient.positions_to_dashboard_dict(positions, time_now_ms)
                # Add total leverage
                leverage = self._position_client.calculate_net_portfolio_leverage(synthetic_hotkey)
                positions_data['total_leverage'] = leverage
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Position data unavailable for {synthetic_hotkey}: {e}")

        # Limit orders data (unfilled orders)
        limit_orders_data = None
        try:
            limit_orders_data = self._limit_order_client.to_dashboard_dict(synthetic_hotkey)
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Limit orders data unavailable for {synthetic_hotkey}: {e}")

        account_size_data = None
        try:
            account_size_data = self._miner_account_client.get_account(synthetic_hotkey)
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Account size data unavailable for {synthetic_hotkey}: {e}")


        # Statistics data (from cached miner statistics - refreshed every 5 minutes)
        statistics_data = None
        try:
            statistics_data = self._statistics_client.get_miner_statistics_for_hotkey(synthetic_hotkey)
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Statistics data unavailable for {synthetic_hotkey}: {e}")

        # Elimination data
        elimination_data = None
        try:
            elimination_data = self._elimination_client.get_elimination(synthetic_hotkey)
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Elimination data unavailable for {synthetic_hotkey}: {e}")

        # Drawdown stats (synthetic hotkeys only)
        drawdown_data = None
        try:
            drawdown_data = self._challenge_period_client.get_drawdown_stats(synthetic_hotkey)
        except Exception as e:
            bt.logging.debug(f"[ENTITY_MANAGER] Drawdown stats unavailable for {synthetic_hotkey}: {e}")

        # 3. Build aggregated response
        subaccount_info_dict = {
            'synthetic_hotkey': synthetic_hotkey,
            'entity_hotkey': entity_hotkey,
            'subaccount_id': subaccount_id,
            'subaccount_uuid': subaccount.subaccount_uuid,
            'asset_class': subaccount.asset_class,
            'account_size': subaccount.account_size,
            'status': subaccount.status,
            'created_at_ms': subaccount.created_at_ms,
            'eliminated_at_ms': subaccount.eliminated_at_ms,
        }
        if subaccount.hl_address:
            subaccount_info_dict['hl_address'] = subaccount.hl_address
        if subaccount.payout_address:
            subaccount_info_dict['payout_address'] = subaccount.payout_address

        return {
            'subaccount_info': subaccount_info_dict,
            'challenge_period': challenge_data,
            'ledger': ledger_data,
            'positions': positions_data,
            'limit_orders': limit_orders_data,
            'account_size_data': account_size_data,
            'statistics': statistics_data,
            'elimination': elimination_data,
            'drawdown': drawdown_data,
        }

    def broadcast_subaccount_dashboard(self, synthetic_hotkey: str) -> None:
        if self.running_unit_tests:
            return

        try:
            self._websocket_client.broadcast_subaccount_dashboard(synthetic_hotkey)
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Dashboard broadcast failed for {synthetic_hotkey}: {e}")

    def get_all_entities(self) -> Dict[str, EntityData]:
        """Get all entities."""
        # Use master lock: copying entire dict
        with self._entities_lock:
            return dict(self.entities)

    def get_hl_leaderboard_data(self) -> dict:
        """
        Build leaderboard data for all HL subaccounts using batch queries.

        Returns a dict with:
        - summary: global metrics (totalTraders, fundedTraders, inChallenge, eliminated, totalVolume)
        - fundedTraders: list of funded trader dicts sorted by PnL descending
        - challengeTraders: list of in-challenge trader dicts sorted by progress descending
        """
        time_now_ms = TimeUtil.now_in_millis()

        # 1. Collect all HL subaccounts (active + eliminated) from entities
        all_hl_subaccounts = []  # (hl_address, subaccount_info, synthetic_hotkey)
        with self._entities_lock:
            for entity_data in self.entities.values():
                for subaccount in entity_data.subaccounts.values():
                    if subaccount.hl_address:
                        all_hl_subaccounts.append((
                            subaccount.hl_address,
                            subaccount,
                            subaccount.synthetic_hotkey
                        ))

        active_subaccounts = [
            (addr, sa, shk) for addr, sa, shk in all_hl_subaccounts
            if sa.status in ('active', 'admin')
        ]
        eliminated_subaccounts = [
            (addr, sa, shk) for addr, sa, shk in all_hl_subaccounts
            if sa.status == 'eliminated'
        ]
        active_hotkeys = [shk for _, _, shk in active_subaccounts]

        # 2. Batch fetch from RPC services
        # Challenge period buckets
        challenge_buckets = {}  # hotkey -> (bucket_str, start_time_ms)
        try:
            testing_miners = self._challenge_period_client.get_testing_miners()
            success_miners = self._challenge_period_client.get_success_miners()
            for hk, start_time in testing_miners.items():
                challenge_buckets[hk] = ('testing', start_time)
            for hk, start_time in success_miners.items():
                challenge_buckets[hk] = ('funded', start_time)
        except Exception as e:
            bt.logging.error(f"[LEADERBOARD] Challenge period fetch failed: {e}")

        # Batch statistics
        batch_stats = {}
        try:
            batch_stats = self._statistics_client.get_miner_statistics_for_hotkeys(active_hotkeys)
        except Exception as e:
            bt.logging.error(f"[LEADERBOARD] Batch statistics fetch failed: {e}")

        # Batch account data
        batch_accounts = {}
        try:
            batch_accounts = self._miner_account_client.get_accounts(active_hotkeys)
        except Exception as e:
            bt.logging.error(f"[LEADERBOARD] Batch accounts fetch failed: {e}")

        # 3. Build trader entries
        funded_traders = []
        challenge_traders = []
        total_volume = 0.0
        total_payouts = 0.0

        for hl_address, subaccount, synthetic_hotkey in active_subaccounts:
            bucket_info = challenge_buckets.get(synthetic_hotkey)
            if not bucket_info:
                # Registered but no trades placed yet — include with null stats at bottom
                since = datetime.fromtimestamp(
                    subaccount.created_at_ms / 1000, tz=timezone.utc
                ).strftime('%b %Y')
                challenge_traders.append({
                    'address': hl_address,
                    'pnl': None,
                    'progress': None,
                    'sharpe': None,
                    'trades': 0,
                    'winRate': None,
                    'volume': 0,
                    'drawdown': None,
                    'since': since,
                    'noTrades': True,
                })
                continue

            bucket_str, bucket_start_time = bucket_info
            stats = batch_stats.get(synthetic_hotkey) or {}
            account = batch_accounts.get(synthetic_hotkey) or {}

            # Extract common fields
            engagement = stats.get('engagement') or {}
            n_positions = engagement.get('n_positions') or 0
            trader_volume = engagement.get('total_volume', 0.0)
            percentage_profitable = engagement.get('percentage_profitable')
            win_rate = round(percentage_profitable * 100, 1) if percentage_profitable is not None else None
            drawdowns = stats.get('drawdowns') or {}
            pnl_info = stats.get('pnl_info') or {}
            raw_pnl = pnl_info.get('raw_pnl', 0.0) if isinstance(pnl_info, dict) else 0.0

            # Sharpe: extract from scores
            sharpe = None
            scores = stats.get('scores') or {}
            for asset_class_scores in scores.values():
                sharpe_data = asset_class_scores.get('sharpe') or {}
                if 'value' in sharpe_data:
                    sharpe = round(sharpe_data['value'], 2)
                    break

            account_size = subaccount.account_size
            balance = account.get('balance', account_size) if isinstance(account, dict) else account_size
            total_realized_pnl = account.get('total_realized_pnl', 0.0) if isinstance(account, dict) else 0.0

            total_volume += trader_volume

            # Since date from created_at_ms
            since = datetime.fromtimestamp(
                subaccount.created_at_ms / 1000, tz=timezone.utc
            ).strftime('%b %Y')

            if bucket_str == 'funded':
                weight_info = stats.get('weight') or {}
                funded_traders.append({
                    'address': hl_address,
                    'pnl': round(total_realized_pnl, 2),
                    'funding': account_size,
                    'sharpe': sharpe,
                    'trades': n_positions,
                    'winRate': win_rate,
                    'volume': trader_volume,
                    'payouts': 0,
                    'since': since,
                    'rank': weight_info.get('rank'),
                })
            elif bucket_str == 'testing':
                # Calculate challenge progress
                asset_class = (subaccount.asset_class or '').lower()
                target_return = (
                    ValiConfig.SUBACCOUNT_CRYPTO_CHALLENGE_RETURNS_THRESHOLD
                    if asset_class == TradePairCategory.CRYPTO.value
                    else ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD
                )

                current_return = None
                progress = 0.0
                drawdown_percent = 0.0

                if isinstance(account_size, (int, float)) and account_size > 0 and isinstance(balance, (int, float)):
                    current_return = balance / account_size
                    returns_pct = (current_return - 1.0) * 100.0
                    target_pct = target_return * 100.0
                    if target_pct > 0:
                        progress = round(min(max((returns_pct / target_pct) * 100.0, 0.0), 100.0), 1)

                    max_return = account.get('max_return', current_return) if isinstance(account, dict) else current_return
                    if isinstance(max_return, (int, float)) and max_return > 0:
                        drawdown_percent = round(max((1.0 - (current_return / max_return)) * 100.0, 0.0), 1)

                challenge_traders.append({
                    'address': hl_address,
                    'pnl': round(total_realized_pnl, 2),
                    'progress': progress,
                    'sharpe': sharpe,
                    'trades': n_positions,
                    'winRate': win_rate,
                    'volume': trader_volume,
                    'drawdown': drawdown_percent,
                    'since': since,
                })

        # 4. Sort
        funded_traders.sort(key=lambda t: t.get('rank') or float('inf'))
        for i, trader in enumerate(funded_traders, 1):
            trader['rank'] = i

        challenge_traders.sort(key=lambda t: (t.get('noTrades', False), -(t.get('pnl') or 0)))

        # 5. Build summary
        summary = {
            'totalPaidOut': 0,
            'totalTraders': len(all_hl_subaccounts),
            'fundedTraders': len(funded_traders),
            'inChallenge': len(challenge_traders),
            'eliminated': len(eliminated_subaccounts),
            'totalVolume': total_volume,
        }

        return {
            'summary': summary,
            'fundedTraders': funded_traders,
            'challengeTraders': challenge_traders,
            'timestamp': time_now_ms,
        }

    # ==================== Challenge Period & Elimination Assessment ====================

    def assess_eliminations(self) -> int:
        """
        Check all active subaccounts against the elimination registry and mark eliminated ones.

        This runs periodically (every 5 minutes via daemon) to sync subaccount status
        with the central elimination registry managed by EliminationManager.

        Returns:
            int: Number of subaccounts newly marked as eliminated
        """
        eliminated_count = 0
        now_ms = TimeUtil.now_in_millis()

        # Get all eliminated hotkeys from the central registry
        eliminated_hotkeys = self._elimination_client.get_eliminated_hotkeys()

        # Use master lock: iterating over all entities
        with self._entities_lock:
            for entity_hotkey, entity_data in self.entities.items():
                for subaccount_id, subaccount in entity_data.subaccounts.items():
                    # Skip if already eliminated
                    if subaccount.status == "eliminated":
                        continue

                    synthetic_hotkey = subaccount.synthetic_hotkey

                    # Check if this synthetic hotkey is in eliminations
                    if synthetic_hotkey in eliminated_hotkeys:
                        # Get elimination details for logging
                        elimination_info = self._elimination_client.get_elimination(synthetic_hotkey)
                        reason = elimination_info.get('reason', 'unknown') if elimination_info else 'unknown'

                        bt.logging.info(
                            f"[ENTITY_MANAGER] Subaccount {synthetic_hotkey} found in eliminations. "
                            f"Reason: {reason}. Marking as eliminated."
                        )

                        # Mark subaccount as eliminated
                        subaccount.status = "eliminated"
                        subaccount.eliminated_at_ms = now_ms
                        eliminated_count += 1

                        self.broadcast_subaccount_dashboard(synthetic_hotkey)

            # Persist changes if any subaccounts were eliminated
            if eliminated_count > 0:
                self._write_entities_from_memory_to_disk()

        if eliminated_count > 0:
            bt.logging.info(
                f"[ENTITY_MANAGER] Elimination assessment complete: "
                f"{eliminated_count} subaccounts newly marked as eliminated"
            )

        return eliminated_count

    def sync_entity_data(self, entities_checkpoint_dict: dict) -> dict:
        """
        Sync entity data from a checkpoint dict (from auto-sync or mothership).

        This merges incoming entity data with existing data:
        - Creates new entities if they don't exist
        - Adds new subaccounts to existing entities
        - Updates subaccount status (active/eliminated)
        - Preserves local-only data (e.g., newer subaccounts)

        Args:
            entities_checkpoint_dict: Dict from checkpoint (entity_hotkey -> EntityData dict)

        Returns:
            dict: Sync statistics (entities_added, subaccounts_added, subaccounts_updated)
        """
        stats = {
            'entities_added': 0,
            'subaccounts_added': 0,
            'subaccounts_updated': 0,
            'entities_skipped': 0
        }

        # Validate input
        if not isinstance(entities_checkpoint_dict, dict):
            bt.logging.warning(f"[ENTITY_MANAGER] Invalid entities_checkpoint_dict type: {type(entities_checkpoint_dict)}. Expected dict.")
            return stats

        if not entities_checkpoint_dict:
            bt.logging.debug("[ENTITY_MANAGER] Empty entities_checkpoint_dict provided, nothing to sync")
            return stats

        # Parse checkpoint dict to EntityData objects (with error handling)
        try:
            incoming_entities = EntityManager.parse_checkpoint_dict(entities_checkpoint_dict)
        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Failed to parse entity checkpoint dict: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return stats

        # Use master lock: modifying entities dict
        with self._entities_lock:
            for entity_hotkey, incoming_entity in incoming_entities.items():
                # Check if entity exists locally
                local_entity = self.entities.get(entity_hotkey)

                if not local_entity:
                    # New entity - add it
                    self.entities[entity_hotkey] = incoming_entity
                    # Create lock for new entity
                    self._entity_locks[entity_hotkey] = threading.RLock()

                    # Register entity with challenge period system (ENTITY bucket - 4x dust weight)
                    self._challenge_period_client.set_miner_bucket(
                        entity_hotkey,
                        MinerBucket.ENTITY,
                        incoming_entity.registered_at_ms
                    )

                    # Update HL address reverse index for all subaccounts
                    for sub in incoming_entity.subaccounts.values():
                        if sub.hl_address:
                            normalized_hl = self._normalize_hl_address(sub.hl_address)
                            if normalized_hl:
                                self._hl_address_to_synthetic[normalized_hl] = sub.synthetic_hotkey
                            if self._miner_account_client:
                                self._miner_account_client.set_hl_address(sub.synthetic_hotkey, sub.hl_address)

                    stats['entities_added'] += 1
                    stats['subaccounts_added'] += len(incoming_entity.subaccounts)
                    bt.logging.info(f"[ENTITY_MANAGER] Added new entity {entity_hotkey} with {len(incoming_entity.subaccounts)} subaccounts from sync")
                else:
                    # Entity exists - merge subaccounts
                    # Use per-entity lock for updates
                    entity_lock = self._get_entity_lock(entity_hotkey)
                    with entity_lock:
                        for sub_id, incoming_sub in incoming_entity.subaccounts.items():
                            local_sub = local_entity.subaccounts.get(sub_id)

                            if not local_sub:
                                # New subaccount - add it
                                local_entity.subaccounts[sub_id] = incoming_sub

                                # Update reverse index
                                with self._entities_lock:
                                    self._uuid_to_hotkey[incoming_sub.subaccount_uuid] = incoming_sub.synthetic_hotkey

                                # Update HL address reverse index
                                if incoming_sub.hl_address:
                                    normalized_hl = self._normalize_hl_address(incoming_sub.hl_address)
                                    if normalized_hl:
                                        self._hl_address_to_synthetic[normalized_hl] = incoming_sub.synthetic_hotkey
                                    if self._miner_account_client:
                                        self._miner_account_client.set_hl_address(incoming_sub.synthetic_hotkey, incoming_sub.hl_address)

                                stats['subaccounts_added'] += 1
                                bt.logging.info(f"[ENTITY_MANAGER] Added subaccount {incoming_sub.synthetic_hotkey} from sync")
                            else:
                                # Subaccount exists - update status if changed
                                if local_sub.status != incoming_sub.status:
                                    old_status = local_sub.status
                                    local_sub.status = incoming_sub.status
                                    local_sub.eliminated_at_ms = incoming_sub.eliminated_at_ms
                                    stats['subaccounts_updated'] += 1
                                    bt.logging.info(f"[ENTITY_MANAGER] Updated subaccount {incoming_sub.synthetic_hotkey} status: {old_status} -> {incoming_sub.status}")

                                # Update HL address if added
                                if incoming_sub.hl_address and not local_sub.hl_address:
                                    local_sub.hl_address = incoming_sub.hl_address
                                    normalized_hl = self._normalize_hl_address(incoming_sub.hl_address)
                                    if normalized_hl:
                                        self._hl_address_to_synthetic[normalized_hl] = incoming_sub.synthetic_hotkey
                                    if self._miner_account_client:
                                        self._miner_account_client.set_hl_address(incoming_sub.synthetic_hotkey, incoming_sub.hl_address)
                                    stats['subaccounts_updated'] += 1

                                # Update payout_address if added
                                if incoming_sub.payout_address and not local_sub.payout_address:
                                    local_sub.payout_address = incoming_sub.payout_address
                                    stats['subaccounts_updated'] += 1

                        # Update next_subaccount_id to prevent ID collisions
                        if incoming_entity.next_subaccount_id > local_entity.next_subaccount_id:
                            local_entity.next_subaccount_id = incoming_entity.next_subaccount_id

            # Persist changes to disk
            self._write_entities_from_memory_to_disk()

        bt.logging.info(f"[ENTITY_MANAGER] Entity sync complete: {stats}")
        return stats

    # ==================== Persistence ====================

    def _write_entities_from_memory_to_disk(self):
        """Write entity data from memory to disk."""
        if self.is_backtesting:
            return

        entity_data = self.to_checkpoint_dict()
        ValiBkpUtils.write_file(self.ENTITY_FILE, entity_data)

    def to_checkpoint_dict(self) -> dict:
        """Get entity data as a checkpoint dict for serialization."""
        # Use master lock: iterating over all entities
        with self._entities_lock:
            checkpoint = {}
            for entity_hotkey, entity_data in self.entities.items():
                checkpoint[entity_hotkey] = entity_data.model_dump()
            return checkpoint

    @staticmethod
    def parse_checkpoint_dict(json_dict: dict) -> Dict[str, EntityData]:
        """Parse checkpoint dict from disk."""
        entities = {}
        for entity_hotkey, entity_dict in json_dict.items():
            # Convert subaccount dicts back to SubaccountInfo objects
            subaccounts_dict = {}
            for sub_id_str, sub_dict in entity_dict.get("subaccounts", {}).items():
                subaccounts_dict[int(sub_id_str)] = SubaccountInfo(**sub_dict)

            entity_dict["subaccounts"] = subaccounts_dict
            entities[entity_hotkey] = EntityData(**entity_dict)

        return entities

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
    ):
        """
        Broadcast SubaccountRegistration synapse to other validators using shared broadcast base.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            subaccount_id: The subaccount ID
            subaccount_uuid: The subaccount UUID
            synthetic_hotkey: The synthetic hotkey
            account_size: Account size in USD (immutable)
            asset_class: Asset class selection (immutable)
            status: Subaccount status (active, admin, etc.)
            hl_address: Optional Hyperliquid address for HL subaccounts
            payout_address: Optional EVM payout address for HL subaccounts
        """
        def create_synapse():
            subaccount_data = {
                "entity_hotkey": entity_hotkey,
                "subaccount_id": subaccount_id,
                "subaccount_uuid": subaccount_uuid,
                "synthetic_hotkey": synthetic_hotkey,
                "account_size": account_size,
                "asset_class": asset_class,
                "status": status,
                "hl_address": hl_address,
                "payout_address": payout_address
            }
            if hl_address:
                subaccount_data["hl_address"] = hl_address
            if payout_address:
                subaccount_data["payout_address"] = payout_address
            return template.protocol.SubaccountRegistration(subaccount_data=subaccount_data)

        self._broadcast_to_validators(
            synapse_factory=create_synapse,
            broadcast_name="SubaccountRegistration",
            context={"synthetic_hotkey": synthetic_hotkey}
        )

    def receive_subaccount_registration_update(self, subaccount_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process an incoming subaccount registration from another validator.
        Ensures idempotent registration (handles duplicates gracefully).

        Args:
            subaccount_data: Dictionary containing entity_hotkey, subaccount_id, subaccount_uuid, synthetic_hotkey
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # SECURITY: Verify sender using shared base class method
            if not self.verify_broadcast_sender(sender_hotkey, "SubaccountRegistration"):
                return False

            # Use master lock: might create new entity, then modify it
            with self._entities_lock:
                # Extract data from the synapse
                entity_hotkey = subaccount_data.get("entity_hotkey")
                subaccount_id = subaccount_data.get("subaccount_id")
                subaccount_uuid = subaccount_data.get("subaccount_uuid")
                synthetic_hotkey = subaccount_data.get("synthetic_hotkey")
                account_size = subaccount_data.get("account_size")
                asset_class = subaccount_data.get("asset_class")
                status = subaccount_data.get("status", "active")  # Default to active for backwards compatibility
                hl_address = subaccount_data.get("hl_address")
                normalized_hl = self._normalize_hl_address(hl_address)
                payout_address = subaccount_data.get("payout_address")

                bt.logging.info(
                    f"[ENTITY_MANAGER] Processing subaccount registration for {synthetic_hotkey}"
                )

                # Validate all required fields are present
                if not all([entity_hotkey, subaccount_id is not None, subaccount_uuid, synthetic_hotkey,
                            account_size, asset_class]):
                    bt.logging.warning(
                        f"[ENTITY_MANAGER] Invalid subaccount registration data - missing required fields: {subaccount_data}"
                    )
                    return False

                # Get or create entity data
                entity_data = self.entities.get(entity_hotkey)
                if not entity_data:
                    # Auto-create entity if doesn't exist (from broadcast)
                    entity_data = EntityData(
                        entity_hotkey=entity_hotkey,
                        subaccounts={},
                        next_subaccount_id=subaccount_id + 1,  # Ensure monotonic ID continues
                        registered_at_ms=TimeUtil.now_in_millis()
                    )
                    self.entities[entity_hotkey] = entity_data
                    # Create lock for this entity
                    self._entity_locks[entity_hotkey] = threading.RLock()
                    bt.logging.info(f"[ENTITY_MANAGER] Auto-created entity {entity_hotkey} from broadcast")

                # Check if subaccount already exists (idempotent)
                if subaccount_id in entity_data.subaccounts:
                    existing_sub = entity_data.subaccounts[subaccount_id]
                    if existing_sub.subaccount_uuid == subaccount_uuid:
                        changed = False
                        # Update status if changed
                        if existing_sub.status != status:
                            bt.logging.info(
                                f"[ENTITY_MANAGER] Updating subaccount {synthetic_hotkey} status: "
                                f"{existing_sub.status} -> {status}"
                            )
                            existing_sub.status = status
                            changed = True
                        # Update hl_address if previously None and now provided
                        if hl_address and not existing_sub.hl_address:
                            existing_sub.hl_address = hl_address
                            if normalized_hl:
                                self._hl_address_to_synthetic[normalized_hl] = synthetic_hotkey
                            if self._miner_account_client:
                                self._miner_account_client.set_hl_address(synthetic_hotkey, hl_address)
                            bt.logging.info(
                                f"[ENTITY_MANAGER] Set hl_address {hl_address} for subaccount {synthetic_hotkey}"
                            )
                            changed = True
                        # Update payout_address if previously None and now provided
                        if payout_address and not existing_sub.payout_address:
                            existing_sub.payout_address = payout_address
                            bt.logging.info(
                                f"[ENTITY_MANAGER] Set payout_address {payout_address} for subaccount {synthetic_hotkey}"
                            )
                            changed = True
                        if changed:
                            self._write_entities_from_memory_to_disk()
                        else:
                            bt.logging.debug(
                                f"[ENTITY_MANAGER] Subaccount {synthetic_hotkey} already exists (idempotent)"
                            )
                        return True
                    else:
                        bt.logging.warning(
                            f"[ENTITY_MANAGER] Subaccount ID conflict for {entity_hotkey}:{subaccount_id}"
                        )
                        return False

                # Create new subaccount info
                now_ms = TimeUtil.now_in_millis()
                hl_address = subaccount_data.get("hl_address")
                payout_address = subaccount_data.get("payout_address")
                subaccount_info = SubaccountInfo(
                    subaccount_id=subaccount_id,
                    subaccount_uuid=subaccount_uuid,
                    synthetic_hotkey=synthetic_hotkey,
                    status=status,
                    created_at_ms=now_ms,
                    account_size=account_size,
                    asset_class=asset_class,
                    hl_address=hl_address,
                    payout_address=payout_address
                )

                # Add to entity
                entity_data.subaccounts[subaccount_id] = subaccount_info

                # Update reverse indices
                with self._entities_lock:
                    self._uuid_to_hotkey[subaccount_uuid] = synthetic_hotkey
                    if normalized_hl:
                        self._hl_address_to_synthetic[normalized_hl] = synthetic_hotkey

                # Propagate hl_address to MinerAccount so buying_power uses the correct HS divisor
                if hl_address and self._miner_account_client:
                    self._miner_account_client.set_hl_address(synthetic_hotkey, hl_address)

                # Update next_subaccount_id if needed
                if subaccount_id >= entity_data.next_subaccount_id:
                    entity_data.next_subaccount_id = subaccount_id + 1

                # Save to disk
                self._write_entities_from_memory_to_disk()

                bt.logging.info(
                    f"[ENTITY_MANAGER] Registered subaccount {synthetic_hotkey} via broadcast"
                )
                return True

        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Error processing subaccount registration: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False

    # ==================== Entity Endpoint URL Methods ====================

    def set_endpoint_url(self, entity_hotkey: str, endpoint_url: str) -> Tuple[bool, str]:
        """
        Set the public endpoint URL for an entity miner.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            endpoint_url: The public-facing endpoint URL (http/https, max 512 chars)

        Returns:
            (success: bool, message: str)
        """
        # Validate URL
        if not endpoint_url or not isinstance(endpoint_url, str):
            return False, "endpoint_url is required"

        if len(endpoint_url) > 512:
            return False, "endpoint_url must be 512 characters or fewer"

        if not endpoint_url.startswith("http://") and not endpoint_url.startswith("https://"):
            return False, "endpoint_url must start with http:// or https://"

        with self._entities_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, f"Entity {entity_hotkey} not found. Register first."

            entity_data.endpoint_url = endpoint_url
            self._write_entities_from_memory_to_disk()

        bt.logging.info(f"[ENTITY_MANAGER] Set endpoint URL for {entity_hotkey}: {endpoint_url}")

        # Broadcast to other validators (non-mothership validators receive this)
        self.broadcast_entity_endpoint_update(entity_hotkey, endpoint_url)

        return True, f"Endpoint URL set successfully: {endpoint_url}"

    def get_endpoint_url_by_address(self, hl_address: str = None, subaccount: str = None) -> Optional[str]:
        """
        Resolve an HL address or synthetic hotkey to the entity's endpoint URL.

        Args:
            hl_address: Hyperliquid address (0x-prefixed)
            subaccount: Synthetic hotkey (entity_hotkey_N)

        Returns:
            The entity's endpoint URL, or None if not found
        """
        entity_hotkey = None

        if hl_address:
            # HL address -> synthetic hotkey -> entity_hotkey
            normalized_hl = self._normalize_hl_address(hl_address)
            with self._entities_lock:
                synthetic = self._hl_address_to_synthetic.get(normalized_hl) if normalized_hl else None
            if synthetic:
                parsed = parse_synthetic_hotkey(synthetic)
                if parsed[0]:
                    entity_hotkey = parsed[0]

        elif subaccount:
            # Synthetic hotkey -> entity_hotkey
            parsed = parse_synthetic_hotkey(subaccount)
            if parsed[0]:
                entity_hotkey = parsed[0]

        if not entity_hotkey:
            return None

        with self._entities_lock:
            entity_data = self.entities.get(entity_hotkey)
            if entity_data:
                return entity_data.endpoint_url

        return None

    def broadcast_entity_endpoint_update(self, entity_hotkey: str, endpoint_url: str):
        """
        Broadcast EntityEndpointUpdate synapse to other validators.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            endpoint_url: The public-facing endpoint URL
        """
        def create_synapse():
            endpoint_data = {
                "entity_hotkey": entity_hotkey,
                "endpoint_url": endpoint_url
            }
            return template.protocol.EntityEndpointUpdate(endpoint_data=endpoint_data)

        self._broadcast_to_validators(
            synapse_factory=create_synapse,
            broadcast_name="EntityEndpointUpdate",
            context={"entity_hotkey": entity_hotkey}
        )

    def receive_entity_endpoint_update(self, endpoint_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process an incoming EntityEndpointUpdate from another validator (mothership).

        Args:
            endpoint_data: Dictionary containing entity_hotkey and endpoint_url
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # SECURITY: Verify sender using shared base class method
            if not self.verify_broadcast_sender(sender_hotkey, "EntityEndpointUpdate"):
                return False

            entity_hotkey = endpoint_data.get("entity_hotkey")
            endpoint_url = endpoint_data.get("endpoint_url")

            if not entity_hotkey or not endpoint_url:
                bt.logging.warning(
                    f"[ENTITY_MANAGER] Invalid endpoint update data - missing required fields: {endpoint_data}"
                )
                return False

            with self._entities_lock:
                entity_data = self.entities.get(entity_hotkey)
                if not entity_data:
                    bt.logging.warning(
                        f"[ENTITY_MANAGER] Entity {entity_hotkey} not found for endpoint update"
                    )
                    return False

                entity_data.endpoint_url = endpoint_url
                self._write_entities_from_memory_to_disk()

            bt.logging.info(
                f"[ENTITY_MANAGER] Received endpoint URL update for {entity_hotkey}: {endpoint_url}"
            )
            return True

        except Exception as e:
            bt.logging.error(f"[ENTITY_MANAGER] Error processing endpoint update: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False

    # ==================== Testing/Admin Methods ====================

    def clear_all_entities(self):
        """Clear all entity data (for testing)."""
        if not self.running_unit_tests:
            raise Exception("Clearing entities is only allowed during unit tests.")

        # Use master lock: clearing entire dict
        with self._entities_lock:
            self.entities.clear()
            self._entity_locks.clear()
            self._hl_address_to_synthetic.clear()
            self._write_entities_from_memory_to_disk()

        bt.logging.info("[ENTITY_MANAGER] Cleared all entity data")
