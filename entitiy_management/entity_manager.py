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
import uuid
import time
import threading
import asyncio
import bittensor as bt
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from pydantic import BaseModel, Field

import template.protocol
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.cache_controller import CacheController
from vali_objects.validator_broadcast_base import ValidatorBroadcastBase
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.statistics.miner_statistics_client import MinerStatisticsClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from time_util.time_util import TimeUtil


class SubaccountInfo(BaseModel):
    """Data structure for a single subaccount."""
    subaccount_id: int = Field(description="Monotonically increasing ID")
    subaccount_uuid: str = Field(description="Unique UUID for this subaccount")
    synthetic_hotkey: str = Field(description="Synthetic hotkey: {entity_hotkey}_{subaccount_id}")
    status: str = Field(default="active", description="Status: active, eliminated, or unknown")
    created_at_ms: int = Field(description="Timestamp when subaccount was created")
    eliminated_at_ms: Optional[int] = Field(default=None, description="Timestamp when subaccount was eliminated")

    # Note: Challenge period tracking has been migrated to ChallengePeriodManager
    # Synthetic hotkeys are added to challenge period bucket and evaluated via inspect()


class EntityData(BaseModel):
    """Data structure for an entity."""
    entity_hotkey: str = Field(description="The VANTA_ENTITY_HOTKEY")
    subaccounts: Dict[int, SubaccountInfo] = Field(default_factory=dict, description="Map subaccount_id -> SubaccountInfo")
    next_subaccount_id: int = Field(default=0, description="Next subaccount ID to assign (monotonic)")
    collateral_amount: float = Field(default=0.0, description="Collateral amount (placeholder)")
    max_subaccounts: int = Field(default=10, description="Maximum allowed subaccounts")
    registered_at_ms: int = Field(description="Timestamp when entity was registered")

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
        is_testnet = (config.netuid == 116) if (config and hasattr(config, 'netuid')) else False

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

        self.ENTITY_FILE = ValiBkpUtils.get_entity_file_location(running_unit_tests=running_unit_tests)

        # Load initial entities from disk
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.ENTITY_FILE)
            self.entities = self.parse_checkpoint_dict(disk_data)
            # Recreate locks for all loaded entities
            for entity_hotkey in self.entities.keys():
                self._entity_locks[entity_hotkey] = threading.RLock()
            bt.logging.info(f"[ENTITY_MANAGER] Loaded {len(self.entities)} entities from disk with per-entity locks")

        bt.logging.info("[ENTITY_MANAGER] EntityManager initialized")

    # ==================== Lock Management ====================

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
        entity_hotkey: str,
        collateral_amount: float = 0.0,
        max_subaccounts: int = None
    ) -> Tuple[bool, str]:
        """
        Register a new entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            collateral_amount: Collateral amount (placeholder)
            max_subaccounts: Maximum allowed subaccounts (default from ValiConfig)

        Returns:
            (success: bool, message: str)
        """
        if max_subaccounts is None:
            max_subaccounts = ValiConfig.ENTITY_MAX_SUBACCOUNTS

        # Use master lock: adding new entity to dict
        with self._entities_lock:
            if entity_hotkey in self.entities:
                return False, f"Entity {entity_hotkey} already registered"

            # TODO: Add collateral verification here
            # collateral_verified = self._verify_collateral(entity_hotkey, collateral_amount)
            # if not collateral_verified:
            #     return False, "Insufficient collateral"

            entity_data = EntityData(
                entity_hotkey=entity_hotkey,
                subaccounts={},
                next_subaccount_id=0,
                collateral_amount=collateral_amount,
                max_subaccounts=max_subaccounts,
                registered_at_ms=TimeUtil.now_in_millis()
            )

            self.entities[entity_hotkey] = entity_data
            # Create lock for this entity
            self._entity_locks[entity_hotkey] = threading.RLock()
            self._write_entities_from_memory_to_disk()

            bt.logging.info(f"[ENTITY_MANAGER] Registered entity {entity_hotkey} with max_subaccounts={max_subaccounts}")
            return True, f"Entity {entity_hotkey} registered successfully"

    def create_subaccount(self, entity_hotkey: str) -> Tuple[bool, Optional[SubaccountInfo], str]:
        """
        Create a new subaccount for an entity.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY

        Returns:
            (success: bool, subaccount_info: Optional[SubaccountInfo], message: str)
        """
        # Use per-entity lock: only operates on single entity
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, None, f"Entity {entity_hotkey} not registered"

            # Check slot allowance
            active_count = len(entity_data.get_active_subaccounts())
            if active_count >= entity_data.max_subaccounts:
                return False, None, f"Entity {entity_hotkey} has reached maximum subaccounts ({entity_data.max_subaccounts})"

            # Generate monotonic ID
            subaccount_id = entity_data.next_subaccount_id
            entity_data.next_subaccount_id += 1

            # Generate UUID and synthetic hotkey
            subaccount_uuid = str(uuid.uuid4())
            synthetic_hotkey = f"{entity_hotkey}_{subaccount_id}"

            # Create subaccount info
            now_ms = TimeUtil.now_in_millis()
            subaccount_info = SubaccountInfo(
                subaccount_id=subaccount_id,
                subaccount_uuid=subaccount_uuid,
                synthetic_hotkey=synthetic_hotkey,
                status="active",
                created_at_ms=now_ms
            )

            # TODO: Transfer collateral from entity to subaccount
            # This should use the collateral SDK to transfer collateral from entity_hotkey to synthetic_hotkey
            # collateral_transfer_amount = calculate_subaccount_collateral(entity_data.collateral_amount, entity_data.max_subaccounts)
            # collateral_sdk.transfer_collateral(from_hotkey=entity_hotkey, to_hotkey=synthetic_hotkey, amount=collateral_transfer_amount)
            bt.logging.info(f"[ENTITY_MANAGER] TODO: Transfer collateral from {entity_hotkey} to {synthetic_hotkey}")

            # TODO: Set account size for the subaccount using ContractClient
            # This should set a fixed account size for the synthetic hotkey
            # from vali_objects.utils.vali_utils import ValiUtils
            # contract_client = ValiUtils.get_contract_client()
            # FIXED_ACCOUNT_SIZE = 1000.0  # Define this constant in ValiConfig
            # contract_client.set_account_size(synthetic_hotkey, FIXED_ACCOUNT_SIZE)
            bt.logging.info(f"[ENTITY_MANAGER] TODO: Set account size for {synthetic_hotkey} using ContractClient.set_account_size()")

            entity_data.subaccounts[subaccount_id] = subaccount_info
            self._write_entities_from_memory_to_disk()

            bt.logging.info(
                f"[ENTITY_MANAGER] Created subaccount {subaccount_id} for entity {entity_hotkey}: {synthetic_hotkey}"
            )
            return True, subaccount_info, f"Subaccount {subaccount_id} created successfully"

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
            self._write_entities_from_memory_to_disk()

            bt.logging.info(
                f"[ENTITY_MANAGER] Eliminated subaccount {subaccount_id} for entity {entity_hotkey}. Reason: {reason}"
            )
            return True, f"Subaccount {subaccount_id} eliminated successfully"

    def get_subaccount_status(self, synthetic_hotkey: str) -> Tuple[bool, Optional[str], str]:
        """
        Get the status of a subaccount by synthetic hotkey.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (found: bool, status: Optional[str], synthetic_hotkey: str)
        """
        if not self.is_synthetic_hotkey(synthetic_hotkey):
            return False, None, synthetic_hotkey

        entity_hotkey, subaccount_id = self.parse_synthetic_hotkey(synthetic_hotkey)

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
        if self.is_synthetic_hotkey(hotkey):
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

            if status != 'active':
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
        - MinerStatisticsClient: Cached statistics (metrics, scores, rankings, etc.)
        - EliminationClient: Elimination status

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            Dict with aggregated dashboard data, or None if subaccount not found
        """
        # 1. Validate subaccount exists
        entity_hotkey, subaccount_id = self.parse_synthetic_hotkey(synthetic_hotkey)
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
            ledger_data = self._debt_ledger_client.get_ledger(synthetic_hotkey)
        except Exception as e:
            bt.logging.debug(f"[ENTITY_MANAGER] Ledger data unavailable for {synthetic_hotkey}: {e}")

        # Position data
        positions_data = None
        try:
            positions = self._position_client.get_positions_for_one_hotkey(synthetic_hotkey)
            if positions:
                positions_data = PositionManagerClient.positions_to_dashboard_dict(positions, time_now_ms)
                # Add total leverage
                leverage = self._position_client.calculate_net_portfolio_leverage(synthetic_hotkey)
                positions_data['total_leverage'] = leverage
        except Exception as e:
            bt.logging.debug(f"[ENTITY_MANAGER] Position data unavailable for {synthetic_hotkey}: {e}")

        # Statistics data (from cached miner statistics - refreshed every 5 minutes)
        statistics_data = None
        try:
            statistics_data = self._statistics_client.get_miner_statistics_for_hotkey(synthetic_hotkey)
        except Exception as e:
            bt.logging.debug(f"[ENTITY_MANAGER] Statistics data unavailable for {synthetic_hotkey}: {e}")

        # Elimination data
        elimination_data = None
        try:
            elimination_data = self._elimination_client.get_elimination(synthetic_hotkey)
        except Exception as e:
            bt.logging.debug(f"[ENTITY_MANAGER] Elimination data unavailable for {synthetic_hotkey}: {e}")

        # 3. Build aggregated response
        return {
            'subaccount_info': {
                'synthetic_hotkey': synthetic_hotkey,
                'entity_hotkey': entity_hotkey,
                'subaccount_id': subaccount_id,
                'status': subaccount.status,
                'created_at_ms': subaccount.created_at_ms,
                'eliminated_at_ms': subaccount.eliminated_at_ms,
            },
            'challenge_period': challenge_data,
            'ledger': ledger_data,
            'positions': positions_data,
            'statistics': statistics_data,
            'elimination': elimination_data,
        }

    def get_all_entities(self) -> Dict[str, EntityData]:
        """Get all entities."""
        # Use master lock: copying entire dict
        with self._entities_lock:
            return dict(self.entities)

    def is_synthetic_hotkey(self, hotkey: str) -> bool:
        """
        Check if a hotkey is synthetic (contains underscore).

        Args:
            hotkey: The hotkey to check

        Returns:
            True if synthetic, False otherwise
        """
        # Edge case: What if an entity hotkey itself contains an underscore?
        # We handle this by checking if the part after the last underscore is a valid integer
        if "_" not in hotkey:
            return False

        # Try to parse as synthetic hotkey
        parts = hotkey.rsplit("_", 1)
        if len(parts) != 2:
            return False

        try:
            int(parts[1])  # Check if last part is a valid integer
            return True
        except ValueError:
            return False

    def parse_synthetic_hotkey(self, synthetic_hotkey: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse a synthetic hotkey into entity_hotkey and subaccount_id.

        Args:
            synthetic_hotkey: The synthetic hotkey ({entity_hotkey}_{subaccount_id})

        Returns:
            (entity_hotkey, subaccount_id) or (None, None) if invalid
        """
        if not self.is_synthetic_hotkey(synthetic_hotkey):
            return None, None

        parts = synthetic_hotkey.rsplit("_", 1)
        entity_hotkey = parts[0]
        try:
            subaccount_id = int(parts[1])
            return entity_hotkey, subaccount_id
        except ValueError:
            return None, None

    def update_collateral(self, entity_hotkey: str, collateral_amount: float) -> Tuple[bool, str]:
        """
        Update collateral for an entity (placeholder).

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            collateral_amount: New collateral amount

        Returns:
            (success: bool, message: str)
        """
        # Use per-entity lock: only operates on single entity
        entity_lock = self._get_entity_lock(entity_hotkey)
        with entity_lock:
            entity_data = self.entities.get(entity_hotkey)
            if not entity_data:
                return False, f"Entity {entity_hotkey} not found"

            # TODO: Verify collateral with collateral SDK
            entity_data.collateral_amount = collateral_amount
            self._write_entities_from_memory_to_disk()

            bt.logging.info(f"[ENTITY_MANAGER] Updated collateral for {entity_hotkey}: {collateral_amount}")
            return True, f"Collateral updated successfully"

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

            # Persist changes if any subaccounts were eliminated
            if eliminated_count > 0:
                self._write_entities_from_memory_to_disk()

        if eliminated_count > 0:
            bt.logging.info(
                f"[ENTITY_MANAGER] Elimination assessment complete: "
                f"{eliminated_count} subaccounts newly marked as eliminated"
            )

        return eliminated_count

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
        synthetic_hotkey: str
    ):
        """
        Broadcast SubaccountRegistration synapse to other validators using shared broadcast base.

        Args:
            entity_hotkey: The VANTA_ENTITY_HOTKEY
            subaccount_id: The subaccount ID
            subaccount_uuid: The subaccount UUID
            synthetic_hotkey: The synthetic hotkey
        """
        def create_synapse():
            subaccount_data = {
                "entity_hotkey": entity_hotkey,
                "subaccount_id": subaccount_id,
                "subaccount_uuid": subaccount_uuid,
                "synthetic_hotkey": synthetic_hotkey
            }
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

                bt.logging.info(
                    f"[ENTITY_MANAGER] Processing subaccount registration for {synthetic_hotkey}"
                )

                if not all([entity_hotkey, subaccount_id is not None, subaccount_uuid, synthetic_hotkey]):
                    bt.logging.warning(
                        f"[ENTITY_MANAGER] Invalid subaccount registration data received: {subaccount_data}"
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
                subaccount_info = SubaccountInfo(
                    subaccount_id=subaccount_id,
                    subaccount_uuid=subaccount_uuid,
                    synthetic_hotkey=synthetic_hotkey,
                    status="active",
                    created_at_ms=now_ms
                )

                # Add to entity
                entity_data.subaccounts[subaccount_id] = subaccount_info

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

    # ==================== Testing/Admin Methods ====================

    def clear_all_entities(self):
        """Clear all entity data (for testing)."""
        if not self.running_unit_tests:
            raise Exception("Clearing entities is only allowed during unit tests.")

        # Use master lock: clearing entire dict
        with self._entities_lock:
            self.entities.clear()
            self._entity_locks.clear()
            self._write_entities_from_memory_to_disk()

        bt.logging.info("[ENTITY_MANAGER] Cleared all entity data")
