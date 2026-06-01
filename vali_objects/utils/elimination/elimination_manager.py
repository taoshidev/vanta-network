# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
EliminationManager - Business logic for elimination management.

This manager contains the heavy business logic for managing eliminations,
while EliminationServer wraps it and exposes methods via RPC.

This follows the same pattern as PerfLedgerManager/PerfLedgerServer.

Usage:
    # Typically created by EliminationServer
    manager = EliminationManager(
        connection_mode=RPCConnectionMode.RPC,
        running_unit_tests=False
    )

    # Process eliminations
    manager.process_eliminations(iteration_epoch)
"""
from dataclasses import dataclass
import shutil
import threading
from enum import Enum

import bittensor as bt

from vali_objects.enums.order_source_enum import OrderSource
from vanta_api.websocket_notifier import WebSocketNotifierClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.position import Position
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.enums.miner_bucket_enum import MinerBucket
from shared_objects.locks.position_lock_client import PositionLockClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair, RPCConnectionMode
from shared_objects.cache_controller import CacheController
from shared_objects.subtensor_ops.metagraph_utils import is_anomalous_hotkey_loss
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.contract.contract_client import ContractClient
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.vali_dataclasses.price_source import PriceSource
from shared_objects.rpc.common_data_client import CommonDataClient
from entity_management.entity_utils import is_synthetic_hotkey
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.position_management.position_utils.position_utils import PositionUtils
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils


# ==================== Elimination Types ====================

class EliminationReason(Enum):
    """Reasons for miner elimination."""
    ZOMBIE = "ZOMBIE"
    INACTIVE = "INACTIVE"
    PLAGIARISM = "PLAGIARISM"
    MAX_TOTAL_DRAWDOWN = "MAX_TOTAL_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_TIME = "FAILED_CHALLENGE_PERIOD_TIME"
    FAILED_PROBATION_TIME = "FAILED_PROBATION_TIME"
    FAILED_CHALLENGE_PERIOD_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN"
    FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN = "FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN"
    FAILED_FUNDED_PERIOD_EOD_DRAWDOWN = "FAILED_FUNDED_PERIOD_EOD_DRAWDOWN"
    LIQUIDATED = "LIQUIDATED"
    DEREGISTERED = "DEREGISTERED"

    @property
    def is_intraday_drawdown(self):
        return self in (self.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN, self.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN)

    @property
    def is_eod_drawdown(self):
        return self in (self.FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN, self.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN)

# Constants for departed hotkeys tracking
DEPARTED_HOTKEYS_KEY = "departed_hotkeys"

@dataclass
class EliminationRow:
    hotkey: str
    reason: str
    elimination_initiated_time_ms: int
    elimination_drawdown_pct: float | None = None
    intraday_drawdown_pct: float | None = None
    eod_drawdown_pct: float | None = None
    bucket_at_elimination: MinerBucket | None = None
    positions_closed: bool = False  # Don't necesesarily have to track, but saves calls to position rpc
    row_added_ms: int | None = None

    def __post_init__(self):
        if self.bucket_at_elimination is not None:
            return

        if self.reason in (EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN.value,
                           EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN.value):
            self.bucket_at_elimination = MinerBucket.SUBACCOUNT_FUNDED
        elif self.reason in (EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN.value,
                           EliminationReason.FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN.value):
            self.bucket_at_elimination = MinerBucket.SUBACCOUNT_CHALLENGE
        elif self.reason in (EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value,
                             EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value):
            self.bucket_at_elimination = MinerBucket.CHALLENGE
        elif self.reason == EliminationReason.FAILED_PROBATION_TIME.value:
            self.bucket_at_elimination = MinerBucket.PROBATION
        elif self.reason == EliminationReason.PLAGIARISM.value:
            self.bucket_at_elimination = MinerBucket.PLAGIARISM

    @classmethod
    def from_dict(cls, d: dict) -> 'EliminationRow':
        raw_bucket = d.get('bucket_at_elimination')
        return cls(
            hotkey=d['hotkey'],
            reason=d['reason'],
            elimination_initiated_time_ms=d['elimination_initiated_time_ms'],
            elimination_drawdown_pct=d.get('elimination_drawdown_pct') or d.get('dd'),
            intraday_drawdown_pct=d.get('intraday_dd'),
            eod_drawdown_pct=d.get('eod_dd'),
            bucket_at_elimination=MinerBucket(raw_bucket) if raw_bucket is not None else None,
            positions_closed=d.get('positions_closed', False),
            row_added_ms=d.get('row_added_ms'),
        )

    def to_dict(self) -> dict:
        return {
            'hotkey': self.hotkey,
            'dd': self.elimination_drawdown_pct,  # TODO remove after dashboard update
            'elimination_drawdown_pct': self.elimination_drawdown_pct,
            'intraday_dd': self.intraday_drawdown_pct,
            'eod_dd': self.eod_drawdown_pct,
            'reason': self.reason,
            'elimination_initiated_time_ms': self.elimination_initiated_time_ms,
            'bucket_at_elimination': self.bucket_at_elimination.value if self.bucket_at_elimination is not None else None,
            'positions_closed': self.positions_closed,
            'row_added_ms': self.row_added_ms,
        }

    def pretty_str(self) -> str:
        """
        Formatted string for slack webhook
        X `<hotkey>`
        > ELIMINATION_REASON
        > Intraday drawdown: 0.00% | EOD drawdown: 0.00%
        > Elimination drawdown: 0.00%
        """
        lines = [
                f"`{self.hotkey}`",
                f"> *{self.reason}*"
                ]

        drawdown_lines = []
        if self.intraday_drawdown_pct is not None:
            drawdown_lines.append(f"Intraday drawdown: {self.intraday_drawdown_pct:.2f}%")
        if self.eod_drawdown_pct is not None:
            drawdown_lines.append(f"EOD drawdown: {self.eod_drawdown_pct:.2f}%")

        if drawdown_lines:
            lines.append(f"> {' | '.join(drawdown_lines)}")

        if self.elimination_drawdown_pct is not None:
            lines.append(f"> Elimination drawdown: {self.elimination_drawdown_pct:.2}%")

        return "\n".join(lines) + "\n"


# ==================== Manager Implementation ====================

class EliminationManager(CacheController):
    """
    Business logic manager for elimination processing.

    Contains the heavy business logic for managing eliminations,
    while EliminationServer wraps it and exposes methods via RPC.

    This follows the same pattern as PerfLedgerManager.

    ## Thread Safety and Lock Ordering

    This manager uses `self.eliminations_lock` (threading.Lock) to protect access
    to shared state: `self.eliminations` and `self.previous_metagraph_hotkeys`.

    ### Lock Type
    - `eliminations_lock` is a non-reentrant lock (threading.Lock)
    - Same thread acquiring twice causes deadlock
    - Private `_locked()` helpers assume lock is already held
    - Public methods acquire lock before calling helpers

    ### Lock Ordering Guarantees
    When multiple locks are needed, acquire in this order to prevent deadlock:
    1. `eliminations_lock` (EliminationManager)
    2. `position_lock` (PositionLockClient) - acquired via `get_lock(hotkey, trade_pair_id)`

    **NEVER acquire in reverse order** (position_lock → eliminations_lock causes circular wait)

    ### Lock Scope Minimization
    Locks are held ONLY during dict operations, never during I/O:
    - Dict reads/writes: Hold lock
    - Disk writes: Hold lock (prevents concurrent file corruption)
    - Network calls (RPC): NO lock
    - Heavy computation: NO lock

    ### No Nested Locks Within EliminationManager
    All methods acquire at most ONE lock (eliminations_lock).
    No method holds eliminations_lock while acquiring another lock.
    The only exception is add_manual_flat_order() which:
    1. Releases eliminations_lock (if held)
    2. Acquires position_lock
    This maintains lock ordering (eliminations → position).

    ### Two-Stage Cache Locking (Server)
    EliminationServer._refresh_cache() uses two-stage locking to avoid nested locks:
    1. Acquire manager's eliminations_lock → get snapshot → release
    2. Acquire cache's _cache_lock → update cache → release
    This prevents nested locking (manager lock inside cache lock).

    ### Locking Patterns

    **Pattern 1: Private _locked() Helpers**
    Private methods ending in `_locked()` assume caller holds eliminations_lock:
    - `_save_eliminations_locked()` - caller MUST hold lock
    - Public `save_eliminations()` acquires lock, calls `_save_eliminations_locked()`

    **Pattern 2: Snapshot for Iteration**
    To avoid RuntimeError during dict iteration, use snapshot pattern:
    ```python
    with self.eliminations_lock:
        snapshot = list(self.eliminations.values())
    # Iterate over snapshot (safe - dict can change without affecting iteration)
    for item in snapshot:
        process(item)
    ```

    **Pattern 3: Atomic Check-Then-Act**
    Prevent TOCTOU races by holding lock during both check and act:
    ```python
    with self.eliminations_lock:
        if condition_check():  # CHECK
            modify_state()      # ACT
            save()              # PERSIST
    ```

    **Pattern 4: Minimize Lock Hold Time**
    For expensive operations, split into: identify (short lock) → process (no lock):
    ```python
    # Step 1: Identify work (short lock)
    with self.eliminations_lock:
        items_to_process = [x for x in data if needs_processing(x)]

    # Step 2: Process work (no lock - I/O operations)
    for item in items_to_process:
        expensive_io_operation(item)
    ```

    ### Methods and Lock Usage

    **Locked Read Methods** (acquire lock for consistent snapshot):
    - `get_eliminated_hotkeys()` - returns set of hotkeys
    - `get_eliminations_from_memory()` - returns list of eliminations
    - `get_eliminations_dict()` - returns dict copy

    **Locked Write Methods** (acquire lock for atomic updates):
    - `append_elimination_row()` - add + save
    - `delete_eliminations()` - delete + save
    - `sync_eliminations()` - atomic clear + repopulate + save
    - `_update_departed_hotkeys()` - atomic read-modify-write of departed state

    **Locked Check-Then-Act** (prevent TOCTOU):
    - `is_hotkey_re_registered()` - check departed + check metagraph

    **Unlocked Methods** (no shared state or read-only):
    - `is_hotkey_eliminated()` - single dict lookup (GIL-protected)
    - `generate_elimination_row()` - pure function
    - `get_elimination()` - dict.get() + deepcopy (GIL-protected)
    """

    def __init__(
        self,
        is_backtesting=False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False,
        serve: bool = False
    ):
        """
        Initialize EliminationManager.

        Args:
            is_backtesting: Whether running in backtesting mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
            running_unit_tests: Whether running in test mode
        """
        self.serve = serve
        # Initialize CacheController (provides metagraph access)
        CacheController.__init__(
            self,
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            connection_mode=connection_mode
        )

        # Create own CommonDataClient (forward compatibility - no parameter passing)
        self._common_data_client = CommonDataClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Create own PerfLedgerClient (forward compatibility - no parameter passing)
        self._perf_ledger_client = PerfLedgerClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        # Create own PositionManagerClient (forward compatibility - no parameter passing)
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Create RPC client for ChallengePeriodManager
        self._challenge_period_client = ChallengePeriodClient(
            connection_mode=connection_mode
        )

        # Use LOCAL mode for WebSocketNotifier in tests (server not started in test mode)
        ws_connection_mode = RPCConnectionMode.LOCAL if running_unit_tests else connection_mode
        self.websocket_notifier_client = WebSocketNotifierClient(connection_mode=ws_connection_mode)
        self.live_price_fetcher_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        # Create own ContractClient (forward compatibility - no parameter passing)
        self._contract_client = ContractClient(
            port=ValiConfig.RPC_CONTRACTMANAGER_PORT,
            connect_immediately=False
        )

        self._position_lock_client = PositionLockClient()

        # Create own LimitOrderClient (forward compatibility - no parameter passing)
        self._limit_order_client = LimitOrderClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self._miner_account_client = MinerAccountClient(
            connect_immediately=False,
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests
        )

        # Create own EntityCollateralClient for slashing on elimination
        from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient
        self._entity_collateral_client = EntityCollateralClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        self.ELIMINATIONS_FILE = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        self.eliminations_lock = threading.Lock()
        self.eliminations: dict[str, EliminationRow] = self._load_eliminations_from_disk()

        if len(self.eliminations) == 0:
            ValiBkpUtils.write_file(self.ELIMINATIONS_FILE, {CacheController.ELIMINATIONS: []})

        # Migrate legacy departed_hotkeys.json into eliminations as DEREGISTERED rows
        departed_from_disk = self._get_departed_hotkeys_from_disk()
        for hotkey, info in departed_from_disk.items():
            if hotkey not in self.eliminations:
                detected_ms = info.get("detected_ms") if isinstance(info, dict) else None
                self.eliminations[hotkey] = EliminationRow(
                    hotkey=hotkey,
                    reason=EliminationReason.DEREGISTERED.value,
                    elimination_initiated_time_ms=detected_ms or TimeUtil.now_in_millis(),
                )

        # Track previous metagraph hotkeys to detect changes
        try:
            self.previous_metagraph_hotkeys = set(self._metagraph_client.get_hotkeys())
        except (AttributeError, RuntimeError):
            # MetagraphClient not connected yet (test mode without server setup)
            self.previous_metagraph_hotkeys = set()

        # TODO remove
        self.first_refresh_ran = False

        self.last_new_eliminations_checked_ms: int = TimeUtil.now_in_millis()

        bt.logging.info(f"[ELIM_MANAGER] EliminationManager initialized with {len(self.eliminations)} eliminations")

    # ==================== Pickle Prevention ====================

    def __getstate__(self):
        """
        Prevent manager from being pickled.

        Managers live inside server processes and should never be serialized.
        If this is called, it indicates an architectural issue where server-side
        objects are being pickled when they should stay in their own process.

        Raises:
            TypeError: Always, with stack trace for debugging
        """
        import traceback
        stack_trace = ''.join(traceback.format_stack())
        raise TypeError(
            f"{self.__class__.__name__} should not be pickled - it lives in a server process.\n"
            f"Managers contain RPC client objects and should never leave their server process.\n"
            f"\nStack trace showing where pickle was attempted:\n{stack_trace}"
        )

    # ==================== Properties ====================

    @property
    def perf_ledger_manager(self):
        """Get perf ledger client (forward compatibility - created internally)."""
        return self._perf_ledger_client

    @property
    def sync_in_progress(self):
        """Get sync_in_progress flag via CommonDataClient."""
        return self._common_data_client.get_sync_in_progress()

    @property
    def sync_epoch(self):
        """Get sync_epoch value via CommonDataClient."""
        return self._common_data_client.get_sync_epoch()

    # ==================== Core Business Logic ====================

    def get_eliminations_lock(self):
        """Get the local eliminations lock (manager-side only)"""
        return self.eliminations_lock

    def _close_position(
        self,
        hotkey: str,
        position: Position,
        position_close_ms: int | None,
        iteration_epoch: int | None = None
    ):
        """Add flat orders for eliminated miner"""
        if position.is_closed_position:
            return False

        if not position_close_ms:
            position_close_ms = TimeUtil.now_in_millis()

        with self._position_lock_client.get_lock(hotkey, position.trade_pair.trade_pair_id):
            position_refreshed = self._position_client.get_miner_position_by_uuid(hotkey, position.position_uuid)
            if position_refreshed is None:
                bt.logging.warning(
                    f"Unexpectedly could not find position with uuid {position.position_uuid} for hotkey {hotkey} "
                    f"and trade pair {position.trade_pair.trade_pair_id}. Not add flat orders"
                )
                return

            position = position_refreshed
            if position.is_closed_position:
                return

            fake_flat_order_time = position_close_ms
            if position.orders and position.orders[-1].processed_ms > position_close_ms:
                bt.logging.warning(
                    f'Unexpectedly found a position with a processed_ms {position.orders[-1].processed_ms} '
                    f'greater than the elimination time {position_close_ms}'
                )
                fake_flat_order_time = position.orders[-1].processed_ms + 1

            flat_order = Position.generate_fake_flat_order(
                    position, fake_flat_order_time, self.live_price_fetcher_client, src=OrderSource.ELIMINATION_FLAT)

            position.add_order(flat_order, self.live_price_fetcher_client)

            # Epoch-based validation
            if iteration_epoch is not None:
                current_epoch = self.sync_epoch
                if current_epoch != iteration_epoch:
                    bt.logging.warning(
                        f"Sync occurred during EliminationManager iteration for {hotkey} {position.trade_pair.trade_pair_id} "
                        f"(epoch {iteration_epoch} -> {current_epoch}). Skipping save to avoid data corruption"
                    )
                    return

            self._position_client.save_miner_position(position, delete_open_position_if_exists=True)
            if self.serve and self.websocket_notifier_client:
                self.websocket_notifier_client.broadcast_position_update(position)

            bt.logging.info(
                f'Added flat order for miner {hotkey} that has been eliminated. '
                f'Trade pair: {position.trade_pair.trade_pair_id}. flat order: {flat_order}. '
                f'position uuid {position.position_uuid}.'
            )



    def process_eliminations(self, iteration_epoch=None):
        """Main elimination processing loop"""
        try:
            # Check if we should process:
            # 1. Process if time-based refresh is due
            # 2. OR process if there are urgent challenge period eliminations
            if not self.first_refresh_ran:
                self.first_refresh_ran = True

            bt.logging.info(
                f"running elimination manager. invalidation data "
                f"{dict(self._perf_ledger_client.get_perf_ledger_hks_to_invalidate())}"
            )

            bt.logging.debug("[ELIM_PROCESS] Starting handle_mdd_eliminations")
            self.handle_mdd_eliminations()

            bt.logging.debug("[ELIM_PROCESS] Starting handle_zombies")
            self.handle_zombies()

            bt.logging.debug("[ELIM_PROCESS] Starting handle_idle_miners")
            self.handle_idle_miners()

            # Run after other checks so MDD/zombie/etc. take priority over DEREGISTERED
            bt.logging.debug("[ELIM_PROCESS] Starting handle_departed_hotkeys")
            self.handle_departed_hotkeys()

            bt.logging.debug("[ELIM_PROCESS] Starting _cleanup_eliminated_miners")
            self._cleanup_eliminated_miners(iteration_epoch)

            bt.logging.debug("[ELIM_PROCESS] Completed successfully")
            # self.set_last_update_time()

            return self._to_slack_message()

        except Exception as e:
            bt.logging.error(f"[ELIM_PROCESS] process_eliminations() failed with exception: {e}", exc_info=True)
            # Re-raise to let RPC framework handle it properly
            raise

    def is_zombie_hotkey(self, hotkey, all_hotkeys_set):
        """Check if hotkey is a zombie"""
        if hotkey == ValiConfig.DEVELOPMENT_HOTKEY or is_synthetic_hotkey(hotkey):
            return False
        return hotkey not in all_hotkeys_set

    def append_elimination_row(
        self,
        hotkey: str,
        reason: str,
        elimination_drawdown_pct: float | None = None,
        intraday_drawdown_pct: float | None = None,
        eod_drawdown_pct: float | None = None,
        elimination_time_ms: int | None = None,
        bucket_at_elimination: MinerBucket | None = None,
    ):
        """
        Add elimination row (thread-safe).

        Acquires eliminations_lock to ensure atomic check-update-save operation.
        """
        bt.logging.info(f"[ELIM_DEBUG] append_elimination_row called for {hotkey}, reason={reason}")
        if hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
            return

        if hotkey in self.eliminations:
            bt.logging.warning(f"[ELIM_DEBUG] Attempted to eliminate {hotkey} already in eliminations "
                               f"(original elimination: {self.eliminations[hotkey]})")
            return

        if not elimination_time_ms:
            elimination_time_ms = TimeUtil.now_in_millis()


        if bucket_at_elimination is None:
            bucket_at_elimination = self._challenge_period_client.get_miner_bucket(hotkey, elimination_time_ms)

        # Empty elimination drawdown for stuff zombie/inactive eliminations
        if elimination_drawdown_pct is None and intraday_drawdown_pct is None and eod_drawdown_pct is None:
            ledger = self.perf_ledger_manager.filtered_ledger_for_scoring([hotkey]).get(hotkey)
            _, elimination_drawdown_pct = LedgerUtils.is_beyond_max_drawdown(ledger)

        elimination_row = EliminationRow(
                hotkey=hotkey,
                reason=reason,
                elimination_initiated_time_ms=elimination_time_ms,
                elimination_drawdown_pct=elimination_drawdown_pct,
                intraday_drawdown_pct=intraday_drawdown_pct,
                eod_drawdown_pct=eod_drawdown_pct,
                bucket_at_elimination=bucket_at_elimination,
                row_added_ms=TimeUtil.now_in_millis(),
        )

        with self.eliminations_lock:
            dict_len_before = len(self.eliminations)
            self.eliminations[hotkey] = elimination_row
            dict_len_after = len(self.eliminations)
            bt.logging.info(f"[ELIM_DEBUG] Eliminations dict grew from {dict_len_before} to {dict_len_after} entries")

            # Save while holding lock to prevent concurrent disk writes
            self._save_eliminations_locked()

        # Slash on new eliminations
        bt.logging.info(f"miner eliminated with hotkey [{hotkey}]. Info [{elimination_row}]")
        is_subaccount = is_synthetic_hotkey(hotkey)
        if not is_subaccount:
            self._contract_client.slash_miner_collateral_proportion(hotkey)

        if is_subaccount and bucket_at_elimination in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            self._entity_collateral_client.try_slash_on_elimination(hotkey)

    def delete_eliminations(self, deleted_hotkeys):
        """
        Delete multiple eliminations (thread-safe).

        Acquires eliminations_lock to ensure atomic delete-save operation.
        """
        with self.eliminations_lock:
            for hotkey in deleted_hotkeys:
                if hotkey in self.eliminations:
                    del self.eliminations[hotkey]
            # Save while holding lock to prevent concurrent disk writes
            self._save_eliminations_locked()

    def handle_mdd_eliminations(self):
        """Check for MDD eliminations."""
        bt.logging.info("checking main competition for maximum drawdown eliminations.")

        # Get MAINCOMP hotkeys from cp_client
        regular_miner_buckets = [bucket for bucket in MinerBucket if bucket.is_regular_miner]
        miner_hotkeys = self._challenge_period_client.get_hotkeys_by_bucket(regular_miner_buckets)

        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(
            hotkeys=miner_hotkeys
        )
        for miner_hotkey, ledger in filtered_ledger.items():
            if miner_hotkey in self.eliminations:
                continue

            miner_exceeds_mdd, drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element=ledger)

            if miner_exceeds_mdd:
                bt.logging.info(f"[ELIMINATION] {miner_hotkey} mdd={drawdown_percentage:.2f}%, mdd elimination disabled for now")
                # TODO enable mdd
                # self.append_elimination_row(miner_hotkey, EliminationReason.MAX_TOTAL_DRAWDOWN.value, elimination_drawdown_pct=drawdown_percentage)

    def handle_zombies(self):
        """Handle zombie miners"""

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        all_hotkeys_set = set(self._metagraph_client.get_hotkeys())

        for hotkey in CacheController.get_directory_names(all_miners_dir):
            corresponding_elimination = self.eliminations.get(hotkey)
            elimination_reason = corresponding_elimination.reason if corresponding_elimination else None
            if elimination_reason:
                continue
            elif self.is_zombie_hotkey(hotkey, all_hotkeys_set):
                self.append_elimination_row(hotkey=hotkey, reason=EliminationReason.ZOMBIE.value)

    def handle_idle_miners(self):
        """Eliminate MAINCOMP, CHALLENGE, and PROBATION miners that have not submitted any orders in the past 60 days."""
        now_ms = TimeUtil.now_in_millis()
        IDLE_ELIMINATION_ACTIVATION_MS = TimeUtil.formatted_date_str_to_millis("2026-06-08 00:00:00")

        bt.logging.info("checking all active buckets for inactive miner eliminations.")

        active_buckets = [MinerBucket.MAINCOMP, MinerBucket.CHALLENGE, MinerBucket.PROBATION,
                          MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA]

        candidate_hotkeys = set()
        candidate_hotkeys.update(self._challenge_period_client.get_hotkeys_by_bucket(active_buckets))

        if not candidate_hotkeys:
            return set()

        hotkey_to_positions = self._position_client.get_positions_for_hotkeys(
            list(candidate_hotkeys), only_open_positions=False
        )

        idle_threshold_ms = ValiConfig.IDLE_MINER_MAXIMUM_MS
        near_idle_threshold_ms = idle_threshold_ms * 0.9
        idle_hotkeys = {}
        near_idle_hotkeys = {}

        bt.logging.info(f"Inactive miner cutoff: {TimeUtil.millis_to_formatted_date_str(now_ms - idle_threshold_ms)}")
        for hotkey, positions in hotkey_to_positions.items():
            if hotkey in self.eliminations:
                continue

            orders = PositionUtils.flatten_orders(positions)
            last_order_ms = max(orders, key=lambda o: o.processed_ms).processed_ms if orders else None

            if last_order_ms is not None and (now_ms - last_order_ms) > idle_threshold_ms:
                idle_hotkeys[hotkey] = last_order_ms
                bt.logging.info(
                    f"inactive miner detected: {hotkey}. Last order: {last_order_ms} ({TimeUtil.millis_to_formatted_date_str(last_order_ms)})"
                )
            elif last_order_ms is not None and (now_ms - last_order_ms) > near_idle_threshold_ms:
                near_idle_hotkeys[hotkey] = last_order_ms
                bt.logging.info(
                    f"Miner approaching inactive threshold: {hotkey}. Last order: {last_order_ms} ({TimeUtil.millis_to_formatted_date_str(last_order_ms)})"
                )

        if now_ms < IDLE_ELIMINATION_ACTIVATION_MS:
            return

        for hotkey, last_order_ms in idle_hotkeys.items():
            bt.logging.info(f"Eliminating inactive miner {hotkey}.")
            self.append_elimination_row(hotkey=hotkey, reason=EliminationReason.INACTIVE.value, elimination_time_ms=last_order_ms+idle_threshold_ms)


    def handle_departed_hotkeys(self):
        """
        Track departed hotkeys (thread-safe).

        Acquires eliminations_lock to ensure atomic read-modify-write of departed_hotkeys
        and previous_metagraph_hotkeys state. After the lock is released, issues
        DEREGISTERED eliminations for any newly departed hotkeys.
        """
        if self.is_backtesting:
            return

        new_departures = set()
        with self.eliminations_lock:
            current_hotkeys = set(self._metagraph_client.get_hotkeys())
            lost_hotkeys = self.previous_metagraph_hotkeys - current_hotkeys
            gained_hotkeys = current_hotkeys - self.previous_metagraph_hotkeys

            if lost_hotkeys:
                bt.logging.debug(f"Metagraph lost hotkeys: {lost_hotkeys}")
            if gained_hotkeys:
                bt.logging.debug(f"Metagraph gained hotkeys: {gained_hotkeys}")

            deregistered_set = {hk for hk, row in self.eliminations.items() if row.reason == EliminationReason.DEREGISTERED.value}
            re_registered_hotkeys = gained_hotkeys & deregistered_set
            if re_registered_hotkeys:
                bt.logging.warning(
                    f"Detected {len(re_registered_hotkeys)} re-registered miners: {re_registered_hotkeys}. "
                    f"These hotkeys were previously de-registered and have re-registered. Their orders will be rejected."
                )

            is_anomalous, _ = is_anomalous_hotkey_loss(lost_hotkeys, len(self.previous_metagraph_hotkeys))
            if lost_hotkeys and not is_anomalous:
                new_departures = lost_hotkeys - set(self.eliminations.keys())
                if new_departures:
                    bt.logging.info(
                        f"Tracked {len(new_departures)} newly departed hotkeys: {new_departures}. "
                        f"Total deregistered: {len(deregistered_set) + len(new_departures)}"
                    )
            elif lost_hotkeys:
                bt.logging.warning(
                    f"Detected anomalous metagraph change: {len(lost_hotkeys)} hotkeys lost "
                    f"({100 * len(lost_hotkeys) / len(self.previous_metagraph_hotkeys):.1f}% of total). "
                    f"Not tracking as departed to avoid false positives."
                )

            self.previous_metagraph_hotkeys = current_hotkeys

        # Issue DEREGISTERED eliminations after releasing the lock (append_elimination_row acquires it)
        for hotkey in new_departures:
            self.append_elimination_row(hotkey, EliminationReason.DEREGISTERED.value)

    def _cleanup_eliminated_miners(self, iteration_epoch: int | None = None):
        """post-elimination cleanup: close open positions and delete unfilled limit orders"""
        now_ms = TimeUtil.now_in_millis()

        with self.eliminations_lock:
            eliminations_snapshot = list(self.eliminations.values())

        write_to_disk = False
        for elimination_data in eliminations_snapshot:
            if not elimination_data.positions_closed:
                hotkey = elimination_data.hotkey

                deleted_limit_orders = self._limit_order_client.delete_all_limit_orders_for_hotkey(hotkey)
                if deleted_limit_orders:
                    bt.logging.info(f"Cancelled {deleted_limit_orders} pending limit orders for eliminated miner [{hotkey}]")

                positions = self._position_client.get_positions_for_one_hotkey(hotkey, only_open_positions=True)
                for p in positions:
                    self._close_position(hotkey, p, elimination_data.elimination_initiated_time_ms, iteration_epoch=iteration_epoch)

                with self.eliminations_lock:
                    self.eliminations[hotkey].positions_closed = True
                    write_to_disk = True

            self._purge_expired_elimination(elimination_data, now_ms)

        if write_to_disk:
            self.save_eliminations()

    def _purge_expired_elimination(self, elimination_data, now_ms):
        """Delete position files and miner directory after the retention window expires.
        Skips funded subaccounts to preserve accrued realized pnl.
        """
        if elimination_data.bucket_at_elimination in (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            return

        is_expired = now_ms - elimination_data.elimination_initiated_time_ms >= ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS
        if not is_expired:
            return

        hotkey = elimination_data.hotkey
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests) + hotkey
        positions = self._position_client.get_positions_for_one_hotkey(hotkey)
        for p in positions:
            self._position_client.delete_position(p.miner_hotkey, p.position_uuid)

        try:
            shutil.rmtree(miner_dir)
        except FileNotFoundError:
            pass

    def _get_departed_hotkeys_from_disk(self) -> dict:
        """Load departed hotkeys from disk (legacy) and default file for migration."""
        import os
        location = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=self.running_unit_tests)
        try:
            departed_data = ValiUtils.get_vali_json_file(location, DEPARTED_HOTKEYS_KEY)
            if departed_data is None:
                departed_data = {}
            if isinstance(departed_data, list):
                departed_data = {hotkey: {"detected_ms": 0} for hotkey in departed_data}
            if departed_data:
                bt.logging.info(f"Migrating {len(departed_data)} departed hotkeys from disk into eliminations")
            return departed_data
        except Exception:
            pass

        base_dir = ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests).replace('/validation/', '')
        default_location = os.path.join(base_dir, 'data', 'default_departed_hotkeys.json')
        try:
            departed_data = ValiUtils.get_vali_json_file(default_location, DEPARTED_HOTKEYS_KEY)
            if departed_data is None:
                departed_data = {}
            if isinstance(departed_data, list):
                departed_data = {hotkey: {"detected_ms": 0} for hotkey in departed_data}
            if departed_data:
                bt.logging.info(f"Migrating {len(departed_data)} departed hotkeys from default file into eliminations")
            return departed_data
        except Exception:
            return {}

    # ==================== Query Methods (used by Server) ====================

    def is_hotkey_eliminated(self, hotkey: str) -> bool:
        """Fast existence check (O(1))"""
        return hotkey in self.eliminations

    def get_elimination(self, hotkey: str) -> dict | None:
        """Get full elimination details"""
        elimination = self.eliminations.get(hotkey)
        return elimination.to_dict() if elimination else None

    def get_dashboard(self, hotkey: str) -> dict | None:
        elimination = self.eliminations.get(hotkey)
        if elimination is None:
            return None
        return {
            "elimination_initiated_time_ms": elimination.elimination_initiated_time_ms,
            "reason": elimination.reason,
            "dd": elimination.elimination_drawdown_pct,
        }

    def get_eliminated_hotkeys(self) -> set[str]:
        """
        Get all eliminated hotkeys (thread-safe).

        Returns a consistent snapshot of eliminated hotkeys.
        """
        with self.eliminations_lock:
            return set(self.eliminations.keys())

    def get_eliminated_hotkeys_by_bucket(self, buckets: list[MinerBucket]) -> set[str]:
        """
        Get eliminated hotkeys whose bucket_at_elimination is in the given list (thread-safe).
        """
        bucket_set = set(buckets)
        with self.eliminations_lock:
            return {hk for hk, row in self.eliminations.items() if row.bucket_at_elimination in bucket_set}

    def get_eliminations_from_memory(self) -> list[dict]:
        """
        Get all eliminations as a list (thread-safe).

        Returns a consistent snapshot of all elimination records.
        """
        with self.eliminations_lock:
            return [v.to_dict() for v in self.eliminations.values()]

    def remove_elimination(self, hotkey: str) -> bool:
        """Remove elimination. Returns True if removed, False if not found."""
        if hotkey in self.eliminations:
            del self.eliminations[hotkey]
            return True
        return False

    def sync_eliminations(self, eliminations_list: list) -> list:
        """
        Sync eliminations from external source (batch update, thread-safe).

        Acquires eliminations_lock to ensure atomic clear-repopulate-save operation.
        This prevents readers from seeing an empty dict during the sync window.

        Returns:
            list of removed hotkeys
        """
        with self.eliminations_lock:
            hotkeys_before = set(self.eliminations.keys())
            hotkeys_after = set(x['hotkey'] for x in eliminations_list)
            removed = [x for x in hotkeys_before if x not in hotkeys_after]
            added = [x for x in hotkeys_after if x not in hotkeys_before]

            bt.logging.info(f'sync_eliminations: removed {len(removed)} {removed}, added {len(added)} {added}')

            # Atomic batch update (clear + repopulate while holding lock)
            self.eliminations.clear()
            for elim in eliminations_list:
                hotkey = elim['hotkey']
                self.eliminations[hotkey] = EliminationRow.from_dict(elim)

            # Save while holding lock to prevent concurrent disk writes
            self._save_eliminations_locked()
            return removed

    def clear_eliminations(self) -> None:
        """Clear all eliminations for testing"""
        if not self.running_unit_tests:
            raise Exception('clear_eliminations can only be called during unit tests')
        ValiBkpUtils.write_file(self.ELIMINATIONS_FILE, {CacheController.ELIMINATIONS: []})
        self.eliminations.clear()

    def clear_departed_hotkeys(self) -> None:
        """Clear all DEREGISTERED eliminations for testing"""
        if not self.running_unit_tests:
            raise Exception('clear_departed_hotkeys can only be called during unit tests')
        with self.eliminations_lock:
            deregistered = [hk for hk, row in self.eliminations.items() if row.reason == EliminationReason.DEREGISTERED.value]
            for hk in deregistered:
                del self.eliminations[hk]
            self._save_eliminations_locked()
        try:
            self.previous_metagraph_hotkeys = set(self._metagraph_client.get_hotkeys())
        except (AttributeError, RuntimeError):
            self.previous_metagraph_hotkeys = set()

    def is_hotkey_re_registered(self, hotkey: str) -> bool:
        """Check if hotkey was deregistered and is now back in the metagraph (thread-safe)."""
        if not hotkey:
            return False
        with self.eliminations_lock:
            row = self.eliminations.get(hotkey)
            if not row or row.reason != EliminationReason.DEREGISTERED.value:
                return False
            return self._metagraph_client.has_hotkey(hotkey)

    def get_departed_hotkeys(self) -> dict[str, dict]:
        """Get all DEREGISTERED eliminations as {hotkey: {detected_ms: ...}}"""
        with self.eliminations_lock:
            return {
                hk: {"detected_ms": row.elimination_initiated_time_ms}
                for hk, row in self.eliminations.items()
                if row.reason == EliminationReason.DEREGISTERED.value
            }

    def get_departed_hotkey_info(self, hotkey: str) -> dict | None:
        """Get departure info for a single hotkey."""
        with self.eliminations_lock:
            row = self.eliminations.get(hotkey)
            if row and row.reason == EliminationReason.DEREGISTERED.value:
                return {"detected_ms": row.elimination_initiated_time_ms}
            return None

    def get_eliminations_dict(self) -> dict[str, dict]:
        """
        Get eliminations dict (copy, thread-safe).

        Returns a consistent snapshot of the eliminations dictionary.
        """
        with self.eliminations_lock:
            return {k: v.to_dict() for k, v in self.eliminations.items()}

    def _load_eliminations_from_disk(self) -> dict[str, EliminationRow]:
        """Load eliminations from disk, returning {hotkey: EliminationRow}."""
        if self.is_backtesting:
            return {}
        try:
            data = ValiUtils.get_vali_json_file(self.ELIMINATIONS_FILE, CacheController.ELIMINATIONS)
            if not data:
                return {}
            result = {e['hotkey']: EliminationRow.from_dict(e) for e in data}
            bt.logging.trace(f"Loaded {len(result)} eliminations from disk")
            return result
        except Exception as e:
            bt.logging.warning(f"Could not load eliminations from disk: {e}. Starting with empty dict.")
            return {}

    def _save_eliminations_to_disk(self, eliminations) -> None:
        """Write eliminations list to disk."""
        if self.is_backtesting:
            return
        if not isinstance(eliminations, list):
            eliminations = list(eliminations)
        ValiBkpUtils.write_file(self.ELIMINATIONS_FILE, {CacheController.ELIMINATIONS: eliminations})
        bt.logging.info(f"[ELIM_DEBUG] Successfully wrote {len(eliminations)} eliminations to disk")

    def _save_eliminations_locked(self):
        """Save eliminations to disk (caller MUST hold eliminations_lock)."""
        self._save_eliminations_to_disk([v.to_dict() for v in self.eliminations.values()])

    def save_eliminations(self):
        """Save eliminations to disk (thread-safe)."""
        with self.eliminations_lock:
            self._save_eliminations_locked()

    def write_eliminations_to_disk(self, eliminations):
        """Write arbitrary eliminations list to disk (public API for restore scripts)."""
        self._save_eliminations_to_disk(eliminations)

    def _to_slack_message(self) -> str | None:
        since_ms = self.last_new_eliminations_checked_ms
        self.last_new_eliminations_checked_ms = TimeUtil.now_in_millis()

        recent = [
            row for row in self.eliminations.values()
            if row.row_added_ms is not None and row.row_added_ms >= since_ms
        ]
        if not recent:
            return None

        msg = f":x: *Eliminations!* (since {TimeUtil.millis_to_formatted_date_str(since_ms)})\n"
        msg += "\n".join(row.pretty_str() for row in recent)

        return msg
