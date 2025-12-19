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
import shutil
import threading
from copy import deepcopy
from enum import Enum
from typing import Dict, Set, List, Optional

import bittensor as bt

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


# ==================== Elimination Types ====================

class EliminationReason(Enum):
    """Reasons for miner elimination."""
    ZOMBIE = "ZOMBIE"
    PLAGIARISM = "PLAGIARISM"
    MAX_TOTAL_DRAWDOWN = "MAX_TOTAL_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_TIME = "FAILED_CHALLENGE_PERIOD_TIME"
    FAILED_CHALLENGE_PERIOD_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_DRAWDOWN"
    LIQUIDATED = "LIQUIDATED"


# Constants for departed hotkeys tracking
DEPARTED_HOTKEYS_KEY = "departed_hotkeys"


# ==================== Manager Implementation ====================

class EliminationManager(CacheController):
    """
    Business logic manager for elimination processing.

    Contains the heavy business logic for managing eliminations,
    while EliminationServer wraps it and exposes methods via RPC.

    This follows the same pattern as PerfLedgerManager.

    ## Thread Safety and Lock Ordering

    This manager uses `self.eliminations_lock` (threading.Lock) to protect access
    to shared state: `self.eliminations`, `self.departed_hotkeys`, and
    `self.previous_metagraph_hotkeys`.

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
    - `handle_first_refresh()` - check + set first_refresh_ran flag
    - `handle_challenge_period_eliminations()` - check exists + add elimination

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
        self.cp_client = ChallengePeriodClient(
            connection_mode=connection_mode
        )

        self.first_refresh_ran = False
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

        # Local dicts (no IPC) - much faster!
        self.eliminations: Dict[str, dict] = {}
        self.departed_hotkeys: Dict[str, dict] = {}
        self.eliminations_lock = threading.Lock()

        # Populate from disk, filtering out development hotkey
        eliminations_from_disk = self.get_eliminations_from_disk()
        filtered_count = 0
        for elim in eliminations_from_disk:
            hotkey = elim['hotkey']
            # Skip development hotkey - it should never be eliminated
            if hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
                filtered_count += 1
                bt.logging.debug(f"[ELIM_INIT] Filtered out DEVELOPMENT_HOTKEY from eliminations during disk load")
                continue
            self.eliminations[hotkey] = elim

        if filtered_count > 0:
            bt.logging.info(f"[ELIM_INIT] Filtered out {filtered_count} DEVELOPMENT_HOTKEY elimination(s) from disk load")

        if len(self.eliminations) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                {CacheController.ELIMINATIONS: []}
            )

        # Initialize departed hotkeys tracking
        self.departed_hotkeys.update(self._get_departed_hotkeys_from_disk())
        if len(self.departed_hotkeys) == 0:
            self._save_departed_hotkeys()

        # Track previous metagraph hotkeys to detect changes
        try:
            self.previous_metagraph_hotkeys = set(self._metagraph_client.get_hotkeys())
        except (AttributeError, RuntimeError):
            # MetagraphClient not connected yet (test mode without server setup)
            self.previous_metagraph_hotkeys = set()

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

    def generate_elimination_row(self, hotkey, current_dd, reason, t_ms=None, price_info=None, return_info=None):
        """Generate elimination row dict."""
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()
        return {
            'hotkey': hotkey,
            'dd': current_dd,
            'reason': reason,
            'elimination_initiated_time_ms': t_ms,
            'price_info': price_info or {},
            'return_info': return_info or {}
        }

    def handle_perf_ledger_eliminations(self, iteration_epoch=None):
        """
        Process performance ledger eliminations (thread-safe).

        Identifies new eliminations, adds them atomically, then handles cleanup.
        Lock scope is minimized - only held during dict operations, not I/O.
        """
        perf_ledger_eliminations = self.perf_ledger_manager.get_perf_ledger_eliminations()

        # Step 1: Identify new eliminations (short lock for read)
        new_eliminations = []
        with self.eliminations_lock:
            for e in perf_ledger_eliminations:
                if e['hotkey'] not in self.eliminations:
                    new_eliminations.append(e)

        if not new_eliminations:
            return

        # Step 2: Add all new eliminations atomically (short lock for batch write)
        with self.eliminations_lock:
            for e in new_eliminations:
                # Double-check (another thread may have added it between step 1 and step 2)
                if e['hotkey'] not in self.eliminations:
                    self.eliminations[e['hotkey']] = e
            # Batch save while holding lock
            self._save_eliminations_locked()

        bt.logging.info(f'Wrote {len(new_eliminations)} perf ledger eliminations to disk')

        # Step 3: Handle cleanup outside lock (I/O operations - no lock needed)
        for e in new_eliminations:
            price_info = e['price_info']
            trade_pair_to_price_source_used_for_elimination_check = {}
            for k, v in price_info.items():
                trade_pair = TradePair.get_latest_tade_pair_from_trade_pair_str(k)
                elimination_initiated_time_ms = e['elimination_initiated_time_ms']
                trade_pair_to_price_source_used_for_elimination_check[trade_pair] = PriceSource(
                    source='elim', open=v, close=v,
                    start_ms=elimination_initiated_time_ms,
                    timespan_ms=1000, websocket=False
                )
            self.handle_eliminated_miner(e['hotkey'], trade_pair_to_price_source_used_for_elimination_check,
                                        iteration_epoch)
            # Skip slashing in test mode (no contract manager)
            if self._contract_client:
                self._contract_client.slash_miner_collateral_proportion(e['hotkey'])

    def add_manual_flat_order(self, hotkey: str, position: Position, corresponding_elimination,
                             source_for_elimination, iteration_epoch=None):
        """Add flat orders for eliminated miner"""
        elimination_time_ms = corresponding_elimination['elimination_initiated_time_ms'] if corresponding_elimination else TimeUtil.now_in_millis()
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

            fake_flat_order_time = elimination_time_ms
            if position.orders and position.orders[-1].processed_ms > elimination_time_ms:
                bt.logging.warning(
                    f'Unexpectedly found a position with a processed_ms {position.orders[-1].processed_ms} '
                    f'greater than the elimination time {elimination_time_ms}'
                )
                fake_flat_order_time = position.orders[-1].processed_ms + 1

            flat_order = Position.generate_fake_flat_order(position, fake_flat_order_time,
                                                           self.live_price_fetcher_client, source_for_elimination)
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
                f'position uuid {position.position_uuid}. Source for elimination {source_for_elimination}'
            )

    def handle_eliminated_miner(self, hotkey: str,
                                trade_pair_to_price_source_used_for_elimination_check: Dict[TradePair, PriceSource],
                                iteration_epoch=None):
        """Handle cleanup for eliminated miner"""
        # Clean up limit orders using internal LimitOrderClient (forward compatibility)
        result = self._limit_order_client.delete_all_limit_orders_for_hotkey(hotkey)
        bt.logging.info(f"Cleaned up limit orders for eliminated miner [{hotkey}]: {result}")

        for p in self._position_client.get_positions_for_one_hotkey(hotkey, only_open_positions=True):
            source_for_elimination = trade_pair_to_price_source_used_for_elimination_check.get(p.trade_pair)
            corresponding_elimination = self.eliminations.get(hotkey)
            if corresponding_elimination:
                self.add_manual_flat_order(hotkey, p, corresponding_elimination,
                                          source_for_elimination, iteration_epoch)

    def handle_challenge_period_eliminations(self, iteration_epoch=None):
        """
        Process challenge period eliminations (thread-safe).

        Atomically checks and adds eliminations to prevent redundant processing.
        Lock scope is minimized - only held during dict operations, not I/O.
        """
        # Check if there are any eliminations to process
        if not self.cp_client.has_elimination_reasons():
                return
        eliminations_snapshot = self.cp_client.get_all_elimination_reasons()

        hotkeys = list(eliminations_snapshot.keys())

        if not hotkeys:
            return

        bt.logging.info(f"[ELIM_DEBUG] Processing {len(hotkeys)} challenge period eliminations: {hotkeys}")
        bt.logging.info(f"[ELIM_DEBUG] Current eliminations dict has {len(self.eliminations)} entries")

        # Collect eliminations that were successfully added
        newly_added_eliminations = []  # [(hotkey, elim_reason, elim_mdd), ...]

        # Process each hotkey individually, popping atomically to avoid race conditions
        for hotkey in hotkeys:
            # Atomically pop the elimination reason (get + remove in one operation)
            elim_data = self.cp_client.pop_elimination_reason(hotkey)
            # Skip if already removed (another thread might have processed it)
            if elim_data is None:
                bt.logging.debug(f"[ELIM_DEBUG] Hotkey {hotkey} already processed/removed")
                continue

            elim_reason = elim_data[0]
            elim_mdd = elim_data[1]

            # Atomic check-then-add: Lock prevents another thread from adding
            # the same elimination between check and add
            with self.eliminations_lock:
                already_eliminated = hotkey in self.eliminations
                if already_eliminated:
                    bt.logging.warning(
                        f"[ELIM_DEBUG] Hotkey {hotkey} is ALREADY in eliminations list. Skipping. "
                        f"Elimination: {self.eliminations[hotkey]}"
                    )
                    continue

                # Add elimination directly (we're already holding the lock)
                bt.logging.info(f"[ELIM_DEBUG] Adding new elimination for {hotkey}")
                elimination_row = self.generate_elimination_row(hotkey, elim_mdd, elim_reason)
                self.eliminations[hotkey] = elimination_row
                # Save while holding lock
                self._save_eliminations_locked()

                # Track that we successfully added this elimination
                newly_added_eliminations.append((hotkey, elim_reason, elim_mdd))

                bt.logging.info(f"[ELIM_DEBUG] Verified {hotkey} was added to eliminations list")

        bt.logging.info(f"[ELIM_DEBUG] After processing, eliminations dict has {len(self.eliminations)} entries")

        # Handle cleanup outside lock (I/O operations - only for newly added eliminations)
        for hotkey, elim_reason, elim_mdd in newly_added_eliminations:
            self.handle_eliminated_miner(hotkey, {}, iteration_epoch)
            # Skip slashing in test mode (no contract manager)
            if self._contract_client:
                self._contract_client.slash_miner_collateral_proportion(hotkey)

    def handle_first_refresh(self, iteration_epoch=None):
        """
        Handle first refresh on startup (thread-safe).

        Acquires eliminations_lock to ensure atomic check-set of first_refresh_ran flag
        and consistent snapshot of eliminated_hotkeys.
        """
        if self.is_backtesting:
            return

        # Atomic check-then-set of first_refresh_ran flag
        with self.eliminations_lock:
            if self.first_refresh_ran:
                return
            self.first_refresh_ran = True
            # Get snapshot of eliminated hotkeys while holding lock
            eliminated_hotkeys = list(self.eliminations.keys())

        # Process outside lock (I/O operations don't need lock)
        hotkey_to_positions = self._position_client.get_positions_for_hotkeys(eliminated_hotkeys,
                                                                              only_open_positions=True)
        for hotkey, open_positions in hotkey_to_positions.items():
            if not open_positions:
                continue
            for p in open_positions:
                self.add_manual_flat_order(hotkey, p, self.eliminations.get(hotkey), None, iteration_epoch)

    def process_eliminations(self, iteration_epoch=None):
        """Main elimination processing loop"""
        try:
            # Check if we should process:
            # 1. Process if time-based refresh is due
            # 2. OR process if there are urgent challenge period eliminations
            refresh_due = self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS)

            # Check for urgent eliminations using cp_client
            has_urgent_eliminations = self.cp_client.has_elimination_reasons()

            if not refresh_due and not has_urgent_eliminations:
                return

            bt.logging.info(
                f"running elimination manager. invalidation data "
                f"{dict(self._perf_ledger_client.get_perf_ledger_hks_to_invalidate())}"
            )

            bt.logging.debug("[ELIM_PROCESS] Starting _update_departed_hotkeys")
            self._update_departed_hotkeys()

            bt.logging.debug("[ELIM_PROCESS] Starting handle_first_refresh")
            self.handle_first_refresh(iteration_epoch)

            bt.logging.debug("[ELIM_PROCESS] Starting handle_perf_ledger_eliminations")
            self.handle_perf_ledger_eliminations(iteration_epoch)

            bt.logging.debug("[ELIM_PROCESS] Starting handle_challenge_period_eliminations")
            self.handle_challenge_period_eliminations(iteration_epoch)

            bt.logging.debug("[ELIM_PROCESS] Starting handle_mdd_eliminations")
            self.handle_mdd_eliminations(iteration_epoch)

            bt.logging.debug("[ELIM_PROCESS] Starting handle_zombies")
            self.handle_zombies(iteration_epoch)

            bt.logging.debug("[ELIM_PROCESS] Starting _delete_eliminated_expired_miners")
            self._delete_eliminated_expired_miners()

            bt.logging.debug("[ELIM_PROCESS] Completed successfully")
            self.set_last_update_time()
        except Exception as e:
            bt.logging.error(f"[ELIM_PROCESS] process_eliminations() failed with exception: {e}", exc_info=True)
            # Re-raise to let RPC framework handle it properly
            raise

    def is_zombie_hotkey(self, hotkey, all_hotkeys_set):
        """Check if hotkey is a zombie"""
        if hotkey == ValiConfig.DEVELOPMENT_HOTKEY or is_synthetic_hotkey(hotkey):
            return False
        return hotkey not in all_hotkeys_set

    def _save_eliminations_locked(self):
        """
        PRIVATE: Save eliminations to disk (caller MUST hold eliminations_lock).

        This is a private helper method. Callers must acquire self.eliminations_lock
        before calling this method to ensure atomic read-then-write to disk.
        """
        if not self.is_backtesting:
            self.write_eliminations_to_disk(list(self.eliminations.values()))

    def save_eliminations(self):
        """
        PUBLIC: Save eliminations to disk (thread-safe).

        Acquires eliminations_lock to ensure atomic read-then-write.
        """
        with self.eliminations_lock:
            self._save_eliminations_locked()

    def write_eliminations_to_disk(self, eliminations):
        """Write eliminations to disk"""
        if not isinstance(eliminations, list):
            eliminations = list(eliminations)
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, vali_eliminations)
        bt.logging.info(f"[ELIM_DEBUG] Successfully wrote {len(eliminations)} eliminations to disk")

    def get_eliminations_from_disk(self) -> list:
        """Load eliminations from disk"""
        location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        try:
            cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
            if cached_eliminations is None:
                cached_eliminations = []
            bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
            return cached_eliminations
        except Exception as e:
            bt.logging.warning(f"Could not load eliminations from disk: {e}. Starting with empty list.")
            return []

    def append_elimination_row(self, hotkey, current_dd, reason, t_ms=None, price_info=None, return_info=None):
        """
        Add elimination row (thread-safe).

        Acquires eliminations_lock to ensure atomic check-update-save operation.
        """
        bt.logging.info(f"[ELIM_DEBUG] append_elimination_row called for {hotkey}, reason={reason}")
        elimination_row = self.generate_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                                        price_info=price_info, return_info=return_info)

        with self.eliminations_lock:
            dict_len_before = len(self.eliminations)
            self.eliminations[hotkey] = elimination_row
            dict_len_after = len(self.eliminations)
            bt.logging.info(f"[ELIM_DEBUG] Eliminations dict grew from {dict_len_before} to {dict_len_after} entries")

            # Save while holding lock to prevent concurrent disk writes
            self._save_eliminations_locked()

        bt.logging.info(f"miner eliminated with hotkey [{hotkey}]. Info [{elimination_row}]")

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

    def handle_mdd_eliminations(self, iteration_epoch=None):
        """Check for MDD eliminations."""
        from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
        bt.logging.info("checking main competition for maximum drawdown eliminations.")

        # Get MAINCOMP hotkeys from cp_client
        challengeperiod_success_hotkeys = self.cp_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)

        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(
            portfolio_only=True, hotkeys=challengeperiod_success_hotkeys
        )
        for miner_hotkey, ledger in filtered_ledger.items():
            if miner_hotkey in self.eliminations:
                continue

            miner_exceeds_mdd, drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element=ledger)

            if miner_exceeds_mdd:
                self.append_elimination_row(miner_hotkey, drawdown_percentage, EliminationReason.MAX_TOTAL_DRAWDOWN.value)
                self.handle_eliminated_miner(miner_hotkey, {}, iteration_epoch)
                # Skip slashing in test mode (no contract manager)
                if self._contract_client:
                    self._contract_client.slash_miner_collateral_proportion(miner_hotkey)

    def handle_zombies(self, iteration_epoch=None):
        """Handle zombie miners"""

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        all_hotkeys_set = set(self._metagraph_client.get_hotkeys())

        for hotkey in CacheController.get_directory_names(all_miners_dir):
            corresponding_elimination = self.eliminations.get(hotkey)
            elimination_reason = corresponding_elimination.get('reason') if corresponding_elimination else None
            if elimination_reason:
                continue
            elif self.is_zombie_hotkey(hotkey, all_hotkeys_set):
                self.append_elimination_row(hotkey=hotkey, current_dd=None, reason=EliminationReason.ZOMBIE.value)
                self.handle_eliminated_miner(hotkey, {}, iteration_epoch)

    def _update_departed_hotkeys(self):
        """
        Track departed hotkeys (thread-safe).

        Acquires eliminations_lock to ensure atomic read-modify-write of departed_hotkeys
        and previous_metagraph_hotkeys state.
        """
        if self.is_backtesting:
            return

        # Acquire lock for entire operation (reads previous state, modifies departed_hotkeys, updates previous state)
        with self.eliminations_lock:
            current_hotkeys = set(self._metagraph_client.get_hotkeys())
            lost_hotkeys = self.previous_metagraph_hotkeys - current_hotkeys
            gained_hotkeys = current_hotkeys - self.previous_metagraph_hotkeys

            if lost_hotkeys:
                bt.logging.debug(f"Metagraph lost hotkeys: {lost_hotkeys}")
            if gained_hotkeys:
                bt.logging.debug(f"Metagraph gained hotkeys: {gained_hotkeys}")

            departed_hotkeys_set = set(self.departed_hotkeys.keys())
            re_registered_hotkeys = gained_hotkeys & departed_hotkeys_set
            if re_registered_hotkeys:
                bt.logging.warning(
                    f"Detected {len(re_registered_hotkeys)} re-registered miners: {re_registered_hotkeys}. "
                    f"These hotkeys were previously de-registered and have re-registered. Their orders will be rejected."
                )

            is_anomalous, _ = is_anomalous_hotkey_loss(lost_hotkeys, len(self.previous_metagraph_hotkeys))
            if lost_hotkeys and not is_anomalous:
                new_departures = lost_hotkeys - departed_hotkeys_set
                if new_departures:
                    current_time_ms = TimeUtil.now_in_millis()
                    for hotkey in new_departures:
                        self.departed_hotkeys[hotkey] = {"detected_ms": current_time_ms}
                    self._save_departed_hotkeys()
                    bt.logging.info(
                        f"Tracked {len(new_departures)} newly departed hotkeys: {new_departures}. "
                        f"Total departed hotkeys: {len(self.departed_hotkeys)}"
                    )
            elif lost_hotkeys:
                bt.logging.warning(
                    f"Detected anomalous metagraph change: {len(lost_hotkeys)} hotkeys lost "
                    f"({100 * len(lost_hotkeys) / len(self.previous_metagraph_hotkeys):.1f}% of total). "
                    f"Not tracking as departed to avoid false positives."
                )

            # Update previous state while still holding lock
            self.previous_metagraph_hotkeys = current_hotkeys

    def _delete_eliminated_expired_miners(self):
        """Delete expired eliminated miners."""
        deleted_hotkeys = set()
        any_challenege_period_changes = False
        now_ms = TimeUtil.now_in_millis()
        metagraph_hotkeys_set = set(self._metagraph_client.get_hotkeys())

        # Get snapshot while holding lock to avoid RuntimeError during iteration
        with self.eliminations_lock:
            eliminations_snapshot = list(self.eliminations.values())

        # Iterate over snapshot (safe - won't crash if dict is modified by other threads)
        for x in eliminations_snapshot:
            hotkey = x['hotkey']
            elimination_initiated_time_ms = x['elimination_initiated_time_ms']

            if now_ms - elimination_initiated_time_ms < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS:
                continue
            if hotkey in metagraph_hotkeys_set:
                bt.logging.trace(f"miner [{hotkey}] has not been deregistered by BT yet. Not deleting miner dir.")
                continue

            if self.cp_client.has_miner(hotkey):
                self.cp_client.remove_miner(hotkey)
                any_challenege_period_changes = True

            # Delete limit orders for eliminated miner (both in-memory and on-disk)
            result = self._limit_order_client.delete_all_limit_orders_for_hotkey(hotkey)
            bt.logging.info(f"Deleted limit orders for expired elimination [{hotkey}]: {result}")


            miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests) + hotkey
            all_positions = self._position_client.get_positions_for_one_hotkey(hotkey)
            for p in all_positions:
                self._position_client.delete_position(p.miner_hotkey, p.position_uuid)
            try:
                shutil.rmtree(miner_dir)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found. Already deleted. [{miner_dir}]")
            bt.logging.info(
                f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. "
                f"reason: [{x['reason']}] Removing miner dir [{miner_dir}]"
            )
            deleted_hotkeys.add(hotkey)

        # Write challengeperiod changes to disk if any changes were made
        if any_challenege_period_changes:
            self.cp_client.write_challengeperiod_from_memory_to_disk()

        if deleted_hotkeys:
            self.delete_eliminations(deleted_hotkeys)

    def _get_departed_hotkeys_from_disk(self) -> dict:
        """Load departed hotkeys from disk"""
        location = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=self.running_unit_tests)
        try:
            departed_data = ValiUtils.get_vali_json_file(location, DEPARTED_HOTKEYS_KEY)
            if departed_data is None:
                departed_data = {}
            if isinstance(departed_data, list):
                bt.logging.info(f"Converting legacy departed hotkeys list to dict format")
                departed_data = {hotkey: {"detected_ms": 0} for hotkey in departed_data}
            bt.logging.trace(f"Loaded {len(departed_data)} departed hotkeys from disk. Dir: {location}")
            return departed_data
        except Exception as e:
            bt.logging.warning(f"Could not load departed hotkeys from disk: {e}. Trying default file...")
            return self._get_departed_hotkeys_from_default_file()

    def _get_departed_hotkeys_from_default_file(self) -> dict:
        """Load departed hotkeys from default file"""
        import os
        base_dir = ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests).replace('/validation/', '')
        default_location = os.path.join(base_dir, 'data', 'default_departed_hotkeys.json')

        try:
            departed_data = ValiUtils.get_vali_json_file(default_location, DEPARTED_HOTKEYS_KEY)
            if departed_data is None:
                departed_data = {}
            if isinstance(departed_data, list):
                bt.logging.info(f"Converting legacy default departed hotkeys list to dict format")
                departed_data = {hotkey: {"detected_ms": 0} for hotkey in departed_data}
            bt.logging.info(f"Loaded {len(departed_data)} departed hotkeys from default file: {default_location}")
            return departed_data
        except Exception as e:
            bt.logging.warning(f"Could not load departed hotkeys from default file: {e}. Starting with empty dict.")
            return {}

    def _save_departed_hotkeys(self):
        """Save departed hotkeys to disk"""
        if not self.is_backtesting:
            departed_dict = dict(self.departed_hotkeys)
            departed_data = {DEPARTED_HOTKEYS_KEY: departed_dict}
            bt.logging.trace(f"Writing {len(departed_dict)} departed hotkeys to disk")
            output_location = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=self.running_unit_tests)
            ValiBkpUtils.write_file(output_location, departed_data)

    # ==================== Query Methods (used by Server) ====================

    def is_hotkey_eliminated(self, hotkey: str) -> bool:
        """Fast existence check (O(1))"""
        return hotkey in self.eliminations

    def get_elimination(self, hotkey: str) -> Optional[dict]:
        """Get full elimination details"""
        elimination = self.eliminations.get(hotkey)
        return deepcopy(elimination) if elimination else None

    def get_eliminated_hotkeys(self) -> Set[str]:
        """
        Get all eliminated hotkeys (thread-safe).

        Returns a consistent snapshot of eliminated hotkeys.
        """
        with self.eliminations_lock:
            return set(self.eliminations.keys())

    def get_eliminations_from_memory(self) -> List[dict]:
        """
        Get all eliminations as a list (thread-safe).

        Returns a consistent snapshot of all elimination records.
        """
        with self.eliminations_lock:
            return list(self.eliminations.values())

    def add_elimination(self, hotkey: str, elimination_data: dict) -> bool:
        """Add or update an elimination record. Returns True if new, False if updated."""
        # Validate required fields
        required_fields = ['hotkey', 'reason', 'elimination_initiated_time_ms']
        for field in required_fields:
            if field not in elimination_data:
                raise ValueError(f"Missing required field: {field}")

        if elimination_data['hotkey'] != hotkey:
            raise ValueError(f"Hotkey mismatch: {hotkey} != {elimination_data['hotkey']}")

        already_exists = hotkey in self.eliminations
        self.eliminations[hotkey] = elimination_data
        return not already_exists

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
            List of removed hotkeys
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
                self.eliminations[hotkey] = elim

            # Save while holding lock to prevent concurrent disk writes
            self._save_eliminations_locked()
            return removed

    def clear_eliminations(self) -> None:
        """Clear all eliminations for testing"""
        if not self.running_unit_tests:
            raise Exception('clear_eliminations can only be called during unit tests')
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
            {CacheController.ELIMINATIONS: []}
        )
        self.eliminations.clear()

    def _load_eliminations_from_disk(self) -> None:
        """
        Load eliminations from disk into memory (for testing recovery scenarios).
        This method reloads the eliminations dict from disk, useful for simulating
        validator restarts in tests.
        """
        if not self.running_unit_tests:
            raise Exception('_load_eliminations_from_disk can only be called during unit tests')

        with self.eliminations_lock:
            # Load from disk
            eliminations_from_disk = self.get_eliminations_from_disk()

            # Clear and repopulate, filtering out development hotkey
            self.eliminations.clear()
            filtered_count = 0

            for elim in eliminations_from_disk:
                hotkey = elim['hotkey']
                # Skip development hotkey - it should never be eliminated
                if hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
                    filtered_count += 1
                    bt.logging.debug(f"[ELIM_RELOAD] Filtered out DEVELOPMENT_HOTKEY from eliminations during disk reload")
                    continue
                self.eliminations[hotkey] = elim

            if filtered_count > 0:
                bt.logging.info(f"[ELIM_RELOAD] Filtered out {filtered_count} DEVELOPMENT_HOTKEY elimination(s) from disk reload")

            bt.logging.info(f"[ELIM_RELOAD] Loaded {len(self.eliminations)} elimination(s) from disk")

    def clear_departed_hotkeys(self) -> None:
        """Clear all departed hotkeys for testing"""
        if not self.running_unit_tests:
            raise Exception('clear_departed_hotkeys can only be called during unit tests')
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=self.running_unit_tests),
            {DEPARTED_HOTKEYS_KEY: {}}
        )
        self.departed_hotkeys.clear()
        # Reset previous_metagraph_hotkeys to current state to avoid false departures
        try:
            self.previous_metagraph_hotkeys = set(self._metagraph_client.get_hotkeys())
        except (AttributeError, RuntimeError):
            # MetagraphClient not connected yet (test mode without server setup)
            self.previous_metagraph_hotkeys = set()

    def is_hotkey_re_registered(self, hotkey: str) -> bool:
        """
        Check if hotkey is re-registered (was departed, now back) - thread-safe.

        Acquires eliminations_lock to prevent TOCTOU (time-of-check, time-of-use) race
        where departed_hotkeys could change between check and metagraph lookup.
        """
        if not hotkey:
            return False

        # Atomic check-then-use: Lock prevents departed_hotkeys from changing
        # between the check and the metagraph lookup
        with self.eliminations_lock:
            # Fast path: Check departed_hotkeys first
            is_departed = hotkey in self.departed_hotkeys
            if not is_departed:
                return False

            # Slow path: Check if back in metagraph
            # Lock is held, so departed_hotkeys can't change during this call
            is_in_metagraph = self._metagraph_client.has_hotkey(hotkey)
            return is_in_metagraph

    def get_departed_hotkeys(self) -> Dict[str, dict]:
        """Get all departed hotkeys"""
        return self.departed_hotkeys

    def get_departed_hotkey_info(self, hotkey: str) -> Optional[dict]:
        """Get departed info for a single hotkey (O(1) lookup)"""
        return self.departed_hotkeys.get(hotkey)

    def get_eliminations_dict(self) -> Dict[str, dict]:
        """
        Get eliminations dict (copy, thread-safe).

        Returns a consistent snapshot of the eliminations dictionary.
        """
        with self.eliminations_lock:
            return dict(self.eliminations)
