# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
EliminationManager RPC Server - Manages elimination state with local (non-IPC) dicts.

This server runs in its own process and exposes elimination management via RPC.
Much faster than IPC managerized dicts (50-200x improvement on batch operations).
"""
import time
import traceback
import shutil
import threading
from copy import deepcopy
from typing import Dict, Set, List, Optional
from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair
from setproctitle import setproctitle
from shared_objects.error_utils import ErrorUtils
from shared_objects.cache_controller import CacheController
from shared_objects.metagraph_utils import is_anomalous_hotkey_loss
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.elimination_manager import EliminationReason, DEPARTED_HOTKEYS_KEY

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource


class EliminationManagerServer(CacheController):
    """
    Server-side elimination manager with local dicts (no IPC overhead).

    All public methods ending in _rpc are exposed via RPC to the client.
    Internal state (eliminations, departed_hotkeys) is kept local to this process.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_rpc_address=None,
                 running_unit_tests=False, shutdown_dict=None, is_backtesting=False,
                 websocket_notifier=None, contract_manager=None, position_locks=None,
                 sync_in_progress=None, slack_notifier=None, sync_epoch=None, limit_order_manager=None):
        super().__init__(metagraph=metagraph, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict
        self.running_unit_tests = running_unit_tests

        # ChallengePeriod manager reference (test mode only - set via property)
        self.challengeperiod_manager = None

        # Create RPC client for ChallengePeriodManager (using address injection pattern)
        self.cp_client = None
        if challengeperiod_rpc_address and not running_unit_tests:
            from vali_objects.rpc.manager_rpc_client import ManagerRPCClient
            host, port = challengeperiod_rpc_address
            self.cp_client = ManagerRPCClient(host, port, ValiConfig.RPC_CHALLENGEPERIOD_SERVICE_NAME)
            connection_success = self.cp_client.connect()
            if connection_success:
                bt.logging.success(
                    f"[ELIM_RPC] Connected to ChallengePeriodManager at {challengeperiod_rpc_address}"
                )
            else:
                bt.logging.warning(
                    f"[ELIM_RPC] Failed to connect to ChallengePeriodManager at {challengeperiod_rpc_address}"
                )
        self.first_refresh_ran = False
        self.websocket_notifier = websocket_notifier
        secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        self.live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
        self.contract_manager = contract_manager
        self.position_locks = position_locks
        self.sync_in_progress = sync_in_progress
        self.slack_notifier = slack_notifier
        self.sync_epoch = sync_epoch
        self.limit_order_manager = limit_order_manager

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
        self.previous_metagraph_hotkeys = set(self.metagraph.get_hotkeys()) if self.metagraph else set()

        # Cache for fast fail-early checks (auto-refreshed by daemon)
        self._eliminations_cache = {}  # {hotkey: elimination_dict}
        self._departed_hotkeys_cache = {}  # {hotkey: departure_info_dict}
        self._cache_lock = threading.Lock()

        # Initial cache population
        self._refresh_cache()

        # Start cache refresh daemon
        if not running_unit_tests:
            self._start_cache_refresh_daemon()

    def _refresh_cache(self):
        """Refresh the fast-lookup caches from current state."""
        with self._cache_lock:
            self._eliminations_cache = dict(self.eliminations)  # Full elimination dicts
            self._departed_hotkeys_cache = dict(self.departed_hotkeys)
            bt.logging.debug(
                f"[CACHE_REFRESH] Refreshed: {len(self._eliminations_cache)} eliminated, "
                f"{len(self._departed_hotkeys_cache)} departed hotkeys"
            )

    def _cache_refresh_loop(self):
        """Background daemon that refreshes cache periodically."""
        setproctitle("vali_EliminationCacheRefresher")
        bt.logging.info(f"Elimination cache refresh daemon started ({ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S}-second interval)")

        while not self.shutdown_dict:
            try:
                time.sleep(ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S)
                self._refresh_cache()
            except Exception as e:
                bt.logging.error(f"Error in cache refresh daemon: {e}")
                time.sleep(ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S)

        bt.logging.info("Elimination cache refresh daemon shutting down")

    def _start_cache_refresh_daemon(self):
        """Start the background cache refresh thread."""
        refresh_thread = threading.Thread(target=self._cache_refresh_loop, daemon=True)
        refresh_thread.start()
        bt.logging.info("Started cache refresh daemon")

    # ==================== RPC Methods (exposed to client) ====================

    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring"""
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "num_eliminations": len(self.eliminations),
            "num_departed_hotkeys": len(self.departed_hotkeys)
        }

    def is_hotkey_eliminated_rpc(self, hotkey: str) -> bool:
        """Fast existence check (O(1))"""
        return hotkey in self.eliminations

    def get_elimination_rpc(self, hotkey: str) -> Optional[dict]:
        """Get full elimination details"""
        elimination = self.eliminations.get(hotkey)
        return deepcopy(elimination) if elimination else None

    def get_eliminated_hotkeys_rpc(self) -> Set[str]:
        """Get all eliminated hotkeys"""
        return set(self.eliminations.keys())

    def get_eliminations_from_memory_rpc(self) -> List[dict]:
        """Get all eliminations as a list"""
        return list(self.eliminations.values())

    def get_eliminations_from_disk_rpc(self) -> list:
        """Load eliminations from disk"""
        return self.get_eliminations_from_disk()

    def append_elimination_row_rpc(self, hotkey: str, current_dd: float, reason: str,
                                    t_ms: int = None, price_info: dict = None, return_info: dict = None) -> None:
        """
        Add elimination row (exposed for testing).

        Args:
            hotkey: The hotkey to eliminate
            current_dd: Current drawdown
            reason: Elimination reason
            t_ms: Optional timestamp in milliseconds
            price_info: Optional price information
            return_info: Optional return information
        """
        self.append_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                    price_info=price_info, return_info=return_info)

    def add_elimination_rpc(self, hotkey: str, elimination_data: dict) -> bool:
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

    def remove_elimination_rpc(self, hotkey: str) -> bool:
        """Remove elimination. Returns True if removed, False if not found."""
        if hotkey in self.eliminations:
            del self.eliminations[hotkey]
            return True
        return False

    def sync_eliminations_rpc(self, eliminations_list: list) -> list:
        """
        Sync eliminations from external source (batch update).
        Returns list of removed hotkeys.
        """
        hotkeys_before = set(self.eliminations.keys())
        hotkeys_after = set(x['hotkey'] for x in eliminations_list)
        removed = [x for x in hotkeys_before if x not in hotkeys_after]
        added = [x for x in hotkeys_after if x not in hotkeys_before]

        bt.logging.info(f'sync_eliminations_rpc: removed {len(removed)} {removed}, added {len(added)} {added}')

        # Batch update (much faster than individual IPC calls)
        self.eliminations.clear()
        for elim in eliminations_list:
            hotkey = elim['hotkey']
            self.eliminations[hotkey] = elim

        self.save_eliminations()
        return removed

    def clear_eliminations_rpc(self) -> None:
        """Clear all eliminations"""
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
            {CacheController.ELIMINATIONS: []}
        )
        self.eliminations.clear()

    def is_hotkey_re_registered_rpc(self, hotkey: str) -> bool:
        """Check if hotkey is re-registered (was departed, now back)"""
        if not hotkey:
            return False

        # Fast path: Check departed_hotkeys first
        is_departed = hotkey in self.departed_hotkeys
        if not is_departed:
            return False

        # Slow path: Check if back in metagraph
        is_in_metagraph = self.metagraph.has_hotkey(hotkey)
        return is_in_metagraph

    def get_departed_hotkeys_rpc(self) -> Dict[str, dict]:
        """Get all departed hotkeys"""
        return self.departed_hotkeys

    def get_departed_hotkey_info_rpc(self, hotkey: str) -> Optional[dict]:
        """Get departed info for a single hotkey (O(1) lookup)"""
        return self.departed_hotkeys.get(hotkey)

    def get_cached_elimination_data_rpc(self) -> tuple:
        """
        Get cached elimination data (always fresh, auto-refreshed by background daemon).

        Returns:
            Tuple of (eliminations_dict, departed_hotkeys_dict)
        """
        with self._cache_lock:
            return (dict(self._eliminations_cache), dict(self._departed_hotkeys_cache))

    def get_eliminations_lock_rpc(self):
        """This method should not be called via RPC - lock is local to server"""
        raise NotImplementedError(
            "get_eliminations_lock() is not available via RPC. "
            "Locking happens automatically on server side."
        )

    def process_eliminations_rpc(self, position_locks=None, iteration_epoch=None) -> None:
        """
        Trigger elimination processing via RPC.
        Uses RPC in both test and production modes.
        """
        self.process_eliminations(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def handle_perf_ledger_eliminations_rpc(self, position_locks=None, iteration_epoch=None) -> None:
        """
        Process performance ledger eliminations (exposed for testing).
        Uses RPC in both test and production modes.
        """
        self.handle_perf_ledger_eliminations(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def get_first_refresh_ran_rpc(self) -> bool:
        """Get the first_refresh_ran flag via RPC."""
        return self.first_refresh_ran

    def set_first_refresh_ran_rpc(self, value: bool) -> None:
        """Set the first_refresh_ran flag via RPC."""
        self.first_refresh_ran = value

    def is_zombie_hotkey_rpc(self, hotkey: str, all_hotkeys_set: set) -> bool:
        """Check if hotkey is a zombie via RPC."""
        return self.is_zombie_hotkey(hotkey, all_hotkeys_set)

    def handle_mdd_eliminations_rpc(self, position_locks=None, iteration_epoch=None) -> None:
        """
        Check for MDD eliminations via RPC.
        Uses RPC in both test and production modes.
        """
        self.handle_mdd_eliminations(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def save_eliminations_rpc(self) -> None:
        """Save eliminations to disk via RPC."""
        self.save_eliminations()

    def write_eliminations_to_disk_rpc(self, eliminations: list) -> None:
        """Write eliminations to disk via RPC."""
        self.write_eliminations_to_disk(eliminations)

    def get_eliminations_dict_rpc(self) -> Dict[str, dict]:
        """Get eliminations dict (copy) via RPC."""
        return dict(self.eliminations)

    def handle_first_refresh_rpc(self, position_locks, iteration_epoch=None) -> None:
        """
        Handle first refresh on startup via RPC.
        Uses RPC in both test and production modes.
        """
        self.handle_first_refresh(position_locks, iteration_epoch)

    def start_daemon_rpc(self) -> None:
        """
        Start the daemon loop (called via RPC after all dependencies are ready).

        Dependencies (position_manager, challengeperiod_manager) must be passed at
        server creation time via __init__ parameters. This method should be called
        after the validator has finished all initialization.
        """
        if hasattr(self, '_daemon_started') and self._daemon_started:
            bt.logging.warning("EliminationManager daemon already started, ignoring")
            return

        import threading

        daemon_thread = threading.Thread(
            target=self.run_update_loop,
            daemon=True
        )
        daemon_thread.start()
        self._daemon_started = True
        bt.logging.success("EliminationManager daemon loop started in server process (via RPC)")

    # ==================== Internal Methods (not exposed) ====================

    def get_eliminations_lock(self):
        """Get the local eliminations lock (server-side only)"""
        return self.eliminations_lock

    def handle_perf_ledger_eliminations(self, position_locks, iteration_epoch=None):
        """Process performance ledger eliminations"""
        perf_ledger_eliminations = self.position_manager.perf_ledger_manager.get_perf_ledger_eliminations()
        n_eliminations = 0
        for e in perf_ledger_eliminations:
            if e['hotkey'] in self.eliminations:
                continue

            n_eliminations += 1
            hotkey = e['hotkey']
            self.eliminations[hotkey] = e

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
                                        position_locks, iteration_epoch)
            self.contract_manager.slash_miner_collateral_proportion(e['hotkey'])

        if n_eliminations:
            self.save_eliminations()
            bt.logging.info(f'Wrote {n_eliminations} perf ledger eliminations to disk')

    def add_manual_flat_order(self, hotkey: str, position: Position, corresponding_elimination,
                             position_locks, source_for_elimination, iteration_epoch=None):
        """Add flat orders for eliminated miner"""
        elimination_time_ms = corresponding_elimination['elimination_initiated_time_ms'] if corresponding_elimination else TimeUtil.now_in_millis()
        with position_locks.get_lock(hotkey, position.trade_pair.trade_pair_id):
            position_refreshed = self.position_manager.get_miner_position_by_uuid(hotkey, position.position_uuid)
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
                                                           self.live_price_fetcher, source_for_elimination)
            position.add_order(flat_order, self.live_price_fetcher)

            # Epoch-based validation
            if self.sync_epoch and iteration_epoch is not None:
                current_epoch = self.sync_epoch.value
                if current_epoch != iteration_epoch:
                    bt.logging.warning(
                        f"Sync occurred during EliminationManager iteration for {hotkey} {position.trade_pair.trade_pair_id} "
                        f"(epoch {iteration_epoch} -> {current_epoch}). Skipping save to avoid data corruption"
                    )
                    return

            self.position_manager.save_miner_position(position, delete_open_position_if_exists=True)
            if self.websocket_notifier:
                self.websocket_notifier.broadcast_position_update(position)
            bt.logging.info(
                f'Added flat order for miner {hotkey} that has been eliminated. '
                f'Trade pair: {position.trade_pair.trade_pair_id}. flat order: {flat_order}. '
                f'position uuid {position.position_uuid}. Source for elimination {source_for_elimination}'
            )

    def handle_eliminated_miner(self, hotkey: str,
                                trade_pair_to_price_source_used_for_elimination_check: Dict[TradePair, PriceSource],
                                position_locks, iteration_epoch=None):
        """Handle cleanup for eliminated miner"""
        # Clean up limit orders
        if self.limit_order_manager:
            try:
                result = self.limit_order_manager.delete_all_limit_orders_for_hotkey(hotkey)
                bt.logging.info(f"Cleaned up limit orders for eliminated miner [{hotkey}]: {result}")
            except Exception as e:
                bt.logging.error(f"Error cleaning up limit orders for eliminated miner [{hotkey}]: {e}")

        for p in self.position_manager.get_positions_for_one_hotkey(hotkey, only_open_positions=True):
            source_for_elimination = trade_pair_to_price_source_used_for_elimination_check.get(p.trade_pair)
            corresponding_elimination = self.eliminations.get(hotkey)
            if corresponding_elimination:
                self.add_manual_flat_order(hotkey, p, corresponding_elimination, position_locks,
                                          source_for_elimination, iteration_epoch)

    def handle_challenge_period_eliminations(self, position_locks, iteration_epoch=None):
        """
        Process challenge period eliminations.

        Uses atomic pop operations to prevent race conditions where ChallengePeriodManager
        might add new eliminations while we're processing.

        In test mode, uses direct reference to challengeperiod_manager.
        In production, uses RPC client (cp_client).
        """
        # Determine which CP interface to use
        cp_interface = None
        use_direct_call = False

        if self.running_unit_tests and self.challengeperiod_manager:
            # Test mode: use direct reference
            cp_interface = self.challengeperiod_manager
            use_direct_call = True
        elif self.cp_client:
            # Production mode: use RPC client
            cp_interface = self.cp_client
            use_direct_call = False
        else:
            # No CP interface available
            return

        # Check if there are any eliminations to process
        if use_direct_call:
            if not cp_interface.has_elimination_reasons():
                return
            eliminations_snapshot = cp_interface.get_all_elimination_reasons()
        else:
            # Production mode: use RPC client (with safe call handling)
            try:
                if not cp_interface.call("has_elimination_reasons_rpc"):
                    return
                eliminations_snapshot = cp_interface.call("get_all_elimination_reasons_rpc")
            except RuntimeError as e:
                # RPC not connected - skip challenge period eliminations
                bt.logging.debug(f"CP client not connected, skipping challenge period eliminations: {e}")
                return

        hotkeys = list(eliminations_snapshot.keys())

        if not hotkeys:
            return

        bt.logging.info(f"[ELIM_DEBUG] Processing {len(hotkeys)} challenge period eliminations: {hotkeys}")
        bt.logging.info(f"[ELIM_DEBUG] Current eliminations dict has {len(self.eliminations)} entries")

        # Process each hotkey individually, popping atomically to avoid race conditions
        for hotkey in hotkeys:
            # Atomically pop the elimination reason (get + remove in one operation)
            # This prevents ChallengePeriodManager from re-adding while we process
            if use_direct_call:
                elim_data = cp_interface.pop_elimination_reason(hotkey)
            else:
                elim_data = cp_interface.call("pop_elimination_reason_rpc", hotkey)

            # Skip if already removed (another thread might have processed it)
            if elim_data is None:
                bt.logging.debug(f"[ELIM_DEBUG] Hotkey {hotkey} already processed/removed")
                continue

            already_eliminated = hotkey in self.eliminations
            if already_eliminated:
                bt.logging.warning(
                    f"[ELIM_DEBUG] Hotkey {hotkey} is ALREADY in eliminations list. Skipping. "
                    f"Elimination: {self.eliminations[hotkey]}"
                )
                continue

            bt.logging.info(f"[ELIM_DEBUG] Adding new elimination for {hotkey}")
            elim_reason = elim_data[0]
            elim_mdd = elim_data[1]
            self.append_elimination_row(hotkey=hotkey, current_dd=elim_mdd, reason=elim_reason)

            # Verify it was added
            if hotkey in self.eliminations:
                bt.logging.info(f"[ELIM_DEBUG] ✓ Verified {hotkey} was added to eliminations list")
            else:
                bt.logging.error(f"[ELIM_DEBUG] ✗ FAILED to add {hotkey} to eliminations list!")

            self.handle_eliminated_miner(hotkey, {}, position_locks, iteration_epoch)
            self.contract_manager.slash_miner_collateral_proportion(hotkey)

        bt.logging.info(f"[ELIM_DEBUG] After processing, eliminations dict has {len(self.eliminations)} entries")

    def handle_first_refresh(self, position_locks, iteration_epoch=None):
        """Handle first refresh on startup"""
        if self.is_backtesting or self.first_refresh_ran:
            return

        eliminated_hotkeys = set(self.eliminations.keys())
        hotkey_to_positions = self.position_manager.get_positions_for_hotkeys(eliminated_hotkeys,
                                                                              only_open_positions=True)
        for hotkey, open_positions in hotkey_to_positions.items():
            if not open_positions:
                continue
            for p in open_positions:
                self.add_manual_flat_order(hotkey, p, self.eliminations.get(hotkey), position_locks, None, iteration_epoch)

        self.first_refresh_ran = True

    def process_eliminations(self, position_locks=None, iteration_epoch=None):
        """Main elimination processing loop"""
        if position_locks is None:
            position_locks = self.position_locks

        # Check if we should process:
        # 1. Process if time-based refresh is due
        # 2. OR process if there are urgent challenge period eliminations
        refresh_due = self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS)

        # Check for urgent eliminations using dual interface pattern
        has_urgent_eliminations = False
        if self.running_unit_tests and self.challengeperiod_manager:
            # Test mode: use direct reference
            has_urgent_eliminations = self.challengeperiod_manager.has_elimination_reasons()
        elif self.cp_client:
            # Production mode: use RPC client (with safe call handling)
            try:
                has_urgent_eliminations = self.cp_client.call("has_elimination_reasons_rpc")
            except RuntimeError as e:
                # RPC not connected - skip urgent eliminations check
                bt.logging.debug(f"CP client not connected, skipping urgent eliminations check: {e}")
                has_urgent_eliminations = False

        if not refresh_due and not has_urgent_eliminations:
            return

        bt.logging.info(
            f"running elimination manager. invalidation data "
            f"{dict(self.position_manager.perf_ledger_manager.perf_ledger_hks_to_invalidate)}"
        )

        self._update_departed_hotkeys()
        self.handle_first_refresh(position_locks, iteration_epoch)
        self.handle_perf_ledger_eliminations(position_locks, iteration_epoch)
        self.handle_challenge_period_eliminations(position_locks, iteration_epoch)
        self.handle_mdd_eliminations(position_locks, iteration_epoch)
        self.handle_zombies(position_locks, iteration_epoch)
        self._delete_eliminated_expired_miners()

        self.set_last_update_time()

    def run_update_loop(self):
        """Main server loop"""
        setproctitle("vali_EliminationManagerServer")
        bt.logging.info("EliminationManagerServer daemon loop running")

        while not self.shutdown_dict:
            try:
                if self.sync_in_progress and self.sync_in_progress.value:
                    bt.logging.debug("EliminationManagerServer: Sync in progress, pausing...")
                    time.sleep(1)
                    continue

                iteration_epoch = self.sync_epoch.value if self.sync_epoch else None
                self.process_eliminations(iteration_epoch=iteration_epoch)
                time.sleep(1)

            except Exception as e:
                error_traceback = traceback.format_exc()
                bt.logging.error(f"Error in EliminationManagerServer update loop: {e}")
                bt.logging.error(error_traceback)

                if self.slack_notifier:
                    error_message = ErrorUtils.format_error_for_slack(
                        error=e, traceback_str=error_traceback,
                        include_operation=True, include_timestamp=True
                    )
                    self.slack_notifier.send_message(
                        f"❌ EliminationManagerServer daemon error!\n{error_message}",
                        level="error"
                    )

                time.sleep(10)

        bt.logging.info("EliminationManagerServer process shutting down")

    def is_zombie_hotkey(self, hotkey, all_hotkeys_set):
        """Check if hotkey is a zombie"""
        if hotkey == ValiConfig.DEVELOPMENT_HOTKEY:
            return False
        return hotkey not in all_hotkeys_set

    def save_eliminations(self):
        """Save eliminations to disk"""
        if not self.is_backtesting:
            self.write_eliminations_to_disk(list(self.eliminations.values()))

    def write_eliminations_to_disk(self, eliminations):
        """Write eliminations to disk"""
        if not isinstance(eliminations, list):
            eliminations = list(eliminations)
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        bt.logging.info(f"[ELIM_DEBUG] Writing {len(eliminations)} eliminations to disk at {output_location}")
        bt.logging.info(f"[ELIM_DEBUG] Hotkeys in elimination list being written: {[x['hotkey'] for x in eliminations]}")
        ValiBkpUtils.write_file(output_location, vali_eliminations)
        bt.logging.info(f"[ELIM_DEBUG] Successfully wrote eliminations to disk")

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
        """Add elimination row"""
        bt.logging.info(f"[ELIM_DEBUG] append_elimination_row called for {hotkey}, reason={reason}")
        elimination_row = self.generate_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                                        price_info=price_info, return_info=return_info)
        dict_len_before = len(self.eliminations)
        self.eliminations[hotkey] = elimination_row
        dict_len_after = len(self.eliminations)
        bt.logging.info(f"[ELIM_DEBUG] Eliminations dict grew from {dict_len_before} to {dict_len_after} entries")

        self.save_eliminations()
        bt.logging.info(f"miner eliminated with hotkey [{hotkey}]. Info [{elimination_row}]")

    def delete_eliminations(self, deleted_hotkeys):
        """Delete multiple eliminations"""
        for hotkey in deleted_hotkeys:
            if hotkey in self.eliminations:
                del self.eliminations[hotkey]
        self.save_eliminations()

    def handle_mdd_eliminations(self, position_locks, iteration_epoch=None):
        """
        Check for MDD eliminations.

        In test mode, uses direct reference to challengeperiod_manager.
        In production, uses RPC client (cp_client).
        """
        from vali_objects.utils.ledger_utils import LedgerUtils
        bt.logging.info("checking main competition for maximum drawdown eliminations.")
        if self.shutdown_dict:
            return

        # Determine which CP interface to use
        cp_interface = None
        use_direct_call = False

        if self.running_unit_tests and self.challengeperiod_manager:
            # Test mode: use direct reference
            cp_interface = self.challengeperiod_manager
            use_direct_call = True
        elif self.cp_client:
            # Production mode: use RPC client
            cp_interface = self.cp_client
            use_direct_call = False
        else:
            # No CP interface available
            return

        # Get MAINCOMP hotkeys
        if use_direct_call:
            challengeperiod_success_hotkeys = cp_interface.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        else:
            # Production mode: use RPC client (with safe call handling)
            try:
                challengeperiod_success_hotkeys = cp_interface.call("get_hotkeys_by_bucket_rpc", MinerBucket.MAINCOMP)
            except RuntimeError as e:
                # RPC not connected - skip MDD eliminations check
                bt.logging.debug(f"CP client not connected, skipping MDD eliminations: {e}")
                return

        filtered_ledger = self.position_manager.perf_ledger_manager.filtered_ledger_for_scoring(
            portfolio_only=True, hotkeys=challengeperiod_success_hotkeys
        )
        for miner_hotkey, ledger in filtered_ledger.items():
            if self.shutdown_dict:
                return
            if miner_hotkey in self.eliminations:
                continue

            miner_exceeds_mdd, drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element=ledger)

            if miner_exceeds_mdd:
                self.append_elimination_row(miner_hotkey, drawdown_percentage, EliminationReason.MAX_TOTAL_DRAWDOWN.value)
                self.handle_eliminated_miner(miner_hotkey, {}, position_locks, iteration_epoch)
                self.contract_manager.slash_miner_collateral_proportion(miner_hotkey)

    def handle_zombies(self, position_locks, iteration_epoch=None):
        """Handle zombie miners"""
        if self.shutdown_dict or self.is_backtesting:
            return

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        all_hotkeys_set = set(self.metagraph.get_hotkeys()) if self.metagraph else set()

        for hotkey in CacheController.get_directory_names(all_miners_dir):
            corresponding_elimination = self.eliminations.get(hotkey)
            elimination_reason = corresponding_elimination.get('reason') if corresponding_elimination else None
            if elimination_reason:
                continue
            elif self.is_zombie_hotkey(hotkey, all_hotkeys_set):
                self.append_elimination_row(hotkey=hotkey, current_dd=None, reason=EliminationReason.ZOMBIE.value)
                self.handle_eliminated_miner(hotkey, {}, position_locks, iteration_epoch)

    def _update_departed_hotkeys(self):
        """Track departed hotkeys"""
        if self.is_backtesting:
            return

        current_hotkeys = set(self.metagraph.get_hotkeys()) if self.metagraph else set()
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

        self.previous_metagraph_hotkeys = current_hotkeys

    def _delete_eliminated_expired_miners(self):
        """Delete expired eliminated miners"""
        deleted_hotkeys = set()
        any_challenege_period_changes = False
        now_ms = TimeUtil.now_in_millis()
        metagraph_hotkeys_set = set(self.metagraph.get_hotkeys()) if self.metagraph else set()

        for x in self.eliminations.values():
            if self.shutdown_dict:
                return
            hotkey = x['hotkey']
            elimination_initiated_time_ms = x['elimination_initiated_time_ms']

            if now_ms - elimination_initiated_time_ms < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS:
                continue
            if hotkey in metagraph_hotkeys_set:
                bt.logging.trace(f"miner [{hotkey}] has not been deregistered by BT yet. Not deleting miner dir.")
                continue

            # Determine which CP interface to use for cleanup
            if self.running_unit_tests and self.challengeperiod_manager:
                # Test mode: use direct reference
                if self.challengeperiod_manager.has_miner(hotkey):
                    self.challengeperiod_manager.remove_miner(hotkey)
                    any_challenege_period_changes = True
            elif self.cp_client:
                # Production mode: use RPC client (with safe call handling)
                try:
                    if self.cp_client.call("has_miner_rpc", hotkey):
                        self.cp_client.call("remove_miner_rpc", hotkey)
                        any_challenege_period_changes = True
                except RuntimeError as e:
                    # RPC not connected - skip challenge period cleanup
                    bt.logging.debug(f"CP client not connected, skipping challenge period cleanup for {hotkey}: {e}")

            miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests) + hotkey
            all_positions = self.position_manager.get_positions_for_one_hotkey(hotkey)
            for p in all_positions:
                self.position_manager.delete_position(p)
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
            if self.running_unit_tests and self.challengeperiod_manager:
                # Test mode: use direct reference
                self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
            elif self.cp_client:
                # Production mode: use RPC client (with safe call handling)
                try:
                    self.cp_client.call("write_challengeperiod_from_memory_to_disk_rpc")
                except RuntimeError as e:
                    # RPC not connected - skip writing challenge period changes
                    bt.logging.debug(f"CP client not connected, skipping challenge period disk write: {e}")

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


def start_elimination_manager_server(
    metagraph, position_manager, challengeperiod_rpc_address,
    running_unit_tests, shutdown_dict, is_backtesting,
    websocket_notifier, contract_manager, position_locks,
    sync_in_progress, slack_notifier, sync_epoch, limit_order_manager,
    address, authkey, server_ready
):
    """
    Entry point for server process.

    Args:
        challengeperiod_rpc_address: Tuple of (host, port) for ChallengePeriodManager RPC server.
                                    Example: ("localhost", 50003)
    """
    from multiprocessing.managers import BaseManager

    server_instance = EliminationManagerServer(
        metagraph=metagraph,
        position_manager=position_manager,
        challengeperiod_rpc_address=challengeperiod_rpc_address,
        running_unit_tests=running_unit_tests,
        shutdown_dict=shutdown_dict,
        is_backtesting=is_backtesting,
        websocket_notifier=websocket_notifier,
        contract_manager=contract_manager,
        position_locks=position_locks,
        sync_in_progress=sync_in_progress,
        slack_notifier=slack_notifier,
        sync_epoch=sync_epoch,
        limit_order_manager=limit_order_manager
    )

    # NOTE: Daemon is NOT started here - it's started explicitly via start_daemon_rpc()
    # after all dependencies (position_manager, challengeperiod_manager) are wired up

    # Register server with manager
    class EliminationManagerRPC(BaseManager):
        pass

    EliminationManagerRPC.register('EliminationManagerServer', callable=lambda: server_instance)

    manager = EliminationManagerRPC(address=address, authkey=authkey)
    rpc_server = manager.get_server()

    bt.logging.success(f"EliminationManagerServer ready on {address}")

    if server_ready:
        server_ready.set()

    # Start serving (blocks forever)
    rpc_server.serve_forever()
