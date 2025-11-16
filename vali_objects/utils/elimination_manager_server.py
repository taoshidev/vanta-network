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

    def __init__(self, metagraph, position_manager, challengeperiod_manager,
                 running_unit_tests=False, shutdown_dict=None, is_backtesting=False,
                 shared_queue_websockets=None, contract_manager=None, position_locks=None,
                 sync_in_progress=None, slack_notifier=None, sync_epoch=None, limit_order_manager=None):
        super().__init__(metagraph=metagraph, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict
        self.challengeperiod_manager = challengeperiod_manager
        self.running_unit_tests = running_unit_tests
        self.first_refresh_ran = False
        self.shared_queue_websockets = shared_queue_websockets
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

        # Populate from disk
        eliminations_from_disk = self.get_eliminations_from_disk()
        for elim in eliminations_from_disk:
            hotkey = elim['hotkey']
            self.eliminations[hotkey] = elim

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
        self.previous_metagraph_hotkeys = set(self.metagraph.hotkeys) if self.metagraph else set()

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
        is_in_metagraph = self.metagraph.has_hotkey(hotkey) if self.metagraph else False
        return is_in_metagraph

    def get_departed_hotkeys_rpc(self) -> Dict[str, dict]:
        """Get all departed hotkeys"""
        return dict(self.departed_hotkeys)

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
            self.contract_manager.slash_miner_collateral_proportion(e['hotkey'], ValiConfig.SLASH_PROPORTION)

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
            if self.shared_queue_websockets:
                self.shared_queue_websockets.put(position.to_websocket_dict())
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
        """Process challenge period eliminations"""
        eliminations_with_reasons = self.challengeperiod_manager.get_all_elimination_reasons()

        if not eliminations_with_reasons:
            return

        hotkeys = list(eliminations_with_reasons.keys())
        bt.logging.info(f"[ELIM_DEBUG] Processing {len(hotkeys)} challenge period eliminations: {hotkeys}")
        bt.logging.info(f"[ELIM_DEBUG] Current eliminations dict has {len(self.eliminations)} entries")

        for hotkey in hotkeys:
            already_eliminated = hotkey in self.eliminations
            if already_eliminated:
                bt.logging.warning(
                    f"[ELIM_DEBUG] Hotkey {hotkey} is ALREADY in eliminations list. Skipping. "
                    f"Elimination: {self.eliminations[hotkey]}"
                )
                continue

            bt.logging.info(f"[ELIM_DEBUG] Adding new elimination for {hotkey}")
            elim_reason = eliminations_with_reasons[hotkey][0]
            elim_mdd = eliminations_with_reasons[hotkey][1]
            self.append_elimination_row(hotkey=hotkey, current_dd=elim_mdd, reason=elim_reason)

            # Verify it was added
            if hotkey in self.eliminations:
                bt.logging.info(f"[ELIM_DEBUG] ✓ Verified {hotkey} was added to eliminations list")
            else:
                bt.logging.error(f"[ELIM_DEBUG] ✗ FAILED to add {hotkey} to eliminations list!")

            self.handle_eliminated_miner(hotkey, {}, position_locks, iteration_epoch)
            self.contract_manager.slash_miner_collateral_proportion(hotkey, ValiConfig.CHALLENGEPERIOD_SLASH_PROPORTION)

        bt.logging.info(f"[ELIM_DEBUG] After processing, eliminations dict has {len(self.eliminations)} entries")
        self.challengeperiod_manager.clear_elimination_reasons()

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

        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS) and \
                not self.challengeperiod_manager.has_elimination_reasons():
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
        bt.logging.info("EliminationManagerServer process started")

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
        """Check for MDD eliminations"""
        from vali_objects.utils.ledger_utils import LedgerUtils
        bt.logging.info("checking main competition for maximum drawdown eliminations.")
        if self.shutdown_dict:
            return
        challengeperiod_success_hotkeys = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)

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
                self.contract_manager.slash_miner_collateral_proportion(miner_hotkey, ValiConfig.SLASH_PROPORTION)

    def handle_zombies(self, position_locks, iteration_epoch=None):
        """Handle zombie miners"""
        if self.shutdown_dict or self.is_backtesting:
            return

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        all_hotkeys_set = set(self.metagraph.hotkeys) if self.metagraph else set()

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

        current_hotkeys = set(self.metagraph.hotkeys) if self.metagraph else set()
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
        metagraph_hotkeys_set = set(self.metagraph.hotkeys) if self.metagraph else set()

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

            if self.challengeperiod_manager.has_miner(hotkey):
                self.challengeperiod_manager.remove_miner(hotkey)
                any_challenege_period_changes = True

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

        if any_challenege_period_changes:
            self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

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
    metagraph, position_manager, challengeperiod_manager,
    running_unit_tests, shutdown_dict, is_backtesting,
    shared_queue_websockets, contract_manager, position_locks,
    sync_in_progress, slack_notifier, sync_epoch, limit_order_manager,
    address, authkey, server_ready
):
    """Entry point for server process"""
    from multiprocessing.managers import BaseManager

    server_instance = EliminationManagerServer(
        metagraph=metagraph,
        position_manager=position_manager,
        challengeperiod_manager=challengeperiod_manager,
        running_unit_tests=running_unit_tests,
        shutdown_dict=shutdown_dict,
        is_backtesting=is_backtesting,
        shared_queue_websockets=shared_queue_websockets,
        contract_manager=contract_manager,
        position_locks=position_locks,
        sync_in_progress=sync_in_progress,
        slack_notifier=slack_notifier,
        sync_epoch=sync_epoch,
        limit_order_manager=limit_order_manager
    )

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
