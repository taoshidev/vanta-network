# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import shutil
from copy import deepcopy
from enum import Enum
from typing import Dict
from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from shared_objects.metagraph_utils import is_anomalous_hotkey_loss
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource

class EliminationReason(Enum):
    ZOMBIE = "ZOMBIE"
    PLAGIARISM = "PLAGIARISM"
    MAX_TOTAL_DRAWDOWN = "MAX_TOTAL_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_TIME = "FAILED_CHALLENGE_PERIOD_TIME"
    FAILED_CHALLENGE_PERIOD_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_DRAWDOWN"
    LIQUIDATED = "LIQUIDATED"

# Constants for departed hotkeys tracking
DEPARTED_HOTKEYS_KEY = "departed_hotkeys"

class EliminationManager(CacheController):
    """"
    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_manager,
                 running_unit_tests=False, shutdown_dict=None, ipc_manager=None, is_backtesting=False,
                 shared_queue_websockets=None, contract_manager=None, position_locks=None,
                 sync_in_progress=None, slack_notifier=None, sync_epoch=None):
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

        if ipc_manager:
            self.eliminations = ipc_manager.list()
            self.departed_hotkeys = ipc_manager.dict()
        else:
            self.eliminations = []
            self.departed_hotkeys = {}
        self.eliminations.extend(self.get_eliminations_from_disk())
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
        self.previous_metagraph_hotkeys = set(self.metagraph.hotkeys) if self.metagraph and self.metagraph.hotkeys else set()

    def handle_perf_ledger_eliminations(self, position_locks, iteration_epoch=None):
        perf_ledger_eliminations = self.position_manager.perf_ledger_manager.get_perf_ledger_eliminations()
        n_eliminations = 0
        for e in perf_ledger_eliminations:
            if self.hotkey_in_eliminations(e['hotkey']):
                continue

            n_eliminations += 1
            self.eliminations.append(e)
            self.eliminations[-1] = e  # ipc list does not update the object without using __setitem__

            price_info = e['price_info']
            trade_pair_to_price_source_used_for_elimination_check = {}
            for k, v in price_info.items():
                trade_pair = TradePair.get_latest_tade_pair_from_trade_pair_str(k)
                elimination_initiated_time_ms = e['elimination_initiated_time_ms']
                trade_pair_to_price_source_used_for_elimination_check[trade_pair] = PriceSource(source='elim', open=v,
                                                                                                close=v,
                                                                                                start_ms=elimination_initiated_time_ms,
                                                                                                timespan_ms=1000,
                                                                                                websocket=False)
            self.handle_eliminated_miner(e['hotkey'], trade_pair_to_price_source_used_for_elimination_check, position_locks)
            self.contract_manager.slash_miner_collateral_proportion(e['hotkey'])

        if n_eliminations:
            self.save_eliminations()
            bt.logging.info(f'Wrote {n_eliminations} perf ledger eliminations to disk')

    def add_manual_flat_order(self, hotkey: str, position: Position, corresponding_elimination, position_locks,
                              source_for_elimination, iteration_epoch=None):
        """
        Add flat orders to the positions for a miner that has been eliminated

        Args:
            iteration_epoch: Epoch captured at start of iteration. Used to detect stale data.
        """
        elimination_time_ms = corresponding_elimination['elimination_initiated_time_ms'] if corresponding_elimination else TimeUtil.now_in_millis()
        with position_locks.get_lock(hotkey, position.trade_pair.trade_pair_id):
            # Position could have updated in the time between mdd_check being called and this function being called
            position_refreshed = self.position_manager.get_miner_position_by_uuid(hotkey, position.position_uuid)
            if position_refreshed is None:
                bt.logging.warning(
                    f"Unexpectedly could not find position with uuid {position.position_uuid} for hotkey {hotkey} and trade pair {position.trade_pair.trade_pair_id}. Not add flat orders")
                return

            position = position_refreshed
            if position.is_closed_position:
                return

            fake_flat_order_time = elimination_time_ms
            if position.orders and position.orders[-1].processed_ms > elimination_time_ms:
                bt.logging.warning(
                    f'Unexpectedly found a position with a processed_ms {position.orders[-1].processed_ms} greater than the elimination time {elimination_time_ms} ')
                fake_flat_order_time = position.orders[-1].processed_ms + 1

            flat_order = Position.generate_fake_flat_order(position, fake_flat_order_time, self.live_price_fetcher, source_for_elimination)
            position.add_order(flat_order, self.live_price_fetcher)

            # Epoch-based validation: check if sync occurred during our iteration
            if self.sync_epoch and iteration_epoch is not None:
                current_epoch = self.sync_epoch.value
                if current_epoch != iteration_epoch:
                    bt.logging.warning(
                        f"Sync occurred during EliminationManager iteration for {hotkey} {position.trade_pair.trade_pair_id} "
                        f"(epoch {iteration_epoch} -> {current_epoch}). "
                        f"Skipping save to avoid data corruption"
                    )
                    return

            self.position_manager.save_miner_position(position, delete_open_position_if_exists=True)
            if self.shared_queue_websockets:
                self.shared_queue_websockets.put(position.to_websocket_dict())
            bt.logging.info(f'Added flat order for miner {hotkey} that has been eliminated. '
                            f'Trade pair: {position.trade_pair.trade_pair_id}. flat order: {flat_order}. '
                            f'position uuid {position.position_uuid}. Source for elimination {source_for_elimination}')

    def handle_eliminated_miner(self, hotkey: str,
                                trade_pair_to_price_source_used_for_elimination_check: Dict[TradePair, PriceSource],
                                position_locks, iteration_epoch=None):

        for p in self.position_manager.get_positions_for_one_hotkey(hotkey, only_open_positions=True):
            source_for_elimination = trade_pair_to_price_source_used_for_elimination_check.get(p.trade_pair)
            corresponding_elimination = self.hotkey_in_eliminations(hotkey)
            if corresponding_elimination:
                self.add_manual_flat_order(hotkey, p, corresponding_elimination, position_locks, source_for_elimination, iteration_epoch)

    def handle_challenge_period_eliminations(self, position_locks, iteration_epoch=None):
        eliminations_with_reasons = self.challengeperiod_manager.eliminations_with_reasons
        if not eliminations_with_reasons:
            return

        hotkeys = list(eliminations_with_reasons.keys())
        for hotkey in hotkeys:
            if self.hotkey_in_eliminations(hotkey):
                continue
            elim_reason = eliminations_with_reasons[hotkey][0]
            elim_mdd = eliminations_with_reasons[hotkey][1]
            self.append_elimination_row(hotkey=hotkey, current_dd=elim_mdd, reason=elim_reason)
            self.handle_eliminated_miner(hotkey, {}, position_locks)
            self.contract_manager.slash_miner_collateral_proportion(hotkey)


        self.challengeperiod_manager.eliminations_with_reasons = {}

    def handle_first_refresh(self, position_locks, iteration_epoch=None):
        if self.is_backtesting or self.first_refresh_ran:
            return

        eliminated_hotkeys = self.get_eliminated_hotkeys()
        hotkey_to_positions = self.position_manager.get_positions_for_hotkeys(eliminated_hotkeys,
                                                                              only_open_positions=True)
        for hotkey, open_positions in hotkey_to_positions.items():
            if not open_positions:
                continue
            for p in open_positions:
                self.add_manual_flat_order(hotkey, p, self.hotkey_in_eliminations(hotkey), position_locks, None, iteration_epoch)

        self.first_refresh_ran = True

    def process_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Process all elimination checks and handle eliminated miners.

        Args:
            position_locks: PositionLocks instance. If None, uses self.position_locks.
                           Parameter kept for backward compatibility.
            iteration_epoch: Epoch captured at start of iteration. Used to detect stale data.
        """
        # Use provided position_locks or fall back to instance variable
        if position_locks is None:
            position_locks = self.position_locks

        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS) and \
                not bool(self.challengeperiod_manager.eliminations_with_reasons):
            return


        bt.logging.info(f"running elimination manager. invalidation data {dict(self.position_manager.perf_ledger_manager.perf_ledger_hks_to_invalidate)}")
        # Update departed hotkeys tracking first to detect re-registrations
        self._update_departed_hotkeys()
        self.handle_first_refresh(position_locks, iteration_epoch)
        self.handle_perf_ledger_eliminations(position_locks, iteration_epoch)
        self.handle_challenge_period_eliminations(position_locks, iteration_epoch)
        self.handle_mdd_eliminations(position_locks, iteration_epoch)
        self.handle_zombies(position_locks, iteration_epoch)
        self._delete_eliminated_expired_miners()

        self.set_last_update_time()

    def run_update_loop(self):
        """
        Run the elimination manager in a continuous loop in its own process.
        This method is designed to run in a separate process and will process
        eliminations continuously until shutdown is signaled.
        """
        import time
        import traceback
        from setproctitle import setproctitle
        from shared_objects.error_utils import ErrorUtils
        setproctitle("vali_EliminationManager")

        bt.logging.info("EliminationManager process started")

        while not self.shutdown_dict:
            try:
                # Check if sync is in progress and pause if so
                if self.sync_in_progress and self.sync_in_progress.value:
                    bt.logging.debug("EliminationManager: Sync in progress, pausing...")
                    time.sleep(1)
                    continue

                # Capture epoch at START of iteration
                iteration_epoch = self.sync_epoch.value if self.sync_epoch else None

                # Run the elimination process with captured epoch
                self.process_eliminations(iteration_epoch=iteration_epoch)

                # Sleep to avoid busy waiting. The refresh_allowed check in process_eliminations
                # will handle the actual refresh timing, but we sleep here to be nice
                # to the CPU when refresh is not allowed
                time.sleep(1)

            except Exception as e:
                error_traceback = traceback.format_exc()
                bt.logging.error(f"Error in EliminationManager update loop: {e}")
                bt.logging.error(error_traceback)

                # Send error notification to Slack
                if self.slack_notifier:
                    error_message = ErrorUtils.format_error_for_slack(
                        error=e,
                        traceback_str=error_traceback,
                        include_operation=True,
                        include_timestamp=True
                    )
                    self.slack_notifier.send_message(
                        f"❌ EliminationManager daemon error!\n{error_message}",
                        level="error"
                    )

                time.sleep(10)  # Wait longer on error before retrying

        bt.logging.info("EliminationManager process shutting down")

    def is_zombie_hotkey(self, hotkey, all_hotkeys_set):
        if hotkey in all_hotkeys_set:
            return False

        return True

    def sync_eliminations(self, dat) -> list:
        # log the difference in hotkeys
        hotkeys_before = set(x['hotkey'] for x in self.eliminations)
        hotkeys_after = set(x['hotkey'] for x in dat)
        removed = [x for x in hotkeys_before if x not in hotkeys_after]
        added = [x for x in hotkeys_after if x not in hotkeys_before]
        bt.logging.info(f'sync_eliminations: removed {len(removed)} {removed}, added {len(added)} {added}')
        # Update the list in place while keeping the reference intact:
        self.eliminations[:] = dat
        self.save_eliminations()
        return removed

    def hotkey_in_eliminations(self, hotkey):
        for x in self.eliminations:
            if x['hotkey'] == hotkey:
                return deepcopy(x)
        return None

    def _delete_eliminated_expired_miners(self):
        deleted_hotkeys = set()
        # self.eliminations were just refreshed in process_eliminations
        any_challenege_period_changes = False
        now_ms = TimeUtil.now_in_millis()
        metagraph_hotkeys_set = set(self.metagraph.hotkeys) if self.metagraph and self.metagraph.hotkeys else set()
        for x in self.eliminations:
            if self.shutdown_dict:
                return
            hotkey = x['hotkey']
            elimination_initiated_time_ms = x['elimination_initiated_time_ms']
            # Don't delete this miner until it hits the minimum elimination time.
            if now_ms - elimination_initiated_time_ms < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS:
                continue
            # We will not delete this miner's cache until it has been deregistered by BT
            if hotkey in metagraph_hotkeys_set:
                bt.logging.trace(f"miner [{hotkey}] has not been deregistered by BT yet. Not deleting miner dir.")
                continue

            # If the miner is no longer in the metagraph, we can remove them from the challengeperiod information
            if hotkey in self.challengeperiod_manager.active_miners:
                self.challengeperiod_manager.active_miners.pop(hotkey)
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
                f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. reason: [{x['reason']}]"
                f"Removing miner dir [{miner_dir}]"
            )
            deleted_hotkeys.add(hotkey)

        # Write the challengeperiod information to disk
        if any_challenege_period_changes:
            self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        if deleted_hotkeys:
            self.delete_eliminations(deleted_hotkeys)

    def save_eliminations(self):
        if not self.is_backtesting:
            self.write_eliminations_to_disk(self.eliminations)

    def write_eliminations_to_disk(self, eliminations):
        if not isinstance(eliminations, list):
            eliminations = list(eliminations)  # proxy list
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        bt.logging.trace(f"Writing [{len(eliminations)}] eliminations from memory to disk: {vali_eliminations}")
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, vali_eliminations)

    def clear_eliminations(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                                {CacheController.ELIMINATIONS: []})
        del self.eliminations[:]

    def get_eliminated_hotkeys(self):
        return set([x['hotkey'] for x in self.eliminations]) if self.eliminations else set()

    def get_eliminations_from_memory(self):
        return list(self.eliminations)  # ListProxy is not JSON serializable

    def get_eliminations_from_disk(self) -> list:
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
        elimination_row = self.generate_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                                        price_info=price_info, return_info=return_info)
        self.eliminations.append(elimination_row)
        self.eliminations[-1] = elimination_row  # ipc list does not update the object without using __setitem__
        self.save_eliminations()
        bt.logging.info(f"miner eliminated with hotkey [{hotkey}]. Info [{elimination_row}]")

    def delete_eliminations(self, deleted_hotkeys):
        # with self.eliminations_lock:
        items_to_remove = [x for x in self.eliminations if x['hotkey'] in deleted_hotkeys]
        for item in items_to_remove:
            self.eliminations.remove(item)
        self.save_eliminations()

    def handle_mdd_eliminations(self, position_locks, iteration_epoch=None):
        """
        Checks the mdd of each miner and eliminates any miners that surpass MAX_TOTAL_DRAWDOWN
        """
        from vali_objects.utils.ledger_utils import LedgerUtils
        bt.logging.info("checking main competition for maximum drawdown eliminations.")
        if self.shutdown_dict:
            return
        challengeperiod_success_hotkeys = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)

        filtered_ledger = self.position_manager.perf_ledger_manager.filtered_ledger_for_scoring(
            portfolio_only=True,
            hotkeys=challengeperiod_success_hotkeys)
        for miner_hotkey, ledger in filtered_ledger.items():
            if self.shutdown_dict:
                return
            if self.hotkey_in_eliminations(miner_hotkey):
                continue

            miner_exceeds_mdd, drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element=ledger)

            if miner_exceeds_mdd:
                self.append_elimination_row(miner_hotkey, drawdown_percentage, EliminationReason.MAX_TOTAL_DRAWDOWN.value)
                self.handle_eliminated_miner(miner_hotkey, {}, position_locks)
                self.contract_manager.slash_miner_collateral_proportion(miner_hotkey, ValiConfig.DRAWDOWN_SLASH_PROPORTION)


    def handle_zombies(self, position_locks, iteration_epoch=None):
        """
        If a miner is no longer in the metagraph and an elimination does not exist for them, we create an elimination
        row for them and add flat orders to their positions. If they have been a zombie for more than
        ELIMINATION_FILE_DELETION_DELAY_MS, delete them
        """
        if self.shutdown_dict or self.is_backtesting:
            return

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        all_hotkeys_set = set(self.metagraph.hotkeys) if self.metagraph and self.metagraph.hotkeys else set()

        for hotkey in CacheController.get_directory_names(all_miners_dir):
            corresponding_elimination = self.hotkey_in_eliminations(hotkey)
            elimination_reason = corresponding_elimination.get('reason') if corresponding_elimination else None
            if elimination_reason:
                continue  # already an elimination and marked for deletion
            elif self.is_zombie_hotkey(hotkey, all_hotkeys_set):
                self.append_elimination_row(hotkey=hotkey, current_dd=None, reason=EliminationReason.ZOMBIE.value)
                self.handle_eliminated_miner(hotkey, {}, position_locks, iteration_epoch)

    def _update_departed_hotkeys(self):
        """
        Track hotkeys that have departed from the metagraph (de-registered).
        Ignores anomalous changes that might indicate network issues.
        Should be called during process_eliminations to keep departed hotkeys up to date.
        """
        if self.is_backtesting:
            return

        current_hotkeys = set(self.metagraph.hotkeys) if self.metagraph and self.metagraph.hotkeys else set()
        lost_hotkeys = self.previous_metagraph_hotkeys - current_hotkeys
        gained_hotkeys = current_hotkeys - self.previous_metagraph_hotkeys

        # Log changes
        if lost_hotkeys:
            bt.logging.debug(f"Metagraph lost hotkeys: {lost_hotkeys}")
        if gained_hotkeys:
            bt.logging.debug(f"Metagraph gained hotkeys: {gained_hotkeys}")

        # Check for re-registered hotkeys
        departed_hotkeys_set = set(self.departed_hotkeys.keys())
        re_registered_hotkeys = gained_hotkeys & departed_hotkeys_set
        if re_registered_hotkeys:
            bt.logging.warning(
                f"Detected {len(re_registered_hotkeys)} re-registered miners: {re_registered_hotkeys}. "
                f"These hotkeys were previously de-registered and have re-registered. "
                f"Their orders will be rejected."
            )

        # Only track legitimate departures (not anomalous drops)
        is_anomalous, _ = is_anomalous_hotkey_loss(lost_hotkeys, len(self.previous_metagraph_hotkeys))
        if lost_hotkeys and not is_anomalous:
            # Add lost hotkeys to departed tracking
            new_departures = lost_hotkeys - departed_hotkeys_set
            if new_departures:
                current_time_ms = TimeUtil.now_in_millis()
                for hotkey in new_departures:
                    self.departed_hotkeys[hotkey] = {
                        "detected_ms": current_time_ms
                    }
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

        # Update previous hotkeys for next iteration
        self.previous_metagraph_hotkeys = current_hotkeys

    def is_hotkey_re_registered(self, hotkey: str) -> bool:
        """
        Check if a hotkey is re-registered (was previously de-registered and has re-registered).

        Args:
            hotkey: The hotkey to check

        Returns:
            True if the hotkey is in the metagraph AND in the departed_hotkeys dict, False otherwise
        """
        if not hotkey:
            return False

        current_hotkeys = set(self.metagraph.hotkeys) if self.metagraph and self.metagraph.hotkeys else set()

        # Re-registered if currently in metagraph AND previously departed (O(1) dict lookup)
        return hotkey in current_hotkeys and hotkey in self.departed_hotkeys

    def _get_departed_hotkeys_from_disk(self) -> dict:
        """Load departed hotkeys from disk.

        Tries to load from validation/departed_hotkeys.json (runtime file).
        If not found, falls back to data/default_departed_hotkeys.json (committed default).

        Returns:
            Dict mapping hotkey -> metadata dict with key: detected_ms
        """
        location = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=self.running_unit_tests)
        try:
            departed_data = ValiUtils.get_vali_json_file(location, DEPARTED_HOTKEYS_KEY)
            if departed_data is None:
                departed_data = {}
            # Handle legacy list format for backwards compatibility
            if isinstance(departed_data, list):
                bt.logging.info(f"Converting legacy departed hotkeys list to dict format")
                departed_data = {hotkey: {"detected_ms": 0} for hotkey in departed_data}
            bt.logging.trace(f"Loaded {len(departed_data)} departed hotkeys from disk. Dir: {location}")
            return departed_data
        except Exception as e:
            bt.logging.warning(f"Could not load departed hotkeys from disk: {e}. Trying default file...")
            # Fall back to default file committed to repo
            return self._get_departed_hotkeys_from_default_file()

    def _get_departed_hotkeys_from_default_file(self) -> dict:
        """Load departed hotkeys from the default file committed to the repository.

        This file (data/default_departed_hotkeys.json) contains all historically departed
        hotkeys and serves as a fallback when the runtime file doesn't exist.

        Returns:
            Dict mapping hotkey -> metadata dict with key: detected_ms
        """
        import os
        base_dir = ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests).replace('/validation/', '')
        default_location = os.path.join(base_dir, 'data', 'default_departed_hotkeys.json')

        try:
            departed_data = ValiUtils.get_vali_json_file(default_location, DEPARTED_HOTKEYS_KEY)
            if departed_data is None:
                departed_data = {}
            # Handle legacy list format for backwards compatibility
            if isinstance(departed_data, list):
                bt.logging.info(f"Converting legacy default departed hotkeys list to dict format")
                departed_data = {hotkey: {"detected_ms": 0} for hotkey in departed_data}
            bt.logging.info(f"Loaded {len(departed_data)} departed hotkeys from default file: {default_location}")
            return departed_data
        except Exception as e:
            bt.logging.warning(f"Could not load departed hotkeys from default file: {e}. Starting with empty dict.")
            return {}

    def _save_departed_hotkeys(self):
        """Save departed hotkeys to disk."""
        if not self.is_backtesting:
            departed_dict = dict(self.departed_hotkeys)  # Convert proxy dict to regular dict
            departed_data = {DEPARTED_HOTKEYS_KEY: departed_dict}
            bt.logging.trace(f"Writing {len(departed_dict)} departed hotkeys to disk")
            output_location = ValiBkpUtils.get_departed_hotkeys_dir(running_unit_tests=self.running_unit_tests)
            ValiBkpUtils.write_file(output_location, departed_data)
