# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import os
import shutil
import time
import traceback
from collections import defaultdict
from multiprocessing import Process
from pickle import UnpicklingError
from typing import List, Dict
import bittensor as bt
from pathlib import Path

from copy import deepcopy
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil, timeme
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.positions_to_snap import positions_to_snap
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import OrderStatus, OrderSource, Order
from vali_objects.utils.position_filtering import PositionFiltering

TARGET_MS = 1764145800000

class PositionManager(CacheController):
    def __init__(self, metagraph=None, running_unit_tests=False,
                 perform_order_corrections=False,
                 perform_compaction=False,
                 is_mothership=False, perf_ledger_manager=None,
                 challengeperiod_manager=None,
                 elimination_manager=None,
                 contract_manager=None,
                 secrets=None,
                 ipc_manager=None,
                 live_price_fetcher=None,
                 is_backtesting=False,
                 shared_queue_websockets=None,
                 split_positions_on_disk_load=False,
                 closed_position_daemon=False):

        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        # Populate memory with positions

        self.perf_ledger_manager = perf_ledger_manager
        self.challengeperiod_manager = challengeperiod_manager
        self.elimination_manager = elimination_manager
        self.shared_queue_websockets = shared_queue_websockets

        self.recalibrated_position_uuids = set()

        self.is_mothership = is_mothership
        self.perform_compaction = perform_compaction
        self.perform_order_corrections = perform_order_corrections
        self.split_positions_on_disk_load = split_positions_on_disk_load

        # Track splitting statistics
        self.split_stats = defaultdict(self._default_split_stats)

        self.contract_manager = contract_manager
        if contract_manager:
            self.cached_miner_account_sizes = deepcopy(self.contract_manager.miner_account_sizes)
        else:
            self.cached_miner_account_sizes = {}

        if ipc_manager:
            self.hotkey_to_positions = ipc_manager.dict()
        else:
            self.hotkey_to_positions = {}
        self.secrets = secrets
        self.live_price_fetcher = live_price_fetcher
        self._populate_memory_positions_for_first_time()
        if closed_position_daemon:
            self.compaction_process = Process(target=self.run_closed_position_daemon_forever, daemon=True)
            self.compaction_process.start()
            bt.logging.info("Started run_closed_position_daemon_forever process.")

    def run_closed_position_daemon_forever(self):
        #try:
        #    self.ensure_position_consistency_serially()
        #except Exception as e:
        #    bt.logging.error(f"Error {e} in initial ensure_position_consistency_serially: {traceback.format_exc()}")
        while True:
            try:
                t0 = time.time()
                self.compact_price_sources()
                bt.logging.info(f'compacted price sources in {time.time() - t0:.2f} seconds')
            except Exception as e:
                bt.logging.error(f"Error {e} in run_closed_position_daemon_forever: {traceback.format_exc()}")
                time.sleep(ValiConfig.PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS)
            time.sleep(ValiConfig.PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS)

    def _default_split_stats(self):
        """Default split statistics for each miner. Used to make defaultdict pickleable."""
        return {
            'n_positions_split': 0,
            'product_return_pre_split': 1.0,
            'product_return_post_split': 1.0
        }

    @timeme
    def _populate_memory_positions_for_first_time(self):
        """
        Load positions from disk into memory and apply position splitting if enabled.
        """
        if self.is_backtesting:
            return

        initial_hk_to_positions = self.get_positions_for_all_miners(from_disk=True)

        # Apply position splitting if enabled on disk load
        if self.split_positions_on_disk_load:
            bt.logging.info("Applying position splitting on disk load...")
            total_hotkeys = len(initial_hk_to_positions)
            hotkeys_with_splits = 0
            hotkeys_with_errors = []

            for hk, positions in initial_hk_to_positions.items():
                split_positions = []
                positions_split_for_hotkey = 0

                for position in positions:
                    try:
                        # Split the position and track stats
                        new_positions, split_info = self.split_position_on_flat(position, track_stats=True)
                        split_positions.extend(new_positions)

                        # Count if this position was actually split
                        if len(new_positions) > 1:
                            positions_split_for_hotkey += 1

                    except Exception as e:
                        bt.logging.error(f"Failed to split position {position.position_uuid} for hotkey {hk}: {e}")
                        bt.logging.error(f"Position details: {len(position.orders)} orders, trade_pair={position.trade_pair}")
                        traceback.print_exc()
                        # Keep the original position if splitting fails
                        split_positions.append(position)
                        if hk not in hotkeys_with_errors:
                            hotkeys_with_errors.append(hk)

                # Track if this hotkey had any splits
                if positions_split_for_hotkey > 0:
                    hotkeys_with_splits += 1

                # Update with split positions
                initial_hk_to_positions[hk] = split_positions

            # Log comprehensive splitting statistics
            self._log_split_stats()

            # Log summary for all hotkeys
            bt.logging.info("=" * 60)
            bt.logging.info("POSITION SPLITTING SUMMARY")
            bt.logging.info("=" * 60)
            bt.logging.info(f"Total hotkeys processed: {total_hotkeys}")
            bt.logging.info(f"Hotkeys with positions split: {hotkeys_with_splits}")
            bt.logging.info(f"Hotkeys with no splits needed: {total_hotkeys - hotkeys_with_splits - len(hotkeys_with_errors)}")
            if hotkeys_with_errors:
                bt.logging.error(f"Hotkeys with splitting errors: {len(hotkeys_with_errors)}")
                for hk in hotkeys_with_errors[:5]:  # Show first 5 errors
                    bt.logging.error(f"  - {hk}")
                if len(hotkeys_with_errors) > 5:
                    bt.logging.error(f"  ... and {len(hotkeys_with_errors) - 5} more")
            bt.logging.info("=" * 60)

        # Load positions into memory
        for hk, positions in initial_hk_to_positions.items():
            if positions:
                self.hotkey_to_positions[hk] = positions

    def ensure_position_consistency_serially(self):
        """
        Ensures position consistency by checking all closed positions for return calculation changes
        and updating them to disk if needed. This should be called before starting main processing loops.
        """
        if self.is_backtesting:
            return

        if not self.live_price_fetcher:
            self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets, disable_ws=True)

        start_time = time.time()
        last_log_time = start_time
        n_positions_checked_for_change = 0
        successful_updates = 0
        failed_updates = 0

        # Calculate total positions for progress tracking
        total_positions = sum(len([p for p in positions if not p.is_open_position])
                            for positions in self.hotkey_to_positions.values())
        bt.logging.info(f'Starting position consistency check on {total_positions} closed positions...')

        # Check all positions and immediately save if return changed
        for hk_index, (hk, positions) in enumerate(self.hotkey_to_positions.items()):
            for p in positions:
                if p.is_open_position:
                    continue
                n_positions_checked_for_change += 1
                original_return = p.return_at_close
                p.rebuild_position_with_updated_orders(self.live_price_fetcher)
                new_return = p.return_at_close
                if new_return != original_return:
                    try:
                        self.save_miner_position(p, delete_open_position_if_exists=False)
                        successful_updates += 1
                    except Exception as e:
                        failed_updates += 1
                        bt.logging.error(f'Failed to update position {p.position_uuid} for hotkey {hk}: {e}')

                # Log progress every 1000 positions or every 5 minutes
                current_time = time.time()
                if n_positions_checked_for_change % 1000 == 0 or (current_time - last_log_time) >= 300:
                    elapsed = current_time - start_time
                    progress_pct = (n_positions_checked_for_change / total_positions) * 100 if total_positions > 0 else 0
                    bt.logging.info(
                        f'Position consistency progress: {n_positions_checked_for_change}/{total_positions} '
                        f'({progress_pct:.1f}%) checked, {successful_updates} updated, {failed_updates} failed. '
                        f'Elapsed: {elapsed:.1f}s'
                    )
                    if (current_time - last_log_time) >= 300:
                        last_log_time = current_time

        # Log final results
        elapsed = time.time() - start_time
        if successful_updates > 0 or failed_updates > 0:
            bt.logging.warning(
                f'Position consistency completed: Updated {successful_updates} positions out of {n_positions_checked_for_change} checked '
                f'for return changes due to difference in return calculation. '
                f'({failed_updates} failures). Serial updates completed in {elapsed:.2f} seconds.'
            )
        else:
            bt.logging.info(f'Position consistency completed: No positions needed return updates out of {n_positions_checked_for_change} checked in {elapsed:.2f} seconds.')

    def filtered_positions_for_scoring(
            self,
            hotkeys: List[str] = None
    ) -> (Dict[str, List[Position]], Dict[str, int]):
        """
        Filter the positions for a set of hotkeys.
        """
        if hotkeys is None:
            hotkeys = self.get_miner_hotkeys_with_at_least_one_position()

        hk_to_first_order_time = {}
        filtered_positions = {}
        for hotkey, miner_positions in self.get_positions_for_hotkeys(hotkeys, sort_positions=True).items():
            if miner_positions:
                hk_to_first_order_time[hotkey] = min([p.orders[0].processed_ms for p in miner_positions])
                filtered_positions[hotkey] = PositionFiltering.filter_positions_for_duration(miner_positions)

        return filtered_positions, hk_to_first_order_time

    def pre_run_setup(self):
        """
        Run this outside of init so that cross object dependencies can be set first. See validator.py
        """
        if self.perform_order_corrections:
            try:
                self.apply_order_corrections()
                time_now_ms = TimeUtil.now_in_millis()
                if time_now_ms < TARGET_MS:
                    self.close_open_orders_for_suspended_trade_pairs()
            except Exception as e:
                bt.logging.error(f"Error applying order corrections: {e}")
                traceback.print_exc()

    def give_erronously_eliminated_miners_another_shot(self, hotkey_to_positions):
        time_now_ms = TimeUtil.now_in_millis()
        if time_now_ms > TARGET_MS:
            return
        # The MDD Checker will immediately eliminate miners if they exceed the maximum drawdown
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        eliminations_to_delete = set()
        for e in eliminations:
            if e['hotkey'] in ('5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X',
                               '5E9Ppyn5DzHGaPQmsHVnkNJDjGd7DstqjHWZpQhWPMbqzNex',
                               '5DoCFr2EoW1CGuYCEXhsuQdWRsgiUMuxGwNt4Xqb5TCptcBW',
                               '5EHpm2UK3CyhH1zZiJmM6erGrzkmVAF9EnT1QLSPhMzQaQHG',
                               '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU',
                               '5CALivVcJBTjYJFMsAkqhppQgq5U2PYW4HejCajHMvTMUgkC',
                               '5FTR8y26ap56vvahaxbB4PYxSkTQFpkQDqZN32uTVcW9cKjy',
                               '5Et6DsfKyfe2PBziKo48XNsTCWst92q8xWLdcFy6hig427qH',
                               '5HYRAnpjhcT45f6udFAbfJXwUmqqeaNvte4sTjuQvDxTaQB3',
                               '5Cd9bVVja2KdgsTiR7rTAh7a4UKVfnAuYAW1bs8BiedUE9JN',
                               '5FmvpMPvurA896m1X19fZXnct3NRXFrY57XVRcQLupb4sNZs',
                               '5DXRG8rCuuF7Lkd46mMbkdDNq52kDdph5PbxrCLAhuKAwkdq',
                               '5CcsBjaLAVfrjsAh6FyaTK4rBikkfQVanEmespwVpDGcE7jP',
                               '5DqxA5rsR5FGCkoZQ2eDnpQu1dBrdqr6EU7ZFKqsnHQQvpVh',
                               '5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ'):
                bt.logging.warning('Removed elimination for hotkey ', e['hotkey'])
                positions = hotkey_to_positions.get(e['hotkey'])
                if positions:
                    self.reopen_force_closed_positions(positions)
                eliminations_to_delete.add(e)

        self.elimination_manager.delete_eliminations(eliminations_to_delete)

    @staticmethod
    def strip_old_price_sources(position: Position, time_now_ms: int) -> int:
        n_removed = 0
        one_week_ago_ms = time_now_ms - 1000 * 60 * 60 * 24 * 7
        for o in position.orders:
            if o.processed_ms < one_week_ago_ms:
                if o.price_sources:
                    o.price_sources = []
                    n_removed += 1
        return n_removed

    def correct_for_tp(self, positions: List[Position], idx, prices, tp, timestamp_ms=None, n_attempts=0,
                       n_corrections=0, unique_corrections=None, pos=None):
        n_attempts += 1
        i = -1

        if pos:
            i = idx
        else:
            for p in positions:
                if p.trade_pair == tp:
                    pos = p
                    i += 1
                    if i == idx:
                        break

            if i != idx:
                bt.logging.warning(f"Could not find position for trade pair {tp.trade_pair_id} at index {idx}. i {i}")
                return n_attempts, n_corrections

        if pos and timestamp_ms:
            # check if the timestamp_ms is outside of 5 minutes of the position's open_ms
            delta_time_min = abs(timestamp_ms - pos.open_ms) / 1000.0 / 60.0
            if delta_time_min > 5.0:
                bt.logging.warning(
                    f"Timestamp ms: {timestamp_ms} is more than 5 minutes away from position open ms: {pos.open_ms}. delta_time_min {delta_time_min}")
                return n_attempts, n_corrections

        if not prices:
            # del position
            if pos:
                self.delete_position(pos)
                unique_corrections.add(pos.position_uuid)
                n_corrections += 1
                return n_attempts, n_corrections

        elif i == idx and pos and len(prices) <= len(pos.orders):
            self.delete_position(pos)
            for i in range(len(prices)):
                pos.orders[i].price = prices[i]

            old_return = pos.return_at_close  # noqa: F841
            pos.rebuild_position_with_updated_orders(self.live_price_fetcher)
            self.save_miner_position(pos, delete_open_position_if_exists=False)
            unique_corrections.add(pos.position_uuid)
            n_corrections += 1
            return n_attempts, n_corrections
        else:
            bt.logging.warning(
                f"Could not correct position for trade pair {tp.trade_pair_id}. i {i}, idx {idx}, len(prices) {len(prices)}, len(pos.orders) {len(pos.orders)}")
        return n_attempts, n_corrections

    def reopen_force_closed_positions(self, positions):
        for position in positions:
            if position.is_closed_position and abs(position.net_leverage) > 0:
                print('rac1:', position.return_at_close)
                print(
                    f"Deleting position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id} nl {position.net_leverage}")
                self.delete_position(position)
                position.reopen_position()
                position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                print('rac2:', position.return_at_close)
                self.save_miner_position(position, delete_open_position_if_exists=False)
                print(f"Reopened position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id}")

    @timeme
    def compact_price_sources(self):
        time_now = TimeUtil.now_in_millis()
        cutoff_time_ms = time_now - 10 * ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS # Generous bound
        n_price_sources_removed = 0
        hotkey_to_positions = self.get_positions_for_all_miners(sort_positions=True)
        for hotkey, positions in hotkey_to_positions.items():
            for position in positions:
                if position.is_open_position:
                    continue # Don't modify open positions as we don't want to deal with locking
                elif any(o.processed_ms > cutoff_time_ms for o in position.orders):
                    continue # Could be subject to retro price correction and we don't want to deal with locking

                n = self.strip_old_price_sources(position, time_now)
                if n:
                    n_price_sources_removed += n
                    self.save_miner_position(position, delete_open_position_if_exists=False)

        bt.logging.info(f'Removed {n_price_sources_removed} price sources from old data.')

    def dedupe_positions(self, positions, miner_hotkey):
        positions_by_trade_pair = defaultdict(list)
        n_positions_deleted = 0
        n_orders_deleted = 0
        n_positions_rebuilt_with_new_orders = 0
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(deepcopy(position))

        for trade_pair, positions in positions_by_trade_pair.items():
            position_uuid_to_dedupe = {}
            for p in positions:
                if p.position_uuid in position_uuid_to_dedupe:
                    # Replace if it has more orders
                    if len(p.orders) > len(position_uuid_to_dedupe[p.position_uuid].orders):
                        old_position = position_uuid_to_dedupe[p.position_uuid]
                        self.delete_position(old_position)
                        position_uuid_to_dedupe[p.position_uuid] = p
                        n_positions_deleted += 1
                    else:
                        self.delete_position(p)
                        n_positions_deleted += 1
                else:
                    position_uuid_to_dedupe[p.position_uuid] = p

            for position in position_uuid_to_dedupe.values():
                order_uuid_to_dedup = {}
                new_orders = []
                any_orders_deleted = False
                for order in position.orders:
                    if order.order_uuid in order_uuid_to_dedup:
                        n_orders_deleted += 1
                        any_orders_deleted = True
                    else:
                        new_orders.append(order)
                        order_uuid_to_dedup[order.order_uuid] = order
                if any_orders_deleted:
                    position.orders = new_orders
                    position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                    self.save_miner_position(position, delete_open_position_if_exists=False)
                    n_positions_rebuilt_with_new_orders += 1
        if n_positions_deleted or n_orders_deleted or n_positions_rebuilt_with_new_orders:
            bt.logging.warning(
                f"Hotkey {miner_hotkey}: Deleted {n_positions_deleted} duplicate positions and {n_orders_deleted} "
                f"duplicate orders across {n_positions_rebuilt_with_new_orders} positions.")

    @timeme
    def apply_order_corrections(self):
        """
        This is our mechanism for manually synchronizing validator orders in situations where a bug prevented an
        order from filling. We are working on a more robust automated synchronization/recovery system.

        11/4/2024 - Metagraph synchronization was set to 5 minutes preventing a new miner from having their orders
        processed by all validators. After verifying that this miner's order should have been sent to all validators,
        we increased the metagraph update frequency to 1 minute to prevent this from happening again. This override
        will correct the order status for this miner.

        4/13/2024 - Price recalibration incorrectly applied to orders made after TwelveData websocket prices were
        implemented. This regressed pricing since the websocket prices are more accurate.

        Errantly closed out open CADCHF positions during a recalibration. Delete these positions that adversely affected
        miners

        One miner was eliminated due to a faulty candle from polygon at the close. We are investigating a workaround
        and have several candidate solutions.

        miner couldn't close position due to temporary bug. deleted position completely.

        # 4/15/24 Verified high lag on order price using Twelve Data

        # 4/17/24 Verified duplicate order sent due to a miner.py script. deleting entire position.

        # 4/19/24 Verified bug on old version of miner.py that delayed order significantly. The PR to reduce miner lag went
         live April 14th and this trade was April 9th

         4/23/24 - position price source flipped from polygon to TD. Need to be consistent within a position.
          Fix coming in next update.

          4/26/24, 5/9/24 - extreme price parsing is giving outliers from bad websocket data. Patch the function and manually correct
          elimination.

          Bug in forex market close due to federal holiday logic 5/27/24. deleted position

          5/30/24 - duplicate order bug. miner.py script updated.

          5.31.24 - validator outage due to twelvedata thread error. add position if not exists.

        """
        now_ms = TimeUtil.now_in_millis()
        if now_ms > TARGET_MS:
            return

        hotkey_to_positions = self.get_positions_for_all_miners(sort_positions=True)
        #self.give_erronously_eliminated_miners_another_shot(hotkey_to_positions)
        n_corrections = 0
        n_attempts = 0
        unique_corrections = set()
        # Wipe miners only once when dynamic challenge period launches
        miners_to_wipe = []
        miners_to_promote = []
        position_uuids_to_delete = []
        wipe_positions = False
        reopen_force_closed_orders = False
        current_eliminations = self.elimination_manager.get_eliminations_from_memory()
        if now_ms < TARGET_MS:
            # All miners that wanted their challenge period restarted
            miners_to_wipe = []   # All miners that should have been promoted
            position_uuids_to_delete = []
            miners_to_promote = []

            for p in positions_to_snap:
                try:
                    pos = Position(**p)
                    hotkey = pos.miner_hotkey
                    # if this hotkey is eliminated, log an error and continue
                    if any(e['hotkey'] == hotkey for e in current_eliminations):
                        bt.logging.error(f"Hotkey {hotkey} is eliminated. Skipping position {pos}.")
                        continue
                    if pos.is_open_position:
                        self.delete_open_position_if_exists(pos)
                    self.save_miner_position(pos)
                    print(f"Added position {pos.position_uuid} for trade pair {pos.trade_pair.trade_pair_id} for hk {pos.miner_hotkey}")
                except Exception as e:
                    print(f"Error adding position {p} {e}")

        #Don't accidentally promote eliminated miners
        for e in current_eliminations:
            if e['hotkey'] in miners_to_promote:
                miners_to_promote.remove(e['hotkey'])

        # Promote miners that would have passed challenge period
        for miner in miners_to_promote:
            if miner in self.challengeperiod_manager.active_miners:
                if self.challengeperiod_manager.active_miners[miner][0] != MinerBucket.MAINCOMP:
                    self.challengeperiod_manager._promote_challengeperiod_in_memory([miner], now_ms)
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        # Wipe miners_to_wipe below
        for k in miners_to_wipe:
            if k not in hotkey_to_positions:
                hotkey_to_positions[k] = []

        n_eliminations_before = len(self.elimination_manager.get_eliminations_from_memory())
        for e in self.elimination_manager.get_eliminations_from_memory():
            if e['hotkey'] in miners_to_wipe:
                self.elimination_manager.delete_eliminations([e['hotkey']])
                print(f"Removed elimination for hotkey {e['hotkey']}")
        n_eliminations_after = len(self.elimination_manager.get_eliminations_from_memory())
        print(f'    n_eliminations_before {n_eliminations_before} n_eliminations_after {n_eliminations_after}')
        update_perf_ledgers = False
        for miner_hotkey, positions in hotkey_to_positions.items():
            n_attempts += 1
            self.dedupe_positions(positions, miner_hotkey)
            if miner_hotkey in miners_to_wipe: # and now_ms < TARGET_MS:
                update_perf_ledgers = True
                bt.logging.info(f"Resetting hotkey {miner_hotkey}")
                n_corrections += 1
                unique_corrections.update([p.position_uuid for p in positions])
                for pos in positions:
                    if wipe_positions:
                        self.delete_position(pos)
                    elif pos.position_uuid in position_uuids_to_delete:
                        print(f'Deleting position {pos.position_uuid} for trade pair {pos.trade_pair.trade_pair_id} for hk {pos.miner_hotkey}')
                        self.delete_position(pos)
                    elif reopen_force_closed_orders:
                        if any((o.src in (1,3)) for o in pos.orders):
                            pos.orders = [o for o in pos.orders if (o.src in (0,2,4))]
                            pos.rebuild_position_with_updated_orders(self.live_price_fetcher)
                            self.save_miner_position(pos)
                            print(f'Removed eliminated orders from position {pos}')
                if miner_hotkey in self.challengeperiod_manager.active_miners:
                    self.challengeperiod_manager.active_miners.pop(miner_hotkey)
                    print(f'Removed challengeperiod status for {miner_hotkey}')

                self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        if update_perf_ledgers:
            perf_ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
            print('n perf ledgers before:', len(perf_ledgers))
            perf_ledgers_new = {k:v for k,v in perf_ledgers.items() if k not in miners_to_wipe}
            print('n perf ledgers after:', len(perf_ledgers_new))
            self.perf_ledger_manager.save_perf_ledgers(perf_ledgers_new)


            """
            if miner_hotkey == '5Cd9bVVja2KdgsTiR7rTAh7a4UKVfnAuYAW1bs8BiedUE9JN' and now_ms < TARGET_MS:
                position_that_should_exist_raw = {"miner_hotkey": "5Cd9bVVja2KdgsTiR7rTAh7a4UKVfnAuYAW1bs8BiedUE9JN",
                                                  "position_uuid": "f5a54d87-26c4-4a73-91b3-d8607b898507", "open_ms": 1734077788550,
                                                  "trade_pair": TradePair.USDJPY, "orders":
                                                      [{"order_type": "LONG", "leverage": 0.25, "price": 152.865, "processed_ms": 1734077788550, "order_uuid": "f5a54d87-26c4-4a73-91b3-d8607b898507", "price_sources": [], "src": 0},
                                                       {"order_type": "LONG", "leverage": 0.25, "price": 153.846, "processed_ms": 1734424931078, "order_uuid": "a53bd995-ad81-4b98-8039-5991abc00374", "price_sources": [], "src": 0},
                                                       {"order_type": "FLAT", "leverage": 0.25, "price": 153.656, "processed_ms": 1734517608513, "order_uuid": "3572eabe-4a4c-4fa2-8262-bf2a8e8ea394", "price_sources": [], "src": 0}],
                                                  "current_return": 1.0009828934026757, "close_ms": 1734517608513, "return_at_close": 1.000926976973672,
                                                  "net_leverage": 0.0, "average_entry_price": 153.3555, "position_type": "FLAT", "is_closed_position": True}

                success = self.enforce_position_state(position_that_should_exist_raw, TradePair.USDJPY, miner_hotkey,
                                                      unique_corrections, overwrite=True)
                n_corrections += success
                n_attempts += 1

            if miner_hotkey == "5HYBzAsTcxDXxHNXBpUJAQ9ZwmaGTwTb24ZBGJpELpG7LPGf" and now_ms < TARGET_MS:
                position_that_should_exist_raw = \
                {"miner_hotkey": "5HYBzAsTcxDXxHNXBpUJAQ9ZwmaGTwTb24ZBGJpELpG7LPGf",
                 "position_uuid": "c1be3244-5125-4bd6-83b7-9f56c84b3387", "open_ms": 1736389802186,
                 "trade_pair": TradePair.BTCUSD, "orders": [
                    {"order_type": "SHORT", "leverage": -0.5, "price": 94432.48, "processed_ms": 1736389802186,
                     "order_uuid": "c1be3244-5125-4bd6-83b7-9f56c84b3387", "price_sources": [
                        {"source": "Polygon_ws", "timespan_ms": 0, "open": 94432.48, "close": 94432.48,
                         "vwap": 94432.48, "high": 94432.48, "low": 94432.48, "start_ms": 1736389802000,
                         "websocket": False, "lag_ms": 186, "volume": 0.04655431},
                        {"source": "Tiingo_gdax_rest", "timespan_ms": 0, "open": 94431.06, "close": 94431.06,
                         "vwap": 94431.06, "high": 94431.06, "low": 94431.06, "start_ms": 1736389800615,
                         "websocket": True, "lag_ms": 1571, "volume": None},
                        {"source": "Polygon_rest", "timespan_ms": 1000, "open": 94237.5, "close": 94200.0,
                         "vwap": 94243.0749, "high": 94246.12, "low": 94200.0, "start_ms": 1736390000000,
                         "websocket": False, "lag_ms": 197814, "volume": 0.01125985}], "src": 0},
                    {"order_type": "FLAT", "leverage": 0.5, "price": 93908.85, "processed_ms": 1736395887370,
                     "order_uuid": "da0075dd-b97a-4cb4-a7d2-8c4e074101c5", "price_sources": [
                        {"source": "Polygon_ws", "timespan_ms": 0, "open": 93908.85, "close": 93908.85,
                         "vwap": 93908.85, "high": 93908.85, "low": 93908.85, "start_ms": 1736395887000,
                         "websocket": True, "lag_ms": 370, "volume": 1.3e-05},
                        {"source": "Tiingo_gdax_rest", "timespan_ms": 0, "open": 93908.85, "close": 93908.85,
                         "vwap": 93908.85, "high": 93908.85, "low": 93908.85, "start_ms": 1736395886709,
                         "websocket": True, "lag_ms": 661, "volume": None}], "src": 0}],
                 "current_return": 1.0027725100516263, "close_ms": 1736395887370, "return_at_close": 1.0022180496021578,
                 "net_leverage": 0.0, "average_entry_price": 94432.48, "position_type": "FLAT",
                 "is_closed_position": True}
                success = self.enforce_position_state(position_that_should_exist_raw, TradePair.BTCUSD, miner_hotkey, unique_corrections)
                n_corrections += success
                n_attempts += 1


            
                    
            if miner_hotkey == '5DX8tSyGrx1QuoR1wL99TWDusvmmWgQW5su3ik2Sc8y8Mqu3':
                n_corrections += self.correct_for_tp(positions, 0, [151.83500671, 151.792], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                n_corrections += self.correct_for_tp(positions, 0, [0.66623, 0.66634], TradePair.CADCHF, unique_corrections)

            if miner_hotkey == '5D4zieKMoRVm477oUyMTZAWZ9orzpiJM8K6ufQQjryiXwpGU':
                n_corrections += self.correct_for_tp(positions, 0, [0.66634, 0.6665], TradePair.CADCHF, unique_corrections)

            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                n_corrections += self.correct_for_tp(positions, 0, None, TradePair.CADCHF, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 0, [151.841, 151.773], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 1, [151.8, 152.302], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5Ec93qtHkKprEaA5EWXrmPmWppMeMiwaY868bpxfkH5ocBxi':
                n_corrections += self.correct_for_tp(positions, 0, [151.808, 151.844], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 1, [151.817, 151.84], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 2, [151.839, 151.809], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 3, [151.772, 151.751], TradePair.USDJPY, unique_corrections)
                n_corrections += self.correct_for_tp(positions, 4, [151.77, 151.748], TradePair.USDJPY, unique_corrections)

            if miner_hotkey == '5Ct1J2jNxb9zeHpsj547BR1nZk4ZD51Bb599tzEWnxyEr4WR':
                n_corrections += self.correct_for_tp(positions, 0, None, TradePair.CADCHF, unique_corrections)
                
            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                correct_for_tp(positions, 2, None, TradePair.EURCHF, timestamp_ms=1712950839925)
            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                correct_for_tp(positions, 0, [0.66242, 0.66464], TradePair.CADCHF)
            if miner_hotkey == '5D4zieKMoRVm477oUyMTZAWZ9orzpiJM8K6ufQQjryiXwpGU':
                correct_for_tp(positions, 0, [111.947, 111.987], TradePair.CADJPY)
            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                correct_for_tp(positions, 0, [151.727, 151.858, 153.0370, 153.0560, 153.0720, 153.2400, 153.2280, 153.2400], TradePair.USDJPY)
            if miner_hotkey == '5DfhKZckZwjCqEcBUsW7jwzA5APCdj5SgZbfK6zzS9bMPuHn':
                correct_for_tp(positions, 0, [111.599, 111.55999756, 111.622], TradePair.CADJPY)
                
            if miner_hotkey == '5C5dGkAZ8P58Rcm7abWwsKRv91h8aqTsvVak2ogJ6wpxSZPw':
                correct_for_tp(positions, 0, [151.73, 151.862, 153.047, 153.051, 153.071, 153.241, 153.225, 153.235], TradePair.USDJPY)
            if miner_hotkey == '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx':
                correct_for_tp(positions, 0, None, TradePair.ETHUSD, timestamp_ms=1713102534971)
            
            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                correct_for_tp(positions, 1, [100.192, 100.711, 100.379], TradePair.AUDJPY)
                correct_for_tp(positions, 1, None, TradePair.GBPJPY, timestamp_ms=1712624748605)
                correct_for_tp(positions, 2, None, TradePair.AUDCAD, timestamp_ms=1712839053529)
                
            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                n_attempts, n_corrections = self.correct_for_tp(positions, 1, None, TradePair.BTCUSD, timestamp_ms=1712671378202, n_attempts=n_attempts, n_corrections=n_corrections, unique_corrections=unique_corrections)

            if miner_hotkey == '5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw':
                n_attempts, n_corrections = self.correct_for_tp(positions, 3, [1.36936, 1.36975], TradePair.USDCAD, n_attempts=n_attempts,
                                                                n_corrections=n_corrections,
                                                                unique_corrections=unique_corrections)
                                                                
            if miner_hotkey == '5Dxqzduahnqw8q3XSUfTcEZGU7xmAsfJubhHZwvXVLN9fSjR':
                self.reopen_force_closed_positions(positions)
                n_corrections += 1
                n_attempts += 1

            if miner_hotkey == '5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t':
                #with open(ValiBkpUtils.get_positions_override_dir() + miner_hotkey + '.json', 'w') as f:
                #    dat = [p.to_json_string() for p in positions]
                #    f.write(json.dumps(dat, cls=CustomEncoder))


                time_now_ms = TimeUtil.now_in_millis()
                if time_now_ms > TARGET_MS:
                    return
                n_attempts += 1
                self.restore_from_position_override(miner_hotkey)
                n_corrections += 1

            if miner_hotkey == "5G3ys2356ovgUivX3endMP7f37LPEjRkzDAM3Km8CxQnErCw":
                time_now_ms = TimeUtil.now_in_millis()
                if time_now_ms > TARGET_MS:
                    return
                position_to_delete = [x for x in positions if x.trade_pair == TradePair.NZDUSD][-1]
                n_attempts, n_corrections = self.correct_for_tp(positions, None, None, TradePair.NZDUSD,
                                                                timestamp_ms=1716906327000, n_attempts=n_attempts,
                                                                n_corrections=n_corrections,
                                                                unique_corrections=unique_corrections,
                                                                pos=position_to_delete)
                                                                
            if miner_hotkey == "5DWmX9m33Tu66Qh12pr41Wk87LWcVkdyM9ZSNJFsks3QritF":
                 time_now_ms = TimeUtil.now_in_millis()
                 if time_now_ms > TARGET_MS:
                     return
                 position_to_delete = sorted([x for x in positions if x.trade_pair == TradePair.SPX], key=lambda x: x.close_ms)[-1]
                 n_attempts, n_corrections = self.correct_for_tp(positions, None, None, TradePair.SPX,
                                                                 timestamp_ms=None, n_attempts=n_attempts,
                                                                 n_corrections=n_corrections,
                                                                 unique_corrections=unique_corrections,
                                                                 pos=position_to_delete)
        """


        #5DCzvCF22vTVhXLtGrd7dBy19iFKKJNxmdSp5uo4C4v6Xx6h
        bt.logging.warning(
            f"Applied {n_corrections} order corrections out of {n_attempts} attempts. unique positions corrected: {len(unique_corrections)}")


    def enforce_position_state(self, position_that_should_exist_raw, trade_pair, miner_hotkey, unique_corrections, overwrite=False):
        position_that_should_exist_raw['trade_pair'] = trade_pair
        for o in position_that_should_exist_raw['orders']:
            o['trade_pair'] = trade_pair
        position = Position.from_dict(position_that_should_exist_raw)
        # check if the position exists on the filesystem
        existing_disk_positions = self.get_positions_for_one_hotkey(miner_hotkey)
        position_exists = False
        for p in existing_disk_positions:
            if p.position_uuid == position.position_uuid:
                position_exists = True
                break
        if not position_exists or overwrite:
            self.save_miner_position(position, delete_open_position_if_exists=True)
            print(f"Added position {position.position_uuid} for trade pair {position.trade_pair.trade_pair_id}")
            unique_corrections.add(position.position_uuid)
            return True
        return False

    def close_open_orders_for_suspended_trade_pairs(self):
        if not self.live_price_fetcher:
            self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets, disable_ws=True)
        tps_to_eliminate = [TradePair.SPX, TradePair.DJI, TradePair.NDX, TradePair.VIX,
                            TradePair.AUDJPY, TradePair.CADJPY, TradePair.CHFJPY,
                            TradePair.EURJPY, TradePair.NZDJPY, TradePair.GBPJPY, TradePair.USDJPY,
                            TradePair.USDMXN]
        if not tps_to_eliminate:
            return
        all_positions = self.get_positions_for_all_miners(sort_positions=True)
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        bt.logging.info(f"Found {len(eliminations)} eliminations on disk.")
        for hotkey, positions in all_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            # Closing all open positions for the specified trade pair
            for position in positions:
                if position.is_closed_position:
                    continue
                if position.trade_pair in tps_to_eliminate:
                    price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(position.trade_pair, TARGET_MS)
                    live_price = price_sources[0].parse_appropriate_price(TARGET_MS, position.trade_pair.is_forex, OrderType.FLAT, position.orders[0].order_type)
                    flat_order = Order(price=live_price,
                                       price_sources=price_sources,
                                       processed_ms=TARGET_MS,
                                       order_uuid=position.position_uuid[::-1], # deterministic across validators. Won't mess with p2p sync
                                       trade_pair=position.trade_pair,
                                       order_type=OrderType.FLAT,
                                       quantity=-position.net_quantity,
                                       src=OrderSource.DEPRECATION_FLAT)

                    position.add_order(flat_order, self.live_price_fetcher)
                    self.save_miner_position(position, delete_open_position_if_exists=True)
                    if self.shared_queue_websockets:
                        self.shared_queue_websockets.put(position.to_websocket_dict())
                    bt.logging.info(
                    f"Position {position.position_uuid} for hotkey {hotkey} and trade pair {position.trade_pair.trade_pair_id} has been closed. Added flat order {flat_order}")


    @staticmethod
    def get_return_per_closed_position(positions: List[Position]) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if position.is_open_position:
                continue
            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms
            closed_position_returns.append(position.return_at_close)

        cumulative_return = 1
        per_position_return = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return *= value
            per_position_return.append(cumulative_return)
        return per_position_return

    @staticmethod
    def get_percent_profitable_positions(positions: List[Position]) -> float:
        if len(positions) == 0:
            return 0.0

        profitable_positions = 0
        n_closed_positions = 0

        for position in positions:
            if position.is_open_position:
                continue

            n_closed_positions += 1
            if position.return_at_close > 1.0:
                profitable_positions += 1

        if n_closed_positions == 0:
            return 0.0

        return profitable_positions / n_closed_positions

    @staticmethod
    def positions_are_the_same(position1: Position, position2: Position | dict) -> (bool, str):
        # Iterate through all the attributes of position1 and compare them to position2.
        # Get attributes programmatically.
        comparing_to_dict = isinstance(position2, dict)
        for attr in dir(position1):
            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if attr.startswith("_") or callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property) \
                    or (attr in ('model_computed_fields', 'model_config', 'model_fields', 'model_fields_set', 'newest_order_age_ms')):
                continue

            value1 = getattr(position1, attr)
            # Check if position2 is a dict and access the value accordingly.
            if comparing_to_dict:
                # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                value2 = position2.get(attr)
            else:
                value2 = getattr(position2, attr, None)

            if value1 != value2:
                return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    def get_miner_position_by_uuid(self, hotkey:str, position_uuid: str) -> Position | None:
        if hotkey not in self.hotkey_to_positions:
            return None
        return self._position_from_list_of_position(hotkey, position_uuid)

    def get_recently_updated_miner_hotkeys(self):
        """
        Identifies and returns a list of directories that have been updated in the last 3 days.
        """
        # Define the path to the directory containing the directories to check
        query_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        # Get the current time
        current_time = time.time()
        # List of directories updated in the last 24 hours
        updated_directory_names = []
        # Get the names of all directories in query_dir
        directory_names = CacheController.get_directory_names(query_dir)
        # Loop through each directory name
        for item in directory_names:
            item_path = Path(query_dir) / item  # Construct the full path
            # Get the last modification time of the directory
            root_last_modified_time_s = self._get_file_mod_time_s(item_path)
            latest_modification_time_s = self._get_latest_file_modification_time_s(item_path, root_last_modified_time_s)
            # Check if the directory was updated in the last 3 days
            if current_time - latest_modification_time_s < 259200:  # 3 days in seconds
                updated_directory_names.append(item)

        return updated_directory_names

    def _get_latest_file_modification_time_s(self, dir_path, root_last_modified_time):
        """
        Recursively finds the max modification time of all files within a directory.
        """
        latest_mod_time_s = root_last_modified_time
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = Path(root) / file
                mod_time = self._get_file_mod_time_s(file_path)
                latest_mod_time_s = max(latest_mod_time_s, mod_time)

        return latest_mod_time_s

    def _get_file_mod_time_s(self, file_path):
        try:
            return os.path.getmtime(file_path)
        except OSError:  # Handle the case where the file is inaccessible
            return 0

    def delete_open_position_if_exists(self, position: Position) -> None:
        # See if we need to delete the open position file
        open_position = self.get_open_position_for_a_miner_trade_pair(position.miner_hotkey,
                                                                      position.trade_pair.trade_pair_id)
        if open_position:
            self.delete_position(open_position)

    def verify_open_position_write(self, miner_dir, updated_position):
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        # Print all files found for dir
        positions = [self._get_position_from_disk(file) for file in all_files]
        if len(positions) == 0:
            return  # First time open position is being saved
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(
                f"More than one open position for miner {updated_position.miner_hotkey} and trade_pair."
                f" {updated_position.trade_pair.trade_pair_id}. Please restore cache. Positions: {positions}")
        elif len(positions) == 1:
            if positions[0].position_uuid != updated_position.position_uuid:
                msg = (
                    f"Attempted to write open position {updated_position.position_uuid} for miner {updated_position.miner_hotkey} "
                    f"and trade_pair {updated_position.trade_pair.trade_pair_id} but found an existing open"
                    f" position with a different position_uuid {positions[0].position_uuid}.")
                raise ValiRecordsMisalignmentException(msg)

        # -------------------------------------------------------------------------------------
        # Make sure the memory positions match the disk positions. Only run this during test
        if not self.running_unit_tests:
            return

        cdf = miner_dir[:-5] + 'closed/'
        positions.extend([self._get_position_from_disk(file) for file in ValiBkpUtils.get_all_files_in_dir(cdf)])

        temp = self.hotkey_to_positions.get(updated_position.miner_hotkey, [])
        positions_memory_by_position_uuid = {}
        for position in temp:
            if position.trade_pair == updated_position.trade_pair:
                positions_memory_by_position_uuid[position.position_uuid] = position
        positions_disk_by_uuid = {p.position_uuid: p for p in positions}
        errors = []
        for position_uuid, position in positions_memory_by_position_uuid.items():
            if position_uuid not in positions_disk_by_uuid:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found in memory but not on disk.")
                continue
            disk_position = positions_disk_by_uuid[position_uuid]
            is_same, diff = self.positions_are_the_same(position, disk_position)
            if not is_same:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found in memory but does not match the position on disk. {diff}")

        for position_uuid, position in positions_disk_by_uuid.items():
            if position_uuid not in positions_memory_by_position_uuid:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found on disk but not in memory.")
                continue
            memory_position = positions_memory_by_position_uuid[position_uuid]
            is_same, diff = self.positions_are_the_same(memory_position, position)
            if not is_same:
                errors.append(
                    f"Position {position_uuid} for miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id} "
                    f"found on disk but does not match the position in memory. {diff}")
        if errors:
            raise ValiRecordsMisalignmentException(
                f"Found errors in miner {updated_position.miner_hotkey} and trade_pair {updated_position.trade_pair.trade_pair_id}. Errors: {errors}."
                f" Disk positions: {positions_disk_by_uuid.keys()}. Memory positions: {positions_memory_by_position_uuid.keys()}. all files {all_files}")
        # -------------------------------------------------------------------------------------

    def _position_from_list_of_position(self, hotkey, position_uuid):
        for p in self.hotkey_to_positions.get(hotkey, []):
            if p.position_uuid == position_uuid:
                return deepcopy(p)  # for unit tests we deepcopy. ipc cache never returns a reference.
        return None

    def get_existing_positions(self, hotkey: str):
        return self.hotkey_to_positions.get(hotkey, [])

    def _save_miner_position_to_memory(self, position: Position):
        # Multiprocessing-safe
        hk = position.miner_hotkey
        existing_positions = self.get_existing_positions(hk)

        # Sanity check
        if position.miner_hotkey in self.hotkey_to_positions and position.position_uuid in existing_positions:
            existing_pos = self._position_from_list_of_position(position.miner_hotkey, position.position_uuid)
            assert existing_pos.trade_pair == position.trade_pair, f"Trade pair mismatch for position {position.position_uuid}. Existing: {existing_pos.trade_pair}, New: {position.trade_pair}"

        new_positions = [p for p in existing_positions if p.position_uuid != position.position_uuid]
        new_positions.append(deepcopy(position))
        self.hotkey_to_positions[hk] = new_positions  # Trigger the update on the multiprocessing Manager


    def save_miner_position(self, position: Position, delete_open_position_if_exists=True) -> None:
        if not self.is_backtesting:
            miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                         position.trade_pair.trade_pair_id,
                                                                         order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                         running_unit_tests=self.running_unit_tests)
            if position.is_closed_position and delete_open_position_if_exists:
                self.delete_open_position_if_exists(position)
            elif position.is_open_position:
                self.verify_open_position_write(miner_dir, position)

            #print(f'Saving position {position.position_uuid} for miner {position.miner_hotkey} and trade pair {position.trade_pair.trade_pair_id} is_open {position.is_open_position}')
            ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)
        self._save_miner_position_to_memory(position)

    def overwrite_position_on_disk(self, position: Position) -> None:
        # delete the position from disk. Try the open position dir and the closed position dir
        self.delete_position(position, check_open_and_closed_dirs=True)
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(position.miner_hotkey,
                                                                     position.trade_pair.trade_pair_id,
                                                                     order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                                                                     running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)
        self._save_miner_position_to_memory(position)

    def clear_all_miner_positions(self, target_hotkey=None):
        self.hotkey_to_positions = {}
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            if target_hotkey and file != target_hotkey:
                continue
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def get_number_of_eliminations(self):
        return len(self.elimination_manager.eliminations)

    def get_number_of_miners_with_any_positions(self):
        ans = 0
        for k, v in self.hotkey_to_positions.items():
            if len(v) > 0:
                ans += 1
        return ans

    def get_extreme_position_order_processed_on_disk_ms(self):
        dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        min_time = float("inf")
        max_time = 0
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                continue
            hotkey = file
            # Read all positions in this directory
            positions = self.get_positions_for_one_hotkey(hotkey)
            for p in positions:
                for o in p.orders:
                    min_time = min(min_time, o.processed_ms)
                    max_time = max(max_time, o.processed_ms)
        return min_time, max_time

    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        temp = self.hotkey_to_positions.get(hotkey, [])
        positions = []
        for p in temp:
            if p.trade_pair.trade_pair_id == trade_pair_id and p.is_open_position:
                positions.append(p)
        if len(positions) > 1:
            raise ValiRecordsMisalignmentException(f"More than one open position for miner {hotkey} and trade_pair."
                                                   f" {trade_pair_id}. Please restore cache. Positions: {positions}")
        return deepcopy(positions[0]) if len(positions) == 1 else None

    def get_filepath_for_position(self, hotkey, trade_pair_id, position_uuid, is_open):
        order_status = OrderStatus.CLOSED if not is_open else OrderStatus.OPEN
        return ValiBkpUtils.get_partitioned_miner_positions_dir(hotkey, trade_pair_id, order_status=order_status,
                                                                running_unit_tests=self.running_unit_tests) + position_uuid

    def delete_position(self, p: Position, check_open_and_closed_dirs=False):
        hotkey = p.miner_hotkey
        trade_pair_id = p.trade_pair.trade_pair_id
        position_uuid = p.position_uuid
        is_open = p.is_open_position
        if check_open_and_closed_dirs:
            file_paths = [self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, True),
                          self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, False)]
        else:
            file_paths = [self.get_filepath_for_position(hotkey, trade_pair_id, position_uuid, is_open)]
        for fp in file_paths:
            if not self.is_backtesting:
                if os.path.exists(fp):
                    os.remove(fp)
                    bt.logging.info(f"Deleted position from disk: {fp}")
            self._delete_position_from_memory(hotkey, position_uuid)

    def _delete_position_from_memory(self, hotkey, position_uuid):
        if hotkey in self.hotkey_to_positions:
            new_positions = [p for p in self.hotkey_to_positions[hotkey] if p.position_uuid != position_uuid]
            if new_positions:
                self.hotkey_to_positions[hotkey] = new_positions
            else:
                del self.hotkey_to_positions[hotkey]

    def calculate_net_portfolio_leverage(self, hotkey: str) -> float:
        """
        Calculate leverage across all open positions
        Normalize each asset class with a multiplier
        """
        positions = self.get_positions_for_one_hotkey(hotkey, only_open_positions=True)

        portfolio_leverage = 0.0
        for position in positions:
            portfolio_leverage += abs(position.get_net_leverage()) * position.trade_pair.leverage_multiplier

        return portfolio_leverage

    @timeme
    def get_positions_for_all_miners(self, from_disk=False, **args) -> dict[str, list[Position]]:
        if from_disk:
            all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
                ValiBkpUtils.get_miner_dir(self.running_unit_tests)
            )
        else:
            all_miner_hotkeys = list(self.hotkey_to_positions.keys())
        return self.get_positions_for_hotkeys(all_miner_hotkeys, from_disk=from_disk, **args)

    @staticmethod
    def positions_to_dashboard_dict(original_positions: list[Position], time_now_ms) -> dict:
        ans = {
            "positions": [],
            "thirty_day_returns": 1.0,
            "all_time_returns": 1.0,
            "n_positions": 0,
            "percentage_profitable": 0.0
        }
        acceptable_position_end_ms = TimeUtil.timestamp_to_millis(
            TimeUtil.generate_start_timestamp(
                ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
            ))
        positions_30_days = [
            position
            for position in original_positions
            if position.open_ms > acceptable_position_end_ms
        ]
        ps_30_days = PositionFiltering.filter_positions_for_duration(positions_30_days)
        return_per_position = PositionManager.get_return_per_closed_position(ps_30_days)
        if len(return_per_position) > 0:
            curr_return = return_per_position[len(return_per_position) - 1]
            ans["thirty_day_returns"] = curr_return

        ps_all_time = PositionFiltering.filter_positions_for_duration(original_positions)
        return_per_position = PositionManager.get_return_per_closed_position(ps_all_time)
        if len(return_per_position) > 0:
            curr_return = return_per_position[len(return_per_position) - 1]
            ans["all_time_returns"] = curr_return
            ans["n_positions"] = len(ps_all_time)
            ans["percentage_profitable"] = PositionManager.get_percent_profitable_positions(ps_all_time)

        for p in original_positions:
            # Don't modify the position object in-place
            # Instead, create the dict representation and modify only the dict
            PositionManager.strip_old_price_sources(p, time_now_ms)

            position_dict = json.loads(str(p), cls=GeneralizedJSONDecoder)
            # Convert None to 0 for JSON serialization (avoids null in JSON)
            # This is safe because we're only modifying the dict, not the position object
            if position_dict.get('close_ms') is None:
                position_dict['close_ms'] = 0

            ans["positions"].append(position_dict)
        return ans

    def _get_account_size_for_order(self, position, order):
        """
        temp method:
        Get the miner's account size for an order
        """
        COLLATERAL_START_TIME_MS = 1755302399000
        if order.processed_ms < COLLATERAL_START_TIME_MS:
            return ValiConfig.DEFAULT_CAPITAL

        if not self.contract_manager:
            return ValiConfig.DEFAULT_CAPITAL

        account_size = self.contract_manager.get_miner_account_size(
                position.miner_hotkey, order.processed_ms, records_dict=self.cached_miner_account_sizes)
        return account_size if account_size is not None else ValiConfig.MIN_CAPITAL

    def _migrate_order_quantities(self, position: Position) -> int:
        """
        temp method:
        Migrate old orders that only have leverage to include quantity.
        Returns number of orders migrated.
        """
        migrated_count = 0

        for order in position.orders:
            if order.quantity is None and order.leverage is not None:
                order.value = order.leverage * position.account_size
                order.quantity = order.value / (order.price * position.trade_pair.lot_size)

                migrated_count += 1

        if migrated_count > 0:
            bt.logging.info(
                f"Migrated order {position.orders[0].order_uuid}: "
                f"leverage={position.orders[0].leverage} → quantity={position.orders[0].quantity}. "
                f"Total order migrations for {position.position_uuid}: {migrated_count}"
            )

        return migrated_count

    def _get_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        file_string = None
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.model_validate_json(file_string)
            if not ans.orders:
                bt.logging.warning(f"Anomalous position has no orders: {ans.to_dict()}")
            else:
                # temp logic:
                # populate order quantity and value field for historical orders.
                needs_order_migration = any(o.quantity is None and o.leverage is not None for o in ans.orders)

                # check if account_size needs migration
                needs_account_size_migration = (ans.account_size == 0 or ans.account_size is None)

                if needs_order_migration or needs_account_size_migration:
                    # Fix account_size first (needed for order quantity calculation)
                    if needs_account_size_migration and ans.orders:
                        ans.account_size = self._get_account_size_for_order(ans, ans.orders[0])
                        bt.logging.info(
                            f"Migrated account_size for position {ans.position_uuid}: "
                            f"0 → {ans.account_size}"
                        )
                    # migrate order quantities if needed
                    if needs_order_migration:
                        self._migrate_order_quantities(ans)

                    # Rebuild to recalculate all position-level fields
                    ans.rebuild_position_with_updated_orders(self.live_price_fetcher)
                    self.save_miner_position(ans, delete_open_position_if_exists=False)
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException(f"Vali position file is missing {file}")
        except UnpicklingError as e:
            raise ValiBkpCorruptDataException(f"file_string is {file_string}, {e}")
        except UnicodeDecodeError as e:
            raise ValiBkpCorruptDataException(
                f" Error {e} for file {file} You may be running an old version of the software. Confirm with the team if you should delete your cache. file string {file_string[:2000] if file_string else None}")
        except Exception as e:
            raise ValiBkpCorruptDataException(f"Error {e} file_path {file} file_string: {file_string}")

    @staticmethod
    def sort_by_close_ms(_position):
        """
        Sort key function for positions.
        Closed positions are sorted by close_ms (ascending).
        Open positions are sorted to the end (infinity).

        This is the canonical sorting method used throughout the codebase.
        """
        return (
            _position.close_ms if _position.is_closed_position else float("inf")
        )

    def exorcise_positions(self, positions, all_files) -> List[Position]:
        """
        1/7/24: Not needed anymore?
        Disk positions can be left in a bad state for a variety of reasons. Let's clean them up here.
        If a dup is encountered, deleted both and let position syncing add the correct one back.
        """
        filtered_positions = []
        position_uuid_to_count = defaultdict(int)
        order_uuid_to_count = defaultdict(int)
        order_uuids_to_purge = set()
        for position in positions:
            position_uuid_to_count[position.position_uuid] += 1
            for order in position.orders:
                order_uuid_to_count[order.order_uuid] += 1
                if order_uuid_to_count[order.order_uuid] > 1:
                    order_uuids_to_purge.add(order.order_uuid)

        for file_name, position in zip(all_files, positions):
            if position_uuid_to_count[position.position_uuid] > 1:
                bt.logging.info(f"Exorcising position from disk due to duplicate position uuid: {file_name} {position}")
                os.remove(file_name)
                continue

            elif not position.orders:
                bt.logging.info(f"Exorcising position from disk due to no orders: {file_name} {position.to_dict()}")
                os.remove(file_name)
                continue

            new_orders = [x for x in position.orders if order_uuid_to_count[x.order_uuid] == 1]
            if len(new_orders) != len(position.orders):
                bt.logging.info(f"Exorcising position from disk due to order mismatch: {file_name} {position}")
                os.remove(file_name)
            else:
                filtered_positions.append(position)
        return filtered_positions

    def get_positions_for_one_hotkey(self,
                                     miner_hotkey: str,
                                     only_open_positions: bool = False,
                                     sort_positions: bool = False,
                                     acceptable_position_end_ms: int = None,
                                     from_disk: bool = False
                                     ) -> List[Position]:

        if from_disk:
            miner_dir = ValiBkpUtils.get_miner_all_positions_dir(miner_hotkey,
                                                                 running_unit_tests=self.running_unit_tests)
            all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
            positions = [self._get_position_from_disk(file) for file in all_files]
        else:
            positions = self.hotkey_to_positions.get(miner_hotkey, [])

        if acceptable_position_end_ms is not None:
            positions = [
                position
                for position in positions
                if position.open_ms > acceptable_position_end_ms
            ]

        if only_open_positions:
            positions = [
                position for position in positions if position.is_open_position
            ]

        if sort_positions:
            positions = sorted(positions, key=self.sort_by_close_ms)

        return positions

    def get_positions_for_hotkeys(self, hotkeys: List[str], eliminations: List = None, **args) -> Dict[
        str, List[Position]]:
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations) if eliminations is not None else set()

        return {
            hotkey: self.get_positions_for_one_hotkey(hotkey, **args)
            for hotkey in hotkeys
            if hotkey not in eliminated_hotkeys
        }

    def get_miner_hotkeys_with_at_least_one_position(self) -> set[str]:
        return set(self.hotkey_to_positions.keys())

    def compute_realtime_drawdown(self, hotkey: str) -> float:
        """
        Compute the realtime drawdown from positions.
        Bypasses perf ledger, since perf ledgers are refreshed in 5 min intervals and may be out of date.
        Used to enable realtime withdrawals based on drawdown.
        """
        # 1. Get existing perf ledger to access historical max portfolio value
        existing_bundle = self.perf_ledger_manager.get_perf_ledgers(
            portfolio_only=True,
            from_disk=False
        )
        portfolio_ledger = existing_bundle.get(hotkey, {}).get('portfolio')

        if not portfolio_ledger or not portfolio_ledger.cps:
            bt.logging.warning(f"No perf ledger found for {hotkey}")
            return 1.0

        # 2. Get historical max portfolio value from existing checkpoints
        portfolio_ledger.init_max_portfolio_value()  # Ensures max_return is set
        max_portfolio_value = portfolio_ledger.max_return

        # 3. Calculate current portfolio value with live prices
        current_portfolio_value = self._calculate_current_portfolio_value(hotkey)

        # 4. Calculate current drawdown
        if max_portfolio_value <= 0:
            return 1.0

        drawdown = min(1.0, current_portfolio_value / max_portfolio_value)

        print(f"Real-time drawdown for {hotkey}: "
                f"{(1-drawdown)*100:.2f}% "
                f"(current: {current_portfolio_value:.4f}, "
                f"max: {max_portfolio_value:.4f})")

        return drawdown

    def _calculate_current_portfolio_value(self, miner_hotkey: str) -> float:
        """
        Calculate current portfolio value with live prices.
        """
        positions = self.get_positions_for_one_hotkey(
            miner_hotkey,
            only_open_positions=False
        )

        if not positions:
            return 1.0  # No positions = starting value

        portfolio_return = 1.0
        now_ms = TimeUtil.now_in_millis()

        for position in positions:
            if position.is_open_position:
                # Get live price for open positions
                price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(
                    position.trade_pair,
                    now_ms
                )

                if price_sources and price_sources[0]:
                    realtime_price = price_sources[0].close
                    # Calculate return with fees at this moment
                    position_return = position.get_open_position_return_with_fees(
                        realtime_price,
                        self.live_price_fetcher,
                        now_ms
                    )
                    portfolio_return *= position_return
                else:
                    # Fallback to last known return
                    portfolio_return *= position.return_at_close
            else:
                # Use stored return for closed positions
                portfolio_return *= position.return_at_close

        return portfolio_return

    def _log_split_stats(self):
        """Log statistics about position splitting."""
        bt.logging.info("=" * 60)
        bt.logging.info("POSITION SPLITTING STATISTICS")
        bt.logging.info("=" * 60)

        total_splits = 0
        for hotkey, stats in self.split_stats.items():
            if stats['n_positions_split'] > 0:
                bt.logging.info(f"Hotkey: {hotkey}")
                bt.logging.info(f"  Number of positions split: {stats['n_positions_split']}")
                bt.logging.info(f"  Product of returns pre-split: {stats['product_return_pre_split']:.6f}")
                bt.logging.info(f"  Product of returns post-split: {stats['product_return_post_split']:.6f}")
                total_splits += stats['n_positions_split']

        bt.logging.info(f"Total positions split across all hotkeys: {total_splits}")
        bt.logging.info("=" * 60)

    def _find_split_points(self, position: Position) -> list[int]:
        """
        Find all valid split points in a position where splitting should occur.
        Returns a list of order indices where splits should happen.
        This is the single source of truth for split logic.
        """
        if len(position.orders) < 2:
            return []

        split_points = []
        cumulative_leverage = 0.0
        previous_sign = None

        for i, order in enumerate(position.orders):
            previous_leverage = cumulative_leverage
            cumulative_leverage += order.leverage

            # Determine the sign of leverage (positive, negative, or zero)
            current_sign = None
            if abs(cumulative_leverage) < 1e-9:
                current_sign = 0
            elif cumulative_leverage > 0:
                current_sign = 1
            else:
                current_sign = -1

            # Check for leverage sign flip
            leverage_flipped = False
            if previous_sign is not None and previous_sign != 0 and current_sign != 0 and previous_sign != current_sign:
                leverage_flipped = True

            # Check for explicit FLAT or implicit flat (leverage reaches zero or flips sign)
            is_explicit_flat = order.order_type == OrderType.FLAT
            is_implicit_flat = (abs(cumulative_leverage) < 1e-9 or leverage_flipped) and not is_explicit_flat

            if is_explicit_flat or is_implicit_flat:
                # Don't split if this is the last order
                if i < len(position.orders) - 1:
                    # Check if the split would create valid sub-positions
                    orders_before = position.orders[:i+1]
                    orders_after = position.orders[i+1:]

                    # Check if first part is valid (2+ orders, doesn't start with FLAT)
                    first_valid = (len(orders_before) >= 2 and
                                 orders_before[0].order_type != OrderType.FLAT)

                    # Check if second part would be valid (at least 1 order, doesn't start with FLAT)
                    second_valid = (len(orders_after) >= 1 and
                                  orders_after[0].order_type != OrderType.FLAT)

                    if first_valid and second_valid:
                        split_points.append(i)
                        cumulative_leverage = 0.0  # Reset for next segment
                        previous_sign = 0
                        continue

            # Update previous sign for next iteration
            previous_sign = current_sign

        return split_points

    def _position_needs_splitting(self, position: Position) -> bool:
        """
        Check if a position would actually be split by split_position_on_flat.
        Uses the same logic as split_position_on_flat but without creating new positions.
        """
        return len(self._find_split_points(position)) > 0

    def split_position_on_flat(self, position: Position, track_stats: bool = False) -> tuple[list[Position], dict]:
        """
        Takes a position, iterates through the orders, and splits the position into multiple positions
        separated by FLAT orders OR when cumulative leverage reaches zero or flips sign (implicit flat).

        Implicit flat is defined as:
        - Cumulative leverage reaches zero (abs(cumulative_leverage) < 1e-9), OR
        - Cumulative leverage flips sign (e.g., from positive to negative or vice versa)

        Uses _find_split_points as the single source of truth for split logic.
        Ensures:
        - CLOSED positions have at least 2 orders
        - OPEN positions can have 1 order
        - No position starts with a FLAT order

        If track_stats is True, updates split_stats with splitting information.

        Returns:
            tuple: (list of positions, split_info dict with 'implicit_flat_splits' and 'explicit_flat_splits')
        """
        try:
            split_points = self._find_split_points(position)

            if not split_points:
                return [position], {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}

            # Track pre-split return if requested
            pre_split_return = position.return_at_close if track_stats else None

            # Count implicit vs explicit flats
            implicit_flat_splits = 0
            explicit_flat_splits = 0

            cumulative_leverage = 0.0
            previous_sign = None

            for i, order in enumerate(position.orders):
                cumulative_leverage += order.leverage

                # Determine the sign of leverage (positive, negative, or zero)
                current_sign = None
                if abs(cumulative_leverage) < 1e-9:
                    current_sign = 0
                elif cumulative_leverage > 0:
                    current_sign = 1
                else:
                    current_sign = -1

                # Check for leverage sign flip
                leverage_flipped = False
                if previous_sign is not None and previous_sign != 0 and current_sign != 0 and previous_sign != current_sign:
                    leverage_flipped = True

                if i in split_points:
                    if order.order_type == OrderType.FLAT:
                        explicit_flat_splits += 1
                    elif abs(cumulative_leverage) < 1e-9 or leverage_flipped:
                        implicit_flat_splits += 1

                # Update previous sign for next iteration
                previous_sign = current_sign

            # Create order groups based on split points
            order_groups = []
            start_idx = 0

            for split_idx in split_points:
                # Add orders up to and including the split point
                order_group = position.orders[start_idx:split_idx + 1]
                order_groups.append(order_group)
                start_idx = split_idx + 1

            # Add remaining orders if any
            if start_idx < len(position.orders):
                order_groups.append(position.orders[start_idx:])

            # Update the original position with the first group
            position.orders = order_groups[0]
            position.rebuild_position_with_updated_orders(self.live_price_fetcher)

            positions = [position]

            # Create new positions for remaining groups
            for order_group in order_groups[1:]:
                new_position = Position(miner_hotkey=position.miner_hotkey,
                                        position_uuid=order_group[0].order_uuid,
                                        open_ms=0,
                                        trade_pair=position.trade_pair,
                                        orders=order_group)
                new_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                positions.append(new_position)

            split_info = {
                'implicit_flat_splits': implicit_flat_splits,
                'explicit_flat_splits': explicit_flat_splits
            }

        except Exception as e:
            bt.logging.error(f"Error during position splitting for {position.miner_hotkey}: {e}")
            bt.logging.error(f"Position UUID: {position.position_uuid}, Orders: {len(position.orders)}")
            # Return original position on error
            return [position], {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}

        # Track stats if requested
        if track_stats and pre_split_return is not None:
            hotkey = position.miner_hotkey
            self.split_stats[hotkey]['n_positions_split'] += 1
            self.split_stats[hotkey]['product_return_pre_split'] *= pre_split_return

            # Calculate post-split product of returns
            for pos in positions:
                if pos.is_closed_position:
                    self.split_stats[hotkey]['product_return_post_split'] *= pos.return_at_close

        return positions, split_info

if __name__ == '__main__':
    from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
    from vali_objects.utils.elimination_manager import EliminationManager
    from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
    from vali_utils import ValiUtils
    bt.logging.enable_info()

    plm = PerfLedgerManager(None)
    secrets = ValiUtils.get_secrets()
    lpf = LivePriceFetcher(secrets, disable_ws=True)
    pm = PositionManager(perf_ledger_manager=plm, live_price_fetcher=lpf)
    elimination_manager = EliminationManager(None, pm, None)
    cpm = ChallengePeriodManager(None, position_manager=pm)
    pm.challengeperiod_manager = cpm
    pm.elimination_manager = elimination_manager
    pm.apply_order_corrections()
