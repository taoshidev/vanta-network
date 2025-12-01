# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import json
import math
import os
import shutil
import time
import traceback
from collections import defaultdict
from pickle import UnpicklingError
from typing import List, Dict
import bittensor as bt
from pathlib import Path
from copy import deepcopy
from shared_objects.cache_controller import CacheController
from shared_objects.rpc_service_base import RPCServiceBase
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
from vali_objects.utils.position_splitter import PositionSplitter

TARGET_MS = 1764145800000

class PositionManager(RPCServiceBase, CacheController):
    def __init__(self, metagraph=None, running_unit_tests=False,
                 perform_order_corrections=False,
                 perform_compaction=False,
                 is_mothership=False, perf_ledger_manager=None,
                 challengeperiod_manager=None,
                 elimination_manager=None,
                 secrets=None,
                 live_price_fetcher=None,
                 is_backtesting=False,
                 websocket_notifier=None,
                 split_positions_on_disk_load=False,
                 closed_position_daemon=False,
                 slack_notifier=None):

        # Initialize RPCServiceBase
        RPCServiceBase.__init__(
            self,
            service_name=ValiConfig.RPC_POSITIONMANAGER_SERVICE_NAME,
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )

        # Initialize CacheController
        CacheController.__init__(self, metagraph=metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)

        self.perf_ledger_manager = perf_ledger_manager
        self.challengeperiod_manager = challengeperiod_manager
        self.elimination_manager = elimination_manager
        self.websocket_notifier = websocket_notifier

        self.recalibrated_position_uuids = set()

        self.is_mothership = is_mothership
        self.perform_compaction = perform_compaction
        self.perform_order_corrections = perform_order_corrections
        self.split_positions_on_disk_load = split_positions_on_disk_load

        # Track splitting statistics
        self.split_stats = defaultdict(self._default_split_stats)

        self.secrets = secrets
        self.live_price_fetcher = live_price_fetcher
        self.start_compaction_daemon = closed_position_daemon

        # Store dependencies needed for server creation
        self.is_backtesting = is_backtesting

        # Start the RPC service (this replaces _start_direct_server and _start_rpc_server)
        self._initialize_service()

        # Position splitting (if enabled) now happens on the server side during startup
        # Compaction daemon (if enabled) now runs on the server side to avoid RPC overhead

    def _create_direct_server(self):
        """Create direct in-memory instance for tests"""
        from vali_objects.utils.position_manager_server import PositionManagerServer

        # When split_positions_on_disk_load is enabled, force disk loading even in test mode
        load_from_disk = True if self.split_positions_on_disk_load else None

        return PositionManagerServer(
            running_unit_tests=self.running_unit_tests,
            is_backtesting=self.is_backtesting,
            load_from_disk=load_from_disk,
            split_positions_on_disk_load=self.split_positions_on_disk_load
        )

    def _start_server_process(self, address, authkey, server_ready):
        """Start RPC server in separate process"""
        from multiprocessing import Process
        from vali_objects.utils.position_manager_server import start_position_manager_server

        # Get EliminationManager address/authkey if elimination_manager exists
        elimination_manager_address = None
        elimination_manager_authkey = None
        if self.elimination_manager and hasattr(self.elimination_manager, '_address') and hasattr(self.elimination_manager, '_authkey'):
            elimination_manager_address = self.elimination_manager._address
            elimination_manager_authkey = self.elimination_manager._authkey

        process = Process(
            target=start_position_manager_server,
            args=(
                address,
                authkey,
                self.running_unit_tests,
                self.is_backtesting,
                self.split_positions_on_disk_load,
                self.start_compaction_daemon,
                server_ready,
                elimination_manager_address,
                elimination_manager_authkey
            ),
            daemon=True
        )
        process.start()
        return process

    def _default_split_stats(self):
        """Default split statistics for each miner. Used to make defaultdict pickleable."""
        return {
            'n_positions_split': 0,
            'product_return_pre_split': 1.0,
            'product_return_post_split': 1.0
        }

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

        # Get all hotkeys from RPC server
        all_hotkeys = self._server_proxy.get_all_hotkeys_rpc()

        # Calculate total positions for progress tracking
        total_positions = 0
        for hk in all_hotkeys:
            positions = self._server_proxy.get_positions_for_one_hotkey_rpc(hk, only_open_positions=False)
            total_positions += len([p for p in positions if not p.is_open_position])

        bt.logging.info(f'Starting position consistency check on {total_positions} closed positions...')

        # Check all positions and immediately save if return changed
        for hk in all_hotkeys:
            positions = self._server_proxy.get_positions_for_one_hotkey_rpc(hk, only_open_positions=False)
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
            hotkeys: List[str] = None,
            include_development_positions=False
    ) -> (Dict[str, List[Position]], Dict[str, int]):
        """
        Filter the positions for a set of hotkeys.
        Excludes development positions by default.
        """
        if hotkeys is None:
            # Get hotkeys from position manager - already filtered by include_development_positions
            hotkeys = list(self.get_miner_hotkeys_with_at_least_one_position(include_development_positions=include_development_positions))
        else:
            # Hotkeys were provided explicitly - filter them if needed
            if not include_development_positions:
                hotkeys = [hk for hk in hotkeys if hk != ValiConfig.DEVELOPMENT_HOTKEY]

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
            if self.challengeperiod_manager.has_miner(miner):
                if self.challengeperiod_manager.get_miner_bucket(miner) != MinerBucket.MAINCOMP:
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
                if self.challengeperiod_manager.has_miner(miner_hotkey):
                    self.challengeperiod_manager.remove_miner(miner_hotkey)
                    print(f'Removed challengeperiod status for {miner_hotkey}')

                self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        if update_perf_ledgers:
            perf_ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
            print('n perf ledgers before:', len(perf_ledgers))
            perf_ledgers_new = {k:v for k,v in perf_ledgers.items() if k not in miners_to_wipe}
            print('n perf ledgers after:', len(perf_ledgers_new))
            self.perf_ledger_manager.save_perf_ledgers(perf_ledgers_new)

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
                    live_price = price_sources[0].parse_appropriate_price(TARGET_MS, position.trade_pair.is_forex, OrderType.FLAT, position)
                    flat_order = Order(price=live_price,
                                       price_sources=price_sources,
                                       processed_ms=TARGET_MS,
                                       order_uuid=position.position_uuid[::-1],
                                       # determinstic across validators. Won't mess with p2p sync
                                       trade_pair=position.trade_pair,
                                       order_type=OrderType.FLAT,
                                       leverage=0,
                                       src=OrderSource.DEPRECATION_FLAT)

                    position.add_order(flat_order, self.live_price_fetcher)
                    self.save_miner_position(position, delete_open_position_if_exists=True)
                    if self.websocket_notifier:
                        self.websocket_notifier.broadcast_position_update(position)
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
            # Skip Pydantic internal attributes to avoid deprecation warnings
            if attr.startswith("_") or (attr in ('model_computed_fields', 'model_config', 'model_fields', 'model_fields_set', '__fields__', 'newest_order_age_ms')):
                continue

            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property):
                continue

            value1 = getattr(position1, attr)
            # Check if position2 is a dict and access the value accordingly.
            if comparing_to_dict:
                # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                value2 = position2.get(attr)
            else:
                value2 = getattr(position2, attr, None)

            # tolerant float comparison
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                value1 = float(value1)
                value2 = float(value2)
                if not math.isclose(value1, value2, rel_tol=1e-9, abs_tol=1e-9):
                    return False, f"{attr} is different. {value1} != {value2}"
            elif value1 != value2:
                return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    def get_miner_position_by_uuid(self, hotkey:str, position_uuid: str) -> Position | None:
        # Use RPC to get the specific position directly
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

        # Use RPC to get positions from server
        temp = self._server_proxy.get_positions_for_one_hotkey_rpc(updated_position.miner_hotkey, only_open_positions=False)
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
        # Use RPC to get specific position
        position = self._server_proxy.get_position_rpc(hotkey, position_uuid)
        # Only deepcopy in direct mode (unit tests) where we access dict directly
        # In RPC mode, pickle serialization already creates a copy
        if position and self.running_unit_tests:
            return deepcopy(position)
        return position

    def get_existing_positions(self, hotkey: str):
        # Use RPC to get positions for hotkey
        return self._server_proxy.get_positions_for_one_hotkey_rpc(hotkey, only_open_positions=False)

    def _save_miner_position_to_memory(self, position: Position):
        # Use RPC to save position to server
        # The server handles deduplication, in-place updates, and validation
        # Only deepcopy in direct mode (unit tests) to protect server's internal state
        # In RPC mode, pickle serialization already creates a copy
        if self.running_unit_tests:
            self._server_proxy.save_miner_position_rpc(deepcopy(position))
        else:
            self._server_proxy.save_miner_position_rpc(position)


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
        # Use RPC to clear positions in server
        self._server_proxy.clear_all_miner_positions_rpc()

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
        # Use RPC to get all hotkeys - if a hotkey is in the dict, it has at least one position
        return len(self._server_proxy.get_all_hotkeys_rpc())

    def get_extreme_position_order_processed_on_disk_ms(self):
        return self._server_proxy.get_extreme_position_order_processed_on_disk_ms_rpc()

    def get_open_position_for_a_miner_trade_pair(self, hotkey: str, trade_pair_id: str) -> Position | None:
        """
        Get the open position for a specific miner and trade pair.
        Uses pure RPC - filtering happens server-side to avoid sending all positions over RPC.
        """
        # Single RPC call with server-side filtering - only returns the matching position (or None)
        position = self._server_proxy.get_open_position_for_trade_pair_rpc(hotkey, trade_pair_id)
        # Only deepcopy in direct mode (unit tests) where we access dict directly
        # In RPC mode, pickle serialization already creates a copy
        if position and self.running_unit_tests:
            return deepcopy(position)
        return position

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
        # Use RPC to delete position from server
        self._server_proxy.delete_position_rpc(hotkey, position_uuid)

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
    def get_positions_for_all_miners(self, include_development_positions=False, **args) -> dict[str, list[Position]]:
        """
        Get positions for all miners from the RPC server.

        Note: This always reads from the server's in-memory positions.
        For tests that need to verify disk persistence, use _read_positions_from_disk_for_tests().
        """
        # Use RPC to get all hotkeys from server
        all_miner_hotkeys = self._server_proxy.get_all_hotkeys_rpc()

        # Filter out development hotkey unless explicitly requested
        if not include_development_positions:
            all_miner_hotkeys = [hk for hk in all_miner_hotkeys if hk != ValiConfig.DEVELOPMENT_HOTKEY]

        return self.get_positions_for_hotkeys(all_miner_hotkeys, **args)

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


    def _get_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        file_string = None
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.model_validate_json(file_string)
            if not ans.orders:
                bt.logging.warning(f"Anomalous position has no orders: {ans.to_dict()}")
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

    def _read_positions_from_disk_for_tests(self, miner_hotkey: str, only_open_positions: bool = False) -> List[Position]:
        """
        Test helper method to read positions directly from disk, bypassing the RPC server.

        ⚠️ WARNING: This method is ONLY for tests that need to verify disk persistence.
        Production code should NEVER call this method - always use get_positions_for_one_hotkey() instead.

        The RPC server architecture dictates that only the server should read from disk normally.
        This helper exists solely to allow tests to verify that the server is correctly
        persisting data to disk.

        Args:
            miner_hotkey: The hotkey to read positions for
            only_open_positions: Whether to filter to only open positions

        Returns:
            List of positions loaded directly from disk files
        """
        miner_dir = ValiBkpUtils.get_miner_all_positions_dir(
            miner_hotkey,
            running_unit_tests=self.running_unit_tests
        )
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        positions = [self._get_position_from_disk(file) for file in all_files]

        if only_open_positions:
            positions = [position for position in positions if position.is_open_position]

        return positions

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
                                     acceptable_position_end_ms: int = None
                                     ) -> List[Position]:
        """
        Get positions for a hotkey from the RPC server.

        Server-side filtering reduces RPC payload and client processing overhead.

        Note: This always reads from the server's in-memory positions.
        For tests that need to verify disk persistence, use _read_positions_from_disk_for_tests().
        """
        # Single RPC call with server-side filtering (timestamp filtering happens on server)
        positions = self._server_proxy.get_positions_for_one_hotkey_rpc(
            miner_hotkey,
            only_open_positions,
            acceptable_position_end_ms
        )

        # Client-side sorting only (can't be done efficiently server-side)
        if sort_positions:
            positions = sorted(positions, key=self.sort_by_close_ms)

        return positions

    def get_positions_for_hotkeys(self, hotkeys: List[str],
                                   filter_eliminations: bool = False, **args) -> Dict[
        str, List[Position]]:
        """
        Get positions for multiple hotkeys from the RPC server (optimized bulk call).

        Server-side filtering reduces RPC payload and client processing overhead.

        Args:
            hotkeys: List of hotkeys to fetch positions for
            filter_eliminations: If True, server will fetch and filter eliminations internally
            **args: Additional filtering parameters (only_open_positions, sort_positions, acceptable_position_end_ms)

        Note: This always reads from the server's in-memory positions.
        For tests that need to verify disk persistence, use _read_positions_from_disk_for_tests().
        """
        # Extract filter/sort parameters
        only_open_positions = args.get('only_open_positions', False)
        sort_positions = args.get('sort_positions', False)
        acceptable_position_end_ms = args.get('acceptable_position_end_ms', None)

        # Single bulk RPC call with server-side filtering (much faster!)
        # Server handles elimination filtering, timestamp filtering, and open/closed filtering
        hotkey_to_positions = self._server_proxy.get_positions_for_hotkeys_rpc(
            hotkeys,
            only_open_positions=only_open_positions,
            filter_eliminations=filter_eliminations,  # Server fetches and filters internally
            acceptable_position_end_ms=acceptable_position_end_ms  # Server-side filtering
        )

        # Client-side sorting only (can't be done efficiently server-side)
        if sort_positions:
            for hotkey, positions in hotkey_to_positions.items():
                hotkey_to_positions[hotkey] = sorted(positions, key=self.sort_by_close_ms)

        return hotkey_to_positions

    def get_miner_hotkeys_with_at_least_one_position(self, include_development_positions=False) -> set[str]:
        # Use RPC to get all hotkeys from server
        hotkeys = set(self._server_proxy.get_all_hotkeys_rpc())

        # Filter out development hotkey unless explicitly requested
        if not include_development_positions and ValiConfig.DEVELOPMENT_HOTKEY in hotkeys:
            hotkeys = hotkeys - {ValiConfig.DEVELOPMENT_HOTKEY}

        return hotkeys

    def compute_realtime_drawdown(self, hotkey: str) -> float:
        """
        Compute the realtime drawdown from positions.
        Bypasses perf ledger, since perf ledgers are refreshed in 5 min intervals and may be out of date.
        Used to enable realtime withdrawals based on drawdown.

        Returns proportion of portfolio value as drawdown. 1.0 -> 0% drawdown, 0.9 -> 10% drawdown
        """
        # 1. Get existing perf ledger to access historical max portfolio value
        existing_bundle = self.perf_ledger_manager.get_perf_ledgers(
            portfolio_only=True,
            from_disk=False
        )
        portfolio_ledger = existing_bundle.get(hotkey)

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
        Delegates to PositionSplitter utility (single source of truth).
        """
        return PositionSplitter.find_split_points(position)

    def _position_needs_splitting(self, position: Position) -> bool:
        """
        Check if a position would actually be split by split_position_on_flat.
        Delegates to PositionSplitter utility (single source of truth).
        """
        return PositionSplitter.position_needs_splitting(position)

    def split_position_on_flat(self, position: Position, track_stats: bool = False) -> tuple[list[Position], dict]:
        """
        Split a position into multiple positions separated by FLAT orders or implicit flats.
        Delegates to PositionSplitter utility (single source of truth).

        If track_stats is True, updates split_stats with splitting information.

        Returns:
            tuple: (list of positions, split_info dict with 'implicit_flat_splits' and 'explicit_flat_splits')
        """
        # Delegate to PositionSplitter for the actual splitting logic
        positions, split_info = PositionSplitter.split_position_on_flat(
            position,
            self.live_price_fetcher,
            track_stats=track_stats
        )

        # Track stats if requested
        if track_stats and 'pre_split_return' in split_info and split_info['pre_split_return'] is not None:
            hotkey = position.miner_hotkey
            self.split_stats[hotkey]['n_positions_split'] += 1
            self.split_stats[hotkey]['product_return_pre_split'] *= split_info['pre_split_return']

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
