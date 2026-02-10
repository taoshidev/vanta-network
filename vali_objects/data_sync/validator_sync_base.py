import time
import traceback
from copy import deepcopy
from collections import defaultdict

from time_util.time_util import TimeUtil
from vali_objects.enums.misc import PositionSyncResult
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position_management.position_utils.position_splitter import PositionSplitter
from vali_objects.vali_dataclasses.position import Position
import bittensor as bt
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient

from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.position_management.position_manager import PositionManager
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from entity_management.entity_client import EntityClient

AUTO_SYNC_ORDER_LAG_MS = 1000 * 60 * 60 * 24

# Create a new type of exception PositionSyncResultException
class PositionSyncResultException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ValidatorSyncBase():
    def __init__(self, order_sync=None, running_unit_tests=False,
                 enable_position_splitting=False, verbose=False, is_mothership=False):
        self.verbose = verbose
        self.running_unit_tests = running_unit_tests
        self.is_mothership = is_mothership
        self.SYNC_LOOK_AROUND_MS = 1000 * 60 * 3
        self.enable_position_splitting = enable_position_splitting
        self._elimination_client = EliminationClient(running_unit_tests=running_unit_tests)
        self._position_manager_client = PositionManagerClient(running_unit_tests=running_unit_tests)
        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(running_unit_tests=running_unit_tests)
        self.last_signal_sync_time_ms = 0
        self.order_sync = order_sync
        self._challenge_period_client = ChallengePeriodClient(running_unit_tests=running_unit_tests)
        # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests)
        # Create own PerfLedgerClient (forward compatibility - no parameter passing)
        # This replaces the old ipc_manager.dict() pattern for perf_ledger_hks_to_invalidate
        self._perf_ledger_client = PerfLedgerClient(running_unit_tests=running_unit_tests)
        # Create own AssetSelectionClient (forward compatibility - no parameter passing)
        self._asset_selection_client = AssetSelectionClient(running_unit_tests=running_unit_tests)
        # Create own LimitOrderClient (forward compatibility - no parameter passing)
        from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
        self._limit_order_client = LimitOrderClient(running_unit_tests=running_unit_tests)
        # Create own MinerAccountClient (forward compatibility - no parameter passing)
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient(running_unit_tests=running_unit_tests)
        # Create own EntityClient (forward compatibility - no parameter passing)
        self._entity_client = EntityClient(running_unit_tests=running_unit_tests)
        self.init_data()

    def init_data(self):
        self.global_stats = defaultdict(int)

        # Order tracking sets (by miner)
        self.miners_with_order_deletions = set()
        self.miners_with_order_insertions = set()
        self.miners_with_order_matches = set()
        self.miners_with_order_updates = set()

        # Position tracking sets (by miner)
        self.miners_with_position_deletions = set()
        self.miners_with_position_insertions = set()
        self.miners_with_position_matches = set()
        self.miners_with_position_updates = set()
        # Clear perf ledger invalidations via RPC
        self._perf_ledger_client.clear_perf_ledger_hks_to_invalidate()


    @property
    def perf_ledger_hks_to_invalidate(self) -> dict:
        """
        Get hotkeys to invalidate from PerfLedgerServer via RPC.

        This property provides backward compatibility for code and tests that access
        perf_ledger_hks_to_invalidate directly. The data is now managed by PerfLedgerServer.
        """
        return self._perf_ledger_client.get_perf_ledger_hks_to_invalidate()


    def sync_positions(self, shadow_mode, candidate_data=None, disk_positions=None) -> dict[str: list[Position]]:
        t0 = time.time()
        self.init_data()
        assert candidate_data, "Candidate data must be provided"

        backup_creation_time_ms = candidate_data['created_timestamp_ms']
        bt.logging.info(f"Automated sync. Shadow mode {shadow_mode}. Found backup creation time: {TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)}")

        candidate_hk_to_positions = {}
        for hk, data in candidate_data['positions'].items():
            positions = data['positions']
            candidate_hk_to_positions[hk] = [Position(**p) for p in positions]

        # The candidate dataset is time lagged. We only delete non-matching data if they occured during the window of the candidate data.
        # We want to account for a few minutes difference of possible orders that came in after a retry.
        if 'hard_snap_cutoff_ms' in candidate_data:
            hard_snap_cutoff_ms = candidate_data['hard_snap_cutoff_ms']
        else:
            hard_snap_cutoff_ms = backup_creation_time_ms - AUTO_SYNC_ORDER_LAG_MS
        bt.logging.info(
            f"Automated sync. hard_snap_cutoff_ms: {TimeUtil.millis_to_formatted_date_str(hard_snap_cutoff_ms)}")

        if self.is_mothership:
            bt.logging.info("Mothership detected")

        # Check if disk_positions was explicitly provided
        disk_positions_provided = disk_positions is not None

        if disk_positions is None:
            disk_positions = self._position_manager_client.get_positions_for_all_miners(sort_positions=True)

        # Detect and delete overlapping positions before sync
        if not shadow_mode:
            overlap_stats = self.detect_and_delete_overlapping_positions(disk_positions)
            # Reload positions after deletions ONLY if we loaded them ourselves
            if overlap_stats['positions_deleted'] > 0 and not disk_positions_provided:
                disk_positions = self._position_manager_client.get_positions_for_all_miners(sort_positions=True)

        eliminations = candidate_data['eliminations']
        if not self.is_mothership:
            # Get current eliminations before sync (use PositionManager's internal elimination client)
            old_eliminated_hotkeys = set(x['hotkey'] for x in self._elimination_client.get_eliminations_from_memory())

            # Sync eliminations and get removed hotkeys
            removed = self._elimination_client.sync_eliminations(eliminations)

            # Get new eliminations after sync
            new_eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
            newly_eliminated = new_eliminated_hotkeys - old_eliminated_hotkeys

            # Invalidate perf ledgers for both removed and newly eliminated miners via RPC
            for hk in removed:
                self._perf_ledger_client.set_hotkey_to_invalidate(hk, 0)
            for hk in newly_eliminated:
                self._perf_ledger_client.set_hotkey_to_invalidate(hk, 0)

        limit_orders_data = candidate_data.get('limit_orders', {})
        if limit_orders_data:
            self._limit_order_client.sync_limit_orders(limit_orders_data)

        challengeperiod_data = candidate_data.get('challengeperiod', {})
        if challengeperiod_data:  # Only in autosync as of now.
            orig_testing_keys = set(self._challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
            orig_success_keys = set(self._challenge_period_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

            challengeperiod_dict = ChallengePeriodManager.parse_checkpoint_dict(challengeperiod_data)
            new_testing_keys = {
                    hotkey for hotkey, (bucket, _, _, _) in challengeperiod_dict.items()
                    if bucket is MinerBucket.CHALLENGE
                    }
            new_success_keys = {
                    hotkey for hotkey, (bucket, _, _, _) in challengeperiod_dict.items()
                    if bucket is MinerBucket.MAINCOMP
                    }

            bt.logging.info(f"Challengeperiod testing sync keys added: {new_testing_keys-orig_testing_keys}\n"
                            f"Challengeperiod testing sync keys removed: {orig_testing_keys - new_testing_keys}\n"
                            f"Challengeperiod success sync keys added: {new_success_keys - orig_success_keys}\n"
                            f"Challengeperiod success sync keys removed: {orig_success_keys - new_success_keys}")
            if not shadow_mode:
                self._challenge_period_client.sync_challenge_period_data(challengeperiod_data)

        # Sync miner account sizes if available
        miner_account_sizes_data = candidate_data.get('miner_account_sizes', {})
        if miner_account_sizes_data:
            if not shadow_mode:
                bt.logging.info(f"Syncing {len(miner_account_sizes_data)} miner account size records from auto sync")
                self._miner_account_client.sync_miner_account_sizes_data(miner_account_sizes_data)

        eliminated_hotkeys = set([e['hotkey'] for e in eliminations])
        # For a healthy validator, the existing positions will always be a superset of the candidate positions
        for hotkey, positions in candidate_hk_to_positions.items():
            if ShutdownCoordinator.is_shutdown():
                return
            if hotkey in eliminated_hotkeys:
                self.global_stats['n_miners_skipped_eliminated'] += 1
                continue
            
            # Deduplicate candidate positions BEFORE processing
            deduped_positions = self._dedupe_candidate_positions(positions)
            candidate_hk_to_positions[hotkey] = deduped_positions
            
            self.global_stats['n_miners_synced'] += 1
            candidate_positions_by_trade_pair = self.partition_positions_by_trade_pair(deduped_positions)
            existing_positions_by_trade_pair = self.partition_positions_by_trade_pair(disk_positions.get(hotkey, []))
            unified_trade_pairs = set(candidate_positions_by_trade_pair.keys()) | set(existing_positions_by_trade_pair.keys())
            for trade_pair in unified_trade_pairs:
                if ShutdownCoordinator.is_shutdown():
                    return
                candidate_positions = candidate_positions_by_trade_pair.get(trade_pair, [])
                existing_positions = existing_positions_by_trade_pair.get(trade_pair, [])

                try:
                    position_to_sync_status, min_timestamp_of_change, stats = self.resolve_positions(candidate_positions, existing_positions, trade_pair, hotkey, hard_snap_cutoff_ms)
                    if min_timestamp_of_change != float('inf'):
                        # Update hotkey invalidation timestamp via RPC (uses min logic internally)
                        self._perf_ledger_client.update_hotkey_to_invalidate(hotkey, min_timestamp_of_change)
                        if not shadow_mode:
                            self.write_modifications(position_to_sync_status, stats)
                except Exception as e:
                    full_traceback = traceback.format_exc()
                    # Slice the last 1000 characters of the traceback
                    limited_traceback = full_traceback[-1000:]
                    bt.logging.error(f"Error syncing positions for hotkey {hotkey} trade pair {trade_pair}. Error: {e} traceback: {limited_traceback}")
                    # If this is PositionSyncResultException, throw it up. Otherwise, log the error and continue.
                    if isinstance(e, PositionSyncResultException):
                        raise e
                    else:
                        self.global_stats['exceptions_seen'] += 1

        # Sync asset selections if available
        asset_selections_data = candidate_data.get('asset_selections', {})
        if asset_selections_data:
            bt.logging.info(f"Syncing {len(asset_selections_data)} miner asset selections from auto sync")
            if not shadow_mode:
                bt.logging.info(f"Syncing {len(asset_selections_data)} miner asset selection records from auto sync")
                self._asset_selection_client.sync_miner_asset_selection_data(asset_selections_data)

        # Sync entity data if available
        entities_data = candidate_data.get('entities', {})
        if entities_data and not shadow_mode:
            bt.logging.info(f"Syncing {len(entities_data)} entity records from auto sync")
            self._entity_client.sync_entity_data(entities_data)
        elif entities_data and shadow_mode:
            bt.logging.info(f"Shadow mode: Would sync {len(entities_data)} entity records (skipped)")

        # Reorganized stats with clear, grouped naming
        # Overview
        self.global_stats['miners_processed'] = self.global_stats['n_miners_synced']
        self.global_stats['miners_eliminated_skipped'] = self.global_stats['n_miners_skipped_eliminated']
        
        # Position outcomes (by unique miners)
        self.global_stats['miners_with_position_updates'] = len(self.miners_with_position_updates)
        self.global_stats['miners_with_position_matches'] = len(self.miners_with_position_matches)
        self.global_stats['miners_with_position_insertions'] = len(self.miners_with_position_insertions)
        self.global_stats['miners_with_position_deletions'] = len(self.miners_with_position_deletions)
        
        # Order outcomes (by unique miners)
        self.global_stats['miners_with_order_updates'] = len(self.miners_with_order_updates)
        self.global_stats['miners_with_order_matches'] = len(self.miners_with_order_matches)
        self.global_stats['miners_with_order_insertions'] = len(self.miners_with_order_insertions)
        self.global_stats['miners_with_order_deletions'] = len(self.miners_with_order_deletions)
        
        # Note: Raw counts use existing legacy keys directly (no duplication)

        # Print reorganized stats with clear grouping
        bt.logging.info("=" * 60)
        bt.logging.info("AUTOSYNC STATISTICS")
        bt.logging.info("=" * 60)
        
        # Overview section
        bt.logging.info("SYNC OVERVIEW:")
        bt.logging.info(f"  miners_processed: {self.global_stats.get('miners_processed', 0)}")
        bt.logging.info(f"  miners_eliminated_skipped: {self.global_stats.get('miners_eliminated_skipped', 0)}")
        bt.logging.info(f"  sync_duration_seconds: {time.time() - t0:.2f}")
        
        # Position outcomes
        bt.logging.info("POSITION OUTCOMES (by unique miners):")
        bt.logging.info(f"  miners_with_position_updates: {self.global_stats.get('miners_with_position_updates', 0)}")
        bt.logging.info(f"  miners_with_position_matches: {self.global_stats.get('miners_with_position_matches', 0)}")
        bt.logging.info(f"  miners_with_position_insertions: {self.global_stats.get('miners_with_position_insertions', 0)}")
        bt.logging.info(f"  miners_with_position_deletions: {self.global_stats.get('miners_with_position_deletions', 0)}")
        
        # Order outcomes
        bt.logging.info("ORDER OUTCOMES (by unique miners):")
        bt.logging.info(f"  miners_with_order_updates: {self.global_stats.get('miners_with_order_updates', 0)}")
        bt.logging.info(f"  miners_with_order_matches: {self.global_stats.get('miners_with_order_matches', 0)}")
        bt.logging.info(f"  miners_with_order_insertions: {self.global_stats.get('miners_with_order_insertions', 0)}")
        bt.logging.info(f"  miners_with_order_deletions: {self.global_stats.get('miners_with_order_deletions', 0)}")
        
        # Order-level operation counts (more actionable metrics)
        bt.logging.info("ORDER OPERATIONS (total counts):")
        bt.logging.info(f"  total_orders_inserted: {self.global_stats.get('orders_inserted', 0)}")
        bt.logging.info(f"  total_orders_updated: {self.global_stats.get('orders_updated', 0)}")
        bt.logging.info(f"  total_orders_deleted: {self.global_stats.get('orders_deleted', 0)}")
        bt.logging.info(f"  total_orders_matched: {self.global_stats.get('orders_matched', 0)}")
        
        # Raw counts (total items and miners processed)
        bt.logging.info("RAW COUNTS (total items and miners processed):")
        bt.logging.info(f"  total_positions_processed: {self.global_stats.get('positions_matched', 0)}")
        bt.logging.info(f"  total_orders_processed: {self.global_stats.get('orders_matched', 0)}")
        bt.logging.info(f"  total_miners_processed: {self.global_stats.get('miners_processed', 0)}")
        
        # Other stats (exclude keys we already displayed above)
        excluded_keys = {
            'miners_processed', 'miners_eliminated_skipped',  # Overview keys
            'n_miners_synced', 'n_miners_skipped_eliminated',  # Legacy overview 
            'positions_matched', 'orders_matched',  # Raw counts (already shown)
            'positions_deleted', 'positions_inserted', 'positions_kept',  # Raw position ops
            'orders_deleted', 'orders_inserted', 'orders_updated', 'orders_kept'  # Order ops (already shown)
        }
        other_stats = {k: v for k, v in self.global_stats.items() 
                      if not k.startswith('miners_with_') and k not in excluded_keys}
        if other_stats:
            bt.logging.info("OTHER STATS:")
            for k, v in other_stats.items():
                bt.logging.info(f"  {k}: {v}")
        
        bt.logging.info("=" * 60)

    def write_modifications(self, position_to_sync_status, stats):
        # Track position sync statuses for global stats
        for position, sync_status in position_to_sync_status.items():
            hk = position.miner_hotkey
            if sync_status == PositionSyncResult.UPDATED:
                self.miners_with_position_updates.add(hk)
            elif sync_status == PositionSyncResult.NOTHING:
                self.miners_with_position_matches.add(hk)
            elif sync_status == PositionSyncResult.INSERTED:
                self.miners_with_position_insertions.add(hk)
            elif sync_status == PositionSyncResult.DELETED:
                self.miners_with_position_deletions.add(hk)
            # KEPT status is handled elsewhere in the existing flow

        # Ensure the enums align with the global stats
        kept_and_matched = stats['kept'] + stats['matched']
        deleted = stats['deleted']
        inserted = stats['inserted']

        # Deletions happen first
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.DELETED:
                deleted -= 1
                if not self.is_mothership:
                    self._position_manager_client.delete_position(position.miner_hotkey, position.position_uuid)

        # Handle multiple open positions for a hotkey - track across ALL sync statuses to prevent duplicates
        prev_open_position = None
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.UPDATED:
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self._position_manager_client.save_miner_position(p)
                kept_and_matched -= 1

        # Insertions happen last so that there is no double open position issue
        # Do NOT reset prev_open_position - we need to track it across all sync statuses
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.INSERTED:
                inserted -= 1
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self._position_manager_client.save_miner_position(p)

        # Handle NOTHING status positions
        # Do NOT reset prev_open_position - we need to track it across all sync statuses
        for position, sync_status in position_to_sync_status.items():
            if sync_status == PositionSyncResult.NOTHING:
                kept_and_matched -= 1
                if not self.is_mothership:
                    positions = self.split_position_on_flat(position)
                    for p in positions:
                        if p.is_open_position:
                            prev_open_position = self.close_older_open_position(p, prev_open_position)
                        self._position_manager_client.save_miner_position(p)

        if kept_and_matched != 0:
            raise PositionSyncResultException(f"kept_and_matched: {kept_and_matched} stats {stats}")
        if deleted != 0:
            raise PositionSyncResultException(f"deleted: {deleted} stats {stats}")
        if inserted != 0:
            raise PositionSyncResultException(f"inserted: {inserted} stats {stats}")

    def close_older_open_position(self, p1: Position, p2: Position):
        """
        p1 and p2 are both open positions for a hotkey+trade pair, so we want to close the older one.
        We add a synthetic FLAT order before closing to maintain position invariants.

        Args:
            p1: The position we're about to save (from sync batch)
            p2: The previous position from the sync batch (can be None)

        Returns:
            The position to keep open (newest by open_ms)
        """
        # First, check if there's already an open position in memory for this miner+trade_pair
        # This catches the case where memory has a different position than what we're syncing
        existing_in_memory = self._position_manager_client.get_open_position_for_trade_pair(
            p1.miner_hotkey,
            p1.trade_pair.trade_pair_id
        )

        # Collect all open positions we need to consider, ensuring uniqueness by UUID
        positions_by_uuid = {}

        # Add positions, keyed by UUID to ensure uniqueness
        if existing_in_memory:
            positions_by_uuid[existing_in_memory.position_uuid] = existing_in_memory
        if p2 is not None:
            positions_by_uuid[p2.position_uuid] = p2
        positions_by_uuid[p1.position_uuid] = p1

        # If there's only one unique position, nothing to close
        if len(positions_by_uuid) == 1:
            return p1

        # Convert to list for sorting
        positions_to_compare = list(positions_by_uuid.values())

        # Sort by open_ms to find the newest position (the one to keep)
        positions_sorted = sorted(positions_to_compare, key=lambda p: p.open_ms)
        position_to_keep = positions_sorted[-1]  # Newest
        positions_to_close = positions_sorted[:-1]  # All older ones

        # Close all older positions
        for position_to_close in positions_to_close:
            self.global_stats['n_positions_closed_duplicate_opens_for_trade_pair'] += 1

            # Add synthetic FLAT order to properly close the position
            close_time_ms = position_to_close.orders[-1].processed_ms + 1
            flat_order = Position.generate_fake_flat_order(position_to_close, close_time_ms, self._live_price_client)
            position_to_close.orders.append(flat_order)
            position_to_close.close_out_position(close_time_ms)
            # Save the closed position back to disk
            self._position_manager_client.save_miner_position(position_to_close, delete_open_position_if_exists=False)

            bt.logging.warning(
                f"Closed duplicate open position {position_to_close.position_uuid} (open_ms={position_to_close.open_ms}) "
                f"in favor of newer position {position_to_keep.position_uuid} (open_ms={position_to_keep.open_ms}) "
                f"for miner {p1.miner_hotkey} trade_pair {p1.trade_pair.trade_pair_id}"
            )

        return position_to_keep  # Return the one to keep open



    def debug_print_pos(self, p):
        print(f'    pos: open {TimeUtil.millis_to_formatted_date_str(p.open_ms)} close {TimeUtil.millis_to_formatted_date_str(p.close_ms) if p.close_ms else "N/A"} uuid {p.position_uuid}')
        for o in p.orders:
            self.debug_print_order(o)
    def debug_print_order(self, o):
        print(f'        order: type {o.order_type} lev {o.leverage} time {TimeUtil.millis_to_formatted_date_str(o.processed_ms)} uuid {o.order_uuid}')

    def positions_aligned(self, p1, p2, timebound_ms=None, validate_num_orders=False):
        p1_initial_position_type = p1.orders[0].order_type
        p2_initial_position_type = p2.orders[0].order_type
        if validate_num_orders and len(p1.orders) != len(p2.orders):
            return False
        if p1_initial_position_type != p2_initial_position_type:
            return False
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        if p1.is_closed_position and p2.is_closed_position:
            return abs(p1.open_ms - p2.open_ms) < timebound_ms and abs(p1.close_ms - p2.close_ms) < timebound_ms
        elif p1.is_open_position and p2.is_open_position:
            return abs(p1.open_ms - p2.open_ms) < timebound_ms
        return False

    def dict_positions_aligned(self, p1, p2, timebound_ms=None, validate_num_orders=False):
        p1_initial_position_type = p1["orders"][0]["order_type"]
        p2_initial_position_type = p2["orders"][0]["order_type"]
        if validate_num_orders and len(p1["orders"]) != len(p2["orders"]):
            return False
        if p1_initial_position_type != p2_initial_position_type:
            return False
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        if p1["is_closed_position"] and p2["is_closed_position"]:
            return abs(p1["open_ms"] - p2["open_ms"]) < timebound_ms and abs(p1["close_ms"] - p2["close_ms"]) < timebound_ms
        elif not p1["is_closed_position"] and not p2["is_closed_position"]:
            return abs(p1["open_ms"] - p2["open_ms"]) < timebound_ms
        return False

    def positions_aligned_strict(self, p1, p2):
        timebound_ms = 1000 * 10
        return (self.positions_aligned(p1, p2, timebound_ms=timebound_ms) and
                self.all_orders_aligned(p1.orders, p2.orders, timebound_ms=timebound_ms))

    def all_orders_aligned(self, orders1, orders2, timebound_ms=None):
        if len(orders1) != len(orders2):
            return False
        for o1, o2 in zip(orders1, orders2):
            if not self.orders_aligned(o1, o2, timebound_ms=timebound_ms):
                return False
        return True

    def positions_order_aligned(self, p1, p2):
        for o1 in p1["orders"]:
            for o2 in p2["orders"]:
                if self.dict_orders_aligned(o1, o2):
                    return True
        return False

    def orders_aligned(self, o1, o2, timebound_ms=None):
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        return ((o1.leverage == o2.leverage) and
                (o1.order_type == o2.order_type) and
                abs(o1.processed_ms - o2.processed_ms) < timebound_ms)

    def dict_orders_aligned(self, o1, o2, timebound_ms=None):
        if timebound_ms is None:
            timebound_ms = self.SYNC_LOOK_AROUND_MS
        return (o1["order_uuid"] == o2["order_uuid"] or
                ((o1["leverage"] == o2["leverage"]) and
                 (o1["order_type"] == o2["order_type"]) and
                 abs(o1["processed_ms"] - o2["processed_ms"]) < timebound_ms))

    def sync_orders(self, ep, cp, hk, trade_pair, hard_snap_cutoff_ms):
        existing_orders = ep.orders
        candidate_orders = cp.orders
        min_timestamp_of_order_change = float('inf')
        metadata_changed = False  # Track if any order metadata changes
        # Positions are synonymous with an order
        assert existing_orders, existing_orders
        assert candidate_orders, candidate_orders

        ret = []
        matched_candidates_by_uuid = set()
        matched_existing_by_uuid = set()
        kept = list()
        matched = list()
        inserted = list()
        stats = defaultdict(int)
        # First pass. Try to match 1:1 based on uuid
        for c in candidate_orders:
            if c.order_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_orders:
                if e.order_uuid in matched_existing_by_uuid:
                    continue
                if e.order_uuid == c.order_uuid:
                    # temp: Update USD conversion rates from candidate data. Can be removed
                    # Check if metadata actually changed before updating
                    if (e.quote_usd_rate != c.quote_usd_rate or e.usd_base_rate != c.usd_base_rate):
                        metadata_changed = True
                        e.quote_usd_rate = c.quote_usd_rate
                        e.usd_base_rate = c.usd_base_rate
                    ret.append(e)
                    matched_candidates_by_uuid |= {c.order_uuid}
                    matched_existing_by_uuid |= {e.order_uuid}
                    stats['matched'] += 1
                    matched.append(e)
                    break

        # Second pass. Try to match 1:1 based on timestamps, leverage, and order type
        for c in candidate_orders:
            if c.order_uuid in matched_candidates_by_uuid:  # already matched
                continue
            for e in existing_orders:
                if e.order_uuid in matched_existing_by_uuid:
                    continue
                if self.orders_aligned(e, c):
                    # temp: Update USD conversion rates from candidate data. Can be removed
                    # Check if metadata actually changed before updating
                    if (e.quote_usd_rate != c.quote_usd_rate or e.usd_base_rate != c.usd_base_rate):
                        metadata_changed = True
                        e.quote_usd_rate = c.quote_usd_rate
                        e.usd_base_rate = c.usd_base_rate
                    matched_candidates_by_uuid |= {c.order_uuid}
                    matched_existing_by_uuid |= {e.order_uuid}
                    ret.append(e)
                    stats['matched'] += 1
                    matched.append(e)
                    break

        # Handle insertions (unmatched candidates)
        for o in candidate_orders:
            if o.order_uuid in matched_candidates_by_uuid:
                continue
            ret.append(o)
            inserted.append(o)
            stats['inserted'] += 1
            min_timestamp_of_order_change = min(min_timestamp_of_order_change, o.processed_ms)

        # Handle unmatched existing orders. Delete if they occurred during the window of the candidate data.
        deleted = []
        for o in existing_orders:
            if o.order_uuid in matched_existing_by_uuid:
                continue
            if o.processed_ms < hard_snap_cutoff_ms:
                stats['deleted'] += 1
                deleted.append(o)
                min_timestamp_of_order_change = min(min_timestamp_of_order_change, o.processed_ms)
            else:
                ret.append(o)
                kept.append(o)
                stats['kept'] += 1

        if stats['deleted']:
            self.global_stats['orders_deleted'] += stats['deleted']
            self.miners_with_order_deletions.add(hk)
        if stats['inserted']:
            self.global_stats['orders_inserted'] += stats['inserted']
            self.miners_with_order_insertions.add(hk)
        if stats['matched']:
            self.global_stats['orders_matched'] += stats['matched']
            self.miners_with_order_matches.add(hk)
        if stats['kept']:
            self.global_stats['orders_kept'] += stats['kept']
        
        # Track order updates (when orders are modified)
        if stats.get('updated', 0) > 0:
            self.global_stats['orders_updated'] = self.global_stats.get('orders_updated', 0) + stats['updated']
            self.miners_with_order_updates.add(hk)

        any_changes = stats['inserted'] + stats['deleted']
        if self.verbose and any_changes:
            print(f'hk {hk} trade pair {trade_pair.trade_pair} - Found {len(candidate_orders)} candidates and'
                  f' {len(existing_orders)} existing orders. stats {stats} min_timestamp_of_order_change {min_timestamp_of_order_change}')

            print('  existing:')
            for x in existing_orders:
                self.debug_print_order(x)
            print('  candidate:')
            for x in candidate_orders:
                self.debug_print_order(x)
            if inserted:
                print('  inserted:')
                for x in inserted:
                    self.debug_print_order(x)
            if deleted:
                print('  deleted:')
                for x in deleted:
                    self.debug_print_order(x)
            if kept:
                print('  kept:')
                for x in kept:
                    self.debug_print_order(x)
            if matched:
                print('  matched:')
                print(f'  {len(matched)} matched orders')

        ans = ret
        ans.sort(key=lambda x: x.processed_ms)
        return ans, min_timestamp_of_order_change, metadata_changed

    def resolve_positions(self, candidate_positions, existing_positions, trade_pair, hk, hard_snap_cutoff_ms):
        min_timestamp_of_change = float('inf')  # If this stays as float('inf), no changes happened
        candidate_positions_original = deepcopy(candidate_positions)
        existing_positions_original = deepcopy(existing_positions)
        ret = []
        matched_candidates_by_uuid = set()
        matched_existing_by_uuid = set()
        # Map position_uuid to syncStatus
        position_to_sync_status = {}
        kept = list()
        inserted = list()
        deleted = list()
        matched = list()
        stats = defaultdict(int)

        # First pass. Try to match 1:1 based on position_uuid
        for c in candidate_positions:
            if c.position_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_positions:
                if e.position_uuid in matched_existing_by_uuid:
                    continue
                if e.position_uuid == c.position_uuid:
                    # temp: sync account_size from candidate to existing position
                    # Check if position metadata will change
                    position_metadata_changed = False
                    if hasattr(c, 'account_size') and hasattr(e, 'account_size'):
                        if e.account_size != c.account_size:
                            position_metadata_changed = True
                            e.account_size = c.account_size

                    e.orders, min_timestamp_of_order_change, order_metadata_changed = self.sync_orders(e, c, hk, trade_pair, hard_snap_cutoff_ms)

                    # If metadata changed but no order insertions/deletions, trigger update
                    if (position_metadata_changed or order_metadata_changed) and min_timestamp_of_order_change == float('inf'):
                        min_timestamp_of_order_change = e.open_ms

                    if min_timestamp_of_order_change != float('inf'):
                        e.rebuild_position_with_updated_orders(self._live_price_client)
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                        position_to_sync_status[e] = PositionSyncResult.UPDATED
                    else:
                        position_to_sync_status[e] = PositionSyncResult.NOTHING
                        # Check if position actually needs splitting before forcing write_modifications
                        if self._position_manager_client and PositionSplitter.position_needs_splitting(e):
                            # Force write_modifications to be called for position splitting
                            min_timestamp_of_change = min(min_timestamp_of_change, e.open_ms)
                    ret.append(e)

                    matched_candidates_by_uuid |= {c.position_uuid}
                    matched_existing_by_uuid |= {e.position_uuid}
                    stats['matched'] += 1
                    matched.append(e)
                    break

        # Second pass. Try to match 1:1 based on timestamps
        for c in candidate_positions:
            if c.position_uuid in matched_candidates_by_uuid:
                continue
            for e in existing_positions:
                if e.position_uuid in matched_existing_by_uuid:
                    continue
                if self.positions_aligned(e, c):
                    # temp: sync account_size from candidate to existing position
                    # Check if position metadata will change
                    position_metadata_changed = False
                    if hasattr(c, 'account_size') and hasattr(e, 'account_size'):
                        if e.account_size != c.account_size:
                            position_metadata_changed = True
                            e.account_size = c.account_size

                    e.orders, min_timestamp_of_order_change, order_metadata_changed = self.sync_orders(e, c, hk, trade_pair, hard_snap_cutoff_ms)

                    # If metadata changed but no order insertions/deletions, trigger update
                    if (position_metadata_changed or order_metadata_changed) and min_timestamp_of_order_change == float('inf'):
                        min_timestamp_of_order_change = e.open_ms

                    if min_timestamp_of_order_change != float('inf'):
                        e.rebuild_position_with_updated_orders(self._live_price_client)
                        min_timestamp_of_change = min(min_timestamp_of_change, min_timestamp_of_order_change)
                        position_to_sync_status[e] = PositionSyncResult.UPDATED
                    else:
                        position_to_sync_status[e] = PositionSyncResult.NOTHING
                        # Check if position actually needs splitting before forcing write_modifications
                        if self._position_manager_client and PositionSplitter.position_needs_splitting(e):
                            # Force write_modifications to be called for position splitting
                            min_timestamp_of_change = min(min_timestamp_of_change, e.open_ms)
                    matched_candidates_by_uuid |= {c.position_uuid}
                    matched_existing_by_uuid |= {e.position_uuid}
                    ret.append(e)
                    matched.append(e)
                    stats['matched'] += 1
                    break

        # Handle insertions (unmatched candidates)
        for p in candidate_positions:
            if p.position_uuid in matched_candidates_by_uuid:
                continue

            stats['inserted'] += 1
            position_to_sync_status[p] = PositionSyncResult.INSERTED
            min_timestamp_of_change = min(min_timestamp_of_change, p.open_ms)
            ret.append(p)
            inserted.append(p)

        # Handle unmatched existing positions. Delete if they occurred during the window of the candidate data.
        for p in existing_positions:
            if p.position_uuid in matched_existing_by_uuid:
                continue
            if p.open_ms < hard_snap_cutoff_ms:
                stats['deleted'] += 1
                deleted.append(p)
                position_to_sync_status[p] = PositionSyncResult.DELETED
                min_timestamp_of_change = min(min_timestamp_of_change, p.open_ms)
            else:
                ret.append(p)
                kept.append(p)
                stats['kept'] += 1
                position_to_sync_status[p] = PositionSyncResult.NOTHING


        if stats['deleted']:
            self.global_stats['positions_deleted'] += stats['deleted']
            self.miners_with_position_deletions.add(hk)
        if stats['inserted']:
            self.global_stats['positions_inserted'] += stats['inserted']
            self.miners_with_position_insertions.add(hk)
        if stats['matched']:
            self.global_stats['positions_matched'] += stats['matched']
            self.miners_with_position_matches.add(hk)
        if stats['kept']:
            self.global_stats['positions_kept'] += stats['kept']
        
        # Track position updates (when positions are modified)
        if stats.get('updated', 0) > 0:
            self.global_stats['positions_updated'] = self.global_stats.get('positions_updated', 0) + stats['updated']
            self.miners_with_position_updates.add(hk)


        if self.verbose and (stats['inserted'] or stats['deleted']):
            print(f'hk {hk} trade pair {trade_pair.trade_pair} - Found {len(candidate_positions)} candidates and'
                  f' {len(existing_positions)} existing positions. stats {stats}')

            print('  existing positions:')
            for x in existing_positions_original:
                self.debug_print_pos(x)
            print('  candidate positions:')
            for x in candidate_positions_original:
                self.debug_print_pos(x)
            if inserted:
                print('  inserted positions:')
                for x in inserted:
                    self.debug_print_pos(x)
            if deleted:
                print('  deleted positions:')
                for x in deleted:
                    self.debug_print_pos(x)
            if matched:
                print('  matched positions:')
                for x in matched:
                    self.debug_print_pos(x)
            if kept:
                print('  kept positions:')
                for x in kept:
                    self.debug_print_pos(x)

        return position_to_sync_status, min_timestamp_of_change, stats

    def _dedupe_candidate_positions(self, positions: list[Position]) -> list[Position]:
        """
        Deduplicate candidate positions in memory before sync processing.
        Similar to position_manager.dedupe_positions but works on in-memory data.
        """
        if not positions:
            return positions
            
        positions_by_trade_pair = defaultdict(list)
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(position)

        deduped_positions = []
        for trade_pair, trade_pair_positions in positions_by_trade_pair.items():
            position_uuid_to_keep = {}
            
            for position in trade_pair_positions:
                if position.position_uuid in position_uuid_to_keep:
                    # Keep the position with more orders (same logic as position_manager.dedupe_positions)
                    existing_position = position_uuid_to_keep[position.position_uuid]
                    if len(position.orders) > len(existing_position.orders):
                        position_uuid_to_keep[position.position_uuid] = position
                    # If current has same or fewer orders, keep the existing one (do nothing)
                else:
                    position_uuid_to_keep[position.position_uuid] = position
            
            # Add the deduplicated positions for this trade pair
            deduped_positions.extend(position_uuid_to_keep.values())
        
        return deduped_positions

    def partition_positions_by_trade_pair(self, positions: list[Position]) -> dict[TradePair, list[Position]]:
        positions_by_trade_pair = defaultdict(list)
        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(deepcopy(position))
        return positions_by_trade_pair

    def split_position_on_flat(self, position: Position) -> list[Position]:
        """
        Delegates position splitting to the PositionManager.
        This maintains the autosync logic while using the centralized splitting implementation.
        """
        if not self._position_manager_client or not self.enable_position_splitting:
            # If no position manager or splitting disabled, return position as-is
            return [position]

        # Use the position manager's split method
        positions, split_info = self._position_manager_client.split_position_on_flat(position, track_stats=False)

        # Track statistics for autosync
        if len(positions) > 1:
            self.global_stats['n_positions_spawned_from_post_flat_orders'] += len(positions) - 1

            # Track implicit flat splits
            if split_info['implicit_flat_splits'] > 0:
                self.global_stats['n_positions_split_on_implicit_flat'] = \
                    self.global_stats.get('n_positions_split_on_implicit_flat', 0) + split_info['implicit_flat_splits']

        return positions

    def detect_and_delete_overlapping_positions(self, disk_positions, current_time_ms=None):
        """
        For each hotkey, analyze positions on a per-trade-pair basis.
        Detects and deletes:
        1. Positions with overlapping time intervals
        2. Multiple open positions for same trade pair
        3. Open positions that are not last chronologically
        4. Positions with inconsistent state (e.g., open position with close_ms set)

        Auto sync will then fill in these positions with valid start/end times.

        Uses interval merging algorithm for efficient O(n log n) overlap detection.

        Args:
            disk_positions: Dict[str, List[Position]] - positions organized by hotkey
            current_time_ms: int - current timestamp, or None to use TimeUtil.now_in_millis()

        Returns:
            Dict with statistics about overlaps detected and positions deleted
        """
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        stats = {
            'hotkeys_checked': 0,
            'trade_pairs_checked': 0,
            'positions_deleted': 0,
            'positions_deleted_overlaps': 0,
            'positions_deleted_invariant_violations': 0,
            'hotkeys_with_overlaps': set(),
            'hotkeys_with_invariant_violations': set(),
            'trade_pairs_with_overlaps': defaultdict(int),
            'trade_pairs_with_invariant_violations': defaultdict(int)
        }

        for hotkey, positions in disk_positions.items():
            if not positions:
                continue

            stats['hotkeys_checked'] += 1

            # Group positions by trade pair
            positions_by_trade_pair = self.partition_positions_by_trade_pair(positions)

            for trade_pair, tp_positions in positions_by_trade_pair.items():
                stats['trade_pairs_checked'] += 1
                positions_to_delete = set()

                # 1. Find all positions with overlapping time intervals
                overlapping_position_uuids = self._find_overlapping_positions_via_merge(tp_positions, current_time_ms)
                if overlapping_position_uuids:
                    positions_to_delete.update(overlapping_position_uuids)
                    stats['positions_deleted_overlaps'] += len(overlapping_position_uuids)
                    stats['hotkeys_with_overlaps'].add(hotkey)
                    stats['trade_pairs_with_overlaps'][trade_pair.trade_pair_id] += 1

                # 2. Find positions violating ordering/state invariants
                invariant_violation_uuids = self._find_positions_violating_invariants(tp_positions, current_time_ms)
                if invariant_violation_uuids:
                    positions_to_delete.update(invariant_violation_uuids)
                    stats['positions_deleted_invariant_violations'] += len(invariant_violation_uuids)
                    stats['hotkeys_with_invariant_violations'].add(hotkey)
                    stats['trade_pairs_with_invariant_violations'][trade_pair.trade_pair_id] += 1

                # Delete all problematic positions
                if positions_to_delete:
                    # Build UUID -> Position map for efficient lookup
                    uuid_to_position = {p.position_uuid: p for p in tp_positions}

                    for position_uuid in positions_to_delete:
                        if not self.is_mothership:
                            position = uuid_to_position[position_uuid]
                            self._position_manager_client.delete_position(position.miner_hotkey, position.position_uuid)
                        stats['positions_deleted'] += 1

                    bt.logging.warning(
                        f"Deleted {len(positions_to_delete)} problematic positions for "
                        f"hotkey {hotkey} trade pair {trade_pair.trade_pair_id} "
                        f"(overlaps: {len(overlapping_position_uuids)}, "
                        f"invariant violations: {len(invariant_violation_uuids)})"
                    )

        # Log summary
        bt.logging.info("=" * 60)
        bt.logging.info("POSITION INTEGRITY CHECK SUMMARY")
        bt.logging.info("=" * 60)
        bt.logging.info(f"Hotkeys checked: {stats['hotkeys_checked']}")
        bt.logging.info(f"Trade pairs checked: {stats['trade_pairs_checked']}")
        bt.logging.info(f"Hotkeys with overlaps: {len(stats['hotkeys_with_overlaps'])}")
        bt.logging.info(f"Hotkeys with invariant violations: {len(stats['hotkeys_with_invariant_violations'])}")
        bt.logging.info(f"Total positions deleted: {stats['positions_deleted']}")
        bt.logging.info(f"  - Due to overlaps: {stats['positions_deleted_overlaps']}")
        bt.logging.info(f"  - Due to invariant violations: {stats['positions_deleted_invariant_violations']}")
        if stats['trade_pairs_with_overlaps']:
            bt.logging.info(f"Trade pairs with overlaps: {dict(stats['trade_pairs_with_overlaps'])}")
        if stats['trade_pairs_with_invariant_violations']:
            bt.logging.info(f"Trade pairs with invariant violations: {dict(stats['trade_pairs_with_invariant_violations'])}")
        bt.logging.info("=" * 60)

        return stats

    def _find_overlapping_positions_via_merge(self, positions: list[Position], current_time_ms: int) -> set:
        """
        Find all positions that have overlapping time ranges using interval merging algorithm.

        Algorithm:
        1. Sort intervals by start time
        2. Iterate through and merge overlapping intervals
        3. Track all positions in each merged group
        4. Return all positions from groups that contain 2+ positions (i.e., overlaps occurred)

        Time complexity: O(n log n) due to sorting

        Args:
            positions: List of positions for a single trade pair
            current_time_ms: Current timestamp (used for open positions' end time)

        Returns:
            Set of position UUIDs that are part of any overlap
        """
        if len(positions) < 2:
            return set()

        # Create intervals: (start_ms, end_ms, position)
        intervals = []
        for position in positions:
            start_ms = position.open_ms
            # Use close_ms if closed, otherwise use current time
            end_ms = position.close_ms if position.is_closed_position else current_time_ms
            intervals.append((start_ms, end_ms, position))

        # Sort by start time
        intervals.sort(key=lambda x: x[0])

        # Merge overlapping intervals and track which positions are in each merged group
        current_group = [intervals[0][2]]  # Start with first position
        current_end = intervals[0][1]
        overlapping_position_uuids = set()

        for i in range(1, len(intervals)):
            start_ms, end_ms, position = intervals[i]

            # Check if current interval overlaps with the merged interval
            # Overlap occurs if start_ms < current_end
            if start_ms < current_end:
                # Overlaps - add to current group
                current_group.append(position)
                # Extend the merged interval's end if necessary
                current_end = max(current_end, end_ms)
                overlapping_position_uuids.add(position.position_uuid)
                overlapping_position_uuids.add(intervals[i-1][2].position_uuid)
            else:
                # No overlap
                # Start new group
                current_group = [position]
                current_end = end_ms

        return overlapping_position_uuids

    def _find_positions_violating_invariants(self, positions: list[Position], current_time_ms: int) -> set:
        """
        Find positions that violate ordering/state invariants that would cause
        perf_ledger assertion failures.

        Detects:
        1. Multiple open positions for the same trade pair (should be at most 1)
        2. Open position exists but is NOT the last position chronologically
        3. Positions with inconsistent state (treats both 0 and None as valid for open positions)
        4. Closed positions without FLAT order as last order

        Args:
            positions: List of positions for a single trade pair
            current_time_ms: Current timestamp (not used currently but kept for consistency)

        Returns:
            Set of position UUIDs that violate invariants
        """
        if len(positions) < 1:
            return set()

        # Sort positions chronologically by close_ms
        # Use the canonical sorting method from PositionManager
        sorted_positions = sorted(positions, key=PositionManager.sort_by_close_ms)

        violation_uuids = set()

        # Count open and closed positions
        open_positions = [p for p in sorted_positions if p.is_open_position]
        n_open = len(open_positions)

        # Violation 1: More than 1 open position
        if n_open > 1:
            # Delete all but the most recent open position (by open_ms)
            open_positions_sorted = sorted(open_positions, key=lambda p: p.open_ms)
            # Keep the last one, delete all others
            for p in open_positions_sorted[:-1]:
                violation_uuids.add(p.position_uuid)
            bt.logging.warning(
                f"INVARIANT VIOLATION: Found {n_open} open positions (max 1 allowed). "
                f"Will delete {len(violation_uuids)} older open positions."
            )

        # Violation 2: If exactly 1 open position, it must be the last in the chronological list
        elif n_open == 1:
            last_position = sorted_positions[-1]
            if not last_position.is_open_position:
                # The open position is NOT last - this is a violation
                open_position = open_positions[0]
                violation_uuids.add(open_position.position_uuid)
                bt.logging.warning(
                    f"INVARIANT VIOLATION: Found 1 open position but it's NOT the last chronologically. "
                    f"Last position is closed (close_ms={last_position.close_ms}). "
                    f"Will delete the misplaced open position {open_position.position_uuid}."
                )

        # Violation 3: Check for positions with inconsistent state
        # Treat both 0 and None as valid for open positions (defensive programming)
        for p in sorted_positions:
            # Skip if already marked for deletion
            if p.position_uuid in violation_uuids:
                continue

            # Check for inconsistent state
            # For open positions: close_ms can be None or 0 (both valid)
            # For closed positions: close_ms must be a real timestamp (not None, not 0)
            if p.is_open_position and p.close_ms is not None and p.close_ms != 0:
                violation_uuids.add(p.position_uuid)
                bt.logging.warning(
                    f"INVARIANT VIOLATION: Position {p.position_uuid} has is_open_position=True "
                    f"but close_ms={p.close_ms} (should be None or 0). Will delete."
                )
            elif p.is_closed_position and (p.close_ms is None or p.close_ms == 0):
                violation_uuids.add(p.position_uuid)
                bt.logging.warning(
                    f"INVARIANT VIOLATION: Position {p.position_uuid} has is_closed_position=True "
                    f"but close_ms={p.close_ms} (should be a real timestamp). Will delete."
                )

        # Violation 4: Check for closed positions without FLAT order as last order
        for p in sorted_positions:
            # Skip if already marked for deletion
            if p.position_uuid in violation_uuids:
                continue

            # Closed positions must have a FLAT order as their last order
            if p.is_closed_position:
                if not p.orders or p.orders[-1].order_type != OrderType.FLAT:
                    violation_uuids.add(p.position_uuid)
                    last_order_type = p.orders[-1].order_type.name if p.orders else "NO_ORDERS"
                    bt.logging.warning(
                        f"INVARIANT VIOLATION: Position {p.position_uuid} is closed "
                        f"but last order is {last_order_type} (should be FLAT). Will delete."
                    )

        return violation_uuids
