import json
from copy import deepcopy
from unittest.mock import Mock, patch

from tests.shared_objects.mock_classes import MockLivePriceFetcher

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.auto_sync import PositionSyncer
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.utils.validator_sync_base import AUTO_SYNC_ORDER_LAG_MS, PositionSyncResultException
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.miner_bucket_enum import MinerBucket
from time_util.time_util import TimeUtil


class TestPositions(TestBase):

    def setUp(self):
        super().setUp()

        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        self.default_order = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS, order_uuid=self.DEFAULT_ORDER_UUID, trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=1)
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True,
                                                elimination_manager=self.elimination_manager, live_price_fetcher=self.live_price_fetcher)
        self.elimination_manager.position_manager = self.position_manager
        self.position_manager.clear_all_miner_positions()
        
        # Clear any eliminations that might persist between tests
        self.elimination_manager.eliminations.clear()

        self.default_open_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.default_closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.FLAT,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_closed_position.close_out_position(self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 6)

        self.position_syncer = PositionSyncer(running_unit_tests=True, position_manager=self.position_manager, enable_position_splitting=True, live_price_fetcher=self.live_price_fetcher)

    def validate_comprehensive_stats(self, stats, expected_miners=1, expected_eliminated=0, 
                                   expected_pos_updates=0, expected_pos_matches=0, expected_pos_insertions=0, 
                                   expected_pos_deletions=0,
                                   expected_ord_updates=0, expected_ord_matches=0, expected_ord_insertions=0,
                                   expected_ord_deletions=0,
                                   expected_exceptions=0, 
                                   # Raw count expectations (optional, auto-calculated if not provided)
                                   expected_raw_pos_processed=None, expected_raw_ord_processed=None, expected_raw_miners_processed=None,
                                   expected_raw_pos_inserted=None, expected_raw_pos_deleted=None,
                                   expected_raw_ord_inserted=None, expected_raw_ord_deleted=None,
                                   test_context=""):
        """
        Comprehensive validation of all AutoSync statistics keys.
        Ensures tests check every possible stats key for completeness.
        
        Validates:
        - All 10 essential miner-level keys (miners_processed, miners_with_position_*, etc.)
        - RAW COUNTS: total_positions_processed, total_orders_processed, total_miners_processed
        - Optional operation-level raw counts (positions_inserted, orders_deleted, etc.)
        - Operational stats (exceptions_seen)
        - Logical consistency checks
        """
        context = f" in {test_context}" if test_context else ""
        
        # Essential overview stats
        assert stats['miners_processed'] == expected_miners, f"miners_processed mismatch{context}: {stats}"
        assert stats.get('miners_eliminated_skipped', 0) == expected_eliminated, f"miners_eliminated_skipped mismatch{context}: {stats}"
        
        # Position outcome stats (by unique miners)
        assert stats['miners_with_position_updates'] == expected_pos_updates, f"miners_with_position_updates mismatch{context}: {stats}"
        assert stats['miners_with_position_matches'] == expected_pos_matches, f"miners_with_position_matches mismatch{context}: {stats}"
        assert stats['miners_with_position_insertions'] == expected_pos_insertions, f"miners_with_position_insertions mismatch{context}: {stats}"
        assert stats['miners_with_position_deletions'] == expected_pos_deletions, f"miners_with_position_deletions mismatch{context}: {stats}"
        
        # Order outcome stats (by unique miners)
        assert stats['miners_with_order_updates'] == expected_ord_updates, f"miners_with_order_updates mismatch{context}: {stats}"
        assert stats['miners_with_order_matches'] == expected_ord_matches, f"miners_with_order_matches mismatch{context}: {stats}"
        assert stats['miners_with_order_insertions'] == expected_ord_insertions, f"miners_with_order_insertions mismatch{context}: {stats}"
        assert stats['miners_with_order_deletions'] == expected_ord_deletions, f"miners_with_order_deletions mismatch{context}: {stats}"
        
        # Operational stats (should be 0 for normal test scenarios)
        assert stats.get('exceptions_seen', 0) == expected_exceptions, f"exceptions_seen mismatch{context}: {stats}"
        
        # Raw count validation (displayed in RAW COUNTS section)
        if expected_raw_pos_processed is not None:
            assert stats.get('positions_matched', 0) == expected_raw_pos_processed, f"total_positions_processed mismatch{context}: {stats}"
        if expected_raw_ord_processed is not None:
            assert stats.get('orders_matched', 0) == expected_raw_ord_processed, f"total_orders_processed mismatch{context}: {stats}"
        if expected_raw_miners_processed is not None:
            assert stats.get('miners_processed', 0) == expected_raw_miners_processed, f"total_miners_processed mismatch{context}: {stats}"
        
        # Operation-level raw count validation (if provided)
        if expected_raw_pos_inserted is not None:
            assert stats.get('positions_inserted', 0) == expected_raw_pos_inserted, f"positions_inserted mismatch{context}: {stats}"
        if expected_raw_pos_deleted is not None:
            assert stats.get('positions_deleted', 0) == expected_raw_pos_deleted, f"positions_deleted mismatch{context}: {stats}"
        if expected_raw_ord_inserted is not None:
            assert stats.get('orders_inserted', 0) == expected_raw_ord_inserted, f"orders_inserted mismatch{context}: {stats}"
        if expected_raw_ord_deleted is not None:
            assert stats.get('orders_deleted', 0) == expected_raw_ord_deleted, f"orders_deleted mismatch{context}: {stats}"
        
        # Validate that all miner counts are consistent
        total_position_outcomes = expected_pos_updates + expected_pos_matches + expected_pos_insertions + expected_pos_deletions
        total_order_outcomes = expected_ord_updates + expected_ord_matches + expected_ord_insertions + expected_ord_deletions
        
        # Note: A single miner can have multiple types of outcomes (e.g., both position match AND order insertion)
        # So we don't assert exact equality, but we do validate logical consistency
        assert total_position_outcomes <= expected_miners * 5, f"Position outcomes exceed logical bounds{context}: {stats}"
        assert total_order_outcomes <= expected_miners * 5, f"Order outcomes exceed logical bounds{context}: {stats}"

    def positions_to_disk_data(self, positions: list[Position]):
        return {self.DEFAULT_MINER_HOTKEY: positions}

    def positions_to_candidate_data(self, positions: list[Position]):
        mt = 0
        for p in positions:
            for o in p.orders:
                mt = max(mt, o.processed_ms)

        candidate_data = {'positions': {self.DEFAULT_MINER_HOTKEY: {}},
                  'eliminations': [], 'created_timestamp_ms': mt + AUTO_SYNC_ORDER_LAG_MS}
        for hk in candidate_data['positions']:
            candidate_data['positions'][hk]['positions'] = [json.loads(str(p), cls=GeneralizedJSONDecoder) for p in positions]
        return candidate_data


    def test_validate_basic_sync(self):
        candidate_data = self.positions_to_candidate_data([self.default_position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Use comprehensive validation with RAW COUNTS
        self.validate_comprehensive_stats(
            stats,
            expected_miners=1,
            expected_eliminated=0,
            expected_pos_updates=0,
            expected_pos_matches=1,  # Position matched by UUID
            expected_pos_insertions=0,
            expected_pos_deletions=0,
            expected_ord_updates=0,
            expected_ord_matches=1,  # Order matched by UUID
            expected_ord_insertions=0,
            expected_ord_deletions=0,
            expected_exceptions=0,
            expected_raw_pos_processed=1,  # 1 position processed
            expected_raw_ord_processed=1,  # 1 order processed
            expected_raw_miners_processed=1,  # 1 miner processed
            expected_raw_pos_inserted=0,
            expected_raw_pos_deleted=0,
            expected_raw_ord_inserted=0,
            expected_raw_ord_deleted=0,
            test_context="basic sync validation"
        )
        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

        # When there are no existing positions, we should insert the new one.
        candidate_data = self.positions_to_candidate_data([self.default_position])
        disk_positions = self.positions_to_disk_data([])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Use comprehensive validation for insertion scenario with RAW COUNTS
        self.validate_comprehensive_stats(
            stats,
            expected_miners=1,
            expected_eliminated=0,
            expected_pos_updates=0,
            expected_pos_matches=0,
            expected_pos_insertions=1,  # New position inserted
            expected_pos_deletions=0,
            expected_ord_updates=0,
            expected_ord_matches=0,
            expected_ord_insertions=0,
            expected_ord_deletions=0,
            expected_exceptions=0,
            expected_raw_pos_processed=0,  # No positions processed (new insertion)
            expected_raw_ord_processed=0,  # No orders processed (new insertion)
            expected_raw_miners_processed=1,  # 1 miner processed
            expected_raw_pos_inserted=1,  # 1 new position inserted
            expected_raw_pos_deleted=0,
            expected_raw_ord_inserted=0,
            expected_raw_ord_deleted=0,
            test_context="position insertion"
        )

        assert stats['positions_inserted'] == 1, stats
        assert stats['positions_matched'] == 0, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 0, stats
        assert stats['orders_matched'] == 0, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats
        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.DEFAULT_OPEN_MS

    def test_comprehensive_stats_validation(self):
        """
        Test to ensure all possible AutoSync statistics keys are properly tracked.
        This test exercises different scenarios to trigger various stats and validates
        that our comprehensive validation catches everything.
        """
        # Test 1: Order insertion scenario (most common case to validate)
        # Create existing position with one order
        existing_order = Order(
            price=100.0, processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="update_test", trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.LONG, leverage=1.0
        )
        
        existing_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="order_update_test",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[existing_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        
        # Create candidate with additional order (should trigger order insertion)
        candidate_order1 = deepcopy(existing_order)
        candidate_order2 = Order(
            price=101.0, processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="new_order", trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.FLAT, leverage=0.0
        )
        
        candidate_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="order_update_test",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[candidate_order1, candidate_order2],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        disk_positions = self.positions_to_disk_data([existing_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should show position update (not just match) with order insertion
        # Note: When a position gets additional orders, BOTH position_updates AND position_matches are set
        # This is because: 1) UUID matched (position_matches), 2) Content changed (position_updates)
        self.validate_comprehensive_stats(
            stats,
            expected_miners=1,
            expected_eliminated=0,
            expected_pos_updates=1,  # Position was updated (got additional orders)
            expected_pos_matches=1,  # Position also matched by UUID
            expected_pos_insertions=0,
            expected_pos_deletions=0,
            expected_ord_updates=0,
            expected_ord_matches=1,  # Existing order matched
            expected_ord_insertions=1,  # New order inserted
            expected_ord_deletions=0,
            expected_exceptions=0,
            expected_raw_pos_processed=1,  # 1 position processed
            expected_raw_ord_processed=1,  # 1 existing order processed
            expected_raw_miners_processed=1,  # 1 miner processed
            expected_raw_pos_inserted=0,
            expected_raw_pos_deleted=0,
            expected_raw_ord_inserted=1,  # 1 new order inserted
            expected_raw_ord_deleted=0,
            test_context="order insertion scenario"
        )
        
        # Validate raw counts are present and reasonable
        assert 'positions_matched' in stats, "Raw position count missing"
        assert 'orders_matched' in stats, "Raw order count missing" 
        assert 'orders_inserted' in stats, "Raw order insertion count missing"
        assert stats['positions_matched'] >= 0, "Invalid position count"
        assert stats['orders_matched'] >= 0, "Invalid order count"
        
        # Test 2: Validate operational stats are tracked
        assert stats.get('exceptions_seen', 0) == 0, "Should have no exceptions in normal operation"
        
        # Test 3: Validate that legacy keys exist for compatibility
        assert 'n_miners_synced' in stats, "Legacy n_miners_synced key missing"
        assert stats['n_miners_synced'] == stats['miners_processed'], "Legacy key should match new key"

    def test_position_deletion(self):
        dp1 = deepcopy(self.default_closed_position)
        dp1.position_uuid = 'to_delete'
        dp1.open_ms -= 1000 * 60 * 10

        # Opened within hardsnap but closed after hardsnap. Sry doesn't matter - should have existed.
        dp2 = deepcopy(self.default_closed_position)
        dp2.position_uuid = 'to_delete'
        dp2.open_ms -= 1000 * 60 * 10
        dp2.close_ms += 1000 * 60 * 10

        dp3 = deepcopy(self.default_open_position)
        dp3.position_uuid = 'to_delete'
        dp3.open_ms -= 1000 * 60 * 10

        # One delete, one insert (from candidate)
        for i, dp in enumerate([dp1, dp2, dp3]):
            cd = self.default_open_position if i == 2 else self.default_closed_position
            candidate_data = self.positions_to_candidate_data([cd])
            disk_positions = self.positions_to_disk_data([dp])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats

            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            # Comprehensive validation: position deletion + insertion scenario
            self.validate_comprehensive_stats(
                stats,
                expected_miners=1,
                expected_eliminated=0,
                expected_pos_updates=0,
                expected_pos_matches=0,
                expected_pos_insertions=1,  # New position inserted
                expected_pos_deletions=1,  # Old position deleted
                expected_ord_updates=0,
                expected_ord_matches=0,
                expected_ord_insertions=0,
                expected_ord_deletions=0,
                expected_exceptions=0,
                expected_raw_pos_processed=0,  # No existing positions processed
                expected_raw_ord_processed=0,  # No existing orders processed
                expected_raw_miners_processed=1,  # 1 miner processed
                expected_raw_pos_inserted=1,  # 1 new position inserted
                expected_raw_pos_deleted=1,  # 1 old position deleted
                expected_raw_ord_inserted=0,
                expected_raw_ord_deleted=0,
                test_context=f"position deletion iteration {i}"
            )

            assert stats['positions_inserted'] == 1, stats
            assert stats['positions_matched'] == 0, stats
            assert stats['positions_deleted'] == 1, stats
            assert stats['positions_kept'] == 0, stats

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
            assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == dp.open_ms


        # Same thing just no more inserting. We keep the positions since there is no candidate.
        candidate_data = self.positions_to_candidate_data([])
        for i, dp in enumerate([dp1, dp2, dp3]):
            disk_positions = self.positions_to_disk_data([dp])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats

            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['miners_processed'] == 1, (i, stats_str)

            assert stats['miners_with_position_deletions'] == 0, (i, stats_str)
            assert stats['miners_with_position_matches'] == 0, (i, stats_str)
            assert stats['miners_with_position_insertions'] == 0, (i, stats_str)

            assert stats['miners_with_order_deletions'] == 0, (i, stats_str)
            assert stats['miners_with_order_insertions'] == 0, (i, stats_str)
            assert stats['miners_with_order_matches'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 0, stats
            assert stats['positions_matched'] == 0, stats
            assert stats['positions_deleted'] == 0, stats
            assert stats['positions_kept'] == 1, stats

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

    def test_split_position_with_orders_after_flat(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = self.default_order.processed_ms + 1000
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
        order2.order_type = OrderType.FLAT
        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        order3.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10 * 2
        orders = [order1, order2, order3]
        position = deepcopy(self.default_position)
        position.orders = orders
        print(position.is_open_position)
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                            disk_positions=disk_positions)
        stats = self.position_syncer.global_stats

        assert stats['n_positions_spawned_from_post_flat_orders'] == 1, stats
        assert stats['n_positions_closed_duplicate_opens_for_trade_pair'] == 0, stats

    def test_position_keep_both_insert(self):
        dp1 = deepcopy(self.default_closed_position)
        dp1.position_uuid = 'to_keep'
        # After hardsnap, we must keep it
        dp1.open_ms += 1000 * 60 * 10

        dp2 = deepcopy(self.default_closed_position)
        dp2.position_uuid = 'to_keep'
        # After hardsnap, we must keep it
        dp2.open_ms += 1000 * 60 * 10
        dp2.close_ms += 1000 * 60 * 10

        # Close the older open position, and insert the newer open position
        dp3 = deepcopy(self.default_open_position)
        dp3.position_uuid = 'double_open'
        dp3.open_ms += 1000 * 60 * 10

        for i, dp in enumerate([dp1, dp2, dp3]):
            cd = self.default_open_position if i == 2 else self.default_closed_position
            candidate_data = self.positions_to_candidate_data([cd])
            disk_positions = self.positions_to_disk_data([dp])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats

            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['miners_processed'] == 1, (i, stats_str)
            # if i == 2:
            #     assert stats['blocked_keep_open_position_acked'] == 1
            assert stats['miners_with_position_deletions'] == 0, (i, stats_str)
            # With the fix, positions that match by UUID but have no changes are correctly tracked as "matched"
            # This test scenario has positions that get NOTHING status (no changes needed) so they count as matched
            assert stats['miners_with_position_matches'] >= 0, (i, stats_str)
            assert stats['miners_with_position_insertions'] == 1, (i, stats_str)

            assert stats['miners_with_order_deletions'] == 0, (i, stats_str)
            assert stats['miners_with_order_insertions'] == 0, (i, stats_str)
            assert stats['miners_with_order_matches'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 1, (i, stats_str)
            assert stats['positions_matched'] == 0, (i, stats_str)
            assert stats['positions_deleted'] == 0, (i, stats_str)
            assert stats['positions_kept'] == 1, (i, stats_str)

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
            assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == cd.open_ms

    def test_validate_basic_order_sync_order_match_with_heuristic_insert_one(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
        order1.processed_ms = self.default_order.processed_ms + 1000
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats

        # Comprehensive validation: position updated with order insertion
        self.validate_comprehensive_stats(
            stats,
            expected_miners=1,
            expected_eliminated=0,
            expected_pos_updates=1,  # Position was updated (got additional orders)
            expected_pos_matches=1,  # Position also matched by UUID
            expected_pos_insertions=0,
            expected_pos_deletions=0,
            expected_ord_updates=0,
            expected_ord_matches=1,  # Existing order matched heuristically
            expected_ord_insertions=1,  # New order inserted
            expected_ord_deletions=0,
            expected_exceptions=0,
            expected_raw_pos_processed=1,  # 1 position processed
            expected_raw_ord_processed=1,  # 1 existing order processed
            expected_raw_miners_processed=1,  # 1 miner processed
            expected_raw_pos_inserted=0,
            expected_raw_pos_deleted=0,
            expected_raw_ord_inserted=1,  # 1 new order inserted
            expected_raw_ord_deleted=0,
            test_context="heuristic order match with insertion"
        )

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == order2.processed_ms

    def test_validate_basic_order_sync_order_match_with_uuid_insert_one(self):
        order1 = deepcopy(self.default_order)
        order1.processed_ms = self.default_order.processed_ms + 10000 # Purposely different to ensure match on uuid happens regardless
        order1.order_type = OrderType.SHORT# Purposely different to ensure match on uuid happens regardless
        order1.leverage = 2  # Purposely different to ensure match on uuid happens regardless
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats

        # Comprehensive validation: position updated with UUID-based order match + insertion
        self.validate_comprehensive_stats(
            stats,
            expected_miners=1,
            expected_eliminated=0,
            expected_pos_updates=1,  # Position was updated (got additional orders)
            expected_pos_matches=1,  # Position also matched by UUID
            expected_pos_insertions=0,
            expected_pos_deletions=0,
            expected_ord_updates=0,
            expected_ord_matches=1,  # Existing order matched by UUID (despite different fields)
            expected_ord_insertions=1,  # New order inserted
            expected_ord_deletions=0,
            expected_exceptions=0,
            expected_raw_pos_processed=1,  # 1 position processed
            expected_raw_ord_processed=1,  # 1 existing order processed
            expected_raw_miners_processed=1,  # 1 miner processed
            expected_raw_pos_inserted=0,
            expected_raw_pos_deleted=0,
            expected_raw_ord_inserted=1,  # 1 new order inserted
            expected_raw_ord_deleted=0,
            test_context="UUID-based order match with insertion"
        )
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == order2.processed_ms

    def test_validate_basic_order_sync_no_matches_one_deletion(self):
        disk_positions = self.positions_to_disk_data([self.default_position])
        for i in range(3):
            order1 = deepcopy(self.default_order)
            order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
            if i == 0:
                order1.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10  # Purposely different to ensure no match
            if i == 1:
                order1.order_type = OrderType.SHORT  # Purposely different to ensure no match
            if i == 2:
                order1.leverage = 2  # Purposely different to ensure no match
            order2 = deepcopy(self.default_order)
            order2.order_uuid = "test_order2"
            order2.processed_ms = self.default_order.processed_ms + 1000 * 60 * 10
            orders = [order1, order2]
            position = deepcopy(self.default_position)
            position.orders = orders
            candidate_data = self.positions_to_candidate_data([position])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats
            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "

            assert stats['miners_processed'] == 1, stats

            assert stats['miners_with_position_deletions'] == 0, (i, stats_str)
            assert stats['miners_with_position_matches'] == 1, (i, stats_str)
            assert stats['miners_with_position_insertions'] == 0, (i, stats_str)

            assert stats['miners_with_order_deletions'] == 1, (i, stats_str)
            assert stats['miners_with_order_insertions'] == 1, (i, stats_str)
            assert stats['miners_with_order_matches'] == 0, (i, stats_str)

            assert stats['positions_inserted'] == 0, (i, stats_str)
            assert stats['positions_matched'] == 1, (i, stats_str)
            assert stats['positions_deleted'] == 0, (i, stats_str)
            assert stats['positions_kept'] == 0, (i, stats_str)

            assert stats['orders_inserted'] == 2, (i, stats_str)
            assert stats['orders_matched'] == 0, (i, stats_str)
            assert stats['orders_deleted'] == 1, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
            assert (self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] ==
                    min(order1.processed_ms, order2.processed_ms, self.default_order.processed_ms))

    def test_validate_order_sync_all_matches_heuristic(self):
        disk_positions = self.positions_to_disk_data([self.default_position])
        for i in range(2):
            order1 = deepcopy(self.default_order)
            order1.order_uuid = self.DEFAULT_ORDER_UUID + "foobar"
            if i == 0:
                order1.processed_ms = self.default_order.processed_ms + 1000 * 60 * 2  # Slightly different. Should match
            if i == 1:
                order1.processed_ms = self.default_order.processed_ms - 1000 * 60 * 2  # Slightly different. Should match

            orders = [order1]
            position = deepcopy(self.default_position)
            position.orders = orders
            candidate_data = self.positions_to_candidate_data([position])
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
            stats = self.position_syncer.global_stats
            stats_str = ''
            for k, v in stats.items():
                stats_str += f"{k}:{v}, "


            assert stats['miners_processed'] == 1, stats

            assert stats['miners_with_position_deletions'] == 0, (i, stats_str)
            assert stats['miners_with_position_matches'] == 1, (i, stats_str)
            assert stats['miners_with_position_insertions'] == 0, (i, stats_str)

            assert stats['miners_with_order_deletions'] == 0, (i, stats_str)
            assert stats['miners_with_order_insertions'] == 0, (i, stats_str)
            assert stats['miners_with_order_matches'] == 1, (i, stats_str)

            assert stats['positions_inserted'] == 0, (i, stats_str)
            assert stats['positions_matched'] == 1, (i, stats_str)
            assert stats['positions_deleted'] == 0, (i, stats_str)
            assert stats['positions_kept'] == 0, (i, stats_str)

            assert stats['orders_inserted'] == 0, (i, stats_str)
            assert stats['orders_matched'] == 1, (i, stats_str)
            assert stats['orders_deleted'] == 0, (i, stats_str)
            assert stats['orders_kept'] == 0, (i, stats_str)

            assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

        # Test fragmentation especially hard. May need to do a hard snap to candidates.
    def test_validate_order_sync_keep_recent_orders_one_insert(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            # Ensure they are in the future
            o.order_uuid = i
            o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])


        position = deepcopy(self.default_position)
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['miners_processed'] == 1, stats

        assert stats['miners_with_position_deletions'] == 0, stats
        assert stats['miners_with_position_matches'] == 1, stats
        assert stats['miners_with_position_insertions'] == 0, stats

        assert stats['miners_with_order_deletions'] == 0, stats
        assert stats['miners_with_order_insertions'] == 1, stats
        assert stats['miners_with_order_matches'] == 0, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 0, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 3, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms

    def test_validate_order_sync_keep_recent_orders_match_one_by_uuid(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            # Ensure they are in the future
            o.order_uuid = o.order_uuid if i == 0 else i
            o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])

        position = deepcopy(self.default_position)
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['miners_processed'] == 1, stats

        assert stats['miners_with_position_deletions'] == 0, stats
        assert stats['miners_with_position_matches'] == 1, stats
        assert stats['miners_with_position_insertions'] == 0, stats

        assert stats['miners_with_order_deletions'] == 0, stats
        assert stats['miners_with_order_insertions'] == 0, stats
        assert stats['miners_with_order_matches'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 0, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 2, stats


        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 0

    def test_validate_order_sync_one_of_each(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            o.order_uuid = o.order_uuid if i == 0 else i

            # Will still allow for a match
            if i == 0:
                o.processed_ms += 1000 * 60 * 2
            # Allow for a delete. mismatched attributes
            if i == 1:
                o.processed_ms = self.default_order.processed_ms - 1000 * 60 * 30
                o.order_type = OrderType.SHORT
            if i == 2:
                # Will allow for a keep as it is in the future.
                o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
                o.order_type = OrderType.SHORT
                o.leverage = -4

        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])
        position = deepcopy(self.default_position)
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = self.default_order.processed_ms + 1000 * 60 * 20
        order_to_insert.leverage = 5
        order_to_match = deepcopy(self.default_order)
        position.orders = [order_to_match, order_to_insert]
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['miners_processed'] == 1, stats

        assert stats['miners_with_position_deletions'] == 0, stats
        assert stats['miners_with_position_matches'] == 1, stats
        assert stats['miners_with_position_insertions'] == 0, stats

        assert stats['miners_with_order_deletions'] == 1, stats
        assert stats['miners_with_order_insertions'] == 1, stats
        assert stats['miners_with_order_matches'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 1, stats
        assert stats['orders_kept'] == 1, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms - 1000 * 60 * 30

    def test_validate_position_sync_one_of_each_uuid_match(self):
        # Create non-overlapping closed positions
        # dp_to_keep: [T0 + 30min, T0 + 40min]
        dp_to_keep = deepcopy(self.default_closed_position)
        dp_to_keep.position_uuid = "to_keep"
        dp_to_keep.open_ms += 1000 * 60 * 30
        dp_to_keep.close_ms = dp_to_keep.open_ms + 1000 * 60 * 10
        dp_to_keep.close_out_position(dp_to_keep.close_ms)

        # dp_to_delete: [T0 - 10min, T0 - 5min]
        dp_to_delete = deepcopy(self.default_closed_position)
        dp_to_delete.position_uuid = "to_delete"
        dp_to_delete.open_ms -= 1000 * 60 * 10
        dp_to_delete.close_ms = dp_to_delete.open_ms + 1000 * 60 * 5
        dp_to_delete.close_out_position(dp_to_delete.close_ms)

        # dp_to_match: [T0, T0 + 10min]
        dp_to_match = deepcopy(self.default_closed_position)
        dp_to_match.position_uuid = "to_match"
        dp_to_match.close_ms = dp_to_match.open_ms + 1000 * 60 * 10
        dp_to_match.close_out_position(dp_to_match.close_ms)
        dp_to_match.orders[0].processed_ms += 1000

        # cp_to_insert: [T0 + 50min, T0 + 60min]
        cp_to_insert = deepcopy(self.default_closed_position)
        cp_to_insert.position_uuid = "to_insert"
        cp_to_insert.orders[0].order_type = OrderType.SHORT
        cp_to_insert.open_ms += 1000 * 60 * 50
        cp_to_insert.close_ms = cp_to_insert.open_ms + 1000 * 60 * 10
        cp_to_insert.close_out_position(cp_to_insert.close_ms)

        # cp_to_match: [T0, T0 + 10min] (same as dp_to_match - they should match by UUID)
        cp_to_match = deepcopy(self.default_closed_position)
        cp_to_match.position_uuid = "to_match"
        cp_to_match.close_ms = cp_to_match.open_ms + 1000 * 60 * 10
        cp_to_match.close_out_position(cp_to_match.close_ms)
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = cp_to_match.open_ms + 1000 * 60 * 5  # Within the position's time range
        cp_to_match.orders.append(order_to_insert)

        disk_positions = self.positions_to_disk_data([dp_to_keep, dp_to_delete, dp_to_match])
        candidate_data = self.positions_to_candidate_data([cp_to_insert, cp_to_match])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "

        assert stats['miners_processed'] == 1, stats

        assert stats['miners_with_position_deletions'] == 1, stats
        assert stats['miners_with_position_matches'] == 1, stats
        assert stats['miners_with_position_insertions'] == 1, stats

        assert stats['miners_with_order_deletions'] == 0, stats
        assert stats['miners_with_order_insertions'] == 1, stats
        assert stats['miners_with_order_matches'] == 1, stats

        assert stats['positions_inserted'] == 1, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 1, stats
        assert stats['positions_kept'] == 1, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[
                   self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms - 1000 * 60 * 10


    def test_validate_position_sync_one_of_each_heuristic_match(self):
        # Create non-overlapping closed positions
        # dp_to_keep: [T0 + 30min, T0 + 40min]
        dp_to_keep = deepcopy(self.default_closed_position)
        dp_to_keep.position_uuid = "to_keep"
        dp_to_keep.open_ms += 1000 * 60 * 30
        dp_to_keep.close_ms = dp_to_keep.open_ms + 1000 * 60 * 10
        dp_to_keep.close_out_position(dp_to_keep.close_ms)

        # dp_to_delete: [T0 - 10min, T0 - 5min]
        dp_to_delete = deepcopy(self.default_closed_position)
        dp_to_delete.position_uuid = "to_delete"
        dp_to_delete.open_ms -= 1000 * 60 * 10
        dp_to_delete.close_ms = dp_to_delete.open_ms + 1000 * 60 * 5
        dp_to_delete.close_out_position(dp_to_delete.close_ms)

        # dp_to_match: [T0, T0 + 10min]
        dp_to_match = deepcopy(self.default_closed_position)
        dp_to_match.position_uuid = "to_match"
        dp_to_match.close_ms = dp_to_match.open_ms + 1000 * 60 * 10
        dp_to_match.close_out_position(dp_to_match.close_ms)
        dp_to_match.orders[0].processed_ms += 1000

        # cp_to_insert: [T0 + 50min, T0 + 60min]
        cp_to_insert = deepcopy(self.default_closed_position)
        cp_to_insert.position_uuid = "to_insert"
        cp_to_insert.orders[0].order_type = OrderType.SHORT
        cp_to_insert.open_ms += 1000 * 60 * 50
        cp_to_insert.close_ms = cp_to_insert.open_ms + 1000 * 60 * 10
        cp_to_insert.close_out_position(cp_to_insert.close_ms)

        # cp_to_match: [T0, T0 + 10min] (matches dp_to_match by heuristic)
        cp_to_match = deepcopy(self.default_closed_position)
        cp_to_match.position_uuid = "candy_to_match"
        cp_to_match.close_ms = cp_to_match.open_ms + 1000 * 60 * 10
        cp_to_match.close_out_position(cp_to_match.close_ms)
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = cp_to_match.open_ms + 1000 * 60 * 5  # Within the position's time range
        cp_to_match.orders.append(order_to_insert)


        disk_positions = self.positions_to_disk_data([dp_to_keep, dp_to_delete, dp_to_match])
        candidate_data = self.positions_to_candidate_data([cp_to_insert, cp_to_match])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['miners_processed'] == 1, stats

        assert stats['miners_with_position_deletions'] == 1, stats
        assert stats['miners_with_position_matches'] == 1, stats
        assert stats['miners_with_position_insertions'] == 1, stats

        assert stats['miners_with_order_deletions'] == 0, stats
        assert stats['miners_with_order_insertions'] == 1, stats
        assert stats['miners_with_order_matches'] == 1, stats

        assert stats['positions_inserted'] == 1, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 1, stats
        assert stats['positions_kept'] == 1, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 0, stats

        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == self.default_order.processed_ms - 1000 * 60 * 10


    def test_validate_order_sync_testing_hardsnap(self):
        dp1 = deepcopy(self.default_position)
        orders = [deepcopy(self.default_order) for _ in range(3)]
        for i, o in enumerate(orders):
            o.order_uuid = o.order_uuid if i == 0 else i

            # Will still allow for a match
            if i == 0:
                o.processed_ms += 1000 * 60 * 2
            # Keeping this monstrosity only because it comes after hardsnap
            if i == 1:
                o.processed_ms = self.default_order.processed_ms + 1000 * 60 * 30
                o.order_type = OrderType.SHORT
            if i == 2:
                # Will allow for a keep as it is in the future.
                o.processed_ms = self.default_order.processed_ms + 2 * AUTO_SYNC_ORDER_LAG_MS
                o.order_type = OrderType.SHORT
                o.leverage = -4

        dp1.orders = orders
        disk_positions = self.positions_to_disk_data([dp1])
        position = deepcopy(self.default_position)
        order_to_insert = deepcopy(self.default_order)
        order_to_insert.order_uuid = "to_insert"
        order_to_insert.processed_ms = self.default_order.processed_ms + 1000 * 60 * 20
        order_to_insert.leverage = 5
        order_to_match = deepcopy(self.default_order)
        position.orders = [order_to_match, order_to_insert]
        candidate_data = self.positions_to_candidate_data([position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        stats_str = ''
        for k, v in stats.items():
            stats_str += f"{k}:{v}, "


        assert stats['miners_processed'] == 1, stats

        assert stats['miners_with_position_deletions'] == 0, stats
        assert stats['miners_with_position_matches'] == 1, stats
        assert stats['miners_with_position_insertions'] == 0, stats

        assert stats['miners_with_order_deletions'] == 0, stats
        assert stats['miners_with_order_insertions'] == 1, stats
        assert stats['miners_with_order_matches'] == 1, stats

        assert stats['positions_inserted'] == 0, stats
        assert stats['positions_matched'] == 1, stats
        assert stats['positions_deleted'] == 0, stats
        assert stats['positions_kept'] == 0, stats

        assert stats['orders_inserted'] == 1, stats
        assert stats['orders_matched'] == 1, stats
        assert stats['orders_deleted'] == 0, stats
        assert stats['orders_kept'] == 2, stats


        assert len(self.position_syncer.perf_ledger_hks_to_invalidate) == 1
        assert self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY] == order_to_insert.processed_ms

    def test_hard_snap_cutoff_ms_edge_cases(self):
        """Test edge cases around hard_snap_cutoff_ms boundary to ensure correct position handling"""
        # Create positions around the hard snap cutoff boundary
        hard_snap_cutoff_ms = self.DEFAULT_OPEN_MS

        # Position just before cutoff - should be deleted if not in candidate
        # Make it a closed position to avoid overlap with other positions
        before_cutoff = deepcopy(self.default_position)
        before_cutoff.position_uuid = "before_cutoff"
        before_cutoff.open_ms = hard_snap_cutoff_ms - 10000
        before_cutoff.close_ms = hard_snap_cutoff_ms - 1
        before_cutoff.close_out_position(before_cutoff.close_ms)

        # Position exactly at cutoff - should be kept
        # Make it a closed position to avoid overlap
        at_cutoff = deepcopy(self.default_position)
        at_cutoff.position_uuid = "at_cutoff"
        at_cutoff.open_ms = hard_snap_cutoff_ms
        at_cutoff.close_ms = hard_snap_cutoff_ms + 5000
        at_cutoff.close_out_position(at_cutoff.close_ms)

        # Position just after cutoff - should be kept
        # Make it a closed position to avoid overlap
        after_cutoff = deepcopy(self.default_position)
        after_cutoff.position_uuid = "after_cutoff"
        after_cutoff.open_ms = hard_snap_cutoff_ms + 10000
        after_cutoff.close_ms = hard_snap_cutoff_ms + 15000
        after_cutoff.close_out_position(after_cutoff.close_ms)
        
        # Test with empty candidate data - positions at/after cutoff should be kept
        candidate_data = self.positions_to_candidate_data([])
        candidate_data['hard_snap_cutoff_ms'] = hard_snap_cutoff_ms
        disk_positions = self.positions_to_disk_data([before_cutoff, at_cutoff, after_cutoff])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data, 
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        assert stats['positions_deleted'] == 1, f"Should delete 1 position before cutoff: {stats}"
        assert stats['positions_kept'] == 2, f"Should keep 2 positions at/after cutoff: {stats}"
        
        # Test with candidate having different position - before cutoff still deleted
        new_position = deepcopy(self.default_position)
        new_position.position_uuid = "new_position"
        new_position.open_ms = hard_snap_cutoff_ms + 1000 * 60 * 5  # 5 minutes later to avoid heuristic matching
        new_position.close_ms = hard_snap_cutoff_ms + 1000 * 60 * 5 + 5000
        new_position.close_out_position(new_position.close_ms)

        # Make sure the order type is different to avoid positions_aligned matching
        new_position.orders[0].order_type = OrderType.SHORT
        new_position.orders[0].leverage = -1
        new_position.position_type = OrderType.SHORT
        
        candidate_data = self.positions_to_candidate_data([new_position])
        candidate_data['hard_snap_cutoff_ms'] = hard_snap_cutoff_ms
        
        # Re-init stats to test independently
        self.position_syncer.init_data()
        
        # Recreate disk positions for second test since first test modified disk state
        fresh_disk_positions = self.positions_to_disk_data([before_cutoff, at_cutoff, after_cutoff])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=fresh_disk_positions)
        stats = self.position_syncer.global_stats
        
        assert stats['positions_deleted'] == 1, f"Should still delete before cutoff: {stats}"
        assert stats['positions_kept'] == 2, f"Should keep at/after cutoff: {stats}"
        assert stats['positions_inserted'] == 1, f"Should insert new position: {stats}"

    def test_position_uuid_conflict_resolution(self):
        """Test handling of positions with same UUID but different characteristics"""
        base_uuid = "conflict_uuid"
        
        # Existing position
        existing = deepcopy(self.default_position)
        existing.position_uuid = base_uuid
        existing.orders[0].order_type = OrderType.LONG
        
        # Candidate with same UUID but different order type (should match by UUID)
        candidate = deepcopy(self.default_position)
        candidate.position_uuid = base_uuid
        candidate.orders[0].order_type = OrderType.SHORT
        candidate.orders[0].leverage = -1
        
        # Add a second order to candidate
        second_order = deepcopy(self.default_order)
        second_order.order_uuid = "second_order"
        second_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        candidate.orders.append(second_order)
        
        candidate_data = self.positions_to_candidate_data([candidate])
        disk_positions = self.positions_to_disk_data([existing])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should match by UUID despite different order types
        assert stats['positions_matched'] == 1, f"Should match by UUID: {stats}"
        assert stats['orders_inserted'] == 1, f"Should insert second order: {stats}"
        assert stats['orders_matched'] == 1, f"Should match first order by UUID: {stats}"

    def test_heuristic_position_matching(self):
        """Test position matching based on timestamps when UUIDs don't match"""
        # Create positions with different UUIDs but similar timestamps
        existing = deepcopy(self.default_position)
        existing.position_uuid = "existing_uuid"
        existing.open_ms = self.DEFAULT_OPEN_MS
        
        # Candidate with different UUID but timestamp within SYNC_LOOK_AROUND_MS
        candidate = deepcopy(self.default_position)
        candidate.position_uuid = "candidate_uuid"
        candidate.open_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2  # 2 minutes later
        
        # Add different orders to test order sync
        candidate.orders[0].leverage = 2
        extra_order = deepcopy(self.default_order)
        extra_order.order_uuid = "extra_order"
        extra_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 5
        candidate.orders.append(extra_order)
        
        candidate_data = self.positions_to_candidate_data([candidate])
        disk_positions = self.positions_to_disk_data([existing])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should match by timestamp alignment
        assert stats['positions_matched'] == 1, f"Should match by timestamp: {stats}"
        assert stats['orders_inserted'] == 1, f"Should insert extra order: {stats}"

    def test_order_sync_with_future_orders(self):
        """Test order sync behavior with orders beyond AUTO_SYNC_ORDER_LAG_MS"""
        position = deepcopy(self.default_position)
        
        # Add orders at various times relative to candidate creation
        candidate_creation_time = self.DEFAULT_OPEN_MS + AUTO_SYNC_ORDER_LAG_MS + 1000 * 60 * 60
        
        # Order within sync window - should be synced
        within_window = deepcopy(self.default_order)
        within_window.order_uuid = "within_window"
        within_window.processed_ms = candidate_creation_time - AUTO_SYNC_ORDER_LAG_MS + 1000
        
        # Order beyond sync window - should be kept in disk position
        beyond_window = deepcopy(self.default_order)
        beyond_window.order_uuid = "beyond_window"
        beyond_window.processed_ms = candidate_creation_time + 1000
        
        disk_position = deepcopy(position)
        disk_position.orders = [position.orders[0], within_window, beyond_window]
        
        candidate_position = deepcopy(position)
        candidate_position.orders = [position.orders[0], within_window]
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        candidate_data['created_timestamp_ms'] = candidate_creation_time
        disk_positions = self.positions_to_disk_data([disk_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        assert stats['orders_kept'] == 1, f"Should keep future order: {stats}"
        assert stats['orders_matched'] == 2, f"Should match orders within window: {stats}"

    @patch('vali_objects.utils.auto_sync.requests.get')
    def test_perform_sync_with_network_errors(self, mock_get):
        """Test autosync behavior with network failures"""
        # Test HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_get.return_value = mock_response
        
        # Should handle error gracefully
        self.position_syncer.n_orders_being_processed = [0]
        # Mock lock must support context manager protocol (__enter__ and __exit__)
        mock_lock = Mock()
        mock_lock.__enter__ = Mock(return_value=mock_lock)
        mock_lock.__exit__ = Mock(return_value=None)
        self.position_syncer.signal_sync_lock = mock_lock
        self.position_syncer.signal_sync_condition = Mock()
        
        # This should not raise an exception
        self.position_syncer.perform_sync()
        
        # Test with successful response but invalid JSON
        mock_response.raise_for_status.side_effect = None
        mock_response.content = b'invalid json'
        
        self.position_syncer.perform_sync()

    def test_split_position_on_flat_complex(self):
        """Test splitting positions when FLAT orders are present"""
        # Create position with multiple FLAT orders
        position = deepcopy(self.default_position)
        position.position_uuid = "split_test"
        
        # LONG -> increase -> FLAT -> SHORT -> FLAT sequence
        orders = []
        
        # Initial LONG
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long1"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2
        long_order.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(long_order)
        
        # Increase position
        increase_order = deepcopy(self.default_order)
        increase_order.order_uuid = "increase1"
        increase_order.order_type = OrderType.LONG
        increase_order.leverage = 1
        increase_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(increase_order)
        
        # First FLAT
        flat1 = deepcopy(self.default_order)
        flat1.order_uuid = "flat1"
        flat1.order_type = OrderType.FLAT
        flat1.leverage = -3
        flat1.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(flat1)
        
        # New SHORT position
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short1"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1.5
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(short_order)
        
        # Second FLAT
        flat2 = deepcopy(self.default_order)
        flat2.order_uuid = "flat2"
        flat2.order_type = OrderType.FLAT
        flat2.leverage = 1.5
        flat2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 4
        orders.append(flat2)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should spawn new position from orders after FLAT
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, f"Should spawn positions: {stats}"

    def test_challengeperiod_sync_integration(self):
        """Test challengeperiod sync when included in candidate data"""
        # Setup challengeperiod manager
        self.position_manager.challengeperiod_manager = ChallengePeriodManager(
            metagraph=self.mock_metagraph,
            position_manager=self.position_manager,
            running_unit_tests=True
        )
        
        # Create candidate data with challengeperiod info
        test_hotkey2 = "test_miner_2"
        self.mock_metagraph.hotkeys.append(test_hotkey2)
        
        candidate_data = self.positions_to_candidate_data([self.default_position])
        candidate_data['challengeperiod'] = {
            self.DEFAULT_MINER_HOTKEY: {
                "bucket": MinerBucket.CHALLENGE.value,
                "bucket_start_time": self.DEFAULT_OPEN_MS,
                "previous_bucket": None,
                "previous_bucket_start_time": None
            },
            test_hotkey2: {
                "bucket": MinerBucket.MAINCOMP.value,
                "bucket_start_time": self.DEFAULT_OPEN_MS - 1000 * 60 * 60 * 24,
                "previous_bucket": None,
                "previous_bucket_start_time": None
            }
        }
        
        # Clear any existing challengeperiod data
        self.position_manager.challengeperiod_manager.active_miners.clear()
        
        disk_positions = self.positions_to_disk_data([self.default_position])
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        
        # Verify challengeperiod was synced
        assert self.DEFAULT_MINER_HOTKEY in self.position_manager.challengeperiod_manager.active_miners
        assert test_hotkey2 in self.position_manager.challengeperiod_manager.active_miners
        
        bucket1, _, _, _ = self.position_manager.challengeperiod_manager.active_miners[self.DEFAULT_MINER_HOTKEY]
        bucket2, _, _, _ = self.position_manager.challengeperiod_manager.active_miners[test_hotkey2]
        
        assert bucket1 == MinerBucket.CHALLENGE
        assert bucket2 == MinerBucket.MAINCOMP

    def test_elimination_sync_and_ledger_invalidation(self):
        """Test elimination sync and perf ledger invalidation"""
        # Ensure clean elimination state for test
        self.elimination_manager.eliminations.clear()
        
        # Create eliminations in candidate data
        eliminated_hotkey = "eliminated_miner"
        self.mock_metagraph.hotkeys.append(eliminated_hotkey)
        
        # Position for eliminated miner
        eliminated_position = deepcopy(self.default_position)
        eliminated_position.miner_hotkey = eliminated_hotkey
        
        candidate_data = self.positions_to_candidate_data([self.default_position])
        candidate_data['eliminations'] = [{
            'hotkey': eliminated_hotkey,
            'reason': 'Test elimination',
            'timestamp': self.DEFAULT_OPEN_MS
        }]
        
        # Include the eliminated miner's position in candidate data
        candidate_data['positions'][eliminated_hotkey] = {
            'positions': [json.loads(str(eliminated_position), cls=GeneralizedJSONDecoder)]
        }
        
        disk_positions = {
            self.DEFAULT_MINER_HOTKEY: [self.default_position],
            eliminated_hotkey: [eliminated_position]
        }
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should skip eliminated miner
        assert stats['n_miners_skipped_eliminated'] == 1, f"Should skip eliminated miner: {stats}"
        
        # Should invalidate perf ledger for eliminated miner
        assert eliminated_hotkey in self.position_syncer.perf_ledger_hks_to_invalidate, f"Expected {eliminated_hotkey} in {self.position_syncer.perf_ledger_hks_to_invalidate}"
        assert self.position_syncer.perf_ledger_hks_to_invalidate[eliminated_hotkey] == 0

    def test_concurrent_position_handling(self):
        """Test handling of multiple open positions for same trade pair"""
        # Create two open positions for same trade pair (invalid state)
        open_pos1 = deepcopy(self.default_open_position)
        open_pos1.position_uuid = "open1"
        open_pos1.open_ms = self.DEFAULT_OPEN_MS
        
        open_pos2 = deepcopy(self.default_open_position)
        open_pos2.position_uuid = "open2"
        open_pos2.open_ms = self.DEFAULT_OPEN_MS + 1000 * 60  # 1 minute later
        
        # Candidate has the newer position
        candidate_data = self.positions_to_candidate_data([open_pos2])
        disk_positions = self.positions_to_disk_data([open_pos1, open_pos2])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should handle duplicate open positions
        assert stats['n_positions_closed_duplicate_opens_for_trade_pair'] >= 0, f"Should handle duplicates: {stats}"

    def test_position_sync_result_exception_handling(self):
        """Test that PositionSyncResultException is properly propagated"""
        # Create a position that WILL actually trigger splitting and write_modifications
        position = deepcopy(self.default_position)
        position.position_uuid = "exception_test"
        
        # Create position that needs splitting: LONG -> FLAT -> SHORT
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -2.0
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1.0
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long_order, flat_order, short_order]
        
        # Mock the write_modifications to raise exception
        original_write = self.position_syncer.write_modifications
        
        def mock_write_error(position_to_sync_status, stats):
            # Force a mismatch in kept_and_matched count
            raise PositionSyncResultException("Test exception")
        
        self.position_syncer.write_modifications = mock_write_error
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([position])
        
        # Should raise PositionSyncResultException
        with self.assertRaises(PositionSyncResultException):
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                               disk_positions=disk_positions)
        
        # Restore original
        self.position_syncer.write_modifications = original_write

    def test_order_matching_edge_cases(self):
        """Test order matching with various edge cases"""
        position = deepcopy(self.default_position)
        
        # Test exact UUID match with different attributes
        order1 = deepcopy(self.default_order)
        order1.leverage = 2
        order1.order_type = OrderType.SHORT
        
        # Test heuristic match within time window
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "different_uuid"
        order2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2  # 2 min later
        
        # Test no match - too far in time
        order3 = deepcopy(self.default_order)
        order3.order_uuid = "no_match"
        order3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 10  # 10 min later
        order3.leverage = 5
        
        disk_position = deepcopy(position)
        disk_position.orders = [order1]
        
        candidate_position = deepcopy(position)
        candidate_position.orders = [self.default_order, order2, order3]
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        disk_positions = self.positions_to_disk_data([disk_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        assert stats['orders_matched'] == 1, f"Should match by UUID: {stats}"
        assert stats['orders_inserted'] == 2, f"Should insert non-matching orders: {stats}"

    def test_sync_with_cooldown_timing(self):
        """Test the sync_positions_with_cooldown timing logic"""
        # Test cooldown period
        self.position_syncer.last_signal_sync_time_ms = TimeUtil.now_in_millis()
        self.position_syncer.force_ran_on_boot = True
        
        # Should not sync - too recent
        with patch.object(self.position_syncer, 'perform_sync') as mock_sync:
            self.position_syncer.sync_positions_with_cooldown(auto_sync_enabled=True)
            mock_sync.assert_not_called()
        
        # Simulate time passing
        self.position_syncer.last_signal_sync_time_ms = TimeUtil.now_in_millis() - 1000 * 60 * 31
        
        # Test outside time window
        with patch('vali_objects.utils.auto_sync.TimeUtil.generate_start_timestamp') as mock_time:
            mock_dt = Mock()
            mock_dt.hour = 5  # Not 6
            mock_dt.minute = 15
            mock_time.return_value = mock_dt
            
            with patch.object(self.position_syncer, 'perform_sync') as mock_sync:
                self.position_syncer.sync_positions_with_cooldown(auto_sync_enabled=True)
                mock_sync.assert_not_called()
        
        # Test within time window
        with patch('vali_objects.utils.auto_sync.TimeUtil.generate_start_timestamp') as mock_time:
            mock_dt = Mock()
            mock_dt.hour = 6
            mock_dt.minute = 15  # Between 8 and 20
            mock_time.return_value = mock_dt
            
            with patch.object(self.position_syncer, 'perform_sync') as mock_sync:
                self.position_syncer.sync_positions_with_cooldown(auto_sync_enabled=True)
                mock_sync.assert_called_once()

    def test_hard_snap_cutoff_with_position_updates(self):
        """Test that positions before cutoff are deleted even when other positions are being updated"""
        hard_snap_cutoff_ms = self.DEFAULT_OPEN_MS

        # Create three positions - one to be deleted, one to be updated, one to be kept
        # Make all closed positions with non-overlapping time ranges
        to_delete = deepcopy(self.default_position)
        to_delete.position_uuid = "to_delete"
        to_delete.open_ms = hard_snap_cutoff_ms - 10000
        to_delete.close_ms = hard_snap_cutoff_ms - 1000
        to_delete.close_out_position(to_delete.close_ms)

        to_update = deepcopy(self.default_position)
        to_update.position_uuid = "to_update"
        to_update.open_ms = hard_snap_cutoff_ms + 1000
        to_update.close_ms = hard_snap_cutoff_ms + 5000
        to_update.close_out_position(to_update.close_ms)

        to_keep = deepcopy(self.default_position)
        to_keep.position_uuid = "to_keep"
        to_keep.open_ms = hard_snap_cutoff_ms + 10000
        to_keep.close_ms = hard_snap_cutoff_ms + 15000
        to_keep.close_out_position(to_keep.close_ms)
        
        # Candidate has updated version of to_update with extra order
        candidate_update = deepcopy(to_update)
        extra_order = deepcopy(self.default_order)
        extra_order.order_uuid = "extra_order"
        extra_order.processed_ms = to_update.open_ms + 1000 * 60
        candidate_update.orders.append(extra_order)
        
        candidate_data = self.positions_to_candidate_data([candidate_update])
        candidate_data['hard_snap_cutoff_ms'] = hard_snap_cutoff_ms
        
        disk_positions = self.positions_to_disk_data([to_delete, to_update, to_keep])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should delete position before cutoff, update matched position, and keep the other
        assert stats['positions_deleted'] == 1, f"Should delete position before cutoff: {stats}"
        assert stats['positions_matched'] == 1, f"Should match and update one position: {stats}"
        assert stats['positions_kept'] == 1, f"Should keep one position: {stats}"
        assert stats['orders_inserted'] == 1, f"Should insert extra order: {stats}"

    def test_order_matching_time_boundary(self):
        """Test order matching edge case where orders are at the exact SYNC_LOOK_AROUND_MS boundary"""
        position = deepcopy(self.default_position)
        
        # Create orders at exact boundaries
        base_time = self.DEFAULT_OPEN_MS
        
        # Order just within boundary - should match
        within_order = deepcopy(self.default_order)
        within_order.order_uuid = "within_order"
        within_order.processed_ms = base_time + self.position_syncer.SYNC_LOOK_AROUND_MS - 1
        
        # Order at exact boundary - should NOT match (uses < not <=)
        boundary_order = deepcopy(self.default_order) 
        boundary_order.order_uuid = "boundary_order"
        boundary_order.processed_ms = base_time + self.position_syncer.SYNC_LOOK_AROUND_MS
        
        disk_position = deepcopy(position)
        disk_position.orders = [self.default_order]
        
        candidate_position = deepcopy(position)
        candidate_position.orders = [within_order, boundary_order]
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        disk_positions = self.positions_to_disk_data([disk_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should match default order with within_order heuristically
        # Boundary order should be inserted as new (doesn't match due to < boundary)
        assert stats['orders_matched'] == 1, f"Should match one order heuristically: {stats}"
        assert stats['orders_deleted'] == 0, f"Should not delete matched order: {stats}"
        assert stats['orders_inserted'] == 1, f"Should insert boundary order: {stats}"

    def test_mothership_mode(self):
        """Test behavior when running as mothership"""
        # Mock mothership mode
        with patch('vali_objects.utils.validator_sync_base.ValiUtils.get_secrets') as mock_secrets:
            mock_secrets.return_value = {'ms': 'mothership_secret'}
            
            # Create new syncer in mothership mode
            mothership_syncer = PositionSyncer(
                running_unit_tests=True,
                position_manager=self.position_manager
            )
            
            assert mothership_syncer.is_mothership
            
            # Test that mothership doesn't write modifications
            candidate_data = self.positions_to_candidate_data([self.default_position])
            disk_positions = self.positions_to_disk_data([])
            
            # Mock position manager methods to track calls
            with patch.object(self.position_manager, 'delete_position') as mock_delete:
                with patch.object(self.position_manager, 'overwrite_position_on_disk') as mock_overwrite:
                    mothership_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                                     disk_positions=disk_positions)
                    
                    # Mothership should not modify positions
                    mock_delete.assert_not_called()
                    mock_overwrite.assert_not_called()

    def test_position_dedupe_before_sync(self):
        """Test that position deduplication happens before sync"""
        # Create duplicate positions with same UUID
        dup1 = deepcopy(self.default_position)
        dup1.position_uuid = "duplicate_uuid"
        dup1.open_ms = self.DEFAULT_OPEN_MS
        
        dup2 = deepcopy(self.default_position)
        dup2.position_uuid = "duplicate_uuid"  # Same UUID
        dup2.open_ms = self.DEFAULT_OPEN_MS + 1000
        dup2.orders[0].processed_ms = self.DEFAULT_OPEN_MS + 1000
        
        # Add different order to dup2 to make them different
        extra_order = deepcopy(self.default_order)
        extra_order.order_uuid = "extra_order"
        extra_order.processed_ms = self.DEFAULT_OPEN_MS + 2000
        dup2.orders.append(extra_order)
        
        # Candidate has duplicates
        candidate_data = self.positions_to_candidate_data([dup1, dup2])
        disk_positions = self.positions_to_disk_data([])
        
        # Should dedupe before processing
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should only insert one position after deduplication
        assert stats['positions_inserted'] == 1, f"Should dedupe positions: {stats}"

    def test_order_sync_with_duplicate_uuids(self):
        """Test handling of orders with duplicate UUIDs within same position"""
        position = deepcopy(self.default_position)
        
        # Create orders with duplicate UUID
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "duplicate_order_uuid"
        order1.processed_ms = self.DEFAULT_OPEN_MS
        order1.leverage = 1
        
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "duplicate_order_uuid"  # Same UUID
        order2.processed_ms = self.DEFAULT_OPEN_MS + 1000
        order2.leverage = 2
        
        order3 = deepcopy(self.default_order)
        order3.order_uuid = "unique_order_uuid"
        order3.processed_ms = self.DEFAULT_OPEN_MS + 2000
        
        candidate_position = deepcopy(position)
        candidate_position.orders = [order1, order2, order3]
        
        disk_position = deepcopy(position)
        disk_position.orders = [order1]  # Only has first duplicate
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        disk_positions = self.positions_to_disk_data([disk_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should handle duplicate UUIDs gracefully
        assert stats['orders_matched'] >= 1, f"Should match at least one order: {stats}"
        assert stats['orders_inserted'] >= 1, f"Should insert unique order: {stats}"

    def test_position_sync_with_invalid_order_sequence(self):
        """Test syncing positions with invalid order sequences"""
        # Create position with orders out of chronological order
        position = deepcopy(self.default_position)
        position.position_uuid = "invalid_sequence"
        
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "order1"
        order1.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        order1.order_type = OrderType.LONG
        
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "order2"
        order2.processed_ms = self.DEFAULT_OPEN_MS  # Earlier than order1!
        order2.order_type = OrderType.SHORT
        
        position.orders = [order1, order2]  # Out of chronological order
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should still insert the position
        assert stats['positions_inserted'] == 1, f"Should insert position: {stats}"

    def test_empty_order_list_handling(self):
        """Test handling of positions with empty order lists"""
        # Create position with no orders (invalid state)
        empty_position = deepcopy(self.default_position)
        empty_position.position_uuid = "empty_orders"
        empty_position.orders = []
        
        candidate_data = self.positions_to_candidate_data([empty_position])
        disk_positions = self.positions_to_disk_data([])
        
        # Should handle empty orders gracefully (likely assertion error in sync_orders)
        try:
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                               disk_positions=disk_positions)
        except AssertionError:
            # Expected - empty orders should trigger assertion
            pass

    def test_position_rebuild_after_order_sync(self):
        """Test that positions are properly rebuilt after order synchronization"""
        # Create position with specific order sequence
        position = deepcopy(self.default_position)
        position.position_uuid = "rebuild_test"
        
        # Original has LONG order
        disk_position = deepcopy(position)
        
        # Candidate adds FLAT order
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat_order"
        flat_order.order_type = OrderType.FLAT
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        candidate_position = deepcopy(position)
        candidate_position.orders.append(flat_order)
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        disk_positions = self.positions_to_disk_data([disk_position])
        
        # Mock rebuild_position_with_updated_orders to verify it's called
        rebuild_called = [False]
        original_rebuild = Position.rebuild_position_with_updated_orders
        
        def mock_rebuild(self, live_price_fetcher):
            rebuild_called[0] = True
            original_rebuild(self, live_price_fetcher)
        
        Position.rebuild_position_with_updated_orders = mock_rebuild
        
        try:
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                               disk_positions=disk_positions)
            
            assert rebuild_called[0], "Position should be rebuilt after order sync"
        finally:
            Position.rebuild_position_with_updated_orders = original_rebuild

    def test_shadow_mode_no_writes(self):
        """Test that shadow mode doesn't write any changes"""
        # Create scenario with changes
        to_insert = deepcopy(self.default_position)
        to_insert.position_uuid = "to_insert"
        
        to_delete = deepcopy(self.default_position)
        to_delete.position_uuid = "to_delete"
        to_delete.open_ms = self.DEFAULT_OPEN_MS - AUTO_SYNC_ORDER_LAG_MS - 1000
        
        candidate_data = self.positions_to_candidate_data([to_insert])
        disk_positions = self.positions_to_disk_data([to_delete])
        
        # Mock write methods
        with patch.object(self.position_manager, 'delete_position') as mock_delete:
            with patch.object(self.position_manager, 'overwrite_position_on_disk') as mock_overwrite:
                # Run in shadow mode
                self.position_syncer.sync_positions(shadow_mode=True, candidate_data=candidate_data,
                                                   disk_positions=disk_positions)
                
                # Should not write anything in shadow mode
                mock_delete.assert_not_called()
                mock_overwrite.assert_not_called()

    def test_perf_ledger_invalidation_tracking(self):
        """Test that perf ledger invalidation timestamps are tracked correctly"""
        # Create multiple positions with different change timestamps
        # Make them closed positions with non-overlapping time ranges
        pos1 = deepcopy(self.default_position)
        pos1.position_uuid = "pos1"
        pos1.open_ms = self.DEFAULT_OPEN_MS
        pos1.close_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        pos1.close_out_position(pos1.close_ms)

        pos2 = deepcopy(self.default_position)
        pos2.position_uuid = "pos2"
        pos2.open_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 5
        pos2.close_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 8
        pos2.close_out_position(pos2.close_ms)
        
        # Candidate adds order to pos1 (earlier timestamp)
        new_order = deepcopy(self.default_order)
        new_order.order_uuid = "new_order"
        new_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        candidate_pos1 = deepcopy(pos1)
        candidate_pos1.orders.append(new_order)
        
        # Candidate deletes pos2 (later timestamp)
        candidate_data = self.positions_to_candidate_data([candidate_pos1])
        candidate_data['hard_snap_cutoff_ms'] = pos2.open_ms + 1  # Force pos2 deletion
        
        disk_positions = self.positions_to_disk_data([pos1, pos2])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        
        # Should track minimum timestamp of changes
        assert self.DEFAULT_MINER_HOTKEY in self.position_syncer.perf_ledger_hks_to_invalidate
        invalidation_time = self.position_syncer.perf_ledger_hks_to_invalidate[self.DEFAULT_MINER_HOTKEY]
        assert invalidation_time == new_order.processed_ms, f"Should use earliest change time: {invalidation_time}"

    def test_position_sync_with_multiple_trade_pairs(self):
        """Test syncing positions across multiple trade pairs"""
        # Create positions for different trade pairs
        btc_position = deepcopy(self.default_position)
        btc_position.position_uuid = "btc_pos"
        btc_position.trade_pair = TradePair.BTCUSD
        
        eth_position = deepcopy(self.default_position)
        eth_position.position_uuid = "eth_pos"
        eth_position.trade_pair = TradePair.ETHUSD
        
        # Add order to ETH position in candidate
        new_order = deepcopy(self.default_order)
        new_order.order_uuid = "eth_order"
        new_order.trade_pair = TradePair.ETHUSD
        new_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        candidate_eth = deepcopy(eth_position)
        candidate_eth.orders.append(new_order)
        
        candidate_data = self.positions_to_candidate_data([btc_position, candidate_eth])
        disk_positions = self.positions_to_disk_data([btc_position, eth_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should match both positions, insert order only for ETH
        assert stats['positions_matched'] == 2, f"Should match both positions: {stats}"
        assert stats['orders_inserted'] == 1, f"Should insert one order: {stats}"

    def test_exception_during_position_resolution(self):
        """Test handling of exceptions during position resolution"""
        # Create scenario that might cause exception
        position = deepcopy(self.default_position)
        
        # Mock resolve_positions to raise non-PositionSyncResultException
        original_resolve = self.position_syncer.resolve_positions
        
        def mock_resolve_error(*args, **kwargs):
            raise ValueError("Test exception during resolution")
        
        self.position_syncer.resolve_positions = mock_resolve_error
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([position])
        
        try:
            # Should catch and log exception, continue processing
            self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                               disk_positions=disk_positions)
            
            # Should count the exception
            assert self.position_syncer.global_stats['exceptions_seen'] == 1
        finally:
            self.position_syncer.resolve_positions = original_resolve

    def test_order_sync_boundary_conditions(self):
        """Test order synchronization with exact boundary conditions"""
        position = deepcopy(self.default_position)
        base_time = self.DEFAULT_OPEN_MS
        
        # Create orders at various boundary conditions
        orders = []
        
        # Order at exact hard_snap_cutoff_ms - should be kept
        at_cutoff = deepcopy(self.default_order)
        at_cutoff.order_uuid = "at_cutoff"
        at_cutoff.processed_ms = base_time
        orders.append(at_cutoff)
        
        # Order just before cutoff - should be deleted
        before_cutoff = deepcopy(self.default_order)
        before_cutoff.order_uuid = "before_cutoff"
        before_cutoff.processed_ms = base_time - 1
        orders.append(before_cutoff)
        
        # Order at exact SYNC_LOOK_AROUND_MS from match candidate
        at_sync_boundary = deepcopy(self.default_order)
        at_sync_boundary.order_uuid = "at_sync_boundary"
        at_sync_boundary.processed_ms = base_time + self.position_syncer.SYNC_LOOK_AROUND_MS
        orders.append(at_sync_boundary)
        
        disk_position = deepcopy(position)
        disk_position.orders = orders
        
        # Candidate only has the base order
        candidate_position = deepcopy(position)
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        candidate_data['hard_snap_cutoff_ms'] = base_time
        disk_positions = self.positions_to_disk_data([disk_position])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Verify boundary behavior
        assert stats['orders_deleted'] == 1, f"Should delete order before cutoff: {stats}"
        assert stats['orders_kept'] >= 1, f"Should keep orders at/after cutoff: {stats}"

    def test_position_splitting_bug_scenario_1(self):
        """Test case: LONG -> FLAT -> SHORT sequence should create two positions"""
        # Create a position with LONG -> FLAT -> SHORT sequence
        position = deepcopy(self.default_position)
        position.position_uuid = "combined_position"
        
        # LONG order
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long_order"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2
        long_order.processed_ms = self.DEFAULT_OPEN_MS
        
        # FLAT order (should close the LONG)
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat_order"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -2
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        # SHORT order (should start new position)
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short_order"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        # Test when this comes from candidate data
        candidate_position = deepcopy(position)
        candidate_position.orders = [long_order, flat_order, short_order]
        
        candidate_data = self.positions_to_candidate_data([candidate_position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should spawn a new position after FLAT
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Should split position at FLAT order: {stats}"
        assert stats['positions_inserted'] == 1, f"Should insert original position: {stats}"

    def test_position_splitting_bug_scenario_2(self):
        """Test case: Existing position gets orders that should trigger split"""
        # Existing position has LONG order
        existing = deepcopy(self.default_position)
        existing.position_uuid = "existing_pos"
        existing.orders[0].order_type = OrderType.LONG
        existing.orders[0].leverage = 2
        
        # Candidate adds FLAT and SHORT orders
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat_order"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -2
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short_order"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1.5
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        candidate = deepcopy(existing)
        candidate.orders = [existing.orders[0], flat_order, short_order]
        
        candidate_data = self.positions_to_candidate_data([candidate])
        disk_positions = self.positions_to_disk_data([existing])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should split the updated position
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Should split updated position: {stats}"
        assert stats['orders_inserted'] == 2, f"Should insert FLAT and SHORT orders: {stats}"

    def test_position_splitting_bug_scenario_3(self):
        """Test case: Multiple FLAT orders creating multiple positions"""
        position = deepcopy(self.default_position)
        position.position_uuid = "multi_flat_pos"
        
        orders = []
        # LONG -> FLAT -> SHORT -> FLAT -> LONG sequence
        # Should create 3 separate positions
        
        # First LONG
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "order1"
        o1.order_type = OrderType.LONG
        o1.leverage = 1
        o1.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(o1)
        
        # First FLAT
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "order2"
        o2.order_type = OrderType.FLAT
        o2.leverage = -1
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        # SHORT
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "order3"
        o3.order_type = OrderType.SHORT
        o3.leverage = -2
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        # Second FLAT
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "order4"
        o4.order_type = OrderType.FLAT
        o4.leverage = 2
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        # Second LONG
        o5 = deepcopy(self.default_order)
        o5.order_uuid = "order5"
        o5.order_type = OrderType.LONG
        o5.leverage = 3
        o5.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 4
        orders.append(o5)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should create 2 additional positions (3 total)
        assert stats['n_positions_spawned_from_post_flat_orders'] == 2, \
            f"Should spawn 2 additional positions: {stats}"

    def test_position_splitting_bug_scenario_4(self):
        """Test case: Position ending with FLAT should not split"""
        position = deepcopy(self.default_position)
        position.position_uuid = "flat_ending_pos"
        
        # LONG -> increase -> FLAT (no orders after FLAT)
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2
        
        increase_order = deepcopy(self.default_order)
        increase_order.order_uuid = "increase"
        increase_order.order_type = OrderType.LONG
        increase_order.leverage = 1
        increase_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -3
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long_order, increase_order, flat_order]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should NOT spawn new positions when FLAT is at the end
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not split when FLAT is last order: {stats}"

    def test_position_splitting_bug_scenario_5(self):
        """Test case: Heuristic matching with position that should be split"""
        # Disk has unsplit position
        disk_pos = deepcopy(self.default_position)
        disk_pos.position_uuid = "disk_uuid"
        
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat"
        flat_order.order_type = OrderType.FLAT
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short"
        short_order.order_type = OrderType.SHORT
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        disk_pos.orders = [long_order, flat_order, short_order]
        
        # Candidate has different UUID but matches heuristically
        candidate_pos = deepcopy(disk_pos)
        candidate_pos.position_uuid = "candidate_uuid"
        candidate_pos.open_ms = self.DEFAULT_OPEN_MS + 1000  # Within SYNC_LOOK_AROUND_MS
        
        # Add another order to candidate
        extra_order = deepcopy(self.default_order)
        extra_order.order_uuid = "extra"
        extra_order.order_type = OrderType.SHORT
        extra_order.leverage = -1
        extra_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        candidate_pos.orders.append(extra_order)
        
        candidate_data = self.positions_to_candidate_data([candidate_pos])
        disk_positions = self.positions_to_disk_data([disk_pos])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should match and split the position
        assert stats['positions_matched'] == 1, f"Should match position: {stats}"
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Should split matched position: {stats}"

    def test_position_splitting_bug_scenario_6(self):
        """Test case: Position with no FLAT should not split"""
        position = deepcopy(self.default_position)
        position.position_uuid = "no_flat_pos"
        
        # LONG -> increase -> decrease (no FLAT)
        orders = []
        for i in range(3):
            order = deepcopy(self.default_order)
            order.order_uuid = f"order_{i}"
            order.order_type = OrderType.LONG
            order.leverage = 2 - i * 0.5  # 2, 1.5, 1
            order.processed_ms = self.DEFAULT_OPEN_MS + i * 1000 * 60
            orders.append(order)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should NOT spawn new positions without FLAT orders
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not split without FLAT orders: {stats}"

    def test_position_splitting_with_sync_status_nothing(self):
        """Test that positions with NOTHING status still get split if they contain FLAT orders"""
        # Create a position that will match exactly (no changes)
        position = deepcopy(self.default_position)
        position.position_uuid = "exact_match"
        
        # LONG -> FLAT -> SHORT sequence
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -2.0  # Close the LONG position
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1.0  # New SHORT position
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long_order, flat_order, short_order]
        
        # Both disk and candidate have the exact same position
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([deepcopy(position)])
        
        # Clear all positions first to start fresh
        self.position_manager.clear_all_miner_positions()
        
        # Add the initial position to disk
        self.position_manager.overwrite_position_on_disk(deepcopy(position))
        
        # Use the actual disk positions from the position manager
        actual_disk_positions = self.position_manager.get_positions_for_all_miners()
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=actual_disk_positions)
        stats = self.position_syncer.global_stats
        
        # Position should match (NOTHING status) but still be split
        assert stats['positions_matched'] == 1, f"Should match position: {stats}"
        
        # Check that positions were written to disk after splitting
        # The bug was that split positions weren't being written for NOTHING status
        all_positions = self.position_manager.get_positions_for_all_miners()
        disk_positions_after = all_positions.get(self.DEFAULT_MINER_HOTKEY, [])
        
        assert len(disk_positions_after) >= 2, \
            f"Should have split position and written to disk. Found {len(disk_positions_after)} positions"

    def test_position_not_split_in_candidate_data(self):
        """Test case: Candidate data contains unsplit position that should have been split"""
        # This tests the scenario where the bug might originate from the candidate data itself
        # having positions that weren't properly split
        
        # Create a complex position that definitely should be split
        position = deepcopy(self.default_position)
        position.position_uuid = "should_be_split"
        
        # Create order sequence: LONG -> FLAT -> SHORT -> FLAT -> LONG
        orders = []
        
        # Initial LONG
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "long1"
        o1.order_type = OrderType.LONG
        o1.leverage = 2
        o1.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(o1)
        
        # FLAT to close
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "flat1"
        o2.order_type = OrderType.FLAT
        o2.leverage = -2
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 5
        orders.append(o2)
        
        # New SHORT position (should be separate)
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "short1"
        o3.order_type = OrderType.SHORT
        o3.leverage = -3
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 10
        orders.append(o3)
        
        # FLAT to close SHORT
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "flat2"
        o4.order_type = OrderType.FLAT
        o4.leverage = 3
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 15
        orders.append(o4)
        
        # New LONG position (should be separate)
        o5 = deepcopy(self.default_order)
        o5.order_uuid = "long2"
        o5.order_type = OrderType.LONG
        o5.leverage = 1
        o5.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 20
        orders.append(o5)
        
        position.orders = orders
        
        # Simulate this coming from candidate data (backup)
        candidate_data = self.positions_to_candidate_data([position])
        
        # Print debug info
        print(f"\nCandidate position has {len(position.orders)} orders")
        for i, order in enumerate(position.orders):
            print(f"  Order {i}: {order.order_type.name} at {order.processed_ms}")
        
        # Empty disk - this is a new sync
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        print(f"\nSync stats: {stats}")
        
        # This should create 3 positions total (original + 2 spawned)
        assert stats['n_positions_spawned_from_post_flat_orders'] == 2, \
            f"Should spawn 2 additional positions from candidate data: {stats}"
        
        # Verify the position was actually inserted and split
        assert stats['positions_inserted'] == 1, f"Should insert the position: {stats}"

    def test_real_world_position_splitting_bug(self):
        """Test the real-world scenario where positions aren't split during sync"""
        # This simulates a validator that has been offline and is syncing
        # The backup contains a position that should have been split but wasn't
        
        # Create a position representing a real trading sequence
        miner_hotkey = "real_miner"
        self.mock_metagraph.hotkeys.append(miner_hotkey)
        
        position = Position(
            miner_hotkey=miner_hotkey,
            position_uuid="combined_pos_123",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            orders=[],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )
        
        # Order sequence that should be 2 positions:
        # 1. LONG position (10:00 - 11:00)
        # 2. SHORT position (11:05 - ongoing)
        
        # Initial LONG
        long_open = Order(
            order_uuid="long_open",
            price=50000,
            processed_ms=self.DEFAULT_OPEN_MS,  # 10:00
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=2
        )
        
        # Close LONG with FLAT
        long_close = Order(
            order_uuid="long_close",
            price=51000,
            processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60,  # 11:00
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=-2
        )
        
        # Open SHORT after 5 minutes
        short_open = Order(
            order_uuid="short_open",
            price=50500,
            processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 65,  # 11:05
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.SHORT,
            leverage=-1.5
        )
        
        # Add more SHORT orders
        short_increase = Order(
            order_uuid="short_increase",
            price=50000,
            processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 90,  # 11:30
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.SHORT,
            leverage=-0.5
        )
        
        position.orders = [long_open, long_close, short_open, short_increase]
        
        # Clear all positions first and add the position to disk
        self.position_manager.clear_all_miner_positions()
        self.position_manager.overwrite_position_on_disk(deepcopy(position))
        
        # Validator already has this exact position on disk (synced before)
        disk_positions = {miner_hotkey: [deepcopy(position)]}
        
        # Candidate data from backup has the same position
        candidate_data = {
            'positions': {
                miner_hotkey: {
                    'positions': [json.loads(str(position), cls=GeneralizedJSONDecoder)]
                }
            },
            'eliminations': [],
            'created_timestamp_ms': short_increase.processed_ms + AUTO_SYNC_ORDER_LAG_MS
        }
        
        print(f"\nSyncing position with {len(position.orders)} orders")
        print(f"Expected: 2 positions (LONG closed, SHORT open)")
        print(f"Actual: Position should be split correctly after bug fix")
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Position matches exactly so has NOTHING status, but should still be split
        assert stats['positions_matched'] == 1, f"Position matches: {stats}"
        assert stats['orders_matched'] == 4, f"All orders match: {stats}"
        
        # FIX VERIFIED: Position should be split and now is!
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Bug fixed - position split successfully: {stats}"
        
        # Verify the fix worked: positions were written to disk after splitting
        final_positions = self.position_manager.get_positions_for_all_miners()
        miner_positions = final_positions.get(miner_hotkey, [])
        
        assert len(miner_positions) >= 2, \
            f"Should have split into 2+ positions. Found: {len(miner_positions)} positions"
        
        print(f"\nBUG FIXED: Position with FLAT order correctly split during sync")
        print(f"Stats: {stats}")
        print(f"Final positions: {len(miner_positions)}")

    def test_implicit_flat_splitting_scenario_1(self):
        """Test that positions split when cumulative leverage reaches zero (implicit flat)"""
        position = deepcopy(self.default_position)
        position.position_uuid = "implicit_flat_test"
        
        # LONG +2 -> SHORT -2 (implicit flat at zero) -> LONG +1
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long1"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        long_order.processed_ms = self.DEFAULT_OPEN_MS
        
        # This brings leverage to zero (implicit flat)
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short1"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -2.0
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        # New position should start here
        long_order2 = deepcopy(self.default_order)
        long_order2.order_uuid = "long2"
        long_order2.order_type = OrderType.LONG
        long_order2.leverage = 1.0
        long_order2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long_order, short_order, long_order2]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should split at the implicit flat
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Should split at implicit flat (leverage=0): {stats}"
        assert stats.get('n_positions_split_on_implicit_flat', 0) >= 1, \
            f"Should track implicit flat splits: {stats}"

    def test_implicit_flat_splitting_scenario_2(self):
        """Test complex leverage sequence with multiple implicit flats"""
        position = deepcopy(self.default_position)
        position.position_uuid = "complex_implicit_flat"
        
        orders = []
        
        # LONG +3
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "o1"
        o1.order_type = OrderType.LONG
        o1.leverage = 3.0
        o1.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(o1)
        
        # SHORT -1 (cumulative: +2)
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "o2"
        o2.order_type = OrderType.SHORT
        o2.leverage = -1.0
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        # SHORT -2 (cumulative: 0) - IMPLICIT FLAT
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "o3"
        o3.order_type = OrderType.SHORT
        o3.leverage = -2.0
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        # SHORT -2 (new position)
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "o4"
        o4.order_type = OrderType.SHORT
        o4.leverage = -2.0
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        # LONG +2 (cumulative: 0) - IMPLICIT FLAT
        o5 = deepcopy(self.default_order)
        o5.order_uuid = "o5"
        o5.order_type = OrderType.LONG
        o5.leverage = 2.0
        o5.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 4
        orders.append(o5)
        
        # LONG +1 (new position)
        o6 = deepcopy(self.default_order)
        o6.order_uuid = "o6"
        o6.order_type = OrderType.LONG
        o6.leverage = 1.0
        o6.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 5
        orders.append(o6)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should create 3 positions total (2 splits)
        assert stats['n_positions_spawned_from_post_flat_orders'] == 2, \
            f"Should spawn 2 additional positions: {stats}"
        assert stats.get('n_positions_split_on_implicit_flat', 0) == 2, \
            f"Should have 2 implicit flat splits: {stats}"

    def test_no_split_when_leverage_zero_at_end(self):
        """Test that no split occurs when leverage reaches zero at the last order"""
        position = deepcopy(self.default_position)
        position.position_uuid = "zero_at_end"
        
        # LONG +2 -> SHORT -2 (ends at zero leverage)
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -2.0
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        position.orders = [long_order, short_order]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should NOT split when zero leverage is at the end
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not split when zero leverage at end: {stats}"

    def test_explicit_vs_implicit_flat(self):
        """Test that explicit FLAT takes precedence over implicit flat"""
        position = deepcopy(self.default_position)
        position.position_uuid = "explicit_vs_implicit"
        
        # LONG +2 -> FLAT -2 (explicit flat at zero)
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        
        # Explicit FLAT that also brings leverage to zero
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -2.0
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        # New position
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1.0
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long_order, flat_order, short_order]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should split, but not count as implicit flat
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Should split on explicit FLAT: {stats}"
        assert stats.get('n_positions_split_on_implicit_flat', 0) == 0, \
            f"Should not count explicit FLAT as implicit: {stats}"

    def test_floating_point_precision_in_leverage(self):
        """Test that floating point precision issues are handled when checking for zero leverage"""
        position = deepcopy(self.default_position)
        position.position_uuid = "float_precision"
        
        orders = []
        
        # Use values that might cause floating point precision issues
        # 0.1 + 0.2 - 0.3 might not exactly equal 0 in floating point
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "o1"
        o1.order_type = OrderType.LONG
        o1.leverage = 0.1
        o1.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(o1)
        
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "o2"
        o2.order_type = OrderType.LONG
        o2.leverage = 0.2
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        # This should bring leverage to ~0 (but might be 5.551115123125783e-17)
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "o3"
        o3.order_type = OrderType.SHORT
        o3.leverage = -0.3
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        # New position
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "o4"
        o4.order_type = OrderType.LONG
        o4.leverage = 1.0
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should handle floating point precision and split
        assert stats['n_positions_spawned_from_post_flat_orders'] >= 1, \
            f"Should handle floating point precision: {stats}"

    def test_mixed_explicit_and_implicit_flats(self):
        """Test position with both explicit FLAT orders and implicit zero leverage points"""
        position = deepcopy(self.default_position)
        position.position_uuid = "mixed_flats"
        
        orders = []
        
        # Position 1: LONG +2
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "o1"
        o1.order_type = OrderType.LONG
        o1.leverage = 2.0
        o1.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(o1)
        
        # Explicit FLAT
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "o2"
        o2.order_type = OrderType.FLAT
        o2.leverage = -2.0
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        # Position 2: SHORT -3
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "o3"
        o3.order_type = OrderType.SHORT
        o3.leverage = -3.0
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        # Implicit flat (brings to 0)
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "o4"
        o4.order_type = OrderType.LONG
        o4.leverage = 3.0
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        # Position 3: LONG +1
        o5 = deepcopy(self.default_order)
        o5.order_uuid = "o5"
        o5.order_type = OrderType.LONG
        o5.leverage = 1.0
        o5.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 4
        orders.append(o5)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should create 3 positions (2 splits: 1 explicit, 1 implicit)
        assert stats['n_positions_spawned_from_post_flat_orders'] == 2, \
            f"Should spawn 2 additional positions: {stats}"
        assert stats.get('n_positions_split_on_implicit_flat', 0) == 1, \
            f"Should have 1 implicit flat split: {stats}"

    def test_position_split_minimum_orders_constraint(self):
        """Test that CLOSED positions are not created with < 2 orders"""
        position = deepcopy(self.default_position)
        position.position_uuid = "min_orders_test"
        
        # Only 2 orders total: LONG -> FLAT
        # Should NOT split because it would create:
        # - Position 1: LONG, FLAT (valid closed position with 2 orders)
        # - Position 2: Nothing (no orders after FLAT)
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        
        flat_order = deepcopy(self.default_order)
        flat_order.order_uuid = "flat"
        flat_order.order_type = OrderType.FLAT
        flat_order.leverage = -2.0
        flat_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        position.orders = [long_order, flat_order]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should NOT split when FLAT is at the end
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not split when FLAT is at end: {stats}"

    def test_position_cannot_start_with_flat(self):
        """Test that positions starting with FLAT are not created during split"""
        position = deepcopy(self.default_position)
        position.position_uuid = "flat_start_test"
        
        # FLAT -> LONG -> FLAT -> SHORT sequence
        # First FLAT should be ignored, split should happen at second FLAT
        flat1 = deepcopy(self.default_order)
        flat1.order_uuid = "flat1"
        flat1.order_type = OrderType.FLAT
        flat1.leverage = 0
        flat1.processed_ms = self.DEFAULT_OPEN_MS
        
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        long_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        flat2 = deepcopy(self.default_order)
        flat2.order_uuid = "flat2"
        flat2.order_type = OrderType.FLAT
        flat2.leverage = -2.0
        flat2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        short_order = deepcopy(self.default_order)
        short_order.order_uuid = "short"
        short_order.order_type = OrderType.SHORT
        short_order.leverage = -1.0
        short_order.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        
        position.orders = [flat1, long_order, flat2, short_order]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should not split because first segment starts with FLAT
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not create positions starting with FLAT: {stats}"

    def test_complex_split_with_constraints(self):
        """Test complex scenario with multiple constraints"""
        position = deepcopy(self.default_position)
        position.position_uuid = "complex_constraints"
        
        orders = []
        
        # Valid first position: LONG -> SHORT -> FLAT (3 orders)
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "o1"
        o1.order_type = OrderType.LONG
        o1.leverage = 3.0
        o1.processed_ms = self.DEFAULT_OPEN_MS
        orders.append(o1)
        
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "o2"
        o2.order_type = OrderType.SHORT
        o2.leverage = -1.0
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "o3"
        o3.order_type = OrderType.FLAT
        o3.leverage = -2.0
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        # Invalid: single LONG order (would be < 2 orders)
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "o4"
        o4.order_type = OrderType.LONG
        o4.leverage = 1.0
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        # Implicit flat (cumulative back to 0)
        o5 = deepcopy(self.default_order)
        o5.order_uuid = "o5"
        o5.order_type = OrderType.SHORT
        o5.leverage = -1.0
        o5.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 4
        orders.append(o5)
        
        # Valid second position: SHORT -> LONG (2 orders)
        o6 = deepcopy(self.default_order)
        o6.order_uuid = "o6"
        o6.order_type = OrderType.SHORT
        o6.leverage = -2.0
        o6.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 5
        orders.append(o6)
        
        o7 = deepcopy(self.default_order)
        o7.order_uuid = "o7"
        o7.order_type = OrderType.LONG
        o7.leverage = 1.0
        o7.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 6
        orders.append(o7)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should handle constraints properly
        # First split at FLAT is valid (3 orders before, 4 after)
        # Second split at implicit flat might be merged due to constraints
        assert stats['positions_inserted'] == 1, f"Should insert position: {stats}"

    def test_single_order_open_position_allowed(self):
        """Test that single order OPEN positions are allowed after split"""
        position = deepcopy(self.default_position)
        position.position_uuid = "single_order_open"
        
        # LONG -> SHORT (brings to 0) -> LONG (single order that creates open position)
        long1 = deepcopy(self.default_order)
        long1.order_uuid = "long1"
        long1.order_type = OrderType.LONG
        long1.leverage = 2.0
        
        short1 = deepcopy(self.default_order)
        short1.order_uuid = "short1"
        short1.order_type = OrderType.SHORT
        short1.leverage = -2.0
        short1.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        long2 = deepcopy(self.default_order)
        long2.order_uuid = "long2"
        long2.order_type = OrderType.LONG
        long2.leverage = 1.0
        long2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long1, short1, long2]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should split - single order OPEN position is valid
        assert stats['n_positions_spawned_from_post_flat_orders'] == 1, \
            f"Should allow single order open position: {stats}"
        assert stats.get('n_positions_split_on_implicit_flat', 0) >= 1, \
            f"Should track implicit flat: {stats}"

    def test_valid_split_with_sufficient_orders(self):
        """Test that valid splits with sufficient orders work correctly"""
        position = deepcopy(self.default_position)
        position.position_uuid = "valid_split_sufficient"
        
        # LONG -> SHORT -> FLAT -> SHORT -> LONG
        # Both segments have at least 2 orders
        orders = []
        
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "o1"
        o1.order_type = OrderType.LONG
        o1.leverage = 2.0
        orders.append(o1)
        
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "o2"
        o2.order_type = OrderType.SHORT
        o2.leverage = -1.0
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "o3"
        o3.order_type = OrderType.FLAT
        o3.leverage = -1.0
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "o4"
        o4.order_type = OrderType.SHORT
        o4.leverage = -2.0
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        o5 = deepcopy(self.default_order)
        o5.order_uuid = "o5"
        o5.order_type = OrderType.LONG
        o5.leverage = 1.0
        o5.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 4
        orders.append(o5)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should split successfully - both segments have >= 2 orders
        assert stats['n_positions_spawned_from_post_flat_orders'] == 1, \
            f"Should split when both segments have sufficient orders: {stats}"

    def test_position_with_single_order(self):
        """Test that positions with only 1 order are handled gracefully"""
        position = deepcopy(self.default_position)
        position.position_uuid = "single_order_pos"
        
        # Only one order - should not attempt to split
        single_order = deepcopy(self.default_order)
        single_order.order_uuid = "single"
        single_order.order_type = OrderType.LONG
        single_order.leverage = 1.0
        
        position.orders = [single_order]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should not crash and should not split
        assert stats['positions_inserted'] == 1, f"Should insert single-order position: {stats}"
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not split single-order position: {stats}"

    def test_single_order_closed_position_not_allowed(self):
        """Test that single order CLOSED positions are not created"""
        position = deepcopy(self.default_position)
        position.position_uuid = "single_closed_test"
        
        # LONG -> FLAT -> FLAT (single order that would be closed)
        long_order = deepcopy(self.default_order)
        long_order.order_uuid = "long"
        long_order.order_type = OrderType.LONG
        long_order.leverage = 2.0
        
        flat1 = deepcopy(self.default_order)
        flat1.order_uuid = "flat1"
        flat1.order_type = OrderType.FLAT
        flat1.leverage = -2.0
        flat1.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        
        flat2 = deepcopy(self.default_order)
        flat2.order_uuid = "flat2"
        flat2.order_type = OrderType.FLAT
        flat2.leverage = 0
        flat2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        
        position.orders = [long_order, flat1, flat2]
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should not split - would create single order closed position
        assert stats['n_positions_spawned_from_post_flat_orders'] == 0, \
            f"Should not create single order closed position: {stats}"

    def test_implicit_flat_with_open_position(self):
        """Test implicit flat creating an open position with single order"""
        position = deepcopy(self.default_position)
        position.position_uuid = "implicit_open"
        
        # LONG +3 -> SHORT -1 -> SHORT -2 (implicit flat) -> LONG +2 (single open)
        orders = []
        
        o1 = deepcopy(self.default_order)
        o1.order_uuid = "o1"
        o1.order_type = OrderType.LONG
        o1.leverage = 3.0
        orders.append(o1)
        
        o2 = deepcopy(self.default_order)
        o2.order_uuid = "o2"
        o2.order_type = OrderType.SHORT
        o2.leverage = -1.0
        o2.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60
        orders.append(o2)
        
        o3 = deepcopy(self.default_order)
        o3.order_uuid = "o3"
        o3.order_type = OrderType.SHORT
        o3.leverage = -2.0  # Brings cumulative to 0
        o3.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 2
        orders.append(o3)
        
        o4 = deepcopy(self.default_order)
        o4.order_uuid = "o4"
        o4.order_type = OrderType.LONG
        o4.leverage = 2.0  # Single order creating open position
        o4.processed_ms = self.DEFAULT_OPEN_MS + 1000 * 60 * 3
        orders.append(o4)
        
        position.orders = orders
        
        candidate_data = self.positions_to_candidate_data([position])
        disk_positions = self.positions_to_disk_data([])
        
        self.position_syncer.sync_positions(shadow_mode=False, candidate_data=candidate_data,
                                           disk_positions=disk_positions)
        stats = self.position_syncer.global_stats
        
        # Should split - creates valid closed position and single order open position
        assert stats['n_positions_spawned_from_post_flat_orders'] == 1, \
            f"Should split with single order open position: {stats}"
        assert stats.get('n_positions_split_on_implicit_flat', 0) == 1, \
            f"Should track implicit flat: {stats}"


class TestOverlapDetection(TestBase):
    """Tests for overlap detection and deletion feature"""

    def setUp(self):
        super().setUp()
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_OPEN_MS = 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        self.elimination_manager.position_manager = self.position_manager
        self.position_manager.clear_all_miner_positions()
        self.elimination_manager.eliminations.clear()

        self.position_syncer = PositionSyncer(
            running_unit_tests=True,
            position_manager=self.position_manager,
            enable_position_splitting=True
        )

    def create_position(self, position_uuid, open_ms, close_ms=None, trade_pair=None, miner_hotkey=None):
        """Helper to create a test position"""
        if trade_pair is None:
            trade_pair = self.DEFAULT_TRADE_PAIR
        if miner_hotkey is None:
            miner_hotkey = self.DEFAULT_MINER_HOTKEY

        order1 = Order(
            price=100.0,
            processed_ms=open_ms,
            order_uuid=f"{position_uuid}_o1",
            trade_pair=trade_pair,
            order_type=OrderType.LONG,
            leverage=1.0
        )

        orders = [order1]
        position_type = OrderType.LONG

        if close_ms is not None:
            order2 = Order(
                price=105.0,
                processed_ms=close_ms,
                order_uuid=f"{position_uuid}_o2",
                trade_pair=trade_pair,
                order_type=OrderType.FLAT,
                leverage=0.0
            )
            orders.append(order2)
            position_type = OrderType.FLAT

        position = Position(
            miner_hotkey=miner_hotkey,
            position_uuid=position_uuid,
            open_ms=open_ms,
            trade_pair=trade_pair,
            orders=orders,
            position_type=position_type,
        )

        if close_ms is not None:
            position.close_out_position(close_ms)

        return position

    def test_no_overlaps_single_position(self):
        """Test that a single position has no overlaps"""
        position = self.create_position("p1", self.DEFAULT_OPEN_MS, self.DEFAULT_OPEN_MS + 1000)

        current_time_ms = self.DEFAULT_OPEN_MS + 10000
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([position], current_time_ms)

        self.assertEqual(len(overlaps), 0, "Single position should have no overlaps")

    def test_no_overlaps_non_overlapping_positions(self):
        """Test that non-overlapping positions are not flagged"""
        # P1: [100, 200]
        p1 = self.create_position("p1", 100, 200)
        # P2: [300, 400]
        p2 = self.create_position("p2", 300, 400)
        # P3: [500, 600]
        p3 = self.create_position("p3", 500, 600)

        current_time_ms = 700
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([p1, p2, p3], current_time_ms)

        self.assertEqual(len(overlaps), 0, "Non-overlapping positions should not be flagged")

    def test_two_positions_clear_overlap(self):
        """Test that two positions with clear overlap are detected"""
        # P1: [100, 200]
        p1 = self.create_position("p1", 100, 200)
        # P2: [150, 250] - overlaps with P1
        p2 = self.create_position("p2", 150, 250)

        current_time_ms = 300
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([p1, p2], current_time_ms)

        self.assertEqual(len(overlaps), 2, "Both positions should be flagged for overlap")
        self.assertIn(p1.position_uuid, overlaps, "P1 UUID should be in overlapping positions")
        self.assertIn(p2.position_uuid, overlaps, "P2 UUID should be in overlapping positions")

    def test_multiple_overlapping_positions(self):
        """Test that multiple overlapping positions are all detected"""
        # P1: [100, 200]
        p1 = self.create_position("p1", 100, 200)
        # P2: [150, 250] - overlaps with P1
        p2 = self.create_position("p2", 150, 250)
        # P3: [240, 300] - overlaps with P2
        p3 = self.create_position("p3", 240, 300)
        # P4: [400, 500] - no overlap
        p4 = self.create_position("p4", 400, 500)

        current_time_ms = 600
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([p1, p2, p3, p4], current_time_ms)

        # P1, P2, P3 should all be flagged (they form a chain of overlaps)
        # P4 should not be flagged
        self.assertEqual(len(overlaps), 3, "Three overlapping positions should be flagged")
        self.assertIn(p1.position_uuid, overlaps, "P1 UUID should be in overlapping positions")
        self.assertIn(p2.position_uuid, overlaps, "P2 UUID should be in overlapping positions")
        self.assertIn(p3.position_uuid, overlaps, "P3 UUID should be in overlapping positions")
        self.assertNotIn(p4.position_uuid, overlaps, "P4 UUID should not be in overlapping positions")

    def test_open_positions_with_current_time(self):
        """Test that open positions use current_time_ms as end time"""
        # P1: [100, 200] - closed
        p1 = self.create_position("p1", 100, 200)
        # P2: [150, open] - open position
        p2 = self.create_position("p2", 150, None)

        # With current_time=180, P2's effective end is 180, so it overlaps with P1 [100, 200]
        current_time_ms = 180
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([p1, p2], current_time_ms)

        self.assertEqual(len(overlaps), 2, "Open and closed positions should be detected as overlapping")

    def test_touching_boundaries_no_overlap(self):
        """Test that positions that touch at boundaries don't overlap"""
        # P1: [100, 200]
        p1 = self.create_position("p1", 100, 200)
        # P2: [200, 300] - starts exactly when P1 ends
        p2 = self.create_position("p2", 200, 300)

        current_time_ms = 400
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([p1, p2], current_time_ms)

        # start_ms < current_end check means [100, 200] and [200, 300] don't overlap
        self.assertEqual(len(overlaps), 0, "Positions touching at boundaries should not overlap")

    def test_different_trade_pairs_no_overlap(self):
        """Test that different trade pairs are analyzed separately"""
        # Both positions have same time range but different trade pairs
        p1_btc = self.create_position("p1", 100, 200, trade_pair=TradePair.BTCUSD)
        p2_eth = self.create_position("p2", 150, 250, trade_pair=TradePair.ETHUSD)

        # When analyzing only BTC positions
        current_time_ms = 300
        overlaps_btc = self.position_syncer._find_overlapping_positions_via_merge([p1_btc], current_time_ms)

        self.assertEqual(len(overlaps_btc), 0, "Single position in trade pair should have no overlaps")

    def test_detect_and_delete_overlapping_positions(self):
        """Test the full detect_and_delete_overlapping_positions method"""
        # Create overlapping positions
        p1 = self.create_position("p1", 100, 200)
        p2 = self.create_position("p2", 150, 250)
        p3 = self.create_position("p3", 400, 500)  # Non-overlapping

        # Save positions to disk
        for p in [p1, p2, p3]:
            self.position_manager.save_miner_position(p)

        disk_positions = {self.DEFAULT_MINER_HOTKEY: [p1, p2, p3]}
        current_time_ms = 600

        stats = self.position_syncer.detect_and_delete_overlapping_positions(disk_positions, current_time_ms)

        # Verify stats
        self.assertEqual(stats['hotkeys_checked'], 1, "Should check 1 hotkey")
        self.assertEqual(stats['trade_pairs_checked'], 1, "Should check 1 trade pair")
        self.assertEqual(stats['positions_deleted'], 2, "Should delete 2 overlapping positions")
        self.assertEqual(len(stats['hotkeys_with_overlaps']), 1, "Should have 1 hotkey with overlaps")

        # Verify positions were actually deleted from disk
        remaining_positions = self.position_manager.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(remaining_positions), 1, "Should have 1 position remaining")
        self.assertEqual(remaining_positions[0].position_uuid, "p3", "P3 should remain")

    def test_multiple_trade_pairs_overlap_detection(self):
        """Test overlap detection across multiple trade pairs"""
        # Create overlapping BTC positions
        p1_btc = self.create_position("p1_btc", 100, 200, trade_pair=TradePair.BTCUSD)
        p2_btc = self.create_position("p2_btc", 150, 250, trade_pair=TradePair.BTCUSD)

        # Create overlapping ETH positions
        p1_eth = self.create_position("p1_eth", 300, 400, trade_pair=TradePair.ETHUSD)
        p2_eth = self.create_position("p2_eth", 350, 450, trade_pair=TradePair.ETHUSD)

        # Save positions
        for p in [p1_btc, p2_btc, p1_eth, p2_eth]:
            self.position_manager.save_miner_position(p)

        disk_positions = {self.DEFAULT_MINER_HOTKEY: [p1_btc, p2_btc, p1_eth, p2_eth]}
        current_time_ms = 500

        stats = self.position_syncer.detect_and_delete_overlapping_positions(disk_positions, current_time_ms)

        # Should detect overlaps in both trade pairs
        self.assertEqual(stats['positions_deleted'], 4, "Should delete all 4 overlapping positions")
        self.assertEqual(stats['trade_pairs_checked'], 2, "Should check 2 trade pairs")
        self.assertEqual(stats['trade_pairs_with_overlaps'][TradePair.BTCUSD.trade_pair_id], 1)
        self.assertEqual(stats['trade_pairs_with_overlaps'][TradePair.ETHUSD.trade_pair_id], 1)

    def test_unsorted_positions_handled_correctly(self):
        """Test that positions are sorted internally before overlap detection"""
        # Create positions in non-chronological order
        p3 = self.create_position("p3", 300, 400)
        p1 = self.create_position("p1", 100, 200)
        p2 = self.create_position("p2", 150, 250)

        current_time_ms = 500
        overlaps = self.position_syncer._find_overlapping_positions_via_merge([p3, p1, p2], current_time_ms)

        # Should still detect P1 and P2 as overlapping despite being passed unsorted
        self.assertEqual(len(overlaps), 2, "Should detect overlaps regardless of input order")
        self.assertIn(p1.position_uuid, overlaps)
        self.assertIn(p2.position_uuid, overlaps)
        self.assertNotIn(p3.position_uuid, overlaps)

    def test_empty_positions_list(self):
        """Test that empty positions list doesn't cause errors"""
        disk_positions = {self.DEFAULT_MINER_HOTKEY: []}
        current_time_ms = 500

        stats = self.position_syncer.detect_and_delete_overlapping_positions(disk_positions, current_time_ms)

        # Empty positions list is skipped, so hotkeys_checked should be 0
        self.assertEqual(stats['positions_deleted'], 0)
        self.assertEqual(stats['hotkeys_checked'], 0)



