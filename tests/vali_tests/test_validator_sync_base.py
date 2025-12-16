"""
Comprehensive tests for ValidatorSyncBase.close_older_open_position logic.

This test file rigorously tests the duplicate open position handling during autosync,
which is critical for maintaining data integrity and preventing the production error:
ValiRecordsMisalignmentException when multiple open positions exist for the same trade pair.
"""
import uuid
from time_util.time_util import TimeUtil
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.data_sync.validator_sync_base import ValidatorSyncBase
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.vali_utils import ValiUtils


class TestValidatorSyncBase(TestBase):
    """
    Test ValidatorSyncBase.close_older_open_position logic with rigorous scenarios.

    Key scenarios tested:
    1. No duplicates - single position, nothing to close
    2. Batch duplicates - p1 and p2 both provided, different UUIDs
    3. Memory duplicate - existing_in_memory has different UUID than p1
    4. Triple duplicate - all three (existing_in_memory, p2, p1) have different UUIDs
    5. Same UUID in batch - p1 and p2 have same UUID (deduplication)
    6. Same UUID memory and batch - existing_in_memory and p1 same UUID (deduplication)
    7. Timestamps matter - verify newest by open_ms is kept
    8. Synthetic FLAT order - verify older positions get closed properly
    9. Save to disk - verify closed positions are saved correctly
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    position_client = None
    live_price_fetcher_client = None
    validator_sync = None

    TEST_HOTKEY = "test_hotkey_validator_sync_base"
    TEST_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_ACCOUNT_SIZE = 100_000

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator."""
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')

        # Create ValidatorSyncBase instance
        cls.validator_sync = ValidatorSyncBase(
            running_unit_tests=True,
            enable_position_splitting=False
        )

    @classmethod
    def tearDownClass(cls):
        """One-time teardown: No action needed (orchestrator manages lifecycle)."""
        pass

    def setUp(self):
        """Per-test setup: Reset data state."""
        self.orchestrator.clear_all_test_data()
        # Reset global stats for each test
        self.validator_sync.init_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()
        # Reset global stats after each test
        self.validator_sync.init_data()

    def create_test_position(self, position_uuid=None, open_ms=None, is_open=True):
        """
        Helper to create a test position with configurable parameters.

        Args:
            position_uuid: UUID for the position (generates new if None)
            open_ms: Timestamp when position opened (uses current time if None)
            is_open: Whether position should be open or closed

        Returns:
            Position object
        """
        if position_uuid is None:
            position_uuid = str(uuid.uuid4())
        if open_ms is None:
            open_ms = TimeUtil.now_in_millis()

        # Create initial LONG order
        orders = [Order(
            order_uuid=str(uuid.uuid4()),
            order_type=OrderType.LONG,
            leverage=0.025,
            price=50000.0,
            quote_usd_rate=1,
            usd_base_rate=1/50000.0,
            processed_ms=open_ms,
            trade_pair=self.TEST_TRADE_PAIR
        )]

        close_ms = None
        position_type = OrderType.LONG

        # Add FLAT order if position should be closed
        if not is_open:
            close_ms = open_ms + 1000 * 60 * 30  # 30 minutes later
            flat_order = Order(
                order_uuid=str(uuid.uuid4()),
                order_type=OrderType.FLAT,
                leverage=0.0,
                price=51000.0,
                quote_usd_rate=1,
                usd_base_rate=1/51000.0,
                processed_ms=close_ms,
                trade_pair=self.TEST_TRADE_PAIR
            )
            orders.append(flat_order)
            position_type = OrderType.FLAT

        position = Position(
            position_uuid=position_uuid,
            miner_hotkey=self.TEST_HOTKEY,
            open_ms=open_ms,
            close_ms=close_ms,
            trade_pair=self.TEST_TRADE_PAIR,
            orders=orders,
            position_type=position_type,
            is_closed_position=not is_open,
            account_size=self.DEFAULT_ACCOUNT_SIZE
        )

        return position

    def test_no_duplicates_single_position(self):
        """Test 1: No duplicates - single position, nothing to close."""
        print("\n" + "="*60)
        print("TEST 1: No duplicates - single position")
        print("="*60)

        p1 = self.create_test_position(open_ms=1000)

        # Call close_older_open_position with no p2
        result = self.validator_sync.close_older_open_position(p1, None)

        # Should return p1 unchanged
        self.assertEqual(result.position_uuid, p1.position_uuid)
        self.assertTrue(result.is_open_position)
        self.assertEqual(len(result.orders), 1)

        # No positions should be closed
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            0
        )

        print("✅ Single position returned unchanged, no duplicates closed")

    def test_batch_duplicates_different_uuids(self):
        """Test 2: Batch duplicates - p1 and p2 both provided, different UUIDs."""
        print("\n" + "="*60)
        print("TEST 2: Batch duplicates - p1 and p2 different UUIDs")
        print("="*60)

        # p2 is older (open_ms=1000), p1 is newer (open_ms=2000)
        p2 = self.create_test_position(open_ms=1000)
        p1 = self.create_test_position(open_ms=2000)

        print(f"p2 (older): UUID={p2.position_uuid[:8]}..., open_ms={p2.open_ms}")
        print(f"p1 (newer): UUID={p1.position_uuid[:8]}..., open_ms={p1.open_ms}")

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1, p2)

        # Should return p1 (newer)
        self.assertEqual(result.position_uuid, p1.position_uuid)
        self.assertTrue(result.is_open_position)

        # Should have closed p2 (older)
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            1
        )

        # Verify p2 was closed with synthetic FLAT order
        # (Note: p2 is modified in-place during close)
        self.assertEqual(len(p2.orders), 2)  # Original LONG + synthetic FLAT
        self.assertEqual(p2.orders[-1].order_type, OrderType.FLAT)
        self.assertTrue(p2.is_closed_position)

        print(f"✅ Newer position {p1.position_uuid[:8]}... kept")
        print(f"✅ Older position {p2.position_uuid[:8]}... closed with synthetic FLAT")

    def test_memory_duplicate_different_uuid(self):
        """Test 3: Memory duplicate - existing_in_memory has different UUID than p1."""
        print("\n" + "="*60)
        print("TEST 3: Memory duplicate - existing position in memory")
        print("="*60)

        # Create and save an older position to memory
        existing_position = self.create_test_position(open_ms=1000)
        self.position_client.save_miner_position(existing_position)

        # Create newer position to sync
        p1 = self.create_test_position(open_ms=2000)

        print(f"Existing (older): UUID={existing_position.position_uuid[:8]}..., open_ms={existing_position.open_ms}")
        print(f"p1 (newer): UUID={p1.position_uuid[:8]}..., open_ms={p1.open_ms}")

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1, None)

        # Should return p1 (newer)
        self.assertEqual(result.position_uuid, p1.position_uuid)
        self.assertTrue(result.is_open_position)

        # Should have closed existing_position (older)
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            1
        )

        print(f"✅ Newer position {p1.position_uuid[:8]}... kept")
        print(f"✅ Older position {existing_position.position_uuid[:8]}... closed")

    def test_triple_duplicate_all_different_uuids(self):
        """Test 4: Triple duplicate - all three (existing_in_memory, p2, p1) have different UUIDs."""
        print("\n" + "="*60)
        print("TEST 4: Triple duplicate - memory, p2, and p1 all different")
        print("="*60)

        # Create and save oldest position to memory
        existing_position = self.create_test_position(open_ms=1000)
        self.position_client.save_miner_position(existing_position)

        # Create middle position
        p2 = self.create_test_position(open_ms=1500)

        # Create newest position
        p1 = self.create_test_position(open_ms=2000)

        print(f"Existing (oldest): UUID={existing_position.position_uuid[:8]}..., open_ms={existing_position.open_ms}")
        print(f"p2 (middle): UUID={p2.position_uuid[:8]}..., open_ms={p2.open_ms}")
        print(f"p1 (newest): UUID={p1.position_uuid[:8]}..., open_ms={p1.open_ms}")

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1, p2)

        # Should return p1 (newest)
        self.assertEqual(result.position_uuid, p1.position_uuid)
        self.assertTrue(result.is_open_position)

        # Should have closed 2 positions (existing and p2)
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            2
        )

        print(f"✅ Newest position {p1.position_uuid[:8]}... kept")
        print(f"✅ Two older positions closed")

    def test_same_uuid_in_batch_deduplication(self):
        """Test 5: Same UUID in batch - p1 and p2 have same UUID (deduplication)."""
        print("\n" + "="*60)
        print("TEST 5: Same UUID in batch - deduplication")
        print("="*60)

        shared_uuid = str(uuid.uuid4())

        # Create positions with same UUID but different timestamps
        p2 = self.create_test_position(position_uuid=shared_uuid, open_ms=1000)
        p1 = self.create_test_position(position_uuid=shared_uuid, open_ms=1000)

        print(f"p2: UUID={p2.position_uuid[:8]}..., open_ms={p2.open_ms}")
        print(f"p1: UUID={p1.position_uuid[:8]}... (same UUID), open_ms={p1.open_ms}")

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1, p2)

        # Should return p1 (only one unique position)
        self.assertEqual(result.position_uuid, p1.position_uuid)
        self.assertTrue(result.is_open_position)

        # Should NOT have closed anything (only 1 unique position)
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            0
        )

        print(f"✅ Deduplication worked - only one position, nothing closed")

    def test_same_uuid_memory_and_batch_deduplication(self):
        """Test 6: Same UUID memory and batch - existing_in_memory and p1 same UUID (deduplication)."""
        print("\n" + "="*60)
        print("TEST 6: Same UUID in memory and batch - deduplication")
        print("="*60)

        shared_uuid = str(uuid.uuid4())

        # Create and save position to memory
        existing_position = self.create_test_position(position_uuid=shared_uuid, open_ms=1000)
        self.position_client.save_miner_position(existing_position)

        # Create p1 with same UUID
        p1 = self.create_test_position(position_uuid=shared_uuid, open_ms=1000)

        print(f"Existing: UUID={existing_position.position_uuid[:8]}..., open_ms={existing_position.open_ms}")
        print(f"p1: UUID={p1.position_uuid[:8]}... (same UUID), open_ms={p1.open_ms}")

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1, None)

        # Should return p1 (only one unique position)
        self.assertEqual(result.position_uuid, p1.position_uuid)
        self.assertTrue(result.is_open_position)

        # Should NOT have closed anything (only 1 unique position)
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            0
        )

        print(f"✅ Deduplication worked - same position in memory and batch")

    def test_timestamps_determine_which_position_kept(self):
        """Test 7: Timestamps matter - verify newest by open_ms is kept."""
        print("\n" + "="*60)
        print("TEST 7: Timestamps determine which position is kept")
        print("="*60)

        # Test keeping older timestamp when p1 is older
        print("\nSubtest 7a: p1 older than p2 - should keep p2")
        p1_old = self.create_test_position(open_ms=1000)
        p2_new = self.create_test_position(open_ms=2000)

        result = self.validator_sync.close_older_open_position(p1_old, p2_new)

        # Should return p2_new (newer)
        self.assertEqual(result.position_uuid, p2_new.position_uuid)
        print(f"✅ Kept newer position {p2_new.position_uuid[:8]}...")

        # Reset stats
        self.validator_sync.init_data()

        # Test keeping newer timestamp when p1 is newer
        print("\nSubtest 7b: p1 newer than p2 - should keep p1")
        p1_new = self.create_test_position(open_ms=2000)
        p2_old = self.create_test_position(open_ms=1000)

        result = self.validator_sync.close_older_open_position(p1_new, p2_old)

        # Should return p1_new (newer)
        self.assertEqual(result.position_uuid, p1_new.position_uuid)
        print(f"✅ Kept newer position {p1_new.position_uuid[:8]}...")

    def test_synthetic_flat_order_added(self):
        """Test 8: Synthetic FLAT order - verify older positions get closed properly."""
        print("\n" + "="*60)
        print("TEST 8: Synthetic FLAT order added to closed position")
        print("="*60)

        # Create two positions
        p2_old = self.create_test_position(open_ms=1000)
        p1_new = self.create_test_position(open_ms=2000)

        # Capture original order count
        original_p2_order_count = len(p2_old.orders)

        print(f"p2_old before close: {original_p2_order_count} orders")

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1_new, p2_old)

        # Verify p2_old was modified
        print(f"p2_old after close: {len(p2_old.orders)} orders")

        # Should have added one FLAT order
        self.assertEqual(len(p2_old.orders), original_p2_order_count + 1)

        # Last order should be FLAT
        self.assertEqual(p2_old.orders[-1].order_type, OrderType.FLAT)

        # Position should be marked as closed
        self.assertTrue(p2_old.is_closed_position)
        self.assertIsNotNone(p2_old.close_ms)

        # FLAT order timestamp should be after last original order
        self.assertGreater(
            p2_old.orders[-1].processed_ms,
            p2_old.orders[-2].processed_ms
        )

        print(f"✅ Synthetic FLAT order added at timestamp {p2_old.orders[-1].processed_ms}")
        print(f"✅ Position marked as closed at {p2_old.close_ms}")

    def test_closed_position_saved_to_disk(self):
        """Test 9: Save to disk - verify closed positions are saved correctly."""
        print("\n" + "="*60)
        print("TEST 9: Closed position saved to disk")
        print("="*60)

        # Create two positions
        p2_old = self.create_test_position(open_ms=1000)
        p1_new = self.create_test_position(open_ms=2000)

        p2_uuid = p2_old.position_uuid

        # Call close_older_open_position
        result = self.validator_sync.close_older_open_position(p1_new, p2_old)

        # Verify closed position was saved to disk
        # Get all positions from disk
        positions_on_disk = self.position_client.get_positions_for_one_hotkey(
            self.TEST_HOTKEY,
            only_open_positions=False
        )

        # Find the closed position
        closed_position_on_disk = None
        for pos in positions_on_disk:
            if pos.position_uuid == p2_uuid:
                closed_position_on_disk = pos
                break

        # Should exist on disk
        self.assertIsNotNone(closed_position_on_disk)

        # Should be marked as closed
        self.assertTrue(closed_position_on_disk.is_closed_position)

        # Should have FLAT order
        self.assertEqual(closed_position_on_disk.orders[-1].order_type, OrderType.FLAT)

        print(f"✅ Closed position {p2_uuid[:8]}... found on disk")
        print(f"✅ Position correctly marked as closed")
        print(f"✅ FLAT order preserved in disk storage")

    def test_complex_scenario_production_bug_reproduction(self):
        """
        Test 10: Complex scenario - Reproduce the production bug.

        Scenario:
        - Memory has position UUID 85c4da75... (from recent signal)
        - Autosync wants to insert position UUID 3eb40617... (from backup)
        - Without fix: ValiRecordsMisalignmentException
        - With fix: Older position closed, newer kept
        """
        print("\n" + "="*60)
        print("TEST 10: Production bug reproduction scenario")
        print("="*60)

        # Simulate production scenario
        # Memory has a newer position from recent signal processing
        memory_position_uuid = "85c4da75-aa68-452f-a670-9ac19e69da29"
        memory_position = self.create_test_position(
            position_uuid=memory_position_uuid,
            open_ms=2000  # Newer
        )
        self.position_client.save_miner_position(memory_position)

        # Autosync wants to insert older position from backup
        backup_position_uuid = "3eb40617-377f-48a5-b719-a0ea304c7c5f"
        backup_position = self.create_test_position(
            position_uuid=backup_position_uuid,
            open_ms=1000  # Older
        )

        print(f"Memory position (newer): UUID={memory_position_uuid[:8]}..., open_ms={memory_position.open_ms}")
        print(f"Backup position (older): UUID={backup_position_uuid[:8]}..., open_ms={backup_position.open_ms}")

        # This simulates autosync calling close_older_open_position before save
        result = self.validator_sync.close_older_open_position(backup_position, None)

        # Should keep the newer memory position
        self.assertEqual(result.position_uuid, memory_position_uuid)

        # Should have closed the older backup position
        self.assertEqual(
            self.validator_sync.global_stats.get('n_positions_closed_duplicate_opens_for_trade_pair', 0),
            1
        )

        # Now when we save, there should be no validation error
        # because the older position was already closed
        self.position_client.save_miner_position(result)

        # Verify final state on disk
        positions_on_disk = self.position_client.get_positions_for_one_hotkey(
            self.TEST_HOTKEY,
            only_open_positions=False
        )

        # Should have 2 positions: 1 open (newer), 1 closed (older)
        self.assertEqual(len(positions_on_disk), 2)

        open_positions = [p for p in positions_on_disk if p.is_open_position]
        closed_positions = [p for p in positions_on_disk if p.is_closed_position]

        self.assertEqual(len(open_positions), 1)
        self.assertEqual(len(closed_positions), 1)

        # Open position should be the newer one
        self.assertEqual(open_positions[0].position_uuid, memory_position_uuid)

        print(f"✅ Production bug scenario handled correctly")
        print(f"✅ Newer position {memory_position_uuid[:8]}... kept open")
        print(f"✅ Older position {backup_position_uuid[:8]}... closed")
        print(f"✅ No ValiRecordsMisalignmentException raised")


if __name__ == '__main__':
    import unittest
    unittest.main()
