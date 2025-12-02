# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test position splitting functionality using modern server/client architecture.

Tests comprehensive position splitting scenarios:
- Explicit FLAT orders
- Implicit flats (leverage reaching zero)
- Leverage sign flips
- Multiple splits in single position
- Split statistics tracking
"""
import unittest

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase

from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestPositionSplitting(TestBase):
    """
    Position splitting tests using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    elimination_client = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_ACCOUNT_SIZE = 100_000

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.elimination_client = cls.orchestrator.get_client('elimination')

        # Initialize metagraph with test miners
        cls.metagraph_client.set_hotkeys([cls.DEFAULT_MINER_HOTKEY, "miner1", "miner2", "miner3"])

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def create_position_with_orders(self, orders_data, miner_hotkey=None):
        """Helper to create a position with specified orders."""
        if miner_hotkey is None:
            miner_hotkey = self.DEFAULT_MINER_HOTKEY

        orders = []
        for i, (order_type, leverage, price) in enumerate(orders_data):
            order = Order(
                price=price,
                processed_ms=1000 + i * 1000,
                order_uuid=f"order_{i}",
                trade_pair=TradePair.BTCUSD,
                order_type=order_type,
                leverage=leverage,
            )
            orders.append(order)

        position = Position(
            miner_hotkey=miner_hotkey,
            position_uuid=f"{miner_hotkey}_test_position_uuid",
            open_ms=1000,
            trade_pair=TradePair.BTCUSD,
            orders=orders,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)

        return position

    def test_position_splitting_always_available(self):
        """Test that position splitting is always available in PositionManager."""
        # Create a position that should be split
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120)
        ])

        # Splitting should always work when called directly
        result, split_info = self.position_client.split_position_on_flat(position)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].orders), 2)  # LONG and FLAT
        self.assertEqual(len(result[1].orders), 1)  # SHORT

    def test_implicit_flat_splitting(self):
        """Test splitting on implicit flat (cumulative leverage reaches zero)."""
        # Create a position where cumulative leverage reaches zero implicitly
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),
            (OrderType.SHORT, -2.0, 110),  # Cumulative leverage = 0
            (OrderType.LONG, 1.0, 120)
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # Should be split into 2 positions
        self.assertEqual(len(result), 2)

        # First position should have LONG and SHORT orders
        self.assertEqual(len(result[0].orders), 2)
        self.assertEqual(result[0].orders[0].order_type, OrderType.LONG)
        self.assertEqual(result[0].orders[1].order_type, OrderType.SHORT)

        # Second position should have LONG order
        self.assertEqual(len(result[1].orders), 1)
        self.assertEqual(result[1].orders[0].order_type, OrderType.LONG)

        # Verify split info
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)

    def test_split_stats_tracking(self):
        """Test that splitting statistics are tracked correctly."""
        # Create a closed position with specific returns
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120),
            (OrderType.FLAT, 0.0, 100),
            (OrderType.LONG, 1.0, 95)  # Add another order after FLAT
        ])
        position.close_out_position(6000)

        # Get the pre-split return for verification
        pre_split_return = position.return_at_close

        # Split with tracking enabled
        result, split_info = self.position_client.split_position_on_flat(position, track_stats=True)

        # Verify split happened
        self.assertEqual(len(result), 3)  # Should split into 3 positions

        # Check stats were updated correctly via client
        stats = self.position_client.get_split_stats(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(stats['n_positions_split'], 1)
        self.assertEqual(stats['product_return_pre_split'], pre_split_return)

        # Calculate expected post-split product
        expected_post_split_product = 1.0
        for pos in result:
            if pos.is_closed_position:
                expected_post_split_product *= pos.return_at_close

        self.assertAlmostEqual(stats['product_return_post_split'], expected_post_split_product, places=6)

    def test_split_positions_on_disk_load(self):
        """Test that positions can be manually split after loading from disk."""
        # Create and save a position that should be split
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120)
        ])
        self.position_client.save_miner_position(position)

        # Load positions from server
        loaded_positions = self.position_client.get_positions_for_one_hotkey(
            self.DEFAULT_MINER_HOTKEY
        )

        # Initially should be 1 position (not split yet)
        self.assertEqual(len(loaded_positions), 1)

        # Split the loaded position manually
        split_result, _ = self.position_client.split_position_on_flat(loaded_positions[0])

        # Check that positions were split
        self.assertEqual(len(split_result), 2)

        # Verify the split happened correctly
        positions_by_order_count = sorted(split_result, key=lambda p: len(p.orders))
        self.assertEqual(len(positions_by_order_count[0].orders), 1)  # SHORT order
        self.assertEqual(len(positions_by_order_count[1].orders), 2)  # LONG and FLAT orders

    def test_no_split_when_no_flat_orders(self):
        """Test that positions without FLAT orders are not split."""
        # Create a position without FLAT orders
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.LONG, 0.5, 110),
            (OrderType.SHORT, -0.5, 120)
        ])

        # Should not be split
        result, split_info = self.position_client.split_position_on_flat(position)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].position_uuid, position.position_uuid)

    def test_multiple_splits_in_one_position(self):
        """Test splitting a position with multiple FLAT orders."""
        # Create a position with multiple FLAT orders
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120),
            (OrderType.FLAT, 0.0, 130),
            (OrderType.LONG, 2.0, 140)
        ])

        # Should be split into 3 positions
        result, split_info = self.position_client.split_position_on_flat(position)
        self.assertEqual(len(result), 3)

        # Verify each split
        self.assertEqual(len(result[0].orders), 2)  # LONG, FLAT
        self.assertEqual(len(result[1].orders), 2)  # SHORT, FLAT
        self.assertEqual(len(result[2].orders), 1)  # LONG

    def test_split_stats_multiple_miners(self):
        """Test that splitting statistics are tracked separately for each miner."""
        # Create positions for different miners
        positions_data = {
            "miner1": [(OrderType.LONG, 1.0, 100), (OrderType.FLAT, 0.0, 110), (OrderType.SHORT, -1.0, 105)],
            "miner2": [(OrderType.SHORT, -1.0, 100), (OrderType.FLAT, 0.0, 90), (OrderType.LONG, 1.0, 85)],
            "miner3": [(OrderType.LONG, 2.0, 100), (OrderType.SHORT, -1.0, 110)]  # No split needed
        }

        for miner, orders_data in positions_data.items():
            position = self.create_position_with_orders(orders_data, miner_hotkey=miner)

            # Split with tracking
            result, split_info = self.position_client.split_position_on_flat(position, track_stats=True)

        # Verify stats for each miner
        stats1 = self.position_client.get_split_stats("miner1")
        self.assertEqual(stats1['n_positions_split'], 1)  # Split once

        stats2 = self.position_client.get_split_stats("miner2")
        self.assertEqual(stats2['n_positions_split'], 1)  # Split once

        # miner3 should have zero splits since no split occurred
        stats3 = self.position_client.get_split_stats("miner3")
        self.assertEqual(stats3['n_positions_split'], 0)

    def test_leverage_flip_positive_to_negative(self):
        """Test implicit flat when leverage flips from positive to negative."""
        # Create a position where leverage flips from positive to negative
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),    # Cumulative: 2.0
            (OrderType.SHORT, -3.0, 110),   # Cumulative: -1.0 (FLIP!)
            (OrderType.LONG, 1.0, 120)     # Cumulative: 0.0
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # Should be split into 2 positions
        self.assertEqual(len(result), 2)

        # First position should have LONG and SHORT orders
        self.assertEqual(len(result[0].orders), 2)
        self.assertEqual(result[0].orders[0].order_type, OrderType.LONG)
        self.assertEqual(result[0].orders[0].leverage, 2.0)
        self.assertEqual(result[0].orders[1].order_type, OrderType.SHORT)
        self.assertEqual(result[0].orders[1].leverage, -2.0)

        # Second position should have LONG order
        self.assertEqual(len(result[1].orders), 1)
        self.assertEqual(result[1].orders[0].order_type, OrderType.LONG)
        self.assertEqual(result[1].orders[0].leverage, 1.0)

        # Verify split info - leverage flip counts as implicit flat
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)

    def test_leverage_flip_negative_to_positive(self):
        """Test implicit flat when leverage flips from negative to positive."""
        # Create a position where leverage flips from negative to positive
        position = self.create_position_with_orders([
            (OrderType.SHORT, -2.0, 100),   # Cumulative: -2.0
            (OrderType.LONG, 3.0, 110),    # Cumulative: 1.0 (FLIP!)
            (OrderType.SHORT, -1.0, 120)    # Cumulative: 0.0
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # Should be split into 2 positions
        self.assertEqual(len(result), 2)

        # Verify split info - leverage flip counts as implicit flat
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)

    def test_multiple_leverage_flips(self):
        """Test multiple leverage flips in a single position."""
        # Create a position with multiple leverage flips
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),     # Cumulative: 2.0
            (OrderType.SHORT, -3.0, 110),    # Cumulative: -1.0 (FLIP 1!)
            (OrderType.LONG, 2.0, 120),     # Cumulative: 1.0 (FLIP 2!)
            (OrderType.SHORT, -2.0, 130),    # Cumulative: -1.0 (FLIP 3!)
            (OrderType.LONG, 1.0, 140)      # Cumulative: 0.0
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # The position will split at multiple points
        self.assertEqual(len(result), 3)

        # Verify split info - 2 implicit flats (1 flip, 1 zero)
        self.assertEqual(split_info['implicit_flat_splits'], 2)
        self.assertEqual(split_info['explicit_flat_splits'], 0)

    def test_no_split_without_flip_or_zero(self):
        """Test that positions don't split without leverage flip or reaching zero."""
        # Create a position where leverage stays positive
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),    # Cumulative: 1.0
            (OrderType.LONG, 0.5, 110),    # Cumulative: 1.5
            (OrderType.SHORT, -0.5, 120),   # Cumulative: 1.0 (still positive)
            (OrderType.LONG, 0.5, 130)     # Cumulative: 1.5
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # Should NOT be split
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].position_uuid, position.position_uuid)

        # Verify split info
        self.assertEqual(split_info['implicit_flat_splits'], 0)
        self.assertEqual(split_info['explicit_flat_splits'], 0)

    def test_mixed_implicit_and_explicit_flats(self):
        """Test position with both implicit flats (leverage flips/zero) and explicit FLAT orders."""
        # Create a position with mixed split points
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),     # Cumulative: 2.0
            (OrderType.SHORT, -2.0, 110),    # Cumulative: 0.0 (implicit - zero)
            (OrderType.LONG, 1.0, 120),     # Cumulative: 1.0
            (OrderType.FLAT, 0.0, 130),     # Explicit FLAT
            (OrderType.SHORT, -2.0, 140),    # Cumulative: -1.0 (implicit - flip)
            (OrderType.LONG, 1.0, 150)      # Cumulative: 0.0
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # Should be split into 3 positions
        self.assertEqual(len(result), 3)

        # Verify split info - 1 implicit (zero) and 1 explicit
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 1)

    def test_leverage_near_zero_threshold(self):
        """Test that leverage values very close to zero are treated as zero."""
        # Create a position where leverage reaches nearly zero (within 1e-9)
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.SHORT, -(1.0 - 1e-10), 110),  # Cumulative: ~1e-10 (treated as 0)
            (OrderType.LONG, 1.0, 120)
        ])

        # Split the position
        result, split_info = self.position_client.split_position_on_flat(position)

        # Should be split into 2 positions
        self.assertEqual(len(result), 2)

        # Verify split info - near-zero counts as implicit flat
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)

    def test_no_split_at_last_order(self):
        """Test that splits don't occur at the last order even if it's a flat."""
        # Create positions ending with various flat conditions
        test_cases = [
            # Explicit FLAT at end
            [(OrderType.LONG, 1.0, 100), (OrderType.FLAT, 0.0, 110)],
            # Implicit flat (zero) at end
            [(OrderType.LONG, 1.0, 100), (OrderType.SHORT, -1.0, 110)],
            # Implicit flat (flip) at end
            [(OrderType.LONG, 2.0, 100), (OrderType.SHORT, -3.0, 110)]
        ]

        for i, orders_data in enumerate(test_cases):
            position = self.create_position_with_orders(orders_data)
            # Use unique UUIDs for each test case
            position.position_uuid = f"{self.DEFAULT_MINER_HOTKEY}_test_case_{i}"

            result, split_info = self.position_client.split_position_on_flat(position)

            # Should NOT be split (flat is at last order)
            self.assertEqual(len(result), 1, f"Test case {i} failed")
            self.assertEqual(result[0].position_uuid, position.position_uuid)
            self.assertEqual(split_info['implicit_flat_splits'], 0)
            self.assertEqual(split_info['explicit_flat_splits'], 0)


if __name__ == '__main__':
    unittest.main()
