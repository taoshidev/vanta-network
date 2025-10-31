# developer: jbonilla
import unittest
from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.mock_classes import MockLivePriceFetcher
from shared_objects.mock_metagraph import MockMetagraph
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order


class TestPositionSplitting(TestBase):
    
    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        self.DEFAULT_ACCOUNT_SIZE = 100_000
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        
    def create_position_with_orders(self, orders_data):
        """Helper to create a position with specified orders."""
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
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_uuid",
            open_ms=1000,
            trade_pair=TradePair.BTCUSD,
            orders=orders,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        
        return position
    
    def test_position_splitting_always_available(self):
        """Test that position splitting is always available in PositionManager."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position that should be split
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120)
        ])
        
        # Splitting should always work when called directly
        result, split_info = position_manager.split_position_on_flat(position)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].orders), 2)  # LONG and FLAT
        self.assertEqual(len(result[1].orders), 1)  # SHORT
    
    
    def test_implicit_flat_splitting(self):
        """Test splitting on implicit flat (cumulative leverage reaches zero)."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        self.elimination_manager.position_manager = position_manager
        
        # Create a position where cumulative leverage reaches zero implicitly
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),
            (OrderType.SHORT, -2.0, 110),  # Cumulative leverage = 0
            (OrderType.LONG, 1.0, 120)
        ])
        
        # Split the position
        result, split_info = position_manager.split_position_on_flat(position)
        
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
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
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
        result, split_info = position_manager.split_position_on_flat(position, track_stats=True)
        
        # Verify split happened
        self.assertEqual(len(result), 3)  # Should split into 3 positions
        
        # Check stats were updated correctly
        stats = position_manager.split_stats[self.DEFAULT_MINER_HOTKEY]
        self.assertEqual(stats['n_positions_split'], 1)
        self.assertEqual(stats['product_return_pre_split'], pre_split_return)
        
        # Calculate expected post-split product
        expected_post_split_product = 1.0
        for pos in result:
            if pos.is_closed_position:
                expected_post_split_product *= pos.return_at_close
        
        self.assertAlmostEqual(stats['product_return_post_split'], expected_post_split_product, places=6)
    
    def test_split_positions_on_disk_load(self):
        """Test that positions are split on disk load when flag is enabled."""
        # First create and save some positions with a normal manager
        position_manager_save = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher,
            # Splitting is always available in PositionManager
        )
        self.elimination_manager.position_manager = position_manager_save
        position_manager_save.clear_all_miner_positions()
        
        # Create and save a position that should be split
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120)
        ])
        position_manager_save.save_miner_position(position)
        
        # Now create a new manager with splitting on disk load enabled
        position_manager_load = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher,
            split_positions_on_disk_load=True  # Enable splitting on disk load
        )
        
        # Check that positions were split on load
        loaded_positions = position_manager_load.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(loaded_positions), 2)
        
        # Verify the split happened correctly
        positions_by_order_count = sorted(loaded_positions, key=lambda p: len(p.orders))
        self.assertEqual(len(positions_by_order_count[0].orders), 1)  # SHORT order
        self.assertEqual(len(positions_by_order_count[1].orders), 2)  # LONG and FLAT orders
    
    def test_no_split_when_no_flat_orders(self):
        """Test that positions without FLAT orders are not split."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position without FLAT orders
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.LONG, 0.5, 110),
            (OrderType.SHORT, -0.5, 120)
        ])
        
        # Should not be split
        result, split_info = position_manager.split_position_on_flat(position)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], position)
    
    def test_multiple_splits_in_one_position(self):
        """Test splitting a position with multiple FLAT orders."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position with multiple FLAT orders
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.FLAT, 0.0, 110),
            (OrderType.SHORT, -1.0, 120),
            (OrderType.FLAT, 0.0, 130),
            (OrderType.LONG, 2.0, 140)
        ])
        
        # Should be split into 3 positions
        result, split_info = position_manager.split_position_on_flat(position)
        self.assertEqual(len(result), 3)
        
        # Verify each split
        self.assertEqual(len(result[0].orders), 2)  # LONG, FLAT
        self.assertEqual(len(result[1].orders), 2)  # SHORT, FLAT
        self.assertEqual(len(result[2].orders), 1)  # LONG
    
    def test_split_stats_multiple_miners(self):
        """Test that splitting statistics are tracked separately for each miner."""
        # Create manager with multiple miners
        mock_metagraph = MockMetagraph(["miner1", "miner2", "miner3"])
        position_manager = PositionManager(
            metagraph=mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher,
            # Splitting is always available in PositionManager
        )
        
        # Create positions for different miners
        positions_data = {
            "miner1": [(OrderType.LONG, 1.0, 100), (OrderType.FLAT, 0.0, 110), (OrderType.SHORT, -1.0, 105)],
            "miner2": [(OrderType.SHORT, -1.0, 100), (OrderType.FLAT, 0.0, 90), (OrderType.LONG, 1.0, 85)],
            "miner3": [(OrderType.LONG, 2.0, 100), (OrderType.SHORT, -1.0, 110)]  # No split needed
        }
        
        for miner, orders_data in positions_data.items():
            orders = []
            for i, (order_type, leverage, price) in enumerate(orders_data):
                order = Order(
                    price=price,
                    processed_ms=1000 + i * 1000,
                    order_uuid=f"{miner}_order_{i}",
                    trade_pair=TradePair.BTCUSD,
                    order_type=order_type,
                    leverage=leverage,
                )
                orders.append(order)
            
            position = Position(
                miner_hotkey=miner,
                position_uuid=f"{miner}_position",
                open_ms=1000,
                trade_pair=TradePair.BTCUSD,
                orders=orders
            )
            position.rebuild_position_with_updated_orders(self.live_price_fetcher)
            
            # Split with tracking
            result, split_info = position_manager.split_position_on_flat(position, track_stats=True)
        
        # Verify stats for each miner
        stats1 = position_manager.split_stats["miner1"]
        self.assertEqual(stats1['n_positions_split'], 1)  # Split once
        
        stats2 = position_manager.split_stats["miner2"]
        self.assertEqual(stats2['n_positions_split'], 1)  # Split once
        
        # miner3 should not have stats since no split occurred
        self.assertNotIn("miner3", position_manager.split_stats)
    
    def test_leverage_flip_positive_to_negative(self):
        """Test implicit flat when leverage flips from positive to negative."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position where leverage flips from positive to negative
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),    # Cumulative: 2.0
            (OrderType.SHORT, -3.0, 110),   # Cumulative: -1.0 (FLIP!)
            (OrderType.LONG, 1.0, 120)     # Cumulative: 0.0
        ])
        
        # Split the position
        result, split_info = position_manager.split_position_on_flat(position)
        
        # Should be split into 2 positions
        self.assertEqual(len(result), 2)
        
        # First position should have LONG and SHORT orders
        self.assertEqual(len(result[0].orders), 2)
        self.assertEqual(result[0].orders[0].order_type, OrderType.LONG)
        self.assertEqual(result[0].orders[0].leverage, 2.0)
        self.assertEqual(result[0].orders[1].order_type, OrderType.SHORT)
        self.assertEqual(result[0].orders[1].leverage, -3.0)
        
        # Second position should have LONG order
        self.assertEqual(len(result[1].orders), 1)
        self.assertEqual(result[1].orders[0].order_type, OrderType.LONG)
        self.assertEqual(result[1].orders[0].leverage, 1.0)
        
        # Verify split info - leverage flip counts as implicit flat
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)
    
    def test_leverage_flip_negative_to_positive(self):
        """Test implicit flat when leverage flips from negative to positive."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position where leverage flips from negative to positive
        position = self.create_position_with_orders([
            (OrderType.SHORT, -2.0, 100),   # Cumulative: -2.0
            (OrderType.LONG, 3.0, 110),    # Cumulative: 1.0 (FLIP!)
            (OrderType.SHORT, -1.0, 120)    # Cumulative: 0.0
        ])
        
        # Split the position
        result, split_info = position_manager.split_position_on_flat(position)
        
        # Should be split into 2 positions
        self.assertEqual(len(result), 2)
        
        # Verify split info - leverage flip counts as implicit flat
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)
    
    def test_multiple_leverage_flips(self):
        """Test multiple leverage flips in a single position."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position with multiple leverage flips
        position = self.create_position_with_orders([
            (OrderType.LONG, 2.0, 100),     # Cumulative: 2.0
            (OrderType.SHORT, -3.0, 110),    # Cumulative: -1.0 (FLIP 1!)
            (OrderType.LONG, 2.0, 120),     # Cumulative: 1.0 (FLIP 2!)
            (OrderType.SHORT, -2.0, 130),    # Cumulative: -1.0 (FLIP 3!)
            (OrderType.LONG, 1.0, 140)      # Cumulative: 0.0
        ])
        
        # Split the position
        result, split_info = position_manager.split_position_on_flat(position)
        
        # The position will split at multiple points:
        # 1. Order 1: leverage flip from +2.0 to -1.0 (valid: 2 orders before, 3 after)
        # 2. After first split, new segment starts with cumulative=0
        #    Order 2: LONG 2.0 -> cum=2.0
        #    Order 3: SHORT -2.0 -> cum=0.0 (zero leverage, valid: 2 orders before, 1 after)
        # Result: 3 positions total
        self.assertEqual(len(result), 3)
        
        # Verify split info - 2 implicit flats (1 flip, 1 zero)
        self.assertEqual(split_info['implicit_flat_splits'], 2)
        self.assertEqual(split_info['explicit_flat_splits'], 0)
    
    def test_no_split_without_flip_or_zero(self):
        """Test that positions don't split without leverage flip or reaching zero."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position where leverage stays positive
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),    # Cumulative: 1.0
            (OrderType.LONG, 0.5, 110),    # Cumulative: 1.5
            (OrderType.SHORT, -0.5, 120),   # Cumulative: 1.0 (still positive)
            (OrderType.LONG, 0.5, 130)     # Cumulative: 1.5
        ])
        
        # Split the position
        result, split_info = position_manager.split_position_on_flat(position)
        
        # Should NOT be split
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], position)
        
        # Verify split info
        self.assertEqual(split_info['implicit_flat_splits'], 0)
        self.assertEqual(split_info['explicit_flat_splits'], 0)
    
    def test_mixed_implicit_and_explicit_flats(self):
        """Test position with both implicit flats (leverage flips/zero) and explicit FLAT orders."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
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
        result, split_info = position_manager.split_position_on_flat(position)
        
        # Should be split into 3 positions
        # Split at: index 1 (zero leverage), index 3 (explicit FLAT)
        # Note: index 4 is NOT a valid split point because it would only leave 1 order after
        self.assertEqual(len(result), 3)
        
        # Verify split info - 1 implicit (zero) and 1 explicit
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 1)
    
    def test_leverage_near_zero_threshold(self):
        """Test that leverage values very close to zero are treated as zero."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create a position where leverage reaches nearly zero (within 1e-9)
        position = self.create_position_with_orders([
            (OrderType.LONG, 1.0, 100),
            (OrderType.SHORT, -(1.0 - 1e-10), 110),  # Cumulative: ~1e-10 (treated as 0)
            (OrderType.LONG, 1.0, 120)
        ])
        
        # Split the position
        result, split_info = position_manager.split_position_on_flat(position)
        
        # Should be split into 2 positions
        self.assertEqual(len(result), 2)
        
        # Verify split info - near-zero counts as implicit flat
        self.assertEqual(split_info['implicit_flat_splits'], 1)
        self.assertEqual(split_info['explicit_flat_splits'], 0)
    
    def test_no_split_at_last_order(self):
        """Test that splits don't occur at the last order even if it's a flat."""
        position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        
        # Create positions ending with various flat conditions
        test_cases = [
            # Explicit FLAT at end
            [(OrderType.LONG, 1.0, 100), (OrderType.FLAT, 0.0, 110)],
            # Implicit flat (zero) at end
            [(OrderType.LONG, 1.0, 100), (OrderType.SHORT, -1.0, 110)],
            # Implicit flat (flip) at end
            [(OrderType.LONG, 2.0, 100), (OrderType.SHORT, -3.0, 110)]
        ]
        
        for orders_data in test_cases:
            position = self.create_position_with_orders(orders_data)
            result, split_info = position_manager.split_position_on_flat(position)
            
            # Should NOT be split (flat is at last order)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], position)
            self.assertEqual(split_info['implicit_flat_splits'], 0)
            self.assertEqual(split_info['explicit_flat_splits'], 0)


if __name__ == '__main__':
    unittest.main()
