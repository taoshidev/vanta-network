"""
Core performance ledger tests.

This file contains the essential tests for performance ledger functionality:
- Basic ledger operations
- Position tracking and calculations
- Return and fee calculations
- Multi-trade pair scenarios
"""

import unittest
from unittest.mock import patch, Mock
import math
from decimal import Decimal

from tests.shared_objects.mock_classes import MockLivePriceFetcher

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS, MS_IN_8_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfLedger,
    PerfLedgerManager,
    PerfCheckpoint,
    TP_ID_PORTFOLIO,
    ParallelizationMode,
    TradePairReturnStatus,
)


class TestPerfLedgerCore(TestBase):
    """Core performance ledger functionality tests."""

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.test_hotkey = "test_miner_core"
        self.now_ms = TimeUtil.now_in_millis()
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        
        self.mmg = MockMetagraph(hotkeys=[self.test_hotkey])
        self.elimination_manager = EliminationManager(self.mmg, None, None, running_unit_tests=True)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )
        self.position_manager.clear_all_miner_positions()

    def validate_perf_ledger(self, ledger: PerfLedger, expected_init_time: int = None):
        """Validate performance ledger structure and attributes."""
        # Basic structure validation
        self.assertIsInstance(ledger, PerfLedger)
        self.assertIsInstance(ledger.cps, list)
        self.assertIsInstance(ledger.initialization_time_ms, int)
        self.assertIsInstance(ledger.max_return, float)
        
        # Time validation
        if expected_init_time:
            self.assertEqual(ledger.initialization_time_ms, expected_init_time)
        
        # Checkpoint sequence validation
        prev_time = 0
        for i, cp in enumerate(ledger.cps):
            self.validate_checkpoint(cp, f"Checkpoint {i}")
            self.assertGreaterEqual(cp.last_update_ms, prev_time, 
                                   f"Checkpoint {i} time should be >= previous")
            prev_time = cp.last_update_ms

    def validate_checkpoint(self, cp: PerfCheckpoint, context: str = ""):
        """Validate checkpoint structure and attributes."""
        # Basic type validation
        self.assertIsInstance(cp.last_update_ms, int, f"{context}: last_update_ms should be int")
        self.assertIsInstance(cp.gain, float, f"{context}: gain should be float")
        self.assertIsInstance(cp.loss, float, f"{context}: loss should be float")
        self.assertIsInstance(cp.n_updates, int, f"{context}: n_updates should be int")
        
        # Portfolio value validation
        self.assertIsInstance(cp.prev_portfolio_ret, float, f"{context}: prev_portfolio_ret should be float")
        self.assertIsInstance(cp.prev_portfolio_spread_fee, float, f"{context}: prev_portfolio_spread_fee should be float")
        self.assertIsInstance(cp.prev_portfolio_carry_fee, float, f"{context}: prev_portfolio_carry_fee should be float")
        
        # Risk metrics validation
        self.assertIsInstance(cp.mdd, float, f"{context}: mdd should be float")
        self.assertIsInstance(cp.mpv, float, f"{context}: mpv should be float")
        
        # Logical constraints
        self.assertGreaterEqual(cp.n_updates, 0, f"{context}: n_updates should be >= 0")
        self.assertGreaterEqual(cp.gain, 0.0, f"{context}: gain should be >= 0")
        self.assertLessEqual(cp.loss, 0.0, f"{context}: loss should be <= 0")
        
        # Carry fee loss validation (allow small negative values due to floating point precision)
        if hasattr(cp, 'carry_fee_loss'):
            self.assertGreaterEqual(cp.carry_fee_loss, -0.01, f"{context}: carry_fee_loss should be reasonable")
        
        # Portfolio values should be reasonable
        self.assertGreater(cp.prev_portfolio_ret, 0.0, f"{context}: portfolio return should be positive")
        self.assertGreater(cp.prev_portfolio_spread_fee, 0.0, f"{context}: spread fee should be positive")
        self.assertGreater(cp.prev_portfolio_carry_fee, 0.0, f"{context}: carry fee should be positive")
        
        # Risk metrics should be reasonable
        self.assertGreater(cp.mdd, 0.0, f"{context}: MDD should be positive")
        self.assertGreater(cp.mpv, 0.0, f"{context}: MPV should be positive")
        
        # Fees should not exceed 100%
        self.assertLessEqual(cp.prev_portfolio_spread_fee, 1.0, f"{context}: spread fee should be <= 1.0")
        self.assertLessEqual(cp.prev_portfolio_carry_fee, 1.0, f"{context}: carry fee should be <= 1.0")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_basic_position_tracking(self, mock_lpf):
        """Test basic position tracking and checkpoint creation."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        base_time = self.now_ms - (10 * MS_IN_24_HOURS)
        
        # Create a simple position
        position = self._create_position(
            "basic_pos", TradePair.BTCUSD,
            base_time, base_time + MS_IN_24_HOURS,
            50000.0, 51000.0,  # 2% gain
            OrderType.LONG
        )
        self.position_manager.save_miner_position(position)
        
        # Update ledger
        plm.update(t_ms=base_time + (2 * MS_IN_24_HOURS))
        
        # Verify
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.test_hotkey, bundles)
        
        bundle = bundles[self.test_hotkey]
        self.assertIn(TradePair.BTCUSD.trade_pair_id, bundle)
        self.assertIn(TP_ID_PORTFOLIO, bundle)
        
        # Validate each ledger thoroughly
        for tp_id, ledger in bundle.items():
            self.validate_perf_ledger(ledger, base_time)
            
            # Check checkpoints exist
            self.assertGreater(len(ledger.cps), 0, f"Ledger {tp_id} should have checkpoints")
            
            # For a 2-day period with 12-hour checkpoints, expect 4 checkpoints minimum
            # The exact count depends on alignment and timing, but should be reasonable
            min_expected_checkpoints = 3  # At least 3 checkpoints for 2-day period
            max_expected_checkpoints = 6  # At most 6 for boundary cases
            self.assertGreaterEqual(len(ledger.cps), min_expected_checkpoints,
                                   f"Expected at least {min_expected_checkpoints} checkpoints, got {len(ledger.cps)} for {tp_id}")
            self.assertLessEqual(len(ledger.cps), max_expected_checkpoints,
                                f"Expected at most {max_expected_checkpoints} checkpoints, got {len(ledger.cps)} for {tp_id}")
            
            # Validate that at least one checkpoint has trading activity
            # We created a position, so there must be activity recorded
            has_activity = any(cp.n_updates > 0 for cp in ledger.cps)
            if tp_id == TradePair.BTCUSD.trade_pair_id:
                # For the specific trade pair we created a position in, must have activity
                self.assertTrue(has_activity, f"BTC ledger must have trading activity - we created a position")
            elif tp_id == TP_ID_PORTFOLIO:
                # Portfolio aggregates individual TPs, so should also have activity
                self.assertTrue(has_activity, f"Portfolio ledger must reflect BTC trading activity")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_return_calculation_accuracy(self, mock_lpf):
        """Test accurate return calculations for various scenarios."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        base_time = self.now_ms - (20 * MS_IN_24_HOURS)
        
        test_cases = [
            # (name, open_price, close_price, order_type, expected_return_sign)
            ("10% gain long", 50000.0, 55000.0, OrderType.LONG, 1),  # positive
            ("10% loss long", 50000.0, 45000.0, OrderType.LONG, -1),  # negative
            ("10% gain short", 50000.0, 45000.0, OrderType.SHORT, 1),  # positive (price fell)
            ("10% loss short", 50000.0, 55000.0, OrderType.SHORT, -1),  # negative (price rose)
            ("no change", 50000.0, 50000.0, OrderType.LONG, 0),  # zero
        ]
        
        for i, (name, open_price, close_price, order_type, expected_sign) in enumerate(test_cases):
            position = self._create_position(
                f"pos_{i}", TradePair.BTCUSD,
                base_time + (i * 2 * MS_IN_24_HOURS),
                base_time + (i * 2 * MS_IN_24_HOURS) + MS_IN_24_HOURS,
                open_price, close_price, order_type
            )
            self.position_manager.save_miner_position(position)
        
        # Update after all positions
        plm.update(t_ms=base_time + (15 * MS_IN_24_HOURS))
        
        # Verify returns
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        bundle = bundles[self.test_hotkey]
        
        # Validate all ledgers
        for tp_id, ledger in bundle.items():
            self.validate_perf_ledger(ledger, base_time)
            
            # Check that we have checkpoints with varied return characteristics
            gains_found = 0
            losses_found = 0
            neutral_found = 0

            for cp in ledger.cps:
                if cp.gain > 0:
                    gains_found += 1
                elif cp.loss < 0:
                    losses_found += 1
                elif cp.n_updates > 0 and cp.gain == 0 and cp.loss == 0:
                    neutral_found += 1

            # Should have variety in returns
            if tp_id == TradePair.BTCUSD.trade_pair_id:
                self.assertGreater(gains_found, 0, f"Ledger {tp_id} should have some gaining checkpoints")
                self.assertGreater(losses_found, 0, f"Ledger {tp_id} should have some losing checkpoints")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_multi_trade_pair_aggregation(self, mock_lpf):
        """Test portfolio aggregation across multiple trade pairs."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        plm.clear_all_ledger_data()
        
        base_time = self.now_ms - (10 * MS_IN_24_HOURS)
        
        # Create positions in different trade pairs
        positions = [
            ("btc", TradePair.BTCUSD, 50000.0, 52000.0),  # 4% gain
            ("eth", TradePair.ETHUSD, 3000.0, 2850.0),    # 5% loss
            ("eur", TradePair.EURUSD, 1.10, 1.12),        # ~1.8% gain
        ]
        
        for name, tp, open_price, close_price in positions:
            position = self._create_position(
                name, tp,
                base_time, base_time + MS_IN_24_HOURS,
                open_price, close_price, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (2 * MS_IN_24_HOURS))
        
        # Verify all trade pairs are tracked
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        bundle = bundles[self.test_hotkey]
        
        # Validate each expected trade pair
        for _, tp, _, _ in positions:
            self.assertIn(tp.trade_pair_id, bundle, f"{tp.trade_pair_id} should be in bundle")
            self.validate_perf_ledger(bundle[tp.trade_pair_id], base_time)
        
        # Portfolio should aggregate all positions
        self.assertIn(TP_ID_PORTFOLIO, bundle, "Portfolio ledger should exist")
        portfolio_ledger = bundle[TP_ID_PORTFOLIO]
        self.validate_perf_ledger(portfolio_ledger, base_time)
        
        # Portfolio should have at least as many checkpoints as individual TPs
        min_individual_cps = min(len(bundle[tp.trade_pair_id].cps) for _, tp, _, _ in positions)
        self.assertGreaterEqual(len(portfolio_ledger.cps), min_individual_cps,
                               "Portfolio should have reasonable checkpoint count")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_fee_calculations(self, mock_lpf):
        """Test carry fee and spread fee calculations."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        plm.clear_all_ledger_data()
        
        base_time = self.now_ms - (10 * MS_IN_24_HOURS)
        
        # Create position held for multiple days (accumulates carry fees)
        position = self._create_position(
            "fee_test", TradePair.BTCUSD,
            base_time, base_time + (5 * MS_IN_24_HOURS),  # 5-day position
            50000.0, 50000.0,  # No price change
            OrderType.LONG
        )
        self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (6 * MS_IN_24_HOURS))
        
        # Check fees
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Validate the ledger structure first
        self.validate_perf_ledger(btc_ledger, base_time)
        
        # Find checkpoint with position and validate fee behavior
        position_checkpoint_found = False
        for i, cp in enumerate(btc_ledger.cps):
            if cp.n_updates > 0 and i != 0:  # Skip initial checkpoint which has an update due to initial spread fee
                position_checkpoint_found = True
                
                # Validate checkpoint structure
                self.validate_checkpoint(cp, "Fee calculation checkpoint")
                
                # Carry fee should be applied over 5 days
                last_cp = btc_ledger.cps[-1]
                self.assertLess(cp.prev_portfolio_carry_fee, 1.0,
                               f"Carry fee should be applied over 5 days #{i}:{cp} {last_cp}")
                self.assertGreater(cp.prev_portfolio_carry_fee, 0.95,
                               f"Carry fee should not be too large for 5 days #{i}:{cp} {last_cp}")
                
                # Spread fee behavior validation
                self.assertLessEqual(cp.prev_portfolio_spread_fee, 1.0)
                self.assertGreater(cp.prev_portfolio_spread_fee, 0.99)
                
                # Additional fee validation - allow small negative values due to floating point precision
                self.assertGreaterEqual(cp.carry_fee_loss, -0.01, "Carry fee loss should be reasonable (small negative values allowed for FP precision)")
                break
        
        self.assertTrue(position_checkpoint_found, "Should find at least one checkpoint with position data")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_checkpoint_time_alignment(self, mock_lpf):
        """Test that checkpoints align to expected time boundaries."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        # Align to checkpoint boundary
        checkpoint_duration = 12 * 60 * 60 * 1000  # 12 hours
        base_time = (self.now_ms // checkpoint_duration) * checkpoint_duration
        base_time -= (5 * MS_IN_24_HOURS)
        
        # Create position
        position = self._create_position(
            "aligned", TradePair.BTCUSD,
            base_time, base_time + MS_IN_24_HOURS,
            50000.0, 50000.0, OrderType.LONG
        )
        self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (2 * MS_IN_24_HOURS))
        
        # Verify checkpoint alignment
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Validate ledger structure
        self.validate_perf_ledger(btc_ledger, base_time)
        
        # Verify checkpoint alignment and structure
        for i, cp in enumerate(btc_ledger.cps):
            # Validate each checkpoint
            self.validate_checkpoint(cp, f"Alignment checkpoint {i}")
            
            # All checkpoints should be aligned to 12-hour boundaries
            self.assertEqual(cp.last_update_ms % checkpoint_duration, 0,
                           f"Checkpoint {i} at {cp.last_update_ms} not aligned to 12-hour boundary")
            
            # Checkpoint time should be reasonable
            self.assertGreaterEqual(cp.last_update_ms, base_time,
                                   f"Checkpoint {i} time should be >= base_time")
            self.assertLessEqual(cp.last_update_ms, base_time + (3 * MS_IN_24_HOURS),
                                f"Checkpoint {i} time should be reasonable")

    def _create_position(self, position_id: str, trade_pair: TradePair, 
                        open_ms: int, close_ms: int, open_price: float, 
                        close_price: float, order_type: OrderType,
                        leverage: float = 1.0) -> Position:
        """Helper to create a position with specified parameters."""
        open_order = Order(
            price=open_price,
            processed_ms=open_ms,
            order_uuid=f"{position_id}_open",
            trade_pair=trade_pair,
            order_type=order_type,
            leverage=leverage if order_type == OrderType.LONG else -leverage,
        )
        
        close_order = Order(
            price=close_price,
            processed_ms=close_ms,
            order_uuid=f"{position_id}_close",
            trade_pair=trade_pair,
            order_type=OrderType.FLAT,
            leverage=0.0,
        )
        
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid=position_id,
            open_ms=open_ms,
            close_ms=close_ms,
            trade_pair=trade_pair,
            orders=[open_order, close_order],
            position_type=OrderType.FLAT,
            is_closed_position=True,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        return position


    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_single_checkpoint_open_ms_tracking(self, mock_lpf):
        """
        Test that a perf ledger with only one checkpoint properly tracks
        the open_ms to match when the position was actually open.
        """
        # Mock the live price fetcher
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        # Create the perf ledger manager
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
            is_testing=True,
        )
        
        # Align to checkpoint boundaries for predictable behavior
        checkpoint_duration = 12 * 60 * 60 * 1000  # 12 hours
        base_time = (self.now_ms // checkpoint_duration) * checkpoint_duration - (5 * MS_IN_24_HOURS)
        
        # Create a position that opens at a specific time and stays open for 3 hours
        position_open_time = base_time + (2 * 60 * 60 * 1000)  # 2 hours after base
        position_close_time = position_open_time + (3 * 60 * 60 * 1000)  # 3 hours later
        
        # Create the position using the helper method
        position = self._create_position(
            "single_checkpoint_pos", TradePair.BTCUSD,
            position_open_time, position_close_time,
            50000.0, 51000.0, OrderType.LONG
        )
        
        # Save the position
        self.position_manager.save_miner_position(position)
        
        # Update the perf ledger at a time after the position closed
        # but within the same checkpoint period
        update_time = position_close_time + (1 * 60 * 60 * 1000)  # 1 hour after close
        plm.update(t_ms=update_time)
        
        # Get the perf ledgers
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.test_hotkey, bundles, "Should have bundle for test hotkey")
        
        # Check BTCUSD ledger
        btc_bundle = bundles[self.test_hotkey]
        self.assertIn(TradePair.BTCUSD.trade_pair_id, btc_bundle, "Should have BTCUSD ledger")
        btc_ledger = btc_bundle[TradePair.BTCUSD.trade_pair_id]
        
        # Verify we have exactly one checkpoint
        self.assertEqual(len(btc_ledger.cps), 1, "Should have exactly one checkpoint")
        
        checkpoint = btc_ledger.cps[0]
        
        # Verify the checkpoint properties
        print(f"\n=== Single Checkpoint Test Results ===")
        print(f"Position open time: {position_open_time} ({TimeUtil.millis_to_formatted_date_str(position_open_time)})")
        print(f"Position close time: {position_close_time} ({TimeUtil.millis_to_formatted_date_str(position_close_time)})")
        print(f"Update time: {update_time} ({TimeUtil.millis_to_formatted_date_str(update_time)})")
        print(f"Checkpoint last_update_ms: {checkpoint.last_update_ms} ({TimeUtil.millis_to_formatted_date_str(checkpoint.last_update_ms)})")
        print(f"Checkpoint accum_ms: {checkpoint.accum_ms} ({checkpoint.accum_ms / (60 * 60 * 1000):.2f} hours)")
        print(f"Checkpoint open_ms: {checkpoint.open_ms} ({checkpoint.open_ms / (60 * 60 * 1000):.2f} hours)")
        print(f"Position was open for: {(position_close_time - position_open_time) / (60 * 60 * 1000):.2f} hours")
        
        # The open_ms should match the duration the position was actually open
        expected_open_duration_ms = position_close_time - position_open_time
        self.assertEqual(
            checkpoint.open_ms,
            expected_open_duration_ms,
            f"Checkpoint open_ms should match position open duration. "
            f"Expected {expected_open_duration_ms}ms ({expected_open_duration_ms/(60*60*1000):.2f}h), "
            f"got {checkpoint.open_ms}ms ({checkpoint.open_ms/(60*60*1000):.2f}h)"
        )
        
        # Verify the checkpoint reflects the position's return (accounting for fees)
        # The checkpoint return will be less than or equal to the raw position return due to fees
        self.assertLessEqual(
            checkpoint.prev_portfolio_ret,
            position.return_at_close,
            msg="Checkpoint return should be less than or equal to position return due to fees"
        )
        self.assertGreater(
            checkpoint.prev_portfolio_ret,
            1.0,
            msg="Checkpoint return should still be profitable"
        )
        
        # Check portfolio ledger
        self.assertIn(TP_ID_PORTFOLIO, btc_bundle, "Should have portfolio ledger")
        portfolio_ledger = btc_bundle[TP_ID_PORTFOLIO]
        self.assertEqual(len(portfolio_ledger.cps), 1, "Portfolio should have one checkpoint")
        
        portfolio_checkpoint = portfolio_ledger.cps[0]
        self.assertEqual(
            portfolio_checkpoint.open_ms,
            expected_open_duration_ms,
            "Portfolio checkpoint should also track correct open duration"
        )

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_single_checkpoint_multiple_positions_sequential(self, mock_lpf):
        """
        Test single checkpoint with multiple positions that open and close sequentially.
        The open_ms should be the sum of all position open durations.
        """
        # Mock the live price fetcher
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
            is_testing=True,
        )
        
        # Align to checkpoint boundaries
        checkpoint_duration = 12 * 60 * 60 * 1000  # 12 hours
        base_time = (self.now_ms // checkpoint_duration) * checkpoint_duration - (5 * MS_IN_24_HOURS)
        
        # Create two positions that don't overlap
        # Position 1: 2-4 hours after base (2 hours duration)
        pos1_open = base_time + (2 * 60 * 60 * 1000)
        pos1_close = base_time + (4 * 60 * 60 * 1000)
        
        position1 = self._create_position(
            "pos1", TradePair.BTCUSD,
            pos1_open, pos1_close,
            50000.0, 51000.0, OrderType.LONG
        )
        self.position_manager.save_miner_position(position1)
        
        # Position 2: 5-7 hours after base (2 hours duration)
        pos2_open = base_time + (5 * 60 * 60 * 1000)
        pos2_close = base_time + (7 * 60 * 60 * 1000)
        
        position2 = self._create_position(
            "pos2", TradePair.BTCUSD,
            pos2_open, pos2_close,
            51000.0, 52000.0, OrderType.LONG
        )
        self.position_manager.save_miner_position(position2)
        
        # Update after both positions are closed
        update_time = base_time + (8 * 60 * 60 * 1000)
        plm.update(t_ms=update_time)
        
        # Get the ledgers
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Should have single checkpoint
        self.assertEqual(len(btc_ledger.cps), 1, "Should have exactly one checkpoint")
        
        checkpoint = btc_ledger.cps[0]
        
        # Total open duration should be sum of both positions
        expected_total_open_ms = (pos1_close - pos1_open) + (pos2_close - pos2_open)
        expected_total_hours = expected_total_open_ms / (60 * 60 * 1000)
        
        print(f"\n=== Sequential Positions Test ===")
        print(f"Position 1 open duration: {(pos1_close - pos1_open)/(60*60*1000):.2f} hours")
        print(f"Position 2 open duration: {(pos2_close - pos2_open)/(60*60*1000):.2f} hours")
        print(f"Expected total open_ms: {expected_total_hours:.2f} hours")
        print(f"Actual checkpoint open_ms: {checkpoint.open_ms/(60*60*1000):.2f} hours")
        
        self.assertEqual(
            checkpoint.open_ms,
            expected_total_open_ms,
            f"Checkpoint open_ms should be sum of position durations. "
            f"Expected {expected_total_hours:.2f}h, got {checkpoint.open_ms/(60*60*1000):.2f}h"
        )


if __name__ == '__main__':
    unittest.main()
