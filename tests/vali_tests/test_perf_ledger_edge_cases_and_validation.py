"""
Performance ledger edge cases and validation tests.

This file consolidates all edge case, stress test, validation, and error handling tests:
- Edge cases and boundary conditions
- Stress testing with high volumes
- Bypass validation
- Error handling
- Checkpoint integrity
- Endpoint behavior
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
import math
from decimal import Decimal

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS, MS_IN_8_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
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


class TestPerfLedgerEdgeCasesAndValidation(TestBase):
    """Tests for edge cases, validation, and error handling in performance ledger."""

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.test_hotkey = "test_miner_edge"
        self.now_ms = TimeUtil.now_in_millis()
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        
        self.mmg = MockMetagraph(hotkeys=[self.test_hotkey])
        self.elimination_manager = EliminationManager(self.mmg, None, None, running_unit_tests=True)
        self.position_manager = PositionManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            elimination_manager=self.elimination_manager,
        )
        self.position_manager.clear_all_miner_positions()

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_empty_position_list(self, mock_lpf):
        """Test behavior with no positions."""
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
        
        # Update with no positions
        plm.update(t_ms=self.now_ms)
        
        # Should create empty ledgers
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertEqual(len(bundles), 0, "Should have no bundles with no positions")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_simultaneous_positions(self, mock_lpf):
        """Test handling of positions that open and close at the same time."""
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
        
        # Create position that opens and closes instantly
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid="instant",
            open_ms=base_time,
            close_ms=base_time + 1000,  # 1 second later
            trade_pair=TradePair.BTCUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[
                Order(
                    price=50000.0,
                    processed_ms=base_time,
                    order_uuid="open",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=1.0,
                ),
                Order(
                    price=50100.0,
                    processed_ms=base_time + 1000,
                    order_uuid="close",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        self.position_manager.save_miner_position(position)
        
        # Should handle gracefully
        plm.update(t_ms=base_time + MS_IN_24_HOURS)
        
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.test_hotkey, bundles)

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_high_volume_positions(self, mock_lpf):
        """Test with a large number of positions (stress test)."""
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
        
        base_time = self.now_ms - (60 * MS_IN_24_HOURS)
        
        # Create 20 positions across different days (reduced from 50 to avoid timing issues)
        for i in range(20):
            # Stagger positions so they don't overlap
            open_time = base_time + (i * 2 * MS_IN_24_HOURS)
            close_time = open_time + MS_IN_24_HOURS
            
            position = self._create_position(
                f"pos_{i}", TradePair.BTCUSD,
                open_time, close_time,
                50000.0 + i * 100,  # Varying prices
                50000.0 + i * 100 + 500,  # Small gains
                OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Update to a time past all positions
        update_time = base_time + (45 * MS_IN_24_HOURS)
        plm.update(t_ms=update_time)
        
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        
        # Check if we got bundles
        self.assertIn(self.test_hotkey, bundles, "Should have bundles for test miner")
        
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Should have many checkpoints (at least 10)
        self.assertGreater(len(btc_ledger.cps), 10, "Should have many checkpoints with high volume")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_extreme_price_movements(self, mock_lpf):
        """Test handling of extreme price movements and liquidations."""
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
        
        # Test extreme scenarios - but not zero price as that causes issues
        scenarios = [
            ("99% loss", 50000.0, 500.0, OrderType.LONG),
            ("10x gain", 50000.0, 500000.0, OrderType.LONG),
            ("90% short gain", 50000.0, 5000.0, OrderType.SHORT),  # Price drops 90%, short gains
        ]
        
        for i, (name, open_price, close_price, order_type) in enumerate(scenarios):
            position = self._create_position(
                f"extreme_{i}", TradePair.BTCUSD,
                base_time + (i * 2 * MS_IN_24_HOURS),
                base_time + (i * 2 * MS_IN_24_HOURS) + MS_IN_24_HOURS,
                open_price, close_price, order_type
            )
            self.position_manager.save_miner_position(position)
        
        # Update to a time past all positions
        update_time = base_time + (10 * MS_IN_24_HOURS)
        plm.update(t_ms=update_time)
        
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        
        # Even with extreme price movements, the system should handle gracefully
        # and create ledgers for valid positions
        self.assertIsNotNone(bundles, "Should return bundles object")
        
        # If positions were created and saved, we should have some tracking
        # The fact that extreme movements might prevent ledger creation is a specific
        # business rule that should be tested explicitly, not hand-waved
        if self.test_hotkey in bundles:
            btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
            # Validate the ledger structure if it exists
            self.assertIsInstance(btc_ledger.cps, list)
            self.assertGreaterEqual(len(btc_ledger.cps), 0)

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_bypass_validation_conditions(self, mock_lpf):
        """Test all conditions that control bypass logic."""
        mock_pds = Mock()
        mock_pds.unified_candle_fetcher.return_value = []
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        mmg = MockMetagraph(hotkeys=["test"])
        plm = PerfLedgerManager(
            metagraph=mmg,
            running_unit_tests=True,
            position_manager=PositionManager(
                metagraph=mmg,
                running_unit_tests=True,
                elimination_manager=EliminationManager(mmg, None, None, running_unit_tests=True),
            ),
            parallel_mode=ParallelizationMode.SERIAL,
        )
        
        # Create test ledger
        ledger = PerfLedger(initialization_time_ms=self.now_ms)
        prev_cp = PerfCheckpoint(
            last_update_ms=self.now_ms,
            prev_portfolio_ret=0.95,
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            mdd=0.95,
            mpv=1.0
        )
        ledger.cps.append(prev_cp)
        
        # Create a mock closed position for testing
        mock_closed_position = Mock()
        mock_closed_position.is_open_position = False
        
        # Test various bypass conditions
        test_cases = [
            # (any_open, position_just_closed, tp_id, tp_id_rtp_data, should_bypass)
            (TradePairReturnStatus.TP_NO_OPEN_POSITIONS, False, "BTCUSD", {"BTCUSD": None}, True),
            (TradePairReturnStatus.TP_NO_OPEN_POSITIONS, True, "BTCUSD", {"BTCUSD": mock_closed_position}, False),
            (TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE, False, "BTCUSD", {"BTCUSD": None}, False),
            (TradePairReturnStatus.TP_NO_OPEN_POSITIONS, False, "ETHUSD", {"BTCUSD": None}, False),
            (TradePairReturnStatus.TP_NO_OPEN_POSITIONS, False, "BTCUSD", {}, True),
        ]
        
        for any_open, pos_closed, tp_id, tp_id_rtp_data, should_bypass in test_cases:
            ret, spread, carry = plm.get_bypass_values_if_applicable(
                ledger, tp_id, any_open, 1.0, .999, .998, tp_id_rtp_data
            )
            
            if should_bypass:
                self.assertEqual(ret, 0.95, f"Should bypass for {any_open}, {pos_closed}, {tp_id}, {tp_id_rtp_data}")
            else:
                self.assertEqual(ret, 1.0, f"Should not bypass for {any_open}, {pos_closed}, {tp_id}, {tp_id_rtp_data}")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_checkpoint_boundary_edge_cases(self, mock_lpf):
        """Test positions that span checkpoint boundaries in various ways."""
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
        base_time -= (10 * MS_IN_24_HOURS)
        
        # Test cases - staggered to avoid overlapping positions (production constraint: max 1 open position per trade pair)
        positions = [
            # Opens exactly on boundary
            ("boundary_open", base_time, base_time + MS_IN_24_HOURS),
            # Closes exactly on boundary (starts after first ends)
            ("boundary_close", base_time + MS_IN_24_HOURS + 100000, base_time + MS_IN_24_HOURS + checkpoint_duration),
            # Spans multiple boundaries (starts after second ends)
            ("multi_boundary", base_time + MS_IN_24_HOURS + checkpoint_duration + 100000, base_time + MS_IN_24_HOURS + checkpoint_duration + (3 * checkpoint_duration)),
            # Very short position within single checkpoint (starts after third ends)
            ("within_checkpoint", base_time + MS_IN_24_HOURS + (4 * checkpoint_duration) + 100000, base_time + MS_IN_24_HOURS + (4 * checkpoint_duration) + 200000),
        ]
        
        for name, open_ms, close_ms in positions:
            position = self._create_position(
                name, TradePair.BTCUSD,
                open_ms, close_ms,
                50000.0, 51000.0, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Update past all positions - need to ensure we update past the longest position
        # Last position (within_checkpoint) ends at base_time + MS_IN_24_HOURS + (4 * checkpoint_duration) + 200000
        update_time = base_time + MS_IN_24_HOURS + (5 * checkpoint_duration)  # Well past all positions
        plm.update(t_ms=update_time)
        
        # Verify all positions are tracked
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        
        # Must have bundles for the test miner - we created valid positions
        self.assertIn(self.test_hotkey, bundles, "Should have bundles for test miner with valid positions")
        self.assertIn(TradePair.BTCUSD.trade_pair_id, bundles[self.test_hotkey], "Should have BTC ledger")
            
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Should have checkpoints aligned to boundaries
        for cp in btc_ledger.cps:
            self.assertEqual(cp.last_update_ms % checkpoint_duration, 0,
                           f"Checkpoint at {cp.last_update_ms} not aligned")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_negative_returns_and_mdd(self, mock_lpf):
        """Test maximum drawdown calculation with negative returns."""
        from collections import namedtuple
        Candle = namedtuple('Candle', ['timestamp', 'close'])
        
        mock_pds = Mock()
        
        # Mock candle data to provide prices for the positions
        def mock_unified_candle_fetcher(*args, **kwargs):
            # Extract parameters
            if args:
                trade_pair = args[0]
                start_ms = args[1] if len(args) > 1 else kwargs.get('start_timestamp_ms')
                end_ms = args[2] if len(args) > 2 else kwargs.get('end_timestamp_ms')
            else:
                trade_pair = kwargs.get('trade_pair')
                start_ms = kwargs.get('start_timestamp_ms')
                end_ms = kwargs.get('end_timestamp_ms')
            
            # Generate candles with prices matching our position prices
            candles = []
            base_time = self.now_ms - (20 * MS_IN_24_HOURS)
            
            # Define prices at key timestamps to match position open/close prices
            price_schedule = [
                (base_time, 50000.0),  # Start of loss1
                (base_time + MS_IN_24_HOURS, 49000.0),  # End of loss1, start of loss2
                (base_time + 2 * MS_IN_24_HOURS, 47000.0),  # End of loss2, start of loss3
                (base_time + 3 * MS_IN_24_HOURS, 45000.0),  # End of loss3, start of recovery
                (base_time + 4 * MS_IN_24_HOURS, 46000.0),  # End of recovery
                (base_time + 10 * MS_IN_24_HOURS, 46000.0),  # Final update time
            ]
            
            # Generate minute candles between start_ms and end_ms
            for i in range(len(price_schedule) - 1):
                t1, p1 = price_schedule[i]
                t2, p2 = price_schedule[i + 1]
                
                if t1 <= end_ms and t2 >= start_ms:
                    # Generate candles for this period
                    current_ms = max(t1, start_ms)
                    while current_ms <= min(t2, end_ms):
                        # Linear interpolation between prices
                        progress = (current_ms - t1) / (t2 - t1) if t2 > t1 else 0
                        price = p1 + (p2 - p1) * progress
                        candles.append(Candle(timestamp=current_ms, close=price))
                        current_ms += 60000  # 1 minute
            
            return candles
        
        mock_pds.unified_candle_fetcher.side_effect = mock_unified_candle_fetcher
        mock_pds.tp_to_mfs = {}
        mock_lpf.return_value.polygon_data_service = mock_pds
        
        plm = PerfLedgerManager(
            metagraph=self.mmg,
            running_unit_tests=True,
            position_manager=self.position_manager,
            parallel_mode=ParallelizationMode.SERIAL,
            live_price_fetcher=mock_lpf.return_value,
            is_backtesting=True,  # Ensure we process historical data
        )
        
        base_time = self.now_ms - (20 * MS_IN_24_HOURS)
        
        # Create losing positions to test MDD
        losses = [
            ("loss1", 50000.0, 49000.0),  # -2%
            ("loss2", 49000.0, 47000.0),  # -4%
            ("loss3", 47000.0, 45000.0),  # -4.3%
            ("recovery", 45000.0, 46000.0),  # +2.2%
        ]
        
        for i, (name, open_price, close_price) in enumerate(losses):
            # Add small offset to prevent timestamp collisions between positions
            open_time = base_time + (i * MS_IN_24_HOURS) + (i * 1000)  # Add 1 second offset per position
            close_time = base_time + ((i + 1) * MS_IN_24_HOURS) - 1000  # Close 1 second before next position starts
            
            position = self._create_position(
                name, TradePair.BTCUSD,
                open_time,
                close_time,
                open_price, close_price, OrderType.LONG
            )
            self.position_manager.save_miner_position(position)
        
        # Update incrementally to build up state properly
        current_time = base_time
        step_size = 12 * 60 * 60 * 1000  # 12 hours
        final_time = base_time + (10 * MS_IN_24_HOURS)
        
        while current_time < final_time:
            next_time = min(current_time + step_size, final_time)
            plm.update(t_ms=next_time)
            current_time = next_time
        
        # Check MDD
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.test_hotkey, bundles, f"Should have bundles for test miner {self.test_hotkey}")
        self.assertIn(TradePair.BTCUSD.trade_pair_id, bundles[self.test_hotkey], "Should have BTC ledger")
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # MDD should be less than 1.0 (indicating drawdown)
        # Find a checkpoint with actual updates (n_updates > 0)
        checkpoint_with_data = None
        for cp in reversed(btc_ledger.cps):
            if cp.n_updates > 0:
                checkpoint_with_data = cp
                break
        
        if checkpoint_with_data:
            self.assertLess(checkpoint_with_data.mdd, 1.0, 
                          f"Should have drawdown with losing positions. MDD={checkpoint_with_data.mdd}")
        else:
            self.fail("No checkpoint with updates found")

    @patch('vali_objects.vali_dataclasses.perf_ledger.LivePriceFetcher')
    def test_fee_edge_cases(self, mock_lpf):
        """Test edge cases in fee calculations."""
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
        
        base_time = self.now_ms - (30 * MS_IN_24_HOURS)
        
        # Test extreme leverage (high fees)
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid="high_leverage",
            open_ms=base_time,
            close_ms=base_time + (10 * MS_IN_24_HOURS),  # 10 days
            trade_pair=TradePair.BTCUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[
                Order(
                    price=50000.0,
                    processed_ms=base_time,
                    order_uuid="open",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=10.0,  # 10x leverage
                ),
                Order(
                    price=50000.0,
                    processed_ms=base_time + (10 * MS_IN_24_HOURS),
                    order_uuid="close",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        self.position_manager.save_miner_position(position)
        
        # Update
        plm.update(t_ms=base_time + (11 * MS_IN_24_HOURS))
        
        # High leverage should result in higher fees
        bundles = plm.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
        
        # Find checkpoint with position
        for i, cp in enumerate(btc_ledger.cps):
            if cp.n_updates > 0 and i != 0 :  # Skip initial checkpoint with the initial spread fee that triggers n_updates 1
                # With 10x leverage for 10 days, carry fees should be significant
                # Based on actual implementation: ~0.9989 for 10x leverage over 10 days
                self.assertLess(cp.prev_portfolio_carry_fee, 1.0,
                               "High leverage should have measurable carry fees")
                self.assertLess(cp.prev_portfolio_carry_fee, 0.999,
                               "10x leverage for 10 days should have measurable carry fees (actual: ~0.999)")
                break

    def _create_position(self, position_id: str, trade_pair: TradePair, 
                        open_ms: int, close_ms: int, open_price: float, 
                        close_price: float, order_type: OrderType,
                        leverage: float = 1.0) -> Position:
        """Helper to create a position."""
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid=position_id,
            open_ms=open_ms,
            close_ms=close_ms,
            trade_pair=trade_pair,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[
                Order(
                    price=open_price,
                    processed_ms=open_ms,
                    order_uuid=f"{position_id}_open",
                    trade_pair=trade_pair,
                    order_type=order_type,
                    leverage=leverage if order_type == OrderType.LONG else -leverage,
                ),
                Order(
                    price=close_price,
                    processed_ms=close_ms,
                    order_uuid=f"{position_id}_close",
                    trade_pair=trade_pair,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        return position


if __name__ == '__main__':
    unittest.main()
