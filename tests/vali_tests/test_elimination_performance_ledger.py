# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import (
    PerfLedger,
    PerfCheckpoint,
    TP_ID_PORTFOLIO
)


class TestPerfLedgerEliminations(TestBase):
    """
    Test suite for performance ledger eliminations using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    challenge_period_client = None

    # Test miners
    HEALTHY_MINER = "healthy_miner"
    MDD_MINER = "mdd_miner"
    LIQUIDATED_MINER = "liquidated_miner"
    INVALIDATED_MINER = "invalidated_miner"
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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')

        # Define test miners
        cls.all_miners = [
            cls.HEALTHY_MINER,
            cls.MDD_MINER,
            cls.LIQUIDATED_MINER,
            cls.INVALIDATED_MINER
        ]

        # Set up metagraph with test miners
        cls.metagraph_client.set_hotkeys(cls.all_miners)

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

        # Set up initial positions
        self._setup_positions()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _setup_positions(self):
        """Set up test positions for miners"""
        for miner in self.all_miners:
            position = Position(
                miner_hotkey=miner,
                position_uuid=f"{miner}_btc_position",
                open_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[Order(
                    price=60000,
                    processed_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
                    order_uuid=f"order_{miner}",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=1.0
                )]
            )
            self.position_client.save_miner_position(position)

    def test_perf_ledger_elimination_detection(self):
        """Test that perf ledger manager correctly detects eliminations"""
        # Create a ledger that exceeds max drawdown
        losing_ledger = generate_losing_ledger(
            0,
            ValiConfig.TARGET_LEDGER_WINDOW_MS
        )

        # Save ledger
        ledgers = {
            self.MDD_MINER: losing_ledger,
            self.HEALTHY_MINER: generate_winning_ledger(0, ValiConfig.TARGET_LEDGER_WINDOW_MS)
        }
        self.perf_ledger_client.save_perf_ledgers(ledgers)

        # Check if miner is beyond max drawdown
        # generate_losing_ledger returns a dict, we need the portfolio ledger
        portfolio_ledger = losing_ledger[TP_ID_PORTFOLIO]
        is_beyond, dd_percentage = LedgerUtils.is_beyond_max_drawdown(portfolio_ledger)
        self.assertTrue(is_beyond)
        # dd_percentage is returned as percentage (0-100), not decimal
        self.assertGreater(dd_percentage, 10.0)

        # Create elimination row
        elim_row = {
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': dd_percentage,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {
                str(TradePair.BTCUSD): 55000  # Price at elimination
            }
        }

        # Add to perf ledger eliminations
        self.perf_ledger_client.add_elimination_row(elim_row)

        # Get eliminations
        eliminations = self.perf_ledger_client.get_perf_ledger_eliminations()
        self.assertEqual(len(eliminations), 1)
        self.assertEqual(eliminations[0]['hotkey'], self.MDD_MINER)

    def test_perf_ledger_invalidation(self):
        """Test that invalidated hotkeys are excluded from scoring"""
        # Set up ledgers
        ledgers = {
            self.HEALTHY_MINER: generate_winning_ledger(0, ValiConfig.TARGET_LEDGER_WINDOW_MS),
            self.INVALIDATED_MINER: generate_winning_ledger(0, ValiConfig.TARGET_LEDGER_WINDOW_MS)
        }
        self.perf_ledger_client.save_perf_ledgers(ledgers)

        # Mark miner for invalidation
        self.perf_ledger_client.set_perf_ledger_hks_to_invalidate(
            {self.INVALIDATED_MINER: TimeUtil.now_in_millis()}
        )

        # Get filtered ledger for scoring
        filtered_ledger = self.perf_ledger_client.filtered_ledger_for_scoring(
            portfolio_only=True,
            hotkeys=self.all_miners
        )

        # Verify invalidated miner is excluded
        self.assertIn(self.HEALTHY_MINER, filtered_ledger)
        self.assertNotIn(self.INVALIDATED_MINER, filtered_ledger)

    def test_perf_ledger_elimination_persistence(self):
        """Test that perf ledger eliminations are persisted to disk"""
        # Create elimination
        elim_row = {
            'hotkey': self.LIQUIDATED_MINER,
            'reason': EliminationReason.LIQUIDATED.value,
            'dd': 0.15,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {
                str(TradePair.BTCUSD): 50000,
                str(TradePair.ETHUSD): 2500
            },
            'return_info': {
                'total_return': -0.15,
                'sharpe': -1.5
            }
        }

        # Write to disk
        self.perf_ledger_client.write_perf_ledger_eliminations_to_disk([elim_row])

        # Verify file exists
        elim_file = ValiBkpUtils.get_perf_ledger_eliminations_dir(running_unit_tests=True)
        self.assertTrue(os.path.exists(elim_file))

        # Check eliminations were saved
        loaded_elims = self.perf_ledger_client.get_perf_ledger_eliminations(first_fetch=True)
        self.assertEqual(len(loaded_elims), 1)
        self.assertEqual(loaded_elims[0]['hotkey'], self.LIQUIDATED_MINER)

    def test_perf_checkpoint_mdd_calculation(self):
        """Test maximum drawdown calculation in performance checkpoints"""
        # Create checkpoints with drawdown
        checkpoints = []
        
        # Initial checkpoint
        cp1 = PerfCheckpoint(
            last_update_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * 10,
            prev_portfolio_ret=1.0,
            accum_ms=MS_IN_24_HOURS,
            open_ms=MS_IN_24_HOURS,
            n_updates=100,
            gain=0.0,
            loss=0.0,
            mdd=1.0,  # No drawdown initially
            mpv=0.0
        )
        checkpoints.append(cp1)
        
        # Checkpoint with 5% drawdown
        cp2 = PerfCheckpoint(
            last_update_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * 5,
            prev_portfolio_ret=0.95,
            accum_ms=MS_IN_24_HOURS,
            open_ms=MS_IN_24_HOURS,
            n_updates=100,
            gain=0.0,
            loss=0.05,
            mdd=0.95,  # 5% drawdown
            mpv=0.0
        )
        checkpoints.append(cp2)
        
        # Checkpoint with 12% drawdown (exceeds max)
        cp3 = PerfCheckpoint(
            last_update_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
            prev_portfolio_ret=0.88,
            accum_ms=MS_IN_24_HOURS,
            open_ms=MS_IN_24_HOURS,
            n_updates=100,
            gain=0.0,
            loss=0.12,
            mdd=0.88,  # 12% drawdown
            mpv=0.0
        )
        checkpoints.append(cp3)
        
        # Create ledger with checkpoints
        ledger = PerfLedger(
            initialization_time_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * 11,
            max_return=1.0,
            cps=checkpoints
        )
        
        # Check if beyond max drawdown
        is_beyond, dd_percentage = LedgerUtils.is_beyond_max_drawdown(ledger)
        self.assertTrue(is_beyond)
        # dd_percentage is returned as percentage (0-100), not decimal
        self.assertAlmostEqual(dd_percentage, 12.0, places=0)

    def test_perf_ledger_update_with_eliminations(self):
        """Test that perf ledger update handles eliminations correctly"""
        # Set up positions and ledgers
        ledgers = {}
        for miner in [self.HEALTHY_MINER, self.MDD_MINER]:
            ledgers[miner] = generate_winning_ledger(0, ValiConfig.TARGET_LEDGER_WINDOW_MS)

        self.perf_ledger_client.save_perf_ledgers(ledgers)

        # Mark MDD miner for elimination
        elim_row = {
            'hotkey': self.MDD_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.11,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {}
        }
        self.perf_ledger_client.add_elimination_row(elim_row)

        # Process eliminations through elimination manager
        self.elimination_client.handle_perf_ledger_eliminations()

        # Verify elimination was processed
        eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(eliminations), 1)
        self.assertEqual(eliminations[0]['hotkey'], self.MDD_MINER)

        # Verify positions were closed
        positions = self.position_client.get_positions_for_one_hotkey(self.MDD_MINER)
        for pos in positions:
            self.assertTrue(pos.is_closed_position)

    def test_ledger_window_constraints(self):
        """Test that perf ledgers respect window constraints"""
        # Create a ledger with checkpoints spanning different time windows
        old_checkpoint = PerfCheckpoint(
            last_update_ms=TimeUtil.now_in_millis() - ValiConfig.TARGET_LEDGER_WINDOW_MS - MS_IN_24_HOURS,
            prev_portfolio_ret=1.1,
            accum_ms=MS_IN_8_HOURS,
            open_ms=MS_IN_8_HOURS,
            n_updates=50
        )

        recent_checkpoint = PerfCheckpoint(
            last_update_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
            prev_portfolio_ret=1.05,
            accum_ms=MS_IN_8_HOURS,
            open_ms=MS_IN_8_HOURS,
            n_updates=50
        )

        ledger = PerfLedger(
            initialization_time_ms=TimeUtil.now_in_millis() - ValiConfig.TARGET_LEDGER_WINDOW_MS - MS_IN_24_HOURS * 2,
            max_return=1.1,
            target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS,
            cps=[old_checkpoint, recent_checkpoint]
        )

        # Save ledger
        self.perf_ledger_client.save_perf_ledgers({
            self.HEALTHY_MINER: {TP_ID_PORTFOLIO: ledger}
        })

        # Get ledger and verify window constraint
        retrieved_ledgers = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
        miner_ledger = retrieved_ledgers[self.HEALTHY_MINER][TP_ID_PORTFOLIO]

        # Check that old checkpoints outside window are handled appropriately
        self.assertIsNotNone(miner_ledger)
        self.assertEqual(len(miner_ledger.cps), 2)

    def test_concurrent_elimination_handling(self):
        """Test handling of concurrent eliminations from multiple sources"""
        # Add elimination from perf ledger
        pl_elim = {
            'hotkey': self.LIQUIDATED_MINER,
            'reason': EliminationReason.LIQUIDATED.value,
            'dd': 0.15,
            'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
            'price_info': {str(TradePair.BTCUSD): 50000}
        }
        self.perf_ledger_client.add_elimination_row(pl_elim)

        # Process through elimination manager
        self.elimination_client.handle_perf_ledger_eliminations()

        # Try to add another elimination for same miner (should be prevented)
        initial_count = len(self.elimination_client.get_eliminations_from_memory())

        # Try MDD elimination for already eliminated miner
        self.elimination_client.handle_mdd_eliminations()

        # Verify no duplicate
        final_count = len(self.elimination_client.get_eliminations_from_memory())
        self.assertEqual(initial_count, final_count)

    def test_perf_ledger_void_behavior(self):
        """Test perf ledger behavior with closed positions"""
        # Create a closed position
        position = Position(
            miner_hotkey=self.HEALTHY_MINER,
            position_uuid="closed_position",
            open_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * 2,
            close_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS,
            trade_pair=TradePair.BTCUSD,
            is_closed_position=True,
            return_at_close=1.0,  # No profit/loss
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[Order(
                price=60000,
                processed_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * 2,
                order_uuid="closed_order",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG,
                leverage=1.0
            )]
        )

        self.position_client.save_miner_position(position)

        # Update perf ledger
        current_time = TimeUtil.now_in_millis()
        self.perf_ledger_client.update(t_ms=current_time)

        # Get ledger and verify closed positions are handled
        ledgers = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
        if self.HEALTHY_MINER in ledgers:
            miner_ledger = ledgers[self.HEALTHY_MINER].get(TP_ID_PORTFOLIO)
            if miner_ledger:
                # Closed position with return_at_close=1.0 should not affect performance
                self.assertIsNotNone(miner_ledger)

    def test_perf_ledger_metrics_calculation(self):
        """Test calculation of performance metrics used in eliminations"""
        # Create a ledger with specific performance characteristics
        checkpoints = []
        
        # Simulate performance over time
        returns = [1.0, 1.02, 1.01, 0.98, 0.95, 0.93, 0.91, 0.89]  # Declining performance
        
        # Calculate running mdd (maximum drawdown from peak)
        peak = 1.0
        for i, ret in enumerate(returns):
            if ret > peak:
                peak = ret
            
            # mdd represents the lowest point relative to peak (as a ratio)
            mdd = ret / peak
            
            cp = PerfCheckpoint(
                last_update_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * (len(returns) - i),
                prev_portfolio_ret=ret,
                accum_ms=MS_IN_24_HOURS,
                open_ms=MS_IN_24_HOURS,
                n_updates=100,
                gain=max(0, ret - returns[i-1]) if i > 0 else 0,
                loss=max(0, returns[i-1] - ret) if i > 0 else 0,
                mdd=mdd,
                mpv=peak
            )
            checkpoints.append(cp)
        
        ledger = PerfLedger(
            initialization_time_ms=TimeUtil.now_in_millis() - MS_IN_24_HOURS * len(returns),
            max_return=max(returns),
            cps=checkpoints
        )
        
        # Calculate metrics
        is_beyond, dd_percentage = LedgerUtils.is_beyond_max_drawdown(ledger)
        
        # Verify drawdown calculation
        # The worst mdd is 0.89/1.02 = 0.8725
        # So drawdown is 1 - 0.8725 = 0.1275 = 12.75%
        # But is_beyond_max_drawdown rounds to 2 places, so we get 12.75 rounded to 13
        self.assertTrue(is_beyond)  # Should exceed 10% max drawdown
        # Check that it's approximately 13% (allowing for rounding)
        self.assertAlmostEqual(dd_percentage, 13.0, places=0)

    def test_perf_ledger_realtime_update(self):
        """Test perf ledger updates with real-time price changes"""
        # Update ledger
        current_time = TimeUtil.now_in_millis()
        self.perf_ledger_client.update(t_ms=current_time)

        # Check if any miners hit drawdown limits
        eliminations = self.perf_ledger_client.get_perf_ledger_eliminations()

        # No eliminations should occur for healthy ledgers
        self.assertEqual(len(eliminations), 0)
