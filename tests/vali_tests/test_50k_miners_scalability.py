"""
Scalability tests for 50k miners.

Tests the scoring system's ability to handle 50,000 miners without running out
of memory or timing out. Uses synthetic data generation to stress test:
- Ledger generation and storage
- Scoring pipeline (metrics, penalties, softmax, normalization)
- Memory usage patterns
- Execution time

Run with:
    python tests/run_vali_testing_suite.py test_50k_miners_scalability.py
"""

import gc
import math
import os
import sys
import time
import tracemalloc
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple
from unittest.mock import patch, MagicMock

import numpy as np

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position_management.position_utils.position_penalties import PositionPenalties
from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_config import TradePair, TradePairCategory, ValiConfig
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger, TP_ID_PORTFOLIO
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position


@dataclass
class ScaleTestResult:
    """Results from a scalability test run."""
    n_miners: int
    memory_peak_mb: float
    memory_current_mb: float
    execution_time_s: float
    scores_generated: int
    success: bool
    error_message: str = ""


class SyntheticDataGenerator:
    """Generates synthetic miner data for stress testing."""

    # Fixed timestamps for deterministic tests
    BASE_TIME_MS = 1735689600000  # 2024-01-01 00:00:00 UTC
    CHECKPOINT_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    @classmethod
    def generate_hotkey(cls, index: int) -> str:
        """Generate a deterministic hotkey for a miner index."""
        return f"5F{'0' * 40}{index:06d}"[-48:]  # 48 char hotkey format

    @classmethod
    def generate_checkpoint(
        cls,
        time_offset_ms: int,
        gain: float = 0.001,
        loss: float = -0.0005,
        mdd: float = 0.98,
        realized_pnl: float = 100.0,
        unrealized_pnl: float = 50.0,
    ) -> PerfCheckpoint:
        """Generate a single checkpoint with realistic values."""
        return PerfCheckpoint(
            last_update_ms=cls.BASE_TIME_MS + time_offset_ms,
            prev_portfolio_ret=1.0 + (gain + loss) * (time_offset_ms / cls.CHECKPOINT_DURATION_MS),
            prev_portfolio_realized_pnl=realized_pnl,
            prev_portfolio_unrealized_pnl=unrealized_pnl,
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            accum_ms=cls.CHECKPOINT_DURATION_MS,
            open_ms=cls.CHECKPOINT_DURATION_MS,
            n_updates=100,
            gain=max(gain, 0),
            loss=min(loss, 0),
            spread_fee_loss=-0.001,
            carry_fee_loss=-0.002,
            mdd=mdd,
            mpv=1.0 + gain,
            realized_pnl=realized_pnl / 240,  # Daily split
            unrealized_pnl=unrealized_pnl,
        )

    @classmethod
    def generate_ledger(
        cls,
        n_days: int = 120,
        performance_profile: str = "average",
        seed: int = None,
    ) -> PerfLedger:
        """
        Generate a realistic PerfLedger with specified number of days.

        Args:
            n_days: Number of days of history (default 120 = full window)
            performance_profile: One of "winning", "losing", "average", "volatile"
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # Performance profiles determine daily return distribution
        profiles = {
            "winning": {"mean": 0.002, "std": 0.01, "mdd_base": 0.97},
            "losing": {"mean": -0.001, "std": 0.015, "mdd_base": 0.92},
            "average": {"mean": 0.0005, "std": 0.012, "mdd_base": 0.95},
            "volatile": {"mean": 0.001, "std": 0.025, "mdd_base": 0.90},
        }
        profile = profiles.get(performance_profile, profiles["average"])

        checkpoints = []
        n_checkpoints = n_days * ValiConfig.DAILY_CHECKPOINTS

        portfolio_value = 1.0
        max_portfolio_value = 1.0
        cumulative_realized_pnl = 0.0

        for i in range(n_checkpoints):
            # Generate realistic daily return
            daily_return = np.random.normal(profile["mean"], profile["std"])

            # Split into gain/loss
            gain = max(daily_return, 0)
            loss = min(daily_return, 0)

            # Update portfolio value
            portfolio_value *= math.exp(daily_return)
            max_portfolio_value = max(max_portfolio_value, portfolio_value)

            # Calculate drawdown
            mdd = min(portfolio_value / max_portfolio_value, profile["mdd_base"])

            # Realistic PnL values (scale with account size)
            realized_delta = np.random.normal(50, 20) if daily_return > 0 else np.random.normal(-30, 15)
            cumulative_realized_pnl += realized_delta
            unrealized_pnl = np.random.normal(0, 100)

            checkpoint = PerfCheckpoint(
                last_update_ms=cls.BASE_TIME_MS + (i + 1) * cls.CHECKPOINT_DURATION_MS,
                prev_portfolio_ret=portfolio_value,
                prev_portfolio_realized_pnl=cumulative_realized_pnl,
                prev_portfolio_unrealized_pnl=unrealized_pnl,
                prev_portfolio_spread_fee=0.999,
                prev_portfolio_carry_fee=0.998,
                accum_ms=cls.CHECKPOINT_DURATION_MS,
                open_ms=cls.CHECKPOINT_DURATION_MS,
                n_updates=np.random.randint(50, 150),
                gain=gain,
                loss=loss,
                spread_fee_loss=-0.0005,
                carry_fee_loss=-0.001,
                mdd=mdd,
                mpv=max_portfolio_value,
                realized_pnl=realized_delta,
                unrealized_pnl=unrealized_pnl,
            )
            checkpoints.append(checkpoint)

        return PerfLedger(
            initialization_time_ms=cls.BASE_TIME_MS,
            max_return=max_portfolio_value,
            target_cp_duration_ms=cls.CHECKPOINT_DURATION_MS,
            target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS,
            cps=checkpoints,
            tp_id=TP_ID_PORTFOLIO,
        )

    @classmethod
    def generate_order(
        cls,
        hotkey: str,
        position_uuid: str,
        order_type: OrderType,
        leverage: float,
        price: float,
        processed_ms: int,
        trade_pair: TradePair,
    ) -> Order:
        """Generate a single order."""
        return Order(
            order_type=order_type,
            leverage=leverage,
            price=price,
            trade_pair=trade_pair,
            processed_ms=processed_ms,
            order_uuid=f"{position_uuid}_{processed_ms}",
        )

    @classmethod
    def generate_position(
        cls,
        hotkey: str,
        trade_pair: TradePair = TradePair.BTCUSD,
        is_closed: bool = True,
        n_orders: int = 3,
        seed: int = None,
    ) -> Position:
        """Generate a realistic position with orders."""
        if seed is not None:
            np.random.seed(seed)

        position_uuid = f"pos_{hotkey}_{seed or 0}"
        open_ms = cls.BASE_TIME_MS + np.random.randint(0, 30 * 24 * 60 * 60 * 1000)

        # Generate orders
        orders = []
        entry_leverage = np.random.uniform(0.05, 0.3) if trade_pair.is_crypto else np.random.uniform(0.5, 3.0)
        entry_price = 50000 if trade_pair.is_crypto else 1.1

        # Entry order - leverage sign must match order type
        is_long = np.random.random() > 0.5
        entry_order_type = OrderType.LONG if is_long else OrderType.SHORT
        entry_order_leverage = entry_leverage if is_long else -entry_leverage

        entry_order = Order(
            order_type=entry_order_type,
            leverage=entry_order_leverage,
            price=entry_price,
            trade_pair=trade_pair,
            processed_ms=open_ms,
            order_uuid=f"{position_uuid}_0",
        )
        orders.append(entry_order)

        # Additional orders - same direction as entry
        current_time = open_ms
        for i in range(1, n_orders - 1):
            current_time += np.random.randint(1000 * 60, 1000 * 60 * 60 * 24)  # 1 min to 24 hours
            add_leverage = abs(entry_order_leverage) * np.random.uniform(0.2, 0.5)
            if not is_long:
                add_leverage = -add_leverage
            order = Order(
                order_type=entry_order_type,
                leverage=add_leverage,
                price=entry_price * np.random.uniform(0.95, 1.05),
                trade_pair=trade_pair,
                processed_ms=current_time,
                order_uuid=f"{position_uuid}_{i}",
            )
            orders.append(order)

        # Exit order
        if is_closed:
            close_ms = current_time + np.random.randint(1000 * 60, 1000 * 60 * 60 * 24)
            exit_order = Order(
                order_type=OrderType.FLAT,
                leverage=-sum(o.leverage for o in orders),
                price=entry_price * np.random.uniform(0.9, 1.1),
                trade_pair=trade_pair,
                processed_ms=close_ms,
                order_uuid=f"{position_uuid}_exit",
            )
            orders.append(exit_order)
        else:
            close_ms = None

        position = Position(
            miner_hotkey=hotkey,
            position_uuid=position_uuid,
            open_ms=open_ms,
            trade_pair=trade_pair,
            orders=orders,
            current_return=np.random.uniform(0.95, 1.1),
            close_ms=close_ms,
            net_leverage=0 if is_closed else entry_leverage,
            return_at_close=np.random.uniform(0.9, 1.15) if is_closed else 1.0,
            is_closed_position=is_closed,
        )

        return position

    @classmethod
    def generate_miner_data(
        cls,
        n_miners: int,
        n_days: int = 120,
        n_positions_per_miner: int = 10,
        performance_distribution: Dict[str, float] = None,
    ) -> Tuple[Dict[str, Dict[str, PerfLedger]], Dict[str, List[Position]], Dict[str, float]]:
        """
        Generate complete synthetic data for n miners.

        Returns:
            ledger_dict: Dict mapping hotkey -> {tp_id -> PerfLedger}
            positions_dict: Dict mapping hotkey -> List[Position]
            account_sizes: Dict mapping hotkey -> account_size (float)
        """
        if performance_distribution is None:
            performance_distribution = {
                "winning": 0.2,
                "average": 0.5,
                "losing": 0.2,
                "volatile": 0.1,
            }

        ledger_dict = {}
        positions_dict = {}
        account_sizes = {}

        # Cumulative distribution for profile selection
        profiles = list(performance_distribution.keys())
        cumulative_probs = np.cumsum(list(performance_distribution.values()))

        for i in range(n_miners):
            hotkey = cls.generate_hotkey(i)

            # Select performance profile
            r = np.random.random()
            profile_idx = np.searchsorted(cumulative_probs, r)
            profile = profiles[min(profile_idx, len(profiles) - 1)]

            # Generate portfolio ledger
            portfolio_ledger = cls.generate_ledger(n_days=n_days, performance_profile=profile, seed=i)

            # Also generate a crypto-specific ledger (subset of portfolio)
            crypto_ledger = cls.generate_ledger(n_days=n_days, performance_profile=profile, seed=i + n_miners)
            crypto_ledger.tp_id = TradePair.BTCUSD.trade_pair_id

            # Generate forex ledger
            forex_ledger = cls.generate_ledger(n_days=n_days, performance_profile=profile, seed=i + 2 * n_miners)
            forex_ledger.tp_id = TradePair.EURUSD.trade_pair_id

            ledger_dict[hotkey] = {
                TP_ID_PORTFOLIO: portfolio_ledger,
                TradePair.BTCUSD.trade_pair_id: crypto_ledger,
                TradePair.EURUSD.trade_pair_id: forex_ledger,
            }

            # Generate positions
            positions = []
            for j in range(n_positions_per_miner):
                trade_pair = TradePair.BTCUSD if j % 2 == 0 else TradePair.EURUSD
                position = cls.generate_position(
                    hotkey=hotkey,
                    trade_pair=trade_pair,
                    is_closed=j < n_positions_per_miner - 1,  # Last position open
                    n_orders=np.random.randint(2, 6),
                    seed=i * n_positions_per_miner + j,
                )
                positions.append(position)

            positions_dict[hotkey] = positions

            # Account sizes - most miners should have adequate collateral
            if np.random.random() < 0.9:  # 90% have adequate collateral
                account_sizes[hotkey] = np.random.uniform(150000, 500000)
            else:  # 10% below minimum
                account_sizes[hotkey] = np.random.uniform(50000, 140000)

        return ledger_dict, positions_dict, account_sizes


class Test50kMinersScalability(TestBase):
    """Test suite for 50k miner scalability."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        # Disable bittensor logging to reduce noise
        import logging
        logging.getLogger('bittensor').setLevel(logging.WARNING)

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        gc.collect()

    def tearDown(self):
        """Clean up after tests."""
        gc.collect()
        super().tearDown()

    def _run_scale_test(
        self,
        n_miners: int,
        n_days: int = 120,
        n_positions_per_miner: int = 5,
    ) -> ScaleTestResult:
        """
        Run a scalability test with the given number of miners.

        Returns ScaleTestResult with timing and memory metrics.
        """
        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        start_time = time.time()
        error_message = ""
        scores_generated = 0
        success = True

        try:
            # Generate synthetic data
            print(f"\n  Generating data for {n_miners:,} miners...")
            gen_start = time.time()

            ledger_dict, positions_dict, account_sizes = SyntheticDataGenerator.generate_miner_data(
                n_miners=n_miners,
                n_days=n_days,
                n_positions_per_miner=n_positions_per_miner,
            )

            gen_time = time.time() - gen_start
            print(f"  Data generation took {gen_time:.2f}s")

            # Calculate asset class min days
            asset_classes = [TradePairCategory.CRYPTO, TradePairCategory.FOREX]
            asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
                ledger_dict, asset_classes
            )

            # Run scoring pipeline
            print(f"  Running scoring pipeline...")
            score_start = time.time()

            # Mock external dependencies
            with patch.object(
                PositionPenalties, 'risk_profile_penalty', return_value=1.0
            ), patch(
                'vali_objects.contract.validator_contract_manager.ValidatorContractManager.min_collateral_penalty',
                side_effect=lambda x: 1.0 if x >= ValiConfig.MIN_COLLATERAL_VALUE else 0.0
            ):
                results = Scoring.compute_results_checkpoint(
                    ledger_dict=ledger_dict,
                    full_positions=positions_dict,
                    asset_class_min_days=asset_class_min_days,
                    evaluation_time_ms=SyntheticDataGenerator.BASE_TIME_MS + n_days * ValiConfig.DAILY_MS,
                    verbose=False,
                    weighting=False,
                    all_miner_account_sizes=account_sizes,
                )

            score_time = time.time() - score_start
            print(f"  Scoring took {score_time:.2f}s")

            scores_generated = len(results)

            # Verify results
            if scores_generated == 0:
                error_message = "No scores generated"
                success = False
            else:
                # Check score normalization (should sum to ~1.0)
                total_score = sum(score for _, score in results)
                if not (0.99 <= total_score <= 1.01):
                    error_message = f"Scores don't sum to 1.0: {total_score}"
                    success = False

                # Verify no NaN or Inf values
                for hotkey, score in results:
                    if math.isnan(score) or math.isinf(score):
                        error_message = f"Invalid score for {hotkey}: {score}"
                        success = False
                        break

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            success = False
            print(f"  ERROR: {error_message}")

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = time.time() - start_time

        return ScaleTestResult(
            n_miners=n_miners,
            memory_peak_mb=peak / 1024 / 1024,
            memory_current_mb=current / 1024 / 1024,
            execution_time_s=execution_time,
            scores_generated=scores_generated,
            success=success,
            error_message=error_message,
        )

    def test_baseline_1k_miners(self):
        """Test baseline performance with 1,000 miners."""
        result = self._run_scale_test(n_miners=1000, n_days=60, n_positions_per_miner=3)

        print(f"\n  === 1K Miners Results ===")
        print(f"  Execution time: {result.execution_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Scores generated: {result.scores_generated:,}")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertEqual(result.scores_generated, 1000)
        # Should complete in under 30 seconds
        self.assertLess(result.execution_time_s, 30, "1K miners took too long")
        # Should use less than 1GB memory
        self.assertLess(result.memory_peak_mb, 1024, "1K miners used too much memory")

    def test_medium_10k_miners(self):
        """Test medium scale with 10,000 miners."""
        result = self._run_scale_test(n_miners=10000, n_days=60, n_positions_per_miner=3)

        print(f"\n  === 10K Miners Results ===")
        print(f"  Execution time: {result.execution_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Scores generated: {result.scores_generated:,}")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertEqual(result.scores_generated, 10000)
        # Should complete in under 5 minutes
        self.assertLess(result.execution_time_s, 300, "10K miners took too long")
        # Should use less than 4GB memory
        self.assertLess(result.memory_peak_mb, 4096, "10K miners used too much memory")

    def test_large_50k_miners(self):
        """Test large scale with 50,000 miners - the main scalability target."""
        result = self._run_scale_test(n_miners=50000, n_days=60, n_positions_per_miner=3)

        print(f"\n  === 50K Miners Results ===")
        print(f"  Execution time: {result.execution_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Scores generated: {result.scores_generated:,}")
        print(f"  Memory per miner: {result.memory_peak_mb * 1024 / 50000:.2f} KB")
        print(f"  Time per miner: {result.execution_time_s * 1000 / 50000:.4f} ms")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertEqual(result.scores_generated, 50000)
        # Should complete in under 30 minutes (1800 seconds)
        self.assertLess(result.execution_time_s, 1800, "50K miners took too long")
        # Should use less than 16GB memory
        self.assertLess(result.memory_peak_mb, 16384, "50K miners used too much memory")

    def test_50k_miners_full_history(self):
        """Test 50k miners with full 120-day history - maximum data scenario."""
        result = self._run_scale_test(n_miners=50000, n_days=120, n_positions_per_miner=5)

        print(f"\n  === 50K Miners Full History Results ===")
        print(f"  Execution time: {result.execution_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Scores generated: {result.scores_generated:,}")
        print(f"  Memory per miner: {result.memory_peak_mb * 1024 / 50000:.2f} KB")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertEqual(result.scores_generated, 50000)
        # Full history might take longer but should still complete in under 1 hour
        self.assertLess(result.execution_time_s, 3600, "50K miners full history took too long")
        # Full history uses more memory but should stay under 32GB
        self.assertLess(result.memory_peak_mb, 32768, "50K miners full history used too much memory")


class TestScoringComponentsScalability(TestBase):
    """Test individual scoring components at scale."""

    def test_metrics_calculation_at_scale(self):
        """Test that individual metric calculations are efficient."""
        n_miners = 50000
        n_days = 120
        n_returns = n_days  # One return per day

        # Generate random returns for all miners
        all_returns = [
            list(np.random.normal(0.001, 0.02, n_returns))
            for _ in range(n_miners)
        ]

        # Generate ledgers for all miners
        all_ledgers = [
            SyntheticDataGenerator.generate_ledger(n_days=n_days, seed=i)
            for i in range(min(100, n_miners))  # Only create 100 actual ledgers for metric tests
        ]

        metrics_to_test = [
            ("sharpe", Metrics.sharpe),
            ("sortino", Metrics.sortino),
            ("omega", Metrics.omega),
            ("statistical_confidence", Metrics.statistical_confidence),
            ("daily_max_drawdown", Metrics.daily_max_drawdown),
        ]

        print(f"\n  Testing metric calculations for {n_miners:,} miners ({n_returns} days each):")

        for metric_name, metric_func in metrics_to_test:
            start_time = time.time()

            results = []
            for returns in all_returns:
                if metric_name == "daily_max_drawdown":
                    result = metric_func(returns)
                else:
                    result = metric_func(returns)
                results.append(result)

            elapsed = time.time() - start_time
            valid_results = [r for r in results if not math.isnan(r) and not math.isinf(r)]

            print(f"    {metric_name}: {elapsed:.3f}s ({elapsed/n_miners*1000:.4f}ms/miner), {len(valid_results):,} valid")

            # Each metric should complete in under 30 seconds for 50k miners
            self.assertLess(elapsed, 30, f"{metric_name} took too long")

    def test_softmax_at_scale(self):
        """Test softmax normalization with 50k miners."""
        n_miners = 50000

        # Generate random scores
        scores = [(f"miner_{i}", np.random.uniform(0, 100)) for i in range(n_miners)]

        start_time = time.time()
        softmax_scores = Scoring.softmax_scores(scores)
        elapsed = time.time() - start_time

        print(f"\n  Softmax for {n_miners:,} miners: {elapsed:.3f}s")

        # Verify results
        self.assertEqual(len(softmax_scores), n_miners)
        total = sum(score for _, score in softmax_scores)
        self.assertAlmostEqual(total, 1.0, places=5)

        # Should complete very quickly (under 1 second)
        self.assertLess(elapsed, 1.0, "Softmax took too long")

    def test_percentile_calculation_at_scale(self):
        """Test percentile calculations with 50k miners."""
        n_miners = 50000

        # Generate random scores
        scores = [(f"miner_{i}", np.random.uniform(-100, 100)) for i in range(n_miners)]

        start_time = time.time()
        percentile_scores = Scoring.miner_scores_percentiles(scores)
        elapsed = time.time() - start_time

        print(f"\n  Percentile calculation for {n_miners:,} miners: {elapsed:.3f}s")

        # Verify results
        self.assertEqual(len(percentile_scores), n_miners)

        # Should complete in under 5 seconds
        self.assertLess(elapsed, 5.0, "Percentile calculation took too long")

    def test_penalty_calculation_at_scale(self):
        """Test penalty calculations with 50k miners."""
        n_miners = 50000
        n_positions_per_miner = 5

        # Generate minimal position data
        positions_dict = {}
        ledger_dict = {}
        account_sizes = {}

        print(f"\n  Generating penalty test data for {n_miners:,} miners...")

        for i in range(n_miners):
            hotkey = SyntheticDataGenerator.generate_hotkey(i)

            # Minimal ledger for penalty testing
            ledger = SyntheticDataGenerator.generate_ledger(n_days=30, seed=i)
            ledger_dict[hotkey] = {TP_ID_PORTFOLIO: ledger}

            # Minimal positions
            positions_dict[hotkey] = [
                SyntheticDataGenerator.generate_position(
                    hotkey=hotkey,
                    is_closed=True,
                    n_orders=2,
                    seed=i * n_positions_per_miner + j,
                )
                for j in range(n_positions_per_miner)
            ]

            account_sizes[hotkey] = 200000 if i % 10 != 0 else 50000

        # Mock risk_profile_penalty to avoid complex calculation
        with patch.object(
            PositionPenalties, 'risk_profile_penalty', return_value=1.0
        ), patch(
            'vali_objects.contract.validator_contract_manager.ValidatorContractManager.min_collateral_penalty',
            side_effect=lambda x: 1.0 if x >= ValiConfig.MIN_COLLATERAL_VALUE else 0.0
        ):
            start_time = time.time()
            penalties = Scoring.miner_penalties(
                positions_dict,
                ledger_dict,
                account_sizes,
            )
            elapsed = time.time() - start_time

        print(f"  Penalty calculation for {n_miners:,} miners: {elapsed:.3f}s")

        # Verify results
        self.assertEqual(len(penalties), n_miners)

        # Count penalized miners
        full_penalty = sum(1 for p in penalties.values() if p == 0)
        partial_penalty = sum(1 for p in penalties.values() if 0 < p < 1)
        no_penalty = sum(1 for p in penalties.values() if p == 1)

        print(f"  Full penalty: {full_penalty}, Partial: {partial_penalty}, None: {no_penalty}")

        # Should complete in under 30 seconds
        self.assertLess(elapsed, 30, "Penalty calculation took too long")


class TestMemoryEfficiency(TestBase):
    """Test memory efficiency of data structures."""

    def test_checkpoint_memory_size(self):
        """Verify checkpoint memory usage is reasonable."""
        checkpoint = SyntheticDataGenerator.generate_checkpoint(0)

        # Estimate size using sys.getsizeof (doesn't capture nested objects well)
        # but gives a rough baseline
        import sys
        base_size = sys.getsizeof(checkpoint)

        # Calculate actual memory by serializing
        dict_size = sys.getsizeof(checkpoint.to_dict())

        print(f"\n  Checkpoint sizes:")
        print(f"    Object: ~{base_size} bytes")
        print(f"    Dict representation: ~{dict_size} bytes")

        # A checkpoint should be well under 1KB
        self.assertLess(base_size, 1024, "Checkpoint too large")

    def test_ledger_memory_size(self):
        """Verify ledger memory usage scales linearly."""
        sizes = []

        for n_days in [30, 60, 120]:
            ledger = SyntheticDataGenerator.generate_ledger(n_days=n_days)

            # Estimate memory
            import sys
            base_size = sys.getsizeof(ledger)
            cps_size = sum(sys.getsizeof(cp) for cp in ledger.cps)
            total_est = base_size + cps_size

            sizes.append((n_days, total_est))
            print(f"\n  Ledger with {n_days} days ({len(ledger.cps)} checkpoints): ~{total_est / 1024:.1f} KB")

        # Verify roughly linear scaling
        ratio_30_to_60 = sizes[1][1] / sizes[0][1]
        ratio_60_to_120 = sizes[2][1] / sizes[1][1]

        # Should scale roughly 2x (with some overhead)
        self.assertLess(ratio_30_to_60, 3.0, "Memory scaling not linear (30->60)")
        self.assertLess(ratio_60_to_120, 3.0, "Memory scaling not linear (60->120)")

    def test_50k_ledger_dict_memory(self):
        """Estimate total memory for 50k miners' ledger dicts."""
        tracemalloc.start()
        gc.collect()

        # Generate data for subset to estimate
        n_sample = 1000
        ledger_dict, _, _ = SyntheticDataGenerator.generate_miner_data(
            n_miners=n_sample,
            n_days=120,
            n_positions_per_miner=0,  # Just ledgers
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_per_miner_kb = peak / 1024 / n_sample
        estimated_50k_gb = memory_per_miner_kb * 50000 / 1024 / 1024

        print(f"\n  Memory estimate for 50k miners:")
        print(f"    Per miner (ledgers only): {memory_per_miner_kb:.1f} KB")
        print(f"    Estimated 50k total: {estimated_50k_gb:.2f} GB")

        # Should estimate under 20GB for ledgers
        self.assertLess(estimated_50k_gb, 20, "Estimated memory too high for 50k miners")


class TestConcurrencyScalability(TestBase):
    """Test concurrent processing capabilities."""

    def test_parallel_metric_calculation(self):
        """Test if metrics can be calculated in parallel efficiently."""
        n_miners = 10000
        n_days = 60

        # Generate data
        all_returns = [
            list(np.random.normal(0.001, 0.02, n_days))
            for _ in range(n_miners)
        ]

        # Sequential timing
        start = time.time()
        seq_results = [Metrics.sharpe(returns) for returns in all_returns]
        seq_time = time.time() - start

        # Parallel timing using ThreadPoolExecutor
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            par_results = list(executor.map(Metrics.sharpe, all_returns))
        par_time = time.time() - start

        print(f"\n  Parallel metric calculation for {n_miners:,} miners:")
        print(f"    Sequential: {seq_time:.3f}s")
        print(f"    Parallel (4 workers): {par_time:.3f}s")
        print(f"    Speedup: {seq_time/par_time:.2f}x")

        # Results should be identical
        for s, p in zip(seq_results, par_results):
            self.assertEqual(s, p)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
