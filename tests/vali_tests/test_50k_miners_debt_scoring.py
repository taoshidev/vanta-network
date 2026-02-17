"""
Scalability tests for 50k miners using DebtBasedScoring.

Tests the debt-based scoring system's ability to handle 50,000 miners.
Uses parallel data generation with multiprocessing for fast setup.

Key aspects tested:
- DebtLedger memory footprint at scale
- DebtBasedScoring.compute_results() performance
- Memory usage patterns matching production

Run with:
    python tests/run_vali_testing_suite.py test_50k_miners_debt_scoring.py

Or directly:
    python -m pytest tests/vali_tests/test_50k_miners_debt_scoring.py -v -s
"""

import gc
import math
import os
import sys
import time
import tracemalloc
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional
from unittest.mock import MagicMock, patch

import numpy as np

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger


@dataclass
class ScaleTestResult:
    """Results from a scalability test run."""
    n_miners: int
    memory_peak_mb: float
    memory_current_mb: float
    data_gen_time_s: float
    scoring_time_s: float
    total_time_s: float
    scores_generated: int
    success: bool
    error_message: str = ""


def generate_hotkey(index: int) -> str:
    """Generate a deterministic hotkey for a miner index."""
    return f"5F{'0' * 40}{index:06d}"[-48:]


def generate_single_miner_debt_ledger(args: Tuple[int, int, int, str]) -> Tuple[str, DebtLedger]:
    """
    Generate a DebtLedger for a single miner.

    Designed to be called in parallel via multiprocessing.

    Args:
        args: Tuple of (miner_index, n_checkpoints, base_time_ms, bucket)

    Returns:
        Tuple of (hotkey, DebtLedger)
    """
    miner_index, n_checkpoints, base_time_ms, bucket = args

    # Use index as seed for reproducibility
    np.random.seed(miner_index)

    hotkey = generate_hotkey(miner_index)

    # Generate checkpoints
    checkpoints = []
    portfolio_value = 1.0
    max_portfolio_value = 1.0
    cumulative_realized_pnl = 0.0

    # Performance profile based on miner index
    profile_idx = miner_index % 4
    profiles = [
        {"mean": 0.002, "std": 0.01, "mdd_base": 0.97},   # winning
        {"mean": -0.001, "std": 0.015, "mdd_base": 0.92},  # losing
        {"mean": 0.0005, "std": 0.012, "mdd_base": 0.95}, # average
        {"mean": 0.001, "std": 0.025, "mdd_base": 0.90},  # volatile
    ]
    profile = profiles[profile_idx]

    checkpoint_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    for i in range(n_checkpoints):
        # Generate realistic checkpoint data
        daily_return = np.random.normal(profile["mean"], profile["std"])

        portfolio_value *= math.exp(daily_return)
        max_portfolio_value = max(max_portfolio_value, portfolio_value)

        # Calculate drawdown
        mdd = min(portfolio_value / max_portfolio_value, profile["mdd_base"])

        # Realistic PnL values
        realized_delta = np.random.normal(50, 20) if daily_return > 0 else np.random.normal(-30, 15)
        cumulative_realized_pnl += realized_delta
        unrealized_pnl = np.random.normal(0, 100)

        # Emissions (realistic values)
        chunk_alpha = np.random.uniform(0.1, 2.0)
        alpha_to_tao = 0.001  # ~0.1% of TAO
        tao_price = 500.0  # $500/TAO
        chunk_tao = chunk_alpha * alpha_to_tao
        chunk_usd = chunk_tao * tao_price

        # Penalties (most miners have no penalties)
        drawdown_penalty = 1.0 if mdd > 0.9 else 0.0
        risk_penalty = np.random.choice([1.0, 0.95, 0.9], p=[0.8, 0.15, 0.05])
        collateral_penalty = 1.0 if miner_index % 10 != 0 else 0.0  # 10% have no collateral
        total_penalty = drawdown_penalty * risk_penalty * collateral_penalty

        checkpoint = DebtCheckpoint(
            timestamp_ms=base_time_ms + (i + 1) * checkpoint_duration_ms,
            # Emissions
            chunk_emissions_alpha=chunk_alpha,
            chunk_emissions_tao=chunk_tao,
            chunk_emissions_usd=chunk_usd,
            avg_alpha_to_tao_rate=alpha_to_tao,
            avg_tao_to_usd_rate=tao_price,
            tao_balance_snapshot=cumulative_realized_pnl * 0.01,
            alpha_balance_snapshot=cumulative_realized_pnl * 10,
            # Performance
            portfolio_return=portfolio_value,
            realized_pnl=realized_delta,
            unrealized_pnl=unrealized_pnl,
            spread_fee_loss=-0.0005 * abs(daily_return),
            carry_fee_loss=-0.001 * abs(daily_return),
            max_drawdown=mdd,
            max_portfolio_value=max_portfolio_value,
            open_ms=checkpoint_duration_ms,
            accum_ms=checkpoint_duration_ms,
            n_updates=np.random.randint(50, 150),
            # Penalties
            drawdown_penalty=drawdown_penalty,
            risk_profile_penalty=risk_penalty,
            min_collateral_penalty=collateral_penalty,
            risk_adjusted_performance_penalty=1.0,
            total_penalty=total_penalty,
            challenge_period_status=bucket,
        )
        checkpoints.append(checkpoint)

    return hotkey, DebtLedger(hotkey=hotkey, checkpoints=checkpoints)


def generate_debt_ledgers_parallel(
    n_miners: int,
    n_days: int = 120,
    n_workers: int = None,
    bucket_distribution: Dict[str, float] = None,
) -> Dict[str, DebtLedger]:
    """
    Generate DebtLedgers for n miners using parallel processing.

    Args:
        n_miners: Number of miners to generate
        n_days: Days of history per miner
        n_workers: Number of parallel workers (default: CPU count)
        bucket_distribution: Distribution of miner buckets

    Returns:
        Dict mapping hotkey -> DebtLedger
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 82)  # Cap at 16 workers

    if bucket_distribution is None:
        bucket_distribution = {
            MinerBucket.MAINCOMP.value: 0.4,
            MinerBucket.PROBATION.value: 0.2,
            MinerBucket.CHALLENGE.value: 0.3,
            MinerBucket.PLAGIARISM.value: 0.05,
            MinerBucket.UNKNOWN.value: 0.05,
        }

    n_checkpoints = n_days * ValiConfig.DAILY_CHECKPOINTS

    # Base time aligned to checkpoint duration
    base_time_ms = 1735689600000  # 2024-01-01 00:00:00 UTC
    base_time_ms = (base_time_ms // ValiConfig.TARGET_CHECKPOINT_DURATION_MS) * ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    # Assign buckets to miners
    buckets = list(bucket_distribution.keys())
    cumulative_probs = np.cumsum(list(bucket_distribution.values()))

    # Prepare arguments for parallel generation
    args_list = []
    for i in range(n_miners):
        r = (i / n_miners)  # Deterministic bucket assignment
        bucket_idx = np.searchsorted(cumulative_probs, r)
        bucket = buckets[min(bucket_idx, len(buckets) - 1)]
        args_list.append((i, n_checkpoints, base_time_ms, bucket))

    # Generate in parallel
    ledger_dict = {}

    if n_workers > 1 and n_miners > 100:
        # Use multiprocessing for large datasets
        with Pool(processes=n_workers) as pool:
            results = pool.map(generate_single_miner_debt_ledger, args_list)

        for hotkey, ledger in results:
            ledger_dict[hotkey] = ledger
    else:
        # Sequential for small datasets
        for args in args_list:
            hotkey, ledger = generate_single_miner_debt_ledger(args)
            ledger_dict[hotkey] = ledger

    return ledger_dict


class MockMetagraph:
    """Mock metagraph for testing."""

    def __init__(self, hotkeys: List[str]):
        self._hotkeys = hotkeys
        self.n_uids = len(hotkeys)
        self._emission = [0.001] * self.n_uids  # 0.001 TAO per tempo per UID
        self.tao_reserve_rao = 1e18  # 1 billion TAO in RAO
        self.alpha_reserve_rao = 1e21  # 1 trillion ALPHA in RAO
        self.tao_to_usd_rate = 500.0

    def get_emission(self):
        return self._emission

    def get_hotkeys(self):
        return self._hotkeys


class MockChallengePeriodClient:
    """Mock challenge period client for testing."""

    def __init__(self, ledger_dict: Dict[str, DebtLedger]):
        # Extract bucket from first checkpoint of each miner
        self._buckets = {}
        for hotkey, ledger in ledger_dict.items():
            if ledger.checkpoints:
                status = ledger.checkpoints[0].challenge_period_status
                self._buckets[hotkey] = MinerBucket(status)
            else:
                self._buckets[hotkey] = MinerBucket.UNKNOWN

    def get_miner_bucket(self, hotkey: str) -> MinerBucket:
        return self._buckets.get(hotkey, MinerBucket.UNKNOWN)


class MockContractClient:
    """Mock contract client for testing."""

    def __init__(self, hotkeys: List[str]):
        # 90% have adequate collateral, 10% don't
        self._account_sizes = {}
        for i, hotkey in enumerate(hotkeys):
            if i % 10 != 0:
                self._account_sizes[hotkey] = np.random.uniform(150000, 500000)
            else:
                self._account_sizes[hotkey] = np.random.uniform(50000, 140000)

    def get_miner_account_size(self, hotkey: str, most_recent: bool = True) -> float:
        return self._account_sizes.get(hotkey, 0.0)


class Test50kMinersDebtScoring(TestBase):
    """Test suite for 50k miner scalability with DebtBasedScoring."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
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
        n_workers: int = None,
    ) -> ScaleTestResult:
        """
        Run a scalability test with the given number of miners.

        Returns ScaleTestResult with timing and memory metrics.
        """
        tracemalloc.start()
        gc.collect()

        start_time = time.time()
        error_message = ""
        scores_generated = 0
        success = True
        data_gen_time = 0
        scoring_time = 0

        try:
            # Step 1: Generate data in parallel
            print(f"\n  Generating DebtLedgers for {n_miners:,} miners ({n_days} days each)...")
            gen_start = time.time()

            ledger_dict = generate_debt_ledgers_parallel(
                n_miners=n_miners,
                n_days=n_days,
                n_workers=n_workers,
            )

            data_gen_time = time.time() - gen_start
            print(f"  Data generation: {data_gen_time:.2f}s ({n_miners / data_gen_time:.0f} miners/sec)")

            # Step 2: Create mocks
            hotkeys = list(ledger_dict.keys())
            metagraph = MockMetagraph(hotkeys)
            challengeperiod_client = MockChallengePeriodClient(ledger_dict)
            contract_client = MockContractClient(hotkeys)

            # Step 3: Run DebtBasedScoring
            print(f"  Running DebtBasedScoring.compute_results()...")
            score_start = time.time()

            current_time_ms = (
                1735689600000 +
                (n_days + 30) * ValiConfig.DAILY_MS  # 30 days after data ends
            )

            results = DebtBasedScoring.compute_results(
                ledger_dict=ledger_dict,
                metagraph=metagraph,
                challengeperiod_client=challengeperiod_client,
                contract_client=contract_client,
                current_time_ms=current_time_ms,
                verbose=False,
                is_testnet=True,
            )

            scoring_time = time.time() - score_start
            print(f"  Scoring: {scoring_time:.2f}s")

            scores_generated = len(results)

            # Verify results
            if scores_generated == 0:
                error_message = "No scores generated"
                success = False
            else:
                # Check score normalization
                total_score = sum(score for _, score in results)
                if not (0.99 <= total_score <= 1.01):
                    error_message = f"Scores don't sum to 1.0: {total_score}"
                    success = False

                # Check for invalid values
                for hotkey, score in results:
                    if math.isnan(score) or math.isinf(score):
                        error_message = f"Invalid score for {hotkey}: {score}"
                        success = False
                        break

        except Exception as e:
            import traceback
            error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            success = False
            print(f"  ERROR: {error_message}")

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = time.time() - start_time

        return ScaleTestResult(
            n_miners=n_miners,
            memory_peak_mb=peak / 1024 / 1024,
            memory_current_mb=current / 1024 / 1024,
            data_gen_time_s=data_gen_time,
            scoring_time_s=scoring_time,
            total_time_s=total_time,
            scores_generated=scores_generated,
            success=success,
            error_message=error_message,
        )

    def test_baseline_1k_miners(self):
        """Test baseline performance with 1,000 miners."""
        result = self._run_scale_test(n_miners=1000, n_days=60)

        print(f"\n  === 1K Miners Results ===")
        print(f"  Data generation: {result.data_gen_time_s:.2f}s")
        print(f"  Scoring time: {result.scoring_time_s:.2f}s")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Scores generated: {result.scores_generated:,}")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        # Scores include burn address
        self.assertGreaterEqual(result.scores_generated, 1000)
        self.assertLess(result.total_time_s, 60, "1K miners took too long")
        self.assertLess(result.memory_peak_mb, 1024, "1K miners used too much memory")

    def test_medium_10k_miners(self):
        """Test medium scale with 10,000 miners."""
        result = self._run_scale_test(n_miners=10000, n_days=60)

        print(f"\n  === 10K Miners Results ===")
        print(f"  Data generation: {result.data_gen_time_s:.2f}s")
        print(f"  Scoring time: {result.scoring_time_s:.2f}s")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"  Scores generated: {result.scores_generated:,}")
        print(f"  Memory per miner: {result.memory_peak_mb * 1024 / 10000:.2f} KB")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertGreaterEqual(result.scores_generated, 10000)
        self.assertLess(result.total_time_s, 300, "10K miners took too long")
        self.assertLess(result.memory_peak_mb, 4096, "10K miners used too much memory")

    def test_large_50k_miners(self):
        """Test large scale with 50,000 miners - the main scalability target."""
        result = self._run_scale_test(n_miners=50000, n_days=60)

        print(f"\n  === 50K Miners Results ===")
        print(f"  Data generation: {result.data_gen_time_s:.2f}s")
        print(f"  Scoring time: {result.scoring_time_s:.2f}s")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB ({result.memory_peak_mb/1024:.2f} GB)")
        print(f"  Scores generated: {result.scores_generated:,}")
        print(f"  Memory per miner: {result.memory_peak_mb * 1024 / 50000:.2f} KB")
        print(f"  Time per miner: {result.total_time_s * 1000 / 50000:.4f} ms")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertGreaterEqual(result.scores_generated, 50000)
        self.assertLess(result.total_time_s, 1800, "50K miners took too long (30 min limit)")
        self.assertLess(result.memory_peak_mb, 16384, "50K miners used too much memory (16GB limit)")

    def test_50k_miners_full_history(self):
        """Test 50k miners with full 120-day history."""
        result = self._run_scale_test(n_miners=50000, n_days=120)

        print(f"\n  === 50K Miners Full History Results ===")
        print(f"  Data generation: {result.data_gen_time_s:.2f}s")
        print(f"  Scoring time: {result.scoring_time_s:.2f}s")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB ({result.memory_peak_mb/1024:.2f} GB)")
        print(f"  Scores generated: {result.scores_generated:,}")

        self.assertTrue(result.success, f"Test failed: {result.error_message}")
        self.assertGreaterEqual(result.scores_generated, 50000)
        self.assertLess(result.total_time_s, 3600, "50K miners full history took too long (1hr limit)")
        self.assertLess(result.memory_peak_mb, 32768, "50K miners full history used too much memory (32GB limit)")


class TestDebtCheckpointMemory(TestBase):
    """Test memory efficiency of DebtCheckpoint data structures."""

    def test_checkpoint_memory_size(self):
        """Verify DebtCheckpoint memory usage."""
        import sys

        checkpoint = DebtCheckpoint(
            timestamp_ms=1735689600000,
            chunk_emissions_alpha=1.0,
            chunk_emissions_tao=0.001,
            chunk_emissions_usd=0.5,
            portfolio_return=1.05,
            realized_pnl=100.0,
            unrealized_pnl=50.0,
            max_drawdown=0.95,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value,
        )

        # Basic size (doesn't include referenced objects)
        base_size = sys.getsizeof(checkpoint)

        print(f"\n  DebtCheckpoint base size: {base_size} bytes")

        # Checkpoint should be compact
        self.assertLess(base_size, 500, "DebtCheckpoint too large")

    def test_ledger_memory_scaling(self):
        """Verify DebtLedger memory scales linearly with checkpoints."""
        tracemalloc.start()

        sizes = []
        for n_days in [30, 60, 120]:
            gc.collect()
            tracemalloc.reset_peak()

            # Generate a single miner's ledger
            _, ledger = generate_single_miner_debt_ledger((0, n_days * 2, 1735689600000, MinerBucket.MAINCOMP.value))

            _, peak = tracemalloc.get_traced_memory()
            sizes.append((n_days, len(ledger.checkpoints), peak / 1024))

            print(f"\n  {n_days} days ({len(ledger.checkpoints)} checkpoints): {peak/1024:.1f} KB")

        tracemalloc.stop()

        # Should scale roughly linearly
        ratio_30_to_60 = sizes[1][2] / sizes[0][2]
        ratio_60_to_120 = sizes[2][2] / sizes[1][2]

        print(f"  Scaling ratio 30->60 days: {ratio_30_to_60:.2f}x")
        print(f"  Scaling ratio 60->120 days: {ratio_60_to_120:.2f}x")

        # Should be roughly 2x (with some overhead)
        self.assertLess(ratio_30_to_60, 3.0, "Memory scaling not linear (30->60)")
        self.assertLess(ratio_60_to_120, 3.0, "Memory scaling not linear (60->120)")

    def test_50k_memory_estimate(self):
        """Estimate total memory for 50k miners."""
        n_sample = 100
        n_days = 120

        tracemalloc.start()
        gc.collect()

        # Generate sample ledgers
        ledger_dict = generate_debt_ledgers_parallel(
            n_miners=n_sample,
            n_days=n_days,
            n_workers=1,  # Sequential for accurate measurement
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_per_miner_kb = peak / 1024 / n_sample
        estimated_50k_gb = memory_per_miner_kb * 50000 / 1024 / 1024

        print(f"\n  Memory estimate for 50k miners (120 days):")
        print(f"    Per miner: {memory_per_miner_kb:.1f} KB")
        print(f"    Estimated 50k total: {estimated_50k_gb:.2f} GB")

        # Should be under 20GB for ledgers alone
        self.assertLess(estimated_50k_gb, 20, "Estimated memory too high for 50k miners")


class TestParallelDataGeneration(TestBase):
    """Test parallel data generation performance."""

    def test_parallel_generation_speedup(self):
        """Test that parallel generation provides speedup."""
        n_miners = 1000
        n_days = 30

        # Sequential
        start = time.time()
        _ = generate_debt_ledgers_parallel(n_miners=n_miners, n_days=n_days, n_workers=1)
        seq_time = time.time() - start

        # Parallel
        n_workers = min(cpu_count(), 8)
        start = time.time()
        _ = generate_debt_ledgers_parallel(n_miners=n_miners, n_days=n_days, n_workers=n_workers)
        par_time = time.time() - start

        speedup = seq_time / par_time if par_time > 0 else 1.0

        print(f"\n  Data generation ({n_miners} miners):")
        print(f"    Sequential: {seq_time:.2f}s")
        print(f"    Parallel ({n_workers} workers): {par_time:.2f}s")
        print(f"    Speedup: {speedup:.2f}x")

        # Should get at least some speedup with parallel generation
        if n_workers > 1:
            self.assertGreater(speedup, 1.2, "Parallel generation should be faster")

    def test_large_scale_generation(self):
        """Test generation of 50k miners."""
        n_miners = 50000
        n_days = 60
        n_workers = min(cpu_count(), 16)

        print(f"\n  Generating {n_miners:,} miners ({n_days} days, {n_workers} workers)...")

        start = time.time()
        ledger_dict = generate_debt_ledgers_parallel(
            n_miners=n_miners,
            n_days=n_days,
            n_workers=n_workers,
        )
        elapsed = time.time() - start

        print(f"  Generated in {elapsed:.2f}s ({n_miners/elapsed:.0f} miners/sec)")

        self.assertEqual(len(ledger_dict), n_miners)
        # Should generate 50k miners in under 5 minutes
        self.assertLess(elapsed, 300, "50k miner generation took too long")


if __name__ == '__main__':
    unittest.main(verbosity=2)
