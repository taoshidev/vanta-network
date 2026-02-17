"""
Debt Scoring Scaling Benchmark: 10k -> 200k miners.

Generates comprehensive benchmarks for the DebtBasedScoring system,
measuring execution time and memory usage across different miner counts.
Produces detailed visualizations showing scaling characteristics.

Run with:
    python tests/vali_tests/test_debt_scoring_scaling_benchmark.py

Or with pytest:
    python -m pytest tests/vali_tests/test_debt_scoring_scaling_benchmark.py -v -s
"""

import gc
import json
import math
import os
import sys
import time
import tracemalloc
import unittest
from dataclasses import dataclass, asdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    n_miners: int
    n_days: int
    n_checkpoints_per_miner: int

    # Timing metrics (seconds)
    data_gen_time_s: float
    scoring_time_s: float
    total_time_s: float

    # Memory metrics (MB)
    memory_peak_mb: float
    memory_current_mb: float
    memory_per_miner_kb: float

    # Throughput metrics
    miners_per_second: float
    scoring_throughput: float  # miners/second for scoring only

    # Validation
    scores_generated: int
    success: bool
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScalingAnalysis:
    """Analysis of scaling behavior across benchmark runs."""
    miner_counts: List[int]

    # Time scaling
    data_gen_times: List[float]
    scoring_times: List[float]
    total_times: List[float]

    # Memory scaling
    peak_memories_mb: List[float]
    memory_per_miner_kb: List[float]

    # Throughput
    throughputs: List[float]

    # Fit coefficients for O(n) scaling
    time_slope: float
    time_intercept: float
    memory_slope: float
    memory_intercept: float


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockMetagraph:
    """Mock metagraph for benchmarking."""

    def __init__(self, hotkeys: List[str]):
        self._hotkeys = hotkeys
        self.n_uids = len(hotkeys)
        self._emission = [0.001] * self.n_uids
        self.tao_reserve_rao = 1e18
        self.alpha_reserve_rao = 1e21
        self.tao_to_usd_rate = 500.0

    def get_emission(self):
        return self._emission

    def get_hotkeys(self):
        return self._hotkeys


class MockChallengePeriodClient:
    """Mock challenge period client for benchmarking."""

    def __init__(self, ledger_dict: Dict[str, DebtLedger]):
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
    """Mock contract client for benchmarking."""

    def __init__(self, hotkeys: List[str]):
        self._account_sizes = {}
        for i, hotkey in enumerate(hotkeys):
            if i % 10 != 0:
                self._account_sizes[hotkey] = np.random.uniform(150000, 500000)
            else:
                self._account_sizes[hotkey] = np.random.uniform(50000, 140000)

    def get_miner_account_size(self, hotkey: str, most_recent: bool = True) -> float:
        return self._account_sizes.get(hotkey, 0.0)


# ============================================================================
# Data Generation
# ============================================================================

def generate_hotkey(index: int) -> str:
    """Generate a deterministic hotkey for a miner index."""
    return f"5F{'0' * 40}{index:06d}"[-48:]


def generate_single_miner_debt_ledger(args: Tuple[int, int, int, str]) -> Tuple[str, DebtLedger]:
    """Generate a DebtLedger for a single miner (for parallel processing)."""
    miner_index, n_checkpoints, base_time_ms, bucket = args

    np.random.seed(miner_index)
    hotkey = generate_hotkey(miner_index)

    checkpoints = []
    portfolio_value = 1.0
    max_portfolio_value = 1.0
    cumulative_realized_pnl = 0.0

    # Performance profiles
    profile_idx = miner_index % 4
    profiles = [
        {"mean": 0.002, "std": 0.01, "mdd_base": 0.97},
        {"mean": -0.001, "std": 0.015, "mdd_base": 0.92},
        {"mean": 0.0005, "std": 0.012, "mdd_base": 0.95},
        {"mean": 0.001, "std": 0.025, "mdd_base": 0.90},
    ]
    profile = profiles[profile_idx]
    checkpoint_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    for i in range(n_checkpoints):
        daily_return = np.random.normal(profile["mean"], profile["std"])
        portfolio_value *= math.exp(daily_return)
        max_portfolio_value = max(max_portfolio_value, portfolio_value)
        mdd = min(portfolio_value / max_portfolio_value, profile["mdd_base"])

        realized_delta = np.random.normal(50, 20) if daily_return > 0 else np.random.normal(-30, 15)
        cumulative_realized_pnl += realized_delta
        unrealized_pnl = np.random.normal(0, 100)

        chunk_alpha = np.random.uniform(0.1, 2.0)
        alpha_to_tao = 0.001
        tao_price = 500.0
        chunk_tao = chunk_alpha * alpha_to_tao
        chunk_usd = chunk_tao * tao_price

        drawdown_penalty = 1.0 if mdd > 0.9 else 0.0
        risk_penalty = np.random.choice([1.0, 0.95, 0.9], p=[0.8, 0.15, 0.05])
        collateral_penalty = 1.0 if miner_index % 10 != 0 else 0.0
        total_penalty = drawdown_penalty * risk_penalty * collateral_penalty

        checkpoint = DebtCheckpoint(
            timestamp_ms=base_time_ms + (i + 1) * checkpoint_duration_ms,
            chunk_emissions_alpha=chunk_alpha,
            chunk_emissions_tao=chunk_tao,
            chunk_emissions_usd=chunk_usd,
            avg_alpha_to_tao_rate=alpha_to_tao,
            avg_tao_to_usd_rate=tao_price,
            tao_balance_snapshot=cumulative_realized_pnl * 0.01,
            alpha_balance_snapshot=cumulative_realized_pnl * 10,
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
) -> Dict[str, DebtLedger]:
    """Generate DebtLedgers for n miners using parallel processing."""
    if n_workers is None:
        n_workers = min(cpu_count(), 128)

    bucket_distribution = {
        MinerBucket.MAINCOMP.value: 0.4,
        MinerBucket.PROBATION.value: 0.2,
        MinerBucket.CHALLENGE.value: 0.3,
        MinerBucket.PLAGIARISM.value: 0.05,
        MinerBucket.UNKNOWN.value: 0.05,
    }

    n_checkpoints = n_days * ValiConfig.DAILY_CHECKPOINTS
    base_time_ms = 1735689600000
    base_time_ms = (base_time_ms // ValiConfig.TARGET_CHECKPOINT_DURATION_MS) * ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    buckets = list(bucket_distribution.keys())
    cumulative_probs = np.cumsum(list(bucket_distribution.values()))

    args_list = []
    for i in range(n_miners):
        r = i / n_miners
        bucket_idx = np.searchsorted(cumulative_probs, r)
        bucket = buckets[min(bucket_idx, len(buckets) - 1)]
        args_list.append((i, n_checkpoints, base_time_ms, bucket))

    ledger_dict = {}

    if n_workers > 1 and n_miners > 100:
        with Pool(processes=n_workers) as pool:
            results = pool.map(generate_single_miner_debt_ledger, args_list)
        for hotkey, ledger in results:
            ledger_dict[hotkey] = ledger
    else:
        for args in args_list:
            hotkey, ledger = generate_single_miner_debt_ledger(args)
            ledger_dict[hotkey] = ledger

    return ledger_dict


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    n_miners: int,
    n_days: int = 60,
    n_workers: int = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark with the specified number of miners."""
    tracemalloc.start()
    gc.collect()

    start_time = time.time()
    error_message = ""
    scores_generated = 0
    success = True
    data_gen_time = 0
    scoring_time = 0
    n_checkpoints = n_days * ValiConfig.DAILY_CHECKPOINTS

    try:
        # Data generation
        if verbose:
            print(f"  Generating {n_miners:,} miners ({n_days} days, {n_checkpoints} checkpoints each)...")

        gen_start = time.time()
        ledger_dict = generate_debt_ledgers_parallel(
            n_miners=n_miners,
            n_days=n_days,
            n_workers=n_workers,
        )
        data_gen_time = time.time() - gen_start

        if verbose:
            print(f"    Data gen: {data_gen_time:.2f}s ({n_miners / data_gen_time:.0f} miners/sec)")

        # Create mocks
        hotkeys = list(ledger_dict.keys())
        metagraph = MockMetagraph(hotkeys)
        challengeperiod_client = MockChallengePeriodClient(ledger_dict)
        contract_client = MockContractClient(hotkeys)

        # Run scoring
        if verbose:
            print(f"  Running DebtBasedScoring.compute_results()...")

        score_start = time.time()
        current_time_ms = 1735689600000 + (n_days + 30) * ValiConfig.DAILY_MS

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
        scores_generated = len(results)

        if verbose:
            print(f"    Scoring: {scoring_time:.2f}s ({n_miners / scoring_time:.0f} miners/sec)")

        # Validate results
        if scores_generated == 0:
            error_message = "No scores generated"
            success = False
        else:
            total_score = sum(score for _, score in results)
            if not (0.99 <= total_score <= 1.01):
                error_message = f"Scores don't sum to 1.0: {total_score}"
                success = False

            for hotkey, score in results:
                if math.isnan(score) or math.isinf(score):
                    error_message = f"Invalid score for {hotkey}: {score}"
                    success = False
                    break

    except Exception as e:
        import traceback
        error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        success = False
        if verbose:
            print(f"  ERROR: {error_message}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = time.time() - start_time
    memory_peak_mb = peak / 1024 / 1024
    memory_per_miner_kb = (peak / 1024) / n_miners if n_miners > 0 else 0

    return BenchmarkResult(
        n_miners=n_miners,
        n_days=n_days,
        n_checkpoints_per_miner=n_checkpoints,
        data_gen_time_s=data_gen_time,
        scoring_time_s=scoring_time,
        total_time_s=total_time,
        memory_peak_mb=memory_peak_mb,
        memory_current_mb=current / 1024 / 1024,
        memory_per_miner_kb=memory_per_miner_kb,
        miners_per_second=n_miners / total_time if total_time > 0 else 0,
        scoring_throughput=n_miners / scoring_time if scoring_time > 0 else 0,
        scores_generated=scores_generated,
        success=success,
        error_message=error_message,
    )


def run_scaling_benchmark(
    miner_counts: List[int] = None,
    n_days: int = 60,
    n_workers: int = None,
    verbose: bool = True,
) -> Tuple[List[BenchmarkResult], ScalingAnalysis]:
    """
    Run benchmarks across multiple miner counts to analyze scaling.

    Returns:
        Tuple of (list of BenchmarkResults, ScalingAnalysis)
    """
    if miner_counts is None:
        miner_counts = [10000, 25000, 50000, 75000, 100000, 150000, 200000]

    results = []

    print("\n" + "=" * 70)
    print("DEBT SCORING SCALING BENCHMARK")
    print("=" * 70)
    print(f"Testing: {', '.join(f'{n:,}' for n in miner_counts)} miners")
    print(f"Days of history: {n_days}")
    print(f"Workers: {n_workers or min(cpu_count(), 128)}")
    print("=" * 70)

    for n_miners in miner_counts:
        print(f"\n[{n_miners:,} MINERS]")

        gc.collect()
        result = run_benchmark(
            n_miners=n_miners,
            n_days=n_days,
            n_workers=n_workers,
            verbose=verbose,
        )
        results.append(result)

        status = "PASS" if result.success else "FAIL"
        print(f"  Result: {status}")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.1f} MB ({result.memory_peak_mb/1024:.2f} GB)")
        print(f"  Memory/miner: {result.memory_per_miner_kb:.2f} KB")
        print(f"  Throughput: {result.miners_per_second:.0f} miners/sec")

        if not result.success:
            print(f"  Error: {result.error_message[:200]}")

    # Compute scaling analysis
    successful_results = [r for r in results if r.success]

    if len(successful_results) >= 2:
        counts = np.array([r.n_miners for r in successful_results])
        times = np.array([r.total_time_s for r in successful_results])
        memories = np.array([r.memory_peak_mb for r in successful_results])

        # Linear regression for scaling coefficients
        time_slope, time_intercept = np.polyfit(counts, times, 1)
        memory_slope, memory_intercept = np.polyfit(counts, memories, 1)

        analysis = ScalingAnalysis(
            miner_counts=[r.n_miners for r in successful_results],
            data_gen_times=[r.data_gen_time_s for r in successful_results],
            scoring_times=[r.scoring_time_s for r in successful_results],
            total_times=[r.total_time_s for r in successful_results],
            peak_memories_mb=[r.memory_peak_mb for r in successful_results],
            memory_per_miner_kb=[r.memory_per_miner_kb for r in successful_results],
            throughputs=[r.miners_per_second for r in successful_results],
            time_slope=time_slope,
            time_intercept=time_intercept,
            memory_slope=memory_slope,
            memory_intercept=memory_intercept,
        )
    else:
        analysis = None

    return results, analysis


# ============================================================================
# Plotting Functions
# ============================================================================

def create_scaling_plots(
    results: List[BenchmarkResult],
    analysis: ScalingAnalysis,
    output_dir: str = None,
    show_plots: bool = False,
) -> str:
    """
    Create comprehensive scaling visualization plots.

    Returns:
        Path to the saved plot file, or None if plotting failed.
    """
    try:
        # Use Agg backend for headless environments (SSH, servers)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available, skipping plots")
        return None

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    successful_results = [r for r in results if r.success]
    if len(successful_results) < 2:
        print("Not enough successful results to plot")
        return None

    # Extract data
    miners = np.array([r.n_miners for r in successful_results])
    miners_k = miners / 1000
    total_times = np.array([r.total_time_s for r in successful_results])
    scoring_times = np.array([r.scoring_time_s for r in successful_results])
    data_gen_times = np.array([r.data_gen_time_s for r in successful_results])
    peak_memories = np.array([r.memory_peak_mb for r in successful_results])
    mem_per_miner = np.array([r.memory_per_miner_kb for r in successful_results])
    throughputs = np.array([r.miners_per_second for r in successful_results])
    scoring_throughputs = np.array([r.scoring_throughput for r in successful_results])

    # Style settings - use a style that works across matplotlib versions
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # Use default style
    colors = {
        'primary': '#2563eb',
        'secondary': '#7c3aed',
        'tertiary': '#059669',
        'accent': '#dc2626',
        'neutral': '#6b7280',
        'light': '#e5e7eb',
    }

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Execution Time vs Miners (main plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(miners_k, total_times, 'o-', color=colors['primary'],
             linewidth=2.5, markersize=8, label='Total Time')
    ax1.plot(miners_k, scoring_times, 's--', color=colors['secondary'],
             linewidth=2, markersize=7, label='Scoring Only')
    ax1.plot(miners_k, data_gen_times, '^:', color=colors['tertiary'],
             linewidth=2, markersize=7, label='Data Generation')

    # Add trend line
    x_fit = np.linspace(miners.min(), miners.max(), 100) / 1000
    y_fit = analysis.time_slope * x_fit * 1000 + analysis.time_intercept
    ax1.plot(x_fit, y_fit, '--', color=colors['neutral'], alpha=0.5,
             label=f'Linear fit (slope={analysis.time_slope*1e6:.2f} ms/miner)')

    ax1.set_xlabel('Number of Miners (thousands)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time Scaling', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # 2. Memory Usage vs Miners
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(miners_k, peak_memories / 1024, 'o-', color=colors['accent'],
             linewidth=2.5, markersize=8)
    ax2.fill_between(miners_k, 0, peak_memories / 1024, alpha=0.2, color=colors['accent'])

    # Add memory fit line
    y_mem_fit = (analysis.memory_slope * x_fit * 1000 + analysis.memory_intercept) / 1024
    ax2.plot(x_fit, y_mem_fit, '--', color=colors['neutral'], alpha=0.5,
             label=f'Linear fit ({analysis.memory_slope*1024:.1f} KB/miner)')

    ax2.set_xlabel('Number of Miners (thousands)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Peak Memory (GB)', fontsize=11, fontweight='bold')
    ax2.set_title('Memory Usage Scaling', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    # 3. Throughput vs Miners
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(miners_k, throughputs, width=miners_k.max()*0.06,
            color=colors['primary'], alpha=0.7, label='Overall')
    ax3.bar(miners_k + miners_k.max()*0.07, scoring_throughputs,
            width=miners_k.max()*0.06, color=colors['secondary'], alpha=0.7,
            label='Scoring Only')

    # Add horizontal line for average
    avg_throughput = np.mean(throughputs)
    ax3.axhline(y=avg_throughput, color=colors['neutral'], linestyle='--',
                alpha=0.7, label=f'Avg: {avg_throughput:.0f}/sec')

    ax3.set_xlabel('Number of Miners (thousands)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Throughput (miners/second)', fontsize=11, fontweight='bold')
    ax3.set_title('Processing Throughput', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(left=0)

    # 4. Memory per Miner (efficiency)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(miners_k, mem_per_miner, 'o-', color=colors['tertiary'],
             linewidth=2.5, markersize=8)
    ax4.fill_between(miners_k, 0, mem_per_miner, alpha=0.2, color=colors['tertiary'])

    ax4.set_xlabel('Number of Miners (thousands)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Memory per Miner (KB)', fontsize=11, fontweight='bold')
    ax4.set_title('Memory Efficiency', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)

    # Add annotation for trend
    if mem_per_miner[-1] < mem_per_miner[0]:
        trend_text = f"Improving: {mem_per_miner[0]:.1f} -> {mem_per_miner[-1]:.1f} KB"
    else:
        trend_text = f"Stable: ~{np.mean(mem_per_miner):.1f} KB/miner"
    ax4.annotate(trend_text, xy=(0.5, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Time Breakdown Stacked Area
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.stackplot(miners_k, data_gen_times, scoring_times,
                  labels=['Data Generation', 'Scoring'],
                  colors=[colors['tertiary'], colors['secondary']], alpha=0.7)
    ax5.plot(miners_k, total_times, 'k-', linewidth=2, label='Total')

    ax5.set_xlabel('Number of Miners (thousands)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_title('Time Breakdown', fontsize=13, fontweight='bold', pad=10)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(left=0)
    ax5.set_ylim(bottom=0)

    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create summary table
    summary_data = [
        ['Metric', 'Min', 'Max', 'Scaling'],
        ['Miners', f'{miners.min():,}', f'{miners.max():,}', '-'],
        ['Total Time', f'{total_times.min():.1f}s', f'{total_times.max():.1f}s',
         f'{analysis.time_slope*1e6:.2f} ms/miner'],
        ['Peak Memory', f'{peak_memories.min():.0f} MB', f'{peak_memories.max():.0f} MB',
         f'{analysis.memory_slope*1024:.1f} KB/miner'],
        ['Throughput', f'{throughputs.min():.0f}/s', f'{throughputs.max():.0f}/s',
         f'Avg: {np.mean(throughputs):.0f}/s'],
        ['Mem/Miner', f'{mem_per_miner.min():.1f} KB', f'{mem_per_miner.max():.1f} KB',
         f'Avg: {np.mean(mem_per_miner):.1f} KB'],
    ]

    table = ax6.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.2, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor(colors['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(summary_data)):
        for j in range(len(summary_data[0])):
            table[(i, j)].set_facecolor(colors['light'] if i % 2 == 0 else 'white')

    ax6.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

    # Add overall title
    fig.suptitle(
        f'Debt-Based Scoring System: Scaling Analysis (10k-200k miners, {successful_results[0].n_days} days)',
        fontsize=15, fontweight='bold', y=0.98
    )

    # Save the figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'scaling_benchmark_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path}")

    # Also save a high-res version
    output_path_hires = os.path.join(output_dir, f'scaling_benchmark_{timestamp}_hires.png')
    plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-res plot saved to: {output_path_hires}")

    # Clean up - always close in headless mode
    plt.close('all')

    return output_path


def save_results_json(
    results: List[BenchmarkResult],
    analysis: ScalingAnalysis,
    output_dir: str = None,
) -> str:
    """Save benchmark results to JSON file."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'scaling_benchmark_{timestamp}.json')

    data = {
        'timestamp': timestamp,
        'results': [r.to_dict() for r in results],
        'analysis': {
            'miner_counts': analysis.miner_counts,
            'time_slope_ms_per_miner': analysis.time_slope * 1000,
            'time_intercept_s': analysis.time_intercept,
            'memory_slope_kb_per_miner': analysis.memory_slope * 1024,
            'memory_intercept_mb': analysis.memory_intercept,
            'avg_throughput': np.mean(analysis.throughputs),
            'avg_memory_per_miner_kb': np.mean(analysis.memory_per_miner_kb),
        } if analysis else None,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")
    return output_path


# ============================================================================
# Test Cases
# ============================================================================

class TestDebtScoringScalingBenchmark(unittest.TestCase):
    """Unit test wrapper for the scaling benchmark."""

    def test_full_scaling_benchmark(self):
        """Run the full scaling benchmark from 10k to 200k miners."""
        miner_counts = [10000, 25000, 50000, 75000, 100000, 150000, 200000]

        results, analysis = run_scaling_benchmark(
            miner_counts=miner_counts,
            n_days=60,
            verbose=True,
        )

        # Verify all runs succeeded
        for result in results:
            self.assertTrue(
                result.success,
                f"Benchmark failed for {result.n_miners} miners: {result.error_message}"
            )

        # Verify scaling is roughly linear (O(n))
        if analysis:
            # Time should scale roughly linearly
            time_ratio = analysis.total_times[-1] / analysis.total_times[0]
            miner_ratio = analysis.miner_counts[-1] / analysis.miner_counts[0]

            # Allow for some overhead, but should be within 3x of linear
            self.assertLess(
                time_ratio / miner_ratio, 3.0,
                f"Time scaling not linear: {time_ratio:.2f}x time for {miner_ratio:.2f}x miners"
            )

            # Memory should also scale roughly linearly
            mem_ratio = analysis.peak_memories_mb[-1] / analysis.peak_memories_mb[0]
            self.assertLess(
                mem_ratio / miner_ratio, 3.0,
                f"Memory scaling not linear: {mem_ratio:.2f}x memory for {miner_ratio:.2f}x miners"
            )

        # Create plots
        if analysis:
            create_scaling_plots(results, analysis, show_plots=False)
            save_results_json(results, analysis)

    def test_quick_scaling_check(self):
        """Quick sanity check with smaller miner counts."""
        miner_counts = [5000, 10000, 20000]

        results, analysis = run_scaling_benchmark(
            miner_counts=miner_counts,
            n_days=30,
            verbose=True,
        )

        for result in results:
            self.assertTrue(result.success, f"Failed at {result.n_miners}: {result.error_message}")
            self.assertGreater(result.scores_generated, 0)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for running the benchmark directly."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run debt scoring scaling benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_debt_scoring_scaling_benchmark.py
  python test_debt_scoring_scaling_benchmark.py --max-miners 100000
  python test_debt_scoring_scaling_benchmark.py --quick
  python test_debt_scoring_scaling_benchmark.py --custom 10000 50000 100000
        """
    )
    parser.add_argument(
        '--max-miners', type=int, default=200000,
        help='Maximum number of miners to test (default: 200000)'
    )
    parser.add_argument(
        '--days', type=int, default=60,
        help='Number of days of history per miner (default: 60)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel workers (default: auto)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with fewer data points'
    )
    parser.add_argument(
        '--custom', type=int, nargs='+',
        help='Custom list of miner counts to test'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--show-plot', action='store_true',
        help='Display plots interactively'
    )

    args = parser.parse_args()

    # Determine miner counts
    if args.custom:
        miner_counts = sorted(args.custom)
    elif args.quick:
        miner_counts = [5000, 10000, 25000, 50000]
    else:
        # Generate counts up to max
        base_counts = [10000, 25000, 50000, 75000, 100000, 150000, 200000]
        miner_counts = [c for c in base_counts if c <= args.max_miners]

    # Run benchmark
    results, analysis = run_scaling_benchmark(
        miner_counts=miner_counts,
        n_days=args.days,
        n_workers=args.workers,
        verbose=True,
    )

    # Generate outputs
    plot_path = None
    json_path = None

    if analysis:
        json_path = save_results_json(results, analysis)

        if not args.no_plot:
            plot_path = create_scaling_plots(
                results,
                analysis,
                show_plots=args.show_plot,
            )

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"Runs: {successful} successful, {failed} failed")

    if analysis:
        print(f"\nScaling Coefficients:")
        print(f"  Time:   {analysis.time_slope*1e6:.2f} ms per miner")
        print(f"  Memory: {analysis.memory_slope*1024:.1f} KB per miner")
        print(f"\nEstimates for 1M miners:")
        est_time = analysis.time_slope * 1_000_000 + analysis.time_intercept
        est_memory = (analysis.memory_slope * 1_000_000 + analysis.memory_intercept) / 1024
        print(f"  Time:   ~{est_time/60:.1f} minutes")
        print(f"  Memory: ~{est_memory:.1f} GB")

    # Print output file locations
    print("\n" + "-" * 70)
    print("OUTPUT FILES:")
    if json_path:
        print(f"  JSON:  {os.path.abspath(json_path)}")
    if plot_path:
        print(f"  Plot:  {os.path.abspath(plot_path)}")
        print(f"  HiRes: {os.path.abspath(plot_path.replace('.png', '_hires.png'))}")
    print("-" * 70)


if __name__ == '__main__':
    main()
