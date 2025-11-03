#!/usr/bin/env python3
"""
Local Debt Ledger Builder & Visualizer

Builds debt ledgers for miners based on their performance ledgers, penalties, and emissions.
Provides comprehensive visualization of debt-based scoring metrics.

This script loads positions from the database and builds complete debt checkpoints.
In single hotkey mode, it generates matplotlib plots showing:
- Penalties over time (drawdown, risk profile, min collateral, total)
- PnL performance (gain, loss, net PnL)
- Emissions received (ALPHA, TAO, USD)
- Portfolio metrics (return, max drawdown)

Usage:
    1. Set TEST_SINGLE_HOTKEY to a miner's hotkey (or None for all miners)
    2. Set SHOULD_PLOT = True to generate visualizations (requires single hotkey)
    3. Run: python runnable/local_debt_ledger.py
"""

import bittensor as bt
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.position_source import PositionSourceManager, PositionSource
from shared_objects.cache_controller import CacheController
from shared_objects.mock_metagraph import MockMetagraph
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_dataclasses.debt_ledger import DebtLedgerManager
from vali_objects.utils.asset_selection_manager import AssetSelectionManager


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Set to a specific hotkey to process single miner, or None for all miners
TEST_SINGLE_HOTKEY = '5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY'

# End time in milliseconds for position loading (None = all positions)
END_TIME_MS = None  # Example: 1736035200000

# Whether to generate matplotlib plots (only works in single hotkey mode)
SHOULD_PLOT = True

# Enable verbose/debug logging
VERBOSE = False

# Whether to use database for positions (True recommended)
USE_DATABASE_POSITIONS = True

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_penalties(debt_checkpoints, hotkey):
    """Plot penalty analysis over time"""
    if not debt_checkpoints:
        bt.logging.warning(f"No debt checkpoints found for {hotkey}")
        return

    bt.logging.info(f"Plotting penalties for {hotkey}")

    # Extract data for plotting
    timestamps = [datetime.fromtimestamp(cp.timestamp_ms / 1000, tz=timezone.utc)
                 for cp in debt_checkpoints]
    drawdown_penalties = [cp.drawdown_penalty for cp in debt_checkpoints]
    risk_profile_penalties = [cp.risk_profile_penalty for cp in debt_checkpoints]
    min_collateral_penalties = [cp.min_collateral_penalty for cp in debt_checkpoints]
    risk_adjusted_penalties = [cp.risk_adjusted_performance_penalty for cp in debt_checkpoints]
    total_penalties = [cp.total_penalty for cp in debt_checkpoints]

    # Get time range for title
    start_date = timestamps[0].strftime('%Y-%m-%d')
    end_date = timestamps[-1].strftime('%Y-%m-%d')

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Calculate min/max for legend labels
    dd_min, dd_max = min(drawdown_penalties), max(drawdown_penalties)
    rp_min, rp_max = min(risk_profile_penalties), max(risk_profile_penalties)
    mc_min, mc_max = min(min_collateral_penalties), max(min_collateral_penalties)
    ra_min, ra_max = min(risk_adjusted_penalties), max(risk_adjusted_penalties)
    total_min, total_max = min(total_penalties), max(total_penalties)

    # Plot all penalties
    ax.plot(timestamps, drawdown_penalties, 'b-', linewidth=2,
           label=f'Drawdown Threshold (min: {dd_min:.4f}, max: {dd_max:.4f})')
    ax.plot(timestamps, risk_profile_penalties, 'r-', linewidth=2,
           label=f'Risk Profile (min: {rp_min:.4f}, max: {rp_max:.4f})')
    ax.plot(timestamps, min_collateral_penalties, 'g-', linewidth=2,
           label=f'Min Collateral (min: {mc_min:.4f}, max: {mc_max:.4f})')
    ax.plot(timestamps, risk_adjusted_penalties, 'orange', linewidth=2,
           label=f'Risk Adjusted Performance (min: {ra_min:.4f}, max: {ra_max:.4f})')
    ax.plot(timestamps, total_penalties, color='purple', linewidth=2.5,
           label=f'Total Penalty (min: {total_min:.4f}, max: {total_max:.4f})')

    # Set title
    ax.set_title(f'Penalty Analysis for {hotkey}\n({start_date} to {end_date})',
                fontsize=14, pad=15)

    # Labels and grid
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Penalty Multiplier', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    plot_filename = f'runnable/debt_ledger_penalties_{hotkey}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    bt.logging.info(f"Penalties plot saved to: {plot_filename}")


def plot_pnl_performance(debt_checkpoints, hotkey):
    """Plot PnL performance over time"""
    if not debt_checkpoints:
        return

    bt.logging.info(f"Plotting PnL performance for {hotkey}")

    # Extract data
    timestamps = [datetime.fromtimestamp(cp.timestamp_ms / 1000, tz=timezone.utc)
                 for cp in debt_checkpoints]
    realized_pnls = [cp.realized_pnl for cp in debt_checkpoints]
    unrealized_pnls = [cp.unrealized_pnl for cp in debt_checkpoints]
    net_pnls = [cp.net_pnl for cp in debt_checkpoints]

    # Get time range
    start_date = timestamps[0].strftime('%Y-%m-%d')
    end_date = timestamps[-1].strftime('%Y-%m-%d')

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot PnL components
    ax.plot(timestamps, realized_pnls, 'g-', linewidth=2, label=f'Realized PnL (total: {sum(realized_pnls):.2f})')
    ax.plot(timestamps, unrealized_pnls, 'orange', linewidth=2, label=f'Unrealized PnL (total: {sum(unrealized_pnls):.2f})')
    ax.plot(timestamps, net_pnls, 'b-', linewidth=2.5, label=f'Net PnL - Payout (total: {sum(net_pnls):.2f})')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)

    # Set title
    ax.set_title(f'PnL Performance for {hotkey}\n({start_date} to {end_date})',
                fontsize=14, pad=15)

    # Labels and grid
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PnL (ALPHA)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    plot_filename = f'runnable/debt_ledger_pnl_{hotkey}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    bt.logging.info(f"PnL plot saved to: {plot_filename}")


def plot_emissions(debt_checkpoints, hotkey):
    """Plot emissions received over time"""
    if not debt_checkpoints:
        return

    bt.logging.info(f"Plotting emissions for {hotkey}")

    # Extract data
    timestamps = [datetime.fromtimestamp(cp.timestamp_ms / 1000, tz=timezone.utc)
                 for cp in debt_checkpoints]
    chunk_alpha = [cp.chunk_emissions_alpha for cp in debt_checkpoints]
    chunk_tao = [cp.chunk_emissions_tao for cp in debt_checkpoints]
    chunk_usd = [cp.chunk_emissions_usd for cp in debt_checkpoints]

    # Get time range
    start_date = timestamps[0].strftime('%Y-%m-%d')
    end_date = timestamps[-1].strftime('%Y-%m-%d')

    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    # Plot emissions
    total_alpha = sum(chunk_alpha)
    total_tao = sum(chunk_tao)
    total_usd = sum(chunk_usd)

    ax1.plot(timestamps, chunk_alpha, 'b-', linewidth=2, label=f'ALPHA (total: {total_alpha:.2f})')
    ax1.plot(timestamps, chunk_tao, 'g-', linewidth=2, label=f'TAO (total: {total_tao:.4f})')
    ax2.plot(timestamps, chunk_usd, 'r-', linewidth=2, label=f'USD (total: ${total_usd:.2f})')

    # Set title
    ax1.set_title(f'Emissions Received for {hotkey}\n({start_date} to {end_date})',
                 fontsize=14, pad=15)

    # Labels and grid
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('ALPHA / TAO', fontsize=12, color='b')
    ax2.set_ylabel('USD', fontsize=12, color='r')
    ax1.grid(True, alpha=0.3)

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legends
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    plot_filename = f'runnable/debt_ledger_emissions_{hotkey}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    bt.logging.info(f"Emissions plot saved to: {plot_filename}")


def plot_portfolio_metrics(debt_checkpoints, hotkey):
    """Plot portfolio return and max drawdown over time"""
    if not debt_checkpoints:
        return

    bt.logging.info(f"Plotting portfolio metrics for {hotkey}")

    # Extract data
    timestamps = [datetime.fromtimestamp(cp.timestamp_ms / 1000, tz=timezone.utc)
                 for cp in debt_checkpoints]
    portfolio_returns = [cp.portfolio_return for cp in debt_checkpoints]
    max_drawdowns = [cp.max_drawdown for cp in debt_checkpoints]

    # Get time range
    start_date = timestamps[0].strftime('%Y-%m-%d')
    end_date = timestamps[-1].strftime('%Y-%m-%d')

    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()

    # Plot metrics
    final_return = portfolio_returns[-1] if portfolio_returns else 1.0
    worst_dd = min(max_drawdowns) if max_drawdowns else 0.0

    ax1.plot(timestamps, portfolio_returns, 'g-', linewidth=2.5,
            label=f'Portfolio Return (final: {final_return:.4f})')
    ax2.plot(timestamps, max_drawdowns, 'r-', linewidth=2.5,
            label=f'Max Drawdown (worst: {worst_dd:.4f})')

    # Add reference lines
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')

    # Set title
    ax1.set_title(f'Portfolio Metrics for {hotkey}\n({start_date} to {end_date})',
                 fontsize=14, pad=15)

    # Labels and grid
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Return (multiplier)', fontsize=12, color='g')
    ax2.set_ylabel('Max Drawdown', fontsize=12, color='r')
    ax1.grid(True, alpha=0.3)

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legends
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # Save plot
    plot_filename = f'runnable/debt_ledger_portfolio_{hotkey}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    bt.logging.info(f"Portfolio metrics plot saved to: {plot_filename}")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Enable logging
    if VERBOSE:
        bt.logging.enable_debug()
    else:
        bt.logging.enable_info()

    # Validate configuration
    if SHOULD_PLOT and not TEST_SINGLE_HOTKEY:
        bt.logging.error("SHOULD_PLOT requires TEST_SINGLE_HOTKEY to be specified")
        exit(1)

    # Initialize components
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)

    # Determine which hotkeys to process
    if TEST_SINGLE_HOTKEY:
        hotkeys_to_process = [TEST_SINGLE_HOTKEY]
        bt.logging.info(f"Processing single hotkey: {TEST_SINGLE_HOTKEY}")
    else:
        hotkeys_to_process = all_hotkeys_on_disk
        bt.logging.info(f"Processing all {len(hotkeys_to_process)} hotkeys")

    # Load positions from database
    hk_to_positions = {}
    if USE_DATABASE_POSITIONS:
        source_type = PositionSource.DATABASE
        bt.logging.info("Using database as position source")

        position_source_manager = PositionSourceManager(source_type)
        hk_to_positions = position_source_manager.load_positions(
            end_time_ms=END_TIME_MS,
            hotkeys=hotkeys_to_process
        )

        if hk_to_positions:
            hotkeys_to_process = list(hk_to_positions.keys())
            bt.logging.info(f"Loaded positions for {len(hotkeys_to_process)} miners from database")

    # Initialize metagraph and managers
    mmg = MockMetagraph(hotkeys=hotkeys_to_process)
    elimination_manager = EliminationManager(mmg, None, None)
    position_manager = PositionManager(
        metagraph=mmg,
        running_unit_tests=False,
        elimination_manager=elimination_manager,
        is_backtesting=True
    )

    # Save loaded positions to position manager
    if hk_to_positions:
        position_count = 0
        for hk, positions in hk_to_positions.items():
            for pos in positions:
                position_manager.save_miner_position(pos)
                position_count += 1
        bt.logging.info(f"Saved {position_count} positions to position manager")

    # Create PerfLedgerManager
    bt.logging.info("Creating PerfLedgerManager...")
    perf_ledger_manager = PerfLedgerManager(
        mmg,
        position_manager=position_manager,
        running_unit_tests=False,
        enable_rss=False,
        build_portfolio_ledgers_only=True
    )

    # Build performance ledgers
    bt.logging.info("Building performance ledgers...")
    if TEST_SINGLE_HOTKEY:
        bt.logging.info(f"Building perf ledger for: {TEST_SINGLE_HOTKEY}")
        perf_ledger_manager.update(testing_one_hotkey=TEST_SINGLE_HOTKEY, t_ms=TimeUtil.now_in_millis())
    else:
        bt.logging.info("Building perf ledgers for all hotkeys")
        perf_ledger_manager.update()

    # Create ValidatorContractManager (with mock data for standalone mode)
    bt.logging.info("Creating ValidatorContractManager...")
    contract_manager = ValidatorContractManager(
        config=None,
        position_manager=position_manager,
        ipc_manager=None,
        metagraph=mmg,
        running_unit_tests=False
    )

    # Create AssetSelectionManager
    bt.logging.info("Creating AssetSelectionManager...")
    asset_selection_manager = AssetSelectionManager(
        config=None,
        metagraph=mmg,
        ipc_manager=None
    )

    # Create DebtLedgerManager
    bt.logging.info("Creating DebtLedgerManager...")
    debt_ledger_manager = DebtLedgerManager(
        perf_ledger_manager=perf_ledger_manager,
        position_manager=position_manager,
        contract_manager=contract_manager,
        asset_selection_manager=asset_selection_manager,
        challengeperiod_manager=position_manager.challengeperiod_manager,
        slack_webhook_url=None,
        start_daemon=False,  # Don't start daemon for local debugging
        ipc_manager=None,
        running_unit_tests=False,
        validator_hotkey=None
    )

    # Build debt ledgers manually (since daemon is not running)
    bt.logging.info("Building debt ledgers...")
    debt_ledger_manager.build_debt_ledgers(verbose=VERBOSE)

    # Print summary
    bt.logging.info("\n" + "="*60)
    bt.logging.info("Debt Ledger Summary")
    bt.logging.info("="*60)
    for hotkey, ledger in debt_ledger_manager.debt_ledgers.items():
        num_checkpoints = len(ledger.checkpoints) if ledger.checkpoints else 0
        bt.logging.info(f"Miner {hotkey[:12]}...: {num_checkpoints} debt checkpoints")

    # Generate plots if requested and in single hotkey mode
    if SHOULD_PLOT and TEST_SINGLE_HOTKEY:
        ledger = debt_ledger_manager.debt_ledgers.get(TEST_SINGLE_HOTKEY)

        if not ledger or not ledger.checkpoints:
            bt.logging.warning(f"No debt ledger found for {TEST_SINGLE_HOTKEY}")
        else:
            bt.logging.info(f"\nGenerating visualizations for {TEST_SINGLE_HOTKEY}")
            bt.logging.info(f"Total checkpoints: {len(ledger.checkpoints)}")

            # Generate all plots
            plot_penalties(ledger.checkpoints, TEST_SINGLE_HOTKEY)
            plot_pnl_performance(ledger.checkpoints, TEST_SINGLE_HOTKEY)
            plot_emissions(ledger.checkpoints, TEST_SINGLE_HOTKEY)
            plot_portfolio_metrics(ledger.checkpoints, TEST_SINGLE_HOTKEY)

            bt.logging.info("\nAll plots generated successfully!")

    bt.logging.info("\nDebtLedger processing complete!")
