#!/usr/bin/env python3
"""
Local Debt Ledger Builder & Visualizer

Builds debt ledgers for miners based on their performance ledgers, penalties, and emissions.
Provides comprehensive visualization of debt-based scoring metrics.

This script uses the DebtLedgerServer in LOCAL mode to build debt ledgers without
connecting to any RPC servers. All managers create their own internal clients.

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

import atexit
import signal
import sys
import bittensor as bt
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from vali_objects.vali_config import RPCConnectionMode
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_server import DebtLedgerServer


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# Set to a specific hotkey to process single miner, or None for all miners
TEST_SINGLE_HOTKEY = '5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY'

# Whether to generate matplotlib plots (only works in single hotkey mode)
SHOULD_PLOT = True

# Enable verbose/debug logging
VERBOSE = False

# Global server reference for cleanup
_debt_ledger_server = None

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

    # Get time range
    start_date = timestamps[0].strftime('%Y-%m-%d')
    end_date = timestamps[-1].strftime('%Y-%m-%d')

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot PnL components
    ax.plot(timestamps, realized_pnls, 'g-', linewidth=2, label=f'Realized PnL (total: {sum(realized_pnls):.2f})')
    ax.plot(timestamps, unrealized_pnls, 'orange', linewidth=2, label=f'Unrealized PnL (total: {sum(unrealized_pnls):.2f})')

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
# CLEANUP FUNCTIONS
# ============================================================================

def cleanup():
    """Cleanup function to properly shutdown servers on exit."""
    global _debt_ledger_server
    if _debt_ledger_server is not None:
        bt.logging.info("Shutting down DebtLedgerServer...")
        try:
            _debt_ledger_server.shutdown()
        except Exception as e:
            bt.logging.warning(f"Error during shutdown: {e}")
        _debt_ledger_server = None
        bt.logging.info("Shutdown complete.")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    bt.logging.info(f"\nReceived signal {signum}, cleaning up...")
    cleanup()
    sys.exit(0)


# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Enable logging
    if VERBOSE:
        bt.logging.enable_debug()
    else:
        bt.logging.enable_info()

    # Validate configuration
    if SHOULD_PLOT and not TEST_SINGLE_HOTKEY:
        bt.logging.error("SHOULD_PLOT requires TEST_SINGLE_HOTKEY to be specified")
        exit(1)

    try:
        # Create DebtLedgerServer in LOCAL mode (no RPC connections, no daemon)
        # The server creates all required managers internally (forward compatibility pattern)
        bt.logging.info("Creating DebtLedgerServer in LOCAL mode...")
        _debt_ledger_server = DebtLedgerServer(
            slack_webhook_url=None,
            running_unit_tests=True,  # Bypass RPC connections
            validator_hotkey=None,
            start_server=False,  # No RPC server needed
            start_daemon=False,  # No daemon - we'll call build manually
            connection_mode=RPCConnectionMode.LOCAL  # Direct mode
        )

        # Build debt ledgers manually (this calls penalty + emissions + debt builds internally)
        bt.logging.info("Building debt ledgers...")
        bt.logging.info("This will build: penalty ledgers -> emissions ledgers -> debt ledgers")
        _debt_ledger_server.run_daemon_iteration()

        # Print summary
        bt.logging.info("\n" + "="*60)
        bt.logging.info("Debt Ledger Summary")
        bt.logging.info("="*60)
        for hotkey, ledger in _debt_ledger_server.debt_ledgers.items():
            num_checkpoints = len(ledger.checkpoints) if ledger.checkpoints else 0
            bt.logging.info(f"Miner {hotkey[:12]}...: {num_checkpoints} debt checkpoints")

        # Generate plots if requested and in single hotkey mode
        if SHOULD_PLOT and TEST_SINGLE_HOTKEY:
            ledger = _debt_ledger_server.debt_ledgers.get(TEST_SINGLE_HOTKEY)

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

    finally:
        # Ensure cleanup runs even on exceptions
        cleanup()
