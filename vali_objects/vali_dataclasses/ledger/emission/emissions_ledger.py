"""
Emissions Ledger - Tracks theta (TAO) emissions for hotkeys in 12-hour UTC chunks

This module builds emissions ledgers by querying on-chain data to track how much theta
has been awarded to each hotkey over its entire history since registration.

Emissions are tracked in 12-hour chunks aligned with UTC day:
- Chunk 1: 00:00 UTC - 12:00 UTC
- Chunk 2: 12:00 UTC - 00:00 UTC (next day)

Each checkpoint stores:
- Chunk emissions (alpha/TAO/USD) for that specific 12-hour period
- Average alpha-to-TAO and TAO-to-USD conversion rates for the chunk
- Number of blocks sampled in the chunk

Cumulative values are calculated dynamically by summing across checkpoints.

Architecture:
- EmissionsCheckpoint: Data for a single 12-hour chunk
- EmissionsLedger: Emissions history for a SINGLE hotkey
- EmissionsLedgerManager: Builds and manages ledgers for multiple hotkeys

Standalone Usage:
    python -m vali_objects.vali_dataclasses.emissions_ledger --hotkey <hotkey> --netuid 8
"""
import os
import gzip
import json
import shutil
import signal
import argparse
import multiprocessing
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import bittensor as bt
import time
import scalecodec

from time_util.time_util import TimeUtil
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.slack_notifier import SlackNotifier


@dataclass
class EmissionsCheckpoint:
    """
    Stores emissions data for a 12-hour UTC chunk.

    Attributes:
        chunk_start_ms: Start timestamp of the 12-hour chunk (milliseconds)
        chunk_end_ms: End timestamp of the 12-hour chunk (milliseconds)
        chunk_emissions: Alpha tokens earned during this specific 12-hour chunk
        chunk_emissions_tao: TAO value of chunk emissions (using avg conversion rate)
        chunk_emissions_usd: USD value of chunk emissions (using avg TAO/USD rate)
        avg_alpha_to_tao_rate: Average alpha-to-TAO conversion rate across the chunk (mandatory)
        avg_tao_to_usd_rate: Average TAO/USD price across the chunk (mandatory)
        num_blocks: Number of blocks sampled in this chunk
        block_start: Block number at chunk start (for reference)
        block_end: Block number at chunk end (for reference)
        tao_balance_snapshot: TAO balance at block_end (for validation)
        alpha_balance_snapshot: ALPHA balance at block_end (for validation)
    """
    chunk_start_ms: int
    chunk_end_ms: int
    chunk_emissions: float
    chunk_emissions_tao: float = 0.0
    chunk_emissions_usd: float = 0.0
    avg_alpha_to_tao_rate: float = 0.0
    avg_tao_to_usd_rate: float = 0.0
    num_blocks: int = 0
    block_start: Optional[int] = None
    block_end: Optional[int] = None
    tao_balance_snapshot: float = 0.0
    alpha_balance_snapshot: float = 0.0

    def __eq__(self, other):
        if not isinstance(other, EmissionsCheckpoint):
            return False
        return (
            self.chunk_start_ms == other.chunk_start_ms
            and self.chunk_end_ms == other.chunk_end_ms
            and abs(self.chunk_emissions - other.chunk_emissions) < 1e-9
        )

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'chunk_start_ms': self.chunk_start_ms,
            'chunk_end_ms': self.chunk_end_ms,
            'chunk_start_utc': datetime.fromtimestamp(self.chunk_start_ms / 1000, tz=timezone.utc).isoformat(),
            'chunk_end_utc': datetime.fromtimestamp(self.chunk_end_ms / 1000, tz=timezone.utc).isoformat(),
            'chunk_emissions': self.chunk_emissions,
            'chunk_emissions_tao': self.chunk_emissions_tao,
            'chunk_emissions_usd': self.chunk_emissions_usd,
            'avg_alpha_to_tao_rate': self.avg_alpha_to_tao_rate,
            'avg_tao_to_usd_rate': self.avg_tao_to_usd_rate,
            'num_blocks': self.num_blocks,
            'block_start': self.block_start,
            'block_end': self.block_end,
            'tao_balance_snapshot': self.tao_balance_snapshot,
            'alpha_balance_snapshot': self.alpha_balance_snapshot,
        }


class EmissionsLedger:
    """
    Emissions ledger for a SINGLE hotkey.

    Stores the complete emissions history as a series of 12-hour checkpoints,
    along with methods to query and visualize the data.
    """

    def __init__(self, hotkey: str, coldkey: str, checkpoints: Optional[List[EmissionsCheckpoint]] = None):
        """
        Initialize emissions ledger for a single hotkey.

        Args:
            hotkey: SS58 address of the hotkey
            coldkey: SS58 address of the coldkey (owner of this hotkey)
            checkpoints: Optional list of emission checkpoints
        """
        self.hotkey = hotkey
        self.coldkey = coldkey
        self.checkpoints: List[EmissionsCheckpoint] = checkpoints or []

    def add_checkpoint(self, checkpoint: EmissionsCheckpoint, target_cp_duration_ms: int):
        """
        Add a checkpoint to the ledger.

        Validates that the new checkpoint is properly aligned with the target checkpoint
        duration and the previous checkpoint (no gaps, no overlaps).

        Args:
            checkpoint: The checkpoint to add
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Raises:
            AssertionError: If checkpoint validation fails
        """
        # Validate checkpoint end time aligns with target duration
        assert checkpoint.chunk_end_ms % target_cp_duration_ms == 0, (
            f"Checkpoint end time {checkpoint.chunk_end_ms} must align with target_cp_duration_ms "
            f"{target_cp_duration_ms} for {self.hotkey}"
        )

        # Validate checkpoint duration is exactly target_cp_duration_ms
        checkpoint_duration_ms = checkpoint.chunk_end_ms - checkpoint.chunk_start_ms
        assert checkpoint_duration_ms == target_cp_duration_ms, (
            f"Checkpoint duration must be exactly {target_cp_duration_ms}ms for {self.hotkey}: "
            f"chunk spans {checkpoint.chunk_start_ms} to {checkpoint.chunk_end_ms} "
            f"({datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} - "
            f"{datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}), "
            f"duration is {checkpoint_duration_ms}ms"
        )

        # If there are existing checkpoints, ensure perfect boundary alignment
        if self.checkpoints:
            prev_checkpoint = self.checkpoints[-1]

            # First check it's after previous checkpoint
            assert checkpoint.chunk_end_ms > prev_checkpoint.chunk_end_ms, (
                f"Checkpoint end time must be after previous checkpoint for {self.hotkey}: "
                f"new checkpoint ends at {checkpoint.chunk_end_ms}, "
                f"but previous checkpoint ends at {prev_checkpoint.chunk_end_ms}"
            )

            # Then check exact spacing - checkpoints must be contiguous
            assert checkpoint.chunk_start_ms == prev_checkpoint.chunk_end_ms, (
                f"Checkpoint boundary misalignment for {self.hotkey}: "
                f"new checkpoint starts at {checkpoint.chunk_start_ms} "
                f"({datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}), "
                f"but previous checkpoint ended at {prev_checkpoint.chunk_end_ms} "
                f"({datetime.fromtimestamp(prev_checkpoint.chunk_end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}). "
                f"Expected perfect alignment (no gaps, no overlaps)."
            )

        self.checkpoints.append(checkpoint)

    def get_checkpoint_at_time(self, timestamp_ms: int, target_cp_duration_ms: int) -> Optional[EmissionsCheckpoint]:
        """
        Get the checkpoint at a specific timestamp (efficient O(1) lookup).

        Uses index calculation instead of scanning since checkpoints are evenly-spaced
        and contiguous (enforced by strict add_checkpoint validation).

        Args:
            timestamp_ms: Exact timestamp to query (should match chunk_end_ms)
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Returns:
            Checkpoint at the exact timestamp, or None if not found

        Raises:
            ValueError: If checkpoint exists at calculated index but timestamp doesn't match (data corruption)
        """
        if not self.checkpoints:
            return None

        # Calculate expected index based on first checkpoint and duration
        first_checkpoint_end_ms = self.checkpoints[0].chunk_end_ms

        # Check if timestamp is before first checkpoint
        if timestamp_ms < first_checkpoint_end_ms:
            return None

        # Calculate index (checkpoints are evenly spaced by target_cp_duration_ms)
        time_diff = timestamp_ms - first_checkpoint_end_ms
        if time_diff % target_cp_duration_ms != 0:
            # Timestamp doesn't align with checkpoint boundaries
            return None

        index = time_diff // target_cp_duration_ms

        # Check if index is within bounds
        if index >= len(self.checkpoints):
            return None

        # Validate the checkpoint at this index has the expected end timestamp
        checkpoint = self.checkpoints[index]
        if checkpoint.chunk_end_ms != timestamp_ms:
            from time_util.time_util import TimeUtil
            raise ValueError(
                f"Data corruption detected for {self.hotkey}: "
                f"checkpoint at index {index} has chunk_end_ms {checkpoint.chunk_end_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(checkpoint.chunk_end_ms)}), "
                f"but expected {timestamp_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(timestamp_ms)}). "
                f"Checkpoints are not properly contiguous."
            )

        return checkpoint

    def get_cumulative_emissions(self) -> float:
        """
        Get total cumulative alpha emissions for this hotkey.

        Calculates by summing chunk_emissions across all checkpoints.

        Returns:
            Total alpha emissions (float)
        """
        return sum(cp.chunk_emissions for cp in self.checkpoints)

    def get_cumulative_emissions_tao(self) -> float:
        """
        Get total cumulative TAO emissions for this hotkey.

        Calculates by summing chunk_emissions_tao across all checkpoints.

        Returns:
            Total TAO emissions (float)
        """
        return sum(cp.chunk_emissions_tao for cp in self.checkpoints)

    def get_cumulative_emissions_usd(self) -> float:
        """
        Get total cumulative USD emissions for this hotkey.

        Calculates by summing chunk_emissions_usd across all checkpoints.

        Returns:
            Total USD emissions (float)
        """
        return sum(cp.chunk_emissions_usd for cp in self.checkpoints)

    def to_dict(self) -> dict:
        """
        Convert ledger to dictionary for serialization.

        Returns:
            Dictionary with hotkey, coldkey, and all checkpoints
        """
        return {
            'hotkey': self.hotkey,
            'coldkey': self.coldkey,
            'total_checkpoints': len(self.checkpoints),
            'cumulative_emissions': self.get_cumulative_emissions(),
            'cumulative_emissions_tao': self.get_cumulative_emissions_tao(),
            'cumulative_emissions_usd': self.get_cumulative_emissions_usd(),
            'checkpoints': [cp.to_dict() for cp in self.checkpoints]
        }

    def print_summary(self):
        """Print a formatted summary of emissions for this hotkey."""
        if not self.checkpoints:
            print(f"\nNo emissions data found for {self.hotkey}")
            return

        print(f"\n{'='*80}")
        print(f"Emissions Summary for {self.hotkey}")
        print(f"{'='*80}")
        print(f"Total Checkpoints: {len(self.checkpoints)}")
        print(f"Total Emissions: {self.get_cumulative_emissions():.6f} alpha (~{self.get_cumulative_emissions_tao():.6f} TAO)")
        print(f"\nFirst 5 Checkpoints:")
        print(f"{'Chunk Start (UTC)':<25} {'Chunk End (UTC)':<25} {'Chunk Alpha':>15} {'Cumulative Alpha':>15}")
        print(f"{'-'*80}")

        cumulative_alpha = 0.0
        for i, checkpoint in enumerate(self.checkpoints[:5]):
            cumulative_alpha += checkpoint.chunk_emissions
            start_dt = datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc)
            print(f"{start_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                  f"{end_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                  f"{checkpoint.chunk_emissions:>15.6f} "
                  f"{cumulative_alpha:>15.6f}")

        if len(self.checkpoints) > 10:
            print(f"{'...':<25} {'...':<25} {'...':>15} {'...':>15}")
            print(f"\nLast 5 Checkpoints:")
            print(f"{'Chunk Start (UTC)':<25} {'Chunk End (UTC)':<25} {'Chunk Alpha':>15} {'Cumulative Alpha':>15}")
            print(f"{'-'*80}")

            # Calculate cumulative up to the start of last 5 checkpoints
            cumulative_alpha = sum(cp.chunk_emissions for cp in self.checkpoints[:-5])
            for checkpoint in self.checkpoints[-5:]:
                cumulative_alpha += checkpoint.chunk_emissions
                start_dt = datetime.fromtimestamp(checkpoint.chunk_start_ms / 1000, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(checkpoint.chunk_end_ms / 1000, tz=timezone.utc)
                print(f"{start_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                      f"{end_dt.strftime('%Y-%m-%d %H:%M:%S'):<25} "
                      f"{checkpoint.chunk_emissions:>15.6f} "
                      f"{cumulative_alpha:>15.6f}")

        print(f"{'='*80}\n")

    def plot_emissions(self, save_path: Optional[str] = None):
        """
        Plot emissions data using matplotlib.

        Creates two subplots:
        1. Bar chart of chunk emissions over time (alpha & TAO)
        2. Line chart of cumulative emissions over time (alpha & TAO)

        Args:
            save_path: Optional path to save the plot (default: display only)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            bt.logging.warning("matplotlib not installed, skipping plot. Install with: pip install matplotlib")
            return

        if not self.checkpoints:
            bt.logging.warning(f"No emissions data found for {self.hotkey}, skipping plot")
            return

        # Extract data for plotting
        chunk_starts = [datetime.fromtimestamp(cp.chunk_start_ms / 1000, tz=timezone.utc) for cp in self.checkpoints]
        chunk_emissions = [cp.chunk_emissions for cp in self.checkpoints]
        chunk_emissions_tao = [cp.chunk_emissions_tao for cp in self.checkpoints]

        # Calculate cumulative values dynamically
        cumulative_emissions = []
        cumulative_emissions_tao = []
        cumulative_alpha = 0.0
        cumulative_tao = 0.0
        for cp in self.checkpoints:
            cumulative_alpha += cp.chunk_emissions
            cumulative_tao += cp.chunk_emissions_tao
            cumulative_emissions.append(cumulative_alpha)
            cumulative_emissions_tao.append(cumulative_tao)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Emissions Analysis for {self.hotkey[:16]}...{self.hotkey[-8:]}', fontsize=14, fontweight='bold')

        # Subplot 1: Chunk Emissions (Bar Chart with dual y-axes)
        ax1_alpha = ax1
        ax1_alpha.bar(chunk_starts, chunk_emissions, width=0.4, alpha=0.6, color='steelblue', edgecolor='navy', label='Alpha')
        ax1_alpha.set_xlabel('Date (UTC)', fontsize=11)
        ax1_alpha.set_ylabel('Alpha per Chunk', fontsize=11, color='steelblue')
        ax1_alpha.tick_params(axis='y', labelcolor='steelblue')
        ax1_alpha.set_title('Emissions per 12-Hour Chunk (Alpha & TAO)', fontsize=12, fontweight='bold')
        ax1_alpha.grid(True, alpha=0.3, linestyle='--')

        # TAO emissions on right y-axis
        ax1_tao = ax1_alpha.twinx()
        ax1_tao.plot(chunk_starts, chunk_emissions_tao, linewidth=2, color='orange', marker='o',
                    markersize=4, alpha=0.8, label='TAO')
        ax1_tao.set_ylabel('TAO per Chunk', fontsize=11, color='orange')
        ax1_tao.tick_params(axis='y', labelcolor='orange')

        # Format x-axis
        ax1_alpha.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1_alpha.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.checkpoints)//20)))
        plt.setp(ax1_alpha.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add statistics text box
        total_emissions = cumulative_emissions[-1] if cumulative_emissions else 0
        total_emissions_tao = cumulative_emissions_tao[-1] if cumulative_emissions_tao else 0
        avg_chunk = sum(chunk_emissions) / len(chunk_emissions) if chunk_emissions else 0
        avg_chunk_tao = sum(chunk_emissions_tao) / len(chunk_emissions_tao) if chunk_emissions_tao else 0
        nonzero_chunks = sum(1 for e in chunk_emissions if e > 0)

        stats_text = f'Total: {total_emissions:.6f} alpha (~{total_emissions_tao:.6f} TAO)\n'
        stats_text += f'Avg/Chunk: {avg_chunk:.6f} alpha (~{avg_chunk_tao:.6f} TAO)\n'
        stats_text += f'Active Chunks: {nonzero_chunks}/{len(self.checkpoints)}'

        ax1_alpha.text(0.98, 0.97, stats_text,
                transform=ax1_alpha.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Subplot 2: Cumulative Emissions (Line Chart with dual y-axes)
        ax2_alpha = ax2
        ax2_alpha.plot(chunk_starts, cumulative_emissions, linewidth=2, color='darkgreen', marker='o',
                      markersize=3, alpha=0.7, label='Alpha')
        ax2_alpha.fill_between(chunk_starts, cumulative_emissions, alpha=0.2, color='lightgreen')
        ax2_alpha.set_xlabel('Date (UTC)', fontsize=11)
        ax2_alpha.set_ylabel('Cumulative Alpha', fontsize=11, color='darkgreen')
        ax2_alpha.tick_params(axis='y', labelcolor='darkgreen')
        ax2_alpha.set_title('Cumulative Emissions Over Time (Alpha & TAO)', fontsize=12, fontweight='bold')
        ax2_alpha.grid(True, alpha=0.3, linestyle='--')

        # TAO emissions on right y-axis
        ax2_tao = ax2_alpha.twinx()
        ax2_tao.plot(chunk_starts, cumulative_emissions_tao, linewidth=2, color='darkorange', marker='s',
                    markersize=3, alpha=0.7, label='TAO')
        ax2_tao.fill_between(chunk_starts, cumulative_emissions_tao, alpha=0.2, color='peachpuff')
        ax2_tao.set_ylabel('Cumulative TAO', fontsize=11, color='darkorange')
        ax2_tao.tick_params(axis='y', labelcolor='darkorange')

        # Format x-axis
        ax2_alpha.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2_alpha.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.checkpoints)//20)))
        plt.setp(ax2_alpha.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add final value annotations
        if chunk_starts and cumulative_emissions:
            ax2_alpha.annotate(f'{total_emissions:.6f} alpha',
                        xy=(chunk_starts[-1], cumulative_emissions[-1]),
                        xytext=(-60, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkgreen'))

            ax2_tao.annotate(f'{total_emissions_tao:.6f} TAO',
                        xy=(chunk_starts[-1], cumulative_emissions_tao[-1]),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='darkorange',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='peachpuff', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkorange'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            bt.logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


class EmissionsLedgerManager:
    """
    Manages emissions tracking for Bittensor hotkeys.

    Queries on-chain data to build a historical record of emissions received
    by hotkeys, organized into 12-hour UTC-aligned chunks.

    The ledger tracks emissions from the time a miner first registered on the subnet
    until the current block, aggregating data into consistent time windows.
    """

    # Stay 12 hours behind current time (allow two chunks for data finality)
    DEFAULT_LAG_TIME_MS = 12 * 60 * 60 * 1000

    # Check for new chunks every hour
    DEFAULT_CHECK_INTERVAL_SECONDS = 3600

    # Bittensor blocks are produced every ~12 seconds
    SECONDS_PER_BLOCK = 12

    # Default offset in days from current time for emissions tracking
    DEFAULT_START_TIME_OFFSET_DAYS = 30

    def __init__(
        self,
        archive_endpoint: str = "wss://archive.chain.opentensor.ai:443",
        netuid: int = 8,
        rate_limit_per_second: float = 1.0,
        running_unit_tests: bool = False,
        slack_webhook_url: Optional[str] = None,
        start_daemon: bool = False,
        validator_hotkey: Optional[str] = None
    ):
        """
        Initialize EmissionsLedger with blockchain connection.

        Note: Creates its own PerfLedgerClient internally (forward compatibility).

        Args:
            archive_endpoint: archive node endpoint for historical queries.
            netuid: Subnet UID to query (default: 8 for mainnet PTN)
            rate_limit_per_second: Maximum queries per second (default: 1.0 for official endpoints)
            running_unit_tests: Whether this is being run in unit tests
            slack_webhook_url: Optional Slack webhook URL for failure notifications
            start_daemon: If True, automatically start daemon process running run_forever (default: False)
            validator_hotkey: Optional validator hotkey for notifications
        """
        # Create PerfLedgerClient internally for accessing perf ledger data
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
        self._perf_ledger_client = PerfLedgerClient()

        # Pickleable attributes
        self.archive_endpoint = archive_endpoint
        self.netuid = netuid
        self.rate_limit_per_second = rate_limit_per_second
        self.last_query_time = 0.0
        self.running_unit_tests = running_unit_tests
        # In-memory ledgers (normal Python dict - managed within DebtLedgerServer process)
        self.emissions_ledgers: Dict[str, EmissionsLedger] = {}
        # Daemon control
        self.running = False
        self.daemon_process: Optional[multiprocessing.Process] = None
        # Slack notifications
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)

        # Non-pickleable components (lazy-initialized when needed)
        self.subtensor = None
        self.live_price_fetcher = None

        self.load_from_disk()
        bt.logging.info("EmissionsLedgerManager initialized (non-pickleable components will be lazy-initialized)")

        # Start daemon process if requested
        if start_daemon:
            self._start_daemon_process()

    def _rate_limit(self):
        """
        Enforce rate limiting by sleeping if necessary.

        Ensures we don't exceed rate_limit_per_second requests per second.
        """
        if self.rate_limit_per_second <= 0:
            return  # No rate limiting

        current_time = time.time()
        time_since_last_query = current_time - self.last_query_time
        min_interval = 1.0 / self.rate_limit_per_second

        if time_since_last_query < min_interval:
            sleep_time = min_interval - time_since_last_query
            time.sleep(sleep_time)

        self.last_query_time = time.time()

    def _start_daemon_process(
        self,
        check_interval_seconds: Optional[int] = None,
        lag_time_ms: Optional[int] = None
    ):
        """
        Start a daemon process running run_forever.

        The process is marked as a daemon so it will automatically terminate
        when the main program exits.

        Args:
            check_interval_seconds: How often to check for new chunks (default: 3600s = 1 hour)
            lag_time_ms: How far behind current time to stay (default: 12 hours)
        """
        if self.daemon_process is not None and self.daemon_process.is_alive():
            bt.logging.warning("Daemon process already running")
            return

        bt.logging.info("Starting emissions ledger daemon process")

        # Create daemon process
        self.daemon_process = multiprocessing.Process(
            target=self.run_forever,
            kwargs={
                'check_interval_seconds': check_interval_seconds,
                'lag_time_ms': lag_time_ms
            },
            daemon=True,
            name="EmissionsLedgerDaemon"
        )

        # Start the process
        self.daemon_process.start()

        bt.logging.info(f"Daemon process started (Process ID: {self.daemon_process.pid})")

    def _extract_block_timestamp(self, block_data: dict, block_number: int) -> int:
        """
        Extract timestamp from Bittensor/Substrate block data.

        Args:
            block_data (dict): Result of substrate.get_block()
            block_number (int): Block number (for logging/errors)

        Returns:
            int: Block timestamp in milliseconds.

        Raises:
            ValueError: If timestamp cannot be extracted.
        """
        if not block_data:
            raise ValueError(f"Block data is None or empty for block {block_number}")

        extrinsics = block_data.get("extrinsics", [])
        if not extrinsics:
            raise ValueError(f"No extrinsics found in block {block_number}")

        # Try all possible shapes for extrinsic call payloads
        for idx, extrinsic in enumerate(extrinsics):
            # Normalize access to dict form
            ext = (
                extrinsic.value
                if hasattr(extrinsic, "value") and isinstance(extrinsic.value, dict)
                else extrinsic
            )
            if not isinstance(ext, dict):
                continue

            # Look for "call" or "method" node
            call_node = ext.get("call") or ext.get("method")
            if not call_node:
                continue

            call = (
                call_node.value
                if hasattr(call_node, "value") and isinstance(call_node.value, dict)
                else call_node
            )
            if not isinstance(call, dict):
                continue

            if call.get("call_module") != "Timestamp" or call.get("call_function") != "set":
                continue

            # Parse possible argument shapes
            call_args = call.get("call_args", [])
            timestamp_ms = None

            # Common case: [{'name': 'now', 'type': 'Compact<u64>', 'value': 1690000000000}]
            if isinstance(call_args, list) and call_args and isinstance(call_args[0], dict):
                arg0 = call_args[0]
                if "value" in arg0:
                    timestamp_ms = int(arg0["value"])

            # Fallback: {'now': 1690000000000}
            elif isinstance(call_args, dict) and "now" in call_args:
                timestamp_ms = int(call_args["now"])

            if timestamp_ms is not None:
                return timestamp_ms

        # If no match found — log structural info
        first_ext = extrinsics[0] if extrinsics else None
        bt.logging.error(
            f"[extract_block_timestamp] Could not parse timestamp for block {block_number}. "
            f"First extrinsic: {type(first_ext)}, keys: {list(first_ext.keys()) if isinstance(first_ext, dict) else 'N/A'}"
        )
        raise ValueError(f"Failed to extract timestamp from block {block_number}")

    def query_alpha_to_tao_rate(self, block_number: int, block_hash: Optional[str] = None) -> float:
        """
        Query the alpha-to-TAO conversion rate at a specific block.
        Raises exception if conversion rate unavailable (fast fail).

        The conversion rate is calculated from subnet pool reserves:
        alpha_price_in_tao = TAO_reserve / Alpha_reserve

        Args:
            block_number: Block number to query
            block_hash: Optional pre-fetched block hash (avoids redundant query)

        Returns:
            Alpha-to-TAO conversion rate

        Raises:
            ValueError: If block hash, pool data, or conversion rate unavailable
        """
        # Ensure subtensor is initialized
        self.instantiate_non_pickleable_components()

        # Get block hash if not provided
        if block_hash is None:
            # Rate limit before block hash query
            self._rate_limit()
            block_hash = self.subtensor.substrate.get_block_hash(block_number)

        if not block_hash:
            raise ValueError(f"Could not get hash for block {block_number}")

        # Rate limit before SubnetTAO query
        self._rate_limit()

        # Query TAO reserve
        tao_reserve_query = self.subtensor.substrate.query(
            module='SubtensorModule',
            storage_function='SubnetTAO',
            params=[self.netuid],
            block_hash=block_hash
        )

        # Rate limit before SubnetAlphaIn query
        self._rate_limit()

        # Query Alpha reserve
        alpha_reserve_query = self.subtensor.substrate.query(
            module='SubtensorModule',
            storage_function='SubnetAlphaIn',
            params=[self.netuid],
            block_hash=block_hash
        )

        if tao_reserve_query is None or alpha_reserve_query is None:
            raise ValueError(f"Pool reserve data not available at block {block_number}")

        # Extract values (stored in RAO)
        tao_reserve_rao = float(tao_reserve_query.value if hasattr(tao_reserve_query, 'value') else tao_reserve_query)
        alpha_reserve_rao = float(alpha_reserve_query.value if hasattr(alpha_reserve_query, 'value') else alpha_reserve_query)

        if alpha_reserve_rao == 0:
            raise ValueError(f"Alpha reserve is zero at block {block_number}")

        # Convert from RAO to tokens (both use 1e9 divisor)
        tao_in_pool = tao_reserve_rao / 1e9
        alpha_in_pool = alpha_reserve_rao / 1e9

        # Calculate alpha price in TAO
        alpha_to_tao_rate = tao_in_pool / alpha_in_pool

        return alpha_to_tao_rate

    def query_tao_to_usd_rate(self, block_number: int, block_timestamp_ms: int) -> float:
        """
        Query TAO/USD rate at a specific block using its timestamp.
        Raises exception if price unavailable (fast fail).

        Args:
            block_number: Block number (for error messages)
            block_timestamp_ms: Block timestamp in milliseconds (pre-extracted)

        Returns:
            TAO/USD price at the block

        Raises:
            ValueError: If price data unavailable
        """
        # Query price using actual block timestamp
        price_source = self.live_price_fetcher.get_close_at_date(
            TradePair.TAOUSD,
            block_timestamp_ms
        )

        # Fast fail if price unavailable
        if price_source is None or price_source.close is None:
            raise ValueError(
                f"Failed to fetch TAO/USD price for block {block_number} "
                f"(timestamp: {TimeUtil.millis_to_formatted_date_str(block_timestamp_ms)})"
            )

        return price_source.close

    def _query_tao_balance_at_block(
        self,
        hotkey_ss58: str,
        block_hash: str
    ) -> float:
        """
        Query free TAO balance for a hotkey at a specific block.

        Args:
            hotkey_ss58: SS58 address of the hotkey
            block_hash: Block hash to query at

        Returns:
            TAO balance in tokens (not RAO)

        Raises:
            ValueError: If query returns None or invalid data
            Exception: If substrate query fails
        """
        # In test mode, return mock balance
        if self.running_unit_tests:
            return 100.0  # Mock TAO balance

        self._rate_limit()

        try:
            account_info = self.subtensor.substrate.query(
                module='System',
                storage_function='Account',
                params=[hotkey_ss58],
                block_hash=block_hash
            )
        except Exception as e:
            raise Exception(
                f"Failed to query TAO balance for {hotkey_ss58} at block {block_hash}: {e}"
            )

        if account_info is None:
            raise ValueError(
                f"TAO balance query returned None for {hotkey_ss58} at block {block_hash}"
            )

        try:
            free_balance_rao = account_info.get('data', {}).get('free', 0)
            tao_balance = float(free_balance_rao) / 1e9
        except Exception as e:
            raise ValueError(
                f"Failed to parse TAO balance from account_info for {hotkey_ss58}: {e}"
            )

        return tao_balance

    def _get_coldkey_for_hotkey(self, hotkey_ss58: str) -> str:
        """
        Get coldkey for a hotkey, checking ledger first then querying substrate.

        If ledger exists but doesn't have coldkey, queries substrate and updates ledger.
        This ensures coldkey is persisted for future lookups and disk saves.

        Args:
            hotkey_ss58: SS58 address of the hotkey

        Returns:
            Coldkey SS58 address

        Raises:
            ValueError: If no coldkey found
            Exception: If substrate query fails
        """
        # Check if ledger exists and has coldkey
        if hotkey_ss58 in self.emissions_ledgers:
            ledger = self.emissions_ledgers[hotkey_ss58]
            if ledger.coldkey:
                return ledger.coldkey

        # In test mode, return mock coldkey
        if self.running_unit_tests:
            mock_coldkey = f"5Mock{hotkey_ss58[5:48]}"  # Mock coldkey based on hotkey
            # Update ledger with mock coldkey if ledger exists
            if hotkey_ss58 in self.emissions_ledgers:
                self.emissions_ledgers[hotkey_ss58].coldkey = mock_coldkey
            return mock_coldkey

        # Query substrate for coldkey
        self._rate_limit()

        try:
            coldkey = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='Owner',
                params=[hotkey_ss58]
            )

            if not coldkey:
                raise ValueError(f"No coldkey found for hotkey {hotkey_ss58}")

            coldkey_str = str(coldkey)

            # Update ledger with coldkey if ledger exists
            if hotkey_ss58 in self.emissions_ledgers:
                self.emissions_ledgers[hotkey_ss58].coldkey = coldkey_str
                bt.logging.debug(f"Updated ledger for {hotkey_ss58[:16]}... with coldkey {coldkey_str[:16]}...")

            return coldkey_str

        except Exception as e:
            raise Exception(
                f"Failed to query coldkey (Owner) for hotkey {hotkey_ss58}: {e}"
            )

    def _query_alpha_balance_at_block(
        self,
        hotkey_ss58: str,
        block_number: int
    ) -> float:
        """
        Query ALPHA balance (staked alpha) for a hotkey at a specific block.

        Uses subtensor.get_stake() which queries the alpha stake amount for a hotkey
        within this subnet. Alpha emissions are automatically staked to the hotkey.

        Args:
            hotkey_ss58: SS58 address of the hotkey
            block_number: Block number to query at

        Returns:
            ALPHA balance in tokens

        Raises:
            ValueError: If query returns invalid data
            Exception: If substrate query fails
        """
        # In test mode, return mock balance
        if self.running_unit_tests:
            return 50.0  # Mock ALPHA balance

        # Get coldkey for this hotkey (checks ledger first, then queries)
        coldkey = self._get_coldkey_for_hotkey(hotkey_ss58)

        try:
            # Rate limit before stake query
            self._rate_limit()

            # Query the alpha stake using get_stake()
            stake_balance = self.subtensor.get_stake(
                coldkey_ss58=coldkey,
                hotkey_ss58=hotkey_ss58,
                netuid=self.netuid,
                block=block_number
            )

            # Convert Balance object to float (note: .tao represents ALPHA when netuid != 0)
            alpha_balance = float(stake_balance.tao)

        except Exception as e:
            raise Exception(
                f"Failed to query ALPHA stake for hotkey {hotkey_ss58} (coldkey {coldkey}) "
                f"at block {block_number}: {e}"
            )

        return alpha_balance

    def instantiate_non_pickleable_components(self):
        """
        Lazy-initialize non-pickleable components (subtensor, live_price_fetcher).

        This is called automatically when needed, allowing the manager to be pickled
        for multiprocessing without serialization errors.
        """
        # Initialize subtensor if not already initialized
        if self.subtensor is None:
            # Skip subtensor initialization in test mode (uses mock data instead)
            if self.running_unit_tests:
                bt.logging.debug("Skipping subtensor initialization in test mode (uses mock data)")
                return

            bt.logging.info(f"Initializing subtensor connection to {self.archive_endpoint}, netuid: {self.netuid}")

            parser = argparse.ArgumentParser()
            bt.subtensor.add_args(parser)
            config = bt.config(parser, args=[])

            # Override the chain endpoint
            config.subtensor.chain_endpoint = self.archive_endpoint

            # Clear network so it uses our custom endpoint
            config.subtensor.network = None

            self.subtensor = bt.subtensor(config=config)
            bt.logging.info(f"Connected to: {self.subtensor.chain_endpoint}")

        # Initialize live price fetcher client if not already initialized
        if self.live_price_fetcher is None:
            bt.logging.info("Initializing live price fetcher client")
            self.live_price_fetcher = LivePriceFetcherClient(running_unit_tests=self.running_unit_tests)

    def _query_rates_for_zero_emission_chunk(
        self,
        chunk_start_block: int,
        chunk_end_block: int
    ) -> tuple[float, float]:
        """
        Query conversion rates for chunks with no emissions.

        When a chunk has no emissions (all hotkeys have zero emissions), we still need
        to set mandatory rate fields. This method queries rates at the chunk midpoint block.

        Args:
            chunk_start_block: Start block of the chunk
            chunk_end_block: End block of the chunk

        Returns:
            Tuple of (avg_alpha_to_tao_rate, avg_tao_to_usd_rate). Returns (0.0, 0.0) on failure.
        """
        # In test mode, return mock rates
        if self.running_unit_tests:
            MOCK_ALPHA_TO_TAO_RATE = 1.0
            MOCK_TAO_TO_USD_RATE = 500.0
            return MOCK_ALPHA_TO_TAO_RATE, MOCK_TAO_TO_USD_RATE

        bt.logging.debug(
            f"No emissions found in chunk, querying rates directly for zero-emission checkpoints"
        )

        # Query rates at a representative block (chunk midpoint)
        midpoint_block = (chunk_start_block + chunk_end_block) // 2

        try:
            # Get block hash for midpoint block
            self._rate_limit()
            midpoint_block_hash = self.subtensor.substrate.get_block_hash(midpoint_block)

            if midpoint_block_hash:
                # Query alpha-to-TAO rate
                avg_alpha_to_tao_rate = self.query_alpha_to_tao_rate(
                    midpoint_block,
                    block_hash=midpoint_block_hash
                )

                # Get block timestamp for TAO/USD rate
                self._rate_limit()
                block_data = self.subtensor.substrate.get_block(block_hash=midpoint_block_hash)
                if block_data:
                    block_timestamp_ms = self._extract_block_timestamp(block_data, midpoint_block)
                    avg_tao_to_usd_rate = self.query_tao_to_usd_rate(
                        midpoint_block,
                        block_timestamp_ms=block_timestamp_ms
                    )
                    return avg_alpha_to_tao_rate, avg_tao_to_usd_rate
        except Exception as e:
            bt.logging.warning(
                f"Failed to query rates for zero-emission chunk: {e}. "
                f"Using default rates (0.0)"
            )

        return 0.0, 0.0

    def build_all_emissions_ledgers_optimized(
        self,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
    ):
        """
        Build emissions ledgers for ALL hotkeys in the subnet efficiently.

        Works IN-PLACE on self.emissions_ledgers, either creating new ledgers or
        appending checkpoints to existing ones with correct cumulative values.

        This is the optimized approach that queries each block once and processes
        all hotkeys simultaneously, sharing block hashes, emission vectors, and
        alpha-to-TAO conversion rates across all hotkeys.

        Emissions checkpoints are aligned with performance ledger checkpoints, ensuring
        consistency with performance tracking.

        Args:
            start_time_ms: Optional start time in milliseconds (default: DEFAULT_START_TIME_OFFSET_DAYS ago)
            end_time_ms: Optional end time (default: current time)
        """

        # In test mode, skip connection setup but still process all business logic
        if not self.running_unit_tests:
            self.instantiate_non_pickleable_components()

        start_exec_time = time.time()
        bt.logging.info("Building emissions ledgers for all hotkeys (aligned with perf ledgers)")
        if self.rate_limit_per_second < 10:
            bt.logging.warning(f"Emissions ledger network rate limit set to {self.rate_limit_per_second} req/sec - queries will be slow")

        # Get all perf ledgers (portfolio only) to use as checkpoint reference
        all_perf_ledgers: dict[str, 'PerfLedger'] = self._perf_ledger_client.get_perf_ledgers(
            portfolio_only=True
        )

        if not all_perf_ledgers:
            raise ValueError("No performance ledgers found - cannot build emissions without perf ledger alignment")

        # Pick a reference portfolio ledger (any miner will do since they're all aligned to same boundaries)
        # Use the one with the most checkpoints for maximum coverage
        reference_portfolio_ledger = None
        reference_hotkey = None
        max_checkpoints = 0

        for hotkey, ledger in all_perf_ledgers.items():
            portfolio_ledger = ledger  # Already a PerfLedger when portfolio_only=True

            if portfolio_ledger and portfolio_ledger.cps:
                if len(portfolio_ledger.cps) > max_checkpoints:
                    max_checkpoints = len(portfolio_ledger.cps)
                    reference_portfolio_ledger = portfolio_ledger
                    reference_hotkey = hotkey

        if not reference_portfolio_ledger:
            raise ValueError("No valid portfolio ledgers found with checkpoints")

        bt.logging.info(
            f"Using portfolio ledger from {reference_hotkey[:16]}...{reference_hotkey[-8:]} "
            f"as reference ({len(reference_portfolio_ledger.cps)} checkpoints, "
            f"target_cp_duration_ms: {reference_portfolio_ledger.target_cp_duration_ms}ms)"
        )

        # Rate limit before initial query (skip in test mode)
        if not self.running_unit_tests:
            self._rate_limit()

        # Verify max UIDs is still 256 (sanity check - has never changed in Bittensor history)
        if self.running_unit_tests:
            # In test mode, assume standard 256 UIDs
            current_max_uids = 256
        else:
            current_max_uids_result = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='SubnetworkN',
                params=[self.netuid]
            )
            current_max_uids = int(current_max_uids_result.value if hasattr(current_max_uids_result, 'value') else current_max_uids_result) if current_max_uids_result else 0
        assert current_max_uids == 256, f"Expected max UIDs to be 256, but got {current_max_uids}. The hardcoded value needs to be updated!"

        # Validate that start_time_ms doesn't conflict with existing data
        checkpoint_info = self.get_checkpoint_info()
        last_computed_chunk_end_ms = checkpoint_info["last_computed_chunk_end_ms"]

        if last_computed_chunk_end_ms > 0 and start_time_ms is not None:
            assert start_time_ms >= last_computed_chunk_end_ms, (
                f"start_time_ms must be >= last computed checkpoint end time. "
                f"Existing data ends at: {TimeUtil.millis_to_formatted_date_str(last_computed_chunk_end_ms)} "
                f"({last_computed_chunk_end_ms}), but requested start_time_ms is: "
                f"{TimeUtil.millis_to_formatted_date_str(start_time_ms)} ({start_time_ms}). "
                f"Caller must provide valid start time that continues from existing data."
            )

        # Default end time is now with lag
        current_time_ms = int(time.time() * 1000)
        if end_time_ms is None:
            # In test mode, use no lag since test data is created "now"
            lag_ms = 0 if self.running_unit_tests else self.DEFAULT_LAG_TIME_MS
            end_time_ms = current_time_ms - lag_ms

        # CRITICAL: Enforce 12-hour lag to ensure we never build checkpoints too close to real-time
        # This prevents incomplete or unreliable data from being included
        # Skip this enforcement in test mode
        if not self.running_unit_tests:
            min_allowed_end_time_ms = current_time_ms - self.DEFAULT_LAG_TIME_MS
            if end_time_ms > min_allowed_end_time_ms:
                bt.logging.warning(
                    f"Requested end_time_ms ({TimeUtil.millis_to_formatted_date_str(end_time_ms)}) "
                    f"is too recent (within {self.DEFAULT_LAG_TIME_MS / 1000 / 3600:.1f} hours of current time). "
                    f"Adjusting to enforce mandatory {self.DEFAULT_LAG_TIME_MS / 1000 / 3600:.1f}-hour lag: "
                    f"{TimeUtil.millis_to_formatted_date_str(min_allowed_end_time_ms)}"
                )
                end_time_ms = min_allowed_end_time_ms

        # Filter perf checkpoints to those within our time range and that are complete (not active)
        target_cp_duration_ms = reference_portfolio_ledger.target_cp_duration_ms
        checkpoints_to_process = []

        # Determine the start cutoff time (use the more restrictive of the two)
        start_cutoff_ms = max(
            last_computed_chunk_end_ms if last_computed_chunk_end_ms > 0 else 0,
            start_time_ms if start_time_ms is not None else 0
        )

        for checkpoint in reference_portfolio_ledger.cps:
            # Skip active checkpoints (incomplete)
            if checkpoint.accum_ms != target_cp_duration_ms:
                continue

            checkpoint_time_ms = checkpoint.last_update_ms

            # Skip checkpoints before our start cutoff time
            if checkpoint_time_ms <= start_cutoff_ms:
                continue

            # Skip checkpoints after our end time
            if checkpoint_time_ms > end_time_ms:
                continue

            checkpoints_to_process.append(checkpoint)

        if not checkpoints_to_process:
            bt.logging.info("No new checkpoints to process")
            return

        bt.logging.info(
            f"Processing {len(checkpoints_to_process)} checkpoints "
            f"(from {TimeUtil.millis_to_formatted_date_str(checkpoints_to_process[0].last_update_ms)} "
            f"to {TimeUtil.millis_to_formatted_date_str(checkpoints_to_process[-1].last_update_ms)})"
        )

        # Get current block for estimating block ranges
        if not self.running_unit_tests:
            self._rate_limit()
            current_block = self.subtensor.get_current_block()
        else:
            # In test mode, estimate current block based on time
            # Bittensor blocks are ~12 seconds, assume we started at block 1M
            current_block = 1000000 + int(time.time() / self.SECONDS_PER_BLOCK)
        current_time_ms = int(time.time() * 1000)

        chunk_count = 0
        hotkeys_processed_cumulative = set()  # Track all unique hotkeys seen so far

        # Track all hotkeys we've seen across all blocks
        # IMPORTANT: Initialize from existing ledgers when resuming to ensure all hotkeys
        # continue to receive zero-emission checkpoints even if they're no longer active.
        # Additional hotkeys will be discovered from blockchain queries in _calculate_emissions_for_all_hotkeys.
        # NOTE: Emissions tracking is independent of perf ledgers - we use perf ledgers only for
        # time boundaries, not to filter which hotkeys get emissions checkpoints.
        all_hotkeys_seen = set(self.emissions_ledgers.keys()) if self.emissions_ledgers else set()

        if all_hotkeys_seen:
            bt.logging.info(f"Resuming with {len(all_hotkeys_seen)} hotkeys from existing ledgers")

        # Iterate over perf ledger checkpoints
        for checkpoint_idx, checkpoint in enumerate(checkpoints_to_process):
            chunk_count += 1
            chunk_start_time = time.time()

            # Checkpoint boundaries from perf ledger
            current_chunk_end_ms = checkpoint.last_update_ms

            # Calculate start time from previous checkpoint or target duration
            if checkpoint_idx == 0:
                # First checkpoint - calculate start from checkpoint duration
                current_chunk_start_ms = current_chunk_end_ms - target_cp_duration_ms
            else:
                # Use previous checkpoint's end time as this checkpoint's start time
                current_chunk_start_ms = checkpoints_to_process[checkpoint_idx - 1].last_update_ms

            # Calculate corresponding blocks using time from current time
            seconds_from_chunk_start = (current_time_ms - current_chunk_start_ms) / 1000
            seconds_from_chunk_end = (current_time_ms - current_chunk_end_ms) / 1000
            chunk_start_block = current_block - int(seconds_from_chunk_start / self.SECONDS_PER_BLOCK)
            chunk_end_block = current_block - int(seconds_from_chunk_end / self.SECONDS_PER_BLOCK)

            # Optional: Log block continuity info for debugging
            # Since we're aligned with perf ledger time boundaries, block numbers are just for reference
            if chunk_count > 1 and self.emissions_ledgers:
                sample_ledger = next(iter(self.emissions_ledgers.values()))
                if sample_ledger.checkpoints and sample_ledger.checkpoints[-1].block_end is not None:
                    previous_block_end = sample_ledger.checkpoints[-1].block_end
                    expected_start_block = previous_block_end + 1
                    block_gap = abs(chunk_start_block - expected_start_block)

                    # Log warning if gap is significant (blocks are for reference only)
                    if block_gap > 100:
                        bt.logging.debug(
                            f"Block estimation drift: previous chunk ended at block {previous_block_end}, "
                            f"new chunk starts at block {chunk_start_block} "
                            f"(gap: {block_gap} blocks). This is normal due to block time variance."
                        )

            blocks_in_chunk = chunk_end_block - chunk_start_block + 1

            # Calculate emissions for ALL hotkeys in this chunk (optimized)
            chunk_results = self._calculate_emissions_for_all_hotkeys(
                chunk_start_block,
                chunk_end_block,
                all_hotkeys_seen,
            )

            # Track hotkeys with activity in this chunk
            hotkeys_active_in_chunk = set()

            # Get shared rate values and num_blocks for zero-emission checkpoints
            # These values are mandatory, so we must always ensure they are set
            shared_avg_alpha_to_tao_rate = 0.0
            shared_avg_tao_to_usd_rate = 0.0
            shared_num_blocks = 0

            if chunk_results:
                # Use rates from first hotkey with emissions (rates are same for all hotkeys)
                first_result = next(iter(chunk_results.values()))
                shared_avg_alpha_to_tao_rate = first_result[3]  # avg_alpha_to_tao_rate (index 3 in 6-tuple)
                shared_avg_tao_to_usd_rate = first_result[4]     # avg_tao_to_usd_rate (index 4 in 6-tuple)
                shared_num_blocks = first_result[5]              # num_blocks (index 5 in 6-tuple)
            elif all_hotkeys_seen:
                # No emissions in this chunk, but we still need to create zero-emission checkpoints
                # with valid rates. Query rates at the chunk midpoint block.
                shared_avg_alpha_to_tao_rate, shared_avg_tao_to_usd_rate = self._query_rates_for_zero_emission_chunk(
                    chunk_start_block,
                    chunk_end_block
                )

            # Query block hash for balance snapshots at checkpoint end
            if self.running_unit_tests:
                # In test mode, use a mock block hash
                end_block_hash = f"0x{'0' * 64}"  # Mock hash
            else:
                self._rate_limit()
                end_block_hash = self.subtensor.substrate.get_block_hash(chunk_end_block)
                if not end_block_hash:
                    raise ValueError(
                        f"Failed to get block_hash for block {chunk_end_block} "
                        f"(chunk {current_chunk_start_ms}-{current_chunk_end_ms})"
                    )

            # Single loop: create checkpoints for ALL hotkeys in all_hotkeys_seen
            # (includes both hotkeys with emissions and hotkeys without emissions)
            for hotkey in all_hotkeys_seen:
                # Check if this hotkey had emissions in this chunk
                if hotkey in chunk_results:
                    # Hotkey has emissions - use data from chunk_results
                    alpha_emissions, tao_emissions, usd_emissions, avg_alpha_to_tao_rate, avg_tao_to_usd_rate, num_blocks = chunk_results[hotkey]

                    # Create ledger if this is a newly discovered hotkey
                    if hotkey not in self.emissions_ledgers:
                        coldkey = self._get_coldkey_for_hotkey(hotkey)
                        self.emissions_ledgers[hotkey] = EmissionsLedger(hotkey, coldkey)

                    # Track if this hotkey had any emissions in this chunk
                    if alpha_emissions > 0:
                        hotkeys_active_in_chunk.add(hotkey)
                        hotkeys_processed_cumulative.add(hotkey)

                    # Query balance snapshots at checkpoint end
                    tao_balance = self._query_tao_balance_at_block(hotkey, end_block_hash)
                    alpha_balance = self._query_alpha_balance_at_block(hotkey, chunk_end_block)

                    checkpoint = EmissionsCheckpoint(
                        chunk_start_ms=current_chunk_start_ms,
                        chunk_end_ms=current_chunk_end_ms,
                        chunk_emissions=alpha_emissions,
                        chunk_emissions_tao=tao_emissions,
                        chunk_emissions_usd=usd_emissions,
                        avg_alpha_to_tao_rate=avg_alpha_to_tao_rate,
                        avg_tao_to_usd_rate=avg_tao_to_usd_rate,
                        num_blocks=num_blocks,
                        block_start=chunk_start_block,
                        block_end=chunk_end_block,
                        tao_balance_snapshot=tao_balance,
                        alpha_balance_snapshot=alpha_balance
                    )
                else:
                    # Hotkey exists but had no emissions in this chunk - create zero-emission checkpoint
                    # Create ledger if this is a newly discovered hotkey (discovered mid-chunk with no emissions)
                    if hotkey not in self.emissions_ledgers:
                        coldkey = self._get_coldkey_for_hotkey(hotkey)
                        self.emissions_ledgers[hotkey] = EmissionsLedger(hotkey, coldkey)

                    # Query balance snapshots at checkpoint end
                    tao_balance = self._query_tao_balance_at_block(hotkey, end_block_hash)
                    alpha_balance = self._query_alpha_balance_at_block(hotkey, chunk_end_block)

                    checkpoint = EmissionsCheckpoint(
                        chunk_start_ms=current_chunk_start_ms,
                        chunk_end_ms=current_chunk_end_ms,
                        chunk_emissions=0.0,
                        chunk_emissions_tao=0.0,
                        chunk_emissions_usd=0.0,
                        avg_alpha_to_tao_rate=shared_avg_alpha_to_tao_rate,
                        avg_tao_to_usd_rate=shared_avg_tao_to_usd_rate,
                        num_blocks=shared_num_blocks,
                        block_start=chunk_start_block,
                        block_end=chunk_end_block,
                        tao_balance_snapshot=tao_balance,
                        alpha_balance_snapshot=alpha_balance
                    )

                # Append checkpoint to ledger
                # IMPORTANT: For IPC-managed dicts, we must retrieve, mutate, and reassign
                # to propagate changes (managed dicts don't track nested mutations)
                ledger = self.emissions_ledgers[hotkey]
                ledger.add_checkpoint(checkpoint, target_cp_duration_ms)
                self.emissions_ledgers[hotkey] = ledger  # Reassign to trigger IPC update

            # Calculate chunk processing time
            chunk_elapsed = time.time() - chunk_start_time

            # Format chunk date range for logging
            chunk_start_dt = datetime.fromtimestamp(current_chunk_start_ms / 1000, tz=timezone.utc)
            chunk_end_dt = datetime.fromtimestamp(current_chunk_end_ms / 1000, tz=timezone.utc)
            date_range = f"{chunk_start_dt.strftime('%Y-%m-%d %H:%M')} - {chunk_end_dt.strftime('%Y-%m-%d %H:%M')} UTC"

            # Log progress for every chunk
            bt.logging.info(
                f"Chunk {chunk_count} ({date_range}): "
                f"{blocks_in_chunk} blocks, "
                f"{chunk_elapsed:.2f}s, "
                f"{len(hotkeys_active_in_chunk)} active hotkeys, "
                f"{len(hotkeys_processed_cumulative)} cumulative unique"
            )

            # Save ledger state after each successful chunk (incremental persistence for crash recovery)
            self.save_to_disk(create_backup=False)

        # Log completion (worked in-place on self.emissions_ledgers)
        elapsed_time = time.time() - start_exec_time
        bt.logging.info(
            f"Built {chunk_count} checkpoints for {len(self.emissions_ledgers)} hotkeys in {elapsed_time:.2f} seconds "
            f"(aligned with perf ledger, target_cp_duration_ms: {target_cp_duration_ms}ms)"
        )

        # Log summary for each hotkey
        for hotkey, ledger in self.emissions_ledgers.items():
            if ledger.checkpoints:
                total_alpha = ledger.get_cumulative_emissions()
                total_tao = ledger.get_cumulative_emissions_tao()
                bt.logging.info(f"  {hotkey[:16]}...{hotkey[-8:]}: {total_alpha:.6f} alpha (~{total_tao:.6f} TAO)")

    def _get_uid_to_hotkey_at_block(self, block_hash: str) -> Dict[int, str]:
        """
        Query UID-to-hotkey mapping at a specific historical block.

        Args:
            block_hash: Block hash to query at

        Returns:
            Dictionary mapping UID to hotkey at that block
        """
        # Ensure subtensor is initialized
        self.instantiate_non_pickleable_components()

        # Rate limit before query_map
        self._rate_limit()

        # Query all Keys entries for this netuid at once (much more efficient than 256 individual queries)
        keys_map = self.subtensor.substrate.query_map(
            module='SubtensorModule',
            storage_function='Keys',
            params=[self.netuid],
            block_hash=block_hash
        )

        # Convert result to dict[uid -> hotkey]
        uid_to_hotkey = {}
        for key, value in keys_map:
            # Key is the UID directly (integer)
            uid = int(key)

            # Value is a ScaleObj containing the hotkey as bytes
            # Convert to SS58 address string
            if hasattr(value, 'value'):
                # value.value is a list of tuples with byte values
                # Example: [(24, 91, 223, ...)] - extract bytes
                if isinstance(value.value, list) and len(value.value) > 0:
                    hotkey_bytes = bytes(value.value[0]) if isinstance(value.value[0], (list, tuple)) else bytes(value.value)
                else:
                    hotkey_bytes = bytes(value.value)

                # Convert bytes to SS58 address
                hotkey = scalecodec.ss58_encode(hotkey_bytes.hex(), ss58_format=42)

                if hotkey:
                    uid_to_hotkey[uid] = hotkey
            else:
                # Fallback: try to use value as-is
                hotkey = str(value)
                if hotkey:
                    uid_to_hotkey[uid] = hotkey

        return uid_to_hotkey

    def _get_mock_emissions_for_tests(self, all_hotkeys_seen: Optional[set] = None) -> Dict[str, tuple[float, float, float, float, float, int]]:
        """
        Get mock emissions data for unit tests (avoids blockchain queries).

        This method returns mock data in the same format as _calculate_emissions_for_all_hotkeys(),
        allowing all downstream business logic to run normally while avoiding network calls.

        Args:
            all_hotkeys_seen: Set to track all hotkeys encountered

        Returns:
            Dictionary mapping hotkey to (alpha_emissions, tao_emissions, usd_emissions,
                                         avg_alpha_to_tao_rate, avg_tao_to_usd_rate, num_blocks)
        """
        # Get hotkeys from perf ledgers to generate mock emissions
        all_perf_ledgers = self._perf_ledger_client.get_perf_ledgers(portfolio_only=False)

        if not all_perf_ledgers:
            return {}

        # Mock constants (realistic but zero to not affect tests)
        MOCK_ALPHA_EMISSIONS = 0.0
        MOCK_ALPHA_TO_TAO_RATE = 1.0
        MOCK_TAO_TO_USD_RATE = 500.0
        MOCK_NUM_BLOCKS = 1

        result = {}
        for hotkey in all_perf_ledgers.keys():
            if all_hotkeys_seen is not None:
                all_hotkeys_seen.add(hotkey)

            # Return tuple: (alpha_emissions, tao_emissions, usd_emissions, avg_alpha_to_tao_rate, avg_tao_to_usd_rate, num_blocks)
            tao_emissions = MOCK_ALPHA_EMISSIONS * MOCK_ALPHA_TO_TAO_RATE
            usd_emissions = tao_emissions * MOCK_TAO_TO_USD_RATE

            result[hotkey] = (
                MOCK_ALPHA_EMISSIONS,
                tao_emissions,
                usd_emissions,
                MOCK_ALPHA_TO_TAO_RATE,
                MOCK_TAO_TO_USD_RATE,
                MOCK_NUM_BLOCKS
            )

        bt.logging.debug(f"Generated mock emissions for {len(result)} hotkeys in test mode")
        return result

    def _calculate_emissions_for_all_hotkeys(
        self,
        start_block: int,
        end_block: int,
        all_hotkeys_seen: Optional[set] = None,
    ) -> Dict[str, tuple[float, float, float, float, float, int]]:
        """
        Calculate emissions for ALL hotkeys between two blocks (OPTIMIZED).

        This method queries each block once and extracts data for all UIDs,
        sharing block hashes and alpha-to-TAO rates across all hotkeys.

        IMPORTANT: Queries UID-to-hotkey mapping at each historical block to
        correctly attribute emissions even if hotkeys changed UIDs over time.

        Args:
            start_block: Starting block (inclusive)
            end_block: Ending block (inclusive)
            all_hotkeys_seen: Set to track all hotkeys encountered

        Returns:
            Dictionary mapping hotkey to (alpha_emissions, tao_emissions, usd_emissions,
                                         avg_alpha_to_tao_rate, avg_tao_to_usd_rate, num_blocks)
            All rate values are guaranteed to be floats (not None).
        """
        # In unit test mode, return mock emissions data to avoid blockchain queries
        # This exercises all downstream business logic (chunking, aggregation, checkpoint creation)
        # while avoiding expensive network calls
        if self.running_unit_tests:
            return self._get_mock_emissions_for_tests(all_hotkeys_seen)
        # Sample blocks at regular intervals
        sample_interval = int(3600 / self.SECONDS_PER_BLOCK)  # ~300 blocks per hour
        sampled_blocks = list(range(start_block, end_block + 1, sample_interval))
        if sampled_blocks[-1] != end_block:
            sampled_blocks.append(end_block)

        # Fail-fast: sampled_blocks must be populated
        if not sampled_blocks:
            raise ValueError(
                f"No sampled blocks found for block range {start_block}-{end_block}"
            )

        # Initialize tracking for emissions
        hotkey_total_alpha = defaultdict(float)
        hotkey_total_tao = defaultdict(float)
        hotkey_total_usd = defaultdict(float)

        # Track sums for averaging (instead of start/end rates)
        sum_alpha_to_tao_rate = 0.0
        sum_tao_to_usd_rate = 0.0
        num_blocks_sampled = 0

        previous_block = start_block
        previous_emissions_by_hotkey = {}
        previous_alpha_to_tao_rate = None
        previous_tao_to_usd_rate = None

        for i, block in enumerate(sampled_blocks):
            # Rate limit before block hash query
            self._rate_limit()

            # Query block hash ONCE for this block
            block_hash = self.subtensor.substrate.get_block_hash(block)
            if not block_hash:
                raise ValueError(f"Block hash query failed for block {block}")

            # Rate limit before block data query
            self._rate_limit()

            # Query block data ONCE for this block (needed for timestamp extraction)
            block_data = self.subtensor.substrate.get_block(block_hash=block_hash)
            if not block_data:
                raise ValueError(f"Block data query failed for block {block}")

            # Extract timestamp from block data
            block_timestamp_ms = self._extract_block_timestamp(block_data, block)

            # Query UID-to-hotkey mapping AT THIS BLOCK (rate limited inside method)
            current_uid_to_hotkey = self._get_uid_to_hotkey_at_block(block_hash)

            # Track new hotkeys
            for hotkey in current_uid_to_hotkey.values():
                if hotkey not in all_hotkeys_seen:
                    all_hotkeys_seen.add(hotkey)

            # Rate limit before Emission query
            self._rate_limit()

            # Query ALL emissions at once (entire vector)
            emission_query = self.subtensor.substrate.query(
                module='SubtensorModule',
                storage_function='Emission',
                params=[self.netuid],
                block_hash=block_hash
            )

            if emission_query is not None:
                emissions_list = emission_query.value if hasattr(emission_query, 'value') else emission_query
            else:
                emissions_list = []

            # Query alpha-to-TAO rate ONCE (shared by all hotkeys) - pass block_hash to avoid redundant query
            alpha_to_tao_rate = self.query_alpha_to_tao_rate(block, block_hash=block_hash)

            # Query TAO-to-USD rate ONCE (shared by all hotkeys) - pass timestamp to avoid redundant query
            tao_to_usd_rate = self.query_tao_to_usd_rate(block, block_timestamp_ms=block_timestamp_ms)

            # Accumulate rates for averaging
            sum_alpha_to_tao_rate += alpha_to_tao_rate
            sum_tao_to_usd_rate += tao_to_usd_rate
            num_blocks_sampled += 1

            # Process emissions for each UID -> hotkey (at this block)
            current_emissions_by_hotkey = defaultdict(float)
            for uid, hotkey in current_uid_to_hotkey.items():
                # Extract emission for this UID from the vector
                if isinstance(emissions_list, (list, tuple)) and uid < len(emissions_list):
                    emission_rao = float(emissions_list[uid])
                    emission_alpha = emission_rao / 1e9  # Convert RAO to alpha
                    emission_per_block = emission_alpha / 360  # Convert per-tempo to per-block
                elif isinstance(emissions_list, dict) and uid in emissions_list:
                    emission_rao = float(emissions_list[uid])
                    emission_alpha = emission_rao / 1e9
                    emission_per_block = emission_alpha / 360
                else:
                    emission_per_block = 0.0

                current_emissions_by_hotkey[hotkey] += emission_per_block

                # Calculate segment emissions if we have previous data
                if hotkey in previous_emissions_by_hotkey:
                    blocks_elapsed = block - previous_block
                    avg_alpha_rate = (previous_emissions_by_hotkey[hotkey] + current_emissions_by_hotkey[hotkey]) / 2
                    segment_emissions_alpha = avg_alpha_rate * blocks_elapsed
                    hotkey_total_alpha[hotkey] += segment_emissions_alpha

                    # Convert to TAO using average conversion rate for this segment
                    avg_alpha_to_tao_rate = (previous_alpha_to_tao_rate + alpha_to_tao_rate) / 2
                    segment_emissions_tao = segment_emissions_alpha * avg_alpha_to_tao_rate
                    hotkey_total_tao[hotkey] += segment_emissions_tao

                    # Convert to USD using average TAO/USD rate for this segment
                    avg_tao_to_usd_rate = (previous_tao_to_usd_rate + tao_to_usd_rate) / 2
                    segment_emissions_usd = segment_emissions_tao * avg_tao_to_usd_rate
                    hotkey_total_usd[hotkey] += segment_emissions_usd

            # Update state for next iteration
            previous_emissions_by_hotkey = current_emissions_by_hotkey
            previous_alpha_to_tao_rate = alpha_to_tao_rate
            previous_tao_to_usd_rate = tao_to_usd_rate
            previous_block = block

        # Calculate average rates (guaranteed to be non-None)
        avg_alpha_to_tao_rate = sum_alpha_to_tao_rate / num_blocks_sampled if num_blocks_sampled > 0 else 0.0
        avg_tao_to_usd_rate = sum_tao_to_usd_rate / num_blocks_sampled if num_blocks_sampled > 0 else 0.0

        # Build result dictionary for ALL hotkeys seen
        results = {}
        for hotkey in all_hotkeys_seen:
            results[hotkey] = (
                hotkey_total_alpha.get(hotkey, 0.0),
                hotkey_total_tao.get(hotkey, 0.0),
                hotkey_total_usd.get(hotkey, 0.0),
                avg_alpha_to_tao_rate,
                avg_tao_to_usd_rate,
                num_blocks_sampled
            )

        return results

    # ============================================================================
    # PERSISTENCE METHODS
    # ============================================================================

    def _get_ledger_path(self) -> str:
        """Get path for emissions ledger file."""
        suffix = "/tests" if self.running_unit_tests else ""
        base_path = ValiConfig.BASE_DIR + f"{suffix}/validation/emissions_ledger.json"
        return base_path + ".gz"

    def save_to_disk(self, create_backup: bool = True):
        """
        Save emissions ledgers to disk with atomic write.

        Validates that all ledgers have the same final checkpoint time before saving
        to ensure ledgers are kept in sync.

        Args:
            create_backup: Whether to create timestamped backup before overwrite

        Raises:
            ValueError: If ledgers have mismatched final checkpoint times
        """
        if not self.emissions_ledgers:
            bt.logging.warning("No ledgers to save")
            return

        # Validate that all ledgers have the same final checkpoint time
        final_checkpoint_times = set()
        for hotkey, ledger in self.emissions_ledgers.items():
            if ledger.checkpoints:
                final_checkpoint_times.add(ledger.checkpoints[-1].chunk_end_ms)

        if len(final_checkpoint_times) > 1:
            # Ledgers are out of sync - this is a critical error
            error_details = []
            for hotkey, ledger in self.emissions_ledgers.items():
                if ledger.checkpoints:
                    last_time = ledger.checkpoints[-1].chunk_end_ms
                    last_time_str = datetime.fromtimestamp(last_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
                    error_details.append(f"  {hotkey[:16]}...{hotkey[-8:]}: {last_time_str} ({last_time})")

            raise ValueError(
                f"Cannot save ledgers: final checkpoint times are not synchronized!\n"
                f"Found {len(final_checkpoint_times)} different final times:\n" +
                "\n".join(error_details[:10]) +  # Show first 10
                (f"\n  ... and {len(error_details) - 10} more" if len(error_details) > 10 else "") +
                "\n\nAll ledgers must be updated to the same time before saving."
            )

        ledger_path = self._get_ledger_path()

        # Build data structure (coldkey now stored in each ledger)
        data = {
            "format_version": "1.0",
            "last_update_ms": int(time.time() * 1000),
            "netuid": self.netuid,
            "archive_endpoint": self.archive_endpoint,
            "ledgers": {}
        }

        for hotkey, ledger in self.emissions_ledgers.items():
            data["ledgers"][hotkey] = ledger.to_dict()  # Includes coldkey automatically

        # Create backup before overwriting
        if create_backup and os.path.exists(ledger_path):
            self._create_backup()

        # Atomic write: temp file -> move
        self._write_compressed(ledger_path, data)

        bt.logging.info(f"Saved {len(self.emissions_ledgers)} emissions ledgers to {ledger_path}")

    def load_from_disk(self) -> int:
        """
        Load existing ledgers from disk.

        Returns:
            Number of ledgers loaded
        """
        ledger_path = self._get_ledger_path()

        if not os.path.exists(ledger_path):
            bt.logging.info("No existing emissions ledger file found")
            return 0

        # Load data
        data = self._read_compressed(ledger_path)

        # Extract metadata
        metadata = {
            "netuid": data.get("netuid"),
            "archive_endpoint": data.get("archive_endpoint"),
            "last_update_ms": data.get("last_update_ms"),
            "format_version": data.get("format_version", "1.0")
        }

        # Reconstruct ledgers
        for hotkey, ledger_dict in data.get("ledgers", {}).items():
            checkpoints = []
            for cp in ledger_dict.get("checkpoints", []):
                checkpoint = EmissionsCheckpoint(
                    chunk_start_ms=cp["chunk_start_ms"],
                    chunk_end_ms=cp["chunk_end_ms"],
                    chunk_emissions=cp["chunk_emissions"],
                    chunk_emissions_tao=cp.get("chunk_emissions_tao", 0.0),
                    chunk_emissions_usd=cp.get("chunk_emissions_usd", 0.0),
                    avg_alpha_to_tao_rate=cp["avg_alpha_to_tao_rate"],
                    avg_tao_to_usd_rate=cp["avg_tao_to_usd_rate"],
                    num_blocks=cp.get("num_blocks", 0),
                    block_start=cp.get("block_start"),
                    block_end=cp.get("block_end"),
                    tao_balance_snapshot=cp.get("tao_balance_snapshot", 0.0),
                    alpha_balance_snapshot=cp.get("alpha_balance_snapshot", 0.0)
                )

                checkpoints.append(checkpoint)

            # Load coldkey from ledger data (empty string if missing - will be lazily queried)
            coldkey = ledger_dict.get("coldkey", "")

            # Note: If coldkey is empty, it will be lazily queried from substrate when first needed
            # by _get_coldkey_for_hotkey() and then persisted on next save (one-time migration)
            if not coldkey:
                bt.logging.debug(f"Coldkey missing for {hotkey[:16]}... in saved data - will be queried lazily when needed")

            self.emissions_ledgers[hotkey] = EmissionsLedger(hotkey=hotkey, coldkey=coldkey, checkpoints=checkpoints)

        bt.logging.info(
            f"Loaded {len(self.emissions_ledgers)} emissions ledgers, "
            f"metadata: {metadata}, "
            f"last update: {TimeUtil.millis_to_formatted_date_str(metadata.get('last_update_ms', 0))}"
        )

        return len(self.emissions_ledgers)

    def _create_backup(self):
        """Create timestamped backup of current ledger file."""
        backup_dir = ValiBkpUtils.get_vali_bkp_dir() + "emissions_ledgers/"
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"{backup_dir}emissions_ledger_{timestamp}.json.gz"

        ledger_path = self._get_ledger_path()

        try:
            shutil.copy2(ledger_path, backup_path)
            bt.logging.info(f"Created backup: {backup_path}")
        except Exception as e:
            bt.logging.warning(f"Failed to create backup: {e}")

    def _write_compressed(self, path: str, data: dict):
        """Write JSON data compressed with gzip (atomic write via temp file)."""
        temp_path = path + ".tmp"
        with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        shutil.move(temp_path, path)

    def _read_compressed(self, path: str) -> dict:
        """Read compressed JSON data."""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    # ============================================================================
    # CHECKPOINT INFO (IMPLICIT FROM LEDGERS)
    # ============================================================================

    def get_checkpoint_info(self) -> dict:
        """
        Extract checkpoint metadata from ledgers (implicit tracking).

        Returns:
            Dictionary with last_computed_chunk_end_ms, last_computed_block, etc.
        """
        if not self.emissions_ledgers:
            return {
                "last_computed_chunk_end_ms": 0,
                "last_computed_block": 0,
                "total_checkpoints": 0,
                "hotkeys_tracked": 0
            }

        last_chunk_end_ms = 0
        last_block = 0
        total_checkpoints = 0

        for ledger in self.emissions_ledgers.values():
            if ledger.checkpoints:
                last_checkpoint = ledger.checkpoints[-1]
                last_chunk_end_ms = max(last_chunk_end_ms, last_checkpoint.chunk_end_ms)
                if last_checkpoint.block_end:
                    last_block = max(last_block, last_checkpoint.block_end)
                total_checkpoints += len(ledger.checkpoints)

        return {
            "last_computed_chunk_end_ms": last_chunk_end_ms,
            "last_computed_block": last_block,
            "total_checkpoints": total_checkpoints,
            "hotkeys_tracked": len(self.emissions_ledgers)
        }

    # ============================================================================
    # DELTA UPDATE METHODS
    # ============================================================================

    def build_emissions_ledgers(self, delta_update: bool = True, lag_time_ms: Optional[int] = None) -> int:
        """
        Build emissions ledgers with control over delta vs full rebuild.

        IMPORTANT: This method is for testing/manual use only. Production code should use
        the automatic daemon which calls build_delta_update() directly.

        Args:
            delta_update: If True, only process new data. If False, clear and rebuild from scratch.
            lag_time_ms: Stay this far behind current time (default: 12 hours)

        Returns:
            Number of chunks built

        Raises:
            RuntimeError: If called in production (running_unit_tests=False)
        """
        if not self.running_unit_tests:
            raise RuntimeError(
                "build_emissions_ledgers() is for testing only. "
                "Production code should not call this method directly. "
                "The emissions ledger daemon handles automatic updates."
            )

        if not delta_update:
            # Clear existing ledgers to force full rebuild
            self.emissions_ledgers.clear()
        return self.build_delta_update(lag_time_ms=lag_time_ms)

    def build_delta_update(self, lag_time_ms: Optional[int] = None) -> int:
        """
        Build emissions ledgers from scratch (full rebuild).

        This method rebuilds ALL emissions ledgers from scratch to ensure they reflect
        any changes in performance ledgers. This is necessary because performance ledgers
        can change, and emissions calculations depend on them.

        This method:
        1. Checks if there are new chunks to compute
        2. Clears existing emissions ledgers
        3. Rebuilds from scratch using the full lookback period

        Args:
            lag_time_ms: Stay this far behind current time (default: 12 hours)

        Returns:
            Number of chunks built
        """
        if lag_time_ms is None:
            # In test mode, use no lag since test data is created "now"
            lag_time_ms = 0 if self.running_unit_tests else self.DEFAULT_LAG_TIME_MS

        if self.rate_limit_per_second < 10:
            bt.logging.warning(f"Emissions ledger network rate limit set to {self.rate_limit_per_second} req/sec - queries will be slow")

        start_time = time.time()

        # Get checkpoint info from existing ledgers to check if update is needed
        checkpoint_info = self.get_checkpoint_info()
        last_computed_chunk_end_ms = checkpoint_info["last_computed_chunk_end_ms"]

        # Calculate new time range
        current_time_ms = int(time.time() * 1000)
        end_time_ms = current_time_ms - lag_time_ms  # Stay behind current time

        # Check if there are new chunks to compute
        if last_computed_chunk_end_ms == 0:
            # First run - perform full build
            bt.logging.info("No existing checkpoint found - performing initial full build")
            return self._build_full(end_time_ms)

        if last_computed_chunk_end_ms >= end_time_ms:
            bt.logging.info(
                f"No new chunks to compute. "
                f"Last computed: {TimeUtil.millis_to_formatted_date_str(last_computed_chunk_end_ms)}, "
                f"Target end: {TimeUtil.millis_to_formatted_date_str(end_time_ms)}"
            )
            return 0

        # Delta update: build from last checkpoint to end_time (NEVER clear existing ledgers!)
        start_time_ms = last_computed_chunk_end_ms

        bt.logging.info(
            f"Delta update from {TimeUtil.millis_to_formatted_date_str(start_time_ms)} "
            f"to {TimeUtil.millis_to_formatted_date_str(end_time_ms)}"
        )

        self.build_all_emissions_ledgers_optimized(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms
        )

        # Calculate chunks built in this delta
        checkpoint_info_after = self.get_checkpoint_info()
        chunks_built = checkpoint_info_after["last_computed_chunk_end_ms"] - last_computed_chunk_end_ms
        chunks_built = chunks_built // ValiConfig.TARGET_CHECKPOINT_DURATION_MS if chunks_built > 0 else 0

        elapsed = time.time() - start_time
        bt.logging.info(
            f"Delta update completed in {elapsed:.2f}s - "
            f"built {chunks_built} new chunks"
        )

        return chunks_built

    def _build_full(self, end_time_ms: int, lookback_days: int = DEFAULT_START_TIME_OFFSET_DAYS) -> int:
        """
        Perform full build (used for initial run).

        Args:
            end_time_ms: End time for build
            lookback_days: How many days to look back

        Returns:
            Number of chunks built
        """
        bt.logging.info(f"Building full emissions ledgers ({lookback_days} day lookback)")

        # Clear existing ledgers for full rebuild
        self.emissions_ledgers.clear()

        # Calculate start_time_ms from lookback_days
        current_time = datetime.now(timezone.utc)
        start_time = current_time - timedelta(days=lookback_days)
        start_time_ms = int(start_time.timestamp() * 1000)

        # Build from scratch (works in-place on self.emissions_ledgers and saves incrementally)
        self.build_all_emissions_ledgers_optimized(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms
        )

        total_chunks = sum(len(ledger.checkpoints) for ledger in self.emissions_ledgers.values())
        bt.logging.info(f"Full build complete - {total_chunks} total chunks")

        return total_chunks

    # ============================================================================
    # QUERY METHODS
    # ============================================================================

    def get_ledger(self, hotkey: str) -> Optional[EmissionsLedger]:
        """Get emissions ledger for a specific hotkey."""
        return self.emissions_ledgers.get(hotkey)

    def get_all_ledgers(self) -> Dict[str, EmissionsLedger]:
        """Get all emissions ledgers."""
        return deepcopy(self.emissions_ledgers)

    def get_earliest_emissions_timestamp(self) -> Optional[int]:
        """
        Get the earliest emissions timestamp across all ledgers (efficient single IPC read).

        Reads the IPC dict once and finds the minimum chunk_start_ms across all ledgers.
        This is more efficient than calling get_ledger() for each hotkey individually.

        Returns:
            Earliest chunk_start_ms across all emissions ledgers, or None if no ledgers exist
        """
        if not self.emissions_ledgers:
            return None

        earliest_ms = None
        # Read IPC dict once - this is the key optimization
        for ledger in self.emissions_ledgers.values():
            if ledger.checkpoints:
                ledger_earliest_ms = ledger.checkpoints[0].chunk_start_ms
                if earliest_ms is None or ledger_earliest_ms < earliest_ms:
                    earliest_ms = ledger_earliest_ms

        return earliest_ms

    def get_cumulative_emissions(self, hotkey: str) -> float:
        """Get cumulative alpha emissions for a hotkey."""
        ledger = self.get_ledger(hotkey)
        return ledger.get_cumulative_emissions() if ledger else 0.0

    def get_cumulative_emissions_tao(self, hotkey: str) -> float:
        """Get cumulative TAO emissions for a hotkey."""
        ledger = self.get_ledger(hotkey)
        return ledger.get_cumulative_emissions_tao() if ledger else 0.0

    def get_cumulative_emissions_usd(self, hotkey: str) -> float:
        """Get cumulative USD emissions for a hotkey."""
        ledger = self.get_ledger(hotkey)
        return ledger.get_cumulative_emissions_usd() if ledger else 0.0

    def print_emissions_summary(self, hotkey: str):
        """Print formatted summary of emissions for a hotkey."""
        ledger = self.get_ledger(hotkey)
        if ledger:
            ledger.print_summary()
        else:
            print(f"\nNo emissions data found for {hotkey}")

    def plot_emissions(self, hotkey: str, save_path: Optional[str] = None):
        """Plot emissions data for a hotkey using matplotlib."""
        ledger = self.get_ledger(hotkey)
        if ledger:
            ledger.plot_emissions(save_path)
        else:
            bt.logging.warning(f"No emissions data found for {hotkey}, skipping plot")

    # ============================================================================
    # DAEMON MODE
    # ============================================================================

    def run_forever(
            self,
            check_interval_seconds: Optional[int] = None,
            lag_time_ms: Optional[int] = None
    ):
        """
        Run as daemon - continuously update emissions ledgers forever.

        Checks for new chunks at regular intervals and performs delta updates.
        Handles graceful shutdown on SIGINT/SIGTERM.

        Features:
        - Exponential backoff on failures (1h -> 2h -> 4h -> 8h -> 16h -> 32h -> max 48h)
        - Slack notifications for failures
        - Automatic retry after backoff
        - Graceful shutdown

        Args:
            check_interval_seconds: How often to check for new chunks (default: 3600s = 1 hour)
            lag_time_ms: How far behind current time to stay (default: 12 hours)
        """
        if check_interval_seconds is None:
            check_interval_seconds = self.DEFAULT_CHECK_INTERVAL_SECONDS

        if lag_time_ms is None:
            lag_time_ms = self.DEFAULT_LAG_TIME_MS

        self.running = True

        # Exponential backoff parameters
        consecutive_failures = 0
        initial_backoff_seconds = 3600  # Start with 1 hour
        max_backoff_seconds = 172800  # Max 48 hours
        backoff_multiplier = 2

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            bt.logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        bt.logging.info("=" * 80)
        bt.logging.info("Emissions Ledger Manager - Daemon Mode")
        bt.logging.info("=" * 80)
        bt.logging.info(f"NetUID: {self.netuid}")
        bt.logging.info(f"Archive Endpoint: {self.archive_endpoint}")
        bt.logging.info(f"Rate Limit: {self.rate_limit_per_second} req/sec")
        bt.logging.info(f"Check Interval: {check_interval_seconds}s ({check_interval_seconds / 3600:.1f} hours)")
        bt.logging.info(f"Lag Time: {lag_time_ms / 1000 / 3600:.1f} hours behind current time")
        bt.logging.info(f"Slack Notifications: {'Enabled' if self.slack_notifier.webhook_url else 'Disabled'}")
        bt.logging.info("=" * 80)

        # Main loop (do-while pattern - executes immediately on first iteration)
        while self.running:
            try:
                # Perform delta update (happens immediately on first iteration)
                bt.logging.info("Checking for new chunks...")
                chunks_added = self.build_delta_update(lag_time_ms=lag_time_ms)

                if chunks_added > 0:
                    bt.logging.info(f"Added {chunks_added} new chunks")
                else:
                    bt.logging.info("No new chunks available")

                # Success - reset failure counter
                if consecutive_failures > 0:
                    bt.logging.info(f"Recovered after {consecutive_failures} failure(s)")
                    # Send recovery alert with VM/git/hotkey context
                    self.slack_notifier.send_ledger_recovery_alert("Emissions Ledger", consecutive_failures)

                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1

                # Calculate backoff for logging
                backoff_seconds = min(
                    initial_backoff_seconds * (backoff_multiplier ** (consecutive_failures - 1)),
                    max_backoff_seconds
                )

                bt.logging.error(
                    f"Error in daemon loop (failure #{consecutive_failures}): {e}",
                    exc_info=True
                )

                # Send Slack alert with VM/git/hotkey context
                self.slack_notifier.send_ledger_failure_alert(
                    "Emissions Ledger",
                    consecutive_failures,
                    e,
                    backoff_seconds
                )

            # Calculate sleep time and sleep (moved to end of loop)
            if self.running:
                if consecutive_failures > 0:
                    # Exponential backoff
                    backoff_seconds = min(
                        initial_backoff_seconds * (backoff_multiplier ** (consecutive_failures - 1)),
                        max_backoff_seconds
                    )
                    next_check_time = time.time() + backoff_seconds
                    next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S UTC')
                    bt.logging.warning(
                        f"Retrying after {consecutive_failures} failure(s). "
                        f"Backoff: {backoff_seconds}s. Next attempt at: {next_check_str}"
                    )
                else:
                    # Normal interval
                    next_check_time = time.time() + check_interval_seconds
                    next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S UTC')
                    bt.logging.info(f"Next check at: {next_check_str}")

                # Sleep in small intervals to allow graceful shutdown
                while self.running and time.time() < next_check_time:
                    time.sleep(10)

        bt.logging.info("Emissions Ledger Manager daemon stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build emissions ledger for Bittensor hotkeys")
    parser.add_argument("--hotkey", type=str, help="Hotkey to display/focus on (optional, displays one plot)", default=None)
    parser.add_argument("--netuid", type=int, default=8, help="Subnet UID (default: 8)")
    parser.add_argument("--network", type=str, default="finney", help="Network name (default: finney)")
    parser.add_argument("--archive-endpoint", type=str, action="append", dest="archive_endpoints",
                       help="Archive node endpoint (can be specified multiple times). Example: wss://archive.chain.opentensor.ai:443")
    parser.add_argument("--start-time-offset-days", type=int, default=EmissionsLedgerManager.DEFAULT_START_TIME_OFFSET_DAYS,
                       help="Number of days to look back from now for emissions tracking")
    parser.add_argument("--show-plot", action="store_true", help="Display plot for specified hotkey (requires --hotkey)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    bt.logging.enable_info()
    if args.verbose:
        bt.logging.enable_debug()

    # Initialize emissions ledger manager
    # EmissionsLedgerManager creates its own PerfLedgerClient internally (forward compatibility)
    bt.logging.info("Initializing emissions ledger manager...")
    emissions_ledger_manager = EmissionsLedgerManager(
        start_daemon=False
    )

    # Calculate end_time_ms with required lag (12 hours behind current time for data finality)
    current_time_ms = int(time.time() * 1000)
    end_time_ms = current_time_ms - EmissionsLedgerManager.DEFAULT_LAG_TIME_MS

    # Determine start_time_ms based on existing data or default lookback
    checkpoint_info = emissions_ledger_manager.get_checkpoint_info()
    last_computed_chunk_end_ms = checkpoint_info["last_computed_chunk_end_ms"]

    if last_computed_chunk_end_ms > 0:
        # Resume from existing data
        start_time_ms = last_computed_chunk_end_ms
        bt.logging.info(f"Resuming from existing data at {TimeUtil.millis_to_formatted_date_str(last_computed_chunk_end_ms)}")
    else:
        # No existing data - use lookback period
        current_time = datetime.now(timezone.utc)
        start_time = current_time - timedelta(days=args.start_time_offset_days)
        start_time_ms = int(start_time.timestamp() * 1000)
        bt.logging.info(f"No existing data - starting from {args.start_time_offset_days} days ago")

    # ALWAYS build ALL ledgers using optimized method
    bt.logging.info("Building emissions ledgers for ALL hotkeys in subnet (optimized mode)")
    bt.logging.info(f"Using {EmissionsLedgerManager.DEFAULT_LAG_TIME_MS / 1000 / 3600:.1f} hour lag time for data finality")
    emissions_ledger_manager.build_all_emissions_ledgers_optimized(
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms
    )

    if len(emissions_ledger_manager.emissions_ledgers) == 0:
        bt.logging.error("No emissions ledgers were built")
        exit(1)

    # Create emissions_ledger_plots directory
    # Get project root (2 levels up from vali_objects/vali_dataclasses/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    plots_dir = os.path.join(project_root, "emissions_ledger_plots")

    os.makedirs(plots_dir, exist_ok=True)
    bt.logging.info(f"Saving plots to: {plots_dir}")

    # Save ALL plots to emissions_ledger_plots/{hotkey}.png
    for hotkey in emissions_ledger_manager.emissions_ledgers.keys():
        plot_path = os.path.join(plots_dir, f"{hotkey}.png")
        try:
            emissions_ledger_manager.plot_emissions(hotkey, save_path=plot_path)
            bt.logging.info(f"Saved plot for {hotkey[:16]}...{hotkey[-8:]}")
        except Exception as e:
            bt.logging.error(f"Error saving plot for {hotkey}: {e}")

    bt.logging.info(f"All plots saved to {plots_dir}")

    # Print summary for specified hotkey or all hotkeys
    if args.hotkey:
        if args.hotkey in emissions_ledger_manager.emissions_ledgers:
            emissions_ledger_manager.print_emissions_summary(args.hotkey)

            # Optionally display the plot for this hotkey
            if args.show_plot:
                bt.logging.info(f"Displaying plot for {args.hotkey}")
                emissions_ledger_manager.plot_emissions(args.hotkey, save_path=None)
        else:
            bt.logging.error(f"Hotkey {args.hotkey} not found in built ledgers")
    else:
        # Print summaries for all hotkeys
        bt.logging.info("Printing summaries for all hotkeys")
        for hotkey in emissions_ledger_manager.emissions_ledgers.keys():
            emissions_ledger_manager.print_emissions_summary(hotkey)
