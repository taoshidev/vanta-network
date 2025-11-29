"""
Debt Ledger - Unified view combining emissions, penalties, and performance data

This module provides a unified DebtLedger structure that combines:
- Emissions data (alpha/TAO/USD) from EmissionsLedger
- Penalty multipliers from PenaltyLedger
- Performance metrics (PnL, fees, drawdown) from PerfLedger

The DebtLedger provides a complete financial picture for each miner, making it
easy for the UI to display comprehensive miner statistics.

Architecture:
- DebtCheckpoint: Data for a single point in time
- DebtLedger: Complete debt history for a SINGLE hotkey
- DebtLedgerManager: Manages ledgers for multiple hotkeys

Usage:
    # Create a debt ledger for a miner
    ledger = DebtLedger(hotkey="5...")

    # Add a checkpoint combining all data sources
    checkpoint = DebtCheckpoint(
        timestamp_ms=1234567890000,
        # Emissions data
        chunk_emissions_alpha=10.5,
        chunk_emissions_tao=0.05,
        chunk_emissions_usd=25.0,
        # Performance data
        portfolio_return=1.15,
        realized_pnl=800.0,
        unrealized_pnl=100.0,
        # ... other fields
    )
    ledger.add_checkpoint(checkpoint)

Standalone Usage:
Use runnable/local_debt_ledger.py for standalone execution with hard-coded configuration.
Edit the configuration variables at the top of that file to customize behavior.

"""
import multiprocessing
import signal
import bittensor as bt
import time
import gzip
import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime, timezone

from vanta_api.slack_notifier import SlackNotifier
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedgerManager, EmissionsLedger
from vali_objects.vali_dataclasses.penalty_ledger import PenaltyLedgerManager, PenaltyLedger
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
from vali_objects.utils.miner_bucket_enum import MinerBucket


@dataclass
class DebtCheckpoint:
    """
    Unified checkpoint combining emissions, penalties, and performance data.

    All data is aligned to a single timestamp representing a snapshot in time
    of the miner's complete financial state.

    Attributes:
        # Timing
        timestamp_ms: Checkpoint timestamp in milliseconds

        # Emissions Data (from EmissionsLedger) - chunk data only, no cumulative
        chunk_emissions_alpha: Alpha tokens earned in this chunk
        chunk_emissions_tao: TAO value earned in this chunk
        chunk_emissions_usd: USD value earned in this chunk
        avg_alpha_to_tao_rate: Average alpha-to-TAO conversion rate for this chunk
        avg_tao_to_usd_rate: Average TAO/USD price for this chunk
        tao_balance_snapshot: TAO balance at checkpoint end (for validation)
        alpha_balance_snapshot: ALPHA balance at checkpoint end (for validation)

        # Performance Data (from PerfLedger)
        # Note: Sourced from PerfCheckpoint attributes - some have different names:
        #   portfolio_return <- gain, max_drawdown <- mdd, max_portfolio_value <- mpv
        portfolio_return: Current portfolio return multiplier (1.0 = break-even)
        realized_pnl: Net realized PnL during this checkpoint period (NOT cumulative across checkpoints)
        unrealized_pnl: Net unrealized PnL during this checkpoint period (NOT cumulative across checkpoints)
        spread_fee_loss: Spread fee losses during this checkpoint period (NOT cumulative)
        carry_fee_loss: Carry fee losses during this checkpoint period (NOT cumulative)
        max_drawdown: Maximum drawdown (worst loss from peak, cumulative)
        max_portfolio_value: Maximum portfolio value achieved (cumulative)
        open_ms: Time with open positions during this checkpoint (milliseconds)
        accum_ms: Time duration of this checkpoint (milliseconds)
        n_updates: Number of performance updates during this checkpoint

        # Penalty Data (from PenaltyLedger)
        drawdown_penalty: Drawdown threshold penalty multiplier
        risk_profile_penalty: Risk profile penalty multiplier
        min_collateral_penalty: Minimum collateral penalty multiplier
        risk_adjusted_performance_penalty: Risk-adjusted performance penalty multiplier
        total_penalty: Combined penalty multiplier (product of all penalties)
        challenge_period_status: Challenge period status (MAINCOMP/CHALLENGE/PROBATION/PLAGIARISM/UNKNOWN)

        # Derived/Computed Fields
        total_fees: Total fees paid (spread + carry)
        return_after_fees: Portfolio return after all fees
        weighted_score: Final score after applying all penalties
    """
    # Timing
    timestamp_ms: int

    # Emissions Data (chunk only, cumulative calculated by summing)
    chunk_emissions_alpha: float = 0.0
    chunk_emissions_tao: float = 0.0
    chunk_emissions_usd: float = 0.0
    avg_alpha_to_tao_rate: float = 0.0
    avg_tao_to_usd_rate: float = 0.0
    tao_balance_snapshot: float = 0.0
    alpha_balance_snapshot: float = 0.0

    # Performance Data
    portfolio_return: float = 1.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    spread_fee_loss: float = 0.0
    carry_fee_loss: float = 0.0
    max_drawdown: float = 1.0
    max_portfolio_value: float = 0.0
    open_ms: int = 0
    accum_ms: int = 0
    n_updates: int = 0

    # Penalty Data
    drawdown_penalty: float = 1.0
    risk_profile_penalty: float = 1.0
    min_collateral_penalty: float = 1.0
    risk_adjusted_performance_penalty: float = 1.0
    total_penalty: float = 1.0
    challenge_period_status: str = None

    def __post_init__(self):
        """Calculate derived fields after initialization"""
        # Set default for challenge_period_status if not provided
        if self.challenge_period_status is None:
            self.challenge_period_status = MinerBucket.UNKNOWN.value
        # Calculate derived financial fields
        self.total_fees = self.spread_fee_loss + self.carry_fee_loss
        self.return_after_fees = self.portfolio_return
        self.weighted_score = self.portfolio_return * self.total_penalty

    def __eq__(self, other):
        if not isinstance(other, DebtCheckpoint):
            return False
        return self.timestamp_ms == other.timestamp_ms

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            # Timing
            'timestamp_ms': self.timestamp_ms,
            'timestamp_utc': datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc).isoformat(),

            # Emissions (chunk only)
            'emissions': {
                'chunk_alpha': self.chunk_emissions_alpha,
                'chunk_tao': self.chunk_emissions_tao,
                'chunk_usd': self.chunk_emissions_usd,
                'avg_alpha_to_tao_rate': self.avg_alpha_to_tao_rate,
                'avg_tao_to_usd_rate': self.avg_tao_to_usd_rate,
                'tao_balance_snapshot': self.tao_balance_snapshot,
                'alpha_balance_snapshot': self.alpha_balance_snapshot,
            },

            # Performance
            'performance': {
                'portfolio_return': self.portfolio_return,
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl,
                'spread_fee_loss': self.spread_fee_loss,
                'carry_fee_loss': self.carry_fee_loss,
                'total_fees': self.total_fees,
                'max_drawdown': self.max_drawdown,
                'max_portfolio_value': self.max_portfolio_value,
                'open_ms': self.open_ms,
                'accum_ms': self.accum_ms,
                'n_updates': self.n_updates,
            },

            # Penalties
            'penalties': {
                'drawdown': self.drawdown_penalty,
                'risk_profile': self.risk_profile_penalty,
                'min_collateral': self.min_collateral_penalty,
                'risk_adjusted_performance': self.risk_adjusted_performance_penalty,
                'cumulative': self.total_penalty,
                'challenge_period_status': self.challenge_period_status,
            },

            # Derived
            'derived': {
                'return_after_fees': self.return_after_fees,
                'weighted_score': self.weighted_score,
            }
        }


class DebtLedger:
    """
    Complete debt/earnings ledger for a SINGLE hotkey.

    Combines emissions, penalties, and performance data into a unified view.
    Stores checkpoints in chronological order.
    """

    def __init__(self, hotkey: str, checkpoints: Optional[List[DebtCheckpoint]] = None):
        """
        Initialize debt ledger for a single hotkey.

        Args:
            hotkey: SS58 address of the hotkey
            checkpoints: Optional list of debt checkpoints
        """
        self.hotkey = hotkey
        self.checkpoints: List[DebtCheckpoint] = checkpoints or []

    def add_checkpoint(self, checkpoint: DebtCheckpoint, target_cp_duration_ms: int):
        """
        Add a checkpoint to the ledger.

        Validates that the new checkpoint is properly aligned with the target checkpoint
        duration and the previous checkpoint (no gaps, no overlaps) - matching emissions ledger strictness.

        Args:
            checkpoint: The checkpoint to add
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Raises:
            AssertionError: If checkpoint validation fails
        """
        # Validate checkpoint timestamp aligns with target duration
        assert checkpoint.timestamp_ms % target_cp_duration_ms == 0, (
            f"Checkpoint timestamp {checkpoint.timestamp_ms} must align with target_cp_duration_ms "
            f"{target_cp_duration_ms} for {self.hotkey}"
        )

        # If there are existing checkpoints, ensure perfect spacing (contiguity)
        if self.checkpoints:
            prev_checkpoint = self.checkpoints[-1]

            # First check it's after previous checkpoint
            assert checkpoint.timestamp_ms > prev_checkpoint.timestamp_ms, (
                f"Checkpoint timestamp must be after previous checkpoint for {self.hotkey}: "
                f"new checkpoint at {checkpoint.timestamp_ms}, "
                f"but previous checkpoint at {prev_checkpoint.timestamp_ms}"
            )

            # Then check exact spacing - checkpoints must be contiguous (no gaps, no overlaps)
            expected_timestamp_ms = prev_checkpoint.timestamp_ms + target_cp_duration_ms
            assert checkpoint.timestamp_ms == expected_timestamp_ms, (
                f"Checkpoint spacing must be exactly {target_cp_duration_ms}ms for {self.hotkey}: "
                f"new checkpoint at {checkpoint.timestamp_ms}, "
                f"previous at {prev_checkpoint.timestamp_ms}, "
                f"expected {expected_timestamp_ms}. "
                f"Expected perfect alignment (no gaps, no overlaps)."
            )

        self.checkpoints.append(checkpoint)

    def get_latest_checkpoint(self) -> Optional[DebtCheckpoint]:
        """Get the most recent checkpoint"""
        return self.checkpoints[-1] if self.checkpoints else None

    def get_checkpoint_at_time(self, timestamp_ms: int, target_cp_duration_ms: int) -> Optional[DebtCheckpoint]:
        """
        Get the checkpoint at a specific timestamp (efficient O(1) lookup).

        Uses index calculation instead of scanning since checkpoints are evenly-spaced
        and contiguous (enforced by strict add_checkpoint validation).

        Args:
            timestamp_ms: Exact timestamp to query
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Returns:
            Checkpoint at the exact timestamp, or None if not found

        Raises:
            ValueError: If checkpoint exists at calculated index but timestamp doesn't match (data corruption)
        """
        if not self.checkpoints:
            return None

        # Calculate expected index based on first checkpoint and duration
        first_checkpoint_ms = self.checkpoints[0].timestamp_ms

        # Check if timestamp is before first checkpoint
        if timestamp_ms < first_checkpoint_ms:
            return None

        # Calculate index (checkpoints are evenly spaced by target_cp_duration_ms)
        time_diff = timestamp_ms - first_checkpoint_ms
        if time_diff % target_cp_duration_ms != 0:
            # Timestamp doesn't align with checkpoint boundaries
            return None

        index = time_diff // target_cp_duration_ms

        # Check if index is within bounds
        if index >= len(self.checkpoints):
            return None

        # Validate the checkpoint at this index has the expected timestamp
        checkpoint = self.checkpoints[index]
        if checkpoint.timestamp_ms != timestamp_ms:
            raise ValueError(
                f"Data corruption detected for {self.hotkey}: "
                f"checkpoint at index {index} has timestamp {checkpoint.timestamp_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(checkpoint.timestamp_ms)}), "
                f"but expected {timestamp_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(timestamp_ms)}). "
                f"Checkpoints are not properly contiguous."
            )

        return checkpoint

    def get_cumulative_emissions_alpha(self) -> float:
        """Get total cumulative alpha emissions by summing chunk emissions"""
        return sum(cp.chunk_emissions_alpha for cp in self.checkpoints)

    def get_cumulative_emissions_tao(self) -> float:
        """Get total cumulative TAO emissions by summing chunk emissions"""
        return sum(cp.chunk_emissions_tao for cp in self.checkpoints)

    def get_cumulative_emissions_usd(self) -> float:
        """Get total cumulative USD emissions by summing chunk emissions"""
        return sum(cp.chunk_emissions_usd for cp in self.checkpoints)

    def get_current_portfolio_return(self) -> float:
        """Get current portfolio return"""
        latest = self.get_latest_checkpoint()
        return latest.portfolio_return if latest else 1.0

    def get_current_weighted_score(self) -> float:
        """Get current weighted score (return * penalties)"""
        latest = self.get_latest_checkpoint()
        return latest.weighted_score if latest else 1.0

    def to_dict(self) -> dict:
        """
        Convert ledger to dictionary for serialization.

        Returns:
            Dictionary with hotkey and all checkpoints
        """
        latest = self.get_latest_checkpoint()

        return {
            'hotkey': self.hotkey,
            'total_checkpoints': len(self.checkpoints),

            # Summary statistics (cumulative emissions calculated by summing)
            'summary': {
                'cumulative_emissions_alpha': self.get_cumulative_emissions_alpha(),
                'cumulative_emissions_tao': self.get_cumulative_emissions_tao(),
                'cumulative_emissions_usd': self.get_cumulative_emissions_usd(),
                'portfolio_return': self.get_current_portfolio_return(),
                'weighted_score': self.get_current_weighted_score(),
                'total_fees': latest.total_fees if latest else 0.0,
            } if latest else {},

            # All checkpoints
            'checkpoints': [cp.to_dict() for cp in self.checkpoints]
        }

    def print_summary(self):
        """Print a formatted summary of the debt ledger"""
        if not self.checkpoints:
            print(f"\nNo debt ledger data found for {self.hotkey}")
            return

        latest = self.get_latest_checkpoint()

        print(f"\n{'='*80}")
        print(f"Debt Ledger Summary for {self.hotkey}")
        print(f"{'='*80}")
        print(f"Total Checkpoints: {len(self.checkpoints)}")
        print(f"\n--- Emissions ---")
        print(f"Total Alpha: {self.get_cumulative_emissions_alpha():.6f}")
        print(f"Total TAO: {self.get_cumulative_emissions_tao():.6f}")
        print(f"Total USD: ${self.get_cumulative_emissions_usd():,.2f}")
        print(f"\n--- Performance ---")
        print(f"Portfolio Return: {latest.portfolio_return:.4f} ({(latest.portfolio_return - 1) * 100:+.2f}%)")
        print(f"Total Fees: ${latest.total_fees:,.2f}")
        print(f"Max Drawdown: {latest.max_drawdown:.4f}")
        print(f"\n--- Penalties ---")
        print(f"Drawdown Penalty: {latest.drawdown_penalty:.4f}")
        print(f"Risk Profile Penalty: {latest.risk_profile_penalty:.4f}")
        print(f"Min Collateral Penalty: {latest.min_collateral_penalty:.4f}")
        print(f"Risk Adjusted Performance Penalty: {latest.risk_adjusted_performance_penalty:.4f}")
        print(f"Cumulative Penalty: {latest.total_penalty:.4f}")
        print(f"\n--- Final Score ---")
        print(f"Weighted Score: {latest.weighted_score:.4f}")
        print(f"{'='*80}\n")

    @staticmethod
    def from_dict(data: dict) -> 'DebtLedger':
        """
        Reconstruct ledger from dictionary.

        Args:
            data: Dictionary containing ledger data

        Returns:
            Reconstructed DebtLedger
        """
        checkpoints = []
        for cp_dict in data.get('checkpoints', []):
            # Extract nested data from the structured format
            if 'emissions' in cp_dict:
                # Structured format from to_dict()
                emissions = cp_dict['emissions']
                performance = cp_dict['performance']
                penalties = cp_dict['penalties']

                checkpoint = DebtCheckpoint(
                    timestamp_ms=cp_dict['timestamp_ms'],
                    # Emissions
                    chunk_emissions_alpha=emissions.get('chunk_alpha', 0.0),
                    chunk_emissions_tao=emissions.get('chunk_tao', 0.0),
                    chunk_emissions_usd=emissions.get('chunk_usd', 0.0),
                    avg_alpha_to_tao_rate=emissions.get('avg_alpha_to_tao_rate', 0.0),
                    avg_tao_to_usd_rate=emissions.get('avg_tao_to_usd_rate', 0.0),
                    tao_balance_snapshot=emissions.get('tao_balance_snapshot', 0.0),
                    alpha_balance_snapshot=emissions.get('alpha_balance_snapshot', 0.0),
                    # Performance
                    portfolio_return=performance.get('portfolio_return', 1.0),
                    realized_pnl=performance.get('realized_pnl', 0.0),
                    unrealized_pnl=performance.get('unrealized_pnl', 0.0),
                    spread_fee_loss=performance.get('spread_fee_loss', 0.0),
                    carry_fee_loss=performance.get('carry_fee_loss', 0.0),
                    max_drawdown=performance.get('max_drawdown', 1.0),
                    max_portfolio_value=performance.get('max_portfolio_value', 0.0),
                    open_ms=performance.get('open_ms', 0),
                    accum_ms=performance.get('accum_ms', 0),
                    n_updates=performance.get('n_updates', 0),
                    # Penalties
                    drawdown_penalty=penalties.get('drawdown', 1.0),
                    risk_profile_penalty=penalties.get('risk_profile', 1.0),
                    min_collateral_penalty=penalties.get('min_collateral', 1.0),
                    risk_adjusted_performance_penalty=penalties.get('risk_adjusted_performance', 1.0),
                    total_penalty=penalties.get('cumulative', 1.0),
                    challenge_period_status=penalties.get('challenge_period_status', MinerBucket.UNKNOWN.value),
                )
            else:
                # Flat format (backward compatibility or alternative format)
                checkpoint = DebtCheckpoint(
                    timestamp_ms=cp_dict['timestamp_ms'],
                    chunk_emissions_alpha=cp_dict.get('chunk_emissions_alpha', 0.0),
                    chunk_emissions_tao=cp_dict.get('chunk_emissions_tao', 0.0),
                    chunk_emissions_usd=cp_dict.get('chunk_emissions_usd', 0.0),
                    avg_alpha_to_tao_rate=cp_dict.get('avg_alpha_to_tao_rate', 0.0),
                    avg_tao_to_usd_rate=cp_dict.get('avg_tao_to_usd_rate', 0.0),
                    tao_balance_snapshot=cp_dict.get('tao_balance_snapshot', 0.0),
                    alpha_balance_snapshot=cp_dict.get('alpha_balance_snapshot', 0.0),
                    portfolio_return=cp_dict.get('portfolio_return', 1.0),
                    realized_pnl=cp_dict.get('realized_pnl', 0.0),
                    unrealized_pnl=cp_dict.get('unrealized_pnl', 0.0),
                    spread_fee_loss=cp_dict.get('spread_fee_loss', 0.0),
                    carry_fee_loss=cp_dict.get('carry_fee_loss', 0.0),
                    max_drawdown=cp_dict.get('max_drawdown', 1.0),
                    max_portfolio_value=cp_dict.get('max_portfolio_value', 0.0),
                    open_ms=cp_dict.get('open_ms', 0),
                    accum_ms=cp_dict.get('accum_ms', 0),
                    n_updates=cp_dict.get('n_updates', 0),
                    drawdown_penalty=cp_dict.get('drawdown_penalty', 1.0),
                    risk_profile_penalty=cp_dict.get('risk_profile_penalty', 1.0),
                    min_collateral_penalty=cp_dict.get('min_collateral_penalty', 1.0),
                    risk_adjusted_performance_penalty=cp_dict.get('risk_adjusted_performance_penalty', 1.0),
                    total_penalty=cp_dict.get('total_penalty', 1.0),
                    challenge_period_status=cp_dict.get('challenge_period_status', MinerBucket.UNKNOWN.value),
                )

            checkpoints.append(checkpoint)

        return DebtLedger(hotkey=data['hotkey'], checkpoints=checkpoints)


class DebtLedgerManager:
    """
    Manages debt ledgers for multiple hotkeys.

    Responsibilities:
    - Combine data from EmissionsLedgerManager, PerfLedgerManager, and PenaltyLedger
    - Build unified DebtCheckpoints by merging data from all three sources
    - Handle serialization/deserialization
    - Provide query methods for UI consumption
    """

    DEFAULT_CHECK_INTERVAL_SECONDS = 3600 * 12  # 12 hours

    def __init__(self, perf_ledger_manager, position_manager, contract_manager, asset_selection_manager,
                 challengeperiod_manager=None, slack_webhook_url=None, start_daemon=True, ipc_manager=None, running_unit_tests=False, validator_hotkey=None):
        self.perf_ledger_manager = perf_ledger_manager

        # IMPORTANT: PenaltyLedgerManager now runs in its own daemon (run_daemon=True)
        # This ensures penalty ledgers refresh every 12 hours UTC-aligned with accurate checkpoint data
        self.penalty_ledger_manager = PenaltyLedgerManager(
            position_manager=position_manager,
            perf_ledger_manager=perf_ledger_manager,
            contract_manager=contract_manager,
            asset_selection_manager=asset_selection_manager,
            challengeperiod_manager=challengeperiod_manager,
            slack_webhook_url=slack_webhook_url,
            run_daemon=True,  # Run penalty ledger in its own daemon
            running_unit_tests=running_unit_tests,
            validator_hotkey=validator_hotkey,
            ipc_manager=ipc_manager
        )

        self.emissions_ledger_manager = EmissionsLedgerManager(slack_webhook_url=slack_webhook_url, start_daemon=False,
                                                               ipc_manager=ipc_manager, perf_ledger_manager=perf_ledger_manager, running_unit_tests=running_unit_tests, validator_hotkey=validator_hotkey)

        self.debt_ledgers: dict[str, DebtLedger] = ipc_manager.dict() if ipc_manager else {}
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)
        self.running_unit_tests = running_unit_tests
        self.running = False

        self.load_data_from_disk()

        if start_daemon:
            self._start_daemon_process()

    # ============================================================================
    # PERSISTENCE METHODS
    # ============================================================================

    def _get_ledger_path(self) -> str:
        """Get path for debt ledger file."""
        suffix = "/tests" if self.running_unit_tests else ""
        base_path = ValiConfig.BASE_DIR + f"{suffix}/validation/debt_ledger.json"
        return base_path + ".gz"

    def save_to_disk(self, create_backup: bool = True):
        """
        Save debt ledgers to disk with atomic write.

        Args:
            create_backup: Whether to create timestamped backup before overwrite
        """
        if not self.debt_ledgers:
            bt.logging.warning("No debt ledgers to save")
            return

        ledger_path = self._get_ledger_path()

        # Build data structure
        data = {
            "format_version": "1.0",
            "last_update_ms": int(time.time() * 1000),
            "ledgers": {}
        }

        for hotkey, ledger in self.debt_ledgers.items():
            data["ledgers"][hotkey] = ledger.to_dict()

        # Atomic write: temp file -> move
        self._write_compressed(ledger_path, data)

        bt.logging.info(f"Saved {len(self.debt_ledgers)} debt ledgers to {ledger_path}")

    def load_data_from_disk(self) -> int:
        """
        Load existing ledgers from disk.

        Returns:
            Number of ledgers loaded
        """
        ledger_path = self._get_ledger_path()

        if not os.path.exists(ledger_path):
            bt.logging.info("No existing debt ledger file found")
            return 0

        # Load data
        data = self._read_compressed(ledger_path)

        # Extract metadata
        metadata = {
            "last_update_ms": data.get("last_update_ms"),
            "format_version": data.get("format_version", "1.0")
        }

        # Reconstruct ledgers
        for hotkey, ledger_dict in data.get("ledgers", {}).items():
            ledger = DebtLedger.from_dict(ledger_dict)
            self.debt_ledgers[hotkey] = ledger

        bt.logging.info(
            f"Loaded {len(self.debt_ledgers)} debt ledgers, "
            f"metadata: {metadata}, "
            f"last update: {TimeUtil.millis_to_formatted_date_str(metadata.get('last_update_ms', 0))}"
        )

        return len(self.debt_ledgers)

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
    # HELPER METHODS
    # ============================================================================

    def get_last_processed_ms(self, miner_hotkey: str) -> int:
        """
        Get the last processed timestamp for a miner's debt ledger.

        This is a helper method to modularize delta update logic.

        Args:
            miner_hotkey: The miner's hotkey

        Returns:
            Last processed timestamp in milliseconds, or 0 if no checkpoints exist
        """
        if miner_hotkey not in self.debt_ledgers:
            return 0

        debt_ledger = self.debt_ledgers[miner_hotkey]
        if not debt_ledger.checkpoints:
            return 0

        last_checkpoint = debt_ledger.get_latest_checkpoint()
        return last_checkpoint.timestamp_ms

    # ============================================================================
    # DAEMON MODE
    # ============================================================================

    def _start_daemon_process(self):
        """Start the daemon process for continuous updates."""
        daemon_process = multiprocessing.Process(
            target=self.run_daemon_forever,
            args=(),
            kwargs={'verbose': False}
        )
        daemon_process.daemon = True
        daemon_process.start()
        bt.logging.info("Started DebtLedgerManager daemon process")

    def get_ledger(self, hotkey: str) -> Optional[DebtLedger]:
        """Get emissions ledger for a specific hotkey."""
        return self.debt_ledgers.get(hotkey)

    def run_daemon_forever(self, check_interval_seconds: Optional[int] = None, verbose: bool = False):
        """
        Run as daemon - continuously update debt ledgers forever.

        Checks for new performance/emissions/penalty data at regular intervals and performs full rebuilds.
        Handles graceful shutdown on SIGINT/SIGTERM.

        Features:
        - Full rebuilds (debt ledgers are derived from emissions + penalties + performance)
        - Periodic refresh (default: every 12 hours)
        - Graceful shutdown
        - Automatic retry on failures

        Args:
            check_interval_seconds: How often to check for new checkpoints (default: 12 hours)
            verbose: Enable detailed logging
        """
        if check_interval_seconds is None:
            check_interval_seconds = self.DEFAULT_CHECK_INTERVAL_SECONDS

        self.running = True

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            bt.logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        bt.logging.info("=" * 80)
        bt.logging.info("Debt Ledger Manager - Daemon Mode")
        bt.logging.info("=" * 80)
        bt.logging.info(f"Check Interval: {check_interval_seconds}s ({check_interval_seconds / 3600:.1f} hours)")
        bt.logging.info(f"Full Rebuild Mode: Enabled (debt ledgers derived from emissions + penalties + perf)")
        bt.logging.info(f"Slack Notifications: {'Enabled' if self.slack_notifier.webhook_url else 'Disabled'}")
        bt.logging.info("=" * 80)

        # Track consecutive failures for exponential backoff
        consecutive_failures = 0
        initial_backoff_seconds = 300  # Start with 5 minutes
        max_backoff_seconds = 3600  # Max 1 hour
        backoff_multiplier = 2

        time.sleep(120) # Initial delay to stagger large ipc reads

        # Main loop
        while self.running:
            try:
                bt.logging.info("="*80)
                bt.logging.info("Starting coordinated ledger update cycle...")
                bt.logging.info("="*80)
                start_time = time.time()

                # IMPORTANT: Update sub-ledgers FIRST in correct order before building debt ledgers
                # This ensures debt ledgers have the latest data from all sources

                # NOTE: Penalty ledgers are now updated in their own dedicated daemon (UTC-aligned 12-hour intervals)
                # This ensures penalty data has accurate challenge period status per checkpoint

                # Step 1: Update emissions ledgers
                bt.logging.info("Step 1/2: Updating emissions ledgers...")
                emissions_start = time.time()
                self.emissions_ledger_manager.build_delta_update()
                bt.logging.info(f"Emissions ledgers updated in {time.time() - emissions_start:.2f}s")

                # Step 2: Build debt ledgers (combines data from penalty + emissions + perf)
                # Penalty ledgers are updated separately by their dedicated daemon
                # IMPORTANT: Debt ledgers ALWAYS do full rebuilds (never delta updates)
                # since they're derived from three sources that can change retroactively
                bt.logging.info("Step 2/2: Building debt ledgers (full rebuild)...")
                debt_start = time.time()
                self.build_debt_ledgers(verbose=verbose, delta_update=False)
                bt.logging.info(f"Debt ledgers built in {time.time() - debt_start:.2f}s")

                elapsed = time.time() - start_time
                bt.logging.info("="*80)
                bt.logging.info(f"Complete update cycle finished in {elapsed:.2f}s")
                bt.logging.info("="*80)

                # Success - reset failure counter
                if consecutive_failures > 0:
                    bt.logging.info(f"Recovered after {consecutive_failures} failure(s)")
                    # Send recovery alert with VM/git/hotkey context
                    self.slack_notifier.send_ledger_recovery_alert("Debt Ledger", consecutive_failures)

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
                    "Debt Ledger",
                    consecutive_failures,
                    e,
                    backoff_seconds
                )

            # Calculate sleep time and sleep
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

        bt.logging.info("Debt Ledger Manager daemon stopped")

    def build_debt_ledgers(self, verbose: bool = False, delta_update: bool = True):
        """
        Build or update debt ledgers for all hotkeys using timestamp-based iteration.

        Iterates over TIMESTAMPS (perf ledger checkpoints), processing ALL hotkeys at each timestamp.
        Saves to disk after each timestamp for crash recovery. Matches emissions ledger pattern.

        In order to create a debt checkpoint, we must have:
        - Corresponding emissions checkpoint for that timestamp
        - Corresponding penalty checkpoint for that timestamp
        - Corresponding perf checkpoint for that timestamp

        IMPORTANT: Builds candidate ledgers first, then atomically swaps them in to prevent race conditions
        where ledgers momentarily disappear during the build process.

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints since last update. If False, rebuild from scratch.
        """
        # Build into candidate dict to prevent race conditions (don't clear existing ledgers yet)
        if delta_update:
            # Delta update: start with copies of existing ledgers
            candidate_ledgers = {}
            for hotkey, existing_ledger in self.debt_ledgers.items():
                # Create a new DebtLedger with copies of existing checkpoints
                candidate_ledgers[hotkey] = DebtLedger(hotkey, checkpoints=list(existing_ledger.checkpoints))
        else:
            # Full rebuild: start from scratch
            candidate_ledgers = {}
            bt.logging.info("Full rebuild mode: building new debt ledgers from scratch")

        # Read all perf ledgers from perf ledger manager
        all_perf_ledgers: Dict[str, Dict[str, any]] = self.perf_ledger_manager.get_perf_ledgers(
            portfolio_only=False
        )

        if not all_perf_ledgers:
            bt.logging.warning("No performance ledgers found")
            return

        # Pick a reference portfolio ledger (use the one with the most checkpoints for maximum coverage)
        reference_portfolio_ledger = None
        reference_hotkey = None
        max_checkpoints = 0

        for hotkey, ledger_dict in all_perf_ledgers.items():
            portfolio_ledger = ledger_dict.get(TP_ID_PORTFOLIO)
            if portfolio_ledger and portfolio_ledger.cps:
                if len(portfolio_ledger.cps) > max_checkpoints:
                    max_checkpoints = len(portfolio_ledger.cps)
                    reference_portfolio_ledger = portfolio_ledger
                    reference_hotkey = hotkey

        if not reference_portfolio_ledger:
            bt.logging.warning("No valid portfolio ledgers found with checkpoints")
            return

        bt.logging.info(
            f"Using portfolio ledger from {reference_hotkey[:16]}...{reference_hotkey[-8:]} "
            f"as reference ({len(reference_portfolio_ledger.cps)} checkpoints, "
            f"target_cp_duration_ms: {reference_portfolio_ledger.target_cp_duration_ms}ms)"
        )

        target_cp_duration_ms = reference_portfolio_ledger.target_cp_duration_ms

        # Determine which checkpoints to process based on delta update mode
        # Find the ledger with the MOST checkpoints (longest history) to use as reference
        # This prevents truncating history when new miners register with few checkpoints
        last_processed_ms = 0
        if delta_update and candidate_ledgers:
            reference_ledger = None
            max_checkpoint_count = 0
            max_last_processed_ms = 0

            # Find ledger with most checkpoints
            for ledger in candidate_ledgers.values():
                if ledger.checkpoints:
                    checkpoint_count = len(ledger.checkpoints)
                    ledger_last_ms = ledger.checkpoints[-1].timestamp_ms

                    if checkpoint_count > max_checkpoint_count:
                        max_checkpoint_count = checkpoint_count
                        reference_ledger = ledger
                        last_processed_ms = ledger_last_ms

                    # Track maximum timestamp for sanity check
                    if ledger_last_ms > max_last_processed_ms:
                        max_last_processed_ms = ledger_last_ms

            if last_processed_ms > 0:
                # Sanity check: reference ledger (most checkpoints) should have the maximum timestamp
                # This validates that the longest-running miner is up-to-date
                assert last_processed_ms == max_last_processed_ms, (
                    f"Reference ledger (most checkpoints: {max_checkpoint_count}) has timestamp "
                    f"{TimeUtil.millis_to_formatted_date_str(last_processed_ms)}, but max timestamp across "
                    f"all ledgers is {TimeUtil.millis_to_formatted_date_str(max_last_processed_ms)}. "
                    f"This indicates the reference ledger is behind, which would cause history truncation."
                )

                bt.logging.info(
                    f"Delta update mode: resuming from {TimeUtil.millis_to_formatted_date_str(last_processed_ms)} "
                    f"(reference ledger with {max_checkpoint_count} checkpoints)"
                )

        # Filter checkpoints to process
        perf_checkpoints_to_process = []
        for checkpoint in reference_portfolio_ledger.cps:
            # Skip active checkpoints (incomplete)
            if checkpoint.accum_ms != target_cp_duration_ms:
                continue

            checkpoint_ms = checkpoint.last_update_ms

            # Skip checkpoints we've already processed in delta update mode
            if delta_update and checkpoint_ms <= last_processed_ms:
                continue

            perf_checkpoints_to_process.append(checkpoint)

        if not perf_checkpoints_to_process:
            bt.logging.info("No new checkpoints to process")
            return

        bt.logging.info(
            f"Processing {len(perf_checkpoints_to_process)} checkpoints "
            f"(from {TimeUtil.millis_to_formatted_date_str(perf_checkpoints_to_process[0].last_update_ms)} "
            f"to {TimeUtil.millis_to_formatted_date_str(perf_checkpoints_to_process[-1].last_update_ms)})"
        )

        # Track all hotkeys we need to process (from perf ledgers)
        all_hotkeys_to_track = set(all_perf_ledgers.keys())

        # Optimization: Find earliest emissions timestamp across all hotkeys to skip early checkpoints
        earliest_emissions_ms = self.emissions_ledger_manager.get_earliest_emissions_timestamp()

        if earliest_emissions_ms:
            bt.logging.info(
                f"Earliest emissions data starts at {TimeUtil.millis_to_formatted_date_str(earliest_emissions_ms)}"
            )

        # Iterate over TIMESTAMPS processing ALL hotkeys at each timestamp
        checkpoint_count = 0
        for perf_checkpoint in perf_checkpoints_to_process:
            checkpoint_count += 1
            checkpoint_start_time = time.time()

            # Skip this entire timestamp if it's before the earliest emissions data
            if earliest_emissions_ms and perf_checkpoint.last_update_ms < earliest_emissions_ms:
                if verbose:
                    bt.logging.info(
                        f"Skipping checkpoint {checkpoint_count} at {TimeUtil.millis_to_formatted_date_str(perf_checkpoint.last_update_ms)} "
                        f"(before earliest emissions data)"
                    )
                continue

            hotkeys_processed_at_checkpoint = 0
            hotkeys_missing_data = []

            # Process ALL hotkeys at this timestamp
            for hotkey in all_hotkeys_to_track:
                # Get ledgers for this hotkey
                ledger_dict = all_perf_ledgers.get(hotkey)
                if not ledger_dict:
                    continue

                portfolio_ledger = ledger_dict.get(TP_ID_PORTFOLIO)
                if not portfolio_ledger or not portfolio_ledger.cps:
                    continue

                # Get this hotkey's perf checkpoint at the current timestamp (efficient O(1) lookup)
                hotkey_perf_checkpoint = portfolio_ledger.get_checkpoint_at_time(
                    perf_checkpoint.last_update_ms, target_cp_duration_ms
                )
                if not hotkey_perf_checkpoint:
                    continue  # This hotkey doesn't have a perf checkpoint at this timestamp

                # Get corresponding penalty checkpoint (efficient O(1) lookup)
                penalty_ledger = self.penalty_ledger_manager.get_penalty_ledger(hotkey)
                penalty_checkpoint = None
                if penalty_ledger:
                    penalty_checkpoint = penalty_ledger.get_checkpoint_at_time(perf_checkpoint.last_update_ms, target_cp_duration_ms)

                # Get corresponding emissions checkpoint (efficient O(1) lookup)
                emissions_ledger = self.emissions_ledger_manager.get_ledger(hotkey)
                emissions_checkpoint = None
                if emissions_ledger:
                    emissions_checkpoint = emissions_ledger.get_checkpoint_at_time(perf_checkpoint.last_update_ms, target_cp_duration_ms)

                # Skip if we don't have both penalty and emissions data
                if not penalty_checkpoint or not emissions_checkpoint:
                    hotkeys_missing_data.append(hotkey)
                    continue

                # Validate timestamps match
                if hotkey_perf_checkpoint.last_update_ms != perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Perf checkpoint timestamp mismatch for {hotkey}: "
                            f"expected {perf_checkpoint.last_update_ms}, got {hotkey_perf_checkpoint.last_update_ms}"
                        )
                    continue

                if penalty_checkpoint.last_processed_ms != perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Penalty checkpoint timestamp mismatch for {hotkey}: "
                            f"expected {perf_checkpoint.last_update_ms}, got {penalty_checkpoint.last_processed_ms}"
                        )
                    continue

                if emissions_checkpoint.chunk_end_ms != perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Emissions checkpoint end time mismatch for {hotkey}: "
                            f"expected {perf_checkpoint.last_update_ms}, got {emissions_checkpoint.chunk_end_ms}"
                        )
                    continue

                # Get or create debt ledger for this hotkey (from candidate ledgers)
                if hotkey in candidate_ledgers:
                    debt_ledger = candidate_ledgers[hotkey]
                else:
                    debt_ledger = DebtLedger(hotkey)

                # Skip if this hotkey already has a checkpoint at this timestamp (delta update safety check)
                if delta_update and debt_ledger.checkpoints:
                    last_checkpoint_ms = debt_ledger.checkpoints[-1].timestamp_ms
                    if perf_checkpoint.last_update_ms <= last_checkpoint_ms:
                        if verbose:
                            bt.logging.info(
                                f"Skipping checkpoint for {hotkey} at {perf_checkpoint.last_update_ms} "
                                f"(already processed, last checkpoint: {last_checkpoint_ms})"
                            )
                        continue

                # Create unified debt checkpoint combining all three sources
                debt_checkpoint = DebtCheckpoint(
                    timestamp_ms=hotkey_perf_checkpoint.last_update_ms,
                    # Emissions data (chunk only - cumulative calculated by summing)
                    chunk_emissions_alpha=emissions_checkpoint.chunk_emissions,
                    chunk_emissions_tao=emissions_checkpoint.chunk_emissions_tao,
                    chunk_emissions_usd=emissions_checkpoint.chunk_emissions_usd,
                    avg_alpha_to_tao_rate=emissions_checkpoint.avg_alpha_to_tao_rate,
                    avg_tao_to_usd_rate=emissions_checkpoint.avg_tao_to_usd_rate,
                    tao_balance_snapshot=emissions_checkpoint.tao_balance_snapshot,
                    alpha_balance_snapshot=emissions_checkpoint.alpha_balance_snapshot,
                    # Performance data - access attributes directly from PerfCheckpoint
                    portfolio_return=hotkey_perf_checkpoint.gain,  # Current portfolio multiplier
                    realized_pnl=hotkey_perf_checkpoint.realized_pnl,  # Net realized PnL during this checkpoint period
                    unrealized_pnl=hotkey_perf_checkpoint.unrealized_pnl,  # Net unrealized PnL during this checkpoint period
                    spread_fee_loss=hotkey_perf_checkpoint.spread_fee_loss,  # Spread fees during this checkpoint
                    carry_fee_loss=hotkey_perf_checkpoint.carry_fee_loss,  # Carry fees during this checkpoint
                    max_drawdown=hotkey_perf_checkpoint.mdd,  # Max drawdown
                    max_portfolio_value=hotkey_perf_checkpoint.mpv,  # Max portfolio value achieved
                    open_ms=hotkey_perf_checkpoint.open_ms,
                    accum_ms=hotkey_perf_checkpoint.accum_ms,
                    n_updates=hotkey_perf_checkpoint.n_updates,
                    # Penalty data
                    drawdown_penalty=penalty_checkpoint.drawdown_penalty,
                    risk_profile_penalty=penalty_checkpoint.risk_profile_penalty,
                    min_collateral_penalty=penalty_checkpoint.min_collateral_penalty,
                    risk_adjusted_performance_penalty=penalty_checkpoint.risk_adjusted_performance_penalty,
                    total_penalty=penalty_checkpoint.total_penalty,
                    challenge_period_status=penalty_checkpoint.challenge_period_status,
                )

                # Add checkpoint to candidate ledger (validates strict contiguity)
                debt_ledger.add_checkpoint(debt_checkpoint, target_cp_duration_ms)
                candidate_ledgers[hotkey] = debt_ledger  # Update candidate ledgers
                hotkeys_processed_at_checkpoint += 1

            # Log progress for this checkpoint
            checkpoint_elapsed = time.time() - checkpoint_start_time
            checkpoint_dt = datetime.fromtimestamp(perf_checkpoint.last_update_ms / 1000, tz=timezone.utc)
            bt.logging.info(
                f"Checkpoint {checkpoint_count}/{len(perf_checkpoints_to_process)} "
                f"({checkpoint_dt.strftime('%Y-%m-%d %H:%M UTC')}): "
                f"{hotkeys_processed_at_checkpoint} hotkeys processed, "
                f"{len(hotkeys_missing_data)} missing data, "
                f"{checkpoint_elapsed:.2f}s"
            )

        # Build completed successfully - atomically swap candidate ledgers into production
        # This prevents race conditions where ledgers momentarily disappear during build
        bt.logging.info(
            f"Build completed successfully: {checkpoint_count} checkpoints for {len(candidate_ledgers)} hotkeys. "
            f"Atomically updating debt ledgers..."
        )

        # IMPORTANT: For IPC-managed dicts, we need to update each key individually to trigger IPC updates
        # To avoid race condition where dict is momentarily empty, we:
        # 1. Delete obsolete keys first (keys in old but not in new)
        # 2. Then add/update new keys
        # This way the dict is never empty - it always has at least the keys being kept
        if hasattr(self.debt_ledgers, '_getvalue'):
            # IPC managed dict - atomic update without clear()
            old_hotkeys = set(self.debt_ledgers.keys())
            new_hotkeys = set(candidate_ledgers.keys())

            # Delete obsolete hotkeys first
            hotkeys_to_delete = old_hotkeys - new_hotkeys
            for hotkey in hotkeys_to_delete:
                del self.debt_ledgers[hotkey]

            # Then add/update all new hotkeys
            for hotkey, ledger in candidate_ledgers.items():
                self.debt_ledgers[hotkey] = ledger
        else:
            # Regular dict - direct assignment
            self.debt_ledgers = candidate_ledgers

        # Save to disk after atomic swap
        bt.logging.info(f"Saving {len(self.debt_ledgers)} debt ledgers to disk...")
        self.save_to_disk(create_backup=False)

        # Final summary
        bt.logging.info(
            f"Debt ledgers updated: {checkpoint_count} checkpoints processed, "
            f"{len(self.debt_ledgers)} hotkeys tracked "
            f"(target_cp_duration_ms: {target_cp_duration_ms}ms)"
        )
