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
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timezone
from time_util.time_util import TimeUtil
from vali_objects.enums.miner_bucket_enum import MinerBucket


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


