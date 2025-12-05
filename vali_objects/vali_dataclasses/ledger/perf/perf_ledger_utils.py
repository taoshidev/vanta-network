"""
Utility functions for performance ledger calculations.

This module contains extracted utility functions to improve testability
and separation of concerns in the performance ledger system.
"""

import math
from typing import Tuple
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint


class PerfLedgerMath:
    """
    Mathematical utility functions for performance ledger calculations.
    
    This class contains static methods for various calculations used in
    performance ledger processing, extracted for better testability.
    """
    
    @staticmethod
    def compute_return_delta(current_value: float, previous_value: float, use_log: bool = True) -> float:
        """
        Compute the delta between two return values.
        
        Args:
            current_value: Current portfolio value/return
            previous_value: Previous portfolio value/return  
            use_log: Whether to use logarithmic calculation (True) or simple percentage (False)
            
        Returns:
            The calculated delta
            
        Raises:
            ValueError: If values are invalid (negative, zero, etc.)
            ZeroDivisionError: If previous_value is zero for percentage calculation
        """
        if current_value <= 0 or previous_value <= 0:
            raise ValueError(f"Values must be positive: current={current_value}, previous={previous_value}")
        
        if use_log:
            return math.log(current_value / previous_value)
        else:
            return (current_value - previous_value) / previous_value
    
    @staticmethod
    def compute_simple_delta(current_value: float, previous_value: float) -> float:
        """
        Compute simple arithmetic delta between two values.
        
        Args:
            current_value: Current value
            previous_value: Previous value
            
        Returns:
            Simple difference (current - previous)
        """
        return current_value - previous_value
    
    @staticmethod
    def update_maximum_drawdown(current_value: float, max_portfolio_value: float, current_mdd: float) -> Tuple[float, float]:
        """
        Update maximum drawdown calculation.
        
        Args:
            current_value: Current portfolio value
            max_portfolio_value: Maximum portfolio value seen so far
            current_mdd: Current maximum drawdown
            
        Returns:
            Tuple of (new_max_portfolio_value, new_mdd)
        """
        new_max_value = max(max_portfolio_value, current_value)
        
        if new_max_value > 0:
            current_drawdown = current_value / new_max_value
            new_mdd = min(current_mdd, current_drawdown)
        else:
            new_mdd = current_mdd
            
        return new_max_value, new_mdd
    
    @staticmethod
    def calculate_fee_delta(current_fee: float, previous_fee: float) -> float:
        """
        Calculate fee delta with proper validation.
        
        Args:
            current_fee: Current fee value
            previous_fee: Previous fee value
            
        Returns:
            Fee delta (should typically be negative, representing loss)
        """
        if current_fee < 0 or previous_fee < 0:
            raise ValueError(f"Fee values must be non-negative: current={current_fee}, previous={previous_fee}")
        
        if current_fee > 1 or previous_fee > 1:
            raise ValueError(f"Fee values must be <= 1: current={current_fee}, previous={previous_fee}")
        
        return current_fee - previous_fee
    
    @staticmethod
    def validate_checkpoint_data(checkpoint: PerfCheckpoint) -> bool:
        """
        Validate checkpoint data for mathematical consistency.
        
        Args:
            checkpoint: The checkpoint to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If critical validation fails
        """
        if checkpoint.last_update_ms < 0:
            raise ValueError("Checkpoint timestamp cannot be negative")
        
        if checkpoint.prev_portfolio_ret <= 0:
            raise ValueError("Portfolio return must be positive")
        
        if checkpoint.gain < 0 or checkpoint.loss > 0:
            raise ValueError("Gains must be non-negative, losses must be non-positive")
        
        if checkpoint.gain > 0 and checkpoint.loss < 0:
            raise ValueError("Cannot have both gains and losses in same checkpoint")
        
        if checkpoint.prev_portfolio_spread_fee < 0 or checkpoint.prev_portfolio_spread_fee > 1:
            raise ValueError("Spread fee must be between 0 and 1")
        
        if checkpoint.prev_portfolio_carry_fee < 0 or checkpoint.prev_portfolio_carry_fee > 1:
            raise ValueError("Carry fee must be between 0 and 1")
        
        return True
    
    @staticmethod
    def calculate_time_weighted_return(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Calculate time-weighted return from a series of checkpoints.
        
        Args:
            checkpoints: List of performance checkpoints
            
        Returns:
            Time-weighted return value
        """
        if not checkpoints:
            return 1.0
        
        total_return = 1.0
        
        for i, cp in enumerate(checkpoints):
            if i == 0:
                # First checkpoint, use as baseline
                continue
            
            prev_cp = checkpoints[i-1]
            period_return = cp.prev_portfolio_ret / prev_cp.prev_portfolio_ret
            
            # Weight by time duration if available
            time_weight = max(1, cp.open_ms) / (1000 * 60 * 60)  # Convert to hours, minimum 1
            
            # Apply time weighting (simple approach)
            weighted_return = math.pow(period_return, min(time_weight / 24, 1.0))  # Cap at daily
            total_return *= weighted_return
        
        return total_return
    
    @staticmethod
    def calculate_volatility(checkpoints: list[PerfCheckpoint]) -> float:
        """
        Calculate return volatility from checkpoints.
        
        Args:
            checkpoints: List of performance checkpoints
            
        Returns:
            Volatility measure
        """
        if len(checkpoints) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(checkpoints)):
            if checkpoints[i-1].prev_portfolio_ret > 0:
                period_return = math.log(checkpoints[i].prev_portfolio_ret / checkpoints[i-1].prev_portfolio_ret)
                returns.append(period_return)
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        
        return math.sqrt(variance)


class PerfLedgerValidator:
    """
    Validation utilities for performance ledger data integrity.
    """
    
    @staticmethod
    def validate_position_consistency(positions: list, expected_miner: str) -> bool:
        """
        Validate that all positions belong to the expected miner and are consistent.
        
        Args:
            positions: List of positions to validate
            expected_miner: Expected miner hotkey
            
        Returns:
            True if all positions are consistent
        """
        for position in positions:
            if position.miner_hotkey != expected_miner:
                raise ValueError(f"Position miner {position.miner_hotkey} does not match expected {expected_miner}")
            
            if not position.orders:
                raise ValueError(f"Position {position.position_uuid} has no orders")
            
            # Validate order sequence
            for i in range(1, len(position.orders)):
                if position.orders[i].processed_ms < position.orders[i-1].processed_ms:
                    raise ValueError(f"Orders in position {position.position_uuid} are not in chronological order")
        
        return True
    
    @staticmethod
    def validate_ledger_integrity(ledger_bundle: dict) -> bool:
        """
        Validate the integrity of a ledger bundle.
        
        Args:
            ledger_bundle: Dictionary of trade pair ID to PerfLedger
            
        Returns:
            True if bundle is valid
        """
        if not isinstance(ledger_bundle, dict):
            raise ValueError("Ledger bundle must be a dictionary")
        
        # Check for portfolio ledger
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import TP_ID_PORTFOLIO
        if TP_ID_PORTFOLIO not in ledger_bundle:
            raise ValueError("Ledger bundle must contain portfolio ledger")
        
        portfolio_ledger = ledger_bundle[TP_ID_PORTFOLIO]
        
        # Validate all ledgers have consistent timing
        for tp_id, ledger in ledger_bundle.items():
            if tp_id == TP_ID_PORTFOLIO:
                continue
            
            if ledger.last_update_ms != portfolio_ledger.last_update_ms:
                raise ValueError(f"Ledger {tp_id} last update time inconsistent with portfolio")
        
        return True