# developer: assistant
# Copyright © 2024 Taoshi Inc

"""
Enhanced mock utilities for comprehensive elimination testing.
Provides robust mocks that closely mirror production behavior.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from unittest.mock import MagicMock
import numpy as np

from shared_objects.mock_metagraph import MockMetagraph as BaseMockMetagraph
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint, TP_ID_PORTFOLIO
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.position import Position
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.position_manager import PositionManager
from tests.shared_objects.mock_classes import (
    MockPositionManager as BaseMockPositionManager,
    MockChallengePeriodManager as BaseMockChallengePeriodManager
)
from vali_objects.scoring.scoring import Scoring


class EnhancedMockMetagraph(BaseMockMetagraph):
    """Enhanced mock metagraph with full attribute support"""

    def __init__(self, hotkeys, neurons=None):
        super().__init__(hotkeys, neurons)

        # Initialize block attributes first
        self.block = 10000  # Default block number
        self.uid_to_block = {i: 1000 for i in range(len(hotkeys))}

        # Initialize all required attributes
        self.n = len(hotkeys)
        self.uids = list(range(len(hotkeys)))
        self.stakes = [100.0] * len(hotkeys)  # Default stake
        self.trust = [1.0] * len(hotkeys)
        self.consensus = [1.0] * len(hotkeys)
        self.incentive = [1.0] * len(hotkeys)
        self.dividends = [0.0] * len(hotkeys)
        self.active = [1] * len(hotkeys)
        self.last_update = [self.block] * len(hotkeys)
        self.validator_permit = [True] * len(hotkeys)
        self.weights = [[0.0] * len(hotkeys) for _ in range(len(hotkeys))]
        self.bonds = [[0.0] * len(hotkeys) for _ in range(len(hotkeys))]

        # Registration tracking
        self.block_at_registration = [1000] * len(hotkeys)
        self.uid_to_hotkey = {i: hk for i, hk in enumerate(hotkeys)}
        self.hotkey_to_uid = {hk: i for i, hk in enumerate(hotkeys)}

        # Attributes for DebtBasedScoring (production code requirements)
        # Emission: TAO per tempo (360 blocks), one value per UID
        # Assume 0.1 TAO per tempo per miner as reasonable default
        self.emission = [0.1] * len(hotkeys)

        # Substrate reserves for ALPHA/TAO conversion
        # Need to mimic multiprocessing.Value objects with .value accessor
        from unittest.mock import MagicMock

        # Reserve values: 1M TAO reserve, 10M ALPHA reserve
        # This gives alpha_to_tao_rate = 1M / 10M = 0.1 TAO per ALPHA
        tao_reserve_mock = MagicMock()
        tao_reserve_mock.value = 1_000_000 * 1e9  # 1M TAO in RAO
        self.tao_reserve_rao = tao_reserve_mock

        alpha_reserve_mock = MagicMock()
        alpha_reserve_mock.value = 10_000_000 * 1e9  # 10M ALPHA in RAO
        self.alpha_reserve_rao = alpha_reserve_mock

        # TAO to USD price: $500 per TAO as reasonable test value
        self.tao_to_usd_rate = 500.0
        
    def sync(self, block=None, lite=True):
        """Mock sync method"""
        if block:
            self.block = block
        return self
        
    def get_uid_for_hotkey(self, hotkey: str) -> Optional[int]:
        """Get UID for a given hotkey"""
        return self.hotkey_to_uid.get(hotkey)
    
    def remove_hotkey(self, hotkey: str):
        """Remove a hotkey from the metagraph (simulate deregistration)"""
        if hotkey in self.hotkeys:
            idx = self.hotkeys.index(hotkey)
            self.hotkeys.remove(hotkey)
            self.n = len(self.hotkeys)
            
            # Update all lists
            if idx < len(self.uids):
                self.uids.pop(idx)
                self.stakes.pop(idx)
                self.trust.pop(idx)
                self.consensus.pop(idx)
                self.incentive.pop(idx)
                self.dividends.pop(idx)
                self.active.pop(idx)
                self.last_update.pop(idx)
                self.validator_permit.pop(idx)
                self.block_at_registration.pop(idx)
            
            # Update mappings
            if hotkey in self.hotkey_to_uid:
                uid = self.hotkey_to_uid[hotkey]
                del self.hotkey_to_uid[hotkey]
                if uid in self.uid_to_hotkey:
                    del self.uid_to_hotkey[uid]
                if uid in self.uid_to_block:
                    del self.uid_to_block[uid]
            
            # Rebuild uid mappings
            self.uid_to_hotkey = {i: hk for i, hk in enumerate(self.hotkeys)}
            self.hotkey_to_uid = {hk: i for i, hk in enumerate(self.hotkeys)}

    def add_hotkey(self, hotkey: str):
        """Add a hotkey to the metagraph (simulate re-registration)"""
        if hotkey not in self.hotkeys:
            self.hotkeys.append(hotkey)
            self.n = len(self.hotkeys)

            # Add to all lists with default values
            new_uid = len(self.uids)
            self.uids.append(new_uid)
            self.stakes.append(100.0)
            self.trust.append(1.0)
            self.consensus.append(1.0)
            self.incentive.append(1.0)
            self.dividends.append(0.0)
            self.active.append(1)
            self.last_update.append(self.block)
            self.validator_permit.append(True)
            self.block_at_registration.append(self.block)

            # Update mappings
            self.uid_to_hotkey[new_uid] = hotkey
            self.hotkey_to_uid[hotkey] = new_uid
            self.uid_to_block[new_uid] = self.block


class EnhancedMockChallengePeriodManager(BaseMockChallengePeriodManager):
    """Enhanced mock challenge period manager with full bucket support"""

    def __init__(self, metagraph, position_manager, perf_ledger_manager, contract_manager, plagiarism_manager, running_unit_tests=True):
        super().__init__(metagraph, position_manager, contract_manager, plagiarism_manager)
        self.perf_ledger_manager = perf_ledger_manager
        self.elimination_manager = position_manager.elimination_manager if position_manager else None

        # Initialize bucket storage
        self.active_miners = {}  # hotkey -> (bucket, timestamp)
        self.eliminations_with_reasons = {}
        self.miner_plagiarism_scores = {}

        # Performance thresholds
        self.elimination_threshold = 0.25  # Bottom 25% get eliminated

    def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> List[str]:
        """Get all hotkeys in a specific bucket, excluding eliminated miners"""
        # Get eliminated hotkeys if elimination_manager is available
        eliminated_hotkeys = set()
        if self.elimination_manager:
            eliminated_hotkeys = set(self.elimination_manager.get_eliminated_hotkeys())

        # Return hotkeys in the bucket that are not eliminated
        return [hk for hk, (b, _, _, _) in self.active_miners.items()
                if b == bucket and hk not in eliminated_hotkeys]

    def remove_eliminated(self):
        """Remove eliminated miners from active_miners"""
        if not self.elimination_manager:
            return

        eliminated_hotkeys = set(self.elimination_manager.get_eliminated_hotkeys())

        # Remove eliminated miners from active_miners
        miners_to_remove = [hk for hk in self.active_miners.keys() if hk in eliminated_hotkeys]
        for hotkey in miners_to_remove:
            del self.active_miners[hotkey]

    def set_miner_bucket(self, hotkey: str, bucket: MinerBucket, timestamp_ms: int = None):
        """Set a miner's bucket"""
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
        self.active_miners[hotkey] = (bucket, timestamp_ms, None, None)
        
    def refresh(self, position_locks):
        """Mock refresh that processes challenge period logic"""
        # Process any pending eliminations
        pass
    
    def _refresh_plagiarism_scores_in_memory_and_disk(self):
        """Mock refresh of plagiarism scores"""
        # In production, this would update plagiarism scores
        # For testing, we just pass
        pass


class EnhancedMockPerfLedgerManager:
    """Enhanced mock perf ledger manager that respects eliminations"""
    
    def __init__(self, metagraph, ipc_manager=None, running_unit_tests=True, perf_ledger_hks_to_invalidate=None):
        from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
        self.base = PerfLedgerManager(metagraph, ipc_manager, running_unit_tests, perf_ledger_hks_to_invalidate or {})
        # Delegate all attributes to base
        self.__dict__.update(self.base.__dict__)
        self.elimination_manager = None  # Set after initialization
        
    def __getattr__(self, name):
        # Delegate to base for any missing attributes
        return getattr(self.base, name)
    
    def __setattr__(self, name, value):
        # Special handling for certain attributes
        if name in ['base', 'elimination_manager']:
            self.__dict__[name] = value
        elif hasattr(self, 'base') and hasattr(self.base, name):
            setattr(self.base, name, value)
        else:
            self.__dict__[name] = value
        
    def filtered_ledger_for_scoring(self, portfolio_only=False, hotkeys=None):
        """Override to exclude eliminated miners"""
        # Get base filtered ledger
        filtered_ledger = self.base.filtered_ledger_for_scoring(portfolio_only, hotkeys)
        
        # Additional filtering for eliminated miners
        if self.elimination_manager:
            eliminations = self.elimination_manager.get_eliminations_from_memory()
            eliminated_hotkeys = {e['hotkey'] for e in eliminations}
            
            # Remove eliminated miners from the ledger
            filtered_ledger = {
                hk: ledger for hk, ledger in filtered_ledger.items() 
                if hk not in eliminated_hotkeys
            }
            
        return filtered_ledger
    
    def save_perf_ledgers(self, ledgers):
        """Save performance ledgers"""
        return self.base.save_perf_ledgers(ledgers)
    
    def clear_perf_ledgers_from_disk(self):
        """Clear performance ledgers from disk"""
        return self.base.clear_perf_ledgers_from_disk()
    
    def update(self, t_ms=None):
        """Update ledgers"""
        # Override to avoid issues with empty positions/orders in tests
        if hasattr(self, 'position_manager') and self.position_manager:
            # Ensure we have valid positions for sorting
            hotkey_to_positions, _ = self.position_manager.get_all_miner_positions()
            
            # Filter out hotkeys with no positions or no orders
            valid_hotkeys = []
            for hotkey, positions in hotkey_to_positions.items():
                if positions and any(pos.orders for pos in positions):
                    valid_hotkeys.append(hotkey)
            
            # Only update if we have valid data
            if valid_hotkeys:
                return self.base.update(t_ms)
        
        # Default: skip update if no valid data
        return


class EnhancedMockPositionManager(BaseMockPositionManager):
    """Enhanced mock position manager with full elimination support"""
    
    def __init__(self, metagraph, perf_ledger_manager, elimination_manager, live_price_fetcher=None):
        super().__init__(metagraph, perf_ledger_manager, elimination_manager, live_price_fetcher)
        self.challengeperiod_manager = None  # Set after initialization
        
        # Track closed positions separately for testing
        self.closed_positions_by_hotkey = defaultdict(list)
        
    def save_miner_position(self, position: Position, delete_open_position_if_exists=True):
        """Override to track closed positions"""
        super().save_miner_position(position, delete_open_position_if_exists)
        
        if position.is_closed_position:
            self.closed_positions_by_hotkey[position.miner_hotkey].append(position)
        
    def filtered_positions_for_scoring(self, hotkeys: List[str] = None):
        """Get positions filtered for scoring"""
        if hotkeys is None:
            hotkeys = self.metagraph.hotkeys
            
        filtered_positions = {}
        all_positions = {}
        
        for hotkey in hotkeys:
            positions = self.get_positions_for_one_hotkey(hotkey, only_open_positions=False)
            if positions:
                all_positions[hotkey] = positions
                # Only include miners with open positions for scoring
                open_positions = [p for p in positions if not p.is_closed_position]
                if open_positions:
                    filtered_positions[hotkey] = open_positions
                    
        return filtered_positions, all_positions


class MockLedgerFactory:
    """Factory for creating test ledgers with specific characteristics"""
    
    @staticmethod
    def create_winning_ledger(
        start_ms: int = 0,
        end_ms: int = ValiConfig.TARGET_LEDGER_WINDOW_MS,
        final_return: float = 1.1,  # 10% gain
        n_checkpoints: int = None,
        max_drawdown: float = None  # Control max drawdown
    ) -> Dict[str, PerfLedger]:
        """Create a ledger with positive returns"""
        if n_checkpoints is None:
            n_checkpoints = (end_ms - start_ms) // ValiConfig.TARGET_CHECKPOINT_DURATION_MS
            
        checkpoints = []
        returns = MockLedgerFactory._generate_return_curve(
            1.0, final_return, n_checkpoints, volatility=0.02,
            max_drawdown=max_drawdown
        )
        
        # Track peak for proper MDD calculation
        peak = 1.0
        for i, ret in enumerate(returns):
            if ret > peak:
                peak = ret
                
            mdd = ret / peak  # Current value relative to peak
            
            cp = PerfCheckpoint(
                last_update_ms=start_ms + i * ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                prev_portfolio_ret=ret,
                accum_ms=ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                open_ms=ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                n_updates=100,
                gain=max(0, ret - returns[i-1]) if i > 0 else 0,
                loss=min(0, ret - returns[i-1]) if i > 0 else 0,
                mdd=mdd,
                mpv=peak
            )
            checkpoints.append(cp)
            
        ledger = PerfLedger(
            initialization_time_ms=start_ms,
            max_return=max(returns),
            cps=checkpoints
        )
        
        return {
            TP_ID_PORTFOLIO: ledger,
            "BTCUSD": ledger  # Same ledger for simplicity
        }
        
    @staticmethod
    def create_losing_ledger(
        start_ms: int = 0,
        end_ms: int = ValiConfig.TARGET_LEDGER_WINDOW_MS,
        final_return: float = 0.88,  # 12% loss (exceeds 10% MDD)
        n_checkpoints: int = None
    ) -> Dict[str, PerfLedger]:
        """Create a ledger with negative returns exceeding MDD"""
        if n_checkpoints is None:
            n_checkpoints = (end_ms - start_ms) // ValiConfig.TARGET_CHECKPOINT_DURATION_MS
            
        checkpoints = []
        returns = MockLedgerFactory._generate_return_curve(
            1.0, final_return, n_checkpoints, volatility=0.03
        )
        
        # Track peak for proper MDD calculation
        peak = 1.0
        for i, ret in enumerate(returns):
            if ret > peak:
                peak = ret
                
            mdd = ret / peak  # Current drawdown from peak
            
            cp = PerfCheckpoint(
                last_update_ms=start_ms + i * ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                prev_portfolio_ret=ret,
                accum_ms=ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                open_ms=ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                n_updates=100,
                gain=max(0, ret - returns[i-1]) if i > 0 else 0,
                loss=min(0, ret - returns[i-1]) if i > 0 else 0,
                mdd=mdd,
                mpv=peak
            )
            checkpoints.append(cp)
            
        ledger = PerfLedger(
            initialization_time_ms=start_ms,
            max_return=max(returns),
            cps=checkpoints
        )
        
        return {
            TP_ID_PORTFOLIO: ledger,
            "BTCUSD": ledger
        }
        
    @staticmethod
    def _generate_return_curve(
        start_value: float,
        end_value: float,
        n_points: int,
        volatility: float = 0.02,
        max_drawdown: float = None
    ) -> List[float]:
        """Generate a return curve with some volatility"""
        import numpy as np
        
        # Set seed for reproducible tests
        np.random.seed(42)
        
        # Linear interpolation with noise
        base_returns = np.linspace(start_value, end_value, n_points)
        
        # Add some realistic volatility
        noise = np.random.normal(0, volatility, n_points)
        noise[0] = 0  # Start exactly at start_value
        
        # Cumulative to ensure we end near end_value
        returns = base_returns + noise
        returns[-1] = end_value  # End exactly at end_value
        
        # If max_drawdown is specified, ensure we don't exceed it
        if max_drawdown is not None:
            peak = start_value
            for i in range(len(returns)):
                if returns[i] > peak:
                    peak = returns[i]
                # Ensure drawdown doesn't exceed max_drawdown
                min_allowed = peak * (1 - max_drawdown)
                if returns[i] < min_allowed:
                    returns[i] = min_allowed
        
        return returns.tolist()


class MockSubtensorWeightSetterHelper:
    """Helper for setting up weight setter tests"""

    @staticmethod
    def create_mock_subtensor():
        """Create a properly configured mock subtensor"""
        mock_subtensor = MagicMock()
        mock_subtensor.set_weights = MagicMock(return_value=(True, "Success"))
        mock_subtensor.get_current_block = MagicMock(return_value=10000)
        return mock_subtensor

    @staticmethod
    def create_mock_wallet():
        """Create a properly configured mock wallet"""
        mock_wallet = MagicMock()
        mock_wallet.hotkey = MagicMock()
        mock_wallet.coldkey = MagicMock()
        return mock_wallet

    @staticmethod
    def create_mock_debt_ledger_manager(hotkeys: List[str] = None, perf_ledger_manager = None):
        """
        Create a properly configured mock debt ledger manager

        Args:
            hotkeys: List of hotkeys to create debt ledgers for. If None, creates empty dict.
            perf_ledger_manager: Optional perf ledger manager to extract checkpoint data from
        """
        from vali_objects.vali_dataclasses.debt_ledger import DebtLedger, DebtCheckpoint

        mock_manager = MagicMock()

        if hotkeys:
            # Create debt ledgers aligned with perf ledgers if available
            mock_manager.debt_ledgers = {}

            for hotkey in hotkeys:
                # Create a debt ledger with at least one checkpoint
                debt_ledger = DebtLedger(hotkey=hotkey)

                # If we have perf ledger manager, extract performance data
                if perf_ledger_manager:
                    perf_ledgers = perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
                    if hotkey in perf_ledgers:
                        miner_ledgers = perf_ledgers[hotkey]
                        portfolio_ledger = miner_ledgers.get(TP_ID_PORTFOLIO)

                        if portfolio_ledger and portfolio_ledger.cps:
                            # Get the latest checkpoint from perf ledger
                            latest_cp = portfolio_ledger.cps[-1]

                            # Create debt checkpoint with matching performance data
                            debt_checkpoint = DebtCheckpoint(
                                timestamp_ms=latest_cp.last_update_ms,
                                # Performance data from perf ledger
                                portfolio_return=latest_cp.prev_portfolio_ret,
                                max_drawdown=latest_cp.mdd,
                                max_portfolio_value=latest_cp.mpv,
                                realized_pnl=latest_cp.realized_pnl if hasattr(latest_cp, 'realized_pnl') else 0.0,
                                unrealized_pnl=latest_cp.unrealized_pnl if hasattr(latest_cp, 'unrealized_pnl') else 0.0,
                                open_ms=latest_cp.open_ms,
                                accum_ms=latest_cp.accum_ms,
                                n_updates=latest_cp.n_updates,
                                # Default penalty values
                                total_penalty=1.0
                            )

                            # Add checkpoint to ledger (use target duration from ValiConfig)
                            debt_ledger.add_checkpoint(debt_checkpoint, ValiConfig.TARGET_CHECKPOINT_DURATION_MS)
                        else:
                            # No perf checkpoint data - create default checkpoint
                            debt_checkpoint = DebtCheckpoint(
                                timestamp_ms=TimeUtil.now_in_millis(),
                                portfolio_return=1.0,
                                total_penalty=1.0
                            )
                            debt_ledger.add_checkpoint(debt_checkpoint, ValiConfig.TARGET_CHECKPOINT_DURATION_MS)
                    else:
                        # Hotkey not in perf ledgers - create default checkpoint
                        debt_checkpoint = DebtCheckpoint(
                            timestamp_ms=TimeUtil.now_in_millis(),
                            portfolio_return=1.0,
                            total_penalty=1.0
                        )
                        debt_ledger.add_checkpoint(debt_checkpoint, ValiConfig.TARGET_CHECKPOINT_DURATION_MS)
                else:
                    # No perf ledger manager - create default checkpoint
                    debt_checkpoint = DebtCheckpoint(
                        timestamp_ms=TimeUtil.now_in_millis(),
                        portfolio_return=1.0,
                        total_penalty=1.0
                    )
                    debt_ledger.add_checkpoint(debt_checkpoint, ValiConfig.TARGET_CHECKPOINT_DURATION_MS)

                mock_manager.debt_ledgers[hotkey] = debt_ledger
        else:
            # Empty dict - weight calculation will handle this gracefully
            mock_manager.debt_ledgers = {}

        return mock_manager


class MockScoring:
    """Mock scoring implementation for testing"""

    @staticmethod
    def compute_results_checkpoint(
        ledger_dict: Dict[str, Dict[str, PerfLedger]],
        full_positions: Dict[str, List[Position]],
        evaluation_time_ms: int = None,
        **kwargs
    ):
        """Mock compute_results_checkpoint that returns simple scores"""
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        results = []

        # Generate scores based on ledger performance
        for hotkey, miner_ledger in ledger_dict.items():
            portfolio_ledger = miner_ledger.get(TP_ID_PORTFOLIO)
            if not portfolio_ledger or not portfolio_ledger.cps:
                continue

            # Calculate simple return
            if len(portfolio_ledger.cps) > 0:
                initial_return = portfolio_ledger.cps[0].prev_portfolio_ret
                final_return = portfolio_ledger.cps[-1].prev_portfolio_ret
                total_return = (final_return / initial_return) - 1.0

                # Simple scoring: positive return = good score
                score = max(0.0, min(1.0, 0.5 + total_return))
            else:
                score = 0.0

            results.append((hotkey, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def score_testing_miners(filtered_ledger, checkpoint_results):
        """Mock score_testing_miners for challenge period miners"""
        # For testing miners, we just pass through the results
        # In production, this would apply special scoring logic
        return checkpoint_results


class MockDebtBasedScoring:
    """Mock debt-based scoring implementation for testing"""

    @staticmethod
    def compute_results(
        ledger_dict: dict,
        metagraph,
        challengeperiod_manager,
        current_time_ms: int = None,
        **kwargs
    ):
        """Mock compute_results that uses actual debt ledger checkpoint data for tests"""
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        results = []

        # Extract scores from debt ledger checkpoints
        for hotkey, debt_ledger in ledger_dict.items():
            # Get the latest checkpoint from the debt ledger
            latest_checkpoint = debt_ledger.get_latest_checkpoint()

            if latest_checkpoint:
                # Use weighted_score which combines portfolio_return and penalties
                # This is what the production debt-based scoring uses
                score = latest_checkpoint.weighted_score
            else:
                # No checkpoint data - assign zero score
                score = 0.0

            results.append((hotkey, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
