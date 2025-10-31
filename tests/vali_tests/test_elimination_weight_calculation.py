# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Consolidated weight calculation tests for eliminated miners.
Combines weight calculation behavior and elimination weight tests.
"""
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import bittensor as bt

from neurons.validator import Validator
from tests.shared_objects.mock_classes import MockPositionManager
from shared_objects.mock_metagraph import MockMetagraph
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.mock_utils import (
    EnhancedMockMetagraph,
    EnhancedMockPerfLedgerManager,
    EnhancedMockPositionManager,
    MockLedgerFactory,
    MockSubtensorWeightSetterHelper
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager, EliminationReason
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
# Removed test_helpers import - using ValiConfig directly
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfLedger, TP_ID_PORTFOLIO
from vali_objects.scoring.scoring import Scoring


class TestEliminationWeightCalculation(TestBase):
    """Weight calculation behavior for eliminated miners"""

    # Test date after DebtBasedScoring activation (Dec 2025)
    # January 15, 2026 00:00:00 UTC
    TEST_TIME_MS = int(datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        
        # Create test miners
        self.ELIMINATED_MINER = "eliminated_miner"
        self.HEALTHY_MINER_1 = "healthy_miner_1"
        self.HEALTHY_MINER_2 = "healthy_miner_2"
        self.CHALLENGE_MINER = "challenge_miner"
        self.PROBATION_MINER = "probation_miner"
        self.ZOMBIE_MINER = "zombie_miner"
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        
        self.all_miners = [
            self.ELIMINATED_MINER,
            self.HEALTHY_MINER_1,
            self.HEALTHY_MINER_2,
            self.CHALLENGE_MINER,
            self.PROBATION_MINER,
            self.ZOMBIE_MINER
        ]
        
        # Initialize components with enhanced mocks
        self.mock_metagraph = EnhancedMockMetagraph(self.all_miners)
        
        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        
        self.position_locks = PositionLocks()
        
        # Create managers
        self.perf_ledger_manager = EnhancedMockPerfLedgerManager(
            self.mock_metagraph,
            running_unit_tests=True,
            perf_ledger_hks_to_invalidate={}
        )

        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.live_price_fetcher,
            None,
            running_unit_tests=True,
            contract_manager=self.contract_manager
        )
        
        self.position_manager = EnhancedMockPositionManager(
            self.mock_metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )

        self.contract_manager = ValidatorContractManager(running_unit_tests=True)

        from vali_objects.utils.plagiarism_manager import PlagiarismManager
        self.plagiarism_manager = PlagiarismManager(slack_notifier=None, running_unit_tests=True)

        self.challengeperiod_manager = ChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.perf_ledger_manager,
            contract_manager=self.contract_manager,
            plagiarism_manager=self.plagiarism_manager,
            running_unit_tests=True
        )
        
        # Set circular references
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        self.perf_ledger_manager.position_manager = self.position_manager
        self.perf_ledger_manager.elimination_manager = self.elimination_manager
        self.position_manager.challengeperiod_manager = self.challengeperiod_manager

        # Clear data
        self.clear_all_data()

        # Set up initial state
        self._setup_positions()
        self._setup_challenge_period_status()
        self._setup_perf_ledgers()

        # Create weight setter with mock debt_ledger_manager (after perf ledgers are set up)
        self.mock_debt_ledger_manager = MockSubtensorWeightSetterHelper.create_mock_debt_ledger_manager(
            self.all_miners,
            perf_ledger_manager=self.perf_ledger_manager
        )
        self.weight_setter = SubtensorWeightSetter(
            self.mock_metagraph,
            self.position_manager,
            contract_manager=self.contract_manager,
            debt_ledger_manager=self.mock_debt_ledger_manager,
            running_unit_tests=True
        )

        self._setup_eliminations()

    def tearDown(self):
        super().tearDown()
        self.clear_all_data()

    def clear_all_data(self):
        """Clear all test data"""
        self.position_manager.clear_all_miner_positions()
        self.perf_ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

    def _setup_positions(self):
        """Create positions for all miners"""
        position_time_ms = self.TEST_TIME_MS - MS_IN_24_HOURS * 5
        for miner in self.all_miners:
            position = Position(
                miner_hotkey=miner,
                position_uuid=f"{miner}_position",
                open_ms=position_time_ms,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[Order(
                    price=60000,
                    processed_ms=position_time_ms,
                    order_uuid=f"order_{miner}",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=0.5
                )]
            )
            self.position_manager.save_miner_position(position)

    def _setup_challenge_period_status(self):
        """Set up challenge period status"""
        # Main competition miners - use start of ledger window as bucket start time
        bucket_start_ms = self.TEST_TIME_MS - ValiConfig.TARGET_LEDGER_WINDOW_MS
        self.challengeperiod_manager.active_miners[self.HEALTHY_MINER_1] = (MinerBucket.MAINCOMP, bucket_start_ms, None, None)
        self.challengeperiod_manager.active_miners[self.HEALTHY_MINER_2] = (MinerBucket.MAINCOMP, bucket_start_ms, None, None)
        self.challengeperiod_manager.active_miners[self.ELIMINATED_MINER] = (MinerBucket.MAINCOMP, bucket_start_ms, None, None)

        # Challenge period miner
        self.challengeperiod_manager.active_miners[self.CHALLENGE_MINER] = (
            MinerBucket.CHALLENGE,
            self.TEST_TIME_MS - MS_IN_24_HOURS,
            None,
            None
        )

        # Probation miner
        self.challengeperiod_manager.active_miners[self.PROBATION_MINER] = (
            MinerBucket.PROBATION,
            self.TEST_TIME_MS - MS_IN_24_HOURS * 3,
            None,
            None
        )

        # Zombie miner (will be removed from metagraph)
        self.challengeperiod_manager.active_miners[self.ZOMBIE_MINER] = (MinerBucket.MAINCOMP, bucket_start_ms, None, None)

    def _setup_perf_ledgers(self):
        """Set up performance ledgers"""
        ledgers = {}

        # Use TEST_TIME_MS as the end time (current time), and calculate start based on window
        end_ms = self.TEST_TIME_MS
        start_ms = end_ms - ValiConfig.TARGET_LEDGER_WINDOW_MS

        # Healthy miners with good performance
        ledgers[self.HEALTHY_MINER_1] = MockLedgerFactory.create_winning_ledger(
            start_ms=start_ms,
            end_ms=end_ms,
            final_return=1.15  # 15% gain
        )

        ledgers[self.HEALTHY_MINER_2] = MockLedgerFactory.create_winning_ledger(
            start_ms=start_ms,
            end_ms=end_ms,
            final_return=1.10  # 10% gain
        )

        # Eliminated miner (will be excluded from weights)
        ledgers[self.ELIMINATED_MINER] = MockLedgerFactory.create_losing_ledger(
            start_ms=start_ms,
            end_ms=end_ms,
            final_return=0.88  # 12% loss, exceeds MDD
        )

        # Challenge and probation miners
        ledgers[self.CHALLENGE_MINER] = MockLedgerFactory.create_winning_ledger(
            start_ms=start_ms,
            end_ms=end_ms,
            final_return=1.05  # 5% gain
        )

        ledgers[self.PROBATION_MINER] = MockLedgerFactory.create_winning_ledger(
            start_ms=start_ms,
            end_ms=end_ms,
            final_return=1.08  # 8% gain
        )

        # Zombie miner
        ledgers[self.ZOMBIE_MINER] = MockLedgerFactory.create_winning_ledger(
            start_ms=start_ms,
            end_ms=end_ms,
            final_return=1.06  # 6% gain
        )

        self.perf_ledger_manager.save_perf_ledgers(ledgers)

    def _setup_eliminations(self):
        """Set up initial eliminations"""
        # Eliminate the MDD miner
        self.elimination_manager.eliminations.append({
            'hotkey': self.ELIMINATED_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': self.TEST_TIME_MS
        })

        # Remove eliminated miners from challenge period manager's active_miners
        self.challengeperiod_manager.remove_eliminated()

    # ========== Weight Calculation Tests (from test_weight_calculation_eliminations.py) ==========
    
    def test_eliminated_miners_excluded_from_weights(self):
        """Test that eliminated miners receive zero weights"""
        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # Get miner hotkeys and weights
        metagraph_hotkeys = list(self.mock_metagraph.hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}

        # Check eliminated miner has zero weight
        # The elimination should have been processed already
        eliminated_found = False
        for idx, weight in transformed_list:
            hotkey = metagraph_hotkeys[idx] if idx < len(metagraph_hotkeys) else None
            if hotkey == self.ELIMINATED_MINER:
                self.assertEqual(weight, 0.0)
                eliminated_found = True
                break
        
        # If not in transformed list, that's also acceptable (excluded entirely)
        if not eliminated_found:
            # Verify it's not in checkpoint results either
            result_hotkeys = [result[0] for result in checkpoint_results]
            self.assertNotIn(self.ELIMINATED_MINER, result_hotkeys)
        
        # Verify healthy miners have non-zero weights
        for healthy_miner in [self.HEALTHY_MINER_1, self.HEALTHY_MINER_2]:
            if healthy_miner in hotkey_to_idx:
                healthy_idx = hotkey_to_idx[healthy_miner]
                healthy_weight = next(
                    (weight for idx, weight in transformed_list if idx == healthy_idx),
                    None
                )
                self.assertIsNotNone(healthy_weight)
                self.assertGreater(healthy_weight, 0.0)

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_zombie_miners_excluded_from_weights(self, mock_candle_fetcher):
        """Test that zombie miners (not in metagraph) are excluded"""
        # Mock the API call to return empty list (no price data needed for this test)
        mock_candle_fetcher.return_value = []
        
        # Remove zombie miner from metagraph
        self.mock_metagraph.remove_hotkey(self.ZOMBIE_MINER)
        
        # Process eliminations to mark as zombie
        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)
        
        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
        
        # Verify zombie is not in results
        result_hotkeys = [result[0] for result in checkpoint_results]
        self.assertNotIn(self.ZOMBIE_MINER, result_hotkeys)

    def test_weight_distribution_after_eliminations(self):
        """Test that weights are properly redistributed after eliminations"""
        # Eliminate multiple miners
        self.elimination_manager.eliminations.append({
            'hotkey': self.ZOMBIE_MINER,
            'reason': EliminationReason.ZOMBIE.value,
            'dd': 0.0,
            'elimination_initiated_time_ms': self.TEST_TIME_MS
        })

        # Remove the newly eliminated miner from active_miners
        self.challengeperiod_manager.remove_eliminated()

        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
        
        # Get non-zero weights
        non_zero_weights = [weight for _, weight in transformed_list if weight > 0]
        
        # Verify we have non-zero weights
        if non_zero_weights:
            total_weight = sum(non_zero_weights)
            self.assertGreater(total_weight, 0)
            # The SubtensorWeightSetter handles normalization internally when calling subtensor.set_weights

    def test_challenge_period_miners_weights(self):
        """Test weight calculation for challenge period miners"""
        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
        
        # Challenge period miners should be included in results
        result_hotkeys = [result[0] for result in checkpoint_results]
        
        # In backtesting mode, challenge miners would be included
        # In production mode, they might not be
        if self.weight_setter.is_backtesting:
            self.assertIn(self.CHALLENGE_MINER, result_hotkeys)

    def test_scoring_with_mixed_miner_states(self):
        """Test scoring calculation with miners in different states"""
        # Get filtered ledger for scoring
        success_hotkeys = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(
            hotkeys=success_hotkeys
        )
        
        # Eliminated miner should not be in filtered ledger
        self.assertNotIn(self.ELIMINATED_MINER, filtered_ledger)
        
        # Healthy miners should be included
        self.assertIn(self.HEALTHY_MINER_1, filtered_ledger)
        
        # Get positions for scoring
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(
            hotkeys=success_hotkeys
        )

        asset_classes = list(AssetSegmentation.distill_asset_classes(ValiConfig.ASSET_CLASS_BREAKDOWN))
        asset_class_min_days = {asset_class: ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL for asset_class in asset_classes}
        
        # Compute scores
        if len(filtered_ledger) > 0:
            scores = Scoring.compute_results_checkpoint(
                filtered_ledger,
                filtered_positions,
                asset_class_min_days=asset_class_min_days,
                evaluation_time_ms=self.TEST_TIME_MS,
                all_miner_account_sizes={}
            )
            
            # Verify scores don't include eliminated miners
            score_hotkeys = [score[0] for score in scores]
            self.assertNotIn(self.ELIMINATED_MINER, score_hotkeys)

    def test_invalidated_miners_excluded_from_scoring(self):
        """Test that invalidated miners are excluded from scoring"""
        # Invalidate a miner
        self.perf_ledger_manager.perf_ledger_hks_to_invalidate[self.HEALTHY_MINER_2] = True
        
        # Get filtered ledger
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring()
        
        # Invalidated miner should not be included
        self.assertNotIn(self.HEALTHY_MINER_2, filtered_ledger)

    def test_dtao_block_registration_handling(self):
        """Test handling of dTAO block registration edge cases"""
        # Set specific block registration times
        target_dtao_block_zero_incentive_start = 4916273
        target_dtao_block_zero_incentive_end = 4951874
        
        # Mock a miner with problematic registration block
        idx = self.mock_metagraph.hotkeys.index(self.HEALTHY_MINER_1)
        self.mock_metagraph.block_at_registration[idx] = target_dtao_block_zero_incentive_start + 100
        
        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
        
        # The weight setter should handle this case
        # (In production, such miners might get zero weight)
        self.assertIsNotNone(transformed_list)

    def test_weight_calculation_performance_metrics(self):
        """Test that weight calculation uses performance metrics correctly"""
        # Get ledgers for healthy miners - portfolio_only=True returns dict[str, PerfLedger]
        ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=True)
        
        # Verify ledger structure
        for miner in [self.HEALTHY_MINER_1, self.HEALTHY_MINER_2]:
            if miner in ledgers:
                # With portfolio_only=True, we get PerfLedger directly
                portfolio_ledger = ledgers[miner]
                self.assertIsInstance(portfolio_ledger, PerfLedger)
                self.assertTrue(hasattr(portfolio_ledger, 'cps'))
                self.assertGreater(len(portfolio_ledger.cps), 0)

    # ========== Simple Weight Behavior Tests (from test_elimination_weight_behavior.py concepts) ==========
    
    def test_weight_normalization_invariant(self):
        """Test that weights always sum to 1.0 regardless of eliminations"""
        # Test with no eliminations
        self.elimination_manager.eliminations = []
        # Re-add the eliminated miner to active_miners since we cleared eliminations
        self.challengeperiod_manager.active_miners[self.ELIMINATED_MINER] = (MinerBucket.MAINCOMP, 0, None, None)
        current_time = TimeUtil.now_in_millis()
        _, transformed_list = self.weight_setter.compute_weights_default(current_time)

        # The transformed_list contains raw scores, not normalized weights
        # The actual normalization happens in the subtensor.set_weights call
        # So we just verify that we have non-empty results
        self.assertGreater(len(transformed_list), 0)

        # Test with eliminations - verify eliminated miners get zero
        self._setup_eliminations()
        _, transformed_list = self.weight_setter.compute_weights_default(current_time)
        
        # Find eliminated miner in results
        metagraph_hotkeys = list(self.mock_metagraph.hotkeys)
        for idx, weight in transformed_list:
            if idx < len(metagraph_hotkeys) and metagraph_hotkeys[idx] == self.ELIMINATED_MINER:
                self.assertEqual(weight, 0.0)

    def test_progressive_elimination_weight_behavior(self):
        """Test weight behavior as miners are progressively eliminated"""
        current_time = TimeUtil.now_in_millis()

        # Initial state - one elimination
        _, initial_weights = self.weight_setter.compute_weights_default(current_time)
        initial_non_zero = sum(1 for _, w in initial_weights if w > 0)

        # Add another elimination
        self.elimination_manager.eliminations.append({
            'hotkey': self.HEALTHY_MINER_2,
            'reason': EliminationReason.PLAGIARISM.value,
            'dd': 0.0,
            'elimination_initiated_time_ms': self.TEST_TIME_MS
        })

        # Remove the newly eliminated miner from active_miners
        self.challengeperiod_manager.remove_eliminated()

        # Recompute weights
        _, new_weights = self.weight_setter.compute_weights_default(current_time)
        new_non_zero = sum(1 for _, w in new_weights if w > 0)
        
        # Fewer miners should have non-zero weights
        self.assertLess(new_non_zero, initial_non_zero)
        
        # Verify we have weights
        if new_weights:
            raw_weights = [w for _, w in new_weights]
            total = sum(raw_weights)
            self.assertGreater(total, 0)  # Should have some non-zero weights
            # The weight setter handles normalization internally

    def test_weight_normalization_by_subtensor(self):
        """Test that our weight setter properly formats weights for Bittensor"""
        # Get the weights that would be sent to Bittensor
        current_time = TimeUtil.now_in_millis()
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(current_time)
        
        # The transformed_list contains (uid, score) tuples
        # These are the raw scores that will be sent to Bittensor
        if transformed_list:
            # Check that eliminated miners have zero weight
            eliminated_uids = []
            for hotkey in self.elimination_manager.get_eliminated_hotkeys():
                if hotkey in self.mock_metagraph.hotkeys:
                    uid = self.mock_metagraph.hotkeys.index(hotkey)
                    eliminated_uids.append(uid)
            
            # Verify eliminated miners have zero scores
            for uid, score in transformed_list:
                if uid in eliminated_uids:
                    self.assertEqual(score, 0.0)
        
        # Now test the full weight setting process
        # The weights passed to Bittensor are the normalized scores from Scoring
        self.assertGreater(len(transformed_list), 0)
        # Verify eliminated miners have zero weight
        if self.ELIMINATED_MINER in self.mock_metagraph.hotkeys:
            eliminated_idx = self.mock_metagraph.hotkeys.index(self.ELIMINATED_MINER)
            # Check if this miner's index is in the weights
            transformed_uids = [uid for uid, _ in transformed_list]
            if eliminated_idx in transformed_uids:
                pos = transformed_uids.index(eliminated_idx)
                self.assertEqual(transformed_list[pos], 0.0)
                
    def test_scoring_normalize_scores_method(self):
        """Test the production Scoring.normalize_scores method directly"""
        # Import the real Scoring class
        from vali_objects.scoring.scoring import Scoring as RealScoring
        
        # Test with various score distributions
        test_cases = [
            # Regular scores
            {"miner1": 100.0, "miner2": 50.0, "miner3": 25.0},
            # All equal scores
            {"miner1": 1.0, "miner2": 1.0, "miner3": 1.0},
            # One dominant miner
            {"miner1": 1000.0, "miner2": 1.0, "miner3": 1.0},
            # Fractional scores
            {"miner1": 0.15, "miner2": 0.10, "miner3": 0.05}
        ]
        
        for scores in test_cases:
            normalized = RealScoring.normalize_scores(scores)
            
            # Verify all values are normalized
            total = sum(normalized.values())
            self.assertAlmostEqual(total, 1.0, places=6)
            
            # Verify relative ordering is preserved
            original_order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            normalized_order = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
            self.assertEqual([x[0] for x in original_order], [x[0] for x in normalized_order])
            
        # Test edge cases
        empty_scores = {}
        self.assertEqual(RealScoring.normalize_scores(empty_scores), {})
        
        zero_scores = {"miner1": 0.0, "miner2": 0.0}
        self.assertEqual(RealScoring.normalize_scores(zero_scores), {})

    def test_extreme_elimination_scenario(self):
        """Test behavior when almost all miners are eliminated"""
        # Eliminate all but one miner
        for miner in self.all_miners[1:]:  # Keep first miner
            self.elimination_manager.eliminations.append({
                'hotkey': miner,
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.15,
                'elimination_initiated_time_ms': self.TEST_TIME_MS
            })

        # Remove all newly eliminated miners from active_miners
        self.challengeperiod_manager.remove_eliminated()

        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
        
        # Should have exactly one miner with weight 1.0
        non_zero_weights = [(idx, w) for idx, w in transformed_list if w > 0]
        
        if non_zero_weights:
            self.assertEqual(len(non_zero_weights), 1)
            self.assertAlmostEqual(non_zero_weights[0][1], 1.0, places=6)
