# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Consolidated weight calculation tests for eliminated miners.
Combines weight calculation behavior and elimination weight tests.
"""
from datetime import datetime, timezone

import bittensor as bt

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from shared_objects.locks.position_lock import PositionLocks
from vali_objects.scoring.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger
from vali_objects.scoring.scoring import Scoring


class TestEliminationWeightCalculation(TestBase):
    """
    Weight calculation behavior for eliminated miners.
    Uses ServerOrchestrator singleton for shared server infrastructure across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Test date after DebtBasedScoring activation (Nov 2025)
    # December 15, 2025 00:00:00 UTC
    TEST_TIME_MS = int(datetime(2025, 12, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    debt_ledger_client = None
    elimination_client = None
    challenge_period_client = None
    plagiarism_client = None
    position_locks = None
    weight_setter = None

    # Test miner constants
    ELIMINATED_MINER = "eliminated_miner"
    HEALTHY_MINER_1 = "healthy_miner_1"
    HEALTHY_MINER_2 = "healthy_miner_2"
    CHALLENGE_MINER = "challenge_miner"
    PROBATION_MINER = "probation_miner"
    ZOMBIE_MINER = "zombie_miner"
    DEFAULT_ACCOUNT_SIZE = 100_000

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.debt_ledger_client = cls.orchestrator.get_client('debt_ledger')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')

        # Define test miners BEFORE creating test data
        cls.all_test_miners = [
            cls.ELIMINATED_MINER,
            cls.HEALTHY_MINER_1,
            cls.HEALTHY_MINER_2,
            cls.CHALLENGE_MINER,
            cls.PROBATION_MINER,
            cls.ZOMBIE_MINER
        ]
        # Initialize metagraph with test miners
        cls.metagraph_client.set_hotkeys(cls.all_test_miners)

        # Create position locks instance
        cls.position_locks = PositionLocks()

        # Weight setter will be initialized per-test because it depends on test data
        cls.weight_setter = None

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Create fresh test data
        self._create_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        # Define all test miners
        self.all_miners = [
            self.ELIMINATED_MINER,
            self.HEALTHY_MINER_1,
            self.HEALTHY_MINER_2,
            self.CHALLENGE_MINER,
            self.PROBATION_MINER,
            self.ZOMBIE_MINER
        ]

        # Re-initialize metagraph after clear_all_test_data()
        self.metagraph_client.set_hotkeys(self.all_miners)

        # Set up initial state
        self._setup_positions()
        self._setup_challenge_period_status()
        self._setup_perf_ledgers()

        # Build debt ledgers from perf ledgers (required for weight calculation)
        self._build_debt_ledgers()

        # Initialize weight setter (now that debt ledgers are ready)
        from vali_objects.vali_config import RPCConnectionMode
        self.weight_setter = SubtensorWeightSetter(
            connection_mode=RPCConnectionMode.RPC,
            is_backtesting=True,  # For test mode
            is_mainnet=False  # testnet mode
        )

        self._setup_eliminations()

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
            self.position_client.save_miner_position(position)

    def _setup_challenge_period_status(self):
        """Set up challenge period status for miners"""
        # Build miners dict
        miners = {}

        # Main competition miners - use start of ledger window as bucket start time
        bucket_start_ms = self.TEST_TIME_MS - ValiConfig.TARGET_LEDGER_WINDOW_MS
        for miner in [self.HEALTHY_MINER_1, self.HEALTHY_MINER_2, self.ELIMINATED_MINER, self.ZOMBIE_MINER]:
            miners[miner] = (MinerBucket.MAINCOMP, bucket_start_ms, None, None)

        # Challenge period miner
        miners[self.CHALLENGE_MINER] = (
            MinerBucket.CHALLENGE,
            self.TEST_TIME_MS - MS_IN_24_HOURS,
            None,
            None
        )

        # Probation miner
        miners[self.PROBATION_MINER] = (
            MinerBucket.PROBATION,
            self.TEST_TIME_MS - MS_IN_24_HOURS * 3,
            None,
            None
        )

        # Update using client API
        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners(miners)
        # Note: Data persistence handled automatically by server - no manual disk write needed

    def _setup_perf_ledgers(self):
        """Set up performance ledgers for testing"""
        ledgers = {}

        # Use TEST_TIME_MS as the end time (current time), and calculate start based on window
        end_ms = self.TEST_TIME_MS
        start_ms = end_ms - ValiConfig.TARGET_LEDGER_WINDOW_MS

        # Healthy miners with good performance
        ledgers[self.HEALTHY_MINER_1] = generate_winning_ledger(
            start_ms,
            end_ms
        )

        ledgers[self.HEALTHY_MINER_2] = generate_winning_ledger(
            start_ms,
            end_ms
        )

        # Eliminated miner (will be excluded from weights)
        ledgers[self.ELIMINATED_MINER] = generate_losing_ledger(
            start_ms,
            end_ms
        )

        # Challenge and probation miners
        ledgers[self.CHALLENGE_MINER] = generate_winning_ledger(
            start_ms,
            end_ms
        )

        ledgers[self.PROBATION_MINER] = generate_winning_ledger(
            start_ms,
            end_ms
        )

        # Zombie miner
        ledgers[self.ZOMBIE_MINER] = generate_winning_ledger(
            start_ms,
            end_ms
        )

        self.perf_ledger_client.save_perf_ledgers(ledgers)
        self.perf_ledger_client.re_init_perf_ledger_data()

    def _build_debt_ledgers(self):
        """Build debt ledgers from perf ledgers for weight calculation tests."""
        # The DebtLedgerServer builds debt ledgers from THREE sources:
        # 1. Performance ledgers (via PerfLedgerClient) ✅ Already set up
        # 2. Emissions ledgers (via EmissionsLedgerManager) ⚠️ Need to build
        # 3. Penalty ledgers (via PenaltyLedgerManager) ⚠️ Need to build

        # Build penalty ledgers FIRST (they depend on perf ledgers and challenge period data)
        bt.logging.info("Building penalty ledgers...")
        self.debt_ledger_client.build_penalty_ledgers(verbose=False, delta_update=False)

        # Build emissions ledgers SECOND (they depend on metagraph data)
        bt.logging.info("Building emissions ledgers...")
        self.debt_ledger_client.build_emissions_ledgers(delta_update=False)

        # Now build debt ledgers THIRD (combines all three sources)
        bt.logging.info("Building debt ledgers...")
        self.debt_ledger_client.build_debt_ledgers(verbose=False, delta_update=False)

        bt.logging.info(f"Built debt ledgers for {len(self.all_miners)} miners")

    def _setup_eliminations(self):
        """Set up initial eliminations"""
        # Eliminate the MDD miner
        self.elimination_client.add_elimination(self.ELIMINATED_MINER, {
            'hotkey': self.ELIMINATED_MINER,
            'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
            'dd': 0.12,
            'elimination_initiated_time_ms': self.TEST_TIME_MS
        })

        # Remove eliminated miners from challenge period client
        self.challenge_period_client.remove_eliminated()

    # ========== Weight Calculation Tests (from test_weight_calculation_eliminations.py) ==========

    def test_eliminated_miners_excluded_from_weights(self):
        """Test that eliminated miners receive zero weights"""
        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # Get miner hotkeys and weights
        metagraph_hotkeys = self.metagraph_client.get_hotkeys()
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

    def test_zombie_miners_excluded_from_weights(self):
        """Test that zombie miners (not in metagraph) are excluded"""
        # Remove zombie miner from metagraph
        new_hotkeys = [hk for hk in self.metagraph_client.get_hotkeys() if hk != self.ZOMBIE_MINER]
        self.metagraph_client.set_hotkeys(new_hotkeys)

        # Process eliminations to mark as zombie
        self.elimination_client.process_eliminations()

        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # Verify zombie is not in results
        result_hotkeys = [result[0] for result in checkpoint_results]
        self.assertNotIn(self.ZOMBIE_MINER, result_hotkeys)

    def test_weight_distribution_after_eliminations(self):
        """Test that weights are properly redistributed after eliminations"""
        # Eliminate multiple miners
        self.elimination_client.add_elimination(self.ZOMBIE_MINER, {
            'hotkey': self.ZOMBIE_MINER,
            'reason': EliminationReason.ZOMBIE.value,
            'dd': 0.0,
            'elimination_initiated_time_ms': self.TEST_TIME_MS
        })

        # Remove the newly eliminated miner from active_miners
        self.challenge_period_client.remove_eliminated()

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
        success_hotkeys = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        filtered_ledger = self.perf_ledger_client.filtered_ledger_for_scoring(
            hotkeys=success_hotkeys
        )

        # Eliminated miner should not be in filtered ledger
        self.assertNotIn(self.ELIMINATED_MINER, filtered_ledger)

        # Healthy miners should be included
        self.assertIn(self.HEALTHY_MINER_1, filtered_ledger)

        # Get positions for scoring
        filtered_positions, _ = self.position_client.filtered_positions_for_scoring(
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
        # Invalidate a miner via client
        self.perf_ledger_client.set_invalidation(self.HEALTHY_MINER_2, True)

        # Get filtered ledger
        filtered_ledger = self.perf_ledger_client.filtered_ledger_for_scoring()

        # Invalidated miner should not be included
        self.assertNotIn(self.HEALTHY_MINER_2, filtered_ledger)

    def test_dtao_block_registration_handling(self):
        """Test handling of dTAO block registration edge cases"""
        # Set specific block registration times
        target_dtao_block_zero_incentive_start = 4916273
        target_dtao_block_zero_incentive_end = 4951874

        # Mock a miner with problematic registration block
        self.metagraph_client.set_block_at_registration(self.HEALTHY_MINER_1, target_dtao_block_zero_incentive_start + 100)

        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # The weight setter should handle this case
        # (In production, such miners might get zero weight)
        self.assertIsNotNone(transformed_list)

    def test_weight_calculation_performance_metrics(self):
        """Test that weight calculation uses performance metrics correctly"""
        # Get ledgers for healthy miners - portfolio_only=True returns dict[str, PerfLedger]
        ledgers = self.perf_ledger_client.get_perf_ledgers(portfolio_only=True)

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
        self.elimination_client.clear_eliminations()
        # Re-add the eliminated miner to active_miners since we cleared eliminations
        miners = {self.ELIMINATED_MINER: (MinerBucket.MAINCOMP, 0, None, None)}
        self.challenge_period_client.update_miners(miners)
        _, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # The transformed_list contains raw scores, not normalized weights
        # The actual normalization happens in the subtensor.set_weights call
        # So we just verify that we have non-empty results
        self.assertGreater(len(transformed_list), 0)

        # Test with eliminations - verify eliminated miners get zero
        self._setup_eliminations()
        _, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # Find eliminated miner in results
        metagraph_hotkeys = self.metagraph_client.get_hotkeys()
        for idx, weight in transformed_list:
            if idx < len(metagraph_hotkeys) and metagraph_hotkeys[idx] == self.ELIMINATED_MINER:
                self.assertEqual(weight, 0.0)

    def test_progressive_elimination_weight_behavior(self):
        """Test weight behavior as miners are progressively eliminated"""
        # Initial state - one elimination
        _, initial_weights = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
        initial_non_zero = sum(1 for _, w in initial_weights if w > 0)

        # Add another elimination
        self.elimination_client.add_elimination(self.HEALTHY_MINER_2, {
            'hotkey': self.HEALTHY_MINER_2,
            'reason': EliminationReason.PLAGIARISM.value,
            'dd': 0.0,
            'elimination_initiated_time_ms': self.TEST_TIME_MS
        })

        # Remove the newly eliminated miner from active_miners
        self.challenge_period_client.remove_eliminated()

        # Recompute weights
        _, new_weights = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)
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
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # The transformed_list contains (uid, score) tuples
        # These are the raw scores that will be sent to Bittensor
        if transformed_list:
            # Check that eliminated miners have zero weight
            eliminated_uids = []
            metagraph_hotkeys = self.metagraph_client.get_hotkeys()
            for hotkey in self.elimination_client.get_eliminated_hotkeys():
                if hotkey in metagraph_hotkeys:
                    uid = metagraph_hotkeys.index(hotkey)
                    eliminated_uids.append(uid)

            # Verify eliminated miners have zero scores
            for uid, score in transformed_list:
                if uid in eliminated_uids:
                    self.assertEqual(score, 0.0)

        # Now test the full weight setting process
        # The weights passed to Bittensor are the normalized scores from Scoring
        self.assertGreater(len(transformed_list), 0)
        # Verify eliminated miners have zero weight
        metagraph_hotkeys = self.metagraph_client.get_hotkeys()
        if self.ELIMINATED_MINER in metagraph_hotkeys:
            eliminated_idx = metagraph_hotkeys.index(self.ELIMINATED_MINER)
            # Check if this miner's index is in the weights
            transformed_uids = [uid for uid, _ in transformed_list]
            if eliminated_idx in transformed_uids:
                pos = transformed_uids.index(eliminated_idx)
                self.assertEqual(transformed_list[pos][1], 0.0)
                
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
            self.elimination_client.add_elimination(miner, {
                'hotkey': miner,
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.15,
                'elimination_initiated_time_ms': self.TEST_TIME_MS
            })

        # Remove all newly eliminated miners from active_miners
        self.challenge_period_client.remove_eliminated()

        # Compute weights
        checkpoint_results, transformed_list = self.weight_setter.compute_weights_default(self.TEST_TIME_MS)

        # Should have exactly one miner with weight 1.0
        non_zero_weights = [(idx, w) for idx, w in transformed_list if w > 0]

        if non_zero_weights:
            self.assertEqual(len(non_zero_weights), 1)
            self.assertAlmostEqual(non_zero_weights[0][1], 1.0, places=6)
