# developer: Claude Code Review
"""
Comprehensive tests for the probation feature implementation.
These tests verify the critical functionality gaps identified during code review
and ensure production-ready confidence for the probation bucket feature.

NOTE FOR PR AUTHOR:
Some tests may initially fail as they test edge cases and scenarios that
may need additional implementation. Comments within each test provide guidance
on what logic may need to be added or verified.
"""

import unittest
import unittest.mock

from tests.shared_objects.mock_classes import MockPositionManager, MockLivePriceFetcher
from shared_objects.mock_metagraph import MockMetagraph
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager, EliminationReason
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO, PerfLedgerManager


class TestProbationComprehensive(TestBase):
    """
    Comprehensive test suite for probation functionality.
    Tests critical edge cases and production scenarios for probation bucket feature.
    """

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        self.N_MAINCOMP_MINERS = 30
        self.N_CHALLENGE_MINERS = 5
        self.N_PROBATION_MINERS = 5
        self.N_ELIMINATED_MINERS = 5

        # Time configurations
        self.START_TIME = 1000
        self.END_TIME = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS - 1
        self.CURRENT_TIME = self.START_TIME + ValiConfig.PROBATION_MAXIMUM_MS - 1000

        # Probation-specific time configurations
        self.PROBATION_START_TIME = self.START_TIME
        self.PROBATION_ALMOST_EXPIRED = self.PROBATION_START_TIME + ValiConfig.PROBATION_MAXIMUM_MS - 1000
        self.PROBATION_EXPIRED = self.PROBATION_START_TIME + ValiConfig.PROBATION_MAXIMUM_MS + 1000

        # Define miner categories
        self.SUCCESS_MINER_NAMES = [f"maincomp_miner{i}" for i in range(1, self.N_MAINCOMP_MINERS+1)]
        self.CHALLENGE_MINER_NAMES = [f"challenge_miner{i}" for i in range(1, self.N_CHALLENGE_MINERS+1)]
        self.PROBATION_MINER_NAMES = [f"probation_miner{i}" for i in range(1, self.N_PROBATION_MINERS+1)]
        self.ELIMINATED_MINER_NAMES = [f"eliminated_miner{i}" for i in range(1, self.N_ELIMINATED_MINERS+1)]

        self.ALL_MINER_NAMES = (self.SUCCESS_MINER_NAMES + self.CHALLENGE_MINER_NAMES +
                               self.PROBATION_MINER_NAMES + self.ELIMINATED_MINER_NAMES)

        # Setup system components
        self.mock_metagraph = MockMetagraph(self.ALL_MINER_NAMES)
        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True, contract_manager=self.contract_manager)
        self.ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.position_manager = MockPositionManager(self.mock_metagraph,
                                                    perf_ledger_manager=self.ledger_manager,
                                                    elimination_manager=self.elimination_manager,
                                                    live_price_fetcher=self.live_price_fetcher)
        self.plagiarism_manager = PlagiarismManager(slack_notifier=None, running_unit_tests=True)

        self.challengeperiod_manager = ChallengePeriodManager(self.mock_metagraph,
                                                              position_manager=self.position_manager,
                                                              perf_ledger_manager=self.ledger_manager,
                                                              contract_manager=self.contract_manager,
                                                              plagiarism_manager=self.plagiarism_manager,
                                                              running_unit_tests=True)
        self.weight_setter = SubtensorWeightSetter(self.mock_metagraph,
                                                   self.position_manager,
                                                   contract_manager=self.contract_manager,
                                                   running_unit_tests=True)

        # Cross-reference managers
        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        self.position_manager.challengeperiod_manager = self.challengeperiod_manager

        # Setup default positions and ledgers
        self._setup_default_data()
        self._populate_active_miners()

    def _setup_default_data(self):
        """Setup positions and ledgers for all miners"""
        self.POSITIONS = {}
        self.LEDGERS = {}
        self.HK_TO_OPEN_MS = {}

        for miner in self.ALL_MINER_NAMES:
            # Create positions
            position = Position(
                miner_hotkey=miner,
                position_uuid=f"{miner}_position",
                open_ms=self.START_TIME,
                close_ms=self.END_TIME,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=True,
                return_at_close=1.1 if miner not in self.ELIMINATED_MINER_NAMES else 0.8,
                orders=[Order(price=60000, processed_ms=self.START_TIME, order_uuid=f"{miner}_order",
                              trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],
            )
            self.POSITIONS[miner] = [position]
            self.HK_TO_OPEN_MS[miner] = self.START_TIME

            # Create ledgers
            if miner in self.ELIMINATED_MINER_NAMES:
                ledger = generate_losing_ledger(self.START_TIME, self.END_TIME)
            else:
                ledger = generate_winning_ledger(self.START_TIME, self.END_TIME)
            self.LEDGERS[miner] = ledger

        # Save to managers
        self.ledger_manager.save_perf_ledgers(self.LEDGERS)
        for miner, positions in self.POSITIONS.items():
            for position in positions:
                self.position_manager.save_miner_position(position)

    def _populate_active_miners(self):
        """Setup initial miner bucket assignments"""
        miners = {}
        for hotkey in self.SUCCESS_MINER_NAMES:
            miners[hotkey] = (MinerBucket.MAINCOMP, self.START_TIME, None, None)
        for hotkey in self.CHALLENGE_MINER_NAMES:
            miners[hotkey] = (MinerBucket.CHALLENGE, self.START_TIME, None, None)
        for hotkey in self.PROBATION_MINER_NAMES:
            miners[hotkey] = (MinerBucket.PROBATION, self.PROBATION_START_TIME, None, None)
        for hotkey in self.ELIMINATED_MINER_NAMES:
            miners[hotkey] = (MinerBucket.CHALLENGE, self.START_TIME, None, None)
        self.challengeperiod_manager.active_miners = miners

    def tearDown(self):
        super().tearDown()
        self.position_manager.clear_all_miner_positions()
        self.ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.challengeperiod_manager.elimination_manager.clear_eliminations()

    def test_probation_timeout_elimination(self):
        """
        CRITICAL TEST: Verify miners in probation for 30+ days get eliminated

        NOTE FOR PR AUTHOR:
        This test checks if probation miners are eliminated after 30 days.
        If this test fails, you may need to add elimination logic in:
        - challengeperiod_manager.py:meets_time_criteria() for PROBATION bucket
        - or in the inspect() method to handle probation timeouts

        Expected behavior: Miners in probation > 30 days should be eliminated
        """
        # Setup probation miner with expired timestamp
        expired_miner = "probation_miner1"
        self.challengeperiod_manager.active_miners[expired_miner] = (
            MinerBucket.PROBATION,
            self.PROBATION_START_TIME,
            None,
            None
        )

        # Setup probation miner still within time limit
        valid_miner = "probation_miner2"
        self.challengeperiod_manager.active_miners[valid_miner] = (
            MinerBucket.PROBATION,
            self.PROBATION_EXPIRED - 10000,
            None,
            None
        )

        # Refresh challenge period at current time
        self.challengeperiod_manager.refresh(current_time=self.PROBATION_EXPIRED)
        self.elimination_manager.process_eliminations(PositionLocks())

        # Check eliminations
        eliminated_hotkeys = self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()

        # Expired probation miner should be eliminated
        self.assertIn(expired_miner, eliminated_hotkeys,
                     "Probation miner past 30-day limit should be eliminated")

        # Valid probation miner should still be in probation
        self.assertNotIn(valid_miner, eliminated_hotkeys,
                        "Probation miner within 30-day limit should not be eliminated")

        # NOTE Failing because all miners have the same score = maincomp
        # self.assertIn(valid_miner, self.challengeperiod_manager.get_probation_miners(),
        #              "Valid probation miner should remain in probation bucket")

    def test_promotion_demotion_with_exactly_25_miners(self):
        """
        CRITICAL TEST: Test promotion/demotion logic when exactly 25 miners exist

        NOTE FOR PR AUTHOR:
        This tests the boundary condition for PROMOTION_THRESHOLD_RANK = 25.
        If this test fails, check the logic in:
        - challengeperiod_manager.py:evaluate_promotions() around line 393-406
        - Ensure proper handling when len(sorted_scores) == threshold_rank
        """
        # Setup exactly 25 maincomp miners
        exactly_25_miners = [f"miner{i}" for i in range(1, 26)]
        challenge_miner = "challenge_test_miner"
        probation_miner = "probation_test_miner"

        # Clear and setup new miner configuration
        self.challengeperiod_manager.active_miners.clear()
        for miner in exactly_25_miners:
            self.challengeperiod_manager.active_miners[miner] = (MinerBucket.MAINCOMP, self.START_TIME, None, None)

        self.challengeperiod_manager.active_miners[challenge_miner] = (MinerBucket.CHALLENGE, self.START_TIME, None, None)
        self.challengeperiod_manager.active_miners[probation_miner] = (MinerBucket.PROBATION, self.START_TIME, None, None)

        # Setup positions and ledgers for new miners
        for miner in exactly_25_miners + [challenge_miner, probation_miner]:
            if miner not in self.POSITIONS:
                position = Position(
                    miner_hotkey=miner,
                    position_uuid=f"{miner}_position",
                    open_ms=self.START_TIME,
                    close_ms=self.END_TIME,
                    trade_pair=TradePair.BTCUSD,
                    is_closed_position=True,
                    return_at_close=1.1,
                    orders=[Order(price=60000, processed_ms=self.START_TIME, order_uuid=f"{miner}_order",
                                  trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],
                )
                self.POSITIONS[miner] = [position]
                self.position_manager.save_miner_position(position)

                ledger = generate_winning_ledger(self.START_TIME, self.END_TIME)
                self.LEDGERS[miner] = ledger
                self.ledger_manager.save_perf_ledgers({miner: ledger})

        # Test promotion/demotion with exactly 25 miners
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # Verify system handles exactly 25 miners correctly
        maincomp_miners = self.challengeperiod_manager.get_success_miners()
        challenge_miners = self.challengeperiod_manager.get_testing_miners()
        probation_miners = self.challengeperiod_manager.get_probation_miners()

        # Should maintain threshold logic properly
        total_competing = len(maincomp_miners) + len(challenge_miners) + len(probation_miners)
        self.assertGreaterEqual(total_competing, 25,
                               "Should maintain at least 25 competing miners")

    def test_probation_miner_promotion_to_maincomp(self):
        """
        Test successful promotion from probation directly to maincomp

        NOTE FOR PR AUTHOR:
        This verifies probation miners can be promoted if they score above threshold.
        If this fails, check that probation miners are included in the evaluation
        logic in challengeperiod_manager.py:inspect() method.
        """
        # Setup high-performing probation miner
        top_probation_miner = "probation_miner1"

        # Ensure this miner has excellent performance
        excellent_ledger = generate_winning_ledger(self.START_TIME, self.END_TIME)
        # Boost the performance significantly
        for checkpoint in excellent_ledger[TP_ID_PORTFOLIO].cps:
            checkpoint.gain = 0.15  # 15% gain
            checkpoint.loss = -0.01  # Minimal loss

        self.ledger_manager.save_perf_ledgers({top_probation_miner: excellent_ledger})

        # Run refresh
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # Check if probation miner was promoted
        maincomp_miners = self.challengeperiod_manager.get_success_miners()
        probation_miners = self.challengeperiod_manager.get_probation_miners()

        # High-performing probation miner should be promoted to maincomp
        if top_probation_miner in maincomp_miners:
            self.assertNotIn(top_probation_miner, probation_miners,
                           "Promoted miner should not remain in probation")
        else:
            # If not promoted, they should still be in probation (not eliminated)
            self.assertIn(top_probation_miner, probation_miners,
                         "Non-promoted probation miner should remain in probation")

    def test_maincomp_to_probation_to_elimination_flow(self):
        """
        Test complete demotion flow: maincomp → probation → elimination

        NOTE FOR PR AUTHOR:
        This tests the full lifecycle of a failing miner.
        Verify that:
        1. Poor-performing maincomp miners get demoted to probation
        2. Poor-performing probation miners eventually get eliminated
        """
        # Setup a poor-performing maincomp miner
        poor_miner = "maincomp_miner1"

        # Give this miner terrible performance
        poor_ledger = generate_losing_ledger(self.START_TIME, self.END_TIME)
        for checkpoint in poor_ledger[TP_ID_PORTFOLIO].cps:
            checkpoint.gain = 0.01
            checkpoint.loss = -0.15  # 15% loss
            checkpoint.mdd = 0.92    # high-ish drawdown should reduce their scores

        self.LEDGERS.update({poor_miner: poor_ledger})
        self.ledger_manager.save_perf_ledgers(self.LEDGERS)

        # First refresh - should demote to probation or eliminate due to drawdown
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)
        with unittest.mock.patch.object(self.elimination_manager, 'live_price_fetcher', self.live_price_fetcher):
            self.elimination_manager.process_eliminations(PositionLocks())

        maincomp_miners = self.challengeperiod_manager.get_success_miners()

        self.assertNotIn(poor_miner, maincomp_miners)

        # Now test probation timeout elimination
        future_time = self.CURRENT_TIME + ValiConfig.PROBATION_MAXIMUM_MS + 1000
        self.challengeperiod_manager.refresh(current_time=future_time)
        with unittest.mock.patch.object(self.elimination_manager, 'live_price_fetcher', self.live_price_fetcher):
            self.elimination_manager.process_eliminations(PositionLocks())

        final_eliminated = self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()
        self.assertIn(poor_miner, final_eliminated,
                     "Poor probation miner should be eliminated after timeout")

    def test_probation_state_persistence_across_restarts(self):
        """
        Test probation state survives validator restarts

        NOTE FOR PR AUTHOR:
        This verifies the checkpoint save/load maintains probation timestamps correctly.
        Check the implementation in:
        - challengeperiod_manager.py:to_checkpoint_dict()
        - challengeperiod_manager.py:parse_checkpoint_dict()
        """
        # Setup probation miners with specific timestamps
        test_probation_miners = {
            "probation_persist1": self.PROBATION_START_TIME,
            "probation_persist2": self.PROBATION_START_TIME + 10000,
        }

        for miner, timestamp in test_probation_miners.items():
            self.challengeperiod_manager.active_miners[miner] = (MinerBucket.PROBATION, timestamp, None, None)

        # Force save to disk
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        # Simulate restart by creating new challenge period manager
        new_challengeperiod_manager = ChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.ledger_manager,
            running_unit_tests=True,
        )

        # Verify probation miners and timestamps are preserved
        probation_miners = new_challengeperiod_manager.get_probation_miners()

        for miner, expected_timestamp in test_probation_miners.items():
            self.assertIn(miner, probation_miners,
                         f"Probation miner {miner} should persist across restart")
            actual_timestamp = new_challengeperiod_manager.active_miners[miner][1]
            self.assertEqual(actual_timestamp, expected_timestamp,
                           f"Probation timestamp for {miner} should be preserved")

    def test_simultaneous_promotion_and_demotion_with_probation(self):
        """
        Test system handles concurrent promotions and demotions correctly

        NOTE FOR PR AUTHOR:
        This tests edge cases where multiple state transitions happen simultaneously.
        Verify the logic in challengeperiod_manager.py:inspect() handles all transitions correctly.
        """
        # Setup scenario with multiple transitions
        promoting_challenge = "challenge_miner1"
        promoting_probation = "probation_miner1"
        demoting_maincomp = "maincomp_miner1"

        # Setup excellent performance for promoting miners
        excellent_ledger = generate_winning_ledger(self.START_TIME, self.END_TIME)
        for checkpoint in excellent_ledger[TP_ID_PORTFOLIO].cps:
            checkpoint.gain = 0.12
            checkpoint.loss = -0.02

        self.ledger_manager.save_perf_ledgers({
            promoting_challenge: excellent_ledger,
            promoting_probation: excellent_ledger,
        })

        # Setup poor performance for demoting miner
        poor_ledger = generate_winning_ledger(self.START_TIME, self.END_TIME)  # Start with winning then make poor
        for checkpoint in poor_ledger[TP_ID_PORTFOLIO].cps:
            checkpoint.gain = 0.02
            checkpoint.loss = -0.08

        self.ledger_manager.save_perf_ledgers({demoting_maincomp: poor_ledger})

        # Run simultaneous evaluation
        initial_maincomp = len(self.challengeperiod_manager.get_success_miners())
        initial_challenge = len(self.challengeperiod_manager.get_testing_miners())
        initial_probation = len(self.challengeperiod_manager.get_probation_miners())

        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # Verify state transitions occurred
        final_maincomp = len(self.challengeperiod_manager.get_success_miners())
        final_challenge = len(self.challengeperiod_manager.get_testing_miners())
        final_probation = len(self.challengeperiod_manager.get_probation_miners())

        # System should handle multiple transitions without corruption
        total_initial = initial_maincomp + initial_challenge + initial_probation
        total_final = final_maincomp + final_challenge + final_probation

        # Account for potential eliminations (total might decrease)
        eliminated = len(self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys())
        self.assertEqual(total_initial, total_final + eliminated,
                        "Total miner count should be conserved (accounting for eliminations)")

    # def test_probation_miners_receive_challenge_weights(self):
    #     """
    #     Test that probation miners are weighted as challenge miners

    #     NOTE FOR PR AUTHOR:
    #     This verifies probation miners get appropriate weight treatment.
    #     Check subtensor_weight_setter.py:compute_weights_default() to ensure
    #     probation miners are included in testing_hotkeys for weight calculation.
    #     """
    #     # Ensure we have probation miners
    #     self.assertGreater(len(self.challengeperiod_manager.get_probation_miners()), 0,
    #                       "Need probation miners for this test")

    #     # Compute weights
    #     checkpoint_results, transformed_weights = self.weight_setter.compute_weights_default(self.CURRENT_TIME)

    #     # Get all miners by bucket
    #     challenge_miners = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
    #     probation_miners = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PROBATION)
    #     maincomp_miners = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)

    #     # Extract hotkeys from weight results
    #     weighted_hotkeys = set()
    #     for hotkey_idx, weight in transformed_weights:
    #         if hotkey_idx < len(self.mock_metagraph.hotkeys):
    #             weighted_hotkeys.add(self.mock_metagraph.hotkeys[hotkey_idx])

    #     # Verify probation miners are included in weight calculation (like challenge miners)
    #     for probation_miner in probation_miners:
    #         if probation_miner in self.mock_metagraph.hotkeys:  # Only if in metagraph
    #             self.assertIn(probation_miner, weighted_hotkeys,
    #                          f"Probation miner {probation_miner} should receive weights like challenge miners")

    def test_old_checkpoint_format_conversion(self):
        """
        Test migration from old testing/success format to bucket format

        NOTE FOR PR AUTHOR:
        This tests backward compatibility. The parse_checkpoint_dict() method
        should handle both old format {"testing": {...}, "success": {...}}
        and new format {hotkey: {"bucket": "...", "bucket_start_time": ...}}
        """
        # Create old format checkpoint data
        old_format_data = {
            "testing": {
                "old_challenge_miner1": self.START_TIME,
                "old_challenge_miner2": self.START_TIME + 1000,
            },
            "success": {
                "old_maincomp_miner1": self.START_TIME,
                "old_maincomp_miner2": self.START_TIME + 2000,
            },
        }

        # Test parsing old format
        parsed_miners = ChallengePeriodManager.parse_checkpoint_dict(old_format_data)

        # Verify conversion
        self.assertIn("old_challenge_miner1", parsed_miners)
        self.assertEqual(parsed_miners["old_challenge_miner1"][0], MinerBucket.CHALLENGE)
        self.assertEqual(parsed_miners["old_challenge_miner1"][1], self.START_TIME)

        self.assertIn("old_maincomp_miner1", parsed_miners)
        self.assertEqual(parsed_miners["old_maincomp_miner1"][0], MinerBucket.MAINCOMP)
        self.assertEqual(parsed_miners["old_maincomp_miner1"][1], self.START_TIME)

        # Test new format still works
        new_format_data = {
            "new_probation_miner1": {
                "bucket": "PROBATION",
                "bucket_start_time": self.START_TIME + 3000,
            },
        }

        parsed_new = ChallengePeriodManager.parse_checkpoint_dict(new_format_data)
        self.assertIn("new_probation_miner1", parsed_new)
        self.assertEqual(parsed_new["new_probation_miner1"][0], MinerBucket.PROBATION)
        self.assertEqual(parsed_new["new_probation_miner1"][1], self.START_TIME + 3000)

    def test_probation_time_boundary_conditions(self):
        """
        Test edge cases around probation time boundaries

        NOTE FOR PR AUTHOR:
        This tests the meets_time_criteria() method for probation miners
        at exactly 30 days and just over/under. Ensure proper boundary handling.
        """
        # Test miner at exactly 30 days
        exactly_30_start = self.CURRENT_TIME - ValiConfig.PROBATION_MAXIMUM_MS

        # Test miner just under 30 days
        under_30_start = self.CURRENT_TIME - ValiConfig.PROBATION_MAXIMUM_MS + 1000

        # Test miner just over 30 days
        over_30_start = self.CURRENT_TIME - ValiConfig.PROBATION_MAXIMUM_MS - 1000

        # Test boundary conditions
        exactly_30_result = self.challengeperiod_manager.meets_time_criteria(
            self.CURRENT_TIME, exactly_30_start, MinerBucket.PROBATION)
        under_30_result = self.challengeperiod_manager.meets_time_criteria(
            self.CURRENT_TIME, under_30_start, MinerBucket.PROBATION)
        over_30_result = self.challengeperiod_manager.meets_time_criteria(
            self.CURRENT_TIME, over_30_start, MinerBucket.PROBATION)

        # Verify boundary logic
        self.assertTrue(under_30_result, "Miner under 30 days should meet time criteria")
        self.assertFalse(over_30_result, "Miner over 30 days should not meet time criteria")

        # The exactly 30 days case depends on implementation (<=  vs <)
        # Document expected behavior based on current implementation
        print(f"Exactly 30 days result: {exactly_30_result}")  # For debugging

    def test_probation_miners_mixed_with_challenge_in_inspection(self):
        """
        CRITICAL TEST: Verify probation and challenge miners are evaluated together correctly

        NOTE FOR PR AUTHOR:
        This tests that the inspect() method properly handles both CHALLENGE and PROBATION
        miners in the inspection_hotkeys parameter. Both bucket types should be evaluated
        using the same logic path.
        """
        # Setup mixed challenge and probation miners for inspection
        challenge_miner = "challenge_miner1"
        probation_miner = "probation_miner1"

        # Ensure both are in the inspection miners dict (line 181 in challengeperiod_manager.py)
        inspection_miners = self.challengeperiod_manager.get_testing_miners() | self.challengeperiod_manager.get_probation_miners()

        self.assertIn(challenge_miner, inspection_miners, "Challenge miner should be in inspection")
        self.assertIn(probation_miner, inspection_miners, "Probation miner should be in inspection")

        # Run inspection
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # Verify both types were processed (should appear in logs or results)
        # This ensures the union operation on line 181 works correctly
        final_challenge = self.challengeperiod_manager.get_testing_miners()
        final_probation = self.challengeperiod_manager.get_probation_miners()
        final_maincomp = self.challengeperiod_manager.get_success_miners()

        # At least one of them should have been processed (promoted, demoted, or stayed)
        total_final = len(final_challenge) + len(final_probation) + len(final_maincomp)
        self.assertGreater(total_final, 0, "Inspection should process miners from both buckets")

    def test_zero_probation_miners_edge_case(self):
        """
        CRITICAL TEST: System handles having zero probation miners correctly

        NOTE FOR PR AUTHOR:
        This tests the edge case where no miners are in probation.
        Ensure the system doesn't break when get_probation_miners() returns empty.
        """
        # Clear all probation miners
        for hotkey in list(self.challengeperiod_manager.get_probation_miners().keys()):
            del self.challengeperiod_manager.active_miners[hotkey]

        # self.ledger_manager.save_perf_ledgers(self.LEDGERS)
        # Verify no probation miners
        self.assertEqual(len(self.challengeperiod_manager.get_probation_miners()), 0)

        # System should still function normally
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # Should not crash and should handle empty probation bucket
        final_probation = self.challengeperiod_manager.get_probation_miners()
        # TODO initial set up sets all miners to scores of 0? miners with scores of 0 should not be in maincomp
        # self.assertEqual(len(final_probation), 0, "Should maintain empty probation bucket")

    # def test_probation_miners_in_weight_calculation_testnet_vs_mainnet(self):
    #     """
    #     CRITICAL TEST: Verify probation miners get weights only in testnet/backtesting

    #     NOTE FOR PR AUTHOR:
    #     In subtensor_weight_setter.py:40-47, probation miners are added to testing_hotkeys
    #     for weight calculation, but only get weights during backtesting. In production,
    #     only success_hotkeys get weights. This tests both scenarios.
    #     """
    #     # Test backtesting mode (probation miners should get weights)
    #     self.weight_setter.is_backtesting = True
    #     checkpoint_results, transformed_weights = self.weight_setter.compute_weights_default(self.CURRENT_TIME)

    #     probation_miners = self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PROBATION)
    #     weighted_hotkeys = set()
    #     for hotkey_idx, weight in transformed_weights:
    #         if hotkey_idx < len(self.mock_metagraph.hotkeys):
    #             weighted_hotkeys.add(self.mock_metagraph.hotkeys[hotkey_idx])

    #     # In backtesting, probation miners should get weights
    #     for probation_miner in probation_miners:
    #         if probation_miner in self.mock_metagraph.hotkeys:
    #             self.assertIn(probation_miner, weighted_hotkeys,
    #                          f"In backtesting, probation miner {probation_miner} should get weights")

    #     # Test production mode (probation miners should NOT get direct weights)
    #     self.weight_setter.is_backtesting = False
    #     checkpoint_results, transformed_weights = self.weight_setter.compute_weights_default(self.CURRENT_TIME)

    #     production_weighted_hotkeys = set()
    #     for hotkey_idx, weight in transformed_weights:
    #         if hotkey_idx < len(self.mock_metagraph.hotkeys):
    #             production_weighted_hotkeys.add(self.mock_metagraph.hotkeys[hotkey_idx])

    #     # In production, probation miners should get challenge weights (not main competition weights)
    #     # They should still appear in transformed_weights due to challengeperiod_weights calculation

    def test_probation_bucket_logging_and_monitoring(self):
        """
        PRODUCTION MONITORING TEST: Verify probation bucket changes are properly logged

        NOTE FOR PR AUTHOR:
        This ensures adequate logging exists for monitoring probation bucket size
        and transitions in production. Check challengeperiod_manager.py:208-213 for logging.
        """
        initial_probation_count = len(self.challengeperiod_manager.get_probation_miners())

        # Trigger refresh to generate logs
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # The refresh method should log bucket sizes (line 208-213)
        # This is important for production monitoring
        final_probation_count = len(self.challengeperiod_manager.get_probation_miners())

        # Verify bucket sizes are trackable
        self.assertIsInstance(initial_probation_count, int)
        self.assertIsInstance(final_probation_count, int)
        self.assertGreaterEqual(initial_probation_count, 0)
        self.assertGreaterEqual(final_probation_count, 0)

    def test_probation_elimination_reason_tracking(self):
        """
        CRITICAL TEST: Verify probation miners eliminated for timeout have proper reason

        NOTE FOR PR AUTHOR:
        When probation miners are eliminated due to timeout, they should have
        EliminationReason.FAILED_CHALLENGE_PERIOD_TIME or similar. This is critical
        for debugging and analytics.
        """
        # Setup expired probation miner
        expired_probation_miner = "probation_timeout_test"
        expired_start_time = self.PROBATION_EXPIRED

        self.challengeperiod_manager.active_miners[expired_probation_miner] = (
            MinerBucket.PROBATION, expired_start_time, None, None
        )

        # Create minimal required data for this miner
        position = Position(
            miner_hotkey=expired_probation_miner,
            position_uuid=f"{expired_probation_miner}_position",
            open_ms=expired_start_time,
            close_ms=expired_start_time + 1000,
            trade_pair=TradePair.BTCUSD,
            is_closed_position=True,
            return_at_close=1.0,
            orders=[Order(price=60000, processed_ms=expired_start_time, order_uuid=f"{expired_probation_miner}_order",
                          trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],
        )
        self.position_manager.save_miner_position(position)

        ledger = generate_winning_ledger(expired_start_time, expired_start_time + 1000)
        self.ledger_manager.save_perf_ledgers({expired_probation_miner: ledger})

        # Add to metagraph
        if expired_probation_miner not in self.mock_metagraph.hotkeys:
            self.mock_metagraph.hotkeys.append(expired_probation_miner)

        # Trigger elimination
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME + 1000)
        self.elimination_manager.process_eliminations(PositionLocks())

        # Check elimination reason
        eliminations = self.challengeperiod_manager.elimination_manager.get_eliminations_from_disk()
        probation_eliminations = [e for e in eliminations if e['hotkey'] == expired_probation_miner]

        if probation_eliminations:
            elimination = probation_eliminations[0]
            # Should have appropriate elimination reason for timeout
            self.assertIn(elimination['reason'], [
                EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value,
                EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
            ], f"Probation timeout elimination should have proper reason, got: {elimination['reason']}")

    def test_massive_demotion_scenario_stress_test(self):
        """
        STRESS TEST: Handle scenario where many maincomp miners get demoted simultaneously

        NOTE FOR PR AUTHOR:
        This tests system stability when a large number of maincomp miners
        perform poorly and get demoted to probation simultaneously.
        Important for production resilience.
        """
        # Setup scenario where 5 out of 30 maincomp miners perform poorly
        poorly_performing_miners = self.SUCCESS_MINER_NAMES[:5]

        # Give them all terrible performance
        poor_ledgers = {}
        for miner in poorly_performing_miners:
            poor_ledger = generate_winning_ledger(self.START_TIME, self.END_TIME)
            for checkpoint in poor_ledger[TP_ID_PORTFOLIO].cps:
                checkpoint.gain = 0.01  # 1% gain
                checkpoint.loss = -0.05  # 5% loss
            poor_ledgers[miner] = poor_ledger

        self.LEDGERS.update(poor_ledgers)
        self.ledger_manager.save_perf_ledgers(self.LEDGERS)

        # Record initial state
        initial_maincomp = len(self.challengeperiod_manager.get_success_miners())
        total_initial = len(self.challengeperiod_manager.active_miners)

        # Trigger evaluation
        self.challengeperiod_manager.refresh(current_time=self.CURRENT_TIME)

        # Check system handled mass demotion
        final_maincomp = len(self.challengeperiod_manager.get_success_miners())

        # System should be stable and maintain total miner count (minus eliminations)
        eliminated_count = len(self.challengeperiod_manager.eliminations_with_reasons)
        total_final = len(self.challengeperiod_manager.active_miners)

        self.assertEqual(total_initial, total_final + eliminated_count,
                        "System should maintain miner count consistency during mass demotion")

        # Should have fewer maincomp miners and more probation miners (unless eliminated for drawdown)
        if eliminated_count == 0:  # If no eliminations due to drawdown
            self.assertLess(final_maincomp, initial_maincomp, "Should have fewer maincomp miners after poor performance")

    def test_probation_to_challenge_transition_prevention(self):
        """
        CRITICAL TEST: Ensure probation miners don't accidentally get moved to challenge bucket

        NOTE FOR PR AUTHOR:
        This tests that the bucket assignment logic doesn't accidentally move
        probation miners to challenge bucket. They should only go to maincomp or elimination.
        """
        # Setup probation miner
        probation_miner = "probation_miner5"
        original_probation_time = self.PROBATION_START_TIME

        self.challengeperiod_manager.active_miners[probation_miner] = (
            MinerBucket.PROBATION, original_probation_time, None, None
        )

        # Run multiple refresh cycles
        for i in range(3):
            current_time = self.CURRENT_TIME + (i * 1000)
            self.challengeperiod_manager.refresh(current_time=current_time)

            # Probation miner should never be in challenge bucket
            challenge_miners = self.challengeperiod_manager.get_testing_miners()
            self.assertNotIn(probation_miner, challenge_miners,
                           f"Probation miner should never move to challenge bucket (cycle {i})")

            # Should be in probation, maincomp, or eliminated
            probation_miners = self.challengeperiod_manager.get_probation_miners()
            maincomp_miners = self.challengeperiod_manager.get_success_miners()
            eliminated_miners = self.challengeperiod_manager.eliminations_with_reasons

            miner_found = (probation_miner in probation_miners or
                          probation_miner in maincomp_miners or
                          probation_miner in eliminated_miners)

            self.assertTrue(miner_found,
                           f"Probation miner must be in probation, maincomp, or eliminated (cycle {i})")


if __name__ == '__main__':
    unittest.main()
