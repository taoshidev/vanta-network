# developer: trdougherty
import copy
from copy import deepcopy

import numpy as np

from tests.shared_objects.mock_classes import (
    MockChallengePeriodManager,
    MockPositionManager,
)
from shared_objects.mock_metagraph import MockMetagraph
from tests.shared_objects.test_utilities import generate_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
import vali_objects.vali_config as vali_file


class TestChallengePeriodUnit(TestBase):

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )


        # For the positions and ledger creation
        self.START_TIME = 1000
        self.END_TIME = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS - 1

        # For time management
        self.CURRENTLY_IN_CHALLENGE = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS - 1    # Evaluation time when inside the challenge period
        self.OUTSIDE_OF_CHALLENGE = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS + 1  # Evaluation time when the challenge period is over

        DAILY_MS = ValiConfig.DAILY_MS
        # Challenge miners must have a minimum amount of trading days before promotion
        self.MIN_PROMOTION_TIME = self.START_TIME + (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS + 1) * DAILY_MS # time when miner can now be promoted
        self.BEFORE_PROMOTION_TIME = self.START_TIME + (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS - 1) * DAILY_MS # time before miner has enough trading days

        # Number of positions
        self.N_POSITIONS_BOUNDS = 20 + 1
        self.N_POSITIONS = self.N_POSITIONS_BOUNDS - 1

        self.EVEN_TIME_DISTRIBUTION = [self.START_TIME + (self.END_TIME - self.START_TIME) * i / self.N_POSITIONS_BOUNDS
                                       for i
                                       in range(self.N_POSITIONS_BOUNDS)]

        self.MINER_NAMES = [f"miner{i}" for i in range(self.N_POSITIONS)] + ["miner"]
        self.SUCCESS_MINER_NAMES = [f"miner{i}" for i in range(1, 26)]
        self.DEFAULT_POSITION = Position(
            miner_hotkey="miner",
            position_uuid="miner",
            orders=[Order(price=60000, processed_ms=self.START_TIME, order_uuid="initial_order", trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],
            net_leverage=0.0,
            open_ms=self.START_TIME,
            close_ms=self.END_TIME,
            trade_pair=TradePair.BTCUSD,
            is_closed_position=True,
            return_at_close=1.0,
        )

        # Generate a positions list with N_POSITIONS positions
        self.DEFAULT_POSITIONS = []
        for i in range(self.N_POSITIONS):
            position = deepcopy(self.DEFAULT_POSITION)
            position.open_ms = int(self.EVEN_TIME_DISTRIBUTION[i])
            position.close_ms = int(self.EVEN_TIME_DISTRIBUTION[i + 1])
            position.is_closed_position = True
            position.position_uuid += str(i)
            position.return_at_close = 1.0
            position.orders[0] = Order(price=60000, processed_ms=int(position.open_ms), order_uuid="order" + str(i), trade_pair=TradePair.BTCUSD,  order_type=OrderType.LONG, leverage=0.1)
            self.DEFAULT_POSITIONS.append(position)

        self.DEFAULT_LEDGER = generate_ledger(
            start_time=self.START_TIME,
            end_time=self.END_TIME,
            gain=0.1,
            loss=-0.08,
            mdd=0.99,
        )

        # Initialize system components
        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)

        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)

        self.position_manager = MockPositionManager(self.mock_metagraph,
                                                    perf_ledger_manager=None,
                                                    elimination_manager=self.elimination_manager)
        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.plagiarism_manager = PlagiarismManager(None, running_unit_tests=True)
        self.challengeperiod_manager = MockChallengePeriodManager(self.mock_metagraph, position_manager=self.position_manager, contract_manager=self.contract_manager, plagiarism_manager=self.plagiarism_manager)
        self.ledger_manager = self.challengeperiod_manager.perf_ledger_manager
        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

        self.position_manager.clear_all_miner_positions()

        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=["miner"])

    def save_and_get_positions(self, base_positions, hotkeys):

        for p in base_positions:
            self.position_manager.save_miner_position(p)

        positions, hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(
            hotkeys=hotkeys)
        assert positions, positions

        return positions, hk_to_first_order_time

    def get_combined_scores_dict(self, miner_scores: dict[str, float], asset_class=None):
        """
        Create a combined scores dict for testing.

        Args:
            miner_scores: dict mapping hotkey to score (0.0 to 1.0)
            asset_class: TradePairCategory, defaults to CRYPTO

        Returns:
            combined_scores_dict in the format expected by inspect()
        """
        if asset_class is None:
            asset_class = vali_file.TradePairCategory.CRYPTO

        combined_scores_dict = {asset_class: {"metrics": {}, "penalties": {}}}
        asset_class_dict = combined_scores_dict[asset_class]

        # Create scores for each metric
        for config_name, config in Scoring.scoring_config.items():
            scores_list = [(hotkey, score) for hotkey, score in miner_scores.items()]
            asset_class_dict["metrics"][config_name] = {
                'scores': scores_list,
                'weight': config['weight']
            }

        # All miners get penalty multiplier of 1 (no penalty)
        asset_class_dict["penalties"] = {hotkey: 1.0 for hotkey in miner_scores.keys()}

        return combined_scores_dict

    def _populate_active_miners(self, *, maincomp=[], challenge=[], probation=[]):
        miners = {}
        for hotkey in maincomp:
            miners[hotkey] = (MinerBucket.MAINCOMP, self.START_TIME, None, None)
        for hotkey in challenge:
            miners[hotkey] = (MinerBucket.CHALLENGE, self.START_TIME, None, None)
        for hotkey in probation:
            miners[hotkey] = (MinerBucket.PROBATION, self.START_TIME, None, None)
        self.challengeperiod_manager.active_miners = miners

    def test_screen_drawdown(self):
        """Test that a high drawdown miner is screened"""
        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        # Returns are strong and consistent
        for i in range(self.N_POSITIONS):
            base_positions[i].return_at_close = 1.1

        # Drawdown is high - 50% drawdown on the first period
        base_ledger[TP_ID_PORTFOLIO].cps[0].mdd = 0.5

        # Drawdown criteria
        max_drawdown = LedgerUtils.instantaneous_max_drawdown(base_ledger[TP_ID_PORTFOLIO])
        max_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)
        self.assertGreater(max_drawdown_percentage, ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE)

        # Check that the miner is successfully screened as failing
        screening_logic, _ = LedgerUtils.is_beyond_max_drawdown(ledger_element=base_ledger[TP_ID_PORTFOLIO])
        self.assertTrue(screening_logic)


    # ------ Time Constrained Tests (Inspect) ------
    def test_failing_remaining_time(self):
        """Miner is not passing, but there is time remaining"""
        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_ledger = {"miner": base_ledger}
        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])

        # Create combined scores dict where miner ranks below PROMOTION_THRESHOLD_RANK (25)
        # Miner gets low score (0.1), success miners fill top 25 ranks with higher scores
        miner_scores = {"miner": 0.1}
        for i in range(ValiConfig.PROMOTION_THRESHOLD_RANK):
            if i < len(self.SUCCESS_MINER_NAMES):
                # Top 25 success miners get scores from 1.0 down to 0.76 (25 miners)
                miner_scores[self.SUCCESS_MINER_NAMES[i]] = 1.0 - (i * 0.01)

        combined_scores_dict = self.get_combined_scores_dict(miner_scores)

        # Check that the miner continues in challenge (time remaining, so not eliminated)
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES[:ValiConfig.PROMOTION_THRESHOLD_RANK],
            probation_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
            combined_scores_dict=combined_scores_dict,
        )
        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))

    def test_failing_no_remaining_time(self):
        """Miner is not passing, and there is no time remaining"""

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            probation_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", list(failing.keys()))

    def test_passing_remaining_time(self):
        """Miner is passing and there is remaining time - they should be promoted"""

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as passing
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            probation_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))

    def test_passing_no_remaining_time(self):
        """Redemption, if they pass right before the challenge period ends and before the next evaluation cycle"""

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as passing
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            probation_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))

    def test_lingering_no_positions(self):
        """Test the scenario where the miner has no positions and has been in the system for a while"""
        base_positions = []

        inspection_positions = {"miner": base_positions}

        _, hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(
            hotkeys=["miner"])

        inspection_ledger = {}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            probation_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertNotIn("miner", passing)
        self.assertIn("miner", list(failing.keys()))

    def test_recently_re_registered_miner(self):
        """
        Test the scenario where the miner is eliminated and registers again. Simulate this with a stale perf ledger
        The positions begin after the perf ledger start therefore the ledger is stale.
        """

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        base_position = deepcopy(self.DEFAULT_POSITION)
        base_position.orders[0].processed_ms = base_ledger[TP_ID_PORTFOLIO].start_time_ms + 1

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions([base_position], ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as testing still
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES,
            probation_hotkeys=[],
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))

    # def test_lingering_with_positions(self):
    #     """Test the scenario where the miner has positions and has been in the system for a while"""
    #     base_positions = deepcopy(self.DEFAULT_POSITIONS)
    #
    #     # Removed requirement of more than one position since it isn't required for dynamic challenge period
    #     base_positions = [base_positions[0]]  # Only one position
    #
    #     base_ledger = deepcopy(self.DEFAULT_LEDGER)
    #
    #     inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
    #     inspection_ledger = {"miner": base_ledger}
    #
    #     inspection_hotkeys = {"miner": self.START_TIME}
    #     current_time = self.OUTSIDE_OF_CHALLENGE
    #
    #     # Check that the miner is screened as testing still
    #     passing, demoted, failing = self.challengeperiod_manager.inspect(
    #         positions=inspection_positions,
    #         ledger=inspection_ledger,
    #         success_hotkeys=self.SUCCESS_MINER_NAMES,
    #         inspection_hotkeys=inspection_hotkeys,
    #         current_time=current_time,
    #         success_scores_dict=self.success_scores_dict,
    #         hk_to_first_order_time=hk_to_first_order_time
    #     )
    #
    #     self.assertNotIn("miner", passing)
    #     self.assertIn("miner", list(failing.keys()))

    def test_just_above_threshold(self):
        """Miner ranking just inside PROMOTION_THRESHOLD_RANK should pass"""

        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        # Create scores where miner ranks at position 24 (within top 25)
        # 23 success miners score higher, miner at 0.77, and 2 success miners score lower
        miner_scores = {}
        for i in range(23):
            if i < len(self.SUCCESS_MINER_NAMES):
                miner_scores[self.SUCCESS_MINER_NAMES[i]] = 1.0 - (i * 0.01)

        miner_scores["miner"] = 0.77  # Rank 24

        # Add 2 more success miners with lower scores who will be demoted
        miner_scores[self.SUCCESS_MINER_NAMES[23]] = 0.76
        miner_scores[self.SUCCESS_MINER_NAMES[24]] = 0.75

        combined_scores_dict = self.get_combined_scores_dict(miner_scores)

        # Check that the miner is promoted (in top 25)
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES[:25],
            probation_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
            combined_scores_dict=combined_scores_dict,
        )
        self.assertIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))
        # miner25 (index 24) should be demoted as they're now rank 26
        self.assertIn(self.SUCCESS_MINER_NAMES[24], demoted)

    def test_just_below_threshold(self):
        """Miner ranking just outside PROMOTION_THRESHOLD_RANK should not be promoted"""

        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        # Create scores where miner ranks at position 26 (just outside top 25)
        # 25 success miners score higher than the test miner
        miner_scores = {}
        for i in range(ValiConfig.PROMOTION_THRESHOLD_RANK):
            if i < len(self.SUCCESS_MINER_NAMES):
                miner_scores[self.SUCCESS_MINER_NAMES[i]] = 1.0 - (i * 0.01)

        miner_scores["miner"] = 0.74  # Rank 26 (just below rank 25's score of 0.76)

        combined_scores_dict = self.get_combined_scores_dict(miner_scores)

        # Check that the miner continues in challenge (not promoted, not eliminated)
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES[:ValiConfig.PROMOTION_THRESHOLD_RANK],
            probation_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
            combined_scores_dict=combined_scores_dict,
        )
        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))

    def test_at_threshold(self):
        """Miner ranking exactly at PROMOTION_THRESHOLD_RANK (rank 25) should pass"""

        current_time = self.CURRENTLY_IN_CHALLENGE

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        # Create scores where miner ranks exactly at position 25 (the threshold)
        # 24 success miners score higher, miner ties with rank 25 at 0.76, 1 miner scores lower
        miner_scores = {}
        for i in range(24):
            if i < len(self.SUCCESS_MINER_NAMES):
                miner_scores[self.SUCCESS_MINER_NAMES[i]] = 1.0 - (i * 0.01)

        miner_scores["miner"] = 0.76  # Ties for rank 25
        miner_scores[self.SUCCESS_MINER_NAMES[24]] = 0.75  # Rank 26, will be demoted

        combined_scores_dict = self.get_combined_scores_dict(miner_scores)

        # Check that the miner is promoted (at threshold rank 25)
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=self.SUCCESS_MINER_NAMES[:25],
            probation_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
            combined_scores_dict=combined_scores_dict,
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))
        # Verify the 26th ranked miner gets demoted
        self.assertIn(self.SUCCESS_MINER_NAMES[24], demoted)

    def test_screen_minimum_interaction(self):
        """
        Miner with passing score and enough trading days should be promoted
        Also includes tests for base cases
        """
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        base_ledger_portfolio = base_ledger[TP_ID_PORTFOLIO]
        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        # Return True if there are enough trading days
        self.assertEqual(ChallengePeriodManager.screen_minimum_interaction(base_ledger_portfolio), True)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        current_time = self.MIN_PROMOTION_TIME

        portfolio_cps = [cp for cp in base_ledger_portfolio.cps if cp.last_update_ms < current_time]
        base_ledger_portfolio.cps = portfolio_cps

        # Check that miner with a passing score passes when they have enough trading days
        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            probation_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))

        # Check two base cases

        base_ledger_portfolio.cps = []
        # Return False if there are no checkpoints
        self.assertEqual(ChallengePeriodManager.screen_minimum_interaction(base_ledger_portfolio), False)

        # Return False if ledger is none
        self.assertEqual(ChallengePeriodManager.screen_minimum_interaction(None), False)

    def test_not_enough_days(self):
        """A miner with a passing score but not enough trading days shouldn't be promoted"""
        base_ledger = deepcopy(self.DEFAULT_LEDGER)
        base_ledger_portfolio = base_ledger[TP_ID_PORTFOLIO]

        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        current_time = self.BEFORE_PROMOTION_TIME

        portfolio_cps = [cp for cp in base_ledger_portfolio.cps if cp.last_update_ms < current_time]
        base_ledger_portfolio.cps = portfolio_cps

        passing, demoted, failing = self.challengeperiod_manager.inspect(
            positions=inspection_positions,
            ledger=inspection_ledger,
            success_hotkeys=[],
            probation_hotkeys=[],
            inspection_hotkeys={"miner": current_time},
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
        )

        self.assertNotIn("miner", passing)
        self.assertNotIn("miner", list(failing.keys()))
