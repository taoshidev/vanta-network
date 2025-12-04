# developer: trdougherty, jbonilla
# Copyright © 2024 Taoshi Inc
"""
ChallengePeriod unit tests using the new client/server architecture.

This test file has been refactored to use real server/client infrastructure
instead of mock classes, following the pattern from test_elimination_core.py.
"""
import unittest
from copy import deepcopy

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import generate_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.scoring.scoring import Scoring
from vali_objects.challenge_period import ChallengePeriodManager
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import TP_ID_PORTFOLIO
import vali_objects.vali_config as vali_file


class TestChallengePeriodUnit(TestBase):
    """
    ChallengePeriod unit tests using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    challenge_period_client = None
    plagiarism_client = None
    asset_selection_client = None

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
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')

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
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        # For the positions and ledger creation
        self.START_TIME = 1000
        self.END_TIME = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS - 1

        # For time management
        self.CURRENTLY_IN_CHALLENGE = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS - 1  # Evaluation time when inside the challenge period
        self.OUTSIDE_OF_CHALLENGE = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS + 1  # Evaluation time when the challenge period is over

        DAILY_MS = ValiConfig.DAILY_MS
        # Challenge miners must have a minimum amount of trading days before promotion
        self.MIN_PROMOTION_TIME = self.START_TIME + (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS + 1) * DAILY_MS  # time when miner can now be promoted
        self.BEFORE_PROMOTION_TIME = self.START_TIME + (ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS - 1) * DAILY_MS  # time before miner has enough trading days

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
            position.orders[0] = Order(price=60000, processed_ms=int(position.open_ms), order_uuid="order" + str(i), trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)
            self.DEFAULT_POSITIONS.append(position)

        self.DEFAULT_LEDGER = generate_ledger(
            start_time=self.START_TIME,
            end_time=self.END_TIME,
            gain=0.1,
            loss=-0.08,
            mdd=0.99,
        )

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Initialize metagraph with test miners
        self.metagraph_client.set_hotkeys(self.MINER_NAMES)

        # Set up asset selection for all miners (required for promotion)
        from vali_objects.vali_config import TradePairCategory
        asset_class_str = TradePairCategory.CRYPTO.value
        asset_selection_data = {}
        for hotkey in self.MINER_NAMES + self.SUCCESS_MINER_NAMES:
            asset_selection_data[hotkey] = asset_class_str

        try:
            self.asset_selection_client.sync_miner_asset_selection_data(asset_selection_data)
        except (BrokenPipeError, ConnectionRefusedError, ConnectionError, EOFError) as e:
            import bittensor as bt
            bt.logging.warning(
                f"Failed to sync asset selection in setUp (server may have crashed): {e}. "
                f"Tests requiring asset selection will fail."
            )

        # Note: Individual tests populate active_miners as needed via _populate_active_miners()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def save_and_get_positions(self, base_positions, hotkeys):
        """Helper to save positions and get filtered positions for scoring with error handling."""
        import bittensor as bt

        try:
            for p in base_positions:
                self.position_client.save_miner_position(p)

            positions, hk_to_first_order_time = self.position_client.filtered_positions_for_scoring(
                hotkeys=hotkeys)
            assert positions, positions

            return positions, hk_to_first_order_time

        except (BrokenPipeError, ConnectionRefusedError, ConnectionError, EOFError) as e:
            bt.logging.warning(
                f"save_and_get_positions failed (server may have crashed): {type(e).__name__}: {e}. "
                f"Returning empty results - test will likely fail."
            )
            # Return empty results to allow test to continue (will fail on assertions)
            return {}, {}
        except AssertionError:
            bt.logging.warning(
                f"save_and_get_positions: No positions returned for hotkeys {hotkeys}. "
                f"This may indicate position_client RPC failure."
            )
            # Re-raise to preserve original test failure behavior
            raise

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
        """Populate active miners using RPC client methods with error handling."""
        import bittensor as bt

        try:
            for hotkey in maincomp:
                self.challenge_period_client.set_miner_bucket(hotkey, MinerBucket.MAINCOMP, self.START_TIME)
            for hotkey in challenge:
                self.challenge_period_client.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, self.START_TIME)
            for hotkey in probation:
                self.challenge_period_client.set_miner_bucket(hotkey, MinerBucket.PROBATION, self.START_TIME)

            # Verify miners were actually registered by checking a sample
            sample_hotkeys = (challenge + maincomp + probation)[:3]  # Check first 3
            for hotkey in sample_hotkeys:
                try:
                    bucket = self.challenge_period_client.get_miner_bucket(hotkey)
                    if bucket is None:
                        bt.logging.warning(
                            f"_populate_active_miners: Verification failed - {hotkey} not found after registration. "
                            f"Server may have crashed."
                        )
                        break
                except Exception as e:
                    bt.logging.warning(
                        f"_populate_active_miners: Verification failed for {hotkey}: {e}. "
                        f"Server may have crashed."
                    )
                    break

        except (BrokenPipeError, ConnectionRefusedError, ConnectionError, EOFError) as e:
            bt.logging.warning(
                f"_populate_active_miners failed (server may have crashed): {type(e).__name__}: {e}. "
                f"Tests relying on this data will likely fail."
            )

    def test_screen_drawdown(self):
        """Test that a high drawdown miner is screened"""
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

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
        # Populate success miners for ranking comparison
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

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
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as passing
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

        base_positions = deepcopy(self.DEFAULT_POSITIONS)
        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.CURRENTLY_IN_CHALLENGE

        # Check that the miner is screened as passing
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

        base_positions = []

        inspection_positions = {"miner": base_positions}

        _, hk_to_first_order_time = self.position_client.filtered_positions_for_scoring(
            hotkeys=["miner"])

        inspection_ledger = {}
        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as failing
        passing, demoted, failing = self.challenge_period_client.inspect(
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

    @unittest.skip('Departed hotkeys flow prevents re-registration.')
    def test_recently_re_registered_miner(self):
        """
        Test the scenario where the miner is eliminated and registers again. Simulate this with a stale perf ledger
        The positions begin after the perf ledger start therefore the ledger is stale.
        """
        # Populate success miners for test context
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

        base_ledger = deepcopy(self.DEFAULT_LEDGER)

        base_position = deepcopy(self.DEFAULT_POSITION)
        base_position.orders[0].processed_ms = base_ledger[TP_ID_PORTFOLIO].start_time_ms + 1

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions([base_position], ["miner"])
        inspection_ledger = {"miner": base_ledger}

        inspection_hotkeys = {"miner": self.START_TIME}
        current_time = self.OUTSIDE_OF_CHALLENGE

        # Check that the miner is screened as testing still
        passing, demoted, failing = self.challenge_period_client.inspect(
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

    def test_just_above_threshold(self):
        """Miner ranking just inside PROMOTION_THRESHOLD_RANK should pass"""
        # Populate success miners for ranking comparison
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

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
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate success miners for ranking comparison
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

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
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate success miners for ranking comparison
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

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
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

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
        passing, demoted, failing = self.challenge_period_client.inspect(
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
        # Populate active miners for test
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES, challenge=["miner"])

        base_ledger = deepcopy(self.DEFAULT_LEDGER)
        base_ledger_portfolio = base_ledger[TP_ID_PORTFOLIO]

        base_positions = deepcopy(self.DEFAULT_POSITIONS)

        inspection_positions, hk_to_first_order_time = self.save_and_get_positions(base_positions, ["miner"])
        inspection_ledger = {"miner": base_ledger}

        current_time = self.BEFORE_PROMOTION_TIME

        portfolio_cps = [cp for cp in base_ledger_portfolio.cps if cp.last_update_ms < current_time]
        base_ledger_portfolio.cps = portfolio_cps

        passing, demoted, failing = self.challenge_period_client.inspect(
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

    # ==================== Race Condition Tests ====================
    # These tests demonstrate race conditions in the ChallengePeriod architecture.
    # They are EXPECTED to fail (crash or produce incorrect results) since proper
    # locking is not implemented. Once locks are added, these tests should pass.

    def test_race_iteration_during_modification(self):
        """
        RC-1: Dictionary iteration crash when dict modified during iteration.

        Real pattern: Client calls get_hotkeys_by_bucket() (iterates active_miners)
        while daemon or another client calls set_miner_bucket() (modifies active_miners).

        Expected failure: RuntimeError: dictionary changed size during iteration
        """
        import threading
        import time

        # Setup: Add 100 miners to challenge bucket
        hotkeys = [f"race_miner_{i}" for i in range(100)]
        for hotkey in hotkeys:
            self.challenge_period_client.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, self.START_TIME)

        errors = []
        iterations_completed = [0]

        def iterator_thread():
            """Simulates client calling get_hotkeys_by_bucket repeatedly"""
            try:
                for _ in range(50):
                    # This iterates over active_miners dict
                    challenge_hotkeys = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
                    iterations_completed[0] += 1
                    time.sleep(0.001)  # Small delay to increase race window
            except RuntimeError as e:
                errors.append(("iterator", str(e)))

        def modifier_thread():
            """Simulates daemon/client modifying active_miners concurrently"""
            try:
                for i in range(50):
                    # Add new miners (modifies active_miners dict)
                    new_hotkey = f"concurrent_miner_{i}"
                    self.challenge_period_client.set_miner_bucket(new_hotkey, MinerBucket.CHALLENGE, self.START_TIME)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("modifier", str(e)))

        # Run both threads concurrently (simulates real RPC scenario)
        t1 = threading.Thread(target=iterator_thread)
        t2 = threading.Thread(target=modifier_thread)

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Expected: RuntimeError during iteration
        # Note: This test may not always fail due to timing, but demonstrates the issue
        if errors:
            # If we caught a RuntimeError, the race condition manifested
            runtime_errors = [e for source, e in errors if "dictionary changed size" in e]
            if runtime_errors:
                self.fail(f"Race condition detected: {runtime_errors[0]}")

        # Even if no crash, verify data consistency
        # All 150 miners should be present (100 initial + 50 added by modifier)
        final_challenge_miners = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
        # NOTE: This assertion may fail if concurrent modifications caused data loss
        expected_count = 100 + 50
        actual_count = len(final_challenge_miners)
        self.assertEqual(
            actual_count,
            expected_count,
            f"Data loss detected: expected {expected_count} miners, got {actual_count}"
        )

    def test_race_concurrent_set_miner_bucket(self):
        """
        RC-4: Read-modify-write race in set_miner_bucket().

        Real pattern: Two clients call set_miner_bucket() for same hotkey concurrently.

        Expected failure: Incorrect is_new return value, or last-writer-wins data loss.
        """
        import threading

        hotkey = "race_hotkey"
        results = []

        def client1_set():
            """Simulates client 1 setting miner bucket"""
            is_new = self.challenge_period_client.set_miner_bucket(
                hotkey, MinerBucket.CHALLENGE, 1000
            )
            results.append(("client1", is_new, MinerBucket.CHALLENGE, 1000))

        def client2_set():
            """Simulates client 2 setting same miner to different bucket"""
            is_new = self.challenge_period_client.set_miner_bucket(
                hotkey, MinerBucket.PROBATION, 2000
            )
            results.append(("client2", is_new, MinerBucket.PROBATION, 2000))

        # Run both threads concurrently
        t1 = threading.Thread(target=client1_set)
        t2 = threading.Thread(target=client2_set)

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Verify results
        self.assertEqual(len(results), 2, "Both threads should complete")

        # Expected: Exactly ONE should return is_new=True, other should return False
        # Actual (without lock): BOTH may return True (race condition)
        is_new_count = sum(1 for _, is_new, _, _ in results if is_new)
        if is_new_count != 1:
            self.fail(
                f"Race condition in set_miner_bucket: {is_new_count} threads returned is_new=True, "
                f"expected exactly 1. Results: {results}"
            )

        # Verify final state (last writer wins, but we don't know which)
        final_bucket = self.challenge_period_client.get_miner_bucket(hotkey)
        final_time = self.challenge_period_client.get_miner_start_time(hotkey)

        # Should match ONE of the writers
        client1_won = (final_bucket == MinerBucket.CHALLENGE and final_time == 1000)
        client2_won = (final_bucket == MinerBucket.PROBATION and final_time == 2000)

        self.assertTrue(
            client1_won or client2_won,
            f"Final state inconsistent: bucket={final_bucket}, time={final_time}"
        )

    def test_race_concurrent_file_writes(self):
        """
        RC-3: Concurrent file writes causing corruption.

        Real pattern: Multiple operations trigger _write_challengeperiod_from_memory_to_disk()
        concurrently (e.g., update_miners, remove_eliminated, refresh).

        Expected failure: File corruption, lost updates, or partial writes.
        """
        import threading
        import time

        # Setup: Add some miners
        for i in range(10):
            self.challenge_period_client.set_miner_bucket(
                f"file_race_miner_{i}", MinerBucket.CHALLENGE, self.START_TIME
            )

        errors = []

        def writer_thread_1():
            """Simulates client 1 bulk updating miners (triggers file write)"""
            try:
                miners_dict = {}
                for i in range(10, 20):
                    # Client expects tuples: (bucket, start_time, prev_bucket, prev_time)
                    miners_dict[f"writer1_miner_{i}"] = (
                        MinerBucket.CHALLENGE,
                        self.START_TIME,
                        None,
                        None
                    )
                self.challenge_period_client.update_miners(miners_dict)
                # Explicit file write to increase contention
                self.challenge_period_client._write_challengeperiod_from_memory_to_disk()
            except Exception as e:
                errors.append(("writer1", str(e)))

        def writer_thread_2():
            """Simulates client 2 removing eliminated (triggers file write)"""
            try:
                # Add and remove miners (triggers disk writes)
                for i in range(20, 30):
                    self.challenge_period_client.set_miner_bucket(
                        f"writer2_miner_{i}", MinerBucket.CHALLENGE, self.START_TIME
                    )
                self.challenge_period_client._write_challengeperiod_from_memory_to_disk()
            except Exception as e:
                errors.append(("writer2", str(e)))

        def writer_thread_3():
            """Simulates daemon refresh (triggers file write)"""
            try:
                # Simulate refresh operations
                time.sleep(0.01)  # Stagger slightly
                self.challenge_period_client._write_challengeperiod_from_memory_to_disk()
            except Exception as e:
                errors.append(("writer3", str(e)))

        # Run all threads concurrently
        threads = [
            threading.Thread(target=writer_thread_1),
            threading.Thread(target=writer_thread_2),
            threading.Thread(target=writer_thread_3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Check for errors
        if errors:
            self.fail(f"File write errors occurred: {errors}")

        # Verify data integrity: All miners should be present
        # Note: Without file lock, last writer may overwrite others' changes
        all_hotkeys = self.challenge_period_client.get_all_miner_hotkeys()

        # We expect at least the miners from all three writers
        # writer1: 10 miners (10-19)
        # writer2: 10 miners (20-29)
        # initial: 10 miners (0-9)
        # Total: 30 miners
        expected_min_count = 30
        actual_count = len(all_hotkeys)

        if actual_count < expected_min_count:
            self.fail(
                f"File write race caused data loss: expected at least {expected_min_count} miners, "
                f"got {actual_count}. Missing miners indicate lost file writes."
            )

    def test_race_daemon_refresh_simulation(self):
        """
        RC-2: Daemon refresh() concurrent with RPC modifications.

        Real pattern: Daemon runs refresh() which does multiple iterations over active_miners
        (get_hotkeys_by_bucket for CHALLENGE/MAINCOMP/PROBATION) plus modifications
        (_promote_challengeperiod_in_memory, _eliminate_challengeperiod_in_memory),
        while clients concurrently call set_miner_bucket().

        Expected failure: RuntimeError during daemon's iterations, or data corruption.
        """
        import threading
        import time

        # Setup: Add miners to different buckets
        for i in range(30):
            self.challenge_period_client.set_miner_bucket(
                f"daemon_test_challenge_{i}", MinerBucket.CHALLENGE, self.START_TIME
            )
        for i in range(20):
            self.challenge_period_client.set_miner_bucket(
                f"daemon_test_maincomp_{i}", MinerBucket.MAINCOMP, self.START_TIME
            )
        for i in range(10):
            self.challenge_period_client.set_miner_bucket(
                f"daemon_test_probation_{i}", MinerBucket.PROBATION, self.START_TIME
            )

        errors = []
        daemon_iterations = [0]

        def daemon_refresh_simulation():
            """Simulates daemon's refresh() method access pattern"""
            try:
                for iteration in range(10):
                    # Daemon refresh pattern: multiple get_hotkeys_by_bucket calls
                    challenge_hks = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
                    maincomp_hks = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
                    probation_hks = self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.PROBATION)

                    # Simulate promotions/demotions (modifies active_miners)
                    if challenge_hks:
                        # Promote first challenge miner
                        self.challenge_period_client.set_miner_bucket(
                            challenge_hks[0], MinerBucket.MAINCOMP, self.START_TIME + iteration
                        )

                    daemon_iterations[0] += 1
                    time.sleep(0.01)  # Simulate refresh interval
            except RuntimeError as e:
                errors.append(("daemon", str(e)))
            except Exception as e:
                errors.append(("daemon_other", str(e)))

        def concurrent_client_modifications():
            """Simulates clients making concurrent modifications"""
            try:
                for i in range(50):
                    # Clients add new miners
                    self.challenge_period_client.set_miner_bucket(
                        f"concurrent_client_miner_{i}", MinerBucket.CHALLENGE, self.START_TIME + i
                    )
                    time.sleep(0.005)  # Faster than daemon to increase race probability
            except Exception as e:
                errors.append(("client", str(e)))

        # Run daemon and client threads concurrently (real scenario)
        daemon_thread = threading.Thread(target=daemon_refresh_simulation)
        client_thread = threading.Thread(target=concurrent_client_modifications)

        daemon_thread.start()
        client_thread.start()
        daemon_thread.join(timeout=10)
        client_thread.join(timeout=10)

        # Check for RuntimeError (dictionary changed size during iteration)
        runtime_errors = [e for source, e in errors if "dictionary changed size" in str(e)]
        if runtime_errors:
            self.fail(
                f"Race condition during daemon refresh: {runtime_errors[0]}. "
                f"Daemon completed {daemon_iterations[0]} iterations before crash."
            )

        # Check for other errors
        if errors:
            self.fail(f"Errors during concurrent daemon/client operations: {errors}")

        # Verify data consistency
        all_hotkeys = self.challenge_period_client.get_all_miner_hotkeys()
        # Should have initial miners (60) + client additions (50) - promotions
        # Exact count is hard to predict due to promotions, but should be > 60
        self.assertGreater(
            len(all_hotkeys), 60,
            f"Data loss detected: expected > 60 miners, got {len(all_hotkeys)}"
        )

    def test_race_bulk_update_visibility(self):
        """
        RC-5: Partial visibility during bulk update_miners().

        Real pattern: Client calls update_miners() with 100 miners while another
        client calls get_hotkeys_by_bucket().

        Expected failure: Reader sees partial state (some miners updated, others not).
        """
        import threading

        partial_reads = []

        def bulk_updater():
            """Simulates sync_challenge_period_data with 100 miners"""
            miners_dict = {}
            for i in range(100):
                # Client expects tuples: (bucket, start_time, prev_bucket, prev_time)
                miners_dict[f"bulk_miner_{i}"] = (
                    MinerBucket.CHALLENGE,
                    self.START_TIME,
                    None,
                    None
                )
            # This updates dict one-by-one internally (dict.update is not atomic)
            self.challenge_period_client.update_miners(miners_dict)

        def concurrent_reader():
            """Simulates client reading during bulk update"""
            import time
            for _ in range(20):
                count = len(self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
                partial_reads.append(count)
                time.sleep(0.001)  # Sample frequently to catch partial states

        # Run concurrently
        updater = threading.Thread(target=bulk_updater)
        reader = threading.Thread(target=concurrent_reader)

        updater.start()
        reader.start()
        updater.join(timeout=5)
        reader.join(timeout=5)

        # Analysis: If locking works, we should see 0 or 100 miners, never partial
        # Without locking: We may see partial states (e.g., 0, 23, 67, 100)
        partial_states = [count for count in partial_reads if 0 < count < 100]

        if partial_states:
            self.fail(
                f"Partial visibility during bulk update detected: saw {len(partial_states)} "
                f"intermediate states. Sample values: {partial_states[:5]}. "
                f"All reads: {sorted(set(partial_reads))}"
            )

        # Verify final state
        final_count = len(self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
        self.assertEqual(final_count, 100, "Not all miners were added")
