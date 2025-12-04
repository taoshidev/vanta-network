# developer: trdougherty, jbonilla
# Copyright © 2024 Taoshi Inc
"""
Integration tests for challenge period management using server/client architecture.
Tests end-to-end challenge period scenarios with real server infrastructure.
"""
from copy import deepcopy
import bittensor as bt

from time_util.time_util import TimeUtil
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import (
    TP_ID_PORTFOLIO,
    PerfLedger,
)


class TestChallengePeriodIntegration(TestBase):
    """
    Integration tests for challenge period management using ServerOrchestrator.

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
    challenge_period_handle = None  # Keep handle for daemon control in tests
    plagiarism_client = None
    asset_selection_client = None

    # Class-level constants
    DEFAULT_MINER_HOTKEY = "test_miner"

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        # This starts servers once and shares them across ALL test classes
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

        # Get challenge period server handle for daemon control in tests
        # (tests manually start/stop daemon as needed)
        cls.challenge_period_handle = cls.orchestrator._servers.get('challenge_period')

        # NOTE: Daemon is NOT started in setUpClass - tests start it manually when needed
        # This prevents daemon refresh() from interfering with test state

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

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Create fresh test data
        self._create_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        self.N_MAINCOMP_MINERS = 30
        self.N_CHALLENGE_MINERS = 5
        self.N_ELIMINATED_MINERS = 5
        self.N_PROBATION_MINERS = 5

        # Time configurations
        self.START_TIME = 1000
        self.END_TIME = self.START_TIME + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS - 1

        # For time management
        self.OUTSIDE_OF_CHALLENGE = self.START_TIME + (2 * ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS) + 1  # Evaluation time when the challenge period is over

        self.N_POSITIONS = 55

        self.EVEN_TIME_DISTRIBUTION = [
            int(self.START_TIME + (self.END_TIME - self.START_TIME) * i / self.N_POSITIONS)
            for i in range(self.N_POSITIONS+1)
        ]

        # Define miner categories
        self.SUCCESS_MINER_NAMES = [f"maincomp_miner{i}" for i in range(1, self.N_MAINCOMP_MINERS+1)]
        self.PROBATION_MINER_NAMES = [f"probation_miner{i}" for i in range(1, self.N_PROBATION_MINERS+1)]
        self.TESTING_MINER_NAMES = [f"challenge_miner{i}" for i in range(1, self.N_CHALLENGE_MINERS+1)]
        self.FAILING_MINER_NAMES = [f"eliminated_miner{i}" for i in range(1, self.N_ELIMINATED_MINERS+1)]

        self.NOT_FAILING_MINER_NAMES = self.SUCCESS_MINER_NAMES + self.TESTING_MINER_NAMES + self.PROBATION_MINER_NAMES
        self.NOT_MAIN_COMP_MINER_NAMES = self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES + self.PROBATION_MINER_NAMES
        self.MINER_NAMES = self.NOT_FAILING_MINER_NAMES + self.FAILING_MINER_NAMES

        # Default characteristics
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_CLOSE_MS = 2000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_ACCOUNT_SIZE = 100_000

        # Default positions
        self.DEFAULT_POSITION = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            close_ms=self.DEFAULT_CLOSE_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            is_closed_position=True,
            return_at_close=1.00,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[Order(price=60000, processed_ms=self.START_TIME, order_uuid="initial_order",
                          trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],

        )

        # Generate a positions list with N_POSITIONS positions
        self.DEFAULT_POSITIONS = []
        for i in range(self.N_POSITIONS):
            position = deepcopy(self.DEFAULT_POSITION)
            position.open_ms = self.EVEN_TIME_DISTRIBUTION[i]
            position.close_ms = self.EVEN_TIME_DISTRIBUTION[i + 1]
            position.position_uuid = f"test_position_{i}"
            position.is_closed_position = True
            position.return_at_close = 1.0
            position.orders[0] = Order(price=60000, processed_ms=int(position.open_ms), order_uuid="order" + str(i),
                                       trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)

            self.DEFAULT_POSITIONS.append(position)



        # Testing information
        self.TESTING_INFORMATION = {x: self.START_TIME for x in self.MINER_NAMES}

        # Set up metagraph with all miner names
        self.metagraph_client.set_hotkeys(self.MINER_NAMES)

        # Set up asset selection for all miners (required for promotion)
        from vali_objects.vali_config import TradePairCategory
        asset_class_str = TradePairCategory.CRYPTO.value
        asset_selection_data = {}
        for hotkey in self.MINER_NAMES:
            asset_selection_data[hotkey] = asset_class_str
        self.asset_selection_client.sync_miner_asset_selection_data(asset_selection_data)

        # Build base ledgers and positions
        self.LEDGERS = {}

        # Build base positions
        self.HK_TO_OPEN_MS = {}

        self.POSITIONS = {}
        for i, miner in enumerate(self.MINER_NAMES):
            positions = deepcopy(self.DEFAULT_POSITIONS)
            i_cutoff = i

            positions = positions[i_cutoff:]
            for position in positions:
                position.miner_hotkey = miner
                self.HK_TO_OPEN_MS[miner] = position.open_ms if miner not in self.HK_TO_OPEN_MS else min(self.HK_TO_OPEN_MS[miner], position.open_ms)

            if miner in self.FAILING_MINER_NAMES:
                ledger = generate_losing_ledger(self.HK_TO_OPEN_MS[miner], self.END_TIME)
            elif miner in self.NOT_FAILING_MINER_NAMES:
                ledger = generate_winning_ledger(self.HK_TO_OPEN_MS[miner], self.END_TIME)

            self.LEDGERS[miner] = ledger
            self.POSITIONS[miner] = positions

        self.perf_ledger_client.save_perf_ledgers(self.LEDGERS)
        self.perf_ledger_client.re_init_perf_ledger_data()  # Force reload after clear+save

        n_perf_ledgers_saved_disk = len(self.perf_ledger_client.get_perf_ledgers(from_disk=True))
        n_perf_ledgers_saved_memory = len(self.perf_ledger_client.get_perf_ledgers(from_disk=False))
        assert n_perf_ledgers_saved_disk == len(self.MINER_NAMES), (n_perf_ledgers_saved_disk, self.LEDGERS, self.MINER_NAMES)
        assert n_perf_ledgers_saved_memory == len(self.MINER_NAMES), (n_perf_ledgers_saved_memory, self.LEDGERS, self.MINER_NAMES)

        for miner, positions in self.POSITIONS.items():
            for position in positions:
                position.position_uuid = f"{miner}_position_{position.open_ms}_{position.close_ms}"
                self.position_client.save_miner_position(position)

        self.max_open_ms = max(self.HK_TO_OPEN_MS.values())

        # Finally update the challenge period to default state
        self.elimination_client.clear_eliminations()

        # Populate initial buckets (FAILING miners are NOT included initially - tests add them if needed)
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=self.TESTING_MINER_NAMES,
                                     probation=self.PROBATION_MINER_NAMES)

    def _populate_active_miners(self, *, maincomp=[], challenge=[], probation=[]):
        miners = {}
        for hotkey in maincomp:
            miners[hotkey] = (MinerBucket.MAINCOMP, self.HK_TO_OPEN_MS[hotkey], None, None)
        for hotkey in challenge:
            miners[hotkey] = (MinerBucket.CHALLENGE, self.HK_TO_OPEN_MS[hotkey], None, None)
        for hotkey in probation:
            miners[hotkey] = (MinerBucket.PROBATION, self.HK_TO_OPEN_MS[hotkey], None, None)
        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners(miners)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()  # Ensure disk matches memory

    def test_refresh_populations(self):
        # Add failing miners to challenge bucket so they can be evaluated
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES,
                                     probation=self.PROBATION_MINER_NAMES)

        # Force-allow refresh by resetting last update time
        self.challenge_period_client.set_last_update_time(0)
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()
        testing_length = len(self.challenge_period_client.get_testing_miners())
        success_length = len(self.challenge_period_client.get_success_miners())
        probation_length = len(self.challenge_period_client.get_probation_miners())
        eliminations_length = len(self.elimination_client.get_eliminations_from_memory())

        # Ensure that all miners that aren't failing end up in testing, success, or probation
        self.assertEqual(testing_length + success_length + probation_length, len(self.NOT_FAILING_MINER_NAMES))
        self.assertEqual(eliminations_length, len(self.FAILING_MINER_NAMES))

    def test_full_refresh(self):
        self.assertEqual(len(self.challenge_period_client.get_testing_miners()), len(self.TESTING_MINER_NAMES))
        self.assertEqual(len(self.challenge_period_client.get_success_miners()), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.elimination_client.get_eliminations_from_memory()), 0)

        inspection_hotkeys = self.challenge_period_client.get_testing_miners()

        for hotkey, inspection_time in inspection_hotkeys.items():
            time_criteria = self.max_open_ms - inspection_time <= ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
            self.assertTrue(time_criteria, f"Time criteria failed for {hotkey}")

        # Add failing miners to challenge bucket so they can be evaluated
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES,
                                     probation=self.PROBATION_MINER_NAMES)

        # Force-allow refresh by resetting last update time
        self.challenge_period_client.set_last_update_time(0)
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()

        elimination_hotkeys_memory = [x['hotkey'] for x in self.elimination_client.get_eliminations_from_memory()]
        elimination_hotkeys_disk = [x['hotkey'] for x in self.elimination_client.get_eliminations_from_disk()]

        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, elimination_hotkeys_memory)
            self.assertIn(miner, elimination_hotkeys_disk)

        for miner in self.SUCCESS_MINER_NAMES:
            self.assertNotIn(miner, elimination_hotkeys_memory)
            self.assertNotIn(miner, elimination_hotkeys_disk)

        for miner in self.TESTING_MINER_NAMES:
            self.assertNotIn(miner, elimination_hotkeys_memory)
            self.assertNotIn(miner, elimination_hotkeys_disk)

    def test_failing_mechanics(self):
        # Add all the challenge period miners
        self.assertListEqual(sorted(self.MINER_NAMES), sorted(self.metagraph_client.get_hotkeys()))
        self.assertListEqual(sorted(self.TESTING_MINER_NAMES), sorted(list(self.challenge_period_client.get_testing_miners().keys())))

        # Let's check the initial state of the challenge period
        self.assertEqual(len(self.challenge_period_client.get_success_miners()), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.elimination_client.get_eliminations_from_memory()), 0)
        self.assertEqual(len(self.challenge_period_client.get_testing_miners()), len(self.TESTING_MINER_NAMES))

        eliminations = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(eliminations), 0)

        self.challenge_period_client.remove_eliminated(eliminations=eliminations)
        self.assertEqual(len(self.elimination_client.get_eliminations_from_memory()), 0)

        self.assertEqual(len(self.challenge_period_client.get_testing_miners()), len(self.TESTING_MINER_NAMES))

        self.challenge_period_client.add_challenge_period_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph_client.get_hotkeys(),
            eliminations=self.elimination_client.get_eliminations_from_memory(),
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )

        self.assertEqual(len(self.challenge_period_client.get_testing_miners()), len(self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES))

        current_time = self.max_open_ms + ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS*ValiConfig.DAILY_MS + 1

        self.challenge_period_client.refresh(current_time)
        self.elimination_client.process_eliminations()

        elimination_keys = self.elimination_client.get_eliminated_hotkeys()

        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, self.metagraph_client.get_hotkeys())
            self.assertIn(miner, elimination_keys)

        eliminations = self.elimination_client.get_eliminated_hotkeys()

        self.assertListEqual(sorted(list(eliminations)), sorted(elimination_keys))

    def test_single_position_no_ledger(self):
        # Cleanup all positions first
        self.position_client.clear_all_miner_positions_and_disk()
        self.perf_ledger_client.clear_all_ledger_data()

        self.challenge_period_client._clear_challengeperiod_in_memory_and_disk()
        self.elimination_client.clear_eliminations()

        position = deepcopy(self.DEFAULT_POSITION)
        position.is_closed_position = False
        position.close_ms = None

        self.position_client.save_miner_position(position)
        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners({self.DEFAULT_MINER_HOTKEY: (MinerBucket.CHALLENGE, self.DEFAULT_OPEN_MS, None, None)})
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

        # Now loading the data
        positions = self.position_client.get_positions_for_hotkeys(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        ledgers = self.perf_ledger_client.get_perf_ledgers(from_disk=True)
        ledgers_memory = self.perf_ledger_client.get_perf_ledgers(from_disk=False)
        self.assertEqual(ledgers, ledgers_memory)

        # First check that there is nothing on the miner
        self.assertEqual(ledgers.get(self.DEFAULT_MINER_HOTKEY, PerfLedger().cps), PerfLedger().cps)
        self.assertEqual(ledgers.get(self.DEFAULT_MINER_HOTKEY, len(PerfLedger().cps)), 0)

        # Check the failing criteria initially
        ledger = ledgers.get(self.DEFAULT_MINER_HOTKEY)
        failing_criteria, _ = LedgerUtils.is_beyond_max_drawdown(
            ledger_element=ledger[TP_ID_PORTFOLIO] if ledger else None,
        )

        self.assertFalse(failing_criteria)

        # Now check the inspect to see where the key went
        challenge_success, challenge_demote, challenge_eliminations = self.challenge_period_client.inspect(
            positions=positions,
            ledger=ledgers,
            success_hotkeys=self.SUCCESS_MINER_NAMES,
            probation_hotkeys=self.PROBATION_MINER_NAMES,
            inspection_hotkeys=self.challenge_period_client.get_testing_miners(),
            current_time=self.max_open_ms,
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
        )
        self.elimination_client.process_eliminations()

        # There should be no promotion or demotion
        self.assertListEqual(challenge_success, [])
        self.assertDictEqual(challenge_eliminations, {})


    def test_promote_testing_miner(self):
        # Add all the challenge period miners
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()

        testing_hotkeys = list(self.challenge_period_client.get_testing_miners().keys())
        success_hotkeys = list(self.challenge_period_client.get_success_miners().keys())

        self.assertIn(self.TESTING_MINER_NAMES[0], testing_hotkeys)
        self.assertNotIn(self.TESTING_MINER_NAMES[0], success_hotkeys)

        self.challenge_period_client.promote_challengeperiod_in_memory(
            hotkeys=[self.TESTING_MINER_NAMES[0]],
            current_time=self.max_open_ms,
        )

        testing_hotkeys = list(self.challenge_period_client.get_testing_miners().keys())
        success_hotkeys = list(self.challenge_period_client.get_success_miners().keys())

        self.assertNotIn(self.TESTING_MINER_NAMES[0], testing_hotkeys)
        self.assertIn(self.TESTING_MINER_NAMES[0], success_hotkeys)

        # Check that the timestamp of the success is the current time of evaluation
        self.assertEqual(
            self.challenge_period_client.get_miner_start_time(self.TESTING_MINER_NAMES[0]),
            self.max_open_ms,
        )

    def test_refresh_elimination_disk(self):
        '''
        Note: The original test attempted to eliminate miners one by one by
        incrementally increasing the time of refresh. However, this did not
        work as expected because the to-be-eliminated minerse already exceeded
        max drawdown limit, causing them to be eliminated immediately on the
        first refresh. Setting it to eliminated_miner1's challenge period
        deadline behaves as intended.
        '''
        self.assertTrue(len(self.challenge_period_client.get_testing_miners()) == len(self.TESTING_MINER_NAMES))
        self.assertTrue(len(self.challenge_period_client.get_success_miners()) == len(self.SUCCESS_MINER_NAMES))
        self.assertTrue(len(self.elimination_client.get_eliminations_from_memory()) == 0)

        # Check the failing miners, to see if they are screened
        for miner in self.FAILING_MINER_NAMES:
            failing_screen, _ = LedgerUtils.is_beyond_max_drawdown(
                ledger_element=self.LEDGERS[miner][TP_ID_PORTFOLIO],
            )
            self.assertEqual(failing_screen, True)

        for miner in self.NOT_FAILING_MINER_NAMES:
            failing_screen, _ = LedgerUtils.is_beyond_max_drawdown(
                ledger_element=self.LEDGERS[miner][TP_ID_PORTFOLIO],
            )
            self.assertEqual(failing_screen, False)

        # Add failing miners to challenge bucket so they can be evaluated
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES,
                                     probation=self.PROBATION_MINER_NAMES)

        refresh_time = self.HK_TO_OPEN_MS['eliminated_miner1'] + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS + 1
        # Force-allow refresh by resetting last update time
        self.challenge_period_client.set_last_update_time(0)
        self.challenge_period_client.refresh(refresh_time)

        elimination_reasons = self.challenge_period_client.get_all_elimination_reasons()
        self.assertEqual(elimination_reasons['eliminated_miner1'][0],
                         EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value)

        self.elimination_client.process_eliminations()

        challenge_success = list(self.challenge_period_client.get_success_miners())
        elimininations = list(self.elimination_client.get_eliminated_hotkeys())
        challenge_testing = list(self.challenge_period_client.get_testing_miners())

        self.assertTrue(len(self.elimination_client.get_eliminations_from_memory()) > 0)
        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, elimininations)
            self.assertNotIn(miner, challenge_testing)
            self.assertNotIn(miner, challenge_success)

    def test_no_positions_miner_filtered(self):
        for hotkey in self.challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE):
            self.challenge_period_client.remove_miner(hotkey)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

        self.assertEqual(len(self.challenge_period_client.get_success_miners()), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.elimination_client.get_eliminated_hotkeys()), 0)
        self.assertEqual(len(self.challenge_period_client.get_testing_miners()), 0)

        # Now going to remove the positions of the miners
        miners_without_positions = self.TESTING_MINER_NAMES[:2]
        # Redeploy the positions
        for miner, positions in self.POSITIONS.items():
            if miner in miners_without_positions:
                for position in positions:
                    self.position_client.delete_position(position.miner_hotkey, position.position_uuid)

        current_time = self.max_open_ms
        self.assertEqual(len(self.challenge_period_client.get_testing_miners()), 0)
        self.challenge_period_client.refresh(current_time=current_time)
        self.elimination_client.process_eliminations()

        for miner in miners_without_positions:
            self.assertIn(miner, self.metagraph_client.get_hotkeys())
            self.assertEqual(current_time, self.challenge_period_client.get_testing_miners()[miner])
            # self.assertNotIn(miner, self.challengeperiod_client.get_testing_miners()) # Miners without positions are not necessarily eliminated
            self.assertNotIn(miner, self.challenge_period_client.get_success_miners())

    def test_disjoint_testing_success(self):
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()

        testing_set = set(self.challenge_period_client.get_testing_miners().keys())
        success_set = set(self.challenge_period_client.get_success_miners().keys())

        self.assertTrue(testing_set.isdisjoint(success_set))

    def test_addition(self):
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()

        self.challenge_period_client.add_challenge_period_testing_in_memory_and_disk(
            new_hotkeys=self.MINER_NAMES,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )

        testing_set = set(self.challenge_period_client.get_testing_miners().keys())
        success_set = set(self.challenge_period_client.get_success_miners().keys())

        self.assertTrue(testing_set.isdisjoint(success_set))

    def test_add_miner_no_positions(self):
        self.challenge_period_client.clear_all_miners()

        # Check if it still stores the miners with no perf ledger
        self.perf_ledger_client.clear_all_ledger_data()

        new_miners = ["miner_no_positions1", "miner_no_positions2"]

        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()
        self.challenge_period_client.add_challenge_period_testing_in_memory_and_disk(
            new_hotkeys=new_miners,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )
        self.assertTrue(len(self.challenge_period_client.get_testing_miners()) == 2)
        self.assertTrue(len(self.challenge_period_client.get_success_miners()) == 0)


        # Now add perf ledgers to check that adding miners without positions still doesn't add them
        self.perf_ledger_client.save_perf_ledgers(self.LEDGERS)
        self.challenge_period_client.add_challenge_period_testing_in_memory_and_disk(
            new_hotkeys=new_miners,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )


        self.assertTrue(len(self.challenge_period_client.get_testing_miners()) == 2)
        self.assertTrue(len(self.challenge_period_client.get_probation_miners()) == 0)
        self.assertTrue(len(self.challenge_period_client.get_success_miners()) == 0)

        all_miners_positions = self.position_client.get_positions_for_hotkeys(self.MINER_NAMES)
        self.assertListEqual(list(all_miners_positions.keys()), self.MINER_NAMES)

        miners_with_one_position = self.position_client.get_miner_hotkeys_with_at_least_one_position()
        miners_with_one_position_sorted = sorted(list(miners_with_one_position))

        self.assertListEqual(miners_with_one_position_sorted, sorted(self.MINER_NAMES))
        self.challenge_period_client.add_challenge_period_testing_in_memory_and_disk(
            new_hotkeys=self.MINER_NAMES,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )

        # All the miners should be passed to testing now
        self.assertListEqual(
            sorted(list(self.challenge_period_client.get_testing_miners().keys())),
            sorted(self.MINER_NAMES + new_miners),
        )

        self.assertListEqual(
            [self.challenge_period_client.get_testing_miners()[hk] for hk in self.MINER_NAMES],
            [self.HK_TO_OPEN_MS[hk] for hk in self.MINER_NAMES],
        )

        self.assertListEqual(
            [self.challenge_period_client.get_testing_miners()[hk] for hk in new_miners],
            [self.START_TIME, self.START_TIME],
        )

        self.assertEqual(len(self.challenge_period_client.get_success_miners()), 0)

    def test_refresh_all_eliminated(self):

        self.assertTrue(len(self.challenge_period_client.get_testing_miners()) == len(self.TESTING_MINER_NAMES))
        self.assertTrue(len(self.challenge_period_client.get_success_miners()) == len(self.SUCCESS_MINER_NAMES))
        self.assertTrue(len(self.elimination_client.get_eliminations_from_memory()) == 0, self.elimination_client.get_eliminations_from_memory())

        for miner in self.MINER_NAMES:
            self.elimination_client.append_elimination_row(miner, -1, "FAILED_CHALLENGE_PERIOD")

        self.challenge_period_client.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        self.elimination_client.process_eliminations()

        self.assertTrue(len(self.challenge_period_client.get_testing_miners()) == 0)
        self.assertTrue(len(self.challenge_period_client.get_success_miners()) == 0)
        self.assertTrue(len(self.elimination_client.get_eliminations_from_memory()) == len(self.MINER_NAMES))

    def test_clear_challengeperiod_in_memory_and_disk(self):
        miners = {
                "test_miner1": (MinerBucket.CHALLENGE, 1, None, None),
                "test_miner2": (MinerBucket.CHALLENGE, 1, None, None),
                "test_miner3": (MinerBucket.CHALLENGE, 1, None, None),
                "test_miner4": (MinerBucket.CHALLENGE, 1, None, None),
                "test_miner5": (MinerBucket.MAINCOMP, 1, None, None),
                "test_miner6": (MinerBucket.MAINCOMP, 1, None, None),
                "test_miner7": (MinerBucket.MAINCOMP, 1, None, None),
                "test_miner8": (MinerBucket.MAINCOMP, 1, None, None),
                }

        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners(miners)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()
        self.challenge_period_client._clear_challengeperiod_in_memory_and_disk()

        testing_keys = list(self.challenge_period_client.get_testing_miners())
        success_keys = list(self.challenge_period_client.get_success_miners())

        self.assertEqual(testing_keys, [])
        self.assertEqual(success_keys, [])

    def test_miner_elimination_reasons_mdd(self):
        """Test that miners are properly being eliminated when beyond mdd"""
        # Add failing miners to challenge bucket so they can be evaluated
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES,
                                     probation=self.PROBATION_MINER_NAMES)

        # Force-allow refresh by resetting last update time
        self.challenge_period_client.set_last_update_time(0)
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()

        eliminations_length = len(self.elimination_client.get_eliminations_from_memory())

        # Ensure that all miners that aren't failing end up in testing or success
        self.assertEqual(eliminations_length, len(self.FAILING_MINER_NAMES))
        for elimination in self.elimination_client.get_eliminations_from_disk():
            self.assertEqual(elimination["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value)

    def test_miner_elimination_reasons_time(self):
        """Test that miners who aren't passing challenge period are properly eliminated for time."""
        # Add failing miners to challenge bucket so they can be evaluated
        self._populate_active_miners(maincomp=self.SUCCESS_MINER_NAMES,
                                     challenge=self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES,
                                     probation=self.PROBATION_MINER_NAMES)

        # Force-allow refresh by resetting last update time
        self.challenge_period_client.set_last_update_time(0)
        self.challenge_period_client.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        self.elimination_client.process_eliminations()
        eliminations_length = len(self.elimination_client.get_eliminations_from_memory())

        # Ensure that all miners that aren't failing end up in testing or success
        self.assertEqual(eliminations_length, len(self.NOT_MAIN_COMP_MINER_NAMES))
        eliminated_for_time = set()
        eliminated_for_mdd = set()

        for elimination in self.elimination_client.get_eliminations_from_disk():
            if elimination["hotkey"] in self.FAILING_MINER_NAMES:
                eliminated_for_mdd.add(elimination["hotkey"])
                continue
            else:
                eliminated_for_time.add(elimination["hotkey"])
                self.assertEqual(elimination["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value)
        self.assertEqual(len(eliminated_for_mdd), len(self.FAILING_MINER_NAMES))
        self.assertEqual(len(eliminated_for_time), len(self.TESTING_MINER_NAMES + self.PROBATION_MINER_NAMES))

    def test_plagiarism_detection_and_elimination(self):
        """Test that miners detected as plagiarists are moved to PLAGIARISM bucket,
        then eliminated after PLAGIARISM_REVIEW_PERIOD_MS (2 weeks)"""

        # Start with a clean state - use first testing miner
        test_miner = self.TESTING_MINER_NAMES[0]

        # Ensure the miner starts in CHALLENGE bucket
        self.assertEqual(self.challenge_period_client.get_miner_bucket(test_miner), MinerBucket.CHALLENGE)

        # Inject plagiarism data via client API (bypasses actual API call)
        plagiarism_time = self.max_open_ms
        plagiarism_data = {test_miner: {"time": plagiarism_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, plagiarism_time)

        # Call refresh - miner should be moved to PLAGIARISM bucket
        self.challenge_period_client.refresh(current_time=self.max_open_ms)
        self.elimination_client.process_eliminations()

        # Verify the miner is in PLAGIARISM bucket
        self.assertEqual(self.challenge_period_client.get_miner_bucket(test_miner), MinerBucket.PLAGIARISM)
        self.assertIn(test_miner, self.challenge_period_client.get_plagiarism_miners())

        # Verify the miner is NOT eliminated yet
        elimination_hotkeys = self.elimination_client.get_eliminated_hotkeys()
        self.assertNotIn(test_miner, elimination_hotkeys)

        # Call refresh 2 weeks later (PLAGIARISM_REVIEW_PERIOD_MS + 1ms)
        elimination_time = plagiarism_time + ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS + 1

        # Re-inject plagiarism data to ensure it persists across the time gap
        # (prevents the server from trying to refresh from API which would return empty)
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, elimination_time)

        self.challenge_period_client.refresh(current_time=elimination_time)
        self.elimination_client.process_eliminations()

        # Verify the miner is now eliminated
        elimination_hotkeys = self.elimination_client.get_eliminated_hotkeys()
        self.assertIn(test_miner, elimination_hotkeys)

        # Verify the miner is no longer in PLAGIARISM bucket or any other bucket
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(test_miner))

        # Verify elimination reason is PLAGIARISM
        eliminations = self.elimination_client.get_eliminations_from_disk()
        plagiarism_elimination_found = False
        for elimination in eliminations:
            if elimination["hotkey"] == test_miner:
                self.assertEqual(elimination["reason"], EliminationReason.PLAGIARISM.value)
                plagiarism_elimination_found = True
                break

        self.assertTrue(plagiarism_elimination_found, f"Could not find plagiarism elimination record for {test_miner}")

    def test_daemon_processes_data_correctly(self):
        """
        Test that the daemon can process and persist miner data correctly.

        This verifies daemon behavior through the client interface:
        1. Daemon runs as a thread in the server process (not separate process)
        2. Daemon thread is different from test process
        3. Data written via client is accessible
        4. Daemon processes refresh operations correctly
        5. Data persists across operations
        """
        import os

        # Get test process info
        test_pid = os.getpid()

        # Get server process PID (from spawn handle)
        server_pid = self.challenge_period_handle.pid

        # Verify test and server are different processes
        self.assertNotEqual(test_pid, server_pid,
                           "Test and server should run in separate processes")

        # Start the daemon via client
        started = self.challenge_period_client.start_daemon()
        self.assertTrue(started, "Daemon should start successfully")

        # Get daemon info from server
        daemon_info = self.challenge_period_client.get_daemon_info()

        # Verify daemon is running
        self.assertTrue(daemon_info["daemon_started"], "Daemon should be marked as started")
        self.assertTrue(daemon_info["daemon_alive"], "Daemon thread should be alive")
        self.assertTrue(daemon_info["daemon_is_thread"], "Daemon should be a thread (not process)")

        # Verify daemon runs in server process (not test process)
        self.assertEqual(daemon_info["server_pid"], server_pid,
                        "Daemon should run in server process")
        self.assertNotEqual(daemon_info["server_pid"], test_pid,
                           "Daemon should NOT run in test process")

        # Verify daemon has a thread ID
        self.assertIsNotNone(daemon_info["daemon_ident"], "Daemon should have a thread ID")

        bt.logging.success(
            f"✓ Daemon architecture verified:\n"
            f"  - Test PID: {test_pid}\n"
            f"  - Server PID: {server_pid}\n"
            f"  - Daemon TID: {daemon_info['daemon_ident']}\n"
            f"  - Daemon is Thread: {daemon_info['daemon_is_thread']}\n"
            f"  - Architecture: Test Process → RPC → Server Process (PID {server_pid}) → Daemon Thread (TID {daemon_info['daemon_ident']})"
        )

        # Now test daemon functionality
        test_hotkey_1 = "daemon_test_miner_1"
        test_hotkey_2 = "daemon_test_miner_2"
        test_time = TimeUtil.now_in_millis()

        # Add miners via client interface
        self.challenge_period_client.set_miner_bucket(
            hotkey=test_hotkey_1,
            bucket=MinerBucket.CHALLENGE,
            start_time=test_time,
            prev_bucket=None,
            prev_time=None
        )

        self.challenge_period_client.set_miner_bucket(
            hotkey=test_hotkey_2,
            bucket=MinerBucket.MAINCOMP,
            start_time=test_time,
            prev_bucket=None,
            prev_time=None
        )

        # Verify data is accessible via client
        bucket_1 = self.challenge_period_client.get_miner_bucket(test_hotkey_1)
        bucket_2 = self.challenge_period_client.get_miner_bucket(test_hotkey_2)

        # Handle both enum and string values
        if isinstance(bucket_1, MinerBucket):
            self.assertEqual(bucket_1, MinerBucket.CHALLENGE)
        else:
            self.assertEqual(bucket_1, MinerBucket.CHALLENGE.value)

        if isinstance(bucket_2, MinerBucket):
            self.assertEqual(bucket_2, MinerBucket.MAINCOMP)
        else:
            self.assertEqual(bucket_2, MinerBucket.MAINCOMP.value)

        # Verify miners exist via client
        self.assertTrue(self.challenge_period_client.has_miner(test_hotkey_1))
        self.assertTrue(self.challenge_period_client.has_miner(test_hotkey_2))

        # Remove the test miners to clean up
        self.challenge_period_client.remove_miner(test_hotkey_1)
        self.challenge_period_client.remove_miner(test_hotkey_2)

        # Verify removal worked
        self.assertFalse(self.challenge_period_client.has_miner(test_hotkey_1))
        self.assertFalse(self.challenge_period_client.has_miner(test_hotkey_2))

        bt.logging.success(
            "✓ Daemon functionality verification complete:\n"
            "  - Data written via client is accessible\n"
            "  - Daemon can be controlled via RPC\n"
            "  - CRUD operations work correctly"
        )

    def test_client_rpc_operations(self):
        """
        Test that client RPC operations work correctly for reading and writing data.

        This verifies:
        1. Client can write data via RPC
        2. Client can read data via RPC
        3. Multiple operations maintain data consistency
        4. Data persists across multiple client calls
        """
        test_time = TimeUtil.now_in_millis()

        # Test 1: Write data via client RPC
        test_hotkey_1 = "rpc_test_miner_1"
        self.challenge_period_client.set_miner_bucket(
            hotkey=test_hotkey_1,
            bucket=MinerBucket.CHALLENGE,
            start_time=test_time,
            prev_bucket=None,
            prev_time=None
        )

        # Test 2: Read data back via client RPC
        bucket_1 = self.challenge_period_client.get_miner_bucket(test_hotkey_1)
        self.assertIsNotNone(bucket_1, "Client should read data via RPC")

        # Verify bucket value
        if isinstance(bucket_1, MinerBucket):
            self.assertEqual(bucket_1, MinerBucket.CHALLENGE)
        else:
            self.assertEqual(bucket_1, MinerBucket.CHALLENGE.value)

        # Test 3: Write multiple miners
        test_hotkey_2 = "rpc_test_miner_2"
        test_hotkey_3 = "rpc_test_miner_3"

        self.challenge_period_client.set_miner_bucket(
            hotkey=test_hotkey_2,
            bucket=MinerBucket.MAINCOMP,
            start_time=test_time,
            prev_bucket=None,
            prev_time=None
        )

        self.challenge_period_client.set_miner_bucket(
            hotkey=test_hotkey_3,
            bucket=MinerBucket.PROBATION,
            start_time=test_time,
            prev_bucket=None,
            prev_time=None
        )

        # Test 4: Verify all miners exist via client
        self.assertTrue(self.challenge_period_client.has_miner(test_hotkey_1))
        self.assertTrue(self.challenge_period_client.has_miner(test_hotkey_2))
        self.assertTrue(self.challenge_period_client.has_miner(test_hotkey_3))

        # Test 5: Verify data persists across multiple reads
        bucket_1_again = self.challenge_period_client.get_miner_bucket(test_hotkey_1)
        bucket_2 = self.challenge_period_client.get_miner_bucket(test_hotkey_2)
        bucket_3 = self.challenge_period_client.get_miner_bucket(test_hotkey_3)

        self.assertIsNotNone(bucket_1_again, "Data should persist")
        self.assertIsNotNone(bucket_2, "Data should persist")
        self.assertIsNotNone(bucket_3, "Data should persist")

        # Test 6: Update existing miner
        self.challenge_period_client.set_miner_bucket(
            hotkey=test_hotkey_1,
            bucket=MinerBucket.MAINCOMP,
            start_time=test_time + 1000,
            prev_bucket=MinerBucket.CHALLENGE,
            prev_time=test_time
        )

        # Verify update worked
        updated_bucket = self.challenge_period_client.get_miner_bucket(test_hotkey_1)
        if isinstance(updated_bucket, MinerBucket):
            self.assertEqual(updated_bucket, MinerBucket.MAINCOMP)
        else:
            self.assertEqual(updated_bucket, MinerBucket.MAINCOMP.value)

        # Test 7: Remove miner via client
        self.challenge_period_client.remove_miner(test_hotkey_1)

        # Verify removal worked
        self.assertFalse(self.challenge_period_client.has_miner(test_hotkey_1))

        bt.logging.success(
            "✓ Client RPC operations verification complete:\n"
            "  - Client can write data via RPC\n"
            "  - Client can read data via RPC\n"
            "  - Multiple operations maintain consistency\n"
            "  - Data persists across client calls"
        )

