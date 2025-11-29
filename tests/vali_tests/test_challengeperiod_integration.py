# developer: trdougherty
from copy import deepcopy
from unittest.mock import patch

from tests.shared_objects.mock_classes import MockPositionManager
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
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    TP_ID_PORTFOLIO,
    PerfLedger,
    PerfLedgerManager,
)
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils


class TestChallengePeriodIntegration(TestBase):

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

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

        # Initialize system components
        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)

        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.elimination_manager = EliminationManager(self.mock_metagraph, self.live_price_fetcher, None, running_unit_tests=True, contract_manager=self.contract_manager)
        self.ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)
        self.ledger_manager.clear_all_ledger_data()
        # Ensure no perf ledgers present
        assert len(self.ledger_manager.get_perf_ledgers()) == 0, self.ledger_manager.get_perf_ledgers()
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
        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

        self.position_manager.clear_all_miner_positions()

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

        self.ledger_manager.save_perf_ledgers(self.LEDGERS)
        n_perf_ledgers_saved_disk = len(self.ledger_manager.get_perf_ledgers(from_disk=True))
        n_perf_ledgers_saved_memory = len(self.ledger_manager.get_perf_ledgers(from_disk=False))
        assert n_perf_ledgers_saved_disk == len(self.MINER_NAMES), (n_perf_ledgers_saved_disk, self.LEDGERS, self.MINER_NAMES)
        assert n_perf_ledgers_saved_memory == len(self.MINER_NAMES), (n_perf_ledgers_saved_memory, self.LEDGERS, self.MINER_NAMES)

        for miner, positions in self.POSITIONS.items():
            for position in positions:
                position.position_uuid = f"{miner}_position_{position.open_ms}_{position.close_ms}"
                self.position_manager.save_miner_position(position)

        self.max_open_ms = max(self.HK_TO_OPEN_MS.values())


        # Finally update the challenge period to default state
        self.challengeperiod_manager.elimination_manager.clear_eliminations()

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
        self.challengeperiod_manager.active_miners = miners

    def tearDown(self):
        super().tearDown()
        # Cleanup and setup
        self.position_manager.clear_all_miner_positions()
        self.ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.challengeperiod_manager.elimination_manager.clear_eliminations()

    def teest_refresh_populations(self):
        self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
        self.elimination_manager.process_eliminations(PositionLocks())
        testing_length = len(self.challengeperiod_manager.get_testing_miners())
        success_length = len(self.challengeperiod_manager.get_success_miners())
        eliminations_length = len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory())

        # Ensure that all miners that aren't failing end up in testing or success
        self.assertEqual(testing_length + success_length, len(self.NOT_FAILING_MINER_NAMES))
        self.assertEqual(eliminations_length, len(self.FAILING_MINER_NAMES))

    def test_full_refresh(self):
        self.assertEqual(len(self.challengeperiod_manager.get_testing_miners()), len(self.TESTING_MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.get_success_miners()), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()), 0)

        inspection_hotkeys = self.challengeperiod_manager.get_testing_miners()

        for hotkey, inspection_time in inspection_hotkeys.items():
            time_criteria = self.max_open_ms - inspection_time <= ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
            self.assertTrue(time_criteria, f"Time criteria failed for {hotkey}")

        self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
        self.elimination_manager.process_eliminations(PositionLocks())

        elimination_hotkeys_memory = [x['hotkey'] for x in self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()]
        elimination_hotkeys_disk = [x['hotkey'] for x in self.challengeperiod_manager.elimination_manager.get_eliminations_from_disk()]

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
        self.assertListEqual(sorted(self.MINER_NAMES), sorted(self.mock_metagraph.hotkeys))
        self.assertListEqual(sorted(self.TESTING_MINER_NAMES), sorted(list(self.challengeperiod_manager.get_testing_miners().keys())))

        # Let's check the initial state of the challenge period
        self.assertEqual(len(self.challengeperiod_manager.get_success_miners()), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()), 0)
        self.assertEqual(len(self.challengeperiod_manager.get_testing_miners()), len(self.TESTING_MINER_NAMES))

        eliminations = self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()
        self.assertEqual(len(eliminations), 0)

        self.challengeperiod_manager.remove_eliminated(eliminations=eliminations)
        self.assertEqual(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()), 0)

        self.assertEqual(len(self.challengeperiod_manager.get_testing_miners()), len(self.TESTING_MINER_NAMES))

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.challengeperiod_manager.metagraph.hotkeys,
            eliminations=self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory(),
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )

        self.assertEqual(len(self.challengeperiod_manager.get_testing_miners()), len(self.TESTING_MINER_NAMES + self.FAILING_MINER_NAMES))

        current_time = current_time=self.max_open_ms + ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS*ValiConfig.DAILY_MS + 1
        self.challengeperiod_manager.refresh(current_time)
        self.elimination_manager.process_eliminations(PositionLocks())

        elimination_keys = self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()

        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, self.mock_metagraph.hotkeys)
            self.assertIn(miner, elimination_keys)

        eliminations = self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()

        self.assertListEqual(sorted(list(eliminations)), sorted(elimination_keys))

    @patch('data_generator.polygon_data_service.PolygonDataService.get_event_before_market_close')
    @patch('data_generator.polygon_data_service.PolygonDataService.get_candles_for_trade_pair')
    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_single_position_no_ledger(self, mock_candle_fetcher, mock_get_candles, mock_market_close):
        # Mock the API calls to return appropriate values for testing
        mock_candle_fetcher.return_value = []
        mock_get_candles.return_value = []
        from vali_objects.utils.live_price_fetcher import PriceSource
        mock_market_close.return_value = PriceSource(open=50000, high=50000, low=50000, close=50000, volume=0, vwap=50000, timestamp=0)
        
        # Cleanup all positions first
        self.position_manager.clear_all_miner_positions()
        self.ledger_manager.clear_perf_ledgers_from_disk()

        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.challengeperiod_manager.elimination_manager.clear_eliminations()

        self.challengeperiod_manager.active_miners = {}

        position = deepcopy(self.DEFAULT_POSITION)
        position.is_closed_position = False
        position.close_ms = None

        self.position_manager.save_miner_position(position)
        self.challengeperiod_manager.active_miners = {self.DEFAULT_MINER_HOTKEY: (MinerBucket.CHALLENGE, self.DEFAULT_OPEN_MS, None, None)}
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        # Now loading the data
        positions = self.position_manager.get_positions_for_hotkeys(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        ledgers = self.ledger_manager.get_perf_ledgers(from_disk=True)
        ledgers_memory = self.ledger_manager.get_perf_ledgers(from_disk=False)
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
        challenge_success, challenge_demote, challenge_eliminations = self.challengeperiod_manager.inspect(
            positions=positions,
            ledger=ledgers,
            success_hotkeys=self.SUCCESS_MINER_NAMES,
            probation_hotkeys=self.PROBATION_MINER_NAMES,
            inspection_hotkeys=self.challengeperiod_manager.get_testing_miners(),
            current_time=self.max_open_ms,
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
        )
        self.elimination_manager.process_eliminations(PositionLocks())
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)

        # There should be no promotion or demotion
        self.assertListEqual(challenge_success, [])
        self.assertDictEqual(challenge_eliminations, {})


    def test_promote_testing_miner(self):
        # Add all the challenge period miners
        self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
        self.elimination_manager.process_eliminations(PositionLocks())

        testing_hotkeys = list(self.challengeperiod_manager.get_testing_miners().keys())
        success_hotkeys = list(self.challengeperiod_manager.get_success_miners().keys())

        self.assertIn(self.TESTING_MINER_NAMES[0], testing_hotkeys)
        self.assertNotIn(self.TESTING_MINER_NAMES[0], success_hotkeys)

        self.challengeperiod_manager._promote_challengeperiod_in_memory(
            hotkeys=[self.TESTING_MINER_NAMES[0]],
            current_time=self.max_open_ms,
        )

        testing_hotkeys = list(self.challengeperiod_manager.get_testing_miners().keys())
        success_hotkeys = list(self.challengeperiod_manager.get_success_miners().keys())

        self.assertNotIn(self.TESTING_MINER_NAMES[0], testing_hotkeys)
        self.assertIn(self.TESTING_MINER_NAMES[0], success_hotkeys)

        # Check that the timestamp of the success is the current time of evaluation
        self.assertEqual(
            self.challengeperiod_manager.active_miners[self.TESTING_MINER_NAMES[0]][1],
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
        self.assertTrue(len(self.challengeperiod_manager.get_testing_miners()) == len(self.TESTING_MINER_NAMES))
        self.assertTrue(len(self.challengeperiod_manager.get_success_miners()) == len(self.SUCCESS_MINER_NAMES))
        self.assertTrue(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()) == 0)

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

        refresh_time = self.HK_TO_OPEN_MS['eliminated_miner1'] + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS + 1
        self.challengeperiod_manager.refresh(refresh_time)

        self.assertEqual(self.challengeperiod_manager.eliminations_with_reasons['eliminated_miner1'][0],
                         EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value)

        self.elimination_manager.process_eliminations(PositionLocks())

        challenge_success = list(self.challengeperiod_manager.get_success_miners())
        elimininations = list(self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys())
        challenge_testing = list(self.challengeperiod_manager.get_testing_miners())

        self.assertTrue(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()) > 0)
        for miner in self.FAILING_MINER_NAMES:
            self.assertIn(miner, elimininations)
            self.assertNotIn(miner, challenge_testing)
            self.assertNotIn(miner, challenge_success)

    def test_no_positions_miner_filtered(self):
        for hotkey in self.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.CHALLENGE):
            del self.challengeperiod_manager.active_miners[hotkey]
        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        self.assertEqual(len(self.challengeperiod_manager.get_success_miners()), len(self.SUCCESS_MINER_NAMES))
        self.assertEqual(len(self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()), 0)
        self.assertEqual(len(self.challengeperiod_manager.get_testing_miners()), 0)

        # Now going to remove the positions of the miners
        miners_without_positions = self.TESTING_MINER_NAMES[:2]
        # Redeploy the positions
        for miner, positions in self.POSITIONS.items():
            if miner in miners_without_positions:
                for position in positions:
                    self.position_manager.delete_position(position)

        current_time = self.max_open_ms
        self.assertEqual(len(self.challengeperiod_manager.get_testing_miners()), 0)
        self.challengeperiod_manager.refresh(current_time=current_time)
        self.elimination_manager.process_eliminations(PositionLocks())

        for miner in miners_without_positions:
            self.assertIn(miner, self.mock_metagraph.hotkeys)
            self.assertEqual(current_time, self.challengeperiod_manager.get_testing_miners()[miner])
            # self.assertNotIn(miner, self.challengeperiod_manager.get_testing_miners()) # Miners without positions are not necessarily eliminated
            self.assertNotIn(miner, self.challengeperiod_manager.get_success_miners())

    def test_disjoint_testing_success(self):
        self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
        self.elimination_manager.process_eliminations(PositionLocks())

        testing_set = set(self.challengeperiod_manager.get_testing_miners().keys())
        success_set = set(self.challengeperiod_manager.get_success_miners().keys())

        self.assertTrue(testing_set.isdisjoint(success_set))

    def test_addition(self):
        self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
        self.elimination_manager.process_eliminations(PositionLocks())

        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.MINER_NAMES,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )

        testing_set = set(self.challengeperiod_manager.get_testing_miners().keys())
        success_set = set(self.challengeperiod_manager.get_success_miners().keys())

        self.assertTrue(testing_set.isdisjoint(success_set))

    def test_add_miner_no_positions(self):
        self.challengeperiod_manager.active_miners = {}

        # Check if it still stores the miners with no perf ledger
        self.ledger_manager.clear_perf_ledgers_from_disk()

        new_miners = ["miner_no_positions1", "miner_no_positions2"]

        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=new_miners,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )
        self.assertTrue(len(self.challengeperiod_manager.get_testing_miners()) == 2)
        self.assertTrue(len(self.challengeperiod_manager.get_success_miners()) == 0)


        # Now add perf ledgers to check that adding miners without positions still doesn't add them
        self.ledger_manager.save_perf_ledgers(self.LEDGERS)
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=new_miners,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )


        self.assertTrue(len(self.challengeperiod_manager.get_testing_miners()) == 2)
        self.assertTrue(len(self.challengeperiod_manager.get_probation_miners()) == 0)
        self.assertTrue(len(self.challengeperiod_manager.get_success_miners()) == 0)

        all_miners_positions = self.challengeperiod_manager.position_manager.get_positions_for_hotkeys(self.MINER_NAMES)
        self.assertListEqual(list(all_miners_positions.keys()), self.MINER_NAMES)

        miners_with_one_position = self.challengeperiod_manager.position_manager.get_miner_hotkeys_with_at_least_one_position()
        miners_with_one_position_sorted = sorted(list(miners_with_one_position))

        self.assertListEqual(miners_with_one_position_sorted, sorted(self.MINER_NAMES))
        self.challengeperiod_manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.MINER_NAMES,
            eliminations=[],
            hk_to_first_order_time=self.HK_TO_OPEN_MS,
            default_time=self.START_TIME,
        )

        # All the miners should be passed to testing now
        self.assertListEqual(
            sorted(list(self.challengeperiod_manager.get_testing_miners().keys())),
            sorted(self.MINER_NAMES + new_miners),
        )

        self.assertListEqual(
            [self.challengeperiod_manager.get_testing_miners()[hk] for hk in self.MINER_NAMES],
            [self.HK_TO_OPEN_MS[hk] for hk in self.MINER_NAMES],
        )

        self.assertListEqual(
            [self.challengeperiod_manager.get_testing_miners()[hk] for hk in new_miners],
            [self.START_TIME, self.START_TIME],
        )

        self.assertEqual(len(self.challengeperiod_manager.get_success_miners()), 0)

    def test_refresh_all_eliminated(self):

        self.assertTrue(len(self.challengeperiod_manager.get_testing_miners()) == len(self.TESTING_MINER_NAMES))
        self.assertTrue(len(self.challengeperiod_manager.get_success_miners()) == len(self.SUCCESS_MINER_NAMES))
        self.assertTrue(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()) == 0, self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory())

        for miner in self.MINER_NAMES:
            self.challengeperiod_manager.elimination_manager.append_elimination_row(miner, -1, "FAILED_CHALLENGE_PERIOD")

        self.challengeperiod_manager.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        self.elimination_manager.process_eliminations(PositionLocks())

        self.assertTrue(len(self.challengeperiod_manager.get_testing_miners()) == 0)
        self.assertTrue(len(self.challengeperiod_manager.get_success_miners()) == 0)
        self.assertTrue(len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory()) == len(self.MINER_NAMES))

    def test_clear_challengeperiod_in_memory_and_disk(self):
        self.active_miners = {
                "test_miner1": (MinerBucket.CHALLENGE, 1),
                "test_miner2": (MinerBucket.CHALLENGE, 1),
                "test_miner3": (MinerBucket.CHALLENGE, 1),
                "test_miner4": (MinerBucket.CHALLENGE, 1),
                "test_miner5": (MinerBucket.MAINCOMP, 1),
                "test_miner6": (MinerBucket.MAINCOMP, 1),
                "test_miner7": (MinerBucket.MAINCOMP, 1),
                "test_miner8": (MinerBucket.MAINCOMP, 1),
                }

        self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()

        testing_keys = list(self.challengeperiod_manager.get_testing_miners())
        success_keys = list(self.challengeperiod_manager.get_success_miners())

        self.assertEqual(testing_keys, [])
        self.assertEqual(success_keys, [])

    def test_miner_elimination_reasons_mdd(self):
        """Test that miners are properly being eliminated when beyond mdd"""
        self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
        self.elimination_manager.process_eliminations(PositionLocks())

        eliminations_length = len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory())

        # Ensure that all miners that aren't failing end up in testing or success
        self.assertEqual(eliminations_length, len(self.FAILING_MINER_NAMES))
        for elimination in self.challengeperiod_manager.elimination_manager.get_eliminations_from_disk():
            self.assertEqual(elimination["reason"], EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value)

    def test_miner_elimination_reasons_time(self):
        """Test that miners who aren't passing challenge period are properly eliminated for time."""
        self.challengeperiod_manager.refresh(current_time=self.OUTSIDE_OF_CHALLENGE)
        self.elimination_manager.process_eliminations(PositionLocks())
        eliminations_length = len(self.challengeperiod_manager.elimination_manager.get_eliminations_from_memory())

        # Ensure that all miners that aren't failing end up in testing or success
        self.assertEqual(eliminations_length, len(self.NOT_MAIN_COMP_MINER_NAMES))
        eliminated_for_time = set()
        eliminated_for_mdd = set()

        for elimination in self.challengeperiod_manager.elimination_manager.get_eliminations_from_disk():
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
        self.assertEqual(self.challengeperiod_manager.get_miner_bucket(test_miner), MinerBucket.CHALLENGE)

        # Mock the plagiarism API to return the test miner as a plagiarist
        plagiarism_time = self.max_open_ms
        plagiarism_data = {test_miner: {"time": plagiarism_time}}

        # Mock get_plagiarism_elimination_scores to return our plagiarism data
        with patch.object(self.plagiarism_manager, 'get_plagiarism_elimination_scores', return_value=plagiarism_data):
            # Call refresh - miner should be moved to PLAGIARISM bucket
            self.challengeperiod_manager.refresh(current_time=self.max_open_ms)
            self.elimination_manager.process_eliminations(PositionLocks())

            # Verify the miner is in PLAGIARISM bucket
            self.assertEqual(self.challengeperiod_manager.get_miner_bucket(test_miner), MinerBucket.PLAGIARISM)
            self.assertIn(test_miner, self.challengeperiod_manager.get_plagiarism_miners())

            # Verify the miner is NOT eliminated yet
            elimination_hotkeys = self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()
            self.assertNotIn(test_miner, elimination_hotkeys)

            # Call refresh 2 weeks later (PLAGIARISM_REVIEW_PERIOD_MS + 1ms)
            elimination_time = plagiarism_time + ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS + 1
            self.challengeperiod_manager.refresh(current_time=elimination_time)
            self.elimination_manager.process_eliminations(PositionLocks())

            # Verify the miner is now eliminated
            elimination_hotkeys = self.challengeperiod_manager.elimination_manager.get_eliminated_hotkeys()
            self.assertIn(test_miner, elimination_hotkeys)

            # Verify the miner is no longer in PLAGIARISM bucket or any other bucket
            self.assertIsNone(self.challengeperiod_manager.get_miner_bucket(test_miner))

            # Verify elimination reason is PLAGIARISM
            eliminations = self.challengeperiod_manager.elimination_manager.get_eliminations_from_disk()
            plagiarism_elimination_found = False
            for elimination in eliminations:
                if elimination["hotkey"] == test_miner:
                    self.assertEqual(elimination["reason"], EliminationReason.PLAGIARISM.value)
                    plagiarism_elimination_found = True
                    break

            self.assertTrue(plagiarism_elimination_found, f"Could not find plagiarism elimination record for {test_miner}")


