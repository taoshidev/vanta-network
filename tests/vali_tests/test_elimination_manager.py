from unittest.mock import patch
from shared_objects.cache_controller import CacheController
from tests.shared_objects.mock_classes import MockPositionManager
from shared_objects.mock_metagraph import MockMetagraph
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.vali_utils import ValiUtils
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
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager


class TestEliminationManager(TestBase):
    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )


        self.MDD_MINER = "miner_mdd"
        self.REGULAR_MINER = "miner_regular"
        self.DEFAULT_ACCOUNT_SIZE = 100_000

        # Initialize system components
        self.mock_metagraph = MockMetagraph([self.MDD_MINER, self.REGULAR_MINER])
        
        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)

        self.contract_manager = ValidatorContractManager(running_unit_tests=True)
        self.elimination_manager = EliminationManager(self.mock_metagraph, self.live_price_fetcher, None, running_unit_tests=True, contract_manager=self.contract_manager)
        self.ledger_manager = PerfLedgerManager(self.mock_metagraph, running_unit_tests=True)
        self.position_manager = MockPositionManager(self.mock_metagraph,
                                                    perf_ledger_manager=self.ledger_manager,
                                                    elimination_manager=self.elimination_manager,
                                                    live_price_fetcher=self.live_price_fetcher)
        self.position_manager.clear_all_miner_positions()
        for hk in self.mock_metagraph.hotkeys:
            mock_position = Position(
                miner_hotkey=hk,
                position_uuid=hk,
                open_ms=1,
                close_ms=2,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                return_at_close=1.00,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[Order(price=60000, processed_ms=1, order_uuid="initial_order",
                              trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],

            )
            self.position_manager.save_miner_position(mock_position)

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        files = CacheController.get_directory_names(all_miners_dir)
        assert len(files) == len(self.mock_metagraph.hotkeys), (all_miners_dir, files, self.mock_metagraph.hotkeys)

        self.position_manager.perf_ledger_manager = self.ledger_manager
        self.elimination_manager.position_manager = self.position_manager
        self.plagiarism_manager = PlagiarismManager(slack_notifier=None, running_unit_tests=True)
        self.challengeperiod_manager = ChallengePeriodManager(self.mock_metagraph,
                                                              position_manager=self.position_manager,
                                                              perf_ledger_manager=self.ledger_manager,
                                                              plagiarism_manager=self.plagiarism_manager,
                                                              running_unit_tests=True)
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

        self.position_locks = PositionLocks()

        self.LEDGERS = {}
        self.LEDGERS[self.MDD_MINER] = generate_losing_ledger(0, ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS)
        self.LEDGERS[self.REGULAR_MINER] = generate_winning_ledger(0, ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS)
        self.ledger_manager.save_perf_ledgers(self.LEDGERS)

        self.challengeperiod_manager.active_miners[self.MDD_MINER] = (MinerBucket.MAINCOMP, 0, None, None)
        self.challengeperiod_manager.active_miners[self.REGULAR_MINER] = (MinerBucket.MAINCOMP, 0, None, None)

    def tearDown(self):
        super().tearDown()
        # Cleanup and setup
        self.position_manager.clear_all_miner_positions()
        self.ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_elimination_for_mdd(self, mock_candle_fetcher):
        # Mock the API call to return empty list (no price data needed for this test)
        mock_candle_fetcher.return_value = []
        
        # Neither miner has been eliminated
        self.assertEqual(len(self.challengeperiod_manager.get_success_miners()), 2)

        self.elimination_manager.process_eliminations(self.position_locks)
        
        # Assert the mock was called
        self.assertTrue(mock_candle_fetcher.called)

        eliminations = self.elimination_manager.get_eliminations_from_disk()
        self.assertEqual(len(eliminations), 1)
        for elimination in eliminations:
            self.assertEqual(elimination["hotkey"], self.MDD_MINER)
            self.assertEqual(elimination["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

        # test_zombie_eliminations
        self.mock_metagraph.hotkeys = []
        self.elimination_manager.process_eliminations(self.position_locks)
        eliminations = self.elimination_manager.get_eliminations_from_disk()
        for elimination in eliminations:
            if elimination["hotkey"] == self.MDD_MINER:
                assert elimination["reason"] == EliminationReason.MAX_TOTAL_DRAWDOWN.value, eliminations
            elif elimination["hotkey"] == self.REGULAR_MINER:
                assert elimination["reason"] == EliminationReason.ZOMBIE.value, eliminations
            else:
                raise Exception(f"Unexpected hotkey in eliminations: {elimination['hotkey']}")



