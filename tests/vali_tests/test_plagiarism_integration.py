import uuid

from tests.shared_objects.mock_classes import (
    MockPlagiarismDetector,
    MockPositionManager, MockLivePriceFetcher,
)
from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.plagiarism_events import PlagiarismEvents
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class TestPlagiarismIntegration(TestBase):

    def setUp(self):

        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )


        self.ONE_DAY_MS = 1000 * 60 * 60 * 24
        self.ONE_HOUR_MS = 1000 * 60 * 60
        self.ONE_MIN_MS = 1000 * 60

        self.N_MINERS = 6
        self.MINER_NAMES = [f"test_miner{i}" for i in range(self.N_MINERS)]
        self.DEFAULT_ACCOUNT_SIZES = 100_000
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.mock_metagraph = MockMetagraph(self.MINER_NAMES)
        self.current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)

        self.position_manager = MockPositionManager(metagraph=self.mock_metagraph, perf_ledger_manager=None,
                                                    elimination_manager=self.elimination_manager)
        self.plagiarism_detector = MockPlagiarismDetector(self.mock_metagraph, self.position_manager)

        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1

        self.position_manager.clear_all_miner_positions()
        self.plagiarism_detector.clear_plagiarism_from_disk()

        self.elimination_manager.clear_eliminations()
        self.position_counter = 0
        PlagiarismEvents.clear_plagiarism_events()

        # Set up miners with postions for btc and eth
        # This will involve setting up 6 positions that aren't plagiarism

        # One position with low leverage and orders a day apart

        self.miner_0_btc_lev = [0.01, 0.02, -0.01]
        self.generate_one_position(hotkey=self.MINER_NAMES[0],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=self.miner_0_btc_lev,
                                   times_apart=[self.ONE_DAY_MS for _ in range(len(self.miner_0_btc_lev))],
                                   open_ms=0,
                                 )
        # One position, short open time
        miner_0_eth_lev = [-0.2]
        self.generate_one_position(hotkey=self.MINER_NAMES[0],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_0_eth_lev,
                                   times_apart=[0],
                                   open_ms=self.ONE_HOUR_MS,
                                   close_ms=self.ONE_HOUR_MS * 6)

        # Two positions, higher leverage each 2.5 days apart with one order
        miner_1_btc_lev_one = [0.5]
        miner_1_btc_close_one = self.ONE_HOUR_MS * 3 + (self.ONE_DAY_MS * 2.5)
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_1_btc_lev_one,
                                   times_apart=[0],
                                   open_ms=self.ONE_HOUR_MS * 3,
                                   close_ms=miner_1_btc_close_one)

        miner_1_btc_lev_two = [-0.5]
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_1_btc_lev_two,
                                   times_apart=[0],
                                   open_ms=miner_1_btc_close_one + (self.ONE_HOUR_MS))

        miner_1_eth_lev_one = [-0.3]
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_1_eth_lev_one,
                                   times_apart=[0],
                                   open_ms=self.ONE_HOUR_MS * 3,
                                   close_ms=miner_1_btc_close_one)
        miner_1_eth_lev_two = [0.2]
        self.generate_one_position(hotkey=self.MINER_NAMES[1],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_1_eth_lev_two,
                                   times_apart=[0],
                                   open_ms=miner_1_btc_close_one + (self.ONE_HOUR_MS),
                                   close_ms=miner_1_btc_close_one)

        # Three positions, somewhat frequent orders (6 hours apart)
        miner_2_btc_lev_one = [0.01, 0.4, -0.1, 0.1]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_2_btc_lev_one,
                                   times_apart=[self.ONE_HOUR_MS * 6 for _ in range(len(miner_2_btc_lev_one))],
                                   open_ms=self.ONE_DAY_MS,
                                   close_ms=self.ONE_DAY_MS * 2.25)
        miner_2_btc_lev_two = [0.01, 0.2, 0.1, -0.2]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_2_btc_lev_two,
                                   times_apart=[self.ONE_HOUR_MS * 6 for _ in range(len(miner_2_btc_lev_two))],
                                   open_ms=self.ONE_DAY_MS * 2.25 + self.ONE_MIN_MS,
                                   close_ms=self.ONE_DAY_MS * 3.5)

        miner_2_btc_lev_three = [-0.1, -0.05, 0.1, -0.1]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_2_btc_lev_three,
                                   times_apart=[self.ONE_HOUR_MS * 6 for _ in range(len(miner_2_btc_lev_three))],
                                   open_ms=self.ONE_DAY_MS * 3.5 + self.ONE_DAY_MS)

        #One Position, One Order
        miner_2_eth_lev = [-0.3]
        self.generate_one_position(hotkey=self.MINER_NAMES[2],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=miner_2_eth_lev,
                                   times_apart=[0],
                                   open_ms=0)

        # Different times apart two positions
        miner_3_btc_lev_one = [0.33, -0.05, -0.05] # 12, 8 hours apart
        self.generate_one_position(hotkey=self.MINER_NAMES[3],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_3_btc_lev_one,
                                   times_apart=[0, self.ONE_HOUR_MS * 12, self.ONE_HOUR_MS * 8],
                                   open_ms=self.ONE_HOUR_MS * 13,
                                   close_ms=self.ONE_DAY_MS * 3)

        miner_3_btc_lev_two = [0.05, 0.1, 0.2, -0.1] # 30 min, 6 hours, 10 min
        self.generate_one_position(hotkey=self.MINER_NAMES[3],
                                   trade_pair=TradePair.BTCUSD,
                                   leverages=miner_3_btc_lev_two,
                                   times_apart=[0, self.ONE_MIN_MS * 30, self.ONE_HOUR_MS * 6, self.ONE_MIN_MS * 10],
                                   open_ms=self.ONE_DAY_MS * 3 + self.ONE_HOUR_MS)

        # Longer Different Times one position
        self.miner_3_eth_lev_one = [0.25, 0.1, -0.2] # 1 day, 2 days
        self.miner_3_eth_times_apart = [0, self.ONE_DAY_MS, self.ONE_DAY_MS * 2]
        self.generate_one_position(hotkey=self.MINER_NAMES[3],
                                   trade_pair=TradePair.ETHUSD,
                                   leverages=self.miner_3_eth_lev_one,
                                   times_apart=[0, self.ONE_DAY_MS, self.ONE_DAY_MS * 2],
                                   open_ms=self.ONE_MIN_MS * 10)

    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position, delete_open_position_if_exists=True)

    def generate_one_position(self, hotkey, trade_pair, leverages, times_apart, open_ms, close_ms=None, times_after=None):
        if times_after is None:
            times_after = [0 for _ in range(len(leverages))]
        if close_ms is None:
            close_ms = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS + 1

        self.position_counter += 1
        position = Position(
            miner_hotkey=hotkey,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + f"pos{self.position_counter}",
            open_ms=open_ms,
            trade_pair=trade_pair,
            account_size=self.DEFAULT_ACCOUNT_SIZES,
        )

        for i in range(len(leverages)):

            if leverages[i] > 0:
                type = OrderType.LONG
            elif leverages[i] < 0:
                type = OrderType.SHORT
            else:
                type = OrderType.FLAT

            order = Order(order_type=type,
                leverage=leverages[i],
                price=1000,
                trade_pair=position.trade_pair,
                processed_ms= open_ms + (i * times_apart[i]) + times_after[i],
                order_uuid=str(uuid.uuid4()))
            self.add_order_to_position_and_save_to_disk(position, order)
        position.close_ms = close_ms

        to_close = close_ms < ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        if to_close:
            close_order = Order(order_type=OrderType.FLAT,
                leverage=0,
                price=1000,
                trade_pair=position.trade_pair,
                processed_ms= close_ms - 1,
                order_uuid=str(uuid.uuid4()))
            self.add_order_to_position_and_save_to_disk(order=close_order, position=position)
        self.position_manager.save_miner_position(position, delete_open_position_if_exists=True)

    def check_one_plagiarist(self, plagiarist_id, victim_id, trade_pair_name):
        self.plagiarism_detector.detect()

        for miner in self.plagiarism_detector.plagiarism_data:
            if miner["plagiarist"] == plagiarist_id:
                self.assertGreaterEqual(miner["overall_score"], 0.95)
                trade_pairs = miner["trade_pairs"]

                # There should only be one trade pair
                self.assertEqual(len(trade_pairs.keys()), 1)

                self.assertIn(trade_pair_name, trade_pairs)

                plagiarism_event = trade_pairs[trade_pair_name]

                # There should only be one victim
                self.assertEqual(len(plagiarism_event["victims"]), 1)

                victim = plagiarism_event["victims"][0]

                self.assertEqual(victim["victim"], victim_id)
                self.assertEqual(victim["victim_trade_pair"], trade_pair_name)

                # Flagged for at least two events
                self.assertGreaterEqual(len(victim["events"]), 2)

                # Flagged for follow orders and single similarity
                event_set = set([event["type"] for event in victim["events"]])
                self.assertIn("follow", event_set)
                self.assertIn("single", event_set)

                for event in victim["events"]:
                    if event["type"] == "follow":
                        self.assertAlmostEqual(event["score"], 1)

                    elif event["type"] == "lag":
                        self.assertGreaterEqual(event["score"], 1)

                    elif event["type"] == "single":
                        self.assertGreaterEqual(event["score"], 0.95)
            else:
                self.assertLess(miner["overall_score"], 0.8)

    def test_no_plagiarism(self):
        # There should be no false positives
        positions = self.position_manager.get_positions_for_hotkeys(
                self.mock_metagraph.hotkeys,
            )

        self.plagiarism_detector.detect(hotkeys= self.mock_metagraph.hotkeys,
            hotkey_positions=positions)

        self.plagiarism_detector._update_plagiarism_scores_in_memory()
        self.assertGreaterEqual(len(self.plagiarism_detector.plagiarism_data), 1)

        for miner, data in self.plagiarism_detector.plagiarism_data.items():
            self.assertLess(data["overall_score"], 0.8)

    def test_plagiarism_scale(self):
        # Plagiarist scales the leverages of another miner with constant time lag of one hour
        # Copies Miner zero bitcoin leverages

        leverages = [x * 1.1 for x in self.miner_0_btc_lev]
        times_apart = [self.ONE_DAY_MS for _ in range(len(self.miner_0_btc_lev))] #same as for miner 0
        times_after = [self.ONE_HOUR_MS for _ in range(len(self.miner_0_btc_lev))]

        self.generate_one_position( hotkey=self.MINER_NAMES[4],
                                    trade_pair=TradePair.BTCUSD,
                                    leverages=leverages,
                                    times_apart=times_apart,
                                    open_ms=0,
                                    times_after=times_after)

        self.check_one_plagiarist(plagiarist_id=self.MINER_NAMES[4], victim_id=self.MINER_NAMES[0], trade_pair_name=TradePair.BTCUSD.name)


    def test_plagiarism_shift(self):
        # Plagiarist shifts the leverages of another miner with constant time lag of one hour
        # Copies Miner three ethereum leverages
        plagiarist_shift = 0.1

        leverages = [x + plagiarist_shift for x in self.miner_3_eth_lev_one]
        times_after = [self.ONE_HOUR_MS for _ in range(len(self.miner_3_eth_lev_one))]

        self.generate_one_position( hotkey=self.MINER_NAMES[4],
                                    trade_pair=TradePair.ETHUSD,
                                    leverages=leverages,
                                    times_apart=self.miner_3_eth_times_apart,
                                    open_ms=self.ONE_MIN_MS * 10,
                                    times_after=times_after)

        self.check_one_plagiarist(plagiarist_id=self.MINER_NAMES[4], victim_id=self.MINER_NAMES[3], trade_pair_name=TradePair.ETHUSD.name)


    def test_plagiarism_variable_scale(self):
        scales = [0.9, 1.1, 0.85]
        leverages = [x * scales[i] for i, x in enumerate(self.miner_0_btc_lev)]
        times_apart = [self.ONE_DAY_MS for _ in range(len(self.miner_0_btc_lev))] #same as for miner 0
        times_after = [self.ONE_HOUR_MS for _ in range(len(self.miner_0_btc_lev))]

        self.generate_one_position( hotkey=self.MINER_NAMES[4],
                                    trade_pair=TradePair.BTCUSD,
                                    leverages=leverages,
                                    times_apart=times_apart,
                                    open_ms=0,
                                    times_after=times_after)

        self.check_one_plagiarist(plagiarist_id=self.MINER_NAMES[4], victim_id=self.MINER_NAMES[0], trade_pair_name=TradePair.BTCUSD.name)

    def test_plagiarism_variable_shift(self):
        # Plagiarist shifts the leverages of another miner with constant time lag of one hour
        # Copies Miner three ethereum leverages
        plagiarist_shifts = [0.1, 0.05, -0.1]

        leverages = [x + plagiarist_shifts[i] for i, x in enumerate(self.miner_3_eth_lev_one)]
        times_after = [self.ONE_HOUR_MS for _ in range(len(self.miner_3_eth_lev_one))]

        self.generate_one_position( hotkey=self.MINER_NAMES[4],
                                    trade_pair=TradePair.ETHUSD,
                                    leverages=leverages,
                                    times_apart=self.miner_3_eth_times_apart,
                                    open_ms=self.ONE_MIN_MS * 10,
                                    times_after=times_after)

        self.check_one_plagiarist(plagiarist_id=self.MINER_NAMES[4], victim_id=self.MINER_NAMES[3], trade_pair_name=TradePair.ETHUSD.name)
