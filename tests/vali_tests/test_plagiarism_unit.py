# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import uuid

from aiofiles.os import access
from tests.shared_objects.mock_classes import MockPlagiarismDetector, MockLivePriceFetcher
from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.plagiarism_definitions import (
    CopySimilarity,
    FollowPercentage,
    LagDetection,
    ThreeCopySimilarity,
    TwoCopySimilarity,
)
from vali_objects.utils.plagiarism_events import PlagiarismEvents
from vali_objects.utils.plagiarism_pipeline import PlagiarismPipeline
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class TestPlagiarismUnit(TestBase):

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.MINER_HOTKEY3 = "test_miner3"
        self.MINER_HOTKEY4 = "test_miner4"
        self.mock_metagraph = MockMetagraph([self.MINER_HOTKEY1, self.MINER_HOTKEY2, self.MINER_HOTKEY3, self.MINER_HOTKEY4])
        self.current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True,
                                                elimination_manager=self.elimination_manager,
                                                live_price_fetcher=self.live_price_fetcher)
        self.elimination_manager.position_manager = self.position_manager
        self.plagiarism_detector = MockPlagiarismDetector(self.mock_metagraph, self.position_manager)
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_ACCOUNT_SIZE = 100_000

        self.eth_position1 = Position(
            miner_hotkey=self.MINER_HOTKEY1,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_eth1",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.ETHUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.eth_position2 = Position(
            miner_hotkey=self.MINER_HOTKEY2,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_eth2",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.ETHUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.eth_position3 = Position(
            miner_hotkey=self.MINER_HOTKEY3,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_eth3",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.ETHUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.btc_position1 = Position(
            miner_hotkey=self.MINER_HOTKEY1,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_btc1",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.btc_position2 = Position(
            miner_hotkey=self.MINER_HOTKEY2,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_btc2",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.btc_position3 = Position(
            miner_hotkey=self.MINER_HOTKEY3,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + "_btc3",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.position_manager.clear_all_miner_positions()
        self.plagiarism_detector.clear_plagiarism_from_disk()

        self.plagiarism_detector.position_manager.elimination_manager.clear_eliminations()
        self.position_counter = 0
        PlagiarismEvents.clear_plagiarism_events()

        self.plagiarism_classes = [FollowPercentage,
                                   LagDetection,
                                   CopySimilarity,
                                   TwoCopySimilarity,
                                   ThreeCopySimilarity]
        self.plagiarism_pipeline = PlagiarismPipeline(self.plagiarism_classes)

    def translate_positions_to_states(self):
        hotkeys = self.mock_metagraph.hotkeys
        positions = self.position_manager.get_positions_for_hotkeys(hotkeys)
        flattened_positions = PositionUtils.flatten(positions)
        positions_list_translated = PositionUtils.translate_current_leverage(flattened_positions, evaluation_time_ms=self.current_time)
        miners, trade_pairs, state_list = PositionUtils.to_state_list(positions_list_translated, current_time=self.current_time)
        state_dict = self.plagiarism_pipeline.state_list_to_dict(miners, trade_pairs, state_list)


        PlagiarismEvents.set_positions(state_dict, miners, trade_pairs, current_time=self.current_time)


    def add_order_to_position_and_save_to_disk(self, position, order):
        position.add_order(order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

    def generate_one_position(self, hotkey, trade_pair, leverages, time_apart, time_after):
        self.position_counter += 1
        position1 = Position(
            miner_hotkey=hotkey,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + f"pos{self.position_counter}",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=trade_pair,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
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
                trade_pair=position1.trade_pair,
                processed_ms= (i * time_apart) + time_after,
                order_uuid=str(uuid.uuid4()))
            self.add_order_to_position_and_save_to_disk(position1, order)

    def generate_plagiarism_position(self, plagiarist_key, victim_key, time_after, victim_leverages, plagiarist_leverages, time_apart):

        self.generate_one_position(plagiarist_key[0], plagiarist_key[1], leverages=plagiarist_leverages, time_apart=time_apart, time_after=time_after)
        self.generate_one_position(victim_key[0], victim_key[1], leverages=victim_leverages, time_apart=time_apart, time_after=0)

    def test_lag_detection_not_similar(self):
        alternate = False
        for i in range(5):
            victim_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24),
                order_uuid=str(uuid.uuid4()))

            plagiarist_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24) + ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS + 1,
                order_uuid=str(uuid.uuid4()))

            # Alternate who is following so that lag threshold shouldn't be passed
            if alternate:
                self.add_order_to_position_and_save_to_disk(self.eth_position2, victim_order)
                self.add_order_to_position_and_save_to_disk(self.eth_position1, plagiarist_order)
            else:
                self.add_order_to_position_and_save_to_disk(self.eth_position1, victim_order)
                self.add_order_to_position_and_save_to_disk(self.eth_position2, plagiarist_order)
            alternate = not alternate

        self.translate_positions_to_states()


        miner_one_lag = LagDetection(self.MINER_HOTKEY1)
        miner_two_lag = LagDetection(self.MINER_HOTKEY2)

        victim_key_one = (self.MINER_HOTKEY2, TradePair.ETHUSD.name)

        miner_one_score = miner_one_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key_one)
        self.assertAlmostEqual(miner_one_score, 1)
        victim_key_two = (self.MINER_HOTKEY1, TradePair.ETHUSD.name)

        miner_two_score = miner_two_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key_two)
        self.assertAlmostEqual(miner_two_score, 1)

        PlagiarismEvents.clear_plagiarism_events()



    def test_lag_detection_plagiarism(self):

        for i in range(0, 5):
            victim_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24),
                order_uuid=str(uuid.uuid4()))

            plagiarist_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24) + 1000 * 60 * 60 * 3, # 3 hours after each other
                order_uuid=str(uuid.uuid4()))

            self.add_order_to_position_and_save_to_disk(self.eth_position1, victim_order)
            self.add_order_to_position_and_save_to_disk(self.eth_position2, plagiarist_order)

        self.translate_positions_to_states()

        miner_one_lag = LagDetection(self.MINER_HOTKEY1)
        miner_two_lag = LagDetection(self.MINER_HOTKEY2)

        miner_two_score = CopySimilarity.score_direct(self.MINER_HOTKEY2, TradePair.ETHUSD.name, self.MINER_HOTKEY1, TradePair.ETHUSD.name)
        self.assertGreaterEqual(miner_two_score, 0.95)
        miner_one_score = CopySimilarity.score_direct(self.MINER_HOTKEY1, TradePair.ETHUSD.name, self.MINER_HOTKEY2, TradePair.ETHUSD.name)
        self.assertGreater(miner_two_score, miner_one_score)

        victim_key_one = (self.MINER_HOTKEY2, TradePair.ETHUSD.name)
        #Consider what the threshold should really be for lag score
        miner_one_score = miner_one_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key_one)
        self.assertLessEqual(miner_one_score, 1)
        victim_key_two = (self.MINER_HOTKEY1, TradePair.ETHUSD.name)

        miner_two_score = miner_two_lag.score_direct(plagiarist_trade_pair=TradePair.ETHUSD.name, victim_key=victim_key_two)
        self.assertGreater(miner_two_score, miner_one_score)

        self.assertGreater(miner_two_score, 1)


        PlagiarismEvents.clear_plagiarism_events()

    def test_follow_similarity_plagiarism(self):
        for i in range(5):
            victim_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24),
                order_uuid=str(uuid.uuid4()))

            plagiarist_order = Order(order_type=OrderType.SHORT,
                leverage=-0.05,
                price=1000,
                trade_pair=TradePair.ETHUSD,
                processed_ms= (i * 1000 * 60 * 60 * 24) + 1000 * 60 * 30, #30 minutes after each other
                order_uuid=str(uuid.uuid4()))

            self.add_order_to_position_and_save_to_disk(self.eth_position1, victim_order)
            self.add_order_to_position_and_save_to_disk(self.eth_position2, plagiarist_order)

        self.translate_positions_to_states()
        miner_one_orders = PlagiarismEvents.positions[(self.MINER_HOTKEY1, TradePair.ETHUSD.name)]
        miner_two_orders = PlagiarismEvents.positions[(self.MINER_HOTKEY2, TradePair.ETHUSD.name)]

        miner_one_differences = FollowPercentage.compute_time_differences(plagiarist_orders=miner_one_orders, victim_orders=miner_two_orders)

        # Miner one is not following miner two
        self.assertCountEqual(miner_one_differences, [])
        average_time_lag_one = FollowPercentage.average_time_lag(differences=miner_one_differences)

        self.assertAlmostEqual(average_time_lag_one, 0)

        follow_percentage_one = FollowPercentage.compute_follow_percentage(victim_orders=miner_two_orders, differences=miner_one_differences)

        self.assertAlmostEqual(follow_percentage_one, 0)

        miner_two_differences = FollowPercentage.compute_time_differences(plagiarist_orders=miner_two_orders, victim_orders=miner_one_orders)
        # All follow times should be 30
        follow_ms = 30 * 1000 * 60

        for diff in miner_two_differences:
            self.assertAlmostEqual(diff, follow_ms/ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS)
        average_time_lag_two = FollowPercentage.average_time_lag(differences=miner_two_differences)

        self.assertAlmostEqual(average_time_lag_two, follow_ms/ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS)

        follow_percentage_two = FollowPercentage.compute_follow_percentage(victim_orders=miner_one_orders, differences=miner_two_differences)

        self.assertAlmostEqual(follow_percentage_two, 1)


        PlagiarismEvents.clear_plagiarism_events()

    def test_follow_similarity_outside(self):
        # Plagiarist follows outside of the order time window
        self.generate_plagiarism_position(plagiarist_key=(self.MINER_HOTKEY2, TradePair.AUDUSD),
                                          victim_key=(self.MINER_HOTKEY1, TradePair.AUDUSD),
                                          time_after=ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS + 1,
                                          victim_leverages= [-0.1 for x in range(5)],
                                          plagiarist_leverages=[-0.1 for x in range(5)],
                                          time_apart=1000 * 60 * 60 * 24 * 2) # 2 days apart

        self.translate_positions_to_states()
        miner_one_orders = PlagiarismEvents.positions[(self.MINER_HOTKEY1, TradePair.AUDUSD.name)]
        miner_two_orders = PlagiarismEvents.positions[(self.MINER_HOTKEY2, TradePair.AUDUSD.name)]

        miner_two_differences = FollowPercentage.compute_time_differences(plagiarist_orders=miner_two_orders, victim_orders=miner_one_orders)

        self.assertCountEqual(miner_two_differences, [])
        average_time_lag_two = FollowPercentage.average_time_lag(differences=miner_two_differences)

        self.assertAlmostEqual(average_time_lag_two, 0)

        follow_percentage_two = FollowPercentage.compute_follow_percentage(victim_orders=miner_one_orders, differences=miner_two_differences)

        self.assertAlmostEqual(follow_percentage_two, 0)

    def test_copy_similarity_plagiarism(self):
        victim_leverages = [-0.1, -0.15, 0.1, -0.1]
        plagiarist_leverages = victim_leverages

        self.generate_plagiarism_position(plagiarist_key=(self.MINER_HOTKEY2, TradePair.AUDCAD),
                                          victim_key=(self.MINER_HOTKEY1, TradePair.AUDCAD),
                                          time_after=ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS // 2,
                                          victim_leverages= victim_leverages,
                                          plagiarist_leverages= plagiarist_leverages,
                                          time_apart=int(ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS * 1.6))

        self.translate_positions_to_states()
        miner_one_orders = PlagiarismEvents.positions[(self.MINER_HOTKEY1, TradePair.AUDCAD.name)]
        miner_two_orders = PlagiarismEvents.positions[(self.MINER_HOTKEY2, TradePair.AUDCAD.name)]

        miner_two_score = CopySimilarity.score_direct(self.MINER_HOTKEY2, TradePair.AUDCAD.name, self.MINER_HOTKEY1, TradePair.AUDCAD.name)
        self.assertGreaterEqual(miner_two_score, 0.95)
        miner_one_score = CopySimilarity.score_direct(self.MINER_HOTKEY1, TradePair.AUDCAD.name, self.MINER_HOTKEY2, TradePair.AUDCAD.name)

        self.assertLess(miner_one_score, miner_two_score)

        miner_two_differences = FollowPercentage.compute_time_differences(plagiarist_orders=miner_two_orders, victim_orders=miner_one_orders)
        self.assertListEqual(miner_two_differences, [(ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS // 2) / ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS for _ in range(len(victim_leverages))])

        average_time_lag_two = FollowPercentage.average_time_lag(differences=miner_two_differences)

        self.assertAlmostEqual(average_time_lag_two * ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS, ValiConfig.PLAGIARISM_ORDER_TIME_WINDOW_MS // 2)

        follow_percentage_two = FollowPercentage.compute_follow_percentage(victim_orders=miner_one_orders, differences=miner_two_differences)

        self.assertAlmostEqual(follow_percentage_two, 1)

        miner_one_differences = FollowPercentage.compute_time_differences(plagiarist_orders=miner_one_orders, victim_orders=miner_two_orders)
        average_time_lag_one = FollowPercentage.average_time_lag(differences=miner_one_differences)
        self.assertAlmostEqual(average_time_lag_one, 0)

    def test_two_copy_similarity_plagiarism(self):
        victim_leverages = [-0.1, -0.2, 0.1, -0.1]
        victim_two_leverages = [-0.3, 0.1, 0.1, 0.05]

        # Plagiarist has the average of the cumulative leverage of two victims
        plagiarist_leverages = [-0.2, -0.05, 0.1, -0.025]

        self.generate_one_position(hotkey=self.MINER_HOTKEY1, trade_pair=TradePair.AUDCAD, leverages=victim_leverages, time_apart=1000 * 60 *60 * 24, time_after=0)

        self.generate_plagiarism_position(plagiarist_key=(self.MINER_HOTKEY3, TradePair.AUDCAD),
                                          victim_key=(self.MINER_HOTKEY2, TradePair.AUDCAD),
                                          plagiarist_leverages=plagiarist_leverages,
                                          victim_leverages=victim_two_leverages,
                                          time_apart=1000 * 60 *60 * 24,
                                          time_after=1000 * 60 * 60 * 3)

        self.translate_positions_to_states()

        two_copy_similarity = TwoCopySimilarity(self.MINER_HOTKEY3)
        two_copy_similarity.score_all(TradePair.AUDCAD.name)
        metadata = two_copy_similarity.summary()

        self.assertListEqual(sorted([self.MINER_HOTKEY1, self.MINER_HOTKEY2]), sorted([x["victim"] for x in metadata.values()]))

        for key, value in metadata.items():
            victim_id = value["victim"]
            if victim_id == self.MINER_HOTKEY1:
                self.assertGreaterEqual(value["score"], 0.8)
            if victim_id == self.MINER_HOTKEY2:
                self.assertGreaterEqual(value["score"], 0.8)

    def test_three_copy_similarity_plagiarism(self):

        victim_leverages = [-0.1, -0.2, 0.1, -0.1]
        victim_two_leverages = [-0.3, 0.1, 0.1, 0.05]
        victim_three_leverages = [-0.2, -0.05, 0.1, -0.025]
        # Plagiarist around 3 other victims
        plagiarist_leverages = [-0.2, -0.05, 0.1, -0.025]

        self.generate_one_position(hotkey=self.MINER_HOTKEY1, trade_pair=TradePair.AUDCAD, leverages=victim_leverages, time_apart=1000 * 60 *60 * 24, time_after=0)
        self.generate_one_position(hotkey=self.MINER_HOTKEY4, trade_pair=TradePair.AUDCAD, leverages=victim_three_leverages, time_apart=1000 * 60 *60 * 24, time_after=0)
        self.generate_plagiarism_position(plagiarist_key=(self.MINER_HOTKEY3, TradePair.AUDCAD),
                                          victim_key=(self.MINER_HOTKEY2, TradePair.AUDCAD),
                                          plagiarist_leverages=plagiarist_leverages,
                                          victim_leverages=victim_two_leverages,
                                          time_apart=1000 * 60 *60 * 24,
                                          time_after=1000 * 60 * 60 * 3)

        self.translate_positions_to_states()

        two_copy_similarity = ThreeCopySimilarity(self.MINER_HOTKEY3)
        two_copy_similarity.score_all(TradePair.AUDCAD.name)
        metadata = two_copy_similarity.summary()

        self.assertListEqual(sorted([self.MINER_HOTKEY1, self.MINER_HOTKEY2, self.MINER_HOTKEY4]), sorted([x["victim"] for x in metadata.values()]))

        for key, value in metadata.items():
            # Assert that all values are above 0.8 (They should all be the same since there are three available)
            self.assertGreaterEqual(value["score"], 0.8)



