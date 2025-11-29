import json
from copy import deepcopy

from bittensor import Balance

from shared_objects.mock_metagraph import MockMetagraph, MockNeuron, MockAxonInfo
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.p2p_syncer import P2PSyncer
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestPositions(TestBase):

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()  # 1718071209000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        self.default_order = Order(price=1, processed_ms=self.DEFAULT_OPEN_MS, order_uuid=self.DEFAULT_ORDER_UUID, trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=1)
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.default_neuron = MockNeuron(axon_info=MockAxonInfo("0.0.0.0"),
                                         stake=Balance(0.0))

        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY])
        self.elimination_manager = EliminationManager(self.mock_metagraph, None, None, running_unit_tests=True)
        self.position_manager = PositionManager(metagraph=self.mock_metagraph, running_unit_tests=True,
                                                elimination_manager=self.elimination_manager)
        self.position_manager.clear_all_miner_positions()
        self.elimination_manager.position_manager = self.position_manager

        self.default_open_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.default_closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_order],
            position_type=OrderType.FLAT,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_closed_position.close_out_position(self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 6)

        self.p2p_syncer = P2PSyncer(running_unit_tests=True, position_manager=self.position_manager)

    def test_get_validators(self):
        neuron1 = deepcopy(self.default_neuron)
        neuron1.stake = Balance(2000.0)
        neuron1.axon_info = MockAxonInfo(ip="test_ip1")

        neuron2 = deepcopy(self.default_neuron)
        neuron2.stake = Balance(0.0)
        neuron2.axon_info = MockAxonInfo(ip="test_ip2")

        neuron3 = deepcopy(self.default_neuron)
        neuron3.stake = Balance(2000.0)

        neuron4 = deepcopy(self.default_neuron)
        neuron4.stake = Balance(0.0)

        self.neurons = [neuron1, neuron2, neuron3, neuron4]
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY], self.neurons)
        validator_axons = self.p2p_syncer.get_validators(self.mock_metagraph.neurons)

        assert len(validator_axons) == 1

    def test_get_largest_staked_validators(self):
        neuron1 = deepcopy(self.default_neuron)
        neuron1.stake = Balance(2000.0)
        neuron1.axon_info = MockAxonInfo(ip="test_ip1")

        neuron2 = deepcopy(self.default_neuron)
        neuron2.stake = Balance(0.0)
        neuron2.axon_info = MockAxonInfo(ip="test_ip2")

        neuron3 = deepcopy(self.default_neuron)
        neuron3.stake = Balance(2000.0)

        neuron4 = deepcopy(self.default_neuron)
        neuron4.stake = Balance(0.0)

        neuron5 = deepcopy(self.default_neuron)
        neuron5.stake = Balance(3000.0)
        neuron5.axon_info = MockAxonInfo(ip="test_ip5")

        neuron6 = deepcopy(self.default_neuron)
        neuron6.stake = Balance(4000.0)
        neuron6.axon_info = MockAxonInfo(ip="test_ip6")

        self.neurons = [neuron1, neuron2, neuron3, neuron4, neuron5, neuron6]
        self.mock_metagraph = MockMetagraph([self.DEFAULT_MINER_HOTKEY], self.neurons)
        validator_axons = self.p2p_syncer.get_largest_staked_validators(2, self.mock_metagraph.neurons)

        assert len(validator_axons) == 2
        assert validator_axons[0].ip == "test_ip6"

    def test_checkpoint_syncing_order_with_median_price(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.price = 1.0
        orders = [order1]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.price = 2.0
        orders = [order1]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.price = 3.0
        orders = [order1]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        # print(json.dumps(checkpoint1, indent=4))

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}

        self.p2p_syncer.create_golden(checkpoints)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1
        assert self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"][0]["price"] == 2.0

    def test_checkpoint_syncing_order_not_in_majority(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order0, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2], "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_checkpoint_syncing_order_not_in_majority_with_multiple_positions(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order0, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        orders = [order3]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string()), json.loads(position2.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        orders = [order1, order2]
        position = deepcopy(self.default_position)
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order4"
        orders = [order3]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string()), json.loads(position2.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2], "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][1]["orders"]) == 2

    def test_checkpoint_syncing_position_not_in_majority(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        order0.processed_ms = TimeUtil.now_in_millis() - 1000 * 60 * 10
        orders = [order0]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position2"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_multiple_positions(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string()), json.loads(position2.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position2.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_multiple_miners(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}
        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position2"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {"diff_miner": {"positions": [json.loads(position.to_json_string())]}}}
        checkpoint4 = {"positions": {"diff_miner": {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3], "test_validator4": [0, checkpoint4]}
        self.p2p_syncer.create_golden(checkpoints)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"]["diff_miner"]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1
        assert len(self.p2p_syncer.golden["positions"]["diff_miner"]["positions"][0]["orders"]) == 1

    def test_checkpoint_syncing_one_of_each_miner(self):
        # TODO
        pass

    def test_checkpoint_syncing_miner_not_in_majority(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        orders = [order0]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position2"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {"diff_miner": {"positions": [json.loads(position.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        orders = [order1]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        checkpoints = {"test_validator1": [0, checkpoint1], "test_validator2": [0, checkpoint2],
                       "test_validator3": [0, checkpoint3]}
        self.p2p_syncer.create_golden(checkpoints)

        # print(json.dumps(self.p2p_syncer.golden, indent=4))

        assert len(self.p2p_syncer.golden["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_heuristic_resolve_positions(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = 100
        orders = [order1]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = 110
        orders = [order2]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        order3.processed_ms = 90
        orders = [order3]
        position3 = deepcopy(self.default_position)
        position3.position_uuid = "test_position3"
        position3.orders = orders
        position3.rebuild_position_with_updated_orders(self.live_price_fetcher)

        order4 = deepcopy(self.default_order)
        order4.order_uuid = "test_order4"
        order4.processed_ms = TimeUtil.now_in_millis() - 1000 * 60 * 10
        orders = [order4]
        position4 = deepcopy(self.default_position)
        position4.position_uuid = "test_position4"
        position4.orders = orders
        position4.rebuild_position_with_updated_orders(self.live_price_fetcher)

        matrix = {'miner_hotkey_1': {self.DEFAULT_TRADE_PAIR: {'validator_hotkey_1': [json.loads(position1.to_json_string())], 'validator_hotkey_2': [json.loads(position2.to_json_string())]}},
                  'miner_hotkey_2': {self.DEFAULT_TRADE_PAIR: {'validator_hotkey_3': [json.loads(position3.to_json_string())], 'validator_hotkey_4': [json.loads(position4.to_json_string())]}}}

        matched_positions = self.p2p_syncer.heuristic_resolve_positions(matrix, 2, set())

        assert len(matched_positions) == 1
        assert matched_positions[0]["position_uuid"] == "test_position1"

    def test_checkpoint_last_order_time(self):
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = 100
        order0 = deepcopy(self.default_order)
        order0.order_uuid = "test_order0"
        order0.processed_ms = 50
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order1"
        order2.processed_ms = 150
        orders = [order1, order2, order0]
        position = deepcopy(self.default_position)
        position.position_uuid = "test_position1"
        position.orders = orders
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position.to_json_string())]}}}

        assert self.p2p_syncer.last_order_time_in_checkpoint(checkpoint) == 150

    def test_find_one_legacy_miners(self):
        num_checkpoints = 2
        order_counts = {"position 1": {"order 1": 2}, "position 2": {"order 2": 1}, "position 2 also": {"order 2 also": 1}}
        miner_to_uuids = {"updated miner": {"positions": ["position 1"], "orders":["order 1"]}, "legacy miner": {"positions": ["position 2", "position 2 also"], "orders": ["order 2", "order 2 also"]}}
        position_counts = {"position 1": 2, "position 2": 1, "position 2 also": 1}
        order_data = {"order 1": [{"processed_ms": 100}, {"processed_ms": 100}], "order 2": [{"processed_ms": 150}], "order 2 also": [{"processed_ms": 150}]}

        legacy_miners = self.p2p_syncer.find_legacy_miners(num_checkpoints, order_counts, miner_to_uuids, position_counts, order_data)

        assert len(legacy_miners) == 1
        assert "legacy miner" in legacy_miners

    def test_find_zero_legacy_miners(self):
        num_checkpoints = 2
        order_counts = {"position 1": {"order 1": 2}, "position 2": {"order 2": 1}}
        miner_to_uuids = {"updated miner": {"positions": ["position 1"], "orders": ["order 1"]},
                          "legacy miner": {"positions": ["position 2"], "orders": ["order 2"]}}
        position_counts = {"position 1": 2, "position 2": 2}
        order_data = {"order 1": [{"processed_ms": 100}, {"processed_ms": 100}], "order 2": [{"processed_ms": 150}, {"processed_ms": 150}]}

        legacy_miners = self.p2p_syncer.find_legacy_miners(num_checkpoints, order_counts, miner_to_uuids,
                                                           position_counts, order_data)

        assert len(legacy_miners) == 0

    def test_position_with_mixed_order_uuids(self):
        """
        Heuristically combine positions with different position_uuids.
        Match the positions on a order-by-order basis, based on heuristics or order_uuid.
        """
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = 1000
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = 2000
        order2.leverage = 0.5
        order2.order_type = "LONG"
        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        order3.leverage = 0.8
        order3.processed_ms = TimeUtil.now_in_millis()
        orders = [order1, order2, order3]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order2x = deepcopy(self.default_order)
        order2x.order_uuid = "test_order2x"
        order2x.processed_ms = 2000
        order2x.leverage = 0.5
        order2x.order_type = "LONG"
        order3 = deepcopy(self.default_order)
        order3.order_uuid = "test_order3"
        order3.leverage = 0.8
        order3.processed_ms = TimeUtil.now_in_millis()
        orders = [order2x, order3]
        position1x = deepcopy(self.default_position)
        position1x.position_uuid = "test_position1x"
        position1x.orders = orders
        position1x.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1x.to_json_string())]}}}

        order3x = deepcopy(self.default_order)
        order3x.order_uuid = "test_order3x"
        order3x.leverage = 0.8
        order3x.processed_ms = TimeUtil.now_in_millis()
        orders = [order3x]
        position1y = deepcopy(self.default_position)
        position1y.position_uuid = "test_position1y"
        position1y.orders = orders
        position1y.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1y.to_json_string())]}}}

        checkpoints = {"test_validator1": [1, checkpoint1], "test_validator2": [1, checkpoint2],"test_validator3": [1, checkpoint3]}

        # print(checkpoints)
        self.p2p_syncer.create_golden(checkpoints)

        # print(self.p2p_syncer.golden)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_order_duplicated_across_multiple_positions(self):
        """
        An order could be duplicated across position A and B on a single validator.
        Heuristic matches have a higher threshold, so must show up on at least >50% of validators.
        """
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = 1000
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = TimeUtil.now_in_millis()
        order2.leverage = 0.5
        order2.order_type = "LONG"
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        orders = [order2]
        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = orders
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string()), json.loads(position2.to_json_string())]}}}

        orders = [order1, order2]
        position1x = deepcopy(self.default_position)
        position1x.position_uuid = "test_position1x"
        position1x.orders = orders
        position1x.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1x.to_json_string())]}}}

        orders = [order1, order2]
        position1y = deepcopy(self.default_position)
        position1y.position_uuid = "test_position1y"
        position1y.orders = orders
        position1y.rebuild_position_with_updated_orders(self.live_price_fetcher)

        orders = [order2]
        position2x = deepcopy(self.default_position)
        position2x.position_uuid = "test_position2x"
        position2x.orders = orders
        position2x.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1y.to_json_string()), json.loads(position2x.to_json_string())]}}}

        orders = [order1, order2]
        position1z = deepcopy(self.default_position)
        position1z.position_uuid = "test_position1z"
        position1z.orders = orders
        position1z.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint4 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1z.to_json_string())]}}}

        orders = [order1, order2]
        position1a = deepcopy(self.default_position)
        position1a.position_uuid = "test_position1a"
        position1a.orders = orders
        position1a.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint5 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1a.to_json_string())]}}}

        checkpoints = {"test_validator1": [1, checkpoint1], "test_validator2": [1, checkpoint2],
                       "test_validator3": [1, checkpoint3], "test_validator4": [1, checkpoint4],
                       "test_validator5": [1, checkpoint5]}

        # print(checkpoints)
        self.p2p_syncer.create_golden(checkpoints)

        # print(self.p2p_syncer.golden)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_order_heuristic_matched_in_same_position(self):
        """
        Heuristic matching for different order_uuid with same position_uuid
        """
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1x"
        order1.processed_ms = 1000
        order1.leverage = 0.5
        order1.order_type = "LONG"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = TimeUtil.now_in_millis()
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1y"
        order1.processed_ms = 1010
        order1.leverage = 0.5
        order1.order_type = "LONG"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = TimeUtil.now_in_millis()
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {"positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1z"
        order1.processed_ms = 990
        order1.leverage = 0.5
        order1.order_type = "LONG"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = TimeUtil.now_in_millis()
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        checkpoints = {"test_validator1": [1, checkpoint1], "test_validator2": [1, checkpoint2],"test_validator3": [1, checkpoint3]}

        # print(checkpoints)
        self.p2p_syncer.create_golden(checkpoints)

        # print(self.p2p_syncer.golden)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_order_heuristic_matched_in_timebound(self):
        """
        same position_uuid, multiple diff order_uuids within timebound_ms of each other. Each should be a separate matched order.
        """
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1x"
        order1.processed_ms = TimeUtil.now_in_millis() - 1000
        order1.leverage = 0.5
        order1.order_type = "LONG"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2x"
        order2.processed_ms = TimeUtil.now_in_millis()
        order2.leverage = 0.5
        order2.order_type = "LONG"
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1y"
        order1.processed_ms = TimeUtil.now_in_millis() - 1000
        order1.leverage = 0.5
        order1.order_type = "LONG"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2y"
        order2.leverage = 0.5
        order2.order_type = "LONG"
        order2.processed_ms = TimeUtil.now_in_millis()
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1z"
        order1.processed_ms = TimeUtil.now_in_millis() - 1000
        order1.leverage = 0.5
        order1.order_type = "LONG"
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2z"
        order2.leverage = 0.5
        order2.order_type = "LONG"
        order2.processed_ms = TimeUtil.now_in_millis()
        orders = [order1, order2]
        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = orders
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint3 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        checkpoints = {"test_validator1": [1, checkpoint1], "test_validator2": [1, checkpoint2],
                       "test_validator3": [1, checkpoint3]}

        # print(checkpoints)
        self.p2p_syncer.create_golden(checkpoints)

        print(self.p2p_syncer.golden)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 1
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 2

    def test_positions_split_up_on_some_validators(self):
        """
        multiple positions may be separate on some validators and combined on others.
        ex: if a validator misses a flat order.
        Ensure order is only in one position.
        """
        order1 = deepcopy(self.default_order)
        order1.order_uuid = "test_order1"
        order1.processed_ms = TimeUtil.now_in_millis() - 1000
        order1.leverage = 0.1
        order2 = deepcopy(self.default_order)
        order2.order_uuid = "test_order2"
        order2.processed_ms = TimeUtil.now_in_millis()
        order2.leverage = 0.2


        position1 = deepcopy(self.default_position)
        position1.position_uuid = "test_position1"
        position1.orders = [order1, order2]
        position1.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint1 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1.to_json_string())]}}}

        position1x = deepcopy(self.default_position)
        position1x.position_uuid = "test_position1"
        position1x.orders = [order1]
        position1x.rebuild_position_with_updated_orders(self.live_price_fetcher)

        position2 = deepcopy(self.default_position)
        position2.position_uuid = "test_position2"
        position2.orders = [order2]
        position2.rebuild_position_with_updated_orders(self.live_price_fetcher)

        checkpoint2 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1x.to_json_string()), json.loads(position2.to_json_string())]}}}

        checkpoint3 = {
            "positions": {self.DEFAULT_MINER_HOTKEY: {"positions": [json.loads(position1x.to_json_string()), json.loads(position2.to_json_string())]}}}

        checkpoints = {"test_validator1": [1, checkpoint1], "test_validator2": [1, checkpoint2],
                       "test_validator3": [1, checkpoint3]}

        self.p2p_syncer.create_golden(checkpoints)

        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"]) == 2
        assert len(self.p2p_syncer.golden["positions"][self.DEFAULT_MINER_HOTKEY]["positions"][0]["orders"]) == 1

    def test_sync_challengeperiod(self):
        checkpoint1 = {"challengeperiod": {"testing": {"miner1": 100, "miner2": 105},
                                           "success": {"miner3": 120, "miner4": 110}}}
        checkpoint2 = {"challengeperiod": {"testing": {"miner1": 100, "miner2": 105},
                                           "success": {"miner3": 130, "miner5": 110}}}
        checkpoint3 = {"challengeperiod": {"testing": {"miner1": 100, "miner6": 165},
                                           "success": {"miner4": 100, "miner5": 110}}}

        checkpoints = {"test_validator1": [1, checkpoint1], "test_validator2": [1, checkpoint2], "test_validator3": [1, checkpoint3]}

        self.p2p_syncer.create_golden(checkpoints)
        assert len(self.p2p_syncer.golden["challengeperiod"]["testing"]) == 2
        assert len(self.p2p_syncer.golden["challengeperiod"]["success"]) == 3
        assert self.p2p_syncer.golden["challengeperiod"]["testing"]["miner1"] == 100
        assert self.p2p_syncer.golden["challengeperiod"]["testing"]["miner2"] == 105
        assert self.p2p_syncer.golden["challengeperiod"]["success"]["miner3"] == 120
        assert self.p2p_syncer.golden["challengeperiod"]["success"]["miner4"] == 100
        assert self.p2p_syncer.golden["challengeperiod"]["success"]["miner5"] == 110






