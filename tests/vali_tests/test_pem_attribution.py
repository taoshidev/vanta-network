"""
Order.relay_hotkey: the permissionless entity miner that relayed a trader's order is recorded on every order the
signal creates, and carried to the orders that come from them later (limit and stop-limit fills, brackets, the
closes of a FLAT_ALL), and it survives serialization. Orders the miner sent itself carry None.
"""
import time
from unittest.mock import Mock

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.utils.order_processor import OrderProcessor
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_dataclasses.price_source import PriceSource

RELAY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


class TestOrderField(TestBase):

    def _order(self, **kwargs):
        return Order(trade_pair=TradePair.BTCUSDC, order_uuid="o-1", processed_ms=1_700_000_000_000, price=100.0,
                     order_type=OrderType.LONG, leverage=0.1, **kwargs)

    def test_defaults_to_none(self):
        self.assertIsNone(self._order().relay_hotkey)

    def test_round_trips_through_serialization(self):
        order = self._order(relay_hotkey=RELAY)
        self.assertEqual(order.to_python_dict()['relay_hotkey'], RELAY)
        self.assertEqual(Order.from_dict(order.to_python_dict()).relay_hotkey, RELAY)
        self.assertEqual(Order(**order.model_dump()).relay_hotkey, RELAY)

    def test_an_order_saved_before_the_field_loads_as_none(self):
        stored = self._order().to_python_dict()
        del stored['relay_hotkey']
        self.assertIsNone(Order.from_dict(stored).relay_hotkey)


class TestProcessorThreadsTheRelay(TestBase):
    """process_vanta_signal hands relay_hotkey to every order it creates."""

    NOW_MS = 1_700_000_000_000

    def setUp(self):
        super().setUp()
        self.limit_order_client = Mock()
        self.market_order_client = Mock()
        self.market_order_client.execute_order.return_value = None
        self.processor = OrderProcessor(limit_order_client=self.limit_order_client,
                                        market_order_client=self.market_order_client,
                                        miner_account_client=Mock())

    def _process(self, signal, order_uuid="o-1", relay_hotkey=RELAY):
        return self.processor.process_vanta_signal("trader", Signal(**signal), order_uuid, self.NOW_MS,
                                                   relay_hotkey=relay_hotkey)

    def test_market(self):
        self._process({"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1})
        self.assertEqual(self.market_order_client.execute_order.call_args.kwargs['relay_hotkey'], RELAY)

    def test_market_without_a_relay(self):
        self._process({"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1},
                      relay_hotkey=None)
        self.assertIsNone(self.market_order_client.execute_order.call_args.kwargs['relay_hotkey'])

    def test_flat_all(self):
        self._process({"execution_type": "FLAT_ALL"}, order_uuid="ALL")
        kwargs = self.market_order_client.close_positions.call_args.kwargs
        self.assertTrue(kwargs['close_all'])
        self.assertEqual(kwargs['relay_hotkey'], RELAY)

    def test_limit_stop_limit_and_bracket(self):
        for signal in (
            {"execution_type": "LIMIT", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1, "limit_price": 90.0},
            {"execution_type": "STOP_LIMIT", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1,
             "limit_price": 110.0, "stop_price": 105.0, "stop_condition": "GTE"},
            {"execution_type": "BRACKET", "trade_pair": "BTCUSDC", "stop_loss": 90.0},
        ):
            with self.subTest(execution_type=signal["execution_type"]):
                self._process(signal)
                order = self.limit_order_client.process_limit_order.call_args.args[1]
                self.assertEqual(order.relay_hotkey, RELAY)

    def test_an_edit_records_the_relay_of_the_edit(self):
        existing = Order(trade_pair=TradePair.BTCUSDC, order_uuid="o-1", processed_ms=self.NOW_MS, price=0.0,
                         order_type=OrderType.LONG, leverage=0.1, execution_type=ExecutionType.LIMIT, limit_price=90.0,
                         src=OrderSource.LIMIT_UNFILLED, relay_hotkey="the-first-relay")
        self.limit_order_client.get_limit_order_by_uuid.return_value = existing.to_python_dict()
        self._process({"execution_type": "LIMIT_EDIT", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1,
                       "limit_price": 95.0})
        edited = self.limit_order_client.process_limit_order.call_args.args[1]
        self.assertTrue(self.limit_order_client.process_limit_order.call_args.kwargs['is_edit'])
        self.assertEqual(edited.relay_hotkey, RELAY)

    def test_a_bulk_bracket_update(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = None
        self._process({"execution_type": "LIMIT_EDIT", "trade_pair": "BTCUSDC", "leverage": 1.0,
                       "bracket_orders": [{"order_uuid": "b-1", "stop_loss": 90.0}]}, order_uuid="unknown")
        order = self.limit_order_client.process_limit_order.call_args.args[1]
        self.assertEqual(order.relay_hotkey, RELAY)


class TestRelayCarriedThroughExecution(TestBase):
    """Against the real servers: fills, brackets, stop-limit children and FLAT_ALL closes keep the relay."""

    orchestrator = None
    MINER = "test_miner"
    TRADE_PAIR = TradePair.BTCUSDC  # not blocked, so the limit order check loop evaluates it

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=ValiUtils.get_secrets(running_unit_tests=True))
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.limit_order_client = cls.orchestrator.get_client('limit_order')
        cls.market_order_client = cls.orchestrator.get_client('market_order')

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys([self.MINER])

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    def _price_source(self, price):
        return PriceSource(source='test', timespan_ms=0, open=price, close=price, vwap=None, high=price, low=price,
                           start_ms=TimeUtil.now_in_millis(), websocket=True, lag_ms=100, bid=price, ask=price)

    def _limit_order(self, relay_hotkey=RELAY, **kwargs):
        fields = dict(trade_pair=self.TRADE_PAIR, order_uuid=f"limit-{TimeUtil.now_in_millis()}",
                      processed_ms=TimeUtil.now_in_millis(), price=0.0, order_type=OrderType.LONG, leverage=0.5,
                      execution_type=ExecutionType.LIMIT, limit_price=49000.0, src=OrderSource.LIMIT_UNFILLED,
                      relay_hotkey=relay_hotkey)
        fields.update(kwargs)
        return Order(**fields)

    def _resting_orders(self):
        by_miner = self.limit_order_client.get_limit_orders_for_trade_pair(self.TRADE_PAIR.trade_pair_id)
        return [Order.from_dict(o) if isinstance(o, dict) else o for o in by_miner.get(self.MINER, [])]

    def _positions(self):
        return self.position_client.get_positions_for_one_hotkey(self.MINER)

    def _fill(self, order):
        self.position_client.save_miner_position(Position(
            miner_hotkey=self.MINER, position_uuid=f"pos-{TimeUtil.now_in_millis()}", open_ms=TimeUtil.now_in_millis(),
            trade_pair=self.TRADE_PAIR, position_type=OrderType.LONG, account_size=1000.0))
        self.live_price_fetcher_client.set_test_price_source(self.TRADE_PAIR, None)
        self.limit_order_client.process_limit_order(self.MINER, order)
        self.assertEqual(self._resting_orders()[0].relay_hotkey, order.relay_hotkey)
        price_source = self._price_source(49000.0)
        self.live_price_fetcher_client.set_test_price_source(self.TRADE_PAIR, price_source)
        self.limit_order_client.fill_limit_order_with_price_source(self.MINER, order, price_source, 49000.0)
        return self._positions()[0].orders[-1]

    def test_a_limit_fill_keeps_the_relay(self):
        filled = self._fill(self._limit_order())
        self.assertEqual(filled.src, OrderSource.LIMIT_FILLED)
        self.assertEqual(filled.relay_hotkey, RELAY)

    def test_a_limit_fill_without_a_relay_has_none(self):
        self.assertIsNone(self._fill(self._limit_order(relay_hotkey=None)).relay_hotkey)

    def test_brackets_take_the_parents_relay(self):
        parent = Order(trade_pair=self.TRADE_PAIR, order_uuid="parent", processed_ms=TimeUtil.now_in_millis(),
                       price=50000.0, order_type=OrderType.LONG, quantity=0.1, execution_type=ExecutionType.MARKET,
                       stop_loss=49000.0, take_profit=51000.0, src=OrderSource.ORGANIC, relay_hotkey=RELAY)
        self.limit_order_client.create_sltp_order(self.MINER, parent)
        brackets = [o for o in self._resting_orders() if o.execution_type == ExecutionType.BRACKET]
        self.assertEqual(len(brackets), 1)
        self.assertEqual(brackets[0].relay_hotkey, RELAY)

    def test_a_triggered_stop_limit_passes_the_relay_to_its_limit_order(self):
        # EURUSD, as in test_stop_limit_orders, where the test price fetcher serves the check loop's sources
        trade_pair = TradePair.EURUSD
        order = self._limit_order(trade_pair=trade_pair, order_uuid="stop-1", execution_type=ExecutionType.STOP_LIMIT,
                                  stop_price=55000.0, stop_condition=StopCondition.GTE, limit_price=56000.0,
                                  src=OrderSource.STOP_LIMIT_UNFILLED)
        self.live_price_fetcher_client.set_test_price_source(trade_pair, None)
        self.limit_order_client.process_limit_order(self.MINER, order)
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(trade_pair, self._price_source(54000.0))
        self.limit_order_client.check_and_fill_limit_orders()
        # Above the stop to trigger it, and above the child's limit so the child rests rather than fills
        time.sleep(0.01)
        self.live_price_fetcher_client.set_test_price_source(trade_pair, self._price_source(57000.0))
        self.limit_order_client.check_and_fill_limit_orders()
        resting = self.limit_order_client.get_limit_orders_for_trade_pair(trade_pair.trade_pair_id).get(self.MINER, [])
        children = [Order.from_dict(o) if isinstance(o, dict) else o for o in resting]
        children = [o for o in children if o.order_uuid == "stop-1-limit"]
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].src, OrderSource.LIMIT_UNFILLED)
        self.assertEqual(children[0].relay_hotkey, RELAY)

    def test_flat_all_closes_carry_the_relay(self):
        self._fill(self._limit_order(relay_hotkey=None))
        self.live_price_fetcher_client.set_test_market_open(True)
        self.live_price_fetcher_client.set_test_price_source(self.TRADE_PAIR, self._price_source(49500.0))
        self.market_order_client.close_positions(hotkey=self.MINER, close_all=True, relay_hotkey=RELAY)
        position = self._positions()[0]
        self.assertTrue(position.is_closed_position)
        self.assertIsNone(position.orders[0].relay_hotkey)
        self.assertEqual(position.orders[-1].order_type, OrderType.FLAT)
        self.assertEqual(position.orders[-1].relay_hotkey, RELAY)
