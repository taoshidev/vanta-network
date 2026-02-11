# developer: jbonilla
import datetime
import json
from copy import deepcopy

from data_generator.polygon_data_service import PolygonDataService
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import (
    CRYPTO_CARRY_FEE_PER_INTERVAL,
    FEE_V6_TIME_MS,
    FOREX_CARRY_FEE_PER_INTERVAL,
    INDICES_CARRY_FEE_PER_INTERVAL,
    Position,
)
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import (
    Order,
)
from vali_objects.enums.order_source_enum import OrderSource


class TestPositions(TestBase):
    """
    Position tests using ServerOrchestrator.

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
    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_POSITION_UUID = "test_position"
    DEFAULT_OPEN_MS = 1000
    DEFAULT_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_ACCOUNT_SIZE = 100_000
    default_position = Position(
        miner_hotkey=DEFAULT_MINER_HOTKEY,
        position_uuid=DEFAULT_POSITION_UUID,
        open_ms=DEFAULT_OPEN_MS,
        trade_pair=DEFAULT_TRADE_PAIR,
        account_size=DEFAULT_ACCOUNT_SIZE,
    )

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
        cls.position_client = cls.orchestrator.get_client('position_manager')

        # Initialize metagraph with test miner
        cls.metagraph_client.set_hotkeys([cls.DEFAULT_MINER_HOTKEY])

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

        # Create fresh test data for this test
        self._create_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data."""
        pass

    # Aliases for backward compatibility with test methods
    @property
    def live_price_fetcher(self):
        """Alias for class-level live_price_fetcher_client."""
        return self.live_price_fetcher_client

    @property
    def position_manager(self):
        """Alias for class-level position_client (provides same interface)."""
        return self.position_client

    def add_order_to_position_and_save(self, position, order):
        position.add_order(order, self.live_price_fetcher, self.position_manager.calculate_net_portfolio_leverage(self.DEFAULT_MINER_HOTKEY))
        self.position_manager.save_miner_position(position)

    def _find_disk_position_from_memory_position(self, position):
        for disk_position in self.position_manager.get_positions_for_one_hotkey(position.miner_hotkey):
            if disk_position.position_uuid == position.position_uuid:
                return disk_position
        raise ValueError(f"Could not find position {position.position_uuid} in disk")

    def validate_intermediate_position_state(self, in_memory_position, expected_state):
        disk_position = self._find_disk_position_from_memory_position(in_memory_position)
        success, reason = PositionManagerClient.positions_are_the_same(in_memory_position, expected_state)
        self.assertTrue(success, "In memory position is not as expected. " + reason)
        success, reason = PositionManagerClient.positions_are_the_same(disk_position, expected_state)
        self.assertTrue(success, "Disc position is not as expected. " + reason)

    def test_profit_position_returns_pre_post_slippage(self):
        """
        The post slippage position returns calculations calculates a realized PnL and an unrealized PnL.
        The pre slippage position returns calculation only calculates returns from the weighted avg entry price and the current price.

        If we set the actual slippage to 0, these returns calculations should be the same.
        """
        import vali_objects.vali_dataclasses.position as position_file
        position_file.ALWAYS_USE_SLIPPAGE = False

        open_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="open_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=1,
        )
        reduce_size_order = Order(
            price=110,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.SHORT,
            leverage=0.5,
        )
        increase_size_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 2000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=1,
        )
        close_order = Order(
            price=110,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 3000,
            order_uuid="close_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.FLAT,
            leverage=0,
        )
        closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.EURUSD,
            orders=[],
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        closed_position.add_order(open_order, self.live_price_fetcher)
        closed_position.add_order(reduce_size_order, self.live_price_fetcher)
        closed_position.add_order(increase_size_order, self.live_price_fetcher)
        closed_position.add_order(close_order, self.live_price_fetcher)

        old_returns_calc = closed_position.current_return

        position_file.ALWAYS_USE_SLIPPAGE = True

        closed_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        new_returns_calc = closed_position.current_return

        assert old_returns_calc == new_returns_calc
        position_file.ALWAYS_USE_SLIPPAGE = None

    def test_loss_position_returns_pre_post_slippage(self):
        """
        The post slippage position returns calculations calculates a realized PnL and an unrealized PnL.
        The pre slippage position returns calculation only calculates returns from the weighted avg entry price and the current price.

        If we set the actual slippage to 0, these returns calculations should be the same.
        """
        import vali_objects.vali_dataclasses.position as position_file
        position_file.ALWAYS_USE_SLIPPAGE = False

        open_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="open_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.SHORT,
            leverage=1,
        )
        reduce_size_order = Order(
            price=110,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=0.5,
        )
        increase_size_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 2000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.SHORT,
            leverage=1,
        )
        close_order = Order(
            price=110,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 3000,
            order_uuid="close_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.FLAT,
            leverage=0,
        )
        closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.EURUSD,
            orders=[],
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        closed_position.add_order(open_order, self.live_price_fetcher)
        closed_position.add_order(reduce_size_order, self.live_price_fetcher)
        closed_position.add_order(increase_size_order, self.live_price_fetcher)
        closed_position.add_order(close_order, self.live_price_fetcher)

        old_returns_calc = closed_position.current_return

        position_file.ALWAYS_USE_SLIPPAGE = True

        closed_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        new_returns_calc = closed_position.current_return

        assert old_returns_calc == new_returns_calc
        position_file.ALWAYS_USE_SLIPPAGE = None

    def test_position_returns_across_slippage_boundary(self):
        """
        Calculates the returns for a position opened before slippage and closed after slippage
        """
        open_order = Order(
            price=152.053,
            slippage=0.00,
            processed_ms=1739929944096,
            order_uuid="open_order",
            trade_pair=TradePair.USDJPY,
            order_type=OrderType.SHORT,
            leverage=-3,
        )
        close_order = Order(
            price=151.821,
            slippage=1.6840600345420394e-05,
            processed_ms=1739938331996,
            order_uuid="close_order",
            trade_pair=TradePair.USDJPY,
            order_type=OrderType.FLAT,
            leverage=3,
        )
        closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.USDJPY,
            orders=[],
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        closed_position.add_order(open_order, self.live_price_fetcher)
        closed_position.add_order(close_order, self.live_price_fetcher)
        assert closed_position.current_return == 1.0045338242380495

    def test_position_returns_one_order(self):
        """
        Calculate and update the returns for a position with a single order.
        """
        open_order = Order(
            price=100,
            slippage=0.01,
            processed_ms=1742910011691,
            order_uuid="open_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.SHORT,
            leverage=-0.1,
        )
        open_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=1742910011691,
            trade_pair=TradePair.BTCUSD,
            orders=[],
            net_leverage=-0.1,
            average_entry_price=100,
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        open_position.add_order(open_order, self.live_price_fetcher)
        assert open_position.current_return == 1

        open_position.set_returns(90, price_fetcher_client=self.live_price_fetcher)
        r1 = open_position.current_return
        assert r1 != 1.0

        open_position.set_returns(80, price_fetcher_client=self.live_price_fetcher)
        r2 = open_position.current_return
        assert r2 != 1.0
        assert r1 < r2

    def test_maximum_leverage_in_interval_monotone_increasing(self):
        position = deepcopy(self.default_position)
        position.orders = []
        for i in range(10):
            o = Order(order_type=OrderType.LONG,
                      leverage=.1 + i / 10,
                      price=100,
                      trade_pair=TradePair.BTCUSD,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i))
            position.orders.append(o)
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        # Test various intervals
        test_intervals = [
            (1000, 1001, 0.1),  # 0.1
            (1000, 1010, 0.3),  # 0.1 + 0.2
            (1000, 1020, 0.6),  # 0.1 + 0.2 + 0.3
            (1000, 1030, 1.0),  # 0.1 + 0.2 + 0.3 + 0.4
            (1000, 1040, 1.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5
            (1000, 1050, 2.1),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6
            (1000, 1060, 2.8),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7
            (1000, 1070, 3.6),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8
            (1000, 1080, 4.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9
            (1000, 1090, 5.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9 + 1.0
            (1000, 1100, 5.5),  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9 + 1.0

            (1085, 1085, 4.5),  # zero interval length is still valid since we check inclusive.
            (1090, 1090, 5.5),  # zero interval length is still valid since we check inclusive.

            (1010, 1020, 0.6),  # max at 1020 unchanged
            (1020, 1030, 1.0),  # max at ... unchanged
            (1030, 1040, 1.5),  # max at ... unchanged
            (1040, 1050, 2.1),  # max at ... unchanged
            (1050, 1060, 2.8),  # max at ... unchanged
            (1060, 1070, 3.6),  # max at ... unchanged
            (1070, 1080, 4.5),  # max at ... unchanged
            (1080, 1090, 5.5),  # 1090 unchanged
            (1090, 1100, 5.5),  # 1090 unchanged

            (1100, 1150, 5.5),
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage
            (1500, 1500, 5.5),
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage

        ]

        for start, end, expected_leverage in test_intervals:
            msg = f"start: {start}, end: {end}, expected_leverage: {expected_leverage}"
            self.assertAlmostEqual(position.max_leverage_seen_in_interval(start, end), expected_leverage, 7, msg)

        # throw an exception for invalid interval
        invalid_intervals = [
            (900, 950),  # Interval before any order timestamps
            (1050, 1000),  # End timestamp smaller than start timestamp
            (-100, 1000),  # Negative start timestamp
            (1000, -100),  # Negative end timestamp
        ]
        for start, end in invalid_intervals:
            msg = f"start: {start}, end: {end}"
            with self.assertRaises(ValueError, msg=msg):
                _ = position.max_leverage_seen_in_interval(start, end)

    def test_maximum_leverage_in_interval_ups_and_downs(self):
        position = deepcopy(self.default_position)
        position.orders = []
        for i in range(10):
            if i % 2 == 0:
                lev = -1
            else:
                lev = 0.5

            ot = OrderType.LONG if lev > 0 else OrderType.SHORT
            o = Order(order_type=ot,
                      leverage=lev,
                      price=100,
                      trade_pair=TradePair.BTCUSD,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i))
            position.orders.append(o)
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        # Test various intervals
        test_intervals = [
            (1000, 1001, 1),  # -1 = -1
            (1000, 1010, 1),  # -1 + 0.5 = -0.5
            (1000, 1020, 1.5),  # -1 + 0.5 - 1 = -1.5
            (1000, 1030, 1.5),  # -1 + 0.5 - 1 + 0.5 = -1
            (1000, 1040, 2.0),  # -1 + 0.5 - 1 + 0.5 - 1 = -2
            (1000, 1050, 2.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -1.5
            (1000, 1060, 2.5),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 = -2.5
            (1000, 1070, 2.5),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -2
            (1000, 1080, 3.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 = -3
            (1000, 1090, 3.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -2.5
            (1000, 1100, 3.0),  # -1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 - 1 + 0.5 = -2.5

            (1085, 1085, 3.0),  # zero interval length is still valid since we check inclusive.
            (1090, 1090, 2.5),  # zero interval length is still valid since we check inclusive.

            (1010, 1020, 1.5),  # max at 1020 unchanged
            (1020, 1030, 1.5),  # max at ... unchanged
            (1030, 1040, 2.0),  # max at ... unchanged
            (1040, 1050, 2.0),  # max at ... unchanged
            (1050, 1060, 2.5),  # max at ... unchanged
            (1060, 1070, 2.5),  # max at ... unchanged
            (1070, 1080, 3.0),  # max at ... unchanged
            (1080, 1090, 3.0),  # 1090 unchanged
            (1090, 1100, 2.5),  # 1090 unchanged

            (1100, 1150, 2.5),
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage
            (1500, 1500, 2.5),
            # Interval after any order timestamps but the position hasn't closed so it is the most recent leverage

        ]

        for start, end, expected_leverage in test_intervals:
            msg = f"start: {start}, end: {end}, expected_leverage: {expected_leverage}"
            self.assertAlmostEqual(position.max_leverage_seen_in_interval(start, end), expected_leverage, 7, msg)

    def test_simple_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=110,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + MS_IN_8_HOURS + 1000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)

        net_value = 1.0 * ValiConfig.DEFAULT_CAPITAL
        net_quantity = net_value / o1.price

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'net_value': net_value,
            'net_quantity': net_quantity,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 10000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0976837374307222,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

        self.assertEqual(position.get_carry_fee(o2.processed_ms)[0], CRYPTO_CARRY_FEE_PER_INTERVAL)

    def test_carry_fee_edge_case(self):
        """
        assert next_update_time_ms > current_time_ms,
        [TimeUtil.millis_to_verbose_formatted_date_str(x) for x in
         (self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms)] + [carry_fee, position]
        AssertionError: ['2024-07-24 04:00:00.000', '2024-07-24 04:00:00.000', '2024-07-24 04:00:00.000', 0.9999979876994218,
         Position(miner_hotkey='5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X',
          position_uuid='6955409a-031e-47df-8614-4488208497a6',
          open_ms=1721228840870,
          trade_pair=<TradePair.BTCUSD: ['BTCUSD', 'BTC/USD', 0.001, 0.001, 20, <TradePairCategory.CRYPTO: 'crypto'>]>,
          orders=[Order(trade_pair=<TradePair.BTCUSD: ['BTCUSD', 'BTC/USD', 0.001, 0.001, 20, <TradePairCategory.CRYPTO: 'crypto'>]>,
                    order_type=<OrderType.LONG: 'LONG'>, leverage=0.001, price=64900.41, processed_ms=1721228840870,
                    order_uuid='6955409a-031e-47df-8614-4488208497a6', price_sources=[])],
           current_return=1.0000177396413983,
           close_ms=None,
           return_at_close=1.0000152272972591,
            net_leverage=0.001,
            average_entry_price=64900.41,
            position_type=<OrderType.LONG: 'LONG'>, is_closed_position=False)]
        """
        timestamp_ms_july_24_2024_4am = datetime.datetime(2024, 7, 24, 4, 0, 0,
                                                          tzinfo=datetime.timezone.utc).timestamp() * 1000
        timestamp_ms_july_24_2024_4am = int(timestamp_ms_july_24_2024_4am)
        t0 = 1721228840870  # Wednesday, July 17, 2024 3:07:20.870 PM
        position = Position(
            open_ms=t0,
            miner_hotkey='5EUTaAo7vCGxvLDWRXRrEuqctPjt9fKZmgkaeFZocWECUe9X',
            position_uuid='6955409a-031e-47df-8614-4488208497a6',
            trade_pair=TradePair.BTCUSD,
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=t0,
                   order_uuid="1000")
        position.add_order(o1, self.live_price_fetcher)
        carry_fee, next_update_time_ms = position.get_carry_fee(timestamp_ms_july_24_2024_4am)
        self.assertNotEqual(next_update_time_ms, timestamp_ms_july_24_2024_4am)
        self.assertEqual(next_update_time_ms, timestamp_ms_july_24_2024_4am + MS_IN_8_HOURS,
                         msg=next_update_time_ms - timestamp_ms_july_24_2024_4am)

    def test_simple_long_position_with_implicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=2.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + 10 * MS_IN_8_HOURS,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)

        net_value = 1.0 * ValiConfig.DEFAULT_CAPITAL
        net_quantity = net_value / o1.price

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'net_value': net_value,
            'net_quantity': net_quantity,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 100000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.993887142229985,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

        self.assertAlmostEqual(position.get_carry_fee(o2.processed_ms)[0], CRYPTO_CARRY_FEE_PER_INTERVAL ** 10, 7)

    def test_simple_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + 1,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=90,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS + 3 * MS_IN_8_HOURS + 1000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        self.assertAlmostEqual(position.get_carry_fee(o1.processed_ms)[0], 1.0)

        net_value = -1.0 * ValiConfig.DEFAULT_CAPITAL
        net_quantity = net_value / o1.price

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0,
            'net_value': net_value,
            'net_quantity': net_quantity,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': 10000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.0974512492292436 ,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)
        self.assertAlmostEqual(position.get_carry_fee(o2.processed_ms)[0], CRYPTO_CARRY_FEE_PER_INTERVAL ** 3)

    def test_liquidated_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 200000.0,
            'net_quantity': 2000.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': -100000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 200000.0,
            'net_quantity': 2000.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 2.0)
        self.assertEqual(position.get_cumulative_leverage(), 4.0)
        self.assertAlmostEqual(position.get_carry_fee(o2.processed_ms)[0], 1.0)

    def test_liquidated_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': -100000.0,
            'net_quantity': -1000.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': -8900000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': -100000.0,
            'net_quantity': -1000.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

    def test_liquidated_short_position_with_no_FLAT(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=.1,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.LONG,
                   leverage=.1,
                   price=9000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")

        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': -100000.0,
            'net_quantity': -1000.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        assert len(position.orders) == 3, position.orders
        assert position.orders[2].src == OrderSource.PRICE_FILLED_ELIMINATION_FLAT
        self.assertGreater(position.orders[2].price_sources[0].lag_ms,
                           1761281990000)  # The lag is high. now_ms - DEFAULT_TESTING_FALLBACK_PRICE_SOURCE.start_ms
        position.orders[2].price_sources[0].lag_ms = PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE.lag_ms
        self.assertEqual(position.orders[2].price_sources, [PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE])

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, position.orders[2]],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': -9888.888888888889,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': -8890111.111111112,
            'net_value': -100000.0,
            'net_quantity': -1000.0,
            'unfilled_orders': []
        })

        # Orders post-liquidation are ignored
        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, position.orders[2]],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': -9888.888888888889,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': -8890111.111111112,
            'net_value': -100000.0,
            'net_quantity': -1000.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

    def test_liquidated_long_position_with_no_FLAT(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=.1,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=.1,
                   price=50,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.998,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 200000.0,
            'net_quantity': 2000.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        assert len(position.orders) == 3, position.orders
        assert position.orders[2].src == OrderSource.PRICE_FILLED_ELIMINATION_FLAT
        self.assertGreater(position.orders[2].price_sources[0].lag_ms, 1761281990000) # The lag is high. now_ms - DEFAULT_TESTING_FALLBACK_PRICE_SOURCE.start_ms
        position.orders[2].price_sources[0].lag_ms = PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE.lag_ms
        self.assertEqual(position.orders[2].price_sources, [PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE])

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, position.orders[2]],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': -10000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': -90000.0,
            'net_value': 200000.0,
            'net_quantity': 2000.0,
            'unfilled_orders': []
        })

        # Orders post-liquidation are ignored
        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save(position, o3)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, position.orders[2]],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': -10000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.0,
            'current_return': 0.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': -90000.0,
            'net_value': 200000.0,
            'net_quantity': 2000.0,
            'unfilled_orders': []
        })

        self.assertEqual(position.max_leverage_seen(), 2.0)
        self.assertEqual(position.get_cumulative_leverage(), 4.0)

    def test_simple_short_position_with_implicit_FLAT(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': -100000.0,
            'net_quantity': -100.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': 50000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 1.4985,
            'current_return': 1.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)

    def test_invalid_leverage_order(self):
        position = deepcopy(self.default_position)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=ValiConfig.ORDER_MIN_LEVERAGE * .999,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=ValiConfig.ORDER_MAX_LEVERAGE * 1.0001,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.SHORT,
                                     leverage=ValiConfig.ORDER_MIN_LEVERAGE * .999,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.SHORT,
                                     leverage=ValiConfig.ORDER_MAX_LEVERAGE * 1.0001,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=0.0,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.SHORT,
                                     leverage=TradePair.BTCUSD.min_leverage * .999,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=-1.0,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

        with self.assertRaises(ValueError):
            position.add_order(Order(order_type=OrderType.LONG,
                                     leverage=TradePair.BTCUSD.min_leverage * .999,
                                     price=100,
                                     trade_pair=TradePair.BTCUSD,
                                     processed_ms=1000,
                                     order_uuid="1000"), self.live_price_fetcher)

    def test_invalid_prices_zero(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=0,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        with self.assertRaises(ValueError):
            position.add_order(o1, self.live_price_fetcher)

    def test_invalid_prices_negative(self):
        with self.assertRaises(ValueError):
            o1 = Order(order_type=OrderType.LONG,  # noqa: F841
                       leverage=1.0,
                       price=-1,
                       trade_pair=TradePair.BTCUSD,
                       processed_ms=1000,
                       order_uuid="1000")

    def test_three_orders_with_longs_no_drawdown(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=2000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=2000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")

        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': .999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 100000.0,
            'net_quantity': 100.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.1,
            'initial_entry_price': 1000,
            'average_entry_price': 1047.6190476190477,
            'cumulative_entry_value': 110000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 1.9978,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 100000.0,
            'net_value': 210000.0,
            'net_quantity': 105.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1047.6190476190477,
            'cumulative_entry_value': 110000.0,
            'realized_pnl': 100_000,
            'close_ms': 5000,
            'return_at_close': 1.9978,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.1)
        self.assertEqual(position.get_cumulative_leverage(), 2.2)

    def test_two_orders_with_a_loss(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 24,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 12,
                   order_uuid="2000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 100000.0,
            'net_quantity': 100.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': -50000.0,
            'close_ms': o2.processed_ms,
            'return_at_close': 0.499,
            'current_return': 0.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.0)
        self.assertEqual(position.get_cumulative_leverage(), 2.0)
        self.assertEqual(position.get_spread_fee(o2.processed_ms), 1.0 - position.trade_pair.fees * 2)
        self.assertEqual(position.get_carry_fee(o2.processed_ms)[0], 1.0)

    def test_three_orders_with_a_loss_and_then_a_gain(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 24,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=500,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 12,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=0.1,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=FEE_V6_TIME_MS - 1000 * 60 * 60 * 4,
                   order_uuid="5000")

        position = deepcopy(self.default_position)
        self.add_order_to_position_and_save(position, o1)
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': 100000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.999,
            'current_return': 1.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 100000.0,
            'net_quantity': 100.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.1,
            'initial_entry_price': 1000,
            'average_entry_price': 916.6666666666666,
            'cumulative_entry_value': 110000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 0.49945,
            'current_return': 0.5,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': -50000.0,
            'net_value': 60000.0,
            'net_quantity': 120.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 1.0,
            'initial_entry_price': 1000,
            'average_entry_price': 916.6666666666666,
            'cumulative_entry_value': 110000.0,
            'realized_pnl': 833.3333333333337,
            'close_ms': None,
            'return_at_close': 1.09868,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 9166.666666666672,
            'net_value': 110000.0,
            'net_quantity': 110.0,
            'unfilled_orders': []
        })

        self.assertEqual(position.max_leverage_seen(), 1.1)
        self.assertAlmostEqual(position.get_cumulative_leverage(), 1.2, 8)

    def test_returns_on_large_price_increase(self):
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=2000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.LONG,
                   leverage=.01,
                   price=39000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")
        o4 = Order(order_type=OrderType.LONG,
                   leverage=.01,
                   price=40000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=4000,
                   order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=40000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save(position, o1)
        self.add_order_to_position_and_save(position, o2)
        self.add_order_to_position_and_save(position, o3)
        self.add_order_to_position_and_save(position, o4)
        self.add_order_to_position_and_save(position, o5)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 1066.152466148805,
            'cumulative_entry_value': 112000.0,
            'realized_pnl': 4090025.641025641,
            'close_ms': 5000,
            'return_at_close': 41.85332812307692,
            'current_return': 41.90025641025641,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.12)
        self.assertEqual(position.get_cumulative_leverage(), 2.24)

    def test_returns_on_many_shorts(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=0.1,
                   price=900,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=.01,
                   price=800,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")
        o4 = Order(order_type=OrderType.SHORT,
                   leverage=.01,
                   price=700,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=4000,
                   order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=600,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save(position, o1)
        self.add_order_to_position_and_save(position, o2)
        self.add_order_to_position_and_save(position, o3)
        self.add_order_to_position_and_save(position, o4)
        self.add_order_to_position_and_save(position, o5)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 984.2720139494332,
            'cumulative_entry_value': -112000.0,
            'realized_pnl': 43726.19047619047,
            'close_ms': 5000,
            'return_at_close': 1.4356521714285715,
            'current_return': 1.4372619047619049,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 1.12)
        self.assertEqual(position.get_cumulative_leverage(), 2.24)

    def test_returns_on_alternating_long_short(self):
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=0.5,
                   price=900,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=2.0,
                   price=800,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=3000,
                   order_uuid="3000")
        o4 = Order(order_type=OrderType.LONG,
                   leverage=2.1,
                   price=750,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=4000,
                   order_uuid="4000")
        o5 = Order(order_type=OrderType.FLAT,
                   leverage=0.0,
                   price=600,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=5000,
                   order_uuid="5000")
        position = deepcopy(self.default_position)

        self.add_order_to_position_and_save(position, o1)
        self.add_order_to_position_and_save(position, o2)
        self.add_order_to_position_and_save(position, o3)
        self.add_order_to_position_and_save(position, o4)
        self.add_order_to_position_and_save(position, o5)

        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3, o4, o5],
            'position_type': OrderType.FLAT,
            'is_closed_position': True,
            'net_leverage': 0.0,
            'initial_entry_price': 1000,
            'average_entry_price': 830.188679245283,
            'cumulative_entry_value': -300000.0,
            'realized_pnl': 31333.33333333332,
            'close_ms': 5000,
            'return_at_close': 1.31005,
            'current_return': 1.3133333333333332,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': self.DEFAULT_OPEN_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': 0.0,
            'net_quantity': 0.0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), 2.5)
        # -1 +.5 - 2.0 + 2.1 = 1.44 (abs 5.6) , (flat from -.4) -> 6.0
        self.assertEqual(position.get_cumulative_leverage(), 6.0)

    def test_error_adding_mismatched_trade_pair(self):
        position = deepcopy(self.default_position)
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=1000,
                   trade_pair=TradePair.EURNZD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=3.0,
                   price=500,
                   trade_pair=TradePair.CADCHF,
                   processed_ms=2000,
                   order_uuid="2000")
        o3 = Order(order_type=OrderType.FLAT,
                   leverage=5.0,
                   price=500,
                   trade_pair=TradePair.SPX,
                   processed_ms=3000,
                   order_uuid="3000")

        for order in [o1, o2, o3]:
            with self.assertRaises(ValueError):
                position.add_order(order, self.live_price_fetcher)

    def test_two_positions_no_collisions(self):
        weekday_time_ms = FEE_V6_TIME_MS + 1000 * 60 * 60 * 24 * 3
        trade_pair1 = TradePair.SPX
        hotkey1 = self.DEFAULT_MINER_HOTKEY
        position1 = Position(
            miner_hotkey=hotkey1,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=weekday_time_ms,
            trade_pair=trade_pair1,
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        trade_pair2 = TradePair.EURJPY
        hotkey2 = self.DEFAULT_MINER_HOTKEY + '_2'
        position2 = Position(
            miner_hotkey=hotkey2,
            position_uuid=self.DEFAULT_POSITION_UUID + '_2',
            open_ms=weekday_time_ms,
            trade_pair=trade_pair2,
            account_size=ValiConfig.DEFAULT_CAPITAL
        )

        o1 = Order(order_type=OrderType.SHORT,
                   leverage=0.4,
                   price=1000,
                   trade_pair=trade_pair1,
                   processed_ms=weekday_time_ms,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=0.4,
                   price=500,
                   trade_pair=trade_pair2,
                   processed_ms=weekday_time_ms,
                   order_uuid="2000",
                   quote_usd_rate=1.0,
                   usd_base_rate=1.0)

        self.add_order_to_position_and_save(position1, o1)
        self.validate_intermediate_position_state(position1, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -0.4,
            'initial_entry_price': 1000,
            'average_entry_price': 1000,
            'cumulative_entry_value': -40000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': position1.miner_hotkey,
            'open_ms': weekday_time_ms,
            'trade_pair': trade_pair1,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': -40000.0,
            'net_quantity': -40.0,
            'unfilled_orders': []
        })

        self.add_order_to_position_and_save(position2, o2)
        print(position2)
        self.validate_intermediate_position_state(position2, {
            'orders': [o2],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -0.4,
            'initial_entry_price': 500,
            'average_entry_price': 500,
            'cumulative_entry_value': -40000.0,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': 1.0,
            'current_return': 1.0,
            'miner_hotkey': position2.miner_hotkey,
            'open_ms': weekday_time_ms,
            'trade_pair': trade_pair2,
            'position_uuid': self.DEFAULT_POSITION_UUID + '_2',
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'net_value': -20000000.0,
            'net_quantity': -0.4,
            'unfilled_orders': []
        })

        self.assertEqual(position1.max_leverage_seen(), 0.4)
        self.assertEqual(position2.max_leverage_seen(), 0.4)
        self.assertEqual(position1.get_cumulative_leverage(), 0.4)
        self.assertEqual(position2.get_cumulative_leverage(), 0.4)

        self.assertEqual(position1.get_carry_fee(o1.processed_ms + MS_IN_24_HOURS)[0],
                         INDICES_CARRY_FEE_PER_INTERVAL ** position1.max_leverage_seen())
        self.assertEqual(position2.get_carry_fee(o2.processed_ms + MS_IN_24_HOURS)[0],
                         FOREX_CARRY_FEE_PER_INTERVAL ** position2.max_leverage_seen())

    def test_leverage_clamping_long(self):
        """Test that exceeding max leverage raises ValueError instead of clamping"""
        position = deepcopy(self.default_position)
        live_price = 69000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage / 2,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        # Add first order successfully
        self.add_order_to_position_and_save(position, o1)

        # Second order should raise ValueError as it would exceed max leverage
        with self.assertRaises(ValueError) as context:
            self.add_order_to_position_and_save(position, o2)

        # Verify position only has first order
        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': TradePair.BTCUSD.max_leverage / 2,
            'net_value': TradePair.BTCUSD.max_leverage / 2 * ValiConfig.DEFAULT_CAPITAL,
            'net_quantity': TradePair.BTCUSD.max_leverage / 2 * ValiConfig.DEFAULT_CAPITAL / live_price,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'cumulative_entry_value': TradePair.BTCUSD.max_leverage / 2 * ValiConfig.DEFAULT_CAPITAL,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': 1000,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage / 2)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage / 2)

    def test_leverage_clamping_skip_long_order(self):
        position = deepcopy(self.default_position)
        live_price = 100000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.max_leverage / 10,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save(position, o2)

        net_value = TradePair.BTCUSD.max_leverage * ValiConfig.DEFAULT_CAPITAL
        net_quantity = net_value / live_price

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': TradePair.BTCUSD.max_leverage,
            'net_value': net_value,
            'net_quantity': net_quantity,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'cumulative_entry_value': TradePair.BTCUSD.max_leverage * ValiConfig.DEFAULT_CAPITAL,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': 1000,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage)

    def test_leverage_clamping_short(self):
        """Test that exceeding max leverage raises ValueError instead of clamping (SHORT)"""
        position = deepcopy(self.default_position)
        live_price = 4444
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-TradePair.BTCUSD.max_leverage * .80,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=-TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        # Add first order successfully
        self.add_order_to_position_and_save(position, o1)

        # Second order should raise ValueError as it would exceed max leverage
        with self.assertRaises(ValueError) as context:
            self.add_order_to_position_and_save(position, o2)

        # Verify position only has first order
        net_value = -TradePair.BTCUSD.max_leverage * 0.80 * ValiConfig.DEFAULT_CAPITAL
        net_quantity = net_value / live_price

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -TradePair.BTCUSD.max_leverage * 0.80,
            'net_value': net_value,
            'net_quantity': net_quantity,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'cumulative_entry_value': -TradePair.BTCUSD.max_leverage * 0.80 * ValiConfig.DEFAULT_CAPITAL,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': 1000,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })
        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage * 0.80)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage * 0.80)

    def test_leverage_clamping_to_small(self):
        position = deepcopy(self.default_position)
        live_price = 4444
        o1 = Order(order_type=OrderType.LONG,
                   leverage=TradePair.BTCUSD.min_leverage * 1.5,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=TradePair.BTCUSD.min_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        # Ensure valueError is thrown. This position's leverage is too small to be considered valid.
        # Instead of clamping, this order should cause an error

        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save(position, o2)

    def test_leverage_clamping_skip_short_order(self):
        position = deepcopy(self.default_position)
        live_price = 999
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=TradePair.BTCUSD.max_leverage / 10,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save(position, o2)

        net_value = -TradePair.BTCUSD.max_leverage * ValiConfig.DEFAULT_CAPITAL
        net_quantity = net_value / live_price

        self.validate_intermediate_position_state(position, {
            'orders': [o1],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -TradePair.BTCUSD.max_leverage,
            'net_value': net_value,
            'net_quantity': net_quantity,
            'initial_entry_price': live_price,
            'average_entry_price': position.average_entry_price,
            'cumulative_entry_value': -TradePair.BTCUSD.max_leverage * ValiConfig.DEFAULT_CAPITAL,
            'realized_pnl': 0,
            'close_ms': None,
            'return_at_close': position.return_at_close,
            'current_return': position.current_return,
            'miner_hotkey': position.miner_hotkey,
            'open_ms': 1000,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

        self.assertEqual(position.max_leverage_seen(), TradePair.BTCUSD.max_leverage)
        self.assertEqual(position.get_cumulative_leverage(), TradePair.BTCUSD.max_leverage)

    def test_long_order_leverage_already_at_portfolio_limit(self):
        """
        3 forex positions: first 2 get us to portfolio max (20), and then 3rd attempts to exceed it.
        """
        # Position 1: AUDCAD at 10.0 leverage (max for forex)
        position = deepcopy(self.default_position)
        position.trade_pair = TradePair.AUDCAD
        o1 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=100,
                   trade_pair=TradePair.AUDCAD,
                   processed_ms=1000,
                   order_uuid="1000",
                   quote_usd_rate=1.0,
                   usd_base_rate=1.0)
        self.add_order_to_position_and_save(position, o1)
        self.position_manager.save_miner_position(position)

        # Position 2: EURUSD at 10.0 leverage (total = 20.0, exactly at forex cap)
        position2 = deepcopy(self.default_position)
        position2.trade_pair = TradePair.EURUSD
        position2.position_uuid = self.DEFAULT_POSITION_UUID + "_2"
        o2 = Order(order_type=OrderType.LONG,
                   leverage=10.0,
                   price=100,
                   trade_pair=TradePair.EURUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        self.add_order_to_position_and_save(position2, o2)
        self.position_manager.save_miner_position(position2)
        self.assertEqual(self.position_manager.calculate_net_portfolio_leverage(self.DEFAULT_MINER_HOTKEY), 20.0)

        # Position 3: GBPUSD at 0.1 leverage would bring total to 20.1, exceeding forex cap
        position3 = deepcopy(self.default_position)
        position3.trade_pair = TradePair.GBPUSD
        o3 = Order(order_type=OrderType.LONG,
                   leverage=0.1,
                   price=100,
                   trade_pair=TradePair.GBPUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        with self.assertRaises(ValueError):
            self.add_order_to_position_and_save(position3, o3)
            self.assertEqual(len(position3.orders), 0)

    def test_long_order_crypto_leverage_exceed_portfolio_limit(self):
        """
        3 crypto positions: 2 positions within the portfolio leverage max (5 for crypto),
        but the third order will exceed the max and raises ValueError
        """
        # Position 1: BTCUSD at 2.0 leverage
        position = deepcopy(self.default_position)
        position.trade_pair=TradePair.BTCUSD
        o1 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000",
                   quote_usd_rate=1.0,
                   usd_base_rate=1.0)
        self.add_order_to_position_and_save(position, o1)
        self.position_manager.save_miner_position(position)

        # Position 2: ETHUSD at 2.4 leverage (total = 4.4)
        position2 = deepcopy(self.default_position)
        position2.position_uuid = self.DEFAULT_POSITION_UUID + "_2"
        position2.trade_pair=TradePair.ETHUSD
        o2 = Order(order_type=OrderType.LONG,
                   leverage=2.4,
                   price=100,
                   trade_pair=TradePair.ETHUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        self.add_order_to_position_and_save(position2, o2)
        self.position_manager.save_miner_position(position2)

        self.assertEqual(self.position_manager.calculate_net_portfolio_leverage(self.DEFAULT_MINER_HOTKEY), 4.4)

        # Position 3: SOLUSD at 0.7 leverage would bring total to 5.1, exceeding crypto cap of 5
        position3 = deepcopy(self.default_position)
        position3.trade_pair=TradePair.SOLUSD
        position3.position_uuid = self.DEFAULT_POSITION_UUID + "_3"
        o3 = Order(order_type=OrderType.LONG,
                   leverage=0.7,
                   price=100,
                   trade_pair=TradePair.SOLUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        # Should raise ValueError instead of clamping
        with self.assertRaises(ValueError) as context:
            self.add_order_to_position_and_save(position3, o3)

    def test_long_order_forex_leverage_exceed_portfolio_limit(self):
        """
        3 forex positions: 2 positions within the portfolio leverage max (20 for forex),
        but the third order will exceed the max and raises ValueError
        """
        # Position 1: AUDCAD at 9.0 leverage
        position = deepcopy(self.default_position)
        position.trade_pair=TradePair.AUDCAD
        o1 = Order(order_type=OrderType.LONG,
                   leverage=9.0,
                   price=100,
                   trade_pair=TradePair.AUDCAD,
                   processed_ms=1000,
                   order_uuid="1000",
                   quote_usd_rate=1.0,
                   usd_base_rate=1.0)
        self.add_order_to_position_and_save(position, o1)
        self.position_manager.save_miner_position(position)

        # Position 2: USDMXN at 9.5 leverage (total = 18.5)
        position2 = deepcopy(self.default_position)
        position2.position_uuid = self.DEFAULT_POSITION_UUID + "_2"
        position2.trade_pair=TradePair.USDMXN
        o2 = Order(order_type=OrderType.LONG,
                   leverage=9.5,
                   price=100,
                   trade_pair=TradePair.USDMXN,
                   processed_ms=1000,
                   order_uuid="1000")
        self.add_order_to_position_and_save(position2, o2)
        self.position_manager.save_miner_position(position2)
        self.assertEqual(self.position_manager.calculate_net_portfolio_leverage(self.DEFAULT_MINER_HOTKEY), 18.5)

        # Position 3: EURUSD at 2.0 leverage would bring total to 20.5, exceeding forex cap of 20
        position3 = deepcopy(self.default_position)
        position3.trade_pair=TradePair.EURUSD
        position3.position_uuid = self.DEFAULT_POSITION_UUID + "_3"
        o3 = Order(order_type=OrderType.LONG,
                   leverage=2.0,
                   price=100,
                   trade_pair=TradePair.EURUSD,
                   processed_ms=2000,
                   order_uuid="2000",
                   quote_usd_rate=1.0,
                   usd_base_rate=1.0)

        # Should raise ValueError instead of clamping
        with self.assertRaises(ValueError) as context:
            self.add_order_to_position_and_save(position3, o3)

    def test_short_order_leverage_exceed_portfolio_limit(self):
        """
        3 crypto SHORT positions: 2 positions within the portfolio leverage max (5 for crypto),
        but the third order will exceed the max and raises ValueError
        """
        # Position 1: BTCUSD SHORT at -2.2 leverage
        position = deepcopy(self.default_position)
        position.trade_pair = TradePair.BTCUSD
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-2.2,
                   price=100,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000",
                   quote_usd_rate=1.0,
                   usd_base_rate=1.0)
        self.add_order_to_position_and_save(position, o1)

        # Position 2: ETHUSD SHORT at -2.3 leverage (total = 4.5)
        position2 = deepcopy(self.default_position)
        position2.trade_pair = TradePair.ETHUSD
        position2.position_uuid = self.DEFAULT_POSITION_UUID + "_2"
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=-2.3,
                   price=100,
                   trade_pair=TradePair.ETHUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        self.add_order_to_position_and_save(position2, o2)
        self.assertEqual(self.position_manager.calculate_net_portfolio_leverage(self.DEFAULT_MINER_HOTKEY), 4.5)

        # Position 3: SOLUSD SHORT at -0.6 leverage would bring total to 5.1, exceeding crypto cap of 5
        position3 = deepcopy(self.default_position)
        position3.trade_pair = TradePair.SOLUSD
        position3.position_uuid = self.DEFAULT_POSITION_UUID + "_3"
        o3 = Order(order_type=OrderType.SHORT,
                   leverage=-0.6,
                   price=100,
                   trade_pair=TradePair.SOLUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        # Should raise ValueError instead of clamping
        with self.assertRaises(ValueError) as context:
            self.add_order_to_position_and_save(position3, o3)

    def test_position_json(self):
        position = deepcopy(self.default_position)
        live_price = 100000
        o1 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        o2 = Order(order_type=OrderType.LONG,
                   leverage=1.0,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        for order in [o1, o2]:
            self.add_order_to_position_and_save(position, order)

        #self.assertEqual(position_json, {})
        dict_repr = position.to_dict()  # Make sure no side effects in the recreated object...
        for x in dict_repr['orders']:
            self.assertFalse('trade_pair' in x, dict_repr)

        position_json = position.to_json_string()
        recreated_object = Position(**json.loads(position_json))
        for x in recreated_object.orders:
            self.assertTrue(hasattr(x, 'trade_pair'), recreated_object)

        recreated_object_json = json.loads(position_json)
        for x in recreated_object_json['orders']:
            self.assertFalse('trade_pair' in x, recreated_object_json)

        #print(f"position json: {position_json}")
        dict_repr = position.to_dict()  # Make sure no side effects in the recreated object...

        recreated_object = Position.model_validate_json(position_json)  #Position(**json.loads(position_json))
        #print(f"recreated object str repr: {recreated_object}")
        #print("recreated object:", recreated_object)
        self.assertTrue(PositionManagerClient.positions_are_the_same(position, recreated_object))
        for x in dict_repr['orders']:
            self.assertFalse('trade_pair' in x, dict_repr)

        for x in recreated_object.orders:
            self.assertTrue(hasattr(x, 'trade_pair'), recreated_object)

    def test_fake_flat_order(self):
        position = deepcopy(self.default_position)
        position.orders = []
        for i in range(10):
            o = Order(order_type=OrderType.LONG,
                      leverage=.1 + i / 10,
                      price=100,
                      trade_pair=TradePair.BTCUSD,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i))
            position.orders.append(o)
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        orig_return = position.return_at_close
        n_orders_orig = len(position.orders)

        fake_flat_order = Order(price=0,
                           processed_ms=12345,
                           order_uuid=position.position_uuid[::-1],
                           # determinstic across validators. Won't mess with p2p sync
                           trade_pair=position.trade_pair,
                           order_type=OrderType.FLAT,
                           leverage=0,
                           src=OrderSource.ELIMINATION_FLAT)
        position.add_order(fake_flat_order, self.live_price_fetcher)
        assert orig_return == position.return_at_close
        assert len(position.orders) == n_orders_orig + 1

        position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        assert orig_return == position.return_at_close
        assert len(position.orders) == n_orders_orig + 1

    def test_deprecated_tp_position(self):
        """
        An open position with a deprecated trade pair should be closed.

        Tests that close_open_orders_for_suspended_trade_pairs correctly identifies
        and closes positions for deprecated trade pairs (SPX, DJI, NDX, VIX).
        """
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.DJI,
            account_size=ValiConfig.DEFAULT_CAPITAL
        )
        for i in range(3):
            o = Order(order_type=OrderType.LONG,
                      leverage=.1 + i / 10,
                      price=100,
                      trade_pair=TradePair.DJI,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i))
            self.add_order_to_position_and_save(position, o)
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        assert len(position.orders) == 3
        assert not position.is_closed_position
        # Server's internal price fetcher client can now connect to real RPC server
        self.position_manager.close_open_orders_for_suspended_trade_pairs()
        position = self._find_disk_position_from_memory_position(position)
        print(position)
        assert len(position.orders) == 4
        assert position.is_closed_position
        assert position.orders[-1].src == OrderSource.DEPRECATION_FLAT


if __name__ == '__main__':
    import unittest

    unittest.main()
