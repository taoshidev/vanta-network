# developer: jbonilla
import json
from copy import deepcopy

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_8_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import (
    Position,
)

FEE_V6_TIME_MS = 1720843707000
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import (
    Order,
)
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource

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
        position_type=OrderType.LONG,
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
        position.add_order(order, self.live_price_fetcher)
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
        With slippage=0.00 on all orders, returns should be the same regardless of slippage application.

        NOTE: leverage values are 2x what they were historically (2/1/2 instead of 1/0.5/1).
        `Position.validate_min_position_size` now rejects orders (including reducing orders)
        that would leave the position below `ValiConfig.FOREX_MIN_POSITION_SIZE_LOTS` (0.01
        lots). At leverage=1 on a $100k account, EURUSD's net_quantity lands at exactly 0.01
        lots after the opening order, and `reduce_size_order` (leverage=0.5) then drops it
        below the floor, raising ValueError. Doubling leverage keeps the same relative order
        sizes/ratios (and therefore the same pre/post-rebuild return equality this test
        checks) while staying comfortably above the minimum lot size.
        """
        open_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="open_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=2,
        )
        reduce_size_order = Order(
            price=110,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.SHORT,
            leverage=1,
        )
        increase_size_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 2000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=2,
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
            # Scaled up (return is scale-invariant) so the reduce_size_order's resulting lot
            # size stays above FOREX_MIN_POSITION_SIZE_LOTS and doesn't hit validate_min_position_size.
            account_size=ValiConfig.DEFAULT_CAPITAL * 10,
            position_type=OrderType.LONG,
        )
        closed_position.add_order(open_order, self.live_price_fetcher)
        closed_position.add_order(reduce_size_order, self.live_price_fetcher)
        closed_position.add_order(increase_size_order, self.live_price_fetcher)
        closed_position.add_order(close_order, self.live_price_fetcher)

        old_returns_calc = closed_position.current_return
        closed_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        new_returns_calc = closed_position.current_return
        assert old_returns_calc == new_returns_calc

    def test_loss_position_returns_pre_post_slippage(self):
        """
        With slippage=0.00 on all orders, returns should be the same on rebuild.

        NOTE: leverage values are 2x what they were historically (2/1/2 instead of 1/0.5/1).
        See the identical note in test_profit_position_returns_pre_post_slippage: at
        leverage=1, `reduce_size_order` drops the position below
        `ValiConfig.FOREX_MIN_POSITION_SIZE_LOTS` and `validate_min_position_size` raises.
        Doubling leverage preserves the same relative order sizes.
        """
        open_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="open_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.SHORT,
            leverage=2,
        )
        reduce_size_order = Order(
            price=110,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=1,
        )
        increase_size_order = Order(
            price=100,
            slippage=0.00,
            processed_ms=self.DEFAULT_OPEN_MS + 2000,
            order_uuid="reduce_size_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.SHORT,
            leverage=2,
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
            # Scaled up (return is scale-invariant) so the reduce_size_order's resulting lot
            # size stays above FOREX_MIN_POSITION_SIZE_LOTS and doesn't hit validate_min_position_size.
            account_size=ValiConfig.DEFAULT_CAPITAL * 10,
            position_type=OrderType.SHORT,
        )
        closed_position.add_order(open_order, self.live_price_fetcher)
        closed_position.add_order(reduce_size_order, self.live_price_fetcher)
        closed_position.add_order(increase_size_order, self.live_price_fetcher)
        closed_position.add_order(close_order, self.live_price_fetcher)

        old_returns_calc = closed_position.current_return
        closed_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        new_returns_calc = closed_position.current_return
        assert old_returns_calc == new_returns_calc

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
            account_size=ValiConfig.DEFAULT_CAPITAL,
            position_type=OrderType.SHORT,
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
            account_size=ValiConfig.DEFAULT_CAPITAL,
            position_type=OrderType.SHORT,
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

    def test_simple_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 1.1,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

    def test_simple_long_position_with_implicit_FLAT(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 2.0,
            'current_return': 2.0,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': FEE_V6_TIME_MS,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

    def test_simple_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 1.1,
            'current_return': 1.1,
            'miner_hotkey': self.DEFAULT_MINER_HOTKEY,
            'open_ms': o1.processed_ms,
            'trade_pair': self.DEFAULT_TRADE_PAIR,
            'position_uuid': self.DEFAULT_POSITION_UUID,
            'account_size': ValiConfig.DEFAULT_CAPITAL,
            'unrealized_pnl': 0,
            'unfilled_orders': []
        })

    def test_liquidated_long_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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

    def test_liquidated_short_position_with_explicit_FLAT(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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

    def test_liquidated_short_position_with_no_FLAT(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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

        # A severe adverse move drives current_return to 0 (liquidated in return terms), but
        # Position no longer auto-inserts a synthetic PRICE_FILLED_ELIMINATION_FLAT order or
        # auto-closes the position on a return-based liquidation — that now happens in a
        # separate service (see elimination_manager.close_all_positions). The position stays
        # open and continues accepting orders until something external closes it.
        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': -9888.888888888889,
            'close_ms': None,
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

        # Further orders are NOT rejected — the position is still open.
        self.add_order_to_position_and_save(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.SHORT,
            'is_closed_position': False,
            'net_leverage': -1.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': -100000.0,
            'realized_pnl': -19777.777777777777,
            'close_ms': None,
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
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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

        # A severe adverse move drives current_return to 0 (liquidated in return terms), but
        # Position no longer auto-inserts a synthetic PRICE_FILLED_ELIMINATION_FLAT order or
        # auto-closes the position on a return-based liquidation — that now happens in a
        # separate service (see elimination_manager.close_all_positions). The position stays
        # open and continues accepting orders until something external closes it.
        self.add_order_to_position_and_save(position, o2)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': -10000.0,
            'close_ms': None,
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

        # Further orders are NOT rejected — the position is still open.
        self.add_order_to_position_and_save(position, o3)
        self.validate_intermediate_position_state(position, {
            'orders': [o1, o2, o3],
            'position_type': OrderType.LONG,
            'is_closed_position': False,
            'net_leverage': 2.0,
            'initial_entry_price': 100,
            'average_entry_price': 100,
            'cumulative_entry_value': 200000.0,
            'realized_pnl': -20000.0,
            'close_ms': None,
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
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 1.5,
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

    def test_invalid_leverage_order(self):
        """
        Test current behavior of a zero-leverage order (other leverage bounds are clamped,
        not rejected, at the MarketOrderManager level rather than Position.add_order).

        NOTE: Position.add_order no longer raises for a zero-leverage order. Because
        `position_type` is now a required field set at construction, a zero-leverage order
        against a flat/empty position produces a zero net delta, which the flatten-detection
        in `_update_position` treats as an implicit FLAT (net_quantity + order.quantity <= 0),
        closing the position with no PnL impact instead of raising ValueError. This looks like
        a validation gap in position.py (zero leverage should probably still be rejected), but
        per instructions we match current behavior here rather than "fixing" production code.
        """
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        order = Order(order_type=OrderType.LONG,
                      leverage=0.0,
                      price=100,
                      trade_pair=TradePair.BTCUSD,
                      processed_ms=1000,
                      order_uuid="1000")
        position.add_order(order, self.live_price_fetcher)
        self.assertTrue(position.is_closed_position)
        self.assertEqual(position.position_type, OrderType.FLAT)
        self.assertEqual(position.current_return, 1.0)

    def test_invalid_prices_zero(self):
        """
        A price of 0 on the first order of a fresh position causes
        `initialize_position_from_first_order` to compute `initial_entry_price == 0`,
        which raises ValueError("Initial entry price must be > 0").
        """
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 2.0,
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
            'return_at_close': 2.0,
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
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 0.5,
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
        position.position_type = None  # reset to force re-derivation from the first order added
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
            'return_at_close': 1.0,
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
            'return_at_close': 0.5,
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
            'return_at_close': 1.1,
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
        position.position_type = None  # reset to force re-derivation from the first order added

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
            'return_at_close': 41.90025641025641,
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
        position.position_type = None  # reset to force re-derivation from the first order added

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
            'return_at_close': 1.4372619047619049,
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
        position.position_type = None  # reset to force re-derivation from the first order added

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
            'return_at_close': 1.3133333333333332,
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

    def test_error_adding_mismatched_trade_pair(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
            account_size=ValiConfig.DEFAULT_CAPITAL,
            position_type=OrderType.SHORT,
        )
        trade_pair2 = TradePair.EURJPY
        hotkey2 = self.DEFAULT_MINER_HOTKEY + '_2'
        position2 = Position(
            miner_hotkey=hotkey2,
            position_uuid=self.DEFAULT_POSITION_UUID + '_2',
            open_ms=weekday_time_ms,
            trade_pair=trade_pair2,
            account_size=ValiConfig.DEFAULT_CAPITAL,
            position_type=OrderType.SHORT,
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

    def test_leverage_clamping_long(self):
        """Test that exceeding max leverage is clamped (not rejected)"""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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

        # Second order would exceed max leverage - with clamping, order is accepted
        # but leverage is clamped to max (note: clamping happens in MarketOrderManager,
        # not in Position.add_order directly - so this test just verifies no error)
        self.add_order_to_position_and_save(position, o2)

        # Verify position has both orders (second order added as-is since clamping
        # happens at MarketOrderManager level, not Position level)
        self.assertEqual(len(position.orders), 2)
        self.assertEqual(position.position_type, OrderType.LONG)
        self.assertFalse(position.is_closed_position)

    def test_leverage_clamping_skip_long_order(self):
        """Test that when position is at max leverage, additional orders are handled (clamped at MarketOrderManager level)"""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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
        # With clamping, this order is accepted (clamping happens at MarketOrderManager level)
        self.add_order_to_position_and_save(position, o2)

        # Both orders should be in position
        self.assertEqual(len(position.orders), 2)
        self.assertEqual(position.position_type, OrderType.LONG)
        self.assertFalse(position.is_closed_position)

    def test_leverage_clamping_short(self):
        """Test that exceeding max leverage is clamped (not rejected) for SHORT"""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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

        # Second order would exceed max - with clamping, order is accepted
        self.add_order_to_position_and_save(position, o2)

        # Both orders should be in position
        self.assertEqual(len(position.orders), 2)
        self.assertEqual(position.position_type, OrderType.SHORT)
        self.assertFalse(position.is_closed_position)

    # def test_leverage_clamping_to_small(self):
    #     position = deepcopy(self.default_position)
    #     live_price = 4444
    #     o1 = Order(order_type=OrderType.LONG,
    #                leverage=TradePair.BTCUSD.min_leverage * 1.5,
    #                price=live_price,
    #                trade_pair=TradePair.BTCUSD,
    #                processed_ms=1000,
    #                order_uuid="1000")
    #     o2 = Order(order_type=OrderType.SHORT,
    #                leverage=TradePair.BTCUSD.min_leverage,
    #                price=live_price,
    #                trade_pair=TradePair.BTCUSD,
    #                processed_ms=2000,
    #                order_uuid="2000")

    #     self.add_order_to_position_and_save(position, o1)
    #     # Ensure valueError is thrown. This position's leverage is too small to be considered valid.
    #     # Instead of clamping, this order should cause an error

    #     with self.assertRaises(ValueError):
    #         self.add_order_to_position_and_save(position, o2)

    def test_leverage_clamping_skip_short_order(self):
        """Test that when SHORT position is at max leverage, additional orders are handled"""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        live_price = 999
        o1 = Order(order_type=OrderType.SHORT,
                   leverage=-TradePair.BTCUSD.max_leverage,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=1000,
                   order_uuid="1000")
        # Note: This order has positive leverage which would reduce the SHORT position
        o2 = Order(order_type=OrderType.SHORT,
                   leverage=TradePair.BTCUSD.max_leverage / 10,
                   price=live_price,
                   trade_pair=TradePair.BTCUSD,
                   processed_ms=2000,
                   order_uuid="2000")

        self.add_order_to_position_and_save(position, o1)
        # With clamping at MarketOrderManager level, order is accepted
        self.add_order_to_position_and_save(position, o2)

        # Both orders should be in position
        self.assertEqual(len(position.orders), 2)
        self.assertEqual(position.position_type, OrderType.SHORT)
        self.assertFalse(position.is_closed_position)

    def test_position_json(self):
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
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

        position_json = str(position)
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
        position.position_type = None  # reset to force re-derivation from the first order added
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
        An open position with a suspended/deprecated trade pair should be force-closed.

        NOTE: The old `close_open_orders_for_suspended_trade_pairs` method (and the
        indices it covered: SPX, DJI, NDX, VIX) no longer exists. The current equivalent
        is `PositionManager.force_close_deprecated_trade_pair_positions`, which is invoked
        internally by `pre_run_setup()` with a hardcoded commodities list
        `[XAUUSD, XAGUSD, BRENTOILUSDC, PAXGUSDC]` (see position_manager.py). We use
        XAUUSD here, and only that client-exposed entrypoint, since the client doesn't
        expose `force_close_deprecated_trade_pair_positions` directly.

        Orders are given an explicit `price_sources` entry because `force_close_position`
        derives its fill price from `Position.last_price_source`, which is only populated
        from `order.price_sources`; without it, `parse_appropriate_price` returns None and
        the close falls back to `ELIMINATION_FLAT` at price 0 instead of `DEPRECATION_FLAT`.
        """
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.XAUUSD,
            account_size=ValiConfig.DEFAULT_CAPITAL,
            position_type=OrderType.LONG,
        )
        for i in range(3):
            o = Order(order_type=OrderType.LONG,
                      leverage=.1 + i / 10,
                      price=100,
                      trade_pair=TradePair.XAUUSD,
                      processed_ms=1000 + i * 10,
                      order_uuid=str(i),
                      price_sources=[PriceSource(source='test', open=100, close=100)])
            self.add_order_to_position_and_save(position, o)
        position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        assert len(position.orders) == 3
        assert not position.is_closed_position
        # Server's internal price fetcher client can now connect to real RPC server
        self.position_manager.pre_run_setup(perform_order_corrections=False)
        position = self._find_disk_position_from_memory_position(position)
        print(position)
        assert len(position.orders) == 4
        assert position.is_closed_position
        assert position.orders[-1].src == OrderSource.DEPRECATION_FLAT

    # ==================== Minimum Position Size Validation Tests ====================
    # Position.validate_order_size (a max-USD-value clamp keyed on a caller-supplied
    # max_position_value) has been removed entirely from position.py with no replacement —
    # that kind of leverage/size capping now happens at the MarketOrderManager level (see
    # other tests' comments referencing "clamped at MarketOrderManager level"). The only
    # size validation left on Position is validate_min_position_size, which raises ValueError
    # if an order would leave a nonzero position below the per-asset-class minimum size. These
    # tests exercise that method instead.

    def test_min_position_size_crypto_order_within_limit(self):
        """A crypto order well above CRYPTO_MIN_POSITION_SIZE_USD ($10) should not raise."""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        position.position_type = OrderType.LONG
        position.net_leverage = 0.0
        position.net_quantity = 0.0
        position.net_value = 0.0
        order = Order(
            price=60000,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="test_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=2.0,
            value=200000,
            quantity=3.33,
        )

        position.validate_min_position_size(order)  # should NOT raise

    def test_min_position_size_crypto_order_below_minimum_raises(self):
        """A crypto order below CRYPTO_MIN_POSITION_SIZE_USD ($10) should raise."""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        position.position_type = OrderType.LONG
        position.net_leverage = 0.0
        position.net_quantity = 0.0
        position.net_value = 0.0
        order = Order(
            price=60000,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="test_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=0.00008,
            value=5,  # below $10 minimum
            quantity=0.0000833,
        )

        with self.assertRaises(ValueError) as ctx:
            position.validate_min_position_size(order)
        self.assertIn("below minimum", str(ctx.exception))

    def test_min_position_size_forex_order_below_minimum_raises(self):
        """A forex order below FOREX_MIN_POSITION_SIZE_LOTS (0.01 lots) should raise."""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        position.position_type = OrderType.LONG
        position.trade_pair = TradePair.EURUSD
        position.net_leverage = 0.0
        position.net_quantity = 0.0
        position.net_value = 0.0
        order = Order(
            price=1.1,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="test_order",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=0.0001,
            value=10,
            quantity=0.005,  # below 0.01 lot minimum
        )

        with self.assertRaises(ValueError) as ctx:
            position.validate_min_position_size(order)
        self.assertIn("below minimum", str(ctx.exception))

    def test_min_position_size_equities_order_below_minimum_raises(self):
        """An equities order below EQUITIES_MIN_POSITION_SIZE_SHARES (0.01 shares) should raise."""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        position.position_type = OrderType.LONG
        position.trade_pair = TradePair.AAPL
        position.net_leverage = 0.0
        position.net_quantity = 0.0
        position.net_value = 0.0
        order = Order(
            price=200,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="test_order",
            trade_pair=TradePair.AAPL,
            order_type=OrderType.LONG,
            leverage=0.00001,
            value=1,
            quantity=0.005,  # below 0.01 share minimum
        )

        with self.assertRaises(ValueError) as ctx:
            position.validate_min_position_size(order)
        self.assertIn("below minimum", str(ctx.exception))

    def test_min_position_size_flat_order_not_limited(self):
        """FLAT orders are exempt from the minimum size check regardless of resulting size."""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        position.position_type = OrderType.LONG
        position.net_leverage = 1.0
        position.net_quantity = 1.67
        position.net_value = 100000

        order = Order(
            price=60000,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="test_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.FLAT,
            leverage=0.0,
            value=-99995,  # would leave a tiny, below-minimum remainder if it were checked
            quantity=-1.665,
        )

        position.validate_min_position_size(order)  # should NOT raise

    def test_min_position_size_reducing_order_still_valid(self):
        """A SHORT order that reduces an existing LONG position but leaves it well above the minimum should not raise."""
        position = deepcopy(self.default_position)
        position.position_type = None  # reset to force re-derivation from the first order added
        initial_order = Order(
            price=60000,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="initial_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.LONG,
            leverage=2.0,
            value=200000,
            quantity=3.33,
        )
        position.orders.append(initial_order)
        position.net_value = 200000
        position.net_leverage = 2.0
        position.net_quantity = 3.33
        position.position_type = OrderType.LONG

        order = Order(
            price=60000,
            processed_ms=self.DEFAULT_OPEN_MS + 1,
            order_uuid="test_order",
            trade_pair=TradePair.BTCUSD,
            order_type=OrderType.SHORT,
            leverage=-0.5,
            value=-50000,
            quantity=-0.83,
        )

        position.validate_min_position_size(order)  # should NOT raise (remaining ~$150k is well above $10 min)

if __name__ == '__main__':
    import unittest

    unittest.main()
