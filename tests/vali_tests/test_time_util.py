# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Time utility tests using modern RPC infrastructure.

Tests TimeUtil functions with proper server/client setup.
"""
from copy import deepcopy
from datetime import datetime, timezone

from shared_objects.common_data_server import CommonDataServer
from shared_objects.metagraph_server import MetagraphServer, MetagraphClient
from shared_objects.port_manager import PortManager
from shared_objects.rpc_client_base import RPCClientBase
from shared_objects.rpc_server_base import RPCServerBase
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS, TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import FEE_V6_TIME_MS, Position
from vali_objects.utils.challengeperiod_server import ChallengePeriodServer
from vali_objects.utils.live_price_server import LivePriceFetcherServer, LivePriceFetcherClient
from vali_objects.utils.position_manager_client import PositionManagerClient
from vali_objects.utils.position_manager_server import PositionManagerServer
from vali_objects.utils.contract_server import ContractServer
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestTimeUtil(TestBase):
    """
    Time utility tests using class-level server setup.

    Server infrastructure is started once in setUpClass and shared across all tests.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level server/client references
    common_data_server = None
    metagraph_server = None
    live_price_fetcher_server = None
    live_price_fetcher_client = None
    contract_server = None
    challengeperiod_server = None
    position_server = None
    position_client = None
    metagraph_client = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_POSITION_UUID = "test_position"
    DEFAULT_OPEN_MS = 1000
    DEFAULT_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_ACCOUNT_SIZE = 100_000

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers (expensive operation done once)."""
        PortManager.force_kill_all_rpc_ports()

        # Start infrastructure servers
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.live_price_fetcher_server = LivePriceFetcherServer(
            secrets=secrets,
            disable_ws=True,
            start_server=True,
            running_unit_tests=True
        )
        cls.common_data_server = CommonDataServer(start_server=True)
        cls.metagraph_server = MetagraphServer(start_server=True, running_unit_tests=True)
        cls.contract_server = ContractServer(start_server=True, running_unit_tests=True)
        cls.challengeperiod_server = ChallengePeriodServer(running_unit_tests=True, start_daemon=False)

        # Create clients
        cls.live_price_fetcher_client = LivePriceFetcherClient()
        cls.metagraph_client = MetagraphClient()
        cls.metagraph_client.set_hotkeys([cls.DEFAULT_MINER_HOTKEY])

        # Start position server after dependencies
        cls.position_server = PositionManagerServer(
            running_unit_tests=True,
            start_server=True
        )
        cls.position_client = PositionManagerClient()

    @classmethod
    def tearDownClass(cls):
        """One-time teardown: Stop all servers."""
        RPCClientBase.disconnect_all()
        RPCServerBase.shutdown_all(force_kill_ports=True)

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        # Clear all data for test isolation
        self.position_client.clear_all_miner_positions_and_disk()

        # Create fresh test data
        self._create_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.position_client.clear_all_miner_positions_and_disk()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.forex_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.EURUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

    def test_n_crypto_intervals(self):
        prev_delta = None
        for i in range(50):
            position = deepcopy(self.default_position)
            o1 = Order(
                order_type=OrderType.LONG,
                leverage=1.0,
                price=100,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1719843814000,
                order_uuid="1000"
            )
            o2 = Order(
                order_type=OrderType.FLAT,
                leverage=0.0,
                price=110,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1719843816000 + i * MS_IN_8_HOURS + i,
                order_uuid="2000"
            )

            position.orders = [o1, o2]
            position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)

            self.assertEqual(position.max_leverage_seen(), 1.0)
            self.assertEqual(position.get_cumulative_leverage(), 2.0)
            n_intervals, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(
                o1.processed_ms, o2.processed_ms
            )
            delta = time_until_next_interval_ms
            if i != 0:
                self.assertEqual(delta + 1, prev_delta, f"delta: {delta}, prev_delta: {prev_delta}")
            prev_delta = delta

            self.assertEqual(n_intervals, i, f"n_intervals: {n_intervals}, i: {i}")

    def test_crypto_edge_case(self):
        t_ms = FEE_V6_TIME_MS + 1000*60*60*4  # 4 hours after start_time # 1720756395630
        position = deepcopy(self.default_position)
        o1 = Order(
            order_type=OrderType.LONG,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=1719596222703,
            order_uuid="1000"
        )
        position.orders = [o1]
        position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
        n_intervals, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(
            position.start_carry_fee_accrual_ms, t_ms
        )
        assert n_intervals == 0, f"n_intervals: {n_intervals}, time_until_next_interval_ms: {time_until_next_interval_ms}"

    def test_parse_iso_to_ms(self):
        test_cases = [
            # Timestamp with milliseconds and UTC offset
            {"iso": "2024-11-20T15:47:40.062000+00:00", "expected": 1732117660062},

            # Timestamp with no milliseconds
            {"iso": "2023-03-01T12:00:00+00:00", "expected": 1677672000000},

            # Timestamp with positive timezone offset
            {"iso": "2023-03-01T15:00:00+03:00", "expected": 1677672000000},

            # Timestamp with negative timezone offset
            {"iso": "2023-03-01T06:00:00-06:00", "expected": 1677672000000},

            # Timestamp with microseconds
            {"iso": "2023-03-01T12:00:00.123456+00:00", "expected": 1677672000123},

            # Timestamp with ns precision
            {"iso": "2023-03-01T12:00:00.123456789+00:00", "expected": 1677672000123},
        ]

        for case in test_cases:
            with self.subTest(iso=case["iso"]):
                ms = TimeUtil.parse_iso_to_ms(case["iso"])
                self.assertEqual(ms, case["expected"], msg=f"Expected {case['expected']} but got {ms}")

    def test_n_forex_intervals(self):
        prev_delta = None
        for i in range(50):
            position = deepcopy(self.forex_position)
            o1 = Order(
                order_type=OrderType.LONG,
                leverage=1.0,
                price=1.1,
                trade_pair=TradePair.EURUSD,
                processed_ms=1719843814000,
                order_uuid="1000"
            )
            o2 = Order(
                order_type=OrderType.FLAT,
                leverage=0.0,
                price=1.2,
                trade_pair=TradePair.EURUSD,
                processed_ms=1719843816000 + i + MS_IN_24_HOURS * i,
                order_uuid="2000"
            )
            position.orders = [o1, o2]
            position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
            self.assertEqual(position.max_leverage_seen(), 1.0)
            self.assertEqual(position.get_cumulative_leverage(), 2.0)
            n_intervals, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_forex_indices(
                o1.processed_ms, o2.processed_ms
            )
            carry_fee, next_update_time_ms = position.crypto_carry_fee(o2.processed_ms)
            assert next_update_time_ms > o2.processed_ms, f"next_update_time_ms: {next_update_time_ms}, o2.processed_ms: {o2.processed_ms}"
            delta = time_until_next_interval_ms
            if i != 0:
                self.assertEqual(delta + 1, prev_delta, f"delta: {delta}, prev_delta: {prev_delta}")
            prev_delta = delta

            self.assertEqual(n_intervals, i, f"n_intervals: {n_intervals}, i: {i}")

    def test_n_intervals_boundary(self):
        for i in range(1, 3):
            # Create a datetime object for 4 AM UTC today
            datetime_utc = datetime(2020, 7, 1, 4 + i*8, 0, 0, tzinfo=timezone.utc)
            t1 = int(datetime_utc.timestamp() * 1000) - 1
            t2 = int(datetime_utc.timestamp() * 1000)
            n_intervals, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(t1, t2)
            delta = time_until_next_interval_ms
            self.assertEqual(n_intervals, 1, f"n_intervals: {n_intervals}")
            self.assertEqual(delta, MS_IN_8_HOURS, f"delta: {delta}")

            t1 = int(datetime_utc.timestamp() * 1000)
            t2 = int(datetime_utc.timestamp() * 1000) + MS_IN_8_HOURS
            n_intervals, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(t1, t2)
            delta = time_until_next_interval_ms
            self.assertEqual(n_intervals, 1, f"n_intervals: {n_intervals}")
            self.assertEqual(delta, MS_IN_8_HOURS, f"delta: {delta}")


if __name__ == '__main__':
    import unittest
    unittest.main()
