# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Time utility tests using modern RPC infrastructure.

Tests TimeUtil functions with proper server/client setup.
"""
from datetime import datetime, timezone

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS, TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import FEE_V6_TIME_MS, Position
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestTimeUtil(TestBase):
    """
    Time utility tests using ServerOrchestrator singleton pattern.

    Server infrastructure is managed by ServerOrchestrator and shared across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    position_client = None
    metagraph_client = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_POSITION_UUID = "test_position"
    DEFAULT_OPEN_MS = 1000
    DEFAULT_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_ACCOUNT_SIZE = 100_000

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
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

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
        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Re-set metagraph hotkeys (cleared by clear_all_test_data)
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY])

        # Create fresh test data
        self._create_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        # No need to create default positions since tests now create fresh instances
        # This avoids deepcopy issues with RPC-backed domain objects
        pass

    def test_n_crypto_intervals(self):
        prev_delta = None
        for i in range(50):
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

            # Create fresh Position with orders (avoid deepcopy to prevent RPC serialization issues)
            position = Position(
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                position_uuid=self.DEFAULT_POSITION_UUID,
                open_ms=self.DEFAULT_OPEN_MS,
                trade_pair=TradePair.BTCUSD,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[o1, o2]
            )
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
        o1 = Order(
            order_type=OrderType.LONG,
            leverage=1.0,
            price=100,
            trade_pair=TradePair.BTCUSD,
            processed_ms=1719596222703,
            order_uuid="1000"
        )

        # Create fresh Position with orders (avoid deepcopy to prevent RPC serialization issues)
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.BTCUSD,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[o1]
        )
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

            # Create fresh Position with orders (avoid deepcopy to prevent RPC serialization issues)
            position = Position(
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                position_uuid=self.DEFAULT_POSITION_UUID,
                open_ms=self.DEFAULT_OPEN_MS,
                trade_pair=TradePair.EURUSD,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[o1, o2]
            )
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
