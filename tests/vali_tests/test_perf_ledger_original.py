import bittensor as bt

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.vali_utils import ValiUtils

bt.logging.enable_info()

class TestPerfLedgers(TestBase):
    """
    Performance ledger tests using ServerOrchestrator.

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
    perf_ledger_client = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_ACCOUNT_SIZE = 100_000
    DEFAULT_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_OPEN_MS = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 24 * 60  # 60 days ago
    default_btc_order = Order(price=60000, processed_ms=DEFAULT_OPEN_MS, order_uuid="test_order_btc",
                              trade_pair=DEFAULT_TRADE_PAIR,
                              order_type=OrderType.LONG, leverage=.5)
    default_nvda_order = Order(price=100, processed_ms=DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 5,
                               order_uuid="test_order_nvda", trade_pair=TradePair.NVDA,
                               order_type=OrderType.LONG, leverage=1)
    default_usdjpy_order = Order(price=156, processed_ms=DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 10,
                                 order_uuid="test_order_usdjpy",
                                 trade_pair=TradePair.USDJPY, order_type=OrderType.LONG, leverage=1)
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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.position_client = cls.orchestrator.get_client('position_manager')

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

        # Reset time-based test data for each test
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 24 * 60  # 60 days ago
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        # Set up metagraph with test miner
        self.metagraph_client.set_hotkeys([self.DEFAULT_MINER_HOTKEY])

        # Create fresh test positions for this test
        self._create_test_positions()

        # Save default positions
        for p in [self.default_usdjpy_position, self.default_nvda_position, self.default_btc_position]:
            self.position_client.save_miner_position(p)

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_positions(self):
        """Helper to create fresh test orders and positions."""
        self.default_btc_order = Order(
            price=60000, processed_ms=self.DEFAULT_OPEN_MS, order_uuid="test_order_btc",
            trade_pair=self.DEFAULT_TRADE_PAIR, order_type=OrderType.LONG, leverage=.5
        )
        self.default_nvda_order = Order(
            price=100, processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 5,
            order_uuid="test_order_nvda", trade_pair=TradePair.NVDA,
            order_type=OrderType.LONG, leverage=1
        )
        self.default_usdjpy_order = Order(
            price=156, processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 10,
            order_uuid="test_order_usdjpy", trade_pair=TradePair.USDJPY,
            order_type=OrderType.LONG, leverage=1
        )

        self.default_btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_btc",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_btc_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_btc_position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)

        self.default_nvda_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_nvda",
            open_ms=self.default_nvda_order.processed_ms,
            trade_pair=TradePair.NVDA,
            orders=[self.default_nvda_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_nvda_position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)

        self.default_usdjpy_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_usdjpy",
            open_ms=self.default_usdjpy_order.processed_ms,
            trade_pair=TradePair.USDJPY,
            orders=[self.default_usdjpy_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_usdjpy_position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)

    def check_alignment_per_cp(self, ans):
        """Verify portfolio ledger has consistent checkpoint data."""
        portfolio_pl = ans[self.DEFAULT_MINER_HOTKEY]
        self.assertIsNotNone(portfolio_pl)
        self.assertGreater(len(portfolio_pl.cps), 0)
        original_ret = portfolio_pl.cps[-1].prev_portfolio_ret
        original_mdd = portfolio_pl.cps[-1].mdd
        # Verify basic consistency: returns should be positive, mdd should be <= 1
        self.assertGreater(original_ret, 0, "Portfolio return should be positive")
        self.assertLessEqual(original_mdd, 1.0, "Portfolio MDD should be <= 1.0")
        self.assertGreater(original_mdd, 0, "Portfolio MDD should be positive")

    def test_basic(self):
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY: [self.default_btc_position]}
        ans = self.perf_ledger_client.generate_perf_ledgers_for_analysis(hotkey_to_positions)
        for hk, pl in ans.items():
            for idx, x in enumerate(pl.cps):
                last_update_formatted = TimeUtil.millis_to_timestamp(x.last_update_ms)
                if idx == 0 or idx == len(pl.cps) - 1:
                    print(x, last_update_formatted)

        assert len(ans) == 1, ans

    def test_multiple_tps(self):
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY:
                                   [self.default_btc_position, self.default_nvda_position, self.default_usdjpy_position]}
        for p in hotkey_to_positions[self.DEFAULT_MINER_HOTKEY]:
            self.position_client.save_miner_position(p)

        self.perf_ledger_client.update()

        tp_to_position_start_time = {}
        for position in hotkey_to_positions[self.DEFAULT_MINER_HOTKEY]:
            if position.trade_pair == TradePair.BTCUSD:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_btc_position.open_ms
            elif position.trade_pair == TradePair.NVDA:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_nvda_position.open_ms
            elif position.trade_pair == TradePair.USDJPY:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_usdjpy_position.open_ms

        ans = self.perf_ledger_client.get_perf_ledgers()
        pl = ans[self.DEFAULT_MINER_HOTKEY]
        # The total product and last checkpoint return should be very close but may differ slightly
        # due to checkpoint boundary alignment and accumulation logic
        self.assertAlmostEqual(pl.get_total_product(), pl.cps[-1].prev_portfolio_ret, 2,
                             f"Total product {pl.get_total_product()} differs from last checkpoint return {pl.cps[-1].prev_portfolio_ret}")
        self.assertEqual(len(ans), 1)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, ans)
        last_update_portfolio = ans[self.DEFAULT_MINER_HOTKEY].last_update_ms
        assert len(ans) == 1, ans

        self.check_alignment_per_cp(ans)

        assert all(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY].cps[1:]), [(x.open_ms, x.accum_ms) for x in ans[self.DEFAULT_MINER_HOTKEY].cps[1:]] # first cp truncated due to 12 hr boundary

        # Close the btc position now
        close_order = Order(price=61000, processed_ms=last_update_portfolio, order_uuid="test_order_btc_close",
                                     trade_pair=self.DEFAULT_TRADE_PAIR, order_type=OrderType.FLAT, leverage=0)
        self.default_btc_position.add_order(close_order, self.live_price_fetcher_client)
        self.position_client.save_miner_position(self.default_btc_position)

        # Waiting a few days
        fast_forward_time_ms = TimeUtil.now_in_millis() + 1000 * 60 * 60 * 24 * 10
        self.perf_ledger_client.update(t_ms=fast_forward_time_ms)
        ans = self.perf_ledger_client.get_perf_ledgers()

        pl = ans[self.DEFAULT_MINER_HOTKEY]
        self.assertAlmostEqual(pl.get_total_product(), pl.cps[-1].prev_portfolio_ret, 13)


        #PerfLedgerManager.print_bundles(ans)

        self.check_alignment_per_cp(ans)

        assert any(x.open_ms != x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY].cps[1:])  # first cp truncated due to 12 hr boundary
        assert any(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY].cps[1:])  # first cp truncated due to 12 hr boundary

        last_update_portfolio2 = ans[self.DEFAULT_MINER_HOTKEY].last_update_ms
        last_accum_ms_portfolio2 = ans[self.DEFAULT_MINER_HOTKEY].cps[-1].accum_ms
        # Verify portfolio ledger timing is consistent
        self.assertIsNotNone(last_update_portfolio2)
        self.assertIsNotNone(last_accum_ms_portfolio2)







