# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Position manager tests using RPC mode with persistent servers.

Architecture:
- All servers started once in setUpClass (expensive operation done once)
- Per-test isolation via data clearing (not server restarts)
- PositionManager and all dependencies use RPC mode
- Tests verify proper client/server communication
"""
import random
from copy import deepcopy

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase

from vali_objects.vali_dataclasses.position import Position
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, PerfCheckpoint, TP_ID_PORTFOLIO


class TestPositionManager(TestBase):
    """
    Position manager tests using ServerOrchestrator.

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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')

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
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.default_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )

    def _find_disk_position_from_memory_position(self, position):
        for disk_position in self.position_client.get_positions_for_one_hotkey(position.miner_hotkey):
            if disk_position.position_uuid == position.position_uuid:
                return disk_position
        raise ValueError(f"Could not find position {position.position_uuid} in disk")

    def validate_positions(self, in_memory_position, expected_state):
        disk_position = self._find_disk_position_from_memory_position(in_memory_position)
        success, reason = PositionManagerClient.positions_are_the_same(in_memory_position, expected_state)
        self.assertTrue(success, "In memory position is not as expected. " + reason)
        success, reason = PositionManagerClient.positions_are_the_same(disk_position, expected_state)
        self.assertTrue(success, "Disc position is not as expected. " + reason)

    def test_creating_closing_and_fetching_multiple_positions(self):
        n_trade_pairs = len(TradePair)
        idx_to_position = {}
        # Create 6 orders per trade pair
        for i in range(n_trade_pairs):
            trade_pair = list(TradePair)[i]
            # Create 5 closed positions
            for j in range(5):
                position = deepcopy(self.default_position)
                position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_{j}"
                position.open_ms = self.DEFAULT_OPEN_MS + 100 * i + j
                position.trade_pair = trade_pair
                position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
                position.close_out_position(position.open_ms + 1)
                idx_to_position[(i, j)] = position
                self.position_client.save_miner_position(position)
            # Create 1 open position
            j = 5
            position = deepcopy(self.default_position)
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_{j}"
            position.open_ms = self.DEFAULT_OPEN_MS + 100 * i + j
            position.trade_pair = trade_pair
            position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
            idx_to_position[(i, j)] = position
            self.position_client.save_miner_position(position)

        all_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY, sort_positions=True)
        self.assertEqual(len(all_disk_positions), n_trade_pairs * 6)
        # Ensure the positions in all_disk_positions are sorted by close_ms.
        t0 = all_disk_positions[0].close_ms
        for i in range(n_trade_pairs):
            for j in range(6):
                n = i * 6 + j
                t1 = all_disk_positions[n].close_ms or float('inf')
                # Ensure the timestamp is increasing.
                self.assertTrue(t0 <= t1, 'timestamps not increasing or a valid timestamp came after a None timestamp')
                t0 = t1


        # Fetch all positions and verify that they are the same as the ones we created
        for i in range(n_trade_pairs):
            for j in range(6):
                expected_position = idx_to_position[(i, j)]
                disk_position = self._find_disk_position_from_memory_position(expected_position)
                self.validate_positions(expected_position, disk_position)

    def test_sorting_and_fetching_positions_with_several_open_positions_for_the_same_trade_pair(self):
        num_positions = 100
        open_time_start = 1000
        open_time_end = 2000
        positions = []

        # Generate and save positions
        for i in range(num_positions):
            open_ms = random.randint(open_time_start, open_time_end)
            close_ms = open_ms + random.randint(1, 1000)
            position = deepcopy(self.default_position)
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}"
            position.open_ms = open_ms
            position.close_out_position(close_ms)
            self.position_client.save_miner_position(position)
            positions.append(position)

        # Add two open positions
        for i in range(2):
            position = deepcopy(self.default_position)
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_open_{i}"
            position.open_ms = random.randint(open_time_start, open_time_end)
            if i == 1:
                # ValiRecordsMisalignmentException is raised server-side when trying to save
                # a second open position for the same trade pair.
                # The exception is logged server-side but may not serialize cleanly through RPC.
                with self.assertRaises(Exception):
                    self.position_client.save_miner_position(position)
            else:
                self.position_client.save_miner_position(position)

        all_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(len(all_disk_positions), num_positions + 1)

        open_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY, only_open_positions=True)
        self.assertEqual(len(open_disk_positions), 1)


    def test_sorting_and_fetching_positions_with_random_close_times_all_closed_positions(self):
            num_positions = 100
            open_time_start = 1000
            open_time_end = 2000
            positions = []

            # Generate and save positions
            for i in range(num_positions):
                open_ms = random.randint(open_time_start, open_time_end)
                close_ms = open_ms + random.randint(1, 1000)
                position = deepcopy(self.default_position)
                # Get a random trade pair
                position.trade_pair = random.choice(list(TradePair))
                position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}"
                position.open_ms = open_ms
                if close_ms:
                    position.close_out_position(close_ms)
                self.position_client.save_miner_position(position)
                positions.append(position)

            # Fetch and sort positions from disk
            all_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                    sort_positions=True)

            # Verify the number of positions fetched matches expectations
            self.assertEqual(len(all_disk_positions), num_positions)

            # Verify that positions are sorted correctly by close_ms, treating None as infinity
            for i in range(1, num_positions):
                prev_close_ms = all_disk_positions[i - 1].close_ms or float('inf')
                curr_close_ms = all_disk_positions[i].close_ms or float('inf')
                self.assertTrue(prev_close_ms <= curr_close_ms, "Positions are not sorted correctly by close_ms")


            # Ensure no open positions are fetched
            all_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                    sort_positions=True, only_open_positions=True)
            self.assertEqual(len(all_disk_positions), 0)

    def test_one_close_and_one_open_order_per_position(self):
        open_time_start = 1000
        open_time_end = 2000
        positions = []

        # Generate and save positions
        for i in range(len(TradePair)):
            open_ms = random.randint(open_time_start, open_time_end)
            close_ms = open_ms + random.randint(1, 1000)
            position = deepcopy(self.default_position)
            # Get a random trade pair
            position.trade_pair = list(TradePair)[i]
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_closed"
            position.open_ms = open_ms
            if close_ms:
                position.close_out_position(close_ms)
            self.position_client.save_miner_position(position)
            positions.append(position)

        for i in range(len(TradePair)):
            open_ms = random.randint(open_time_start, open_time_end)
            close_ms = None
            position = deepcopy(self.default_position)
            # Get a random trade pair
            position.trade_pair = list(TradePair)[i]
            position.position_uuid = f"{self.DEFAULT_POSITION_UUID}_{i}_open"
            position.open_ms = open_ms

            self.position_client.save_miner_position(position)
            positions.append(position)

        # Fetch and sort positions from disk
        all_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                sort_positions=True)

        # Verify the number of positions fetched matches expectations
        self.assertEqual(len(all_disk_positions), 2 * len(TradePair))

        # Verify that positions are sorted correctly by close_ms, treating None as infinity
        for i in range(1, 2 * len(TradePair)):
            prev_close_ms = all_disk_positions[i - 1].close_ms or float('inf')
            curr_close_ms = all_disk_positions[i].close_ms or float('inf')
            self.assertTrue(prev_close_ms <= curr_close_ms, "Positions are not sorted correctly by close_ms")

        # Ensure all open positions are fetched
        open_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                 sort_positions=True,
                                                                                 only_open_positions=True)
        self.assertEqual(len(open_disk_positions), len(TradePair))

        all_disk_positions = self.position_client.get_positions_for_one_hotkey(self.DEFAULT_MINER_HOTKEY,
                                                                                sort_positions=True)
        self.assertEqual(len(all_disk_positions), 2 * len(TradePair))

    def test_compute_realtime_drawdown_with_various_drawdowns(self):
        """Test compute_realtime_drawdown across range of drawdown percentages using proper client/server architecture"""
        from time_util.time_util import TimeUtil

        max_portfolio_value = 2.0
        base_time_ms = TimeUtil.now_in_millis()

        # Test cases: (current_value, expected_drawdown_ratio, description)
        test_cases = [
            (2.0, 1.0, "0% drawdown"),
            (1.98, 0.99, "1% drawdown"),
            (1.96, 0.98, "2% drawdown"),
            (1.90, 0.95, "5% drawdown"),
            (1.80, 0.90, "10% drawdown"),
            (1.70, 0.85, "15% drawdown"),
            (1.50, 0.75, "25% drawdown"),
        ]

        for current_value, expected_ratio, description in test_cases:
            with self.subTest(description=description):
                # Clear data for each subtest
                self.position_client.clear_all_miner_positions_and_disk()
                self.perf_ledger_client.save_perf_ledgers({})

                # Create a PerfLedger with checkpoints that have the max portfolio value
                checkpoint = PerfCheckpoint(
                    last_update_ms=base_time_ms,
                    prev_portfolio_ret=1.0,
                    mpv=max_portfolio_value,  # Max portfolio value
                    mdd=1.0
                )
                perf_ledger = PerfLedger(
                    initialization_time_ms=base_time_ms - 1000000,
                    max_return=max_portfolio_value,
                    cps=[checkpoint],
                    tp_id=TP_ID_PORTFOLIO # Mark as portfolio ledger
                )

                # Save the perf ledger in V2 format: {hotkey: {asset_class: PerfLedger}}
                self.perf_ledger_client.save_perf_ledgers({
                    self.DEFAULT_MINER_HOTKEY: {TP_ID_PORTFOLIO: perf_ledger}
                })

                # Create a position with return_at_close = current_value to get the desired current portfolio value
                position = deepcopy(self.default_position)
                position.position_uuid = f"drawdown_test_{description.replace(' ', '_')}"
                position.open_ms = base_time_ms - 1000
                position.close_out_position(base_time_ms)
                # Manually set the return to the desired value for testing
                position.return_at_close = current_value
                self.position_client.save_miner_position(position)

                # Compute drawdown using the actual method through the client
                drawdown = self.position_client.compute_realtime_drawdown(self.DEFAULT_MINER_HOTKEY)

                # Assert
                self.assertAlmostEqual(drawdown, expected_ratio, places=4,
                                      msg=f"{description}: Expected {expected_ratio}, got {drawdown}")

    # ==================== RACE CONDITION TESTS ====================
    # These tests demonstrate race conditions in PositionManager when accessed concurrently.
    # Based on actual access patterns in the codebase (market_order_manager.py, etc.)
    # EXPECTED TO FAIL until proper locking is implemented.

    def test_race_condition_concurrent_saves_different_trade_pairs_index_desync(self):
        """
        Race #1: Index desynchronization when saving positions for different trade pairs concurrently.

        Real scenario: Multiple miners send signals at the same time for different trade pairs.
        The market_order_manager processes these in parallel (different trade pairs = different locks).

        Access pattern from market_order_manager.py:203:
        - Thread A: save_miner_position(position_btc)  [BTC/USD]
        - Thread B: save_miner_position(position_eth)  [ETH/USD]

        Expected failure: Index (hotkey_to_open_positions) gets out of sync with main dict (hotkey_to_positions).
        """
        import threading
        import time

        miner_hotkey = "test_miner_race1"
        exceptions = []

        def save_btc_position():
            try:
                position = deepcopy(self.default_position)
                position.miner_hotkey = miner_hotkey
                position.position_uuid = "btc_position"
                position.trade_pair = TradePair.BTCUSD
                position.open_ms = 1000
                # Simulate processing time
                time.sleep(0.01)
                self.position_client.save_miner_position(position)
            except Exception as e:
                exceptions.append(("BTC", e))

        def save_eth_position():
            try:
                position = deepcopy(self.default_position)
                position.miner_hotkey = miner_hotkey
                position.position_uuid = "eth_position"
                position.trade_pair = TradePair.ETHUSD
                position.open_ms = 1001
                # Simulate processing time
                time.sleep(0.01)
                self.position_client.save_miner_position(position)
            except Exception as e:
                exceptions.append(("ETH", e))

        # Run concurrently (different trade pairs, so no position lock conflict)
        threads = [
            threading.Thread(target=save_btc_position),
            threading.Thread(target=save_eth_position)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for exceptions
        if exceptions:
            self.fail(f"Exceptions during concurrent saves: {exceptions}")

        # Verify both positions were saved
        all_positions = self.position_client.get_positions_for_one_hotkey(miner_hotkey)
        self.assertEqual(len(all_positions), 2, "Both positions should be saved")

        # CRITICAL: Verify index is in sync
        # This is where the race condition manifests - index may have wrong count
        btc_open = self.position_client.get_open_position_for_trade_pair(miner_hotkey, TradePair.BTCUSD.trade_pair_id)
        eth_open = self.position_client.get_open_position_for_trade_pair(miner_hotkey, TradePair.ETHUSD.trade_pair_id)

        # Both should be found in index
        self.assertIsNotNone(btc_open, "BTC position should be in open index")
        self.assertIsNotNone(eth_open, "ETH position should be in open index")

        # Verify UUIDs match
        self.assertEqual(btc_open.position_uuid, "btc_position")
        self.assertEqual(eth_open.position_uuid, "eth_position")

    def test_race_condition_open_to_closed_transition_stale_read(self):
        """
        Race #2: Stale read when position transitions from open to closed.

        Real scenario: One thread closes a position while another reads leverage.
        From market_order_manager.py:193: calculate_net_portfolio_leverage is called
        from position_manager.py:605: iterates over hotkey_to_open_positions

        Access pattern:
        - Thread A: save_miner_position(closed_position)  → transitions open→closed
        - Thread B: calculate_net_portfolio_leverage(hotkey) → reads open positions

        Expected failure: Thread B might see position in open index that's actually closed,
        or get RuntimeError from dict changing during iteration.
        """
        import threading
        import time

        miner_hotkey = "test_miner_race2"

        # Create and save open position
        position = deepcopy(self.default_position)
        position.miner_hotkey = miner_hotkey
        position.position_uuid = "position_to_close"
        position.trade_pair = TradePair.BTCUSD
        position.open_ms = 1000
        self.position_client.save_miner_position(position)

        # Verify it's in the open index
        open_pos = self.position_client.get_open_position_for_trade_pair(miner_hotkey, TradePair.BTCUSD.trade_pair_id)
        self.assertIsNotNone(open_pos, "Position should be open initially")

        leverage_results = []
        exceptions = []

        def close_position():
            """Simulate closing the position (open → closed transition)"""
            try:
                time.sleep(0.005)  # Let reader thread start first
                closed_pos = deepcopy(position)
                closed_pos.close_out_position(2000)
                self.position_client.save_miner_position(closed_pos)
            except Exception as e:
                exceptions.append(("close", e))

        def read_leverage():
            """Simulate leverage calculation (reads open index)"""
            try:
                # Read leverage multiple times to increase race window
                for _ in range(10):
                    leverage = self.position_client.calculate_net_portfolio_leverage(miner_hotkey)
                    leverage_results.append(leverage)
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(("leverage", e))

        # Run concurrently
        threads = [
            threading.Thread(target=close_position),
            threading.Thread(target=read_leverage)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for RuntimeError (dict changed during iteration)
        if exceptions:
            for name, exc in exceptions:
                if isinstance(exc, RuntimeError) and "dictionary changed size" in str(exc):
                    self.fail(f"RuntimeError from dict mutation during iteration: {exc}")

        # Verify final state: position should be closed
        final_pos = self.position_client.get_position(miner_hotkey, "position_to_close")
        self.assertIsNotNone(final_pos)
        self.assertTrue(final_pos.is_closed_position, "Position should be closed")

        # CRITICAL: Verify it's NOT in open index anymore
        open_after_close = self.position_client.get_open_position_for_trade_pair(miner_hotkey, TradePair.BTCUSD.trade_pair_id)
        self.assertIsNone(open_after_close, "Position should NOT be in open index after closing")

    def test_race_condition_duplicate_open_positions_toctou(self):
        """
        Race #3: Duplicate open positions due to TOCTOU in validation.

        Real scenario: Two threads try to open positions for the same trade pair simultaneously.
        From position_manager.py:914-927: save_miner_position has check-then-act gap.

        Access pattern:
        - Thread A: save_miner_position(position_A, BTC/USD)
        - Thread B: save_miner_position(position_B, BTC/USD)  ← DUPLICATE!

        Timeline:
        T1: Thread A validates - no open position found
        T2: Thread B validates - no open position found  ← RACE WINDOW
        T3: Thread A saves position_A
        T4: Thread B saves position_B  ← BOTH SAVED!

        Expected failure: Both positions get saved, violating business rule of
        "one open position per trade pair".

        Note: In production, position_lock prevents this for same hotkey+trade_pair,
        but this test shows what happens WITHOUT the lock (demonstrating the core race).
        """
        import threading

        miner_hotkey = "test_miner_race3"
        saved_positions = []
        exceptions = []

        def save_position_a():
            try:
                position = deepcopy(self.default_position)
                position.miner_hotkey = miner_hotkey
                position.position_uuid = "position_a"
                position.trade_pair = TradePair.BTCUSD
                position.open_ms = 1000
                self.position_client.save_miner_position(position)
                saved_positions.append("A")
            except Exception as e:
                exceptions.append(("A", e))

        def save_position_b():
            try:
                position = deepcopy(self.default_position)
                position.miner_hotkey = miner_hotkey
                position.position_uuid = "position_b"
                position.trade_pair = TradePair.BTCUSD  # SAME trade pair!
                position.open_ms = 1001
                self.position_client.save_miner_position(position)
                saved_positions.append("B")
            except Exception as e:
                exceptions.append(("B", e))

        # Run concurrently
        threads = [
            threading.Thread(target=save_position_a),
            threading.Thread(target=save_position_b)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # One should succeed, one should raise ValiRecordsMisalignmentException
        # But due to race condition, BOTH might succeed (that's the bug!)

        # Check how many actually got saved
        all_positions = self.position_client.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)

        # EXPECTED BEHAVIOR: Only 1 open position (one thread blocked)
        # ACTUAL BUG: Might have 2 open positions (both saved due to TOCTOU)
        if len(all_positions) > 1:
            self.fail(f"RACE CONDITION: Found {len(all_positions)} open positions for same trade pair (should be max 1)")

        # Verify only one position is in the open index
        open_pos = self.position_client.get_open_position_for_trade_pair(miner_hotkey, TradePair.BTCUSD.trade_pair_id)
        self.assertIsNotNone(open_pos, "Should have one open position in index")

        # If both saved, we have a corruption: main dict has 2, but index has only 1 (last write wins)
        self.assertEqual(len(all_positions), 1, "Should have exactly 1 open position")

    def test_race_condition_iteration_during_modification(self):
        """
        Race #4: RuntimeError from dict mutation during iteration.

        Real scenario: Daemon or scorer iterates positions while RPC calls modify them.
        From position_manager.py:1037-1048: compact_price_sources iterates hotkey_to_positions
        From position_manager.py:906-912: get_positions_for_all_miners iterates

        Access pattern:
        - Thread A: get_positions_for_all_miners() → iterates dict
        - Thread B: save_miner_position() → adds new hotkey to dict

        Expected failure: RuntimeError: dictionary changed size during iteration
        """
        import threading
        import time

        # Pre-populate with some positions
        for i in range(10):
            position = deepcopy(self.default_position)
            position.miner_hotkey = f"miner_{i}"
            position.position_uuid = f"position_{i}"
            position.trade_pair = TradePair.BTCUSD
            position.open_ms = 1000 + i
            self.position_client.save_miner_position(position)

        exceptions = []

        def iterator_thread():
            """Simulate daemon or scorer iterating all positions"""
            try:
                for _ in range(20):
                    # This calls get_positions_for_all_miners which iterates hotkey_to_positions
                    all_positions = self.position_client.get_positions_for_all_miners()
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(("iterator", e))

        def modifier_thread():
            """Simulate RPC calls adding new positions"""
            try:
                for i in range(10, 20):
                    position = deepcopy(self.default_position)
                    position.miner_hotkey = f"miner_{i}"  # NEW hotkey
                    position.position_uuid = f"position_{i}"
                    position.trade_pair = TradePair.BTCUSD
                    position.open_ms = 1000 + i
                    self.position_client.save_miner_position(position)
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(("modifier", e))

        # Run concurrently
        threads = [
            threading.Thread(target=iterator_thread),
            threading.Thread(target=modifier_thread)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for RuntimeError
        for name, exc in exceptions:
            if isinstance(exc, RuntimeError) and "dictionary changed size" in str(exc):
                self.fail(f"RuntimeError from {name}: {exc}")

    def test_race_condition_delete_during_save(self):
        """
        Race #5: Lost update when delete and save happen concurrently.

        Real scenario: One thread deletes a position while another saves it.
        From position_manager.py:422-443: delete_position
        From position_manager.py:914-944: save_miner_position

        Expected failure: Non-deterministic outcome - position might exist or not,
        or index might be out of sync.
        """
        import threading

        miner_hotkey = "test_miner_race5"

        # Create and save initial position
        position = deepcopy(self.default_position)
        position.miner_hotkey = miner_hotkey
        position.position_uuid = "position_to_delete_save"
        position.trade_pair = TradePair.BTCUSD
        position.open_ms = 1000
        self.position_client.save_miner_position(position)

        exceptions = []

        def delete_thread():
            try:
                self.position_client.delete_position(miner_hotkey, "position_to_delete_save")
            except Exception as e:
                exceptions.append(("delete", e))

        def save_thread():
            try:
                # Save the same position (might be updating it)
                updated_pos = deepcopy(position)
                updated_pos.open_ms = 2000  # Modify something
                self.position_client.save_miner_position(updated_pos)
            except Exception as e:
                exceptions.append(("save", e))

        # Run concurrently
        threads = [
            threading.Thread(target=delete_thread),
            threading.Thread(target=save_thread)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if exceptions:
            # KeyError is possible if delete happens between dict checks
            for name, exc in exceptions:
                if isinstance(exc, KeyError):
                    self.fail(f"KeyError from {name} indicates race condition: {exc}")

        # Check final state - position might exist or not (non-deterministic)
        final_pos = self.position_client.get_position(miner_hotkey, "position_to_delete_save")
        open_pos = self.position_client.get_open_position_for_trade_pair(miner_hotkey, TradePair.BTCUSD.trade_pair_id)

        # INVARIANT: If position exists in main dict, it should be in index (if open)
        if final_pos is not None and final_pos.is_open_position:
            self.assertIsNotNone(open_pos, "Open position must be in index if in main dict")
            self.assertEqual(open_pos.position_uuid, final_pos.position_uuid, "Index must point to same position")

        # If position doesn't exist in main dict, it shouldn't be in index
        if final_pos is None:
            self.assertIsNone(open_pos, "Position should not be in index if not in main dict")

    def test_race_condition_leverage_calculation_during_position_close(self):
        """
        Race #6: Incorrect leverage calculation when position closes during iteration.

        Real scenario: Leverage calculation iterates open positions while another thread closes one.
        From market_order_manager.py:193: calculate_net_portfolio_leverage is called in critical path.
        From position_manager.py:605-606: iterates hotkey_to_open_positions[hotkey].values()

        This is a CRITICAL race because leverage limits are enforced for financial risk management.
        Wrong leverage → wrong elimination decisions → financial loss.

        Expected failure: RuntimeError during iteration, or wrong leverage value.
        """
        import threading
        import time

        miner_hotkey = "test_miner_race6"

        # Create multiple open positions
        for i, trade_pair in enumerate([TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD]):
            position = deepcopy(self.default_position)
            position.miner_hotkey = miner_hotkey
            position.position_uuid = f"position_{i}"
            position.trade_pair = trade_pair
            position.open_ms = 1000 + i
            self.position_client.save_miner_position(position)

        leverage_readings = []
        exceptions = []

        def leverage_reader():
            """Simulate multiple leverage calculations (like order processing)"""
            try:
                for _ in range(50):
                    leverage = self.position_client.calculate_net_portfolio_leverage(miner_hotkey)
                    leverage_readings.append(leverage)
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(("leverage", e))

        def position_closer():
            """Simulate closing positions (removes from open index)"""
            try:
                time.sleep(0.01)  # Let reader start
                # Close one position
                pos = self.position_client.get_position(miner_hotkey, "position_0")
                pos.close_out_position(2000)
                self.position_client.save_miner_position(pos)
            except Exception as e:
                exceptions.append(("closer", e))

        # Run concurrently
        threads = [
            threading.Thread(target=leverage_reader),
            threading.Thread(target=position_closer)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for RuntimeError
        for name, exc in exceptions:
            if isinstance(exc, RuntimeError) and "dictionary changed size" in str(exc):
                self.fail(f"CRITICAL: RuntimeError during leverage calculation from {name}: {exc}")

        # Verify final state: should have 2 open positions
        open_positions = self.position_client.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)
        self.assertEqual(len(open_positions), 2, "Should have 2 open positions after closing 1")

    def test_race_condition_stress_index_desync_via_client(self):
        """
        STRESS TEST: High concurrency stress test using RPC clients.

        This test creates many concurrent RPC calls to trigger races on the server side.
        The RPC server uses threading to handle concurrent calls, so multiple threads
        on the client side = multiple threads on the server side = race conditions!

        Why this triggers races:
        1. 100 client threads making concurrent RPC calls
        2. Server's BaseManager uses threading to handle these concurrently
        3. No locks in PositionManager = race conditions
        4. Aggressive timing (minimal delays)

        Expected failure: Index desync, duplicate positions, or RuntimeError
        """
        import threading
        import random

        miner_hotkey = "stress_test_miner"
        exceptions = []
        race_detected = []

        def concurrent_saver(thread_id):
            """Aggressively save positions via RPC client"""
            try:
                for i in range(10):
                    # Random trade pair to increase dict mutation variety
                    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, TradePair.SOLUSD]
                    trade_pair = random.choice(trade_pairs)

                    position = Position(
                        miner_hotkey=miner_hotkey,
                        position_uuid=f"pos_t{thread_id}_i{i}",
                        trade_pair=trade_pair,
                        open_ms=1000 + thread_id * 100 + i,
                        account_size=self.DEFAULT_ACCOUNT_SIZE
                    )

                    # RPC call - will be handled by server thread
                    self.position_client.save_miner_position(position)

                    # Immediately check index consistency via client
                    open_pos = self.position_client.get_open_position_for_trade_pair(
                        miner_hotkey,
                        trade_pair.trade_pair_id
                    )
                    main_pos = self.position_client.get_position(miner_hotkey, position.position_uuid)

                    # Check for desync
                    if main_pos is not None and main_pos.is_open_position:
                        if open_pos is None:
                            race_detected.append(f"Thread {thread_id}: Position in main dict but NOT in index!")
                        elif open_pos.position_uuid != main_pos.position_uuid:
                            race_detected.append(f"Thread {thread_id}: Index points to different position!")

            except Exception as e:
                exceptions.append((f"saver_{thread_id}", e))

        def concurrent_reader():
            """Continuously iterate positions to trigger RuntimeError"""
            try:
                for _ in range(100):
                    # These RPC calls iterate dicts on server side
                    # Should trigger "RuntimeError: dictionary changed size during iteration"
                    all_pos = self.position_client.get_positions_for_all_miners()
                    # Also try iterating open positions
                    leverage = self.position_client.calculate_net_portfolio_leverage(miner_hotkey)
            except Exception as e:
                exceptions.append(("reader", e))

        # Create 100 writer threads + 10 reader threads (high concurrency)
        threads = []
        for i in range(100):
            threads.append(threading.Thread(target=concurrent_saver, args=(i,)))
        for i in range(10):
            threads.append(threading.Thread(target=concurrent_reader))

        # Start all threads simultaneously
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Report findings
        if exceptions:
            # Check for RuntimeError (dict mutation during iteration)
            runtime_errors = [e for name, e in exceptions if isinstance(e, RuntimeError)]
            if runtime_errors:
                self.fail(f"RuntimeError detected (dict changed during iteration): {runtime_errors[0]}")

            # Check for KeyError (race in dict access)
            key_errors = [e for name, e in exceptions if isinstance(e, KeyError)]
            if key_errors:
                self.fail(f"KeyError detected (race condition): {key_errors[0]}")

        # Check for index desync
        if race_detected:
            self.fail(f"INDEX DESYNC DETECTED:\n" + "\n".join(race_detected[:5]))

        # Final consistency check via client
        all_positions = self.position_client.get_positions_for_one_hotkey(miner_hotkey)
        for position in all_positions:
            if position.is_open_position:
                trade_pair_id = position.trade_pair.trade_pair_id
                index_pos = self.position_client.get_open_position_for_trade_pair(miner_hotkey, trade_pair_id)

                # This should NEVER fail if index is in sync
                if index_pos is None:
                    self.fail(f"CRITICAL: Open position {position.position_uuid} is in main dict but NOT in index!")

                # Verify it's the correct position in index
                main_dict_positions = [p for p in all_positions
                                      if p.trade_pair == position.trade_pair and p.is_open_position]
                if len(main_dict_positions) > 1:
                    self.fail(f"DUPLICATE OPEN POSITIONS: Found {len(main_dict_positions)} open positions for {trade_pair_id}")

    def test_race_condition_stress_duplicate_positions_via_client(self):
        """
        STRESS TEST: Try to create duplicate open positions by exploiting TOCTOU gap.

        This test hammers the same trade pair with concurrent RPC saves to trigger
        the validation bypass. All threads try to create an open position for the
        same miner/trade_pair combination simultaneously.

        Expected failure: Multiple open positions for same trade pair (violates business rule)
        """
        import threading

        miner_hotkey = "duplicate_stress_miner"
        exceptions = []
        successful_saves = []

        def try_save_duplicate(thread_id):
            """All threads try to save position for SAME trade pair via RPC"""
            try:
                position = Position(
                    miner_hotkey=miner_hotkey,
                    position_uuid=f"duplicate_attempt_{thread_id}",
                    trade_pair=TradePair.BTCUSD,  # SAME trade pair for all threads!
                    open_ms=1000 + thread_id,
                    account_size=self.DEFAULT_ACCOUNT_SIZE
                )

                # RPC save - server will handle concurrently with threading
                self.position_client.save_miner_position(position)
                successful_saves.append(thread_id)

            except Exception as e:
                exceptions.append((f"thread_{thread_id}", e))

        # Launch 50 threads all trying to create open position for same trade pair
        threads = [threading.Thread(target=try_save_duplicate, args=(i,)) for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check results via client
        all_positions = self.position_client.get_positions_for_one_hotkey(miner_hotkey)
        open_positions = [p for p in all_positions
                         if p.is_open_position and p.trade_pair == TradePair.BTCUSD]

        # CRITICAL: Should only have 1 open position, but race might allow multiple
        if len(open_positions) > 1:
            self.fail(
                f"RACE CONDITION: {len(open_positions)} open positions for same trade pair! "
                f"UUIDs: {[p.position_uuid for p in open_positions]}"
            )

        # Should have exactly 1
        self.assertEqual(len(open_positions), 1,
                        f"Should have exactly 1 open position, got {len(open_positions)}")


if __name__ == '__main__':
    import unittest

    unittest.main()
