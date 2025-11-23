# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test elimination manager functionality using modern server/client architecture.
Tests MDD eliminations and zombie detection.
"""
from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import (
    generate_losing_ledger,
    generate_winning_ledger,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationReason
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order


class TestEliminationManager(TestBase):
    """
    Test elimination manager using server/client architecture.
    Uses ServerOrchestrator singleton for shared server infrastructure across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    challenge_period_client = None
    plagiarism_client = None

    MDD_MINER = "miner_mdd"
    REGULAR_MINER = "miner_regular"
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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')

        # Initialize metagraph with test miners
        cls.metagraph_client.set_hotkeys([cls.MDD_MINER, cls.REGULAR_MINER])

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

        # Create fresh test data
        self._setup_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _setup_test_data(self):
        """Helper to create fresh test data for each test."""
        # Set up metagraph with test miners
        self.metagraph_client.set_hotkeys([self.MDD_MINER, self.REGULAR_MINER])

        # Create initial positions for both miners
        for miner in [self.MDD_MINER, self.REGULAR_MINER]:
            mock_position = Position(
                miner_hotkey=miner,
                position_uuid=miner,
                open_ms=1,
                close_ms=2,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=False,
                return_at_close=1.00,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[Order(price=60000, processed_ms=1, order_uuid="initial_order",
                              trade_pair=TradePair.BTCUSD, order_type=OrderType.LONG, leverage=0.1)],
            )
            self.position_client.save_miner_position(mock_position)

        # Set up performance ledgers
        ledgers = {}
        ledgers[self.MDD_MINER] = generate_losing_ledger(0, ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS)
        ledgers[self.REGULAR_MINER] = generate_winning_ledger(0, ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS)
        self.perf_ledger_client.save_perf_ledgers(ledgers)

        # Set up challenge period status
        miners = {}
        miners[self.MDD_MINER] = (MinerBucket.MAINCOMP, 0, None, None)
        miners[self.REGULAR_MINER] = (MinerBucket.MAINCOMP, 0, None, None)
        self.challenge_period_client.update_miners(miners)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

    def test_elimination_for_mdd(self):
        """Test MDD elimination and zombie detection"""
        # Neither miner has been eliminated initially
        self.assertEqual(len(self.challenge_period_client.get_success_miners()), 2)

        # Process eliminations (no position_locks parameter needed)
        self.elimination_client.process_eliminations()

        # Check MDD miner was eliminated
        eliminations = self.elimination_client.get_eliminations_from_disk()
        self.assertEqual(len(eliminations), 1)
        for elimination in eliminations:
            self.assertEqual(elimination["hotkey"], self.MDD_MINER)
            self.assertEqual(elimination["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)

        # Test zombie eliminations - remove all miners from metagraph
        self.metagraph_client.set_hotkeys([])
        self.elimination_client.process_eliminations()

        # Both miners should now be eliminated
        eliminations = self.elimination_client.get_eliminations_from_disk()
        self.assertEqual(len(eliminations), 2)

        for elimination in eliminations:
            if elimination["hotkey"] == self.MDD_MINER:
                # MDD miner keeps original MDD reason
                self.assertEqual(elimination["reason"], EliminationReason.MAX_TOTAL_DRAWDOWN.value)
            elif elimination["hotkey"] == self.REGULAR_MINER:
                # Regular miner becomes zombie
                self.assertEqual(elimination["reason"], EliminationReason.ZOMBIE.value)
            else:
                raise Exception(f"Unexpected hotkey in eliminations: {elimination['hotkey']}")

    # ==================== Race Condition Tests ====================
    # These tests demonstrate race conditions that exist due to missing lock usage.
    # They are EXPECTED to fail/flake until proper locking is implemented.
    # Each test models real access patterns from production code.

    def test_race_concurrent_append_elimination_row_disk_corruption(self):
        """
        Test RC-1 & RC-10: Concurrent append_elimination_row() causes disk corruption.

        Real-world scenario:
        - Thread 1: handle_mdd_eliminations() calls append_elimination_row("miner1", ...)
        - Thread 2: handle_challenge_period_eliminations() calls append_elimination_row("miner2", ...)
        - Both call save_eliminations() → write same file → last-write-wins → data loss

        Expected success (with locks): All 20 eliminations saved correctly.
        """
        import threading
        import time

        # Clear eliminations
        self.elimination_client.clear_eliminations()

        # Generate 20 test miners
        test_miners = [f"race_miner_{i}" for i in range(20)]

        results = {"success": 0, "errors": []}

        def append_elimination(hotkey):
            """Simulate concurrent elimination from different handlers via RPC"""
            try:
                # Use client API - this creates RPC calls that the server handles concurrently
                self.elimination_client.append_elimination_row(
                    hotkey=hotkey,
                    current_dd=0.08,
                    reason="RACE_TEST_MDD",
                    t_ms=1000 + hash(hotkey) % 1000  # Different timestamps
                )
                results["success"] += 1
            except Exception as e:
                results["errors"].append(str(e))

        # Launch 20 concurrent threads (simulates multiple RPC calls arriving simultaneously)
        threads = [threading.Thread(target=append_elimination, args=(miner,)) for miner in test_miners]

        # Start all threads simultaneously
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Give filesystem a moment to settle
        time.sleep(0.1)

        # Verify results
        self.assertEqual(len(results["errors"]), 0, f"Unexpected errors: {results['errors']}")

        # Check memory state
        eliminations_in_memory = self.elimination_client.get_eliminations_from_memory()
        self.assertEqual(len(eliminations_in_memory), 20,
                        f"Expected 20 eliminations in memory, got {len(eliminations_in_memory)}")

        # Check disk state (WITH LOCKS: All should be saved)
        eliminations_from_disk = self.elimination_client.get_eliminations_from_disk()

        # WITH LOCKS: All 20 eliminations should be on disk
        self.assertEqual(len(eliminations_from_disk), 20,
                        f"Expected 20 eliminations on disk, got {len(eliminations_from_disk)}. "
                        f"Lock protection ensures all concurrent writes are serialized.")

    def test_race_sync_eliminations_clear_window(self):
        """
        Test RC-3: sync_eliminations() clear window causes empty dict reads.

        Real-world scenario:
        - Thread 1: validator_sync_base.py calls sync_eliminations() (clears dict, repopulates)
        - Thread 2: Daemon calls process_eliminations() → handle_mdd_eliminations() → reads eliminations
        - Thread 2 reads between clear and repopulate → sees empty dict
        - Thread 2 thinks no miners eliminated → re-eliminates already-eliminated miners

        Expected success (with locks): Reader never sees 0 eliminations.
        """
        import threading
        import time

        # Clear and prepopulate with 50 eliminations
        self.elimination_client.clear_eliminations()
        initial_miners = [f"initial_miner_{i}" for i in range(50)]
        for miner in initial_miners:
            self.elimination_client.append_elimination_row(
                hotkey=miner,
                current_dd=0.05,
                reason="INITIAL_SETUP"
            )

        # Verify setup
        self.assertEqual(len(self.elimination_client.get_eliminations_from_memory()), 50)

        read_results = []
        stop_reading = threading.Event()

        def continuous_reader():
            """Continuously read eliminations via client (simulates concurrent reads)"""
            while not stop_reading.is_set():
                eliminations = self.elimination_client.get_eliminations_from_memory()
                read_results.append(len(eliminations))
                time.sleep(0.0001)  # Tight loop to catch any race window

        def sync_operation():
            """Sync to new elimination list (simulates validator sync)"""
            time.sleep(0.01)  # Let reader get going first

            # Create new list of 30 eliminations (different miners)
            new_eliminations = [
                {
                    'hotkey': f"synced_miner_{i}",
                    'dd': 0.10,
                    'reason': 'SYNCED',
                    'elimination_initiated_time_ms': TimeUtil.now_in_millis(),
                    'price_info': {},
                    'return_info': {}
                }
                for i in range(30)
            ]

            # Call sync_eliminations via client (this clears then repopulates)
            self.elimination_client.sync_eliminations(new_eliminations)

        # Start reader thread
        reader = threading.Thread(target=continuous_reader, daemon=True)
        reader.start()

        # Start sync operation
        syncer = threading.Thread(target=sync_operation)
        syncer.start()
        syncer.join()

        # Let reader run a bit more
        time.sleep(0.05)
        stop_reading.set()
        reader.join(timeout=1.0)

        # Analyze results
        # WITH LOCKS: Reader should NEVER see 0 eliminations during sync
        zero_reads = read_results.count(0)

        self.assertEqual(zero_reads, 0,
                        f"Reader saw empty dict {zero_reads} times during sync! "
                        f"Lock should prevent readers from seeing empty dict. "
                        f"Sample reads: {read_results[:20]}")

    def test_race_iteration_during_modification_crash(self):
        """
        Test RC-4: process_eliminations() is safe during concurrent append operations.

        Real-world scenario:
        - Thread 1: Daemon calls process_eliminations() which internally iterates eliminations
        - Thread 2: RPC call to append_elimination_row() modifies dict
        - WITHOUT FIX: Python raises RuntimeError: dictionary changed size during iteration
        - WITH FIX: Snapshot pattern prevents crash

        Expected success: No crash, both operations complete successfully.
        """
        import threading
        import time

        # Prepopulate with 50 eliminations
        self.elimination_client.clear_eliminations()
        current_time_ms = TimeUtil.now_in_millis()
        # Use recent timestamps so eliminations won't be deleted during test
        for i in range(50):
            self.elimination_client.append_elimination_row(
                hotkey=f"iter_miner_{i}",
                current_dd=0.05,
                reason="ITER_TEST",
                t_ms=current_time_ms  # Recent timestamp - won't be deleted
            )

        iteration_results = {"crashed": False, "error": None, "completed": False}
        modification_count = {"count": 0}

        def process_thread():
            """Call process_eliminations which iterates over eliminations"""
            try:
                # This calls _delete_eliminated_expired_miners internally
                # which uses snapshot pattern to avoid crashes
                self.elimination_client.process_eliminations()
                iteration_results["completed"] = True
            except RuntimeError as e:
                if "dictionary changed size during iteration" in str(e):
                    iteration_results["crashed"] = True
                    iteration_results["error"] = str(e)
                else:
                    raise

        def modifier_thread():
            """Add more eliminations during iteration (simulates concurrent RPC calls)"""
            time.sleep(0.005)  # Let process_eliminations get started

            for i in range(50, 70):
                self.elimination_client.append_elimination_row(
                    hotkey=f"new_miner_{i}",
                    current_dd=0.06,
                    reason="CONCURRENT_ADD",
                    t_ms=current_time_ms  # Recent timestamp
                )
                modification_count["count"] += 1
                time.sleep(0.001)

        # Start both threads
        processor = threading.Thread(target=process_thread)
        modifier = threading.Thread(target=modifier_thread)

        processor.start()
        modifier.start()

        processor.join()
        modifier.join()

        # Verify modifications happened
        self.assertGreater(modification_count["count"], 0, "Modifier thread didn't run")

        # WITH FIX: Should NOT crash because snapshot pattern protects iteration
        self.assertFalse(iteration_results["crashed"],
                        f"process_eliminations crashed with RuntimeError: {iteration_results['error']}. "
                        f"The snapshot pattern should prevent this crash.")

        self.assertTrue(iteration_results["completed"],
                       "process_eliminations should have completed successfully using snapshot pattern.")

    def test_race_concurrent_departed_hotkeys_updates(self):
        """
        Test RC-6: Concurrent process_eliminations() calls safely track departed hotkeys.

        Real-world scenario:
        - Thread 1: Daemon calls process_eliminations() → _update_departed_hotkeys()
        - Thread 2: Client calls process_eliminations() via RPC → _update_departed_hotkeys()
        - Both read previous_metagraph_hotkeys, both modify departed_hotkeys
        - WITHOUT FIX: Race causes lost departed hotkey tracking
        - WITH FIX: Lock ensures atomic updates

        Expected success (with locks): All departures tracked correctly.
        """
        import threading
        import time

        # Clear eliminations first
        self.elimination_client.clear_eliminations()

        # Setup initial metagraph state with 10 hotkeys
        initial_hotkeys = [f"departed_test_{i}" for i in range(10)]
        self.metagraph_client.set_hotkeys(initial_hotkeys)

        # Clear departed hotkeys AFTER setting metagraph (this also resets previous_metagraph_hotkeys)
        # This ensures we start with clean state and don't track setUp() hotkeys as departed
        self.elimination_client.clear_departed_hotkeys()

        # NOW remove 5 hotkeys from metagraph (departed_test_0 through departed_test_4)
        remaining_hotkeys = [f"departed_test_{i}" for i in range(5, 10)]
        self.metagraph_client.set_hotkeys(remaining_hotkeys)

        def process_eliminations_thread():
            """Call process_eliminations via client (models real RPC calls)"""
            # Small delay to increase race window (all threads call at similar time)
            time.sleep(0.001)

            # Call process_eliminations via client API (this internally calls _update_departed_hotkeys)
            # In production, this would be an RPC call from another process
            # With locks: Should safely detect the departed hotkeys
            # Without locks: Threads race, some updates lost
            self.elimination_client.process_eliminations()

        # Launch 5 concurrent threads, all calling process_eliminations()
        # This models: daemon thread + multiple concurrent RPC calls from clients
        threads = [threading.Thread(target=process_eliminations_thread) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify departed hotkeys were tracked (use client API)
        departed_hotkeys = self.elimination_client.get_departed_hotkeys()

        # WITH LOCKS: All 5 departures should be tracked (first thread detects and records them)
        # The lock ensures atomic read-modify-write of departed_hotkeys and previous_metagraph_hotkeys
        # Subsequent threads see previous_metagraph_hotkeys already updated, so don't re-track
        self.assertEqual(len(departed_hotkeys), 5,
                        f"Expected 5 departed hotkeys, got {len(departed_hotkeys)}. "
                        f"Departed: {list(departed_hotkeys.keys())}. "
                        f"Lock should ensure all departures are tracked correctly.")
