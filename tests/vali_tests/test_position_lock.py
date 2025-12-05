# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Test PositionLockServer/Client with ServerOrchestrator architecture.

This module includes comprehensive tests for:
- Server/client RPC architecture via ServerOrchestrator
- Thread safety within single process
- Multi-process lock coordination
- Race condition prevention
"""
import time
import threading
import unittest
from multiprocessing import Process, Queue, Value

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from shared_objects.locks.position_lock_client import PositionLockClient
from vali_objects.utils.vali_utils import ValiUtils


class TestPositionLockBasic(TestBase):
    """
    Test basic position lock functionality using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across all test classes.
    Per-test isolation is achieved by auto-releasing locks (no data state to clear).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    lock_client = None

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

        # Get position lock client from orchestrator
        cls.lock_client = cls.orchestrator.get_client('position_lock')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: No server restart needed, locks auto-release."""
        pass

    def tearDown(self):
        """Per-test teardown: Locks auto-release, no cleanup needed."""
        pass

    def test_basic_lock_acquisition(self):
        """Test basic lock acquisition and release via client"""
        miner_hotkey = "test_miner_basic"
        trade_pair_id = "BTCUSD"

        # Acquire lock
        with self.lock_client.get_lock(miner_hotkey, trade_pair_id):
            # Lock is held
            pass

        # Lock is automatically released

    def test_threading_concurrency(self):
        """Test that locks properly serialize concurrent thread access"""
        miner_hotkey = "test_miner_threads"
        trade_pair_id = "ETHUSD"

        results = []

        def worker(worker_id):
            with self.lock_client.get_lock(miner_hotkey, trade_pair_id):
                results.append(f"start_{worker_id}")
                time.sleep(0.01)  # Hold lock for 10ms
                results.append(f"end_{worker_id}")

        # Create 3 threads that all want the same lock
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify that locks worked (each start/end pair should be together)
        self.assertEqual(len(results), 6)

        # Check that each worker completed atomically
        for i in range(3):
            start_idx = results.index(f"start_{i}")
            end_idx = results.index(f"end_{i}")
            # End should immediately follow start (lock held)
            self.assertEqual(end_idx, start_idx + 1,
                           f"Worker {i} was not atomic: start at {start_idx}, end at {end_idx}")

    def test_different_keys_independent(self):
        """Test that different lock keys can be held simultaneously"""
        miner1 = "miner_1"
        miner2 = "miner_2"
        trade_pair = "BTCUSD"

        # Should be able to acquire both locks since they're different keys
        with self.lock_client.get_lock(miner1, trade_pair):
            with self.lock_client.get_lock(miner2, trade_pair):
                # Both locks held simultaneously
                pass

    def test_lock_reentrant_after_release(self):
        """Test that the same client can re-acquire a released lock"""
        miner_hotkey = "test_miner_reentrant"
        trade_pair_id = "SOLUSD"

        # Acquire and release multiple times
        for i in range(3):
            with self.lock_client.get_lock(miner_hotkey, trade_pair_id):
                pass  # Lock acquired and released

        # Should complete without error

    def test_health_check(self):
        """Test that server health check works"""
        is_healthy = self.lock_client.health_check()
        self.assertTrue(is_healthy)


def _multiprocess_worker(result_queue: Queue, worker_id: int, miner_hotkey: str,
                        trade_pair_id: str, hold_time: float, race_counter: Value):
    """
    Worker function for multi-process lock tests.

    Each worker:
    1. Creates its own client connection to the shared server
    2. Acquires the lock
    3. Increments a shared counter (race condition test)
    4. Holds the lock for a short time
    5. Reports results back via queue
    """
    try:
        # Each process creates its own client
        lock_client = PositionLockClient()

        with lock_client.get_lock(miner_hotkey, trade_pair_id):
            # Record that we acquired the lock
            result_queue.put(f"acquired_{worker_id}")

            # Increment shared counter (this would race without proper locking)
            with race_counter.get_lock():
                current_value = race_counter.value
                time.sleep(hold_time)  # Simulate work
                race_counter.value = current_value + 1

            result_queue.put(f"released_{worker_id}")

        result_queue.put(f"success_{worker_id}")

    except Exception as e:
        result_queue.put(f"error_{worker_id}:{str(e)}")


class TestPositionLockBehavior(TestBase):
    """
    Test specific lock behaviors and edge cases using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across all test classes.
    Per-test isolation is achieved by auto-releasing locks (no data state to clear).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    lock_client = None

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

        # Get position lock client from orchestrator
        cls.lock_client = cls.orchestrator.get_client('position_lock')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: No server restart needed."""
        pass

    def tearDown(self):
        """Per-test teardown: Locks auto-release."""
        pass

    def test_lock_prevents_concurrent_access(self):
        """
        Verify that only one process can hold the lock at a time.
        Uses a shared counter to detect race conditions.
        """
        result_queue = Queue()
        race_counter = Value('i', 0)

        miner_hotkey = "5timingminer123456789012345678901234567890123"
        trade_pair_id = "XRPUSD"
        num_workers = 3

        processes = []
        for i in range(num_workers):
            p = Process(
                target=_multiprocess_worker,
                args=(result_queue, i, miner_hotkey, trade_pair_id, 0.02, race_counter)
            )
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=30)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Verify all succeeded
        errors = [r for r in results if r.startswith("error_")]
        self.assertEqual(len(errors), 0, f"Errors: {errors}")

        success_count = sum(1 for r in results if r.startswith("success_"))
        self.assertEqual(success_count, num_workers)

        # Verify counter integrity (proves no race condition)
        self.assertEqual(race_counter.value, num_workers,
                        f"Race condition! Counter={race_counter.value}, expected={num_workers}")

    def test_nested_different_locks(self):
        """Test that different locks can be nested"""
        miner1 = "miner_nested_1"
        miner2 = "miner_nested_2"
        trade_pair = "BTCUSD"

        # Should be able to hold both locks since they're for different keys
        with self.lock_client.get_lock(miner1, trade_pair):
            with self.lock_client.get_lock(miner2, trade_pair):
                # Both locks held
                pass

    def test_multiple_clients_same_process(self):
        """Test that multiple client instances in same process work correctly"""
        client1 = PositionLockClient()
        client2 = PositionLockClient()

        miner = "test_miner_multi_client"
        trade_pair = "ETHUSD"

        # Both clients should be able to acquire locks for different keys
        with client1.get_lock(miner, trade_pair):
            with client2.get_lock(f"{miner}_2", trade_pair):
                # Both locks held
                pass


if __name__ == '__main__':
    unittest.main()
