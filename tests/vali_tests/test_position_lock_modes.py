"""
Test PositionLocks with different modes (local, ipc, rpc).
"""
import time
import threading
import unittest
from vali_objects.utils.position_lock import PositionLocks


class TestPositionLockModes(unittest.TestCase):
    """Test all three lock modes work correctly"""

    def test_local_mode(self):
        """Test local mode (threading locks)"""
        locks = PositionLocks(mode='local')

        miner_hotkey = "test_miner"
        trade_pair_id = "BTCUSD"

        # Test basic lock acquisition and release
        with locks.get_lock(miner_hotkey, trade_pair_id):
            # Lock is held
            pass

        # Lock is released
        self.assertEqual(locks.mode, 'local')

    def test_local_mode_concurrency(self):
        """Test local mode handles concurrent access"""
        locks = PositionLocks(mode='local')

        miner_hotkey = "test_miner"
        trade_pair_id = "BTCUSD"

        results = []

        def worker(worker_id):
            with locks.get_lock(miner_hotkey, trade_pair_id):
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
        # Results should be like: [start_0, end_0, start_1, end_1, start_2, end_2]
        # NOT interleaved like: [start_0, start_1, end_0, end_1]
        self.assertEqual(len(results), 6)

        # Check that each worker completed atomically
        for i in range(3):
            start_idx = results.index(f"start_{i}")
            end_idx = results.index(f"end_{i}")
            # End should immediately follow start (lock held)
            self.assertEqual(end_idx, start_idx + 1,
                           f"Worker {i} was not atomic: start at {start_idx}, end at {end_idx}")

    def test_ipc_mode(self):
        """Test IPC mode (multiprocessing Manager locks)"""
        locks = PositionLocks(mode='ipc')

        miner_hotkey = "test_miner"
        trade_pair_id = "ETHUSD"

        # Test basic lock acquisition and release
        with locks.get_lock(miner_hotkey, trade_pair_id):
            # Lock is held
            pass

        # Lock is released
        self.assertEqual(locks.mode, 'ipc')

    def test_rpc_mode(self):
        """Test RPC mode (lock server)"""
        locks = PositionLocks(mode='rpc', running_unit_tests=True)

        miner_hotkey = "test_miner"
        trade_pair_id = "SOLUSD"

        # Test basic lock acquisition and release
        with locks.get_lock(miner_hotkey, trade_pair_id):
            # Lock is held
            pass

        # Lock is released
        self.assertEqual(locks.mode, 'rpc')

        # Cleanup
        locks.shutdown()

    def test_rpc_mode_health_check(self):
        """Test RPC mode health check"""
        locks = PositionLocks(mode='rpc', running_unit_tests=True)

        # Health check should return True
        is_healthy = locks.health_check()
        self.assertTrue(is_healthy)

        # Cleanup
        locks.shutdown()

    def test_rpc_mode_concurrency(self):
        """Test RPC mode handles concurrent access"""
        locks = PositionLocks(mode='rpc', running_unit_tests=True)

        miner_hotkey = "test_miner"
        trade_pair_id = "ETHUSD"

        results = []

        def worker(worker_id):
            with locks.get_lock(miner_hotkey, trade_pair_id):
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

        # Verify that locks worked
        self.assertEqual(len(results), 6)

        # Check that each worker completed atomically
        for i in range(3):
            start_idx = results.index(f"start_{i}")
            end_idx = results.index(f"end_{i}")
            # End should immediately follow start (lock held)
            self.assertEqual(end_idx, start_idx + 1,
                           f"Worker {i} was not atomic: start at {start_idx}, end at {end_idx}")

        # Cleanup
        locks.shutdown()

    def test_legacy_is_backtesting_param(self):
        """Test backward compatibility with is_backtesting parameter"""
        locks = PositionLocks(is_backtesting=True)
        self.assertEqual(locks.mode, 'local')

    def test_legacy_use_ipc_param(self):
        """Test backward compatibility with use_ipc parameter"""
        locks = PositionLocks(use_ipc=True)
        self.assertEqual(locks.mode, 'ipc')

    def test_mode_override(self):
        """Test that explicit mode parameter overrides legacy params"""
        # mode='rpc' should win even with use_ipc=True
        locks = PositionLocks(use_ipc=True, mode='rpc')
        self.assertEqual(locks.mode, 'rpc')

        # mode='local' should win even with is_backtesting=False
        locks = PositionLocks(is_backtesting=False, mode='local')
        self.assertEqual(locks.mode, 'local')


if __name__ == '__main__':
    unittest.main()
