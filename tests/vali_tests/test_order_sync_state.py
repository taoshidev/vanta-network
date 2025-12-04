# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Unit tests for OrderSyncState class.

Tests the thread-safe state management for coordinating order processing vs. position sync.
"""
import threading
import time
import unittest

from vali_objects.data_sync.order_sync_state import OrderSyncState


class TestOrderSyncState(unittest.TestCase):
    """Test OrderSyncState functionality."""

    def test_basic_counter(self):
        """Test basic increment/decrement functionality."""
        sync = OrderSyncState()

        self.assertEqual(sync.get_order_count(), 0, "Initial count should be 0")

        sync.increment_order_count()
        self.assertEqual(sync.get_order_count(), 1, "Count should be 1 after increment")

        sync.decrement_order_count()
        self.assertEqual(sync.get_order_count(), 0, "Count should be 0 after decrement")

    def test_context_manager(self):
        """Test context manager auto-increment/decrement."""
        sync = OrderSyncState()

        self.assertEqual(sync.get_order_count(), 0)

        with sync.begin_order():
            self.assertEqual(sync.get_order_count(), 1, "Count should be 1 inside context")

        self.assertEqual(sync.get_order_count(), 0, "Count should be 0 after context exit")

    def test_context_manager_with_exception(self):
        """Test context manager decrements even on exception."""
        sync = OrderSyncState()

        with self.assertRaises(ValueError):
            with sync.begin_order():
                self.assertEqual(sync.get_order_count(), 1)
                raise ValueError("Test exception")

        self.assertEqual(sync.get_order_count(), 0, "Count should be 0 even after exception")

    def test_sync_waiting_flag(self):
        """Test sync_waiting flag."""
        sync = OrderSyncState()

        self.assertFalse(sync.is_sync_waiting(), "Should not be waiting initially")

        # Simulate sync starting to wait
        def sync_thread():
            with sync.begin_sync():
                time.sleep(0.1)

        thread = threading.Thread(target=sync_thread)
        thread.start()
        time.sleep(0.05)  # Let sync start waiting

        self.assertTrue(sync.is_sync_waiting(), "Should be waiting during sync")

        thread.join()

        self.assertFalse(sync.is_sync_waiting(), "Should not be waiting after sync completes")

    def test_wait_for_orders(self):
        """Test that sync waits for orders to complete."""
        sync = OrderSyncState()

        # Start an order
        sync.increment_order_count()

        # Try to start sync (should wait)
        sync_started = [False]
        sync_completed = [False]

        def sync_thread():
            sync_started[0] = True
            with sync.begin_sync():
                sync_completed[0] = True

        thread = threading.Thread(target=sync_thread)
        thread.start()
        time.sleep(0.05)  # Let sync start

        self.assertTrue(sync_started[0], "Sync thread should have started")
        self.assertFalse(sync_completed[0], "Sync should be waiting for order")
        self.assertTrue(sync.is_sync_waiting(), "Sync should be in waiting state")

        # Complete the order
        sync.decrement_order_count()

        thread.join(timeout=1.0)

        self.assertTrue(sync_completed[0], "Sync should complete after order finishes")
        self.assertFalse(sync.is_sync_waiting(), "Sync should no longer be waiting")

    def test_multiple_concurrent_orders(self):
        """Test multiple orders incrementing/decrementing concurrently."""
        sync = OrderSyncState()

        def process_order(order_id):
            with sync.begin_order():
                time.sleep(0.01)  # Simulate order processing

        # Start 5 concurrent orders
        threads = [threading.Thread(target=process_order, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # All orders should be done
        self.assertEqual(sync.get_order_count(), 0, "All orders should be complete")
        self.assertFalse(sync.is_sync_waiting(), "Sync should not be waiting")

    def test_get_state_dict(self):
        """Test get_state_dict returns correct info."""
        sync = OrderSyncState()

        state = sync.get_state_dict()

        self.assertIn('n_orders_being_processed', state)
        self.assertIn('sync_waiting', state)
        self.assertIn('last_sync_start_ms', state)
        self.assertIn('last_sync_complete_ms', state)
        self.assertIn('time_since_last_sync_ms', state)

        self.assertEqual(state['n_orders_being_processed'], 0)
        self.assertEqual(state['sync_waiting'], False)

    def test_repr(self):
        """Test string representation."""
        sync = OrderSyncState()

        repr_str = repr(sync)
        self.assertIn('OrderSyncState', repr_str)
        self.assertIn('orders=', repr_str)
        self.assertIn('sync_waiting=', repr_str)

    def test_sync_tracking_timestamps(self):
        """Test that sync timestamps are tracked correctly."""
        sync = OrderSyncState()

        # Initial state
        state = sync.get_state_dict()
        self.assertEqual(state['last_sync_start_ms'], 0)
        self.assertEqual(state['last_sync_complete_ms'], 0)
        self.assertIsNone(state['time_since_last_sync_ms'])

        # Perform a sync
        with sync.begin_sync():
            time.sleep(0.01)

        # Check timestamps were updated
        state = sync.get_state_dict()
        self.assertGreater(state['last_sync_start_ms'], 0, "Sync start should be tracked")
        self.assertGreater(state['last_sync_complete_ms'], 0, "Sync complete should be tracked")
        self.assertIsNotNone(state['time_since_last_sync_ms'], "Time since sync should be calculated")
        self.assertGreaterEqual(state['time_since_last_sync_ms'], 0, "Time since sync should be non-negative")

    def test_early_rejection_scenario(self):
        """Test the early rejection use case (order arrives while sync is waiting)."""
        sync = OrderSyncState()

        # Start an order
        sync.increment_order_count()

        # Start sync (will wait for order)
        rejection_count = [0]

        def sync_thread():
            with sync.begin_sync():
                time.sleep(0.1)

        thread = threading.Thread(target=sync_thread)
        thread.start()
        time.sleep(0.05)  # Let sync start waiting

        # Simulate new orders arriving - they should be rejected
        for _ in range(3):
            if sync.is_sync_waiting():
                rejection_count[0] += 1

        self.assertEqual(rejection_count[0], 3, "All 3 orders should have been rejected")

        # Complete the first order
        sync.decrement_order_count()
        thread.join()

        # Now sync is done, new orders should not be rejected
        self.assertFalse(sync.is_sync_waiting())


if __name__ == '__main__':
    unittest.main()
