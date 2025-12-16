# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
OrderSyncState - Thread-safe state tracking for order processing vs. position sync coordination.

This replaces the hacky `n_orders_being_processed = [0]` pattern with a proper class.
"""
import threading
from time_util.time_util import TimeUtil


class OrderSyncState:
    """
    Thread-safe state tracker for coordinating order processing and position sync.

    Replaces the pattern of passing around:
    - signal_sync_lock (threading.Lock)
    - signal_sync_condition (threading.Condition)
    - n_orders_being_processed ([0])  # List-of-size-1 hack

    With a single, cleaner object that encapsulates all related state.

    Usage in validator.py:
        # Initialize
        self.order_sync = OrderSyncState()

        # In receive_signal()
        if self.order_sync.is_sync_waiting():
            synapse.error_message = "Sync in progress"
            return synapse

        with self.order_sync.begin_order():
            # Process order...
            pass
        # Auto-decrements on context exit

    Usage in PositionSyncer:
        # Wait for orders to complete
        self.order_sync.wait_for_orders()

        # Perform sync with automatic flag management
        with self.order_sync.begin_sync():
            # Sync positions...
            pass
    """

    def __init__(self):
        # Core state
        self._n_orders_being_processed = 0
        self._sync_waiting = False
        self._last_sync_start_ms = 0
        self._last_sync_complete_ms = 0

        # Synchronization primitives
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    # ==================== Order Processing Methods ====================

    def increment_order_count(self) -> int:
        """
        Increment the order counter (called when order processing starts).

        Returns:
            New order count after increment
        """
        with self._lock:
            self._n_orders_being_processed += 1
            return self._n_orders_being_processed

    def decrement_order_count(self) -> int:
        """
        Decrement the order counter and notify waiters if count reaches 0.

        Returns:
            New order count after decrement
        """
        with self._lock:
            self._n_orders_being_processed -= 1
            if self._n_orders_being_processed == 0:
                self._condition.notify_all()
            return self._n_orders_being_processed

    def get_order_count(self) -> int:
        """Get current number of orders being processed (thread-safe read)."""
        with self._lock:
            return self._n_orders_being_processed

    class OrderContext:
        """Context manager for order processing (auto-increment/decrement)."""
        def __init__(self, state: 'OrderSyncState'):
            self.state = state

        def __enter__(self):
            self.state.increment_order_count()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.state.decrement_order_count()
            return False  # Don't suppress exceptions

    def begin_order(self) -> OrderContext:
        """
        Context manager for order processing (auto-increments/decrements counter).

        Usage:
            with order_sync.begin_order():
                # Process order...
                pass
        """
        return self.OrderContext(self)

    # ==================== Sync Coordination Methods ====================

    def is_sync_waiting(self) -> bool:
        """
        Check if sync is waiting for orders to complete (thread-safe, fast).

        Use this in receive_signal() for early rejection:
            if self.order_sync.is_sync_waiting():
                return "Sync in progress"
        """
        with self._lock:
            return self._sync_waiting

    def wait_for_orders(self, timeout_seconds: float = None) -> bool:
        """
        Wait for all in-flight orders to complete (blocks until count == 0).

        This is called by PositionSyncer before starting sync.

        Args:
            timeout_seconds: Optional timeout (None = wait forever)

        Returns:
            True if orders completed, False if timeout
        """
        with self._lock:
            # Set sync_waiting flag BEFORE waiting
            self._sync_waiting = True
            self._last_sync_start_ms = TimeUtil.now_in_millis()

            # Wait for order count to reach 0
            while self._n_orders_being_processed > 0:
                if timeout_seconds is not None:
                    # Wait with timeout
                    if not self._condition.wait(timeout=timeout_seconds):
                        # Timeout occurred
                        self._sync_waiting = False
                        return False
                else:
                    # Wait indefinitely
                    self._condition.wait()

            # Orders complete, sync can proceed
            return True

    def mark_sync_complete(self):
        """Mark sync as complete (clears sync_waiting flag)."""
        with self._lock:
            self._sync_waiting = False
            self._last_sync_complete_ms = TimeUtil.now_in_millis()

    class SyncContext:
        """Context manager for sync operations (auto-manages sync_waiting flag)."""
        def __init__(self, state: 'OrderSyncState', timeout_seconds: float = None):
            self.state = state
            self.timeout_seconds = timeout_seconds
            self.acquired = False

        def __enter__(self):
            self.acquired = self.state.wait_for_orders(self.timeout_seconds)
            if not self.acquired:
                raise TimeoutError("Timeout waiting for orders to complete")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.state.mark_sync_complete()
            return False  # Don't suppress exceptions

    def begin_sync(self, timeout_seconds: float = None) -> SyncContext:
        """
        Context manager for sync operations (auto-waits for orders and clears flag).

        Usage:
            with order_sync.begin_sync():
                # Sync positions...
                pass
        """
        return self.SyncContext(self, timeout_seconds)

    # ==================== Status/Debug Methods ====================

    def get_state_dict(self) -> dict:
        """
        Get current state as a dict (useful for logging/debugging).

        Returns:
            {
                'n_orders_being_processed': int,
                'sync_waiting': bool,
                'last_sync_start_ms': int,
                'last_sync_complete_ms': int,
                'time_since_last_sync_ms': int
            }
        """
        with self._lock:
            now_ms = TimeUtil.now_in_millis()
            return {
                'n_orders_being_processed': self._n_orders_being_processed,
                'sync_waiting': self._sync_waiting,
                'last_sync_start_ms': self._last_sync_start_ms,
                'last_sync_complete_ms': self._last_sync_complete_ms,
                'time_since_last_sync_ms': now_ms - self._last_sync_complete_ms if self._last_sync_complete_ms else None,
            }

    def __repr__(self) -> str:
        """String representation for debugging."""
        state = self.get_state_dict()
        return (f"OrderSyncState(orders={state['n_orders_being_processed']}, "
                f"sync_waiting={state['sync_waiting']}, "
                f"time_since_sync={state['time_since_last_sync_ms']}ms)")
