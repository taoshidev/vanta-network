
import threading
from sortedcontainers import SortedList
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig

def sorted_list_key(x):
    return x[0]

class RecentEventTracker:
    """
    Thread-safe tracker for recent price events.

    This class is accessed concurrently by:
    - Background websocket threads (Polygon, Tiingo) writing events
    - RPC client threads reading events
    - Cleanup operations removing old events

    All public methods are thread-safe and use an RLock to protect shared state.
    """

    def __init__(self):
        # RLock allows reentrant locking (methods can call each other)
        self._lock = threading.RLock()

        self.events = SortedList(key=sorted_list_key)  # Assuming each event is a tuple (timestamp, event_data)
        self.timestamp_to_event = {}

    def add_event(self, event, is_forex_quote=False, tp_debug_str: str = None):
        """Thread-safe event addition."""
        event_time_ms = event.start_ms

        with self._lock:
            # Check and return must be atomic to prevent TOCTOU
            if self._timestamp_exists_unsafe(event_time_ms):
                #print(f'Duplicate timestamp {TimeUtil.millis_to_formatted_date_str(event_time_ms)} for tp {tp_debug_str} ignored')
                return

            self.events.add((event_time_ms, event))
            self.timestamp_to_event[event_time_ms] = (event, ([event.bid], [event.ask]) if is_forex_quote else None)
            #print(f"Added event at {TimeUtil.millis_to_formatted_date_str(event_time_ms)}")

            # Cleanup called within lock - prevents concurrent cleanup races
            self._cleanup_old_events_unsafe()
            #print(event, tp_debug_str)

    def get_event_by_timestamp(self, timestamp_ms):
        """Thread-safe event retrieval by timestamp."""
        with self._lock:
            return self.timestamp_to_event.get(timestamp_ms, (None, None))

    def timestamp_exists(self, timestamp_ms):
        """Thread-safe timestamp existence check."""
        with self._lock:
            return self._timestamp_exists_unsafe(timestamp_ms)

    def _timestamp_exists_unsafe(self, timestamp_ms):
        """
        Internal unsafe version for use within locked sections.
        DO NOT call this without holding self._lock!
        """
        return timestamp_ms in self.timestamp_to_event

    @staticmethod
    def forex_median_price(arr):
        """Static method - no locking needed."""
        median_price = arr[len(arr) // 2] if len(arr) % 2 == 1 else (arr[len(arr) // 2] + arr[len(arr) // 2 - 1]) / 2.0
        return median_price

    def update_prices_for_median(self, t_ms, bid_price, ask_price):
        """
        Thread-safe median price update for forex quotes.

        Multiple websocket sources can update the same forex timestamp,
        so this operation must be atomic.
        """
        with self._lock:
            existing_event, prices = self.timestamp_to_event.get(t_ms, (None, None))

            if not prices:
                return

            # Append and sort operations must be atomic
            prices[0].append(bid_price)
            prices[0].sort()
            prices[1].append(ask_price)
            prices[1].sort()

            median_bid = self.forex_median_price(prices[0])
            median_ask = self.forex_median_price(prices[1])

            # Update event fields
            midpoint = (median_bid + median_ask) / 2.0
            existing_event.open = existing_event.close = existing_event.high = existing_event.low = midpoint
            existing_event.bid = median_bid
            existing_event.ask = median_ask

    def _cleanup_old_events(self):
        """
        Thread-safe cleanup wrapper.
        Acquires lock and calls unsafe version.
        """
        with self._lock:
            self._cleanup_old_events_unsafe()

    def _cleanup_old_events_unsafe(self):
        """
        Internal unsafe cleanup - MUST be called with lock held.

        This is separated to allow add_event() to call cleanup
        without double-locking (since add_event already holds lock).
        """
        current_time_ms = TimeUtil.now_in_millis()
        oldest_valid_time_ms = current_time_ms - ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS

        # Loop must be atomic with dict deletion to prevent KeyError
        while self.events and self.events[0][0] < oldest_valid_time_ms:
            removed_event = self.events.pop(0)
            # This del can raise KeyError if another thread already deleted
            # But now it's impossible because we hold the lock
            del self.timestamp_to_event[removed_event[0]]

    def get_events_in_range(self, start_time_ms, end_time_ms):
        """
        Thread-safe retrieval of events in time range.

        Returns a NEW list (copy) to prevent iterator invalidation.
        Callers can safely iterate the returned list without holding the lock.

        Args:
            start_time_ms (int): The start timestamp in milliseconds.
            end_time_ms (int): The end timestamp in milliseconds.

        Returns:
            list: A NEW list of events (event_data) within the specified time range.
        """
        with self._lock:
            if len(self.events) == 0:
                return []

            # Bisect operations and slicing must be atomic
            start_idx = self.events.bisect_left((start_time_ms,))
            end_idx = self.events.bisect_right((end_time_ms + 1,))

            # Return a NEW list (copy) - prevents iterator invalidation
            # Even if events are added/removed after this returns,
            # the caller's list remains valid
            return [event[1] for event in self.events[start_idx:end_idx]]

    def get_closest_event(self, timestamp_ms):
        """
        Thread-safe retrieval of closest event to timestamp.

        All index operations are protected by lock to prevent
        IndexError from concurrent cleanup.
        """
        with self._lock:
            if len(self.events) == 0:
                return None

            # All index accesses must be atomic with length check
            idx = self.events.bisect_left((timestamp_ms,))

            if idx == 0:
                return self.events[0][1]
            elif idx == len(self.events):
                return self.events[-1][1]
            else:
                before = self.events[idx - 1]
                after = self.events[idx]
                return after[1] if (after[0] - timestamp_ms) < (timestamp_ms - before[0]) else before[1]

    def count_events(self):
        """Thread-safe event count."""
        with self._lock:
            return len(self.events)

    def clear_all_events(self, running_unit_tests: bool = False):
        """
        Thread-safe method to clear all events.

        WARNING: This should ONLY be used in unit tests for cleanup between tests.
        In production, this would discard valuable websocket price data.

        Args:
            running_unit_tests: Must be True to proceed. Safety check to prevent accidental production use.

        Raises:
            RuntimeError: If called in production mode (running_unit_tests=False)
        """
        if not running_unit_tests:
            raise RuntimeError("clear_all_events() can only be called in unit test mode")

        with self._lock:
            self.events.clear()
            self.timestamp_to_event.clear()

    def clear_and_add_event(self, event, is_forex_quote=False, tp_debug_str: str = None, running_unit_tests: bool = False):
        """
        Atomically clear all events and add a new event.

        This method is critical for test isolation - it ensures that when injecting test prices,
        stale test prices from previous test runs are cleared before adding the new price.
        Without this atomicity, there's a race window where:
        1. clear_all_events() releases lock
        2. Another thread could add an event
        3. add_event() acquires lock and adds
        Result: Unintended events remain in tracker

        This atomic operation prevents _get_best_price_source() from selecting stale
        test prices when computing median prices across multiple data sources.

        Args:
            event: The event to add after clearing
            is_forex_quote: Whether this is a forex quote (for median price tracking)
            tp_debug_str: Debug string for logging
            running_unit_tests: Must be True to proceed. Safety check to prevent accidental production use.

        Raises:
            RuntimeError: If called in production mode (running_unit_tests=False)
        """
        if not running_unit_tests:
            raise RuntimeError("clear_and_add_event() can only be called in unit test mode")

        with self._lock:
            # Clear all events first
            self.events.clear()
            self.timestamp_to_event.clear()

            # Now add the new event (using unsafe internal logic since we already hold lock)
            event_time_ms = event.start_ms

            # No need to check for duplicates since we just cleared everything
            self.events.add((event_time_ms, event))
            self.timestamp_to_event[event_time_ms] = (event, ([event.bid], [event.ask]) if is_forex_quote else None)

            # No cleanup needed since we just cleared everything
