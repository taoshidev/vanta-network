import unittest
import time
import threading
import sys
from threading import Thread
from unittest.mock import patch

from sortedcontainers import SortedList

from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker


class TestRecentEventTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = RecentEventTracker()

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_add_event(self, mock_time):
        # First event added
        mock_time.return_value = 10000000  # Mock time set for the first event
        event = PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0, bid=95, ask=96)
        self.tracker.add_event(event, is_forex_quote=True)
        existing_event = self.tracker.get_event_by_timestamp(mock_time.return_value)
        self.assertEqual(existing_event[0], event)
        self.assertEqual(existing_event[1], ([event.bid], [event.ask]))

        # Assert the first event is added correctly
        self.assertEqual(len(self.tracker.events), 1)
        self.assertEqual(self.tracker.events[0][1], event)

        # Add second event, update mock time first
        mock_time.return_value = 10000000 + 1000 * 60 * 4  # Update mock time for second event. 4 minutes later
        event2 = PriceSource(start_ms=mock_time.return_value, open=102.0, close=106.0)
        self.tracker.add_event(event2)

        # Assert the first two events are still there as it's within 5 minutes
        self.assertEqual(len(self.tracker.events), 2)

        mock_time.return_value = 10000000 + 1000 * 60 * 6  # Update mock time for third event. 2 minutes later
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=103.0, close=107.0))

        # Now only two events should remain after cleanup
        self.assertEqual(len(self.tracker.events), 2)

        # Get most event
        most_recent_event = self.tracker.get_closest_event(mock_time.return_value)
        self.assertEqual(most_recent_event.open, 103.0)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_cleanup_old_events(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + 3000, open=102.0, close=106.0))

        # Forward time to trigger cleanup
        mock_time.return_value += 1000 * 60 * 5  # 5 minutes later
        self.tracker._cleanup_old_events()

        # Both events still there (edge case test)
        self.assertEqual(len(self.tracker.events), 2)

        mock_time.return_value += 1
        self.tracker._cleanup_old_events()
        #First event should be removed
        self.assertEqual(len(self.tracker.events), 1)

    def test_get_events_in_range(self):
        self.tracker.events = SortedList([(1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),
                                          (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0))],
                                         key=lambda x: x[0])
        events = self.tracker.get_events_in_range(1000000, 1002000)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].open, 100.0)

    def test_get_closest_event(self):
        self.tracker.events = SortedList([(1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),
                                          (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0))],
                                         key=lambda x: x[0])
        closest_event = self.tracker.get_closest_event(1001500)
        self.assertEqual(closest_event.start_ms, 1000000)

    def test_get_events_in_range2(self):
        self.tracker.events = SortedList([
            (995000, PriceSource(start_ms=995000, open=99.0, close=104.0)),  # Before range
            (1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),  # Start of range
            (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0)),  # After range start
            (1005000, PriceSource(start_ms=1005000, open=103.0, close=107.0)),  # After range
        ], key=lambda x: x[0])

        # Target range partially includes some events
        events = self.tracker.get_events_in_range(1000000, 1004000)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].open, 100.0)
        self.assertEqual(events[1].open, 102.0)

        # Test for no events in range
        events_empty = self.tracker.get_events_in_range(990000, 994998)
        self.assertEqual(len(events_empty), 0, events_empty)

    def test_get_closest_event2(self):
        self.tracker.events = SortedList([
            (995000, PriceSource(start_ms=995000, open=99.0, close=104.0)),
            (1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),
            (1001500, PriceSource(start_ms=1001500, open=101.0, close=106.0)),  # Exact match
            (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0)),
            (1004500, PriceSource(start_ms=1004500, open=103.0, close=107.0)),
        ], key=lambda x: x[0])

        # Target timestamp exactly matches one event
        closest_event = self.tracker.get_closest_event(1001500)
        self.assertEqual(closest_event.start_ms, 1001500)

        closest_event_equidistant = self.tracker.get_closest_event(1002250 + 1)
        self.assertEqual(closest_event_equidistant.start_ms, 1003000, closest_event)
        closest_event_equidistant = self.tracker.get_closest_event(1002250 - 1)
        self.assertEqual(closest_event_equidistant.start_ms, 1001500, closest_event)

        # No events, should return None
        self.tracker.events = SortedList()
        no_event = self.tracker.get_closest_event(1001500)
        self.assertIsNone(no_event)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_concurrent_add_events(self, mock_time):
        def add_events(tracker, events):
            for event in events:
                tracker.add_event(event)

        mock_time.return_value = 10000000
        events1 = [PriceSource(start_ms=mock_time.return_value + i * 1000, open=100.0 + i, close=105.0 + i) for i in range(50)]
        events2 = [PriceSource(start_ms=mock_time.return_value + i * 1000, open=200.0 + i, close=205.0 + i) for i in range(50, 100)]

        thread1 = Thread(target=add_events, args=(self.tracker, events1))
        thread2 = Thread(target=add_events, args=(self.tracker, events2))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Check for the correct number of events
        self.assertEqual(len(self.tracker.events), 100)
        # Check that events are sorted correctly
        all_events = list(self.tracker.events)
        self.assertTrue(all(e1[0] <= e2[0] for e1, e2 in zip(all_events[:-1], all_events[1:])))

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_event_overlap(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=101.0, close=106.0))  # Exact same start time
        self.assertEqual(len(self.tracker.events), 1)
        # self.assertNotEqual(self.tracker.events[0][1], self.tracker.events[1][1])
        # self.assertEqual(self.tracker.events[0][0], self.tracker.events[1][0])

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_precise_timing(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + 1, open=101.0, close=106.0))  # 1 millisecond later
        self.assertEqual(len(self.tracker.events), 2)
        self.assertNotEqual(self.tracker.events[0][1], self.tracker.events[1][1])
        self.assertTrue(self.tracker.events[1][0] - self.tracker.events[0][0] == 1)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_boundary_event_removal(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + 299999, open=101.0, close=106.0))  # Just under 5 minutes

        mock_time.return_value = mock_time.return_value + 300000  # Exactly 5 minutes later
        self.tracker._cleanup_old_events()

        self.assertEqual(len(self.tracker.events), 2)

        mock_time.return_value = mock_time.return_value + 1
        self.tracker._cleanup_old_events()

        self.assertEqual(len(self.tracker.events), 1)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_efficiency_of_cleanup(self, mock_time):
        mock_time.return_value = 10000000
        for i in range(100):
            self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + i * 1000, open=100.0 + i, close=105.0 + i))

        mock_time.return_value += + 1000 * 60 * 6  # Forward time to ensure some events are old
        self.tracker._cleanup_old_events()

        # Verify that the method has cleaned up exactly the right amount of events
        self.assertTrue(len(self.tracker.events) < 100)
        self.assertTrue(all(event[0] >= 10000000 + 1000 * 60 for event in self.tracker.events))


class TestRecentEventTrackerRaceConditions(unittest.TestCase):
    """
    Tests demonstrating race conditions in RecentEventTracker when accessed concurrently.
    These tests model the ACTUAL access patterns in the codebase:
    - Websocket threads continuously adding events
    - RPC threads reading events while websockets write
    - Multiple threads reading/writing simultaneously

    EXPECTED BEHAVIOR: These tests will FAIL intermittently (or consistently under load)
    due to lack of thread-safety mechanisms in RecentEventTracker.

    Common failure modes:
    - RuntimeError: list changed size during iteration
    - IndexError: list index out of range
    - AssertionError: data corruption/inconsistency
    - Stale reads: missing recent events

    NOTE: Python's GIL makes some race conditions subtle. These tests use:
    - Threading barriers to force exact concurrent access
    - No sleeps to maximize contention
    - Large iteration counts to increase probability
    - Direct testing of vulnerable code paths
    """

    def setUp(self):
        self.tracker = RecentEventTracker()
        self.errors = []
        self.base_time = 10000000
        self.corruption_detected = []

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_websocket_write_rpc_read(self, mock_time):
        """
        RC-01: Simulates Polygon/Tiingo websocket threads writing while RPC clients read.

        ACTUAL PATTERN:
        - polygon_data_service.py:382 - handle_msg() calls add_event()
        - live_price_fetcher.py:126 - RPC thread calls get_events_in_range()

        EXPECTED FAILURE: RuntimeError, IndexError, or partial/missing data
        """
        mock_time.return_value = self.base_time
        read_results = []
        write_count = [0]

        def websocket_writer():
            """Simulates continuous websocket data (like Polygon/Tiingo)."""
            try:
                for i in range(500):
                    ps = PriceSource(
                        start_ms=self.base_time + i * 100,
                        open=100.0 + i * 0.01,
                        close=100.0 + i * 0.01,
                        high=100.0 + i * 0.01,
                        low=100.0 + i * 0.01,
                        vwap=100.0 + i * 0.01,
                        websocket=True,
                        lag_ms=10
                    )
                    self.tracker.add_event(ps)
                    write_count[0] += 1
                    # Small delay to allow interleaving with readers
                    time.sleep(0.0001)
            except Exception as e:
                self.errors.append(('websocket_writer', e, type(e).__name__))

        def rpc_reader():
            """Simulates RPC client reading events (like get_latest_price)."""
            try:
                for _ in range(100):
                    # Read events in range - this calls get_events_in_range()
                    events = self.tracker.get_events_in_range(
                        self.base_time,
                        self.base_time + 1000000
                    )
                    read_results.append(len(events))
                    # Iterate the results (common pattern in codebase)
                    for event in events:
                        _ = event.close  # Access event data
                    time.sleep(0.0005)
            except Exception as e:
                self.errors.append(('rpc_reader', e, type(e).__name__))

        # Start 2 websocket writers (Polygon + Tiingo) and 3 RPC readers (multiple clients)
        threads = []

        # 2 websocket threads (like Polygon and Tiingo)
        for i in range(2):
            t = threading.Thread(target=websocket_writer, name=f'WebSocket-{i}')
            threads.append(t)
            t.start()

        # 3 RPC reader threads (like multiple position managers, perf ledgers, etc.)
        for i in range(3):
            t = threading.Thread(target=rpc_reader, name=f'RPC-Reader-{i}')
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=30)

        # Check for errors (race conditions manifest as exceptions)
        if self.errors:
            self.fail(f"Race condition detected! Errors: {self.errors}")

        # Verify data consistency
        final_events = self.tracker.get_events_in_range(self.base_time, self.base_time + 1000000)
        self.assertGreater(len(final_events), 0, "Should have events after concurrent writes")

        # Verify all events are sorted
        for i in range(len(final_events) - 1):
            self.assertLessEqual(
                final_events[i].start_ms,
                final_events[i + 1].start_ms,
                "Events should be sorted by timestamp"
            )

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_cleanup_during_read(self, mock_time):
        """
        RC-02: Simulates cleanup removing events while RPC threads are reading.

        ACTUAL PATTERN:
        - recent_event_tracker.py:50 - _cleanup_old_events() pops from list
        - recent_event_tracker.py:84 - get_closest_event() accesses by index

        EXPECTED FAILURE: IndexError when indices become invalid after cleanup
        """
        # Pre-populate with old events
        mock_time.return_value = self.base_time
        for i in range(200):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        def cleanup_thread():
            """Continuously trigger cleanup (simulates time advancing)."""
            try:
                for i in range(50):
                    # Advance time to trigger cleanup
                    mock_time.return_value = self.base_time + (i + 1) * 10000
                    self.tracker._cleanup_old_events()
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(('cleanup_thread', e, type(e).__name__))

        def reader_thread():
            """Reads events while cleanup is removing them."""
            try:
                for _ in range(100):
                    # These operations use indices internally
                    closest = self.tracker.get_closest_event(self.base_time + 50000)
                    events = self.tracker.get_events_in_range(self.base_time, self.base_time + 100000)
                    if closest:
                        _ = closest.close
                    for e in events:
                        _ = e.close
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(('reader_thread', e, type(e).__name__))

        # Start 1 cleanup thread and 3 reader threads
        threads = []

        cleanup = threading.Thread(target=cleanup_thread, name='Cleanup')
        threads.append(cleanup)
        cleanup.start()

        for i in range(3):
            reader = threading.Thread(target=reader_thread, name=f'Reader-{i}')
            threads.append(reader)
            reader.start()

        for t in threads:
            t.join(timeout=10)

        if self.errors:
            self.fail(f"Race condition during cleanup! Errors: {self.errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_forex_median_update_toctou(self, mock_time):
        """
        RC-03 & RC-08: TOCTOU race in timestamp_exists() + concurrent median updates.

        ACTUAL PATTERN:
        - polygon_data_service.py:278 - Check timestamp_exists(), then update_prices_for_median()
        - tiingo_data_service.py:195 - Same pattern
        - Both could receive same forex timestamp simultaneously

        EXPECTED FAILURE: Corrupted median calculation, lost updates, list modification during sort
        """
        mock_time.return_value = self.base_time

        # Add initial forex event
        initial_event = PriceSource(
            start_ms=self.base_time,
            open=1.1000,
            close=1.1000,
            high=1.1000,
            low=1.1000,
            vwap=1.1000,
            websocket=True,
            lag_ms=0,
            bid=1.1000,
            ask=1.1010
        )
        self.tracker.add_event(initial_event, is_forex_quote=True)

        median_updates = [0]

        def polygon_websocket():
            """Simulates Polygon receiving forex quotes."""
            try:
                for i in range(100):
                    # Check if timestamp exists (TOCTOU vulnerable)
                    if self.tracker.timestamp_exists(self.base_time):
                        # Update median - concurrent list modification!
                        bid = 1.1000 + (i * 0.0001)
                        ask = 1.1010 + (i * 0.0001)
                        self.tracker.update_prices_for_median(self.base_time, bid, ask)
                        median_updates[0] += 1
                    time.sleep(0.0001)
            except Exception as e:
                self.errors.append(('polygon_websocket', e, type(e).__name__))

        def tiingo_websocket():
            """Simulates Tiingo receiving forex quotes for same timestamp."""
            try:
                for i in range(100):
                    # Same TOCTOU pattern
                    if self.tracker.timestamp_exists(self.base_time):
                        bid = 1.0995 + (i * 0.0001)
                        ask = 1.1005 + (i * 0.0001)
                        self.tracker.update_prices_for_median(self.base_time, bid, ask)
                        median_updates[0] += 1
                    time.sleep(0.0001)
            except Exception as e:
                self.errors.append(('tiingo_websocket', e, type(e).__name__))

        # Start both websocket threads
        polygon = threading.Thread(target=polygon_websocket, name='Polygon-Forex')
        tiingo = threading.Thread(target=tiingo_websocket, name='Tiingo-Forex')

        polygon.start()
        tiingo.start()

        polygon.join(timeout=5)
        tiingo.join(timeout=5)

        if self.errors:
            self.fail(f"Race in forex median update! Errors: {self.errors}")

        # Verify event still exists and has valid median
        event, prices = self.tracker.get_event_by_timestamp(self.base_time)
        self.assertIsNotNone(event, "Event should still exist")
        if prices:
            # Check that bid/ask lists are properly sorted
            self.assertEqual(prices[0], sorted(prices[0]), "Bid prices should be sorted")
            self.assertEqual(prices[1], sorted(prices[1]), "Ask prices should be sorted")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_multiple_rpc_clients_concurrent_reads(self, mock_time):
        """
        RC-07: Multiple RPC clients reading simultaneously while websockets write.

        ACTUAL PATTERN:
        - Position manager calls get_latest_price() via RPC
        - Perf ledger calls get_candles() via RPC
        - Both eventually call get_events_in_range() on same tracker
        - Websockets continuously adding events

        EXPECTED FAILURE: Inconsistent reads, missing events, iterator corruption
        """
        mock_time.return_value = self.base_time

        # Pre-populate with some events
        for i in range(100):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0 + i * 0.1,
                close=100.0 + i * 0.1,
                high=100.0 + i * 0.1,
                low=100.0 + i * 0.1,
                vwap=100.0 + i * 0.1,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        read_counts = []

        def continuous_writer():
            """Websocket continuously adding events."""
            try:
                for i in range(100, 300):
                    ps = PriceSource(
                        start_ms=self.base_time + i * 1000,
                        open=100.0 + i * 0.1,
                        close=100.0 + i * 0.1,
                        high=100.0 + i * 0.1,
                        low=100.0 + i * 0.1,
                        vwap=100.0 + i * 0.1,
                        websocket=True,
                        lag_ms=0
                    )
                    self.tracker.add_event(ps)
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(('continuous_writer', e, type(e).__name__))

        def rpc_client_reader(client_id):
            """Each RPC client reading events."""
            try:
                for _ in range(50):
                    # Get events in range
                    events = self.tracker.get_events_in_range(
                        self.base_time,
                        self.base_time + 500000
                    )
                    read_counts.append(len(events))

                    # Iterate and access data (common pattern)
                    for event in events:
                        _ = event.close + event.open

                    # Also get closest event
                    closest = self.tracker.get_closest_event(self.base_time + 150000)
                    if closest:
                        _ = closest.close

                    time.sleep(0.002)
            except Exception as e:
                self.errors.append((f'rpc_client_{client_id}', e, type(e).__name__))

        # Start 1 writer and 5 concurrent readers (simulating multiple RPC clients)
        threads = []

        writer = threading.Thread(target=continuous_writer, name='WebSocket')
        threads.append(writer)
        writer.start()

        for i in range(5):
            reader = threading.Thread(target=rpc_client_reader, args=(i,), name=f'RPC-Client-{i}')
            threads.append(reader)
            reader.start()

        for t in threads:
            t.join(timeout=15)

        if self.errors:
            self.fail(f"Race with multiple RPC clients! Errors: {self.errors}")

        # Verify reads were consistent (should see increasing event counts as writer adds)
        self.assertGreater(len(read_counts), 0, "Should have read events")

        # Final consistency check
        final_events = self.tracker.get_events_in_range(self.base_time, self.base_time + 500000)
        self.assertGreater(len(final_events), 100, "Should have accumulated events")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_sortedlist_internal_corruption(self, mock_time):
        """
        RC-01 variant: Stress test SortedList internal tree structure under concurrent writes.

        SortedList uses a B-tree internally. Concurrent add() calls without locks can corrupt
        the tree structure, leading to:
        - Lost elements
        - Duplicate elements
        - Incorrect bisect results
        - Tree invariant violations

        EXPECTED FAILURE: Data loss, incorrect counts, or corrupted tree
        """
        mock_time.return_value = self.base_time

        n_events_per_thread = 200
        n_writer_threads = 4

        def aggressive_writer(thread_id, base_offset):
            """Aggressively write events to stress SortedList."""
            try:
                for i in range(n_events_per_thread):
                    ps = PriceSource(
                        start_ms=self.base_time + base_offset + i,
                        open=100.0 + thread_id,
                        close=100.0 + thread_id,
                        high=100.0 + thread_id,
                        low=100.0 + thread_id,
                        vwap=100.0 + thread_id,
                        websocket=True,
                        lag_ms=0
                    )
                    self.tracker.add_event(ps)
                    # No sleep - maximize contention
            except Exception as e:
                self.errors.append((f'writer_{thread_id}', e, type(e).__name__))

        # Start multiple aggressive writers
        threads = []
        for i in range(n_writer_threads):
            t = threading.Thread(
                target=aggressive_writer,
                args=(i, i * n_events_per_thread),
                name=f'Writer-{i}'
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        if self.errors:
            self.fail(f"SortedList corruption! Errors: {self.errors}")

        # Verify data integrity
        expected_count = n_writer_threads * n_events_per_thread
        actual_count = len(self.tracker.events)

        # Without locks, we expect data loss
        self.assertEqual(
            actual_count, expected_count,
            f"Data loss detected! Expected {expected_count} events, got {actual_count}. "
            f"This indicates SortedList corruption due to concurrent access without locks."
        )

        # Verify sorted order
        all_events = list(self.tracker.events)
        for i in range(len(all_events) - 1):
            self.assertLessEqual(
                all_events[i][0], all_events[i + 1][0],
                "SortedList order violated!"
            )

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_barrier_synchronized_access(self, mock_time):
        """
        AGGRESSIVE TEST: Use barrier to force EXACT concurrent access.

        This test uses threading.Barrier to ensure all threads start
        accessing the tracker at the EXACT same moment, maximizing
        the probability of race conditions.

        EXPECTED FAILURE: IndexError, data corruption, or lost updates
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(50):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        n_threads = 10
        barrier = threading.Barrier(n_threads)
        operations_completed = [0]

        def concurrent_accessor(thread_id):
            """All threads wait at barrier, then access simultaneously."""
            try:
                # Wait for all threads to be ready
                barrier.wait()

                # NOW all threads execute this simultaneously
                for i in range(100):
                    if thread_id % 2 == 0:
                        # Writer thread
                        ps = PriceSource(
                            start_ms=self.base_time + 1000000 + thread_id * 1000 + i,
                            open=100.0,
                            close=100.0,
                            high=100.0,
                            low=100.0,
                            vwap=100.0,
                            websocket=True,
                            lag_ms=0
                        )
                        self.tracker.add_event(ps)
                    else:
                        # Reader thread - this is vulnerable!
                        events = self.tracker.get_events_in_range(
                            self.base_time,
                            self.base_time + 2000000
                        )
                        # Iterate - can crash if list modified during iteration
                        for e in events:
                            _ = e.close

                operations_completed[0] += 1
            except Exception as e:
                self.errors.append((f'thread_{thread_id}', e, type(e).__name__))

        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=concurrent_accessor, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        if self.errors:
            self.fail(f"Barrier-synchronized race detected! Errors: {self.errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_direct_list_iteration_corruption(self, mock_time):
        """
        AGGRESSIVE TEST: Directly test list iteration while modifying.

        This tests the EXACT vulnerable pattern:
        - Thread A: Iterate self.events
        - Thread B: Modify self.events via add_event()

        EXPECTED FAILURE: RuntimeError or IndexError
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(100):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        iteration_errors = []

        def aggressive_iterator():
            """Iterate the internal events list repeatedly."""
            try:
                for _ in range(1000):
                    # Direct iteration of internal list
                    for timestamp, event in self.tracker.events:
                        _ = event.close
                        # NO SLEEP - maximize contention
            except Exception as e:
                iteration_errors.append(('iterator', e, type(e).__name__))

        def aggressive_modifier():
            """Modify the list while iteration happens."""
            try:
                for i in range(1000):
                    ps = PriceSource(
                        start_ms=self.base_time + 100000 + i,
                        open=100.0,
                        close=100.0,
                        high=100.0,
                        low=100.0,
                        vwap=100.0,
                        websocket=True,
                        lag_ms=0
                    )
                    # This modifies self.events.add() and also calls cleanup
                    self.tracker.add_event(ps)
                    # NO SLEEP - maximize contention
            except Exception as e:
                iteration_errors.append(('modifier', e, type(e).__name__))

        # Start 3 iterators and 2 modifiers
        threads = []
        for i in range(3):
            t = threading.Thread(target=aggressive_iterator, name=f'Iterator-{i}')
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=aggressive_modifier, name=f'Modifier-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        if iteration_errors:
            self.fail(f"List iteration corruption! Errors: {iteration_errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_index_access_after_length_check(self, mock_time):
        """
        AGGRESSIVE TEST: TOCTOU on length check then index access.

        Pattern:
        - Thread A: len(events) returns N
        - Thread B: Removes event (cleanup)
        - Thread A: Access events[N-1] - CRASH!

        This is the EXACT pattern in get_closest_event()
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(200):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        def index_accessor():
            """Access by index after checking length (vulnerable pattern)."""
            try:
                for _ in range(500):
                    # TOCTOU vulnerable pattern
                    if len(self.tracker.events) > 0:
                        # Between this check and the access, cleanup could remove events!
                        last_event = self.tracker.events[-1]
                        _ = last_event[1].close

                        if len(self.tracker.events) > 50:
                            middle_event = self.tracker.events[len(self.tracker.events) // 2]
                            _ = middle_event[1].close
            except Exception as e:
                self.errors.append(('index_accessor', e, type(e).__name__))

        def aggressive_cleanup():
            """Trigger cleanup to remove events."""
            try:
                for i in range(100):
                    # Advance time significantly to trigger cleanup
                    mock_time.return_value = self.base_time + (i + 1) * 10000
                    self.tracker._cleanup_old_events()
                    # NO SLEEP
            except Exception as e:
                self.errors.append(('cleanup', e, type(e).__name__))

        # Start 4 index accessors and 2 cleanup threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=index_accessor, name=f'Accessor-{i}')
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=aggressive_cleanup, name=f'Cleanup-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        if self.errors:
            self.fail(f"Index TOCTOU race! Errors: {self.errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_bisect_during_modification(self, mock_time):
        """
        AGGRESSIVE TEST: Test bisect operations during concurrent modifications.

        get_events_in_range() uses bisect_left() and bisect_right() which
        assume the list doesn't change during the operation.

        EXPECTED FAILURE: Incorrect bisect results, IndexError, or corruption
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(500):
            ps = PriceSource(
                start_ms=self.base_time + i * 100,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        bisect_results = []

        def bisect_user():
            """Use bisect operations like get_events_in_range() does."""
            try:
                for i in range(500):
                    # This uses bisect internally
                    events = self.tracker.get_events_in_range(
                        self.base_time + 10000,
                        self.base_time + 40000
                    )
                    bisect_results.append(len(events))

                    # Also test get_closest_event which uses bisect
                    closest = self.tracker.get_closest_event(self.base_time + 25000)
                    if closest:
                        _ = closest.close
            except Exception as e:
                self.errors.append(('bisect_user', e, type(e).__name__))

        def list_modifier():
            """Modify list while bisect operations happen."""
            try:
                for i in range(500):
                    ps = PriceSource(
                        start_ms=self.base_time + 50000 + i * 10,
                        open=100.0,
                        close=100.0,
                        high=100.0,
                        low=100.0,
                        vwap=100.0,
                        websocket=True,
                        lag_ms=0
                    )
                    self.tracker.add_event(ps)
            except Exception as e:
                self.errors.append(('modifier', e, type(e).__name__))

        # 5 bisect users, 3 modifiers
        threads = []
        for i in range(5):
            t = threading.Thread(target=bisect_user, name=f'Bisect-{i}')
            threads.append(t)
            t.start()

        for i in range(3):
            t = threading.Thread(target=list_modifier, name=f'Modifier-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        if self.errors:
            self.fail(f"Bisect corruption! Errors: {self.errors}")

        # Check for inconsistent bisect results
        if len(bisect_results) > 0:
            # Results should be relatively consistent since we're querying same range
            # But with races, we might see wild variations
            min_result = min(bisect_results)
            max_result = max(bisect_results)
            variation = max_result - min_result

            # Some variation is expected as events are added, but extreme variation
            # indicates bisect corruption
            if variation > 100:
                self.corruption_detected.append(
                    f"Extreme bisect result variation: {min_result} to {max_result}"
                )

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_timestamp_dict_concurrent_access(self, mock_time):
        """
        AGGRESSIVE TEST: Test timestamp_to_event dict concurrent access.

        Pattern:
        - Thread A: add_event() writes to dict
        - Thread B: get_event_by_timestamp() reads from dict
        - Thread C: update_prices_for_median() modifies dict value

        EXPECTED FAILURE: KeyError, corrupted values, or lost updates
        """
        mock_time.return_value = self.base_time

        # Add initial forex event
        initial = PriceSource(
            start_ms=self.base_time,
            open=1.1000,
            close=1.1000,
            high=1.1000,
            low=1.1000,
            vwap=1.1000,
            websocket=True,
            lag_ms=0,
            bid=1.1000,
            ask=1.1010
        )
        self.tracker.add_event(initial, is_forex_quote=True)

        dict_access_results = []

        def dict_reader():
            """Read from timestamp_to_event dict."""
            try:
                for _ in range(1000):
                    event, prices = self.tracker.get_event_by_timestamp(self.base_time)
                    if event:
                        _ = event.close
                    if prices:
                        _ = len(prices[0])
                    dict_access_results.append(1)
            except Exception as e:
                self.errors.append(('reader', e, type(e).__name__))

        def dict_updater():
            """Update median prices (modifies dict values)."""
            try:
                for i in range(1000):
                    bid = 1.1000 + (i * 0.00001)
                    ask = 1.1010 + (i * 0.00001)
                    self.tracker.update_prices_for_median(self.base_time, bid, ask)
            except Exception as e:
                self.errors.append(('updater', e, type(e).__name__))

        def dict_checker():
            """Check timestamp_exists."""
            try:
                for _ in range(1000):
                    exists = self.tracker.timestamp_exists(self.base_time)
                    self.assertTrue(exists, "Timestamp should exist")
            except Exception as e:
                self.errors.append(('checker', e, type(e).__name__))

        # 3 readers, 3 updaters, 2 checkers
        threads = []
        for i in range(3):
            t = threading.Thread(target=dict_reader, name=f'Reader-{i}')
            threads.append(t)
            t.start()

        for i in range(3):
            t = threading.Thread(target=dict_updater, name=f'Updater-{i}')
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=dict_checker, name=f'Checker-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        if self.errors:
            self.fail(f"Dict concurrent access errors! Errors: {self.errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_list_sort_during_append(self, mock_time):
        """
        AGGRESSIVE TEST: Test update_prices_for_median list.sort() during append.

        update_prices_for_median() does:
        1. prices[0].append(bid)
        2. prices[0].sort()

        If two threads do this simultaneously:
        - Thread A appends, then Thread B appends, then A sorts, then B sorts
        - Result: corrupted sort order or lost values

        EXPECTED FAILURE: Incorrect sort order or list corruption
        """
        mock_time.return_value = self.base_time

        # Add forex event
        initial = PriceSource(
            start_ms=self.base_time,
            open=1.1000,
            close=1.1000,
            high=1.1000,
            low=1.1000,
            vwap=1.1000,
            websocket=True,
            lag_ms=0,
            bid=1.1000,
            ask=1.1010
        )
        self.tracker.add_event(initial, is_forex_quote=True)

        def concurrent_median_updater(thread_id, base_value):
            """Update median prices concurrently."""
            try:
                for i in range(200):
                    bid = base_value + (i * 0.0001)
                    ask = base_value + 0.001 + (i * 0.0001)
                    self.tracker.update_prices_for_median(self.base_time, bid, ask)
                    # NO SLEEP - maximize contention during append + sort
            except Exception as e:
                self.errors.append((f'updater_{thread_id}', e, type(e).__name__))

        # 6 threads all updating median for same timestamp
        threads = []
        for i in range(6):
            t = threading.Thread(
                target=concurrent_median_updater,
                args=(i, 1.1000 + i * 0.01),
                name=f'MedianUpdater-{i}'
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        if self.errors:
            self.fail(f"List sort/append race! Errors: {self.errors}")

        # Check final state
        event, prices = self.tracker.get_event_by_timestamp(self.base_time)
        if prices:
            # Verify lists are actually sorted
            bid_list = prices[0]
            ask_list = prices[1]

            # Check if sorted
            is_bid_sorted = bid_list == sorted(bid_list)
            is_ask_sorted = ask_list == sorted(ask_list)

            if not is_bid_sorted or not is_ask_sorted:
                self.fail(
                    f"Sort corruption detected! "
                    f"Bid sorted: {is_bid_sorted}, Ask sorted: {is_ask_sorted}. "
                    f"Bid list: {bid_list[:10]}... Ask list: {ask_list[:10]}..."
                )

            # Check for duplicates (could indicate race corruption)
            bid_unique = len(set(bid_list))
            if bid_unique < len(bid_list) * 0.9:  # Allow some duplicates from rounding
                self.corruption_detected.append(
                    f"Excessive duplicates in bid list: {len(bid_list)} total, {bid_unique} unique"
                )


class TestRecentEventTrackerGILReleaseRaces(unittest.TestCase):
    """
    FINAL AGGRESSIVE TESTS: Force GIL release to expose true race conditions.

    The previous tests all passed because Python's GIL serializes bytecode execution.
    These tests FORCE the GIL to be released, creating TRUE concurrent execution.

    Techniques used:
    1. time.sleep(0) - Explicitly yields GIL
    2. sys.setswitchinterval() - Increases context switch frequency
    3. I/O operations - Force GIL release
    4. Injecting yields between critical operations

    If these tests STILL pass, it means:
    - GIL is providing strong protection for SortedList operations
    - BUT: Production websockets do I/O, which releases GIL continuously
    - We MUST still add locks as defensive measure
    """

    def setUp(self):
        self.tracker = RecentEventTracker()
        self.errors = []
        self.base_time = 10000000
        # Increase context switch frequency (default is 0.005 seconds)
        self.original_switchinterval = sys.getswitchinterval()
        sys.setswitchinterval(0.00001)  # Switch every 10 microseconds

    def tearDown(self):
        # Restore original switch interval
        sys.setswitchinterval(self.original_switchinterval)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_with_explicit_gil_release(self, mock_time):
        """
        FORCE GIL RELEASE: Inject time.sleep(0) between critical operations.

        time.sleep(0) explicitly releases the GIL, allowing another thread to run.
        This creates TRUE concurrent execution.

        EXPECTED FAILURE: Iterator corruption or IndexError
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(100):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        def iterator_with_yields():
            """Iterate with explicit GIL releases."""
            try:
                for _ in range(500):
                    for timestamp, event in self.tracker.events:
                        _ = event.close
                        # FORCE GIL RELEASE - another thread will run NOW
                        time.sleep(0)
            except Exception as e:
                self.errors.append(('iterator', e, type(e).__name__))

        def modifier_with_yields():
            """Modify with explicit GIL releases."""
            try:
                for i in range(500):
                    # Split add_event into steps with yields
                    ps = PriceSource(
                        start_ms=self.base_time + 100000 + i,
                        open=100.0,
                        close=100.0,
                        high=100.0,
                        low=100.0,
                        vwap=100.0,
                        websocket=True,
                        lag_ms=0
                    )
                    # Yield before modification
                    time.sleep(0)
                    self.tracker.add_event(ps)
                    # Yield after modification
                    time.sleep(0)
            except Exception as e:
                self.errors.append(('modifier', e, type(e).__name__))

        # 3 iterators and 2 modifiers, all yielding GIL constantly
        threads = []
        for i in range(3):
            t = threading.Thread(target=iterator_with_yields, name=f'Iterator-{i}')
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=modifier_with_yields, name=f'Modifier-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if self.errors:
            self.fail(f"GIL-release race detected! Errors: {self.errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_list_slice_with_concurrent_modification(self, mock_time):
        """
        Test the EXACT pattern in get_events_in_range() with forced GIL releases.

        get_events_in_range() does:
        1. bisect_left() to find start index
        2. bisect_right() to find end index
        3. self.events[start_idx:end_idx] to slice

        With GIL releases, another thread can modify between these steps.
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(200):
            ps = PriceSource(
                start_ms=self.base_time + i * 100,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        slice_results = []

        def slicer_with_yields():
            """Get events in range with yields between operations."""
            try:
                for _ in range(500):
                    # Manually implement get_events_in_range with yields
                    events = self.tracker.events
                    time.sleep(0)  # YIELD - allow modifier to run

                    if len(events) == 0:
                        continue

                    start_idx = events.bisect_left((self.base_time + 5000,))
                    time.sleep(0)  # YIELD - list could change here!

                    end_idx = events.bisect_right((self.base_time + 15000,))
                    time.sleep(0)  # YIELD - list could change here!

                    # Now slice - indices might be invalid!
                    result = [event[1] for event in events[start_idx:end_idx]]
                    slice_results.append(len(result))
            except Exception as e:
                self.errors.append(('slicer', e, type(e).__name__))

        def aggressive_modifier():
            """Add and remove events constantly."""
            try:
                for i in range(500):
                    ps = PriceSource(
                        start_ms=self.base_time + 50000 + i * 10,
                        open=100.0,
                        close=100.0,
                        high=100.0,
                        low=100.0,
                        vwap=100.0,
                        websocket=True,
                        lag_ms=0
                    )
                    time.sleep(0)  # YIELD
                    self.tracker.add_event(ps)
                    time.sleep(0)  # YIELD

                    # Also trigger cleanup
                    if i % 50 == 0:
                        mock_time.return_value = self.base_time + i * 1000
                        self.tracker._cleanup_old_events()
                        time.sleep(0)
            except Exception as e:
                self.errors.append(('modifier', e, type(e).__name__))

        threads = []
        for i in range(4):
            t = threading.Thread(target=slicer_with_yields, name=f'Slicer-{i}')
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=aggressive_modifier, name=f'Modifier-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if self.errors:
            self.fail(f"Slice race with GIL release! Errors: {self.errors}")

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_median_update_with_yields(self, mock_time):
        """
        Test update_prices_for_median() with forced GIL releases.

        This exposes the append + sort race by yielding between operations.
        """
        mock_time.return_value = self.base_time

        # Add forex event
        initial = PriceSource(
            start_ms=self.base_time,
            open=1.1000,
            close=1.1000,
            high=1.1000,
            low=1.1000,
            vwap=1.1000,
            websocket=True,
            lag_ms=0,
            bid=1.1000,
            ask=1.1010
        )
        self.tracker.add_event(initial, is_forex_quote=True)

        def manual_median_update(thread_id, base_value):
            """Manually implement update_prices_for_median with yields."""
            try:
                for i in range(300):
                    bid = base_value + (i * 0.0001)
                    ask = base_value + 0.001 + (i * 0.0001)

                    # Get the event
                    event, prices = self.tracker.get_event_by_timestamp(self.base_time)
                    time.sleep(0)  # YIELD

                    if prices:
                        # Append bid
                        prices[0].append(bid)
                        time.sleep(0)  # YIELD - another thread could append now!

                        # Sort bid
                        prices[0].sort()
                        time.sleep(0)  # YIELD

                        # Append ask
                        prices[1].append(ask)
                        time.sleep(0)  # YIELD - another thread could append now!

                        # Sort ask
                        prices[1].sort()
                        time.sleep(0)  # YIELD

                        # Calculate median and update event
                        median_bid = self.tracker.forex_median_price(prices[0])
                        median_ask = self.tracker.forex_median_price(prices[1])
                        event.close = (median_bid + median_ask) / 2.0
            except Exception as e:
                self.errors.append((f'updater_{thread_id}', e, type(e).__name__))

        # 8 threads all updating, with constant yielding
        threads = []
        for i in range(8):
            t = threading.Thread(
                target=manual_median_update,
                args=(i, 1.1000 + i * 0.01),
                name=f'MedianUpdater-{i}'
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if self.errors:
            self.fail(f"Median update race with yields! Errors: {self.errors}")

        # Check final state - lists should still be sorted
        event, prices = self.tracker.get_event_by_timestamp(self.base_time)
        if prices:
            bid_list = prices[0]
            ask_list = prices[1]

            is_bid_sorted = bid_list == sorted(bid_list)
            is_ask_sorted = ask_list == sorted(ask_list)

            if not is_bid_sorted or not is_ask_sorted:
                self.fail(
                    f"Sort corruption with GIL releases! "
                    f"Bid sorted: {is_bid_sorted}, Ask sorted: {is_ask_sorted}"
                )

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_race_cleanup_with_index_access_and_yields(self, mock_time):
        """
        Test cleanup racing with PUBLIC API access, with explicit yields.

        This tests the ACTUAL production pattern:
        - Thread A: calls get_closest_event() (protected by lock)
        - Thread B: cleanup removes events (protected by lock)
        - With locks: operations are serialized, no race
        - Without locks: would have IndexError

        Updated to use public API methods (matches production usage).
        """
        mock_time.return_value = self.base_time

        # Pre-populate
        for i in range(300):
            ps = PriceSource(
                start_ms=self.base_time + i * 1000,
                open=100.0,
                close=100.0,
                high=100.0,
                low=100.0,
                vwap=100.0,
                websocket=True,
                lag_ms=0
            )
            self.tracker.add_event(ps)

        results = []

        def public_api_accessor_with_yields():
            """
            Access via public API methods with yields.
            This matches actual production usage.
            """
            try:
                for _ in range(500):
                    # Use public API (like production does)
                    count = self.tracker.count_events()
                    time.sleep(0)  # YIELD

                    if count > 0:
                        # Use get_closest_event (thread-safe)
                        closest = self.tracker.get_closest_event(self.base_time + 150000)
                        if closest:
                            _ = closest.close
                        time.sleep(0)

                        # Use get_events_in_range (thread-safe)
                        events = self.tracker.get_events_in_range(
                            self.base_time,
                            self.base_time + 300000
                        )
                        results.append(len(events))
                        time.sleep(0)
            except Exception as e:
                self.errors.append(('accessor', e, type(e).__name__))

        def cleanup_with_yields():
            """Cleanup with yields."""
            try:
                for i in range(100):
                    mock_time.return_value = self.base_time + (i + 1) * 10000
                    time.sleep(0)  # YIELD
                    self.tracker._cleanup_old_events()
                    time.sleep(0)  # YIELD
            except Exception as e:
                self.errors.append(('cleanup', e, type(e).__name__))

        threads = []
        for i in range(5):
            t = threading.Thread(target=public_api_accessor_with_yields, name=f'Accessor-{i}')
            threads.append(t)
            t.start()

        for i in range(2):
            t = threading.Thread(target=cleanup_with_yields, name=f'Cleanup-{i}')
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if self.errors:
            self.fail(f"Public API race with yields! Errors: {self.errors}")

        # With locks, this should complete successfully with no errors
        self.assertEqual(len(self.errors), 0, "Thread-safe operations should not error")


if __name__ == '__main__':
    unittest.main()
