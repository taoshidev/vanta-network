
from sortedcontainers import SortedList
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig

def sorted_list_key(x):
    return x[0]

class RecentEventTracker:
    def __init__(self):
        self.events = SortedList(key=sorted_list_key)  # Assuming each event is a tuple (timestamp, event_data)
        self.timestamp_to_event = {}

    def add_event(self, event, is_forex_quote=False, tp_debug_str: str = None):
        event_time_ms = event.start_ms
        if self.timestamp_exists(event_time_ms):
            #print(f'Duplicate timestamp {TimeUtil.millis_to_formatted_date_str(event_time_ms)} for tp {tp_debug_str} ignored')
            return
        self.events.add((event_time_ms, event))
        self.timestamp_to_event[event_time_ms] = (event, ([event.bid], [event.ask]) if is_forex_quote else None)
        #print(f"Added event at {TimeUtil.millis_to_formatted_date_str(event_time_ms)}")
        self._cleanup_old_events()
        #print(event, tp_debug_str)

    def get_event_by_timestamp(self, timestamp_ms):
        # Already locked by caller
        return self.timestamp_to_event.get(timestamp_ms, (None, None))

    def timestamp_exists(self, timestamp_ms):
        return timestamp_ms in self.timestamp_to_event

    @staticmethod
    def forex_median_price(arr):
        median_price = arr[len(arr) // 2] if len(arr) % 2 == 1 else (arr[len(arr) // 2] + arr[len(arr) // 2 - 1]) / 2.0
        return median_price

    def update_prices_for_median(self, t_ms, bid_price, ask_price):
        existing_event, prices = self.get_event_by_timestamp(t_ms)
        if prices:
            prices[0].append(bid_price)
            prices[0].sort()
            prices[1].append(ask_price)
            prices[1].sort()
            median_bid = self.forex_median_price(prices[0])
            median_ask = self.forex_median_price(prices[1])
            existing_event.open = existing_event.close = existing_event.high = existing_event.low = (median_bid + median_ask) / 2.0
            existing_event.bid = median_bid
            existing_event.ask = median_ask

    def _cleanup_old_events(self):
        # Don't lock here, as this method is called from within a lock
        current_time_ms = TimeUtil.now_in_millis()
        # Calculate the oldest valid time once, outside the loop
        oldest_valid_time_ms = current_time_ms - ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS
        while self.events and self.events[0][0] < oldest_valid_time_ms:
            removed_event = self.events.pop(0)
            del self.timestamp_to_event[removed_event[0]]

    def get_events_in_range(self, start_time_ms, end_time_ms):
        """
            Get all events that have timestamps between start_time_ms and end_time_ms, inclusive.

            Args:
            start_time_ms (int): The start timestamp in milliseconds.
            end_time_ms (int): The end timestamp in milliseconds.

            Returns:
            list: A list of events (event_data) within the specified time range.
        """
        if self.count_events() == 0:
            return []
        # Find the index of the first event greater than or equal to start_time_ms
        start_idx = self.events.bisect_left((start_time_ms,))
        # Find the index of the first event strictly greater than end_time_ms
        end_idx = self.events.bisect_right((end_time_ms + 1,))  # to include events at end_time_ms
        # Retrieve all events within the range [start_idx, end_idx)
        return [event[1] for event in self.events[start_idx:end_idx]]

    def get_closest_event(self, timestamp_ms):
        #print(f"Looking for event at {TimeUtil.millis_to_formatted_date_str(timestamp_ms)}")
        if self.count_events() == 0:
            return None

        # Find the event closest to the given timestamp
        if not timestamp_ms:
            timestamp_ms = TimeUtil.now_in_millis()

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
        # Return the number of events currently stored
        return len(self.events)
