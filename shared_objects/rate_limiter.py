# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

import time

class RateLimiter:
    def __init__(self, max_requests_per_window=10, rate_limit_window_duration_seconds=60):
        """
        Initializes the Rate Limiter with configurable limits and window sizes.

        Parameters:
        - max_requests_per_window: The maximum number of requests a miner is allowed to make in a given time window.
        - rate_limit_window_duration_seconds: The duration of the rate limiting window in seconds.

        The rate limiter uses a sliding window mechanism to determine whether new requests from miners are allowed
        based on their request history within the current window.
        """
        self.max_requests_per_window = max_requests_per_window
        self.rate_limit_window_duration_seconds = rate_limit_window_duration_seconds
        # A dictionary to track the request history for each miner. The key is the miner's hotkey, and the value is a tuple
        # containing the start timestamp of the current rate limit window and the number of requests made in this window.
        self.requests_history = {}

    def is_allowed(self, miner_hotkey):
        """
        Evaluates if a request from the specified miner is allowed under the current rate limit policy.

        Parameters:
        - miner_hotkey: Unique identifier for the miner making the request.

        Returns:
        - A tuple (is_request_allowed: bool, wait_time_seconds: float), where:
          - is_request_allowed indicates if the miner's request is within the rate limit.
          - wait_time_seconds is the time the miner should wait before making another request if the rate limit is exceeded.
        """
        current_time_s = time.time()
        window_start_timestamp, request_count = self.requests_history.get(miner_hotkey, (0, 0))

        # Determine if the current request is within the rate limit window.
        if current_time_s - window_start_timestamp > self.rate_limit_window_duration_seconds:
            # If outside the window, reset the count for this miner.
            self.requests_history[miner_hotkey] = (current_time_s, 1)
            return True, 0.0

        if request_count < self.max_requests_per_window:
            # If within limits, increment the request count and allow the request.
            self.requests_history[miner_hotkey] = (window_start_timestamp, request_count + 1)
            return True, 0.0
        else:
            # If limit is exceeded, calculate the remaining wait time.
            wait_time_seconds = self.rate_limit_window_duration_seconds - (current_time_s - window_start_timestamp)
            return False, wait_time_seconds
