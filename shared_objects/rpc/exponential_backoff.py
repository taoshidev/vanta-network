# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Exponential Backoff Strategy for Daemon Error Handling.

This module provides a reusable exponential backoff strategy for handling
daemon failures with smart defaults based on daemon execution frequency.
"""
import bittensor as bt


class ExponentialBackoff:
    """
    Manages exponential backoff for daemon failures.

    Tracks consecutive failures and calculates appropriate backoff times
    using exponential strategy: initial_backoff * (2 ^ (failures - 1))

    Smart defaults based on daemon interval:
    - Fast daemons (<60s): 10s initial, 300s max
    - Medium daemons (60s-3600s): 60s initial, 600s max
    - Slow daemons (>=3600s): 300s initial, 3600s max

    Example:
        backoff = ExponentialBackoff(daemon_interval_s=1.0)

        # On failure
        backoff.record_failure()
        sleep_time = backoff.calculate_backoff()
        time.sleep(sleep_time)

        # On success
        backoff.reset()
    """

    def __init__(
        self,
        daemon_interval_s: float,
        initial_backoff_s: float = None,
        max_backoff_s: float = None,
        service_name: str = "Service"
    ):
        """
        Initialize exponential backoff strategy.

        Args:
            daemon_interval_s: Daemon execution interval (used for smart defaults)
            initial_backoff_s: Initial backoff time in seconds (None for auto)
            max_backoff_s: Maximum backoff time in seconds (None for auto)
            service_name: Name for logging
        """
        self.service_name = service_name
        self.daemon_interval_s = daemon_interval_s
        self._consecutive_failures = 0

        # Smart defaults based on daemon interval
        if initial_backoff_s is None:
            if daemon_interval_s >= 3600:  # >= 1 hour: heavyweight daemons
                initial_backoff_s = 300.0   # 5 minutes
            elif daemon_interval_s >= 60:   # >= 1 minute: medium weight
                initial_backoff_s = 60.0    # 1 minute
            else:                            # < 1 minute: lightweight/fast daemons
                initial_backoff_s = 10.0    # 10 seconds

        if max_backoff_s is None:
            if daemon_interval_s >= 3600:  # >= 1 hour: heavyweight daemons
                max_backoff_s = 3600.0      # 1 hour
            elif daemon_interval_s >= 60:   # >= 1 minute: medium weight
                max_backoff_s = 600.0       # 10 minutes
            else:                            # < 1 minute: lightweight/fast daemons
                max_backoff_s = 300.0       # 5 minutes

        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s

        # Log configuration
        bt.logging.debug(
            f"{service_name} backoff: "
            f"interval={daemon_interval_s:.0f}s, "
            f"initial={initial_backoff_s:.0f}s, "
            f"max={max_backoff_s:.0f}s"
        )

    def record_failure(self) -> None:
        """Record a failure, incrementing the failure counter."""
        self._consecutive_failures += 1

    def reset(self) -> None:
        """Reset failure counter (call on successful iteration)."""
        if self._consecutive_failures > 0:
            bt.logging.info(
                f"{self.service_name} recovered after "
                f"{self._consecutive_failures} failure(s)"
            )
            self._consecutive_failures = 0

    def calculate_backoff(self) -> float:
        """
        Calculate backoff time based on consecutive failures.

        Uses exponential strategy: initial_backoff * (2 ^ (failures - 1))
        Capped at max_backoff_s.

        Returns:
            Backoff time in seconds
        """
        if self._consecutive_failures == 0:
            return 0.0

        backoff_s = min(
            self.initial_backoff_s * (2 ** (self._consecutive_failures - 1)),
            self.max_backoff_s
        )
        return backoff_s

    @property
    def consecutive_failures(self) -> int:
        """Get number of consecutive failures."""
        return self._consecutive_failures

    @property
    def has_failed(self) -> bool:
        """Check if any failures have been recorded."""
        return self._consecutive_failures > 0
