# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Watchdog Monitor for Daemon Hang Detection.

This module provides a watchdog thread that monitors daemon heartbeats
and sends alerts when a daemon appears to be hung.
"""
import time
import threading
import bittensor as bt
from time_util.time_util import TimeUtil
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator


class WatchdogMonitor:
    """
    Monitors daemon heartbeats and alerts on hangs.

    Runs a background thread that checks for heartbeat updates and
    sends Slack alerts if the daemon appears stuck.

    Example:
        watchdog = WatchdogMonitor(
            service_name="MyService",
            hang_timeout_s=60.0,
            slack_notifier=notifier
        )
        watchdog.start()

        # In daemon loop
        watchdog.update_heartbeat("processing")
        do_work()
        watchdog.update_heartbeat("idle")

        # Cleanup
        watchdog.stop()
    """

    def __init__(
        self,
        service_name: str,
        hang_timeout_s: float = 60.0,
        slack_notifier=None,
        check_interval_s: float = 5.0
    ):
        """
        Initialize watchdog monitor.

        Args:
            service_name: Name of the service being monitored
            hang_timeout_s: Seconds before alerting on hang (default: 60)
            slack_notifier: Optional SlackNotifier for alerts
            check_interval_s: How often to check heartbeat (default: 5)
        """
        self.service_name = service_name
        self.hang_timeout_s = hang_timeout_s
        self.slack_notifier = slack_notifier
        self.check_interval_s = check_interval_s

        self._last_heartbeat_ms = TimeUtil.now_in_millis()
        self._current_operation = "initializing"
        self._watchdog_alerted = False
        self._watchdog_thread: threading.Thread = None
        self._started = False

    def start(self) -> None:
        """Start the watchdog monitoring thread."""
        if self._started:
            bt.logging.warning(f"{self.service_name} watchdog already started")
            return

        self._started = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name=f"{self.service_name}_Watchdog"
        )
        self._watchdog_thread.start()
        bt.logging.info(
            f"{self.service_name} watchdog started "
            f"(timeout: {self.hang_timeout_s}s)"
        )

    def stop(self) -> None:
        """Stop the watchdog monitoring thread."""
        self._started = False

    def update_heartbeat(self, operation: str) -> None:
        """
        Update heartbeat timestamp and current operation.

        Call this regularly from daemon loops to indicate liveness.

        Args:
            operation: Description of current operation (e.g., "processing", "idle")
        """
        self._last_heartbeat_ms = TimeUtil.now_in_millis()
        self._current_operation = operation
        self._watchdog_alerted = False  # Reset alert flag on activity

    def _watchdog_loop(self) -> None:
        """Background thread that monitors heartbeat and alerts on hangs."""
        while not ShutdownCoordinator.is_shutdown() and self._started:
            time.sleep(self.check_interval_s)

            if ShutdownCoordinator.is_shutdown() or not self._started:
                continue

            elapsed_s = (TimeUtil.now_in_millis() - self._last_heartbeat_ms) / 1000.0

            if elapsed_s > self.hang_timeout_s and not self._watchdog_alerted:
                self._watchdog_alerted = True
                hang_msg = (
                    f"⚠️ {self.service_name} Daemon Hang Detected!\n"
                    f"Operation: {self._current_operation}\n"
                    f"No heartbeat for {elapsed_s:.1f}s "
                    f"(threshold: {self.hang_timeout_s}s)\n"
                    f"The daemon may be stuck and require investigation."
                )
                bt.logging.error(hang_msg)
                if self.slack_notifier:
                    self.slack_notifier.send_message(hang_msg, level="error")

        bt.logging.debug(f"{self.service_name} watchdog shutting down")

    @property
    def last_heartbeat_ms(self) -> int:
        """Get timestamp of last heartbeat."""
        return self._last_heartbeat_ms

    @property
    def current_operation(self) -> str:
        """Get current operation description."""
        return self._current_operation

    @property
    def watchdog_alerted(self) -> bool:
        """Check if watchdog has alerted on a hang."""
        return self._watchdog_alerted

    def get_status(self) -> dict:
        """
        Get watchdog status for health checks.

        Returns:
            Dict with heartbeat info and alert status
        """
        elapsed_since_heartbeat = TimeUtil.now_in_millis() - self._last_heartbeat_ms
        return {
            "operation": self._current_operation,
            "last_heartbeat_ms": self._last_heartbeat_ms,
            "elapsed_since_heartbeat_ms": elapsed_since_heartbeat,
            "watchdog_alerted": self._watchdog_alerted,
            "hang_timeout_s": self.hang_timeout_s
        }
