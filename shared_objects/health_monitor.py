# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Process Health Monitor with Auto-Restart.

This module provides health monitoring for multiprocessing.Process instances
with automatic restart capability and Slack notifications.
"""
import time
import threading
import traceback
from typing import Callable, Optional
from multiprocessing import Process
import bittensor as bt
from shared_objects.shutdown_coordinator import ShutdownCoordinator


class HealthMonitor:
    """
    Monitors process health with auto-restart capability.

    Runs a background thread that checks if a process is alive and
    automatically restarts it on failure (if enabled).

    Example:
        def restart_callback():
            # Logic to restart the process
            return new_process

        monitor = HealthMonitor(
            process=my_process,
            restart_callback=restart_callback,
            service_name="MyService",
            health_check_interval_s=30.0,
            enable_auto_restart=True,
            slack_notifier=notifier
        )
        monitor.start()

        # Later...
        if not monitor.is_alive():
            print("Process died!")

        monitor.stop()
    """

    def __init__(
        self,
        process: Process,
        restart_callback: Callable[[], Process],
        service_name: str,
        health_check_interval_s: float = 30.0,
        enable_auto_restart: bool = True,
        slack_notifier=None
    ):
        """
        Initialize health monitor.

        Args:
            process: The multiprocessing.Process to monitor
            restart_callback: Callable that returns a new Process instance on restart
            service_name: Name for logging and alerts
            health_check_interval_s: Seconds between health checks (default: 30)
            enable_auto_restart: Auto-restart if process dies (default: True)
            slack_notifier: Optional SlackNotifier for alerts
        """
        self.process = process
        self.restart_callback = restart_callback
        self.service_name = service_name
        self.health_check_interval_s = health_check_interval_s
        self.enable_auto_restart = enable_auto_restart
        self.slack_notifier = slack_notifier

        self._health_thread: Optional[threading.Thread] = None
        self._stopped = False

    def start(self) -> None:
        """Start background health monitoring thread."""
        if self._health_thread is not None:
            bt.logging.warning(f"{self.service_name} health monitor already started")
            return

        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True,
            name=f"{self.service_name}_HealthMonitor"
        )
        self._health_thread.start()
        bt.logging.info(
            f"{self.service_name} health monitoring started "
            f"(interval: {self.health_check_interval_s}s, "
            f"auto_restart: {self.enable_auto_restart})"
        )

    def stop(self) -> None:
        """Stop health monitoring."""
        self._stopped = True
        if self._health_thread:
            bt.logging.debug(f"{self.service_name} health monitoring stopped")

    def _health_loop(self) -> None:
        """Background thread monitoring process health."""
        while not ShutdownCoordinator.is_shutdown() and not self._stopped:
            time.sleep(self.health_check_interval_s)

            if ShutdownCoordinator.is_shutdown() or self._stopped:
                break

            if not self.is_alive():
                exit_code = self.process.exitcode if self.process else None
                error_msg = (
                    f"🔴 {self.service_name} process died!\n"
                    f"PID: {self.process.pid if self.process else 'N/A'}\n"
                    f"Exit code: {exit_code}\n"
                    f"Auto-restart: {'Enabled' if self.enable_auto_restart else 'Disabled'}"
                )
                bt.logging.error(error_msg)

                if self.slack_notifier:
                    self.slack_notifier.send_message(error_msg, level="error")

                if self.enable_auto_restart and not self._stopped:
                    self._restart()

        bt.logging.debug(f"{self.service_name} health loop exiting")

    def _restart(self) -> None:
        """Restart the process using the restart callback."""
        bt.logging.info(f"{self.service_name} restarting process...")

        try:
            # Call the restart callback to get a new process
            self.process = self.restart_callback()

            restart_msg = (
                f"✅ {self.service_name} process restarted successfully "
                f"(new PID: {self.process.pid})"
            )
            bt.logging.success(restart_msg)

            if self.slack_notifier:
                self.slack_notifier.send_message(restart_msg, level="info")

        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = (
                f"❌ {self.service_name} process restart failed: {e}\n"
                f"Manual intervention required!"
            )
            bt.logging.error(error_msg)
            bt.logging.error(error_trace)

            if self.slack_notifier:
                self.slack_notifier.send_message(
                    f"{error_msg}\n\nError:\n{error_trace[:500]}",
                    level="error"
                )

    def is_alive(self) -> bool:
        """Check if monitored process is running."""
        return self.process is not None and self.process.is_alive()

    @property
    def pid(self) -> Optional[int]:
        """Get process ID of monitored process."""
        return self.process.pid if self.process else None

    def get_status(self) -> dict:
        """
        Get health monitor status.

        Returns:
            Dict with process health info
        """
        return {
            "service": self.service_name,
            "pid": self.pid,
            "is_alive": self.is_alive(),
            "auto_restart_enabled": self.enable_auto_restart,
            "check_interval_s": self.health_check_interval_s
        }
