"""
Unified SlackNotifier for PTN miners and validators.

This module provides a comprehensive Slack notification system that combines:
- Simple alert messaging with rate limiting (validator server monitoring)
- Rich formatted messages with dual-channel support (errors vs general)
- Optional metrics tracking and daily summaries (miner signal processing)
- Server monitoring alerts (websocket, REST, ledgers)
- Plagiarism detection notifications

Supports both simple and advanced use cases with opt-in complexity.
"""

import json
import os
import socket
import subprocess
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
from collections import defaultdict
import bittensor as bt

# Try to use requests library, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False


class SlackNotifier:
    """
    Unified Slack notification handler for both miners and validators.

    Features:
    - Simple alerts with rate limiting
    - Dual-channel messaging (main + error channels)
    - Daily metrics summaries (optional)
    - Server monitoring alerts (websocket, REST, ledgers)
    - Signal processing summaries
    - Plagiarism notifications

    Examples:
        # Simple validator usage (server monitoring)
        notifier = SlackNotifier(webhook_url=url, hotkey=hotkey)
        notifier.send_ledger_failure_alert("Debt Ledger", 3, "Connection timeout", 60)

        # Miner usage with metrics
        notifier = SlackNotifier(
            hotkey=hotkey,
            webhook_url=url,
            is_miner=True,
            enable_metrics=True,
            enable_daily_summary=True
        )
        notifier.send_signal_summary(signal_data)
    """

    def __init__(
        self,
        hotkey: str,
        webhook_url: Optional[str] = None,
        error_webhook_url: Optional[str] = None,
        min_interval_seconds: int = 300,
        is_miner: bool = False,
        enable_metrics: bool = False,
        enable_daily_summary: bool = False
    ):
        """
        Initialize SlackNotifier with flexible configuration.

        Args:
            hotkey: Node hotkey for identification
            webhook_url: Primary Slack webhook URL (falls back to SLACK_WEBHOOK_URL env var)
            error_webhook_url: Separate webhook for errors (optional, defaults to webhook_url)
            min_interval_seconds: Minimum seconds between same alert type (default 300)
            is_miner: Whether this is a miner node (affects metrics and summaries)
            enable_metrics: Enable daily/lifetime metrics tracking
            enable_daily_summary: Enable automated daily summary reports (requires enable_metrics)
        """
        # Core settings
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.error_webhook_url = error_webhook_url or self.webhook_url
        self.hotkey = hotkey
        self.enabled = bool(self.webhook_url)
        self.is_miner = is_miner
        self.node_type = "Miner" if is_miner else "Validator"

        # Rate limiting
        self.min_interval = min_interval_seconds
        self.last_alert_time = {}
        self.message_cooldown_lock = threading.Lock()

        # System info
        self.vm_hostname = self._get_vm_hostname()
        self.git_branch = self._get_git_branch()

        # Metrics (optional) - only initialize if enabled
        self.enable_metrics = enable_metrics
        self.enable_daily_summary = enable_daily_summary and enable_metrics

        if self.enable_metrics:
            self.vm_ip = self._get_vm_ip()
            self.startup_time = datetime.now(timezone.utc)
            self.daily_summary_lock = threading.Lock()
            self.metrics_file = f"{self.node_type.lower()}_lifetime_metrics.json"
            self.lifetime_metrics = self._load_lifetime_metrics()
            self.daily_metrics = self._reset_daily_metrics()

            if self.enable_daily_summary:
                self._start_daily_summary_thread()
        else:
            self.vm_ip = None

        if not self.webhook_url:
            bt.logging.warning("No Slack webhook URL configured. Notifications disabled.")

    # ========== Core Messaging Methods ==========

    def send_alert(self, message: str, alert_key: Optional[str] = None, force: bool = False) -> bool:
        """
        Send alert with rate limiting (simple interface from vanta_api version).

        This is the simplest interface for sending alerts. It sends plain text messages
        with optional rate limiting to prevent spam.

        Args:
            message: Message text to send
            alert_key: Unique key for rate limiting (e.g., "websocket_down")
            force: Bypass rate limiting if True

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.webhook_url:
            bt.logging.info(f"[Slack] Would send (no webhook): {message}")
            return False

        # Rate limiting
        if not force and alert_key:
            now = time.time()
            with self.message_cooldown_lock:
                last_time = self.last_alert_time.get(alert_key, 0)
                if now - last_time < self.min_interval:
                    bt.logging.debug(f"[Slack] Skipping '{alert_key}' (rate limited)")
                    return False
                self.last_alert_time[alert_key] = now

        return self._send_simple_message(message, self.webhook_url)

    def send_message(
        self,
        message: str,
        level: str = "info",
        bypass_cooldown: bool = False,
        use_attachments: bool = True
    ) -> bool:
        """
        Send message with level-based routing (miner_objects interface).

        This interface provides more features:
        - Routes errors/warnings to separate channel
        - Optional rich formatting with attachments
        - Level-based color coding
        - System info footer

        Args:
            message: Message to send
            level: Message level ("error", "warning", "success", "info")
            bypass_cooldown: Skip rate limiting if True
            use_attachments: Use rich formatting (True) or simple text (False)

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        # Cooldown check
        if not bypass_cooldown:
            message_key = message.split('\n')[0][:50]
            with self.message_cooldown_lock:
                current_time = time.time()
                last_time = self.last_alert_time.get(message_key, 0)
                if current_time - last_time < self.min_interval:
                    bt.logging.debug(f"[Slack] Message suppressed (cooldown): {message_key}")
                    return False
                self.last_alert_time[message_key] = current_time

        # Determine webhook based on level
        webhook_url = self.error_webhook_url if level in ["error", "warning"] else self.webhook_url

        # Send with or without rich formatting
        if use_attachments:
            return self._send_rich_message(message, level, webhook_url)
        else:
            return self._send_simple_message(message, webhook_url)

    # ========== Server Monitoring Alerts (from vanta_api) ==========

    def send_websocket_down_alert(self, pid: int, exit_code: int, host: str, port: int) -> bool:
        """
        Send formatted alert for websocket server failure.

        Args:
            pid: Process ID that crashed
            exit_code: Exit code of the process
            host: Websocket host
            port: Websocket port

        Returns:
            bool: True if sent successfully
        """
        message = self._format_server_alert(
            "WebSocket Server Down",
            pid, exit_code, f"ws://{host}:{port}"
        )
        return self.send_alert(message, alert_key="websocket_down")

    def send_rest_down_alert(self, pid: int, exit_code: int, host: str, port: int) -> bool:
        """
        Send formatted alert for REST server failure.

        Args:
            pid: Process ID that crashed
            exit_code: Exit code of the process
            host: REST API host
            port: REST API port

        Returns:
            bool: True if sent successfully
        """
        message = self._format_server_alert(
            "REST API Server Down",
            pid, exit_code, f"http://{host}:{port}"
        )
        return self.send_alert(message, alert_key="rest_down")

    def send_recovery_alert(self, service_name: str) -> bool:
        """
        Send alert when service recovers.

        Args:
            service_name: Name of the recovered service

        Returns:
            bool: True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":white_check_mark: *{service_name} Recovered*\n"
            f"*Time:* {timestamp}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"Service is back online after auto-restart"
        )
        return self.send_alert(message, alert_key=f"{service_name}_recovery", force=True)

    def send_restart_alert(self, service_name: str, restart_count: int, new_pid: int) -> bool:
        """
        Send alert when service is being restarted.

        Args:
            service_name: Name of the service being restarted
            restart_count: Current restart attempt number
            new_pid: New process ID after restart

        Returns:
            bool: True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":arrows_counterclockwise: *{service_name} Auto-Restarting*\n"
            f"*Time:* {timestamp}\n"
            f"*Restart Attempt:* {restart_count}/3\n"
            f"*New PID:* {new_pid}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"Attempting automatic recovery..."
        )
        return self.send_alert(message, alert_key=f"{service_name}_restart")

    def send_critical_alert(self, service_name: str, error_msg: str) -> bool:
        """
        Send critical alert when auto-restart fails.

        Args:
            service_name: Name of the failed service
            error_msg: Error message describing the failure

        Returns:
            bool: True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":red_circle: *CRITICAL: {service_name} Auto-Restart Failed*\n"
            f"*Time:* {timestamp}\n"
            f"*Error:* {error_msg}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"*Action:* MANUAL INTERVENTION REQUIRED"
        )
        return self.send_alert(message, alert_key=f"{service_name}_critical", force=True)

    def send_ledger_failure_alert(
        self,
        ledger_type: str,
        consecutive_failures: int,
        error_msg: str,
        backoff_seconds: int
    ) -> bool:
        """
        Send formatted alert for ledger update failures.

        Args:
            ledger_type: Type of ledger (e.g., "Debt Ledger", "Emissions Ledger")
            consecutive_failures: Number of consecutive failures
            error_msg: Error message (will be truncated to 200 chars)
            backoff_seconds: Backoff time before next retry

        Returns:
            bool: True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":rotating_light: *{ledger_type} - Update Failed*\n"
            f"*Time:* {timestamp}\n"
            f"*Consecutive Failures:* {consecutive_failures}\n"
            f"*Error:* {str(error_msg)[:200]}\n"
            f"*Next Retry:* {backoff_seconds}s backoff\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"*Action:* Will retry automatically. Check logs if failures persist."
        )
        alert_key = f"{ledger_type.lower().replace(' ', '_')}_failure"
        return self.send_alert(message, alert_key=alert_key)

    def send_ledger_recovery_alert(self, ledger_type: str, consecutive_failures: int) -> bool:
        """
        Send alert when ledger service recovers.

        Args:
            ledger_type: Type of ledger (e.g., "Debt Ledger", "Emissions Ledger")
            consecutive_failures: Number of failures before recovery

        Returns:
            bool: True if sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":white_check_mark: *{ledger_type} - Recovered*\n"
            f"*Time:* {timestamp}\n"
            f"*Failed Attempts:* {consecutive_failures}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"Service is back to normal"
        )
        alert_key = f"{ledger_type.lower().replace(' ', '_')}_recovery"
        return self.send_alert(message, alert_key=alert_key, force=True)

    # ========== Miner/Signal Processing (from miner_objects) ==========

    def send_signal_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send a per-signal processing summary to Slack. Only shows mothership data
        (synchronous). Validator results are tracked asynchronously in daily metrics.

        Args:
            summary_data: Dictionary from SignalMetrics.to_summary() with keys:
                - signal_uuid, trade_pair_id, timestamp
                - mothership_success, mothership_error, mothership_time_ms, mothership_retries
                - exception

        Returns:
            bool: True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            # Determine status and channel
            if summary_data.get("exception"):
                status = "EXCEPTION"
                color = "#ff0000"
                webhook_url = self.error_webhook_url
            elif summary_data.get("mothership_success"):
                status = "SUCCESS"
                color = "#00ff00"
                webhook_url = self.webhook_url
            else:
                status = "FAILED"
                color = "#ff0000"
                webhook_url = self.error_webhook_url

            fields = [
                {
                    "title": "Status | Trade Pair",
                    "value": f"{status} | {summary_data.get('trade_pair_id', 'Unknown')}",
                    "short": True
                },
                {
                    "title": "Order UUID",
                    "value": summary_data.get("signal_uuid", "Unknown")[:16],
                    "short": True
                },
                {
                    "title": "Mothership",
                    "value": f"{summary_data.get('mothership_time_ms', 0):.0f}ms | retries: {summary_data.get('mothership_retries', 0)}",
                    "short": True
                }
            ]

            # Add mothership error if present
            mothership_error = summary_data.get("mothership_error")
            if mothership_error:
                fields.append({
                    "title": "Mothership Error",
                    "value": str(mothership_error)[:200],
                    "short": False
                })

            # Add exception if present
            if summary_data.get("exception"):
                fields.append({
                    "title": "Exception",
                    "value": str(summary_data["exception"])[:200],
                    "short": False
                })

            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Signal: {status}",
                    "fields": fields,
                    "footer": f"Taoshi {self.node_type} Monitor",
                    "ts": int(time.time())
                }]
            }

            return self._send_payload(webhook_url, payload)

        except Exception as e:
            bt.logging.error(f"Failed to send Slack summary: {e}")
            return False

    def send_plagiarism_demotion_notification(self, target_hotkey: str) -> bool:
        """
        Send notification when a miner is demoted due to plagiarism.

        Args:
            target_hotkey: Hotkey of the miner being demoted

        Returns:
            bool: True if sent successfully
        """
        if not self.enabled:
            return False

        message = (
            f"🚨 Miner Demoted for Plagiarism\n\n"
            f"Miner ...{target_hotkey[-8:]} has been demoted to PLAGIARISM bucket due to detected plagiarism behavior."
        )
        return self.send_message(message, level="warning")

    def send_plagiarism_promotion_notification(self, target_hotkey: str) -> bool:
        """
        Send notification when a miner is promoted from plagiarism back to probation.

        Args:
            target_hotkey: Hotkey of the miner being promoted

        Returns:
            bool: True if sent successfully
        """
        if not self.enabled:
            return False

        message = (
            f"✅ Miner Restored from Plagiarism\n\n"
            f"Miner ...{target_hotkey[-8:]} has been promoted from PLAGIARISM bucket back to PROBATION."
        )
        return self.send_message(message, level="success")

    def send_plagiarism_elimination_notification(self, target_hotkey: str) -> bool:
        """
        Send notification when a miner is eliminated from plagiarism.

        Args:
            target_hotkey: Hotkey of the miner being eliminated

        Returns:
            bool: True if sent successfully
        """
        if not self.enabled:
            return False

        message = f"🚨 Miner Eliminated for Plagiarism\n\nMiner ...{target_hotkey[-8:]}"
        return self.send_message(message, level="warning")

    # ========== Metrics (optional, from miner_objects) ==========

    def update_daily_metrics(self, summary_data: Dict[str, Any]):
        """
        Update daily metrics with signal-level data (mothership + trade pair + success/fail).

        Args:
            summary_data: Dictionary from SignalMetrics.to_summary()
        """
        if not self.enable_metrics:
            return

        with self.daily_summary_lock:
            trade_pair_id = summary_data.get("trade_pair_id", "Unknown")
            self.daily_metrics["trade_pair_counts"][trade_pair_id] += 1

            mothership_time = summary_data.get("mothership_time_ms", 0)
            if mothership_time > 0:
                self.daily_metrics["mothership_times_ms"].append(mothership_time)

            self.daily_metrics["mothership_retries_total"] += summary_data.get("mothership_retries", 0)

            if summary_data.get("exception") or not summary_data.get("mothership_success"):
                self.daily_metrics["signals_failed"] += 1
            else:
                self.daily_metrics["signals_processed"] += 1
                self.lifetime_metrics["total_lifetime_signals"] += 1

    def update_other_validators_metrics(self, total: int, succeeded: int, errors: Dict[str, str], elapsed_ms: float):
        """
        Update daily metrics with async validator results. Called from background thread
        after fire-and-forget validator queries complete.

        Args:
            total: Number of validators queried
            succeeded: Number that succeeded
            errors: Dict of hotkey -> error message for failures
            elapsed_ms: Total time for the query batch
        """
        if not self.enable_metrics:
            return

        with self.daily_summary_lock:
            self.daily_metrics["other_validators_total"] += total
            self.daily_metrics["other_validators_succeeded"] += succeeded
            if elapsed_ms > 0:
                self.daily_metrics["other_validators_times_ms"].append(elapsed_ms)
            for hotkey, _error_msg in errors.items():
                self.daily_metrics["other_validators_errors"][hotkey] += 1

    # ========== Internal Helper Methods ==========

    def _send_simple_message(self, message: str, webhook_url: str) -> bool:
        """
        Send plain text message without attachments.

        Args:
            message: Message text to send
            webhook_url: Webhook URL to send to

        Returns:
            bool: True if sent successfully
        """
        try:
            payload = {
                "text": message,
                "username": f"PTN {self.node_type} Monitor",
                "icon_emoji": ":rotating_light:"
            }

            return self._send_payload(webhook_url, payload)

        except Exception as e:
            bt.logging.error(f"[Slack] Error sending simple message: {e}")
            return False

    def _send_rich_message(self, message: str, level: str, webhook_url: str) -> bool:
        """
        Send message with rich formatting using attachments.

        Args:
            message: Message to send
            level: Message level for color coding
            webhook_url: Webhook URL to send to

        Returns:
            bool: True if sent successfully
        """
        try:
            # Color coding for different message levels
            color_map = {
                "error": "#ff0000",
                "warning": "#ff9900",
                "success": "#00ff00",
                "info": "#0099ff"
            }

            fields = [
                {
                    "title": f"{self.node_type} Alert",
                    "value": message,
                    "short": False
                }
            ]

            # Add system info if available
            if self.vm_ip:
                fields.append({
                    "title": f"VM IP | {self.node_type} Hotkey",
                    "value": f"{self.vm_ip} | ...{self.hotkey[-8:]}",
                    "short": True
                })
                fields.append({
                    "title": "Script Uptime | Git Branch",
                    "value": f"{self._get_uptime_str()} | {self.git_branch}",
                    "short": True
                })
            else:
                fields.append({
                    "title": f"{self.node_type} Hotkey",
                    "value": f"...{self.hotkey[-8:]}",
                    "short": True
                })
                fields.append({
                    "title": "Git Branch",
                    "value": self.git_branch,
                    "short": True
                })

            payload = {
                "attachments": [{
                    "color": color_map.get(level, "#808080"),
                    "fields": fields,
                    "footer": f"Taoshi {self.node_type} Notification",
                    "ts": int(time.time())
                }]
            }

            return self._send_payload(webhook_url, payload)

        except Exception as e:
            bt.logging.error(f"[Slack] Error sending rich message: {e}")
            return False

    def _send_payload(self, webhook_url: str, payload: Dict[str, Any]) -> bool:
        """
        Send JSON payload to Slack webhook.

        Args:
            webhook_url: Webhook URL
            payload: JSON payload to send

        Returns:
            bool: True if sent successfully
        """
        try:
            if HAS_REQUESTS:
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                success = response.status_code == 200
            else:
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    webhook_url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                success = False
                with urllib.request.urlopen(req, timeout=10) as response:
                    success = response.status == 200

            if success:
                bt.logging.info(f"[Slack] Message sent successfully")
            return success

        except Exception as e:
            bt.logging.error(f"[Slack] Error sending payload: {e}")
            return False

    def _format_server_alert(self, title: str, pid: int, exit_code: int, endpoint: str) -> str:
        """
        Format server monitoring alert message.

        Args:
            title: Alert title
            pid: Process ID
            exit_code: Exit code
            endpoint: Server endpoint

        Returns:
            str: Formatted alert message
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        return (
            f":rotating_light: *{title}!*\n"
            f"*Time:* {timestamp}\n"
            f"*PID:* {pid}\n"
            f"*Exit Code:* {exit_code}\n"
            f"*Endpoint:* {endpoint}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"*Action:* Check validator logs immediately"
        )

    def _get_uptime_str(self) -> str:
        """
        Get formatted uptime string.

        Returns:
            str: Formatted uptime (e.g., "3.5 days" or "12.3 hours")
        """
        if not self.enable_metrics:
            return "N/A"

        current_uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
        total_uptime = self.lifetime_metrics.get("total_uptime_seconds", 0) + current_uptime

        if total_uptime >= 86400:
            return f"{total_uptime / 86400:.1f} days"
        else:
            return f"{total_uptime / 3600:.1f} hours"

    def _get_vm_hostname(self) -> str:
        """Get VM hostname."""
        try:
            return socket.gethostname()
        except Exception as e:
            bt.logging.error(f"Failed to get hostname: {e}")
            return "Unknown Hostname"

    def _get_vm_ip(self) -> str:
        """Get VM IP address."""
        if not HAS_REQUESTS:
            return "Unknown IP"
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            return response.text
        except Exception:
            try:
                hostname = socket.gethostname()
                return socket.gethostbyname(hostname)
            except Exception:
                return "Unknown IP"

    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            branch = result.stdout.strip()
            return branch if branch else "Unknown Branch"
        except Exception as e:
            bt.logging.error(f"Failed to get git branch: {e}")
            return "Unknown Branch"

    def _load_lifetime_metrics(self) -> Dict[str, Any]:
        """
        Load persistent metrics from file.

        Returns:
            dict: Lifetime metrics
        """
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            bt.logging.warning(f"Failed to load lifetime metrics: {e}")

        # Default metrics
        return {
            "total_lifetime_signals": 0,
            "total_uptime_seconds": 0,
            "last_shutdown_time": None
        }

    def _save_lifetime_metrics(self):
        """Save persistent metrics to file."""
        if not self.enable_metrics:
            return

        try:
            # Update uptime
            current_session_uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
            self.lifetime_metrics["total_uptime_seconds"] += current_session_uptime
            self.lifetime_metrics["last_shutdown_time"] = datetime.now(timezone.utc).isoformat()

            with open(self.metrics_file, 'w') as f:
                json.dump(self.lifetime_metrics, f)
        except Exception as e:
            bt.logging.error(f"Failed to save lifetime metrics: {e}")

    def _reset_daily_metrics(self) -> Dict[str, Any]:
        """
        Reset daily metrics.

        Returns:
            dict: Fresh daily metrics dictionary
        """
        return {
            "signals_processed": 0,
            "signals_failed": 0,
            "trade_pair_counts": defaultdict(int),
            "mothership_times_ms": [],
            "mothership_retries_total": 0,
            # Updated asynchronously by background thread
            "other_validators_total": 0,
            "other_validators_succeeded": 0,
            "other_validators_errors": defaultdict(int),  # hotkey -> count
            "other_validators_times_ms": [],
        }

    def _send_daily_summary(self):
        """Send daily summary report."""
        if not self.enable_metrics:
            return

        with self.daily_summary_lock:
            try:
                uptime_str = self._get_uptime_str()

                # Signal counts
                total_succeeded = self.daily_metrics["signals_processed"]
                total_failed = self.daily_metrics["signals_failed"]
                total_today = total_succeeded + total_failed
                success_rate = (total_succeeded / max(1, total_today)) * 100

                # Trade pair breakdown (top 10)
                trade_pairs = sorted(
                    self.daily_metrics["trade_pair_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                trade_pair_str = ", ".join([f"{pair}: {count}" for pair, count in trade_pairs]) or "None"

                # Mothership response time stats
                ms_times = self.daily_metrics["mothership_times_ms"]
                if ms_times:
                    sorted_times = sorted(ms_times)
                    n = len(sorted_times)
                    avg_ms = sum(sorted_times) / n
                    median_ms = (sorted_times[n // 2] + sorted_times[(n - 1) // 2]) / 2
                    p95_ms = sorted_times[min(int(0.95 * n), n - 1)]
                    mothership_str = f"Avg: {avg_ms:.0f}ms | Median: {median_ms:.0f}ms | P95: {p95_ms:.0f}ms\nRetries total: {self.daily_metrics['mothership_retries_total']}"
                else:
                    mothership_str = "No data"

                # Other validators stats
                ov_total = self.daily_metrics["other_validators_total"]
                ov_succeeded = self.daily_metrics["other_validators_succeeded"]
                ov_times = self.daily_metrics["other_validators_times_ms"]
                if ov_total > 0:
                    ov_rate = (ov_succeeded / max(1, ov_total)) * 100
                    ov_avg = sum(ov_times) / len(ov_times) if ov_times else 0
                    other_val_str = f"Reached: {ov_total} | Success: {ov_rate:.0f}% | Avg: {ov_avg:.0f}ms"
                else:
                    other_val_str = "No data"

                fields = [
                    {
                        "title": "Daily Summary Report",
                        "value": f"Report for {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                        "short": False
                    },
                    {
                        "title": f"{self.node_type} Hotkey",
                        "value": f"...{self.hotkey[-8:]}",
                        "short": True
                    },
                    {
                        "title": "Script Uptime",
                        "value": uptime_str,
                        "short": True
                    },
                    {
                        "title": "System Info",
                        "value": f"Host: {self.vm_hostname}\nIP: {self.vm_ip}\nBranch: {self.git_branch}",
                        "short": True
                    },
                    {
                        "title": "Lifetime Signals",
                        "value": str(self.lifetime_metrics["total_lifetime_signals"]),
                        "short": True
                    },
                    {
                        "title": "Today's Signals",
                        "value": f"{total_today} ({total_succeeded} ok, {total_failed} failed)",
                        "short": True
                    },
                    {
                        "title": "Success Rate",
                        "value": f"{success_rate:.1f}%",
                        "short": True
                    },
                    {
                        "title": "Trade Pairs",
                        "value": trade_pair_str,
                        "short": False
                    },
                    {
                        "title": "Mothership Stats",
                        "value": mothership_str,
                        "short": False
                    },
                    {
                        "title": "Other Validators",
                        "value": other_val_str,
                        "short": False
                    }
                ]

                # Failing validators (top 5 by error count)
                ov_errors = self.daily_metrics["other_validators_errors"]
                if ov_errors:
                    top_failures = sorted(ov_errors.items(), key=lambda x: x[1], reverse=True)[:5]
                    fail_str = "\n".join([f"...{hk[-8:]}: {cnt} errors" for hk, cnt in top_failures])
                    fields.append({
                        "title": "Failing Validators (top 5)",
                        "value": fail_str,
                        "short": False
                    })

                payload = {
                    "attachments": [{
                        "color": "#4CAF50",
                        "fields": fields,
                        "footer": f"Taoshi {self.node_type} Daily Summary",
                        "ts": int(time.time())
                    }]
                }

                self._send_payload(self.webhook_url, payload)

                # Reset daily metrics after successful send
                self.daily_metrics = self._reset_daily_metrics()

            except Exception as e:
                bt.logging.error(f"Failed to send daily summary: {e}")

    def _start_daily_summary_thread(self):
        """Start the daily summary background thread."""
        if not self.enabled or not self.enable_daily_summary:
            return

        def daily_summary_loop():
            while True:
                try:
                    now = datetime.now(timezone.utc)
                    # Calculate seconds until next midnight UTC
                    next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    if next_midnight <= now:
                        next_midnight = next_midnight + timedelta(days=1)

                    sleep_seconds = (next_midnight - now).total_seconds()
                    time.sleep(sleep_seconds)

                    # Send daily summary
                    self._send_daily_summary()

                except Exception as e:
                    bt.logging.error(f"Error in daily summary thread: {e}")
                    time.sleep(3600)  # Sleep 1 hour on error

        summary_thread = threading.Thread(target=daily_summary_loop, daemon=True)
        summary_thread.start()

    def shutdown(self):
        """Clean shutdown - save metrics."""
        if self.enable_metrics:
            try:
                self._save_lifetime_metrics()
            except Exception as e:
                bt.logging.error(f"Error during shutdown: {e}")

    def __getstate__(self):
        """Prepare object for pickling - exclude unpicklable threading.Lock."""
        state = self.__dict__.copy()
        # Remove the unpicklable locks
        state.pop('daily_summary_lock', None)
        state.pop('message_cooldown_lock', None)
        return state

    def __setstate__(self, state):
        """Restore object after unpickling - recreate threading.Lock."""
        self.__dict__.update(state)
        # Recreate the locks in the new process
        self.message_cooldown_lock = threading.Lock()
        if self.enable_metrics:
            self.daily_summary_lock = threading.Lock()
