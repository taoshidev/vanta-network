import json
import os
import socket
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime
import bittensor as bt


class SlackNotifier:
    """Utility for sending Slack notifications with rate limiting."""

    def __init__(self, webhook_url=None, min_interval_seconds=300, hotkey=None):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL (can also be set via SLACK_WEBHOOK_URL env var)
            min_interval_seconds: Minimum seconds between same alert type (default 5 minutes)
            hotkey: Validator hotkey for identification in alerts
        """
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.min_interval = min_interval_seconds
        self.last_alert_time = {}  # Track last alert time per alert_key
        self.hotkey = hotkey
        self.vm_hostname = self._get_vm_hostname()
        self.git_branch = self._get_git_branch()

        if not self.webhook_url:
            bt.logging.warning("No Slack webhook URL configured. Notifications disabled.")

    def send_alert(self, message, alert_key=None, force=False):
        """
        Send alert to Slack with rate limiting.

        Args:
            message: Message text to send
            alert_key: Unique key for this alert type (for rate limiting)
            force: If True, bypass rate limiting

        Returns:
            bool: True if sent, False if skipped or failed
        """
        if not self.webhook_url:
            bt.logging.info(f"[Slack] Would send (no webhook configured): {message}")
            return False

        # Rate limiting
        if not force and alert_key:
            now = time.time()
            last_time = self.last_alert_time.get(alert_key, 0)
            if now - last_time < self.min_interval:
                bt.logging.debug(f"[Slack] Skipping alert '{alert_key}' (rate limited)")
                return False
            self.last_alert_time[alert_key] = now

        try:
            # Format payload
            payload = {
                "text": message,
                "username": "PTN Validator Monitor",
                "icon_emoji": ":rotating_light:"
            }

            # Send request
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    bt.logging.info(f"[Slack] Alert sent: {message[:50]}...")
                    return True
                else:
                    bt.logging.error(f"[Slack] Failed to send alert: HTTP {response.status}")
                    return False

        except urllib.error.URLError as e:
            bt.logging.error(f"[Slack] Network error sending alert: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"[Slack] Error sending alert: {e}")
            return False

    def _get_vm_hostname(self) -> str:
        """Get the VM's hostname"""
        try:
            return socket.gethostname()
        except Exception as e:
            bt.logging.error(f"Failed to get hostname: {e}")
            return "Unknown Hostname"

    def _get_git_branch(self) -> str:
        """Get the current git branch"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            branch = result.stdout.strip()
            if branch:
                return branch
            return "Unknown Branch"
        except Exception as e:
            bt.logging.error(f"Failed to get git branch: {e}")
            return "Unknown Branch"

    def send_websocket_down_alert(self, pid, exit_code, host, port):
        """Send formatted alert for websocket server failure."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":rotating_light: *WebSocket Server Down!*\n"
            f"*Time:* {timestamp}\n"
            f"*PID:* {pid}\n"
            f"*Exit Code:* {exit_code}\n"
            f"*Endpoint:* ws://{host}:{port}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"*Action:* Check validator logs immediately"
        )
        return self.send_alert(message, alert_key="websocket_down")

    def send_rest_down_alert(self, pid, exit_code, host, port):
        """Send formatted alert for REST server failure."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hotkey_display = f"...{self.hotkey[-8:]}" if self.hotkey else "Unknown"
        message = (
            f":rotating_light: *REST API Server Down!*\n"
            f"*Time:* {timestamp}\n"
            f"*PID:* {pid}\n"
            f"*Exit Code:* {exit_code}\n"
            f"*Endpoint:* http://{host}:{port}\n"
            f"*VM Name:* {self.vm_hostname}\n"
            f"*Validator Hotkey:* {hotkey_display}\n"
            f"*Git Branch:* {self.git_branch}\n"
            f"*Action:* Check validator logs immediately"
        )
        return self.send_alert(message, alert_key="rest_down")

    def send_recovery_alert(self, service_name):
        """Send alert when service recovers."""
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

    def send_restart_alert(self, service_name, restart_count, new_pid):
        """Send alert when service is being restarted."""
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

    def send_critical_alert(self, service_name, error_msg):
        """Send critical alert when auto-restart fails."""
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

    def send_ledger_failure_alert(self, ledger_type, consecutive_failures, error_msg, backoff_seconds):
        """
        Send formatted alert for ledger update failures.

        Args:
            ledger_type: Type of ledger (e.g., "Debt Ledger", "Emissions Ledger", "Penalty Ledger")
            consecutive_failures: Number of consecutive failures
            error_msg: Error message (will be truncated to 200 chars)
            backoff_seconds: Backoff time before next retry
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
        return self.send_alert(message, alert_key=f"{ledger_type.lower().replace(' ', '_')}_failure")

    def send_ledger_recovery_alert(self, ledger_type, consecutive_failures):
        """
        Send alert when ledger service recovers.

        Args:
            ledger_type: Type of ledger (e.g., "Debt Ledger", "Emissions Ledger", "Penalty Ledger")
            consecutive_failures: Number of failures before recovery
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
        return self.send_alert(message, alert_key=f"{ledger_type.lower().replace(' ', '_')}_recovery", force=True)
