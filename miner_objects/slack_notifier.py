# Enhanced SlackNotifier with separate channels, daily summaries, and error categorization
import json
import socket
import requests
import threading
import time
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
from collections import defaultdict
import bittensor as bt


class SlackNotifier:
    """Handles all Slack notifications for miners and validators with enhanced features"""

    def __init__(self, hotkey, webhook_url: Optional[str] = None, error_webhook_url: Optional[str] = None, is_miner: bool = True):
        self.webhook_url = webhook_url
        self.hotkey = hotkey
        self.error_webhook_url = error_webhook_url or webhook_url  # Fallback to main if not provided
        self.enabled = bool(webhook_url)
        self.is_miner = is_miner
        self.node_type = "Miner" if is_miner else "Validator"
        self.vm_ip = self._get_vm_ip()
        self.vm_hostname = self._get_vm_hostname()
        self.git_branch = self._get_git_branch()

        # Daily summary tracking
        self.startup_time = datetime.now(timezone.utc)
        self.daily_summary_lock = threading.Lock()
        self.last_summary_date = None

        # Persistent metrics (survive restarts)
        self.metrics_file = f"{self.node_type.lower()}_lifetime_metrics.json"
        self.lifetime_metrics = self._load_lifetime_metrics()

        # Daily metrics (reset each day)
        self.daily_metrics = {
            "signals_processed": 0,
            "signals_failed": 0,
            "validator_response_times": [],  # All individual validator response times in ms
            "validator_counts": [],
            "trade_pair_counts": defaultdict(int),
            "successful_validators": set(),
            "error_categories": defaultdict(int),
            "failing_validators": defaultdict(int)
        }

        # Message cooldown tracking (prevents spam)
        self.message_cooldown_seconds = 5 * 60  # 5 minutes
        self.last_message_times = {}  # message_key -> timestamp
        self.message_cooldown_lock = threading.Lock()

        # Start daily summary thread
        self._start_daily_summary_thread()

    def _get_vm_ip(self) -> str:
        """Get the VM's IP address"""
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            return response.text
        except Exception as e:
            try:
                bt.logging.error(f"Got exception: {e}")
                hostname = socket.gethostname()
                return socket.gethostbyname(hostname)
            except Exception as e2:
                bt.logging.error(f"Got exception: {e2}")
                return "Unknown IP"

    def _get_vm_hostname(self) -> str:
        """Get the VM's hostname"""
        try:
            return socket.gethostname()
        except Exception as e:
            bt.logging.error(f"Got exception: {e}")
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

    def _load_lifetime_metrics(self) -> Dict[str, Any]:
        """Load persistent metrics from file
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            bt.logging.warning(f"Failed to load lifetime metrics: {e}")
        """
        # Default metrics
        return {
            "total_lifetime_signals": 0,
            "total_uptime_seconds": 0,
            "last_shutdown_time": None
        }

    def _save_lifetime_metrics(self):
        """Save persistent metrics to file"""
        try:
            # Update uptime
            if self.lifetime_metrics.get("last_shutdown_time"):
                last_shutdown = datetime.fromisoformat(self.lifetime_metrics["last_shutdown_time"])
                downtime = (self.startup_time - last_shutdown).total_seconds()
                # Only add if downtime was reasonable (less than 7 days)
                if 0 < downtime < 7 * 24 * 3600:
                    pass  # Don't add downtime to uptime

            current_session_uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
            self.lifetime_metrics["total_uptime_seconds"] += current_session_uptime
            self.lifetime_metrics["last_shutdown_time"] = datetime.now(timezone.utc).isoformat()

            with open(self.metrics_file, 'w') as f:
                json.dump(self.lifetime_metrics, f)
        except Exception as e:
            bt.logging.error(f"Failed to save lifetime metrics: {e}")

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages"""
        error_lower = error_message.lower()

        if any(keyword in error_lower for keyword in ['timeout', 'timed out', 'time out']):
            return "Timeout"
        elif any(keyword in error_lower for keyword in ['connection', 'connect', 'refused', 'unreachable']):
            return "Connection Failed"
        elif any(keyword in error_lower for keyword in ['invalid', 'decode', 'parse', 'json', 'format']):
            return "Invalid Response"
        elif any(keyword in error_lower for keyword in ['network', 'dns', 'resolve']):
            return "Network Error"
        else:
            return "Other"

    def _start_daily_summary_thread(self):
        """Start the daily summary thread"""
        if not self.enabled:
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

                    # Send daily summary (only makes sense for miners at this moment)
                    if self.is_miner:
                        self._send_daily_summary()

                except Exception as e:
                    bt.logging.error(f"Error in daily summary thread: {e}")
                    time.sleep(3600)  # Sleep 1 hour on error

        summary_thread = threading.Thread(target=daily_summary_loop, daemon=True)
        summary_thread.start()

    def _get_uptime_str(self) -> str:
        """Get formatted uptime string"""
        current_uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
        total_uptime = self.lifetime_metrics["total_uptime_seconds"] + current_uptime

        if total_uptime >= 86400:
            return f"{total_uptime / 86400:.1f} days"
        else:
            return f"{total_uptime / 3600:.1f} hours"


    def _send_daily_summary(self):
        """Send daily summary report"""
        with self.daily_summary_lock:
            try:
                # Calculate uptime
                uptime_str = self._get_uptime_str()

                # Validator response time stats
                response_times = self.daily_metrics["validator_response_times"]
                if response_times:
                    best_response_time = min(response_times)
                    worst_response_time = max(response_times)
                    avg_response_time = sum(response_times) / len(response_times)
                    # Calculate median
                    sorted_times = sorted(response_times)
                    n = len(sorted_times)
                    median_response_time = (sorted_times[n // 2] + sorted_times[(n - 1) // 2]) / 2
                    # Calculate 95th percentile
                    p95_index = int(0.95 * n)
                    p95_response_time = sorted_times[min(p95_index, n - 1)]
                else:
                    best_response_time = worst_response_time = avg_response_time = median_response_time = p95_response_time = 0

                # Validator count stats
                val_counts = self.daily_metrics["validator_counts"]
                if val_counts:
                    min_validators = min(val_counts)
                    max_validators = max(val_counts)
                    avg_validators = sum(val_counts) / len(val_counts)
                else:
                    min_validators = max_validators = avg_validators = 0

                # Success rate
                total_today = self.daily_metrics["signals_processed"]
                failed_today = self.daily_metrics["signals_failed"]
                success_rate = ((total_today - failed_today) / max(1, total_today)) * 100

                # Trade pair breakdown (top 10)
                trade_pairs = sorted(
                    self.daily_metrics["trade_pair_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                trade_pair_str = ", ".join([f"{pair}: {count}" for pair, count in trade_pairs]) or "None"

                # Error category breakdown
                error_categories = dict(self.daily_metrics["error_categories"])
                error_str = ", ".join([f"{cat}: {count}" for cat, count in error_categories.items()]) or "None"

                fields = [
                    {
                        "title": "📊 Daily Summary Report",
                        "value": f"Automated daily report for {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                        "short": False
                    },
                    {
                        "title": f"🕒 {self.node_type} Hotkey",
                        "value": f"...{self.hotkey[-8:]}",
                        "short": True
                    },
                    {
                        "title": "Script Uptime",
                        "value": uptime_str,
                        "short": True
                    },
                    {
                        "title": "📈 Lifetime Signals",
                        "value": str(self.lifetime_metrics["total_lifetime_signals"]),
                        "short": True
                    },
                    {
                        "title": "📅 Today's Signals",
                        "value": str(total_today),
                        "short": True
                    },
                    {
                        "title": "✅ Success Rate",
                        "value": f"{success_rate:.1f}%",
                        "short": True
                    },
                    {
                        "title": "⚡ Validator Response Times (ms)",
                        "value": f"Best: {best_response_time:.0f}ms\nWorst: {worst_response_time:.0f}ms\nAvg: {avg_response_time:.0f}ms\nMedian: {median_response_time:.0f}ms\n95th %ile: {p95_response_time:.0f}ms",
                        "short": True
                    },
                    {
                        "title": "🔗 Validator Counts",
                        "value": f"Min: {min_validators}\nMax: {max_validators}\nAvg: {avg_validators:.1f}",
                        "short": True
                    },
                    {
                        "title": "💱 Trade Pairs",
                        "value": trade_pair_str,
                        "short": False
                    },
                    {
                        "title": "✨ Unique Validators",
                        "value": str(len(self.daily_metrics["successful_validators"])),
                        "short": True
                    },
                    {
                        "title": "🖥️ System Info",
                        "value": f"Host: {self.vm_hostname}\nIP: {self.vm_ip}\nBranch: {self.git_branch}",
                        "short": True
                    }
                ]

                if error_categories:
                    fields.append({
                        "title": "❌ Error Categories",
                        "value": error_str,
                        "short": False
                    })

                payload = {
                    "attachments": [{
                        "color": "#4CAF50",  # Green for summary
                        "fields": fields,
                        "footer": f"Taoshi {self.node_type} Daily Summary",
                        "ts": int(time.time())
                    }]
                }

                # Send to main channel (not error channel)
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                response.raise_for_status()

                # Reset daily metrics after successful send
                self.daily_metrics = {
                    "signals_processed": 0,
                    "signals_failed": 0,
                    "validator_response_times": [],
                    "validator_counts": [],
                    "trade_pair_counts": defaultdict(int),
                    "successful_validators": set(),
                    "error_categories": defaultdict(int),
                    "failing_validators": defaultdict(int)
                }

            except Exception as e:
                bt.logging.error(f"Failed to send daily summary: {e}")

    def send_message(self, message: str, level: str = "info", bypass_cooldown: bool = False):
        """Send a message to appropriate Slack channel based on level

        Args:
            message: The message to send
            level: Message level (error, warning, success, info)
            bypass_cooldown: If True, skip cooldown check (for critical messages)
        """
        if not self.enabled:
            return

        # Check cooldown unless bypassed
        if not bypass_cooldown:
            # Create a message key from the first line/emoji to group similar messages
            message_key = message.split('\n')[0][:50]  # First 50 chars of first line

            with self.message_cooldown_lock:
                current_time = time.time()
                last_send_time = self.last_message_times.get(message_key, 0)

                # Check if we're still in cooldown period
                if current_time - last_send_time < self.message_cooldown_seconds:
                    time_remaining = int(self.message_cooldown_seconds - (current_time - last_send_time))
                    bt.logging.debug(f"Slack message suppressed (cooldown). Retry in {time_remaining}s: {message_key}")
                    return

                # Update last send time
                self.last_message_times[message_key] = current_time

        try:
            # Determine which webhook to use
            if level in ["error", "warning"]:
                webhook_url = self.error_webhook_url
            else:
                webhook_url = self.webhook_url

            # Color coding for different message levels
            color_map = {
                "error": "#ff0000",
                "warning": "#ff9900",
                "success": "#00ff00",
                "info": "#0099ff"
            }

            payload = {
                "attachments": [{
                    "color": color_map.get(level, "#808080"),
                    "fields": [
                        {
                            "title": f"{self.node_type} Alert",
                            "value": message,
                            "short": False
                        },
                        {
                            "title": f"VM IP | {self.node_type} Hotkey",
                            "value": f"{self.vm_ip} | ...{self.hotkey[-8:]}",
                            "short": True
                        },
                        {
                            "title": "Script Uptime | Git Branch",
                            "value": f"{self._get_uptime_str()} | {self.git_branch}",
                            "short": True
                        }
                    ],
                    "footer": f"Taoshi {self.node_type} Notification",
                    "ts": int(time.time())
                }]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

        except Exception as e:
            bt.logging.error(f"Failed to send Slack notification: {e}")

    def update_daily_metrics(self, signal_data: Dict[str, Any]):
        """Update daily metrics with signal processing data"""
        with self.daily_summary_lock:
            # Update trade pair counts
            trade_pair_id = signal_data.get("trade_pair_id", "Unknown")
            self.daily_metrics["trade_pair_counts"][trade_pair_id] += 1

            # Update validator response times (individual validator times in ms)
            if "validator_response_times" in signal_data:
                validator_times = signal_data["validator_response_times"].values()
                self.daily_metrics["validator_response_times"].extend(validator_times)

            # Update validator counts
            if "validators_attempted" in signal_data:
                self.daily_metrics["validator_counts"].append(signal_data["validators_attempted"])

            # Track successful validators
            if "validator_response_times" in signal_data:
                self.daily_metrics["successful_validators"].update(signal_data["validator_response_times"].keys())

            # Update error categories
            if signal_data.get("validator_errors"):
                for validator_hotkey, errors in signal_data["validator_errors"].items():
                    for error in errors:
                        category = self._categorize_error(error)
                        self.daily_metrics["error_categories"][category] += 1
                        self.daily_metrics["failing_validators"][validator_hotkey] += 1

            # Update signal counts
            if signal_data.get("exception"):
                self.daily_metrics["signals_failed"] += 1
            else:
                self.daily_metrics["signals_processed"] += 1
                # Update lifetime metrics
                self.lifetime_metrics["total_lifetime_signals"] += 1
                #self._save_lifetime_metrics()

    def send_signal_summary(self, summary_data: Dict[str, Any]):
        """Send a formatted signal processing summary to appropriate Slack channel"""
        if not self.enabled:
            return

        try:
            # Update daily metrics first
            self.update_daily_metrics(summary_data)

            # Determine overall status and which channel to use
            if summary_data.get("exception") or not summary_data.get('validators_succeeded'):
                status = "❌ Failed"
                color = "#ff0000"
                webhook_url = self.error_webhook_url
            elif summary_data.get("all_high_trust_succeeded", False):
                status = "✅ Success"
                color = "#00ff00"
                webhook_url = self.webhook_url
            else:
                status = "⚠️ Partial Success"
                color = "#ff9900"
                webhook_url = self.error_webhook_url

            # Build enhanced fields
            fields = [
                {
                    "title": "Status | Trade Pair",
                    "value": status + " | " + summary_data.get("trade_pair_id", "Unknown"),
                    "short": True
                },
                {
                    "title": f"{self.node_type} Hotkey | Order UUID",
                    "value": "..." + summary_data.get("miner_hotkey", "Unknown")[-8:] + f" | {summary_data.get('signal_uuid', 'Unknown')[:12]}...",
                },
                {
                    "title": "VM IP | Script Uptime",
                    "value": f"{self.vm_ip} | {self._get_uptime_str()}",
                    "short": True
                },
                {
                    "title": "Validators (succeeded/attempted)",
                    "value": f"{summary_data.get('validators_succeeded', 0)}/{summary_data.get('validators_attempted', 0)}",
                    "short": True
                }
            ]

            # Add error categorization if present
            if summary_data.get("validator_errors"):
                error_categories = defaultdict(int)
                for validator_errors in summary_data["validator_errors"].values():
                    for error in validator_errors:
                        category = self._categorize_error(error)
                        error_categories[category] += 1

                if error_categories:
                    error_summary = ", ".join([f"{cat}: {count}" for cat, count in error_categories.items()])
                    error_messages_truncated = []
                    for e in summary_data.get("validator_errors", {}).values():
                        e = str(e)
                        if len(e) > 100:
                            error_messages_truncated.append(e[100:300])
                        else:
                            error_messages_truncated.append(e)
                    fields.append({
                        "title": "🔍 Error Info",
                        "value": error_summary + "\n" + "\n".join(error_messages_truncated),
                        "short": False
                        })

            # Add validator response times if present
            if summary_data.get("validator_response_times"):
                response_times = summary_data["validator_response_times"]
                unique_times = set(response_times.values())

                if len(unique_times) > len(response_times) * 0.3:
                    # Granular per-validator times
                    sorted_times = sorted(response_times.items(), key=lambda x: x[1], reverse=True)
                    response_time_str = "Individual validator response times:\n"
                    for validator, time_taken in sorted_times[:10]:
                        response_time_str += f"• ...{validator[-8:]}: {time_taken}ms\n"
                    if len(sorted_times) > 10:
                        response_time_str += f"... and {len(sorted_times) - 10} more validators"
                else:
                    # Batch processing times
                    time_groups = defaultdict(list)
                    for validator, time_taken in response_times.items():
                        time_groups[time_taken].append(validator)

                    sorted_groups = sorted(time_groups.items(), key=lambda x: x[0], reverse=True)
                    response_time_str = "Response times by retry attempt:\n"
                    for time_taken, validators in sorted_groups:
                        validator_count = len(validators)
                        example_validators = ", ".join(["..." + v[-8:] for v in validators[:3]])
                        if validator_count > 3:
                            example_validators += f" (+{validator_count - 3} more)"
                        response_time_str += f"• {time_taken}ms: {validator_count} validators ({example_validators})\n"

                fields.append({
                    "title": "⏱️ Validator Response Times",
                    "value": response_time_str.strip(),
                    "short": False
                })

                avg_time = summary_data.get("average_response_time", 0)
                if avg_time > 0:
                    fields.append({
                        "title": "Avg Response",
                        "value": f"{avg_time}ms",
                        "short": True
                    })

            # Add error details if present
            if summary_data.get("exception"):
                fields.append({
                    "title": "💥 Error Details",
                    "value": str(summary_data["exception"])[:200],
                    "short": False
                })

            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Signal Processing Summary - {status}",
                    "fields": fields,
                    "footer": f"Taoshi {self.node_type} Monitor",
                    "ts": int(time.time())
                }]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

        except Exception as e:
            bt.logging.error(f"Failed to send Slack summary: {e}")

    def send_plagiarism_demotion_notification(self, hotkey: str):
        """Send notification when a miner is demoted due to plagiarism"""
        if not self.enabled:
            return

        message = f"🚨 Miner Demoted for Plagiarism\n\nMiner ...{hotkey[-8:]} has been demoted to PLAGIARISM bucket due to detected plagiarism behavior."
        self.send_message(message, level="warning")

    def send_plagiarism_promotion_notification(self, hotkey: str):
        """Send notification when a miner is promoted from plagiarism back to probation"""
        if not self.enabled:
            return

        message = f"✅ Miner Restored from Plagiarism\n\nMiner ...{hotkey[-8:]} has been promoted from PLAGIARISM bucket back to PROBATION."
        self.send_message(message, level="success")

    def send_plagiarism_elimination_notification(self, hotkey: str):
        """Send notification when a miner is eliminated from plagiarism"""
        if not self.enabled:
            return

        message = f"🚨 Miner Eliminated for Plagiarism\n\nMiner ...{hotkey[-8:]}"
        self.send_message(message, level="warning")

    def shutdown(self):
        """Clean shutdown - save metrics"""
        try:
            self._save_lifetime_metrics()
        except Exception as e:
            bt.logging.error(f"Error during shutdown: {e}")

    def __getstate__(self):
        """Prepare object for pickling - exclude unpicklable threading.Lock"""
        state = self.__dict__.copy()
        # Remove the unpicklable locks
        state.pop('daily_summary_lock', None)
        state.pop('message_cooldown_lock', None)
        return state

    def __setstate__(self, state):
        """Restore object after unpickling - recreate threading.Lock"""
        self.__dict__.update(state)
        # Recreate the locks in the new process
        self.daily_summary_lock = threading.Lock()
        self.message_cooldown_lock = threading.Lock()