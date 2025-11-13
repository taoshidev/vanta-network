import json
import os
import time
import traceback
import threading
from multiprocessing import Process, Manager
from vanta_api.rest_server import VantaRestServer
from vanta_api.websocket_server import WebSocketServer
from vanta_api.slack_notifier import SlackNotifier

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


def start_rest_server(shared_queue, host="127.0.0.1", port=48888, refresh_interval=15, position_manager=None,
                      contract_manager=None, miner_statistics_manager=None, request_core_manager=None,
                      asset_selection_manager=None, debt_ledger_manager=None, limit_order_manager=None):
    """Starts the REST API server in a separate process."""
    try:
        print(f"[REST] Step 1/4: Starting REST server process with host={host}, port={port}")

        # Get default API keys file path
        api_keys_file = ValiBkpUtils.get_api_keys_file_path()
        print(f"[REST] Step 2/4: API keys file path: {api_keys_file}")

        # Create and run the REST server
        print(f"[REST] Step 3/4: Creating VantaRestServer instance...")
        rest_server = VantaRestServer(
            api_keys_file=api_keys_file,
            shared_queue=shared_queue,
            host=host,
            port=port,
            refresh_interval=refresh_interval,
            position_manager=position_manager,
            contract_manager=contract_manager,
            miner_statistics_manager=miner_statistics_manager,
            request_core_manager=request_core_manager,
            asset_selection_manager=asset_selection_manager,
            debt_ledger_manager=debt_ledger_manager,
            limit_order_manager=limit_order_manager
        )
        print(f"[REST] Step 4/4: PTNRestServer created successfully, starting server...")
        rest_server.run()
    except Exception as e:
        print(f"[REST] FATAL ERROR in REST server process: {e}")
        print(f"[REST] Exception type: {type(e).__name__}")
        print(traceback.format_exc())
        raise


def start_websocket_server(shared_queue, host="localhost", port=8765, refresh_interval=15):
    """Starts the WebSocket server in a separate process."""
    try:
        # Get default API keys file path
        api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        print(f"Starting WebSocket server process with host={host}, port={port}")

        # Create and run the WebSocket server with the shared queue
        print(f"Creating WebSocketServer instance...")
        websocket_server = WebSocketServer(
            api_keys_file=api_keys_file,
            shared_queue=shared_queue,
            host=host,
            port=port,
            refresh_interval=refresh_interval
        )
        print(f"WebSocketServer instance created, calling run()...")
        websocket_server.run()
        print(f"WebSocketServer.run() returned (this shouldn't happen unless shutting down)")
    except Exception as e:
        print(f"FATAL: Exception in WebSocket server process: {type(e).__name__}: {e}")
        print(f"Full traceback:")
        print(traceback.format_exc())
        raise


class APIManager:
    """Manages API services and processes."""

    def __init__(self, shared_queue, refresh_interval=15,
                 rest_host="127.0.0.1", rest_port=48888,
                 ws_host="localhost", ws_port=8765,
                 position_manager=None, contract_manager=None,
                 miner_statistics_manager=None, request_core_manager=None,
                 asset_selection_manager=None, slack_webhook_url=None, debt_ledger_manager=None,
                 validator_hotkey=None, limit_order_manager=None):
        """Initialize API management with shared queue and server configurations.

        Args:
            shared_queue: Multiprocessing.Queue for WebSocket messaging (required)
            refresh_interval: How often to check for API key changes (seconds)
            rest_host: Host address for the REST API server
            rest_port: Port for the REST API server
            ws_host: Host address for the WebSocket server
            ws_port: Port for the WebSocket server
            position_manager: PositionManager instance (optional) for fast miner positions
            contract_manager: ValidatorContractManager instance (optional) for collateral operations
            slack_webhook_url: Slack webhook URL for health alerts (optional)
            validator_hotkey: Validator hotkey for identification in alerts (optional)
        """
        if shared_queue is None:
            raise ValueError("shared_queue cannot be None - a valid queue is required")

        self.shared_queue = shared_queue
        self.refresh_interval = refresh_interval

        # Server configurations
        self.rest_host = rest_host
        self.rest_port = rest_port
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.position_manager = position_manager
        self.contract_manager = contract_manager
        self.miner_statistics_manager = miner_statistics_manager
        self.request_core_manager = request_core_manager
        self.asset_selection_manager = asset_selection_manager
        self.debt_ledger_manager = debt_ledger_manager

        # Initialize Slack notifier
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)
        self.health_monitor_thread = None
        self.shutdown_event = threading.Event()

        # Process references (set in run())
        self.rest_process = None
        self.ws_process = None

        # Restart throttling
        self.rest_restart_times = []  # Track restart timestamps
        self.ws_restart_times = []    # Track restart timestamps
        self.max_restarts_per_window = 3
        self.restart_window_seconds = 300  # 5 minutes
        self.restart_lock = threading.Lock()  # Protect restart operations
        self.limit_order_manager = limit_order_manager

        # Get default API keys file path
        self.api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        # Verify API keys file exists
        if not os.path.exists(self.api_keys_file):
            print(f"WARNING: API keys file '{self.api_keys_file}' not found!")
        else:
            print(f"API keys file found at: {self.api_keys_file}")
            # Check if it's a valid JSON file
            try:
                with open(self.api_keys_file, "r") as f:
                    keys = json.load(f)
                print(f"API keys file contains {len(keys)} keys")
            except Exception as e:
                print(f"ERROR reading API keys file: {e}")

    def _can_restart(self, service_name, restart_times):
        """
        Check if a service can be restarted based on throttling rules.

        Args:
            service_name: Name of the service for logging
            restart_times: List of recent restart timestamps

        Returns:
            bool: True if restart is allowed, False if throttled
        """
        current_time = time.time()

        # Remove restart times outside the window
        cutoff_time = current_time - self.restart_window_seconds
        restart_times[:] = [t for t in restart_times if t > cutoff_time]

        # Check if we've hit the limit
        if len(restart_times) >= self.max_restarts_per_window:
            print(f"[APIManager] {service_name} restart THROTTLED: "
                  f"{len(restart_times)} restarts in last {self.restart_window_seconds}s (max: {self.max_restarts_per_window})")
            return False

        return True

    def _restart_rest_server(self):
        """
        Restart the REST server process.

        Returns:
            bool: True if restart was attempted, False if throttled
        """
        with self.restart_lock:
            # Check throttling
            if not self._can_restart("REST Server", self.rest_restart_times):
                self.slack_notifier.send_critical_alert(
                    "REST Server",
                    f"Auto-restart failed: exceeded {self.max_restarts_per_window} restarts in {self.restart_window_seconds}s"
                )
                return False

            # Record restart attempt
            self.rest_restart_times.append(time.time())
            restart_count = len(self.rest_restart_times)

            print(f"[APIManager] Attempting to restart REST server (attempt {restart_count}/{self.max_restarts_per_window})...")

            # Terminate old process
            if self.rest_process and self.rest_process.is_alive():
                print(f"[APIManager] Terminating old REST process (PID: {self.rest_process.pid})...")
                self.rest_process.terminate()
                self.rest_process.join(timeout=5)
                if self.rest_process.is_alive():
                    print(f"[APIManager] Force killing REST process...")
                    self.rest_process.kill()

            # Create new process
            self.rest_process = Process(
                target=start_rest_server,
                args=(self.shared_queue, self.rest_host, self.rest_port, self.refresh_interval,
                      self.position_manager, self.contract_manager, self.miner_statistics_manager,
                      self.request_core_manager, self.asset_selection_manager, self.debt_ledger_manager),
                name="RestServer"
            )
            self.rest_process.start()

            print(f"[APIManager] REST server restarted (new PID: {self.rest_process.pid})")
            self.slack_notifier.send_restart_alert("REST Server", restart_count, self.rest_process.pid)

            return True

    def _restart_websocket_server(self):
        """
        Restart the WebSocket server process.

        Returns:
            bool: True if restart was attempted, False if throttled
        """
        with self.restart_lock:
            # Check throttling
            if not self._can_restart("WebSocket Server", self.ws_restart_times):
                self.slack_notifier.send_critical_alert(
                    "WebSocket Server",
                    f"Auto-restart failed: exceeded {self.max_restarts_per_window} restarts in {self.restart_window_seconds}s"
                )
                return False

            # Record restart attempt
            self.ws_restart_times.append(time.time())
            restart_count = len(self.ws_restart_times)

            print(f"[APIManager] Attempting to restart WebSocket server (attempt {restart_count}/{self.max_restarts_per_window})...")

            # Terminate old process
            if self.ws_process and self.ws_process.is_alive():
                print(f"[APIManager] Terminating old WebSocket process (PID: {self.ws_process.pid})...")
                self.ws_process.terminate()
                self.ws_process.join(timeout=5)
                if self.ws_process.is_alive():
                    print(f"[APIManager] Force killing WebSocket process...")
                    self.ws_process.kill()

            # Create new process
            self.ws_process = Process(
                target=start_websocket_server,
                args=(self.shared_queue, self.ws_host, self.ws_port, self.refresh_interval),
                name="WebSocketServer"
            )
            self.ws_process.start()

            print(f"[APIManager] WebSocket server restarted (new PID: {self.ws_process.pid})")
            self.slack_notifier.send_restart_alert("WebSocket Server", restart_count, self.ws_process.pid)

            return True

    def _health_monitor_daemon(self):
        """
        Daemon thread that monitors process health, attempts automatic restarts, and sends Slack alerts.
        Runs independently of the main monitoring loop.
        """
        import socket

        print("[HealthMonitor] Daemon thread started")
        ws_was_down = False
        rest_was_down = False
        check_count = 0

        # Grace period: Don't send alerts during initial startup (60 seconds / 6 checks)
        STARTUP_GRACE_CHECKS = 6
        print(f"[HealthMonitor] Startup grace period: {STARTUP_GRACE_CHECKS} checks ({STARTUP_GRACE_CHECKS * 10} seconds)")

        def check_port_listening(host, port, timeout=2):
            """Check if a port is actually listening and accepting connections."""
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0  # 0 means success
            except Exception as e:
                print(f"[HealthMonitor] Error checking port {host}:{port}: {e}")
                return False

        while not self.shutdown_event.is_set():
            try:
                check_count += 1
                in_grace_period = check_count <= STARTUP_GRACE_CHECKS

                # Log heartbeat every 6 checks (1 minute)
                if check_count % 6 == 0:
                    print(f"[HealthMonitor] Heartbeat #{check_count}: "
                          f"WS={'UP' if self.ws_process.is_alive() else 'DOWN'}, "
                          f"REST={'UP' if self.rest_process.is_alive() else 'DOWN'}")

                # Check WebSocket server health
                ws_process_alive = self.ws_process.is_alive()
                ws_port_open = check_port_listening(self.ws_host, self.ws_port)
                ws_healthy = ws_process_alive and ws_port_open

                if not ws_healthy:
                    # Only act after grace period
                    if not ws_was_down and not in_grace_period:
                        print(f"[HealthMonitor] WebSocket server DOWN! "
                              f"Process alive: {ws_process_alive}, "
                              f"Port {self.ws_port} open: {ws_port_open}, "
                              f"PID: {self.ws_process.pid}, Exit: {self.ws_process.exitcode}")
                        self.slack_notifier.send_websocket_down_alert(
                            pid=self.ws_process.pid,
                            exit_code=self.ws_process.exitcode,
                            host=self.ws_host,
                            port=self.ws_port
                        )
                        ws_was_down = True

                        # Attempt automatic restart
                        print("[HealthMonitor] Attempting to restart WebSocket server...")
                        self._restart_websocket_server()

                        # Give new process time to start up
                        time.sleep(30)

                    elif in_grace_period:
                        # During grace period, just log
                        print(f"[HealthMonitor] WebSocket not ready yet (startup grace period, check {check_count}/{STARTUP_GRACE_CHECKS})")
                else:
                    # Only send recovery alerts if we actually detected a failure (not during startup)
                    if ws_was_down:
                        print("[HealthMonitor] WebSocket server RECOVERED!")
                        self.slack_notifier.send_recovery_alert("WebSocket Server")
                        ws_was_down = False

                # Check REST server health
                rest_process_alive = self.rest_process.is_alive()
                rest_port_open = check_port_listening(self.rest_host, self.rest_port)
                rest_healthy = rest_process_alive and rest_port_open

                if not rest_healthy:
                    # Only act after grace period
                    if not rest_was_down and not in_grace_period:
                        print(f"[HealthMonitor] REST server DOWN! "
                              f"Process alive: {rest_process_alive}, "
                              f"Port {self.rest_port} open: {rest_port_open}, "
                              f"PID: {self.rest_process.pid}, Exit: {self.rest_process.exitcode}")
                        self.slack_notifier.send_rest_down_alert(
                            pid=self.rest_process.pid,
                            exit_code=self.rest_process.exitcode,
                            host=self.rest_host,
                            port=self.rest_port
                        )
                        rest_was_down = True

                        # Attempt automatic restart
                        print("[HealthMonitor] Attempting to restart REST server...")
                        self._restart_rest_server()

                        # Give new process time to start up
                        time.sleep(30)

                    elif in_grace_period:
                        # During grace period, just log
                        print(f"[HealthMonitor] REST server not ready yet (startup grace period, check {check_count}/{STARTUP_GRACE_CHECKS})")
                else:
                    # Only send recovery alerts if we actually detected a failure (not during startup)
                    if rest_was_down:
                        print("[HealthMonitor] REST server RECOVERED!")
                        self.slack_notifier.send_recovery_alert("REST Server")
                        rest_was_down = False

                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"[HealthMonitor] Error in health check: {e}")
                traceback.print_exc()
                time.sleep(10)

        print("[HealthMonitor] Daemon thread stopped")

    def run(self):
        """Main entry point to run REST API and WebSocket server with automatic restart capability."""
        print("Starting API services with automatic restart enabled...")

        # Start REST server process with host/port configuration
        self.rest_process = Process(
            target=start_rest_server,
            args=(self.shared_queue, self.rest_host, self.rest_port, self.refresh_interval, self.position_manager,
                  self.contract_manager, self.miner_statistics_manager, self.request_core_manager,
                  self.asset_selection_manager, self.debt_ledger_manager,  self.limit_order_manager),
            name="RestServer"
        )
        self.rest_process.start()
        print(f"REST API server process started (PID: {self.rest_process.pid}) at http://{self.rest_host}:{self.rest_port}")

        # Start WebSocket server process with host/port configuration
        self.ws_process = Process(
            target=start_websocket_server,
            args=(self.shared_queue, self.ws_host, self.ws_port, self.refresh_interval),
            name="WebSocketServer"
        )
        self.ws_process.start()
        print(f"WebSocket server process started (PID: {self.ws_process.pid}) at ws://{self.ws_host}:{self.ws_port}")

        # Start health monitor daemon thread (now uses self.rest_process and self.ws_process)
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_daemon,
            daemon=True,
            name="HealthMonitor"
        )
        self.health_monitor_thread.start()
        print("Health monitor daemon thread started (with auto-restart enabled)")

        # Keep main thread alive - health monitoring happens in daemon thread
        try:
            while True:
                time.sleep(60)  # Just keep alive, daemon handles all monitoring

        except KeyboardInterrupt:
            print("\nShutting down API services due to keyboard interrupt...")

            # Signal health monitor to stop
            self.shutdown_event.set()

            # Terminate processes
            if self.rest_process.is_alive():
                print(f"Terminating REST server process (PID: {self.rest_process.pid})...")
                self.rest_process.terminate()
            if self.ws_process.is_alive():
                print(f"Terminating WebSocket server process (PID: {self.ws_process.pid})...")
                self.ws_process.terminate()

            # Wait for clean shutdown
            self.rest_process.join(timeout=10)
            self.ws_process.join(timeout=10)
            print("API services shutdown complete.")


if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the API services")
    parser.add_argument("--rest-host", type=str, default="127.0.0.1", help="Host for the REST server")
    parser.add_argument("--rest-port", type=int, default=48888, help="Port for the REST server")
    parser.add_argument("--ws-host", type=str, default="localhost", help="Host for the WebSocket server")
    parser.add_argument("--ws-port", type=int, default=8765, help="Port for the WebSocket server")

    args = parser.parse_args()

    # Create a manager for the shared queue
    mp_manager = Manager()
    shared_queue = mp_manager.Queue()

    # Create test message
    shared_queue.put({"type": "test", "message": "This is a test message", "timestamp": int(time.time() * 1000)})

    # Create and run the API manager with command-line arguments
    api_manager = APIManager(
        shared_queue=shared_queue,
        rest_host=args.rest_host,
        rest_port=args.rest_port,
        ws_host=args.ws_host,
        ws_port=args.ws_port
    )
    api_manager.run()
