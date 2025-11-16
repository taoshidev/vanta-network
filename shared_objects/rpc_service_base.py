"""
RPC Service Base Class - Common infrastructure for RPC client/server services.

This module provides a base class that consolidates common patterns across all RPC services:
- Process lifecycle management (start, stop, cleanup)
- Stale server cleanup (port conflict resolution)
- Client connection with retries
- Direct vs RPC mode for unit tests
- Secure authkey generation
- Readiness signaling via Event
- Optional health checking with auto-restart

Example usage:

    class MyServiceClient(RPCServiceBase):
        def __init__(self, some_dependency, running_unit_tests=False):
            super().__init__(
                service_name="MyService",
                port=50003,
                running_unit_tests=running_unit_tests,
                enable_health_check=True  # Optional
            )
            self.some_dependency = some_dependency

            # Start the service (RPC or direct mode)
            self._initialize_service()

        def _create_direct_server(self):
            '''Create direct in-memory instance for tests'''
            return MyService(self.some_dependency)

        def _start_server_process(self, address, authkey, server_ready):
            '''Start RPC server in separate process'''
            from multiprocessing import Process

            def server_main():
                server = MyService(self.some_dependency)
                self._serve_rpc(server, address, authkey, server_ready)

            process = Process(target=server_main, daemon=True)
            process.start()
            return process

        # Client API methods - proxy to self._server_proxy
        def some_method(self, arg):
            return self._server_proxy.some_method_rpc(arg)

With health checking:

    class MyServiceClient(RPCServiceBase):
        def __init__(self, slack_notifier=None, running_unit_tests=False):
            super().__init__(
                service_name="MyService",
                port=50003,
                running_unit_tests=running_unit_tests,
                enable_health_check=True,
                health_check_interval_s=60,
                max_consecutive_failures=3,
                enable_auto_restart=True,
                slack_notifier=slack_notifier
            )
            self._initialize_service()

        # Server must implement health_check_rpc() method:
        # def health_check_rpc(self) -> dict:
        #     return {"status": "ok", "timestamp_ms": TimeUtil.now_in_millis()}
"""
import os
import time
import secrets
import signal
import subprocess
import traceback
import bittensor as bt
from abc import ABC, abstractmethod
from multiprocessing import Event, Process
from multiprocessing.managers import BaseManager
from typing import Optional, Tuple, Dict
from setproctitle import setproctitle
from time_util.time_util import TimeUtil
from shared_objects.error_utils import ErrorUtils
from shared_objects.port_manager import PortManager


class RPCServiceBase(ABC):
    """
    Abstract base class for RPC client/server services.

    Provides common functionality for:
    - Process lifecycle (start, stop, cleanup)
    - Client connection with retries
    - Direct vs RPC mode (for unit tests)
    - Secure authkey generation
    - Readiness signaling
    - Stale server cleanup

    Subclasses must implement:
    - _create_direct_server(): Create in-memory server for tests
    - _start_server_process(): Start RPC server in separate process
    """

    def __init__(
        self,
        service_name: str,
        port: int,
        running_unit_tests: bool = False,
        enable_health_check: bool = False,
        health_check_interval_s: int = 60,
        max_consecutive_failures: int = 3,
        enable_auto_restart: bool = True,
        slack_notifier=None
    ):
        """
        Initialize the RPC service base.

        Args:
            service_name: Name of the service (for logging and process naming)
            port: Port number for RPC server
            running_unit_tests: Whether running in unit test mode (uses direct mode)
            enable_health_check: Whether to enable health checking
            health_check_interval_s: Seconds between health checks (default: 60)
            max_consecutive_failures: Max failures before triggering restart (default: 3)
            enable_auto_restart: Whether to auto-restart on health check failures (default: True)
            slack_notifier: Optional SlackNotifier for health check alerts
        """
        self.service_name = service_name
        self.port = port
        self.running_unit_tests = running_unit_tests

        # Process management
        self._server_process: Optional[Process] = None
        self._client_manager: Optional[BaseManager] = None
        self._server_proxy = None  # Proxy object for RPC calls (or direct instance in test mode)

        # Connection settings
        self._address = ('localhost', port)
        self._authkey: Optional[bytes] = None
        self._max_connection_retries = 5
        self._connection_retry_delay_s = 1.0

        # Health check settings
        self.enable_health_check = enable_health_check
        self.health_check_interval_s = health_check_interval_s
        self.health_check_interval_ms = health_check_interval_s * 1000
        self.max_consecutive_failures = max_consecutive_failures
        self.enable_auto_restart = enable_auto_restart
        self.slack_notifier = slack_notifier

        # Health check state
        self._last_health_check_time_ms = 0
        self._consecutive_failures = 0
        self._health_check_enabled_for_instance = False  # Set to True after successful init

    def _initialize_service(self):
        """
        Initialize the service in either direct or RPC mode.
        Call this from subclass __init__ after setting up dependencies.
        """
        if self.running_unit_tests:
            bt.logging.info(f"{self.service_name} using direct in-memory mode (unit tests)")
            self._server_proxy = self._create_direct_server()
            # No health checks in unit test mode
        else:
            bt.logging.info(f"{self.service_name} starting RPC mode on port {self.port}")
            self._start_rpc_mode()
            # Enable health checks after successful initialization
            if self.enable_health_check:
                self._health_check_enabled_for_instance = True
                bt.logging.success(
                    f"{self.service_name} health checks enabled "
                    f"(interval: {self.health_check_interval_s}s, "
                    f"max_failures: {self.max_consecutive_failures}, "
                    f"auto_restart: {self.enable_auto_restart})"
                )

    @abstractmethod
    def _create_direct_server(self):
        """
        Create a direct in-memory server instance for unit tests.

        Returns:
            Server instance (not proxied, direct Python object)

        Example:
            return MyService(dependency1, dependency2, running_unit_tests=True)
        """
        raise NotImplementedError("Subclass must implement _create_direct_server()")

    @abstractmethod
    def _start_server_process(self, address: Tuple[str, int], authkey: bytes, server_ready: Event) -> Process:
        """
        Start the RPC server in a separate process.

        Args:
            address: (host, port) tuple for RPC server
            authkey: Authentication key for RPC connection
            server_ready: Event to signal when server is ready to accept connections

        Returns:
            Process object for the server process

        Example:
            def server_main():
                from setproctitle import setproctitle
                setproctitle(f"vali_{self.service_name}")

                server = MyService(...)
                self._serve_rpc(server, address, authkey, server_ready)

            process = Process(target=server_main, daemon=True)
            process.start()
            return process
        """
        raise NotImplementedError("Subclass must implement _start_server_process()")

    def _serve_rpc(self, server_instance, address: Tuple[str, int], authkey: bytes, server_ready: Event):
        """
        Helper method to serve an RPC server instance.

        This is a convenience method that subclasses can call from _start_server_process().
        It handles the BaseManager setup and serve_forever() call.

        Args:
            server_instance: The server object to expose via RPC
            address: (host, port) tuple
            authkey: Authentication key
            server_ready: Event to signal when ready
        """
        class ServiceManager(BaseManager):
            pass

        # Register the service with the manager
        ServiceManager.register(self.service_name, callable=lambda: server_instance)

        # Create manager and get server
        manager = ServiceManager(address=address, authkey=authkey)
        server = manager.get_server()

        bt.logging.success(f"{self.service_name} server ready on {address}")

        # Signal that server is ready
        if server_ready:
            server_ready.set()
            bt.logging.debug(f"{self.service_name} readiness event set")

        # Start serving (blocks forever)
        server.serve_forever()

    def _start_rpc_mode(self):
        """Start the service in RPC mode with process and client connection."""
        # Generate secure authentication key
        self._authkey = secrets.token_bytes(32)

        # Cleanup any stale servers on this port
        self._cleanup_stale_server()

        # Create readiness event
        server_ready = Event()

        # Start server process (subclass implements this)
        self._server_process = self._start_server_process(
            self._address,
            self._authkey,
            server_ready
        )

        if not self._server_process:
            raise RuntimeError(f"{self.service_name} _start_server_process() returned None")

        bt.logging.info(
            f"{self.service_name} server process started (PID: {self._server_process.pid})"
        )

        # Wait for server to signal readiness
        if not server_ready.wait(timeout=10.0):
            bt.logging.error(f"{self.service_name} server failed to start within 10 seconds")
            self._server_process.terminate()
            raise TimeoutError(f"{self.service_name} server startup timeout")

        bt.logging.debug(f"{self.service_name} server signaled ready, connecting client...")

        # Connect client
        self._connect_client()

    def _connect_client(self):
        """Connect the RPC client to the server with retries."""
        # Create client manager class
        class ClientManager(BaseManager):
            pass

        # Register the service type (client just needs to know the type name)
        ClientManager.register(self.service_name)

        # Retry connection with backoff
        for attempt in range(self._max_connection_retries):
            try:
                manager = ClientManager(address=self._address, authkey=self._authkey)
                manager.connect()

                # Get the proxy object
                self._server_proxy = getattr(manager, self.service_name)()
                self._client_manager = manager

                bt.logging.success(
                    f"{self.service_name} client connected to server at {self._address}"
                )
                return

            except Exception as e:
                if attempt < self._max_connection_retries - 1:
                    bt.logging.warning(
                        f"{self.service_name} connection failed (attempt {attempt + 1}/"
                        f"{self._max_connection_retries}): {e}. Retrying in "
                        f"{self._connection_retry_delay_s}s..."
                    )
                    time.sleep(self._connection_retry_delay_s)
                else:
                    bt.logging.error(
                        f"{self.service_name} failed to connect after "
                        f"{self._max_connection_retries} attempts: {e}"
                    )
                    raise

    def _cleanup_stale_server(self):
        """
        Kill any existing process using the service's RPC port.

        This prevents "Address already in use" errors when restarting.
        Based on cleanup_stale_position_manager_server() from position_manager_server.py.
        """
        if os.name != 'posix':
            bt.logging.debug(f"{self.service_name} port cleanup only supported on POSIX systems")
            return

        try:
            # Find processes using the port
            result = subprocess.run(
                ['lsof', '-ti', f':{self.port}'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')

                for pid_str in pids:
                    try:
                        pid = int(pid_str)

                        # Check if it's one of our server processes
                        cmd_result = subprocess.run(
                            ['ps', '-p', str(pid), '-o', 'comm='],
                            capture_output=True,
                            text=True,
                            timeout=1
                        )

                        if cmd_result.returncode == 0:
                            process_name = cmd_result.stdout.strip()

                            # Only kill if it looks like our server
                            # (matches service name or is a python process)
                            if self.service_name in process_name or 'python' in process_name:
                                bt.logging.warning(
                                    f"Killing stale {self.service_name} server process "
                                    f"(PID: {pid}, port: {self.port})"
                                )

                                try:
                                    os.kill(pid, signal.SIGTERM)

                                    # Wait briefly for graceful termination
                                    # Poll to see if process terminated
                                    deadline = time.time() + 1.0
                                    while time.time() < deadline:
                                        try:
                                            # Check if process still exists
                                            os.kill(pid, 0)  # Signal 0 checks existence
                                            time.sleep(0.05)
                                        except ProcessLookupError:
                                            # Process terminated
                                            break

                                    # Force kill if still alive
                                    try:
                                        os.kill(pid, signal.SIGKILL)
                                    except ProcessLookupError:
                                        pass  # Already dead

                                except ProcessLookupError:
                                    pass  # Process already terminated

                    except (ValueError, subprocess.TimeoutExpired) as e:
                        bt.logging.trace(f"Error checking process {pid_str}: {e}")

        except FileNotFoundError:
            bt.logging.trace(f"{self.service_name}: lsof command not available, skipping port cleanup")
        except subprocess.TimeoutExpired:
            bt.logging.warning(f"{self.service_name} port cleanup timed out")
        except Exception as e:
            bt.logging.warning(f"{self.service_name} error during port cleanup: {e}")

    def _is_process_alive_safe(self, process: Optional[Process]) -> bool:
        """
        Safely check if a process is alive, handling fork scenarios.

        Multiprocessing.Process.is_alive() checks that _parent_pid == os.getpid(),
        which fails if the current process forked (e.g., PM2 daemonization).

        Args:
            process: Process object to check (or None)

        Returns:
            bool: True if process is alive, False otherwise (or if not checkable)
        """
        if not process:
            return False

        try:
            return process.is_alive()
        except AssertionError as e:
            # AssertionError: can only test a child process
            # This happens when we're in a forked child trying to check a parent's process
            if "can only test a child process" in str(e):
                bt.logging.debug(
                    f"{self.service_name} process check failed (forked child context), "
                    f"assuming process is not manageable from this context"
                )
                return False
            raise

    def health_check(self, current_time_ms: Optional[int] = None) -> bool:
        """
        Perform health check on the RPC server.

        Features:
        - Rate limiting (only checks every health_check_interval_s)
        - Consecutive failure tracking
        - Auto-restart on max_consecutive_failures
        - Slack notifications
        - Recovery detection

        Args:
            current_time_ms: Current timestamp in milliseconds (defaults to now)

        Returns:
            bool: True if healthy, False if unhealthy

        Note:
            Server must implement health_check_rpc() method returning:
            {"status": "ok", "timestamp_ms": <timestamp>, ...}
        """
        # Health checks only enabled in RPC mode
        if not self._health_check_enabled_for_instance:
            return True

        # If server_proxy is None, server is dead/unreachable
        # This should NEVER happen with proper deployment - it indicates either:
        # 1. Server process crashed (auto-restart will handle)
        # 2. Process was forked after RPC initialization (MISCONFIGURATION)
        if not self._server_proxy:
            bt.logging.error(
                f"❌ {self.service_name} health check failed: _server_proxy is None\n"
                f"This indicates a serious issue:\n"
                f"  • Server process may have crashed, OR\n"
                f"  • Process was forked after RPC initialization (DEPLOYMENT ERROR)\n"
                f"\n"
                f"If using PM2: Ensure validator is started WITHOUT fork mode.\n"
                f"RPC services must be initialized BEFORE any process forking."
            )

            self._consecutive_failures += 1

            if self._consecutive_failures >= self.max_consecutive_failures:
                if self.slack_notifier:
                    self.slack_notifier.send_message(
                        f"🔴 {self.service_name} _server_proxy is None after {self._consecutive_failures} checks\n"
                        f"This may indicate:\n"
                        f"1. Server crash (auto-restart will attempt recovery)\n"
                        f"2. Process fork AFTER RPC init (check deployment config)\n"
                        f"\n"
                        f"Auto-restart: {'Enabled' if self.enable_auto_restart else 'Disabled'}",
                        level="error"
                    )

                self._trigger_restart("_server_proxy is None")

            return False

        # Get current time
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        # Rate limiting: only check if enough time has passed
        if current_time_ms - self._last_health_check_time_ms < self.health_check_interval_ms:
            return True  # Skip check, assume healthy

        self._last_health_check_time_ms = current_time_ms

        try:
            # Call the health_check_rpc method on the server
            health_status = self._server_proxy.health_check_rpc()

            if health_status.get("status") == "ok":
                # Health check succeeded
                if self._consecutive_failures > 0:
                    # Recovery detected - server came back online
                    recovery_msg = (
                        f"✅ {self.service_name} recovered after "
                        f"{self._consecutive_failures} failed health checks"
                    )
                    bt.logging.success(recovery_msg)

                    if self.slack_notifier:
                        self.slack_notifier.send_message(
                            f"{recovery_msg}\n"
                            f"Server timestamp: {health_status.get('timestamp_ms', 'unknown')}",
                            level="info"
                        )

                # Reset failure counter
                self._consecutive_failures = 0
                return True

            else:
                # Health check returned non-ok status
                self._consecutive_failures += 1
                bt.logging.warning(
                    f"{self.service_name} health check returned status: "
                    f"{health_status.get('status', 'unknown')} "
                    f"(failure {self._consecutive_failures}/{self.max_consecutive_failures})"
                )

                # Trigger restart if we've hit the failure threshold
                if self._consecutive_failures >= self.max_consecutive_failures:
                    self._trigger_restart("non-ok status")

                return False

        except Exception as e:
            # Health check RPC call failed (connection error, timeout, etc.)
            self._consecutive_failures += 1
            error_trace = traceback.format_exc()

            bt.logging.warning(
                f"{self.service_name} health check failed "
                f"(failure {self._consecutive_failures}/{self.max_consecutive_failures}): {e}"
            )
            bt.logging.trace(error_trace)

            # Trigger restart if we've hit the failure threshold
            if self._consecutive_failures >= self.max_consecutive_failures:
                self._trigger_restart(f"RPC error: {str(e)}")

            return False

    def _trigger_restart(self, reason: str):
        """
        Trigger a restart of the RPC server due to health check failures.

        Args:
            reason: Description of why restart was triggered
        """
        restart_msg = (
            f"🔄 {self.service_name} triggering restart after "
            f"{self._consecutive_failures} consecutive health check failures\n"
            f"Reason: {reason}"
        )

        bt.logging.error(restart_msg)

        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"{restart_msg}\n"
                f"Auto-restart enabled: {self.enable_auto_restart}",
                level="error"
            )

        if not self.enable_auto_restart:
            bt.logging.warning(
                f"{self.service_name} auto-restart is disabled, manual intervention required"
            )
            return

        try:
            # Shutdown existing server
            bt.logging.info(f"{self.service_name} shutting down for restart...")
            self.shutdown()

            # Wait for port to actually be released (usually <50ms)
            if not PortManager.wait_for_port_release(self.port, timeout=5.0):
                bt.logging.warning(
                    f"{self.service_name} port {self.port} still in use after shutdown, "
                    f"attempting restart anyway"
                )
            else:
                bt.logging.debug(f"{self.service_name} port {self.port} released successfully")

            # Restart the server
            bt.logging.info(f"{self.service_name} restarting server...")
            self._start_rpc_mode()

            # Reset failure counter after successful restart
            self._consecutive_failures = 0

            restart_success_msg = f"✅ {self.service_name} successfully restarted"
            bt.logging.success(restart_success_msg)

            if self.slack_notifier:
                self.slack_notifier.send_message(restart_success_msg, level="info")

        except Exception as e:
            error_trace = traceback.format_exc()
            restart_error_msg = (
                f"❌ {self.service_name} restart failed: {str(e)}\n"
                f"Manual intervention required!"
            )

            bt.logging.error(restart_error_msg)
            bt.logging.error(error_trace)

            if self.slack_notifier:
                self.slack_notifier.send_message(
                    f"{restart_error_msg}\n\nError details:\n{error_trace[:500]}",
                    level="error"
                )

            # Re-raise to ensure caller is aware of failure
            raise

    def shutdown(self):
        """Shutdown the RPC service cleanly."""
        # For direct mode (unit tests), nothing to clean up
        if self.running_unit_tests:
            self._server_proxy = None
            bt.logging.debug(f"{self.service_name} shutdown (direct mode)")
            return

        # For RPC mode: close client connection and terminate server process
        if self._client_manager:
            try:
                bt.logging.debug(f"{self.service_name} shutting down RPC client connection")
                self._client_manager.shutdown()
            except Exception as e:
                bt.logging.trace(f"{self.service_name} error shutting down RPC client: {e}")
            finally:
                self._client_manager = None
                self._server_proxy = None

        # Use safe process check (handles fork scenarios)
        if self._is_process_alive_safe(self._server_process):
            bt.logging.debug(
                f"{self.service_name} terminating RPC server process "
                f"(PID: {self._server_process.pid})"
            )
            self._server_process.terminate()
            self._server_process.join(timeout=2)

            # Check again after terminate
            if self._is_process_alive_safe(self._server_process):
                bt.logging.warning(
                    f"{self.service_name} force killing RPC server process "
                    f"(PID: {self._server_process.pid})"
                )
                self._server_process.kill()
                self._server_process.join()

            # Wait for port to actually be released (usually <50ms)
            if PortManager.wait_for_port_release(self.port, timeout=3.0):
                bt.logging.debug(
                    f"{self.service_name} port {self.port} released successfully"
                )
            else:
                bt.logging.warning(
                    f"{self.service_name} port {self.port} still in use after shutdown"
                )

            bt.logging.success(f"{self.service_name} RPC server shutdown complete")

    def __del__(self):
        """
        Cleanup: terminate the RPC server process when service is destroyed.

        IMPORTANT: Only cleanup if we're in the same process that created the server.
        If the process was forked after initialization, __del__ in the child should
        NOT attempt cleanup (that would break the parent's server).
        """
        # Safety check: Only cleanup if running_unit_tests or if we actually own the server
        # In forked child processes, we don't want to cleanup parent's resources
        if self.running_unit_tests:
            # Unit test mode - safe to cleanup
            self.shutdown()
        elif self._server_process:
            try:
                # Check if we can access the server process (will fail if forked)
                _ = self._server_process.is_alive()
                # If we get here, we own the process - safe to cleanup
                self.shutdown()
            except (AssertionError, AttributeError):
                # Forked child context or invalid process - DO NOT cleanup
                # This prevents breaking the parent's RPC servers
                pass
