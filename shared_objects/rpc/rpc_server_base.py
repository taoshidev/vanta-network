# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
RPC Server Base Class - Unified infrastructure for all RPC servers.

This module provides a base class that consolidates common patterns across all RPC servers:
- RPC server lifecycle (start, stop)
- Daemon thread with watchdog monitoring
- Standardized health_check_rpc()
- Slack notifications on hang/failure
- Standard shutdown handling

Example usage:

    class MyServer(RPCServerBase):
        def __init__(self, metagraph, **kwargs):
            super().__init__(
                service_name="MyService",
                port=ValiConfig.RPC_MYSERVICE_PORT,
                **kwargs
            )
            self.metagraph = metagraph
            # ... initialize server-specific state

        def run_daemon_iteration(self):
            '''Single iteration of daemon work.'''
            if self._is_shutdown():
                return
            # Process work here
            self.do_some_work()

        def get_daemon_name(self) -> str:
            return "vali_MyServiceDaemon"

        # Server-specific RPC methods
        def my_method_rpc(self, arg):
            return self._do_something(arg)

Usage in validator.py:

    # Initialize ShutdownCoordinator once at startup (uses shared memory)
    from shared_objects.shutdown_coordinator import ShutdownCoordinator
    ShutdownCoordinator.initialize()

    # Create server with auto-start (no shutdown_dict needed!)
    my_server = MyServer(
        metagraph=self.metagraph,
        start_server=True,
        start_daemon=True
    )

    # Or deferred start
    my_server = MyServer(
        metagraph=self.metagraph,
        start_server=False,
        start_daemon=False
    )
    # ... later
    my_server.start_rpc_server()
    my_server.start_daemon()

    # Shutdown coordination via singleton
    ShutdownCoordinator.signal_shutdown("Validator shutting down")
"""
import time
import socket
import inspect
import threading
import traceback
import bittensor as bt
from abc import ABC, abstractmethod
from multiprocessing import Process, Event
from multiprocessing.managers import BaseManager
from typing import Optional, Callable
from setproctitle import setproctitle
from time_util.time_util import TimeUtil
from shared_objects.error_utils import ErrorUtils
from shared_objects.rpc.port_manager import PortManager
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from shared_objects.rpc.exponential_backoff import ExponentialBackoff
from shared_objects.rpc.watchdog_monitor import WatchdogMonitor
from shared_objects.rpc.health_monitor import HealthMonitor
from shared_objects.rpc.server_registry import ServerRegistry
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


def _enable_tcp_nodelay_on_listener(server) -> None:
    """
    Enable TCP_NODELAY on server's listener socket to eliminate Nagle's algorithm delays.

    Nagle's algorithm buffers small packets (adding 40-200ms delay per message) to reduce
    network overhead. For localhost RPC with many small messages, this kills performance.

    Enabling TCP_NODELAY on the listener ensures all accepted connections inherit this setting,
    reducing RPC latency from ~150ms to <5ms on localhost.

    Args:
        server: BaseManager.get_server() instance
    """
    try:
        # Access the listener socket (path: server.listener._listener._socket)
        if hasattr(server, 'listener') and server.listener is not None:
            listener = server.listener
            # multiprocessing.connection.Listener wraps the actual SocketListener in _listener
            if hasattr(listener, '_listener') and listener._listener is not None:
                socket_listener = listener._listener
                # SocketListener has the actual socket in _socket
                if hasattr(socket_listener, '_socket') and socket_listener._socket is not None:
                    sock = socket_listener._socket
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    bt.logging.debug("TCP_NODELAY enabled on RPC server listener socket")
    except Exception as e:
        # Non-critical optimization - log but don't fail
        bt.logging.trace(f"Failed to enable TCP_NODELAY on server: {e}")

class ServerProcessHandle:
    """
    Handle for managing a spawned server process with health monitoring.

    Provides:
    - Health monitoring (is process alive?)
    - Auto-restart if process dies
    - Graceful shutdown

    Created by RPCServerBase.spawn_process() - don't instantiate directly.
    Uses HealthMonitor for monitoring logic.
    """

    def __init__(
        self,
        process: Process,
        entry_point: Callable,
        entry_kwargs: dict,
        slack_notifier=None,
        health_check_interval_s: float = 30.0,
        enable_auto_restart: bool = True,
        service_name: str = "RPCServer"
    ):
        self.process = process
        self.entry_point = entry_point
        self.entry_kwargs = entry_kwargs
        self.service_name = service_name

        # Create health monitor with restart callback
        self._health_monitor = HealthMonitor(
            process=process,
            restart_callback=self._create_restart_callback(),
            service_name=service_name,
            health_check_interval_s=health_check_interval_s,
            enable_auto_restart=enable_auto_restart,
            slack_notifier=slack_notifier
        )

        # Start health monitoring
        self._health_monitor.start()

    def _create_restart_callback(self) -> Callable[[], Process]:
        """Create callback for health monitor to restart process."""
        def restart() -> Process:
            # Create new process with same entry point and args
            new_process = Process(
                target=self.entry_point,
                kwargs=self.entry_kwargs,
                daemon=True
            )
            new_process.start()
            self.process = new_process
            return new_process
        return restart

    def is_alive(self) -> bool:
        """Check if server process is running."""
        return self._health_monitor.is_alive()

    def stop(self, timeout: float = 5.0):
        """
        Stop the server process gracefully.

        Args:
            timeout: Seconds to wait for graceful shutdown before force kill
        """
        # Stop health monitoring first
        self._health_monitor.stop()

        if self.process is None or not self.process.is_alive():
            bt.logging.debug(f"{self.service_name} process already stopped")
            return

        bt.logging.info(f"{self.service_name} stopping process (PID: {self.process.pid})...")

        # Terminate gracefully
        self.process.terminate()
        self.process.join(timeout=timeout)

        # Force kill if still alive
        if self.process.is_alive():
            bt.logging.warning(f"{self.service_name} force killing process")
            self.process.kill()
            self.process.join()

        bt.logging.info(f"{self.service_name} process stopped")

    @property
    def pid(self) -> Optional[int]:
        """Get process ID."""
        return self._health_monitor.pid

    def __repr__(self):
        status = "alive" if self.is_alive() else "stopped"
        return f"ServerProcessHandle({self.service_name}, pid={self.pid}, {status})"


class RPCServerBase(ABC):
    """
    Abstract base class for all RPC servers with unified daemon management.

    Features:
    - RPC server lifecycle (start, stop)
    - Daemon thread with watchdog monitoring
    - Standardized health_check_rpc()
    - Slack notifications on hang/failure
    - Standard shutdown handling
    - Automatic instance tracking for test cleanup (via ServerRegistry)

    Subclasses must implement:
    - run_daemon_iteration(): Single iteration of daemon work
    - get_daemon_name(): Process title for setproctitle
    """
    service_name = None
    service_port = None

    @classmethod
    def shutdown_all(cls, force_kill_ports: bool = True) -> None:
        """
        Shutdown all active server instances.

        Delegates to ServerRegistry.shutdown_all().

        Args:
            force_kill_ports: If True, force-kill any processes still using RPC ports
                            after graceful shutdown (default: True)

        Example:
            def tearDown(self):
                RPCServerBase.shutdown_all()
        """
        ServerRegistry.shutdown_all(force_kill_ports=force_kill_ports)

    @classmethod
    def force_kill_ports(cls, ports: list) -> None:
        """
        Force-kill any processes using the specified ports.
        Delegates to ServerRegistry.force_kill_ports().
        """
        ServerRegistry.force_kill_ports(ports)

    @classmethod
    def force_kill_all_rpc_ports(cls) -> None:
        """
        Force-kill any processes using any known RPC port.
        Delegates to ServerRegistry.force_kill_all_rpc_ports().
        """
        ServerRegistry.force_kill_all_rpc_ports()

    def __init__(
        self,
        service_name: str,
        port: int,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = True,
        daemon_interval_s: float = 1.0,
        hang_timeout_s: float = 60.0,
        process_health_check_interval_s: float = 30.0,
        enable_process_auto_restart: bool = True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        initial_backoff_s: float = None,
        max_backoff_s: float = None,
        daemon_stagger_s: float = 0.0
    ):
        """
        Initialize the RPC server base.

        Args:
            service_name: Name of the service (for logging and RPC registration)
            port: Port number for RPC server
            slack_notifier: Optional SlackNotifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            daemon_interval_s: Seconds between daemon iterations (default: 1.0)
            hang_timeout_s: Seconds before watchdog alerts on hang (default: 60.0)
            process_health_check_interval_s: Seconds between process health checks (default: 30.0)
            enable_process_auto_restart: Whether to auto-restart dead processes (default: True)
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - don't start RPC server, used for in-process testing
                - RPC (1): Normal RPC mode - start server and accept network connections
                Default: RPC
            initial_backoff_s: Initial backoff time in seconds for exponential backoff on daemon failures.
                If None (default), auto-calculated based on daemon_interval_s:
                - Fast daemons (<60s): 10s initial backoff
                - Medium daemons (60s-3600s): 60s initial backoff
                - Slow daemons (>=3600s): 300s initial backoff
            max_backoff_s: Maximum backoff time in seconds for exponential backoff.
                If None (default), auto-calculated based on daemon_interval_s:
                - Fast daemons (<60s): 300s (5 min) max backoff
                - Medium daemons (60s-3600s): 600s (10 min) max backoff
                - Slow daemons (>=3600s): 3600s (1 hour) max backoff
            daemon_stagger_s: Initial delay in seconds before first daemon iteration to stagger startup (default: 0.0)

        Note: Shutdown coordination is now handled via ShutdownCoordinator singleton.
              No need to pass shutdown_dict parameter anymore.
        """
        self.connection_mode = connection_mode
        self.service_name = service_name
        self.port = port
        self.slack_notifier = slack_notifier
        self.daemon_interval_s = daemon_interval_s
        self.hang_timeout_s = hang_timeout_s
        self.process_health_check_interval_s = process_health_check_interval_s
        self.enable_process_auto_restart = enable_process_auto_restart
        self.daemon_stagger_s = daemon_stagger_s

        # Local shutdown flag - checked by _is_shutdown() to avoid RPC calls during shutdown
        # This prevents zombie threads when servers are shutting down
        self._local_shutdown = False

        # Create exponential backoff strategy for daemon failures
        self._backoff = ExponentialBackoff(
            daemon_interval_s=daemon_interval_s,
            initial_backoff_s=initial_backoff_s,
            max_backoff_s=max_backoff_s,
            service_name=service_name
        )

        # Daemon state
        self._daemon_thread: Optional[threading.Thread] = None
        self._daemon_started = False
        self._first_iteration = True  # Track first daemon iteration for stagger delay

        # RPC server state (thread-based)
        self._rpc_server = None
        self._rpc_thread: Optional[threading.Thread] = None
        self._server_ready = threading.Event()

        # Process-based server state
        self._server_process: Optional[Process] = None
        self._process_health_thread: Optional[threading.Thread] = None
        self._server_process_factory: Optional[Callable] = None

        # Start server if requested and in RPC mode
        if start_server and self.connection_mode == RPCConnectionMode.RPC:
            self.start_rpc_server()

        # Create watchdog monitor for hang detection (only created, not started yet)
        self._watchdog = WatchdogMonitor(
            service_name=service_name,
            hang_timeout_s=hang_timeout_s,
            slack_notifier=slack_notifier
        )

        if start_daemon:
            self.start_daemon()

        # Register instance for tracking (enables shutdown_all() for test cleanup)
        ServerRegistry.register(self)

    # ==================== Properties ====================

    def _is_shutdown(self) -> bool:
        """
        Check if shutdown has been signaled.

        Checks:
        1. ShutdownCoordinator.is_shutdown() - global singleton
        2. self._local_shutdown - local flag (prevents RPC calls during shutdown)

        Returns:
            True if shutdown is in progress

        Usage:
            # In daemon loops
            while not self._is_shutdown():
                do_work()

            # In methods
            if self._is_shutdown():
                return
        """
        # Check global shutdown coordinator (fast, local check with optional RPC fallback)
        if ShutdownCoordinator.is_shutdown():
            return True

        # Check local shutdown flag (prevents RPC calls during shutdown)
        if self._local_shutdown:
            return True

        return False

    # ==================== Abstract Methods ====================

    @abstractmethod
    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called repeatedly by daemon loop.

        Subclasses implement business logic here (e.g., process eliminations,
        check challenge periods, update prices).

        This method should:
        - Check self._is_shutdown() before long operations
        - Handle exceptions gracefully (or let base class handle them)
        - Complete in reasonable time to avoid watchdog alerts

        Example:
            def run_daemon_iteration(self):
                if self._is_shutdown():
                    return
                self.process_pending_eliminations()
                self.cleanup_expired_entries()
        """
        raise NotImplementedError("Subclass must implement run_daemon_iteration()")

    def get_daemon_name(self) -> str:
        """
        Return process title for setproctitle.

        Uses the service_name class attribute to generate a consistent daemon name.
        Subclasses should NOT override this method.

        Returns:
            str: Process title in format "vali_{service_name}"
        """
        return f"vali_{self.service_name}"

    # ==================== RPC Server Lifecycle ====================

    def start_rpc_server(self):
        """
        Start the RPC server (exposes all _rpc methods).

        The server runs in a background thread and accepts connections
        from RPCClientBase instances.
        """
        if self._rpc_server is not None:
            bt.logging.warning(f"{self.service_name} RPC server already started")
            return

        start_time = time.time()

        # Cleanup any stale servers on this port
        self._cleanup_stale_server()

        # Use 127.0.0.1 instead of 'localhost' to avoid IPv6/IPv4 fallback delays
        # 'localhost' can trigger ~170ms delay due to IPv6 ::1 timeout then IPv4 127.0.0.1 fallback
        address = ('127.0.0.1', self.port)
        authkey = ValiConfig.get_rpc_authkey(self.service_name, self.port)

        class ServerManager(BaseManager):
            pass

        # Register self as the service
        ServerManager.register(self.service_name, callable=lambda: self)

        try:
            manager = ServerManager(address=address, authkey=authkey)
            self._rpc_server = manager.get_server()

            # Enable TCP_NODELAY to eliminate Nagle's algorithm delays (~150ms -> <5ms)
            _enable_tcp_nodelay_on_listener(self._rpc_server)

            # Start serving in background thread
            self._rpc_thread = threading.Thread(
                target=self._serve_forever,
                daemon=True,
                name=f"{self.service_name}_RPC"
            )
            self._rpc_thread.start()

            # Wait for server to be ready
            if not self._server_ready.wait(timeout=5.0):
                bt.logging.warning(f"{self.service_name} RPC server may not be fully ready")

            elapsed_ms = (time.time() - start_time) * 1000
            bt.logging.success(f"{self.service_name} RPC server started on port {self.port} ({elapsed_ms:.0f}ms)")

        except Exception as e:
            bt.logging.error(f"{self.service_name} failed to start RPC server: {e}")
            raise

    def _serve_forever(self):
        """Internal method to run RPC server (called in thread)."""
        self._server_ready.set()
        try:
            self._rpc_server.serve_forever()
        except Exception as e:
            if not self._is_shutdown():
                bt.logging.error(f"{self.service_name} RPC server error: {e}")

    def stop_rpc_server(self):
        """Stop the RPC server and release the port."""
        if self._rpc_server:
            start_time = time.time()
            port = self.port  # Save port before clearing server
            rpc_server = self._rpc_server
            rpc_thread = self._rpc_thread

            # Clear references first to prevent other code from using them
            self._rpc_server = None
            self._rpc_thread = None
            self._server_ready.clear()

            try:
                # For multiprocessing.managers.Server, set stop_event to signal serve_forever to exit
                if hasattr(rpc_server, 'stop_event'):
                    rpc_server.stop_event.set()

                # Close the listener to interrupt any blocking accept() call
                if hasattr(rpc_server, 'listener'):
                    try:
                        rpc_server.listener.close()
                    except Exception:
                        pass

            except Exception as e:
                bt.logging.trace(f"{self.service_name} RPC server shutdown error: {e}")

            # Wait for the RPC thread to finish (short timeout)
            if rpc_thread and rpc_thread.is_alive():
                rpc_thread.join(timeout=0.5)
                if rpc_thread.is_alive():
                    bt.logging.trace(f"{self.service_name} RPC thread still alive after join timeout")

            # Clean up any lingering RPC connections/resources (prevents semaphore leaks)
            # The Server object maintains internal state that needs explicit cleanup
            try:
                # Close all tracked connections and clear internal registries
                if hasattr(rpc_server, 'id_to_obj'):
                    rpc_server.id_to_obj.clear()
                if hasattr(rpc_server, 'id_to_refcount'):
                    rpc_server.id_to_refcount.clear()
                if hasattr(rpc_server, 'id_to_local_proxy_obj'):
                    rpc_server.id_to_local_proxy_obj.clear()
            except Exception as e:
                bt.logging.trace(f"{self.service_name} error clearing server registries: {e}")

            elapsed_ms = (time.time() - start_time) * 1000
            bt.logging.info(f"{self.service_name} RPC server stopped ({elapsed_ms:.0f}ms)")

    def _cleanup_stale_server(self):
        """
        Aggressively kill any existing process using this port.

        Uses SIGKILL for immediate termination - designed for test cleanup
        where we need fast, reliable port release.
        """
        if PortManager.is_port_free(self.port):
            return

        bt.logging.warning(f"{self.service_name} port {self.port} in use, forcing cleanup...")
        PortManager.force_kill_port(self.port)

        # Wait for OS to release the port after killing process
        # Usually completes in <50ms, but allow up to 2 seconds
        if not PortManager.wait_for_port_release(self.port, timeout=2.0):
            bt.logging.warning(
                f"{self.service_name} port {self.port} still not free after cleanup and 2s wait. "
                f"Attempting to start anyway (SO_REUSEADDR may work)"
            )

    # ==================== Process-Based Server Lifecycle ====================

    def start_server_process(self, process_factory: Callable[[], None]) -> None:
        """
        Start the RPC server in a separate process with health monitoring.

        This is an alternative to start_rpc_server() for when you want the server
        to run in its own process (better isolation, can survive crashes).

        Args:
            process_factory: A callable that creates and runs the server.
                           This function will be called in a new process and should
                           block (e.g., call serve_forever()). It should also be
                           callable again for restarts.

        Example:
            def create_server():
                server = MyServer(...)
                RPCServerBase.serve_rpc(
                    server_instance=server,
                    service_name="MyService",
                    address=('localhost', port),
                    authkey=authkey,
                    server_ready=server_ready
                )

            self.start_server_process(create_server)
        """
        if self._server_process is not None and self._server_process.is_alive():
            bt.logging.warning(f"{self.service_name} server process already running")
            return

        # Cleanup any stale servers on this port
        self._cleanup_stale_server()

        # Store factory for restarts
        self._server_process_factory = process_factory

        # Start the server process
        self._start_server_process_internal()

        # Start health monitoring thread (only in RPC mode)
        if self.connection_mode == RPCConnectionMode.RPC:
            self._process_health_thread = threading.Thread(
                target=self._process_health_loop,
                daemon=True,
                name=f"{self.service_name}_ProcessHealth"
            )
            self._process_health_thread.start()
            bt.logging.info(
                f"{self.service_name} process health monitoring started "
                f"(interval: {self.process_health_check_interval_s}s, "
                f"auto_restart: {self.enable_process_auto_restart})"
            )

    def _start_server_process_internal(self) -> None:
        """Internal method to start/restart the server process."""
        if self._server_process_factory is None:
            raise RuntimeError(f"{self.service_name} no process factory set")

        # Create and start the process
        self._server_process = Process(
            target=self._server_process_factory,
            daemon=True,
            name=f"{self.service_name}_Process"
        )
        self._server_process.start()

        bt.logging.success(
            f"{self.service_name} server process started (PID: {self._server_process.pid})"
        )

    def _process_health_loop(self) -> None:
        """Background thread that monitors server process health."""
        bt.logging.info(f"{self.service_name} process health loop started")

        while not self._is_shutdown():
            time.sleep(self.process_health_check_interval_s)

            if self._is_shutdown():
                break

            if self._server_process is None:
                continue

            # Check if process is alive
            if not self._server_process.is_alive():
                exit_code = self._server_process.exitcode
                error_msg = (
                    f"🔴 {self.service_name} server process died!\n"
                    f"Exit code: {exit_code}\n"
                    f"Auto-restart: {'Enabled' if self.enable_process_auto_restart else 'Disabled'}"
                )
                bt.logging.error(error_msg)

                if self.slack_notifier:
                    self.slack_notifier.send_message(error_msg, level="error")

                if self.enable_process_auto_restart:
                    self._restart_server_process()

        bt.logging.debug(f"{self.service_name} process health loop shutting down")

    def _restart_server_process(self) -> None:
        """Restart the server process after it died."""
        bt.logging.info(f"{self.service_name} restarting server process...")

        try:
            # Cleanup port
            self._cleanup_stale_server()

            # Wait for port to be released
            if not PortManager.wait_for_port_release(self.port, timeout=5.0):
                bt.logging.warning(
                    f"{self.service_name} port {self.port} still in use, attempting restart anyway"
                )

            # Start new process
            self._start_server_process_internal()

            restart_msg = f"✅ {self.service_name} server process restarted successfully"
            bt.logging.success(restart_msg)

            if self.slack_notifier:
                self.slack_notifier.send_message(restart_msg, level="info")

        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = (
                f"❌ {self.service_name} server process restart failed: {e}\n"
                f"Manual intervention required!"
            )
            bt.logging.error(error_msg)
            bt.logging.error(error_trace)

            if self.slack_notifier:
                self.slack_notifier.send_message(
                    f"{error_msg}\n\nError:\n{error_trace[:500]}",
                    level="error"
                )

    def stop_server_process(self) -> None:
        """Stop the server process."""
        if self._server_process is None:
            return

        if self._server_process.is_alive():
            bt.logging.info(
                f"{self.service_name} terminating server process (PID: {self._server_process.pid})"
            )
            self._server_process.terminate()
            self._server_process.join(timeout=1.0)

            if self._server_process.is_alive():
                bt.logging.warning(f"{self.service_name} force killing server process")
                self._server_process.kill()
                self._server_process.join(timeout=0.5)

        self._server_process = None
        bt.logging.info(f"{self.service_name} server process stopped")

    def is_server_process_alive(self) -> bool:
        """Check if the server process is running."""
        return self._server_process is not None and self._server_process.is_alive()

    # ==================== Daemon Lifecycle ====================

    def start_daemon(self):
        """
        Start the daemon loop with watchdog monitoring.

        The daemon calls run_daemon_iteration() repeatedly with
        daemon_interval_s seconds between iterations.

        A watchdog thread monitors for hangs and sends alerts.
        """
        if self._daemon_started:
            bt.logging.warning(f"{self.service_name} daemon already started")
            return

        # Start daemon thread
        self._daemon_thread = threading.Thread(
            target=self._daemon_loop,
            daemon=True,
            name=f"{self.service_name}_Daemon"
        )
        self._daemon_thread.start()
        self._daemon_started = True

        # Start watchdog (monitors for hangs) - only in RPC mode
        if self.connection_mode == RPCConnectionMode.RPC:
            self._watchdog.start()

        bt.logging.success(f"{self.service_name} daemon started (interval: {self.daemon_interval_s}s)")

    def _daemon_loop(self):
        """Main daemon loop - calls run_daemon_iteration() repeatedly."""
        setproctitle(self.get_daemon_name())
        bt.logging.info(f"{self.service_name} daemon running")

        while not self._is_shutdown():
            try:
                # Check shutdown before processing
                if self._is_shutdown():
                    break

                # Initial stagger delay on first iteration (if configured)
                if self._first_iteration and self.daemon_stagger_s > 0:
                    bt.logging.info(
                        f"{self.service_name} first daemon iteration - "
                        f"staggering startup by {self.daemon_stagger_s:.0f}s..."
                    )
                    time.sleep(self.daemon_stagger_s)
                    self._first_iteration = False
                    # Check shutdown again after stagger sleep
                    if self._is_shutdown():
                        break

                self._watchdog.update_heartbeat("processing")
                self.run_daemon_iteration()
                self._watchdog.update_heartbeat("idle")

                # Success - reset backoff
                self._backoff.reset()

                time.sleep(self.daemon_interval_s)

            except Exception as e:
                # If shutting down, exit gracefully without error handling
                if self._is_shutdown():
                    break

                # Record failure and calculate backoff
                self._backoff.record_failure()
                backoff_seconds = self._backoff.calculate_backoff()

                # Log error with failure count and backoff time
                bt.logging.error(
                    f"{self.service_name} daemon error (failure #{self._backoff.consecutive_failures}): {e}",
                    exc_info=True
                )

                # Send Slack alert if notifier is available
                if self.slack_notifier:
                    error_trace = traceback.format_exc()
                    error_message = ErrorUtils.format_error_for_slack(
                        error=e,
                        traceback_str=error_trace,
                        include_operation=True,
                        include_timestamp=True
                    )
                    self.slack_notifier.send_message(
                        f"❌ {self.service_name} daemon error (failure #{self._backoff.consecutive_failures})!\n"
                        f"Next retry in {backoff_seconds:.0f}s\n{error_message}",
                        level="error"
                    )

                # Sleep for backoff duration
                bt.logging.info(f"{self.service_name} backing off for {backoff_seconds:.0f}s before retry")
                time.sleep(backoff_seconds)

        # Skip logging to avoid race condition with pytest closing stdout/stderr
        # bt.logging.info(f"{self.service_name} daemon shutting down")

    def stop_daemon(self):
        """Signal daemon to stop (via shutdown_dict)."""
        # Daemon checks shutdown_dict and will exit naturally
        self._daemon_started = False
        bt.logging.info(f"{self.service_name} daemon stop signaled")

    # ==================== Standard RPC Methods ====================

    def get_health_check_details(self) -> dict:
        """
        Hook for subclasses to add service-specific health check details.

        Override this method to add custom fields to health check response.
        DO NOT override health_check_rpc() - use this hook instead.

        Returns:
            dict: Service-specific health check fields (empty dict by default)

        Example:
            def get_health_check_details(self) -> dict:
                return {
                    "num_ledgers": len(self._manager.hotkey_to_perf_bundle),
                    "cache_status": "active" if self._cache else "empty"
                }
        """
        return {}

    def health_check_rpc(self) -> dict:
        """
        Standard health check - all servers expose this.

        Returns dict with base fields plus service-specific details from get_health_check_details().
        Subclasses should NOT override this method - override get_health_check_details() instead.
        """
        watchdog_status = self._watchdog.get_status()
        base_health = {
            "status": "ok",
            "service": self.service_name,
            "timestamp_ms": TimeUtil.now_in_millis(),
            "daemon_running": self._daemon_started,
            "last_heartbeat_ms": watchdog_status["last_heartbeat_ms"],
            "current_operation": watchdog_status["operation"]
        }

        # Merge with service-specific details (subclass hook)
        service_details = self.get_health_check_details()
        if service_details:
            base_health.update(service_details)

        return base_health

    def start_daemon_rpc(self) -> bool:
        """
        Start daemon via RPC (for deferred start).

        Returns:
            bool: True if daemon was started, False if already running
        """
        if self._daemon_started and self._daemon_thread and self._daemon_thread.is_alive():
            bt.logging.warning(f"[{self.service_name}] Daemon already running")
            return False

        self.start_daemon()
        bt.logging.success(f"[{self.service_name}] Daemon started via RPC")
        return True

    def stop_daemon_rpc(self) -> bool:
        """
        Stop daemon via RPC.

        Returns:
            bool: True if daemon was stopped, False if not running
        """
        if not self._daemon_thread or not self._daemon_thread.is_alive():
            bt.logging.warning(f"[{self.service_name}] Daemon not running")
            return False

        self.stop_daemon()
        bt.logging.success(f"[{self.service_name}] Daemon stopped via RPC")
        return True

    def is_daemon_running_rpc(self) -> bool:
        """
        Check if daemon is running via RPC.

        Returns:
            bool: True if daemon is running, False otherwise
        """
        return self._daemon_thread is not None and self._daemon_thread.is_alive()

    def get_daemon_info_rpc(self) -> dict:
        """
        Get daemon information for testing/debugging.

        Returns:
            dict: {
                "daemon_started": bool,
                "daemon_alive": bool,
                "daemon_ident": int (thread ID),
                "server_pid": int (process ID),
                "daemon_is_thread": bool
            }
        """
        import os
        import threading

        info = {
            "daemon_started": self._daemon_started,
            "daemon_alive": self._daemon_thread.is_alive() if self._daemon_thread else False,
            "daemon_ident": self._daemon_thread.ident if self._daemon_thread else None,
            "server_pid": os.getpid(),
            "daemon_is_thread": isinstance(self._daemon_thread, threading.Thread) if self._daemon_thread else None
        }
        return info

    def get_daemon_status_rpc(self) -> dict:
        """Get daemon status via RPC."""
        watchdog_status = self._watchdog.get_status()
        return {
            "running": self._daemon_started,
            **watchdog_status
        }

    # ==================== Shutdown ====================

    def shutdown(self):
        """Gracefully shutdown server and daemon."""
        bt.logging.info(f"{self.service_name} shutting down...")
        # Set local shutdown flag FIRST to stop daemon loops from making RPC calls
        # This prevents KeyError when CommonDataServer is shutdown before other servers
        self._local_shutdown = True

        # Signal shutdown via ShutdownCoordinator (new pattern)
        # This is safe to call multiple times
        ShutdownCoordinator.signal_shutdown(f"{self.service_name} shutdown requested")

        self.stop_daemon()
        self._watchdog.stop()
        self.stop_rpc_server()
        self.stop_server_process()
        # Unregister from instance tracking
        ServerRegistry.unregister(self)
        bt.logging.info(f"{self.service_name} shutdown complete")

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_rpc_server') and self._rpc_server:
            try:
                self.stop_rpc_server()
            except Exception:
                pass
        if hasattr(self, '_server_process') and self._server_process:
            try:
                self.stop_server_process()
            except Exception:
                pass

    @classmethod
    def entry_point_start_server(cls, **kwargs):
        """
        Entry point for server RPC process.

        Creates server instance in-process. Constructor never spawns, so no recursion risk.

        Args:
            **kwargs: Server constructor parameters plus:
                server_ready: Optional Event to signal when ready (auto-removed)
                health_check_interval_s: Ignored (ServerProcessHandle parameter, auto-removed)
                enable_auto_restart: Ignored (ServerProcessHandle parameter, auto-removed)
        """
        assert cls.service_name, f"{cls.__name__} must set service_name class attribute"
        assert cls.service_port, f"{cls.__name__} must set service_port class attribute"

        # Set process title for monitoring
        setproctitle(f"vali_{cls.service_name}")

        # Extract and remove ServerProcessHandle-specific parameters
        server_ready = kwargs.pop('server_ready', None)
        kwargs.pop('health_check_interval_s', None)
        kwargs.pop('enable_auto_restart', None)

        # Add required server parameters
        kwargs['start_server'] = True
        kwargs['connection_mode'] = RPCConnectionMode.RPC

        # Filter kwargs to only include parameters the server's __init__ accepts
        # This allows spawn_process() to pass standard parameters while servers
        # can have different constructor signatures
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}

        # Keep only kwargs that the server accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        # Log if we're filtering out any parameters (for debugging)
        filtered_out = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out:
            bt.logging.debug(
                f"[{cls.service_name}] Filtered out unsupported parameters: {filtered_out}"
            )

        # Create server in-process (constructor never spawns, so no recursion)
        server_instance = cls(**filtered_kwargs)

        bt.logging.success(f"[SERVER] {cls.service_name} ready on port {cls.service_port}")

        if server_ready:
            server_ready.set()

        # Block until shutdown (uses ShutdownCoordinator)
        while not ShutdownCoordinator.is_shutdown():
            time.sleep(1)

        # Graceful shutdown
        server_instance.shutdown()
        bt.logging.info(f"{cls.service_name} process exiting")


    @classmethod
    def spawn_process(
        cls,
        slack_notifier=None,
        start_daemon=False,
        is_backtesting=False,
        running_unit_tests=False,
        health_check_interval_s: float = 30.0,
        enable_auto_restart: bool = True,
        wait_for_ready: bool = True,
        ready_timeout: float = 30.0,
        **server_kwargs
    ) -> 'ServerProcessHandle':
        """
        Spawn server in separate process with automatic readiness waiting.

        By default, this method blocks until the server is ready to accept connections.
        This prevents "Connection refused" errors when clients connect immediately.

        Args:
            slack_notifier: Optional SlackNotifier for alerts
            start_daemon: Whether to start daemon immediately in spawned process
            is_backtesting: Whether running in backtesting mode
            running_unit_tests: Whether running in test mode
            health_check_interval_s: Seconds between health checks (default: 30.0)
            enable_auto_restart: Whether to auto-restart if process dies (default: True)
            wait_for_ready: Whether to wait for server to be ready before returning (default: True)
            ready_timeout: Seconds to wait for server readiness (default: 30.0)
            **server_kwargs: Additional server-specific constructor parameters
                            (e.g., secrets for LivePriceFetcherServer, market_order_manager for LimitOrderServer)

        Returns:
            ServerProcessHandle: Handle for managing the spawned process

        Example:
            # Simple usage (blocks until server is ready)
            handle = ChallengePeriodServer.spawn_process()
            client = ChallengePeriodClient()  # No connection errors!

            # With server-specific parameters
            handle = LivePriceFetcherServer.spawn_process(secrets=secrets, disable_ws=True)
            client = LivePriceFetcherClient()

            # Async spawning (don't wait for ready)
            handle = ChallengePeriodServer.spawn_process(wait_for_ready=False)
            # ... do other work ...
            # Eventually create client when you know server is ready
        """

        entry_kwargs = {
            'slack_notifier': slack_notifier,
            'start_daemon': start_daemon,
            'is_backtesting': is_backtesting,
            'running_unit_tests': running_unit_tests,
            'health_check_interval_s': health_check_interval_s,
            'enable_auto_restart': enable_auto_restart,
            **server_kwargs  # Pass through server-specific parameters
        }

        if not cls.service_name:
            raise Exception('No service name provided')
        if not cls.service_port:
            raise Exception('No service port provided')

        # Detailed timing breakdown to track spawn performance
        start_time = time.time()

        # Always create server_ready event internally for clean API
        t0 = time.time()
        server_ready = Event()
        entry_kwargs['server_ready'] = server_ready
        t1 = time.time()

        # Create and start the process
        process = Process(
            target=cls.entry_point_start_server,
            kwargs=entry_kwargs,
            daemon=True
        )
        t2 = time.time()
        process.start()
        t3 = time.time()

        # Calculate timing breakdown
        elapsed_ms = (t3 - start_time) * 1000
        event_ms = (t1 - t0) * 1000
        create_ms = (t2 - t1) * 1000
        start_ms = (t3 - t2) * 1000

        bt.logging.success(
            f"{cls.service_name} process spawned (PID: {process.pid}) ({elapsed_ms:.0f}ms) "
            f"[event={event_ms:.0f}ms, create={create_ms:.0f}ms, start={start_ms:.0f}ms]"
        )

        # Wait for server to be ready (unless wait_for_ready=False)
        if wait_for_ready:
            t4 = time.time()
            if server_ready.wait(timeout=ready_timeout):
                t5 = time.time()
                ready_ms = (t5 - t4) * 1000
                total_ms = (t5 - start_time) * 1000
                bt.logging.success(
                    f"{cls.service_name} server ready ({total_ms:.0f}ms) [ready={ready_ms:.0f}ms]"
                )
            else:
                # Check if process died during startup
                if not process.is_alive():
                    exit_code = process.exitcode
                    error_msg = (
                        f"❌ {cls.service_name} process died during startup!\n"
                        f"Exit code: {exit_code}\n"
                        f"Common causes:\n"
                        f"  - Stale shutdown flag (should be fixed by reset_on_attach=True)\n"
                        f"  - Missing dependencies or configuration\n"
                        f"  - Port already in use\n"
                        f"  - Initialization error in server __init__"
                    )
                    bt.logging.error(error_msg)
                    if slack_notifier:
                        slack_notifier.send_message(error_msg, level="error")
                    raise RuntimeError(
                        f"{cls.service_name} process died during startup (exit code: {exit_code})"
                    )
                else:
                    bt.logging.warning(
                        f"{cls.service_name} server may not be fully ready after {ready_timeout}s timeout. "
                        f"Process is alive but didn't signal ready."
                    )

        # Create and return handle with health monitoring
        handle = ServerProcessHandle(
            process=process,
            entry_point=cls.entry_point_start_server,
            entry_kwargs=entry_kwargs,
            slack_notifier=slack_notifier,
            health_check_interval_s=health_check_interval_s,
            enable_auto_restart=enable_auto_restart,
            service_name=cls.service_name
        )

        return handle

    # ==================== Static Helpers for Subprocess Servers ====================

    @staticmethod
    def serve_rpc(
        server_instance,
        service_name: str,
        address: tuple,
        authkey: bytes,
        server_ready=None
    ):
        """
        Helper to serve an RPC server instance (for subprocess-based servers).

        This is a convenience method for starting servers in separate processes.
        Use this when the server runs in its own process via multiprocessing.Process.

        Args:
            server_instance: The server object to expose via RPC
            service_name: Name to register the service under
            address: (host, port) tuple for the server
            authkey: Authentication key for RPC connections
            server_ready: Optional Event to signal when server is ready

        Example usage in a separate process entry point:

            def start_my_server(address, authkey, server_ready):
                from setproctitle import setproctitle
                setproctitle("vali_MyServerProcess")

                server = MyServer(...)
                RPCServerBase.serve_rpc(
                    server_instance=server,
                    service_name="MyService",
                    address=address,
                    authkey=authkey,
                    server_ready=server_ready
                )

            # Start via multiprocessing
            process = Process(target=start_my_server, args=(address, authkey, server_ready))
            process.start()
        """
        class ServiceManager(BaseManager):
            pass

        # Register the service with the manager
        ServiceManager.register(service_name, callable=lambda: server_instance)

        # Create manager and get server
        manager = ServiceManager(address=address, authkey=authkey)
        server = manager.get_server()

        bt.logging.success(f"{service_name} server ready on {address}")

        # Signal that server is ready
        if server_ready:
            server_ready.set()
            bt.logging.debug(f"{service_name} readiness event set")

        # Start serving (blocks forever)
        server.serve_forever()
