# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
RPC Client Base Class - Unified lightweight client for connecting to RPC servers.

This module provides a base class for RPC clients that:
- Connect to existing RPC servers (no server ownership)
- Can be created in any process
- Support LOCAL mode via set_direct_server() for in-process testing
- Provide generic call() method for dynamic RPC calls
- Pickle support for subprocess handoff

Example usage:

    class MyServiceClient(RPCClientBase):
        def __init__(self, port=None, connection_mode=RPCConnectionMode.RPC):
            super().__init__(
                service_name="MyService",
                port=port or ValiConfig.RPC_MYSERVICE_PORT,
                connection_mode=connection_mode
            )

        # Typed method wrappers (preferred for IDE support)
        def some_method(self, arg) -> str:
            return self._server.some_method_rpc(arg)

        def another_method(self, x, y) -> int:
            return self._server.another_method_rpc(x, y)

Generic call() usage (for dynamic method names):

    client = MyServiceClient()
    result = client.call("some_method_rpc", arg1, kwarg1=value)

LOCAL mode usage (bypass RPC for testing):

    # In tests, bypass RPC and use direct server reference
    client = MyServiceClient(connection_mode=RPCConnectionMode.LOCAL)
    client.set_direct_server(server_instance)
    # Now client._server returns server_instance directly
"""
import os
import time
import socket
import threading
import bittensor as bt
from multiprocessing.managers import BaseManager
from typing import Optional, Any, Dict
from abc import abstractmethod

from vali_objects.vali_config import ValiConfig, RPCConnectionMode


# Store original socket class for restoration
_original_socket = socket.socket
_socket_patched = False


def _patch_socket_for_nodelay():
    """
    Monkey-patch socket.socket to enable TCP_NODELAY on all TCP sockets.

    This is necessary because multiprocessing.managers creates sockets dynamically
    for each RPC call rather than keeping persistent connections. We can't access
    these sockets directly, so we patch socket creation at the source.

    Only patches once (thread-safe via class-level flag check).
    """
    global _socket_patched

    if _socket_patched:
        return

    class TCPNodeDelaySocket(socket.socket):
        """Socket subclass that automatically enables TCP_NODELAY for TCP sockets."""

        def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
            super().__init__(family, type, proto, fileno)

            # Enable TCP_NODELAY for TCP sockets (eliminates Nagle's algorithm delays)
            if family == socket.AF_INET and type == socket.SOCK_STREAM:
                try:
                    self.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except (OSError, AttributeError):
                    # Socket might not support TCP_NODELAY (e.g., not connected yet)
                    pass

        def connect(self, address):
            """Override connect to ensure TCP_NODELAY is set after connection."""
            super().connect(address)
            # Ensure TCP_NODELAY is set (in case __init__ was too early)
            try:
                self.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except (OSError, AttributeError):
                pass

    # Replace socket.socket globally
    socket.socket = TCPNodeDelaySocket
    _socket_patched = True
    bt.logging.debug("Socket patched to enable TCP_NODELAY for all RPC connections")


class RPCClientBase:
    """
    Lightweight RPC client base - connects to existing server.

    Can be created in ANY process. No server ownership.
    Supports pickle for subprocess handoff.

    Features:
    - Lazy connection on first use (no blocking during __init__)
    - Automatic connection with retries
    - Test mode support via set_direct_server()
    - Generic call() method for dynamic RPC calls
    - Pickle support for subprocess handoff
    - Automatic instance tracking for test cleanup
    - Sequential instance IDs per service (for debugging/monitoring)

    Lazy connection eliminates server startup ordering concerns - clients can be
    created before their target servers are running. Connection happens automatically
    on first method call.

    Subclasses just need to:
    1. Call super().__init__ with service_name and port
    2. Implement typed methods that delegate to self._server
    """

    # Class-level registry of all active client instances (for test cleanup)
    _active_instances: list = []
    _registry_lock = threading.Lock()

    # Track instance counts per service name for sequential IDs
    _instance_counts: Dict[str, int] = {}

    @classmethod
    def disconnect_all(cls, reset_counts: bool = True) -> None:
        """
        Disconnect all active client instances.

        Call this in test tearDown (before RPCServerBase.shutdown_all()) to ensure
        all clients are disconnected before servers are shut down. This prevents
        clients from holding connections that block server shutdown.

        Args:
            reset_counts: If True (default), reset instance counts so IDs start fresh.
                         Set to False if you want cumulative counts across test runs.

        Example:
            def tearDown(self):
                RPCClientBase.disconnect_all()
                RPCServerBase.shutdown_all()
        """
        with cls._registry_lock:
            instances = list(cls._active_instances)
            cls._active_instances.clear()
            if reset_counts:
                cls._instance_counts.clear()

        for instance in instances:
            try:
                instance.disconnect()
            except Exception as e:
                bt.logging.trace(f"Error disconnecting {instance.service_name}Client: {e}")

        bt.logging.debug(f"Disconnected {len(instances)} RPC client instances")

    @classmethod
    def get_instance_counts(cls) -> Dict[str, int]:
        """
        Get current instance counts per service name.

        Useful for debugging/monitoring to see how many clients of each type exist.

        Returns:
            Dict mapping service_name -> total instances created
        """
        with cls._registry_lock:
            return dict(cls._instance_counts)

    @classmethod
    def _register_instance(cls, instance: 'RPCClientBase') -> int:
        """
        Register a new client instance for tracking.

        Returns:
            int: The sequential instance ID for this service
        """
        with cls._registry_lock:
            cls._active_instances.append(instance)

            # Assign sequential ID per service name
            service_name = instance.service_name
            if service_name not in cls._instance_counts:
                cls._instance_counts[service_name] = 0
            cls._instance_counts[service_name] += 1
            instance_id = cls._instance_counts[service_name]

        bt.logging.debug(
            f"{service_name}Client #{instance_id} registered (port={instance.port})"
        )
        return instance_id

    @classmethod
    def _unregister_instance(cls, instance: 'RPCClientBase') -> None:
        """Unregister a client instance."""
        with cls._registry_lock:
            if instance in cls._active_instances:
                cls._active_instances.remove(instance)

    def __init__(
        self,
        service_name: str,
        port: int,
        max_retries: int = 5,
        retry_delay_s: float = 1.0,
        connect_immediately: bool = False,
        warning_threshold: int = 2,
        local_cache_refresh_period_ms: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize RPC client.

        Args:
            service_name: Name of the RPC service to connect to
            port: Port number of the RPC server
            max_retries: Maximum connection retry attempts (default: 60)
            retry_delay_s: Delay between retries in seconds (default: 1.0)
            connect_immediately: If True, connect in __init__. If False (default), connect
                lazily on first method call. Lazy connection is preferred to avoid blocking
                during initialization and eliminate server startup ordering concerns.
            warning_threshold: Number of retries before logging warnings (default: 30)
            local_cache_refresh_period_ms: If not None, spawn a daemon thread that refreshes
                a local cache at this interval. Subclasses must implement populate_cache().
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
                Default: RPC
        """
        self.connection_mode = connection_mode
        self.service_name = service_name
        self.port = port
        # Use 127.0.0.1 instead of 'localhost' to avoid IPv6/IPv4 fallback delays
        # 'localhost' can trigger ~170ms delay due to IPv6 ::1 timeout then IPv4 127.0.0.1 fallback
        self._address = ('127.0.0.1', port)
        self._authkey = ValiConfig.get_rpc_authkey(service_name, port)
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._warning_threshold = warning_threshold

        # Connection state
        self._manager: Optional[BaseManager] = None
        self._proxy = None
        self._connected = False

        # Direct server reference (used in LOCAL mode)
        self._direct_server = None

        # Local cache state
        self._local_cache_refresh_period_ms = local_cache_refresh_period_ms
        self._local_cache: Dict[str, Any] = {}
        self._local_cache_lock = threading.Lock()
        self._cache_refresh_thread: Optional[threading.Thread] = None
        self._cache_refresh_shutdown = threading.Event()

        # Register instance for tracking (enables disconnect_all() for test cleanup)
        # Store sequential ID for debugging/monitoring
        # IMPORTANT: Must be set BEFORE connect() since connect() uses it in logging
        self._instance_id = RPCClientBase._register_instance(self)

        # Connect if requested and in RPC mode
        if connect_immediately and self.connection_mode == RPCConnectionMode.RPC:
            self.connect()

        # Start local cache refresh daemon if configured and in RPC mode
        if local_cache_refresh_period_ms is not None and self.connection_mode == RPCConnectionMode.RPC:
            self._start_cache_refresh_daemon()

    @property
    def _server(self):
        """
        Returns the server interface (direct or proxy).

        In LOCAL mode: returns _direct_server (no RPC overhead)
        In RPC mode: returns _proxy (RPC connection)

        Connects lazily on first access if not already connected.
        This eliminates server startup ordering concerns - clients can be
        created before their target servers are running.

        Subclasses should use self._server to access RPC methods:
            return self._server.some_method_rpc(arg)
        """
        if self._direct_server is not None:
            return self._direct_server

        # Lazy connection: connect on first use if not already connected (RPC mode only)
        if self._proxy is None and not self._connected and self.connection_mode == RPCConnectionMode.RPC:
            self.connect()

        return self._proxy

    def connect(self, max_retries: int = None, retry_delay: float = None) -> bool:
        """
        Connect to the RPC server with retries.

        Args:
            max_retries: Override default max retries (optional)
            retry_delay: Override default retry delay (optional)

        Returns:
            bool: True if connected successfully

        Raises:
            ConnectionError: If connection fails after all retries
        """
        if self._connected and self._proxy is not None:
            return True

        if self._direct_server is not None:
            # Test mode - no connection needed
            return True

        max_retries = max_retries or self._max_retries
        retry_delay = retry_delay or self._retry_delay_s

        # Create client manager class
        class ClientManager(BaseManager):
            pass

        # Register the service type
        ClientManager.register(self.service_name)

        # Patch socket.socket to enable TCP_NODELAY (only once globally)
        _patch_socket_for_nodelay()

        # Retry connection with backoff
        last_error = None
        start_time = time.time()
        for attempt in range(1, max_retries + 1):
            try:
                # Detailed timing breakdown to identify bottleneck
                t0 = time.time()
                manager = ClientManager(address=self._address, authkey=self._authkey)
                t1 = time.time()
                manager.connect()
                t2 = time.time()

                # Get the proxy object (TCP_NODELAY now enabled via socket patch)
                self._proxy = getattr(manager, self.service_name)()
                t3 = time.time()
                self._manager = manager
                self._connected = True

                # Log success with detailed timing breakdown
                elapsed_ms = (t3 - start_time) * 1000
                manager_create_ms = (t1 - t0) * 1000
                connect_ms = (t2 - t1) * 1000
                proxy_ms = (t3 - t2) * 1000

                if attempt > 1:
                    bt.logging.success(
                        f"{self.service_name}Client #{self._instance_id} connected to server at {self._address} "
                        f"after {attempt} attempts ({elapsed_ms:.0f}ms) "
                        f"[create={manager_create_ms:.0f}ms, connect={connect_ms:.0f}ms, proxy={proxy_ms:.0f}ms]"
                    )
                else:
                    bt.logging.success(
                        f"{self.service_name}Client #{self._instance_id} connected to server at {self._address} ({elapsed_ms:.0f}ms) "
                        f"[create={manager_create_ms:.0f}ms, connect={connect_ms:.0f}ms, proxy={proxy_ms:.0f}ms]"
                    )
                return True

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Log based on threshold to reduce noise during startup
                    if attempt >= self._warning_threshold:
                        bt.logging.warning(
                            f"{self.service_name}Client connection failed (attempt {attempt}/"
                            f"{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                    else:
                        bt.logging.trace(
                            f"{self.service_name}Client connection failed (attempt {attempt}/"
                            f"{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                    time.sleep(retry_delay)
                else:
                    bt.logging.error(
                        f"{self.service_name}Client failed to connect after "
                        f"{max_retries} attempts: {e}"
                    )

        raise ConnectionError(
            f"Failed to connect to {self.service_name} at {self._address}: {last_error}"
        )

    def call(self, method_name: str, *args, **kwargs) -> Any:
        """
        Generic method to call any RPC method by name.

        Args:
            method_name: Name of the RPC method to call (e.g., "some_method_rpc")
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            The result from the RPC call

        Raises:
            RuntimeError: If not connected
            AttributeError: If method doesn't exist on remote service

        Example:
            result = client.call("get_data_rpc", key="some_key")
        """
        if self._server is None:
            raise RuntimeError(f"Not connected to {self.service_name}")

        try:
            method = getattr(self._server, method_name)
            result = method(*args, **kwargs)

            bt.logging.trace(
                f"{self.service_name}Client.{method_name}(*{args}, **{kwargs}) -> {type(result)}"
            )

            return result

        except AttributeError as e:
            bt.logging.error(
                f"{self.service_name}Client method '{method_name}' not found: {e}"
            )
            raise
        except Exception as e:
            bt.logging.error(
                f"{self.service_name}Client RPC call failed: {method_name}: {e}"
            )
            raise

    def is_connected(self) -> bool:
        """Check if client is connected (or has direct server)."""
        if self._direct_server is not None:
            return True
        return self._connected and self._proxy is not None

    def health_check(self) -> dict:
        """
        Get health status from server.

        All RPC servers inherit from RPCServerBase which provides health_check_rpc().
        This is a standard method available on all servers.

        Returns:
            dict: Health status with 'status', 'service', 'timestamp_ms' and service-specific info
        """
        return self._server.health_check_rpc()

    def start_daemon(self) -> bool:
        """
        Start the daemon thread remotely via RPC.

        All RPC servers inherit from RPCServerBase which provides start_daemon_rpc().
        This is a standard method available on all servers.

        Returns:
            bool: True if daemon was started, False if already running
        """
        return self._server.start_daemon_rpc()

    def disconnect(self):
        """Disconnect from the server."""
        start_time = time.time()

        # Stop cache refresh daemon if running
        if self._cache_refresh_thread is not None:
            self._cache_refresh_shutdown.set()
            self._cache_refresh_thread.join(timeout=2.0)
            self._cache_refresh_thread = None

        # Clean up manager connection (prevents semaphore leaks)
        # BaseManager creates IPC resources that need explicit cleanup
        if self._manager is not None:
            try:
                # Shutdown the manager's connection to the server
                # This releases semaphores and shared memory used for IPC
                if hasattr(self._manager, '_state'):
                    # Manager has internal state tracking the connection
                    # Setting to None allows garbage collection of resources
                    self._manager._state = None
                if hasattr(self._manager, '_Client'):
                    # Close the connection to the server
                    # This prevents lingering socket connections
                    try:
                        if self._manager._Client is not None:
                            self._manager._Client.close()
                    except Exception:
                        pass
            except Exception as e:
                bt.logging.trace(f"{self.service_name}Client error during manager cleanup: {e}")

        self._manager = None
        self._proxy = None
        self._connected = False
        self._direct_server = None

        # Unregister from instance tracking
        RPCClientBase._unregister_instance(self)
        elapsed_ms = (time.time() - start_time) * 1000
        bt.logging.debug(f"{self.service_name}Client disconnected ({elapsed_ms:.0f}ms)")

    # ==================== Local Cache Support ====================

    def _start_cache_refresh_daemon(self) -> None:
        """Start the background cache refresh daemon thread."""
        if self._cache_refresh_thread is not None and self._cache_refresh_thread.is_alive():
            return  # Already running

        self._cache_refresh_shutdown.clear()
        self._cache_refresh_thread = threading.Thread(
            target=self._cache_refresh_loop,
            daemon=True,
            name=f"{self.service_name}CacheRefresh"
        )
        self._cache_refresh_thread.start()
        bt.logging.info(
            f"[{self.service_name}] Local cache refresh daemon started "
            f"(interval: {self._local_cache_refresh_period_ms}ms)"
        )

    def _cache_refresh_loop(self) -> None:
        """
        Background daemon that periodically refreshes the local cache.

        Calls populate_cache() at the configured interval to pull fresh data
        from the server and store it locally for fast access.
        """
        refresh_interval_s = self._local_cache_refresh_period_ms / 1000.0

        while not self._cache_refresh_shutdown.is_set():
            try:
                # Call subclass-specific populate_cache implementation
                start_time = time.perf_counter()
                new_cache = self.populate_cache()
                refresh_ms = (time.perf_counter() - start_time) * 1000

                # Atomic cache update under lock
                with self._local_cache_lock:
                    self._local_cache = new_cache

                bt.logging.debug(
                    f"[{self.service_name}] Local cache refreshed in {refresh_ms:.2f}ms "
                    f"({len(new_cache) if isinstance(new_cache, dict) else 'N/A'} entries)"
                )

            except Exception as e:
                bt.logging.error(f"[{self.service_name}] Error refreshing local cache: {e}")

            # Wait for next refresh cycle (interruptible)
            self._cache_refresh_shutdown.wait(timeout=refresh_interval_s)

        bt.logging.info(f"[{self.service_name}] Local cache refresh daemon stopped")

    def populate_cache(self) -> Dict[str, Any]:
        """
        Populate the local cache with data from the server.

        Subclasses that use local_cache_refresh_period_ms MUST override this method
        to fetch and return the cache data structure.

        Returns:
            Dict containing the cache data. Structure is subclass-specific.

        Example implementation:
            def populate_cache(self) -> Dict[str, Any]:
                # Fetch data from server via RPC
                eliminations = self._server.get_eliminations_dict_rpc()
                departed = self._server.get_departed_hotkeys_rpc()
                return {
                    "eliminations": eliminations,
                    "departed_hotkeys": departed
                }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement populate_cache() "
            f"when using local_cache_refresh_period_ms"
        )

    def get_local_cache(self) -> Dict[str, Any]:
        """
        Get a thread-safe copy of the local cache.

        Returns:
            Dict containing the cached data (copy for thread safety)
        """
        with self._local_cache_lock:
            return dict(self._local_cache)

    def get_from_local_cache(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the local cache by key.

        Args:
            key: The key to look up in the cache
            default: Default value if key not found

        Returns:
            The cached value or default
        """
        with self._local_cache_lock:
            return self._local_cache.get(key, default)

    # ==================== Pickle Support for Subprocess Handoff ====================

    def __getstate__(self):
        """
        Prepare object for pickling (when passed to child processes).

        The unpickled object will reconnect to the existing RPC server.

        Subclasses can override _prepare_state_for_pickle() to add service-specific
        attributes that need special handling.
        """
        bt.logging.debug(
            f"[{self.service_name}_PICKLE] __getstate__ called in PID {os.getpid()}"
        )

        state = self.__dict__.copy()

        # Mark as needing reconnection after unpickle
        state['_needs_reconnect'] = True

        # Don't pickle proxy/manager objects (they're not picklable)
        state['_manager'] = None
        state['_proxy'] = None

        # Don't pickle cache-related unpicklable objects
        state['_local_cache_lock'] = None
        state['_cache_refresh_thread'] = None
        state['_cache_refresh_shutdown'] = None

        # Apply subclass-specific excludes/transforms
        self._prepare_state_for_pickle(state)

        return state

    def _prepare_state_for_pickle(self, state: dict) -> None:
        """
        Hook for subclasses to customize pickle state preparation.

        Override this method to handle service-specific unpicklable attributes.
        Common patterns:
        - Set locks to None: state['_my_lock'] = None
        - Convert defaultdicts to dicts: state['my_dict'] = dict(self.my_dict)

        Args:
            state: The state dict being prepared for pickling (modify in place)
        """
        pass  # Base implementation does nothing

    def __setstate__(self, state):
        """
        Restore object after unpickling (in child process).

        Automatically reconnects to existing RPC server.

        Subclasses can override _restore_unpicklable_state() to restore
        service-specific attributes that couldn't be pickled.
        """
        bt.logging.debug(
            f"[{state.get('service_name', 'RPC')}_UNPICKLE] __setstate__ called in PID {os.getpid()}"
        )

        self.__dict__.update(state)

        # Restore subclass-specific unpicklable state
        self._restore_unpicklable_state(state)

        # In LOCAL mode, nothing to reconnect
        if self.connection_mode == RPCConnectionMode.LOCAL:
            bt.logging.debug(f"[{self.service_name}_UNPICKLE] LOCAL mode - no reconnection needed")
            return

        # Reconnect to existing RPC server (RPC mode)
        if state.get('_needs_reconnect', False):
            bt.logging.debug(
                f"[{self.service_name}_UNPICKLE] Reconnecting to RPC server on port {self.port}"
            )

            # Use faster retry settings for unpickle reconnection
            original_retries = self._max_retries
            original_delay = self._retry_delay_s
            self._max_retries = 5  # Fewer retries - server should be running
            self._retry_delay_s = 0.5

            try:
                self.connect()
                bt.logging.success(
                    f"[{self.service_name}_UNPICKLE] Reconnected to RPC server at {self._address}"
                )
            except Exception as e:
                # Always fail loudly to catch architectural issues where clients are pickled
                import traceback
                stack_trace = ''.join(traceback.format_stack())
                raise RuntimeError(
                    f"[{self.service_name}_UNPICKLE] Failed to reconnect after unpickle: {e}\n"
                    f"This indicates clients are being pickled when they shouldn't be.\n"
                    f"Clients embedded in server managers should never leave their process.\n"
                    f"\nStack trace showing unpickle location:\n{stack_trace}"
                ) from e
            finally:
                self._max_retries = original_retries
                self._retry_delay_s = original_delay

    def _restore_unpicklable_state(self, state: dict) -> None:
        """
        Hook for subclasses to restore service-specific unpicklable state.

        Override this method to restore attributes that couldn't be pickled.
        Common patterns:
        - Recreate locks: self._my_lock = threading.Lock()

        Args:
            state: The state dict that was unpickled (for reference)
        """
        # Restore cache-related objects
        self._local_cache_lock = threading.Lock()
        self._cache_refresh_shutdown = threading.Event()
        self._cache_refresh_thread = None

        # Restart cache refresh daemon if it was configured and in RPC mode
        if (self._local_cache_refresh_period_ms is not None
                and self.connection_mode == RPCConnectionMode.RPC):
            self._start_cache_refresh_daemon()

    @property
    def instance_id(self) -> int:
        """Get the sequential instance ID for this client."""
        return getattr(self, '_instance_id', 0)

    def __repr__(self):
        mode = self.connection_mode.name
        instance_id = self.instance_id
        if self.connection_mode == RPCConnectionMode.LOCAL:
            return f"{self.__class__.__name__}(#{instance_id}, port={self.port}, mode={mode})"
        status = "connected" if self._connected else "disconnected"
        return f"{self.__class__.__name__}(#{instance_id}, port={self.port}, mode={mode}, {status})"
