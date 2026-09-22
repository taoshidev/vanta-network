# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from multiprocessing.managers import BaseManager
from typing import Optional, Any, Dict

from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.log import logger
from shared_objects.error_utils import ErrorUtils


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
    logger.debug("Socket patched to enable TCP_NODELAY for all RPC connections")


class RPCCallTimeoutError(TimeoutError):
    """An RPC call exceeded the client's per-call timeout.

    The backing service is alive-but-unresponsive (or mid-GC / mid-heavy-pass);
    the caller has been freed instead of blocking forever. Callers should treat
    this as a retryable service-unavailable condition.

    Deliberately short-circuited by _invoke_rpc BEFORE its transient-error check:
    this subclasses TimeoutError (an OSError), so ErrorUtils.is_transient_rpc_error()
    would otherwise read it as a dead transport and run the reconnect+retry cycle.
    A timeout means the opposite — the connection is fine and the SERVER IS STILL
    EXECUTING the call — so reconnecting achieves nothing, while re-running the
    method would pile more work onto an already-wedged service and risk
    double-applying a non-idempotent write.
    """

    def __init__(self, service_name: str, method_name: str, timeout_s: float):
        self.service_name = service_name
        self.method_name = method_name
        self.timeout_s = timeout_s
        super().__init__(
            f"{service_name}.{method_name} timed out after {timeout_s}s "
            f"(service alive but unresponsive)"
        )


class _ResilientRPCProxy:
    """
    Transparent stand-in for the raw multiprocessing BaseManager proxy.

    Every attribute access returns a callable that routes the RPC method through the owning
    client's _invoke_rpc(), so that ALL call sites get uniform behavior — both the typed method
    wrappers (self._server.foo_rpc(...)) and the generic call() path. Before this wrapper, only
    call() self-healed, and the ~20 clients that call the proxy directly (elimination,
    position_manager, metagraph, ...) would cache a dead proxy forever after a state-server
    restart and raise on every subsequent call.

    Routing through _invoke_rpc() is what gives every call BOTH protections, for the two
    distinct ways a backing service fails:
      - server GONE (bounced during a deploy): reconnect-on-server-bounce, with the transport
        probe / circuit breaker that keep a business error from being retried.
      - server WEDGED (alive but unresponsive): a per-call timeout. The raw proxy call blocks in
        conn.recv() with NO timeout, so a wedged service pinned the calling thread forever — in
        the REST process that permanently drained Waitress's 32-thread pool and took the whole
        API down until restart. _invoke_rpc runs each proxy call on the client's small executor
        and waits at most rpc_call_timeout_s, raising RPCCallTimeoutError instead of hanging
        (see _call_proxy_method / _invoke_bounded).

    A timed-out worker thread stays parked on its own thread-local connection until the service
    recovers; the caller is freed immediately. When every worker is parked, new calls queue
    behind them and time out promptly — a natural fail-fast while the service is wedged that
    self-heals on recovery. The timeout is CLIENT-SIDE ONLY: the server keeps executing the
    dispatched method to completion. Design RPC methods to be idempotent/retry-safe, and give
    known-long operations a larger per-client rpc_call_timeout_s rather than relying on the
    default.

    __getattr__ only fires for attributes not found normally, so the real `_client` attribute
    is never routed through the RPC path.
    """

    def __init__(self, client: 'RPCClientBase'):
        self._client = client

    def __getattr__(self, name: str):
        # Never route Python special/dunder lookups (copy, pickle, len, ... protocols) into an
        # RPC call — RPC methods never look like __x__. Let them fail as normal missing attrs.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)

        client = self.__dict__['_client']

        def _method(*args, **kwargs):
            return client._invoke_rpc(name, args, kwargs)

        return _method


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

    # Bounded reconnect for the self-heal path in _invoke_rpc: up to _RECONNECT_MAX_RETRIES cycles
    # of (rebuild connection, retry the call), polling every _RECONNECT_RETRY_DELAY_S. Poll FAST and
    # OFTEN rather than sleeping long: a recovered call returns the instant the restarted server is
    # ready (~1s in practice), and the in-call budget (~cycles x delay) stays small so a bounce
    # doesn't tie up executor threads for tens of seconds — longer outages hand off to caller-level
    # retry (order placer / daemon loop / the next call's default connect). The settle sleep is kept
    # OUTSIDE _conn_lock, and each cycle's connect() is a single fast attempt (max_retries=1), so the
    # lock is never held across a sleep. Reconnecting has no side effects, so retrying the CONNECT is
    # always safe; only the method re-run (up to _RECONNECT_MAX_RETRIES times) carries at-least-once
    # semantics (safe here: reads dominate, order writes are UUID-deduped, sync-epoch is
    # staleness-tolerant). Callers needing fail-fast (probes) pass retry=False to _invoke_rpc.
    _RECONNECT_MAX_RETRIES = 5
    _RECONNECT_RETRY_DELAY_S = 1.0
    # Circuit breaker: once a full self-heal cycle (or a lazy connect) has failed, further calls
    # within this window fail FAST (single attempt, quick connect, no settle sleeps) instead of
    # each paying the full multi-cycle penalty. Without it, a state-tier restart turned every
    # call from every one of the 32 axon/waitress worker threads into ~4-8s of blocked thread —
    # saturating both pools within seconds and queueing even health checks. The first call after
    # the window (or any success) closes the breaker.
    _BACKOFF_WINDOW_S = 5.0

    # Class-level registry of all active client instances (for test cleanup)
    _active_instances: list = []
    _registry_lock = threading.Lock()

    # Track instance counts per service name for sequential IDs
    _instance_counts: Dict[str, int] = {}

    # Default per-call RPC timeout (seconds). Generous on purpose: the goal is
    # to convert an INFINITE hang into a bounded failure, not to police normal
    # latency. Override per client via rpc_call_timeout_s=, or set <= 0 to
    # disable bounding for a client that makes legitimately unbounded calls.
    RPC_CALL_TIMEOUT_S = 60.0
    # Executor size per client: enough to pump concurrent healthy calls from a
    # full Waitress pool without serializing them, small enough that parked
    # (timed-out) workers stay cheap. Note the deliberate failure semantics:
    # while a service is wedged, its parked workers consume slots and queued
    # calls time out promptly (fail-fast); healthy-service queuing is transient
    # (calls complete in ms). Tune upward for clients that legitimately carry
    # heavy concurrent fan-out; tune rpc_call_timeout_s (not workers) for
    # legitimately slow calls.
    RPC_EXECUTOR_WORKERS = 16

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
                logger.debug(f"Error disconnecting {instance.service_name}Client: {e}")

        logger.debug(f"Disconnected {len(instances)} RPC client instances")

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

        logger.debug(
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
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        rpc_call_timeout_s: float = None,
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
        # Monotonic generation, bumped on every successful connect(). The self-heal path captures
        # the generation before a call and only rebuilds if it is unchanged after a failure — this
        # de-dups reconnects when a burst of executor threads all fail on the same dead proxy, so
        # they reuse the first thread's rebuilt connection instead of tearing it down N times.
        self._connection_generation = 0
        # Serializes (re)connect/disconnect so that under concurrent load (this client is shared
        # across the axon's executor threads) a transient-error storm doesn't spawn many managers or
        # tear down a freshly-rebuilt connection. Reentrant: connect() may be entered via _server.
        self._conn_lock = threading.RLock()
        # Circuit breaker state: monotonic-ish wall-clock deadline until which calls fail fast
        # (see _BACKOFF_WINDOW_S). 0 = breaker closed. Read/written unlocked (float assignment is
        # atomic; a racy read only costs one extra fast/slow attempt).
        self._backoff_until = 0.0

        # Resilient proxy returned by the _server property in RPC mode. Bound to this client so
        # every method call routes through _invoke_rpc()'s reconnect-on-bounce logic.
        self._resilient_proxy = _ResilientRPCProxy(self)

        # Direct server reference (used in LOCAL mode)
        self._direct_server = None

        # Per-call timeout machinery (applied inside _invoke_rpc, see _call_proxy_method).
        # None -> class default; a value <= 0 disables bounding entirely and makes _server hand
        # back the raw proxy (see the _server property).
        self._rpc_call_timeout_s = (
            rpc_call_timeout_s if rpc_call_timeout_s is not None else self.RPC_CALL_TIMEOUT_S
        )
        self._rpc_executor: Optional[ThreadPoolExecutor] = None
        self._rpc_executor_lock = threading.Lock()
        self._rpc_timeout_log_ts: Dict[str, float] = {}

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
        Returns the server interface (direct or resilient proxy).

        In LOCAL mode: returns _direct_server (no RPC overhead, direct method calls).
        In RPC mode: returns the _ResilientRPCProxy wrapper — every method call on it routes
        through _invoke_rpc(), which lazily connects on first use, self-heals a poisoned proxy
        after a server restart, AND bounds the call at rpc_call_timeout_s so a wedged service
        cannot pin the caller forever. The wrapper is returned WITHOUT forcing a connection here
        so client construction stays non-blocking and free of server-startup ordering concerns.
        rpc_call_timeout_s <= 0 opts the client out of both and hands back the raw proxy.

        Subclasses use self._server to access RPC methods exactly as before:
            return self._server.some_method_rpc(arg)
        """
        if self._direct_server is not None:
            return self._direct_server

        if self.connection_mode == RPCConnectionMode.RPC:
            if self._rpc_call_timeout_s <= 0:
                # Opt-out escape hatch: the RAW proxy, i.e. neither the per-call timeout nor the
                # _invoke_rpc reconnect/self-heal wrapper. Reserved for a client that makes
                # legitimately unbounded calls (and for tests that want production proxy
                # semantics); no production client sets rpc_call_timeout_s <= 0 today.
                return self._ensure_proxy()
            return self._resilient_proxy

        # Non-RPC, non-direct (should not happen in practice) - return raw proxy.
        return self._proxy

    def _ensure_proxy(self):
        """
        Return a live RPC proxy, connecting lazily on first use.

        Kept separate from the _server property so the resilient wrapper can (re)fetch the
        current proxy without recursing back through the wrapper.

        This lazy connect is intentionally NOT serialized by _conn_lock (the self-heal path is).
        Two first-time callers can therefore race and each build a manager; the race is benign —
        both connect to the same live server, the last write to self._proxy wins, and the orphaned
        manager is GC'd. Locking here would add contention to the common already-connected path.
        """
        # Keyed on _proxy alone (not `and not self._connected`): a racing teardown must never
        # leave this returning None while _connected looks True — connect() itself no-ops when
        # already connected, so the extra call in a race is harmless.
        if self._proxy is None and self.connection_mode == RPCConnectionMode.RPC:
            self.connect()
        return self._proxy

    def _get_rpc_executor(self) -> ThreadPoolExecutor:
        """Lazily create the small executor that carries bounded RPC calls."""
        if self._rpc_executor is None:
            with self._rpc_executor_lock:
                if self._rpc_executor is None:
                    self._rpc_executor = ThreadPoolExecutor(
                        max_workers=self.RPC_EXECUTOR_WORKERS,
                        thread_name_prefix=f"rpc-{self.service_name}",
                    )
        return self._rpc_executor

    def _call_proxy_method(self, proxy, method_name: str, args, kwargs):
        """
        Invoke ONE method on the raw proxy, bounded by the per-call timeout when enabled.

        Single funnel for every raw-proxy invocation in _invoke_rpc (first attempt, each
        self-heal retry, and the transport probe), so no path can reach the unbounded
        conn.recv() that drained the REST worker pool. rpc_call_timeout_s <= 0 calls straight
        through (raw-proxy behavior).
        """
        method = getattr(proxy, method_name)
        if self._rpc_call_timeout_s <= 0:
            return method(*args, **kwargs)
        return self._invoke_bounded(method_name, method, args, kwargs)

    def _invoke_bounded(self, method_name: str, bound_method, args, kwargs):
        """Run one proxy method call with the per-call timeout (see _ResilientRPCProxy)."""
        timeout_s = self._rpc_call_timeout_s
        future = self._get_rpc_executor().submit(bound_method, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            # Python >= 3.11 aliases concurrent.futures.TimeoutError to the builtin
            # TimeoutError, so a TimeoutError raised by the SERVER'S own logic (e.g. a
            # position-lock timeout under contention, transported verbatim by
            # multiprocessing) lands in this handler too. Only a future that has NOT
            # completed is a real client-side wait timeout; a completed one carries the
            # call's own outcome and must propagate untouched — relabelling it would hide a
            # business error behind a bogus "exceeded Ns" and rob _invoke_rpc's transport
            # probe of the chance to classify it.
            if future.done():
                completed_error = future.exception(timeout=0)
                if completed_error is not None:
                    raise completed_error
                return future.result()
            # cancel() succeeds only for QUEUED calls (they never hit the wire);
            # a call already running keeps its worker parked until the service
            # recovers. Either way this caller is freed now.
            cancelled = future.cancel()
            self._log_rpc_timeout(method_name, timeout_s, cancelled)
            raise RPCCallTimeoutError(self.service_name, method_name, timeout_s) from None

    # Sustained outages produce one timeout per call; full-volume ERROR logging
    # would flood the logs with identical lines. Log ERROR at most once per
    # method per RPC_TIMEOUT_LOG_INTERVAL_S; the rest drop to DEBUG.
    RPC_TIMEOUT_LOG_INTERVAL_S = 30.0

    def _log_rpc_timeout(self, method_name: str, timeout_s: float, cancelled: bool) -> None:
        now = time.monotonic()
        detail = (
            f"{self.service_name}Client.{method_name} exceeded {timeout_s}s — freeing caller; "
            f"{'call was still queued (never dispatched)' if cancelled else 'worker remains parked until the service recovers'}"
        )
        with self._rpc_executor_lock:
            last = self._rpc_timeout_log_ts.get(method_name, 0.0)
            should_error = now - last >= self.RPC_TIMEOUT_LOG_INTERVAL_S
            if should_error:
                self._rpc_timeout_log_ts[method_name] = now
        if should_error:
            logger.error(detail)
        else:
            logger.debug(detail)

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
                # Bump so the self-heal path can tell a freshly-rebuilt connection from the dead
                # one a concurrent thread failed on (see _invoke_rpc / _connection_generation).
                self._connection_generation += 1

                # Log success with detailed timing breakdown
                elapsed_ms = (t3 - start_time) * 1000
                manager_create_ms = (t1 - t0) * 1000
                connect_ms = (t2 - t1) * 1000
                proxy_ms = (t3 - t2) * 1000

                if attempt > 1:
                    logger.info(
                        f"{self.service_name}Client #{self._instance_id} connected to server at {self._address} "
                        f"after {attempt} attempts ({elapsed_ms:.0f}ms) "
                        f"[create={manager_create_ms:.0f}ms, connect={connect_ms:.0f}ms, proxy={proxy_ms:.0f}ms]"
                    )
                else:
                    logger.info(
                        f"{self.service_name}Client #{self._instance_id} connected to server at {self._address} ({elapsed_ms:.0f}ms) "
                        f"[create={manager_create_ms:.0f}ms, connect={connect_ms:.0f}ms, proxy={proxy_ms:.0f}ms]"
                    )
                return True

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Log based on threshold to reduce noise during startup
                    if attempt >= self._warning_threshold:
                        logger.warning(
                            f"{self.service_name}Client connection failed (attempt {attempt}/"
                            f"{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                    else:
                        logger.debug(
                            f"{self.service_name}Client connection failed (attempt {attempt}/"
                            f"{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"{self.service_name}Client failed to connect after "
                        f"{max_retries} attempts: {e}"
                    )

        raise ConnectionError(
            f"Failed to connect to {self.service_name} at {self._address}: {last_error}"
        )

    def set_direct_server(self, server_instance):
        """
        Set direct server reference for LOCAL mode operation.

        In LOCAL mode, the client bypasses RPC and calls methods directly
        on the provided server instance. This eliminates RPC overhead and
        port conflicts, allowing multiple processes (e.g., miners) to run
        on the same machine.

        Args:
            server_instance: The server instance to call methods on directly

        Example:
            # Create client in LOCAL mode
            client = MyClient(connection_mode=RPCConnectionMode.LOCAL)

            # Set direct server reference
            client.set_direct_server(server_instance)

            # Now all RPC calls go directly to server_instance
            result = client.some_method()  # calls server_instance.some_method_rpc()
        """
        if self.connection_mode != RPCConnectionMode.LOCAL:
            logger.warning(
                f"{self.service_name}Client.set_direct_server() called but connection_mode is {self.connection_mode}, "
                f"not LOCAL. This may cause unexpected behavior."
            )
        self._direct_server = server_instance
        logger.debug(f"{self.service_name}Client: Direct server reference set (LOCAL mode)")

    def call(self, method_name: str, *args, **kwargs) -> Any:
        """
        Generic method to call any RPC method by name.

        Thin wrapper over the central _invoke_rpc() choke point, so the generic call() path and
        the typed self._server.foo_rpc() path share identical reconnect-on-server-bounce behavior.

        Args:
            method_name: Name of the RPC method to call (e.g., "some_method_rpc")
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            The result from the RPC call

        Raises:
            AttributeError: If method doesn't exist on remote service

        Example:
            result = client.call("get_data_rpc", key="some_key")
        """
        return self._invoke_rpc(method_name, args, kwargs)

    def _invoke_rpc(self, method_name: str, args: tuple = (), kwargs: dict = None,
                    retry: bool = True) -> Any:
        """
        Central choke point for EVERY RPC method call (typed wrappers via _ResilientRPCProxy and
        the generic call() path both route here).

        Behavior:
          1. LOCAL/direct mode: call straight through to the in-process server, no transport.
          2. RPC mode: invoke on the (lazily-connected) proxy. On a *transient* transport error
             (a state server bouncing during a deploy — see ErrorUtils.is_transient_rpc_error),
             drop the poisoned connection and retry the whole reconnect+call cycle a bounded
             number of times, with a short settle between attempts. Business-logic errors are
             re-raised untouched and never trigger a reconnect/retry.
          3. Every proxy invocation here (first attempt, each self-heal retry, and the transport
             probe) goes through _call_proxy_method, i.e. it is bounded at rpc_call_timeout_s.
             A server that is WEDGED rather than gone therefore raises RPCCallTimeoutError, which
             is re-raised IMMEDIATELY — no reconnect, no retry, no connection reset: the
             transport is healthy and the server is still executing the call, so re-running it
             would multiply load and could double-apply a non-idempotent write. That keeps the
             retry=False guarantee below intact for a wedge as well as for a bounce.

        Why retry the CYCLE (not just the call): a just-restarted server accepts the TCP connect
        a moment before its manager can actually serve method calls, so a single reconnect+retry
        can still hit a broken pipe on the send. Cycling reconnect+call with a brief settle rides
        out that readiness window. Reconnecting has no side effects; only the method re-run carries
        at-least-once semantics (safe here: reads dominate, order writes are UUID-deduped, sync-epoch
        is staleness-tolerant).

        retry=False -> fail-fast: attempt once, and on a transient error drop the poisoned
        connection (so the NEXT call reconnects) but re-raise immediately instead of cycling. Use
        for liveness probes (health_check) and any future non-idempotent method that must not be
        auto-retried — the caller sees the failure at once rather than blocking on the self-heal.

        Reconnect is de-duped across threads via _connection_generation: this client is shared
        across the axon's executor threads, so a transient-error burst would otherwise have every
        thread rebuild the connection. Only the first thread whose captured generation still
        matches rebuilds; the rest reuse its fresh connection.
        """
        kwargs = kwargs or {}

        # LOCAL / direct-server mode: no RPC transport, nothing to reconnect.
        if self._direct_server is not None:
            return getattr(self._direct_server, method_name)(*args, **kwargs)

        # Circuit breaker: while open (a recent self-heal or connect already failed), this call
        # gets ONE fast attempt — quick connect, no self-heal cycles, no settle sleeps — so an
        # outage costs each caller ~0.3s instead of ~4-8s of blocked thread. Any success closes
        # the breaker.
        breaker_open = time.time() < self._backoff_until

        # ---- First attempt (lazy-connects on first use) ----
        try:
            if breaker_open and self._proxy is None:
                self.connect(max_retries=1, retry_delay=0.25)
            proxy = self._ensure_proxy()
            if proxy is None:
                # connect() returned without establishing a transport. Surface the explicit
                # "Not connected" contract instead of letting getattr(None, method) raise an
                # AttributeError, which the classifier below would read as a permanent
                # business error.
                raise RuntimeError(f"Not connected to {self.service_name}")
        except Exception:
            # Even the connect failed — open (or extend) the breaker so concurrent/subsequent
            # calls fail fast instead of each paying the full connect-retry penalty.
            self._backoff_until = time.time() + self._BACKOFF_WINDOW_S
            raise
        generation = self._connection_generation
        try:
            result = self._call_proxy_method(proxy, method_name, args, kwargs)
            # Do NOT interpolate args/kwargs here: this runs on EVERY call (all typed wrappers
            # route through here now), and repr-ing large payloads (metagraph/position lists) on
            # the hot path would cost even when trace logging is disabled. Name + result type only.
            logger.debug(f"{self.service_name}Client.{method_name} -> {type(result).__name__}")
            if self._backoff_until:
                self._backoff_until = 0.0
            return result
        except RPCCallTimeoutError:
            # Client-side timeout: the service is alive but WEDGED and is STILL EXECUTING this
            # call. Nothing to reconnect (the transport is fine) and nothing safe to retry, so
            # free the caller now — that is the whole point of the bound — and let caller-level
            # retry decide. MUST precede the is_transient_rpc_error() check: RPCCallTimeoutError
            # is a TimeoutError, i.e. an OSError, so it would otherwise be misread as a dead
            # transport and dragged through the full reconnect+retry cycle.
            raise
        except Exception as e:
            if not ErrorUtils.is_transient_rpc_error(e):
                # Business rejection / missing method / etc. — do NOT reconnect or retry.
                raise
            # Type says transient — but multiprocessing transports server-raised exceptions
            # VERBATIM, so an OSError subclass raised by the server's own business logic (e.g.
            # a position-lock TimeoutError under contention) is type-identical to a dead socket.
            # Disambiguate by probing the SAME connection: if it still serves calls, the error
            # came from the server's logic — re-executing the method would multiply real work
            # (6x the full order pipeline per the review) for nothing. Probe cost is one cheap
            # RPC, on error paths only.
            if self._transport_probe_ok(proxy):
                raise
            last_error = e

        # Fail-fast opt-out: drop the poisoned connection (de-duped) so the next call reconnects,
        # then re-raise now instead of running the multi-cycle self-heal.
        if not retry or breaker_open:
            self._backoff_until = time.time() + self._BACKOFF_WINDOW_S
            with self._conn_lock:
                if self._connection_generation == generation:
                    self._reset_connection()
            raise last_error

        # ---- Self-heal: bounded cycles of (rebuild connection, retry the call) ----
        for attempt in range(1, self._RECONNECT_MAX_RETRIES + 1):
            logger.warning(
                f"{self.service_name}Client.{method_name} transient RPC error ({last_error!r}); "
                f"reconnecting and retrying (attempt {attempt}/{self._RECONNECT_MAX_RETRIES})..."
            )

            # Rebuild the connection (de-duped): only if no other thread has rebuilt since our
            # captured generation. A concurrent rebuild -> reuse its fresh proxy.
            with self._conn_lock:
                if self._connection_generation == generation:
                    self._reset_connection()
                    try:
                        # Single fast attempt: the outer loop owns the retry cadence and its
                        # settle sleep runs OUTSIDE this lock, so connect() must not sleep here.
                        self.connect(max_retries=1, retry_delay=self._RECONNECT_RETRY_DELAY_S)
                        proxy = self._proxy
                    except Exception as reconnect_err:
                        # Server still down; state is reset so the next cycle retries the connect.
                        last_error = reconnect_err
                        proxy = None
                else:
                    proxy = self._proxy
                generation = self._connection_generation

            if proxy is not None:
                try:
                    result = self._call_proxy_method(proxy, method_name, args, kwargs)
                    logger.info(
                        f"{self.service_name}Client.{method_name} recovered after reconnect "
                        f"(attempt {attempt})."
                    )
                    return result
                except RPCCallTimeoutError:
                    # Reconnected fine, then the call itself wedged — same reasoning as the
                    # first attempt: free the caller instead of cycling on a live-but-stuck
                    # service.
                    raise
                except Exception as e2:
                    if not ErrorUtils.is_transient_rpc_error(e2):
                        raise
                    if self._transport_probe_ok(proxy):
                        # Fresh connection works — this is the server's own OSError-family
                        # business exception; stop re-executing (see first-attempt comment).
                        raise
                    last_error = e2

            # Brief settle so a just-restarted server becomes ready before the next cycle.
            if attempt < self._RECONNECT_MAX_RETRIES:
                time.sleep(self._RECONNECT_RETRY_DELAY_S)

        # All recovery cycles exhausted — open the circuit breaker (subsequent calls fail fast
        # for _BACKOFF_WINDOW_S), drop the (still-dead) connection so a later call reconnects
        # cleanly, and surface the failure to the caller (placer retry / daemon loop).
        self._backoff_until = time.time() + self._BACKOFF_WINDOW_S
        with self._conn_lock:
            if self._connection_generation == generation:
                self._reset_connection()
        logger.error(
            f"{self.service_name}Client.{method_name} still failing after "
            f"{self._RECONNECT_MAX_RETRIES} reconnect attempts: {last_error!r}"
        )
        raise last_error

    def _transport_probe_ok(self, proxy) -> bool:
        """
        True if the connection that just raised is actually alive — meaning the exception was
        raised by the SERVER'S code (transported verbatim by multiprocessing) rather than by the
        transport itself. Every RPCServerBase exposes health_check_rpc.

        BaseProxy connections are thread-local, and the failed call ran on a worker of this
        client's bounded executor, so the probe is issued through that same executor: with the
        worker just freed by the failure it normally reuses that very connection. Even when the
        pool hands it a different worker the verdict holds — a dead/restarted server fails the
        probe on any connection, a live one answers on any connection.
        """
        if proxy is None:
            return False
        try:
            # Bounded like every other call: an unbounded probe against a WEDGED service would
            # hang this thread forever — precisely the failure the per-call timeout exists to
            # prevent. A probe that times out is not provably alive, so it reports False and the
            # caller treats the connection as dead.
            self._call_proxy_method(proxy, "health_check_rpc", (), {})
            return True
        except Exception:
            return False

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
        # Liveness probe: fail fast (retry=False) rather than block on the multi-cycle self-heal.
        # A poisoned connection is still dropped, so the next real call reconnects.
        return self._invoke_rpc("health_check_rpc", retry=False)

    def start_daemon(self) -> bool:
        """
        Start the daemon thread remotely via RPC.

        All RPC servers inherit from RPCServerBase which provides start_daemon_rpc().
        This is a standard method available on all servers.

        Returns:
            bool: True if daemon was started, False if already running
        """
        return self._server.start_daemon_rpc()

    def stop_daemon(self) -> bool:
        """
        Stop the daemon thread remotely via RPC.

        All RPC servers inherit from RPCServerBase which provides stop_daemon_rpc().
        This is a standard method available on all servers.

        Returns:
            bool: True if daemon was stopped, False if not running
        """
        return self._server.stop_daemon_rpc()

    def _teardown_transport(self) -> None:
        """
        Close the BaseManager connection and clear proxy/connected state.

        Shared by disconnect() (full teardown) and _reset_connection() (transient self-heal).
        Cleans up the IPC resources BaseManager holds so they don't leak across reconnects.
        """
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
                logger.debug(f"{self.service_name}Client error during manager cleanup: {e}")

        # Write order is load-bearing: _connected FIRST, then _proxy. _ensure_proxy reads these
        # unlocked; the reverse order exposed (_proxy=None, _connected=True), which skipped the
        # reconnect and produced getattr(None, method) -> AttributeError — misclassified as a
        # permanent business error on the order path.
        self._connected = False
        self._manager = None
        self._proxy = None

    def _reset_connection(self) -> None:
        """
        Drop ONLY the transport so the next call reconnects, used by the transient self-heal
        path in _invoke_rpc().

        Unlike disconnect(), this intentionally does NOT unregister the instance or stop the
        cache-refresh daemon — a server bounce is a transient blip, not a teardown. (The old
        call()-only self-heal used the full disconnect(), which de-registered long-lived clients
        from instance tracking and permanently killed their cache-refresh daemon on every blip.)

        Callers should hold self._conn_lock.
        """
        self._teardown_transport()

    def disconnect(self):
        """Disconnect from the server."""
        # Stop cache refresh daemon if running
        if self._cache_refresh_thread is not None:
            self._cache_refresh_shutdown.set()
            self._cache_refresh_thread.join(timeout=2.0)
            self._cache_refresh_thread = None

        # Clean up manager connection (prevents semaphore leaks)
        # BaseManager creates IPC resources that need explicit cleanup
        self._teardown_transport()
        self._direct_server = None
        if self._rpc_executor is not None:
            # Don't wait: parked workers may be blocked on a dead service.
            self._rpc_executor.shutdown(wait=False, cancel_futures=True)
            self._rpc_executor = None

        # Unregister from instance tracking
        RPCClientBase._unregister_instance(self)
        # Skip logging disconnect to avoid race condition with pytest closing stdout/stderr

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
        logger.info(
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

                logger.debug(
                    f"[{self.service_name}] Local cache refreshed in {refresh_ms:.2f}ms "
                    f"({len(new_cache) if isinstance(new_cache, dict) else 'N/A'} entries)"
                )

            except Exception as e:
                logger.error(f"[{self.service_name}] Error refreshing local cache: {e}")

            # Wait for next refresh cycle (interruptible)
            self._cache_refresh_shutdown.wait(timeout=refresh_interval_s)

        # Skip logging to avoid race condition with pytest closing stdout/stderr
        # logger.info(f"[{self.service_name}] Local cache refresh daemon stopped")

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
                return {"eliminations": eliminations}
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
        logger.debug(
            f"[{self.service_name}_PICKLE] __getstate__ called in PID {os.getpid()}"
        )

        state = self.__dict__.copy()

        # Mark as needing reconnection after unpickle
        state['_needs_reconnect'] = True

        # Don't pickle proxy/manager objects (they're not picklable)
        state['_manager'] = None
        state['_proxy'] = None

        # Don't pickle the resilient proxy (holds a back-reference to self -> would recurse) or
        # the connection lock (threading.RLock is not picklable). Both are recreated in __setstate__.
        state['_resilient_proxy'] = None
        state['_conn_lock'] = None

        # Don't pickle cache-related unpicklable objects
        state['_local_cache_lock'] = None
        state['_cache_refresh_thread'] = None
        state['_cache_refresh_shutdown'] = None

        # Don't pickle bounded-call machinery (executor/lock are unpicklable; both are
        # recreated lazily after unpickle). '_bounded_proxy' is the pre-merge name of the
        # bounded facade that now lives inside _invoke_rpc — kept nulled so a state pickled by
        # an older build never carries a stale facade object across the process boundary.
        state['_rpc_executor'] = None
        state['_rpc_executor_lock'] = None
        state['_bounded_proxy'] = None

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
        logger.debug(
            f"[{state.get('service_name', 'RPC')}_UNPICKLE] __setstate__ called in PID {os.getpid()}"
        )

        self.__dict__.update(state)

        # Recreate transient objects dropped in __getstate__.
        self._conn_lock = threading.RLock()
        self._resilient_proxy = _ResilientRPCProxy(self)
        # Tolerate pickles produced before _connection_generation existed.
        if not hasattr(self, '_connection_generation'):
            self._connection_generation = 0

        # Restore subclass-specific unpicklable state
        self._restore_unpicklable_state(state)

        # In LOCAL mode, nothing to reconnect
        if self.connection_mode == RPCConnectionMode.LOCAL:
            logger.debug(f"[{self.service_name}_UNPICKLE] LOCAL mode - no reconnection needed")
            return

        # Reconnect to existing RPC server (RPC mode)
        if state.get('_needs_reconnect', False):
            logger.debug(
                f"[{self.service_name}_UNPICKLE] Reconnecting to RPC server on port {self.port}"
            )

            # Use faster retry settings for unpickle reconnection
            original_retries = self._max_retries
            original_delay = self._retry_delay_s
            self._max_retries = 5  # Fewer retries - server should be running
            self._retry_delay_s = 0.5

            try:
                self.connect()
                logger.info(
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

        # Restore bounded-call machinery (the executor is recreated lazily on first use)
        self._rpc_executor = None
        self._rpc_executor_lock = threading.Lock()
        self._rpc_timeout_log_ts = {}

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
