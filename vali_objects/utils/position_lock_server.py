"""
Position Lock Server - RPC service for managing position locks across processes.

Provides centralized lock management to avoid IPC overhead of multiprocessing.Manager.
"""
import bittensor as bt
import threading
from typing import Tuple, Dict
from shared_objects.rpc_service_base import RPCServiceBase


class PositionLockServer:
    """
    Server-side position lock manager with local dict storage.

    Locks are held server-side. Clients call acquire_rpc/release_rpc instead of
    getting lock objects. This avoids the problem of trying to proxy Lock objects
    across processes.
    """

    def __init__(self):
        """Initialize the lock server."""
        # Local dict to store locks (faster than IPC dict)
        # Use threading.Lock since all RPC access goes through this server process
        self.locks: Dict[Tuple[str, str], threading.Lock] = {}
        self.locks_dict_lock = threading.Lock()  # Protect dict mutations

        bt.logging.info("PositionLockServer initialized")

    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring."""
        from time_util.time_util import TimeUtil
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "num_locks": len(self.locks)
        }

    def _get_or_create_lock(self, miner_hotkey: str, trade_pair_id: str) -> threading.Lock:
        """
        Get or create a lock for the given key (internal method).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID

        Returns:
            threading.Lock object
        """
        lock_key = (miner_hotkey, trade_pair_id)

        # Check if lock exists (read-only, no lock needed for speed)
        lock = self.locks.get(lock_key)
        if lock is not None:
            return lock

        # Lock doesn't exist - acquire dict lock to create it
        with self.locks_dict_lock:
            # Double-check (another thread might have created it)
            lock = self.locks.get(lock_key)
            if lock is not None:
                return lock

            # Create new threading lock (all RPC access goes through this server process)
            lock = threading.Lock()
            self.locks[lock_key] = lock

            bt.logging.trace(
                f"[LOCK_SERVER] Created lock for {miner_hotkey[:8]}.../{trade_pair_id}"
            )

            return lock

    def acquire_rpc(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0) -> bool:
        """
        Acquire lock for the given key (blocks until available or timeout).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if lock was acquired, False if timeout
        """
        lock = self._get_or_create_lock(miner_hotkey, trade_pair_id)
        acquired = lock.acquire(timeout=timeout)

        if not acquired:
            bt.logging.warning(
                f"[LOCK_SERVER] Failed to acquire lock for {miner_hotkey[:8]}.../{trade_pair_id} after {timeout}s"
            )

        return acquired

    def release_rpc(self, miner_hotkey: str, trade_pair_id: str) -> bool:
        """
        Release lock for the given key.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID

        Returns:
            bool: True if released successfully, False if error
        """
        lock_key = (miner_hotkey, trade_pair_id)
        lock = self.locks.get(lock_key)

        if lock is None:
            bt.logging.warning(
                f"[LOCK_SERVER] Attempted to release non-existent lock for {miner_hotkey[:8]}.../{trade_pair_id}"
            )
            return False

        try:
            lock.release()
            return True
        except RuntimeError as e:
            # Lock was not held (already released)
            bt.logging.warning(
                f"[LOCK_SERVER] Error releasing lock for {miner_hotkey[:8]}.../{trade_pair_id}: {e}"
            )
            return False


def start_position_lock_server(address, authkey, server_ready):
    """Entry point for server process."""
    from multiprocessing.managers import BaseManager
    from setproctitle import setproctitle

    setproctitle("vali_PositionLockServer")

    server_instance = PositionLockServer()

    # Register server with manager
    class PositionLockRPC(BaseManager):
        pass

    PositionLockRPC.register('PositionLockServer', callable=lambda: server_instance)

    manager = PositionLockRPC(address=address, authkey=authkey)
    rpc_server = manager.get_server()

    bt.logging.success(f"PositionLockServer ready on {address}")

    if server_ready:
        server_ready.set()

    # Start serving (blocks forever)
    rpc_server.serve_forever()


class PositionLockProxy:
    """
    Context manager proxy for position locks.

    Calls acquire_rpc/release_rpc on the server instead of trying to
    proxy Lock objects across processes.
    """

    def __init__(self, server_proxy, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0):
        """
        Initialize lock proxy.

        Args:
            server_proxy: RPC proxy to PositionLockServer
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Lock acquisition timeout in seconds
        """
        self.server = server_proxy
        self.miner_hotkey = miner_hotkey
        self.trade_pair_id = trade_pair_id
        self.timeout = timeout
        self.acquired = False

    def __enter__(self):
        """Acquire lock via RPC."""
        self.acquired = self.server.acquire_rpc(self.miner_hotkey, self.trade_pair_id, self.timeout)
        if not self.acquired:
            raise TimeoutError(
                f"Failed to acquire lock for {self.miner_hotkey}/{self.trade_pair_id} after {self.timeout}s"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock via RPC."""
        if self.acquired:
            self.server.release_rpc(self.miner_hotkey, self.trade_pair_id)
        return False  # Don't suppress exceptions


class PositionLockManagerClient(RPCServiceBase):
    """
    RPC client for PositionLockServer.

    Provides centralized lock management with better performance than
    multiprocessing.Manager IPC overhead.

    Supports two modes:
    - Server mode: Starts its own server (default)
    - Client-only mode: Connects to existing server (for multi-process scenarios)

    Usage:
        # First instance (starts server)
        server_locks = PositionLockManagerClient(running_unit_tests=False)

        # Additional instances (connect to existing server)
        client_locks = PositionLockManagerClient(running_unit_tests=False, client_only=True)

        # Or pass server_locks to other processes (automatically becomes client_only)
        with client_locks.get_lock(hotkey, trade_pair_id):
            # Critical section - only one thread/process can execute this
            pass
    """

    def __init__(self, running_unit_tests: bool = False, client_only: bool = False):
        """
        Initialize the position lock manager client.

        Args:
            running_unit_tests: If True, use direct in-memory mode
            client_only: If True, only connect to existing server (don't start new server)
        """
        self.client_only = client_only

        super().__init__(
            service_name="PositionLockServer",
            port=50008,  # Dedicated port for lock server
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True
        )
        self._initialize_service()

    def _initialize_service(self):
        """
        Initialize service in client-only mode or full mode.

        In client-only mode:
        - Skip starting the server process
        - Just connect to existing server
        - Used when passing locks to child processes
        """
        if self.client_only and not self.running_unit_tests:
            bt.logging.info(f"{self.service_name} client-only mode: connecting to existing server")

            # Use a known authkey (must match what server used)
            # In production, all clients on same host connect to same server
            import hashlib
            # Use stable authkey based on port (all clients/servers use same key)
            self._authkey = hashlib.sha256(f"PositionLockServer_{self.port}".encode()).digest()[:32]

            # Connect to existing server
            self._connect_client()

            # Enable health checks for client-only mode too
            if self.enable_health_check:
                self._health_check_enabled_for_instance = True
        else:
            # Normal mode: start server or use direct mode
            super()._initialize_service()

    def _create_direct_server(self):
        """Create direct in-memory server for tests."""
        return PositionLockServer()

    def _start_server_process(self, address, authkey, server_ready):
        """Start RPC server in separate process."""
        from multiprocessing import Process

        # Store authkey for client-only mode
        # In server mode, we know the authkey - save it for future client-only instances
        import hashlib
        self._authkey = hashlib.sha256(f"PositionLockServer_{self.port}".encode()).digest()[:32]

        process = Process(
            target=start_position_lock_server,
            args=(address, self._authkey, server_ready),
            daemon=True
        )
        process.start()
        return process

    def __getstate__(self):
        """
        Prepare object for pickling (when passed to child processes).

        When pickled, mark as client-only so child processes connect
        to existing server instead of trying to start their own.
        """
        state = self.__dict__.copy()

        # When unpickled in child process, become client-only
        state['client_only'] = True

        # Don't pickle process/proxy objects (they're not picklable anyway)
        state['_server_process'] = None
        state['_client_manager'] = None
        state['_server_proxy'] = None

        return state

    def __setstate__(self, state):
        """
        Restore object after unpickling (in child process).

        Automatically reconnects to existing server.
        """
        self.__dict__.update(state)

        # Reinitialize connection to existing server
        # This will use client_only=True from pickled state
        self._initialize_service()

    def get_lock(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0):
        """
        Get a lock proxy for the given key.

        Returns a context manager that acquires/releases the lock via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Lock acquisition timeout in seconds

        Returns:
            PositionLockProxy: Context manager for the lock

        Usage:
            with lock_client.get_lock(hotkey, pair_id):
                # Critical section
                pass
        """
        return PositionLockProxy(self._server_proxy, miner_hotkey, trade_pair_id, timeout)
