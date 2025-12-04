# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Position Lock Server - RPC service for managing position locks across processes.

Provides centralized lock management to avoid IPC overhead of multiprocessing.Manager.

Architecture:
- PositionLockServer inherits from RPCServerBase for unified infrastructure
- PositionLockClient inherits from RPCClientBase for lightweight RPC access
- PositionLockProxy provides context manager for acquire/release pattern

Usage:
    # Server (typically started by validator)
    server = PositionLockServer(
        start_server=True,
        start_daemon=False  # No daemon needed for lock service
    )

    # Client (can be created in any process)
    client = PositionLockClient()
    with client.get_lock(hotkey, trade_pair_id):
        # Critical section
        pass
"""
import bittensor as bt
import threading
from typing import Tuple, Dict

from shared_objects.rpc.rpc_server_base import RPCServerBase
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig


class PositionLockServer(RPCServerBase):
    """
    Server-side position lock manager with local dict storage.

    Locks are held server-side. Clients call acquire_rpc/release_rpc instead of
    getting lock objects. This avoids the problem of trying to proxy Lock objects
    across processes.

    Inherits from RPCServerBase for unified RPC infrastructure, though this service
    doesn't require a daemon (locks are passive - only respond to acquire/release).
    """
    service_name = ValiConfig.RPC_POSITIONLOCK_SERVICE_NAME
    service_port = ValiConfig.RPC_POSITIONLOCK_PORT

    def __init__(
        self,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False  # No daemon needed for lock service
    ):
        """
        Initialize the lock server.

        Args:
            running_unit_tests: Whether running in unit test mode
            slack_notifier: Optional SlackNotifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon (not needed for locks)
        """
        # Local dict to store locks (faster than IPC dict)
        # Use threading.Lock since all RPC access goes through this server process
        self.locks: Dict[Tuple[str, str], threading.Lock] = {}
        self.locks_dict_lock = threading.Lock()  # Protect dict mutations

        # Initialize base class
        # daemon_interval_s: 60s (slow interval since daemon does nothing)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms
        daemon_interval_s = 60.0
        hang_timeout_s = daemon_interval_s * 2.0  # 120s (2x interval)

        super().__init__(
            service_name=ValiConfig.RPC_POSITIONLOCK_SERVICE_NAME,
            port=ValiConfig.RPC_POSITIONLOCK_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s
        )

        bt.logging.info("PositionLockServer initialized")

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Daemon iteration (no-op for lock service).

        Position locks are passive - they only respond to acquire/release requests.
        No background processing needed.
        """
        # No background processing needed for lock management
        pass

    # ==================== Lock RPC Methods ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
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

    def get_lock_count_rpc(self) -> int:
        """Get the number of locks currently tracked."""
        return len(self.locks)


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
            server_proxy: RPC proxy to PositionLockServer (or direct server in test mode)
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


class PositionLockClient(RPCClientBase):
    """
    Lightweight RPC client for PositionLockServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_POSITIONLOCK_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct PositionLockServer instance.

    Usage:
        # Production mode - connects to existing server
        client = PositionLockClient()
        with client.get_lock(hotkey, pair_id):
            # Critical section
            pass

        # Test mode - direct server access
        client = PositionLockClient(running_unit_tests=True)
        client.set_direct_server(server_instance)
    """

    def __init__(self, port: int = None, running_unit_tests: bool = False):
        """
        Initialize position lock client.

        Args:
            port: Port number of the position lock server (default: ValiConfig.RPC_POSITIONLOCK_PORT)
            running_unit_tests: If True, don't connect via RPC (use set_direct_server() instead)
        """
        super().__init__(
            service_name=ValiConfig.RPC_POSITIONLOCK_SERVICE_NAME,
            port=port or ValiConfig.RPC_POSITIONLOCK_PORT,
            connect_immediately=False
        )

    # ==================== Lock Methods ====================

    def get_lock(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0) -> PositionLockProxy:
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
            with client.get_lock(hotkey, pair_id):
                # Critical section
                pass
        """
        return PositionLockProxy(self._server, miner_hotkey, trade_pair_id, timeout)

    def acquire(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0) -> bool:
        """
        Acquire lock directly (without context manager).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Lock acquisition timeout in seconds

        Returns:
            bool: True if lock was acquired, False if timeout
        """
        return self._server.acquire_rpc(miner_hotkey, trade_pair_id, timeout)

    def release(self, miner_hotkey: str, trade_pair_id: str) -> bool:
        """
        Release lock directly (without context manager).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID

        Returns:
            bool: True if released successfully, False if error
        """
        return self._server.release_rpc(miner_hotkey, trade_pair_id)

    # ==================== Health Check ====================

    def health_check(self) -> dict:
        """Get health status from server."""
        return self._server.health_check_rpc()

    def get_lock_count(self) -> int:
        """Get the number of locks currently tracked."""
        return self._server.get_lock_count_rpc()
