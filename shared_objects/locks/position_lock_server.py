# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
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
import threading
from typing import Tuple, Dict

from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.vali_config import ValiConfig
from shared_objects.log import logger
from time_util.time_util import TimeUtil


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
        # Current owner of each held lock: key -> (owner_token, held_at_ms). Absent = not held.
        # The token gives releases holder identity: threading.Lock has none, so without it a
        # lease-reclaimed holder's eventual release would free the RECLAIMER's lock, cascading to
        # multiple concurrent writers on one (hotkey, trade_pair). Guarded by locks_dict_lock.
        self.lock_owner: Dict[Tuple[str, str], Tuple[int, float]] = {}
        self._next_owner_token = 0
        self._lock_lease_ms = ValiConfig.POSITION_LOCK_LEASE_MS

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

        logger.info("PositionLockServer initialized")

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

            logger.debug(
                f"[LOCK_SERVER] Created lock for {miner_hotkey}.../{trade_pair_id}"
            )

            return lock

    def acquire_rpc(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0):
        """
        Acquire lock for the given key (blocks until available or timeout).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Maximum time to wait in seconds

        Returns:
            int owner token (truthy) if acquired — pass it back to release_rpc so only the
            actual holder can release; False if timeout. Truthiness preserves the legacy
            bool contract for callers that only check success.
        """
        lock_key = (miner_hotkey, trade_pair_id)
        lock = self._get_or_create_lock(miner_hotkey, trade_pair_id)

        # Lease reclaim (never-release protection): if this lock has been HELD past the lease, the
        # holder is presumed dead (crashed between acquire and release — a real hazard now that the
        # holder is the crashable vanta-orders process, not core). Force-release so the (hotkey,
        # trade_pair) is not wedged forever. The lease is FAR above any real hold (see
        # POSITION_LOCK_LEASE_MS), so a live-but-slow holder is never reclaimed. Serialized under
        # locks_dict_lock so only one reclaim fires; dropping the owner entry first makes the dead
        # holder's late release (if it ever arrives) a token-mismatch no-op.
        now_ms = TimeUtil.now_in_millis()
        with self.locks_dict_lock:
            owner = self.lock_owner.get(lock_key)
            if owner is not None and (now_ms - owner[1]) > self._lock_lease_ms:
                self.lock_owner.pop(lock_key, None)
                try:
                    lock.release()
                    logger.warning(
                        f"[LOCK_SERVER] Reclaimed stale lock for {miner_hotkey}.../{trade_pair_id} "
                        f"held {(now_ms - owner[1]) / 1000:.1f}s (> {self._lock_lease_ms / 1000:.0f}s lease) "
                        f"— presumed crashed holder"
                    )
                except RuntimeError:
                    pass  # already free; nothing to reclaim

        acquired = lock.acquire(timeout=timeout)

        if acquired:
            with self.locks_dict_lock:
                self._next_owner_token += 1
                token = self._next_owner_token
                self.lock_owner[lock_key] = (token, TimeUtil.now_in_millis())
            return token

        logger.warning(
            f"[LOCK_SERVER] Failed to acquire lock for {miner_hotkey}.../{trade_pair_id} after {timeout}s"
        )
        return False

    def release_rpc(self, miner_hotkey: str, trade_pair_id: str, token: int = None) -> bool:
        """
        Release lock for the given key.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            token: Owner token from acquire_rpc. When provided, the release is a no-op unless
                the token matches the current owner — a lease-reclaimed holder's late release
                must not free the lock its reclaimer now holds. None = legacy unconditional
                release (version-skew tolerance only).

        Returns:
            bool: True if released successfully, False if error/stale
        """
        lock_key = (miner_hotkey, trade_pair_id)
        lock = self.locks.get(lock_key)

        if lock is None:
            logger.warning(
                f"[LOCK_SERVER] Attempted to release non-existent lock for {miner_hotkey}.../{trade_pair_id}"
            )
            return False

        with self.locks_dict_lock:
            owner = self.lock_owner.get(lock_key)
            if token is not None and (owner is None or owner[0] != token):
                logger.warning(
                    f"[LOCK_SERVER] Stale release for {miner_hotkey}.../{trade_pair_id} "
                    f"(token {token}, current owner {owner[0] if owner else None}) — ignored; "
                    f"the lock was lease-reclaimed and is (or will be) held by a newer owner"
                )
                return False
            self.lock_owner.pop(lock_key, None)
            try:
                lock.release()
                return True
            except RuntimeError as e:
                # Lock was not held (already released — possibly reclaimed by the lease above)
                logger.warning(
                    f"[LOCK_SERVER] Error releasing lock for {miner_hotkey}.../{trade_pair_id}: {e}"
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
        self.token = None

    def __enter__(self):
        """Acquire lock via RPC. Keeps the owner token so only THIS holder's exit releases."""
        self.token = self.server.acquire_rpc(self.miner_hotkey, self.trade_pair_id, self.timeout)
        self.acquired = bool(self.token)
        if not self.acquired:
            raise TimeoutError(
                f"Failed to acquire lock for {self.miner_hotkey}/{self.trade_pair_id} after {self.timeout}s"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock via RPC (token-checked: a lease-reclaimed hold releases as a no-op)."""
        if self.acquired:
            self.server.release_rpc(self.miner_hotkey, self.trade_pair_id, self.token)
        return False  # Don't suppress exceptions


