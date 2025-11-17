"""
Position Lock Server - Dedicated IPC server for managing position locks.

Provides centralized lock management to reduce contention and improve performance.
"""
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import uuid
import bittensor as bt


@dataclass
class LockState:
    """State of a single lock"""
    lock: threading.Lock
    holder: Optional[str] = None  # client_id holding the lock
    lock_token: Optional[str] = None  # unique token for this lock acquisition
    acquired_at: Optional[int] = None  # timestamp in ms
    total_acquisitions: int = 0
    total_wait_ms: float = 0
    max_hold_ms: float = 0


@dataclass
class LockMetrics:
    """Metrics for lock performance"""
    lock_key: Tuple[str, str]
    total_acquisitions: int = 0
    total_contentions: int = 0  # how many times had to wait
    avg_wait_ms: float = 0
    max_wait_ms: float = 0
    avg_hold_ms: float = 0
    max_hold_ms: float = 0
    current_holder: Optional[str] = None


class PositionLockServer:
    """
    Dedicated server for managing position locks via RPC.

    Provides traditional lock acquire/release semantics with better scheduling
    and monitoring than multiprocessing.Manager locks.

    Key benefits:
    - Centralized lock management (single process)
    - Better scheduling (FIFO fairness per lock)
    - Comprehensive metrics (wait times, hold times, contention)
    - Timeout detection and cleanup
    - No multiprocessing.Manager overhead
    """

    def __init__(self, orphan_timeout_ms: int = 60000):
        """
        Initialize the lock server.

        Args:
            orphan_timeout_ms: Time after which to auto-release locks (prevents deadlocks from crashed clients)
        """
        # Lock states - Key: (miner_hotkey, trade_pair_id), Value: LockState
        self.locks: Dict[Tuple[str, str], LockState] = {}
        self.locks_dict_lock = threading.Lock()  # Protect the locks dict itself

        # Metrics
        self.metrics: Dict[Tuple[str, str], LockMetrics] = {}
        self.global_metrics = {
            'total_acquisitions': 0,
            'total_releases': 0,
            'total_timeouts': 0,
            'active_locks': 0
        }

        # Orphan detection
        self.orphan_timeout_ms = orphan_timeout_ms
        self.orphan_checker_thread = None
        self.shutdown_event = threading.Event()

        bt.logging.info(
            f"PositionLockServer initialized (orphan_timeout={orphan_timeout_ms}ms)"
        )

    def acquire_lock(self, lock_key: Tuple[str, str], client_id: str,
                    timeout_ms: int = 30000) -> dict:
        """
        Acquire a lock for the given key.

        Args:
            lock_key: (miner_hotkey, trade_pair_id) tuple
            client_id: Unique identifier for the requesting client
            timeout_ms: Maximum time to wait for lock acquisition

        Returns:
            dict with:
                - success: bool
                - lock_token: str (if successful, used for release)
                - message: str (error message if failed)
                - wait_ms: float (time spent waiting)
        """
        start_ms = int(time.time() * 1000)

        # Get or create lock state
        with self.locks_dict_lock:
            if lock_key not in self.locks:
                self.locks[lock_key] = LockState(lock=threading.Lock())
                self.metrics[lock_key] = LockMetrics(lock_key=lock_key)

            lock_state = self.locks[lock_key]
            metrics = self.metrics[lock_key]

        # Try to acquire the lock
        timeout_s = timeout_ms / 1000.0
        acquired = lock_state.lock.acquire(timeout=timeout_s)

        wait_ms = int(time.time() * 1000) - start_ms

        if acquired:
            # Successfully acquired lock
            lock_token = f"{client_id}_{uuid.uuid4()}"
            lock_state.holder = client_id
            lock_state.lock_token = lock_token
            lock_state.acquired_at = int(time.time() * 1000)
            lock_state.total_acquisitions += 1

            # Update metrics
            metrics.total_acquisitions += 1
            if wait_ms > 0:
                metrics.total_contentions += 1
            metrics.avg_wait_ms = (
                (metrics.avg_wait_ms * (metrics.total_acquisitions - 1) + wait_ms)
                / metrics.total_acquisitions
            )
            metrics.max_wait_ms = max(metrics.max_wait_ms, wait_ms)
            metrics.current_holder = client_id

            self.global_metrics['total_acquisitions'] += 1
            self.global_metrics['active_locks'] += 1

            bt.logging.trace(
                f"[LOCK_SERVER] Lock acquired: {lock_key} by {client_id[:8]}... "
                f"(wait={wait_ms}ms, token={lock_token[:8]}...)"
            )

            return {
                'success': True,
                'lock_token': lock_token,
                'wait_ms': wait_ms,
                'message': None
            }
        else:
            # Timeout waiting for lock
            metrics.total_contentions += 1
            self.global_metrics['total_timeouts'] += 1

            current_holder = lock_state.holder or "unknown"
            bt.logging.warning(
                f"[LOCK_SERVER] Lock acquisition timeout: {lock_key} by {client_id[:8]}... "
                f"(waited {wait_ms}ms, held by {current_holder[:8]}...)"
            )

            return {
                'success': False,
                'lock_token': None,
                'wait_ms': wait_ms,
                'message': f"Lock acquisition timeout after {wait_ms}ms (held by {current_holder})"
            }

    def release_lock(self, lock_key: Tuple[str, str], lock_token: str) -> dict:
        """
        Release a previously acquired lock.

        Args:
            lock_key: (miner_hotkey, trade_pair_id) tuple
            lock_token: Token returned from acquire_lock

        Returns:
            dict with:
                - success: bool
                - hold_ms: float (how long lock was held)
                - message: str (error message if failed)
        """
        with self.locks_dict_lock:
            lock_state = self.locks.get(lock_key)

        if lock_state is None:
            bt.logging.error(f"[LOCK_SERVER] Release failed: lock_key {lock_key} not found")
            return {
                'success': False,
                'hold_ms': 0,
                'message': f"Lock {lock_key} not found"
            }

        if lock_state.lock_token != lock_token:
            bt.logging.error(
                f"[LOCK_SERVER] Release failed: invalid token for {lock_key} "
                f"(expected {lock_state.lock_token[:8] if lock_state.lock_token else 'None'}..., "
                f"got {lock_token[:8]}...)"
            )
            return {
                'success': False,
                'hold_ms': 0,
                'message': "Invalid lock token"
            }

        # Calculate hold time
        hold_ms = int(time.time() * 1000) - lock_state.acquired_at if lock_state.acquired_at else 0

        # Update metrics
        metrics = self.metrics.get(lock_key)
        if metrics:
            metrics.avg_hold_ms = (
                (metrics.avg_hold_ms * (metrics.total_acquisitions - 1) + hold_ms)
                / metrics.total_acquisitions
            )
            metrics.max_hold_ms = max(metrics.max_hold_ms, hold_ms)
            metrics.current_holder = None

        # Release the lock
        lock_state.holder = None
        lock_state.lock_token = None
        lock_state.acquired_at = None
        lock_state.lock.release()

        self.global_metrics['total_releases'] += 1
        self.global_metrics['active_locks'] -= 1

        bt.logging.trace(
            f"[LOCK_SERVER] Lock released: {lock_key} (held for {hold_ms}ms)"
        )

        return {
            'success': True,
            'hold_ms': hold_ms,
            'message': None
        }

    def get_metrics(self, lock_key: Optional[Tuple[str, str]] = None) -> dict:
        """
        Get metrics for a specific lock or global metrics.

        Args:
            lock_key: Optional lock key to get metrics for

        Returns:
            dict with metrics
        """
        if lock_key:
            metrics = self.metrics.get(lock_key)
            if metrics:
                return {
                    'lock_key': lock_key,
                    'total_acquisitions': metrics.total_acquisitions,
                    'total_contentions': metrics.total_contentions,
                    'avg_wait_ms': metrics.avg_wait_ms,
                    'max_wait_ms': metrics.max_wait_ms,
                    'avg_hold_ms': metrics.avg_hold_ms,
                    'max_hold_ms': metrics.max_hold_ms,
                    'current_holder': metrics.current_holder
                }
            return None
        else:
            return {
                'global': self.global_metrics.copy(),
                'per_key': {
                    key: {
                        'total_acquisitions': m.total_acquisitions,
                        'total_contentions': m.total_contentions,
                        'avg_wait_ms': m.avg_wait_ms,
                        'avg_hold_ms': m.avg_hold_ms,
                        'current_holder': m.current_holder
                    }
                    for key, m in self.metrics.items()
                }
            }

    def _check_orphaned_locks(self):
        """
        Background thread that checks for orphaned locks and auto-releases them.
        Prevents deadlocks from crashed/disconnected clients.
        """
        bt.logging.info("[LOCK_SERVER] Orphan checker started")

        while not self.shutdown_event.wait(timeout=5.0):
            now_ms = int(time.time() * 1000)
            orphaned_count = 0

            with self.locks_dict_lock:
                for lock_key, lock_state in self.locks.items():
                    if lock_state.acquired_at is not None:
                        hold_time_ms = now_ms - lock_state.acquired_at

                        if hold_time_ms > self.orphan_timeout_ms:
                            # Lock held too long - force release
                            bt.logging.warning(
                                f"[LOCK_SERVER] Force-releasing orphaned lock: {lock_key} "
                                f"(held by {lock_state.holder[:8] if lock_state.holder else 'unknown'}... "
                                f"for {hold_time_ms}ms)"
                            )

                            lock_state.holder = None
                            lock_state.lock_token = None
                            lock_state.acquired_at = None
                            lock_state.lock.release()

                            self.global_metrics['active_locks'] -= 1
                            orphaned_count += 1

            if orphaned_count > 0:
                bt.logging.warning(
                    f"[LOCK_SERVER] Released {orphaned_count} orphaned locks"
                )

    def start(self):
        """Start the lock server and orphan checker"""
        # Start orphan checker thread
        self.orphan_checker_thread = threading.Thread(
            target=self._check_orphaned_locks,
            daemon=True,
            name="LockOrphanChecker"
        )
        self.orphan_checker_thread.start()

        bt.logging.success("PositionLockServer started")

    def shutdown(self):
        """Graceful shutdown"""
        bt.logging.info("Shutting down PositionLockServer...")
        self.shutdown_event.set()

        if self.orphan_checker_thread:
            self.orphan_checker_thread.join(timeout=10.0)

        # Log final metrics
        bt.logging.info(
            f"[LOCK_SERVER] Final metrics: "
            f"acquisitions={self.global_metrics['total_acquisitions']}, "
            f"releases={self.global_metrics['total_releases']}, "
            f"timeouts={self.global_metrics['total_timeouts']}, "
            f"active={self.global_metrics['active_locks']}"
        )

        bt.logging.success("PositionLockServer shutdown complete")


# Singleton instance for IPC server (started separately)
_lock_server_instance = None


def get_lock_server() -> PositionLockServer:
    """Get or create the singleton lock server instance"""
    global _lock_server_instance
    if _lock_server_instance is None:
        _lock_server_instance = PositionLockServer()
        _lock_server_instance.start()
    return _lock_server_instance
