from multiprocessing import Lock as MPLock
from threading import Lock
import bittensor as bt
from multiprocessing import Manager
import time
import uuid
from typing import Tuple, Optional, Dict
from contextlib import contextmanager


class LockContextManager:
    """Context manager for lock acquisition/release"""

    def __init__(self, lock_impl, lock_key: Tuple[str, str]):
        """
        Initialize context manager.

        Args:
            lock_impl: The lock implementation (LocalLocks, IPCLocks, or RPCLocks)
            lock_key: (miner_hotkey, trade_pair_id) tuple
        """
        self.lock_impl = lock_impl
        self.lock_key = lock_key
        self.lock_obj = None
        self.lock_token = None
        self.acquired_at = None

    def __enter__(self):
        """Acquire the lock"""
        self.acquired_at = time.perf_counter()

        if hasattr(self.lock_impl, 'acquire_lock'):
            # RPC mode - returns lock token
            result = self.lock_impl.acquire_lock(self.lock_key)
            if not result['success']:
                raise TimeoutError(result['message'])
            self.lock_token = result['lock_token']
        else:
            # Local or IPC mode - traditional lock object
            self.lock_obj = self.lock_impl.get_lock(*self.lock_key)
            self.lock_obj.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock"""
        try:
            if self.lock_token:
                # RPC mode
                result = self.lock_impl.release_lock(self.lock_key, self.lock_token)
                if not result['success']:
                    bt.logging.error(f"[LOCK] Release failed: {result['message']}")
            elif self.lock_obj:
                # Local or IPC mode
                self.lock_obj.__exit__(exc_type, exc_val, exc_tb)
        finally:
            elapsed_ms = (time.perf_counter() - self.acquired_at) * 1000 if self.acquired_at else 0
            bt.logging.trace(f"[LOCK] Released {self.lock_key} after {elapsed_ms:.2f}ms")


class LocalLocks:
    """
    Local threading-based locks for single process / testing.
    Fastest option but only works within a single process.
    """

    def __init__(self, hotkey_to_positions=None):
        self.locks: Dict[Tuple[str, str], Lock] = {}
        self._lock_factory = Lock

        if hotkey_to_positions:
            for hotkey, positions in hotkey_to_positions.items():
                for p in positions:
                    key = (hotkey, p.trade_pair.trade_pair_id)
                    if key not in self.locks:
                        self.locks[key] = self._lock_factory()

    def get_lock(self, miner_hotkey: str, trade_pair_id: str):
        """Get or create a lock for the given key"""
        lock_key = (miner_hotkey, trade_pair_id)
        if lock_key not in self.locks:
            self.locks[lock_key] = self._lock_factory()
        return self.locks[lock_key]


class IPCLocks:
    """
    IPC-based locks using multiprocessing.Manager (current implementation).
    Works across processes but has Manager overhead.
    """

    def __init__(self, hotkey_to_positions=None):
        # Create dedicated IPC manager
        ipc_manager = Manager()
        bt.logging.info(
            f"IPCLocks: Created dedicated IPC manager (PID: {ipc_manager._process.pid})"
        )

        # IPC-backed data structure - proxy objects are picklable
        self.locks = ipc_manager.dict()
        self._lock_factory = ipc_manager.Lock

        if hotkey_to_positions:
            for hotkey, positions in hotkey_to_positions.items():
                for p in positions:
                    key = (hotkey, p.trade_pair.trade_pair_id)
                    if key not in self.locks:
                        self.locks[key] = self._lock_factory()

    def get_lock(self, miner_hotkey: str, trade_pair_id: str):
        """Get or create a lock for the given key"""
        lock_key = (miner_hotkey, trade_pair_id)
        lock_lookup_start = time.perf_counter()
        ret = self.locks.get(lock_key, None)
        lock_lookup_ms = (time.perf_counter() - lock_lookup_start) * 1000

        if ret is None:
            ret = self._lock_factory()
            lock_creation_start = time.perf_counter()
            self.locks[lock_key] = ret
            lock_creation_ms = (time.perf_counter() - lock_creation_start) * 1000
            bt.logging.trace(
                f"[LOCK_MGR] Created new lock for {miner_hotkey[:8]}.../{trade_pair_id} "
                f"(lookup={lock_lookup_ms:.2f}ms, creation={lock_creation_ms:.2f}ms)"
            )
        else:
            bt.logging.trace(
                f"[LOCK_MGR] Retrieved existing lock for {miner_hotkey[:8]}.../{trade_pair_id} "
                f"(lookup={lock_lookup_ms:.2f}ms)"
            )

        return ret


class RPCLocks:
    """
    RPC-based locks using PositionLockServer (new implementation).
    Best performance for production multi-process validators.
    """

    def __init__(self, server_address: str = 'localhost', server_port: int = 8766):
        """
        Initialize RPC lock client.

        Args:
            server_address: Address of PositionLockServer
            server_port: Port of PositionLockServer
        """
        self.server_address = server_address
        self.server_port = server_port
        self.client_id = str(uuid.uuid4())

        # Import here to avoid circular dependency
        from vali_objects.utils.position_lock_server import get_lock_server

        # For now, use in-process server (same process, different thread)
        # TODO: Make this actual RPC when running as separate process
        self.server = get_lock_server()

        bt.logging.info(
            f"RPCLocks: Connected to PositionLockServer (client_id={self.client_id[:8]}...)"
        )

    def acquire_lock(self, lock_key: Tuple[str, str], timeout_ms: int = 30000) -> dict:
        """
        Acquire a lock via RPC.

        Args:
            lock_key: (miner_hotkey, trade_pair_id) tuple
            timeout_ms: Maximum time to wait

        Returns:
            dict with success, lock_token, wait_ms, message
        """
        return self.server.acquire_lock(lock_key, self.client_id, timeout_ms)

    def release_lock(self, lock_key: Tuple[str, str], lock_token: str) -> dict:
        """
        Release a lock via RPC.

        Args:
            lock_key: (miner_hotkey, trade_pair_id) tuple
            lock_token: Token from acquire_lock

        Returns:
            dict with success, hold_ms, message
        """
        return self.server.release_lock(lock_key, lock_token)

    def get_metrics(self, lock_key: Optional[Tuple[str, str]] = None) -> dict:
        """Get lock metrics from server"""
        return self.server.get_metrics(lock_key)


class PositionLocks:
    """
    Facade for position lock management with multiple backend modes.

    Supports three modes:
    - 'local': Threading locks (fastest, single process only)
    - 'ipc': Multiprocessing Manager locks (current, multi-process)
    - 'rpc': Dedicated lock server (new, best for production)

    Usage:
        # Local mode (tests, single process)
        locks = PositionLocks(mode='local')

        # IPC mode (current implementation, fallback)
        locks = PositionLocks(mode='ipc')

        # RPC mode (new, recommended for production)
        locks = PositionLocks(mode='rpc')

        # Use the lock
        with locks.get_lock(miner_hotkey, trade_pair_id):
            # ... do work ...
    """

    def __init__(self, hotkey_to_positions=None, is_backtesting=False,
                 use_ipc=False, mode: Optional[str] = None,
                 server_address: str = 'localhost', server_port: int = 8766):
        """
        Initialize PositionLocks with specified mode.

        Args:
            hotkey_to_positions: Initial positions to create locks for
            is_backtesting: If True, use local mode (legacy param)
            use_ipc: If True, use IPC mode (legacy param)
            mode: Explicit mode selection: 'local', 'ipc', or 'rpc'
            server_address: Address for RPC mode
            server_port: Port for RPC mode
        """
        # Determine mode from parameters
        if mode is None:
            if is_backtesting:
                mode = 'local'
            elif use_ipc:
                mode = 'ipc'
            else:
                mode = 'local'

        self.mode = mode
        self.is_backtesting = is_backtesting

        # Create appropriate implementation
        if mode == 'local':
            self.impl = LocalLocks(hotkey_to_positions)
            bt.logging.info("PositionLocks: Using LOCAL mode (threading locks)")

        elif mode == 'ipc':
            self.impl = IPCLocks(hotkey_to_positions)
            bt.logging.info("PositionLocks: Using IPC mode (multiprocessing Manager)")

        elif mode == 'rpc':
            self.impl = RPCLocks(server_address, server_port)
            bt.logging.info(
                f"PositionLocks: Using RPC mode (lock server at {server_address}:{server_port})"
            )

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local', 'ipc', or 'rpc'")

    def get_lock(self, miner_hotkey: str, trade_pair_id: str):
        """
        Get a context manager for the lock.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID

        Returns:
            Context manager that acquires/releases the lock

        Usage:
            with position_locks.get_lock(hotkey, pair_id):
                # ... do work while holding lock ...
        """
        lock_key = (miner_hotkey, trade_pair_id)
        return LockContextManager(self.impl, lock_key)

    def get_metrics(self, lock_key: Optional[Tuple[str, str]] = None) -> Optional[dict]:
        """
        Get metrics for lock performance (RPC mode only).

        Args:
            lock_key: Optional specific lock to get metrics for

        Returns:
            dict with metrics or None if not supported in current mode
        """
        if hasattr(self.impl, 'get_metrics'):
            return self.impl.get_metrics(lock_key)
        return None
