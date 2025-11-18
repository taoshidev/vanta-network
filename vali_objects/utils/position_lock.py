from multiprocessing import Manager
from threading import Lock
import bittensor as bt
from typing import Tuple, Optional, Dict


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
    IPC-based locks using multiprocessing.Manager (legacy fallback).
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
        ret = self.locks.get(lock_key, None)

        if ret is None:
            ret = self._lock_factory()
            self.locks[lock_key] = ret
            bt.logging.trace(
                f"[LOCK_MGR] Created new lock for {miner_hotkey[:8]}.../{trade_pair_id}"
            )
        else:
            bt.logging.trace(
                f"[LOCK_MGR] Retrieved existing lock for {miner_hotkey[:8]}.../{trade_pair_id}"
            )

        return ret


class PositionLocks:
    """
    Facade for position lock management with multiple backend modes.

    Supports three modes:
    - 'local': Threading locks (fastest, single process only)
    - 'ipc': Multiprocessing Manager locks (legacy, multi-process)
    - 'rpc': Dedicated lock server (recommended for production)

    Usage:
        # Local mode (tests, single process)
        locks = PositionLocks(mode='local')

        # IPC mode (legacy fallback)
        locks = PositionLocks(mode='ipc')

        # RPC mode (recommended for production)
        locks = PositionLocks(mode='rpc')

        # Use the lock
        with locks.get_lock(miner_hotkey, trade_pair_id):
            # ... do work ...
    """

    def __init__(self, hotkey_to_positions=None, is_backtesting=False,
                 use_ipc=False, mode: Optional[str] = None,
                 running_unit_tests: bool = False):
        """
        Initialize PositionLocks with specified mode.

        Args:
            hotkey_to_positions: Initial positions to create locks for (not used in RPC mode)
            is_backtesting: If True, use local mode (legacy param)
            use_ipc: If True, use IPC mode (legacy param)
            mode: Explicit mode selection: 'local', 'ipc', or 'rpc'
            running_unit_tests: Whether running in unit test mode
        """
        # Determine mode from parameters
        if mode is None:
            if is_backtesting or running_unit_tests:
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
            # Import here to avoid circular dependency
            from vali_objects.utils.position_lock_server import PositionLockManagerClient

            self.impl = PositionLockManagerClient(running_unit_tests=running_unit_tests)
            bt.logging.info("PositionLocks: Using RPC mode (dedicated lock server)")

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local', 'ipc', or 'rpc'")

    def get_lock(self, miner_hotkey: str, trade_pair_id: str):
        """
        Get a lock for the given key.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID

        Returns:
            Lock object that can be used as a context manager

        Usage:
            with position_locks.get_lock(hotkey, pair_id):
                # ... do work while holding lock ...
        """
        return self.impl.get_lock(miner_hotkey, trade_pair_id)

    def health_check(self, current_time_ms: Optional[int] = None) -> bool:
        """
        Perform health check on the lock service (RPC mode only).

        Args:
            current_time_ms: Current timestamp in milliseconds

        Returns:
            bool: True if healthy, False otherwise
        """
        if hasattr(self.impl, 'health_check'):
            return self.impl.health_check(current_time_ms)
        return True  # Local/IPC modes are always "healthy"

    def shutdown(self):
        """Shutdown the lock service (RPC mode only)."""
        if hasattr(self.impl, 'shutdown'):
            self.impl.shutdown()
