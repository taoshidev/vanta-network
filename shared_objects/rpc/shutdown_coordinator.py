"""
ShutdownCoordinator - Cross-process shutdown flag using shared memory.

No RPC. No refresher thread. Processes can poll whenever they want.
"""

import struct
import time
from multiprocessing import shared_memory
from typing import Optional
import bittensor as bt
from time_util.time_util import TimeUtil


class ShutdownCoordinator:
    _SHM_NAME = "global_shutdown_flag"
    _SHM_SIZE = 8

    _initialized = False
    _shm = None

    _shutdown_reason: Optional[str] = None
    _shutdown_time_ms: Optional[int] = None

    @classmethod
    def initialize(cls, reset_on_attach: bool = False):
        """
        Initialize ShutdownCoordinator by creating or attaching to shared memory.

        Args:
            reset_on_attach: If True and shared memory already exists, reset the shutdown
                           flag to 0 (not shutdown). This should be True for the main
                           validator process to clear any stale shutdown state from
                           previous runs. Child processes should leave this False.
        """
        if cls._initialized:
            return

        try:
            # Try creating the shared memory block
            cls._shm = shared_memory.SharedMemory(
                name=cls._SHM_NAME, create=True, size=cls._SHM_SIZE
            )
            # Initialize flag to 0
            struct.pack_into("q", cls._shm.buf, 0, 0)
            bt.logging.info("[ShutdownCoordinator] Created shared-memory shutdown flag.")
        except FileExistsError:
            # Already exists (another process created it or stale from previous run)
            cls._shm = shared_memory.SharedMemory(name=cls._SHM_NAME)

            if reset_on_attach:
                # Reset flag to 0 (clear stale shutdown state from crashed/killed processes)
                struct.pack_into("q", cls._shm.buf, 0, 0)
                bt.logging.info("[ShutdownCoordinator] Attached to existing shutdown flag and reset to 0.")
            else:
                # Read current state for logging
                current_value = struct.unpack_from("q", cls._shm.buf, 0)[0]
                bt.logging.info(
                    f"[ShutdownCoordinator] Attached to existing shutdown flag (current value: {current_value})."
                )

        cls._initialized = True

    @classmethod
    def _read_flag(cls) -> int:
        cls.initialize()
        return struct.unpack_from("q", cls._shm.buf, 0)[0]

    @classmethod
    def _write_flag(cls, value: int):
        cls.initialize()
        struct.pack_into("q", cls._shm.buf, 0, value)

    @classmethod
    def is_shutdown(cls) -> bool:
        """Read shared memory directly."""
        return cls._read_flag() == 1

    @classmethod
    def signal_shutdown(cls, reason: str = "User requested shutdown"):
        """Writes shutdown flag to shared memory."""

        cls.initialize()

        # If already shutdown, no need to re-write
        if cls.is_shutdown():
            return

        cls._write_flag(1)
        cls._shutdown_reason = reason
        cls._shutdown_time_ms = TimeUtil.now_in_millis()

        bt.logging.warning(f"[SHUTDOWN] Shutdown signaled: {reason}")

    @classmethod
    def wait_for_shutdown(cls, timeout: Optional[float] = None) -> bool:
        """
        Poll shared memory at 100ms intervals (customizable).
        """
        cls.initialize()

        start = time.time()
        while True:
            if cls.is_shutdown():
                return True

            if timeout is not None and (time.time() - start) >= timeout:
                return False

            time.sleep(0.1)  # polling interval

    @classmethod
    def reset(cls):
        """Reset for tests only."""
        cls.initialize()
        cls._write_flag(0)
        cls._shutdown_reason = None
        cls._shutdown_time_ms = None

    @classmethod
    def cleanup_stale_memory(cls):
        """
        Cleanup stale shared memory from previous crashed/killed processes.

        This is useful as a defensive measure during startup to ensure a clean state.
        Safe to call even if shared memory doesn't exist.

        Usage:
            # At validator startup, before initializing
            ShutdownCoordinator.cleanup_stale_memory()
            ShutdownCoordinator.initialize(reset_on_attach=True)
        """
        try:
            # Try to unlink existing shared memory
            shm = shared_memory.SharedMemory(name=cls._SHM_NAME)
            shm.close()
            shm.unlink()
            bt.logging.info("[ShutdownCoordinator] Cleaned up stale shared memory")
        except FileNotFoundError:
            # No stale memory to clean up
            bt.logging.debug("[ShutdownCoordinator] No stale shared memory to clean up")
        except Exception as e:
            # Log but don't fail - initialize() will handle it
            bt.logging.warning(f"[ShutdownCoordinator] Error cleaning up shared memory: {e}")

    @classmethod
    def get_shutdown_info(cls) -> dict:
        cls.initialize()
        return {
            "is_shutdown": cls.is_shutdown(),
            "reason": cls._shutdown_reason,
            "shutdown_time_ms": cls._shutdown_time_ms,
        }


# Convenience functions
def is_shutdown() -> bool:
    return ShutdownCoordinator.is_shutdown()

def signal_shutdown(reason: str = "User requested shutdown"):
    ShutdownCoordinator.signal_shutdown(reason)