from multiprocessing import Lock as MPLock
from threading import Lock
import bittensor as bt
from multiprocessing import Manager
import time

class PositionLocks:
    """
    Updating positions in the validator is vulnerable to race conditions on a per-miner and per-trade-pair basis. This
    class aims to solve that problem by locking the positions for a given miner and trade pair.

    For multiprocessing mode, uses Manager locks that can be shared across processes.
    """
    def __init__(self, hotkey_to_positions=None, is_backtesting=False, use_ipc=False):
        self.is_backtesting = is_backtesting

        # Create dedicated IPC manager if requested (unless backtesting)
        self.use_ipc = use_ipc and not is_backtesting
        if self.use_ipc:
            ipc_manager = Manager()  # Don't store it - proxy objects are picklable
            bt.logging.info(
                f"PositionLocks: Created dedicated IPC manager "
                f"(PID: {ipc_manager._process.pid})"
            )
            # IPC-backed data structure - proxy objects ARE picklable
            self.locks = ipc_manager.dict()
            # Store lock factory for creating new locks
            self._lock_factory = ipc_manager.Lock
        else:
            # Local (non-IPC) data structure for tests or backtesting
            self.locks = {}
            if is_backtesting:
                self._lock_factory = Lock
            else:
                self._lock_factory = MPLock

        if hotkey_to_positions:
            for hotkey, positions in hotkey_to_positions.items():
                for p in positions:
                    key = (hotkey, p.trade_pair.trade_pair_id)
                    if key not in self.locks:
                        self.locks[key] = self._lock_factory()
        #self.global_lock = Lock()

    def get_lock(self, miner_hotkey:str, trade_pair_id:str):
        #bt.logging.info(f"Getting lock for miner_hotkey [{miner_hotkey}] and trade_pair [{trade_pair}].")
        lock_key = (miner_hotkey, trade_pair_id)
        lock_lookup_start = time.perf_counter()
        ret = self.locks.get(lock_key, None)
        lock_lookup_ms = (time.perf_counter() - lock_lookup_start) * 1000

        if ret is None:
            ret = self._lock_factory()
            lock_creation_start = time.perf_counter()
            self.locks[lock_key] = ret
            lock_creation_ms = (time.perf_counter() - lock_creation_start) * 1000
            bt.logging.trace(f"[LOCK_MGR] Created new lock for {miner_hotkey[:8]}.../{trade_pair_id} (lookup={lock_lookup_ms:.2f}ms, creation={lock_creation_ms:.2f}ms)")
        else:
            bt.logging.trace(f"[LOCK_MGR] Retrieved existing lock for {miner_hotkey[:8]}.../{trade_pair_id} (lookup={lock_lookup_ms:.2f}ms)")

        return ret

    #def cleanup_locks(self, active_miner_hotkeys):
    #    with self.global_lock:  # Ensure thread-safe modification of the locks dictionary
    #        keys_to_delete = [key for key in self.locks.keys() if key[0] not in active_miner_hotkeys]
    #        for key in keys_to_delete:
    #            del self.locks[key]

