"""
Lock-order tests for MinerAccountManager (order-execution freeze on testnet, 2026-10-16).

MinerAccountManager has two locks: _accounts_lock (an RLock guarding self.accounts) and _disk_lock
(serializing writes of miner_account_sizes.json). The only safe order is _accounts_lock, then
_disk_lock. Before the fix the hourly snapshot daemon took them the other way round:
take_account_snapshot -> _save_accounts_to_disk held _disk_lock and then asked for _accounts_lock in
accounts_dict(), while process_fees (like every RPC mutator) held _accounts_lock and then asked for
_disk_lock in _save_accounts_to_disk(). When the two met, both waited forever, every other account
RPC queued behind _accounts_lock, and every market order timed out after 60 s.

The lock waits in these tests are bounded (LockMonitor), so a deadlock fails the test after
LOCK_WAIT_S instead of hanging the suite, and the stuck threads unwind.
"""
import ast
import inspect
import json
import os
import shutil
import tempfile
import threading
import traceback
import unittest
from typing import Optional
from unittest import mock

from shared_objects.rpc.server_registry import ServerRegistry
from time_util.time_util import TimeUtil
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account import miner_account_manager as miner_account_manager_module
from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.miner_account.miner_account_server import MinerAccountServer
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import RPCConnectionMode

ACCOUNTS = "_accounts_lock"
DISK = "_disk_lock"

HOTKEY = "5LockOrderTestMinerHotkey"
# Thread names from the py-spy dump of the frozen MinerAccountServer.
DAEMON_THREAD = "MinerAccountServer_Daemon"
FEES_THREAD = "rpc-MinerAccountServer_0"

LOCK_WAIT_S = 5.0    # these locks are held for milliseconds here; a wait this long is a deadlock
STEP_WAIT_S = 10.0   # a hook waiting for the other thread to reach its step
JOIN_S = 30.0


class LockWaitTimeout(RuntimeError):
    """Raised in a thread whose wait for a monitored lock exceeded LockMonitor.wait_timeout_s."""


def _call_path() -> str:
    """The miner_account frames of the current stack, outermost first (for failure messages)."""
    frames = [f for f in traceback.extract_stack() if "miner_account" in f.filename.replace("\\", "/").split("/")]
    return " -> ".join(f"{f.name}:{f.lineno}" for f in frames)


class _MonitoredLock:
    """Drop-in for the manager's lock objects (the manager only uses them as context managers)."""

    def __init__(self, inner, name: str, monitor: "LockMonitor"):
        self._inner = inner
        self.name = name
        self._monitor = monitor

    def __enter__(self):
        self._monitor._before_acquire(self.name)
        if not self._inner.acquire(timeout=self._monitor.wait_timeout_s):
            self._monitor._wait_timed_out(self.name)
        self._monitor._acquired(self.name)
        return self

    def __exit__(self, *exc_info):
        self._monitor._released(self.name)
        self._inner.release()
        return False


class LockMonitor:
    """Instruments a MinerAccountManager's two locks for one test.

    - Records every acquisition of _accounts_lock by a thread that holds _disk_lock but not
      _accounts_lock: the inverted order. (Re-entering the RLock a thread already holds cannot block,
      so it does not count.)
    - Bounds every lock wait: a wait longer than wait_timeout_s is recorded and raises LockWaitTimeout
      in the waiting thread.
    - Runs one-shot hooks keyed by (thread name, lock name, "before" | "after") so a test can force an
      interleaving. "before" runs just before the acquire, "after" runs holding the lock.
    """

    def __init__(self, manager: MinerAccountManager, wait_timeout_s: float = LOCK_WAIT_S):
        self.wait_timeout_s = wait_timeout_s
        self.order_violations = []
        self.wait_timeouts = []
        self._hooks = {}
        self._local = threading.local()
        self._mutex = threading.Lock()
        manager._accounts_lock = _MonitoredLock(manager._accounts_lock, ACCOUNTS, self)
        manager._disk_lock = _MonitoredLock(manager._disk_lock, DISK, self)

    def on(self, thread_name: str, lock_name: str, phase: str, hook) -> None:
        self._hooks[(thread_name, lock_name, phase)] = hook

    def _held(self) -> dict:
        if not hasattr(self._local, "held"):
            self._local.held = {}
        return self._local.held

    def _run_hook(self, lock_name: str, phase: str) -> None:
        with self._mutex:
            hook = self._hooks.pop((threading.current_thread().name, lock_name, phase), None)
        if hook is not None:
            hook()

    def _before_acquire(self, lock_name: str) -> None:
        held = self._held()
        if lock_name == ACCOUNTS and held.get(DISK) and not held.get(ACCOUNTS):
            with self._mutex:
                self.order_violations.append(
                    f"{threading.current_thread().name} took _accounts_lock while holding _disk_lock: {_call_path()}"
                )
        self._run_hook(lock_name, "before")

    def _acquired(self, lock_name: str) -> None:
        held = self._held()
        held[lock_name] = held.get(lock_name, 0) + 1
        self._run_hook(lock_name, "after")

    def _released(self, lock_name: str) -> None:
        held = self._held()
        held[lock_name] -= 1
        if not held[lock_name]:
            del held[lock_name]

    def _wait_timed_out(self, lock_name: str) -> None:
        message = (
            f"{threading.current_thread().name} waited over {self.wait_timeout_s}s for {lock_name} "
            f"holding {sorted(self._held()) or 'nothing'}: {_call_path()}"
        )
        with self._mutex:
            self.wait_timeouts.append(message)
        raise LockWaitTimeout(message)


def _start_thread(name: str, target, errors: list) -> threading.Thread:
    def run():
        try:
            target()
        except Exception as e:
            errors.append(f"{name}: {e!r}")

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


class _MinerAccountFilesInTempDir(unittest.TestCase):
    """Points the manager's files at a temp dir so these tests never touch the shared test files."""

    def setUp(self):
        tmp_dir = tempfile.mkdtemp(prefix="miner_account_lock_order_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        self.accounts_file = os.path.join(tmp_dir, "miner_account_sizes.json")
        self._patch(ValiBkpUtils, "get_miner_account_sizes_file_location", return_value=self.accounts_file)
        self._patch(ValiBkpUtils, "get_asset_selections_file_location",
                    return_value=os.path.join(tmp_dir, "asset_selections.json"))
        # Snapshots also append to per-miner .jsonl logs, after the save; not what these tests are about.
        self._patch(ValiBkpUtils, "log_snapshot")

    def _patch(self, target, attribute: str, **kwargs):
        patcher = mock.patch.object(target, attribute, **kwargs)
        patched = patcher.start()
        self.addCleanup(patcher.stop)
        return patched

    def make_manager(self) -> MinerAccountManager:
        manager = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        manager.set_miner_account_size(HOTKEY, collateral_balance_theta=100.0, timestamp_ms=TimeUtil.now_in_millis())
        return manager

    def make_server(self) -> MinerAccountServer:
        # In-process only: keep it out of the process-wide registry other tests share.
        with mock.patch.object(ServerRegistry, "register"):
            server = MinerAccountServer(
                running_unit_tests=True, start_server=False, connection_mode=RPCConnectionMode.LOCAL
            )
        server._manager.set_miner_account_size(
            HOTKEY, collateral_balance_theta=100.0, timestamp_ms=TimeUtil.now_in_millis()
        )
        return server

    def fees_on_disk(self, hotkey: str) -> float:
        with open(self.accounts_file) as f:
            return json.load(f)[hotkey][-1]["total_fees_paid"]


class TestMinerAccountLockOrderDeadlock(_MinerAccountFilesInTempDir):

    def test_hourly_snapshot_racing_process_fees_does_not_deadlock(self):
        """Replays the 2026-10-16 freeze through the same server entry points as the py-spy stacks.

        MinerAccountServer_Daemon: run_daemon_iteration -> take_account_snapshot -> _save_accounts_to_disk.
        rpc-MinerAccountServer_0: process_fees -> manager.process_fees -> _save_accounts_to_disk.

        The daemon is held where it owns _disk_lock until the fee thread owns _accounts_lock; then
        each goes on. With the old order the daemon next asks for _accounts_lock and the fee thread
        for _disk_lock: deadlock, both lock waits time out. With one lock order the daemon never
        asks for _accounts_lock under _disk_lock, so both calls finish and the fee is persisted.
        """
        server = self.make_server()
        manager = server._manager
        monitor = LockMonitor(manager)

        daemon_holds_disk = threading.Event()
        fees_hold_accounts = threading.Event()

        def daemon_has_disk_lock():
            daemon_holds_disk.set()
            fees_hold_accounts.wait(STEP_WAIT_S)

        monitor.on(DAEMON_THREAD, DISK, "after", daemon_has_disk_lock)
        monitor.on(FEES_THREAD, ACCOUNTS, "before", lambda: daemon_holds_disk.wait(STEP_WAIT_S))
        monitor.on(FEES_THREAD, ACCOUNTS, "after", fees_hold_accounts.set)

        server._first_snapshot_iteration = False  # the first daemon pass only schedules the next hour
        errors = []
        daemon = _start_thread(DAEMON_THREAD, server.run_daemon_iteration, errors)
        fees = _start_thread(FEES_THREAD, lambda: server.process_fees({HOTKEY: 12.5}), errors)
        daemon.join(JOIN_S)
        fees.join(JOIN_S)

        self.assertEqual(
            monitor.wait_timeouts + monitor.order_violations, [],
            "deadlock: the snapshot daemon and process_fees each waited for the lock the other held",
        )
        self.assertTrue(daemon_holds_disk.is_set() and fees_hold_accounts.is_set(),
                        "the test did not reach the interleaving it replays")
        self.assertFalse(daemon.is_alive() or fees.is_alive(), "a thread is still blocked")
        self.assertEqual(errors, [])
        self.assertAlmostEqual(manager.accounts[HOTKEY].total_fees_paid, 12.5)
        self.assertAlmostEqual(self.fees_on_disk(HOTKEY), 12.5)


class TestMinerAccountSnapshotOrdering(_MinerAccountFilesInTempDir):
    """The snapshot is cut under _accounts_lock and written under _disk_lock alone, so two savers can
    reach the disk in the opposite order to the one they cut their snapshots in. The older one must
    never land last."""

    def test_snapshot_cut_before_a_newer_save_does_not_overwrite_it(self):
        manager = self.make_manager()
        monitor = LockMonitor(manager)
        daemon_cut_snapshot = threading.Event()
        fees_saved = threading.Event()

        def daemon_about_to_write():
            daemon_cut_snapshot.set()
            fees_saved.wait(STEP_WAIT_S)

        monitor.on(DAEMON_THREAD, DISK, "before", daemon_about_to_write)

        errors = []
        daemon = _start_thread(DAEMON_THREAD, manager.take_account_snapshot, errors)
        self.assertTrue(daemon_cut_snapshot.wait(STEP_WAIT_S))
        manager.process_fees({HOTKEY: 7.0})  # cuts a newer snapshot and writes it first
        fees_saved.set()
        daemon.join(JOIN_S)

        self.assertFalse(daemon.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(monitor.wait_timeouts, [])
        self.assertAlmostEqual(
            self.fees_on_disk(HOTKEY), 7.0,
            msg="the daemon's older snapshot overwrote the newer one written by process_fees",
        )

    def test_reload_discards_a_snapshot_cut_before_it(self):
        """re_init_account_sizes makes memory equal the file again; a snapshot cut before the reload
        is older than both and must not be written after it."""
        manager = self.make_manager()
        manager.process_fees({HOTKEY: 1.0})
        manager.accounts[HOTKEY].total_fees_paid = 99.0  # in memory only; the reload drops it
        monitor = LockMonitor(manager)
        daemon_cut_snapshot = threading.Event()
        reloaded = threading.Event()

        def daemon_about_to_write():
            daemon_cut_snapshot.set()
            reloaded.wait(STEP_WAIT_S)

        monitor.on(DAEMON_THREAD, DISK, "before", daemon_about_to_write)

        errors = []
        daemon = _start_thread(DAEMON_THREAD, manager.take_account_snapshot, errors)
        self.assertTrue(daemon_cut_snapshot.wait(STEP_WAIT_S))
        manager.re_init_account_sizes()
        reloaded.set()
        daemon.join(JOIN_S)

        self.assertFalse(daemon.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(monitor.wait_timeouts, [])
        self.assertAlmostEqual(manager.accounts[HOTKEY].total_fees_paid, 1.0)
        self.assertAlmostEqual(self.fees_on_disk(HOTKEY), 1.0,
                               msg="a snapshot cut before the reload was written over the reloaded state")


class TestMinerAccountLockOrderGuards(_MinerAccountFilesInTempDir):

    def test_every_persisting_entry_point_takes_accounts_lock_before_disk_lock(self):
        """Runtime guard: drive every method that loads or saves through the monitored locks."""
        manager = self.make_manager()
        monitor = LockMonitor(manager)
        now_ms = TimeUtil.now_in_millis()

        manager.set_miner_account_size(HOTKEY, collateral_balance_theta=200.0, timestamp_ms=now_ms)
        manager.set_miner_account_size("5LockOrderSecondMiner", collateral_balance_theta=50.0, timestamp_ms=now_ms)
        with mock.patch.object(manager, "verify_broadcast_sender", return_value=True):
            manager.receive_collateral_record_update(
                {"hotkey": HOTKEY, "account_size": 1234.0, "account_size_theta": 300.0, "update_time_ms": now_ms + 1},
                sender_hotkey="mothership",
            )
        manager.take_account_snapshot()                                    # the hourly daemon
        manager.take_account_snapshot(hotkey=HOTKEY, reset_snapshot=True)  # the validator REST endpoint
        manager.reset_account(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        manager.update_unrealized_pnl({HOTKEY: 3.0})
        manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        manager.set_miner_buckets({HOTKEY: MinerBucket.SUBACCOUNT_FUNDED})
        manager.set_hl_address(HOTKEY, None)
        manager.set_leverage_tier(HOTKEY, None)
        manager.update_asset_selection(HOTKEY, MinerAssetClass.CRYPTO)
        manager.process_order_buy(HOTKEY, order_value_usd=10.0, borrowed_amount=0.0, fee_usd=0.01)
        manager.process_order_sell(HOTKEY, entry_value_usd=10.0, realized_pnl=0.5, loan_repaid=0.0, fee_usd=0.01)
        manager.process_fees({HOTKEY: 0.5})
        manager.process_dividend_income({HOTKEY: 0.25})
        manager.rebuild_account_state_from_positions(HOTKEY, [])
        manager.sync_miner_account_sizes_data(manager.accounts_dict())
        manager.re_init_account_sizes()
        manager.delete_miner_account_size("5LockOrderSecondMiner")

        self.assertEqual(monitor.order_violations, [])
        self.assertEqual(monitor.wait_timeouts, [])
        # Everything above was saved: the file round-trips to what is in memory.
        reloaded = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        self.assertEqual(reloaded.accounts_dict(), manager.accounts_dict())

    def test_no_method_takes_accounts_lock_inside_disk_lock(self):
        """Static guard over MinerAccountManager's source: nothing inside a `with self._disk_lock`
        block may take _accounts_lock, directly or through another method of the class."""
        tree = ast.parse(inspect.getsource(miner_account_manager_module))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MinerAccountManager")
        methods = {fn.name: fn for fn in cls.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))}

        # Methods that take _accounts_lock themselves or through self.<method>() calls.
        takes_accounts = {name for name, fn in methods.items() if _acquires(fn, ACCOUNTS)}
        while True:
            callers = {name for name, fn in methods.items() if _self_method_calls(fn) & takes_accounts}
            if callers <= takes_accounts:
                break
            takes_accounts |= callers
        self.assertIn("accounts_dict", takes_accounts)

        violations = []
        disk_blocks = 0
        for name, fn in methods.items():
            for node in ast.walk(fn):
                if not isinstance(node, (ast.With, ast.AsyncWith)):
                    continue
                locks = [_lock_name(item.context_expr) for item in node.items]
                if DISK not in locks:
                    continue
                disk_blocks += 1
                if ACCOUNTS in locks[locks.index(DISK) + 1:]:
                    violations.append(f"{name}:{node.lineno} takes _accounts_lock after _disk_lock in one `with`")
                for stmt in node.body:
                    if _acquires(stmt, ACCOUNTS):
                        violations.append(f"{name}:{stmt.lineno} takes _accounts_lock inside `with self._disk_lock`")
                    for callee in sorted(_self_method_calls(stmt) & takes_accounts):
                        violations.append(
                            f"{name}:{stmt.lineno} calls self.{callee}(), which takes _accounts_lock, "
                            f"inside `with self._disk_lock`"
                        )

        self.assertGreater(disk_blocks, 0, "no `with self._disk_lock` found; update this guard if the lock moved")
        self.assertEqual(violations, [])


def _lock_name(expr) -> Optional[str]:
    """ACCOUNTS or DISK for `self._accounts_lock` / `self._disk_lock`, else None."""
    if (isinstance(expr, ast.Attribute) and expr.attr in (ACCOUNTS, DISK)
            and isinstance(expr.value, ast.Name) and expr.value.id == "self"):
        return expr.attr
    return None


def _acquires(node, lock: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, (ast.With, ast.AsyncWith)) and any(_lock_name(i.context_expr) == lock for i in sub.items):
            return True
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "acquire"
                and _lock_name(sub.func.value) == lock):
            return True
    return False


def _self_method_calls(node) -> set:
    return {
        sub.func.attr for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "self"
    }


if __name__ == "__main__":
    unittest.main()
