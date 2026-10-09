"""
Unit coverage for server subprocess startup/shutdown under PM2 restarts.

- _reap_server_processes: children that ignore SIGTERM (the forked-handler case) are SIGKILLed
  within one shared deadline, not one timeout per child.
- ignore_termination_signals_in_child: terminate() is a no-op; only the flag or SIGKILL stops a child.
- The orchestrator's SIGTERM handler actually runs shutdown_all_servers (it used to no-op).
- spawn_process readiness: waits for a slow-loading server instead of warning after 30s, fails
  fast on a dead child, and stops waiting on shutdown.
- wait_for_state_tier: core blocks until vanta-state's ports listen.
- ShutdownCoordinator.initialize_fresh: a stale segment's existing mapping (an orphan of a killed
  run) is not reset to "running".

Processes are created with the 'fork' context explicitly. That is the Linux default the validator
runs with; macOS defaults to 'spawn', which would hide the inherited-signal-handler behavior.
"""
import multiprocessing
import os
import signal
import struct
import threading
import time
import uuid
from multiprocessing import shared_memory

import pytest

import shared_objects.rpc.rpc_server_base as rsb
from shared_objects.rpc.port_manager import PortManager
from shared_objects.rpc.rpc_server_base import RPCServerBase, ignore_termination_signals_in_child
from shared_objects.rpc.server_orchestrator import ServerOrchestrator
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator

fork_ctx = (multiprocessing.get_context("fork")
            if "fork" in multiprocessing.get_all_start_methods() else None)
pytestmark = pytest.mark.skipif(fork_ctx is None, reason="needs the fork start method")


# ---------------------------------------------------------------------------------------------
# Child process targets (module level)
# ---------------------------------------------------------------------------------------------

def _inherited_early_return_handler(signum, frame):
    """What a forked child inherits from vanta-state: returns early once shutdown is signaled."""
    return


def _straggler_ignoring_sigterm():
    """A child stuck in shutdown() that ignores SIGTERM: only SIGKILL stops it."""
    ignore_termination_signals_in_child()
    while True:
        time.sleep(0.05)


def _exits_after(delay_s):
    ignore_termination_signals_in_child()
    time.sleep(delay_s)


class _FakeHandle:
    def __init__(self, process):
        self.process = process
        self.monitoring_stopped = False

    def stop_monitoring(self):
        self.monitoring_stopped = True


def _start(target, *args):
    p = fork_ctx.Process(target=target, args=args, daemon=True)
    p.start()
    return p


# ---------------------------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------------------------

def test_child_ignores_terminate_after_signal_reset():
    """Documents the contract: terminate() can't stop a server child; kill() can."""
    old = signal.signal(signal.SIGTERM, _inherited_early_return_handler)
    try:
        p = _start(_straggler_ignoring_sigterm)
    finally:
        signal.signal(signal.SIGTERM, old)
    try:
        time.sleep(0.2)
        p.terminate()
        p.join(0.5)
        assert p.is_alive(), "SIGTERM must be ignored; children stop via the flag or SIGKILL"
    finally:
        p.kill()
        p.join(2)
    assert not p.is_alive()


def test_reap_kills_all_stragglers_against_one_shared_deadline():
    """Two stragglers cost one grace window + one reap window, not 5s each in sequence (the old
    per-handle terminate()+join(5) that overran vanta-state's 9s alarm)."""
    stragglers = [_start(_straggler_ignoring_sigterm) for _ in range(2)]
    polite = _start(_exits_after, 0.1)
    handles = [(f"s{i}", _FakeHandle(p)) for i, p in enumerate(stragglers)] + [("polite", _FakeHandle(polite))]
    try:
        grace_s = 0.5
        t0 = time.monotonic()
        ServerOrchestrator._reap_server_processes(handles, time.time() + grace_s)
        elapsed = time.monotonic() - t0

        assert elapsed < grace_s + ServerOrchestrator._SERVER_KILL_REAP_S + 0.5
        for _, h in handles:
            assert not h.process.is_alive()
            assert h.monitoring_stopped
        assert polite.exitcode == 0  # exited by itself inside the grace window: not killed
        assert all(p.exitcode == -signal.SIGKILL for p in stragglers)
    finally:
        for p in stragglers + [polite]:
            if p.is_alive():
                p.kill()
                p.join(2)


def test_reap_does_not_kill_children_that_exit_within_grace():
    children = [_start(_exits_after, 0.2) for _ in range(3)]
    handles = [(f"c{i}", _FakeHandle(p)) for i, p in enumerate(children)]
    ServerOrchestrator._reap_server_processes(handles, time.time() + 3.0)
    assert [p.exitcode for p in children] == [0, 0, 0]


def test_orchestrator_signal_handler_runs_shutdown(monkeypatch):
    """The handler used to set _shutting_down before calling shutdown_all_servers, which then
    returned immediately: core's SIGTERM never set the flag or reaped its subprocesses."""
    captured = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: captured.setdefault(sig, handler))
    monkeypatch.setattr("atexit.register", lambda fn: None)

    orch = object.__new__(ServerOrchestrator)
    orch._register_cleanup_handlers()
    calls = []
    # Record whether the real method would have proceeded past its _shutting_down guard.
    orch.shutdown_all_servers = lambda: calls.append(orch._shutting_down)

    with pytest.raises(SystemExit):
        captured[signal.SIGTERM](signal.SIGTERM, None)
    assert calls == [False]


# ---------------------------------------------------------------------------------------------
# Startup readiness
# ---------------------------------------------------------------------------------------------

class _FakeProcess:
    """Stands in for multiprocessing.Process inside spawn_process; `script` drives the child."""
    script = ("ready", 0.0)

    def __init__(self, target=None, kwargs=None, daemon=None):
        self._ready_event = kwargs["server_ready"]
        self._alive = False
        self.exitcode = None
        self.pid = 4242

    def start(self):
        self._alive = True
        kind, delay = self.script

        def run():
            time.sleep(delay)
            if kind == "ready":
                self._ready_event.set()
            elif kind == "die":
                self._alive = False
                self.exitcode = 1

        threading.Thread(target=run, daemon=True).start()

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        pass

    def kill(self):
        self._alive = False


class _SpawnTestServer(RPCServerBase):
    service_name = "SpawnTestSvc"
    service_port = 59996


@pytest.fixture
def fake_spawn(monkeypatch):
    monkeypatch.setattr(rsb, "Process", _FakeProcess)
    monkeypatch.setattr(_SpawnTestServer, "SPAWN_READY_PROGRESS_LOG_S", 0.2)
    handles = []
    yield handles
    for h in handles:
        h.stop_monitoring()


def test_spawn_waits_for_a_slow_loading_server(fake_spawn, monkeypatch):
    """A server still loading its state past the old 30s window must be waited for, not
    returned 'maybe not ready' so the first client connect is refused."""
    monkeypatch.setattr(_FakeProcess, "script", ("ready", 1.0))
    monkeypatch.setattr(_SpawnTestServer, "SPAWN_READY_TIMEOUT_S", 10.0)
    t0 = time.monotonic()
    handle = _SpawnTestServer.spawn_process()
    fake_spawn.append(handle)
    assert time.monotonic() - t0 >= 1.0
    assert handle.process._ready_event.is_set()


def test_spawn_fails_fast_when_the_child_dies(fake_spawn, monkeypatch):
    monkeypatch.setattr(_FakeProcess, "script", ("die", 0.2))
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="died during startup"):
        _SpawnTestServer.spawn_process()  # default cap is 900s; must not wait for it
    assert time.monotonic() - t0 < 3.0


def test_spawn_stops_waiting_on_shutdown(fake_spawn, monkeypatch):
    monkeypatch.setattr(_FakeProcess, "script", ("hang", 0.0))
    flag = {"down": False}
    monkeypatch.setattr(rsb.ShutdownCoordinator, "is_shutdown", classmethod(lambda cls: flag["down"]))
    threading.Timer(0.3, lambda: flag.update(down=True)).start()
    t0 = time.monotonic()
    handle = _SpawnTestServer.spawn_process()  # returns the handle so shutdown can reap it
    fake_spawn.append(handle)
    assert time.monotonic() - t0 < 3.0


def test_spawn_alerts_but_continues_at_the_cap(fake_spawn, monkeypatch):
    monkeypatch.setattr(_FakeProcess, "script", ("hang", 0.0))
    sent = []

    class _Slack:
        def send_message(self, msg, level=None):
            sent.append((msg, level))

    handle = _SpawnTestServer.spawn_process(slack_notifier=_Slack(), ready_timeout=0.5)
    fake_spawn.append(handle)
    assert sent and sent[0][1] == "error" and "not ready" in sent[0][0]


# ---------------------------------------------------------------------------------------------
# Core waits for vanta-state
# ---------------------------------------------------------------------------------------------

def test_wait_for_state_tier_blocks_until_every_port_listens(monkeypatch):
    orch = ServerOrchestrator.get_instance()
    calls = {"n": 0}

    def listening(port, host="localhost", timeout=0.1):
        calls["n"] += 1
        return calls["n"] > len(ServerOrchestrator.VANTA_STATE_SERVERS)  # nothing up on the 1st pass

    monkeypatch.setattr(PortManager, "is_port_listening", staticmethod(listening))
    orch.wait_for_state_tier(timeout_s=5, poll_s=0.01)
    assert calls["n"] > len(ServerOrchestrator.VANTA_STATE_SERVERS)


def test_wait_for_state_tier_raises_on_timeout(monkeypatch):
    orch = ServerOrchestrator.get_instance()
    monkeypatch.setattr(PortManager, "is_port_listening", staticmethod(lambda *a, **k: False))
    with pytest.raises(RuntimeError, match="vanta-state servers not listening"):
        orch.wait_for_state_tier(timeout_s=0.1, poll_s=0.01)


# ---------------------------------------------------------------------------------------------
# Stale shutdown flag
# ---------------------------------------------------------------------------------------------

def test_initialize_fresh_does_not_revive_orphans(monkeypatch):
    """An orphan of a killed run still maps the old segment with flag=1. The new run must get a
    fresh segment rather than resetting the old one in place to 0, which would make the orphan
    resume serving on the ports the new run needs."""
    name = f"vt{os.getpid() % 100000}{uuid.uuid4().hex[:6]}"  # macOS caps shm names at 31 chars
    monkeypatch.setattr(ShutdownCoordinator, "_SHM_NAME", name)
    monkeypatch.setattr(ShutdownCoordinator, "_initialized", False)
    monkeypatch.setattr(ShutdownCoordinator, "_shm", None)

    orphan_view = shared_memory.SharedMemory(name=name, create=True, size=8)
    struct.pack_into("q", orphan_view.buf, 0, 1)  # the previous run had signaled shutdown
    try:
        ShutdownCoordinator.initialize_fresh()
        assert ShutdownCoordinator.is_shutdown() is False  # this run starts clean...
        assert struct.unpack_from("q", orphan_view.buf, 0)[0] == 1  # ...and the orphan still exits
    finally:
        ShutdownCoordinator.cleanup()
        orphan_view.close()
