# developer: Taoshidev
# Copyright (c) 2026 Taoshi Inc
"""
Tests for the bounded RPC call layer (RPCClientBase + _BoundedProxy).

Regression context: typed clients invoke RPC methods directly on self._server,
and the raw BaseManager proxy call blocks in conn.recv() with NO timeout. A
wedged (alive but unresponsive) service therefore pinned the calling thread
forever — in the REST process that permanently drained Waitress's 32-thread
pool and took the whole API down until restart. The _BoundedProxy facade frees
the caller after rpc_call_timeout_s with RPCCallTimeoutError instead.

These tests run against a REAL multiprocessing.managers.BaseManager server
(in-process, daemon thread) so the exact production proxy semantics —
thread-local connections, remote exception propagation — are exercised.
"""
import threading
import time
import unittest
from multiprocessing.managers import BaseManager

from shared_objects.rpc.rpc_client_base import (
    RPCCallTimeoutError,
    RPCClientBase,
)
from vali_objects.vali_config import RPCConnectionMode

_AUTHKEY = b"rpc-timeout-test"


class _WedgeableService:
    """Test service whose wedge_rpc blocks until released (or 30s safety cap)."""

    def __init__(self):
        self._gate = threading.Event()

    def echo_rpc(self, value):
        return value

    def raise_rpc(self):
        raise ValueError("boom from remote")

    def wedge_rpc(self):
        self._gate.wait(30)
        return "unwedged"

    def release_rpc(self):
        self._gate.set()
        return True

    def rearm_rpc(self):
        self._gate.clear()
        return True


class _SvcManager(BaseManager):
    pass


class TestBoundedRPCCalls(unittest.TestCase):
    server_addr = None
    _service = None

    @classmethod
    def setUpClass(cls):
        cls._service = _WedgeableService()
        _SvcManager.register("TestSvc", callable=lambda: cls._service)
        mgr = _SvcManager(address=("127.0.0.1", 0), authkey=_AUTHKEY)
        server = mgr.get_server()
        cls.server_addr = server.address
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def _make_client(self, timeout_s=0.5):
        """RPC-mode client wired to the real in-process BaseManager server."""

        class _ClientManager(BaseManager):
            pass

        _ClientManager.register("TestSvc")
        cm = _ClientManager(address=self.server_addr, authkey=_AUTHKEY)
        cm.connect()

        client = RPCClientBase(
            service_name="TestSvc",
            port=self.server_addr[1],
            connection_mode=RPCConnectionMode.RPC,
            rpc_call_timeout_s=timeout_s,
        )
        client._manager = cm
        client._proxy = cm.TestSvc()
        client._connected = True
        self.addCleanup(client.disconnect)
        return client

    def setUp(self):
        # Re-arm the wedge gate between tests.
        self._service._gate.clear()

    def test_healthy_call_passes_through(self):
        client = self._make_client()
        self.assertEqual(client._server.echo_rpc({"k": [1, 2]}), {"k": [1, 2]})

    def test_remote_exception_propagates_unchanged(self):
        client = self._make_client()
        with self.assertRaises(Exception) as ctx:
            client._server.raise_rpc()
        self.assertIn("boom from remote", str(ctx.exception))

    def test_wedged_service_frees_caller_within_timeout(self):
        client = self._make_client(timeout_s=0.5)
        start = time.monotonic()
        with self.assertRaises(RPCCallTimeoutError) as ctx:
            client._server.wedge_rpc()
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 3.0, f"caller was pinned for {elapsed:.1f}s")
        self.assertEqual(ctx.exception.service_name, "TestSvc")
        self.assertEqual(ctx.exception.method_name, "wedge_rpc")

        # Service recovers -> the SAME client works again immediately.
        self._service._gate.set()
        self.assertEqual(client._server.echo_rpc("after-recovery"), "after-recovery")

    def test_concurrent_wedged_calls_all_freed(self):
        client = self._make_client(timeout_s=0.5)
        errors = []

        def call_wedge():
            try:
                client._server.wedge_rpc()
                errors.append("call unexpectedly succeeded")
            except RPCCallTimeoutError:
                pass
            except Exception as e:  # pragma: no cover
                errors.append(repr(e))

        threads = [threading.Thread(target=call_wedge) for _ in range(3)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.monotonic() - start
        self.assertEqual(errors, [])
        self.assertLess(elapsed, 5.0, f"concurrent callers pinned for {elapsed:.1f}s")
        self._service._gate.set()

    def test_local_mode_bypasses_facade_entirely(self):
        client = RPCClientBase(
            service_name="TestSvc",
            port=self.server_addr[1],
            connection_mode=RPCConnectionMode.LOCAL,
        )
        self.addCleanup(client.disconnect)
        direct = _WedgeableService()
        client.set_direct_server(direct)
        # Identity: LOCAL mode returns the direct object, no wrapper, no executor.
        self.assertIs(client._server, direct)
        self.assertIsNone(client._rpc_executor)

    def test_timeout_opt_out_returns_raw_proxy(self):
        client = self._make_client(timeout_s=0)  # <= 0 disables bounding
        self.assertIs(client._server, client._proxy)

    def test_disconnected_client_raises_runtime_error(self):
        client = RPCClientBase(
            service_name="TestSvc",
            port=self.server_addr[1],
            connection_mode=RPCConnectionMode.RPC,
            rpc_call_timeout_s=0.5,
        )
        self.addCleanup(client.disconnect)
        client.connect = lambda *a, **k: False  # keep _proxy None
        with self.assertRaises(RuntimeError) as ctx:
            client._server.echo_rpc("x")
        self.assertIn("Not connected", str(ctx.exception))

    def test_pickle_state_excludes_bounded_machinery(self):
        client = self._make_client()
        client._get_rpc_executor()  # force executor creation
        state = client.__getstate__()
        self.assertIsNone(state["_rpc_executor"])
        self.assertIsNone(state["_rpc_executor_lock"])
        self.assertIsNone(state["_bounded_proxy"])


if __name__ == "__main__":
    unittest.main()
