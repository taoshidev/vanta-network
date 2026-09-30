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

    def test_timeout_logging_is_rate_limited(self):
        """Sustained timeouts must not flood ERROR logs: one ERROR per method per
        RPC_TIMEOUT_LOG_INTERVAL_S; the rest drop to DEBUG."""
        from unittest.mock import patch as mock_patch

        client = self._make_client(timeout_s=0.2)
        with mock_patch("shared_objects.rpc.rpc_client_base.logger") as mock_logger:
            for _ in range(3):
                with self.assertRaises(RPCCallTimeoutError):
                    client._server.wedge_rpc()
        # The patched logger is module-global, so other clients' background threads in the
        # same test process can log through it too — count only this method's timeout lines.
        def timeout_lines(mock_method):
            return [c for c in mock_method.call_args_list if "wedge_rpc exceeded" in str(c.args[0])]
        self.assertEqual(len(timeout_lines(mock_logger.error)), 1)
        self.assertEqual(len(timeout_lines(mock_logger.debug)), 2)
        self._service._gate.set()

    def test_per_call_timeout_overrides_client_default(self):
        """_invoke_rpc(timeout_s=...) bounds that call alone: a call that outlasts the client's
        default succeeds under a longer per-call bound, and the default still applies elsewhere."""
        client = self._make_client(timeout_s=0.2)
        threading.Timer(0.6, self._service._gate.set).start()
        self.assertEqual(client._invoke_rpc("wedge_rpc", timeout_s=5.0), "unwedged")

        self._service._gate.clear()
        with self.assertRaises(RPCCallTimeoutError) as ctx:
            client._invoke_rpc("wedge_rpc")
        self.assertEqual(ctx.exception.timeout_s, 0.2)
        self._service._gate.set()

    def test_chain_clients_use_the_chain_timeout(self):
        """Clients whose every call waits on the chain must not inherit the 60s default."""
        from shared_objects.subtensor_ops.subtensor_ops_client import SubtensorOpsClient
        from vali_objects.contract.contract_client import ContractClient
        from vali_objects.vali_config import ValiConfig

        self.assertGreater(ValiConfig.RPC_CHAIN_CALL_TIMEOUT_S, RPCClientBase.RPC_CALL_TIMEOUT_S)
        self.assertEqual(ContractClient.RPC_CALL_TIMEOUT_S, ValiConfig.RPC_CHAIN_CALL_TIMEOUT_S)
        self.assertEqual(SubtensorOpsClient.RPC_CALL_TIMEOUT_S, ValiConfig.RPC_CHAIN_CALL_TIMEOUT_S)
        client = ContractClient(connect_immediately=False)
        self.addCleanup(client.disconnect)
        self.assertEqual(client._rpc_call_timeout_s, ValiConfig.RPC_CHAIN_CALL_TIMEOUT_S)

    def test_pickle_state_excludes_bounded_machinery(self):
        client = self._make_client()
        client._get_rpc_executor()  # force executor creation
        state = client.__getstate__()
        self.assertIsNone(state["_rpc_executor"])
        self.assertIsNone(state["_rpc_executor_lock"])
        self.assertIsNone(state["_bounded_proxy"])


if __name__ == "__main__":
    unittest.main()
