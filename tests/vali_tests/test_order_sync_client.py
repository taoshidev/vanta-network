# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for the cross-process order/sync coordination (spec R2.1 + R2.4):
  - CommonDataServer's begin_order/end_order/wait_for_orders/sync_waiting RPC methods
  - Token-based in-flight tracking with TTL reaping (a crashed producer can't deadlock sync)
  - OrderSyncClient adapter (drop-in for OrderSyncState, backed by CommonDataServer)

Mirrors tests/vali_tests/test_order_sync_state.py so the RPC port is proven to preserve the
same semantics, plus the behaviors the in-memory version never needed: fail-open (coordination
unreachable) and TTL self-healing (producer crash). Uses LOCAL mode + set_direct_server: the
client routes calls straight to a server object in-process, and the server's threading.Condition
coordinates across the test's threads exactly as it would across RPC worker threads in production.
"""
import threading
import time
import unittest

from shared_objects.rpc.common_data_server import CommonDataServer
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.server_registry import ServerRegistry
from vali_objects.data_sync.order_sync_client import OrderSyncClient
from vali_objects.vali_config import RPCConnectionMode


class TestOrderSyncClient(unittest.TestCase):
    """Cross-process coordination parity + fail-open + TTL reaping."""

    def setUp(self):
        # Fresh in-process CommonDataServer + wired OrderSyncClient per test (LOCAL mode).
        self.server = CommonDataServer(
            start_server=False,
            connection_mode=RPCConnectionMode.LOCAL,
            running_unit_tests=True,
        )
        cdc = CommonDataClient(connection_mode=RPCConnectionMode.LOCAL, running_unit_tests=True)
        cdc.set_direct_server(self.server)
        self.sync = OrderSyncClient(common_data_client=cdc)

    def tearDown(self):
        # Unregister only (NOT shutdown()) — shutdown() would trip the shared-memory shutdown
        # flag and leak into other tests. Direct unregister frees the name/port for the next test.
        ServerRegistry.unregister(self.server)

    def test_basic_counter(self):
        self.assertEqual(self.sync.get_order_count(), 0)
        token = self.sync._client.begin_order()
        self.assertEqual(self.sync.get_order_count(), 1)
        self.sync._client.end_order(token)
        self.assertEqual(self.sync.get_order_count(), 0)

    def test_end_order_unknown_token_noop(self):
        """Ending an unknown/None token is a harmless no-op (idempotent, e.g. already reaped)."""
        self.assertEqual(self.sync._client.end_order(999), 0)
        self.assertEqual(self.sync._client.end_order(None), 0)
        self.assertEqual(self.sync.get_order_count(), 0)

    def test_context_manager(self):
        self.assertEqual(self.sync.get_order_count(), 0)
        with self.sync.begin_order():
            self.assertEqual(self.sync.get_order_count(), 1)
        self.assertEqual(self.sync.get_order_count(), 0)

    def test_context_manager_with_exception(self):
        with self.assertRaises(ValueError):
            with self.sync.begin_order():
                self.assertEqual(self.sync.get_order_count(), 1)
                raise ValueError("Test exception")
        self.assertEqual(self.sync.get_order_count(), 0, "Count must decrement even on exception")

    def test_sync_waiting_flag(self):
        self.assertFalse(self.sync.is_sync_waiting())

        def sync_thread():
            with self.sync.begin_sync():
                time.sleep(0.1)

        t = threading.Thread(target=sync_thread)
        t.start()
        time.sleep(0.05)
        self.assertTrue(self.sync.is_sync_waiting(), "Should be waiting during sync")
        t.join()
        self.assertFalse(self.sync.is_sync_waiting(), "Should clear after sync completes")

    def test_wait_for_orders_blocks_until_drained(self):
        token = self.sync._client.begin_order()  # one order in flight

        sync_completed = [False]

        def sync_thread():
            with self.sync.begin_sync():
                sync_completed[0] = True

        t = threading.Thread(target=sync_thread)
        t.start()
        time.sleep(0.05)
        self.assertFalse(sync_completed[0], "Sync must wait for the in-flight order")
        self.assertTrue(self.sync.is_sync_waiting())

        self.sync._client.end_order(token)  # order finishes -> notify
        t.join(timeout=1.0)
        self.assertTrue(sync_completed[0], "Sync should proceed once orders drain")
        self.assertFalse(self.sync.is_sync_waiting())

    def test_wait_for_orders_timeout(self):
        """wait_for_orders returns False on timeout and clears sync_waiting."""
        self.sync._client.begin_order()  # never drained, TTL default (won't reap in 0.1s)
        result = self.sync.wait_for_orders(timeout_seconds=0.1)
        self.assertFalse(result, "Should time out with an order still in flight")
        self.assertFalse(self.sync.is_sync_waiting(), "sync_waiting must be cleared on timeout")

    def test_multiple_concurrent_orders(self):
        def process_order():
            with self.sync.begin_order():
                time.sleep(0.01)

        threads = [threading.Thread(target=process_order) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.sync.get_order_count(), 0)
        self.assertFalse(self.sync.is_sync_waiting())

    def test_clear_test_state_resets_orders(self):
        self.sync._client.begin_order()
        self.server.set_sync_waiting_rpc(True)
        self.server.clear_test_state_rpc()
        self.assertEqual(self.sync.get_order_count(), 0)
        self.assertFalse(self.sync.is_sync_waiting())

    # ==================== TTL reaping (spec R2.4) ====================

    def test_stale_inflight_reaped(self):
        """A crashed producer (begin, never end) must not linger — the TTL reaps it."""
        self.server._inflight_ttl_ms = 50
        self.sync._client.begin_order()  # token dropped; simulates producer crash mid-order
        self.assertEqual(self.sync.get_order_count(), 1)
        time.sleep(0.08)  # exceed TTL
        self.assertEqual(self.sync.get_order_count(), 0, "Stale registration should be reaped")

    def test_wait_for_orders_proceeds_after_reap(self):
        """Sync proceeds once an abandoned registration is reaped, with no end_order ever arriving."""
        self.server._inflight_ttl_ms = 50
        self.server._WAIT_POLL_SECONDS = 0.02  # tighten poll so the test is fast
        self.sync._client.begin_order()  # never ended (crashed producer)
        # begin_sync() enters only if wait_for_orders returned True (else it raises TimeoutError).
        with self.sync.begin_sync(timeout_seconds=2.0):
            pass  # reaching here proves sync drained via reap rather than timing out
        # mark_sync_complete (on context exit) clears the flag.
        self.assertFalse(self.sync.is_sync_waiting())

    def test_inflight_label_visible_for_debug(self):
        """The debug label (order_uuid + context) is queryable while an order is in flight."""
        self.server._inflight_ttl_ms = 10_000
        with self.sync.begin_order(label="hk123:BTCUSD:uuid-abc"):
            state = self.server.get_order_sync_state_rpc()
            self.assertEqual(state["in_flight_count"], 1)
            self.assertEqual(len(state["in_flight_sample"]), 1)
            self.assertEqual(state["in_flight_sample"][0]["label"], "hk123:BTCUSD:uuid-abc")
            self.assertGreaterEqual(state["in_flight_sample"][0]["age_ms"], 0)
            self.assertFalse(state["in_flight_sample_truncated"])
        # Cleared after the context exits.
        self.assertEqual(self.server.get_order_sync_state_rpc()["in_flight_count"], 0)

    def test_inflight_map_bounded_and_sample_capped_under_backlog(self):
        """A large stuck backlog: begin_order keeps reaping, and the debug payload stays bounded."""
        self.server._inflight_ttl_ms = 10_000  # keep them all "live" so none reap during the test
        self.server._IN_FLIGHT_SAMPLE = 5
        for i in range(200):
            self.sync._client.begin_order(label=f"order-{i}")
        state = self.server.get_order_sync_state_rpc()
        self.assertEqual(state["in_flight_count"], 200, "count reflects the true total")
        self.assertEqual(len(state["in_flight_sample"]), 5, "sample is capped")
        self.assertTrue(state["in_flight_sample_truncated"])
        self.assertEqual(state["in_flight_sample"][0]["label"], "order-0", "oldest first")

    def test_begin_order_reaps_stale_prefix(self):
        """begin_order itself reaps expired entries, so an inflow stall can't grow the map forever."""
        self.server._inflight_ttl_ms = 40
        self.sync._client.begin_order(label="stale-1")
        self.sync._client.begin_order(label="stale-2")
        self.assertEqual(self.sync.get_order_count(), 2)
        time.sleep(0.06)  # both exceed TTL
        # A fresh begin should reap the two stale ones as a prefix, leaving only the new entry.
        self.sync._client.begin_order(label="fresh")
        self.assertEqual(self.sync.get_order_count(), 1)

    def test_live_order_not_reaped(self):
        """A still-live order (younger than TTL) must NOT be reaped out from under sync."""
        self.server._inflight_ttl_ms = 10_000  # 10s: order stays live
        token = self.sync._client.begin_order()
        time.sleep(0.05)
        self.assertEqual(self.sync.get_order_count(), 1, "Live order must survive")
        self.sync._client.end_order(token)
        self.assertEqual(self.sync.get_order_count(), 0)

    # ==================== Fail-open behavior (spec R2.3) ====================

    class _RaisingClient:
        """Stub whose coordination calls always fail (simulates core/CommonData down)."""
        def is_sync_waiting(self):
            raise ConnectionError("CommonDataServer unreachable")

        def begin_order(self, label=None):
            raise ConnectionError("CommonDataServer unreachable")

        def end_order(self, token=None):
            raise ConnectionError("CommonDataServer unreachable")

    def test_fail_open_is_sync_waiting(self):
        """If coordination is unreachable, is_sync_waiting degrades to False (orders not blocked)."""
        sync = OrderSyncClient(common_data_client=self._RaisingClient())
        self.assertFalse(sync.is_sync_waiting())

    def test_fail_open_begin_order(self):
        """Unreachable coordination => proceed (fail-open), NOT reject; distinct from 'syncing'."""
        sync = OrderSyncClient(common_data_client=self._RaisingClient())
        admission = sync.begin_order(label="x")
        self.assertFalse(admission.rejected, "unreachable => proceed, not reject")
        self.assertIsNone(admission.token, "proceeds unregistered")

    # ==================== Atomic sync gate — TOCTOU fix (R2.5) ====================

    def test_begin_order_refused_during_sync(self):
        """Once sync_waiting is set, the gate refuses registration (None) so no order slips in."""
        self.server.set_sync_waiting_rpc(True)
        self.assertIsNone(self.sync._client.begin_order(), "server refuses registration during sync")
        admission = self.sync.begin_order(label="late-order")
        self.assertTrue(admission.rejected, "adapter surfaces the refusal so the caller rejects")
        self.assertIsNone(admission.token)
        self.assertEqual(self.sync.get_order_count(), 0, "nothing registered during sync")

    def test_admitted_order_blocks_sync_then_new_orders_refused(self):
        """Order admitted BEFORE sync => sync waits for it; meanwhile NEW orders are gate-refused."""
        admission = self.sync.begin_order(label="early")
        self.assertFalse(admission.rejected)
        self.assertEqual(self.sync.get_order_count(), 1)

        sync_done = [False]

        def sync_thread():
            with self.sync.begin_sync(timeout_seconds=1.0):
                sync_done[0] = True

        t = threading.Thread(target=sync_thread)
        t.start()
        time.sleep(0.05)
        self.assertFalse(sync_done[0], "sync must wait for the already-admitted order")
        # A new order arriving now (sync_waiting set) is refused by the atomic gate.
        self.assertTrue(self.sync.begin_order(label="late").rejected)
        # End the early order -> count drains -> sync proceeds.
        self.sync._client.end_order(admission.token)
        t.join(timeout=1.0)
        self.assertTrue(sync_done[0], "sync proceeds once the admitted order ends")

    def test_sync_gate_lease_expires_when_owner_dies(self):
        """Core hard-killed mid-sync: no mark_sync_complete, no heartbeat — the gate must
        auto-clear after the lease instead of rejecting orders until the next sync (~24h)."""
        self.assertTrue(self.server.wait_for_orders_rpc(timeout_seconds=0.1))
        self.assertTrue(self.server.is_sync_waiting_rpc())
        # Owner dies: renewals stop. Backdate the last renewal past the lease.
        self.server._sync_lease_renewed_ms -= (self.server._sync_lease_ms + 1)
        self.assertFalse(self.server.is_sync_waiting_rpc(), "expired gate auto-clears")
        self.assertIsNotNone(self.server.begin_order_rpc("post-expiry-order"),
                             "orders are admitted again after the stale gate expires")

    def test_sync_gate_lease_renewal_keeps_gate_held(self):
        """A live owner's heartbeat keeps the gate held past the lease window."""
        self.assertTrue(self.server.wait_for_orders_rpc(timeout_seconds=0.1))
        self.server._sync_lease_renewed_ms -= (self.server._sync_lease_ms + 1)
        # Renewal arrives before anyone observes the gate — but a renewal on an
        # already-expired-but-unobserved gate is still honored (sync_waiting is True).
        self.assertTrue(self.server.renew_sync_lease_rpc())
        self.assertTrue(self.server.is_sync_waiting_rpc(), "renewed gate stays held")
        self.assertIsNone(self.server.begin_order_rpc("order-during-sync"), "orders still gated")
        self.server.mark_sync_complete_rpc()
        self.assertFalse(self.server.is_sync_waiting_rpc())

    def test_sync_side_does_not_fail_open(self):
        """The sync side must NOT swallow errors — it cannot safely rewrite positions blind."""
        class _RaisingWait:
            def wait_for_orders(self, timeout_seconds=None):
                raise ConnectionError("CommonDataServer unreachable")
        sync = OrderSyncClient(common_data_client=_RaisingWait())
        with self.assertRaises(ConnectionError):
            sync.wait_for_orders(timeout_seconds=0.1)


if __name__ == '__main__':
    unittest.main()
