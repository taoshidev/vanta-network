# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for the authoritative order-UUID dedup (spec R2.6):
  - CommonDataServer check_and_add / release / exists / seed RPC methods (+ capacity eviction)
  - OrderUuidDedupClient wrapper (local read-cache + server-authoritative claim)

Covers the two scenarios the whole feature exists for:
  - MULTI-INSTANCE overlap: two clients on one server, only one claim wins.
  - RETRY safety (R4.1): claim -> apply fails -> release -> retry re-claims successfully.

LOCAL mode + set_direct_server routes client calls straight to an in-process server object.
"""
import unittest
from types import SimpleNamespace

from shared_objects.rpc.common_data_server import CommonDataServer
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.server_registry import ServerRegistry
from vali_objects.order_uuid_dedup_client import OrderUuidDedupClient
from vali_objects.vali_config import RPCConnectionMode


def _pos(order_uuids):
    """Minimal position-like object: has .orders, each with .order_uuid."""
    return SimpleNamespace(orders=[SimpleNamespace(order_uuid=u) for u in order_uuids])


class TestOrderUuidDedup(unittest.TestCase):

    def setUp(self):
        self.server = CommonDataServer(
            start_server=False,
            connection_mode=RPCConnectionMode.LOCAL,
            running_unit_tests=True,
        )
        self.dedup = self._new_client()

    def tearDown(self):
        ServerRegistry.unregister(self.server)

    def _new_client(self) -> OrderUuidDedupClient:
        """A fresh dedup client (own local cache) wired to the shared server — models one instance."""
        cdc = CommonDataClient(connection_mode=RPCConnectionMode.LOCAL, running_unit_tests=True)
        cdc.set_direct_server(self.server)
        return OrderUuidDedupClient(common_data_client=cdc)

    # ==================== Core claim semantics ====================

    def test_check_and_add_claims_once(self):
        self.assertTrue(self.dedup.check_and_add("uuid-1"), "first claim succeeds")
        self.assertFalse(self.dedup.check_and_add("uuid-1"), "second claim is a duplicate")

    def test_exists_reflects_claim(self):
        self.assertFalse(self.dedup.exists("uuid-x"))
        self.dedup.check_and_add("uuid-x")
        self.assertTrue(self.dedup.exists("uuid-x"))

    def test_falsy_uuid_not_dedup_able(self):
        """None/empty uuid can't be claimed or deduped -> always 'proceed', never recorded."""
        self.assertTrue(self.dedup.check_and_add(None))
        self.assertTrue(self.dedup.check_and_add(""))
        self.assertFalse(self.dedup.exists(None))
        self.assertEqual(self.server.order_uuid_count_rpc(), 0)

    def test_release_allows_reclaim(self):
        self.assertTrue(self.dedup.check_and_add("uuid-r"))
        self.assertFalse(self.dedup.check_and_add("uuid-r"), "still claimed")
        self.dedup.release("uuid-r")
        self.assertFalse(self.dedup.exists("uuid-r"), "released locally")
        self.assertTrue(self.dedup.check_and_add("uuid-r"), "re-claim after release succeeds")

    # ==================== Seeding from position history ====================

    def test_add_initial_uuids_seeds_server_and_local(self):
        hk_to_positions = {
            "hk1": [_pos(["a", "b"]), _pos(["c"])],
            "hk2": [_pos(["d"])],
        }
        self.dedup.add_initial_uuids(hk_to_positions)
        for u in ["a", "b", "c", "d"]:
            self.assertTrue(self.server.order_uuid_exists_rpc(u), f"{u} seeded on server")
            self.assertTrue(self.dedup.exists(u), f"{u} warmed in local cache")
        # A seeded uuid is treated as a duplicate (already applied historically).
        self.assertFalse(self.dedup.check_and_add("a"))

    # ==================== Capacity eviction ====================

    def test_capacity_eviction_fifo(self):
        self.server._order_uuid_capacity = 3
        for u in ["u1", "u2", "u3"]:
            self.dedup.check_and_add(u)
        self.assertEqual(self.server.order_uuid_count_rpc(), 3)
        self.dedup.check_and_add("u4")  # evicts u1 (oldest)
        self.assertEqual(self.server.order_uuid_count_rpc(), 3)
        self.assertFalse(self.server.order_uuid_exists_rpc("u1"), "oldest evicted")
        self.assertTrue(self.server.order_uuid_exists_rpc("u4"))

    # ==================== The two scenarios this feature exists for ====================

    def test_multi_instance_overlap_single_winner(self):
        """Two vanta-orders instances race the same order uuid — exactly one claim wins."""
        instance_a = self.dedup
        instance_b = self._new_client()  # separate local cache, same server
        self.assertTrue(instance_a.check_and_add("dup-order"))
        self.assertFalse(instance_b.check_and_add("dup-order"),
                         "the second instance must see the duplicate via the server, not its empty local cache")

    def test_retry_after_transient_failure(self):
        """R4.1: claim -> apply fails -> release -> the placer's retry re-claims and succeeds."""
        uuid = "order-retry"
        self.assertTrue(self.dedup.check_and_add(uuid), "attempt 1 claims")
        # simulate apply raising a transient error -> release
        self.dedup.release(uuid)
        self.assertTrue(self.dedup.check_and_add(uuid), "retry re-claims after release")

    def test_retry_after_success_is_deduped(self):
        """If attempt 1 succeeded (claim kept), a retry (lost ack) is correctly rejected."""
        uuid = "order-committed"
        self.assertTrue(self.dedup.check_and_add(uuid))
        # attempt 1 applied successfully -> NO release
        self.assertFalse(self.dedup.check_and_add(uuid), "retry after success must be deduped")

    def test_clear_test_state_resets_dedup(self):
        self.dedup.check_and_add("x")
        self.server.clear_test_state_rpc()
        self.assertEqual(self.server.order_uuid_count_rpc(), 0)


if __name__ == '__main__':
    unittest.main()
