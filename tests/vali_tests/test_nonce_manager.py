# developer: Taoshidev
# Copyright (c) 2025 Taoshi Inc
"""
Tests for vanta_api.nonce_manager.NonceManager — the REST replay-attack guard
used by ValidatorRestServer (signed-request nonce + timestamp validation).

Coverage moved here from the deleted WebSocket entity-auth tests: the WS server
no longer performs its own nonce handling, so NonceManager is now the sole home
for this logic.
"""
import unittest
from unittest.mock import patch

from vanta_api.nonce_manager import NonceManager


# A fixed "now" so timestamp math is fully deterministic. All tests that reach
# the freshness/future checks patch NonceManager's clock to this value.
NOW_MS = 1_700_000_000_000
TTL_MS = 5 * 60 * 1000  # NonceManager default


def _patch_now(*values):
    """Patch NonceManager's clock. One value per is_valid_request() call
    (each call reads the clock exactly once)."""
    return patch(
        "vanta_api.nonce_manager.TimeUtil.now_in_millis",
        side_effect=list(values),
    )


class TestNonceManagerValidation(unittest.TestCase):
    """is_valid_request: freshness, future-dating, and replay detection."""

    def setUp(self):
        self.mgr = NonceManager()
        self.addr = "5FreshAddress"

    def test_fresh_request_accepted_and_recorded(self):
        with _patch_now(NOW_MS):
            ok, err = self.mgr.is_valid_request(self.addr, "n1", NOW_MS)
        self.assertTrue(ok)
        self.assertEqual(err, "")
        # Nonce is now tracked for replay detection.
        self.assertIn("n1", self.mgr.used_nonces[self.addr])
        self.assertEqual(self.mgr.nonce_timestamps[self.addr]["n1"], NOW_MS)

    def test_replayed_nonce_rejected(self):
        with _patch_now(NOW_MS, NOW_MS):
            ok1, _ = self.mgr.is_valid_request(self.addr, "n1", NOW_MS)
            ok2, err2 = self.mgr.is_valid_request(self.addr, "n1", NOW_MS)
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertEqual(err2, "Nonce already used")

    def test_expired_timestamp_rejected_and_not_recorded(self):
        stale_ts = NOW_MS - TTL_MS - 1  # just past the window
        with _patch_now(NOW_MS):
            ok, err = self.mgr.is_valid_request(self.addr, "n1", stale_ts)
        self.assertFalse(ok)
        self.assertIn("expired", err.lower())
        # Rejected before the mark step — must not be tracked.
        self.assertNotIn("n1", self.mgr.used_nonces[self.addr])

    def test_expired_boundary_is_inclusive(self):
        # current_time - timestamp == ttl is NOT expired (check is strict >).
        with _patch_now(NOW_MS):
            ok, err = self.mgr.is_valid_request(self.addr, "n1", NOW_MS - TTL_MS)
        self.assertTrue(ok, err)

    def test_future_timestamp_rejected(self):
        future_ts = NOW_MS + 60 * 1000 + 1  # just past the 1-minute tolerance
        with _patch_now(NOW_MS):
            ok, err = self.mgr.is_valid_request(self.addr, "n1", future_ts)
        self.assertFalse(ok)
        self.assertIn("future", err.lower())
        self.assertNotIn("n1", self.mgr.used_nonces[self.addr])

    def test_future_within_tolerance_accepted(self):
        # Exactly at the tolerance edge is allowed (check is strict >).
        with _patch_now(NOW_MS):
            ok, err = self.mgr.is_valid_request(self.addr, "n1", NOW_MS + 60 * 1000)
        self.assertTrue(ok, err)

    def test_same_nonce_across_addresses_is_independent(self):
        # Nonces are namespaced per address; identical nonce on two addresses is fine.
        with _patch_now(NOW_MS, NOW_MS):
            ok1, _ = self.mgr.is_valid_request("5AddrA", "shared", NOW_MS)
            ok2, _ = self.mgr.is_valid_request("5AddrB", "shared", NOW_MS)
        self.assertTrue(ok1)
        self.assertTrue(ok2)

    def test_expired_nonce_can_be_reused_after_cleanup(self):
        # First use at NOW; second attempt a full TTL later. The stale entry is
        # cleaned inside is_valid_request, so the same nonce is accepted again.
        later = NOW_MS + TTL_MS + 1
        with _patch_now(NOW_MS, later):
            ok1, _ = self.mgr.is_valid_request(self.addr, "n1", NOW_MS)
            ok2, err2 = self.mgr.is_valid_request(self.addr, "n1", later)
        self.assertTrue(ok1)
        self.assertTrue(ok2, err2)


class TestNonceManagerCleanup(unittest.TestCase):
    """_cleanup_expired_nonces: prunes stale entries from both tracking dicts."""

    def setUp(self):
        self.mgr = NonceManager()
        self.addr = "5CleanupAddr"

    def _seed(self, **nonce_to_ts):
        for nonce, ts in nonce_to_ts.items():
            self.mgr.used_nonces[self.addr].add(nonce)
            self.mgr.nonce_timestamps[self.addr][nonce] = ts

    def test_removes_only_expired(self):
        self._seed(old=NOW_MS - TTL_MS - 1, fresh=NOW_MS)
        self.mgr._cleanup_expired_nonces(self.addr, NOW_MS)
        # Expired one gone from both structures; fresh one kept.
        self.assertNotIn("old", self.mgr.used_nonces[self.addr])
        self.assertNotIn("old", self.mgr.nonce_timestamps[self.addr])
        self.assertIn("fresh", self.mgr.used_nonces[self.addr])
        self.assertIn("fresh", self.mgr.nonce_timestamps[self.addr])

    def test_boundary_nonce_is_kept(self):
        # Exactly TTL old is not expired (strict >).
        self._seed(edge=NOW_MS - TTL_MS)
        self.mgr._cleanup_expired_nonces(self.addr, NOW_MS)
        self.assertIn("edge", self.mgr.used_nonces[self.addr])

    def test_empty_address_is_noop(self):
        # Never-seen address must not raise.
        self.mgr._cleanup_expired_nonces("5NeverSeen", NOW_MS)
        self.assertEqual(len(self.mgr.nonce_timestamps["5NeverSeen"]), 0)

    def test_all_expired_clears_address(self):
        self._seed(a=NOW_MS - TTL_MS - 10, b=NOW_MS - TTL_MS - 5)
        self.mgr._cleanup_expired_nonces(self.addr, NOW_MS)
        self.assertEqual(len(self.mgr.used_nonces[self.addr]), 0)
        self.assertEqual(len(self.mgr.nonce_timestamps[self.addr]), 0)


class TestNonceManagerConfig(unittest.TestCase):
    """Constructor wiring."""

    def test_default_ttl(self):
        self.assertEqual(NonceManager().ttl_ms, 5 * 60 * 1000)

    def test_custom_ttl_enforced(self):
        mgr = NonceManager(ttl_ms=1000)
        stale_ts = NOW_MS - 1001
        with _patch_now(NOW_MS):
            ok, err = mgr.is_valid_request("5Addr", "n1", stale_ts)
        self.assertFalse(ok)
        self.assertIn("1000ms", err)


if __name__ == "__main__":
    unittest.main()
