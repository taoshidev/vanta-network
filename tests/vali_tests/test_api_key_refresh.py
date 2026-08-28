# developer: Taoshidev
# Copyright (c) 2026 Taoshi Inc
"""
Tests for vanta_api.api_key_refresh.APIKeyMixin publication semantics and the
APIMetricsTracker alias resolution.

Regression context: the refresh thread REBINDS api_keys_data / api_key_to_alias /
accessible_api_keys. APIMetricsTracker used to capture the alias DICT OBJECT at
construction, so every rebind orphaned its reference and keys added after startup
logged as "unknown_key" until restart. The tracker now accepts a zero-arg provider
and re-resolves the live mapping per lookup.
"""
import json
import os
import tempfile
import unittest

from vanta_api.api_key_refresh import APIKeyMixin
from vanta_api.base_rest_server import APIMetricsTracker


def _write_keys(path, keys: dict):
    with open(path, "w") as f:
        json.dump(keys, f)
    # Force a newer mtime so load_api_keys' modification check always fires.
    stat = os.stat(path)
    os.utime(path, (stat.st_atime, stat.st_mtime + 2))


class TestAPIKeyMixinPublication(unittest.TestCase):
    """load_api_keys publishes a consistent snapshot via rebinds."""

    def test_reload_publishes_consistent_state(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "api_keys.json")
            _write_keys(path, {"alice": {"key": "key-a", "tier": 100}})
            mixin = APIKeyMixin(path)

            self.assertTrue(mixin.is_valid_api_key("key-a"))
            self.assertEqual(mixin.get_api_key_tier("key-a"), 100)

            old_alias = mixin.api_key_to_alias
            _write_keys(path, {
                "alice": {"key": "key-a", "tier": 100},
                "bob": {"key": "key-b", "tier": 30},
            })
            mixin.load_api_keys()

            # New key fully visible: validity, tier, and alias all consistent.
            self.assertTrue(mixin.is_valid_api_key("key-b"))
            self.assertEqual(mixin.get_api_key_tier("key-b"), 30)
            self.assertEqual(mixin.api_key_to_alias["key-b"], "bob")
            # Published by REBIND, not in-place mutation: concurrent readers
            # iterating the old dict must be able to finish on their snapshot.
            self.assertIsNot(mixin.api_key_to_alias, old_alias)
            # The old snapshot is untouched.
            self.assertNotIn("key-b", old_alias)

    def test_every_accessible_key_has_tier_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "api_keys.json")
            _write_keys(path, {
                "alice": {"key": "key-a", "tier": 100},
                "legacy_user": "legacy-key",
            })
            mixin = APIKeyMixin(path)
            for key in mixin.accessible_api_keys:
                self.assertIn(key, mixin.api_keys_data)
                self.assertIn(key, mixin.api_key_to_alias)


class TestAPIMetricsTrackerAliasResolution(unittest.TestCase):
    """The tracker must see alias-map REBINDS (provider form) and still accept a
    plain dict (legacy form)."""

    def test_provider_sees_rebound_mapping(self):
        class Holder:
            def __init__(self):
                self.api_key_to_alias = {"key-a": "alice"}

        holder = Holder()
        tracker = APIMetricsTracker(log_interval_minutes=60,
                                    api_key_mapping=lambda: holder.api_key_to_alias)
        self.assertEqual(tracker._get_user_id_from_api_key("key-a"), "alice")

        # Simulate the refresh thread REBINDING the mapping.
        holder.api_key_to_alias = {"key-a": "alice", "key-b": "bob"}
        self.assertEqual(tracker._get_user_id_from_api_key("key-b"), "bob")

    def test_legacy_dict_form_still_works(self):
        tracker = APIMetricsTracker(log_interval_minutes=60,
                                    api_key_mapping={"key-a": "alice"})
        self.assertEqual(tracker._get_user_id_from_api_key("key-a"), "alice")
        self.assertEqual(tracker._get_user_id_from_api_key("nope"), "unknown_key")


if __name__ == "__main__":
    unittest.main()
